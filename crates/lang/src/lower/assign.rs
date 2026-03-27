use crate::ast::{self, AssignOp, Ident, Type};
use crate::hir;
use crate::span::Span;
use internment::Intern;

use super::{FuncLower, LowerCtx, LowerError, alloc_assign_temp, lower_expr};

fn field_index_for_assign(
    ctx: &LowerCtx,
    target_ty: &Type,
    field_name: Ident,
    span: Span,
) -> Result<u16, LowerError> {
    match target_ty {
        Type::Struct { name, .. } => ctx
            .shared
            .tcx
            .struct_field_index(*name, field_name)
            .ok_or_else(|| LowerError::UnsupportedAssign {
                span,
                detail: format!("unknown field '{field_name}' on struct '{name}'"),
            })
            .map(|i| i as u16),
        Type::DataRef { name, .. } => ctx
            .shared
            .tcx
            .struct_field_index(*name, field_name)
            .ok_or_else(|| LowerError::UnsupportedAssign {
                span,
                detail: format!("unknown field '{field_name}' on dataref '{name}'"),
            })
            .map(|i| i as u16),
        Type::NamedTuple(fields) => fields
            .iter()
            .position(|(label, _)| *label == field_name)
            .ok_or_else(|| LowerError::UnsupportedAssign {
                span,
                detail: format!("unknown field '{field_name}' on named tuple"),
            })
            .map(|i| i as u16),
        other => Err(LowerError::UnsupportedAssign {
            span,
            detail: format!("field assignment on unsupported type '{other}'"),
        }),
    }
}

enum AssignAccessStep<'a> {
    Field {
        field_index: u16,
        value_type: Type,
    },
    Index {
        index_expr: &'a ast::ExprNode,
        value_type: Type,
    },
}

struct AssignAccessChain<'a> {
    root: Ident,
    steps: Vec<AssignAccessStep<'a>>,
}

impl AssignAccessStep<'_> {
    fn value_type(&self) -> &Type {
        match self {
            AssignAccessStep::Field { value_type, .. } => value_type,
            AssignAccessStep::Index { value_type, .. } => value_type,
        }
    }
}

fn extract_assign_access_chain<'a>(
    mut expr: &'a ast::ExprNode,
    ctx: &LowerCtx,
    span: Span,
) -> Result<AssignAccessChain<'a>, LowerError> {
    let mut steps_outer_first = vec![];
    loop {
        match &expr.node.kind {
            ast::ExprKind::Ident(name) => {
                steps_outer_first.reverse();
                return Ok(AssignAccessChain {
                    root: *name,
                    steps: steps_outer_first,
                });
            }
            ast::ExprKind::Field(field_access) => {
                let target = &field_access.node.target;
                let field_name = field_access.node.field;
                let target_ty = ctx.expr_type(target.node.id, span)?;
                let field_index = field_index_for_assign(ctx, &target_ty, field_name, span)?;
                let value_type = ctx.expr_type(expr.node.id, span)?;
                steps_outer_first.push(AssignAccessStep::Field {
                    field_index,
                    value_type,
                });
                expr = target;
            }
            ast::ExprKind::Index(index_node) => {
                let target = &index_node.node.target;
                let index_expr = &index_node.node.index;
                let value_type = ctx.expr_type(expr.node.id, span)?;
                steps_outer_first.push(AssignAccessStep::Index {
                    index_expr,
                    value_type,
                });
                expr = target;
            }
            _ => {
                return Err(LowerError::UnsupportedAssign {
                    span,
                    detail:
                        "assignment target must be a variable, field access, or index expression"
                            .to_string(),
                });
            }
        }
    }
}

fn lower_assign_to_chain(
    target: &ast::ExprNode,
    value_ast: &ast::ExprNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    let chain = extract_assign_access_chain(target, ctx, span)?;
    let root_local = *fc
        .local_map
        .get(&chain.root)
        .ok_or(LowerError::UnknownLocal {
            name: chain.root,
            span,
        })?;
    let root_ty = fc.locals[root_local.0 as usize].ty.clone();
    let n = chain.steps.len();

    if n == 1 {
        return match &chain.steps[0] {
            AssignAccessStep::Field { field_index, .. } => {
                let value = lower_expr(value_ast, ctx, fc, out)?;
                Ok(hir::Stmt {
                    span,
                    kind: hir::StmtKind::SetField {
                        object: root_local,
                        field_index: *field_index,
                        value,
                    },
                })
            }
            AssignAccessStep::Index { index_expr, .. } => {
                let index = lower_expr(index_expr, ctx, fc, out)?;
                let value = lower_expr(value_ast, ctx, fc, out)?;
                Ok(hir::Stmt {
                    span,
                    kind: hir::StmtKind::SetIndex {
                        object: root_local,
                        index: Box::new(index),
                        value,
                    },
                })
            }
        };
    }

    let mut index_temp_ids: Vec<Option<hir::LocalId>> = vec![None; n];
    for (i, step) in chain.steps[..n - 1].iter().enumerate() {
        if let AssignAccessStep::Index { index_expr, .. } = step {
            let idx_ty = ctx.expr_type(index_expr.node.id, span)?;
            let idx_local = alloc_assign_temp(fc, idx_ty);
            index_temp_ids[i] = Some(idx_local);
            let hir_idx = lower_expr(index_expr, ctx, fc, out)?;
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: idx_local,
                    init: hir_idx,
                },
            });
        }
    }

    let mut value_temp_ids: Vec<hir::LocalId> = vec![];
    for i in 0..n - 1 {
        let ty = chain.steps[i].value_type().clone();
        value_temp_ids.push(alloc_assign_temp(fc, ty));
    }

    for i in 0..n - 1 {
        let source_expr = if i == 0 {
            hir::Expr::new(root_ty.clone(), span, hir::ExprKind::Local(root_local))
        } else {
            hir::Expr::new(
                chain.steps[i - 1].value_type().clone(),
                span,
                hir::ExprKind::Local(value_temp_ids[i - 1]),
            )
        };
        let init = match &chain.steps[i] {
            AssignAccessStep::Field { field_index, .. } => hir::Expr::new(
                chain.steps[i].value_type().clone(),
                span,
                hir::ExprKind::FieldGet { object: Box::new(source_expr), index: *field_index },
            ),
            AssignAccessStep::Index { .. } => {
                let idx_local = index_temp_ids[i].expect("index temp for non-leaf Index step");
                let idx_ty = fc.locals[idx_local.0 as usize].ty.clone();
                hir::Expr::new(
                    chain.steps[i].value_type().clone(),
                    span,
                    hir::ExprKind::IndexGet {
                        target: Box::new(source_expr),
                        index: Box::new(hir::Expr::new(
                            idx_ty,
                            span,
                            hir::ExprKind::Local(idx_local),
                        )),
                    },
                )
            }
        };
        out.push(hir::Stmt {
            span,
            kind: hir::StmtKind::Let {
                local: value_temp_ids[i],
                init,
            },
        });
    }

    let value = lower_expr(value_ast, ctx, fc, out)?;

    let parent = value_temp_ids[n - 2];
    let leaf_stmt = match &chain.steps[n - 1] {
        AssignAccessStep::Field { field_index, .. } => hir::Stmt {
            span,
            kind: hir::StmtKind::SetField {
                object: parent,
                field_index: *field_index,
                value,
            },
        },
        AssignAccessStep::Index { index_expr, .. } => {
            let index = lower_expr(index_expr, ctx, fc, out)?;
            hir::Stmt {
                span,
                kind: hir::StmtKind::SetIndex {
                    object: parent,
                    index: Box::new(index),
                    value,
                },
            }
        }
    };
    out.push(leaf_stmt);

    let mut write_back: Vec<hir::Stmt> = vec![];
    for k in (0..n - 1).rev() {
        let new_value = hir::Expr::new(
            chain.steps[k].value_type().clone(),
            span,
            hir::ExprKind::Local(value_temp_ids[k]),
        );
        let stmt = match &chain.steps[k] {
            AssignAccessStep::Field { field_index, .. } => {
                let object = if k == 0 {
                    root_local
                } else {
                    value_temp_ids[k - 1]
                };
                hir::Stmt {
                    span,
                    kind: hir::StmtKind::SetField {
                        object,
                        field_index: *field_index,
                        value: new_value,
                    },
                }
            }
            AssignAccessStep::Index { .. } => {
                let object = if k == 0 {
                    root_local
                } else {
                    value_temp_ids[k - 1]
                };
                let idx_local = index_temp_ids[k].expect("index temp for write-back");
                let idx_ty = fc.locals[idx_local.0 as usize].ty.clone();
                hir::Stmt {
                    span,
                    kind: hir::StmtKind::SetIndex {
                        object,
                        index: Box::new(hir::Expr::new(
                            idx_ty,
                            span,
                            hir::ExprKind::Local(idx_local),
                        )),
                        value: new_value,
                    },
                }
            }
        };
        write_back.push(stmt);
    }

    let mut iter = write_back.into_iter();
    let last = iter.next_back().expect("n > 1 implies write-back");
    for s in iter {
        out.push(s);
    }
    Ok(last)
}

pub(super) fn lower_assign(
    assign_node: &ast::AssignNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    if assign_node.node.op != AssignOp::Assign {
        return Err(LowerError::UnsupportedAssign {
            span,
            detail: format!(
                "compound assignment '{}' is not supported in HIR v1",
                assign_node.node.op
            ),
        });
    }

    match &assign_node.node.target.node.kind {
        ast::ExprKind::Ident(name) => {
            let local_id = *fc
                .local_map
                .get(name)
                .ok_or(LowerError::UnknownLocal { name: *name, span })?;
            let value = lower_expr(&assign_node.node.value, ctx, fc, out)?;
            Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: local_id,
                    value,
                },
            })
        }

        ast::ExprKind::Field(field_access) => {
            let target_expr = &field_access.node.target;
            let target_ty = ctx.expr_type(target_expr.node.id, span)?;
            if let Type::Extern { name } = &target_ty {
                let field_name = field_access.node.field;
                let qualified = Ident(Intern::new(format!("{name}::__set_{field_name}")));
                let extern_id =
                    *ctx.shared
                        .externs
                        .get(&qualified)
                        .ok_or_else(|| LowerError::UnsupportedAssign {
                            span,
                            detail: format!("unknown extern field setter '{qualified}'"),
                        })?;
                let receiver = lower_expr(target_expr, ctx, fc, out)?;
                let value = lower_expr(&assign_node.node.value, ctx, fc, out)?;
                Ok(hir::Stmt {
                    span,
                    kind: hir::StmtKind::Expr(hir::Expr::new(
                        Type::Void,
                        span,
                        hir::ExprKind::CallExtern { extern_id, args: vec![receiver, value] },
                    )),
                })
            } else {
                lower_assign_to_chain(
                    &assign_node.node.target,
                    &assign_node.node.value,
                    span,
                    ctx,
                    fc,
                    out,
                )
            }
        }
        ast::ExprKind::Index(_) => lower_assign_to_chain(
            &assign_node.node.target,
            &assign_node.node.value,
            span,
            ctx,
            fc,
            out,
        ),

        _ => Err(LowerError::UnsupportedAssign {
            span,
            detail: "assignment target must be a variable, field access, or index expression"
                .to_string(),
        }),
    }
}
