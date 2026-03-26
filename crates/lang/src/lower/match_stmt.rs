use crate::ast::{self, BinaryOp, Lit, Pattern, Type};
use crate::hir;
use crate::span::Span;

use super::{
    FuncLower, LowerCtx, LowerError, lower_block, lower_expr, register_named_local,
    resolve_variant_index,
};

fn lower_arm_body(
    body_expr: &ast::ExprNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
) -> Result<hir::Block, LowerError> {
    if let ast::ExprKind::Block(block_node) = &body_expr.node.kind {
        lower_block(block_node, ctx, fc, is_func_body, ret_ty)
    } else {
        let span = body_expr.span;
        let mut arm_stmts = vec![];
        let hir_expr = lower_expr(body_expr, ctx, fc, &mut arm_stmts)?;
        let kind = if is_func_body && !ret_ty.is_void() {
            hir::StmtKind::Return(Some(hir_expr))
        } else {
            hir::StmtKind::Expr(hir_expr)
        };
        arm_stmts.push(hir::Stmt { span, kind });
        Ok(hir::Block { stmts: arm_stmts })
    }
}

pub(super) fn lower_match_stmts(
    match_node: &ast::MatchNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    let scrutinee_expr = lower_expr(&match_node.node.scrutinee, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty.clone(),
    });

    match &scrutinee_ty {
        Type::Enum { name, type_args } => lower_match_enum(
            match_node,
            span,
            ctx,
            fc,
            is_func_body,
            ret_ty,
            scrutinee_expr,
            scrutinee_local,
            &scrutinee_ty,
            *name,
            type_args,
        ),
        _ => lower_match_non_enum(
            match_node,
            span,
            ctx,
            fc,
            is_func_body,
            ret_ty,
            scrutinee_expr,
            scrutinee_local,
            &scrutinee_ty,
            out,
        ),
    }
}

fn lower_match_enum(
    match_node: &ast::MatchNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
    scrutinee_expr: hir::Expr,
    scrutinee_local: hir::LocalId,
    scrutinee_ty: &Type,
    enum_name: ast::Ident,
    type_args: &[Type],
) -> Result<hir::Stmt, LowerError> {
    let mut arms: Vec<hir::MatchArm> = vec![];
    let mut else_body: Option<hir::MatchElse> = None;

    for arm in &match_node.node.arms {
        let mark = fc.enter_scope();

        match &arm.node.pattern.node {
            Pattern::Wildcard => {
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                else_body = Some(hir::MatchElse {
                    binding: None,
                    body,
                });
            }

            Pattern::Ident(name) => {
                let binding_local = register_named_local(fc, *name, scrutinee_ty.clone());
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                else_body = Some(hir::MatchElse {
                    binding: Some((binding_local, false)),
                    body,
                });
            }

            Pattern::VarIdent(name) => {
                let binding_local = register_named_local(fc, *name, scrutinee_ty.clone());
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                else_body = Some(hir::MatchElse {
                    binding: Some((binding_local, true)),
                    body,
                });
            }

            Pattern::EnumUnit {
                qualifier: _,
                variant,
            } => {
                let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    body,
                });
            }

            Pattern::EnumTuple {
                qualifier: _,
                variant,
                fields: subpatterns,
            } => {
                let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                let field_types = ctx
                    .shared
                    .tcx
                    .enum_variant_field_types(enum_name, *variant, type_args)
                    .unwrap_or_default();

                let mut bindings = vec![];
                for (field_idx, subpat) in subpatterns.iter().enumerate() {
                    match &subpat.node {
                        Pattern::Ident(binding_name) => {
                            let field_ty =
                                field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                            let local = register_named_local(fc, *binding_name, field_ty);
                            bindings.push(hir::MatchBinding {
                                field_index: field_idx as u16,
                                local,
                                mutable: false,
                            });
                        }
                        Pattern::VarIdent(binding_name) => {
                            let field_ty =
                                field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                            let local = register_named_local(fc, *binding_name, field_ty);
                            bindings.push(hir::MatchBinding {
                                field_index: field_idx as u16,
                                local,
                                mutable: true,
                            });
                        }
                        _ => {}
                    }
                }
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings,
                    body,
                });
            }

            Pattern::EnumStruct {
                qualifier: _,
                variant,
                fields: field_patterns,
                has_rest: _,
            } => {
                let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                let field_names = ctx
                    .shared
                    .tcx
                    .enum_variant_field_names(enum_name, *variant)
                    .unwrap_or_default();
                let field_types = ctx
                    .shared
                    .tcx
                    .enum_variant_field_types(enum_name, *variant, type_args)
                    .unwrap_or_default();

                let mut bindings = vec![];
                for (pat_field_name, subpat) in field_patterns {
                    match &subpat.node {
                        Pattern::Ident(binding_name) => {
                            let field_idx = field_names
                                .iter()
                                .position(|n| n == pat_field_name)
                                .unwrap_or(0);
                            let field_ty =
                                field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                            let local = register_named_local(fc, *binding_name, field_ty);
                            bindings.push(hir::MatchBinding {
                                field_index: field_idx as u16,
                                local,
                                mutable: false,
                            });
                        }
                        Pattern::VarIdent(binding_name) => {
                            let field_idx = field_names
                                .iter()
                                .position(|n| n == pat_field_name)
                                .unwrap_or(0);
                            let field_ty =
                                field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                            let local = register_named_local(fc, *binding_name, field_ty);
                            bindings.push(hir::MatchBinding {
                                field_index: field_idx as u16,
                                local,
                                mutable: true,
                            });
                        }
                        _ => {}
                    }
                }
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings,
                    body,
                });
            }

            other => {
                return Err(LowerError::UnsupportedExprKind {
                    span,
                    kind: format!("unsupported match pattern '{}'", other.variant_name()),
                });
            }
        }

        fc.leave_scope(mark);
    }

    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            arms,
            else_body,
        },
    })
}

fn lower_match_non_enum(
    match_node: &ast::MatchNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
    scrutinee_expr: hir::Expr,
    scrutinee_local: hir::LocalId,
    scrutinee_ty: &Type,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Let {
            local: scrutinee_local,
            init: scrutinee_expr,
        },
    });

    let mut cond_arms: Vec<(hir::Expr, hir::Block)> = vec![];
    let mut else_block: Option<hir::Block> = None;

    for arm in &match_node.node.arms {
        let mark = fc.enter_scope();

        match &arm.node.pattern.node {
            Pattern::Wildcard => {
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                else_block = Some(body);
            }

            Pattern::Ident(name) => {
                let local = register_named_local(fc, *name, scrutinee_ty.clone());
                let mut body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                let binding_stmt = hir::Stmt {
                    span,
                    kind: hir::StmtKind::Let {
                        local,
                        init: hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local),
                    },
                };
                body.stmts.insert(0, binding_stmt);
                else_block = Some(body);
            }

            Pattern::VarIdent(name) => {
                let local = register_named_local(fc, *name, scrutinee_ty.clone());
                let mut body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                let binding_stmt = hir::Stmt {
                    span,
                    kind: hir::StmtKind::Let {
                        local,
                        init: hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local),
                    },
                };
                body.stmts.insert(0, binding_stmt);
                else_block = Some(body);
            }

            Pattern::Lit(lit) => {
                let cond = build_lit_cond(scrutinee_ty, scrutinee_local, lit, span)?;
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                cond_arms.push((cond, body));
            }

            Pattern::Tuple(subpats) => {
                let elem_types = match scrutinee_ty {
                    Type::Tuple(elems) => elems.clone(),
                    Type::NamedTuple(fields) => fields.iter().map(|(_, ty)| ty.clone()).collect(),
                    _ => vec![],
                };

                let mut conditions: Vec<hir::Expr> = vec![];
                let mut preamble: Vec<hir::Stmt> = vec![];
                let mut is_catchall = true;

                for (i, subpat) in subpats.iter().enumerate() {
                    let elem_ty = elem_types.get(i).cloned().unwrap_or(Type::Void);
                    match &subpat.node {
                        Pattern::Lit(lit) => {
                            is_catchall = false;
                            let elem_expr = hir::Expr {
                                ty: elem_ty,
                                span,
                                kind: hir::ExprKind::TupleIndex {
                                    tuple: Box::new(hir::Expr::local(
                                        scrutinee_ty.clone(),
                                        span,
                                        scrutinee_local,
                                    )),
                                    index: i as u16,
                                },
                            };
                            let rhs = build_rhs_from_lit(lit, span)?;
                            conditions.push(hir::Expr::binary(
                                Type::Bool,
                                span,
                                BinaryOp::Eq,
                                elem_expr,
                                rhs,
                            ));
                        }
                        Pattern::Ident(name) => {
                            let local = register_named_local(fc, *name, elem_ty.clone());
                            preamble.push(hir::Stmt {
                                span,
                                kind: hir::StmtKind::Let {
                                    local,
                                    init: hir::Expr {
                                        ty: elem_ty,
                                        span,
                                        kind: hir::ExprKind::TupleIndex {
                                            tuple: Box::new(hir::Expr::local(
                                                scrutinee_ty.clone(),
                                                span,
                                                scrutinee_local,
                                            )),
                                            index: i as u16,
                                        },
                                    },
                                },
                            });
                        }
                        Pattern::VarIdent(name) => {
                            let local = register_named_local(fc, *name, elem_ty.clone());
                            preamble.push(hir::Stmt {
                                span,
                                kind: hir::StmtKind::Let {
                                    local,
                                    init: hir::Expr {
                                        ty: elem_ty,
                                        span,
                                        kind: hir::ExprKind::TupleIndex {
                                            tuple: Box::new(hir::Expr::local(
                                                scrutinee_ty.clone(),
                                                span,
                                                scrutinee_local,
                                            )),
                                            index: i as u16,
                                        },
                                    },
                                },
                            });
                        }
                        _ => {}
                    }
                }

                let mut body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                for (i, stmt) in preamble.into_iter().enumerate() {
                    body.stmts.insert(i, stmt);
                }

                if is_catchall {
                    else_block = Some(body);
                } else {
                    let cond = conditions
                        .into_iter()
                        .reduce(|acc, c| hir::Expr::binary(Type::Bool, span, BinaryOp::And, acc, c))
                        .unwrap();
                    cond_arms.push((cond, body));
                }
            }

            other => {
                return Err(LowerError::UnsupportedExprKind {
                    span,
                    kind: format!("unsupported match pattern '{}'", other.variant_name()),
                });
            }
        }

        fc.leave_scope(mark);
    }

    let mut current_else = else_block;
    for (cond, then_block) in cond_arms.into_iter().rev() {
        let if_stmt = hir::Stmt {
            span,
            kind: hir::StmtKind::If {
                cond,
                then_block,
                else_block: current_else,
            },
        };
        current_else = Some(hir::Block {
            stmts: vec![if_stmt],
        });
    }

    let outermost = current_else.ok_or_else(|| LowerError::UnsupportedExprKind {
        span,
        kind: "match with no arms".into(),
    })?;

    if outermost.stmts.len() == 1 {
        if let hir::StmtKind::If { .. } = &outermost.stmts[0].kind {
            return Ok(outermost.stmts.into_iter().next().unwrap());
        }
    }

    let true_cond = hir::Expr {
        ty: Type::Bool,
        span,
        kind: hir::ExprKind::Bool(true),
    };
    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::If {
            cond: true_cond,
            then_block: outermost,
            else_block: None,
        },
    })
}

fn build_rhs_from_lit(lit: &Lit, span: Span) -> Result<hir::Expr, LowerError> {
    Ok(match lit {
        Lit::Int(v) => hir::Expr::int_lit(span, *v),
        Lit::Bool(v) => hir::Expr {
            ty: Type::Bool,
            span,
            kind: hir::ExprKind::Bool(*v),
        },
        Lit::String(s) => hir::Expr {
            ty: Type::String,
            span,
            kind: hir::ExprKind::String(s.clone()),
        },
        Lit::Float(_) | Lit::Nil => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: "float and nil literal patterns not supported in match".into(),
            });
        }
    })
}

fn build_lit_cond(
    scrutinee_ty: &Type,
    scrutinee_local: hir::LocalId,
    lit: &Lit,
    span: Span,
) -> Result<hir::Expr, LowerError> {
    let lhs = hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local);
    let rhs = build_rhs_from_lit(lit, span)?;
    Ok(hir::Expr::binary(Type::Bool, span, BinaryOp::Eq, lhs, rhs))
}
