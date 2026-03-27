use crate::ast::{self, BinaryOp, Lit, Pattern, Type, UnaryOp};
use crate::hir;
use crate::span::Span;
use crate::typecheck::ConstValue;

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
                let span_key = (arm.node.pattern.span.start, arm.node.pattern.span.end);
                if let Some(cv) = ctx.shared.tcx.const_pattern_values.get(&span_key) {
                    let cond = build_const_cond(scrutinee_ty, scrutinee_local, cv, span);
                    let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                    cond_arms.push((cond, body));
                } else {
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
                            let elem_expr = hir::Expr::new(
                                elem_ty,
                                span,
                                hir::ExprKind::TupleIndex {
                                    tuple: Box::new(hir::Expr::local(
                                        scrutinee_ty.clone(),
                                        span,
                                        scrutinee_local,
                                    )),
                                    index: i as u16,
                                },
                            );
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
                                    init: hir::Expr::new(
                                        elem_ty,
                                        span,
                                        hir::ExprKind::TupleIndex {
                                            tuple: Box::new(hir::Expr::local(
                                                scrutinee_ty.clone(),
                                                span,
                                                scrutinee_local,
                                            )),
                                            index: i as u16,
                                        },
                                    ),
                                },
                            });
                        }
                        Pattern::VarIdent(name) => {
                            let local = register_named_local(fc, *name, elem_ty.clone());
                            preamble.push(hir::Stmt {
                                span,
                                kind: hir::StmtKind::Let {
                                    local,
                                    init: hir::Expr::new(
                                        elem_ty,
                                        span,
                                        hir::ExprKind::TupleIndex {
                                            tuple: Box::new(hir::Expr::local(
                                                scrutinee_ty.clone(),
                                                span,
                                                scrutinee_local,
                                            )),
                                            index: i as u16,
                                        },
                                    ),
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

    if outermost.stmts.len() == 1
        && let hir::StmtKind::If { .. } = &outermost.stmts[0].kind
    {
        return Ok(outermost.stmts.into_iter().next().unwrap());
    }

    let true_cond = hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(true));
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
        Lit::Bool(v) => hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(*v)),
        Lit::String(s) => hir::Expr::new(Type::String, span, hir::ExprKind::String(s.clone())),
        Lit::Float { .. } | Lit::Nil => {
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

fn build_rhs_from_const_value(cv: &ConstValue, span: Span) -> hir::Expr {
    match cv {
        ConstValue::Int(n) => hir::Expr::int_lit(span, *n),
        ConstValue::Float(f) => hir::Expr::new(Type::Float, span, hir::ExprKind::Float(*f)),
        ConstValue::Double(d) => hir::Expr::new(Type::Double, span, hir::ExprKind::Double(*d)),
        ConstValue::Bool(b) => hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(*b)),
        ConstValue::String(s) => {
            hir::Expr::new(Type::String, span, hir::ExprKind::String(s.clone()))
        }
        ConstValue::Nil => unreachable!("Nil cannot appear as a const pattern in match"),
    }
}

fn build_const_cond(
    scrutinee_ty: &Type,
    scrutinee_local: hir::LocalId,
    cv: &ConstValue,
    span: Span,
) -> hir::Expr {
    let lhs = hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local);
    let rhs = build_rhs_from_const_value(cv, span);
    hir::Expr::binary(Type::Bool, span, BinaryOp::Eq, lhs, rhs)
}

fn build_non_enum_cond_preamble(
    pattern: &Pattern,
    scrutinee_ty: &Type,
    scrutinee_local: hir::LocalId,
    span: Span,
    fc: &mut FuncLower,
) -> Result<(Option<hir::Expr>, Vec<hir::Stmt>), LowerError> {
    match pattern {
        Pattern::Lit(lit) => {
            let cond = build_lit_cond(scrutinee_ty, scrutinee_local, lit, span)?;
            Ok((Some(cond), vec![]))
        }

        Pattern::Ident(name) => {
            let local = register_named_local(fc, *name, scrutinee_ty.clone());
            let binding = hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local,
                    init: hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local),
                },
            };
            Ok((None, vec![binding]))
        }

        Pattern::VarIdent(name) => {
            let local = register_named_local(fc, *name, scrutinee_ty.clone());
            let binding = hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local,
                    init: hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local),
                },
            };
            Ok((None, vec![binding]))
        }

        Pattern::Wildcard => Ok((None, vec![])),

        Pattern::Tuple(subpats) => {
            let elem_types: Vec<Type> = match scrutinee_ty {
                Type::Tuple(elems) => elems.clone(),
                Type::NamedTuple(fields) => fields.iter().map(|(_, ty)| ty.clone()).collect(),
                _ => vec![],
            };
            let mut conditions: Vec<hir::Expr> = vec![];
            let mut preamble: Vec<hir::Stmt> = vec![];

            for (i, subpat) in subpats.iter().enumerate() {
                let elem_ty = elem_types.get(i).cloned().unwrap_or(Type::Void);
                match &subpat.node {
                    Pattern::Lit(lit) => {
                        let elem_expr = hir::Expr::new(
                            elem_ty,
                            span,
                            hir::ExprKind::TupleIndex {
                                tuple: Box::new(hir::Expr::local(
                                    scrutinee_ty.clone(),
                                    span,
                                    scrutinee_local,
                                )),
                                index: i as u16,
                            },
                        );
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
                                init: hir::Expr::new(
                                    elem_ty,
                                    span,
                                    hir::ExprKind::TupleIndex {
                                        tuple: Box::new(hir::Expr::local(
                                            scrutinee_ty.clone(),
                                            span,
                                            scrutinee_local,
                                        )),
                                        index: i as u16,
                                    },
                                ),
                            },
                        });
                    }
                    Pattern::VarIdent(name) => {
                        let local = register_named_local(fc, *name, elem_ty.clone());
                        preamble.push(hir::Stmt {
                            span,
                            kind: hir::StmtKind::Let {
                                local,
                                init: hir::Expr::new(
                                    elem_ty,
                                    span,
                                    hir::ExprKind::TupleIndex {
                                        tuple: Box::new(hir::Expr::local(
                                            scrutinee_ty.clone(),
                                            span,
                                            scrutinee_local,
                                        )),
                                        index: i as u16,
                                    },
                                ),
                            },
                        });
                    }
                    _ => {
                        // skip wildcard and rest
                    }
                }
            }

            let cond = conditions
                .into_iter()
                .reduce(|acc, c| hir::Expr::binary(Type::Bool, span, BinaryOp::And, acc, c));
            Ok((cond, preamble))
        }

        other => Err(LowerError::UnsupportedExprKind {
            span,
            kind: format!(
                "unsupported pattern '{}' in non-enum context",
                other.variant_name()
            ),
        }),
    }
}

pub(super) fn lower_if_let(
    if_let_node: &ast::IfLetNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    let scrutinee_expr = lower_expr(&if_let_node.node.value, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty.clone(),
    });

    match &scrutinee_ty {
        Type::Enum {
            name: enum_name,
            type_args,
        } => {
            let enum_name = *enum_name;
            let mark = fc.enter_scope();
            let pattern = &if_let_node.node.pattern.node;

            match pattern {
                Pattern::Wildcard => {
                    let body =
                        lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                    fc.leave_scope(mark);
                    return Ok(hir::Stmt {
                        span,
                        kind: hir::StmtKind::Match {
                            scrutinee_init: Box::new(scrutinee_expr),
                            scrutinee: scrutinee_local,
                            arms: vec![],
                            else_body: Some(hir::MatchElse {
                                binding: None,
                                body,
                            }),
                        },
                    });
                }
                Pattern::Ident(name) => {
                    let binding = register_named_local(fc, *name, scrutinee_ty.clone());
                    let body =
                        lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                    fc.leave_scope(mark);
                    return Ok(hir::Stmt {
                        span,
                        kind: hir::StmtKind::Match {
                            scrutinee_init: Box::new(scrutinee_expr),
                            scrutinee: scrutinee_local,
                            arms: vec![],
                            else_body: Some(hir::MatchElse {
                                binding: Some((binding, false)),
                                body,
                            }),
                        },
                    });
                }
                Pattern::VarIdent(name) => {
                    let binding = register_named_local(fc, *name, scrutinee_ty.clone());
                    let body =
                        lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                    fc.leave_scope(mark);
                    return Ok(hir::Stmt {
                        span,
                        kind: hir::StmtKind::Match {
                            scrutinee_init: Box::new(scrutinee_expr),
                            scrutinee: scrutinee_local,
                            arms: vec![],
                            else_body: Some(hir::MatchElse {
                                binding: Some((binding, true)),
                                body,
                            }),
                        },
                    });
                }
                _ => {}
            }

            // named variant pattern goes into a specific match arm
            let arm = match pattern {
                Pattern::EnumUnit { variant, .. } => {
                    let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                    let body =
                        lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                    hir::MatchArm {
                        variant: variant_idx,
                        bindings: vec![],
                        body,
                    }
                }

                Pattern::EnumTuple {
                    variant,
                    fields: subpatterns,
                    ..
                } => {
                    let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                    let field_types = ctx
                        .shared
                        .tcx
                        .enum_variant_field_types(enum_name, *variant, type_args)
                        .unwrap_or_default();
                    let mut bindings = vec![];
                    for (field_idx, subpat) in subpatterns.iter().enumerate() {
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        match &subpat.node {
                            Pattern::Ident(name) => {
                                let local = register_named_local(fc, *name, field_ty);
                                bindings.push(hir::MatchBinding {
                                    field_index: field_idx as u16,
                                    local,
                                    mutable: false,
                                });
                            }
                            Pattern::VarIdent(name) => {
                                let local = register_named_local(fc, *name, field_ty);
                                bindings.push(hir::MatchBinding {
                                    field_index: field_idx as u16,
                                    local,
                                    mutable: true,
                                });
                            }
                            _ => {}
                        }
                    }
                    let body =
                        lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                    hir::MatchArm {
                        variant: variant_idx,
                        bindings,
                        body,
                    }
                }

                Pattern::EnumStruct {
                    variant,
                    fields: field_patterns,
                    ..
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
                        let field_idx = field_names
                            .iter()
                            .position(|n| n == pat_field_name)
                            .unwrap_or(0);
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        match &subpat.node {
                            Pattern::Ident(binding_name) => {
                                let local = register_named_local(fc, *binding_name, field_ty);
                                bindings.push(hir::MatchBinding {
                                    field_index: field_idx as u16,
                                    local,
                                    mutable: false,
                                });
                            }
                            Pattern::VarIdent(binding_name) => {
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
                    let body =
                        lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                    hir::MatchArm {
                        variant: variant_idx,
                        bindings,
                        body,
                    }
                }

                other => {
                    fc.leave_scope(mark);
                    return Err(LowerError::UnsupportedExprKind {
                        span,
                        kind: format!(
                            "unsupported if-let pattern '{}' for enum type",
                            other.variant_name()
                        ),
                    });
                }
            };

            fc.leave_scope(mark);

            let else_body = match &if_let_node.node.else_block {
                Some(else_block) => hir::MatchElse {
                    binding: None,
                    body: lower_block(else_block, ctx, fc, is_func_body, ret_ty)?,
                },
                None => hir::MatchElse {
                    binding: None,
                    body: hir::Block { stmts: vec![] },
                },
            };

            Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::Match {
                    scrutinee_init: Box::new(scrutinee_expr),
                    scrutinee: scrutinee_local,
                    arms: vec![arm],
                    else_body: Some(else_body),
                },
            })
        }

        _ => {
            // non enum emit scrutinee Let, then build if/else from the pattern
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: scrutinee_local,
                    init: scrutinee_expr,
                },
            });

            let mark = fc.enter_scope();

            let (cond_opt, preamble) = build_non_enum_cond_preamble(
                &if_let_node.node.pattern.node,
                &scrutinee_ty,
                scrutinee_local,
                span,
                fc,
            )?;

            let mut then_block =
                lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
            for (i, stmt) in preamble.into_iter().enumerate() {
                then_block.stmts.insert(i, stmt);
            }

            fc.leave_scope(mark);

            let else_block = match &if_let_node.node.else_block {
                Some(b) => Some(lower_block(b, ctx, fc, is_func_body, ret_ty)?),
                None => None,
            };

            let cond = cond_opt
                .unwrap_or_else(|| hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(true)));

            Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond,
                    then_block,
                    else_block,
                },
            })
        }
    }
}

pub(super) fn lower_let_else(
    let_else_node: &ast::LetElseNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Stmt>, LowerError> {
    let scrutinee_expr = lower_expr(&let_else_node.node.value, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty.clone(),
    });

    let pattern = &let_else_node.node.pattern.node;

    // catch all patterns (Wildcard, Ident, VarIdent) always match regardless of scrutinee type
    match pattern {
        Pattern::Wildcard => {
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: scrutinee_local,
                    init: scrutinee_expr,
                },
            });
            return Ok(None);
        }
        Pattern::Ident(name) => {
            let binding_local = register_named_local(fc, *name, scrutinee_ty.clone());
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: scrutinee_local,
                    init: scrutinee_expr,
                },
            });
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: binding_local,
                    init: hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local),
                },
            });
            return Ok(None);
        }
        Pattern::VarIdent(name) => {
            let binding_local = register_named_local(fc, *name, scrutinee_ty.clone());
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: scrutinee_local,
                    init: scrutinee_expr,
                },
            });
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: binding_local,
                    init: hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local),
                },
            });
            return Ok(None);
        }
        _ => {}
    }

    match &scrutinee_ty {
        Type::Enum {
            name: enum_name,
            type_args,
        } => {
            let enum_name = *enum_name;

            let arm = match pattern {
                Pattern::EnumUnit { variant, .. } => {
                    let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                    hir::MatchArm {
                        variant: variant_idx,
                        bindings: vec![],
                        body: hir::Block { stmts: vec![] },
                    }
                }

                Pattern::EnumTuple {
                    variant,
                    fields: subpatterns,
                    ..
                } => {
                    let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                    let field_types = ctx
                        .shared
                        .tcx
                        .enum_variant_field_types(enum_name, *variant, type_args)
                        .unwrap_or_default();
                    let mut bindings = vec![];
                    for (field_idx, subpat) in subpatterns.iter().enumerate() {
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        match &subpat.node {
                            Pattern::Ident(name) => {
                                let local = register_named_local(fc, *name, field_ty);
                                bindings.push(hir::MatchBinding {
                                    field_index: field_idx as u16,
                                    local,
                                    mutable: false,
                                });
                            }
                            Pattern::VarIdent(name) => {
                                let local = register_named_local(fc, *name, field_ty);
                                bindings.push(hir::MatchBinding {
                                    field_index: field_idx as u16,
                                    local,
                                    mutable: true,
                                });
                            }
                            _ => {}
                        }
                    }
                    hir::MatchArm {
                        variant: variant_idx,
                        bindings,
                        body: hir::Block { stmts: vec![] },
                    }
                }

                Pattern::EnumStruct {
                    variant,
                    fields: field_patterns,
                    ..
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
                        let field_idx = field_names
                            .iter()
                            .position(|n| n == pat_field_name)
                            .unwrap_or(0);
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        match &subpat.node {
                            Pattern::Ident(binding_name) => {
                                let local = register_named_local(fc, *binding_name, field_ty);
                                bindings.push(hir::MatchBinding {
                                    field_index: field_idx as u16,
                                    local,
                                    mutable: false,
                                });
                            }
                            Pattern::VarIdent(binding_name) => {
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
                    hir::MatchArm {
                        variant: variant_idx,
                        bindings,
                        body: hir::Block { stmts: vec![] },
                    }
                }

                other => {
                    return Err(LowerError::UnsupportedExprKind {
                        span,
                        kind: format!(
                            "unsupported let-else pattern '{}' for enum type",
                            other.variant_name()
                        ),
                    });
                }
            };

            let else_body =
                lower_block(&let_else_node.node.else_block, ctx, fc, false, &Type::Void)?;

            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Match {
                    scrutinee_init: Box::new(scrutinee_expr),
                    scrutinee: scrutinee_local,
                    arms: vec![arm],
                    else_body: Some(hir::MatchElse {
                        binding: None,
                        body: else_body,
                    }),
                },
            });

            Ok(None)
        }

        _ => {
            // non enum emit scrutinee Let, build guard (inverted condition), then bindings
            out.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: scrutinee_local,
                    init: scrutinee_expr,
                },
            });

            let (cond_opt, preamble) =
                build_non_enum_cond_preamble(pattern, &scrutinee_ty, scrutinee_local, span, fc)?;

            // if there's a condition emit an inverted guard that runs the else_block on mismatch
            if let Some(cond) = cond_opt {
                let neg_cond = hir::Expr::new(
                    Type::Bool,
                    span,
                    hir::ExprKind::Unary {
                        op: UnaryOp::Not,
                        expr: Box::new(cond),
                    },
                );
                let else_body =
                    lower_block(&let_else_node.node.else_block, ctx, fc, false, &Type::Void)?;
                out.push(hir::Stmt {
                    span,
                    kind: hir::StmtKind::If {
                        cond: neg_cond,
                        then_block: else_body,
                        else_block: None,
                    },
                });
            }

            // emit binding extractions (always after the guard, since else_block diverges)
            for stmt in preamble {
                out.push(stmt);
            }

            Ok(None)
        }
    }
}
