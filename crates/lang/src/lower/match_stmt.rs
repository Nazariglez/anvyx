use internment::Intern;

use super::{
    FuncLower, LowerCtx, LowerError, lower_block, lower_expr, register_named_local,
    resolve_variant_index,
};
use crate::{
    ast::{self, BinaryOp, FloatSuffix, Ident, Lit, Pattern, Type, UnaryOp},
    hir,
    span::Span,
    typecheck::ConstValue,
};

fn needs_write_through(arms: &[hir::MatchArm], else_body: Option<&hir::MatchElse>) -> bool {
    let arms_have_var = arms.iter().any(|a| a.bindings.iter().any(|b| b.mutable));
    let else_has_var = else_body.is_some_and(|e| matches!(e.binding, Some((_, true))));
    arms_have_var || else_has_var
}

pub(super) fn alloc_write_through(
    scrutinee_expr: &hir::Expr,
    fc: &mut FuncLower,
) -> Option<hir::MatchWriteThrough> {
    let hir::ExprKind::Local(original) = scrutinee_expr.kind else {
        return None;
    };
    let ref_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: Type::Void,
        is_ref: false,
    });
    Some(hir::MatchWriteThrough {
        original,
        ref_local,
    })
}

struct MatchLowerCtx<'a> {
    match_node: &'a ast::MatchNode,
    span: Span,
    ctx: &'a LowerCtx<'a>,
    fc: &'a mut FuncLower,
    is_func_body: bool,
    ret_ty: &'a Type,
    scrutinee_expr: hir::Expr,
    scrutinee_local: hir::LocalId,
    scrutinee_ty: &'a Type,
}

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
        is_ref: false,
    });

    if let Type::Enum { name, type_args } = &scrutinee_ty {
        let mcx = MatchLowerCtx {
            match_node,
            span,
            ctx,
            fc,
            is_func_body,
            ret_ty,
            scrutinee_expr,
            scrutinee_local,
            scrutinee_ty: &scrutinee_ty,
        };
        lower_match_enum(mcx, *name, type_args)
    } else {
        let mcx = MatchLowerCtx {
            match_node,
            span,
            ctx,
            fc,
            is_func_body,
            ret_ty,
            scrutinee_expr,
            scrutinee_local,
            scrutinee_ty: &scrutinee_ty,
        };
        lower_match_non_enum(mcx, out)
    }
}

fn lower_match_enum(
    mcx: MatchLowerCtx,
    enum_name: Ident,
    type_args: &[Type],
) -> Result<hir::Stmt, LowerError> {
    let MatchLowerCtx {
        match_node,
        span,
        ctx,
        fc,
        is_func_body,
        ret_ty,
        scrutinee_expr,
        scrutinee_local,
        scrutinee_ty,
    } = mcx;
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

            Pattern::EnumUnit { variant, .. } | Pattern::InferredEnumUnit { variant } => {
                let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    guard: None,
                    body,
                });
            }

            Pattern::EnumTuple {
                variant,
                fields: subpatterns,
                ..
            }
            | Pattern::InferredEnumTuple {
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
                    guard: None,
                    body,
                });
            }

            Pattern::EnumStruct {
                variant,
                fields: field_patterns,
                ..
            }
            | Pattern::InferredEnumStruct {
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
                    guard: None,
                    body,
                });
            }

            Pattern::Nil => {
                let none_ident = Ident(Intern::new("None".to_string()));
                let variant_idx = resolve_variant_index(ctx, span, enum_name, none_ident)?;
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    guard: None,
                    body,
                });
            }

            Pattern::Optional(inner) => {
                let some_ident = Ident(Intern::new("Some".to_string()));
                let variant_idx = resolve_variant_index(ctx, span, enum_name, some_ident)?;
                let field_types = ctx
                    .shared
                    .tcx
                    .enum_variant_field_types(enum_name, some_ident, type_args)
                    .unwrap_or_default();
                let field_ty = field_types.first().cloned().unwrap_or(Type::Void);
                let mut bindings = vec![];
                match &inner.node {
                    Pattern::Ident(name) => {
                        let local = register_named_local(fc, *name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: 0,
                            local,
                            mutable: false,
                        });
                    }
                    Pattern::VarIdent(name) => {
                        let local = register_named_local(fc, *name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: 0,
                            local,
                            mutable: true,
                        });
                    }
                    Pattern::Wildcard => {}
                    _ => {
                        return Err(LowerError::UnsupportedExprKind {
                            span,
                            kind: "unsupported inner optional pattern".into(),
                        });
                    }
                }
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings,
                    guard: None,
                    body,
                });
            }

            Pattern::Or(alternatives) => {
                for alt in alternatives {
                    let alt_mark = fc.enter_scope();
                    match &alt.node {
                        Pattern::Wildcard => {
                            let body =
                                lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                            else_body = Some(hir::MatchElse {
                                binding: None,
                                body,
                            });
                        }
                        Pattern::Ident(name) => {
                            let binding_local =
                                register_named_local(fc, *name, scrutinee_ty.clone());
                            let body =
                                lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                            else_body = Some(hir::MatchElse {
                                binding: Some((binding_local, false)),
                                body,
                            });
                        }
                        Pattern::VarIdent(name) => {
                            let binding_local =
                                register_named_local(fc, *name, scrutinee_ty.clone());
                            let body =
                                lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                            else_body = Some(hir::MatchElse {
                                binding: Some((binding_local, true)),
                                body,
                            });
                        }
                        Pattern::EnumUnit { variant, .. }
                        | Pattern::InferredEnumUnit { variant } => {
                            let variant_idx =
                                resolve_variant_index(ctx, span, enum_name, *variant)?;
                            let body =
                                lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                            arms.push(hir::MatchArm {
                                variant: variant_idx,
                                bindings: vec![],
                                guard: None,
                                body,
                            });
                        }
                        Pattern::EnumTuple {
                            variant,
                            fields: subpatterns,
                            ..
                        }
                        | Pattern::InferredEnumTuple {
                            variant,
                            fields: subpatterns,
                        } => {
                            let variant_idx =
                                resolve_variant_index(ctx, span, enum_name, *variant)?;
                            let field_types = ctx
                                .shared
                                .tcx
                                .enum_variant_field_types(enum_name, *variant, type_args)
                                .unwrap_or_default();
                            let mut bindings = vec![];
                            for (field_idx, subpat) in subpatterns.iter().enumerate() {
                                match &subpat.node {
                                    Pattern::Ident(binding_name) => {
                                        let field_ty = field_types
                                            .get(field_idx)
                                            .cloned()
                                            .unwrap_or(Type::Void);
                                        let local =
                                            register_named_local(fc, *binding_name, field_ty);
                                        bindings.push(hir::MatchBinding {
                                            field_index: field_idx as u16,
                                            local,
                                            mutable: false,
                                        });
                                    }
                                    Pattern::VarIdent(binding_name) => {
                                        let field_ty = field_types
                                            .get(field_idx)
                                            .cloned()
                                            .unwrap_or(Type::Void);
                                        let local =
                                            register_named_local(fc, *binding_name, field_ty);
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
                                lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                            arms.push(hir::MatchArm {
                                variant: variant_idx,
                                bindings,
                                guard: None,
                                body,
                            });
                        }
                        Pattern::EnumStruct {
                            variant,
                            fields: field_patterns,
                            ..
                        }
                        | Pattern::InferredEnumStruct {
                            variant,
                            fields: field_patterns,
                            ..
                        } => {
                            let variant_idx =
                                resolve_variant_index(ctx, span, enum_name, *variant)?;
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
                                        let field_ty = field_types
                                            .get(field_idx)
                                            .cloned()
                                            .unwrap_or(Type::Void);
                                        let local =
                                            register_named_local(fc, *binding_name, field_ty);
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
                                        let field_ty = field_types
                                            .get(field_idx)
                                            .cloned()
                                            .unwrap_or(Type::Void);
                                        let local =
                                            register_named_local(fc, *binding_name, field_ty);
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
                                lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                            arms.push(hir::MatchArm {
                                variant: variant_idx,
                                bindings,
                                guard: None,
                                body,
                            });
                        }
                        Pattern::Nil => {
                            let none_ident = Ident(Intern::new("None".to_string()));
                            let variant_idx =
                                resolve_variant_index(ctx, span, enum_name, none_ident)?;
                            let body =
                                lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                            arms.push(hir::MatchArm {
                                variant: variant_idx,
                                bindings: vec![],
                                guard: None,
                                body,
                            });
                        }
                        Pattern::Optional(inner) => {
                            let some_ident = Ident(Intern::new("Some".to_string()));
                            let variant_idx =
                                resolve_variant_index(ctx, span, enum_name, some_ident)?;
                            let field_types = ctx
                                .shared
                                .tcx
                                .enum_variant_field_types(enum_name, some_ident, type_args)
                                .unwrap_or_default();
                            let field_ty = field_types.first().cloned().unwrap_or(Type::Void);
                            match &inner.node {
                                Pattern::Ident(name) => {
                                    let local = register_named_local(fc, *name, field_ty);
                                    let body = lower_arm_body(
                                        &arm.node.body,
                                        ctx,
                                        fc,
                                        is_func_body,
                                        ret_ty,
                                    )?;
                                    arms.push(hir::MatchArm {
                                        variant: variant_idx,
                                        bindings: vec![hir::MatchBinding {
                                            field_index: 0,
                                            local,
                                            mutable: false,
                                        }],
                                        guard: None,
                                        body,
                                    });
                                }
                                Pattern::VarIdent(name) => {
                                    let local = register_named_local(fc, *name, field_ty);
                                    let body = lower_arm_body(
                                        &arm.node.body,
                                        ctx,
                                        fc,
                                        is_func_body,
                                        ret_ty,
                                    )?;
                                    arms.push(hir::MatchArm {
                                        variant: variant_idx,
                                        bindings: vec![hir::MatchBinding {
                                            field_index: 0,
                                            local,
                                            mutable: true,
                                        }],
                                        guard: None,
                                        body,
                                    });
                                }
                                Pattern::Wildcard => {
                                    let body = lower_arm_body(
                                        &arm.node.body,
                                        ctx,
                                        fc,
                                        is_func_body,
                                        ret_ty,
                                    )?;
                                    arms.push(hir::MatchArm {
                                        variant: variant_idx,
                                        bindings: vec![],
                                        guard: None,
                                        body,
                                    });
                                }
                                Pattern::Lit(lit) => {
                                    let field_expr = hir::Expr::new(
                                        field_ty.clone(),
                                        span,
                                        hir::ExprKind::FieldGet {
                                            object: Box::new(hir::Expr::local(
                                                scrutinee_ty.clone(),
                                                span,
                                                scrutinee_local,
                                            )),
                                            index: 0,
                                        },
                                    );
                                    let rhs = build_rhs_from_lit(lit, span)?;
                                    let guard = hir::Expr::binary(
                                        Type::Bool,
                                        span,
                                        BinaryOp::Eq,
                                        field_expr,
                                        rhs,
                                    );
                                    let body = lower_arm_body(
                                        &arm.node.body,
                                        ctx,
                                        fc,
                                        is_func_body,
                                        ret_ty,
                                    )?;
                                    arms.push(hir::MatchArm {
                                        variant: variant_idx,
                                        bindings: vec![],
                                        guard: Some(guard),
                                        body,
                                    });
                                }
                                Pattern::Range {
                                    start,
                                    end,
                                    inclusive,
                                } => {
                                    let field_expr = hir::Expr::new(
                                        field_ty.clone(),
                                        span,
                                        hir::ExprKind::FieldGet {
                                            object: Box::new(hir::Expr::local(
                                                scrutinee_ty.clone(),
                                                span,
                                                scrutinee_local,
                                            )),
                                            index: 0,
                                        },
                                    );
                                    let ge = if let Some(s) = start {
                                        let rhs = build_rhs_from_lit(s, span)?;
                                        Some(hir::Expr::binary(
                                            Type::Bool,
                                            span,
                                            BinaryOp::GreaterThanEq,
                                            field_expr.clone(),
                                            rhs,
                                        ))
                                    } else {
                                        None
                                    };
                                    let lt = if let Some(e) = end {
                                        let rhs = build_rhs_from_lit(e, span)?;
                                        let op = if *inclusive {
                                            BinaryOp::LessThanEq
                                        } else {
                                            BinaryOp::LessThan
                                        };
                                        Some(hir::Expr::binary(
                                            Type::Bool,
                                            span,
                                            op,
                                            field_expr.clone(),
                                            rhs,
                                        ))
                                    } else {
                                        None
                                    };
                                    let guard = match (ge, lt) {
                                        (Some(ge), Some(lt)) => hir::Expr::binary(
                                            Type::Bool,
                                            span,
                                            BinaryOp::And,
                                            ge,
                                            lt,
                                        ),
                                        (Some(only), None) | (None, Some(only)) => only,
                                        (None, None) => {
                                            unreachable!(
                                                "range pattern must have at least one bound"
                                            )
                                        }
                                    };
                                    let body = lower_arm_body(
                                        &arm.node.body,
                                        ctx,
                                        fc,
                                        is_func_body,
                                        ret_ty,
                                    )?;
                                    arms.push(hir::MatchArm {
                                        variant: variant_idx,
                                        bindings: vec![],
                                        guard: Some(guard),
                                        body,
                                    });
                                }
                                _ => {
                                    return Err(LowerError::UnsupportedExprKind {
                                        span,
                                        kind: "unsupported inner optional pattern".into(),
                                    });
                                }
                            }
                        }
                        other => {
                            return Err(LowerError::UnsupportedExprKind {
                                span,
                                kind: format!(
                                    "unsupported pattern '{}' in or-pattern",
                                    other.variant_name()
                                ),
                            });
                        }
                    }
                    fc.leave_scope(alt_mark);
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

    let write_through = if needs_write_through(&arms, else_body.as_ref()) {
        alloc_write_through(&scrutinee_expr, fc)
    } else {
        None
    };

    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            write_through,
            arms,
            else_body,
        },
    })
}

fn lower_match_non_enum(
    mcx: MatchLowerCtx,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    let MatchLowerCtx {
        match_node,
        span,
        ctx,
        fc,
        is_func_body,
        ret_ty,
        scrutinee_expr,
        scrutinee_local,
        scrutinee_ty,
    } = mcx;
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

            Pattern::Range {
                start,
                end,
                inclusive,
            } => {
                let cond = build_range_cond(
                    scrutinee_ty,
                    scrutinee_local,
                    start.as_ref(),
                    end.as_ref(),
                    *inclusive,
                    span,
                )?;
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
                        Pattern::Ident(name) | Pattern::VarIdent(name) => {
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

            Pattern::Or(alternatives) => {
                let mut or_conditions: Vec<hir::Expr> = vec![];
                let mut is_catchall = false;
                let mut preamble: Vec<hir::Stmt> = vec![];

                for alt in alternatives {
                    match &alt.node {
                        Pattern::Lit(lit) => {
                            let cond = build_lit_cond(scrutinee_ty, scrutinee_local, lit, span)?;
                            or_conditions.push(cond);
                        }
                        Pattern::Range {
                            start,
                            end,
                            inclusive,
                        } => {
                            let cond = build_range_cond(
                                scrutinee_ty,
                                scrutinee_local,
                                start.as_ref(),
                                end.as_ref(),
                                *inclusive,
                                span,
                            )?;
                            or_conditions.push(cond);
                        }
                        Pattern::Ident(name) => {
                            let span_key = (alt.span.start, alt.span.end);
                            if let Some(cv) = ctx.shared.tcx.const_pattern_values.get(&span_key) {
                                let cond =
                                    build_const_cond(scrutinee_ty, scrutinee_local, cv, span);
                                or_conditions.push(cond);
                            } else {
                                is_catchall = true;
                                let local = register_named_local(fc, *name, scrutinee_ty.clone());
                                preamble.push(hir::Stmt {
                                    span,
                                    kind: hir::StmtKind::Let {
                                        local,
                                        init: hir::Expr::local(
                                            scrutinee_ty.clone(),
                                            span,
                                            scrutinee_local,
                                        ),
                                    },
                                });
                            }
                        }
                        Pattern::VarIdent(name) => {
                            is_catchall = true;
                            let local = register_named_local(fc, *name, scrutinee_ty.clone());
                            preamble.push(hir::Stmt {
                                span,
                                kind: hir::StmtKind::Let {
                                    local,
                                    init: hir::Expr::local(
                                        scrutinee_ty.clone(),
                                        span,
                                        scrutinee_local,
                                    ),
                                },
                            });
                        }
                        Pattern::Wildcard => {
                            is_catchall = true;
                        }
                        other => {
                            return Err(LowerError::UnsupportedExprKind {
                                span,
                                kind: format!(
                                    "unsupported pattern '{}' in or-pattern",
                                    other.variant_name()
                                ),
                            });
                        }
                    }
                }

                let mut body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                for (i, stmt) in preamble.into_iter().enumerate() {
                    body.stmts.insert(i, stmt);
                }

                if is_catchall {
                    else_block = Some(body);
                } else if !or_conditions.is_empty() {
                    let cond = or_conditions
                        .into_iter()
                        .reduce(|acc, c| hir::Expr::binary(Type::Bool, span, BinaryOp::Or, acc, c))
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
        Lit::Float { value, suffix } => match suffix {
            Some(FloatSuffix::D) => {
                hir::Expr::new(Type::Double, span, hir::ExprKind::Double(*value))
            }
            Some(FloatSuffix::F) | None => {
                hir::Expr::new(Type::Float, span, hir::ExprKind::Float(*value as f32))
            }
        },
        Lit::Nil => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: "nil literal pattern not supported in match".into(),
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

fn build_range_cond(
    scrutinee_ty: &Type,
    scrutinee_local: hir::LocalId,
    start: Option<&Lit>,
    end: Option<&Lit>,
    inclusive: bool,
    span: Span,
) -> Result<hir::Expr, LowerError> {
    let scr = hir::Expr::local(scrutinee_ty.clone(), span, scrutinee_local);

    let ge = if let Some(s) = start {
        let rhs = build_rhs_from_lit(s, span)?;
        Some(hir::Expr::binary(
            Type::Bool,
            span,
            BinaryOp::GreaterThanEq,
            scr.clone(),
            rhs,
        ))
    } else {
        None
    };

    let lt = if let Some(e) = end {
        let rhs = build_rhs_from_lit(e, span)?;
        let op = if inclusive {
            BinaryOp::LessThanEq
        } else {
            BinaryOp::LessThan
        };
        Some(hir::Expr::binary(Type::Bool, span, op, scr.clone(), rhs))
    } else {
        None
    };

    Ok(match (ge, lt) {
        (Some(ge), Some(lt)) => hir::Expr::binary(Type::Bool, span, BinaryOp::And, ge, lt),
        (Some(only), None) | (None, Some(only)) => only,
        (None, None) => unreachable!("range pattern must have at least one bound"),
    })
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

        Pattern::Range {
            start,
            end,
            inclusive,
        } => {
            let cond = build_range_cond(
                scrutinee_ty,
                scrutinee_local,
                start.as_ref(),
                end.as_ref(),
                *inclusive,
                span,
            )?;
            Ok((Some(cond), vec![]))
        }

        Pattern::Ident(name) | Pattern::VarIdent(name) => {
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
                    Pattern::Ident(name) | Pattern::VarIdent(name) => {
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
        is_ref: false,
    });

    if let Type::Enum {
        name: enum_name,
        type_args,
    } = &scrutinee_ty
    {
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
                        write_through: None,
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
                        write_through: None,
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
                let write_through = alloc_write_through(&scrutinee_expr, fc);
                return Ok(hir::Stmt {
                    span,
                    kind: hir::StmtKind::Match {
                        scrutinee_init: Box::new(scrutinee_expr),
                        scrutinee: scrutinee_local,
                        write_through,
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
            Pattern::EnumUnit { variant, .. } | Pattern::InferredEnumUnit { variant } => {
                let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                let body =
                    lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    guard: None,
                    body,
                }
            }

            Pattern::EnumTuple {
                variant,
                fields: subpatterns,
                ..
            }
            | Pattern::InferredEnumTuple {
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
                    guard: None,
                    body,
                }
            }

            Pattern::EnumStruct {
                variant,
                fields: field_patterns,
                ..
            }
            | Pattern::InferredEnumStruct {
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
                    guard: None,
                    body,
                }
            }

            Pattern::Nil => {
                let none_ident = Ident(Intern::new("None".to_string()));
                let variant_idx = resolve_variant_index(ctx, span, enum_name, none_ident)?;
                let body =
                    lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    guard: None,
                    body,
                }
            }

            Pattern::Optional(inner) => {
                let some_ident = Ident(Intern::new("Some".to_string()));
                let variant_idx = resolve_variant_index(ctx, span, enum_name, some_ident)?;
                let field_types = ctx
                    .shared
                    .tcx
                    .enum_variant_field_types(enum_name, some_ident, type_args)
                    .unwrap_or_default();
                let field_ty = field_types.first().cloned().unwrap_or(Type::Void);
                let mut bindings = vec![];
                match &inner.node {
                    Pattern::Ident(name) => {
                        let local = register_named_local(fc, *name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: 0,
                            local,
                            mutable: false,
                        });
                    }
                    Pattern::VarIdent(name) => {
                        let local = register_named_local(fc, *name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: 0,
                            local,
                            mutable: true,
                        });
                    }
                    Pattern::Wildcard => {}
                    _ => {
                        fc.leave_scope(mark);
                        return Err(LowerError::UnsupportedExprKind {
                            span,
                            kind: "unsupported inner optional pattern".into(),
                        });
                    }
                }
                let body =
                    lower_block(&if_let_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
                hir::MatchArm {
                    variant: variant_idx,
                    bindings,
                    guard: None,
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

        let has_var = arm.bindings.iter().any(|b| b.mutable);
        let write_through = if has_var {
            alloc_write_through(&scrutinee_expr, fc)
        } else {
            None
        };

        Ok(hir::Stmt {
            span,
            kind: hir::StmtKind::Match {
                scrutinee_init: Box::new(scrutinee_expr),
                scrutinee: scrutinee_local,
                write_through,
                arms: vec![arm],
                else_body: Some(else_body),
            },
        })
    } else {
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

        let cond =
            cond_opt.unwrap_or_else(|| hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(true)));

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
        is_ref: false,
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
        Pattern::Ident(name) | Pattern::VarIdent(name) => {
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

    if let Type::Enum {
        name: enum_name,
        type_args,
    } = &scrutinee_ty
    {
        let enum_name = *enum_name;

        let arm = match pattern {
            Pattern::EnumUnit { variant, .. } | Pattern::InferredEnumUnit { variant } => {
                let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
                hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    guard: None,
                    body: hir::Block { stmts: vec![] },
                }
            }

            Pattern::EnumTuple {
                variant,
                fields: subpatterns,
                ..
            }
            | Pattern::InferredEnumTuple {
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
                    guard: None,
                    body: hir::Block { stmts: vec![] },
                }
            }

            Pattern::EnumStruct {
                variant,
                fields: field_patterns,
                ..
            }
            | Pattern::InferredEnumStruct {
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
                    guard: None,
                    body: hir::Block { stmts: vec![] },
                }
            }

            Pattern::Nil => {
                let none_ident = Ident(Intern::new("None".to_string()));
                let variant_idx = resolve_variant_index(ctx, span, enum_name, none_ident)?;
                hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    guard: None,
                    body: hir::Block { stmts: vec![] },
                }
            }

            Pattern::Optional(inner) => {
                let some_ident = Ident(Intern::new("Some".to_string()));
                let variant_idx = resolve_variant_index(ctx, span, enum_name, some_ident)?;
                let field_types = ctx
                    .shared
                    .tcx
                    .enum_variant_field_types(enum_name, some_ident, type_args)
                    .unwrap_or_default();
                let field_ty = field_types.first().cloned().unwrap_or(Type::Void);
                let mut bindings = vec![];
                match &inner.node {
                    Pattern::Ident(name) => {
                        let local = register_named_local(fc, *name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: 0,
                            local,
                            mutable: false,
                        });
                    }
                    Pattern::VarIdent(name) => {
                        let local = register_named_local(fc, *name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: 0,
                            local,
                            mutable: true,
                        });
                    }
                    Pattern::Wildcard => {}
                    _ => {
                        return Err(LowerError::UnsupportedExprKind {
                            span,
                            kind: "unsupported inner optional pattern".into(),
                        });
                    }
                }
                hir::MatchArm {
                    variant: variant_idx,
                    bindings,
                    guard: None,
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

        let else_body = lower_block(&let_else_node.node.else_block, ctx, fc, false, &Type::Void)?;

        out.push(hir::Stmt {
            span,
            kind: hir::StmtKind::Match {
                scrutinee_init: Box::new(scrutinee_expr),
                scrutinee: scrutinee_local,
                write_through: None,
                arms: vec![arm],
                else_body: Some(hir::MatchElse {
                    binding: None,
                    body: else_body,
                }),
            },
        });

        Ok(None)
    } else {
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

pub(super) fn lower_while_let(
    while_let_node: &ast::WhileLetNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Stmt>, LowerError> {
    let scrutinee_expr = lower_expr(&while_let_node.node.value, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty.clone(),
        is_ref: false,
    });

    let cond = hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(true));

    let body = if let Type::Enum { .. } = scrutinee_ty {
        let inner = lower_while_let_enum(
            while_let_node,
            span,
            ctx,
            fc,
            scrutinee_expr,
            scrutinee_local,
        )?;
        hir::Block { stmts: vec![inner] }
    } else {
        let mut body_stmts = vec![];
        let inner = lower_while_let_non_enum(
            while_let_node,
            span,
            ctx,
            fc,
            scrutinee_expr,
            scrutinee_local,
            &mut body_stmts,
        )?;
        body_stmts.push(inner);
        hir::Block { stmts: body_stmts }
    };

    Ok(Some(hir::Stmt {
        span,
        kind: hir::StmtKind::While { cond, body },
    }))
}

fn lower_while_let_enum(
    while_let_node: &ast::WhileLetNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    scrutinee_expr: hir::Expr,
    scrutinee_local: hir::LocalId,
) -> Result<hir::Stmt, LowerError> {
    let break_body = hir::Block {
        stmts: vec![hir::Stmt {
            span,
            kind: hir::StmtKind::Break,
        }],
    };
    let scrutinee_ty = scrutinee_expr.ty.clone();
    let Type::Enum {
        name: enum_name,
        type_args,
    } = &scrutinee_ty
    else {
        return Err(LowerError::UnsupportedExprKind {
            span,
            kind: "while-let enum scrutinee is not an enum type".into(),
        });
    };
    let enum_name = *enum_name;
    let type_args = type_args.clone();
    let mark = fc.enter_scope();
    let pattern = &while_let_node.node.pattern.node;

    match pattern {
        Pattern::Wildcard => {
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            fc.leave_scope(mark);
            return Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::Match {
                    scrutinee_init: Box::new(scrutinee_expr),
                    scrutinee: scrutinee_local,
                    write_through: None,
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
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            fc.leave_scope(mark);
            return Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::Match {
                    scrutinee_init: Box::new(scrutinee_expr),
                    scrutinee: scrutinee_local,
                    write_through: None,
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
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            fc.leave_scope(mark);
            let write_through = alloc_write_through(&scrutinee_expr, fc);
            return Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::Match {
                    scrutinee_init: Box::new(scrutinee_expr),
                    scrutinee: scrutinee_local,
                    write_through,
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

    let arm = match pattern {
        Pattern::EnumUnit { variant, .. } | Pattern::InferredEnumUnit { variant } => {
            let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            hir::MatchArm {
                variant: variant_idx,
                bindings: vec![],
                guard: None,
                body,
            }
        }

        Pattern::EnumTuple {
            variant,
            fields: subpatterns,
            ..
        }
        | Pattern::InferredEnumTuple {
            variant,
            fields: subpatterns,
        } => {
            let variant_idx = resolve_variant_index(ctx, span, enum_name, *variant)?;
            let field_types = ctx
                .shared
                .tcx
                .enum_variant_field_types(enum_name, *variant, &type_args)
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
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            hir::MatchArm {
                variant: variant_idx,
                bindings,
                guard: None,
                body,
            }
        }

        Pattern::EnumStruct {
            variant,
            fields: field_patterns,
            ..
        }
        | Pattern::InferredEnumStruct {
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
                .enum_variant_field_types(enum_name, *variant, &type_args)
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
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            hir::MatchArm {
                variant: variant_idx,
                bindings,
                guard: None,
                body,
            }
        }

        Pattern::Nil => {
            let none_ident = Ident(Intern::new("None".to_string()));
            let variant_idx = resolve_variant_index(ctx, span, enum_name, none_ident)?;
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            hir::MatchArm {
                variant: variant_idx,
                bindings: vec![],
                guard: None,
                body,
            }
        }

        Pattern::Optional(inner) => {
            let some_ident = Ident(Intern::new("Some".to_string()));
            let variant_idx = resolve_variant_index(ctx, span, enum_name, some_ident)?;
            let field_types = ctx
                .shared
                .tcx
                .enum_variant_field_types(enum_name, some_ident, &type_args)
                .unwrap_or_default();
            let field_ty = field_types.first().cloned().unwrap_or(Type::Void);
            let mut bindings = vec![];
            match &inner.node {
                Pattern::Ident(name) => {
                    let local = register_named_local(fc, *name, field_ty);
                    bindings.push(hir::MatchBinding {
                        field_index: 0,
                        local,
                        mutable: false,
                    });
                }
                Pattern::VarIdent(name) => {
                    let local = register_named_local(fc, *name, field_ty);
                    bindings.push(hir::MatchBinding {
                        field_index: 0,
                        local,
                        mutable: true,
                    });
                }
                Pattern::Wildcard => {}
                _ => {
                    fc.leave_scope(mark);
                    return Err(LowerError::UnsupportedExprKind {
                        span,
                        kind: "unsupported inner optional pattern".into(),
                    });
                }
            }
            let body = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
            hir::MatchArm {
                variant: variant_idx,
                bindings,
                guard: None,
                body,
            }
        }

        other => {
            fc.leave_scope(mark);
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: format!(
                    "unsupported while-let pattern '{}' for enum type",
                    other.variant_name()
                ),
            });
        }
    };

    fc.leave_scope(mark);

    let has_var = arm.bindings.iter().any(|b| b.mutable);
    let write_through = if has_var {
        alloc_write_through(&scrutinee_expr, fc)
    } else {
        None
    };

    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            write_through,
            arms: vec![arm],
            else_body: Some(hir::MatchElse {
                binding: None,
                body: break_body,
            }),
        },
    })
}

fn lower_while_let_non_enum(
    while_let_node: &ast::WhileLetNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    scrutinee_expr: hir::Expr,
    scrutinee_local: hir::LocalId,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    let break_body = hir::Block {
        stmts: vec![hir::Stmt {
            span,
            kind: hir::StmtKind::Break,
        }],
    };
    let scrutinee_ty = scrutinee_expr.ty.clone();
    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Let {
            local: scrutinee_local,
            init: scrutinee_expr,
        },
    });

    let mark = fc.enter_scope();

    let (cond_opt, preamble) = build_non_enum_cond_preamble(
        &while_let_node.node.pattern.node,
        &scrutinee_ty,
        scrutinee_local,
        span,
        fc,
    )?;

    let mut then_block = lower_block(&while_let_node.node.body, ctx, fc, false, &Type::Void)?;
    for (i, stmt) in preamble.into_iter().enumerate() {
        then_block.stmts.insert(i, stmt);
    }

    fc.leave_scope(mark);

    let cond =
        cond_opt.unwrap_or_else(|| hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(true)));

    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::If {
            cond,
            then_block,
            else_block: Some(break_body),
        },
    })
}
