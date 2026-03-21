use crate::ast::{self, Pattern, Type};
use crate::hir;
use crate::span::Span;

use super::{
    FuncLower, LowerCtx, LowerError,
    lower_block, lower_expr, register_named_local, resolve_variant_index,
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

    let (enum_name, type_args) = match &scrutinee_ty {
        Type::Enum { name, type_args } => (*name, type_args.clone()),
        other => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: format!("match on non-enum type '{other}'"),
            });
        }
    };

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
                    binding: Some(binding_local),
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
                    .enum_variant_field_types(enum_name, *variant, &type_args)
                    .unwrap_or_default();

                let mut bindings = vec![];
                for (field_idx, subpat) in subpatterns.iter().enumerate() {
                    if let Pattern::Ident(binding_name) = &subpat.node {
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        let local = register_named_local(fc, *binding_name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: field_idx as u16,
                            local,
                        });
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
                    if let Pattern::Ident(binding_name) = &subpat.node {
                        let field_idx = field_names
                            .iter()
                            .position(|n| n == pat_field_name)
                            .unwrap_or(0);
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        let local = register_named_local(fc, *binding_name, field_ty);
                        bindings.push(hir::MatchBinding {
                            field_index: field_idx as u16,
                            local,
                        });
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
