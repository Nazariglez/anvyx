use crate::ast::{self, BinaryOp, Ident, Pattern, Stmt, StringPart, Type};
use crate::hir;
use crate::span::Span;
use internment::Intern;

use super::{
    FuncLower, LowerCtx, LowerError,
    alloc_and_bind, lower_assign, lower_expr, lower_for, lower_if_let, lower_let_else,
    lower_match_stmts, register_named_local,
};

pub(super) fn lower_block(
    block: &ast::BlockNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
) -> Result<hir::Block, LowerError> {
    let scope_mark = if !is_func_body {
        Some(fc.enter_scope())
    } else {
        None
    };

    let mut stmts = vec![];

    for stmt_node in &block.node.stmts {
        if let Some(hir_stmt) = lower_stmt(stmt_node, ctx, fc, &mut stmts)? {
            stmts.push(hir_stmt);
        }
    }

    if let Some(tail_expr) = &block.node.tail {
        let s = lower_expr_as_stmt(
            tail_expr,
            tail_expr.span,
            ctx,
            fc,
            is_func_body,
            ret_ty,
            &mut stmts,
        )?;
        stmts.push(s);
    }

    if let Some(mark) = scope_mark {
        fc.leave_scope(mark);
    }

    Ok(hir::Block { stmts })
}

fn lower_expr_as_stmt(
    expr_node: &ast::ExprNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    match &expr_node.node.kind {
        ast::ExprKind::If(if_node) => lower_if(if_node, span, ctx, fc, is_func_body, ret_ty, out),
        ast::ExprKind::Assign(assign_node) => lower_assign(assign_node, span, ctx, fc, out),
        ast::ExprKind::Match(match_node) => {
            lower_match_stmts(match_node, span, ctx, fc, is_func_body, ret_ty, out)
        }
        ast::ExprKind::IfLet(if_let_node) => {
            lower_if_let(if_let_node, span, ctx, fc, is_func_body, ret_ty, out)
        }
        _ => {
            let hir_expr = lower_expr(expr_node, ctx, fc, out)?;
            let kind = if is_func_body && !ret_ty.is_void() {
                hir::StmtKind::Return(Some(hir_expr))
            } else {
                hir::StmtKind::Expr(hir_expr)
            };
            Ok(hir::Stmt { span, kind })
        }
    }
}

fn lower_stmt(
    stmt_node: &ast::StmtNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Stmt>, LowerError> {
    let span = stmt_node.span;

    match &stmt_node.node {
        Stmt::Binding(binding_node) => {
            let binding = &binding_node.node;

            match &binding.pattern.node {
                Pattern::Ident(name) => {
                    let name = *name;
                    let ty = ctx.expr_type(binding.value.node.id, span)?;

                    let local_id = hir::LocalId(fc.locals.len() as u32);
                    let init = lower_expr(&binding.value, ctx, fc, out)?;
                    fc.locals.push(hir::Local {
                        name: Some(name),
                        ty,
                    });
                    fc.bind_local(name, local_id);

                    Ok(Some(hir::Stmt {
                        span,
                        kind: hir::StmtKind::Let {
                            local: local_id,
                            init,
                        },
                    }))
                }
                Pattern::Struct {
                    name: type_name,
                    fields,
                } => {
                    let type_name = *type_name;
                    let rhs_ty = ctx.expr_type(binding.value.node.id, span)?;
                    let rhs_expr = lower_expr(&binding.value, ctx, fc, out)?;
                    let scrutinee_local = alloc_and_bind(fc, span, out, rhs_ty.clone(), rhs_expr);

                    match &rhs_ty {
                        Type::Struct { .. } => {
                            for (field_name, subpat) in fields {
                                let Pattern::Ident(binding_name) = &subpat.node else {
                                    return Err(LowerError::UnsupportedPattern {
                                        span: subpat.span,
                                    });
                                };

                                let field_index = ctx
                                    .shared
                                    .tcx
                                    .struct_field_index(type_name, *field_name)
                                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                                        span,
                                        kind: format!(
                                            "unknown field '{field_name}' on struct '{type_name}'"
                                        ),
                                    })? as u16;

                                let field_ty = ctx
                                    .shared
                                    .tcx
                                    .struct_field_type(type_name, *field_name)
                                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                                        span,
                                        kind: format!(
                                            "unknown field type for '{field_name}' on '{type_name}'"
                                        ),
                                    })?;

                                let local_id =
                                    register_named_local(fc, *binding_name, field_ty.clone());

                                out.push(hir::Stmt {
                                    span,
                                    kind: hir::StmtKind::Let {
                                        local: local_id,
                                        init: hir::Expr {
                                            ty: field_ty,
                                            span,
                                            kind: hir::ExprKind::FieldGet {
                                                object: Box::new(hir::Expr {
                                                    ty: rhs_ty.clone(),
                                                    span,
                                                    kind: hir::ExprKind::Local(scrutinee_local),
                                                }),
                                                index: field_index,
                                            },
                                        },
                                    },
                                });
                            }
                        }
                        Type::Extern { name: extern_name } => {
                            let extern_name = *extern_name;
                            for (field_name, subpat) in fields {
                                let Pattern::Ident(binding_name) = &subpat.node else {
                                    return Err(LowerError::UnsupportedPattern {
                                        span: subpat.span,
                                    });
                                };

                                let qualified = Ident(Intern::new(format!(
                                    "{extern_name}::__get_{field_name}"
                                )));
                                let extern_id = *ctx.shared.externs.get(&qualified).ok_or_else(|| {
                                    LowerError::UnsupportedExprKind {
                                        span,
                                        kind: format!("unknown extern field getter '{qualified}'"),
                                    }
                                })?;

                                let field_ty = ctx
                                    .shared
                                    .tcx
                                    .get_extern_type(extern_name)
                                    .and_then(|def| def.fields.get(field_name))
                                    .map(|f| f.ty.clone())
                                    .unwrap_or(Type::Void);

                                let local_id =
                                    register_named_local(fc, *binding_name, field_ty.clone());

                                out.push(hir::Stmt {
                                    span,
                                    kind: hir::StmtKind::Let {
                                        local: local_id,
                                        init: hir::Expr {
                                            ty: field_ty,
                                            span,
                                            kind: hir::ExprKind::CallExtern {
                                                extern_id,
                                                args: vec![hir::Expr {
                                                    ty: rhs_ty.clone(),
                                                    span,
                                                    kind: hir::ExprKind::Local(scrutinee_local),
                                                }],
                                            },
                                        },
                                    },
                                });
                            }
                        }
                        other => {
                            return Err(LowerError::UnsupportedExprKind {
                                span,
                                kind: format!("struct destructure on unsupported type '{other}'"),
                            });
                        }
                    }

                    Ok(None)
                }
                _ => Err(LowerError::UnsupportedPattern {
                    span: binding.pattern.span,
                }),
            }
        }

        Stmt::Expr(expr_node) => Ok(Some(lower_expr_as_stmt(
            expr_node,
            span,
            ctx,
            fc,
            false,
            &Type::Void,
            out,
        )?)),

        Stmt::Return(return_node) => {
            let value = match &return_node.node.value {
                Some(expr) => Some(lower_expr(expr, ctx, fc, out)?),
                None => None,
            };
            Ok(Some(hir::Stmt {
                span,
                kind: hir::StmtKind::Return(value),
            }))
        }

        Stmt::While(while_node) => {
            let cond = lower_expr(&while_node.node.cond, ctx, fc, out)?;
            let body = lower_block(&while_node.node.body, ctx, fc, false, &Type::Void)?;
            Ok(Some(hir::Stmt {
                span,
                kind: hir::StmtKind::While { cond, body },
            }))
        }

        Stmt::Break => Ok(Some(hir::Stmt {
            span,
            kind: hir::StmtKind::Break,
        })),

        Stmt::Continue => Ok(Some(hir::Stmt {
            span,
            kind: hir::StmtKind::Continue,
        })),

        Stmt::For(for_node) => lower_for(for_node, span, ctx, fc, out),

        Stmt::ExternFunc(_) => Ok(None),
        Stmt::ExternType(_) => Ok(None),
        Stmt::Import(_) => Ok(None),

        Stmt::Func(_) => Err(LowerError::UnsupportedStmtKind {
            span,
            kind: "nested function".to_string(),
        }),

        Stmt::Struct(_) => Ok(None),

        Stmt::Enum(_) => Ok(None),

        Stmt::LetElse(let_else_node) => lower_let_else(let_else_node, span, ctx, fc, out),

        Stmt::Const(_) => Ok(None),
    }
}

fn lower_if(
    if_node: &ast::IfNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Stmt, LowerError> {
    let cond = lower_expr(&if_node.node.cond, ctx, fc, out)?;
    let then_block = lower_block(&if_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
    let else_block = match &if_node.node.else_block {
        Some(b) => Some(lower_block(b, ctx, fc, is_func_body, ret_ty)?),
        None => None,
    };
    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::If {
            cond,
            then_block,
            else_block,
        },
    })
}

pub(super) fn lower_string_interp(
    parts: &[StringPart],
    span: Span,
    ty: &Type,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::ExprKind, LowerError> {
    let mut hir_parts: Vec<hir::Expr> = vec![];

    for part in parts {
        let expr = match part {
            StringPart::Text(s) => hir::Expr {
                ty: Type::String,
                span,
                kind: hir::ExprKind::String(s.clone()),
            },
            StringPart::Expr(e) => {
                let lowered = lower_expr(e, ctx, fc, out)?;
                if lowered.ty == Type::String {
                    lowered
                } else {
                    hir::Expr {
                        ty: Type::String,
                        span,
                        kind: hir::ExprKind::ToString(Box::new(lowered)),
                    }
                }
            }
        };
        hir_parts.push(expr);
    }

    match hir_parts.len() {
        0 => Ok(hir::ExprKind::String(String::new())),
        1 => Ok(hir_parts.remove(0).kind),
        _ => {
            let mut iter = hir_parts.into_iter();
            let first = iter.next().unwrap();
            let folded = iter.fold(first, |acc, rhs| {
                hir::Expr::binary(ty.clone(), span, BinaryOp::Add, acc, rhs)
            });
            Ok(folded.kind)
        }
    }
}
