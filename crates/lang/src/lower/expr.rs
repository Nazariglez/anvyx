use std::collections::HashMap;

use crate::{
    ast::{self, ArrayLen, BinaryOp, Ident, Lit, Type, UnaryOp},
    builtin::Builtin,
    hir,
    span::Span,
    typecheck::{ConstValue, FieldDefault},
};
use internment::Intern;

use super::{
    FuncLower, LowerCtx, LowerError, alloc_and_bind, alloc_assign_temp, emit_counter_increment,
    extern_binary_op_key, extern_unary_op_key, lower_block, lower_block_to_target, lower_if_let,
    lower_match_stmts, lower_string_interp, mangle_generic_name, mangle_method_spec_name,
    register_named_local, resolve_enum_type_id, resolve_struct_type_id, resolve_variant_index,
};

fn lower_args(
    args: &[ast::ExprNode],
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Vec<hir::Expr>, LowerError> {
    args.iter()
        .map(|arg| lower_expr(arg, ctx, fc, out))
        .collect()
}

fn const_value_to_hir_expr(val: &ConstValue, span: Span) -> hir::Expr {
    let (ty, kind) = match val {
        ConstValue::Int(n) => (Type::Int, hir::ExprKind::Int(*n)),
        ConstValue::Float(f) => (Type::Float, hir::ExprKind::Float(*f)),
        ConstValue::Double(d) => (Type::Double, hir::ExprKind::Double(*d)),
        ConstValue::Bool(b) => (Type::Bool, hir::ExprKind::Bool(*b)),
        ConstValue::String(s) => (Type::String, hir::ExprKind::String(s.clone())),
        ConstValue::Nil => (Type::Infer, hir::ExprKind::Nil),
    };
    hir::Expr::new(ty, span, kind)
}

pub(super) fn lower_expr(
    ast_expr: &ast::ExprNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let span = ast_expr.span;
    let ty = ctx.expr_type(ast_expr.node.id, span)?;

    let kind = match &ast_expr.node.kind {
        ast::ExprKind::Ident(name) => {
            if let Some(cv) = ctx.shared.tcx.const_values.get(&ast_expr.node.id) {
                match cv {
                    ConstValue::Int(n) => hir::ExprKind::Int(*n),
                    ConstValue::Float(f) => hir::ExprKind::Float(*f),
                    ConstValue::Double(d) => hir::ExprKind::Double(*d),
                    ConstValue::Bool(b) => hir::ExprKind::Bool(*b),
                    ConstValue::String(s) => hir::ExprKind::String(s.clone()),
                    ConstValue::Nil => hir::ExprKind::Nil,
                }
            } else if let Some(&local_id) = fc.local_map.get(name) {
                hir::ExprKind::Local(local_id)
            } else if let Some(&func_id) = ctx.shared.funcs.get(name) {
                hir::ExprKind::CreateClosure {
                    func: func_id,
                    captures: vec![],
                }
            } else {
                return Err(LowerError::UnknownLocal { name: *name, span });
            }
        }

        ast::ExprKind::Lit(lit) => match lit {
            Lit::Int(v) => hir::ExprKind::Int(*v),
            Lit::Float { value: v, .. } => match &ty {
                Type::Float => hir::ExprKind::Float(*v as f32),
                Type::Double => hir::ExprKind::Double(*v),
                _ => unreachable!("float literal resolved to non-float type"),
            },
            Lit::Bool(v) => hir::ExprKind::Bool(*v),
            Lit::String(v) => hir::ExprKind::String(v.clone()),
            Lit::Nil => hir::ExprKind::Nil,
        },

        ast::ExprKind::Unary(u) => {
            let inner = lower_expr(&u.node.expr, ctx, fc, out)?;
            if let Some(kind) = try_lower_extern_unary_op(u.node.op, &inner, span, ctx)? {
                kind
            } else {
                hir::ExprKind::Unary {
                    op: u.node.op,
                    expr: Box::new(inner),
                }
            }
        }

        ast::ExprKind::Binary(b) => {
            if b.node.op == BinaryOp::Coalesce {
                return lower_coalesce_expr(b, ty, span, ctx, fc, out);
            }
            let lhs = lower_expr(&b.node.left, ctx, fc, out)?;
            let rhs = lower_expr(&b.node.right, ctx, fc, out)?;
            if let Some(kind) = try_lower_extern_binary_op(b.node.op, &lhs, &rhs, span, ctx)? {
                kind
            } else {
                hir::ExprKind::Binary {
                    op: b.node.op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }
            }
        }

        ast::ExprKind::StringInterp(parts) => lower_string_interp(parts, span, &ty, ctx, fc, out)?,

        ast::ExprKind::Call(c) => {
            return lower_call_expr(c, ty, span, ctx, fc, out);
        }

        ast::ExprKind::StructLiteral(lit) => {
            return lower_struct_literal_expr(lit, ty, span, ctx, fc, out);
        }

        ast::ExprKind::Tuple(elements) => {
            let mut lowered = vec![];
            for el in elements {
                lowered.push(lower_expr(el, ctx, fc, out)?);
            }
            hir::ExprKind::TupleLiteral { elements: lowered }
        }

        ast::ExprKind::Field(field_access) => {
            if let Some(cv) = ctx.shared.tcx.const_values.get(&ast_expr.node.id) {
                match cv {
                    ConstValue::Int(n) => hir::ExprKind::Int(*n),
                    ConstValue::Float(f) => hir::ExprKind::Float(*f),
                    ConstValue::Double(d) => hir::ExprKind::Double(*d),
                    ConstValue::Bool(b) => hir::ExprKind::Bool(*b),
                    ConstValue::String(s) => hir::ExprKind::String(s.clone()),
                    ConstValue::Nil => hir::ExprKind::Nil,
                }
            } else {
                return lower_field_expr(field_access, ty, span, ctx, fc, out);
            }
        }

        ast::ExprKind::TupleIndex(tuple_idx) => {
            let tuple = lower_expr(&tuple_idx.node.target, ctx, fc, out)?;
            hir::ExprKind::TupleIndex {
                tuple: Box::new(tuple),
                index: tuple_idx.node.index as u16,
            }
        }

        ast::ExprKind::ArrayLiteral(lit) => {
            let mut elements = vec![];
            for e in &lit.node.elements {
                elements.push(lower_expr(e, ctx, fc, out)?);
            }

            match &ty {
                Type::Array { .. } => hir::ExprKind::ArrayLiteral { elements },
                Type::List { .. } => hir::ExprKind::ListLiteral { elements },
                other => {
                    return Err(LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("array literal with unexpected type '{other}'"),
                    });
                }
            }
        }

        ast::ExprKind::ArrayFill(fill) => {
            let value = lower_expr(&fill.node.value, ctx, fc, out)?;

            let len = match &ty {
                Type::Array {
                    len: ArrayLen::Fixed(n),
                    ..
                } => *n,
                Type::List { .. } => match &fill.node.len.node.kind {
                    ast::ExprKind::Lit(Lit::Int(n)) => *n as usize,
                    _ => match ctx.shared.tcx.const_values.get(&fill.node.len.node.id) {
                        Some(ConstValue::Int(n)) => *n as usize,
                        _ => {
                            return Err(LowerError::UnsupportedExprKind {
                                span,
                                kind: "list fill with non-literal length".to_string(),
                            });
                        }
                    },
                },
                other => {
                    return Err(LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("array fill with unexpected type '{other}'"),
                    });
                }
            };

            match &ty {
                Type::Array { .. } => hir::ExprKind::ArrayFill {
                    value: Box::new(value),
                    len,
                },
                Type::List { .. } => hir::ExprKind::ListFill {
                    value: Box::new(value),
                    len,
                },
                _ => unreachable!(),
            }
        }

        ast::ExprKind::Index(index_node) => {
            if index_node.node.safe {
                return lower_safe_index_expr(index_node, ty, span, ctx, fc, out);
            }
            let target = lower_expr(&index_node.node.target, ctx, fc, out)?;
            let index = lower_expr(&index_node.node.index, ctx, fc, out)?;
            hir::ExprKind::IndexGet {
                target: Box::new(target),
                index: Box::new(index),
            }
        }

        ast::ExprKind::MapLiteral(lit) => {
            let mut entries = vec![];
            for (k, v) in &lit.node.entries {
                let key = lower_expr(k, ctx, fc, out)?;
                let value = lower_expr(v, ctx, fc, out)?;
                entries.push((key, value));
            }

            hir::ExprKind::MapLiteral { entries }
        }

        ast::ExprKind::Cast(cast_node) => {
            let inner = lower_expr(&cast_node.node.expr, ctx, fc, out)?;
            if inner.ty == ty {
                return Ok(inner);
            }
            hir::ExprKind::Cast(Box::new(inner))
        }

        ast::ExprKind::If(if_node) => {
            return lower_if_as_expr(if_node, ty, span, ctx, fc, out);
        }

        ast::ExprKind::IfLet(if_let_node) => {
            return lower_if_let_as_expr(if_let_node, ty, span, ctx, fc, out);
        }

        ast::ExprKind::Match(match_node) => {
            return lower_match_as_expr(match_node, ty, span, ctx, fc, out);
        }

        ast::ExprKind::Block(block_node) => {
            return lower_block_as_expr(block_node, ty, span, ctx, fc, out);
        }

        ast::ExprKind::NamedTuple(fields) => {
            let mut lowered = vec![];
            for (_label, expr) in fields {
                lowered.push(lower_expr(expr, ctx, fc, out)?);
            }
            hir::ExprKind::TupleLiteral { elements: lowered }
        }

        ast::ExprKind::Lambda(lambda) => {
            return lower_lambda(lambda, ast_expr.node.id, ty, span, ctx, fc);
        }

        other => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: other.variant_name().to_string(),
            });
        }
    };

    Ok(hir::Expr::new(ty, span, kind))
}

fn try_lower_extern_binary_op(
    op: BinaryOp,
    lhs: &hir::Expr,
    rhs: &hir::Expr,
    span: Span,
    ctx: &LowerCtx,
) -> Result<Option<hir::ExprKind>, LowerError> {
    let (actual_op, wrap_not) = if op == BinaryOp::NotEq {
        (BinaryOp::Eq, true)
    } else {
        (op, false)
    };

    // try left dispatch, lhs is extern
    let result = if let Type::Extern { name } = &lhs.ty {
        let extern_def = ctx.shared.tcx.get_extern_type(*name);
        extern_def.and_then(|def| {
            def.operators
                .iter()
                .find(|o| o.op == actual_op && !o.self_on_right && o.other_ty == rhs.ty)
                .map(|o| {
                    (
                        extern_binary_op_key(*name, actual_op, &o.other_ty, false),
                        o.ret.clone(),
                    )
                })
        })
    } else {
        None
    };

    // if left dispatch didn't match, try right dispatch, rhs is extern
    let result = result.or_else(|| {
        if let Type::Extern { name } = &rhs.ty {
            let extern_def = ctx.shared.tcx.get_extern_type(*name);
            extern_def.and_then(|def| {
                def.operators
                    .iter()
                    .find(|o| o.op == actual_op && o.self_on_right && o.other_ty == lhs.ty)
                    .map(|o| {
                        (
                            extern_binary_op_key(*name, actual_op, &o.other_ty, true),
                            o.ret.clone(),
                        )
                    })
            })
        } else {
            None
        }
    });

    let Some((key, eq_ret_ty)) = result else {
        return Ok(None);
    };

    let extern_id =
        *ctx.shared
            .externs
            .get(&key)
            .ok_or_else(|| LowerError::UnsupportedExprKind {
                span,
                kind: format!("missing extern ID for operator key '{key}'"),
            })?;

    let call_expr = hir::ExprKind::CallExtern {
        extern_id,
        args: vec![lhs.clone(), rhs.clone()],
    };

    if wrap_not {
        Ok(Some(hir::ExprKind::Unary {
            op: UnaryOp::Not,
            expr: Box::new(hir::Expr::new(eq_ret_ty, span, call_expr)),
        }))
    } else {
        Ok(Some(call_expr))
    }
}

fn try_lower_extern_unary_op(
    op: UnaryOp,
    inner: &hir::Expr,
    span: Span,
    ctx: &LowerCtx,
) -> Result<Option<hir::ExprKind>, LowerError> {
    let Type::Extern { name } = &inner.ty else {
        return Ok(None);
    };

    let Some(def) = ctx.shared.tcx.get_extern_type(*name) else {
        return Ok(None);
    };

    let Some(_op_def) = def.unary_operators.iter().find(|o| o.op == op) else {
        return Ok(None);
    };

    let key = extern_unary_op_key(*name, op);
    let extern_id =
        *ctx.shared
            .externs
            .get(&key)
            .ok_or_else(|| LowerError::UnsupportedExprKind {
                span,
                kind: format!("missing extern ID for unary operator key '{key}'"),
            })?;

    Ok(Some(hir::ExprKind::CallExtern {
        extern_id,
        args: vec![inner.clone()],
    }))
}

fn lower_coalesce_expr(
    b: &ast::BinaryNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    // desugar `a ?? b` into a match on Option
    let scrutinee_expr = lower_expr(&b.node.left, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();
    let inner_ty = scrutinee_ty.option_inner().cloned().unwrap_or(ty.clone());

    let enum_name = match &scrutinee_ty {
        Type::Enum { name, .. } => *name,
        _ => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: "coalesce on non-optional type".to_string(),
            });
        }
    };

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty,
    });

    let result_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: ty.clone(),
    });

    let some_variant = ctx
        .shared
        .tcx
        .enum_variant_index(enum_name, Ident(Intern::new("Some".to_string())))
        .unwrap_or(1);
    let inner_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: inner_ty.clone(),
    });

    let some_arm = hir::MatchArm {
        variant: some_variant,
        bindings: vec![hir::MatchBinding {
            field_index: 0,
            local: inner_local,
            mutable: false,
        }],
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: hir::Expr::local(inner_ty, span, inner_local),
                },
            }],
        },
    };

    let rhs_expr = lower_expr(&b.node.right, ctx, fc, out)?;
    let none_else = hir::MatchElse {
        binding: None,
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: rhs_expr,
                },
            }],
        },
    };

    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            arms: vec![some_arm],
            else_body: Some(none_else),
        },
    });

    Ok(hir::Expr::local(ty, span, result_local))
}

fn lower_safe_field_expr(
    field_access: &ast::FieldAccessNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let scrutinee_expr = lower_expr(&field_access.node.target, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let (enum_name, inner_ty) = match &scrutinee_ty {
        Type::Enum { name, .. } => {
            let inner = scrutinee_ty.option_inner().cloned().ok_or_else(|| {
                LowerError::UnsupportedExprKind {
                    span,
                    kind: "safe field access on non-optional type".to_string(),
                }
            })?;
            (*name, inner)
        }
        _ => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: "safe field access on non-optional type".to_string(),
            });
        }
    };

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty,
    });

    let result_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: ty.clone(),
    });

    let some_variant = ctx
        .shared
        .tcx
        .enum_variant_index(enum_name, Ident(Intern::new("Some".to_string())))
        .unwrap_or(1);

    let inner_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: inner_ty.clone(),
    });

    let field_name = field_access.node.field;
    let field_index = match &inner_ty {
        Type::Struct { name, .. } => ctx
            .shared
            .tcx
            .struct_field_index(*name, field_name)
            .ok_or_else(|| LowerError::UnsupportedExprKind {
                span,
                kind: format!("unknown field '{field_name}' on struct '{name}'"),
            })? as u16,
        Type::DataRef { name, .. } => ctx
            .shared
            .tcx
            .struct_field_index(*name, field_name)
            .ok_or_else(|| LowerError::UnsupportedExprKind {
                span,
                kind: format!("unknown field '{field_name}' on dataref '{name}'"),
            })? as u16,
        Type::NamedTuple(fields) => fields
            .iter()
            .position(|(label, _)| *label == field_name)
            .ok_or_else(|| LowerError::UnsupportedExprKind {
                span,
                kind: format!("unknown field '{field_name}' on named tuple"),
            })? as u16,
        other => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: format!("safe field access on unsupported inner type '{other}'"),
            });
        }
    };

    let field_ty = ty.option_inner().cloned().unwrap_or(ty.clone());

    let field_get = hir::Expr::new(
        field_ty.clone(),
        span,
        hir::ExprKind::FieldGet {
            object: Box::new(hir::Expr::local(inner_ty.clone(), span, inner_local)),
            index: field_index,
        },
    );

    let option_type_id = resolve_enum_type_id(ctx, span, enum_name)?;
    let wrapped = hir::Expr::new(
        ty.clone(),
        span,
        hir::ExprKind::EnumLiteral {
            type_id: option_type_id,
            variant: some_variant,
            fields: vec![field_get],
        },
    );

    let some_arm = hir::MatchArm {
        variant: some_variant,
        bindings: vec![hir::MatchBinding {
            field_index: 0,
            local: inner_local,
            mutable: false,
        }],
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: wrapped,
                },
            }],
        },
    };

    let none_expr = hir::Expr::new(ty.clone(), span, hir::ExprKind::Nil);
    let none_else = hir::MatchElse {
        binding: None,
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: none_expr,
                },
            }],
        },
    };

    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            arms: vec![some_arm],
            else_body: Some(none_else),
        },
    });

    Ok(hir::Expr::local(ty, span, result_local))
}

fn lower_safe_index_expr(
    index_node: &ast::IndexNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let scrutinee_expr = lower_expr(&index_node.node.target, ctx, fc, out)?;
    let index_expr = lower_expr(&index_node.node.index, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let (enum_name, inner_ty) = match &scrutinee_ty {
        Type::Enum { name, .. } => {
            let inner = scrutinee_ty.option_inner().cloned().ok_or_else(|| {
                LowerError::UnsupportedExprKind {
                    span,
                    kind: "safe index on non-optional type".to_string(),
                }
            })?;
            (*name, inner)
        }
        _ => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: "safe index on non-optional type".to_string(),
            });
        }
    };

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty,
    });

    let result_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: ty.clone(),
    });

    let some_variant = ctx
        .shared
        .tcx
        .enum_variant_index(enum_name, Ident(Intern::new("Some".to_string())))
        .unwrap_or(1);

    let inner_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: inner_ty.clone(),
    });

    let elem_ty = ty.option_inner().cloned().unwrap_or(ty.clone());

    let index_get = hir::Expr::new(
        elem_ty,
        span,
        hir::ExprKind::IndexGet {
            target: Box::new(hir::Expr::local(inner_ty.clone(), span, inner_local)),
            index: Box::new(index_expr),
        },
    );

    let option_type_id = resolve_enum_type_id(ctx, span, enum_name)?;
    let wrapped = hir::Expr::new(
        ty.clone(),
        span,
        hir::ExprKind::EnumLiteral {
            type_id: option_type_id,
            variant: some_variant,
            fields: vec![index_get],
        },
    );

    let some_arm = hir::MatchArm {
        variant: some_variant,
        bindings: vec![hir::MatchBinding {
            field_index: 0,
            local: inner_local,
            mutable: false,
        }],
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: wrapped,
                },
            }],
        },
    };

    let none_expr = hir::Expr::new(ty.clone(), span, hir::ExprKind::Nil);
    let none_else = hir::MatchElse {
        binding: None,
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: none_expr,
                },
            }],
        },
    };

    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            arms: vec![some_arm],
            else_body: Some(none_else),
        },
    });

    Ok(hir::Expr::local(ty, span, result_local))
}

fn lower_safe_call_expr(
    c: &ast::CallNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let ast::ExprKind::Field(field) = &c.node.func.node.kind else {
        return Err(LowerError::UnsupportedExprKind {
            span,
            kind: "safe call on non-method expression".to_string(),
        });
    };

    let scrutinee_expr = lower_expr(&field.node.target, ctx, fc, out)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let (enum_name, inner_ty) = match &scrutinee_ty {
        Type::Enum { name, .. } => {
            let inner = scrutinee_ty.option_inner().cloned().ok_or_else(|| {
                LowerError::UnsupportedExprKind {
                    span,
                    kind: "safe call on non-optional type".to_string(),
                }
            })?;
            (*name, inner)
        }
        _ => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: "safe call on non-optional type".to_string(),
            });
        }
    };

    let pre_args = lower_args(&c.node.args, ctx, fc, out)?;

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty,
    });

    let result_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: ty.clone(),
    });

    let some_variant = ctx
        .shared
        .tcx
        .enum_variant_index(enum_name, Ident(Intern::new("Some".to_string())))
        .unwrap_or(1);

    let inner_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: inner_ty.clone(),
    });

    let option_type_id = resolve_enum_type_id(ctx, span, enum_name)?;
    let method_name = field.node.field;
    let inner_result_ty = ty.option_inner().cloned().unwrap_or(ty.clone());

    let call_expr =
        if let Some(internal_name) = ctx.shared.tcx.extend_call_target(c.node.func.node.id) {
            let &func_id = ctx
                .shared
                .funcs
                .get(&internal_name)
                .ok_or(LowerError::UnknownFunc {
                    name: internal_name,
                    span,
                })?;
            let receiver = hir::Expr::local(inner_ty.clone(), span, inner_local);
            let mut args = vec![receiver];
            args.extend(pre_args);
            hir::Expr::new(
                inner_result_ty,
                span,
                hir::ExprKind::Call {
                    func: func_id,
                    args,
                },
            )
        } else if let Type::Extern { name: type_name } = &inner_ty {
            let qualified = Ident(Intern::new(format!("{type_name}::{method_name}")));
            let &extern_id = ctx.shared.externs.get(&qualified).ok_or_else(|| {
                LowerError::UnsupportedExprKind {
                    span,
                    kind: format!("unknown extern method '{method_name}' on '{type_name}'"),
                }
            })?;
            let receiver = hir::Expr::local(inner_ty.clone(), span, inner_local);
            let mut args = vec![receiver];
            args.extend(pre_args);
            hir::Expr::new(
                inner_result_ty,
                span,
                hir::ExprKind::CallExtern { extern_id, args },
            )
        } else if let Type::Struct {
            name: struct_name,
            type_args,
        } = &inner_ty
        {
            let mangled = mangle_method_spec_name(*struct_name, method_name, type_args);
            let &func_id =
                ctx.shared
                    .funcs
                    .get(&mangled)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown method '{method_name}' on struct '{struct_name}'"),
                    })?;
            let receiver = hir::Expr::local(inner_ty.clone(), span, inner_local);
            let mut args = vec![receiver];
            args.extend(pre_args);
            let defaults = ctx
                .shared
                .tcx
                .method_param_defaults(*struct_name, method_name);
            for val in defaults.iter().skip(c.node.args.len()).flatten() {
                args.push(const_value_to_hir_expr(val, span));
            }
            hir::Expr::new(
                inner_result_ty,
                span,
                hir::ExprKind::Call {
                    func: func_id,
                    args,
                },
            )
        } else {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: format!(
                    "safe method call '?.{method_name}()' on unsupported inner type '{inner_ty}'"
                ),
            });
        };

    let wrapped = hir::Expr::new(
        ty.clone(),
        span,
        hir::ExprKind::EnumLiteral {
            type_id: option_type_id,
            variant: some_variant,
            fields: vec![call_expr],
        },
    );

    let some_arm = hir::MatchArm {
        variant: some_variant,
        bindings: vec![hir::MatchBinding {
            field_index: 0,
            local: inner_local,
            mutable: false,
        }],
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: wrapped,
                },
            }],
        },
    };

    let none_expr = hir::Expr::new(ty.clone(), span, hir::ExprKind::Nil);
    let none_else = hir::MatchElse {
        binding: None,
        body: hir::Block {
            stmts: vec![hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: result_local,
                    value: none_expr,
                },
            }],
        },
    };

    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            arms: vec![some_arm],
            else_body: Some(none_else),
        },
    });

    Ok(hir::Expr::local(ty, span, result_local))
}

fn lower_if_as_expr(
    if_node: &ast::IfNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let result_local = alloc_assign_temp(fc, ty.clone());
    let cond = lower_expr(&if_node.node.cond, ctx, fc, out)?;
    let then_block = lower_block_to_target(&if_node.node.then_block, result_local, ctx, fc)?;
    let else_block = match &if_node.node.else_block {
        Some(b) => Some(lower_block_to_target(b, result_local, ctx, fc)?),
        None => None,
    };
    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::If {
            cond,
            then_block,
            else_block,
        },
    });
    Ok(hir::Expr::local(ty, span, result_local))
}

fn lower_if_let_as_expr(
    if_let_node: &ast::IfLetNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let result_local = alloc_assign_temp(fc, ty.clone());
    let stmt = lower_if_let(if_let_node, span, ctx, fc, false, &Type::Void, out)?;
    let modified = inject_assign_target(stmt, result_local);
    out.push(modified);
    Ok(hir::Expr::local(ty, span, result_local))
}

fn lower_match_as_expr(
    match_node: &ast::MatchNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let result_local = alloc_assign_temp(fc, ty.clone());
    let stmt = lower_match_stmts(match_node, span, ctx, fc, false, &Type::Void, out)?;
    let modified = inject_assign_target(stmt, result_local);
    out.push(modified);
    Ok(hir::Expr::local(ty, span, result_local))
}

fn lower_block_as_expr(
    block_node: &ast::BlockNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let result_local = alloc_assign_temp(fc, ty.clone());
    let block = lower_block_to_target(block_node, result_local, ctx, fc)?;
    out.extend(block.stmts);
    Ok(hir::Expr::local(ty, span, result_local))
}

fn inject_assign_target(stmt: hir::Stmt, target: hir::LocalId) -> hir::Stmt {
    let kind = match stmt.kind {
        hir::StmtKind::Expr(expr) => hir::StmtKind::Assign {
            local: target,
            value: expr,
        },
        hir::StmtKind::If {
            cond,
            then_block,
            else_block,
        } => hir::StmtKind::If {
            cond,
            then_block: inject_assign_target_block(then_block, target),
            else_block: else_block.map(|b| inject_assign_target_block(b, target)),
        },
        hir::StmtKind::Match {
            scrutinee_init,
            scrutinee,
            arms,
            else_body,
        } => {
            let arms = arms
                .into_iter()
                .map(|arm| hir::MatchArm {
                    variant: arm.variant,
                    bindings: arm.bindings,
                    body: inject_assign_target_block(arm.body, target),
                })
                .collect();
            let else_body = else_body.map(|e| hir::MatchElse {
                binding: e.binding,
                body: inject_assign_target_block(e.body, target),
            });
            hir::StmtKind::Match {
                scrutinee_init,
                scrutinee,
                arms,
                else_body,
            }
        }
        other => other,
    };
    hir::Stmt {
        span: stmt.span,
        kind,
    }
}

fn inject_assign_target_block(mut block: hir::Block, target: hir::LocalId) -> hir::Block {
    if let Some(last) = block.stmts.pop() {
        block.stmts.push(inject_assign_target(last, target));
    }
    block
}

fn lower_call_expr(
    c: &ast::CallNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    if let ast::ExprKind::Field(field) = &c.node.func.node.kind
        && (field.node.safe || c.node.safe)
    {
        return lower_safe_call_expr(c, ty, span, ctx, fc, out);
    }
    if let Some(internal_name) = ctx.shared.tcx.extend_call_target(c.node.func.node.id) {
        return lower_extend_call(c, internal_name, ty, span, ctx, fc, out);
    }
    if let Some(expr) = try_lower_method_call(c, &ty, span, ctx, fc, out)? {
        return Ok(expr);
    }
    if let Some(expr) = try_lower_enum_constructor(c, &ty, span, ctx, fc, out)? {
        return Ok(expr);
    }
    let is_named_call = match &c.node.func.node.kind {
        ast::ExprKind::Ident(name) => {
            ctx.shared.funcs.contains_key(name)
                || ctx.shared.externs.contains_key(name)
                || Builtin::from_name(name.0.as_ref()).is_some()
                || ctx.shared.tcx.call_type_args(c.node.func.node.id).is_some()
        }
        ast::ExprKind::Field(_) => {
            let field_ty = ctx.expr_type(c.node.func.node.id, span)?;
            !field_ty.is_func()
        }
        _ => false,
    };
    if !is_named_call {
        let callee_ty = ctx.expr_type(c.node.func.node.id, span)?;
        if callee_ty.is_func() {
            return lower_closure_call(c, ty, span, ctx, fc, out);
        }
    }
    let kind = lower_direct_call(c, span, ctx, fc, out)?;
    Ok(hir::Expr::new(ty, span, kind))
}

fn lower_closure_call(
    c: &ast::CallNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let callee = lower_expr(&c.node.func, ctx, fc, out)?;
    let args = lower_args(&c.node.args, ctx, fc, out)?;
    Ok(hir::Expr::new(
        ty,
        span,
        hir::ExprKind::CallClosure {
            callee: Box::new(callee),
            args,
        },
    ))
}

fn lower_lambda(
    lambda: &ast::LambdaNode,
    expr_id: ast::ExprId,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &FuncLower,
) -> Result<hir::Expr, LowerError> {
    let (func_params, func_ret) = match &ty {
        Type::Func { params, ret } => (params.clone(), (**ret).clone()),
        _ => unreachable!("lambda must have Type::Func"),
    };

    let captures = ctx
        .shared
        .tcx
        .lambda_captures
        .get(&expr_id)
        .cloned()
        .unwrap_or_default();

    let lambda_func_id = ctx.shared.alloc_func_id();

    let mut lambda_fc = FuncLower {
        locals: vec![],
        local_map: HashMap::new(),
        scope_log: vec![],
    };

    for (name, capture_ty) in &captures {
        register_named_local(&mut lambda_fc, *name, capture_ty.clone());
    }

    for (i, param) in lambda.node.params.iter().enumerate() {
        let param_ty = param.ty.clone().unwrap_or_else(|| func_params[i].clone());
        register_named_local(&mut lambda_fc, param.name, param_ty);
    }

    let params_len = lambda_fc.locals.len() as u32;

    let body = match &lambda.node.body.node.kind {
        ast::ExprKind::Block(block_node) => {
            lower_block(block_node, ctx, &mut lambda_fc, true, &func_ret)?
        }
        _ => {
            let mut body_stmts = vec![];
            let body_expr = lower_expr(&lambda.node.body, ctx, &mut lambda_fc, &mut body_stmts)?;
            let stmt_kind = if !func_ret.is_void() {
                hir::StmtKind::Return(Some(body_expr))
            } else {
                hir::StmtKind::Expr(body_expr)
            };
            body_stmts.push(hir::Stmt {
                span,
                kind: stmt_kind,
            });
            hir::Block { stmts: body_stmts }
        }
    };

    let lambda_name = Ident(Intern::new(format!("__lambda_{}", lambda_func_id.0)));

    ctx.shared.register_lambda_func(hir::Func {
        id: lambda_func_id,
        name: lambda_name,
        locals: lambda_fc.locals,
        params_len,
        ret: func_ret,
        body,
        span,
    });

    let capture_exprs: Vec<hir::Expr> = captures
        .iter()
        .map(|(name, capture_ty)| {
            let local_id = *fc
                .local_map
                .get(name)
                .expect("captured variable must exist in enclosing scope");
            hir::Expr::new(capture_ty.clone(), span, hir::ExprKind::Local(local_id))
        })
        .collect();

    Ok(hir::Expr::new(
        ty,
        span,
        hir::ExprKind::CreateClosure {
            func: lambda_func_id,
            captures: capture_exprs,
        },
    ))
}

fn lower_extend_call(
    c: &ast::CallNode,
    internal_name: Ident,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    let &func_id = ctx
        .shared
        .funcs
        .get(&internal_name)
        .ok_or(LowerError::UnknownFunc {
            name: internal_name,
            span,
        })?;

    let args = match &c.node.func.node.kind {
        ast::ExprKind::Field(field) => {
            let is_qualified = matches!(
                &field.node.target.node.kind,
                ast::ExprKind::Ident(name) if ctx.shared.tcx.is_module_name(*name)
            );
            if is_qualified {
                lower_args(&c.node.args, ctx, fc, out)?
            } else {
                let receiver = lower_expr(&field.node.target, ctx, fc, out)?;
                let mut args = vec![receiver];
                args.extend(lower_args(&c.node.args, ctx, fc, out)?);
                args
            }
        }
        _ => lower_args(&c.node.args, ctx, fc, out)?,
    };

    Ok(hir::Expr::new(
        ty,
        span,
        hir::ExprKind::Call {
            func: func_id,
            args,
        },
    ))
}

fn lower_map_desugar(
    c: &ast::CallNode,
    method: &str,
    key_ty: &Type,
    value_ty: &Type,
    result_ty: &Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Expr>, LowerError> {
    let ast::ExprKind::Field(field) = &c.node.func.node.kind else {
        unreachable!()
    };
    let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
        unreachable!()
    };

    let map_ty = Type::Map {
        key: Box::new(key_ty.clone()),
        value: Box::new(value_ty.clone()),
    };

    let mark = fc.enter_scope();

    let m_expr = lower_expr(&field.node.target, ctx, fc, out)?;
    let m_local = alloc_and_bind(fc, span, out, map_ty.clone(), m_expr);

    let len_expr = hir::Expr::new(
        Type::Int,
        span,
        hir::ExprKind::MapLen {
            map: Box::new(hir::Expr::local(map_ty.clone(), span, m_local)),
        },
    );
    let len_local = alloc_and_bind(fc, span, out, Type::Int, len_expr);
    let step_local = alloc_and_bind(fc, span, out, Type::Int, hir::Expr::int_lit(span, 1));
    let i_local = alloc_and_bind(fc, span, out, Type::Int, hir::Expr::int_lit(span, 0));

    let cond = hir::Expr::binary(
        Type::Bool,
        span,
        BinaryOp::LessThan,
        hir::Expr::local(Type::Int, span, i_local),
        hir::Expr::local(Type::Int, span, len_local),
    );

    let mut body_stmts = vec![];

    let entry_ty = Type::Tuple(vec![key_ty.clone(), value_ty.clone()]);

    let make_entry_expr = || {
        hir::Expr::new(
            entry_ty.clone(),
            span,
            hir::ExprKind::MapEntryAt {
                map: Box::new(hir::Expr::local(map_ty.clone(), span, m_local)),
                index: Box::new(hir::Expr::local(Type::Int, span, i_local)),
            },
        )
    };

    let result_expr = match method {
        "map_values" => {
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                result_ty.clone(),
                hir::Expr::new(
                    result_ty.clone(),
                    span,
                    hir::ExprKind::MapLiteral { entries: vec![] },
                ),
            );
            let entry_local = alloc_and_bind(
                fc,
                span,
                &mut body_stmts,
                entry_ty.clone(),
                make_entry_expr(),
            );
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, value_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: hir::Expr::new(
                        value_ty.clone(),
                        span,
                        hir::ExprKind::TupleIndex {
                            tuple: Box::new(hir::Expr::local(entry_ty.clone(), span, entry_local)),
                            index: 1,
                        },
                    ),
                },
            });
            let new_value_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            let key_expr = hir::Expr::new(
                key_ty.clone(),
                span,
                hir::ExprKind::TupleIndex {
                    tuple: Box::new(hir::Expr::local(entry_ty.clone(), span, entry_local)),
                    index: 0,
                },
            );
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Expr(hir::Expr::new(
                    Type::Void,
                    span,
                    hir::ExprKind::CollectionMut {
                        object: result_local,
                        method: hir::CollectionMethod::MapInsert,
                        args: vec![key_expr, new_value_expr],
                    },
                )),
            });
            hir::Expr::local(result_ty.clone(), span, result_local)
        }
        "filter" => {
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                result_ty.clone(),
                hir::Expr::new(
                    result_ty.clone(),
                    span,
                    hir::ExprKind::MapLiteral { entries: vec![] },
                ),
            );
            let entry_local = alloc_and_bind(
                fc,
                span,
                &mut body_stmts,
                entry_ty.clone(),
                make_entry_expr(),
            );
            let key_param_name = lambda.node.params[0].name;
            let key_param_local = register_named_local(fc, key_param_name, key_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: key_param_local,
                    init: hir::Expr::new(
                        key_ty.clone(),
                        span,
                        hir::ExprKind::TupleIndex {
                            tuple: Box::new(hir::Expr::local(entry_ty.clone(), span, entry_local)),
                            index: 0,
                        },
                    ),
                },
            });
            let value_param_name = lambda.node.params[1].name;
            let value_param_local = register_named_local(fc, value_param_name, value_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: value_param_local,
                    init: hir::Expr::new(
                        value_ty.clone(),
                        span,
                        hir::ExprKind::TupleIndex {
                            tuple: Box::new(hir::Expr::local(entry_ty.clone(), span, entry_local)),
                            index: 1,
                        },
                    ),
                },
            });
            let pred_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            let insert_stmt = hir::Stmt {
                span,
                kind: hir::StmtKind::Expr(hir::Expr::new(
                    Type::Void,
                    span,
                    hir::ExprKind::CollectionMut {
                        object: result_local,
                        method: hir::CollectionMethod::MapInsert,
                        args: vec![
                            hir::Expr::local(key_ty.clone(), span, key_param_local),
                            hir::Expr::local(value_ty.clone(), span, value_param_local),
                        ],
                    },
                )),
            };
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond: pred_expr,
                    then_block: hir::Block {
                        stmts: vec![insert_stmt],
                    },
                    else_block: None,
                },
            });
            hir::Expr::local(result_ty.clone(), span, result_local)
        }
        _ => unreachable!(),
    };

    emit_counter_increment(
        &mut body_stmts,
        span,
        i_local,
        &Type::Int,
        hir::Expr::local(Type::Int, span, step_local),
        BinaryOp::Add,
    );

    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::While {
            cond,
            body: hir::Block { stmts: body_stmts },
        },
    });

    fc.leave_scope(mark);

    Ok(Some(result_expr))
}

fn lower_list_desugar(
    c: &ast::CallNode,
    method: &str,
    elem_ty: &Type,
    result_ty: &Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Expr>, LowerError> {
    let ast::ExprKind::Field(field) = &c.node.func.node.kind else {
        unreachable!()
    };

    let collection_ty = Type::List {
        elem: Box::new(elem_ty.clone()),
    };

    let mark = fc.enter_scope();

    let xs_expr = lower_expr(&field.node.target, ctx, fc, out)?;
    let xs_local = alloc_and_bind(fc, span, out, collection_ty.clone(), xs_expr);

    let len_expr = hir::Expr::new(
        Type::Int,
        span,
        hir::ExprKind::CollectionLen {
            collection: Box::new(hir::Expr::local(collection_ty.clone(), span, xs_local)),
        },
    );
    let len_local = alloc_and_bind(fc, span, out, Type::Int, len_expr);
    let step_local = alloc_and_bind(fc, span, out, Type::Int, hir::Expr::int_lit(span, 1));
    let i_local = alloc_and_bind(fc, span, out, Type::Int, hir::Expr::int_lit(span, 0));

    let cond = hir::Expr::binary(
        Type::Bool,
        span,
        BinaryOp::LessThan,
        hir::Expr::local(Type::Int, span, i_local),
        hir::Expr::local(Type::Int, span, len_local),
    );

    let mut body_stmts = vec![];

    let make_elem_expr = || {
        hir::Expr::new(
            elem_ty.clone(),
            span,
            hir::ExprKind::IndexGet {
                target: Box::new(hir::Expr::local(collection_ty.clone(), span, xs_local)),
                index: Box::new(hir::Expr::local(Type::Int, span, i_local)),
            },
        )
    };

    let result_expr = match method {
        "map" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                result_ty.clone(),
                hir::Expr::new(
                    result_ty.clone(),
                    span,
                    hir::ExprKind::ListLiteral { elements: vec![] },
                ),
            );
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let body_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Expr(hir::Expr::new(
                    Type::Void,
                    span,
                    hir::ExprKind::CollectionMut {
                        object: result_local,
                        method: hir::CollectionMethod::ListPush,
                        args: vec![body_expr],
                    },
                )),
            });
            hir::Expr::local(result_ty.clone(), span, result_local)
        }
        "filter" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                result_ty.clone(),
                hir::Expr::new(
                    result_ty.clone(),
                    span,
                    hir::ExprKind::ListLiteral { elements: vec![] },
                ),
            );
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let pred_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            let push_stmt = hir::Stmt {
                span,
                kind: hir::StmtKind::Expr(hir::Expr::new(
                    Type::Void,
                    span,
                    hir::ExprKind::CollectionMut {
                        object: result_local,
                        method: hir::CollectionMethod::ListPush,
                        args: vec![hir::Expr::local(elem_ty.clone(), span, param_local)],
                    },
                )),
            };
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond: pred_expr,
                    then_block: hir::Block {
                        stmts: vec![push_stmt],
                    },
                    else_block: None,
                },
            });
            hir::Expr::local(result_ty.clone(), span, result_local)
        }
        "for_each" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let body_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Expr(body_expr),
            });
            hir::Expr::new(Type::Void, span, hir::ExprKind::Nil)
        }
        "any" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                Type::Bool,
                hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(false)),
            );
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let pred_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond: pred_expr,
                    then_block: hir::Block {
                        stmts: vec![
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Assign {
                                    local: result_local,
                                    value: hir::Expr::new(
                                        Type::Bool,
                                        span,
                                        hir::ExprKind::Bool(true),
                                    ),
                                },
                            },
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Break,
                            },
                        ],
                    },
                    else_block: None,
                },
            });
            hir::Expr::local(Type::Bool, span, result_local)
        }
        "all" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                Type::Bool,
                hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(true)),
            );
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let pred_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            let negated = hir::Expr::new(
                Type::Bool,
                span,
                hir::ExprKind::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(pred_expr),
                },
            );
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond: negated,
                    then_block: hir::Block {
                        stmts: vec![
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Assign {
                                    local: result_local,
                                    value: hir::Expr::new(
                                        Type::Bool,
                                        span,
                                        hir::ExprKind::Bool(false),
                                    ),
                                },
                            },
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Break,
                            },
                        ],
                    },
                    else_block: None,
                },
            });
            hir::Expr::local(Type::Bool, span, result_local)
        }
        "find" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let opt_ty = result_ty.clone();
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                opt_ty.clone(),
                hir::Expr::new(opt_ty.clone(), span, hir::ExprKind::Nil),
            );
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let pred_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            let option_name = Ident(Intern::new("Option".to_string()));
            let some_name = Ident(Intern::new("Some".to_string()));
            let type_id = resolve_enum_type_id(ctx, span, option_name)?;
            let variant = resolve_variant_index(ctx, span, option_name, some_name)?;
            let some_expr = hir::Expr::new(
                opt_ty.clone(),
                span,
                hir::ExprKind::EnumLiteral {
                    type_id,
                    variant,
                    fields: vec![hir::Expr::local(elem_ty.clone(), span, param_local)],
                },
            );
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond: pred_expr,
                    then_block: hir::Block {
                        stmts: vec![
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Assign {
                                    local: result_local,
                                    value: some_expr,
                                },
                            },
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Break,
                            },
                        ],
                    },
                    else_block: None,
                },
            });
            hir::Expr::local(opt_ty, span, result_local)
        }
        "find_index" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let opt_ty = result_ty.clone();
            let result_local = alloc_and_bind(
                fc,
                span,
                out,
                opt_ty.clone(),
                hir::Expr::new(opt_ty.clone(), span, hir::ExprKind::Nil),
            );
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let pred_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            let option_name = Ident(Intern::new("Option".to_string()));
            let some_name = Ident(Intern::new("Some".to_string()));
            let type_id = resolve_enum_type_id(ctx, span, option_name)?;
            let variant = resolve_variant_index(ctx, span, option_name, some_name)?;
            let some_expr = hir::Expr::new(
                opt_ty.clone(),
                span,
                hir::ExprKind::EnumLiteral {
                    type_id,
                    variant,
                    fields: vec![hir::Expr::local(Type::Int, span, i_local)],
                },
            );
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond: pred_expr,
                    then_block: hir::Block {
                        stmts: vec![
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Assign {
                                    local: result_local,
                                    value: some_expr,
                                },
                            },
                            hir::Stmt {
                                span,
                                kind: hir::StmtKind::Break,
                            },
                        ],
                    },
                    else_block: None,
                },
            });
            hir::Expr::local(opt_ty, span, result_local)
        }
        "count" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
                unreachable!()
            };
            let result_local =
                alloc_and_bind(fc, span, out, Type::Int, hir::Expr::int_lit(span, 0));
            let param_name = lambda.node.params[0].name;
            let param_local = register_named_local(fc, param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let pred_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            let increment = hir::Expr::binary(
                Type::Int,
                span,
                BinaryOp::Add,
                hir::Expr::local(Type::Int, span, result_local),
                hir::Expr::int_lit(span, 1),
            );
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::If {
                    cond: pred_expr,
                    then_block: hir::Block {
                        stmts: vec![hir::Stmt {
                            span,
                            kind: hir::StmtKind::Assign {
                                local: result_local,
                                value: increment,
                            },
                        }],
                    },
                    else_block: None,
                },
            });
            hir::Expr::local(Type::Int, span, result_local)
        }
        "fold" => {
            let ast::ExprKind::Lambda(lambda) = &c.node.args[1].node.kind else {
                unreachable!()
            };
            let init_expr = lower_expr(&c.node.args[0], ctx, fc, out)?;
            let acc_local = alloc_and_bind(fc, span, out, result_ty.clone(), init_expr);
            let acc_param_name = lambda.node.params[0].name;
            fc.bind_local(acc_param_name, acc_local);
            let elem_param_name = lambda.node.params[1].name;
            let param_local = register_named_local(fc, elem_param_name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: param_local,
                    init: make_elem_expr(),
                },
            });
            let body_expr = lower_expr(&lambda.node.body, ctx, fc, &mut body_stmts)?;
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: acc_local,
                    value: body_expr,
                },
            });
            hir::Expr::local(result_ty.clone(), span, acc_local)
        }
        _ => unreachable!(),
    };

    emit_counter_increment(
        &mut body_stmts,
        span,
        i_local,
        &Type::Int,
        hir::Expr::local(Type::Int, span, step_local),
        BinaryOp::Add,
    );

    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::While {
            cond,
            body: hir::Block { stmts: body_stmts },
        },
    });

    fc.leave_scope(mark);

    Ok(Some(result_expr))
}

fn lower_sort_by(
    c: &ast::CallNode,
    elem_ty: &Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
) -> Result<Option<hir::Expr>, LowerError> {
    let ast::ExprKind::Field(field) = &c.node.func.node.kind else {
        return Ok(None);
    };

    // Receiver must be a simple local identifier for in-place mutation.
    let ast::ExprKind::Ident(root_name) = &field.node.target.node.kind else {
        return Ok(None);
    };

    let Some(&collection) = fc.local_map.get(root_name) else {
        return Ok(None);
    };

    let ast::ExprKind::Lambda(lambda) = &c.node.args[0].node.kind else {
        unreachable!("sort_by desugaring requires lambda arg")
    };

    let lambda_expr_id = c.node.args[0].node.id;
    let captures = ctx
        .shared
        .tcx
        .lambda_captures
        .get(&lambda_expr_id)
        .cloned()
        .unwrap_or_default();

    // Captures not supported via the opcode path — no closure-based sort infrastructure.
    if !captures.is_empty() {
        return Ok(None);
    }

    let lambda_func_id = ctx.shared.alloc_func_id();

    let mut lambda_fc = FuncLower {
        locals: vec![],
        local_map: HashMap::new(),
        scope_log: vec![],
    };

    for param in &lambda.node.params {
        let param_ty = param.ty.clone().unwrap_or_else(|| elem_ty.clone());
        register_named_local(&mut lambda_fc, param.name, param_ty);
    }

    let params_len = lambda_fc.locals.len() as u32;

    let body = match &lambda.node.body.node.kind {
        ast::ExprKind::Block(block_node) => {
            lower_block(block_node, ctx, &mut lambda_fc, true, &Type::Bool)?
        }
        _ => {
            let mut body_stmts = vec![];
            let body_expr = lower_expr(&lambda.node.body, ctx, &mut lambda_fc, &mut body_stmts)?;
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Return(Some(body_expr)),
            });
            hir::Block { stmts: body_stmts }
        }
    };

    let lambda_name = Ident(Intern::new(format!("__sort_cmp_{}", lambda_func_id.0)));

    ctx.shared.register_lambda_func(hir::Func {
        id: lambda_func_id,
        name: lambda_name,
        locals: lambda_fc.locals,
        params_len,
        ret: Type::Bool,
        body,
        span,
    });

    Ok(Some(hir::Expr::new(
        Type::Void,
        span,
        hir::ExprKind::SortBy {
            collection,
            comparator: lambda_func_id,
        },
    )))
}

fn try_lower_method_call(
    c: &ast::CallNode,
    ty: &Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Expr>, LowerError> {
    let ast::ExprKind::Field(field) = &c.node.func.node.kind else {
        return Ok(None);
    };
    if let ast::ExprKind::Ident(root_name) = &field.node.target.node.kind {
        let is_type_name = ctx.shared.tcx.is_module_name(*root_name)
            || ctx.shared.enum_type_ids.contains_key(root_name)
            || ctx.shared.struct_type_ids.contains_key(root_name)
            || ctx.shared.tcx.get_extern_type(*root_name).is_some();
        if is_type_name {
            return Ok(None);
        }

        let method_name = field.node.field;
        let target_ty = ctx.expr_type(field.node.target.node.id, span)?;

        let collection_method = match (&target_ty, method_name.0.as_ref().as_str()) {
            (Type::List { .. }, "push") => Some(hir::CollectionMethod::ListPush),
            (Type::List { .. }, "pop") => Some(hir::CollectionMethod::ListPop),
            (Type::Map { .. }, "insert") => Some(hir::CollectionMethod::MapInsert),
            (Type::Map { .. }, "remove") => Some(hir::CollectionMethod::MapRemove),
            _ => None,
        };
        if let Some(method) = collection_method {
            let local_id = *fc
                .local_map
                .get(root_name)
                .ok_or(LowerError::UnknownLocal {
                    name: *root_name,
                    span,
                })?;
            let args = lower_args(&c.node.args, ctx, fc, out)?;
            return Ok(Some(hir::Expr::new(
                ty.clone(),
                span,
                hir::ExprKind::CollectionMut {
                    object: local_id,
                    method,
                    args,
                },
            )));
        }

        if let Type::Extern { name: type_name } = &target_ty {
            let qualified = Ident(Intern::new(format!("{type_name}::{method_name}")));
            if let Some(&extern_id) = ctx.shared.externs.get(&qualified) {
                let receiver = lower_expr(&field.node.target, ctx, fc, out)?;
                let mut args = vec![receiver];
                args.extend(lower_args(&c.node.args, ctx, fc, out)?);
                return Ok(Some(hir::Expr::new(
                    ty.clone(),
                    span,
                    hir::ExprKind::CallExtern { extern_id, args },
                )));
            }
        }
    }

    let method_name = field.node.field;
    let target_ty = ctx.expr_type(field.node.target.node.id, span)?;

    if let Type::List { elem } = &target_ty {
        let method_str = method_name.0.as_ref().as_str();

        let sort_by_lambda = method_str == "sort_by"
            && c.node.args.len() == 1
            && matches!(c.node.args[0].node.kind, ast::ExprKind::Lambda(_));
        if sort_by_lambda {
            return lower_sort_by(c, elem, span, ctx, fc);
        }

        let single_lambda = matches!(
            method_str,
            "map" | "filter" | "for_each" | "any" | "all" | "find" | "find_index" | "count"
        ) && c.node.args.len() == 1
            && matches!(c.node.args[0].node.kind, ast::ExprKind::Lambda(_));
        let fold_lambda = method_str == "fold"
            && c.node.args.len() == 2
            && matches!(c.node.args[1].node.kind, ast::ExprKind::Lambda(_));
        if single_lambda || fold_lambda {
            return lower_list_desugar(c, method_str, elem, ty, span, ctx, fc, out);
        }
    }

    if let Type::Map { key, value } = &target_ty {
        let method_str = method_name.0.as_ref().as_str();
        if matches!(method_str, "map_values" | "filter")
            && c.node.args.len() == 1
            && matches!(c.node.args[0].node.kind, ast::ExprKind::Lambda(_))
        {
            return lower_map_desugar(c, method_str, key, value, ty, span, ctx, fc, out);
        }
    }

    if let Type::Struct {
        name: struct_name,
        type_args,
    } = &target_ty
    {
        let mangled = mangle_method_spec_name(*struct_name, method_name, type_args);
        if let Some(&func_id) = ctx.shared.funcs.get(&mangled) {
            let receiver = lower_expr(&field.node.target, ctx, fc, out)?;
            let mut args = vec![receiver];
            args.extend(lower_args(&c.node.args, ctx, fc, out)?);
            let defaults = ctx
                .shared
                .tcx
                .method_param_defaults(*struct_name, method_name);
            for val in defaults.iter().skip(c.node.args.len()).flatten() {
                args.push(const_value_to_hir_expr(val, span));
            }
            return Ok(Some(hir::Expr::new(
                ty.clone(),
                span,
                hir::ExprKind::Call {
                    func: func_id,
                    args,
                },
            )));
        }
    }

    Ok(None)
}

fn try_lower_enum_constructor(
    c: &ast::CallNode,
    ty: &Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Expr>, LowerError> {
    let Type::Enum {
        name: enum_name, ..
    } = ty
    else {
        return Ok(None);
    };
    let ast::ExprKind::Field(field) = &c.node.func.node.kind else {
        return Ok(None);
    };
    let ast::ExprKind::Ident(_) = &field.node.target.node.kind else {
        return Ok(None);
    };

    let enum_name = *enum_name;
    let variant_name = field.node.field;
    let type_id = resolve_enum_type_id(ctx, span, enum_name)?;
    let variant = resolve_variant_index(ctx, span, enum_name, variant_name)?;
    let fields = lower_args(&c.node.args, ctx, fc, out)?;
    Ok(Some(hir::Expr::new(
        ty.clone(),
        span,
        hir::ExprKind::EnumLiteral {
            type_id,
            variant,
            fields,
        },
    )))
}

fn lower_direct_call(
    c: &ast::CallNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::ExprKind, LowerError> {
    let (callee_name, module_name) = match &c.node.func.node.kind {
        ast::ExprKind::Ident(name) => (*name, None),
        ast::ExprKind::Field(field) => {
            // module qualified call module.func(args), resolve to the function name
            if let ast::ExprKind::Ident(mod_name) = &field.node.target.node.kind {
                if ctx.shared.tcx.is_module_name(*mod_name) {
                    (field.node.field, Some(*mod_name))
                } else if ctx.shared.tcx.get_extern_type(*mod_name).is_some() {
                    let method_name = field.node.field;
                    (
                        Ident(Intern::new(format!("{mod_name}::{method_name}"))),
                        None,
                    )
                } else {
                    return Err(LowerError::NonDirectCall { span });
                }
            } else if let ast::ExprKind::Field(inner_field) = &field.node.target.node.kind {
                // nested facade.submodule.func(args), two levels of field access
                if let ast::ExprKind::Ident(outer_module) = &inner_field.node.target.node.kind {
                    if ctx.shared.tcx.is_module_name(*outer_module) {
                        (field.node.field, None)
                    } else {
                        return Err(LowerError::NonDirectCall { span });
                    }
                } else {
                    return Err(LowerError::NonDirectCall { span });
                }
            } else {
                return Err(LowerError::NonDirectCall { span });
            }
        }
        _ => return Err(LowerError::NonDirectCall { span }),
    };
    let mut args = lower_args(&c.node.args, ctx, fc, out)?;

    let inject_defaults = |args: &mut Vec<hir::Expr>, defaults: &[Option<ConstValue>]| {
        for val in defaults.iter().skip(c.node.args.len()).flatten() {
            args.push(const_value_to_hir_expr(val, span));
        }
    };

    // builtins take precedence over user functions and externs of the same name
    if let Some(builtin) = Builtin::from_name(callee_name.0.as_ref()) {
        Ok(hir::ExprKind::CallBuiltin { builtin, args })
    } else if let Some(&func_id) = ctx.shared.funcs.get(&callee_name) {
        let defaults = match module_name {
            Some(m) => ctx.shared.tcx.module_func_param_defaults(m, callee_name),
            None => ctx.shared.tcx.func_param_defaults(callee_name),
        };
        inject_defaults(&mut args, defaults);
        Ok(hir::ExprKind::Call {
            func: func_id,
            args,
        })
    } else if let Some(&extern_id) = ctx.shared.externs.get(&callee_name) {
        Ok(hir::ExprKind::CallExtern { extern_id, args })
    } else if let Some((func_name, type_args)) = ctx.shared.tcx.call_type_args(c.node.func.node.id)
    {
        let defaults = ctx.shared.tcx.func_param_defaults(*func_name);
        inject_defaults(&mut args, defaults);
        let mangled = mangle_generic_name(*func_name, type_args);
        let &func_id = ctx
            .shared
            .funcs
            .get(&mangled)
            .ok_or(LowerError::UnknownFunc {
                name: callee_name,
                span,
            })?;
        Ok(hir::ExprKind::Call {
            func: func_id,
            args,
        })
    } else {
        Err(LowerError::UnknownFunc {
            name: callee_name,
            span,
        })
    }
}

fn lower_struct_literal_expr(
    lit: &ast::StructLiteralNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    // enum struct variant Event.Move { dx: 5, dy: 10 }
    if let Some(enum_name) = lit.node.qualifier {
        let variant_name = lit.node.name;
        let type_id = resolve_enum_type_id(ctx, span, enum_name)?;
        let variant = resolve_variant_index(ctx, span, enum_name, variant_name)?;
        let field_names = ctx
            .shared
            .tcx
            .enum_variant_field_names(enum_name, variant_name)
            .ok_or_else(|| LowerError::UnsupportedExprKind {
                span,
                kind: format!(
                    "variant '{variant_name}' on enum '{enum_name}' is not a struct variant"
                ),
            })?;
        let mut fields = vec![];
        for decl_name in &field_names {
            let (_, expr) = lit
                .node
                .fields
                .iter()
                .find(|(name, _)| *name == *decl_name)
                .expect("typechecker ensures all declared fields are provided");
            fields.push(lower_expr(expr, ctx, fc, out)?);
        }
        return Ok(hir::Expr::new(
            ty,
            span,
            hir::ExprKind::EnumLiteral {
                type_id,
                variant,
                fields,
            },
        ));
    }

    if let Type::Extern { name } = &ty {
        let init_name = Ident(Intern::new(format!("{name}::__init__")));
        let extern_id =
            *ctx.shared
                .externs
                .get(&init_name)
                .ok_or_else(|| LowerError::UnsupportedExprKind {
                    span,
                    kind: format!("extern type '{name}' has no __init__ handler"),
                })?;
        let field_order = ctx
            .shared
            .tcx
            .extern_type_field_order(*name)
            .ok_or_else(|| LowerError::UnsupportedExprKind {
                span,
                kind: format!("unknown extern type '{name}'"),
            })?;
        let mut args = vec![];
        for field_name in field_order {
            let (_, expr) = lit
                .node
                .fields
                .iter()
                .find(|(name, _)| *name == *field_name)
                .expect("typechecker ensures all declared fields are provided");
            args.push(lower_expr(expr, ctx, fc, out)?);
        }
        return Ok(hir::Expr::new(
            ty,
            span,
            hir::ExprKind::CallExtern { extern_id, args },
        ));
    }

    let struct_name = lit.node.name;
    let type_id = resolve_struct_type_id(ctx, span, struct_name)?;
    let field_names = ctx
        .shared
        .tcx
        .struct_field_names(struct_name)
        .ok_or_else(|| LowerError::UnsupportedExprKind {
            span,
            kind: format!("unknown struct '{struct_name}'"),
        })?;
    // lower fields in declaration order, synthesizing defaults for omitted fields
    let mut fields = vec![];
    for decl_name in &field_names {
        match lit.node.fields.iter().find(|(name, _)| *name == *decl_name) {
            Some((_, expr)) => {
                fields.push(lower_expr(expr, ctx, fc, out)?);
            }
            None => {
                let default = ctx
                    .shared
                    .tcx
                    .struct_field_default(struct_name, *decl_name)
                    .expect("typechecker ensures omitted fields have defaults");
                let field_ty = ctx
                    .shared
                    .tcx
                    .struct_field_type(struct_name, *decl_name)
                    .expect("field type exists");
                fields.push(synthesize_default_hir_expr(default, &field_ty, span));
            }
        }
    }
    let kind = if matches!(ty, Type::DataRef { .. }) {
        hir::ExprKind::DataRefLiteral { type_id, fields }
    } else {
        hir::ExprKind::StructLiteral { type_id, fields }
    };
    Ok(hir::Expr::new(ty, span, kind))
}

fn synthesize_default_hir_expr(default: &FieldDefault, field_ty: &Type, span: Span) -> hir::Expr {
    let kind = match default {
        FieldDefault::Const(cv) => match cv {
            ConstValue::Int(n) => hir::ExprKind::Int(*n),
            ConstValue::Float(f) => hir::ExprKind::Float(*f),
            ConstValue::Double(d) => hir::ExprKind::Double(*d),
            ConstValue::Bool(b) => hir::ExprKind::Bool(*b),
            ConstValue::String(s) => hir::ExprKind::String(s.clone()),
            ConstValue::Nil => hir::ExprKind::Nil,
        },
        FieldDefault::EmptyArray => match field_ty {
            Type::Array { .. } => hir::ExprKind::ArrayLiteral { elements: vec![] },
            Type::List { .. } => hir::ExprKind::ListLiteral { elements: vec![] },
            _ => unreachable!("typechecker validated empty array default against field type"),
        },
        FieldDefault::EmptyMap => hir::ExprKind::MapLiteral { entries: vec![] },
    };
    hir::Expr::new(field_ty.clone(), span, kind)
}

fn lower_field_expr(
    field_access: &ast::FieldAccessNode,
    ty: Type,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<hir::Expr, LowerError> {
    if field_access.node.safe {
        return lower_safe_field_expr(field_access, ty, span, ctx, fc, out);
    }

    let target = &field_access.node.target;
    let field_name = field_access.node.field;

    if let Ok(target_ty) = ctx.expr_type(target.node.id, span) {
        match target_ty {
            Type::Struct { name, .. } => {
                let index = ctx
                    .shared
                    .tcx
                    .struct_field_index(name, field_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown field '{field_name}' on struct '{name}'"),
                    })? as u16;
                let object = lower_expr(target, ctx, fc, out)?;
                return Ok(hir::Expr::new(
                    ty,
                    span,
                    hir::ExprKind::FieldGet {
                        object: Box::new(object),
                        index,
                    },
                ));
            }
            Type::DataRef { name, .. } => {
                let index = ctx
                    .shared
                    .tcx
                    .struct_field_index(name, field_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown field '{field_name}' on dataref '{name}'"),
                    })? as u16;
                let object = lower_expr(target, ctx, fc, out)?;
                return Ok(hir::Expr::new(
                    ty,
                    span,
                    hir::ExprKind::FieldGet {
                        object: Box::new(object),
                        index,
                    },
                ));
            }
            Type::NamedTuple(fields) => {
                let index = fields
                    .iter()
                    .position(|(label, _)| *label == field_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown field '{field_name}' on named tuple"),
                    })? as u16;
                let object = lower_expr(target, ctx, fc, out)?;
                return Ok(hir::Expr::new(
                    ty,
                    span,
                    hir::ExprKind::FieldGet {
                        object: Box::new(object),
                        index,
                    },
                ));
            }
            Type::Extern { name } => {
                let qualified = Ident(Intern::new(format!("{name}::__get_{field_name}")));
                let extern_id = *ctx.shared.externs.get(&qualified).ok_or_else(|| {
                    LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown extern field getter '{qualified}'"),
                    }
                })?;
                let object = lower_expr(target, ctx, fc, out)?;
                return Ok(hir::Expr::new(
                    ty,
                    span,
                    hir::ExprKind::CallExtern {
                        extern_id,
                        args: vec![object],
                    },
                ));
            }
            other => {
                return Err(LowerError::UnsupportedExprKind {
                    span,
                    kind: format!("field access on unsupported type '{other}'"),
                });
            }
        }
    }

    if let Type::Enum {
        name: enum_name, ..
    } = &ty
    {
        let enum_name = *enum_name;
        let type_id = resolve_enum_type_id(ctx, span, enum_name)?;
        let variant = resolve_variant_index(ctx, span, enum_name, field_name)?;
        return Ok(hir::Expr::new(
            ty,
            span,
            hir::ExprKind::EnumLiteral {
                type_id,
                variant,
                fields: vec![],
            },
        ));
    }

    Err(LowerError::UnsupportedExprKind {
        span,
        kind: "field access on type with no type information".to_string(),
    })
}
