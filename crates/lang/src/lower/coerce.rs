use crate::{
    ast::Type,
    hir::{self, ExprKind, StmtKind},
    ir_meta::VariantShape,
    prelude_enums::OPTION_TYPE_ID,
};

struct CoerceCtx {
    func_params: Vec<Vec<Type>>,
    struct_fields: Vec<(String, Vec<Type>)>,
    enum_variant_fields: Vec<Vec<Vec<Type>>>,
}

/// Wraps T in Some(v) wherever they flow into an Option<T> position
/// must run before ownership analysis so inserted nodes get correct ownership annotations
pub fn coerce_optionals(program: &mut hir::Program) {
    let ctx = CoerceCtx {
        func_params: program
            .funcs
            .iter()
            .map(|f| {
                f.locals[..f.params_len as usize]
                    .iter()
                    .map(|l| l.ty.clone())
                    .collect()
            })
            .collect(),
        struct_fields: program
            .aggregate_meta
            .iter()
            .map(|m| {
                (
                    m.name.clone(),
                    m.fields.iter().map(|f| f.ty.clone()).collect(),
                )
            })
            .collect(),
        enum_variant_fields: program
            .enum_meta
            .iter()
            .map(|e| {
                e.variants
                    .iter()
                    .map(|v| match &v.shape {
                        VariantShape::Unit => vec![],
                        VariantShape::Tuple(types) => types.clone(),
                        VariantShape::Struct(fields) => {
                            fields.iter().map(|f| f.ty.clone()).collect()
                        }
                    })
                    .collect()
            })
            .collect(),
    };

    for func in &mut program.funcs {
        let ret_ty = func.ret.clone();
        let local_types: Vec<Type> = func.locals.iter().map(|l| l.ty.clone()).collect();
        coerce_block(&mut func.body, &ctx, &local_types, &ret_ty);
    }
}

fn coerce_block(block: &mut hir::Block, ctx: &CoerceCtx, locals: &[Type], ret_ty: &Type) {
    for stmt in &mut block.stmts {
        coerce_stmt(stmt, ctx, locals, ret_ty);
    }
}

fn coerce_stmt(stmt: &mut hir::Stmt, ctx: &CoerceCtx, locals: &[Type], ret_ty: &Type) {
    match &mut stmt.kind {
        StmtKind::Let { local, init } => {
            coerce_expr_tree(init, ctx);
            let target = &locals[local.0 as usize];
            try_coerce(init, target);
        }
        StmtKind::Assign { local, value } => {
            coerce_expr_tree(value, ctx);
            let target = &locals[local.0 as usize];
            try_coerce(value, target);
        }
        StmtKind::Expr(expr) => {
            coerce_expr_tree(expr, ctx);
        }
        StmtKind::Return(Some(expr)) => {
            coerce_expr_tree(expr, ctx);
            try_coerce(expr, ret_ty);
        }
        StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            coerce_expr_tree(cond, ctx);
            coerce_block(then_block, ctx, locals, ret_ty);
            if let Some(b) = else_block {
                coerce_block(b, ctx, locals, ret_ty);
            }
        }
        StmtKind::While { cond, body } => {
            coerce_expr_tree(cond, ctx);
            coerce_block(body, ctx, locals, ret_ty);
        }
        StmtKind::SetField {
            object,
            field_index,
            value,
        } => {
            coerce_expr_tree(value, ctx);
            let obj_ty = &locals[object.0 as usize];
            if let Some(field_ty) = resolve_struct_field_type(ctx, obj_ty, *field_index) {
                try_coerce(value, &field_ty);
            }
        }
        StmtKind::SetIndex { index, value, .. } => {
            coerce_expr_tree(index, ctx);
            coerce_expr_tree(value, ctx);
        }
        StmtKind::Match {
            scrutinee_init,
            arms,
            else_body,
            ..
        } => {
            coerce_expr_tree(scrutinee_init, ctx);
            for arm in arms {
                if let Some(guard) = &mut arm.guard {
                    coerce_expr_tree(guard, ctx);
                }
                coerce_block(&mut arm.body, ctx, locals, ret_ty);
            }
            if let Some(else_arm) = else_body {
                coerce_block(&mut else_arm.body, ctx, locals, ret_ty);
            }
        }
    }
}

fn coerce_expr_tree(expr: &mut hir::Expr, ctx: &CoerceCtx) {
    match &mut expr.kind {
        ExprKind::Call { func, args, .. } => {
            for arg in args.iter_mut() {
                coerce_expr_tree(arg, ctx);
            }
            let func_idx = func.0 as usize;
            if let Some(param_types) = ctx.func_params.get(func_idx) {
                for (arg, param_ty) in args.iter_mut().zip(param_types) {
                    try_coerce(arg, param_ty);
                }
            }
        }
        ExprKind::StructLiteral { type_id, fields } => {
            for field in fields.iter_mut() {
                coerce_expr_tree(field, ctx);
            }
            if let Some((_, field_types)) = ctx.struct_fields.get(*type_id as usize) {
                for (field, field_ty) in fields.iter_mut().zip(field_types) {
                    try_coerce(field, field_ty);
                }
            }
        }
        ExprKind::EnumLiteral {
            type_id,
            variant,
            fields,
        } => {
            for field in fields.iter_mut() {
                coerce_expr_tree(field, ctx);
            }
            if let Some(field_types) = ctx
                .enum_variant_fields
                .get(*type_id as usize)
                .and_then(|v| v.get(*variant as usize))
            {
                for (field, field_ty) in fields.iter_mut().zip(field_types) {
                    try_coerce(field, field_ty);
                }
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            coerce_expr_tree(lhs, ctx);
            coerce_expr_tree(rhs, ctx);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast(inner)
        | ExprKind::ToString(inner)
        | ExprKind::Format(inner, _)
        | ExprKind::FieldGet { object: inner, .. }
        | ExprKind::TupleIndex { tuple: inner, .. }
        | ExprKind::UnwrapOptional(inner)
        | ExprKind::CollectionLen { collection: inner }
        | ExprKind::MapLen { map: inner }
        | ExprKind::ArrayFill { value: inner, .. }
        | ExprKind::ListFill { value: inner, .. } => {
            coerce_expr_tree(inner, ctx);
        }
        ExprKind::CallBuiltin { args, .. }
        | ExprKind::CallExtern { args, .. }
        | ExprKind::TupleLiteral { elements: args }
        | ExprKind::ArrayLiteral { elements: args }
        | ExprKind::ListLiteral { elements: args }
        | ExprKind::CollectionMut { args, .. } => {
            for arg in args {
                coerce_expr_tree(arg, ctx);
            }
        }
        ExprKind::CreateClosure { captures, .. } => {
            for cap in captures {
                coerce_expr_tree(cap, ctx);
            }
        }
        ExprKind::CallClosure { callee, args, .. } => {
            coerce_expr_tree(callee, ctx);
            for arg in args {
                coerce_expr_tree(arg, ctx);
            }
        }
        ExprKind::DataRefLiteral { fields, .. } => {
            for field in fields {
                coerce_expr_tree(field, ctx);
            }
        }
        ExprKind::IndexGet { target, index } | ExprKind::MapEntryAt { map: target, index } => {
            coerce_expr_tree(target, ctx);
            coerce_expr_tree(index, ctx);
        }
        ExprKind::Slice {
            target, start, end, ..
        } => {
            coerce_expr_tree(target, ctx);
            coerce_expr_tree(start, ctx);
            coerce_expr_tree(end, ctx);
        }
        ExprKind::MapLiteral { entries } => {
            for (k, v) in entries {
                coerce_expr_tree(k, ctx);
                coerce_expr_tree(v, ctx);
            }
        }
        ExprKind::Local(_)
        | ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Double(_)
        | ExprKind::Bool(_)
        | ExprKind::String(_)
        | ExprKind::Nil
        | ExprKind::SortBy { .. } => {}
    }
}

/// If target is Option<T> and expr is T (not already optional), wrap in Some(expr)
fn try_coerce(expr: &mut hir::Expr, target: &Type) {
    if !target.is_option() || expr.ty.is_option() || matches!(expr.kind, ExprKind::Nil) {
        return;
    }
    let span = expr.span;
    let placeholder = hir::Expr::new(Type::Void, span, ExprKind::Nil);
    let owned = std::mem::replace(expr, placeholder);
    *expr = hir::Expr::new(
        target.clone(),
        span,
        ExprKind::EnumLiteral {
            type_id: OPTION_TYPE_ID,
            variant: 1,
            fields: vec![owned],
        },
    );
}

fn resolve_struct_field_type(ctx: &CoerceCtx, obj_ty: &Type, field_index: u16) -> Option<Type> {
    let name = match obj_ty {
        Type::Struct { name, .. } | Type::DataRef { name, .. } => name.to_string(),
        _ => return None,
    };
    ctx.struct_fields
        .iter()
        .find(|(n, _)| *n == name)
        .and_then(|(_, types)| types.get(field_index as usize).cloned())
}
