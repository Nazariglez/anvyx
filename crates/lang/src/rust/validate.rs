use std::collections::{BTreeSet, HashSet};

use crate::{
    ast::{BinaryOp, FormatKind, FormatSign, Type},
    builtin::Builtin,
    hir::{self, ExprKind, FuncId, StmtKind},
    ir_meta::{AggregateKind, VariantMeta, VariantShape},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Feature {
    FmtF32,
    FmtF64,
    UsesMap,
    UsesCollections,
}

#[derive(Debug, Default)]
pub struct HelperFlags {
    pub stringify_aggregates: BTreeSet<u32>,
    pub stringify_enums: BTreeSet<u32>,
    features: BTreeSet<Feature>,
}

impl HelperFlags {
    pub fn fmt_f32(&self) -> bool {
        self.features.contains(&Feature::FmtF32)
    }

    pub fn fmt_f64(&self) -> bool {
        self.features.contains(&Feature::FmtF64)
    }

    pub fn uses_map(&self) -> bool {
        self.features.contains(&Feature::UsesMap)
    }

    pub fn uses_collections(&self) -> bool {
        self.features.contains(&Feature::UsesCollections)
    }

    fn enable(&mut self, feature: Feature) {
        self.features.insert(feature);
    }
}

pub struct RustPlan {
    pub main: FuncId,
    pub funcs: Vec<FuncId>,
    pub used_aggregates: BTreeSet<u32>,
    pub used_enums: BTreeSet<u32>,
    pub helper_flags: HelperFlags,
}

pub fn validate(program: &hir::Program) -> Result<RustPlan, String> {
    let main_id = program
        .funcs
        .iter()
        .find(|f| f.name.to_string() == "main")
        .map(|f| f.id)
        .ok_or_else(|| "Rust backend: no `main` function found".to_string())?;

    let mut reachable: HashSet<FuncId> = HashSet::new();
    let mut worklist = vec![main_id];
    while let Some(fid) = worklist.pop() {
        if !reachable.insert(fid) {
            continue;
        }
        collect_called_funcs(program, &program.funcs[fid.0 as usize], &mut worklist);
    }

    let mut used_aggregates = BTreeSet::new();
    let mut used_enums = BTreeSet::new();
    let mut helper_flags = HelperFlags::default();
    let mut errors = vec![];

    {
        let mut ctx = ValidationCtx {
            program,
            used_aggregates: &mut used_aggregates,
            used_enums: &mut used_enums,
            helper_flags: &mut helper_flags,
            errors: &mut errors,
        };
        for &fid in &reachable {
            let func = &program.funcs[fid.0 as usize];
            validate_func(&mut ctx, func);
        }
    }

    // used_aggregates must not contain DataRef types
    for &type_id in &used_aggregates {
        let meta = &program.aggregate_meta[type_id as usize];
        if meta.kind == AggregateKind::DataRef {
            errors.push(format!(
                "Rust backend: unsupported dataref type `{}` in reachable code",
                meta.name
            ));
        }
    }

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    // validate display functions for every stringified aggregate,
    //even when they are only reached through nested fields
    let mut extra_worklist: Vec<FuncId> = helper_flags
        .stringify_aggregates
        .iter()
        .filter_map(|&type_id| program.aggregate_meta[type_id as usize].display_func)
        .filter(|fid| !reachable.contains(fid))
        .collect();

    {
        let mut ctx = ValidationCtx {
            program,
            used_aggregates: &mut used_aggregates,
            used_enums: &mut used_enums,
            helper_flags: &mut helper_flags,
            errors: &mut errors,
        };
        while let Some(fid) = extra_worklist.pop() {
            if !reachable.insert(fid) {
                continue;
            }
            let func = &program.funcs[fid.0 as usize];
            collect_called_funcs(program, func, &mut extra_worklist);
            validate_func(&mut ctx, func);
        }
    }

    // build ordered func list, program order, reachable, excluding main
    let funcs: Vec<FuncId> = program
        .funcs
        .iter()
        .filter(|f| f.id != main_id && reachable.contains(&f.id))
        .map(|f| f.id)
        .collect();

    Ok(RustPlan {
        main: main_id,
        funcs,
        used_aggregates,
        used_enums,
        helper_flags,
    })
}

fn collect_called_funcs(program: &hir::Program, func: &hir::Func, out: &mut Vec<FuncId>) {
    collect_block_calls(program, &func.body, out);
}

fn collect_block_calls(program: &hir::Program, block: &hir::Block, out: &mut Vec<FuncId>) {
    for stmt in &block.stmts {
        collect_stmt_calls(program, stmt, out);
    }
}

fn collect_stmt_calls(program: &hir::Program, stmt: &hir::Stmt, out: &mut Vec<FuncId>) {
    match &stmt.kind {
        StmtKind::Let { init: e, .. }
        | StmtKind::Assign { value: e, .. }
        | StmtKind::Expr(e)
        | StmtKind::Return(Some(e)) => collect_expr_calls(program, e, out),
        StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            collect_expr_calls(program, cond, out);
            collect_block_calls(program, then_block, out);
            if let Some(b) = else_block {
                collect_block_calls(program, b, out);
            }
        }
        StmtKind::While { cond, body } => {
            collect_expr_calls(program, cond, out);
            collect_block_calls(program, body, out);
        }
        StmtKind::SetField { value, .. } => collect_expr_calls(program, value, out),
        StmtKind::SetIndex { index, value, .. } => {
            collect_expr_calls(program, index, out);
            collect_expr_calls(program, value, out);
        }
        StmtKind::Match {
            scrutinee_init,
            arms,
            else_body,
            ..
        } => {
            collect_expr_calls(program, scrutinee_init, out);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_expr_calls(program, guard, out);
                }
                collect_block_calls(program, &arm.body, out);
            }
            if let Some(else_arm) = else_body {
                collect_block_calls(program, &else_arm.body, out);
            }
        }
    }
}

fn collect_expr_calls(program: &hir::Program, expr: &hir::Expr, out: &mut Vec<FuncId>) {
    match &expr.kind {
        ExprKind::Call { func, args, .. } => {
            out.push(*func);
            for arg in args {
                collect_expr_calls(program, arg, out);
            }
        }
        ExprKind::CreateClosure { func, captures } => {
            out.push(*func);
            for cap in captures {
                collect_expr_calls(program, cap, out);
            }
        }
        ExprKind::SortBy { comparator, .. } => {
            out.push(*comparator);
        }
        ExprKind::CallBuiltin { builtin, args } => {
            if *builtin == Builtin::Println
                && let Some(arg) = args.first()
            {
                track_display_func(program, &arg.ty, out);
            }
            for arg in args {
                collect_expr_calls(program, arg, out);
            }
        }
        ExprKind::CallExtern { args, .. } | ExprKind::CollectionMut { args, .. } => {
            for arg in args {
                collect_expr_calls(program, arg, out);
            }
        }
        ExprKind::CallClosure { callee, args, .. } => {
            collect_expr_calls(program, callee, out);
            for arg in args {
                collect_expr_calls(program, arg, out);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_expr_calls(program, lhs, out);
            collect_expr_calls(program, rhs, out);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast(inner)
        | ExprKind::UnwrapOptional(inner)
        | ExprKind::Format(inner, _) => {
            collect_expr_calls(program, inner, out);
        }
        ExprKind::ToString(inner) => {
            track_display_func(program, &inner.ty, out);
            collect_expr_calls(program, inner, out);
        }
        ExprKind::StructLiteral { fields, .. }
        | ExprKind::DataRefLiteral { fields, .. }
        | ExprKind::EnumLiteral { fields, .. } => {
            for f in fields {
                collect_expr_calls(program, f, out);
            }
        }
        ExprKind::ArrayLiteral { elements }
        | ExprKind::ListLiteral { elements }
        | ExprKind::TupleLiteral { elements } => {
            for e in elements {
                collect_expr_calls(program, e, out);
            }
        }
        ExprKind::FieldGet { object, .. } => collect_expr_calls(program, object, out),
        ExprKind::TupleIndex { tuple, .. } => collect_expr_calls(program, tuple, out),
        ExprKind::ArrayFill { value, .. } | ExprKind::ListFill { value, .. } => {
            collect_expr_calls(program, value, out);
        }
        ExprKind::IndexGet { target, index } => {
            collect_expr_calls(program, target, out);
            collect_expr_calls(program, index, out);
        }
        ExprKind::Slice {
            target, start, end, ..
        } => {
            collect_expr_calls(program, target, out);
            collect_expr_calls(program, start, out);
            collect_expr_calls(program, end, out);
        }
        ExprKind::CollectionLen { collection } => collect_expr_calls(program, collection, out),
        ExprKind::MapLen { map } => collect_expr_calls(program, map, out),
        ExprKind::MapEntryAt { map, index } => {
            collect_expr_calls(program, map, out);
            collect_expr_calls(program, index, out);
        }
        ExprKind::MapLiteral { entries } => {
            for (k, v) in entries {
                collect_expr_calls(program, k, out);
                collect_expr_calls(program, v, out);
            }
        }
        ExprKind::Local(_)
        | ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Double(_)
        | ExprKind::Bool(_)
        | ExprKind::String(_)
        | ExprKind::Nil => {}
    }
}

fn track_display_func(program: &hir::Program, ty: &Type, out: &mut Vec<FuncId>) {
    if let Type::Struct { name, .. } = ty
        && let Some((_, meta)) = program.find_aggregate_by_name(&name.to_string())
        && let Some(func_id) = meta.display_func
    {
        out.push(func_id);
    }
}

fn collect_variant_stringify_needs(
    program: &hir::Program,
    variant: &VariantMeta,
    helper_flags: &mut HelperFlags,
    used_aggregates: &mut BTreeSet<u32>,
    used_enums: &mut BTreeSet<u32>,
) {
    let field_types: Vec<&Type> = match &variant.shape {
        VariantShape::Tuple(types) => types.iter().collect(),
        VariantShape::Struct(fields) => fields.iter().map(|f| &f.ty).collect(),
        VariantShape::Unit => vec![],
    };
    for t in field_types {
        collect_stringify_needs(program, t, helper_flags, used_aggregates, used_enums);
    }
}

fn collect_stringify_needs(
    program: &hir::Program,
    ty: &Type,
    helper_flags: &mut HelperFlags,
    used_aggregates: &mut BTreeSet<u32>,
    used_enums: &mut BTreeSet<u32>,
) {
    match ty {
        Type::Float => helper_flags.enable(Feature::FmtF32),
        Type::Double => helper_flags.enable(Feature::FmtF64),
        Type::Enum { .. } if ty.is_option() => {
            if let Some(inner) = ty.option_inner() {
                collect_stringify_needs(program, inner, helper_flags, used_aggregates, used_enums);
            }
        }
        Type::Enum { name, .. } => {
            let name_str = name.to_string();
            if let Some((i, meta)) = program
                .enum_meta
                .iter()
                .enumerate()
                .find(|(_, m)| m.name == name_str)
            {
                let type_id = i as u32;
                used_enums.insert(type_id);
                if helper_flags.stringify_enums.insert(type_id) {
                    for variant in &meta.variants {
                        collect_variant_stringify_needs(
                            program,
                            variant,
                            helper_flags,
                            used_aggregates,
                            used_enums,
                        );
                    }
                }
            }
        }
        Type::Struct { name, .. } => {
            let name_str = name.to_string();
            if let Some((id, meta)) = program.find_aggregate_by_name(&name_str) {
                used_aggregates.insert(id);
                if helper_flags.stringify_aggregates.insert(id) && meta.display_func.is_none() {
                    let field_types: Vec<Type> = meta.fields.iter().map(|f| f.ty.clone()).collect();
                    for t in &field_types {
                        collect_stringify_needs(
                            program,
                            t,
                            helper_flags,
                            used_aggregates,
                            used_enums,
                        );
                    }
                }
            }
        }
        Type::Tuple(elems) => {
            for e in elems {
                collect_stringify_needs(program, e, helper_flags, used_aggregates, used_enums);
            }
        }
        Type::UnresolvedName(name) => {
            let name_str = name.to_string();
            if let Some((id, meta)) = program.find_aggregate_by_name(&name_str) {
                used_aggregates.insert(id);
                if helper_flags.stringify_aggregates.insert(id) && meta.display_func.is_none() {
                    let field_types: Vec<Type> = meta.fields.iter().map(|f| f.ty.clone()).collect();
                    for t in &field_types {
                        collect_stringify_needs(
                            program,
                            t,
                            helper_flags,
                            used_aggregates,
                            used_enums,
                        );
                    }
                }
            } else if let Some((i, meta)) = program
                .enum_meta
                .iter()
                .enumerate()
                .find(|(_, m)| m.name == name_str)
            {
                let type_id = i as u32;
                used_enums.insert(type_id);
                if helper_flags.stringify_enums.insert(type_id) {
                    for variant in &meta.variants {
                        collect_variant_stringify_needs(
                            program,
                            variant,
                            helper_flags,
                            used_aggregates,
                            used_enums,
                        );
                    }
                }
            }
        }
        _ => {}
    }
}

struct ValidationCtx<'a> {
    program: &'a hir::Program,
    used_aggregates: &'a mut BTreeSet<u32>,
    used_enums: &'a mut BTreeSet<u32>,
    helper_flags: &'a mut HelperFlags,
    errors: &'a mut Vec<String>,
}

fn validate_func(ctx: &mut ValidationCtx, func: &hir::Func) {
    let fn_name = func.name.to_string();

    for local in &func.locals[..func.params_len as usize] {
        validate_type(&local.ty, &fn_name, ctx.errors);
        track_type_usage(ctx.program, &local.ty, ctx.used_aggregates, ctx.used_enums);
        track_collection_flags(&local.ty, ctx.helper_flags);
    }

    validate_type(&func.ret, &fn_name, ctx.errors);
    track_type_usage(ctx.program, &func.ret, ctx.used_aggregates, ctx.used_enums);
    track_collection_flags(&func.ret, ctx.helper_flags);

    validate_block(ctx, func, &func.body, &fn_name);
}

fn validate_block(ctx: &mut ValidationCtx, func: &hir::Func, block: &hir::Block, fn_name: &str) {
    for stmt in &block.stmts {
        validate_stmt(ctx, func, stmt, fn_name);
    }
}

fn validate_stmt(ctx: &mut ValidationCtx, func: &hir::Func, stmt: &hir::Stmt, fn_name: &str) {
    match &stmt.kind {
        StmtKind::Let { local, init } => {
            let local_info = &func.locals[local.0 as usize];
            validate_type(&local_info.ty, fn_name, ctx.errors);
            track_type_usage(
                ctx.program,
                &local_info.ty,
                ctx.used_aggregates,
                ctx.used_enums,
            );
            validate_expr(ctx, func, init, fn_name);
        }
        StmtKind::Assign { value: e, .. }
        | StmtKind::Expr(e)
        | StmtKind::Return(Some(e))
        | StmtKind::SetField { value: e, .. } => {
            validate_expr(ctx, func, e, fn_name);
        }
        StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            validate_expr(ctx, func, cond, fn_name);
            validate_block(ctx, func, then_block, fn_name);
            if let Some(b) = else_block {
                validate_block(ctx, func, b, fn_name);
            }
        }
        StmtKind::While { cond, body } => {
            validate_expr(ctx, func, cond, fn_name);
            validate_block(ctx, func, body, fn_name);
        }
        StmtKind::SetIndex {
            object,
            index,
            value,
        } => {
            let obj_ty = &func.locals[object.0 as usize].ty;
            track_collection_flags(obj_ty, ctx.helper_flags);
            validate_expr(ctx, func, index, fn_name);
            validate_expr(ctx, func, value, fn_name);
        }
        StmtKind::Match {
            scrutinee_init,
            arms,
            else_body,
            ..
        } => {
            validate_expr(ctx, func, scrutinee_init, fn_name);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    validate_expr(ctx, func, guard, fn_name);
                }
                validate_block(ctx, func, &arm.body, fn_name);
            }
            if let Some(else_arm) = else_body {
                validate_block(ctx, func, &else_arm.body, fn_name);
            }
        }
    }
}

fn validate_expr(ctx: &mut ValidationCtx, func: &hir::Func, expr: &hir::Expr, fn_name: &str) {
    match &expr.kind {
        ExprKind::Local(id) => {
            let local_ty = &func.locals[id.0 as usize].ty;
            track_type_usage(ctx.program, local_ty, ctx.used_aggregates, ctx.used_enums);
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Double(_)
        | ExprKind::Bool(_)
        | ExprKind::String(_)
        | ExprKind::Nil => {}
        ExprKind::Cast(inner) => {
            validate_type(&expr.ty, fn_name, ctx.errors);
            track_type_usage(ctx.program, &expr.ty, ctx.used_aggregates, ctx.used_enums);
            validate_expr(ctx, func, inner, fn_name);
        }
        ExprKind::ToString(inner) => {
            collect_stringify_needs(
                ctx.program,
                &inner.ty,
                ctx.helper_flags,
                ctx.used_aggregates,
                ctx.used_enums,
            );
            validate_expr(ctx, func, inner, fn_name);
        }
        ExprKind::Format(inner, spec) => {
            let is_default = spec.align.is_none()
                && spec.sign == FormatSign::Default
                && !spec.zero_pad
                && spec.width.is_none()
                && spec.precision.is_none()
                && spec.kind == FormatKind::Default;
            if is_default {
                collect_stringify_needs(
                    ctx.program,
                    &inner.ty,
                    ctx.helper_flags,
                    ctx.used_aggregates,
                    ctx.used_enums,
                );
            }
            validate_expr(ctx, func, inner, fn_name);
        }
        ExprKind::Unary { expr: inner, .. } => {
            validate_expr(ctx, func, inner, fn_name);
        }
        ExprKind::Binary { lhs, rhs, op } => {
            if *op == BinaryOp::Add && expr.ty.is_str() {
                if !lhs.ty.is_str() {
                    collect_stringify_needs(
                        ctx.program,
                        &lhs.ty,
                        ctx.helper_flags,
                        ctx.used_aggregates,
                        ctx.used_enums,
                    );
                }
                if !rhs.ty.is_str() {
                    collect_stringify_needs(
                        ctx.program,
                        &rhs.ty,
                        ctx.helper_flags,
                        ctx.used_aggregates,
                        ctx.used_enums,
                    );
                }
            }
            validate_expr(ctx, func, lhs, fn_name);
            validate_expr(ctx, func, rhs, fn_name);
        }
        ExprKind::Call { args, .. } => {
            for arg in args {
                validate_expr(ctx, func, arg, fn_name);
            }
        }
        ExprKind::CallBuiltin { builtin, args } => {
            if *builtin == Builtin::Println && !args.is_empty() {
                collect_stringify_needs(
                    ctx.program,
                    &args[0].ty,
                    ctx.helper_flags,
                    ctx.used_aggregates,
                    ctx.used_enums,
                );
            }
            for arg in args {
                validate_expr(ctx, func, arg, fn_name);
            }
        }
        ExprKind::StructLiteral { type_id, fields } => {
            ctx.used_aggregates.insert(*type_id);
            for field in fields {
                validate_expr(ctx, func, field, fn_name);
            }
        }
        ExprKind::EnumLiteral {
            type_id, fields, ..
        } => {
            ctx.used_enums.insert(*type_id);
            for field in fields {
                validate_expr(ctx, func, field, fn_name);
            }
        }
        ExprKind::FieldGet { object, .. } => {
            validate_expr(ctx, func, object, fn_name);
        }
        ExprKind::TupleLiteral { elements } => {
            for e in elements {
                validate_expr(ctx, func, e, fn_name);
            }
        }
        ExprKind::TupleIndex { tuple, .. } => {
            validate_expr(ctx, func, tuple, fn_name);
        }
        ExprKind::CallExtern { .. } => {
            ctx.errors.push(format!(
                "Rust backend: unsupported extern call in function `{fn_name}`"
            ));
        }
        ExprKind::CreateClosure { .. } => {
            ctx.errors.push(format!(
                "Rust backend: unsupported closure creation in function `{fn_name}`"
            ));
        }
        ExprKind::CallClosure { .. } => {
            ctx.errors.push(format!(
                "Rust backend: unsupported closure call in function `{fn_name}`"
            ));
        }
        ExprKind::DataRefLiteral { .. } => {
            ctx.errors.push(format!(
                "Rust backend: unsupported dataref literal in function `{fn_name}`"
            ));
        }
        ExprKind::ArrayLiteral { elements } | ExprKind::ListLiteral { elements } => {
            track_collection_flags(&expr.ty, ctx.helper_flags);
            for e in elements {
                validate_expr(ctx, func, e, fn_name);
            }
        }
        ExprKind::ArrayFill { value, .. } | ExprKind::ListFill { value, .. } => {
            track_collection_flags(&expr.ty, ctx.helper_flags);
            validate_expr(ctx, func, value, fn_name);
        }
        ExprKind::MapLiteral { entries } => {
            track_collection_flags(&expr.ty, ctx.helper_flags);
            for (k, v) in entries {
                validate_expr(ctx, func, k, fn_name);
                validate_expr(ctx, func, v, fn_name);
            }
        }
        ExprKind::IndexGet { target, index } => {
            track_collection_flags(&target.ty, ctx.helper_flags);
            validate_expr(ctx, func, target, fn_name);
            validate_expr(ctx, func, index, fn_name);
        }
        ExprKind::Slice {
            target, start, end, ..
        } => {
            track_collection_flags(&target.ty, ctx.helper_flags);
            validate_expr(ctx, func, target, fn_name);
            validate_expr(ctx, func, start, fn_name);
            validate_expr(ctx, func, end, fn_name);
        }
        ExprKind::CollectionLen { collection } => {
            track_collection_flags(&collection.ty, ctx.helper_flags);
            validate_expr(ctx, func, collection, fn_name);
        }
        ExprKind::MapLen { map } => {
            track_collection_flags(&map.ty, ctx.helper_flags);
            validate_expr(ctx, func, map, fn_name);
        }
        ExprKind::MapEntryAt { map, index } => {
            track_collection_flags(&map.ty, ctx.helper_flags);
            validate_expr(ctx, func, map, fn_name);
            validate_expr(ctx, func, index, fn_name);
        }
        ExprKind::CollectionMut { object, args, .. } => {
            let obj_ty = &func.locals[object.0 as usize].ty;
            track_collection_flags(obj_ty, ctx.helper_flags);
            for arg in args {
                validate_expr(ctx, func, arg, fn_name);
            }
        }
        ExprKind::SortBy { .. } => {
            ctx.errors.push(format!(
                "Rust backend: unsupported sort in function `{fn_name}`"
            ));
        }
        ExprKind::UnwrapOptional(_) => {
            ctx.errors.push(format!(
                "Rust backend: unsupported optional unwrap in function `{fn_name}`"
            ));
        }
    }
}

fn validate_type(ty: &Type, fn_name: &str, errors: &mut Vec<String>) {
    match ty {
        Type::Int
        | Type::Float
        | Type::Double
        | Type::Bool
        | Type::String
        | Type::Void
        | Type::UnresolvedName(_) => {}
        Type::Struct { type_args, .. } if type_args.is_empty() => {}
        Type::Enum { type_args, .. } => {
            for arg in type_args {
                validate_type(arg, fn_name, errors);
            }
        }
        Type::Tuple(elems) => {
            for e in elems {
                validate_type(e, fn_name, errors);
            }
        }
        Type::Struct { name, .. } | Type::DataRef { name, .. } | Type::Extern { name } => {
            errors.push(format!(
                "Rust backend: unsupported type `{name}` in function `{fn_name}`"
            ));
        }
        Type::List { elem } | Type::Array { elem, .. } | Type::ArrayView { elem } => {
            validate_type(elem, fn_name, errors);
        }
        Type::Map { key, value } => {
            validate_type(key, fn_name, errors);
            validate_type(value, fn_name, errors);
        }
        Type::Func { .. } | Type::NamedTuple(_) | Type::Var(_) | Type::Infer | Type::Any => {
            errors.push(format!(
                "Rust backend: unsupported type `{ty}` in function `{fn_name}`"
            ));
        }
    }
}

fn track_type_usage(
    program: &hir::Program,
    ty: &Type,
    used_aggregates: &mut BTreeSet<u32>,
    used_enums: &mut BTreeSet<u32>,
) {
    match ty {
        Type::Struct { name, .. } => {
            if let Some((id, _)) = program.find_aggregate_by_name(&name.to_string()) {
                used_aggregates.insert(id);
            }
        }
        Type::Enum { name, .. } => {
            let name_str = name.to_string();
            if let Some((i, _)) = program
                .enum_meta
                .iter()
                .enumerate()
                .find(|(_, m)| m.name == name_str)
            {
                used_enums.insert(i as u32);
            }
        }
        Type::Tuple(elems) => {
            for e in elems {
                track_type_usage(program, e, used_aggregates, used_enums);
            }
        }
        Type::List { elem } | Type::Array { elem, .. } | Type::ArrayView { elem } => {
            track_type_usage(program, elem, used_aggregates, used_enums);
        }
        Type::Map { key, value } => {
            track_type_usage(program, key, used_aggregates, used_enums);
            track_type_usage(program, value, used_aggregates, used_enums);
        }
        _ => {}
    }
}

fn track_collection_flags(ty: &Type, flags: &mut HelperFlags) {
    match ty {
        Type::List { .. } | Type::Array { .. } | Type::ArrayView { .. } => {
            flags.enable(Feature::UsesCollections);
        }
        Type::Map { .. } => {
            flags.enable(Feature::UsesCollections);
            flags.enable(Feature::UsesMap);
        }
        Type::Tuple(elems) => {
            for e in elems {
                track_collection_flags(e, flags);
            }
        }
        Type::Enum { type_args, .. } | Type::Struct { type_args, .. } => {
            for arg in type_args {
                track_collection_flags(arg, flags);
            }
        }
        _ => {}
    }
}
