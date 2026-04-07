use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
};

use crate::{
    ast::{BinaryOp, FormatAlign, FormatKind, FormatSign, FormatSpec, Type, UnaryOp},
    backend_names,
    builtin::Builtin,
    hir::{self, CollectionMethod, ExprKind, LocalId, Ownership, StmtKind},
    ir_meta::{AggregateMeta, EnumMeta, VariantMeta, VariantShape},
    prelude_enums::OPTION_TYPE_ID,
};

struct WriteBackEntry {
    original_name: String,
    kind: WriteBackKind,
}

enum WriteBackKind {
    Field {
        field_index: u16,
        variant: u16,
        scrutinee_ty: Type,
    },
    Whole,
}

type WriteBackMap = HashMap<LocalId, WriteBackEntry>;

pub fn emit(program: &hir::Program, plan: &super::validate::RustPlan) -> Result<String, String> {
    let mut out = String::new();
    writeln!(out, "#![allow(unused, non_snake_case)]").unwrap();
    writeln!(out).unwrap();

    let main_fn = &program.funcs[plan.main.0 as usize];

    writeln!(out, "fn main() {{").unwrap();

    for &type_id in &plan.used_aggregates {
        let meta = &program.aggregate_meta[type_id as usize];
        emit_struct_def(&mut out, meta, 1)?;
    }

    for &type_id in &plan.used_enums {
        let is_opt = type_id == OPTION_TYPE_ID;
        if is_opt {
            // use rust native Option instead of our custom Option
            continue;
        }
        let meta = &program.enum_meta[type_id as usize];
        emit_enum_def(&mut out, meta, 1)?;
    }

    let mut float_temp = |ty_name: &str| {
        writeln!(out, "    fn __fmt_{ty_name}(v: {ty_name}) -> String {{").unwrap();
        writeln!(out, "        if v.fract() == 0.0 && v.is_finite() {{ format!(\"{{v:.1}}\") }} else {{ format!(\"{{v}}\") }}").unwrap();
        writeln!(out, "    }}").unwrap();
    };

    if plan.helper_flags.fmt_f32() {
        float_temp("f32");
    }
    if plan.helper_flags.fmt_f64() {
        float_temp("f64");
    }

    if plan.helper_flags.uses_map() {
        writeln!(out, "    use std::collections::HashMap;").unwrap();
    }

    if plan.helper_flags.uses_collections() {
        out.push_str(concat!(
            "    fn __check_index(len: usize, idx: i64) -> usize {\n",
            "        if idx < 0 || idx as usize >= len {\n",
            "            panic!(\"index {idx} out of bounds for length {len}\");\n",
            "        }\n",
            "        idx as usize\n",
            "    }\n",
            "    fn __check_slice(len: usize, start: i64, end: i64) -> std::ops::Range<usize> {\n",
            "        let s = if start < 0 { 0 } else { start as usize };\n",
            "        let e = if end < 0 { 0 } else { end as usize };\n",
            "        if s > e || e > len {\n",
            "            panic!(\"slice {s}..{e} out of bounds for length {len}\");\n",
            "        }\n",
            "        s..e\n",
            "    }\n",
        ));
    }

    for &type_id in &plan.helper_flags.stringify_aggregates {
        let meta = &program.aggregate_meta[type_id as usize];
        super::stringify::emit_struct_fmt_helper(&mut out, program, meta, 1)?;
    }

    for &type_id in &plan.helper_flags.stringify_enums {
        let meta = &program.enum_meta[type_id as usize];
        super::stringify::emit_enum_fmt_helper(&mut out, program, meta, 1)?;
    }

    for &func_id in &plan.funcs {
        let func = &program.funcs[func_id.0 as usize];
        emit_inner_fn(&mut out, program, func, 1)?;
    }

    emit_temp_declarations(&mut out, main_fn, 1)?;
    emit_block(
        &mut out,
        program,
        main_fn,
        &main_fn.body,
        1,
        &HashMap::new(),
    )?;
    writeln!(out, "}}").unwrap();

    Ok(out)
}

fn emit_inner_fn(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    indent: usize,
) -> Result<(), String> {
    write_indent(out, indent);
    write!(out, "fn {}(", mangle_func_name(&func.name.to_string())).unwrap();

    for i in 0..func.params_len as usize {
        if i > 0 {
            write!(out, ", ").unwrap();
        }
        let name = local_name(func, LocalId(i as u32));
        let local = &func.locals[i];
        let ty = emit_type(&local.ty)?;
        if local.is_ref {
            write!(out, "mut {name}: &mut {ty}").unwrap();
        } else {
            write!(out, "mut {name}: {ty}").unwrap();
        }
    }
    write!(out, ")").unwrap();

    if func.ret != Type::Void {
        let ret_ty = emit_type(&func.ret)?;
        write!(out, " -> {ret_ty}").unwrap();
    }
    writeln!(out, " {{").unwrap();

    emit_temp_declarations(out, func, indent + 1)?;
    emit_block(out, program, func, &func.body, indent + 1, &HashMap::new())?;

    write_indent(out, indent);
    writeln!(out, "}}").unwrap();
    Ok(())
}

fn emit_struct_def(out: &mut String, meta: &AggregateMeta, indent: usize) -> Result<(), String> {
    write_indent(out, indent);
    writeln!(out, "#[derive(Clone)]").unwrap();
    write_indent(out, indent);
    write!(out, "struct {} {{", meta.name).unwrap();
    if meta.fields.is_empty() {
        writeln!(out, "}}").unwrap();
    } else {
        writeln!(out).unwrap();
        for field in &meta.fields {
            write_indent(out, indent + 1);
            writeln!(out, "{}: {},", field.name, emit_type(&field.ty)?).unwrap();
        }
        write_indent(out, indent);
        writeln!(out, "}}").unwrap();
    }
    Ok(())
}

fn emit_enum_def(out: &mut String, meta: &EnumMeta, indent: usize) -> Result<(), String> {
    write_indent(out, indent);
    writeln!(out, "#[derive(Clone)]").unwrap();
    write_indent(out, indent);
    writeln!(out, "enum {} {{", meta.name).unwrap();
    for variant in &meta.variants {
        write_indent(out, indent + 1);
        match &variant.shape {
            VariantShape::Unit => writeln!(out, "{},", variant.name).unwrap(),
            VariantShape::Tuple(types) => {
                let type_strs: Vec<String> =
                    types.iter().map(emit_type).collect::<Result<_, _>>()?;
                writeln!(out, "{}({}),", variant.name, type_strs.join(", ")).unwrap();
            }
            VariantShape::Struct(fields) => {
                write!(out, "{} {{ ", variant.name).unwrap();
                for (j, field) in fields.iter().enumerate() {
                    if j > 0 {
                        write!(out, ", ").unwrap();
                    }
                    write!(out, "{}: {}", field.name, emit_type(&field.ty)?).unwrap();
                }
                writeln!(out, " }},").unwrap();
            }
        }
    }
    write_indent(out, indent);
    writeln!(out, "}}").unwrap();
    Ok(())
}

fn emit_block(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    block: &hir::Block,
    indent: usize,
    wb: &WriteBackMap,
) -> Result<(), String> {
    for stmt in &block.stmts {
        emit_stmt(out, program, func, stmt, indent, wb)?;
    }
    Ok(())
}

fn emit_stmt(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    stmt: &hir::Stmt,
    indent: usize,
    wb: &WriteBackMap,
) -> Result<(), String> {
    match &stmt.kind {
        StmtKind::Let { local, init } => {
            let name = local_name(func, *local);
            let info = &func.locals[local.0 as usize];
            let ty = emit_type(&info.ty)?;
            write_indent(out, indent);
            write!(out, "let mut {name}: {ty} = ").unwrap();
            emit_expr(out, program, func, init)?;
            writeln!(out, ";").unwrap();
            Ok(())
        }
        StmtKind::Assign { local, value } => {
            let name = local_name(func, *local);
            write_indent(out, indent);
            write!(out, "{name} = ").unwrap();
            emit_expr(out, program, func, value)?;
            writeln!(out, ";").unwrap();
            if let Some(entry) = wb.get(local) {
                let binding_ty = &func.locals[local.0 as usize].ty;
                emit_write_back(out, program, &name, binding_ty, entry, indent)?;
            }
            Ok(())
        }
        StmtKind::Expr(expr) => emit_expr_stmt(out, program, func, expr, indent),
        StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            write_indent(out, indent);
            write!(out, "if ").unwrap();
            emit_expr(out, program, func, cond)?;
            writeln!(out, " {{").unwrap();
            emit_block(out, program, func, then_block, indent + 1, wb)?;
            if let Some(b) = else_block {
                write_indent(out, indent);
                writeln!(out, "}} else {{").unwrap();
                emit_block(out, program, func, b, indent + 1, wb)?;
                write_indent(out, indent);
                writeln!(out, "}}").unwrap();
            } else {
                write_indent(out, indent);
                writeln!(out, "}}").unwrap();
            }
            Ok(())
        }
        StmtKind::While { cond, body } => {
            write_indent(out, indent);
            write!(out, "while ").unwrap();
            emit_expr(out, program, func, cond)?;
            writeln!(out, " {{").unwrap();
            emit_block(out, program, func, body, indent + 1, wb)?;
            write_indent(out, indent);
            writeln!(out, "}}").unwrap();
            Ok(())
        }
        StmtKind::Break => {
            write_indent(out, indent);
            writeln!(out, "break;").unwrap();
            Ok(())
        }
        StmtKind::Continue => {
            write_indent(out, indent);
            writeln!(out, "continue;").unwrap();
            Ok(())
        }
        StmtKind::Return(Some(expr)) => {
            write_indent(out, indent);
            write!(out, "return ").unwrap();
            emit_expr(out, program, func, expr)?;
            writeln!(out, ";").unwrap();
            Ok(())
        }
        StmtKind::Return(None) => {
            write_indent(out, indent);
            writeln!(out, "return;").unwrap();
            Ok(())
        }
        StmtKind::SetField {
            object,
            field_index,
            value,
        } => {
            let obj_local = &func.locals[object.0 as usize];
            let field_name = resolve_field_name(program, &obj_local.ty, *field_index)?;
            write_indent(out, indent);
            let obj_name = local_name(func, *object);
            write!(out, "{obj_name}.{field_name} = ").unwrap();
            emit_expr(out, program, func, value)?;
            writeln!(out, ";").unwrap();
            if let Some(entry) = wb.get(object) {
                let binding_ty = &func.locals[object.0 as usize].ty;
                emit_write_back(out, program, &obj_name, binding_ty, entry, indent)?;
            }
            Ok(())
        }
        StmtKind::SetIndex {
            object,
            index,
            value,
        } => {
            let obj_ty = &func.locals[object.0 as usize].ty;
            let obj_name = local_name(func, *object);
            write_indent(out, indent);
            if obj_ty.is_map() {
                write!(out, "{obj_name}.insert(").unwrap();
                emit_expr(out, program, func, index)?;
                write!(out, ", ").unwrap();
                emit_expr(out, program, func, value)?;
                writeln!(out, ");").unwrap();
            } else {
                write!(out, "{{ let __idx = __check_index({obj_name}.len(), ").unwrap();
                emit_expr(out, program, func, index)?;
                write!(out, "); {obj_name}[__idx] = ").unwrap();
                emit_expr(out, program, func, value)?;
                writeln!(out, "; }}").unwrap();
            }
            Ok(())
        }
        StmtKind::Match {
            scrutinee_init,
            scrutinee,
            write_through,
            arms,
            else_body,
        } => {
            let scrutinee_name = local_name(func, *scrutinee);
            write_indent(out, indent);
            write!(out, "{scrutinee_name} = ").unwrap();
            emit_expr(out, program, func, scrutinee_init)?;
            writeln!(out, ";").unwrap();

            let scrutinee_ty = &func.locals[scrutinee.0 as usize].ty;

            if scrutinee_ty.is_option() {
                emit_option_match(
                    out,
                    program,
                    func,
                    &scrutinee_name,
                    *scrutinee,
                    write_through.as_ref(),
                    arms,
                    else_body.as_ref(),
                    indent,
                )?;
                return Ok(());
            }

            let enum_meta = resolve_enum_meta(program, scrutinee_ty)?;

            write_indent(out, indent);
            writeln!(out, "match {scrutinee_name} {{").unwrap();

            for arm in arms {
                let arm_wb = build_arm_write_back(func, write_through.as_ref(), arm, scrutinee_ty);
                emit_match_arm(
                    out,
                    program,
                    func,
                    enum_meta,
                    arm,
                    *scrutinee,
                    indent + 1,
                    &arm_wb,
                )?;
            }

            if let Some(else_arm) = else_body {
                let else_wb = build_else_write_back(func, write_through.as_ref(), else_arm);
                emit_match_else(out, program, func, else_arm, indent + 1, &else_wb)?;
            }

            write_indent(out, indent);
            writeln!(out, "}}").unwrap();
            Ok(())
        }
    }
}

fn emit_expr_stmt(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    expr: &hir::Expr,
    indent: usize,
) -> Result<(), String> {
    write_indent(out, indent);
    emit_expr(out, program, func, expr)?;
    writeln!(out, ";").unwrap();
    Ok(())
}

pub(super) fn emit_expr(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    expr: &hir::Expr,
) -> Result<(), String> {
    match &expr.kind {
        ExprKind::Local(id) => {
            let name = local_name(func, *id);
            write!(out, "{name}").unwrap();
            if expr.ownership != Ownership::Move && !is_copy_type(&func.locals[id.0 as usize].ty) {
                write!(out, ".clone()").unwrap();
            }
        }
        ExprKind::Int(v) => write!(out, "{v}_i64").unwrap(),
        ExprKind::Float(v) => write!(out, "{v:?}_f32").unwrap(),
        ExprKind::Double(v) => write!(out, "{v:?}_f64").unwrap(),
        ExprKind::Bool(v) => write!(out, "{v}").unwrap(),
        ExprKind::String(s) => write!(out, "String::from(\"{}\")", escape_str(s)).unwrap(),
        ExprKind::Nil => {
            if expr.ty.is_option() {
                write!(out, "None").unwrap();
            } else {
                write!(out, "()").unwrap();
            }
        }
        ExprKind::ToString(inner) => {
            super::stringify::emit_to_anvyx_string(out, program, func, inner)?;
        }
        ExprKind::Binary { op, lhs, rhs } => {
            emit_binary(out, program, func, *op, lhs, rhs, &expr.ty)?;
        }
        ExprKind::Unary { op, expr: inner } => {
            let op_str = match op {
                UnaryOp::Neg => "-",
                UnaryOp::Not | UnaryOp::BitNot => "!",
            };
            write!(out, "({op_str}").unwrap();
            emit_expr(out, program, func, inner)?;
            write!(out, ")").unwrap();
        }
        ExprKind::Call {
            func: func_id,
            args,
            ref_mask,
        } => {
            let callee = &program.funcs[func_id.0 as usize];
            write!(out, "{}(", mangle_func_name(&callee.name.to_string())).unwrap();
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                if ref_mask[i] {
                    write!(out, "&mut ").unwrap();
                    // Bypass clone logic — we're passing by &mut, so no copy needed
                    if let ExprKind::Local(id) = &arg.kind {
                        write!(out, "{}", local_name(func, *id)).unwrap();
                    } else {
                        emit_expr(out, program, func, arg)?;
                    }
                } else {
                    emit_expr(out, program, func, arg)?;
                }
            }
            write!(out, ")").unwrap();
        }
        ExprKind::CallBuiltin { builtin, args } => {
            emit_builtin_expr(out, program, func, *builtin, args)?;
        }
        ExprKind::Cast(inner) => {
            let target = emit_type(&expr.ty)?;
            write!(out, "(").unwrap();
            emit_expr(out, program, func, inner)?;
            write!(out, " as {target})").unwrap();
        }
        ExprKind::StructLiteral { type_id, fields } => {
            let meta = &program.aggregate_meta[*type_id as usize];
            write!(out, "{} {{ ", meta.name).unwrap();
            for (i, (field_meta, field_expr)) in meta.fields.iter().zip(fields).enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{}: ", field_meta.name).unwrap();
                emit_expr(out, program, func, field_expr)?;
            }
            write!(out, " }}").unwrap();
        }
        ExprKind::FieldGet { object, index } => {
            let field_name = resolve_field_name(program, &object.ty, *index)?;
            emit_expr(out, program, func, object)?;
            write!(out, ".{field_name}").unwrap();
        }
        ExprKind::EnumLiteral {
            type_id,
            variant,
            fields,
        } => {
            if *type_id == OPTION_TYPE_ID {
                if *variant == 1 {
                    write!(out, "Some(").unwrap();
                    emit_expr(out, program, func, &fields[0])?;
                    write!(out, ")").unwrap();
                } else {
                    write!(out, "None").unwrap();
                }
                return Ok(());
            }
            let meta = &program.enum_meta[*type_id as usize];
            let variant_meta = &meta.variants[*variant as usize];
            write!(out, "{}::{}", meta.name, variant_meta.name).unwrap();
            match &variant_meta.shape {
                VariantShape::Unit => {}
                VariantShape::Tuple(_) => {
                    write!(out, "(").unwrap();
                    for (i, field) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(out, ", ").unwrap();
                        }
                        emit_expr(out, program, func, field)?;
                    }
                    write!(out, ")").unwrap();
                }
                VariantShape::Struct(field_defs) => {
                    write!(out, " {{ ").unwrap();
                    for (i, (fmeta, field)) in field_defs.iter().zip(fields).enumerate() {
                        if i > 0 {
                            write!(out, ", ").unwrap();
                        }
                        write!(out, "{}: ", fmeta.name).unwrap();
                        emit_expr(out, program, func, field)?;
                    }
                    write!(out, " }}").unwrap();
                }
            }
        }
        ExprKind::TupleLiteral { elements } => {
            write!(out, "(").unwrap();
            for (i, elem) in elements.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                emit_expr(out, program, func, elem)?;
            }
            write!(out, ")").unwrap();
        }
        ExprKind::TupleIndex { tuple, index } => {
            emit_expr(out, program, func, tuple)?;
            write!(out, ".{index}").unwrap();
        }
        ExprKind::Format(inner, spec) => {
            let is_default = spec.align.is_none()
                && spec.sign == FormatSign::Default
                && !spec.zero_pad
                && spec.width.is_none()
                && spec.precision.is_none()
                && spec.kind == FormatKind::Default;

            if is_default {
                super::stringify::emit_to_anvyx_string(out, program, func, inner)?;
            } else {
                let fmt = build_format_spec(spec);
                write!(out, "format!(\"{fmt}\", ").unwrap();
                emit_expr(out, program, func, inner)?;
                write!(out, ")").unwrap();
            }
        }
        ExprKind::ArrayLiteral { elements } | ExprKind::ListLiteral { elements } => {
            write!(out, "vec![").unwrap();
            for (i, elem) in elements.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                emit_expr(out, program, func, elem)?;
            }
            write!(out, "]").unwrap();
        }
        ExprKind::ArrayFill { value, len } | ExprKind::ListFill { value, len } => {
            write!(out, "vec![").unwrap();
            emit_expr(out, program, func, value)?;
            write!(out, "; {len}]").unwrap();
        }
        ExprKind::MapLiteral { entries } => {
            if entries.is_empty() {
                write!(out, "HashMap::new()").unwrap();
            } else {
                write!(out, "{{ let mut __m = HashMap::new(); ").unwrap();
                for (k, v) in entries {
                    write!(out, "__m.insert(").unwrap();
                    emit_expr(out, program, func, k)?;
                    write!(out, ", ").unwrap();
                    emit_expr(out, program, func, v)?;
                    write!(out, "); ").unwrap();
                }
                write!(out, "__m }}").unwrap();
            }
        }
        ExprKind::IndexGet { target, index } => {
            if target.ty.is_map() {
                emit_target_no_clone(out, program, func, target)?;
                write!(out, ".get(&").unwrap();
                emit_expr(out, program, func, index)?;
                write!(out, ").cloned()").unwrap();
            } else {
                emit_target_no_clone(out, program, func, target)?;
                write!(out, "[__check_index(").unwrap();
                emit_target_no_clone(out, program, func, target)?;
                write!(out, ".len(), ").unwrap();
                emit_expr(out, program, func, index)?;
                write!(out, ")]").unwrap();
                if !is_copy_type(&expr.ty) {
                    write!(out, ".clone()").unwrap();
                }
            }
        }
        ExprKind::Slice {
            target,
            start,
            end,
            inclusive,
        } => {
            emit_target_no_clone(out, program, func, target)?;
            write!(out, "[__check_slice(").unwrap();
            emit_target_no_clone(out, program, func, target)?;
            write!(out, ".len(), ").unwrap();
            emit_expr(out, program, func, start)?;
            write!(out, ", ").unwrap();
            if *inclusive {
                write!(out, "(").unwrap();
                emit_expr(out, program, func, end)?;
                write!(out, ") + 1").unwrap();
            } else {
                emit_expr(out, program, func, end)?;
            }
            write!(out, ")].to_vec()").unwrap();
        }
        ExprKind::CollectionLen { collection } | ExprKind::MapLen { map: collection } => {
            write!(out, "(").unwrap();
            emit_target_no_clone(out, program, func, collection)?;
            write!(out, ".len() as i64)").unwrap();
        }
        ExprKind::MapEntryAt { map, index } => {
            write!(out, "{{ let __e = ").unwrap();
            emit_target_no_clone(out, program, func, map)?;
            write!(out, ".iter().nth(").unwrap();
            emit_expr(out, program, func, index)?;
            write!(
                out,
                " as usize).unwrap(); (__e.0.clone(), __e.1.clone()) }}"
            )
            .unwrap();
        }
        ExprKind::CollectionMut {
            object,
            method,
            args,
        } => {
            let obj_name = local_name(func, *object);
            match method {
                CollectionMethod::ListPush => {
                    write!(out, "{obj_name}.push(").unwrap();
                    emit_expr(out, program, func, &args[0])?;
                    write!(out, ")").unwrap();
                }
                CollectionMethod::ListPop => {
                    write!(out, "{obj_name}.pop()").unwrap();
                }
                CollectionMethod::MapInsert => {
                    write!(out, "{obj_name}.insert(").unwrap();
                    emit_expr(out, program, func, &args[0])?;
                    write!(out, ", ").unwrap();
                    emit_expr(out, program, func, &args[1])?;
                    write!(out, ")").unwrap();
                }
                CollectionMethod::MapRemove => {
                    write!(out, "{obj_name}.remove(&").unwrap();
                    emit_expr(out, program, func, &args[0])?;
                    write!(out, ")").unwrap();
                }
            }
        }
        other => {
            return Err(format!(
                "Rust backend: unsupported expression kind: {other:?}"
            ));
        }
    }
    Ok(())
}

fn resolve_enum_meta<'a>(program: &'a hir::Program, ty: &Type) -> Result<&'a EnumMeta, String> {
    match ty {
        Type::Enum { name, .. } => program
            .enum_meta
            .iter()
            .find(|m| m.name == name.to_string())
            .ok_or_else(|| format!("Rust backend: unknown enum {name}")),
        other => Err(format!("Rust backend: match on non-enum type: {other:?}")),
    }
}

fn emit_match_arm(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    enum_meta: &EnumMeta,
    arm: &hir::MatchArm,
    scrutinee: LocalId,
    indent: usize,
    wb: &WriteBackMap,
) -> Result<(), String> {
    let variant_meta = &enum_meta.variants[arm.variant as usize];
    let enum_name = &enum_meta.name;

    let guard_bindings = if arm.guard.is_some() {
        build_guard_field_map(variant_meta)
    } else {
        HashMap::new()
    };

    write_indent(out, indent);
    write!(out, "{enum_name}::{}", variant_meta.name).unwrap();
    emit_variant_pattern(out, func, variant_meta, &arm.bindings, &guard_bindings);

    if let Some(guard) = &arm.guard {
        write!(out, " if ").unwrap();
        emit_guard_expr(out, program, func, guard, scrutinee, &guard_bindings)?;
    }

    writeln!(out, " => {{").unwrap();
    emit_block(out, program, func, &arm.body, indent + 1, wb)?;
    write_indent(out, indent);
    writeln!(out, "}}").unwrap();
    Ok(())
}

fn emit_match_else(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    else_arm: &hir::MatchElse,
    indent: usize,
    wb: &WriteBackMap,
) -> Result<(), String> {
    write_indent(out, indent);
    match &else_arm.binding {
        Some((local, mutable)) => {
            let name = local_name(func, *local);
            if *mutable {
                write!(out, "mut ").unwrap();
            }
            write!(out, "{name}").unwrap();
        }
        None => write!(out, "_").unwrap(),
    }
    writeln!(out, " => {{").unwrap();
    emit_block(out, program, func, &else_arm.body, indent + 1, wb)?;
    write_indent(out, indent);
    writeln!(out, "}}").unwrap();
    Ok(())
}

fn emit_option_match(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    scrutinee_name: &str,
    scrutinee: LocalId,
    write_through: Option<&hir::MatchWriteThrough>,
    arms: &[hir::MatchArm],
    else_body: Option<&hir::MatchElse>,
    indent: usize,
) -> Result<(), String> {
    let scrutinee_ty = &func.locals[scrutinee.0 as usize].ty;
    write_indent(out, indent);
    writeln!(out, "match {scrutinee_name} {{").unwrap();

    for arm in arms {
        let arm_wb = build_arm_write_back(func, write_through, arm, scrutinee_ty);
        write_indent(out, indent + 1);
        if arm.variant == 1 {
            // Some variant
            let has_guard = arm.guard.is_some();
            if let Some(binding) = arm.bindings.first() {
                let name = local_name(func, binding.local);
                if binding.mutable {
                    write!(out, "Some(mut {name})").unwrap();
                } else {
                    write!(out, "Some({name})").unwrap();
                }
            } else if has_guard {
                write!(out, "Some(__g0)").unwrap();
            } else {
                write!(out, "Some(_)").unwrap();
            }
            if let Some(guard) = &arm.guard {
                let mut field_map = HashMap::new();
                field_map.insert(0u16, "__g0".to_string());
                write!(out, " if ").unwrap();
                emit_guard_expr(out, program, func, guard, scrutinee, &field_map)?;
            }
        } else {
            // None variant
            write!(out, "None").unwrap();
        }
        writeln!(out, " => {{").unwrap();
        emit_block(out, program, func, &arm.body, indent + 2, &arm_wb)?;
        write_indent(out, indent + 1);
        writeln!(out, "}}").unwrap();
    }

    if let Some(else_arm) = else_body {
        let else_wb = build_else_write_back(func, write_through, else_arm);
        emit_match_else(out, program, func, else_arm, indent + 1, &else_wb)?;
    }

    write_indent(out, indent);
    writeln!(out, "}}").unwrap();
    Ok(())
}

fn build_arm_write_back(
    func: &hir::Func,
    write_through: Option<&hir::MatchWriteThrough>,
    arm: &hir::MatchArm,
    scrutinee_ty: &Type,
) -> WriteBackMap {
    let mut wb = WriteBackMap::new();
    let Some(wt) = write_through else {
        return wb;
    };
    let original_name = local_name(func, wt.original);
    for binding in &arm.bindings {
        if binding.mutable {
            wb.insert(
                binding.local,
                WriteBackEntry {
                    original_name: original_name.clone(),
                    kind: WriteBackKind::Field {
                        field_index: binding.field_index,
                        variant: arm.variant,
                        scrutinee_ty: scrutinee_ty.clone(),
                    },
                },
            );
        }
    }
    wb
}

fn build_else_write_back(
    func: &hir::Func,
    write_through: Option<&hir::MatchWriteThrough>,
    else_arm: &hir::MatchElse,
) -> WriteBackMap {
    let mut wb = WriteBackMap::new();
    let Some(wt) = write_through else {
        return wb;
    };
    if let Some((binding_local, true)) = else_arm.binding {
        wb.insert(
            binding_local,
            WriteBackEntry {
                original_name: local_name(func, wt.original),
                kind: WriteBackKind::Whole,
            },
        );
    }
    wb
}

fn emit_write_back(
    out: &mut String,
    program: &hir::Program,
    binding_name: &str,
    binding_ty: &Type,
    entry: &WriteBackEntry,
    indent: usize,
) -> Result<(), String> {
    let original = &entry.original_name;
    let clone_suffix = if is_copy_type(binding_ty) {
        ""
    } else {
        ".clone()"
    };
    match &entry.kind {
        WriteBackKind::Whole => {
            write_indent(out, indent);
            writeln!(out, "{original} = {binding_name}{clone_suffix};").unwrap();
            Ok(())
        }
        WriteBackKind::Field {
            field_index,
            variant,
            scrutinee_ty,
        } => emit_field_write_back(
            out,
            program,
            original,
            binding_name,
            clone_suffix,
            *field_index,
            *variant,
            scrutinee_ty,
            indent,
        ),
    }
}

fn emit_field_write_back(
    out: &mut String,
    program: &hir::Program,
    original_name: &str,
    binding_name: &str,
    clone_suffix: &str,
    field_index: u16,
    variant: u16,
    scrutinee_ty: &Type,
    indent: usize,
) -> Result<(), String> {
    if scrutinee_ty.is_option() {
        write_indent(out, indent);
        writeln!(out, "if let Some(ref mut __wb) = {original_name} {{ *__wb = {binding_name}{clone_suffix}; }}").unwrap();
        return Ok(());
    }

    let enum_meta = resolve_enum_meta(program, scrutinee_ty)?;
    let variant_meta = &enum_meta.variants[variant as usize];
    let enum_name = &enum_meta.name;

    match &variant_meta.shape {
        VariantShape::Unit => Ok(()),
        VariantShape::Tuple(types) => {
            write_indent(out, indent);
            write!(out, "if let {enum_name}::{}(", variant_meta.name).unwrap();
            for i in 0..types.len() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                if i as u16 == field_index {
                    write!(out, "ref mut __wb").unwrap();
                } else {
                    write!(out, "_").unwrap();
                }
            }
            writeln!(
                out,
                ") = {original_name} {{ *__wb = {binding_name}{clone_suffix}; }}"
            )
            .unwrap();
            Ok(())
        }
        VariantShape::Struct(fields) => {
            let field_name = &fields[field_index as usize].name;
            write_indent(out, indent);
            writeln!(out, "if let {enum_name}::{} {{ {field_name}: ref mut __wb, .. }} = {original_name} {{ *__wb = {binding_name}{clone_suffix}; }}", variant_meta.name).unwrap();
            Ok(())
        }
    }
}

fn build_guard_field_map(variant_meta: &VariantMeta) -> HashMap<u16, String> {
    let mut map = HashMap::new();
    match &variant_meta.shape {
        VariantShape::Unit => {}
        VariantShape::Tuple(types) => {
            for i in 0..types.len() {
                map.insert(i as u16, format!("__g{i}"));
            }
        }
        VariantShape::Struct(fields) => {
            for (i, field) in fields.iter().enumerate() {
                map.insert(i as u16, format!("__g_{}", field.name));
            }
        }
    }
    map
}

fn emit_variant_pattern(
    out: &mut String,
    func: &hir::Func,
    variant_meta: &VariantMeta,
    bindings: &[hir::MatchBinding],
    guard_bindings: &HashMap<u16, String>,
) {
    match &variant_meta.shape {
        VariantShape::Unit => {}
        VariantShape::Tuple(types) => {
            write!(out, "(").unwrap();
            for i in 0..types.len() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                if let Some(b) = bindings.iter().find(|b| b.field_index == i as u16) {
                    let name = local_name(func, b.local);
                    if b.mutable {
                        write!(out, "mut ").unwrap();
                    }
                    write!(out, "{name}").unwrap();
                } else if let Some(syn) = guard_bindings.get(&(i as u16)) {
                    write!(out, "{syn}").unwrap();
                } else {
                    write!(out, "_").unwrap();
                }
            }
            write!(out, ")").unwrap();
        }
        VariantShape::Struct(fields) => {
            write!(out, " {{ ").unwrap();
            let mut first = true;
            for b in bindings {
                if !first {
                    write!(out, ", ").unwrap();
                }
                first = false;
                let field_name = &fields[b.field_index as usize].name;
                let name = local_name(func, b.local);
                if b.mutable {
                    write!(out, "mut ").unwrap();
                }
                write!(out, "{field_name}: {name}").unwrap();
            }
            for (idx, syn) in guard_bindings {
                if bindings.iter().any(|b| b.field_index == *idx) {
                    continue;
                }
                if !first {
                    write!(out, ", ").unwrap();
                }
                first = false;
                let field_name = &fields[*idx as usize].name;
                write!(out, "{field_name}: {syn}").unwrap();
            }
            let total_bound = bindings.len()
                + guard_bindings
                    .keys()
                    .filter(|idx| !bindings.iter().any(|b| b.field_index == **idx))
                    .count();
            if total_bound < fields.len() {
                if !first {
                    write!(out, ", ").unwrap();
                }
                write!(out, "..").unwrap();
            }
            write!(out, " }}").unwrap();
        }
    }
}

fn emit_guard_expr(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    expr: &hir::Expr,
    scrutinee: LocalId,
    field_map: &HashMap<u16, String>,
) -> Result<(), String> {
    match &expr.kind {
        ExprKind::FieldGet { object, index } => {
            if let ExprKind::Local(id) = &object.kind
                && *id == scrutinee
                && let Some(syn) = field_map.get(index)
            {
                write!(out, "{syn}").unwrap();
                return Ok(());
            }
            emit_expr(out, program, func, expr)
        }
        ExprKind::Binary { op, lhs, rhs } => {
            let op_str = match op {
                BinaryOp::Eq => "==",
                BinaryOp::NotEq => "!=",
                BinaryOp::LessThan => "<",
                BinaryOp::LessThanEq => "<=",
                BinaryOp::GreaterThan => ">",
                BinaryOp::GreaterThanEq => ">=",
                BinaryOp::And => "&&",
                BinaryOp::Or => "||",
                other => return Err(format!("Rust backend: unsupported guard op: {other:?}")),
            };
            write!(out, "(").unwrap();
            emit_guard_expr(out, program, func, lhs, scrutinee, field_map)?;
            write!(out, " {op_str} ").unwrap();
            emit_guard_expr(out, program, func, rhs, scrutinee, field_map)?;
            write!(out, ")").unwrap();
            Ok(())
        }
        _ => emit_expr(out, program, func, expr),
    }
}

fn resolve_field_name(program: &hir::Program, ty: &Type, index: u16) -> Result<String, String> {
    match ty {
        Type::Struct { name, .. } | Type::DataRef { name, .. } | Type::UnresolvedName(name) => {
            let (_, meta) = program
                .find_aggregate_by_name(&name.to_string())
                .ok_or_else(|| format!("Rust backend: unknown struct {name}"))?;
            Ok(meta.fields[index as usize].name.clone())
        }
        other => Err(format!(
            "Rust backend: field access on non-struct type: {other:?}"
        )),
    }
}

fn emit_binary(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    op: BinaryOp,
    lhs: &hir::Expr,
    rhs: &hir::Expr,
    result_ty: &Type,
) -> Result<(), String> {
    if op == BinaryOp::Add && result_ty.is_str() {
        write!(out, "format!(\"{{}}{{}}\", ").unwrap();
        if lhs.ty.is_str() {
            emit_expr(out, program, func, lhs)?;
        } else {
            super::stringify::emit_to_anvyx_string(out, program, func, lhs)?;
        }
        write!(out, ", ").unwrap();
        if rhs.ty.is_str() {
            emit_expr(out, program, func, rhs)?;
        } else {
            super::stringify::emit_to_anvyx_string(out, program, func, rhs)?;
        }
        write!(out, ")").unwrap();
        return Ok(());
    }

    let op_str = match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Rem => "%",
        BinaryOp::Eq => "==",
        BinaryOp::NotEq => "!=",
        BinaryOp::LessThan => "<",
        BinaryOp::GreaterThan => ">",
        BinaryOp::LessThanEq => "<=",
        BinaryOp::GreaterThanEq => ">=",
        BinaryOp::And => "&&",
        BinaryOp::Or => "||",
        BinaryOp::Xor => "^",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitOr => "|",
        BinaryOp::Shl => "<<",
        BinaryOp::Shr => ">>",
        BinaryOp::Coalesce => {
            return Err("Rust backend: coalesce operator not yet supported".into());
        }
    };

    write!(out, "(").unwrap();
    emit_expr(out, program, func, lhs)?;
    write!(out, " {op_str} ").unwrap();
    emit_expr(out, program, func, rhs)?;
    write!(out, ")").unwrap();
    Ok(())
}

fn emit_builtin_expr(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    builtin: Builtin,
    args: &[hir::Expr],
) -> Result<(), String> {
    match builtin {
        Builtin::Println => {
            let can_use_display = matches!(args[0].ty, Type::Int | Type::Bool | Type::String);
            if can_use_display {
                write!(out, "println!(\"{{}}\", ").unwrap();
                emit_expr(out, program, func, &args[0])?;
                write!(out, ")").unwrap();
            } else {
                write!(out, "println!(\"{{}}\", ").unwrap();
                super::stringify::emit_to_anvyx_string(out, program, func, &args[0])?;
                write!(out, ")").unwrap();
            }
            Ok(())
        }
        Builtin::Assert => {
            write!(out, "assert!(").unwrap();
            emit_expr(out, program, func, &args[0])?;
            write!(out, ")").unwrap();
            Ok(())
        }
        Builtin::AssertMsg => {
            write!(out, "assert!(").unwrap();
            emit_expr(out, program, func, &args[0])?;
            write!(out, ", \"{{}}\", ").unwrap();
            emit_expr(out, program, func, &args[1])?;
            write!(out, ")").unwrap();
            Ok(())
        }
    }
}

fn build_format_spec(spec: &FormatSpec) -> String {
    let mut s = String::from("{:");
    if let Some(align) = &spec.align {
        s.push(spec.fill);
        s.push(match align {
            FormatAlign::Left => '<',
            FormatAlign::Right => '>',
            FormatAlign::Center => '^',
        });
    }
    if spec.sign == FormatSign::Always {
        s.push('+');
    }
    if spec.zero_pad {
        s.push('0');
    }
    if let Some(w) = spec.width {
        write!(s, "{w}").unwrap();
    }
    if let Some(p) = spec.precision {
        write!(s, ".{p}").unwrap();
    }
    match spec.kind {
        FormatKind::Default => {}
        FormatKind::Hex => s.push('x'),
        FormatKind::HexUpper => s.push('X'),
        FormatKind::Binary => s.push('b'),
        FormatKind::Exp => s.push('e'),
        FormatKind::ExpUpper => s.push('E'),
    }
    s.push('}');
    s
}

pub(super) fn emit_type(ty: &Type) -> Result<String, String> {
    match ty {
        Type::Int => Ok("i64".into()),
        Type::Float => Ok("f32".into()),
        Type::Double => Ok("f64".into()),
        Type::Bool => Ok("bool".into()),
        Type::String => Ok("String".into()),
        Type::Void => Ok("()".into()),
        Type::Struct { name, type_args } if type_args.is_empty() => Ok(name.to_string()),
        // UnresolvedName survives into HIR locals/return types for user-defined struct/enum names
        Type::UnresolvedName(name) => Ok(name.to_string()),
        Type::Enum { name, type_args } if type_args.is_empty() => Ok(name.to_string()),
        Type::Enum { name, type_args } => {
            let args: Vec<String> = type_args.iter().map(emit_type).collect::<Result<_, _>>()?;
            Ok(format!("{}<{}>", name, args.join(", ")))
        }
        Type::Tuple(types) => {
            let parts: Vec<String> = types.iter().map(emit_type).collect::<Result<_, _>>()?;
            Ok(format!("({})", parts.join(", ")))
        }
        Type::List { elem } | Type::Array { elem, .. } | Type::ArrayView { elem } => {
            let inner = emit_type(elem)?;
            Ok(format!("Vec<{inner}>"))
        }
        Type::Map { key, value } => {
            let k = emit_type(key)?;
            let v = emit_type(value)?;
            Ok(format!("HashMap<{k}, {v}>"))
        }
        other => Err(format!("Rust backend: unsupported type: {other:?}")),
    }
}

fn is_copy_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Int | Type::Float | Type::Double | Type::Bool | Type::Void
    )
}

fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            c => out.push(c),
        }
    }
    out
}

pub(super) fn write_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        write!(out, "    ").unwrap();
    }
}

fn local_name(func: &hir::Func, id: LocalId) -> String {
    match &func.locals[id.0 as usize].name {
        Some(name) if name.to_string() == "self" => "self_".to_string(),
        Some(name) => name.to_string(),
        None => format!("__tmp{}", id.0),
    }
}

/// Emit without "clone" when the caller needs the original local container
fn emit_target_no_clone(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    expr: &hir::Expr,
) -> Result<(), String> {
    if let ExprKind::Local(id) = &expr.kind {
        write!(out, "{}", local_name(func, *id)).unwrap();
        Ok(())
    } else {
        emit_expr(out, program, func, expr)
    }
}

pub(super) fn mangle_func_name(name: &str) -> String {
    backend_names::mangle_for_rust(name)
}

fn collect_let_locals(block: &hir::Block, set: &mut HashSet<LocalId>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { local, .. } => {
                set.insert(*local);
            }
            StmtKind::If {
                then_block,
                else_block,
                ..
            } => {
                collect_let_locals(then_block, set);
                if let Some(b) = else_block {
                    collect_let_locals(b, set);
                }
            }
            StmtKind::While { body, .. } => {
                collect_let_locals(body, set);
            }
            StmtKind::Match {
                arms, else_body, ..
            } => {
                for arm in arms {
                    collect_let_locals(&arm.body, set);
                }
                if let Some(e) = else_body {
                    collect_let_locals(&e.body, set);
                }
            }
            _ => {}
        }
    }
}

fn emit_temp_declarations(out: &mut String, func: &hir::Func, indent: usize) -> Result<(), String> {
    let mut let_locals = HashSet::new();
    collect_let_locals(&func.body, &mut let_locals);
    for (i, local) in func
        .locals
        .iter()
        .enumerate()
        .skip(func.params_len as usize)
    {
        let id = LocalId(i as u32);
        if local.name.is_none() && !let_locals.contains(&id) {
            let ty = emit_type(&local.ty)?;
            write_indent(out, indent);
            writeln!(out, "let mut __tmp{i}: {ty};").unwrap();
        }
    }
    Ok(())
}
