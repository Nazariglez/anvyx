use std::fmt::Write;

use super::emit::{emit_expr, mangle_func_name, write_indent};
use crate::{
    ast::Type,
    hir,
    ir_meta::{AggregateMeta, EnumMeta, VariantShape},
};

fn emit_tuple_fields_stringify(
    out: &mut String,
    program: &hir::Program,
    temp_var: &str,
    elems: &[Type],
) -> Result<(), String> {
    for (i, elem_ty) in elems.iter().enumerate() {
        if i > 0 {
            write!(out, " __s.push_str(\", \");").unwrap();
        }
        write!(out, " __s.push_str(&").unwrap();
        emit_stringify_var(out, program, &format!("{temp_var}.{i}"), elem_ty)?;
        write!(out, ");").unwrap();
    }
    Ok(())
}

/// Emit Rust code that evaluates to the Anvyx string representation of `expr`
pub fn emit_to_anvyx_string(
    out: &mut String,
    program: &hir::Program,
    func: &hir::Func,
    expr: &hir::Expr,
) -> Result<(), String> {
    match &expr.ty {
        Type::Int | Type::Bool => {
            emit_expr(out, program, func, expr)?;
            write!(out, ".to_string()").unwrap();
        }
        Type::String => {
            emit_expr(out, program, func, expr)?;
        }
        Type::Float => {
            write!(out, "__fmt_f32(").unwrap();
            emit_expr(out, program, func, expr)?;
            write!(out, ")").unwrap();
        }
        Type::Double => {
            write!(out, "__fmt_f64(").unwrap();
            emit_expr(out, program, func, expr)?;
            write!(out, ")").unwrap();
        }
        Type::Enum { .. } if expr.ty.is_option() => {
            let inner_ty = expr.ty.option_inner().unwrap().clone();
            let type_name = inner_ty.to_string();
            write!(out, "{{ let __opt = ").unwrap();
            emit_expr(out, program, func, expr)?;
            write!(out, "; match __opt {{ Some(__ov) => {{ let mut __s = String::from(\".Some(\"); __s.push_str(&").unwrap();
            emit_stringify_var(out, program, "__ov", &inner_ty)?;
            write!(
                out,
                "); __s.push(')'); __s }}, None => \".None<{type_name}>\".to_string() }} }}"
            )
            .unwrap();
        }
        Type::Enum { name, .. } | Type::Struct { name, .. } | Type::UnresolvedName(name) => {
            write!(out, "__fmt_{name}(&").unwrap();
            emit_expr(out, program, func, expr)?;
            write!(out, ")").unwrap();
        }
        Type::Tuple(elems) => {
            write!(out, "{{ let __tup = ").unwrap();
            emit_expr(out, program, func, expr)?;
            write!(out, "; let mut __s = String::from(\"(\");").unwrap();
            emit_tuple_fields_stringify(out, program, "__tup", elems)?;
            write!(out, " __s.push(')'); __s }}").unwrap();
        }
        Type::Void => {
            write!(out, "String::new()").unwrap();
        }
        other => {
            return Err(format!("Rust backend: cannot stringify type `{other}`"));
        }
    }
    Ok(())
}

/// Emit Rust code that stringifies the value named by var_name, always producing an owned String
pub fn emit_stringify_var(
    out: &mut String,
    program: &hir::Program,
    var_name: &str,
    ty: &Type,
) -> Result<(), String> {
    match ty {
        Type::Int | Type::Bool => {
            write!(out, "{var_name}.to_string()").unwrap();
        }
        Type::String => {
            write!(out, "{var_name}.clone()").unwrap();
        }
        Type::Float => {
            write!(out, "__fmt_f32({var_name}.clone())").unwrap();
        }
        Type::Double => {
            write!(out, "__fmt_f64({var_name}.clone())").unwrap();
        }
        Type::Enum { .. } if ty.is_option() => {
            let inner_ty = ty.option_inner().unwrap().clone();
            let type_name = inner_ty.to_string();
            write!(out, "{{ let __vo = {var_name}.clone(); match __vo {{ Some(__ov) => {{ let mut __s = String::from(\".Some(\"); __s.push_str(&").unwrap();
            emit_stringify_var(out, program, "__ov", &inner_ty)?;
            write!(
                out,
                "); __s.push(')'); __s }}, None => \".None<{type_name}>\".to_string() }} }}"
            )
            .unwrap();
        }
        Type::Enum { name, .. } | Type::Struct { name, .. } | Type::UnresolvedName(name) => {
            write!(out, "__fmt_{name}(&{var_name})").unwrap();
        }
        Type::Tuple(elems) => {
            write!(
                out,
                "{{ let __tv = {var_name}.clone(); let mut __s = String::from(\"(\");"
            )
            .unwrap();
            emit_tuple_fields_stringify(out, program, "__tv", elems)?;
            write!(out, " __s.push(')'); __s }}").unwrap();
        }
        Type::Void => {
            write!(out, "String::new()").unwrap();
        }
        other => {
            return Err(format!("Rust backend: cannot stringify type `{other}`"));
        }
    }
    Ok(())
}

pub fn emit_struct_fmt_helper(
    out: &mut String,
    program: &hir::Program,
    meta: &AggregateMeta,
    indent: usize,
) -> Result<(), String> {
    write_indent(out, indent);
    writeln!(
        out,
        "fn __fmt_{}(v: &{}) -> String {{",
        meta.name, meta.name
    )
    .unwrap();

    if let Some(func_id) = meta.display_func {
        let func = &program.funcs[func_id.0 as usize];
        let mangled = mangle_func_name(&func.name.to_string());
        write_indent(out, indent + 1);
        writeln!(out, "{mangled}(v.clone())").unwrap();
    } else {
        write_indent(out, indent + 1);
        writeln!(out, "let mut __s = String::from(\"{}(\");", meta.name).unwrap();
        for (i, field) in meta.fields.iter().enumerate() {
            if i > 0 {
                write_indent(out, indent + 1);
                writeln!(out, "__s.push_str(\", \");").unwrap();
            }
            write_indent(out, indent + 1);
            writeln!(out, "__s.push_str(\"{}: \");", field.name).unwrap();
            write_indent(out, indent + 1);
            write!(out, "__s.push_str(&").unwrap();
            emit_stringify_var(out, program, &format!("v.{}", field.name), &field.ty)?;
            writeln!(out, ");").unwrap();
        }
        write_indent(out, indent + 1);
        writeln!(out, "__s.push(')');").unwrap();
        write_indent(out, indent + 1);
        writeln!(out, "__s").unwrap();
    }

    write_indent(out, indent);
    writeln!(out, "}}").unwrap();
    Ok(())
}

pub fn emit_enum_fmt_helper(
    out: &mut String,
    program: &hir::Program,
    meta: &EnumMeta,
    indent: usize,
) -> Result<(), String> {
    write_indent(out, indent);
    writeln!(
        out,
        "fn __fmt_{}(v: &{}) -> String {{",
        meta.name, meta.name
    )
    .unwrap();
    write_indent(out, indent + 1);
    writeln!(out, "match v {{").unwrap();

    for variant in &meta.variants {
        write_indent(out, indent + 2);
        match &variant.shape {
            VariantShape::Unit => {
                writeln!(
                    out,
                    "{}::{} => \"{}.{}\".to_string(),",
                    meta.name, variant.name, meta.name, variant.name
                )
                .unwrap();
            }
            VariantShape::Tuple(types) => {
                let bindings: Vec<String> = (0..types.len()).map(|i| format!("__v{i}")).collect();
                writeln!(
                    out,
                    "{}::{}({}) => {{",
                    meta.name,
                    variant.name,
                    bindings.join(", ")
                )
                .unwrap();
                write_indent(out, indent + 3);
                writeln!(
                    out,
                    "let mut __s = String::from(\"{}.{}(\");",
                    meta.name, variant.name
                )
                .unwrap();
                for (i, (binding, ty)) in bindings.iter().zip(types.iter()).enumerate() {
                    if i > 0 {
                        write_indent(out, indent + 3);
                        writeln!(out, "__s.push_str(\", \");").unwrap();
                    }
                    write_indent(out, indent + 3);
                    write!(out, "__s.push_str(&").unwrap();
                    emit_stringify_var(out, program, binding, ty)?;
                    writeln!(out, ");").unwrap();
                }
                write_indent(out, indent + 3);
                writeln!(out, "__s.push(')');").unwrap();
                write_indent(out, indent + 3);
                writeln!(out, "__s").unwrap();
                write_indent(out, indent + 2);
                writeln!(out, "}}").unwrap();
            }
            VariantShape::Struct(fields) => {
                let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
                writeln!(
                    out,
                    "{}::{} {{ {} }} => {{",
                    meta.name,
                    variant.name,
                    names.join(", ")
                )
                .unwrap();
                write_indent(out, indent + 3);
                writeln!(
                    out,
                    "let mut __s = String::from(\"{}.{}(\");",
                    meta.name, variant.name
                )
                .unwrap();
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write_indent(out, indent + 3);
                        writeln!(out, "__s.push_str(\", \");").unwrap();
                    }
                    write_indent(out, indent + 3);
                    writeln!(out, "__s.push_str(\"{}: \");", field.name).unwrap();
                    write_indent(out, indent + 3);
                    write!(out, "__s.push_str(&").unwrap();
                    emit_stringify_var(out, program, &field.name, &field.ty)?;
                    writeln!(out, ");").unwrap();
                }
                write_indent(out, indent + 3);
                writeln!(out, "__s.push(')');").unwrap();
                write_indent(out, indent + 3);
                writeln!(out, "__s").unwrap();
                write_indent(out, indent + 2);
                writeln!(out, "}}").unwrap();
            }
        }
    }

    write_indent(out, indent + 1);
    writeln!(out, "}}").unwrap();
    write_indent(out, indent);
    writeln!(out, "}}").unwrap();

    Ok(())
}
