use std::collections::HashMap;

use crate::ast::{self, BinaryOp, Ident, Stmt, Type, UnaryOp};
use crate::hir;
use crate::span::Span;
use crate::typecheck::ExternTypeDef;
use internment::Intern;

use super::{FuncLower, LowerCtx, LowerError, SharedCtx};

pub(super) fn mangle_generic_name(name: Ident, type_args: &[Type]) -> Ident {
    let suffix = type_args
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join("$");
    Ident(Intern::new(format!("{name}${suffix}")))
}

pub(super) fn register_extern_decl(
    name: Ident,
    params: Vec<Type>,
    ret: Type,
    next_extern_id: &mut u32,
    externs: &mut HashMap<Ident, hir::ExternId>,
    extern_decls: &mut Vec<hir::ExternDecl>,
) {
    if externs.contains_key(&name) {
        return;
    }
    let id = hir::ExternId(*next_extern_id);
    *next_extern_id += 1;
    externs.insert(name, id);
    extern_decls.push(hir::ExternDecl {
        id,
        name,
        params,
        ret,
    });
}

fn binary_op_key_str(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Rem => "rem",
        BinaryOp::Eq => "eq",
        _ => unreachable!("unsupported extern binary op: {op}"),
    }
}

fn unary_op_key_str(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        _ => unreachable!("unsupported extern unary op: {op}"),
    }
}

pub(super) fn extern_binary_op_key(
    type_name: Ident,
    op: BinaryOp,
    other_ty: &Type,
    self_on_right: bool,
) -> Ident {
    let op_str = binary_op_key_str(op);
    let other = other_ty.to_string();
    if self_on_right {
        Ident(Intern::new(format!("{type_name}::__op_r{op_str}__{other}")))
    } else {
        Ident(Intern::new(format!("{type_name}::__op_{op_str}__{other}")))
    }
}

pub(super) fn extern_unary_op_key(type_name: Ident, op: UnaryOp) -> Ident {
    let op_str = unary_op_key_str(op);
    Ident(Intern::new(format!("{type_name}::__op_{op_str}")))
}

pub(super) fn register_extern_type_members(
    type_name: Ident,
    extern_def: &ExternTypeDef,
    next_extern_id: &mut u32,
    externs: &mut HashMap<Ident, hir::ExternId>,
    extern_decls: &mut Vec<hir::ExternDecl>,
) {
    for (field_name, field_def) in &extern_def.fields {
        let getter_name = Ident(Intern::new(format!("{type_name}::__get_{field_name}")));
        register_extern_decl(
            getter_name,
            vec![Type::Extern { name: type_name }],
            field_def.ty.clone(),
            next_extern_id,
            externs,
            extern_decls,
        );

        let setter_name = Ident(Intern::new(format!("{type_name}::__set_{field_name}")));
        register_extern_decl(
            setter_name,
            vec![Type::Extern { name: type_name }, field_def.ty.clone()],
            Type::Void,
            next_extern_id,
            externs,
            extern_decls,
        );
    }

    for (method_name, method_def) in &extern_def.methods {
        let qualified = Ident(Intern::new(format!("{type_name}::{method_name}")));
        let mut params = vec![Type::Extern { name: type_name }];
        params.extend(method_def.params.iter().map(|p| p.ty.clone()));
        register_extern_decl(
            qualified,
            params,
            method_def.ret.clone(),
            next_extern_id,
            externs,
            extern_decls,
        );
    }

    for (method_name, method_def) in &extern_def.statics {
        let qualified = Ident(Intern::new(format!("{type_name}::{method_name}")));
        let params = method_def.params.iter().map(|p| p.ty.clone()).collect();
        register_extern_decl(
            qualified,
            params,
            method_def.ret.clone(),
            next_extern_id,
            externs,
            extern_decls,
        );
    }

    if extern_def.has_init {
        let init_name = Ident(Intern::new(format!("{type_name}::__init__")));
        let params = extern_def
            .field_order
            .iter()
            .map(|name| extern_def.fields[name].ty.clone())
            .collect();
        register_extern_decl(
            init_name,
            params,
            Type::Extern { name: type_name },
            next_extern_id,
            externs,
            extern_decls,
        );
    }

    for op_def in &extern_def.operators {
        let key = extern_binary_op_key(type_name, op_def.op, &op_def.other_ty, op_def.self_on_right);
        let params = if op_def.self_on_right {
            vec![op_def.other_ty.clone(), Type::Extern { name: type_name }]
        } else {
            vec![Type::Extern { name: type_name }, op_def.other_ty.clone()]
        };
        register_extern_decl(
            key,
            params,
            op_def.ret.clone(),
            next_extern_id,
            externs,
            extern_decls,
        );
    }

    for op_def in &extern_def.unary_operators {
        let key = extern_unary_op_key(type_name, op_def.op);
        register_extern_decl(
            key,
            vec![Type::Extern { name: type_name }],
            op_def.ret.clone(),
            next_extern_id,
            externs,
            extern_decls,
        );
    }
}

pub(super) fn collect_declarations<'a>(
    stmts: impl Iterator<Item = &'a ast::StmtNode>,
    ctx: &mut SharedCtx,
    func_nodes: &mut Vec<&'a ast::FuncNode>,
    next_func_id: &mut u32,
    next_extern_id: &mut u32,
    extern_decls: &mut Vec<hir::ExternDecl>,
    skip_existing: bool,
) {
    for stmt_node in stmts {
        match &stmt_node.node {
            Stmt::Func(func_node) => {
                if !func_node.node.type_params.is_empty() {
                    continue;
                }
                if skip_existing && ctx.funcs.contains_key(&func_node.node.name) {
                    continue;
                }
                let id = hir::FuncId(*next_func_id);
                *next_func_id += 1;
                ctx.funcs.insert(func_node.node.name, id);
                func_nodes.push(func_node);
            }
            Stmt::ExternFunc(extern_node) => {
                if skip_existing && ctx.externs.contains_key(&extern_node.node.name) {
                    continue;
                }
                let id = hir::ExternId(*next_extern_id);
                *next_extern_id += 1;
                ctx.externs.insert(extern_node.node.name, id);
                extern_decls.push(hir::ExternDecl {
                    id,
                    name: extern_node.node.name,
                    params: extern_node.node.params.iter().map(|p| p.ty.clone()).collect(),
                    ret: extern_node.node.ret.clone(),
                });
            }
            Stmt::ExternType(ext) => {
                if let Some(extern_def) = ctx.tcx.get_extern_type(ext.node.name).cloned() {
                    register_extern_type_members(
                        ext.node.name,
                        &extern_def,
                        next_extern_id,
                        &mut ctx.externs,
                        extern_decls,
                    );
                }
            }
            Stmt::Import(import_node) => {
                if let ast::ImportKind::Selective(items) = &import_node.node.kind {
                    for item in items {
                        let Some(alias) = item.alias else { continue };
                        if let Some(&func_id) = ctx.funcs.get(&item.name) {
                            ctx.funcs.insert(alias, func_id);
                        } else if let Some(&extern_id) = ctx.externs.get(&item.name) {
                            ctx.externs.insert(alias, extern_id);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

pub(super) fn alloc_assign_temp(fc: &mut FuncLower, ty: Type) -> hir::LocalId {
    let id = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local { name: None, ty });
    id
}

pub(super) fn alloc_and_bind(
    fc: &mut FuncLower,
    span: Span,
    out: &mut Vec<hir::Stmt>,
    ty: Type,
    init: hir::Expr,
) -> hir::LocalId {
    let local = alloc_assign_temp(fc, ty);
    out.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Let { local, init },
    });
    local
}

pub(super) fn emit_counter_increment(
    body_stmts: &mut Vec<hir::Stmt>,
    span: Span,
    i_local: hir::LocalId,
    counter_ty: &Type,
    increment: hir::Expr,
    inc_op: BinaryOp,
) {
    body_stmts.push(hir::Stmt {
        span,
        kind: hir::StmtKind::Assign {
            local: i_local,
            value: hir::Expr::binary(
                counter_ty.clone(),
                span,
                inc_op,
                hir::Expr::local(counter_ty.clone(), span, i_local),
                increment,
            ),
        },
    });
}

pub(super) fn register_named_local(fc: &mut FuncLower, name: Ident, ty: Type) -> hir::LocalId {
    let id = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: Some(name),
        ty,
    });
    fc.bind_local(name, id);
    id
}

pub(super) fn resolve_variant_index(
    ctx: &LowerCtx,
    span: Span,
    enum_name: Ident,
    variant: Ident,
) -> Result<u16, LowerError> {
    ctx.shared
        .tcx
        .enum_variant_index(enum_name, variant)
        .ok_or_else(|| LowerError::UnsupportedExprKind {
            span,
            kind: format!("unknown variant '{variant}' on enum '{enum_name}'"),
        })
}

pub(super) fn resolve_enum_type_id(ctx: &LowerCtx, span: Span, name: Ident) -> Result<u32, LowerError> {
    ctx.shared.enum_type_ids
        .get(&name)
        .copied()
        .ok_or_else(|| LowerError::UnsupportedExprKind {
            span,
            kind: format!("unknown enum '{name}'"),
        })
}

pub(super) fn resolve_struct_type_id(ctx: &LowerCtx, span: Span, name: Ident) -> Result<u32, LowerError> {
    ctx.shared.struct_type_ids
        .get(&name)
        .copied()
        .ok_or_else(|| LowerError::UnsupportedExprKind {
            span,
            kind: format!("unknown struct '{name}'"),
        })
}
