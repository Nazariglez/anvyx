use std::collections::HashMap;

use internment::Intern;

use super::{FuncLower, LowerCtx, LowerError, SharedCtx};
use crate::{
    ast::{self, BinaryOp, ConstParam, Ident, Mutability, Stmt, Type, UnaryOp},
    backend_names, hir,
    span::Span,
    typecheck::ExternTypeDef,
};

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
    if self_on_right {
        Ident(Intern::new(format!(
            "{type_name}::__op_r{op_str}__{other_ty}"
        )))
    } else {
        Ident(Intern::new(format!(
            "{type_name}::__op_{op_str}__{other_ty}"
        )))
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
        let key =
            extern_binary_op_key(type_name, op_def.op, &op_def.other_ty, op_def.self_on_right);
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
                if !func_node.node.type_params.is_empty() || !func_node.node.const_params.is_empty()
                {
                    continue;
                }
                if skip_existing && ctx.funcs.contains_key(&func_node.node.name) {
                    continue;
                }
                let id = hir::FuncId(*next_func_id);
                *next_func_id += 1;
                ctx.funcs.insert(func_node.node.name, id);
                ctx.func_asts.insert(func_node.node.name, func_node.clone());
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
                    params: extern_node
                        .node
                        .params
                        .iter()
                        .map(|p| p.ty.clone())
                        .collect(),
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

pub(super) fn resolve_extend_ty(ty: &Type, ctx: &SharedCtx) -> Option<Type> {
    match ty {
        Type::UnresolvedName(name) => {
            if ctx.struct_type_ids.contains_key(name) {
                if ctx.tcx.is_dataref(*name) {
                    Some(Type::DataRef {
                        name: *name,
                        type_args: vec![],
                    })
                } else {
                    Some(Type::Struct {
                        name: *name,
                        type_args: vec![],
                    })
                }
            } else if ctx.enum_type_ids.contains_key(name) {
                Some(Type::Enum {
                    name: *name,
                    type_args: vec![],
                })
            } else if ctx.tcx.get_extern_type(*name).is_some() {
                Some(Type::Extern { name: *name })
            } else {
                None
            }
        }
        Type::Struct { name, type_args } if !type_args.is_empty() => {
            if ctx.enum_type_ids.contains_key(name) {
                Some(Type::Enum {
                    name: *name,
                    type_args: type_args.clone(),
                })
            } else if ctx.tcx.is_dataref(*name) {
                Some(Type::DataRef {
                    name: *name,
                    type_args: type_args.clone(),
                })
            } else {
                Some(ty.clone())
            }
        }
        other => Some(other.clone()),
    }
}

pub(super) fn register_extend_declarations<'a>(
    stmts: impl Iterator<Item = &'a ast::StmtNode>,
    module_path: &[String],
    ctx: &mut SharedCtx,
    next_func_id: &mut u32,
    skip_existing: bool,
) {
    let module_str = module_path.join("::");
    for stmt_node in stmts {
        let Stmt::Extend(node) = &stmt_node.node else {
            continue;
        };
        if !node.node.type_params.is_empty() || !node.node.const_params.is_empty() {
            continue;
        }
        let Some(resolved_ty) = resolve_extend_ty(&node.node.ty, ctx) else {
            continue;
        };
        for method in &node.node.methods {
            if method.node.params.is_empty() {
                continue;
            }
            if method.node.params[0].name.0.as_ref() != "self" {
                continue;
            }
            let internal_name =
                backend_names::encode_extend_name(&module_str, &resolved_ty, method.node.name);
            if skip_existing && ctx.funcs.contains_key(&internal_name) {
                continue;
            }
            let id = hir::FuncId(*next_func_id);
            *next_func_id += 1;
            ctx.funcs.insert(internal_name, id);
        }
    }
}

pub(super) fn alloc_assign_temp(fc: &mut FuncLower, ty: Type) -> hir::LocalId {
    let id = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty,
        is_ref: false,
    });
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
        is_ref: false,
    });
    fc.bind_local(name, id);
    id
}

pub(super) fn register_param_local(
    fc: &mut FuncLower,
    name: Ident,
    ty: Type,
    mutability: Mutability,
) {
    let local_id = register_named_local(fc, name, ty);
    if mutability == Mutability::Mutable {
        fc.locals[local_id.0 as usize].is_ref = true;
    }
}

pub(super) fn register_const_param_locals(
    fc: &mut FuncLower,
    const_params: &[ConstParam],
    const_args: &[usize],
) -> Vec<(hir::LocalId, usize)> {
    const_params
        .iter()
        .zip(const_args.iter())
        .map(|(param, &value)| {
            let local = register_named_local(fc, param.name, Type::Int);
            (local, value)
        })
        .collect()
}

pub(super) fn prepend_const_param_stmts(
    body: &mut hir::Block,
    locals: Vec<(hir::LocalId, usize)>,
    span: Span,
) {
    for (local, value) in locals.into_iter().rev() {
        body.stmts.insert(
            0,
            hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local,
                    init: hir::Expr::int_lit(span, value as i64),
                },
            },
        );
    }
}

pub(super) fn build_method_ref_mask(
    is_var_self: bool,
    method_params: &[ast::Param],
    args_len: usize,
) -> Vec<bool> {
    let mut ref_mask = vec![false; args_len];
    if is_var_self {
        ref_mask[0] = true;
    }
    for (i, param) in method_params.iter().enumerate() {
        if param.mutability == Mutability::Mutable && i + 1 < ref_mask.len() {
            ref_mask[i + 1] = true;
        }
    }
    ref_mask
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

pub(super) fn resolve_enum_type_id(
    ctx: &LowerCtx,
    span: Span,
    name: Ident,
) -> Result<u32, LowerError> {
    ctx.shared
        .enum_type_ids
        .get(&name)
        .copied()
        .ok_or_else(|| LowerError::UnsupportedExprKind {
            span,
            kind: format!("unknown enum '{name}'"),
        })
}

pub(super) fn resolve_struct_type_id(
    ctx: &LowerCtx,
    span: Span,
    name: Ident,
) -> Result<u32, LowerError> {
    ctx.shared
        .struct_type_ids
        .get(&name)
        .copied()
        .ok_or_else(|| LowerError::UnsupportedExprKind {
            span,
            kind: format!("unknown struct '{name}'"),
        })
}
