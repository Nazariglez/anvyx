use std::collections::HashMap;

use internment::Intern;

use crate::ast::{self, Ident, Type, TypeVarId, VariantKind};
use crate::hir;
use crate::typecheck::{TypeChecker, subst_type};
use crate::vm::meta::{EnumMeta, StructMeta, VariantMeta, VariantMetaKind};

use super::{
    FuncLower, LowerCtx, LowerError, SharedCtx, collect_declarations, lower_block,
    mangle_generic_name, register_extend_declarations, register_named_local,
};

pub fn lower_program(
    ast: &ast::Program,
    tcx: &TypeChecker,
    module_list: &[(Vec<String>, Vec<ast::StmtNode>)],
) -> Result<hir::Program, LowerError> {
    let mut struct_type_ids: HashMap<Ident, u32> = HashMap::new();
    let mut next_type_id = 0u32;
    for name in tcx.struct_names() {
        struct_type_ids.entry(name).or_insert_with(|| {
            let id = next_type_id;
            next_type_id += 1;
            id
        });
    }
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            if let ast::Stmt::Struct(s) = &stmt_node.node {
                struct_type_ids.entry(s.node.name).or_insert_with(|| {
                    let id = next_type_id;
                    next_type_id += 1;
                    id
                });
            }
        }
    }

    let mut enum_type_ids: HashMap<Ident, u32> = HashMap::new();
    for name in tcx.enum_names() {
        enum_type_ids.entry(name).or_insert_with(|| {
            let id = next_type_id;
            next_type_id += 1;
            id
        });
    }
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            if let ast::Stmt::Enum(e) = &stmt_node.node {
                enum_type_ids.entry(e.node.name).or_insert_with(|| {
                    let id = next_type_id;
                    next_type_id += 1;
                    id
                });
            }
        }
    }

    let struct_count = struct_type_ids.len();

    let mut struct_meta_slots = vec![None; struct_count];
    for (name, &type_id) in &struct_type_ids {
        let field_names = tcx
            .struct_field_names(*name)
            .unwrap_or_default()
            .iter()
            .map(|f| f.to_string())
            .collect();
        struct_meta_slots[type_id as usize] = Some(StructMeta {
            name: name.to_string(),
            field_names,
            to_string_fn: None,
        });
    }
    let mut struct_meta: Vec<StructMeta> =
        struct_meta_slots.into_iter().map(|m| m.unwrap()).collect();

    let enum_count = enum_type_ids.len();
    let mut enum_meta_slots = vec![None; enum_count];
    for (name, &type_id) in &enum_type_ids {
        let variants = tcx
            .enum_variant_kinds(*name)
            .unwrap_or_default()
            .into_iter()
            .map(|(vname, vkind)| {
                let kind = match vkind {
                    VariantKind::Unit => VariantMetaKind::Unit,
                    VariantKind::Tuple(types) => VariantMetaKind::Tuple(types.len()),
                    VariantKind::Struct(fields) => {
                        VariantMetaKind::Struct(fields.iter().map(|f| f.name.to_string()).collect())
                    }
                };
                VariantMeta {
                    name: vname.to_string(),
                    kind,
                }
            })
            .collect();
        let idx = type_id as usize - struct_count;
        enum_meta_slots[idx] = Some(EnumMeta {
            name: name.to_string(),
            variants,
        });
    }
    let enum_meta: Vec<EnumMeta> = enum_meta_slots.into_iter().map(|m| m.unwrap()).collect();

    let mut shared = SharedCtx {
        tcx,
        funcs: HashMap::new(),
        externs: HashMap::new(),
        struct_type_ids,
        enum_type_ids,
    };

    let mut func_nodes: Vec<&ast::FuncNode> = vec![];
    let mut next_func_id = 0u32;
    let mut next_extern_id = 0u32;
    let mut extern_decls: Vec<hir::ExternDecl> = vec![];

    for (path, stmts) in module_list {
        collect_declarations(
            stmts.iter(),
            &mut shared,
            &mut func_nodes,
            &mut next_func_id,
            &mut next_extern_id,
            &mut extern_decls,
            true,
        );
        register_extend_declarations(stmts.iter(), path, &mut shared, &mut next_func_id, true);
    }

    collect_declarations(
        ast.stmts.iter(),
        &mut shared,
        &mut func_nodes,
        &mut next_func_id,
        &mut next_extern_id,
        &mut extern_decls,
        false,
    );
    register_extend_declarations(ast.stmts.iter(), &[], &mut shared, &mut next_func_id, false);

    let mut spec_registrations: Vec<(Ident, crate::typecheck::SpecializationKey)> = vec![];
    for spec_key in shared.tcx.specializations().keys() {
        let spec_result = &shared.tcx.specializations()[spec_key];
        if spec_result.err.is_some() {
            continue;
        }
        if shared.tcx.generic_template(spec_key.func_name).is_none() {
            continue;
        }
        let mangled = mangle_generic_name(spec_key.func_name, &spec_key.type_args);
        if shared.funcs.contains_key(&mangled) {
            continue;
        }
        let id = hir::FuncId(next_func_id);
        next_func_id += 1;
        shared.funcs.insert(mangled, id);
        spec_registrations.push((mangled, spec_key.clone()));
    }

    let mut struct_methods: Vec<(Ident, u32, &ast::Method)> = vec![];
    let to_string_name = Ident(Intern::new("to_string".to_string()));
    let self_ident = Ident(Intern::new("self".to_string()));
    for stmt_node in ast.stmts.iter() {
        if let ast::Stmt::Struct(s) = &stmt_node.node {
            let struct_name = s.node.name;
            let Some(&type_id) = shared.struct_type_ids.get(&struct_name) else {
                continue;
            };
            if !s.node.type_params.is_empty() {
                continue;
            }
            for method in &s.node.methods {
                if method.receiver.is_none() {
                    continue;
                }
                if !method.type_params.is_empty() {
                    continue;
                }
                let mangled = Ident(Intern::new(format!("{struct_name}::{}", method.name)));
                if !shared.funcs.contains_key(&mangled) {
                    let id = hir::FuncId(next_func_id);
                    next_func_id += 1;
                    shared.funcs.insert(mangled, id);
                    struct_methods.push((struct_name, type_id, method));
                }
            }
        }
    }
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            if let ast::Stmt::Struct(s) = &stmt_node.node {
                let struct_name = s.node.name;
                let Some(&type_id) = shared.struct_type_ids.get(&struct_name) else {
                    continue;
                };
                if !s.node.type_params.is_empty() {
                    continue;
                }
                for method in &s.node.methods {
                    if method.receiver.is_none() {
                        continue;
                    }
                    if !method.type_params.is_empty() {
                        continue;
                    }
                    let mangled = Ident(Intern::new(format!("{struct_name}::{}", method.name)));
                    if !shared.funcs.contains_key(&mangled) {
                        let id = hir::FuncId(next_func_id);
                        next_func_id += 1;
                        shared.funcs.insert(mangled, id);
                        struct_methods.push((struct_name, type_id, method));
                    }
                }
            }
        }
    }

    let shared = shared;

    let ctx = LowerCtx {
        shared: &shared,
        type_overrides: None,
    };
    let mut funcs = vec![];
    for func_node in func_nodes {
        funcs.push(lower_func(func_node, &ctx)?);
    }

    for (mangled, spec_key) in &spec_registrations {
        let spec_result = &shared.tcx.specializations()[spec_key];
        let template = shared
            .tcx
            .generic_template(spec_key.func_name)
            .expect("template must exist as checked during pre-registration");

        let &id = shared
            .funcs
            .get(mangled)
            .expect("mangled name was just registered");

        let type_params = &template.node.type_params;
        let subst: HashMap<TypeVarId, Type> = type_params
            .iter()
            .zip(spec_key.type_args.iter())
            .map(|(param, arg)| (param.id, arg.clone()))
            .collect();

        let specialized_params: Vec<Type> = template
            .node
            .params
            .iter()
            .map(|p| subst_type(&p.ty, &subst))
            .collect();
        let specialized_ret = subst_type(&template.node.ret, &subst);

        let spec_ctx = LowerCtx {
            shared: &shared,
            type_overrides: Some(&spec_result.body_types),
        };

        let mut fc = FuncLower {
            locals: vec![],
            local_map: HashMap::new(),
            scope_log: vec![],
        };

        for (param, ty) in template.node.params.iter().zip(specialized_params.iter()) {
            register_named_local(&mut fc, param.name, ty.clone());
        }
        let params_len = fc.locals.len() as u32;

        let body = lower_block(
            &template.node.body,
            &spec_ctx,
            &mut fc,
            true,
            &specialized_ret,
        )?;

        funcs.push(hir::Func {
            id,
            name: *mangled,
            locals: fc.locals,
            params_len,
            ret: specialized_ret,
            body,
            span: template.span,
        });
    }

    for (struct_name, type_id, method) in &struct_methods {
        let mangled = Ident(Intern::new(format!("{struct_name}::{}", method.name)));
        let &id = shared
            .funcs
            .get(&mangled)
            .expect("struct method func id must exist");

        let mut fc = FuncLower {
            locals: vec![],
            local_map: HashMap::new(),
            scope_log: vec![],
        };

        let self_type = Type::Struct {
            name: *struct_name,
            type_args: vec![],
        };
        register_named_local(&mut fc, self_ident, self_type);

        for param in &method.params {
            register_named_local(&mut fc, param.name, param.ty.clone());
        }
        let params_len = fc.locals.len() as u32;

        let body = lower_block(&method.body, &ctx, &mut fc, true, &method.ret)?;

        funcs.push(hir::Func {
            id,
            name: mangled,
            locals: fc.locals,
            params_len,
            ret: method.ret.clone(),
            body,
            span: method.body.span,
        });

        if method.name == to_string_name && shared.tcx.struct_to_string_body(*struct_name).is_some() {
            struct_meta[*type_id as usize].to_string_fn = Some(id.0 as usize);
        }
    }

    for (path, stmts) in module_list {
        let module_str = path.join("::");
        for stmt_node in stmts {
            let ast::Stmt::Extend(node) = &stmt_node.node else { continue };
            let resolved_ty = resolve_extend_ty(&node.node.ty, &shared);
            let Some(resolved_ty) = resolved_ty else { continue };
            let type_str = format!("{resolved_ty}");
            for method in &node.node.methods {
                if method.node.params.is_empty() { continue; }
                if method.node.params[0].name.0.as_ref() != "self" { continue; }
                let internal_name = Ident(Intern::new(format!(
                    "__extend::{}::{}::{}",
                    module_str, type_str, method.node.name
                )));
                let &id = shared.funcs.get(&internal_name).expect("extend method registered in collect_declarations");
                funcs.push(lower_extend_method(method, &resolved_ty, id, internal_name, &ctx)?);
            }
        }
    }

    for stmt_node in &ast.stmts {
        let ast::Stmt::Extend(node) = &stmt_node.node else { continue };
        let resolved_ty = resolve_extend_ty(&node.node.ty, &shared);
        let Some(resolved_ty) = resolved_ty else { continue };
        let type_str = format!("{resolved_ty}");
        for method in &node.node.methods {
            if method.node.params.is_empty() { continue; }
            if method.node.params[0].name.0.as_ref() != "self" { continue; }
            let internal_name = Ident(Intern::new(format!(
                "__extend::::{}::{}",
                type_str, method.node.name
            )));
            let &id = shared.funcs.get(&internal_name).expect("extend method registered in collect_declarations");
            funcs.push(lower_extend_method(method, &resolved_ty, id, internal_name, &ctx)?);
        }
    }

    funcs.sort_by_key(|f| f.id.0);

    Ok(hir::Program {
        funcs,
        externs: extern_decls,
        struct_meta,
        enum_meta,
    })
}

fn resolve_extend_ty(ty: &Type, shared: &SharedCtx) -> Option<Type> {
    match ty {
        Type::UnresolvedName(name) => {
            if shared.struct_type_ids.contains_key(name) {
                Some(Type::Struct { name: *name, type_args: vec![] })
            } else if shared.enum_type_ids.contains_key(name) {
                Some(Type::Enum { name: *name, type_args: vec![] })
            } else if shared.tcx.get_extern_type(*name).is_some() {
                Some(Type::Extern { name: *name })
            } else {
                None
            }
        }
        other => Some(other.clone()),
    }
}

fn lower_extend_method(
    method: &ast::ExtendMethodNode,
    self_ty: &Type,
    id: hir::FuncId,
    name: Ident,
    ctx: &LowerCtx,
) -> Result<hir::Func, LowerError> {
    let mut fc = FuncLower {
        locals: vec![],
        local_map: HashMap::new(),
        scope_log: vec![],
    };

    for (i, param) in method.node.params.iter().enumerate() {
        let ty = if i == 0 { self_ty.clone() } else { param.ty.clone() };
        register_named_local(&mut fc, param.name, ty);
    }
    let params_len = fc.locals.len() as u32;

    let ret = method.node.ret.clone();
    let body = lower_block(&method.node.body, ctx, &mut fc, true, &ret)?;

    Ok(hir::Func {
        id,
        name,
        locals: fc.locals,
        params_len,
        ret,
        body,
        span: method.span,
    })
}

fn lower_func(func_node: &ast::FuncNode, ctx: &LowerCtx) -> Result<hir::Func, LowerError> {
    let func = &func_node.node;
    let id = *ctx
        .shared
        .funcs
        .get(&func.name)
        .expect("func id must exist after pass 1");

    let mut fc = FuncLower {
        locals: vec![],
        local_map: HashMap::new(),
        scope_log: vec![],
    };

    // register parameters as locals first
    for param in &func.params {
        register_named_local(&mut fc, param.name, param.ty.clone());
    }
    let params_len = fc.locals.len() as u32;

    let body = lower_block(&func.body, ctx, &mut fc, true, &func.ret)?;

    Ok(hir::Func {
        id,
        name: func.name,
        locals: fc.locals,
        params_len,
        ret: func.ret.clone(),
        body,
        span: func_node.span,
    })
}
