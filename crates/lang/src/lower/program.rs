use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

use internment::Intern;

use super::{
    FuncLower, LowerCtx, LowerError, SharedCtx, analyze_ownership, collect_declarations,
    lower_block, mangle_generic_name, mangle_method_spec_name, register_extend_declarations,
    register_named_local, register_param_local,
};
use crate::{
    ast::{self, Ident, MethodReceiver, Type, TypeVarId, VariantKind},
    hir,
    prelude_enums::OPTION_TYPE_ID,
    typecheck::{ExtendSpecKey, MethodSpecKey, TypeChecker, resolve_type_param_names, subst_type},
    vm::{
        cycle_collector::make_dataref_vtable,
        meta::{EnumMeta, StructMeta, VariantMeta, VariantMetaKind},
    },
};

pub fn lower_program(
    ast: &ast::Program,
    tcx: &TypeChecker,
    module_list: &[(Vec<String>, Vec<ast::StmtNode>)],
) -> Result<hir::Program, LowerError> {
    let mut qualified_names: HashMap<Ident, String> = HashMap::new();
    for (path, stmts) in module_list {
        let prefix = path.join("::");
        for stmt_node in stmts {
            match &stmt_node.node {
                ast::Stmt::Struct(s) => {
                    qualified_names
                        .entry(s.node.name)
                        .or_insert_with(|| format!("{}::{}", prefix, s.node.name));
                }
                ast::Stmt::DataRef(s) => {
                    qualified_names
                        .entry(s.node.name)
                        .or_insert_with(|| format!("{}::{}", prefix, s.node.name));
                }
                ast::Stmt::Enum(e) => {
                    qualified_names
                        .entry(e.node.name)
                        .or_insert_with(|| format!("{}::{}", prefix, e.node.name));
                }
                _ => {}
            }
        }
    }

    let mut struct_type_ids: HashMap<Ident, u32> = HashMap::new();
    let mut struct_next_id = 0u32;
    for name in tcx.struct_names() {
        struct_type_ids.entry(name).or_insert_with(|| {
            let id = struct_next_id;
            struct_next_id += 1;
            id
        });
    }
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            if let ast::Stmt::Struct(s) = &stmt_node.node {
                struct_type_ids.entry(s.node.name).or_insert_with(|| {
                    let id = struct_next_id;
                    struct_next_id += 1;
                    id
                });
            }
            if let ast::Stmt::DataRef(s) = &stmt_node.node {
                struct_type_ids.entry(s.node.name).or_insert_with(|| {
                    let id = struct_next_id;
                    struct_next_id += 1;
                    id
                });
            }
        }
    }

    let mut enum_type_ids: HashMap<Ident, u32> = HashMap::new();

    // Reserve well-known ID for Option
    let option_ident = Ident(Intern::new("Option".to_string()));
    enum_type_ids.insert(option_ident, OPTION_TYPE_ID);
    let mut enum_next_id = OPTION_TYPE_ID + 1;

    for name in tcx.enum_names() {
        enum_type_ids.entry(name).or_insert_with(|| {
            let id = enum_next_id;
            enum_next_id += 1;
            id
        });
    }
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            if let ast::Stmt::Enum(e) = &stmt_node.node {
                enum_type_ids.entry(e.node.name).or_insert_with(|| {
                    let id = enum_next_id;
                    enum_next_id += 1;
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
            .map(ToString::to_string)
            .collect();
        let is_dataref = tcx.is_dataref(*name);
        let cycle_capable = tcx.is_cycle_capable(*name);
        let short_name = name.to_string();
        let vtable = if is_dataref {
            let vtable_name = qualified_names
                .get(name)
                .cloned()
                .unwrap_or_else(|| short_name.clone());
            Some(make_dataref_vtable(&vtable_name, cycle_capable))
        } else {
            None
        };
        struct_meta_slots[type_id as usize] = Some(StructMeta {
            name: short_name,
            field_names,
            to_string_fn: None,
            is_dataref,
            cycle_capable,
            vtable,
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
        let idx = type_id as usize;
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
        next_func_id: Cell::new(0),
        lambda_funcs: RefCell::new(vec![]),
        func_asts: HashMap::new(),
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

    let mut method_spec_registrations: Vec<(Ident, MethodSpecKey)> = vec![];
    for (spec_key, spec_result) in shared.tcx.method_specializations() {
        if spec_result.err.is_some() {
            continue;
        }
        if spec_key.type_args.is_empty() {
            continue;
        }
        let mangled = mangle_method_spec_name(
            spec_key.struct_name,
            spec_key.method_name,
            &spec_key.type_args,
        );
        if shared.funcs.contains_key(&mangled) {
            continue;
        }
        let id = hir::FuncId(next_func_id);
        next_func_id += 1;
        shared.funcs.insert(mangled, id);
        method_spec_registrations.push((mangled, spec_key.clone()));
    }

    let mut extend_spec_registrations: Vec<(Ident, ExtendSpecKey)> = vec![];
    for (spec_key, spec_result) in shared.tcx.extend_specializations() {
        if spec_result.err.is_some() {
            continue;
        }

        let template = shared.tcx.get_generic_extend_template(
            spec_key.base_name,
            spec_key.method_name,
            &spec_key.target_type,
        );
        let Some(template) = template else { continue };

        let mangled = spec_key.mangle(&template.source_module);
        if shared.funcs.contains_key(&mangled) {
            continue;
        }

        let id = hir::FuncId(next_func_id);
        next_func_id += 1;
        shared.funcs.insert(mangled, id);
        extend_spec_registrations.push((mangled, spec_key.clone()));
    }

    let mut struct_methods: Vec<(Ident, u32, &ast::Method, bool)> = vec![];
    let to_string_name = Ident(Intern::new("to_string".to_string()));
    let self_ident = Ident(Intern::new("self".to_string()));
    for stmt_node in &ast.stmts {
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
                if let std::collections::hash_map::Entry::Vacant(e) = shared.funcs.entry(mangled) {
                    let id = hir::FuncId(next_func_id);
                    next_func_id += 1;
                    e.insert(id);
                    struct_methods.push((struct_name, type_id, method, false));
                }
            }
        }
        if let ast::Stmt::DataRef(s) = &stmt_node.node {
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
                if let std::collections::hash_map::Entry::Vacant(e) = shared.funcs.entry(mangled) {
                    let id = hir::FuncId(next_func_id);
                    next_func_id += 1;
                    e.insert(id);
                    struct_methods.push((struct_name, type_id, method, true));
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
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        shared.funcs.entry(mangled)
                    {
                        let id = hir::FuncId(next_func_id);
                        next_func_id += 1;
                        e.insert(id);
                        struct_methods.push((struct_name, type_id, method, false));
                    }
                }
            }
            if let ast::Stmt::DataRef(s) = &stmt_node.node {
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
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        shared.funcs.entry(mangled)
                    {
                        let id = hir::FuncId(next_func_id);
                        next_func_id += 1;
                        e.insert(id);
                        struct_methods.push((struct_name, type_id, method, true));
                    }
                }
            }
        }
    }

    shared.next_func_id.set(next_func_id);
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
            defer_stack: vec![],
            loop_defer_depth: None,
        };

        for (param, ty) in template.node.params.iter().zip(specialized_params.iter()) {
            register_param_local(&mut fc, param.name, ty.clone(), param.mutability);
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

    for (mangled, spec_key) in &method_spec_registrations {
        let spec_result = &shared.tcx.method_specializations()[spec_key];
        let Some(struct_def) = shared.tcx.get_struct(spec_key.struct_name) else {
            continue;
        };
        let Some(method) = struct_def.methods.get(&spec_key.method_name) else {
            continue;
        };

        let &id = shared
            .funcs
            .get(mangled)
            .expect("method spec func id must exist");

        let n_struct = struct_def.type_params.len();
        let struct_type_args = &spec_key.type_args[..n_struct];
        let method_type_args = &spec_key.type_args[n_struct..];

        let subst: HashMap<TypeVarId, Type> = struct_def
            .type_params
            .iter()
            .zip(struct_type_args.iter())
            .chain(method.type_params.iter().zip(method_type_args.iter()))
            .map(|(param, arg)| (param.id, arg.clone()))
            .collect();

        let specialized_params: Vec<Type> = method
            .params
            .iter()
            .map(|p| subst_type(&p.ty, &subst))
            .collect();
        let specialized_ret = subst_type(&method.ret, &subst);
        let self_type = struct_def.make_type(spec_key.struct_name, struct_type_args.to_vec());

        let spec_ctx = LowerCtx {
            shared: &shared,
            type_overrides: Some(&spec_result.body_types),
        };

        let mut fc = FuncLower {
            locals: vec![],
            local_map: HashMap::new(),
            scope_log: vec![],
            defer_stack: vec![],
            loop_defer_depth: None,
        };

        if method.receiver.is_some() {
            register_named_local(&mut fc, self_ident, self_type);
            if matches!(method.receiver, Some(MethodReceiver::Var)) {
                fc.locals[0].is_ref = true;
            }
        }
        for (param, ty) in method.params.iter().zip(specialized_params.iter()) {
            register_param_local(&mut fc, param.name, ty.clone(), param.mutability);
        }
        let params_len = fc.locals.len() as u32;

        let body = lower_block(&method.body, &spec_ctx, &mut fc, true, &specialized_ret)?;

        funcs.push(hir::Func {
            id,
            name: *mangled,
            locals: fc.locals,
            params_len,
            ret: specialized_ret,
            body,
            span: method.body.span,
        });
    }

    for (struct_name, type_id, method, is_dataref) in &struct_methods {
        let mangled = Ident(Intern::new(format!("{struct_name}::{}", method.name)));
        let &id = shared
            .funcs
            .get(&mangled)
            .expect("struct method func id must exist");

        let mut fc = FuncLower {
            locals: vec![],
            local_map: HashMap::new(),
            scope_log: vec![],
            defer_stack: vec![],
            loop_defer_depth: None,
        };

        let self_type = if *is_dataref {
            Type::DataRef {
                name: *struct_name,
                type_args: vec![],
            }
        } else {
            Type::Struct {
                name: *struct_name,
                type_args: vec![],
            }
        };
        register_named_local(&mut fc, self_ident, self_type);
        if matches!(method.receiver, Some(MethodReceiver::Var)) {
            fc.locals[0].is_ref = true;
        }

        for param in &method.params {
            register_param_local(&mut fc, param.name, param.ty.clone(), param.mutability);
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

        if method.name == to_string_name && shared.tcx.struct_to_string_body(*struct_name).is_some()
        {
            struct_meta[*type_id as usize].to_string_fn = Some(id.0 as usize);
        }
    }

    for (path, stmts) in module_list {
        let module_str = path.join("::");
        for stmt_node in stmts {
            let ast::Stmt::Extend(node) = &stmt_node.node else {
                continue;
            };
            if !node.node.type_params.is_empty() {
                continue;
            }
            let resolved_ty = resolve_extend_ty(&node.node.ty, &shared);
            let Some(resolved_ty) = resolved_ty else {
                continue;
            };
            let type_str = format!("{resolved_ty}");
            for method in &node.node.methods {
                if method.node.params.is_empty() {
                    continue;
                }
                if method.node.params[0].name.0.as_ref() != "self" {
                    continue;
                }
                let internal_name = Ident(Intern::new(format!(
                    "__extend::{}::{}::{}",
                    module_str, type_str, method.node.name
                )));
                let &id = shared
                    .funcs
                    .get(&internal_name)
                    .expect("extend method registered in collect_declarations");
                funcs.push(lower_extend_method(
                    method,
                    &resolved_ty,
                    id,
                    internal_name,
                    &ctx,
                )?);
            }
        }
    }

    for stmt_node in &ast.stmts {
        let ast::Stmt::Extend(node) = &stmt_node.node else {
            continue;
        };
        if !node.node.type_params.is_empty() {
            continue;
        }
        let resolved_ty = resolve_extend_ty(&node.node.ty, &shared);
        let Some(resolved_ty) = resolved_ty else {
            continue;
        };
        let type_str = format!("{resolved_ty}");
        for method in &node.node.methods {
            if method.node.params.is_empty() {
                continue;
            }
            if method.node.params[0].name.0.as_ref() != "self" {
                continue;
            }
            let internal_name = Ident(Intern::new(format!(
                "__extend::::{}::{}",
                type_str, method.node.name
            )));
            let &id = shared
                .funcs
                .get(&internal_name)
                .expect("extend method registered in collect_declarations");
            funcs.push(lower_extend_method(
                method,
                &resolved_ty,
                id,
                internal_name,
                &ctx,
            )?);
        }
    }

    for (mangled, spec_key) in &extend_spec_registrations {
        let spec_result = &shared.tcx.extend_specializations()[spec_key];
        let template = shared
            .tcx
            .get_generic_extend_template(
                spec_key.base_name,
                spec_key.method_name,
                &spec_key.target_type,
            )
            .expect("template must exist — registered during pre-registration");

        let &id = shared
            .funcs
            .get(mangled)
            .expect("mangled name was just registered");

        let subst: HashMap<TypeVarId, Type> = template
            .type_params
            .iter()
            .zip(spec_key.type_args.iter())
            .map(|(param, arg)| (param.id, arg.clone()))
            .collect();
        let target_resolved =
            resolve_type_param_names(&template.target_type, &template.type_params);
        let self_ty = subst_type(&target_resolved, &subst);

        let method = &template.method.node;

        let specialized_params: Vec<Type> = method
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == 0 {
                    self_ty.clone()
                } else {
                    let resolved = resolve_type_param_names(&p.ty, &template.type_params);
                    subst_type(&resolved, &subst)
                }
            })
            .collect();
        let raw_ret = resolve_type_param_names(&method.ret, &template.type_params);
        let specialized_ret = subst_type(&raw_ret, &subst);

        let spec_ctx = LowerCtx {
            shared: &shared,
            type_overrides: Some(&spec_result.body_types),
        };

        let mut fc = FuncLower {
            locals: vec![],
            local_map: HashMap::new(),
            scope_log: vec![],
            defer_stack: vec![],
            loop_defer_depth: None,
        };

        for (param, ty) in method.params.iter().zip(specialized_params.iter()) {
            register_param_local(&mut fc, param.name, ty.clone(), param.mutability);
        }
        let params_len = fc.locals.len() as u32;

        let body = lower_block(&method.body, &spec_ctx, &mut fc, true, &specialized_ret)?;

        funcs.push(hir::Func {
            id,
            name: *mangled,
            locals: fc.locals,
            params_len,
            ret: specialized_ret,
            body,
            span: template.method.span,
        });
    }

    funcs.extend(shared.lambda_funcs.borrow_mut().drain(..));
    funcs.sort_by_key(|f| f.id.0);

    let mut program = hir::Program {
        funcs,
        externs: extern_decls,
        struct_meta,
        enum_meta,
    };

    analyze_ownership(&mut program);

    Ok(program)
}

fn resolve_extend_ty(ty: &Type, shared: &SharedCtx) -> Option<Type> {
    match ty {
        Type::UnresolvedName(name) => {
            if shared.struct_type_ids.contains_key(name) {
                if shared.tcx.is_dataref(*name) {
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
            } else if shared.enum_type_ids.contains_key(name) {
                Some(Type::Enum {
                    name: *name,
                    type_args: vec![],
                })
            } else if shared.tcx.get_extern_type(*name).is_some() {
                Some(Type::Extern { name: *name })
            } else {
                None
            }
        }
        Type::Struct { name, type_args } if !type_args.is_empty() => {
            if shared.enum_type_ids.contains_key(name) {
                Some(Type::Enum {
                    name: *name,
                    type_args: type_args.clone(),
                })
            } else if shared.tcx.is_dataref(*name) {
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
        defer_stack: vec![],
        loop_defer_depth: None,
    };

    for (i, param) in method.node.params.iter().enumerate() {
        let ty = if i == 0 {
            self_ty.clone()
        } else {
            param.ty.clone()
        };
        register_param_local(&mut fc, param.name, ty, param.mutability);
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
        defer_stack: vec![],
        loop_defer_depth: None,
    };

    // register parameters as locals first
    for param in &func.params {
        register_param_local(&mut fc, param.name, param.ty.clone(), param.mutability);
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
