use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};

use internment::Intern;

use super::{
    FuncLower, LowerCtx, LowerError, SharedCtx, analyze_ownership, collect_declarations,
    lower_block, prepend_const_param_stmts, register_const_param_locals,
    register_extend_declarations, register_named_local, register_param_local, resolve_extend_ty,
};
use crate::{
    ast::{self, Ident, MethodReceiver, Type, TypeVarId, VariantKind},
    backend_names, hir,
    ir_meta::{AggregateKind, AggregateMeta, EnumMeta, FieldMeta, VariantMeta, VariantShape},
    prelude_enums::OPTION_TYPE_ID,
    typecheck::{
        ExtendSpecKey, MethodSpecKey, SpecializationKey, TypecheckResult, build_const_subst,
        resolve_type_param_names, subst_type,
    },
};

type SpecRegistrations = (
    Vec<(Ident, SpecializationKey)>,
    Vec<(Ident, MethodSpecKey)>,
    Vec<(Ident, ExtendSpecKey)>,
);

pub fn lower_program(
    ast: &ast::Program,
    tcx: &TypecheckResult,
    module_list: &[(Vec<String>, Vec<ast::StmtNode>)],
) -> Result<hir::Program, LowerError> {
    for (path, _) in module_list {
        tcx.module_check_context(path)
            .expect("module lowering requires a stored check context for every resolved module");
    }

    let (qualified_names, struct_type_ids, enum_type_ids) = assign_type_ids(tcx, module_list);
    let (mut aggregate_meta, enum_meta) =
        build_type_metadata(tcx, &struct_type_ids, &enum_type_ids, &qualified_names);

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

    let (spec_registrations, method_spec_registrations, extend_spec_registrations) =
        lower_all_specializations(&mut shared, &mut next_func_id);

    let mut struct_methods: Vec<(Ident, u32, &ast::Method, bool)> = vec![];
    let to_string_name = Ident(Intern::new("to_string".to_string()));
    let self_ident = Ident(Intern::new("self".to_string()));
    collect_struct_methods_from(
        ast.stmts.iter(),
        &mut shared,
        &mut next_func_id,
        &mut struct_methods,
    );
    for (_path, stmts) in module_list {
        collect_struct_methods_from(
            stmts.iter(),
            &mut shared,
            &mut next_func_id,
            &mut struct_methods,
        );
    }

    shared.next_func_id.set(next_func_id);
    let shared = shared;

    let ctx = LowerCtx {
        shared: &shared,
        type_overrides: None,
        binding_type_overrides: None,
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
        let const_subst = build_const_subst(&template.node.const_params, &spec_key.const_args);

        let specialized_params: Vec<Type> = template
            .node
            .params
            .iter()
            .map(|p| subst_type(&p.ty, &subst, &const_subst))
            .collect();
        let specialized_ret = subst_type(&template.node.ret, &subst, &const_subst);

        let spec_ctx = LowerCtx {
            shared: &shared,
            type_overrides: Some(&spec_result.body_types),
            binding_type_overrides: Some(&spec_result.binding_types),
        };

        let mut fc = FuncLower::new();

        for (param, ty) in template.node.params.iter().zip(specialized_params.iter()) {
            register_param_local(&mut fc, param.name, ty.clone(), param.mutability);
        }
        let params_len = fc.locals.len() as u32;

        let const_param_locals =
            register_const_param_locals(&mut fc, &template.node.const_params, &spec_key.const_args);

        let mut body = lower_block(
            &template.node.body,
            &spec_ctx,
            &mut fc,
            true,
            &specialized_ret,
        )?;

        prepend_const_param_stmts(&mut body, const_param_locals, template.span);

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
            .map(|p| subst_type(&p.ty, &subst, &HashMap::new()))
            .collect();
        let specialized_ret = subst_type(&method.ret, &subst, &HashMap::new());
        let self_type = struct_def.make_type(spec_key.struct_name, struct_type_args.to_vec());

        let spec_ctx = LowerCtx {
            shared: &shared,
            type_overrides: Some(&spec_result.body_types),
            binding_type_overrides: Some(&spec_result.binding_types),
        };

        let mut fc = FuncLower::new();

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

        let mut fc = FuncLower::new();

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
            aggregate_meta[*type_id as usize].display_func = Some(id);
        }
    }

    for (path, stmts) in module_list {
        let module_str = path.join("::");
        lower_extend_methods_from(stmts.iter(), &module_str, &shared, &ctx, &mut funcs)?;
    }
    lower_extend_methods_from(ast.stmts.iter(), "", &shared, &ctx, &mut funcs)?;

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
        let const_subst = build_const_subst(&template.const_params, &spec_key.const_args);
        let target_resolved =
            resolve_type_param_names(&template.target_type, &template.type_params);
        let self_ty = subst_type(&target_resolved, &subst, &const_subst);

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
                    subst_type(&resolved, &subst, &const_subst)
                }
            })
            .collect();
        let raw_ret = resolve_type_param_names(&method.ret, &template.type_params);
        let specialized_ret = subst_type(&raw_ret, &subst, &const_subst);

        let spec_ctx = LowerCtx {
            shared: &shared,
            type_overrides: Some(&spec_result.body_types),
            binding_type_overrides: Some(&spec_result.binding_types),
        };

        let mut fc = FuncLower::new();

        for (param, ty) in method.params.iter().zip(specialized_params.iter()) {
            register_param_local(&mut fc, param.name, ty.clone(), param.mutability);
        }
        let params_len = fc.locals.len() as u32;

        let const_param_locals =
            register_const_param_locals(&mut fc, &template.const_params, &spec_key.const_args);

        let mut body = lower_block(&method.body, &spec_ctx, &mut fc, true, &specialized_ret)?;

        prepend_const_param_stmts(&mut body, const_param_locals, template.method.span);

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
        aggregate_meta,
        enum_meta,
    };

    super::coerce::coerce_optionals(&mut program);
    analyze_ownership(&mut program);

    Ok(program)
}

fn assign_type_ids(
    tcx: &TypecheckResult,
    module_list: &[(Vec<String>, Vec<ast::StmtNode>)],
) -> (
    HashMap<Ident, String>,
    HashMap<Ident, u32>,
    HashMap<Ident, u32>,
) {
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

    (qualified_names, struct_type_ids, enum_type_ids)
}

fn build_type_metadata(
    tcx: &TypecheckResult,
    struct_type_ids: &HashMap<Ident, u32>,
    enum_type_ids: &HashMap<Ident, u32>,
    qualified_names: &HashMap<Ident, String>,
) -> (Vec<AggregateMeta>, Vec<EnumMeta>) {
    let struct_count = struct_type_ids.len();
    let mut aggregate_meta_slots = vec![None; struct_count];
    for (name, &type_id) in struct_type_ids {
        let field_name_idents = tcx.struct_field_names(*name).unwrap_or_default();
        let fields = field_name_idents
            .iter()
            .map(|f| FieldMeta {
                name: f.to_string(),
                ty: tcx.struct_field_type(*name, *f).unwrap_or(Type::Void),
            })
            .collect();
        let is_dataref = tcx.is_dataref(*name);
        let cycle_capable = tcx.is_cycle_capable(*name);
        let short_name = name.to_string();
        let qualified_name = qualified_names
            .get(name)
            .cloned()
            .unwrap_or_else(|| short_name.clone());
        aggregate_meta_slots[type_id as usize] = Some(AggregateMeta {
            name: short_name,
            qualified_name,
            kind: if is_dataref {
                AggregateKind::DataRef
            } else {
                AggregateKind::Struct
            },
            fields,
            display_func: None,
            cycle_capable,
        });
    }
    let aggregate_meta = aggregate_meta_slots
        .into_iter()
        .map(|m| m.unwrap())
        .collect();

    let enum_count = enum_type_ids.len();
    let mut enum_meta_slots = vec![None; enum_count];
    for (name, &type_id) in enum_type_ids {
        let variants = tcx
            .enum_variant_kinds(*name)
            .unwrap_or_default()
            .into_iter()
            .map(|(vname, vkind)| {
                let shape = match vkind {
                    VariantKind::Unit => VariantShape::Unit,
                    VariantKind::Tuple(types) => VariantShape::Tuple(types.clone()),
                    VariantKind::Struct(fields) => VariantShape::Struct(
                        fields
                            .iter()
                            .map(|f| FieldMeta {
                                name: f.name.to_string(),
                                ty: f.ty.clone(),
                            })
                            .collect(),
                    ),
                };
                VariantMeta {
                    name: vname.to_string(),
                    shape,
                }
            })
            .collect();
        let qualified_name = qualified_names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string());
        enum_meta_slots[type_id as usize] = Some(EnumMeta {
            name: name.to_string(),
            qualified_name,
            variants,
        });
    }
    let enum_meta = enum_meta_slots.into_iter().map(|m| m.unwrap()).collect();

    (aggregate_meta, enum_meta)
}

fn lower_all_specializations(shared: &mut SharedCtx, next_func_id: &mut u32) -> SpecRegistrations {
    let mut spec_registrations: Vec<(Ident, SpecializationKey)> = vec![];
    for spec_key in shared.tcx.specializations().keys() {
        let spec_result = &shared.tcx.specializations()[spec_key];
        if spec_result.err.is_some() {
            continue;
        }
        if shared.tcx.generic_template(spec_key.func_name).is_none() {
            continue;
        }
        let mangled = backend_names::encode_specialization_name(
            spec_key.func_name,
            &spec_key.type_args,
            &spec_key.const_args,
        );
        if shared.funcs.contains_key(&mangled) {
            continue;
        }
        let id = hir::FuncId(*next_func_id);
        *next_func_id += 1;
        shared.funcs.insert(mangled, id);
        spec_registrations.push((mangled, spec_key.clone()));
    }

    let mut method_spec_registrations: Vec<(Ident, MethodSpecKey)> = vec![];
    for (spec_key, spec_result) in shared.tcx.method_specializations() {
        if spec_result.err.is_some() {
            continue;
        }
        if spec_key.type_args.is_empty() && spec_key.const_args.is_empty() {
            continue;
        }
        let mangled = backend_names::encode_method_specialization_name(
            spec_key.struct_name,
            spec_key.method_name,
            &spec_key.type_args,
            &spec_key.const_args,
        );
        if shared.funcs.contains_key(&mangled) {
            continue;
        }
        let id = hir::FuncId(*next_func_id);
        *next_func_id += 1;
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
        let mangled =
            backend_names::encode_extend_specialization_name(spec_key, &template.source_module);
        if shared.funcs.contains_key(&mangled) {
            continue;
        }
        let id = hir::FuncId(*next_func_id);
        *next_func_id += 1;
        shared.funcs.insert(mangled, id);
        extend_spec_registrations.push((mangled, spec_key.clone()));
    }

    (
        spec_registrations,
        method_spec_registrations,
        extend_spec_registrations,
    )
}

fn lower_extend_method(
    method: &ast::ExtendMethodNode,
    self_ty: &Type,
    id: hir::FuncId,
    name: Ident,
    ctx: &LowerCtx,
) -> Result<hir::Func, LowerError> {
    let mut fc = FuncLower::new();

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

fn collect_struct_methods_from<'a>(
    stmts: impl Iterator<Item = &'a ast::StmtNode>,
    shared: &mut SharedCtx,
    next_func_id: &mut u32,
    struct_methods: &mut Vec<(Ident, u32, &'a ast::Method, bool)>,
) {
    for stmt_node in stmts {
        let (struct_name, type_params, const_params, methods, is_dataref) = match &stmt_node.node {
            ast::Stmt::Struct(s) => (
                s.node.name,
                &s.node.type_params,
                &s.node.const_params,
                &s.node.methods,
                false,
            ),
            ast::Stmt::DataRef(s) => (
                s.node.name,
                &s.node.type_params,
                &s.node.const_params,
                &s.node.methods,
                true,
            ),
            _ => continue,
        };
        let Some(&type_id) = shared.struct_type_ids.get(&struct_name) else {
            continue;
        };
        if !type_params.is_empty() || !const_params.is_empty() {
            continue;
        }
        for method in methods {
            if method.receiver.is_none()
                || !method.type_params.is_empty()
                || !method.const_params.is_empty()
            {
                continue;
            }
            let mangled = Ident(Intern::new(format!("{struct_name}::{}", method.name)));
            if let std::collections::hash_map::Entry::Vacant(e) = shared.funcs.entry(mangled) {
                let id = hir::FuncId(*next_func_id);
                *next_func_id += 1;
                e.insert(id);
                struct_methods.push((struct_name, type_id, method, is_dataref));
            }
        }
    }
}

fn lower_extend_methods_from<'a>(
    stmts: impl Iterator<Item = &'a ast::StmtNode>,
    module_prefix: &str,
    shared: &SharedCtx,
    ctx: &LowerCtx,
    funcs: &mut Vec<hir::Func>,
) -> Result<(), LowerError> {
    for stmt_node in stmts {
        let ast::Stmt::Extend(node) = &stmt_node.node else {
            continue;
        };
        if !node.node.type_params.is_empty() || !node.node.const_params.is_empty() {
            continue;
        }
        let Some(resolved_ty) = resolve_extend_ty(&node.node.ty, shared) else {
            continue;
        };
        for method in &node.node.methods {
            if method.node.params.is_empty() || method.node.params[0].name.0.as_ref() != "self" {
                continue;
            }
            let internal_name =
                backend_names::encode_extend_name(module_prefix, &resolved_ty, method.node.name);
            let &id = shared
                .funcs
                .get(&internal_name)
                .expect("extend method registered in collect_declarations");
            funcs.push(lower_extend_method(
                method,
                &resolved_ty,
                id,
                internal_name,
                ctx,
            )?);
        }
    }
    Ok(())
}

fn lower_func(func_node: &ast::FuncNode, ctx: &LowerCtx) -> Result<hir::Func, LowerError> {
    let func = &func_node.node;
    let id = *ctx
        .shared
        .funcs
        .get(&func.name)
        .expect("func id must exist after pass 1");

    let mut fc = FuncLower::new();

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
