use std::collections::HashMap;

use crate::ast::{self, Ident, Type, TypeVarId};
use crate::hir;
use crate::typecheck::{TypeChecker, subst_type};

use super::{
    FuncLower, LowerCtx, LowerError, SharedCtx,
    collect_declarations, lower_block, mangle_generic_name, register_named_local,
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

    for (_path, stmts) in module_list {
        collect_declarations(
            stmts.iter(),
            &mut shared,
            &mut func_nodes,
            &mut next_func_id,
            &mut next_extern_id,
            &mut extern_decls,
            true,
        );
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

    let shared = shared;

    let ctx = LowerCtx { shared: &shared, type_overrides: None };
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

    Ok(hir::Program {
        funcs,
        externs: extern_decls,
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
