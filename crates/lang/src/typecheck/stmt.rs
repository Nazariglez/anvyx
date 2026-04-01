use std::collections::{HashMap, HashSet};

use internment::Intern;

use super::{
    annotations::{AnnotationTarget, extract_deprecated, validate_annotations},
    composite::{is_all_nil_array_literal, is_empty_map_literal},
    const_eval::{
        ConstDef, build_const_dependency_graph, collect_const_decls, eval_const_expr,
        validate_const_expr,
    },
    constraint::TypeRef,
    control::{block_always_diverges, check_for, check_while, check_while_let, is_if_without_else},
    decl::{check_body_common, check_func, check_struct},
    error::{Diagnostic, DiagnosticKind},
    expr::check_expr,
    infer::type_from_fn,
    pattern::{check_pattern, is_refutable},
    types::{
        Deprecated, EnumDef, ExtendEntry, ExtendMethodDef, ExternFieldDef, ExternMethodDef,
        ExternOpDef, ExternTypeDef, ExternUnaryOpDef, GenericExtendTemplate, ModuleDef,
        ModuleExtendEntry, ModuleGenericExtendEntry, StructDef, TypeChecker, build_param_info,
        unwrap_opt_typ, validate_map_key_type,
    },
    unify::contains_infer,
    visit::type_any,
};
use crate::{
    ast::{
        ArrayLen, BlockNode, ConstDeclNode, DeferBody, ExprId, ExprKind, ExprNode, ExternFunc,
        ExternTypeMember, Func, FuncNode, FuncParam, Ident, ImportKind, Mutability, Param,
        ReturnNode, Stmt, StmtNode, Type, TypeParam, VariantKind, Visibility,
    },
    span::Span,
};

pub(super) fn check_block_stmts(
    stmts: &[StmtNode],
    tail: Option<&ExprNode>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected_tail: Option<&Type>,
) -> Option<ExprId> {
    let in_function = !type_checker.return_types.is_empty();

    type_checker.push_scope();
    collect_scope_types(stmts, type_checker, errors);

    let const_snapshot = if in_function {
        type_checker.const_scope_stack.push(HashSet::new());
        Some(type_checker.const_defs.clone())
    } else {
        evaluate_module_consts(stmts, type_checker, errors);
        None
    };

    for stmt in stmts {
        check_stmt(stmt, type_checker, errors);
    }

    let tail_id = tail.map(|expr| {
        let _ = check_expr(expr, type_checker, errors, expected_tail);
        expr.node.id
    });

    if let Some(snapshot) = const_snapshot {
        type_checker.const_scope_stack.pop();
        type_checker.const_defs = snapshot;
    }

    type_checker.pop_scope();
    tail_id
}

fn evaluate_module_consts(
    stmts: &[StmtNode],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let decls = collect_const_decls(stmts);
    if decls.is_empty() {
        return;
    }

    let order = match build_const_dependency_graph(&decls) {
        Ok(order) => order,
        Err(err) => {
            errors.push(err);
            return;
        }
    };

    for idx in order {
        let decl = decls[idx];
        process_const_decl(decl, type_checker, errors);
    }
}

fn process_const_decl(
    decl: &ConstDeclNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let name = decl.node.name;

    let known_consts: HashSet<Ident> = type_checker.const_defs.keys().copied().collect();
    if let Err(err) = validate_const_expr(&decl.node.value, &known_consts) {
        errors.push(err);
        return;
    }

    let annotated_ty = decl
        .node
        .ty
        .as_ref()
        .map(|ty| type_checker.resolve_type(ty));

    let _ = check_expr(
        &decl.node.value,
        type_checker,
        errors,
        annotated_ty.as_ref(),
    );

    let const_value = match eval_const_expr(&decl.node.value, &type_checker.const_defs) {
        Ok(val) => val,
        Err(err) => {
            errors.push(err);
            return;
        }
    };

    let value_ty = const_value.ty();
    if let Some(ref ann_ty) = annotated_ty
        && *ann_ty != value_ty
    {
        errors.push(Diagnostic::new(
            decl.span,
            DiagnosticKind::ConstTypeMismatch {
                expected: ann_ty.clone(),
                got: value_ty,
            },
        ));
        return;
    }

    let final_ty = annotated_ty.unwrap_or_else(|| const_value.ty());

    type_checker.const_defs.insert(
        name,
        ConstDef {
            ty: final_ty,
            value: const_value,
            visibility: decl.node.visibility,
        },
    );
}

pub(super) fn check_block_expr(
    block: &BlockNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected_tail: Option<&Type>,
) -> (Type, Option<ExprId>) {
    let last_expr_id = check_block_stmts(
        &block.node.stmts,
        block.node.tail.as_deref(),
        type_checker,
        errors,
        expected_tail,
    );
    let Some(id) = last_expr_id else {
        return (Type::Void, None);
    };

    let ty = type_checker
        .get_type(id)
        .cloned()
        .map_or(Type::Void, |(_, ty)| ty);

    (ty, Some(id))
}

struct FuncRegistration {
    func_ty: Type,
    param_info: Vec<(Ident, Mutability)>,
    type_params: Option<Vec<TypeParam>>,
    template: Option<FuncNode>,
}

fn build_func_registration(
    func: &Func,
    node: &FuncNode,
    resolve: impl Fn(&Type) -> Type,
) -> FuncRegistration {
    let func_ty = resolve(&type_from_fn(func));
    let param_info = build_param_info(&func.params);
    let type_params = (!func.type_params.is_empty()).then(|| func.type_params.clone());
    let template = type_params.is_some().then(|| node.clone());
    FuncRegistration {
        func_ty,
        param_info,
        type_params,
        template,
    }
}

fn build_extern_func_registration(
    extern_func: &ExternFunc,
    resolve: impl Fn(&Type) -> Type,
) -> (Type, Vec<(Ident, Mutability)>) {
    let func_ty = Type::Func {
        params: extern_func
            .params
            .iter()
            .map(|p| FuncParam::new(resolve(&p.ty), matches!(p.mutability, Mutability::Mutable)))
            .collect(),
        ret: Box::new(resolve(&extern_func.ret)),
    };
    let param_info = build_param_info(&extern_func.params);
    (func_ty, param_info)
}

fn push_reexport_collision(
    errors: &mut Vec<Diagnostic>,
    span: Span,
    name: Ident,
    existing: &str,
    source: &str,
    help: &str,
) {
    errors.push(
        Diagnostic::new(
            span,
            DiagnosticKind::ReExportCollision {
                name,
                first_source: existing.to_string(),
                second_source: source.to_string(),
            },
        )
        .with_help(help),
    );
}

fn merge_symbol(name: Ident, bind_as: Ident, source: &ModuleDef, target: &mut ModuleDef) {
    if let Some(ty) = source.funcs.get(&name) {
        target.funcs.insert(bind_as, ty.clone());
        target.all_names.insert(bind_as);
        if let Some(param_info) = source.func_param_info.get(&name) {
            target.func_param_info.insert(bind_as, param_info.clone());
        }
        if let Some(tp) = source.func_type_params.get(&name) {
            target.func_type_params.insert(bind_as, tp.clone());
        }
        if let Some(tmpl) = source.generic_func_templates.get(&name) {
            target.generic_func_templates.insert(bind_as, tmpl.clone());
        }
        if let Some(defaults) = source.func_param_defaults.get(&name) {
            target.func_param_defaults.insert(bind_as, defaults.clone());
        }
    } else if let Some(struct_def) = source.struct_defs.get(&name) {
        target.struct_defs.insert(bind_as, struct_def.clone());
        target.all_names.insert(bind_as);
    } else if let Some(enum_def) = source.enum_defs.get(&name) {
        target.enum_defs.insert(bind_as, enum_def.clone());
        target.all_names.insert(bind_as);
    } else if let Some(extern_def) = source.extern_types.get(&name) {
        target.extern_types.insert(bind_as, extern_def.clone());
        target.all_names.insert(bind_as);
    } else if let Some(const_def) = source.const_defs.get(&name) {
        target.const_defs.insert(bind_as, const_def.clone());
        target.all_names.insert(bind_as);
    }
}

fn inject_module_item(
    name: Ident,
    bind_as: Ident,
    module_def: &ModuleDef,
    module_path: &[String],
    type_checker: &mut TypeChecker,
) {
    if let Some(ty) = module_def.funcs.get(&name) {
        type_checker.set_var(bind_as, ty.clone(), false);
        if let Some(param_info) = module_def.func_param_info.get(&name) {
            type_checker
                .func_param_info
                .insert(bind_as, param_info.clone());
        }
        if let Some(tp) = module_def.func_type_params.get(&name) {
            type_checker.func_type_params.insert(bind_as, tp.clone());
        }
        if let Some(tmpl) = module_def.generic_func_templates.get(&name) {
            type_checker
                .generic_func_templates
                .insert(bind_as, tmpl.clone());
            type_checker
                .generic_func_source_module
                .insert(bind_as, module_path.to_vec());
        }
        if let Some(defaults) = module_def.func_param_defaults.get(&name) {
            type_checker
                .func_param_defaults
                .insert(bind_as, defaults.clone());
        }
    } else if let Some(struct_def) = module_def.struct_defs.get(&name) {
        type_checker.struct_defs.insert(bind_as, struct_def.clone());
    } else if let Some(enum_def) = module_def.enum_defs.get(&name) {
        type_checker.enum_defs.insert(bind_as, enum_def.clone());
    } else if let Some(extern_def) = module_def.extern_types.get(&name) {
        type_checker
            .extern_type_defs
            .insert(bind_as, extern_def.clone());
    } else if let Some(sub_module) = module_def.re_exported_modules.get(&name) {
        type_checker.module_defs.insert(bind_as, sub_module.clone());
    } else if let Some(const_def) = module_def.const_defs.get(&name) {
        type_checker.const_defs.insert(bind_as, const_def.clone());
    }

    // symbol not found is silently skipped here, check_stmt will report the error
}

fn build_extern_type_def(
    has_init: bool,
    members: &[ExternTypeMember],
    resolve: impl Fn(&Type) -> Type,
) -> ExternTypeDef {
    let mut field_order = vec![];
    let mut fields = HashMap::new();
    let mut methods = HashMap::new();
    let mut statics = HashMap::new();
    let mut operators = vec![];
    let mut unary_operators = vec![];

    for member in members {
        match member {
            ExternTypeMember::Field { name, ty, computed } => {
                if !computed {
                    field_order.push(*name);
                }
                fields.insert(*name, ExternFieldDef { ty: resolve(ty) });
            }
            ExternTypeMember::Method {
                name,
                receiver,
                params,
                ret,
                ..
            } => {
                methods.insert(
                    *name,
                    ExternMethodDef {
                        receiver: Some(*receiver),
                        params: params
                            .iter()
                            .map(|p| Param {
                                mutability: p.mutability,
                                name: p.name,
                                ty: resolve(&p.ty),
                                default: p.default.clone(),
                            })
                            .collect(),
                        ret: resolve(ret),
                    },
                );
            }
            ExternTypeMember::StaticMethod {
                name, params, ret, ..
            } => {
                statics.insert(
                    *name,
                    ExternMethodDef {
                        receiver: None,
                        params: params
                            .iter()
                            .map(|p| Param {
                                mutability: p.mutability,
                                name: p.name,
                                ty: resolve(&p.ty),
                                default: p.default.clone(),
                            })
                            .collect(),
                        ret: resolve(ret),
                    },
                );
            }
            ExternTypeMember::Operator {
                op,
                other_ty,
                ret,
                self_on_right,
            } => {
                operators.push(ExternOpDef {
                    op: *op,
                    other_ty: resolve(other_ty),
                    ret: resolve(ret),
                    self_on_right: *self_on_right,
                });
            }
            ExternTypeMember::UnaryOperator { op, ret } => {
                unary_operators.push(ExternUnaryOpDef {
                    op: *op,
                    ret: resolve(ret),
                });
            }
        }
    }

    ExternTypeDef {
        has_init,
        field_order,
        fields,
        methods,
        statics,
        operators,
        unary_operators,
    }
}

pub(super) fn collect_scope_types(
    stmts: &[StmtNode],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    // register extern type names before anything else, so they can be
    // resolved when building function signatures that reference them
    for stmt in stmts {
        if let Stmt::ExternType(node) = &stmt.node {
            type_checker.extern_type_defs.insert(
                node.node.name,
                ExternTypeDef {
                    has_init: false,
                    field_order: vec![],
                    fields: HashMap::new(),
                    methods: HashMap::new(),
                    statics: HashMap::new(),
                    operators: vec![],
                    unary_operators: vec![],
                },
            );
        }
    }

    let mut seen: HashSet<Ident> = HashSet::new();
    let mut imported: HashSet<Ident> = HashSet::new();
    let mut imported_modules: HashSet<Ident> = HashSet::new();

    for stmt in stmts {
        match &stmt.node {
            Stmt::ExternType(node) => {
                let name = node.node.name;
                if seen.insert(name) {
                    let def = build_extern_type_def(node.node.has_init, &node.node.members, |ty| {
                        type_checker.resolve_type(ty)
                    });
                    type_checker.extern_type_defs.insert(name, def);
                } else {
                    errors.push(Diagnostic::new(
                        stmt.span,
                        DiagnosticKind::DuplicateTypeDefinition { name },
                    ));
                }
            }

            Stmt::ExternFunc(node) => {
                let extern_func = &node.node;
                let (func_ty, param_info) =
                    build_extern_func_registration(extern_func, |ty| type_checker.resolve_type(ty));
                type_checker.set_var(extern_func.name, func_ty, false);
                type_checker
                    .func_param_info
                    .insert(extern_func.name, param_info);
            }

            Stmt::Func(node) => {
                let func = &node.node;
                let FuncRegistration {
                    func_ty,
                    param_info,
                    type_params,
                    template,
                } = build_func_registration(func, node, |ty| type_checker.resolve_type(ty));
                type_checker.set_var(func.name, func_ty, false);
                type_checker.func_param_info.insert(func.name, param_info);
                if let Some(tp) = type_params {
                    type_checker.func_type_params.insert(func.name, tp);
                }
                if let Some(tmpl) = template {
                    type_checker.generic_func_templates.insert(func.name, tmpl);
                }
                if let Deprecated::Yes(reason) = extract_deprecated(&func.annotations) {
                    type_checker.func_deprecated.insert(func.name, reason);
                }
            }

            Stmt::Struct(node) => {
                let name = node.node.name;
                if seen.insert(name) {
                    type_checker
                        .struct_defs
                        .insert(name, StructDef::from_ast(&node.node, false));
                } else {
                    errors.push(Diagnostic::new(
                        stmt.span,
                        DiagnosticKind::DuplicateTypeDefinition { name },
                    ));
                }
            }

            Stmt::DataRef(node) => {
                let name = node.node.name;
                if seen.insert(name) {
                    type_checker
                        .struct_defs
                        .insert(name, StructDef::from_ast(&node.node, true));
                } else {
                    errors.push(Diagnostic::new(
                        stmt.span,
                        DiagnosticKind::DuplicateTypeDefinition { name },
                    ));
                }
            }

            Stmt::Enum(node) => {
                let name = node.node.name;
                if seen.insert(name) {
                    type_checker
                        .enum_defs
                        .insert(name, EnumDef::from_ast(&node.node));
                } else {
                    errors.push(Diagnostic::new(
                        stmt.span,
                        DiagnosticKind::DuplicateTypeDefinition { name },
                    ));
                }
            }

            Stmt::Import(node) => {
                let import = &node.node;
                let path_key: Vec<String> = import.path.iter().map(ToString::to_string).collect();

                let Some(module_def) = type_checker.resolved_module_defs.get(&path_key).cloned()
                else {
                    // module wasn't resolved (std.*, unresolved error, etc.)
                    continue;
                };

                // Activate extend methods from the imported module for all import forms
                let binding_name = match &import.kind {
                    ImportKind::Module => *import.path.last().expect("import path cannot be empty"),
                    ImportKind::ModuleAs(alias) => *alias,
                    ImportKind::Selective(_) | ImportKind::Wildcard => {
                        *import.path.last().expect("import path cannot be empty")
                    }
                };
                for entry in &module_def.extend_methods {
                    let key = (entry.ty.clone(), entry.name);
                    let entries = type_checker.extend_defs.entry(key).or_default();
                    let already_registered = entries.iter().any(|e| e.source_module == path_key);
                    if !already_registered {
                        entries.push(ExtendEntry {
                            source_module: path_key.clone(),
                            binding: binding_name,
                            def: entry.def.clone(),
                        });
                    }
                }
                for entry in &module_def.generic_extend_methods {
                    let key = (entry.base_name, entry.method_name);
                    let entries = type_checker
                        .generic_extend_templates
                        .entry(key)
                        .or_default();
                    let already_registered = entries
                        .iter()
                        .any(|e| e.source_module == path_key && e.target_type == entry.target_type);
                    if !already_registered {
                        entries.push(GenericExtendTemplate {
                            type_params: entry.type_params.clone(),
                            target_type: entry.target_type.clone(),
                            method: entry.method.clone(),
                            source_module: path_key.clone(),
                            binding: binding_name,
                        });
                    }
                }

                match &import.kind {
                    ImportKind::Module => {
                        let binding = *import.path.last().expect("import path cannot be empty");
                        if imported_modules.contains(&binding) {
                            errors.push(Diagnostic::new(
                                stmt.span,
                                DiagnosticKind::ModuleBindingConflict { name: binding },
                            ));
                        } else {
                            imported_modules.insert(binding);
                            type_checker.module_defs.insert(binding, module_def);
                        }
                    }
                    ImportKind::ModuleAs(alias) => {
                        if imported_modules.contains(alias) {
                            errors.push(Diagnostic::new(
                                stmt.span,
                                DiagnosticKind::ModuleBindingConflict { name: *alias },
                            ));
                        } else {
                            imported_modules.insert(*alias);
                            type_checker.module_defs.insert(*alias, module_def);
                        }
                    }
                    ImportKind::Selective(items) => {
                        for item in items {
                            let bind_as = item.alias.unwrap_or(item.name);
                            let existing = if imported.contains(&bind_as) {
                                Some("a previously imported name")
                            } else if seen.contains(&bind_as) {
                                Some("a locally defined type")
                            } else {
                                None
                            };
                            if let Some(existing) = existing {
                                errors.push(Diagnostic::new(
                                    stmt.span,
                                    DiagnosticKind::ImportNameConflict {
                                        name: bind_as,
                                        existing,
                                    },
                                ));
                            } else {
                                inject_module_item(
                                    item.name,
                                    bind_as,
                                    &module_def,
                                    &path_key,
                                    type_checker,
                                );
                                imported.insert(bind_as);
                                seen.insert(bind_as);
                            }
                        }
                    }
                    ImportKind::Wildcard => {
                        let names: Vec<Ident> = module_def.all_public_names().collect();
                        for name in names {
                            let existing = if imported.contains(&name) {
                                Some("a previously imported name")
                            } else if seen.contains(&name) {
                                Some("a locally defined type")
                            } else {
                                None
                            };
                            if let Some(existing) = existing {
                                errors.push(Diagnostic::new(
                                    stmt.span,
                                    DiagnosticKind::ImportNameConflict { name, existing },
                                ));
                            } else {
                                inject_module_item(
                                    name,
                                    name,
                                    &module_def,
                                    &path_key,
                                    type_checker,
                                );
                                imported.insert(name);
                                seen.insert(name);
                            }
                        }
                        let sub_module_names: Vec<Ident> =
                            module_def.re_exported_modules.keys().copied().collect();
                        for name in sub_module_names {
                            let existing = if imported.contains(&name) {
                                Some("a previously imported name")
                            } else if seen.contains(&name) {
                                Some("a locally defined type")
                            } else {
                                None
                            };
                            if let Some(existing) = existing {
                                errors.push(Diagnostic::new(
                                    stmt.span,
                                    DiagnosticKind::ImportNameConflict { name, existing },
                                ));
                            } else {
                                inject_module_item(
                                    name,
                                    name,
                                    &module_def,
                                    &path_key,
                                    type_checker,
                                );
                                imported.insert(name);
                                seen.insert(name);
                            }
                        }
                    }
                }
            }

            Stmt::Extend(node) => {
                let decl = &node.node;
                if !decl.type_params.is_empty() {
                    // generic template are registered by check_extend_decl, not here
                    continue;
                }
                let resolved_ty = type_checker.resolve_type(&decl.ty);
                if !is_valid_concrete_extend_type(&resolved_ty) {
                    continue;
                }

                for method in &decl.methods {
                    if method.node.params.is_empty() {
                        continue;
                    }
                    let self_param = &method.node.params[0];
                    if self_param.name.0.as_ref() != "self" {
                        continue;
                    }
                    if self_param.ty != Type::Infer {
                        continue;
                    }

                    let type_str = format!("{resolved_ty}");
                    let module_str = type_checker
                        .current_module_path
                        .as_ref()
                        .map(|p| p.join("::"))
                        .unwrap_or_default();
                    let internal_name = Ident(Intern::new(format!(
                        "__extend::{}::{}::{}",
                        module_str, type_str, method.node.name
                    )));

                    let mut params = method.node.params.clone();
                    params[0].ty = resolved_ty.clone();
                    let ret = type_checker.resolve_type(&method.node.ret);

                    let source_module =
                        type_checker.current_module_path.clone().unwrap_or_default();
                    let def = ExtendMethodDef {
                        params,
                        ret,
                        internal_name,
                    };
                    type_checker
                        .extend_defs
                        .entry((resolved_ty.clone(), method.node.name))
                        .or_default()
                        .push(ExtendEntry {
                            source_module,
                            binding: Ident(Intern::new(String::new())),
                            def,
                        });
                }
            }
            _ => {}
        }
    }
}

pub(super) fn build_module_def_with_reexports(
    stmts: &[StmtNode],
    type_checker: &TypeChecker,
    module_path: &[String],
) -> (ModuleDef, Vec<Diagnostic>) {
    let mut module_def = ModuleDef::default();
    let mut errors: Vec<Diagnostic> = vec![];

    let mut local_extern_types = HashSet::new();
    for stmt in stmts {
        if let Stmt::ExternType(node) = &stmt.node {
            local_extern_types.insert(node.node.name);
        }
    }

    let resolve = |ty: &Type| -> Type {
        fn pre_resolve(ty: &Type, local: &HashSet<Ident>) -> Type {
            match ty {
                Type::UnresolvedName(name) if local.contains(name) => Type::Extern { name: *name },
                Type::Func { params, ret } => Type::Func {
                    params: params
                        .iter()
                        .map(|p| FuncParam::new(pre_resolve(&p.ty, local), p.mutable))
                        .collect(),
                    ret: Box::new(pre_resolve(ret, local)),
                },
                _ => ty.clone(),
            }
        }
        let pre = pre_resolve(ty, &local_extern_types);
        type_checker.resolve_type(&pre)
    };

    let mut reexported_from: HashMap<Ident, String> = HashMap::new();

    for stmt in stmts {
        match &stmt.node {
            Stmt::Func(node) => {
                let func = &node.node;
                module_def.all_names.insert(func.name);
                if func.visibility != Visibility::Public {
                    continue;
                }
                let FuncRegistration {
                    func_ty,
                    param_info,
                    type_params,
                    template,
                } = build_func_registration(func, node, resolve);
                module_def.funcs.insert(func.name, func_ty);
                module_def.func_param_info.insert(func.name, param_info);
                if let Some(tp) = type_params {
                    module_def.func_type_params.insert(func.name, tp);
                }
                if let Some(tmpl) = template {
                    module_def.generic_func_templates.insert(func.name, tmpl);
                }
                reexported_from.insert(func.name, "<local>".to_string());
            }
            Stmt::Struct(node) => {
                let decl = &node.node;
                module_def.all_names.insert(decl.name);
                if decl.visibility != Visibility::Public {
                    continue;
                }
                module_def
                    .struct_defs
                    .insert(decl.name, StructDef::from_ast(decl, false));

                reexported_from.insert(decl.name, "<local>".to_string());
            }
            Stmt::DataRef(node) => {
                let decl = &node.node;
                module_def.all_names.insert(decl.name);
                if decl.visibility != Visibility::Public {
                    continue;
                }
                module_def
                    .struct_defs
                    .insert(decl.name, StructDef::from_ast(decl, true));

                reexported_from.insert(decl.name, "<local>".to_string());
            }
            Stmt::Enum(node) => {
                let decl = &node.node;
                module_def.all_names.insert(decl.name);
                if decl.visibility != Visibility::Public {
                    continue;
                }
                module_def
                    .enum_defs
                    .insert(decl.name, EnumDef::from_ast(decl));

                reexported_from.insert(decl.name, "<local>".to_string());
            }
            Stmt::Import(node) => {
                let import = &node.node;
                if import.visibility != Visibility::Public {
                    continue;
                }

                let path_key: Vec<String> = import.path.iter().map(ToString::to_string).collect();
                let source_label = path_key.last().cloned().unwrap_or_default();

                let Some(source_def) = type_checker.resolved_module_defs.get(&path_key).cloned()
                else {
                    continue;
                };

                match &import.kind {
                    ImportKind::Selective(items) => {
                        for item in items {
                            let bind_as = item.alias.unwrap_or(item.name);
                            if let Some(existing) = reexported_from.get(&bind_as) {
                                push_reexport_collision(
                                    &mut errors,
                                    node.span,
                                    bind_as,
                                    existing,
                                    &source_label,
                                    "use selective re-exports or aliasing to resolve the conflict",
                                );
                            } else {
                                merge_symbol(item.name, bind_as, &source_def, &mut module_def);
                                reexported_from.insert(bind_as, source_label.clone());
                            }
                        }
                    }
                    ImportKind::Wildcard => {
                        for name in source_def.all_public_names().collect::<Vec<_>>() {
                            if let Some(existing) = reexported_from.get(&name) {
                                push_reexport_collision(
                                    &mut errors,
                                    node.span,
                                    name,
                                    existing,
                                    &source_label,
                                    "use selective re-exports or aliasing to resolve the conflict",
                                );
                            } else {
                                merge_symbol(name, name, &source_def, &mut module_def);
                                reexported_from.insert(name, source_label.clone());
                            }
                        }
                    }
                    ImportKind::Module => {
                        let binding = *import.path.last().expect("import path cannot be empty");
                        if let Some(existing) = reexported_from.get(&binding) {
                            push_reexport_collision(
                                &mut errors,
                                node.span,
                                binding,
                                existing,
                                &source_label,
                                "use `pub import X as alias;` to rename the re-exported module",
                            );
                        } else {
                            module_def.re_exported_modules.insert(binding, source_def);
                            reexported_from.insert(binding, source_label.clone());
                        }
                    }
                    ImportKind::ModuleAs(alias) => {
                        if let Some(existing) = reexported_from.get(alias) {
                            push_reexport_collision(
                                &mut errors,
                                node.span,
                                *alias,
                                existing,
                                &source_label,
                                "use a different alias to resolve the conflict",
                            );
                        } else {
                            module_def.re_exported_modules.insert(*alias, source_def);
                            reexported_from.insert(*alias, source_label.clone());
                        }
                    }
                }
            }
            Stmt::ExternFunc(node) => {
                let extern_func = &node.node;
                module_def.all_names.insert(extern_func.name);
                let (func_ty, param_info) = build_extern_func_registration(extern_func, resolve);
                module_def.funcs.insert(extern_func.name, func_ty);
                module_def
                    .func_param_info
                    .insert(extern_func.name, param_info);
            }
            Stmt::ExternType(node) => {
                let name = node.node.name;
                module_def.all_names.insert(name);
                let def = build_extern_type_def(node.node.has_init, &node.node.members, resolve);
                module_def.extern_types.insert(name, def);
            }
            Stmt::Const(node) => {
                module_def.all_names.insert(node.node.name);
            }
            Stmt::Extend(node) => {
                let decl = &node.node;
                if decl.visibility != Visibility::Public {
                    continue;
                }

                if !decl.type_params.is_empty() {
                    // build the exported extend target, unresolved names
                    // use a name-based "Struct", while typed targets resolve directly
                    let target_type = if let Type::UnresolvedName(name) = &decl.ty {
                        let type_args: Vec<Type> = decl
                            .type_params
                            .iter()
                            .map(|p| Type::UnresolvedName(p.name))
                            .collect();
                        Type::Struct {
                            name: *name,
                            type_args,
                        }
                    } else {
                        resolve(&decl.ty)
                    };
                    let Some(base_key) = extend_base_key(&target_type) else {
                        continue;
                    };
                    for method in &decl.methods {
                        if method.node.params.is_empty() {
                            continue;
                        }
                        if method.node.params[0].name.0.as_ref() != "self" {
                            continue;
                        }
                        module_def
                            .generic_extend_methods
                            .push(ModuleGenericExtendEntry {
                                base_name: base_key,
                                type_params: decl.type_params.clone(),
                                target_type: target_type.clone(),
                                method_name: method.node.name,
                                method: method.clone(),
                            });
                    }
                    continue;
                }

                let resolved_ty = resolve(&decl.ty);
                if !is_valid_concrete_extend_type(&resolved_ty) {
                    continue;
                }
                let module_str = module_path.join("::");
                for method in &decl.methods {
                    if method.node.params.is_empty() {
                        continue;
                    }
                    let self_param = &method.node.params[0];
                    if self_param.name.0.as_ref() != "self" {
                        continue;
                    }
                    let type_str = format!("{resolved_ty}");
                    let internal_name = Ident(Intern::new(format!(
                        "__extend::{}::{}::{}",
                        module_str, type_str, method.node.name
                    )));
                    let mut params = method.node.params.clone();
                    params[0].ty = resolved_ty.clone();
                    let ret = resolve(&method.node.ret);
                    let def = ExtendMethodDef {
                        params,
                        ret,
                        internal_name,
                    };
                    module_def.extend_methods.push(ModuleExtendEntry {
                        ty: resolved_ty.clone(),
                        name: method.node.name,
                        def,
                    });
                }
            }
            _ => {}
        }
    }

    (module_def, errors)
}

pub(super) fn check_stmt(
    stmt: &StmtNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    match &stmt.node {
        Stmt::Import(node) => {
            let import = &node.node;
            let path_key: Vec<String> = import.path.iter().map(ToString::to_string).collect();

            let Some(module_def) = type_checker.resolved_module_defs.get(&path_key) else {
                return;
            };
            let module_def = module_def.clone();
            let module_name = *import.path.last().expect("import path cannot be empty");

            if let ImportKind::Selective(items) = &import.kind {
                for item in items {
                    let is_public = module_def.funcs.contains_key(&item.name)
                        || module_def.struct_defs.contains_key(&item.name)
                        || module_def.enum_defs.contains_key(&item.name)
                        || module_def.extern_types.contains_key(&item.name)
                        || module_def.re_exported_modules.contains_key(&item.name)
                        || module_def.const_defs.contains_key(&item.name);
                    if !is_public {
                        let err_kind = if module_def.all_names.contains(&item.name) {
                            DiagnosticKind::PrivateModuleMember {
                                module: module_name,
                                member: item.name,
                            }
                        } else {
                            DiagnosticKind::UnknownModuleMember {
                                module: module_name,
                                member: item.name,
                            }
                        };
                        errors.push(Diagnostic::new(node.span, err_kind));
                    }
                }
            }
        }
        Stmt::ExternFunc(_) | Stmt::ExternType(_) => {}
        Stmt::Func(node) => {
            validate_annotations(&node.node.annotations, AnnotationTarget::Func, errors);
            check_func(node, type_checker, errors);
        }
        Stmt::Struct(node) | Stmt::DataRef(node) => {
            validate_annotations(&node.node.annotations, AnnotationTarget::Struct, errors);
            check_struct(node, type_checker, errors);
        }
        Stmt::Enum(node) => {
            validate_annotations(&node.node.annotations, AnnotationTarget::Enum, errors);
            let decl = &node.node;
            for variant in &decl.variants {
                validate_annotations(&variant.annotations, AnnotationTarget::Variant, errors);
                let has_any = match &variant.kind {
                    VariantKind::Unit => false,
                    VariantKind::Tuple(types) => types.iter().any(Type::contains_any),
                    VariantKind::Struct(fields) => fields.iter().any(|f| f.ty.contains_any()),
                };
                if has_any {
                    errors.push(Diagnostic::new(
                        node.span,
                        DiagnosticKind::AnyTypeNotAllowed,
                    ));
                }
                if let VariantKind::Struct(fields) = &variant.kind {
                    for field in fields {
                        validate_annotations(&field.annotations, AnnotationTarget::Field, errors);
                    }
                }
            }
        }
        Stmt::Expr(node) => {
            let _ = check_expr(node, type_checker, errors, None);
        }
        Stmt::Binding(node) => check_binding(node, type_checker, errors),
        Stmt::LetElse(node) => check_let_else(node, type_checker, errors),
        Stmt::Return(node) => check_ret(node, type_checker, errors),
        Stmt::While(node) => check_while(node, type_checker, errors),
        Stmt::WhileLet(node) => check_while_let(node, type_checker, errors),
        Stmt::For(node) => check_for(node, type_checker, errors),
        Stmt::Break => check_break(stmt.span, type_checker, errors),
        Stmt::Continue => check_continue(stmt.span, type_checker, errors),
        Stmt::Const(decl) => {
            // module-level consts are already processed by evaluate_module_consts,
            // only handle the function-body case here
            if type_checker.return_types.is_empty() {
                return;
            }

            let name = decl.node.name;

            let already_in_scope = type_checker
                .const_scope_stack
                .last()
                .is_some_and(|scope| scope.contains(&name));
            if already_in_scope {
                errors.push(Diagnostic::new(
                    stmt.span,
                    DiagnosticKind::DuplicateConst { name },
                ));
                return;
            }

            let known_consts: HashSet<Ident> = type_checker.const_defs.keys().copied().collect();
            if let Err(err) = validate_const_expr(&decl.node.value, &known_consts) {
                errors.push(err);
                return;
            }

            let annotated_ty = decl
                .node
                .ty
                .as_ref()
                .map(|ty| type_checker.resolve_type(ty));

            let _ = check_expr(
                &decl.node.value,
                type_checker,
                errors,
                annotated_ty.as_ref(),
            );

            let const_value = match eval_const_expr(&decl.node.value, &type_checker.const_defs) {
                Ok(val) => val,
                Err(err) => {
                    errors.push(err);
                    return;
                }
            };

            let value_ty = const_value.ty();
            if let Some(ref ann_ty) = annotated_ty
                && *ann_ty != value_ty
            {
                errors.push(Diagnostic::new(
                    decl.span,
                    DiagnosticKind::ConstTypeMismatch {
                        expected: ann_ty.clone(),
                        got: value_ty,
                    },
                ));
                return;
            }

            let final_ty = annotated_ty.unwrap_or_else(|| const_value.ty());
            type_checker.const_defs.insert(
                name,
                ConstDef {
                    ty: final_ty,
                    value: const_value,
                    visibility: decl.node.visibility,
                },
            );

            if let Some(scope) = type_checker.const_scope_stack.last_mut() {
                scope.insert(name);
            }
        }
        Stmt::Extend(node) => check_extend_decl(node, type_checker, errors),
        Stmt::Defer(node) => check_defer(node, type_checker, errors),
    }
}

pub(super) fn extend_base_key(ty: &Type) -> Option<Ident> {
    match ty {
        Type::Struct { name, .. }
        | Type::DataRef { name, .. }
        | Type::Enum { name, .. }
        | Type::Extern { name } => Some(*name),
        Type::List { .. } => Some(Ident(Intern::new("__List".to_string()))),
        Type::Map { .. } => Some(Ident(Intern::new("__Map".to_string()))),
        Type::Tuple(_) => Some(Ident(Intern::new("__Tuple".to_string()))),
        Type::Array { .. } => Some(Ident(Intern::new("__Array".to_string()))),
        _ => None,
    }
}

fn is_valid_concrete_extend_type(ty: &Type) -> bool {
    match ty {
        Type::Int
        | Type::Float
        | Type::Double
        | Type::Bool
        | Type::String
        | Type::Extern { .. } => true,
        Type::Struct { type_args, .. }
        | Type::DataRef { type_args, .. }
        | Type::Enum { type_args, .. } => type_args.iter().all(is_valid_concrete_extend_type),
        Type::List { elem } => is_valid_concrete_extend_type(elem),
        Type::Map { key, value } => {
            is_valid_concrete_extend_type(key) && is_valid_concrete_extend_type(value)
        }
        Type::Tuple(fields) => fields.iter().all(is_valid_concrete_extend_type),
        _ => false,
    }
}

fn check_extend_decl(
    node: &crate::ast::ExtendDeclNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let decl = &node.node;

    if !decl.type_params.is_empty() {
        check_generic_extend_decl(node, type_checker, errors);
        return;
    }

    let resolved_ty = type_checker.resolve_type(&decl.ty);

    if !is_valid_concrete_extend_type(&resolved_ty) {
        errors.push(Diagnostic::new(
            node.span,
            DiagnosticKind::ExtendUnsupportedType { ty: resolved_ty },
        ));
        return;
    }

    for method in &decl.methods {
        // Validate self param exists
        if method.node.params.is_empty() || method.node.params[0].name.0.as_ref() != "self" {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::ExtendMethodMissingSelf {
                    method: method.node.name,
                },
            ));
            continue;
        }

        // Validate no type annotation on self (ty must be Infer)
        if method.node.params[0].ty != Type::Infer {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::ExtendSelfTypeAnnotation {
                    method: method.node.name,
                },
            ));
            continue;
        }

        // Native conflict check
        let has_native_conflict = match &resolved_ty {
            Type::Struct { name, .. } | Type::DataRef { name, .. } => type_checker
                .get_struct(*name)
                .is_some_and(|def| def.methods.contains_key(&method.node.name)),
            Type::Extern { name } => type_checker
                .get_extern_type(*name)
                .is_some_and(|def| def.methods.contains_key(&method.node.name)),
            _ => false,
        };
        if has_native_conflict {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::ExtendMethodConflict {
                    ty: resolved_ty.clone(),
                    method: method.node.name,
                },
            ));
            continue;
        }

        // Duplicate check: count entries from current module (source_module == []) for this key
        let key = (resolved_ty.clone(), method.node.name);
        let local_count = type_checker.extend_defs.get(&key).map_or(0, |entries| {
            entries
                .iter()
                .filter(|e| e.source_module.is_empty())
                .count()
        });
        if local_count > 1 {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::DuplicateExtendMethod {
                    ty: resolved_ty.clone(),
                    method: method.node.name,
                },
            ));
            // Don't skip — still typecheck the body
        }

        // Build param list for body checking
        let is_var_self = matches!(method.node.params[0].mutability, Mutability::Mutable);
        let mut params: Vec<(Ident, Type, bool)> =
            vec![(method.node.params[0].name, resolved_ty.clone(), is_var_self)];
        for p in &method.node.params[1..] {
            params.push((
                p.name,
                type_checker.resolve_type(&p.ty),
                matches!(p.mutability, Mutability::Mutable),
            ));
        }

        let ret_ty = type_checker.resolve_type(&method.node.ret);
        check_body_common(
            &params,
            &method.node.body,
            &ret_ty,
            method.span,
            type_checker,
            errors,
        );
    }
}

fn check_generic_extend_decl(
    node: &crate::ast::ExtendDeclNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let decl = &node.node;

    // build the full extend target type, constructing legacy name-based forms manually and resolving typed forms directly
    let target_type = if let Type::UnresolvedName(base_name) = &decl.ty {
        let resolved = type_checker.resolve_type(&decl.ty);
        let type_args: Vec<Type> = decl
            .type_params
            .iter()
            .map(|p| Type::UnresolvedName(p.name))
            .collect();
        match &resolved {
            Type::Struct { name, .. } => {
                let def_tp = type_checker
                    .get_struct(*name)
                    .map_or(0, |d| d.type_params.len());
                if def_tp == 0 {
                    errors.push(Diagnostic::new(
                        node.span,
                        DiagnosticKind::ExtendTypeParamsOnNonGeneric {
                            ty_name: *base_name,
                        },
                    ));
                    return;
                }
                if decl.type_params.len() != def_tp {
                    errors.push(Diagnostic::new(
                        node.span,
                        DiagnosticKind::ExtendTypeParamCountMismatch {
                            ty_name: *base_name,
                            expected: def_tp,
                            found: decl.type_params.len(),
                        },
                    ));
                    return;
                }
                Type::Struct {
                    name: *name,
                    type_args,
                }
            }
            Type::DataRef { name, .. } => {
                let def_tp = type_checker
                    .get_struct(*name)
                    .map_or(0, |d| d.type_params.len());
                if def_tp == 0 {
                    errors.push(Diagnostic::new(
                        node.span,
                        DiagnosticKind::ExtendTypeParamsOnNonGeneric {
                            ty_name: *base_name,
                        },
                    ));
                    return;
                }
                if decl.type_params.len() != def_tp {
                    errors.push(Diagnostic::new(
                        node.span,
                        DiagnosticKind::ExtendTypeParamCountMismatch {
                            ty_name: *base_name,
                            expected: def_tp,
                            found: decl.type_params.len(),
                        },
                    ));
                    return;
                }
                Type::DataRef {
                    name: *name,
                    type_args,
                }
            }
            Type::Enum { name, .. } => {
                let def_tp = type_checker
                    .get_enum(*name)
                    .map_or(0, |d| d.type_params.len());
                if def_tp == 0 {
                    errors.push(Diagnostic::new(
                        node.span,
                        DiagnosticKind::ExtendTypeParamsOnNonGeneric {
                            ty_name: *base_name,
                        },
                    ));
                    return;
                }
                if decl.type_params.len() != def_tp {
                    errors.push(Diagnostic::new(
                        node.span,
                        DiagnosticKind::ExtendTypeParamCountMismatch {
                            ty_name: *base_name,
                            expected: def_tp,
                            found: decl.type_params.len(),
                        },
                    ));
                    return;
                }
                Type::Enum {
                    name: *name,
                    type_args,
                }
            }
            _ => {
                errors.push(Diagnostic::new(
                    node.span,
                    DiagnosticKind::ExtendUnsupportedType { ty: resolved },
                ));
                return;
            }
        }
    } else {
        // new path: "extend<T> type_expr", resolve concrete parts, type params stay as UnresolvedName
        type_checker.resolve_type(&decl.ty)
    };

    // extract the base key for hashmap lookup
    let Some(base_key) = extend_base_key(&target_type) else {
        errors.push(Diagnostic::new(
            node.span,
            DiagnosticKind::ExtendUnsupportedType { ty: target_type },
        ));
        return;
    };

    // validate all declared type params appear somewhere in target_type
    for param in &decl.type_params {
        let used = type_any(
            &target_type,
            &mut |t| matches!(t, Type::UnresolvedName(n) if *n == param.name),
        );
        if !used {
            errors.push(Diagnostic::new(
                node.span,
                DiagnosticKind::ExtendUnusedTypeParam {
                    param_name: param.name,
                },
            ));
            return;
        }
    }

    for method in &decl.methods {
        // validate self param exists
        if method.node.params.is_empty() || method.node.params[0].name.0.as_ref() != "self" {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::ExtendMethodMissingSelf {
                    method: method.node.name,
                },
            ));
            continue;
        }

        // validate no type annotation on self
        if method.node.params[0].ty != Type::Infer {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::ExtendSelfTypeAnnotation {
                    method: method.node.name,
                },
            ));
            continue;
        }

        // native conflict check (only for named struct/dataref types)
        let has_native_conflict = match &target_type {
            Type::Struct { name, .. } | Type::DataRef { name, .. } => type_checker
                .get_struct(*name)
                .is_some_and(|def| def.methods.contains_key(&method.node.name)),
            _ => false,
        };
        if has_native_conflict {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::ExtendMethodConflict {
                    ty: target_type.clone(),
                    method: method.node.name,
                },
            ));
            continue;
        }

        // duplicate check, reject only if there's already a local template with the same pattern
        let key = (base_key, method.node.name);
        let has_same_pattern =
            type_checker
                .generic_extend_templates
                .get(&key)
                .is_some_and(|entries| {
                    entries
                        .iter()
                        .any(|e| e.source_module.is_empty() && e.target_type == target_type)
                });
        if has_same_pattern {
            errors.push(Diagnostic::new(
                method.span,
                DiagnosticKind::DuplicateExtendMethod {
                    ty: target_type.clone(),
                    method: method.node.name,
                },
            ));
            continue;
        }

        // store template, body is NOT typechecked here (lazy specialization)
        type_checker
            .generic_extend_templates
            .entry(key)
            .or_default()
            .push(GenericExtendTemplate {
                type_params: decl.type_params.clone(),
                target_type: target_type.clone(),
                method: method.clone(),
                source_module: vec![],
                binding: Ident(Intern::new(String::new())),
            });
    }
}

pub(super) fn check_binding(
    binding: &crate::ast::BindingNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let node = &binding.node;
    if let Some(annot_ty) = &node.ty
        && annot_ty.contains_any()
    {
        errors.push(Diagnostic::new(
            binding.span,
            DiagnosticKind::AnyTypeNotAllowed,
        ));
    }
    let expected = node
        .ty
        .as_ref()
        .map(|annot_ty| type_checker.resolve_type(annot_ty));
    check_expr(&node.value, type_checker, errors, expected.as_ref());

    if is_if_without_else(&node.value) {
        errors.push(
            Diagnostic::new(node.value.span, DiagnosticKind::IfMissingElse).with_help(
                "add an `else` branch, or end the `if` with `;` if you meant a statement",
            ),
        );
    }

    let val_ref = TypeRef::Expr(node.value.node.id);

    let value_ty = type_checker
        .get_type(node.value.node.id)
        .map_or(Type::Infer, |(_, ty)| ty.clone());

    let binding_ty = if let Some(annot_ty) = &node.ty {
        let resolved_annot = resolve_array_infer_annotation(annot_ty, &value_ty);
        let resolved_annot = type_checker.resolve_type(&resolved_annot);

        // validate map type annotation are keyable
        if let Type::Map { key, .. } = &resolved_annot {
            validate_map_key_type(binding.span, key, type_checker, errors);
        }

        let should_retag_literal = is_array_literal_with_infer_elem(&node.value, &value_ty)
            || is_array_lit_with_list_annotation(&node.value, &resolved_annot);

        if should_retag_literal {
            update_array_literal_typ(&node.value, &resolved_annot, type_checker);
        }

        let annot_ref = TypeRef::concrete(&resolved_annot);
        type_checker.constrain_assignable(binding.span, val_ref, annot_ref, errors);

        resolved_annot
    } else {
        if is_all_nil_array_literal(&node.value) {
            errors.push(
                Diagnostic::new(node.value.span, DiagnosticKind::ArrayAllNilAmbiguous)
                    .with_help("add a type annotation, e.g. `let a: [int?; _] = [nil, nil];`"),
            );
        }
        if is_empty_map_literal(&node.value) {
            errors.push(Diagnostic::new(
                node.value.span,
                DiagnosticKind::MapEmptyLiteralNoContext,
            ));
        }
        value_ty
    };

    let mutable = matches!(node.mutability, Mutability::Mutable);
    check_pattern(&node.pattern, &binding_ty, mutable, type_checker, errors);
}

pub(super) fn check_let_else(
    let_else_node: &crate::ast::LetElseNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let node = &let_else_node.node;

    let value_ty = check_expr(&node.value, type_checker, errors, None);

    if !is_refutable(&node.pattern.node, &value_ty, type_checker) {
        errors.push(Diagnostic::new(
            node.pattern.span,
            DiagnosticKind::LetElseIrrefutable,
        ));
        check_pattern(&node.pattern, &value_ty, false, type_checker, errors);
        return;
    }

    type_checker.push_scope();
    check_block_stmts(
        &node.else_block.node.stmts,
        node.else_block.node.tail.as_deref(),
        type_checker,
        errors,
        None,
    );
    type_checker.pop_scope();

    if !block_always_diverges(&node.else_block) {
        errors.push(Diagnostic::new(
            node.else_block.span,
            DiagnosticKind::LetElseMustDiverge,
        ));
    }

    check_pattern(&node.pattern, &value_ty, false, type_checker, errors);
}

fn is_array_literal_with_infer_elem(expr: &ExprNode, value_ty: &Type) -> bool {
    let is_array_literal = matches!(
        &expr.node.kind,
        ExprKind::ArrayLiteral(_) | ExprKind::ArrayFill(_)
    );
    let has_infer_elem = match unwrap_opt_typ(value_ty) {
        Type::Array { elem, .. } => contains_infer(elem),
        _ => false,
    };
    is_array_literal && has_infer_elem
}

fn is_array_lit_with_list_annotation(expr: &ExprNode, annot_ty: &Type) -> bool {
    let is_array_literal = matches!(
        &expr.node.kind,
        ExprKind::ArrayLiteral(_) | ExprKind::ArrayFill(_)
    );
    let is_list = unwrap_opt_typ(annot_ty).is_list();
    is_array_literal && is_list
}

fn update_array_literal_typ(expr: &ExprNode, annot_ty: &Type, type_checker: &mut TypeChecker) {
    let base_annot = unwrap_opt_typ(annot_ty);

    match &expr.node.kind {
        ExprKind::ArrayLiteral(lit) => match base_annot {
            Type::Array { elem, len } => {
                let new_ty = Type::Array {
                    elem: elem.clone(),
                    len: *len,
                };
                type_checker.set_type(expr.node.id, new_ty, expr.span);
                for el in &lit.node.elements {
                    type_checker.set_type(el.node.id, *elem.clone(), el.span);
                }
            }
            Type::List { elem } => {
                let new_ty = Type::List { elem: elem.clone() };
                type_checker.set_type(expr.node.id, new_ty, expr.span);
                for el in &lit.node.elements {
                    type_checker.set_type(el.node.id, *elem.clone(), el.span);
                }
            }
            _ => {}
        },
        ExprKind::ArrayFill(fill) => match base_annot {
            Type::Array { elem, len } => {
                let new_ty = Type::Array {
                    elem: elem.clone(),
                    len: *len,
                };
                type_checker.set_type(expr.node.id, new_ty, expr.span);
                type_checker.set_type(fill.node.value.node.id, *elem.clone(), fill.node.value.span);
            }
            Type::List { elem } => {
                let new_ty = Type::List { elem: elem.clone() };
                type_checker.set_type(expr.node.id, new_ty, expr.span);
                type_checker.set_type(fill.node.value.node.id, *elem.clone(), fill.node.value.span);
            }
            _ => {}
        },
        _ => {}
    }
}

fn resolve_array_infer_annotation(annot_ty: &Type, value_ty: &Type) -> Type {
    match annot_ty {
        Type::Array {
            elem,
            len: ArrayLen::Infer,
        } => {
            let inferred_len = match value_ty {
                Type::Array {
                    len: ArrayLen::Fixed(n),
                    ..
                } => ArrayLen::Fixed(*n),
                _ => ArrayLen::Infer,
            };
            Type::Array {
                elem: elem.clone(),
                len: inferred_len,
            }
        }
        Type::Array { elem, len } => {
            let resolved_elem = match value_ty {
                Type::Array {
                    elem: value_elem, ..
                } if value_elem.is_infer() => elem.clone(),
                _ => elem.clone(),
            };
            Type::Array {
                elem: resolved_elem,
                len: *len,
            }
        }
        Type::List { elem } => {
            let resolved_elem = match value_ty {
                Type::Array {
                    elem: value_elem, ..
                } if value_elem.is_infer() => elem.clone(),
                _ => elem.clone(),
            };
            Type::List {
                elem: resolved_elem,
            }
        }
        annot_ty if annot_ty.is_option() => {
            let inner = annot_ty.option_inner().expect("is_option guarantees inner");
            Type::option_of(resolve_array_infer_annotation(inner, value_ty))
        }
        _ => annot_ty.clone(),
    }
}

pub(super) fn check_ret(
    ret: &ReturnNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    if type_checker.in_defer {
        errors.push(Diagnostic::new(ret.span, DiagnosticKind::ReturnInDefer));
        return;
    }

    let node = &ret.node;

    // if return is outside a function then we just return (although this shouldn't happen)
    let Some(expected_ret) = type_checker.current_return_type().cloned() else {
        return;
    };

    type_checker.mark_explicit_return();

    match (&node.value, &expected_ret) {
        // returning a value in a non-void fn needs constraining
        (Some(value_expr), expected_ty) => {
            check_expr(value_expr, type_checker, errors, Some(expected_ty));
            let expr_ref = TypeRef::Expr(value_expr.node.id);
            let ret_ref = TypeRef::concrete(expected_ty);
            type_checker.constrain_assignable(ret.span, expr_ref, ret_ref, errors);
        }

        // returning nothing in a void fn is fine
        (None, Type::Void) => {}

        // returning nothing in a non-void fn is invalid
        (None, expected_ty) => {
            errors.push(Diagnostic::new(
                ret.span,
                DiagnosticKind::MismatchedTypes {
                    expected: expected_ty.clone(),
                    found: Type::Void,
                },
            ));
        }
    }
}

fn check_break(span: Span, type_checker: &mut TypeChecker, errors: &mut Vec<Diagnostic>) {
    if type_checker.in_defer {
        errors.push(Diagnostic::new(span, DiagnosticKind::BreakInDefer));
        return;
    }
    if !type_checker.in_loop() {
        errors.push(
            Diagnostic::new(span, DiagnosticKind::BreakOutsideLoop)
                .with_help("break can only appear inside `while` or `for` loops"),
        );
    }
}

fn check_continue(span: Span, type_checker: &mut TypeChecker, errors: &mut Vec<Diagnostic>) {
    if type_checker.in_defer {
        errors.push(Diagnostic::new(span, DiagnosticKind::ContinueInDefer));
        return;
    }
    if !type_checker.in_loop() {
        errors.push(
            Diagnostic::new(span, DiagnosticKind::ContinueOutsideLoop)
                .with_help("continue can only appear inside `while` or `for` loops"),
        );
    }
}

fn check_defer(
    defer_node: &crate::ast::DeferNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let prev_in_defer = type_checker.in_defer;
    type_checker.in_defer = true;

    match &defer_node.node.body {
        DeferBody::Expr(expr) => {
            check_expr(expr, type_checker, errors, None);
        }
        DeferBody::Block(block) => {
            check_block_stmts(
                &block.node.stmts,
                block.node.tail.as_deref(),
                type_checker,
                errors,
                None,
            );
        }
    }

    type_checker.in_defer = prev_in_defer;
}
