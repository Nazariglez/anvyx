use crate::{
    ast::{
        ArrayLen, BlockNode, ExprId, ExprKind, ExprNode, Ident, ImportKind, Mutability, ReturnNode,
        Stmt, StmtNode, Type,
    },
    span::Span,
};
use std::collections::HashMap;

use super::{
    composite::{is_all_nil_array_literal, is_empty_map_literal},
    constraint::TypeRef,
    control::{check_for, check_while, is_if_without_else},
    decl::{check_func, check_struct},
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    infer::type_from_fn,
    pattern::check_pattern,
    types::{
        EnumDef, EnumVariantDef, MethodDef, ModuleDef, StructDef, TypeChecker, is_keyable,
        keyable_reason, unwrap_opt_typ,
    },
    unify::contains_infer,
};

pub(super) fn check_block_stmts(
    stmts: &[StmtNode],
    tail: Option<&ExprNode>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Option<ExprId> {
    type_checker.push_scope();
    collect_scope_types(stmts, type_checker);

    for stmt in stmts {
        check_stmt(stmt, type_checker, errors);
    }

    let tail_id = tail.map(|expr| {
        let _ = check_expr(expr, type_checker, errors);
        expr.node.id
    });

    type_checker.pop_scope();
    tail_id
}

pub(super) fn check_block_expr(
    block: &BlockNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> (Type, Option<ExprId>) {
    let last_expr_id = check_block_stmts(
        &block.node.stmts,
        block.node.tail.as_deref(),
        type_checker,
        errors,
    );
    let Some(id) = last_expr_id else {
        return (Type::Void, None);
    };

    let ty = type_checker
        .get_type(id)
        .cloned()
        .map(|(_, ty)| ty)
        .unwrap_or(Type::Void);

    (ty, Some(id))
}

pub(super) fn collect_scope_types(stmts: &[StmtNode], type_checker: &mut TypeChecker) {
    // register extern type names before anything else, so they can be
    // resolved when building function signatures that reference them
    for stmt in stmts {
        if let Stmt::ExternType(node) = &stmt.node {
            type_checker.extern_type_defs.insert(node.node.name);
        }
    }

    for stmt in stmts {
        match &stmt.node {
            Stmt::ExternType(_) => {
                // already handled in the pre-pass above
            }

            Stmt::ExternFunc(node) => {
                let extern_func = &node.node;
                let func_ty = Type::Func {
                    params: extern_func
                        .params
                        .iter()
                        .map(|p| type_checker.resolve_type(&p.ty))
                        .collect(),
                    ret: Box::new(type_checker.resolve_type(&extern_func.ret)),
                };
                type_checker.set_var(extern_func.name, func_ty, false);
                let param_info: Vec<_> = extern_func
                    .params
                    .iter()
                    .map(|p| (p.name, p.mutability))
                    .collect();
                type_checker
                    .func_param_info
                    .insert(extern_func.name, param_info);
            }

            Stmt::Func(node) => {
                let func = &node.node;
                type_checker.set_var(func.name, type_from_fn(func), false);

                let param_info: Vec<_> =
                    func.params.iter().map(|p| (p.name, p.mutability)).collect();
                type_checker.func_param_info.insert(func.name, param_info);

                // if the function is generic, store the type parameters and template
                // because they are not fully typechecked at definition time until they are used
                let is_generic = !func.type_params.is_empty();
                if is_generic {
                    type_checker
                        .func_type_params
                        .insert(func.name, func.type_params.clone());

                    // store the function ast for later instantiation
                    type_checker
                        .generic_func_templates
                        .insert(func.name, node.clone());
                }
            }

            Stmt::Struct(node) => {
                let decl = &node.node;

                let methods = decl
                    .methods
                    .iter()
                    .map(|method| {
                        (
                            method.name,
                            MethodDef {
                                type_params: method.type_params.clone(),
                                receiver: method.receiver,
                                params: method.params.clone(),
                                ret: method.ret.clone(),
                                body: method.body.clone(),
                            },
                        )
                    })
                    .collect::<HashMap<_, _>>();

                type_checker.struct_defs.insert(
                    decl.name,
                    StructDef {
                        type_params: decl.type_params.clone(),
                        fields: decl.fields.clone(),
                        methods,
                    },
                );
            }

            Stmt::Enum(node) => {
                let decl = &node.node;

                let variants = decl
                    .variants
                    .iter()
                    .map(|v| EnumVariantDef {
                        name: v.name,
                        kind: v.kind.clone(),
                    })
                    .collect();

                type_checker.enum_defs.insert(
                    decl.name,
                    EnumDef {
                        type_params: decl.type_params.clone(),
                        variants,
                    },
                );
            }

            Stmt::Import(node) => {
                let import = &node.node;
                let path_key: Vec<String> = import.path.iter().map(|id| id.to_string()).collect();

                // clone the stmts to release the borrow on type_checker before calling build_module_def_from_stmts
                let module_stmts = type_checker.resolved_module_stmts.get(&path_key).cloned();
                let Some(stmts) = module_stmts else {
                    // module wasn't resolved (std.*, unresolved error, etc.)
                    continue;
                };

                let module_def = build_module_def_from_stmts(&stmts, type_checker);

                match &import.kind {
                    ImportKind::Module => {
                        let binding = *import.path.last().expect("import path cannot be empty");
                        type_checker.module_defs.insert(binding, module_def);
                    }
                    ImportKind::ModuleAs(alias) => {
                        type_checker.module_defs.insert(*alias, module_def);
                    }
                    ImportKind::Selective(items) => {
                        for item in items {
                            let bind_as = item.alias.unwrap_or(item.name);
                            inject_module_item(item.name, bind_as, &module_def, type_checker);
                        }
                    }
                    ImportKind::Wildcard => {
                        let names: Vec<Ident> = module_def
                            .funcs
                            .keys()
                            .chain(module_def.struct_defs.keys())
                            .chain(module_def.enum_defs.keys())
                            .copied()
                            .collect();
                        for name in names {
                            inject_module_item(name, name, &module_def, type_checker);
                        }
                    }
                }
            }

            _ => {}
        }
    }
}

fn build_module_def_from_stmts(stmts: &[StmtNode], type_checker: &TypeChecker) -> ModuleDef {
    let mut module_def = ModuleDef::default();

    for stmt in stmts {
        match &stmt.node {
            Stmt::Func(node) => {
                let func = &node.node;
                let raw_ty = type_from_fn(func);
                let func_ty = type_checker.resolve_type(&raw_ty);
                module_def.funcs.insert(func.name, func_ty);

                let param_info: Vec<_> =
                    func.params.iter().map(|p| (p.name, p.mutability)).collect();
                module_def.func_param_info.insert(func.name, param_info);

                if !func.type_params.is_empty() {
                    module_def
                        .func_type_params
                        .insert(func.name, func.type_params.clone());
                    module_def
                        .generic_func_templates
                        .insert(func.name, node.clone());
                }
            }
            Stmt::Struct(node) => {
                let decl = &node.node;
                let methods = decl
                    .methods
                    .iter()
                    .map(|m| {
                        (
                            m.name,
                            MethodDef {
                                type_params: m.type_params.clone(),
                                receiver: m.receiver,
                                params: m.params.clone(),
                                ret: m.ret.clone(),
                                body: m.body.clone(),
                            },
                        )
                    })
                    .collect();
                module_def.struct_defs.insert(
                    decl.name,
                    StructDef {
                        type_params: decl.type_params.clone(),
                        fields: decl.fields.clone(),
                        methods,
                    },
                );
            }
            Stmt::Enum(node) => {
                let decl = &node.node;
                let variants = decl
                    .variants
                    .iter()
                    .map(|v| EnumVariantDef {
                        name: v.name,
                        kind: v.kind.clone(),
                    })
                    .collect();
                module_def.enum_defs.insert(
                    decl.name,
                    EnumDef {
                        type_params: decl.type_params.clone(),
                        variants,
                    },
                );
            }
            _ => {}
        }
    }

    module_def
}

fn inject_module_item(
    name: Ident,
    bind_as: Ident,
    module_def: &ModuleDef,
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
        }
    } else if let Some(struct_def) = module_def.struct_defs.get(&name) {
        type_checker.struct_defs.insert(bind_as, struct_def.clone());
    } else if let Some(enum_def) = module_def.enum_defs.get(&name) {
        type_checker.enum_defs.insert(bind_as, enum_def.clone());
    }

    // symbol not found is silently skipped here, check_stmt will report the error
}

pub(super) fn check_stmt(
    stmt: &StmtNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    match &stmt.node {
        Stmt::Import(node) => {
            let import = &node.node;
            let path_key: Vec<String> = import.path.iter().map(|id| id.to_string()).collect();

            let module_stmts = type_checker.resolved_module_stmts.get(&path_key).cloned();
            let Some(stmts) = module_stmts else {
                return;
            };

            let module_def = build_module_def_from_stmts(&stmts, type_checker);
            let module_name = *import.path.last().expect("import path cannot be empty");

            if let ImportKind::Selective(items) = &import.kind {
                for item in items {
                    let exists = module_def.funcs.contains_key(&item.name)
                        || module_def.struct_defs.contains_key(&item.name)
                        || module_def.enum_defs.contains_key(&item.name);
                    if !exists {
                        errors.push(TypeErr::new(
                            node.span,
                            TypeErrKind::UnknownModuleMember {
                                module: module_name,
                                member: item.name,
                            },
                        ));
                    }
                }
            }
        }
        Stmt::ExternFunc(_) => {}
        Stmt::ExternType(_) => {}
        Stmt::Func(node) => check_func(node, type_checker, errors),
        Stmt::Struct(node) => check_struct(node, type_checker, errors),
        Stmt::Enum(_) => {
            // enum declarations are collected in collect_scope_types
            // no additional checking needed at this point
        }
        Stmt::Expr(node) => {
            let _ = check_expr(node, type_checker, errors);
        }
        Stmt::Binding(node) => check_binding(node, type_checker, errors),
        Stmt::Return(node) => check_ret(node, type_checker, errors),
        Stmt::While(node) => check_while(node, type_checker, errors),
        Stmt::For(node) => check_for(node, type_checker, errors),
        Stmt::Break => check_break(stmt.span, type_checker, errors),
        Stmt::Continue => check_continue(stmt.span, type_checker, errors),
    }
}

pub(super) fn check_binding(
    binding: &crate::ast::BindingNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let node = &binding.node;
    check_expr(&node.value, type_checker, errors);

    if is_if_without_else(&node.value) {
        errors.push(
            TypeErr::new(node.value.span, TypeErrKind::IfMissingElse).with_help(
                "add an `else` branch, or end the `if` with `;` if you meant a statement",
            ),
        );
    }

    let val_ref = TypeRef::Expr(node.value.node.id);

    let value_ty = type_checker
        .get_type(node.value.node.id)
        .map(|(_, ty)| ty.clone())
        .unwrap_or(Type::Infer);

    let binding_ty = match &node.ty {
        Some(annot_ty) => {
            let resolved_annot = resolve_array_infer_annotation(annot_ty, &value_ty);
            let resolved_annot = type_checker.resolve_type(&resolved_annot);

            // validate map type annotation are keyable
            if let Type::Map { key, .. } = &resolved_annot {
                let is_optional_key = key.is_option();
                let is_infer = matches!(key.as_ref(), Type::Infer);
                let is_float_key = matches!(key.as_ref(), Type::Float);
                if is_optional_key {
                    errors.push(TypeErr::new(
                        binding.span,
                        TypeErrKind::MapOptionalKeyNotAllowed {
                            found: (**key).clone(),
                        },
                    ));
                } else if is_float_key {
                    errors.push(TypeErr::new(binding.span, TypeErrKind::MapKeyFloat));
                } else if !is_keyable(key, type_checker) && !is_infer {
                    let mut err = TypeErr::new(
                        binding.span,
                        TypeErrKind::MapKeyNotKeyable {
                            found: (**key).clone(),
                        },
                    );
                    if let Some(reason) = keyable_reason(key, type_checker) {
                        err.notes.push(reason);
                    }
                    errors.push(err);
                }
            }

            let should_retag_literal = is_array_literal_with_infer_elem(&node.value, &value_ty)
                || is_array_lit_with_list_annotation(&node.value, &resolved_annot);

            if should_retag_literal {
                update_array_literal_typ(&node.value, &resolved_annot, type_checker);
            }

            let annot_ref = TypeRef::Concrete(resolved_annot.clone());
            type_checker.constrain_assignable(binding.span, val_ref, annot_ref, errors);

            resolved_annot
        }
        None => {
            if is_all_nil_array_literal(&node.value) {
                errors.push(
                    TypeErr::new(node.value.span, TypeErrKind::ArrayAllNilAmbiguous)
                        .with_help("add a type annotation, e.g. `let a: [int?; _] = [nil, nil];`"),
                );
            }
            if is_empty_map_literal(&node.value) {
                errors.push(TypeErr::new(
                    node.value.span,
                    TypeErrKind::MapEmptyLiteralNoContext,
                ));
            }
            value_ty
        }
    };

    let mutable = matches!(node.mutability, Mutability::Mutable);
    check_pattern(&node.pattern, &binding_ty, mutable, type_checker, errors);
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
    errors: &mut Vec<TypeErr>,
) {
    let node = &ret.node;

    // if return is outside a function then we just return (although this shouldn't happen)
    let Some(expected_ret) = type_checker.current_return_type().cloned() else {
        return;
    };

    type_checker.mark_explicit_return();

    match (&node.value, &expected_ret) {
        // returning a value in a non-void fn needs constraining
        (Some(value_expr), expected_ty) => {
            check_expr(value_expr, type_checker, errors);
            let expr_ref = TypeRef::Expr(value_expr.node.id);
            let ret_ref = TypeRef::Concrete(expected_ty.clone());
            type_checker.constrain_assignable(ret.span, expr_ref, ret_ref, errors);
        }

        // returning nothing in a void fn is fine
        (None, Type::Void) => {}

        // returning nothing in a non-void fn is invalid
        (None, expected_ty) => {
            errors.push(TypeErr::new(
                ret.span,
                TypeErrKind::MismatchedTypes {
                    expected: expected_ty.clone(),
                    found: Type::Void,
                },
            ));
        }
    }
}

fn check_break(span: Span, type_checker: &mut TypeChecker, errors: &mut Vec<TypeErr>) {
    if !type_checker.in_loop() {
        errors.push(
            TypeErr::new(span, TypeErrKind::BreakOutsideLoop)
                .with_help("break can only appear inside `while` or `for` loops"),
        );
    }
}

fn check_continue(span: Span, type_checker: &mut TypeChecker, errors: &mut Vec<TypeErr>) {
    if !type_checker.in_loop() {
        errors.push(
            TypeErr::new(span, TypeErrKind::ContinueOutsideLoop)
                .with_help("continue can only appear inside `while` or `for` loops"),
        );
    }
}
