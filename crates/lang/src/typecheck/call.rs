use crate::{
    ast::{
        CallNode, ExprKind, ExprNode, FieldAccessNode, FuncNode, Ident, Lit, MethodReceiver,
        Mutability, Type, TypeParam, VariantKind,
    },
    span::Span,
};
use internment::Intern;
use std::collections::HashMap;

use super::{
    constraint::{TypeRef, resolve_constraints},
    decl::{check_body_common, check_fn_body},
    error::{TypeErr, TypeErrKind},
    expr::{check_expr, root_ident},
    infer::{
        build_param_ref, build_subst, constrain_slots_from_type, create_inference_slots,
        infer_type_args_from_call, instantiate_func_type, subst_type,
    },
    types::{
        EnumDef, ExternMethodDef, ExternTypeDef, MethodContext, MethodDef, MethodSpecKey,
        ModuleDef, SpecializationKey, SpecializationResult, StructDef, TypeChecker,
    },
};

pub(super) fn check_var_param_args(
    params: impl IntoIterator<Item = (Ident, Mutability)>,
    args: &[ExprNode],
    type_checker: &TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    for ((param_name, mutability), arg) in params.into_iter().zip(args.iter()) {
        if mutability != Mutability::Mutable {
            continue;
        }

        let Some(root) = root_ident(arg) else {
            errors.push(
                TypeErr::new(
                    arg.span,
                    TypeErrKind::VarParamNotLvalue { param: param_name },
                )
                .with_help("pass a variable declared with 'var', not a literal or expression"),
            );
            continue;
        };

        let Some(info) = type_checker.get_var(root) else {
            continue;
        };

        if !info.mutable {
            errors.push(
                TypeErr::new(
                    arg.span,
                    TypeErrKind::VarParamImmutableBinding {
                        param: param_name,
                        binding: root,
                    },
                )
                .with_help("declare with 'var' to allow mutation"),
            );
        }
    }
}

fn check_receiver_mutability(
    target: &ExprNode,
    type_label: Ident,
    method_name: Ident,
    type_checker: &TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let Some(root) = root_ident(target) else {
        return;
    };
    let Some(info) = type_checker.get_var(root) else {
        return;
    };
    if !info.mutable {
        errors.push(
            TypeErr::new(
                target.span,
                TypeErrKind::MutatingMethodOnImmutable {
                    struct_name: type_label,
                    method: method_name,
                },
            )
            .with_help("declare with 'var' to allow calling mutating methods"),
        );
    }
}

fn list_type_label() -> Ident {
    use std::sync::LazyLock;
    static LABEL: LazyLock<Ident> = LazyLock::new(|| Ident(Intern::new("list".to_string())));
    *LABEL
}

fn map_type_label() -> Ident {
    use std::sync::LazyLock;
    static LABEL: LazyLock<Ident> = LazyLock::new(|| Ident(Intern::new("map".to_string())));
    *LABEL
}

pub(super) fn check_list_method(
    call: &CallNode,
    target: &ExprNode,
    method_name: Ident,
    elem: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let label = list_type_label();
    let node = &call.node;
    match method_name.0.as_ref().as_str() {
        "push" => {
            check_receiver_mutability(target, label, method_name, type_checker, errors);
            check_call_signature(
                call.span,
                std::slice::from_ref(elem),
                &Type::Void,
                &node.args,
                type_checker,
                errors,
            )
        }
        "pop" => {
            check_receiver_mutability(target, label, method_name, type_checker, errors);
            check_call_signature(
                call.span,
                &[],
                &Type::option_of(elem.clone()),
                &node.args,
                type_checker,
                errors,
            )
        }
        _ => {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::UnknownMethod {
                    struct_name: label,
                    method: method_name,
                },
            ));
            Type::Infer
        }
    }
}

pub(super) fn check_map_method(
    call: &CallNode,
    target: &ExprNode,
    method_name: Ident,
    key: &Type,
    value: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let label = map_type_label();
    let node = &call.node;
    match method_name.0.as_ref().as_str() {
        "insert" => {
            check_receiver_mutability(target, label, method_name, type_checker, errors);
            check_call_signature(
                call.span,
                &[key.clone(), value.clone()],
                &Type::Void,
                &node.args,
                type_checker,
                errors,
            )
        }
        "remove" => {
            check_receiver_mutability(target, label, method_name, type_checker, errors);
            check_call_signature(
                call.span,
                std::slice::from_ref(key),
                &Type::option_of(value.clone()),
                &node.args,
                type_checker,
                errors,
            )
        }
        _ => {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::UnknownMethod {
                    struct_name: label,
                    method: method_name,
                },
            ));
            Type::Infer
        }
    }
}

fn check_generic_call(
    call: &CallNode,
    func_name: Ident,
    func_ty: &Type,
    type_params: &[TypeParam],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
    expected: Option<&Type>,
) -> Type {
    let node = &call.node;
    let has_type_args = !node.type_args.is_empty();

    let type_args = if has_type_args {
        let same_param_count = type_params.len() == node.type_args.len();
        if !same_param_count {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::GenericArgNumMismatch {
                    expected: type_params.len(),
                    found: node.type_args.len(),
                },
            ));
            return Type::Infer;
        }

        let Type::Func { params, ret: _ } = func_ty else {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::NotAFunction {
                    expr_type: func_ty.clone(),
                },
            ));
            return Type::Infer;
        };

        let params_count = params.len() == node.args.len();
        if !params_count {
            let instantiated_ty =
                instantiate_func_type(type_params, func_ty, &node.type_args, call.span, errors);
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::MismatchedTypes {
                    expected: instantiated_ty.unwrap_or_else(|| func_ty.clone()),
                    found: Type::Func {
                        params: vec![Type::Infer; node.args.len()],
                        ret: Box::new(Type::Infer),
                    },
                },
            ));
            return Type::Infer;
        }

        let subst = build_subst(type_params, &node.type_args);

        for (arg_expr, param_ty) in node.args.iter().zip(params.iter()) {
            let instantiated = subst_type(param_ty, &subst);
            check_and_constrain_arg(arg_expr, &instantiated, type_checker, errors);
        }

        node.type_args.clone()
    } else {
        let Type::Func { params, ret } = func_ty else {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::NotAFunction {
                    expr_type: func_ty.clone(),
                },
            ));
            return Type::Infer;
        };

        let Some(inferred) = infer_type_args_from_call(
            call.span,
            type_params,
            params,
            &node.args,
            func_ty.clone(),
            ret,
            expected,
            type_checker,
            errors,
        ) else {
            return Type::Infer;
        };

        inferred
    };

    if let Some(param_info) = type_checker.func_param_info.get(&func_name).cloned() {
        check_var_param_args(param_info, &node.args, type_checker, errors);
    }

    let ret = instantiate_and_check_fn(func_name, &type_args, call.span, type_checker, errors);
    type_checker
        .resolved_call_type_args
        .insert(node.func.node.id, (func_name, type_args));
    ret
}

pub(super) fn check_call(
    call: &CallNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
    expected: Option<&Type>,
) -> Type {
    let node = &call.node;

    if let ExprKind::Field(field_access) = &node.func.node.kind
        && let Some(result) = try_check_method_call(call, field_access, type_checker, errors)
    {
        return result;
    }

    let func_ty = check_expr(&node.func, type_checker, errors, None);

    // try to get the function name to look up type parameters
    let func_name = match &node.func.node.kind {
        ExprKind::Ident(ident) => Some(*ident),
        _ => None,
    };

    // lookup type params for generic functions
    let type_params = func_name
        .and_then(|name| type_checker.func_type_params.get(&name))
        .cloned()
        .unwrap_or_default();

    let is_generic = !type_params.is_empty();
    let has_type_args = !node.type_args.is_empty();

    // handle generic function calls with template instantiation
    match (is_generic, has_type_args) {
        (false, true) => {
            errors.push(TypeErr::new(call.span, TypeErrKind::NotGenericFunction));
            return Type::Infer;
        }
        (true, _) => {
            let Some(name) = func_name else {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::NotAFunction {
                        expr_type: func_ty.clone(),
                    },
                ));
                return Type::Infer;
            };
            return check_generic_call(
                call,
                name,
                &func_ty,
                &type_params,
                type_checker,
                errors,
                expected,
            );
        }
        (false, false) => {}
    }

    // fallback to normal call
    let result = check_call_with_type(call, func_ty, type_checker, errors);

    if let Some(name) = func_name
        && let Some(param_info) = type_checker.func_param_info.get(&name).cloned()
    {
        check_var_param_args(param_info, &node.args, type_checker, errors);
    }

    result
}

fn check_and_constrain_arg(
    arg_expr: &ExprNode,
    param_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    check_expr(arg_expr, type_checker, errors, Some(param_ty));
    let arg_ref = TypeRef::Expr(arg_expr.node.id);
    let param_ref = TypeRef::concrete(param_ty);
    type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
}

pub(super) fn check_call_signature(
    call_span: Span,
    param_types: &[Type],
    ret_type: &Type,
    args: &[ExprNode],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    if args.len() != param_types.len() {
        let expected = Type::Func {
            params: param_types.to_vec(),
            ret: Box::new(ret_type.clone()),
        };
        let found = Type::Func {
            params: vec![Type::Infer; args.len()],
            ret: Box::new(Type::Infer),
        };
        errors.push(TypeErr::new(
            call_span,
            TypeErrKind::MismatchedTypes { expected, found },
        ));
        return Type::Infer;
    }

    for (arg_expr, param_ty) in args.iter().zip(param_types.iter()) {
        check_expr(arg_expr, type_checker, errors, Some(param_ty));
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let param_ref = TypeRef::concrete(param_ty);
        type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
    }

    ret_type.clone()
}

fn check_call_with_type(
    call: &CallNode,
    func_ty: Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &call.node;

    match &func_ty {
        Type::Func { params, ret } => {
            check_call_signature(call.span, params, ret, &node.args, type_checker, errors)
        }
        _ => {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::NotAFunction { expr_type: func_ty },
            ));
            Type::Infer
        }
    }
}

pub(super) fn check_module_func_call(
    call: &CallNode,
    module_name: Ident,
    func_name: Ident,
    module_def: &ModuleDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
    expected: Option<&Type>,
) -> Type {
    let Some(func_ty) = module_def.funcs.get(&func_name).cloned() else {
        let err_kind = if module_def.all_names.contains(&func_name) {
            TypeErrKind::PrivateModuleMember {
                module: module_name,
                member: func_name,
            }
        } else {
            TypeErrKind::UnknownModuleMember {
                module: module_name,
                member: func_name,
            }
        };
        errors.push(TypeErr::new(call.span, err_kind));
        return Type::Infer;
    };

    let type_params = module_def.func_type_params.get(&func_name).cloned();
    let is_generic = type_params.as_ref().is_some_and(|tp| !tp.is_empty());

    if is_generic {
        let type_params = type_params.unwrap();
        let template = module_def
            .generic_func_templates
            .get(&func_name)
            .cloned()
            .expect("generic template must exist when type_params are present");

        // temporarily inject template into the global maps so instantiate_and_check_fn can find it
        let prev_tp = type_checker.func_type_params.remove(&func_name);
        let prev_tmpl = type_checker.generic_func_templates.remove(&func_name);
        type_checker
            .func_type_params
            .insert(func_name, type_params.clone());
        type_checker
            .generic_func_templates
            .insert(func_name, template);

        type_checker.push_scope();
        for (name, ty) in &module_def.funcs {
            if *name != func_name {
                type_checker.set_var(*name, ty.clone(), false);
            }
        }

        let node = &call.node;
        let has_explicit_type_args = !node.type_args.is_empty();

        let result = if has_explicit_type_args {
            let ret = instantiate_and_check_fn(
                func_name,
                &node.type_args,
                call.span,
                type_checker,
                errors,
            );
            type_checker
                .resolved_call_type_args
                .insert(node.func.node.id, (func_name, node.type_args.clone()));
            ret
        } else {
            let Type::Func { params, ret } = &func_ty else {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::NotAFunction { expr_type: func_ty },
                ));
                restore_generic_maps(type_checker, func_name, prev_tp, prev_tmpl);
                return Type::Infer;
            };

            let Some(inferred) = infer_type_args_from_call(
                call.span,
                &type_params,
                params,
                &node.args,
                func_ty.clone(),
                ret,
                expected,
                type_checker,
                errors,
            ) else {
                restore_generic_maps(type_checker, func_name, prev_tp, prev_tmpl);
                return Type::Infer;
            };

            let ret =
                instantiate_and_check_fn(func_name, &inferred, call.span, type_checker, errors);
            type_checker
                .resolved_call_type_args
                .insert(node.func.node.id, (func_name, inferred));
            ret
        };

        type_checker.pop_scope();
        restore_generic_maps(type_checker, func_name, prev_tp, prev_tmpl);

        if let Some(param_info) = module_def.func_param_info.get(&func_name).cloned() {
            check_var_param_args(param_info, &call.node.args, type_checker, errors);
        }

        return result;
    }

    let result = check_call_with_type(call, func_ty, type_checker, errors);

    if let Some(param_info) = module_def.func_param_info.get(&func_name).cloned() {
        check_var_param_args(param_info, &call.node.args, type_checker, errors);
    }

    result
}

fn restore_generic_maps(
    tc: &mut TypeChecker,
    name: Ident,
    prev_tp: Option<Vec<TypeParam>>,
    prev_tmpl: Option<FuncNode>,
) {
    match prev_tp {
        Some(tp) => {
            tc.func_type_params.insert(name, tp);
        }
        None => {
            tc.func_type_params.remove(&name);
        }
    }
    match prev_tmpl {
        Some(tmpl) => {
            tc.generic_func_templates.insert(name, tmpl);
        }
        None => {
            tc.generic_func_templates.remove(&name);
        }
    }
}

fn try_check_method_call(
    call: &CallNode,
    field_access: &FieldAccessNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Option<Type> {
    let method_name = field_access.node.field;
    let target = &field_access.node.target;

    if let ExprKind::Ident(type_name) = &target.node.kind
        && let Some(enum_def) = type_checker.get_enum(*type_name).cloned()
    {
        return Some(check_enum_tuple_variant(
            call,
            *type_name,
            method_name,
            &enum_def,
            type_checker,
            errors,
        ));
    }

    if let ExprKind::Ident(type_name) = &target.node.kind
        && let Some(struct_def) = type_checker.get_struct(*type_name).cloned()
    {
        return Some(check_static_method_call(
            call,
            *type_name,
            method_name,
            &struct_def,
            type_checker,
            errors,
        ));
    }

    let target_ty = check_expr(target, type_checker, errors, None);
    if let Type::Struct { name, type_args } = &target_ty
        && let Some(struct_def) = type_checker.get_struct(*name).cloned()
    {
        return Some(check_instance_method_call(
            call,
            *name,
            method_name,
            type_args,
            &struct_def,
            Some(target),
            type_checker,
            errors,
        ));
    }

    match &target_ty {
        Type::List { elem } => {
            return Some(check_list_method(
                call,
                target,
                method_name,
                elem.as_ref(),
                type_checker,
                errors,
            ));
        }
        Type::Map { key, value } => {
            return Some(check_map_method(
                call,
                target,
                method_name,
                key.as_ref(),
                value.as_ref(),
                type_checker,
                errors,
            ));
        }
        _ => {}
    }

    None
}

fn check_enum_tuple_variant(
    call: &CallNode,
    enum_name: Ident,
    variant_name: Ident,
    enum_def: &EnumDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let variant = enum_def.variants.iter().find(|v| v.name == variant_name);

    let Some(variant) = variant else {
        errors.push(TypeErr::new(
            call.span,
            TypeErrKind::UnknownEnumVariant {
                enum_name,
                variant_name,
            },
        ));
        return Type::Infer;
    };

    let VariantKind::Tuple(expected_types) = &variant.kind else {
        // not a tuple variant
        match &variant.kind {
            VariantKind::Unit => {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::EnumVariantNotTuple {
                        enum_name,
                        variant_name,
                    },
                ));
            }
            VariantKind::Struct(_) => {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::EnumVariantNotTuple {
                        enum_name,
                        variant_name,
                    },
                ));
            }
            VariantKind::Tuple(_) => unreachable!(),
        }
        return Type::Infer;
    };

    let node = &call.node;
    let is_generic = !enum_def.type_params.is_empty();

    // create inference slots for generic enums
    let slots = if is_generic {
        let call_id = type_checker.next_call_id();
        create_inference_slots(&enum_def.type_params, type_checker, call_id)
    } else {
        HashMap::new()
    };

    // check argument count
    if node.args.len() != expected_types.len() {
        errors.push(TypeErr::new(
            call.span,
            TypeErrKind::EnumVariantArityMismatch {
                enum_name,
                variant_name,
                expected: expected_types.len(),
                found: node.args.len(),
            },
        ));
        return Type::Infer;
    }

    // check each argument
    for (arg_expr, expected_ty) in node.args.iter().zip(expected_types.iter()) {
        check_expr(arg_expr, type_checker, errors, None);
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let expected_ref = if is_generic {
            if let Some((_, arg_ty)) = type_checker.get_type(arg_expr.node.id) {
                let arg_ty = arg_ty.clone();
                constrain_slots_from_type(
                    expected_ty,
                    &arg_ty,
                    &slots,
                    arg_expr.span,
                    type_checker,
                    errors,
                );
            }
            build_param_ref(expected_ty, &slots, type_checker)
        } else {
            let resolved = type_checker.resolve_type(expected_ty);
            TypeRef::concrete(&resolved)
        };
        type_checker.constrain_assignable(arg_expr.span, arg_ref, expected_ref, errors);
    }

    if is_generic {
        resolve_constraints(type_checker, errors);
    }

    // build the result type
    let type_args = if is_generic {
        enum_def
            .type_params
            .iter()
            .map(|param| {
                let slot_name = slots.get(&param.id).expect("slot exists");
                type_checker
                    .get_var(*slot_name)
                    .map(|info| info.ty.clone())
                    .unwrap_or(Type::Infer)
            })
            .collect()
    } else {
        vec![]
    };

    Type::Enum {
        name: enum_name,
        type_args,
    }
}

fn check_generic_static_method(
    call: &CallNode,
    struct_name: Ident,
    method_name: Ident,
    method: &MethodDef,
    struct_def: &StructDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &call.node;
    let is_struct_generic = !struct_def.type_params.is_empty();
    let has_explicit_type_args = !node.type_args.is_empty();

    let (struct_type_args, method_type_args) = if has_explicit_type_args {
        let same_count = node.type_args.len() == method.type_params.len();
        if !same_count {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::GenericArgNumMismatch {
                    expected: method.type_params.len(),
                    found: node.type_args.len(),
                },
            ));
            return Type::Infer;
        }

        let struct_type_args = if is_struct_generic {
            let param_templates: Vec<Type> = method.params.iter().map(|p| p.ty.clone()).collect();
            let expected_ty = Type::Func {
                params: param_templates.clone(),
                ret: Box::new(method.ret.clone()),
            };
            let Some(inferred) = infer_type_args_from_call(
                call.span,
                &struct_def.type_params,
                &param_templates,
                &node.args,
                expected_ty,
                &method.ret,
                None,
                type_checker,
                errors,
            ) else {
                return Type::Infer;
            };
            inferred
        } else {
            vec![]
        };

        (struct_type_args, node.type_args.clone())
    } else {
        let all_type_params: Vec<_> = struct_def
            .type_params
            .iter()
            .chain(method.type_params.iter())
            .cloned()
            .collect();

        let param_templates: Vec<Type> = method.params.iter().map(|p| p.ty.clone()).collect();
        let expected_ty = Type::Func {
            params: param_templates.clone(),
            ret: Box::new(method.ret.clone()),
        };

        let Some(all_inferred) = infer_type_args_from_call(
            call.span,
            &all_type_params,
            &param_templates,
            &node.args,
            expected_ty,
            &method.ret,
            None,
            type_checker,
            errors,
        ) else {
            return Type::Infer;
        };

        let n_struct = struct_def.type_params.len();
        let struct_type_args = all_inferred[..n_struct].to_vec();
        let method_type_args = all_inferred[n_struct..].to_vec();
        (struct_type_args, method_type_args)
    };

    check_var_param_args(
        method.params.iter().map(|p| (p.name, p.mutability)),
        &node.args,
        type_checker,
        errors,
    );

    instantiate_method_body(
        struct_name,
        method_name,
        &struct_type_args,
        &method_type_args,
        struct_def,
        method,
        call.span,
        type_checker,
        errors,
    )
}

pub(super) fn check_static_method_call(
    call: &CallNode,
    struct_name: Ident,
    method_name: Ident,
    struct_def: &StructDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let Some(method) = struct_def.methods.get(&method_name) else {
        errors.push(TypeErr::new(
            call.span,
            TypeErrKind::UnknownMethod {
                struct_name,
                method: method_name,
            },
        ));
        return Type::Infer;
    };

    if method.receiver.is_some() {
        errors.push(TypeErr::new(
            call.span,
            TypeErrKind::InstanceMethodOnType {
                struct_name,
                method: method_name,
            },
        ));
        return Type::Infer;
    }

    let node = &call.node;
    let has_method_type_params = !method.type_params.is_empty();
    let is_struct_generic = !struct_def.type_params.is_empty();

    if has_method_type_params {
        let method = method.clone();
        return check_generic_static_method(
            call,
            struct_name,
            method_name,
            &method,
            struct_def,
            type_checker,
            errors,
        );
    }

    if is_struct_generic {
        let param_templates: Vec<Type> = method.params.iter().map(|p| p.ty.clone()).collect();
        let expected_ty = Type::Func {
            params: param_templates.clone(),
            ret: Box::new(method.ret.clone()),
        };
        let Some(inferred_type_args) = infer_type_args_from_call(
            call.span,
            &struct_def.type_params,
            &param_templates,
            &node.args,
            expected_ty,
            &method.ret,
            None,
            type_checker,
            errors,
        ) else {
            return Type::Infer;
        };

        let subst = build_subst(&struct_def.type_params, &inferred_type_args);

        let ret_ty = subst_type(&method.ret, &subst);

        check_var_param_args(
            method.params.iter().map(|p| (p.name, p.mutability)),
            &node.args,
            type_checker,
            errors,
        );

        return ret_ty;
    }

    let param_types = method
        .params
        .iter()
        .map(|p| p.ty.clone())
        .collect::<Vec<_>>();
    let ret_type = method.ret.clone();
    let result = check_call_signature(
        call.span,
        &param_types,
        &ret_type,
        &node.args,
        type_checker,
        errors,
    );

    check_var_param_args(
        method.params.iter().map(|p| (p.name, p.mutability)),
        &node.args,
        type_checker,
        errors,
    );

    result
}

#[allow(clippy::too_many_arguments)]
fn check_generic_instance_method(
    call: &CallNode,
    struct_name: Ident,
    method_name: Ident,
    type_args: &[Type],
    struct_def: &StructDef,
    method: &MethodDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &call.node;

    // apply struct substitution first, leaving method type vars in place
    let struct_subst = build_subst(&struct_def.type_params, type_args);

    let partially_subst_params: Vec<Type> = method
        .params
        .iter()
        .map(|p| subst_type(&p.ty, &struct_subst))
        .collect();

    let has_explicit_type_args = !node.type_args.is_empty();

    let method_type_args = if has_explicit_type_args {
        let same_count = node.type_args.len() == method.type_params.len();
        if !same_count {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::GenericArgNumMismatch {
                    expected: method.type_params.len(),
                    found: node.type_args.len(),
                },
            ));
            return Type::Infer;
        }

        let mut combined_subst = build_subst(&struct_def.type_params, type_args);
        combined_subst.extend(build_subst(&method.type_params, &node.type_args));

        for (arg_expr, param_ty) in node.args.iter().zip(partially_subst_params.iter()) {
            let instantiated = subst_type(param_ty, &combined_subst);
            check_and_constrain_arg(arg_expr, &instantiated, type_checker, errors);
        }

        node.type_args.clone()
    } else {
        let partially_subst_ret = subst_type(&method.ret, &struct_subst);
        let expected_ty = Type::Func {
            params: partially_subst_params.clone(),
            ret: Box::new(partially_subst_ret.clone()),
        };

        let Some(inferred) = infer_type_args_from_call(
            call.span,
            &method.type_params,
            &partially_subst_params,
            &node.args,
            expected_ty,
            &partially_subst_ret,
            None,
            type_checker,
            errors,
        ) else {
            return Type::Infer;
        };

        inferred
    };

    check_var_param_args(
        method.params.iter().map(|p| (p.name, p.mutability)),
        &node.args,
        type_checker,
        errors,
    );

    instantiate_method_body(
        struct_name,
        method_name,
        type_args,
        &method_type_args,
        struct_def,
        method,
        call.span,
        type_checker,
        errors,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check_instance_method_call(
    call: &CallNode,
    struct_name: Ident,
    method_name: Ident,
    type_args: &[Type],
    struct_def: &StructDef,
    target: Option<&ExprNode>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let Some(method) = struct_def.methods.get(&method_name) else {
        errors.push(TypeErr::new(
            call.span,
            TypeErrKind::UnknownMethod {
                struct_name,
                method: method_name,
            },
        ));
        return Type::Infer;
    };

    if method.receiver.is_none() {
        errors.push(TypeErr::new(
            call.span,
            TypeErrKind::StaticMethodOnValue {
                struct_name,
                method: method_name,
            },
        ));
        return Type::Infer;
    }

    if let Some(target) = target
        && matches!(method.receiver, Some(MethodReceiver::Var))
    {
        check_receiver_mutability(target, struct_name, method_name, type_checker, errors);
    }

    let node = &call.node;
    let has_method_type_params = !method.type_params.is_empty();

    if has_method_type_params {
        let method = method.clone();
        return check_generic_instance_method(
            call,
            struct_name,
            method_name,
            type_args,
            struct_def,
            &method,
            type_checker,
            errors,
        );
    }

    let subst = build_subst(&struct_def.type_params, type_args);

    let param_types: Vec<Type> = method
        .params
        .iter()
        .map(|p| subst_type(&p.ty, &subst))
        .collect();
    let ret_type = subst_type(&method.ret, &subst);

    let result = check_call_signature(
        call.span,
        &param_types,
        &ret_type,
        &node.args,
        type_checker,
        errors,
    );

    check_var_param_args(
        method.params.iter().map(|p| (p.name, p.mutability)),
        &node.args,
        type_checker,
        errors,
    );

    result
}

pub(super) fn check_extern_instance_method_call(
    call: &CallNode,
    type_name: Ident,
    method_name: Ident,
    method: &ExternMethodDef,
    target: Option<&ExprNode>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    if let Some(target) = target
        && matches!(method.receiver, Some(MethodReceiver::Var))
    {
        check_receiver_mutability(target, type_name, method_name, type_checker, errors);
    }

    let param_types: Vec<Type> = method.params.iter().map(|p| p.ty.clone()).collect();

    let result = check_call_signature(
        call.span,
        &param_types,
        &method.ret,
        &call.node.args,
        type_checker,
        errors,
    );

    check_var_param_args(
        method.params.iter().map(|p| (p.name, p.mutability)),
        &call.node.args,
        type_checker,
        errors,
    );

    result
}

pub(super) fn check_extern_static_method_call(
    call: &CallNode,
    type_name: Ident,
    method_name: Ident,
    extern_def: &ExternTypeDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let Some(method) = extern_def.statics.get(&method_name) else {
        if extern_def.methods.contains_key(&method_name) {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::InstanceMethodOnType {
                    struct_name: type_name,
                    method: method_name,
                },
            ));
        } else {
            errors.push(TypeErr::new(
                call.span,
                TypeErrKind::ExternUnknownMethod {
                    type_name,
                    method: method_name,
                },
            ));
        }
        return Type::Infer;
    };

    let param_types: Vec<Type> = method.params.iter().map(|p| p.ty.clone()).collect();

    let result = check_call_signature(
        call.span,
        &param_types,
        &method.ret,
        &call.node.args,
        type_checker,
        errors,
    );

    check_var_param_args(
        method.params.iter().map(|p| (p.name, p.mutability)),
        &call.node.args,
        type_checker,
        errors,
    );

    result
}

fn generic_context_strings(
    name: &str,
    type_params: &[TypeParam],
    type_args: &[Type],
) -> (String, String) {
    let args_str = type_args
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let label = format!("in this instantiation of '{name}<{args_str}>'");
    let mapping = type_params
        .iter()
        .zip(type_args.iter())
        .map(|(p, a)| format!("{} = {a}", p.name))
        .collect::<Vec<_>>()
        .join(", ");
    let note = format!("where {mapping}");
    (label, note)
}

fn report_cached_spec_error(
    cached: &SpecializationResult,
    context_name: &str,
    type_params: &[TypeParam],
    type_args: &[Type],
    call_span: Span,
    errors: &mut Vec<TypeErr>,
) {
    if let Some((body_span, err_kind)) = &cached.err {
        let (label, note) = generic_context_strings(context_name, type_params, type_args);
        errors.push(
            TypeErr::new(*body_span, err_kind.clone())
                .with_secondary(call_span, label)
                .with_note(note),
        );
    }
}

fn report_instantiation_errors(
    body_errors: Vec<TypeErr>,
    context_name: &str,
    type_params: &[TypeParam],
    type_args: &[Type],
    call_span: Span,
    errors: &mut Vec<TypeErr>,
) {
    let (label, note) = generic_context_strings(context_name, type_params, type_args);
    for mut err in body_errors {
        err.secondary.push((call_span, label.clone()));
        err.notes.push(note.clone());
        errors.push(err);
    }
}

fn instantiate_and_check_fn(
    func_name: Ident,
    type_args: &[Type],
    call_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    // look up type parameters early so they are available for cache-hit context
    let Some(type_params) = type_checker.func_type_params.get(&func_name).cloned() else {
        errors.push(TypeErr::new(
            call_span,
            TypeErrKind::UnknownFunction { name: func_name },
        ));
        return Type::Infer;
    };

    let cache_key = SpecializationKey {
        func_name,
        type_args: type_args.to_vec(),
    };

    // check cache first
    if let Some(cached) = type_checker.specialization_cache.get(&cache_key) {
        report_cached_spec_error(
            cached,
            &func_name.to_string(),
            &type_params,
            type_args,
            call_span,
            errors,
        );
        return cached.ret_ty.clone();
    }

    // look up the generic function template
    let Some(fn_template) = type_checker.generic_func_templates.get(&func_name).cloned() else {
        errors.push(TypeErr::new(
            call_span,
            TypeErrKind::UnknownFunction { name: func_name },
        ));
        return Type::Infer;
    };

    // verify arity
    let same_param_count = type_params.len() == type_args.len();
    if !same_param_count {
        errors.push(TypeErr::new(
            call_span,
            TypeErrKind::GenericArgNumMismatch {
                expected: type_params.len(),
                found: type_args.len(),
            },
        ));
        return Type::Infer;
    }

    // build substitution map to convert TypeVarId -> concrete Type
    let subst = build_subst(&type_params, type_args);

    let func = &fn_template.node;

    // compute the specialized parameter and return types
    let specialized_param_types: Vec<Type> = func
        .params
        .iter()
        .map(|p| subst_type(&type_checker.resolve_type(&p.ty), &subst))
        .collect();
    let specialized_ret = subst_type(&type_checker.resolve_type(&func.ret), &subst);

    // typecheck the body with specialized types, capturing expression types per specialization
    let prev_snapshot = type_checker.spec_type_snapshot.take();
    type_checker.spec_type_snapshot = Some(HashMap::new());

    let module_scope = type_checker
        .generic_func_source_module
        .get(&func_name)
        .and_then(|path| type_checker.resolved_module_defs.get(path).cloned());
    if let Some(ref module_def) = module_scope {
        type_checker.push_scope();
        for (name, ty) in &module_def.funcs {
            if *name != func_name {
                type_checker.set_var(*name, ty.clone(), false);
            }
        }
    }

    let mut body_errors = vec![];
    check_fn_body(
        func,
        &specialized_param_types,
        specialized_ret.clone(),
        call_span,
        type_checker,
        &mut body_errors,
    );

    if module_scope.is_some() {
        type_checker.pop_scope();
    }

    let body_types = type_checker.spec_type_snapshot.take().unwrap_or_default();
    type_checker.spec_type_snapshot = prev_snapshot;

    // cache the result preserving the body span of the first error
    let cached_err = body_errors.first().map(|err| (err.span, err.kind.clone()));
    type_checker.specialization_cache.insert(
        cache_key,
        SpecializationResult {
            ret_ty: specialized_ret.clone(),
            err: cached_err,
            body_types,
        },
    );

    // report body errors with original spans, attaching call-site context
    report_instantiation_errors(
        body_errors,
        &func_name.to_string(),
        &type_params,
        type_args,
        call_span,
        errors,
    );

    specialized_ret
}

#[allow(clippy::too_many_arguments)]
fn instantiate_method_body(
    struct_name: Ident,
    method_name: Ident,
    struct_type_args: &[Type],
    method_type_args: &[Type],
    struct_def: &StructDef,
    method: &MethodDef,
    call_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let all_type_args: Vec<Type> = struct_type_args
        .iter()
        .chain(method_type_args.iter())
        .cloned()
        .collect();

    // collect all type params (struct params first, then method params) mirroring all_type_args
    let all_type_params: Vec<TypeParam> = struct_def
        .type_params
        .iter()
        .chain(method.type_params.iter())
        .cloned()
        .collect();

    let cache_key = MethodSpecKey {
        struct_name,
        method_name,
        type_args: all_type_args.clone(),
    };

    if let Some(cached) = type_checker.method_spec_cache.get(&cache_key) {
        let context_name = format!("{struct_name}.{method_name}");
        report_cached_spec_error(
            cached,
            &context_name,
            &all_type_params,
            &all_type_args,
            call_span,
            errors,
        );
        return cached.ret_ty.clone();
    }

    let mut subst = build_subst(&struct_def.type_params, struct_type_args);
    subst.extend(build_subst(&method.type_params, method_type_args));

    let specialized_param_types: Vec<Type> = method
        .params
        .iter()
        .map(|p| subst_type(&p.ty, &subst))
        .collect();
    let specialized_ret = subst_type(&method.ret, &subst);

    let self_type = Type::Struct {
        name: struct_name,
        type_args: struct_type_args.to_vec(),
    };
    let self_ident = Ident(Intern::new("self".to_string()));

    let mut params: Vec<(Ident, Type, bool)> = vec![];
    if let Some(receiver) = method.receiver {
        let self_mutable = matches!(receiver, MethodReceiver::Var);
        params.push((self_ident, self_type, self_mutable));
    }
    for (param, specialized_ty) in method.params.iter().zip(specialized_param_types.iter()) {
        params.push((
            param.name,
            specialized_ty.clone(),
            matches!(param.mutability, Mutability::Mutable),
        ));
    }

    type_checker.push_method_context(MethodContext {
        struct_name,
        receiver: method.receiver,
    });

    let mut body_errors = vec![];
    check_body_common(
        &params,
        &method.body,
        &specialized_ret,
        call_span,
        type_checker,
        &mut body_errors,
    );

    type_checker.pop_method_context();

    // cache the result preserving the body span of the first error
    // method specializations don't need body_types (methods are not lowered via specialization cache)
    let cached_err = body_errors.first().map(|err| (err.span, err.kind.clone()));
    type_checker.method_spec_cache.insert(
        cache_key,
        SpecializationResult {
            ret_ty: specialized_ret.clone(),
            err: cached_err,
            body_types: HashMap::new(),
        },
    );

    // report body errors with original spans, attaching call-site context
    let context_name = format!("{struct_name}.{method_name}");
    report_instantiation_errors(
        body_errors,
        &context_name,
        &all_type_params,
        &all_type_args,
        call_span,
        errors,
    );

    specialized_ret
}

pub(super) fn type_call_on_base(
    base_ty: &Type,
    call_node: &CallNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    // check each argument
    for arg in &call_node.node.args {
        check_expr(arg, type_checker, errors, None);
    }

    let Type::Func { params, ret } = base_ty else {
        if matches!(base_ty, Type::Infer) {
            return Type::Infer;
        }
        errors.push(TypeErr::new(
            call_node.span,
            TypeErrKind::NotAFunction {
                expr_type: base_ty.clone(),
            },
        ));
        return Type::Infer;
    };

    // check argument count
    if call_node.node.args.len() != params.len() {
        errors.push(TypeErr::new(
            call_node.span,
            TypeErrKind::MismatchedTypes {
                expected: base_ty.clone(),
                found: Type::Func {
                    params: vec![Type::Infer; call_node.node.args.len()],
                    ret: Box::new(Type::Infer),
                },
            },
        ));
        return Type::Infer;
    }

    for (arg_expr, param_ty) in call_node.node.args.iter().zip(params.iter()) {
        if matches!(arg_expr.node.kind, ExprKind::Lit(Lit::Float { .. })) {
            check_expr(arg_expr, type_checker, errors, Some(param_ty));
        }
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let param_ref = TypeRef::concrete(param_ty);
        type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
    }

    (**ret).clone()
}
