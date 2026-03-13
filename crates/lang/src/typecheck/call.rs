use crate::{
    ast::{
        CallNode, ExprKind, ExprNode, FieldAccessNode, Ident, MethodReceiver, Mutability, Type,
        TypeVarId, VariantKind,
    },
    span::Span,
};
use std::collections::HashMap;

use super::{
    constraint::TypeRef,
    decl::check_fn_body,
    error::{TypeErr, TypeErrKind},
    expr::{check_expr, root_ident},
    infer::{
        create_inference_slots, infer_type_args_from_call, instantiate_func_type, subst_type,
        type_to_ref_with_inference,
    },
    types::{EnumDef, SpecializationKey, SpecializationResult, StructDef, TypeChecker},
};

pub(super) fn check_var_param_args(
    params: &[(Ident, Mutability)],
    args: &[ExprNode],
    type_checker: &TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    for ((param_name, mutability), arg) in params.iter().zip(args.iter()) {
        if *mutability != Mutability::Mutable {
            continue;
        }

        let Some(root) = root_ident(arg) else {
            errors.push(
                TypeErr::new(arg.span, TypeErrKind::VarParamNotLvalue { param: *param_name })
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
                        param: *param_name,
                        binding: root,
                    },
                )
                .with_help("declare with 'var' to allow mutation"),
            );
        }
    }
}

fn check_mutating_receiver(
    target: &ExprNode,
    struct_name: Ident,
    method_name: Ident,
    receiver: Option<MethodReceiver>,
    type_checker: &TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let is_mutating = matches!(receiver, Some(MethodReceiver::Var));
    if !is_mutating {
        return;
    }

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
                    struct_name,
                    method: method_name,
                },
            )
            .with_help("declare with 'var' to allow calling mutating methods"),
        );
    }
}

pub(super) fn check_call(
    call: &CallNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &call.node;

    if let ExprKind::Field(field_access) = &node.func.node.kind
        && let Some(result) = try_check_method_call(call, field_access, type_checker, errors)
    {
        return result;
    }

    let func_ty = check_expr(&node.func, type_checker, errors);

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
        // non generic functions with type params are invalid
        (false, true) => {
            errors.push(TypeErr::new(call.span, TypeErrKind::NotGenericFunction));
            return Type::Infer;
        }

        // generic function without type args -> infer type args, then instantiate
        (true, false) => {
            let Some(name) = func_name else {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::NotAFunction {
                        expr_type: func_ty.clone(),
                    },
                ));
                return Type::Infer;
            };

            let Type::Func { params, ret: _ } = &func_ty else {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::NotAFunction {
                        expr_type: func_ty.clone(),
                    },
                ));
                return Type::Infer;
            };

            let Some(inferred_type_args) = infer_type_args_from_call(
                call.span,
                &type_params,
                params,
                &node.args,
                func_ty.clone(),
                type_checker,
                errors,
            ) else {
                return Type::Infer;
            };

            if let Some(param_info) = type_checker.func_param_info.get(&name).cloned() {
                check_var_param_args(&param_info, &node.args, type_checker, errors);
            }

            return instantiate_and_check_fn(
                name,
                &inferred_type_args,
                call.span,
                type_checker,
                errors,
            );
        }

        // generic function with type args -> explicit instantiation
        (true, true) => {
            // error if not a function
            let Some(name) = func_name else {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::NotAFunction {
                        expr_type: func_ty.clone(),
                    },
                ));
                return Type::Infer;
            };

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

            // check arguments against the instantiated parameter types
            let Type::Func { params, ret: _ } = &func_ty else {
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::NotAFunction {
                        expr_type: func_ty.clone(),
                    },
                ));
                return Type::Infer;
            };

            // build map substitution for parameter type checking
            let subst = type_params
                .iter()
                .zip(node.type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect::<HashMap<TypeVarId, _>>();

            let params_count = params.len() == node.args.len();
            if !params_count {
                let instantiated_ty = instantiate_func_type(
                    &type_params,
                    &func_ty,
                    &node.type_args,
                    call.span,
                    errors,
                );
                errors.push(TypeErr::new(
                    call.span,
                    TypeErrKind::MismatchedTypes {
                        expected: instantiated_ty.unwrap_or(func_ty.clone()),
                        found: Type::Func {
                            params: vec![Type::Infer; node.args.len()],
                            ret: Box::new(Type::Infer),
                        },
                    },
                ));
                return Type::Infer;
            }

            // check each argument against the substituted parameter type
            for (arg_expr, param_ty) in node.args.iter().zip(params.iter()) {
                check_expr(arg_expr, type_checker, errors);
                let instantiated_param_ty = subst_type(param_ty, &subst);
                let arg_ref = TypeRef::Expr(arg_expr.node.id);
                let param_ref = TypeRef::Concrete(instantiated_param_ty);
                type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
            }

            if let Some(param_info) = type_checker.func_param_info.get(&name).cloned() {
                check_var_param_args(&param_info, &node.args, type_checker, errors);
            }

            return instantiate_and_check_fn(
                name,
                &node.type_args,
                call.span,
                type_checker,
                errors,
            );
        }

        // non generic function without type args then must be a normal call
        (false, false) => {}
    }

    // fallback to normal call
    let result = check_call_with_type(call, func_ty, type_checker, errors);

    if let Some(name) = func_name {
        if let Some(param_info) = type_checker.func_param_info.get(&name).cloned() {
            check_var_param_args(&param_info, &node.args, type_checker, errors);
        }
    }

    result
}

pub(super) fn check_call_signature(
    call_span: Span,
    param_types: &[Type],
    ret_type: &Type,
    args: &[ExprNode],
    mismatch_found_type: Option<Type>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    if args.len() != param_types.len() {
        let expected = Type::Func {
            params: param_types.to_vec(),
            ret: Box::new(ret_type.clone()),
        };
        let found = mismatch_found_type.unwrap_or_else(|| Type::Func {
            params: vec![Type::Infer; args.len()],
            ret: Box::new(Type::Infer),
        });
        errors.push(TypeErr::new(
            call_span,
            TypeErrKind::MismatchedTypes { expected, found },
        ));
        return Type::Infer;
    }

    for (arg_expr, param_ty) in args.iter().zip(param_types.iter()) {
        check_expr(arg_expr, type_checker, errors);
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let param_ref = TypeRef::Concrete(param_ty.clone());
        type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
    }

    ret_type.clone()
}

/// Check a function call given the function type
pub(super) fn check_call_with_type(
    call: &CallNode,
    func_ty: Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &call.node;

    match func_ty.clone() {
        Type::Func { params, ret } => {
            let ret = *ret;
            check_call_signature(
                call.span,
                &params,
                &ret,
                &node.args,
                Some(Type::Func {
                    params: params.clone(),
                    ret: Box::new(ret.clone()),
                }),
                type_checker,
                errors,
            )
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

pub(super) fn try_check_method_call(
    call: &CallNode,
    field_access: &FieldAccessNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Option<Type> {
    let method_name = field_access.node.field;
    let target = &field_access.node.target;

    // check for enum tuple variant construction first
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

    let target_ty = check_expr(target, type_checker, errors);
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
    let slots = is_generic
        .then(|| {
            let call_id = type_checker.next_call_id();
            create_inference_slots(&enum_def.type_params, type_checker, call_id)
        })
        .unwrap_or_default();

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
        check_expr(arg_expr, type_checker, errors);
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let expected_ref = if is_generic {
            type_to_ref_with_inference(expected_ty, &slots)
        } else {
            let resolved = type_checker.resolve_type(expected_ty);
            TypeRef::Concrete(resolved)
        };
        type_checker.constrain_assignable(arg_expr.span, arg_ref, expected_ref, errors);
    }

    // build the result type
    let type_args = is_generic
        .then(|| {
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
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Type::Enum {
        name: enum_name,
        type_args,
    }
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
    let is_generic = !struct_def.type_params.is_empty();
    if is_generic {
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
            type_checker,
            errors,
        ) else {
            return Type::Infer;
        };

        let subst: HashMap<TypeVarId, Type> = struct_def
            .type_params
            .iter()
            .zip(inferred_type_args.iter())
            .map(|(param, arg)| (param.id, arg.clone()))
            .collect();

        let ret_ty = subst_type(&method.ret, &subst);
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
        None,
        type_checker,
        errors,
    );

    let param_info: Vec<_> = method
        .params
        .iter()
        .map(|p| (p.name, p.mutability))
        .collect();
    check_var_param_args(&param_info, &node.args, type_checker, errors);

    result
}

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

    if let Some(target) = target {
        check_mutating_receiver(target, struct_name, method_name, method.receiver, type_checker, errors);
    }

    let node = &call.node;

    let subst: HashMap<TypeVarId, Type> = struct_def
        .type_params
        .iter()
        .zip(type_args.iter())
        .map(|(param, arg)| (param.id, arg.clone()))
        .collect();

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
        None,
        type_checker,
        errors,
    );

    let param_info: Vec<_> = method
        .params
        .iter()
        .map(|p| (p.name, p.mutability))
        .collect();
    check_var_param_args(&param_info, &node.args, type_checker, errors);

    result
}

/// Instantiates a generic function with concrete type arguments and typechecks the specialized body
pub(super) fn instantiate_and_check_fn(
    func_name: Ident,
    type_args: &[Type],
    call_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let cache_key = SpecializationKey {
        func_name,
        type_args: type_args.to_vec(),
    };

    // check cache first
    if let Some(cached) = type_checker.specialization_cache.get(&cache_key) {
        // if there was an error we report it with the current call site span
        if let Some(err_kind) = &cached.err_kind {
            errors.push(TypeErr::new(call_span, err_kind.clone()));
        }
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

    // look up type parameters
    let Some(type_params) = type_checker.func_type_params.get(&func_name).cloned() else {
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
    let subst: HashMap<TypeVarId, Type> = type_params
        .iter()
        .zip(type_args.iter())
        .map(|(param, arg)| (param.id, arg.clone()))
        .collect();

    let func = &fn_template.node;

    // compute the specialized parameter and return types
    let specialized_param_types: Vec<Type> = func
        .params
        .iter()
        .map(|p| subst_type(&p.ty, &subst))
        .collect();
    let specialized_ret = subst_type(&func.ret, &subst);

    // typecheck the body with specialized types
    let mut body_errors = vec![];
    check_fn_body(
        func,
        &specialized_param_types,
        specialized_ret.clone(),
        call_span,
        type_checker,
        &mut body_errors,
    );

    // cache the result
    let err_kind = body_errors.first().map(|err| err.kind.clone());
    type_checker.specialization_cache.insert(
        cache_key,
        SpecializationResult {
            ret_ty: specialized_ret.clone(),
            err_kind,
        },
    );

    // report any errors from the body with the call site span
    for err in body_errors {
        errors.push(TypeErr::new(call_span, err.kind));
    }

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
        check_expr(arg, type_checker, errors);
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

    // constrain argument types
    for (arg_expr, param_ty) in call_node.node.args.iter().zip(params.iter()) {
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let param_ref = TypeRef::Concrete(param_ty.clone());
        type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
    }

    (**ret).clone()
}
