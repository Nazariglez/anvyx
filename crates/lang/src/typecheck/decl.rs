use crate::{
    ast::{BlockNode, Func, FuncNode, Ident, MethodReceiver, Mutability, StructDeclNode, Type},
    span::Span,
};
use internment::Intern;

use super::{
    constraint::TypeRef,
    error::{TypeErr, TypeErrKind},
    infer::type_from_fn,
    stmt::check_block_expr,
    types::{MethodContext, MethodDef, StructDef, TypeChecker},
};

pub(super) fn check_body_common(
    params: &[(Ident, Type, bool)],
    body: &BlockNode,
    ret_ty: &Type,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    type_checker.push_scope();
    type_checker.push_return_type(ret_ty.clone(), None);

    for (name, ty, mutable) in params {
        type_checker.set_var(*name, ty.clone(), *mutable);
    }

    let expected_tail = if ret_ty.is_void() { None } else { Some(ret_ty) };
    let (body_ty, last_expr_id) = check_block_expr(body, type_checker, errors, expected_tail);
    let had_explicit_return = type_checker.has_explicit_return();

    if ret_ty.is_void() {
        if !body_ty.is_void() {
            errors.push(TypeErr::new(
                error_span,
                TypeErrKind::MismatchedTypes {
                    expected: Type::Void,
                    found: body_ty,
                },
            ));
        }
    } else if let Some(last_id) = last_expr_id {
        let expr_ref = TypeRef::Expr(last_id);
        let ret_ref = TypeRef::concrete(ret_ty);
        type_checker.constrain_assignable(error_span, expr_ref, ret_ref, errors);
    } else if !had_explicit_return {
        errors.push(TypeErr::new(
            error_span,
            TypeErrKind::MismatchedTypes {
                expected: ret_ty.clone(),
                found: Type::Void,
            },
        ));
    }

    type_checker.pop_return_type();
    type_checker.pop_scope();
}

pub(super) fn check_fn_body(
    func: &Func,
    param_types: &[Type],
    ret_ty: Type,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let params: Vec<(Ident, Type, bool)> = func
        .params
        .iter()
        .zip(param_types.iter())
        .map(|(param, ty)| {
            (
                param.name,
                ty.clone(),
                matches!(param.mutability, Mutability::Mutable),
            )
        })
        .collect();

    check_body_common(
        &params,
        &func.body,
        &ret_ty,
        error_span,
        type_checker,
        errors,
    );
}

pub(super) fn check_method_body(
    struct_name: Ident,
    method_name: Ident,
    struct_def: &StructDef,
    method: &MethodDef,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    if !method.type_params.is_empty() {
        for method_param in &method.type_params {
            let shadows_struct = struct_def
                .type_params
                .iter()
                .any(|sp| sp.name == method_param.name);
            if shadows_struct {
                errors.push(TypeErr::new(
                    error_span,
                    TypeErrKind::MethodTypeParamShadowsStruct {
                        struct_name,
                        method: method_name,
                        param: method_param.name,
                    },
                ));
            }
        }
        return;
    }

    let self_type = Type::Struct {
        name: struct_name,
        type_args: struct_def
            .type_params
            .iter()
            .map(|tp| Type::Var(tp.id))
            .collect(),
    };

    // build the param list, prepending self when there is a receiver
    let mut params: Vec<(Ident, Type, bool)> = vec![];
    if let Some(receiver) = method.receiver {
        let self_ident = Ident(Intern::new("self".to_string()));
        let self_mutable = matches!(receiver, MethodReceiver::Var);
        params.push((self_ident, self_type, self_mutable));
    }
    for param in &method.params {
        params.push((
            param.name,
            param.ty.clone(),
            matches!(param.mutability, Mutability::Mutable),
        ));
    }

    type_checker.push_method_context(MethodContext {
        struct_name,
        receiver: method.receiver,
    });

    check_body_common(
        &params,
        &method.body,
        &method.ret,
        error_span,
        type_checker,
        errors,
    );

    type_checker.pop_method_context();
}

pub(super) fn check_func(
    fn_node: &FuncNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let func = &fn_node.node;

    for param in &func.params {
        if param.ty.contains_any() {
            errors.push(TypeErr::new(fn_node.span, TypeErrKind::AnyTypeNotAllowed));
        }
    }
    if func.ret.contains_any() {
        errors.push(TypeErr::new(fn_node.span, TypeErrKind::AnyTypeNotAllowed));
    }

    // if the function is generic we skip checking here
    // it will be done at instantiation time with concrete types
    let is_generic = !func.type_params.is_empty();
    if is_generic {
        return;
    }

    let Some(info) = type_checker.get_var(func.name) else {
        errors.push(TypeErr::new(
            fn_node.span,
            TypeErrKind::UnknownFunction { name: func.name },
        ));

        return;
    };
    let ty = &info.ty;

    if !matches!(ty, Type::Func { .. }) {
        errors.push(TypeErr::new(
            fn_node.span,
            TypeErrKind::MismatchedTypes {
                expected: type_from_fn(func),
                found: ty.clone(),
            },
        ));

        return;
    }

    // build param types from the functions declared parameters
    let param_types: Vec<Type> = func
        .params
        .iter()
        .map(|p| type_checker.resolve_type(&p.ty))
        .collect();

    let ret_ty = type_checker.resolve_type(&func.ret);

    check_fn_body(
        func,
        &param_types,
        ret_ty,
        fn_node.span,
        type_checker,
        errors,
    );
}

pub(super) fn check_struct(
    struct_node: &StructDeclNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let decl = &struct_node.node;
    let struct_name = decl.name;

    for field in &decl.fields {
        if field.ty.contains_any() {
            errors.push(TypeErr::new(struct_node.span, TypeErrKind::AnyTypeNotAllowed));
        }
    }

    let Some(struct_def) = type_checker.get_struct(struct_name).cloned() else {
        return;
    };

    let to_string_ident = Ident(Intern::new("to_string".to_string()));
    if let Some(method_def) = struct_def.methods.get(&to_string_ident) {
        validate_to_string_signature(struct_name, method_def, struct_node.span, errors);
    }

    for method in &decl.methods {
        if let Some(method_def) = struct_def.methods.get(&method.name) {
            check_method_body(
                struct_name,
                method.name,
                &struct_def,
                method_def,
                method.body.span,
                type_checker,
                errors,
            );
        }
    }
}

fn validate_to_string_signature(
    struct_name: Ident,
    method: &MethodDef,
    span: Span,
    errors: &mut Vec<TypeErr>,
) {
    match method.receiver {
        None => {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::InvalidToStringSignature {
                    struct_name,
                    reason: "must have a 'self' receiver".to_string(),
                },
            ));
            return;
        }
        Some(MethodReceiver::Var) => {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::InvalidToStringSignature {
                    struct_name,
                    reason: "receiver must be 'self', not 'var self'".to_string(),
                },
            ));
        }
        Some(MethodReceiver::Value) => {}
    }

    if method.ret != Type::String {
        let found = method.ret.clone();
        errors.push(TypeErr::new(
            span,
            TypeErrKind::InvalidToStringSignature {
                struct_name,
                reason: format!("must return 'string', found '{found}'"),
            },
        ));
    }

    if !method.params.is_empty() {
        errors.push(TypeErr::new(
            span,
            TypeErrKind::InvalidToStringSignature {
                struct_name,
                reason: "must take no parameters besides 'self'".to_string(),
            },
        ));
    }

    if !method.type_params.is_empty() {
        errors.push(TypeErr::new(
            span,
            TypeErrKind::InvalidToStringSignature {
                struct_name,
                reason: "must not be generic".to_string(),
            },
        ));
    }
}
