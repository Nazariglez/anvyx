use crate::{
    ast::{Func, FuncNode, Ident, MethodReceiver, Mutability, StructDeclNode, Type},
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

pub(super) fn check_fn_body(
    func: &Func,
    param_types: &[Type],
    ret_ty: Type,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    type_checker.push_scope();
    type_checker.push_return_type(ret_ty.clone(), None);

    // bind parameters into scope with the provided types
    for (param, ty) in func.params.iter().zip(param_types.iter()) {
        let mutable = matches!(param.mutability, Mutability::Mutable);
        type_checker.set_var(param.name, ty.clone(), mutable);
    }

    // treat the block as an expression for implicit returns
    let (body_ty, last_expr_id) = check_block_expr(&func.body, type_checker, errors);

    // void fn cannot have trailing expressions (at least they are void too)
    let is_void_fn = ret_ty.is_void();
    if is_void_fn {
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
        // if there is a last expression, it must be assignable to return type
        let expr_ref = TypeRef::Expr(last_id);
        let ret_ref = TypeRef::Concrete(ret_ty.clone());
        type_checker.constrain_assignable(error_span, expr_ref, ret_ref, errors);
    } else if !type_checker.has_explicit_return() {
        // no implicit or explicit return, fn with non-void return is invalid
        errors.push(TypeErr::new(
            error_span,
            TypeErrKind::MismatchedTypes {
                expected: ret_ty,
                found: Type::Void,
            },
        ));
    }

    type_checker.pop_return_type();
    type_checker.pop_scope();
}

pub(super) fn check_method_body(
    struct_name: Ident,
    struct_def: &StructDef,
    method: &MethodDef,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    if !method.type_params.is_empty() {
        // FIXME: method generics are not supported yet
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

    type_checker.push_scope();
    type_checker.push_return_type(method.ret.clone(), None);
    type_checker.push_method_context(MethodContext {
        struct_name,
        receiver: method.receiver,
    });

    if let Some(receiver) = method.receiver {
        let self_ident = Ident(Intern::new("self".to_string()));
        let self_mutable = matches!(receiver, MethodReceiver::Var);
        type_checker.set_var(self_ident, self_type, self_mutable);
    }

    for param in &method.params {
        let mutable = matches!(param.mutability, Mutability::Mutable);
        type_checker.set_var(param.name, param.ty.clone(), mutable);
    }

    let (body_ty, last_expr_id) = check_block_expr(&method.body, type_checker, errors);
    let had_explicit_return = type_checker.has_explicit_return();

    if method.ret.is_void() {
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
        let ret_ref = TypeRef::Concrete(method.ret.clone());
        type_checker.constrain_assignable(error_span, expr_ref, ret_ref, errors);
    } else if !had_explicit_return {
        errors.push(TypeErr::new(
            error_span,
            TypeErrKind::MismatchedTypes {
                expected: method.ret.clone(),
                found: Type::Void,
            },
        ));
    }

    type_checker.pop_method_context();
    type_checker.pop_return_type();
    type_checker.pop_scope();
}

pub(super) fn check_func(
    fn_node: &FuncNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let func = &fn_node.node;

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
    let param_types: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();

    check_fn_body(
        func,
        &param_types,
        func.ret.clone(),
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

    let Some(struct_def) = type_checker.get_struct(struct_name).cloned() else {
        return;
    };

    for method in &decl.methods {
        if let Some(method_def) = struct_def.methods.get(&method.name) {
            check_method_body(
                struct_name,
                &struct_def,
                method_def,
                method.body.span,
                type_checker,
                errors,
            );
        }
    }
}
