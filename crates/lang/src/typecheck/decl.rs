use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use internment::Intern;

use super::{
    annotations::{AnnotationTarget, normalize_annotations},
    const_eval::{ConstValue, eval_const_expr, validate_const_expr},
    constraint::TypeRef,
    error::{Diagnostic, DiagnosticKind},
    infer::type_from_fn,
    stmt::check_block_expr,
    types::{
        FieldDefault, MethodContext, MethodDef, StructDef, TypeChecker, type_references_generic,
    },
};
use crate::{
    ast::{
        AggregateDeclNode, ArrayLen, BlockNode, ExprId, ExprKind, Func, FuncNode, Ident,
        MethodReceiver, Mutability, Param, StructField, Type, TypeParam,
    },
    span::Span,
};

static TO_STRING_IDENT: LazyLock<Ident> =
    LazyLock::new(|| Ident(Intern::new("to_string".to_string())));

pub(super) fn validate_block_return(
    ret_ty: &Type,
    body_ty: Type,
    last_expr_id: Option<ExprId>,
    had_explicit_return: bool,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    if ret_ty.is_void() {
        if !body_ty.is_void() {
            errors.push(Diagnostic::new(
                error_span,
                DiagnosticKind::MismatchedTypes {
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
        errors.push(Diagnostic::new(
            error_span,
            DiagnosticKind::MismatchedTypes {
                expected: ret_ty.clone(),
                found: Type::Void,
            },
        ));
    }
}

pub(super) fn check_body_common(
    params: &[(Ident, Type, bool)],
    body: &BlockNode,
    ret_ty: &Type,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    type_checker.push_scope();
    type_checker.push_return_type(ret_ty.clone(), None);

    for (name, ty, mutable) in params {
        type_checker.set_var(*name, ty.clone(), *mutable);
    }

    let expected_tail = if ret_ty.is_void() { None } else { Some(ret_ty) };
    let (body_ty, last_expr_id) = check_block_expr(body, type_checker, errors, expected_tail);
    let had_explicit_return = type_checker.has_explicit_return();

    validate_block_return(
        ret_ty,
        body_ty,
        last_expr_id,
        had_explicit_return,
        error_span,
        type_checker,
        errors,
    );

    type_checker.pop_return_type();
    type_checker.pop_scope();
}

pub(super) fn check_fn_body(
    func: &Func,
    param_types: &[Type],
    ret_ty: &Type,
    error_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
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
        ret_ty,
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
    errors: &mut Vec<Diagnostic>,
) {
    if !method.type_params.is_empty() {
        for method_param in &method.type_params {
            let shadows_struct = struct_def
                .type_params
                .iter()
                .any(|sp| sp.name == method_param.name);
            if shadows_struct {
                errors.push(Diagnostic::new(
                    error_span,
                    DiagnosticKind::MethodTypeParamShadowsStruct {
                        kind: struct_def.kind.keyword(),
                        struct_name,
                        method: method_name,
                        param: method_param.name,
                    },
                ));
            }
        }
        return;
    }

    if !struct_def.type_params.is_empty() {
        return;
    }

    let self_type = struct_def.make_type(
        struct_name,
        struct_def
            .type_params
            .iter()
            .map(|tp| Type::Var(tp.id))
            .collect(),
    );

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
    errors: &mut Vec<Diagnostic>,
) {
    let func = &fn_node.node;

    for param in &func.params {
        if param.ty.contains_any() {
            errors.push(Diagnostic::new(
                fn_node.span,
                DiagnosticKind::AnyTypeNotAllowed,
            ));
        }
    }
    if func.ret.contains_any() {
        errors.push(Diagnostic::new(
            fn_node.span,
            DiagnosticKind::AnyTypeNotAllowed,
        ));
    }

    // if the function is generic we skip checking here
    // it will be done at instantiation time with concrete types
    let is_generic = !func.type_params.is_empty() || !func.const_params.is_empty();
    if is_generic {
        return;
    }

    let Some(info) = type_checker.get_var(func.name) else {
        errors.push(Diagnostic::new(
            fn_node.span,
            DiagnosticKind::UnknownFunction { name: func.name },
        ));

        return;
    };
    let ty = &info.ty;

    if !matches!(ty, Type::Func { .. }) {
        errors.push(Diagnostic::new(
            fn_node.span,
            DiagnosticKind::MismatchedTypes {
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

    let has_defaults = func.params.iter().any(|p| p.default.is_some());
    if has_defaults {
        let defaults = validate_param_defaults(
            &func.params,
            &func.type_params,
            func.name,
            fn_node.span,
            type_checker,
            errors,
        );
        type_checker.func_param_defaults.insert(func.name, defaults);
    }

    check_fn_body(
        func,
        &param_types,
        &ret_ty,
        fn_node.span,
        type_checker,
        errors,
    );
}

fn validate_param_defaults(
    params: &[Param],
    type_params: &[TypeParam],
    owner_name: Ident,
    owner_span: Span,
    type_checker: &TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Vec<Option<ConstValue>> {
    let known_consts: HashSet<Ident> = type_checker.ctx.const_defs.keys().copied().collect();
    let mut seen_default = false;
    let mut defaults = Vec::with_capacity(params.len());

    for param in params {
        match &param.default {
            None => {
                if seen_default {
                    errors.push(Diagnostic::new(
                        owner_span,
                        DiagnosticKind::RequiredParamAfterOptional {
                            func: owner_name,
                            param: param.name,
                        },
                    ));
                }
                defaults.push(None);
            }
            Some(expr) => {
                seen_default = true;

                if type_references_generic(&param.ty, type_params) {
                    errors.push(Diagnostic::new(
                        expr.span,
                        DiagnosticKind::ParamDefaultOnGenericType {
                            func: owner_name,
                            param: param.name,
                        },
                    ));
                    defaults.push(None);
                    continue;
                }

                if validate_const_expr(expr, &known_consts).is_err() {
                    errors.push(Diagnostic::new(
                        expr.span,
                        DiagnosticKind::ParamDefaultNotConst {
                            func: owner_name,
                            param: param.name,
                        },
                    ));
                    defaults.push(None);
                    continue;
                }

                let Ok(value) = eval_const_expr(expr, &type_checker.ctx.const_defs) else {
                    errors.push(Diagnostic::new(
                        expr.span,
                        DiagnosticKind::ParamDefaultNotConst {
                            func: owner_name,
                            param: param.name,
                        },
                    ));
                    defaults.push(None);
                    continue;
                };

                let resolved_ty = type_checker.resolve_type(&param.ty);
                if value == ConstValue::Nil {
                    if !resolved_ty.is_option() {
                        errors.push(Diagnostic::new(
                            expr.span,
                            DiagnosticKind::ParamDefaultTypeMismatch {
                                func: owner_name,
                                param: param.name,
                                expected: resolved_ty,
                                found: Type::Void,
                            },
                        ));
                        defaults.push(None);
                        continue;
                    }
                } else {
                    let value_ty = value.ty();
                    if value_ty != resolved_ty {
                        errors.push(Diagnostic::new(
                            expr.span,
                            DiagnosticKind::ParamDefaultTypeMismatch {
                                func: owner_name,
                                param: param.name,
                                expected: resolved_ty,
                                found: value_ty,
                            },
                        ));
                        defaults.push(None);
                        continue;
                    }
                }

                defaults.push(Some(value));
            }
        }
    }

    defaults
}

fn validate_field_defaults(
    kind: &'static str,
    struct_name: Ident,
    fields: &[StructField],
    type_params: &[TypeParam],
    struct_def: &mut StructDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let known_consts: HashSet<Ident> = type_checker.ctx.const_defs.keys().copied().collect();
    for field in fields {
        let Some(expr) = &field.default else {
            continue;
        };

        if type_references_generic(&field.ty, type_params) {
            errors.push(Diagnostic::new(
                expr.span,
                DiagnosticKind::FieldDefaultOnGenericType {
                    kind,
                    struct_name,
                    field: field.name,
                },
            ));
            continue;
        }

        let resolved_ty = type_checker.resolve_type(&field.ty);

        if let ExprKind::ArrayLiteral(arr) = &expr.node.kind
            && arr.node.elements.is_empty()
        {
            let is_array_or_list = matches!(resolved_ty, Type::Array { .. } | Type::List { .. });
            if is_array_or_list {
                struct_def
                    .field_defaults
                    .insert(field.name, FieldDefault::EmptyArray);
            } else {
                errors.push(Diagnostic::new(
                    expr.span,
                    DiagnosticKind::FieldDefaultTypeMismatch {
                        kind,
                        struct_name,
                        field: field.name,
                        expected: resolved_ty,
                        found: Type::Array {
                            elem: Type::Infer.boxed(),
                            len: ArrayLen::Fixed(0),
                        },
                    },
                ));
            }
            continue;
        }

        if let ExprKind::MapLiteral(map) = &expr.node.kind
            && map.node.entries.is_empty()
        {
            let is_map = matches!(resolved_ty, Type::Map { .. });
            if is_map {
                struct_def
                    .field_defaults
                    .insert(field.name, FieldDefault::EmptyMap);
            } else {
                errors.push(Diagnostic::new(
                    expr.span,
                    DiagnosticKind::FieldDefaultTypeMismatch {
                        kind,
                        struct_name,
                        field: field.name,
                        expected: resolved_ty,
                        found: Type::Map {
                            key: Type::Infer.boxed(),
                            value: Type::Infer.boxed(),
                        },
                    },
                ));
            }
            continue;
        }

        if validate_const_expr(expr, &known_consts).is_err() {
            errors.push(Diagnostic::new(
                expr.span,
                DiagnosticKind::FieldDefaultNotConst {
                    kind,
                    struct_name,
                    field: field.name,
                },
            ));
            continue;
        }

        let Ok(value) = eval_const_expr(expr, &type_checker.ctx.const_defs) else {
            errors.push(Diagnostic::new(
                expr.span,
                DiagnosticKind::FieldDefaultNotConst {
                    kind,
                    struct_name,
                    field: field.name,
                },
            ));
            continue;
        };

        if value == ConstValue::Nil {
            if !resolved_ty.is_option() {
                errors.push(Diagnostic::new(
                    expr.span,
                    DiagnosticKind::FieldDefaultTypeMismatch {
                        kind,
                        struct_name,
                        field: field.name,
                        expected: resolved_ty,
                        found: Type::Void,
                    },
                ));
                continue;
            }
        } else {
            let value_ty = value.ty();
            if value_ty != resolved_ty {
                errors.push(Diagnostic::new(
                    expr.span,
                    DiagnosticKind::FieldDefaultTypeMismatch {
                        kind,
                        struct_name,
                        field: field.name,
                        expected: resolved_ty,
                        found: value_ty,
                    },
                ));
                continue;
            }
        }

        struct_def
            .field_defaults
            .insert(field.name, FieldDefault::Const(value));
    }
}

pub(super) fn check_struct(
    struct_node: &AggregateDeclNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let decl = &struct_node.node;
    let struct_name = decl.name;
    let kind = decl.kind.keyword();

    let mut field_annotations = HashMap::new();
    for field in &decl.fields {
        let f_ann = normalize_annotations(&field.annotations, AnnotationTarget::Field, errors);
        field_annotations.insert(field.name, f_ann);
        if field.ty.contains_any() {
            errors.push(Diagnostic::new(
                struct_node.span,
                DiagnosticKind::AnyTypeNotAllowed,
            ));
        }
    }

    let Some(mut struct_def) = type_checker.get_struct(struct_name).cloned() else {
        return;
    };

    struct_def.field_annotations = field_annotations;

    validate_field_defaults(
        kind,
        struct_name,
        &decl.fields,
        &decl.type_params,
        &mut struct_def,
        type_checker,
        errors,
    );

    for method in &decl.methods {
        if let Some(method_def) = struct_def.methods.get_mut(&method.name) {
            method_def.annotations =
                normalize_annotations(&method.annotations, AnnotationTarget::InlineMethod, errors);

            if method.name == *TO_STRING_IDENT && method_def.annotations.has_internal() {
                errors.push(Diagnostic::new(
                    method.body.span,
                    DiagnosticKind::InternalOnToString,
                ));
            }
        }

        let has_defaults = method.params.iter().any(|p| p.default.is_some());
        if has_defaults && let Some(method_def) = struct_def.methods.get_mut(&method.name) {
            let defaults = validate_param_defaults(
                &method.params,
                &method.type_params,
                method.name,
                method.body.span,
                type_checker,
                errors,
            );
            method_def.param_defaults = defaults;
        }
    }

    type_checker
        .ctx
        .struct_defs
        .insert(struct_name, struct_def.clone());

    if let Some(method_def) = struct_def.methods.get(&*TO_STRING_IDENT) {
        validate_to_string_signature(kind, struct_name, method_def, struct_node.span, errors);
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
    kind: &'static str,
    struct_name: Ident,
    method: &MethodDef,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) {
    match method.receiver {
        None => {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::InvalidToStringSignature {
                    kind,
                    struct_name,
                    reason: "must have a 'self' receiver".to_string(),
                },
            ));
            return;
        }
        Some(MethodReceiver::Var) => {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::InvalidToStringSignature {
                    kind,
                    struct_name,
                    reason: "receiver must be 'self', not 'var self'".to_string(),
                },
            ));
        }
        Some(MethodReceiver::Value) => {}
    }

    if method.ret != Type::String {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::InvalidToStringSignature {
                kind,
                struct_name,
                reason: format!("must return 'string', found '{}'", method.ret),
            },
        ));
    }

    if !method.params.is_empty() {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::InvalidToStringSignature {
                kind,
                struct_name,
                reason: "must take no parameters besides 'self'".to_string(),
            },
        ));
    }

    if !method.type_params.is_empty() {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::InvalidToStringSignature {
                kind,
                struct_name,
                reason: "must not be generic".to_string(),
            },
        ));
    }
}
