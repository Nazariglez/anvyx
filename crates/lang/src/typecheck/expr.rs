use std::collections::HashMap;

use super::{
    composite::{
        check_array_fill, check_array_literal, check_map_literal, check_named_tuple, check_range,
        check_struct_lit, check_tuple, check_tuple_index, validate_field_names,
    },
    constraint::TypeRef,
    control::{check_if, check_if_let, check_match},
    decl::validate_block_return,
    error::{Diagnostic, DiagnosticKind},
    infer::{build_subst, subst_type},
    ops::{check_assign, check_binary, check_unary},
    postfix::{check_postfix_chain, collect_postfix_chain},
    stmt::check_block_expr,
    types::TypeChecker,
};
use crate::{
    ast::{
        CastNode, ExprId, ExprKind, ExprNode, FloatSuffix, FormatKind, FormatSign, FormatSpec,
        FuncParam, Ident, InferredEnumArgs, InferredEnumNode, LambdaNode, Lit, StringPart,
        StructField, Type, TypeVarId, VariantKind,
    },
    span::Span,
};

pub(super) fn check_expr(
    expr_node: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Type {
    // route all postfix expressions through the shared chain checker
    if matches!(
        expr_node.node.kind,
        ExprKind::Field(_) | ExprKind::Index(_) | ExprKind::Call(_)
    ) {
        let (base, chain) = collect_postfix_chain(expr_node);
        return check_postfix_chain(expr_node, base, &chain, type_checker, errors, expected);
    }

    let expr = &expr_node.node;
    let ty = match &expr.kind {
        ExprKind::Ident(ident) => {
            if let Some(info) = type_checker.get_var(*ident) {
                let ty = info.ty.clone();
                type_checker.track_capture(*ident);
                ty
            } else if let Some(const_def) = type_checker.get_const(*ident) {
                let val = const_def.value.clone();
                let ty = const_def.ty.clone();
                type_checker.const_values.insert(expr_node.node.id, val);
                ty
            } else {
                errors.push(Diagnostic::new(
                    expr_node.span,
                    DiagnosticKind::UnknownVariable { name: *ident },
                ));
                Type::Infer
            }
        }
        ExprKind::Block(spanned) => {
            let (block_ty, _) = check_block_expr(spanned, type_checker, errors, expected);
            block_ty
        }
        ExprKind::Lit(lit) => match lit {
            Lit::Float { suffix, .. } => resolve_float_type(*suffix, expected),
            _ => type_from_lit(lit),
        },
        ExprKind::Binary(bin) => check_binary(bin, type_checker, errors),
        ExprKind::Unary(unary) => check_unary(unary, type_checker, errors),
        ExprKind::Assign(assign) => check_assign(assign, type_checker, errors),
        ExprKind::If(if_node) => check_if(if_node, type_checker, errors),
        ExprKind::IfLet(if_let_node) => check_if_let(if_let_node, type_checker, errors),
        ExprKind::Tuple(elements) => check_tuple(elements, type_checker, errors),
        ExprKind::NamedTuple(elements) => {
            check_named_tuple(elements, expr_node.span, type_checker, errors)
        }
        ExprKind::TupleIndex(index_node) => check_tuple_index(index_node, type_checker, errors),
        ExprKind::StructLiteral(lit_node) => check_struct_lit(lit_node, type_checker, errors),
        ExprKind::Range(range_node) => check_range(range_node, type_checker, errors),
        ExprKind::ArrayLiteral(lit_node) => check_array_literal(lit_node, type_checker, errors),
        ExprKind::ArrayFill(fill_node) => check_array_fill(fill_node, type_checker, errors),
        ExprKind::MapLiteral(lit_node) => check_map_literal(lit_node, type_checker, errors),
        ExprKind::Match(match_node) => check_match(match_node, type_checker, errors),
        ExprKind::StringInterp(parts) => check_string_interp(parts, type_checker, errors),
        ExprKind::Cast(cast_node) => check_cast(cast_node, expr.id, type_checker, errors),
        ExprKind::Lambda(lambda_node) => {
            check_lambda(lambda_node, expr_node, type_checker, errors, expected)
        }
        ExprKind::InferredEnum(node) => {
            check_inferred_enum(node, expr_node.span, type_checker, errors, expected)
        }
        ExprKind::Field(_) | ExprKind::Index(_) | ExprKind::Call(_) => {
            unreachable!("postfix expressions should be routed through check_postfix_chain")
        }
        ExprKind::IntrinsicCall(_) => {
            unreachable!("IntrinsicCall should have been resolved before typechecking")
        }
    };

    type_checker.set_type(expr_node.node.id, ty.clone(), expr_node.span);
    ty
}

pub(super) fn root_ident(expr: &ExprNode) -> Option<Ident> {
    match &expr.node.kind {
        ExprKind::Ident(name) => Some(*name),
        ExprKind::Field(field) => root_ident(&field.node.target),
        ExprKind::Index(index) => root_ident(&index.node.target),
        _ => None,
    }
}

pub(super) fn field_path(expr: &ExprNode) -> Option<(Ident, Vec<Ident>)> {
    match &expr.node.kind {
        ExprKind::Ident(name) => Some((*name, vec![])),
        ExprKind::Field(field) => {
            let (root, mut path) = field_path(&field.node.target)?;
            path.push(field.node.field);
            Some((root, path))
        }
        _ => None,
    }
}

pub(super) fn type_from_lit(lit: &Lit) -> Type {
    match lit {
        Lit::Int(_) => Type::Int,
        Lit::Float { .. } => unreachable!("float literals handled in check_expr"),
        Lit::Bool(_) => Type::Bool,
        Lit::String(_) => Type::String,
        Lit::Nil => Type::option_of(Type::Infer),
    }
}

fn resolve_float_type(suffix: Option<FloatSuffix>, expected: Option<&Type>) -> Type {
    match suffix {
        Some(FloatSuffix::F) => Type::Float,
        Some(FloatSuffix::D) => Type::Double,
        None => match expected {
            Some(Type::Double) => Type::Double,
            _ => Type::Float,
        },
    }
}

fn check_cast(
    cast_node: &CastNode,
    expr_id: ExprId,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let from_ty = check_expr(&cast_node.node.expr, type_checker, errors, None);
    let to_ty = type_checker.resolve_type(&cast_node.node.target);

    let is_primitive_cast = matches!(
        (&from_ty, &to_ty),
        (Type::Int, Type::Float | Type::Double)
            | (Type::Float, Type::Int | Type::Double)
            | (Type::Double, Type::Int | Type::Float)
    );
    let is_same_type = from_ty == to_ty;

    if !is_primitive_cast && !is_same_type {
        if let Some(entry) = type_checker
            .ctx
            .cast_defs
            .get(&(from_ty.clone(), to_ty.clone()))
        {
            type_checker
                .user_cast_targets
                .insert(expr_id, entry.internal_name);
        } else {
            let help = format!("define 'cast from(v: {from_ty})' in an 'extend {to_ty}' block");
            errors.push(
                Diagnostic::new(
                    cast_node.span,
                    DiagnosticKind::InvalidCast {
                        from: from_ty,
                        to: to_ty.clone(),
                    },
                )
                .with_help(help),
            );
        }
    }

    to_ty
}

fn check_string_interp(
    parts: &[StringPart],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    for part in parts {
        let StringPart::Expr(expr_node, fmt) = part else {
            continue;
        };
        let ty = check_expr(expr_node, type_checker, errors, None);
        if let Some(spec) = fmt {
            validate_format_spec(&spec.node, &ty, spec.span, errors);
        }
    }
    Type::String
}

fn validate_format_spec(
    spec: &FormatSpec,
    expr_type: &Type,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) {
    if matches!(expr_type, Type::Infer | Type::Void) {
        return;
    }

    match spec.kind {
        FormatKind::Hex | FormatKind::HexUpper | FormatKind::Binary => {
            if !matches!(expr_type, Type::Int) {
                let label = match spec.kind {
                    FormatKind::Hex => "x",
                    FormatKind::HexUpper => "X",
                    FormatKind::Binary => "b",
                    _ => unreachable!(),
                };
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::InvalidFormatSpec {
                        reason: format!("format type '{label}' requires int, found '{expr_type}'"),
                    },
                ));
                return;
            }
        }
        FormatKind::Exp | FormatKind::ExpUpper => {
            if !matches!(expr_type, Type::Float | Type::Double) {
                let label = match spec.kind {
                    FormatKind::Exp => "e",
                    FormatKind::ExpUpper => "E",
                    _ => unreachable!(),
                };
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::InvalidFormatSpec {
                        reason: format!(
                            "format type '{label}' requires float or double, found '{expr_type}'"
                        ),
                    },
                ));
                return;
            }
        }
        FormatKind::Default => {}
    }

    if spec.precision.is_some() && !matches!(expr_type, Type::Float | Type::Double | Type::String) {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::InvalidFormatSpec {
                reason: format!("precision not supported for '{expr_type}'"),
            },
        ));
        return;
    }

    if matches!(spec.sign, FormatSign::Always)
        && !matches!(expr_type, Type::Int | Type::Float | Type::Double)
    {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::InvalidFormatSpec {
                reason: format!("sign format requires a numeric type, found '{expr_type}'"),
            },
        ));
    }
}

fn check_lambda(
    lambda_node: &LambdaNode,
    expr_node: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Type {
    let lambda = &lambda_node.node;

    // extract expected param/return types from context
    let (expected_params, expected_ret): (Option<&[FuncParam]>, Option<&Type>) = match expected {
        Some(Type::Func { params, ret }) => (Some(params.as_slice()), Some(ret.as_ref())),
        _ => (None, None),
    };

    // resolve parameter types
    let mut param_types: Vec<Type> = Vec::with_capacity(lambda.params.len());
    if let Some(ep) = expected_params {
        if lambda.params.len() != ep.len() {
            errors.push(Diagnostic::new(
                expr_node.span,
                DiagnosticKind::LambdaParamCountMismatch {
                    expected: ep.len(),
                    found: lambda.params.len(),
                },
            ));
            return Type::Infer;
        }
        for (param, expected_param_fp) in lambda.params.iter().zip(ep.iter()) {
            let ty = match &param.ty {
                Some(annotated) => {
                    let resolved = type_checker.resolve_type(annotated);
                    if resolved != expected_param_fp.ty {
                        errors.push(Diagnostic::new(
                            expr_node.span,
                            DiagnosticKind::MismatchedTypes {
                                expected: expected_param_fp.ty.clone(),
                                found: resolved.clone(),
                            },
                        ));
                    }
                    resolved
                }
                None => expected_param_fp.ty.clone(),
            };
            param_types.push(ty);
        }
    } else {
        for param in &lambda.params {
            let ty = if let Some(annotated) = &param.ty {
                type_checker.resolve_type(annotated)
            } else {
                errors.push(Diagnostic::new(
                    expr_node.span,
                    DiagnosticKind::CannotInferLambdaParam { name: param.name },
                ));
                Type::Infer
            };
            param_types.push(ty);
        }
    }

    // determine return type hint
    let ret_hint: Option<Type> = match &lambda.ret_type {
        Some(rt) => Some(type_checker.resolve_type(rt)),
        None => expected_ret.cloned(),
    };

    // push lambda boundary, captures stack, and scope
    let boundary = type_checker.scopes.len();
    type_checker.lambda_boundaries.push(boundary);
    type_checker.current_lambda_captures.push(HashMap::new());
    type_checker.push_scope();

    for (param, ty) in lambda.params.iter().zip(param_types.iter()) {
        type_checker.set_var(param.name, ty.clone(), param.mutable);
    }

    // push return type onto stack so return statements inside are checked correctly
    let effective_ret = ret_hint.clone().unwrap_or(Type::Infer);
    type_checker.push_return_type(effective_ret.clone(), None);

    let actual_ret = if let ExprKind::Block(block_node) = &lambda.body.node.kind {
        if matches!(effective_ret, Type::Infer) {
            // no return type hint, infer from body
            let (body_ty, last_expr_id) = check_block_expr(block_node, type_checker, errors, None);
            if last_expr_id.is_some() {
                body_ty
            } else {
                Type::Void
            }
        } else {
            let expected_tail = if effective_ret.is_void() {
                None
            } else {
                Some(&effective_ret)
            };
            let (body_ty, last_expr_id) =
                check_block_expr(block_node, type_checker, errors, expected_tail);
            let had_explicit_return = type_checker.has_explicit_return();

            validate_block_return(
                &effective_ret,
                body_ty,
                last_expr_id,
                had_explicit_return,
                expr_node.span,
                type_checker,
                errors,
            );
            effective_ret.clone()
        }
    } else {
        // expression body
        let body_ty = check_expr(&lambda.body, type_checker, errors, ret_hint.as_ref());
        if matches!(effective_ret, Type::Infer) {
            body_ty
        } else {
            let body_ref = TypeRef::Expr(lambda.body.node.id);
            let ret_ref = TypeRef::concrete(&effective_ret);
            type_checker.constrain_assignable(expr_node.span, body_ref, ret_ref, errors);
            effective_ret.clone()
        }
    };

    type_checker.pop_return_type();
    type_checker.pop_scope();

    let captures = type_checker
        .current_lambda_captures
        .pop()
        .unwrap_or_default();
    type_checker.lambda_boundaries.pop();

    let capture_list: Vec<(Ident, Type)> = captures.into_iter().collect();
    type_checker
        .lambda_captures
        .insert(expr_node.node.id, capture_list);

    let cast_accepts: Vec<bool> = lambda.params.iter().map(|p| p.cast_accept).collect();
    if cast_accepts.iter().any(|&f| f) {
        type_checker
            .lambda_cast_accepts
            .insert(expr_node.node.id, cast_accepts);
    }

    Type::Func {
        params: param_types
            .into_iter()
            .zip(lambda.params.iter())
            .map(|(ty, lp)| FuncParam::new(ty, lp.mutable))
            .collect(),
        ret: Box::new(actual_ret),
    }
}

fn check_inferred_enum(
    node: &InferredEnumNode,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Type {
    let resolved_expected = expected.map(|t| type_checker.resolve_type(t));
    let Some(Type::Enum {
        name: enum_name,
        type_args,
        ..
    }) = resolved_expected.as_ref()
    else {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::CannotInferEnumVariant {
                variant: node.node.variant,
            },
        ));
        return Type::Infer;
    };

    let enum_name = *enum_name;
    let type_args = type_args.clone();
    let variant_name = node.node.variant;

    let Some(enum_def) = type_checker.get_enum(enum_name).cloned() else {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::UnknownEnum { name: enum_name },
        ));
        return Type::Infer;
    };

    let Some(variant) = enum_def.variants.iter().find(|v| v.name == variant_name) else {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::UnknownEnumVariant {
                enum_name,
                variant_name,
            },
        ));
        return Type::Infer;
    };

    enum_def.check_deprecation(enum_name, variant, span, errors);

    let type_params = &enum_def.type_params;
    match (&node.node.args, &variant.kind) {
        (InferredEnumArgs::Unit, VariantKind::Unit) => enum_def.make_type(enum_name, type_args),
        (InferredEnumArgs::Unit, _) => {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::EnumVariantNotUnit {
                    enum_name,
                    variant_name,
                },
            ));
            Type::Infer
        }
        (InferredEnumArgs::Tuple(args), VariantKind::Tuple(expected_types)) => {
            if args.len() != expected_types.len() {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::EnumVariantArityMismatch {
                        enum_name,
                        variant_name,
                        expected: expected_types.len(),
                        found: args.len(),
                    },
                ));
                return Type::Infer;
            }
            let subst = build_subst(type_params, &type_args);
            for (arg_expr, field_ty) in args.iter().zip(expected_types.iter()) {
                let substituted = subst_type(field_ty, &subst, &HashMap::new());
                check_expr(arg_expr, type_checker, errors, Some(&substituted));
                let arg_ref = TypeRef::Expr(arg_expr.node.id);
                let expected_ref = TypeRef::concrete(&substituted);
                type_checker.constrain_assignable(arg_expr.span, arg_ref, expected_ref, errors);
            }
            enum_def.make_type(enum_name, type_args)
        }
        (InferredEnumArgs::Tuple(_), _) => {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::EnumVariantNotTuple {
                    enum_name,
                    variant_name,
                },
            ));
            Type::Infer
        }
        (InferredEnumArgs::Struct(fields), VariantKind::Struct(expected_fields)) => {
            let subst = build_subst(type_params, &type_args);
            validate_and_constrain_enum_struct_fields(
                fields,
                expected_fields,
                span,
                enum_name,
                variant_name,
                &subst,
                type_checker,
                errors,
            );
            enum_def.make_type(enum_name, type_args)
        }
        (InferredEnumArgs::Struct(_), _) => {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::EnumVariantNotStruct {
                    enum_name,
                    variant_name,
                },
            ));
            Type::Infer
        }
    }
}

fn validate_and_constrain_enum_struct_fields(
    fields: &[(Ident, ExprNode)],
    expected_fields: &[StructField],
    span: Span,
    enum_name: Ident,
    variant_name: Ident,
    subst: &HashMap<TypeVarId, Type>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let field_type_map: HashMap<Ident, &Type> =
        expected_fields.iter().map(|f| (f.name, &f.ty)).collect();
    for (name, field_expr) in fields {
        let field_expected = field_type_map
            .get(name)
            .map(|ty| subst_type(ty, subst, &HashMap::new()));
        check_expr(field_expr, type_checker, errors, field_expected.as_ref());
    }
    let provided: Vec<(Ident, Span)> = fields.iter().map(|(n, e)| (*n, e.span)).collect();
    validate_field_names(
        &provided,
        span,
        expected_fields,
        false,
        |field| DiagnosticKind::EnumVariantDuplicateField {
            enum_name,
            variant_name,
            field,
        },
        |field| DiagnosticKind::EnumVariantUnknownField {
            enum_name,
            variant_name,
            field,
        },
        |field| DiagnosticKind::EnumVariantMissingField {
            enum_name,
            variant_name,
            field,
        },
        None,
        errors,
    );
    for (name, field_expr) in fields {
        if let Some(expected_def) = expected_fields.iter().find(|f| f.name == *name) {
            let substituted = subst_type(&expected_def.ty, subst, &HashMap::new());
            let field_ref = TypeRef::Expr(field_expr.node.id);
            let expected_ref = TypeRef::concrete(&substituted);
            type_checker.constrain_assignable(field_expr.span, field_ref, expected_ref, errors);
        }
    }
}
