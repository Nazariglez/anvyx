use crate::ast::{
    CastNode, ExprId, ExprKind, ExprNode, FloatSuffix, Ident, LambdaNode, Lit, StringPart, Type,
};
use std::collections::HashMap;

use super::{
    composite::{
        check_array_fill, check_array_literal, check_map_literal, check_named_tuple, check_range,
        check_struct_lit, check_tuple, check_tuple_index,
    },
    constraint::TypeRef,
    control::{check_if, check_if_let, check_match},
    error::{TypeErr, TypeErrKind},
    ops::{check_assign, check_binary, check_unary},
    postfix::{check_postfix_chain, collect_postfix_chain},
    stmt::check_block_expr,
    types::TypeChecker,
};

pub(super) fn check_expr(
    expr_node: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
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
                errors.push(TypeErr::new(
                    expr_node.span,
                    TypeErrKind::UnknownVariable { name: *ident },
                ));
                Type::Infer
            }
        }
        ExprKind::Block(spanned) => {
            let (block_ty, _) = check_block_expr(spanned, type_checker, errors, None);
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
        ExprKind::Cast(cast_node) => check_cast(cast_node, type_checker, errors),
        ExprKind::Lambda(lambda_node) => {
            check_lambda(lambda_node, expr_node, type_checker, errors, expected)
        }
        ExprKind::Field(_) | ExprKind::Index(_) | ExprKind::Call(_) => {
            unreachable!("postfix expressions should be routed through check_postfix_chain")
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
            Some(Type::Float) => Type::Float,
            Some(Type::Double) => Type::Double,
            _ => Type::Float,
        },
    }
}

fn check_cast(
    cast_node: &CastNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let from_ty = check_expr(&cast_node.node.expr, type_checker, errors, None);
    let to_ty = &cast_node.node.target;

    let valid = match (&from_ty, to_ty) {
        (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,
        (Type::Int, Type::Double) | (Type::Double, Type::Int) => true,
        (Type::Float, Type::Double) | (Type::Double, Type::Float) => true,
        _ => from_ty == *to_ty,
    };

    if !valid {
        errors.push(TypeErr::new(
            cast_node.span,
            TypeErrKind::InvalidCast {
                from: from_ty,
                to: to_ty.clone(),
            },
        ));
    }

    to_ty.clone()
}

fn check_string_interp(
    parts: &[StringPart],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    for part in parts {
        let StringPart::Expr(expr_node) = part else {
            continue;
        };
        check_expr(expr_node, type_checker, errors, None);
    }
    Type::String
}

fn check_lambda(
    lambda_node: &LambdaNode,
    expr_node: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
    expected: Option<&Type>,
) -> Type {
    let lambda = &lambda_node.node;

    // extract expected param/return types from context
    let (expected_params, expected_ret): (Option<&[Type]>, Option<&Type>) = match expected {
        Some(Type::Func { params, ret }) => (Some(params.as_slice()), Some(ret.as_ref())),
        _ => (None, None),
    };

    // resolve parameter types
    let mut param_types: Vec<Type> = Vec::with_capacity(lambda.params.len());
    if let Some(ep) = expected_params {
        if lambda.params.len() != ep.len() {
            errors.push(TypeErr::new(
                expr_node.span,
                TypeErrKind::LambdaParamCountMismatch {
                    expected: ep.len(),
                    found: lambda.params.len(),
                },
            ));
            return Type::Infer;
        }
        for (param, expected_param_ty) in lambda.params.iter().zip(ep.iter()) {
            let ty = match &param.ty {
                Some(annotated) => {
                    let resolved = type_checker.resolve_type(annotated);
                    if resolved != *expected_param_ty {
                        errors.push(TypeErr::new(
                            expr_node.span,
                            TypeErrKind::MismatchedTypes {
                                expected: expected_param_ty.clone(),
                                found: resolved.clone(),
                            },
                        ));
                    }
                    resolved
                }
                None => expected_param_ty.clone(),
            };
            param_types.push(ty);
        }
    } else {
        for param in &lambda.params {
            let ty = match &param.ty {
                Some(annotated) => type_checker.resolve_type(annotated),
                None => {
                    errors.push(TypeErr::new(
                        expr_node.span,
                        TypeErrKind::CannotInferLambdaParam { name: param.name },
                    ));
                    Type::Infer
                }
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
        type_checker.set_var(param.name, ty.clone(), false);
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

            if effective_ret.is_void() {
                if !body_ty.is_void() {
                    errors.push(TypeErr::new(
                        expr_node.span,
                        TypeErrKind::MismatchedTypes {
                            expected: Type::Void,
                            found: body_ty,
                        },
                    ));
                }
            } else if let Some(last_id) = last_expr_id {
                let expr_ref = TypeRef::Expr(last_id);
                let ret_ref = TypeRef::concrete(&effective_ret);
                type_checker.constrain_assignable(expr_node.span, expr_ref, ret_ref, errors);
            } else if !had_explicit_return {
                errors.push(TypeErr::new(
                    expr_node.span,
                    TypeErrKind::MismatchedTypes {
                        expected: effective_ret.clone(),
                        found: Type::Void,
                    },
                ));
            }
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

    Type::Func {
        params: param_types,
        ret: Box::new(actual_ret),
    }
}
