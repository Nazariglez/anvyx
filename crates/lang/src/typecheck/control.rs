use std::collections::HashSet;

use crate::ast::{ExprKind, ExprNode, ForNode, Ident, IfNode, MatchNode, Type, WhileNode};
use crate::span::Span;

use super::{
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    pattern::{check_match_pattern, check_pattern},
    stmt::{check_block_expr, check_block_stmts},
    types::{EnumDef, TypeChecker},
    unify::{contains_infer, unify_types},
};

pub(super) fn check_while(
    while_node: &WhileNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let node = &while_node.node;
    let cond_ty = check_expr(&node.cond, type_checker, errors);
    let maybe_bool = cond_ty.is_bool() || cond_ty.is_infer();
    if !maybe_bool {
        errors.push(TypeErr::new(
            node.cond.span,
            TypeErrKind::WhileConditionNotBool {
                found: cond_ty.clone(),
            },
        ));
        return;
    }

    type_checker.enter_loop();
    let _ = check_block_stmts(&node.body.node.stmts, type_checker, errors);
    type_checker.exit_loop();
}

pub(super) fn check_for(
    for_node: &ForNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let node = &for_node.node;

    let iterable_ty = check_expr(&node.iterable, type_checker, errors);
    let item_ty = extract_range_item_type(&iterable_ty, for_node.span, errors);

    if let Some(ref step_expr) = node.step {
        let step_ty = check_expr(step_expr, type_checker, errors);
        let item_is_int = matches!(item_ty, Type::Int | Type::Infer);
        let step_is_int = matches!(step_ty, Type::Int | Type::Infer);
        if !item_is_int || !step_is_int {
            errors.push(TypeErr::new(
                step_expr.span,
                TypeErrKind::ForStepNotInt {
                    item_ty: item_ty.clone(),
                    step_ty,
                },
            ));
        }
    }

    type_checker.push_scope();
    type_checker.enter_loop();

    check_pattern(&node.pattern, &item_ty, type_checker, errors);

    let _ = check_block_stmts(&node.body.node.stmts, type_checker, errors);

    type_checker.exit_loop();
    type_checker.pop_scope();
}

pub(super) fn extract_range_item_type(ty: &Type, span: Span, errors: &mut Vec<TypeErr>) -> Type {
    match ty {
        Type::Struct { name, type_args } => {
            let name_str = name.0.as_ref();
            let is_range = name_str == "Range" || name_str == "RangeInclusive";
            if is_range && type_args.len() == 1 {
                return type_args[0].clone();
            }
            errors.push(TypeErr::new(
                span,
                TypeErrKind::ForIterableNotRange { found: ty.clone() },
            ));
            Type::Infer
        }
        Type::Infer => Type::Infer,
        _ => {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::ForIterableNotRange { found: ty.clone() },
            ));
            Type::Infer
        }
    }
}

pub(super) fn check_if(
    if_node: &IfNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &if_node.node;

    let cond_ty = check_expr(&node.cond, type_checker, errors);
    let maybe_bool = cond_ty.is_bool() || cond_ty.is_infer();
    if !maybe_bool {
        errors.push(TypeErr::new(
            node.cond.span,
            TypeErrKind::IfConditionNotBool { found: cond_ty },
        ));
    }

    let (then_ty, then_expr_id) = check_block_expr(&node.then_block, type_checker, errors);

    // if there is no else block then the type is void and this must be a statment
    let Some(else_block) = &node.else_block else {
        return Type::Void;
    };

    let (else_ty, else_expr_id) = check_block_expr(else_block, type_checker, errors);

    // unify branch types
    let same_ty = then_ty == else_ty;
    if same_ty {
        return then_ty;
    }

    // handle T vs Infer? case where one branch is nil
    let then_is_nil = is_optional_with_infer(&then_ty);
    let else_is_nil = is_optional_with_infer(&else_ty);

    if !then_ty.is_optional() && else_is_nil {
        let result_ty = Type::Optional(then_ty.boxed());
        if let Some(id) = else_expr_id {
            type_checker.set_type(id, result_ty.clone(), else_block.span);
        }
        return result_ty;
    }

    if !else_ty.is_optional() && then_is_nil {
        let result_ty = Type::Optional(else_ty.boxed());
        if let Some(id) = then_expr_id {
            type_checker.set_type(id, result_ty.clone(), node.then_block.span);
        }
        return result_ty;
    }

    // check if unifiable
    let is_unifiable = contains_infer(&then_ty) || contains_infer(&else_ty);
    if !is_unifiable {
        errors.push(TypeErr::new(
            if_node.span,
            TypeErrKind::MismatchedTypes {
                expected: then_ty.clone(),
                found: else_ty.clone(),
            },
        ));
        return Type::Infer;
    }

    // return the type of the branch that doesn't contain infer
    if contains_infer(&then_ty) {
        else_ty
    } else {
        then_ty
    }
}

pub(super) fn is_optional_with_infer(ty: &Type) -> bool {
    matches!(ty, Type::Optional(inner) if inner.as_ref().is_infer())
}

pub(super) fn is_if_without_else(expr: &ExprNode) -> bool {
    match &expr.node.kind {
        ExprKind::If(if_node) => if_node.node.else_block.is_none(),
        _ => false,
    }
}

pub(super) fn check_match(
    match_node: &MatchNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let scrutinee = &match_node.node.scrutinee;
    let scrutinee_ty = check_expr(scrutinee, type_checker, errors);

    let Type::Enum {
        name: enum_name, ..
    } = &scrutinee_ty
    else {
        errors.push(TypeErr::new(
            scrutinee.span,
            TypeErrKind::MatchScrutineeNotEnum {
                found: scrutinee_ty.clone(),
            },
        ));
        return Type::Infer;
    };

    let Some(enum_def) = type_checker.get_enum(*enum_name) else {
        errors.push(TypeErr::new(
            scrutinee.span,
            TypeErrKind::UnknownEnum { name: *enum_name },
        ));
        return Type::Infer;
    };

    let enum_def = enum_def.clone();
    let mut covered_variants: HashSet<Ident> = HashSet::new();
    let mut has_wildcard = false;
    let mut arm_types: Vec<Type> = vec![];

    for arm in &match_node.node.arms {
        type_checker.push_scope();

        check_match_pattern(
            &arm.node.pattern,
            &scrutinee_ty,
            &enum_def,
            &mut covered_variants,
            &mut has_wildcard,
            type_checker,
            errors,
        );

        let arm_ty = check_expr(&arm.node.body, type_checker, errors);
        arm_types.push(arm_ty);

        type_checker.pop_scope();
    }

    check_exhaustiveness(
        &enum_def,
        &covered_variants,
        has_wildcard,
        match_node.span,
        errors,
    );
    unify_arm_types(&arm_types, match_node.span, errors)
}

fn check_exhaustiveness(
    enum_def: &EnumDef,
    covered: &HashSet<Ident>,
    has_wildcard: bool,
    span: Span,
    errors: &mut Vec<TypeErr>,
) {
    if has_wildcard {
        return;
    }

    let all_variants: HashSet<Ident> = enum_def.variants.iter().map(|v| v.name).collect();
    let missing: Vec<Ident> = all_variants.difference(covered).cloned().collect();
    if !missing.is_empty() {
        errors.push(TypeErr::new(
            span,
            TypeErrKind::NonExhaustiveMatch { missing },
        ));
    }
}

fn unify_arm_types(arm_types: &[Type], span: Span, errors: &mut Vec<TypeErr>) -> Type {
    if arm_types.is_empty() {
        return Type::Void;
    }

    let mut result = arm_types[0].clone();
    for ty in arm_types.iter().skip(1) {
        let mut unify_errors = vec![];
        match unify_types(&result, ty, span, &mut unify_errors) {
            Some(unified) => result = unified,
            None => {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::MatchArmTypeMismatch {
                        expected: result.clone(),
                        found: ty.clone(),
                    },
                ));
            }
        }
    }
    result
}
