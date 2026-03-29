use std::collections::HashSet;

use crate::ast::{
    self, BlockNode, ExprKind, ExprNode, ForNode, Ident, IfLetNode, IfNode, Lit, MatchNode,
    Pattern, Stmt, Type, WhileLetNode, WhileNode,
};
use crate::span::Span;
use internment::Intern;

use super::{
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    pattern::{check_match_pattern, check_pattern, check_pattern_in_match},
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
    let cond_ty = check_expr(&node.cond, type_checker, errors, None);
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
    let _ = check_block_stmts(
        &node.body.node.stmts,
        node.body.node.tail.as_deref(),
        type_checker,
        errors,
        None,
    );
    type_checker.exit_loop();
}

pub(super) fn check_while_let(
    while_let_node: &WhileLetNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let node = &while_let_node.node;
    let value_ty = check_expr(&node.value, type_checker, errors, None);

    type_checker.push_scope();
    check_pattern(&node.pattern, &value_ty, false, type_checker, errors);

    type_checker.enter_loop();
    let _ = check_block_stmts(
        &node.body.node.stmts,
        node.body.node.tail.as_deref(),
        type_checker,
        errors,
        None,
    );
    type_checker.exit_loop();

    type_checker.pop_scope();
}

pub(super) fn check_for(
    for_node: &ForNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    let node = &for_node.node;

    let iterable_ty = check_expr(&node.iterable, type_checker, errors, None);
    let item_ty = extract_iterable_item_type(&iterable_ty, for_node.span, errors);

    let is_map = iterable_ty.is_map();

    if let Some(ref step_expr) = node.step {
        let step_ty = check_expr(step_expr, type_checker, errors, None);

        if is_map {
            errors.push(TypeErr::new(
                step_expr.span,
                TypeErrKind::ForMapStepNotAllowed,
            ));
        } else {
            let is_seq = is_sequence_type(&iterable_ty);
            let step_is_int = matches!(step_ty, Type::Int | Type::Infer);

            if is_seq {
                if !step_is_int {
                    errors.push(TypeErr::new(
                        step_expr.span,
                        TypeErrKind::ForStepNotInt {
                            item_ty: item_ty.clone(),
                            step_ty,
                        },
                    ));
                }
            } else {
                let item_is_int = matches!(item_ty, Type::Int | Type::Infer);
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
        }
    }

    if is_map && node.reversed {
        errors.push(TypeErr::new(
            for_node.span,
            TypeErrKind::ForMapRevNotAllowed,
        ));
    }

    let is_range_from =
        matches!(&iterable_ty, Type::Struct { name, .. } if name.0.as_ref() == "RangeFrom");
    if is_range_from && node.reversed {
        errors.push(TypeErr::new(
            for_node.span,
            TypeErrKind::ForRangeFromRevNotAllowed,
        ));
    }

    let effective_ty = infer_for_effective_type(&node.pattern, &item_ty, &iterable_ty);

    type_checker.push_scope();
    type_checker.enter_loop();

    check_pattern(&node.pattern, &effective_ty, false, type_checker, errors);

    let _ = check_block_stmts(
        &node.body.node.stmts,
        node.body.node.tail.as_deref(),
        type_checker,
        errors,
        None,
    );

    type_checker.exit_loop();
    type_checker.pop_scope();
}

pub(super) fn extract_iterable_item_type(ty: &Type, span: Span, errors: &mut Vec<TypeErr>) -> Type {
    match ty {
        Type::Struct { name, type_args } => {
            let name_str = name.0.as_ref();
            let is_range =
                name_str == "Range" || name_str == "RangeInclusive" || name_str == "RangeFrom";
            if is_range && type_args.len() == 1 {
                return type_args[0].clone();
            }
            errors.push(TypeErr::new(
                span,
                TypeErrKind::ForIterableNotSupported { found: ty.clone() },
            ));
            Type::Infer
        }
        Type::Array { elem, .. } => *elem.clone(),
        Type::List { elem } => *elem.clone(),
        Type::ArrayView { elem } => *elem.clone(),
        Type::Map { key, value } => Type::Tuple(vec![*key.clone(), *value.clone()]),
        Type::Infer => Type::Infer,
        _ => {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::ForIterableNotSupported { found: ty.clone() },
            ));
            Type::Infer
        }
    }
}

fn is_range_type(ty: &Type) -> bool {
    matches!(ty, Type::Struct { name, .. } if {
        let s = name.0.as_ref();
        s == "Range" || s == "RangeInclusive" || s == "RangeFrom"
    })
}

fn is_sequence_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Array { .. } | Type::List { .. } | Type::ArrayView { .. }
    )
}

fn infer_for_effective_type(
    pattern: &ast::PatternNode,
    item_ty: &Type,
    iterable_ty: &Type,
) -> Type {
    if is_range_type(iterable_ty) {
        return item_ty.clone();
    }

    let Pattern::Tuple(subs) = &pattern.node else {
        return item_ty.clone();
    };

    if subs.len() != 2 {
        return item_ty.clone();
    }

    if let Some(elems) = item_ty.tuple_element_types()
        && elems.len() == 2
    {
        if let Pattern::Tuple(inner_subs) = &subs[1].node {
            let t2_matches = elems[1]
                .tuple_element_types()
                .is_some_and(|t2_elems| t2_elems.len() == inner_subs.len());
            if t2_matches {
                return item_ty.clone();
            }
            return Type::Tuple(vec![Type::Int, item_ty.clone()]);
        }
        return item_ty.clone();
    }

    Type::Tuple(vec![Type::Int, item_ty.clone()])
}

pub(super) fn check_if(
    if_node: &IfNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &if_node.node;

    let cond_ty = check_expr(&node.cond, type_checker, errors, None);
    let maybe_bool = cond_ty.is_bool() || cond_ty.is_infer();
    if !maybe_bool {
        errors.push(TypeErr::new(
            node.cond.span,
            TypeErrKind::IfConditionNotBool { found: cond_ty },
        ));
    }

    let (then_ty, then_expr_id) = check_block_expr(&node.then_block, type_checker, errors, None);

    // if there is no else block then the type is void and this must be a statment
    let Some(else_block) = &node.else_block else {
        return Type::Void;
    };

    let (else_ty, else_expr_id) = check_block_expr(else_block, type_checker, errors, None);

    // unify branch types
    let same_ty = then_ty == else_ty;
    if same_ty {
        return then_ty;
    }

    // handle T vs nil case where one branch is nil
    let then_is_nil = then_ty.is_option_with_infer();
    let else_is_nil = else_ty.is_option_with_infer();

    if !then_ty.is_option() && else_is_nil {
        let result_ty = Type::option_of(then_ty);
        if let Some(id) = else_expr_id {
            type_checker.set_type(id, result_ty.clone(), else_block.span);
        }
        return result_ty;
    }

    if !else_ty.is_option() && then_is_nil {
        let result_ty = Type::option_of(else_ty);
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

pub(super) fn is_if_without_else(expr: &ExprNode) -> bool {
    match &expr.node.kind {
        ExprKind::If(if_node) => if_node.node.else_block.is_none(),
        _ => false,
    }
}

pub(super) fn block_always_diverges(block: &BlockNode) -> bool {
    if block.node.tail.is_some() {
        return false;
    }
    matches!(
        block.node.stmts.last().map(|s| &s.node),
        Some(Stmt::Return(_) | Stmt::Break | Stmt::Continue)
    )
}

pub(super) fn check_if_let(
    if_let_node: &IfLetNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &if_let_node.node;

    let value_ty = check_expr(&node.value, type_checker, errors, None);

    type_checker.push_scope();
    check_pattern(&node.pattern, &value_ty, false, type_checker, errors);
    let (then_ty, then_expr_id) = check_block_expr(&node.then_block, type_checker, errors, None);
    type_checker.pop_scope();

    let Some(else_block) = &node.else_block else {
        return Type::Void;
    };

    let (else_ty, else_expr_id) = check_block_expr(else_block, type_checker, errors, None);

    // unify branch types
    let same_ty = then_ty == else_ty;
    if same_ty {
        return then_ty;
    }

    let then_is_nil = then_ty.is_option_with_infer();
    let else_is_nil = else_ty.is_option_with_infer();

    if !then_ty.is_option() && else_is_nil {
        let result_ty = Type::option_of(then_ty);
        if let Some(id) = else_expr_id {
            type_checker.set_type(id, result_ty.clone(), else_block.span);
        }
        return result_ty;
    }

    if !else_ty.is_option() && then_is_nil {
        let result_ty = Type::option_of(else_ty);
        if let Some(id) = then_expr_id {
            type_checker.set_type(id, result_ty.clone(), node.then_block.span);
        }
        return result_ty;
    }

    let is_unifiable = contains_infer(&then_ty) || contains_infer(&else_ty);
    if !is_unifiable {
        errors.push(TypeErr::new(
            if_let_node.span,
            TypeErrKind::MismatchedTypes {
                expected: then_ty.clone(),
                found: else_ty.clone(),
            },
        ));
        return Type::Infer;
    }

    if contains_infer(&then_ty) {
        else_ty
    } else {
        then_ty
    }
}

pub(super) fn check_match(
    match_node: &MatchNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let scrutinee = &match_node.node.scrutinee;
    let scrutinee_ty = check_expr(scrutinee, type_checker, errors, None);

    match &scrutinee_ty {
        Type::Enum {
            name: enum_name, ..
        } => check_match_enum(match_node, &scrutinee_ty, *enum_name, type_checker, errors),
        Type::Bool => check_match_bool(match_node, &scrutinee_ty, type_checker, errors),
        Type::Int | Type::Float | Type::Double | Type::String => {
            check_match_scalar(match_node, &scrutinee_ty, type_checker, errors)
        }
        Type::Tuple(_) | Type::NamedTuple(_) => {
            check_match_tuple(match_node, &scrutinee_ty, type_checker, errors)
        }
        Type::Infer => Type::Infer,
        _ => {
            errors.push(TypeErr::new(
                scrutinee.span,
                TypeErrKind::UnsupportedMatchScrutinee {
                    found: scrutinee_ty.clone(),
                },
            ));
            Type::Infer
        }
    }
}

fn check_match_enum(
    match_node: &MatchNode,
    scrutinee_ty: &Type,
    enum_name: Ident,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let scrutinee = &match_node.node.scrutinee;

    let Some(enum_def) = type_checker.get_enum(enum_name) else {
        errors.push(TypeErr::new(
            scrutinee.span,
            TypeErrKind::UnknownEnum { name: enum_name },
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
            scrutinee_ty,
            &enum_def,
            &mut covered_variants,
            &mut has_wildcard,
            type_checker,
            errors,
        );

        let arm_ty = check_expr(&arm.node.body, type_checker, errors, None);
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

fn check_match_bool(
    match_node: &MatchNode,
    scrutinee_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let mut has_true = false;
    let mut has_false = false;
    let mut has_wildcard = false;
    let mut arm_types: Vec<Type> = vec![];

    for arm in &match_node.node.arms {
        type_checker.push_scope();

        match &arm.node.pattern.node {
            Pattern::Lit(Lit::Bool(true)) => has_true = true,
            Pattern::Lit(Lit::Bool(false)) => has_false = true,
            Pattern::Ident(name) => {
                if type_checker.get_const(*name).is_none() {
                    has_wildcard = true;
                }
            }
            Pattern::Wildcard | Pattern::VarIdent(_) => has_wildcard = true,
            Pattern::Or(alternatives) => {
                for alt in alternatives {
                    match &alt.node {
                        Pattern::Lit(Lit::Bool(true)) => has_true = true,
                        Pattern::Lit(Lit::Bool(false)) => has_false = true,
                        Pattern::Wildcard | Pattern::VarIdent(_) => has_wildcard = true,
                        Pattern::Ident(name) => {
                            if type_checker.get_const(*name).is_none() {
                                has_wildcard = true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        check_pattern_in_match(&arm.node.pattern, scrutinee_ty, type_checker, errors);

        let arm_ty = check_expr(&arm.node.body, type_checker, errors, None);
        arm_types.push(arm_ty);

        type_checker.pop_scope();
    }

    let exhaustive = (has_true && has_false) || has_wildcard;
    if !exhaustive {
        let mut missing = vec![];
        if !has_true {
            missing.push(Ident(Intern::new("true".to_string())));
        }
        if !has_false {
            missing.push(Ident(Intern::new("false".to_string())));
        }
        errors.push(TypeErr::new(
            match_node.span,
            TypeErrKind::NonExhaustiveMatch { missing },
        ));
    }

    unify_arm_types(&arm_types, match_node.span, errors)
}

fn check_match_scalar(
    match_node: &MatchNode,
    scrutinee_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let mut has_wildcard = false;
    let mut arm_types: Vec<Type> = vec![];

    for arm in &match_node.node.arms {
        type_checker.push_scope();

        match &arm.node.pattern.node {
            Pattern::Ident(name) => {
                if type_checker.get_const(*name).is_none() {
                    has_wildcard = true;
                }
            }
            Pattern::Wildcard | Pattern::VarIdent(_) => has_wildcard = true,
            _ => {}
        }

        check_pattern_in_match(&arm.node.pattern, scrutinee_ty, type_checker, errors);

        let arm_ty = check_expr(&arm.node.body, type_checker, errors, None);
        arm_types.push(arm_ty);

        type_checker.pop_scope();
    }

    if !has_wildcard {
        errors.push(TypeErr::new(
            match_node.span,
            TypeErrKind::NonExhaustiveMatchNoCatchAll,
        ));
    }

    unify_arm_types(&arm_types, match_node.span, errors)
}

fn check_match_tuple(
    match_node: &MatchNode,
    scrutinee_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let mut has_wildcard = false;
    let mut arm_types: Vec<Type> = vec![];

    for arm in &match_node.node.arms {
        type_checker.push_scope();

        match &arm.node.pattern.node {
            Pattern::Ident(name) => {
                if type_checker.get_const(*name).is_none() {
                    has_wildcard = true;
                }
            }
            Pattern::Wildcard | Pattern::VarIdent(_) => has_wildcard = true,
            _ => {}
        }

        check_pattern_in_match(&arm.node.pattern, scrutinee_ty, type_checker, errors);

        let arm_ty = check_expr(&arm.node.body, type_checker, errors, None);
        arm_types.push(arm_ty);

        type_checker.pop_scope();
    }

    if !has_wildcard {
        errors.push(TypeErr::new(
            match_node.span,
            TypeErrKind::NonExhaustiveMatchNoCatchAll,
        ));
    }

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
