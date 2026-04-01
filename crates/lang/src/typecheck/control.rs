use std::collections::HashSet;

use internment::Intern;

use super::{
    error::{Diagnostic, DiagnosticKind},
    expr::check_expr,
    pattern::{
        check_match_pattern, check_pattern, check_pattern_in_match, pattern_has_var_binding,
    },
    stmt::{check_block_expr, check_block_stmts},
    types::{EnumDef, TypeChecker},
    unify::{contains_infer, unify_types},
};
use crate::{
    ast::{
        self, BlockNode, ExprId, ExprKind, ExprNode, ForNode, Ident, IfLetNode, IfNode, Lit,
        MatchNode, Pattern, Stmt, Type, WhileLetNode, WhileNode,
    },
    span::Span,
};

fn check_bare_catchall_on_optional(
    pattern: &ast::PatternNode,
    value_ty: &Type,
    keyword: &str,
    errors: &mut Vec<Diagnostic>,
) {
    if !value_ty.is_option() {
        return;
    }
    if let Pattern::Ident(name) | Pattern::VarIdent(name) = &pattern.node {
        errors.push(
            Diagnostic::new(
                pattern.span,
                DiagnosticKind::BareCatchAllOnOptional { pattern_name: *name },
            )
            .with_help(format!(
                "use '{keyword} {name}? = ...' to unwrap the optional, or 'let {name} = ...' if you want the full optional type",
            )),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn unify_branch_types(
    then_ty: Type,
    else_ty: Type,
    then_expr_id: Option<ExprId>,
    else_expr_id: Option<ExprId>,
    then_span: Span,
    else_span: Span,
    parent_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    if then_ty == else_ty {
        return then_ty;
    }

    let then_is_nil = then_ty.is_option_with_infer();
    let else_is_nil = else_ty.is_option_with_infer();

    if !then_ty.is_option() && else_is_nil {
        let result_ty = Type::option_of(then_ty);
        if let Some(id) = else_expr_id {
            type_checker.set_type(id, result_ty.clone(), else_span);
        }
        return result_ty;
    }

    if !else_ty.is_option() && then_is_nil {
        let result_ty = Type::option_of(else_ty);
        if let Some(id) = then_expr_id {
            type_checker.set_type(id, result_ty.clone(), then_span);
        }
        return result_ty;
    }

    let is_unifiable = contains_infer(&then_ty) || contains_infer(&else_ty);
    if !is_unifiable {
        errors.push(Diagnostic::new(
            parent_span,
            DiagnosticKind::MismatchedTypes {
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

fn validate_var_scrutinee(
    scrutinee: &ExprNode,
    type_checker: &TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> bool {
    if let ExprKind::Ident(name) = &scrutinee.node.kind {
        match type_checker.get_var(*name) {
            Some(info) if info.mutable => true,
            _ => {
                errors.push(
                    Diagnostic::new(scrutinee.span, DiagnosticKind::VarPatternOnImmutable)
                        .with_help("declare the scrutinee with 'var' to allow write-through"),
                );
                false
            }
        }
    } else {
        errors.push(
            Diagnostic::new(scrutinee.span, DiagnosticKind::VarPatternOnImmutable)
                .with_help("var binding in pattern requires a simple variable as scrutinee"),
        );
        false
    }
}

fn pattern_is_catch_all(pattern: &Pattern, type_checker: &TypeChecker) -> bool {
    match pattern {
        Pattern::Wildcard | Pattern::VarIdent(_) => true,
        Pattern::Ident(name) => type_checker.get_const(*name).is_none(),
        Pattern::Or(alts) => alts
            .iter()
            .any(|alt| pattern_is_catch_all(&alt.node, type_checker)),
        _ => false,
    }
}

fn collect_bool_coverage(pattern: &Pattern, has_true: &mut bool, has_false: &mut bool) {
    match pattern {
        Pattern::Lit(Lit::Bool(true)) => *has_true = true,
        Pattern::Lit(Lit::Bool(false)) => *has_false = true,
        Pattern::Or(alts) => {
            for alt in alts {
                collect_bool_coverage(&alt.node, has_true, has_false);
            }
        }
        _ => {}
    }
}

pub(super) fn check_while(
    while_node: &WhileNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let node = &while_node.node;
    let cond_ty = check_expr(&node.cond, type_checker, errors, None);
    let maybe_bool = cond_ty.is_bool() || cond_ty.is_infer();
    if !maybe_bool {
        errors.push(Diagnostic::new(
            node.cond.span,
            DiagnosticKind::WhileConditionNotBool {
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
    errors: &mut Vec<Diagnostic>,
) {
    let node = &while_let_node.node;
    let value_ty = check_expr(&node.value, type_checker, errors, None);

    check_bare_catchall_on_optional(&node.pattern, &value_ty, "while let", errors);

    let mutable = if pattern_has_var_binding(&node.pattern) {
        validate_var_scrutinee(&node.value, type_checker, errors)
    } else {
        false
    };

    type_checker.push_scope();
    check_pattern(&node.pattern, &value_ty, mutable, type_checker, errors);

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
    errors: &mut Vec<Diagnostic>,
) {
    let node = &for_node.node;

    let iterable_ty = check_expr(&node.iterable, type_checker, errors, None);
    let item_ty = extract_iterable_item_type(&iterable_ty, for_node.span, errors);

    let is_map = iterable_ty.is_map();

    if let Some(ref step_expr) = node.step {
        let step_ty = check_expr(step_expr, type_checker, errors, None);

        if is_map {
            errors.push(Diagnostic::new(
                step_expr.span,
                DiagnosticKind::ForMapStepNotAllowed,
            ));
        } else {
            let is_seq = is_sequence_type(&iterable_ty);
            let step_is_int = matches!(step_ty, Type::Int | Type::Infer);

            if is_seq {
                if !step_is_int {
                    errors.push(Diagnostic::new(
                        step_expr.span,
                        DiagnosticKind::ForStepNotInt {
                            item_ty: item_ty.clone(),
                            step_ty,
                        },
                    ));
                }
            } else {
                let item_is_int = matches!(item_ty, Type::Int | Type::Infer);
                if !item_is_int || !step_is_int {
                    errors.push(Diagnostic::new(
                        step_expr.span,
                        DiagnosticKind::ForStepNotInt {
                            item_ty: item_ty.clone(),
                            step_ty,
                        },
                    ));
                }
            }
        }
    }

    if is_map && node.reversed {
        errors.push(Diagnostic::new(
            for_node.span,
            DiagnosticKind::ForMapRevNotAllowed,
        ));
    }

    let is_range_from =
        matches!(&iterable_ty, Type::Struct { name, .. } if name.0.as_ref() == "RangeFrom");
    if is_range_from && node.reversed {
        errors.push(Diagnostic::new(
            for_node.span,
            DiagnosticKind::ForRangeFromRevNotAllowed,
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

pub(super) fn extract_iterable_item_type(
    ty: &Type,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    match ty {
        Type::Struct { name, type_args } => {
            let name_str = name.0.as_ref();
            let is_range =
                name_str == "Range" || name_str == "RangeInclusive" || name_str == "RangeFrom";
            if is_range && type_args.len() == 1 {
                return type_args[0].clone();
            }
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::ForIterableNotSupported { found: ty.clone() },
            ));
            Type::Infer
        }
        Type::Array { elem, .. } => *elem.clone(),
        Type::List { elem } => *elem.clone(),
        Type::ArrayView { elem } => *elem.clone(),
        Type::Map { key, value } => Type::Tuple(vec![*key.clone(), *value.clone()]),
        Type::Infer => Type::Infer,
        _ => {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::ForIterableNotSupported { found: ty.clone() },
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
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let node = &if_node.node;

    let cond_ty = check_expr(&node.cond, type_checker, errors, None);
    let maybe_bool = cond_ty.is_bool() || cond_ty.is_infer();
    if !maybe_bool {
        errors.push(Diagnostic::new(
            node.cond.span,
            DiagnosticKind::IfConditionNotBool { found: cond_ty },
        ));
    }

    let (then_ty, then_expr_id) = check_block_expr(&node.then_block, type_checker, errors, None);

    // if there is no else block then the type is void and this must be a statment
    let Some(else_block) = &node.else_block else {
        return Type::Void;
    };

    let (else_ty, else_expr_id) = check_block_expr(else_block, type_checker, errors, None);

    unify_branch_types(
        then_ty,
        else_ty,
        then_expr_id,
        else_expr_id,
        node.then_block.span,
        else_block.span,
        if_node.span,
        type_checker,
        errors,
    )
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
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let node = &if_let_node.node;

    let value_ty = check_expr(&node.value, type_checker, errors, None);

    check_bare_catchall_on_optional(&node.pattern, &value_ty, "if let", errors);

    let mutable = if pattern_has_var_binding(&node.pattern) {
        validate_var_scrutinee(&node.value, type_checker, errors)
    } else {
        false
    };

    type_checker.push_scope();
    check_pattern(&node.pattern, &value_ty, mutable, type_checker, errors);
    let (then_ty, then_expr_id) = check_block_expr(&node.then_block, type_checker, errors, None);
    type_checker.pop_scope();

    let Some(else_block) = &node.else_block else {
        return Type::Void;
    };

    let (else_ty, else_expr_id) = check_block_expr(else_block, type_checker, errors, None);

    unify_branch_types(
        then_ty,
        else_ty,
        then_expr_id,
        else_expr_id,
        node.then_block.span,
        else_block.span,
        if_let_node.span,
        type_checker,
        errors,
    )
}

pub(super) fn check_match(
    match_node: &MatchNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let scrutinee = &match_node.node.scrutinee;
    let scrutinee_ty = check_expr(scrutinee, type_checker, errors, None);

    let any_var = match_node
        .node
        .arms
        .iter()
        .any(|arm| pattern_has_var_binding(&arm.node.pattern));
    if any_var {
        validate_var_scrutinee(scrutinee, type_checker, errors);
    }

    match &scrutinee_ty {
        Type::Enum {
            name: enum_name, ..
        } => check_match_enum(match_node, &scrutinee_ty, *enum_name, type_checker, errors),
        Type::Bool
        | Type::Int
        | Type::Float
        | Type::Double
        | Type::String
        | Type::Tuple(_)
        | Type::NamedTuple(_) => {
            check_match_non_enum(match_node, &scrutinee_ty, type_checker, errors)
        }
        Type::Infer => Type::Infer,
        _ => {
            errors.push(Diagnostic::new(
                scrutinee.span,
                DiagnosticKind::UnsupportedMatchScrutinee {
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
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let scrutinee = &match_node.node.scrutinee;

    let Some(enum_def) = type_checker.get_enum(enum_name) else {
        errors.push(Diagnostic::new(
            scrutinee.span,
            DiagnosticKind::UnknownEnum { name: enum_name },
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

fn check_match_non_enum(
    match_node: &MatchNode,
    scrutinee_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let is_bool = scrutinee_ty.is_bool();
    let mut has_wildcard = false;
    let mut has_true = false;
    let mut has_false = false;
    let mut arm_types: Vec<Type> = vec![];

    for arm in &match_node.node.arms {
        type_checker.push_scope();

        has_wildcard |= pattern_is_catch_all(&arm.node.pattern.node, type_checker);
        if is_bool {
            collect_bool_coverage(&arm.node.pattern.node, &mut has_true, &mut has_false);
        }

        check_pattern_in_match(&arm.node.pattern, scrutinee_ty, type_checker, errors);
        let arm_ty = check_expr(&arm.node.body, type_checker, errors, None);
        arm_types.push(arm_ty);

        type_checker.pop_scope();
    }

    if is_bool {
        let exhaustive = (has_true && has_false) || has_wildcard;
        if !exhaustive {
            let mut missing = vec![];
            if !has_true {
                missing.push(Ident(Intern::new("true".to_string())));
            }
            if !has_false {
                missing.push(Ident(Intern::new("false".to_string())));
            }
            errors.push(Diagnostic::new(
                match_node.span,
                DiagnosticKind::NonExhaustiveMatch { missing },
            ));
        }
    } else if !has_wildcard {
        errors.push(Diagnostic::new(
            match_node.span,
            DiagnosticKind::NonExhaustiveMatchNoCatchAll,
        ));
    }

    unify_arm_types(&arm_types, match_node.span, errors)
}

fn check_exhaustiveness(
    enum_def: &EnumDef,
    covered: &HashSet<Ident>,
    has_wildcard: bool,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) {
    if has_wildcard {
        return;
    }

    let all_variants: HashSet<Ident> = enum_def.variants.iter().map(|v| v.name).collect();
    let missing: Vec<Ident> = all_variants.difference(covered).copied().collect();
    if !missing.is_empty() {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::NonExhaustiveMatch { missing },
        ));
    }
}

fn unify_arm_types(arm_types: &[Type], span: Span, errors: &mut Vec<Diagnostic>) -> Type {
    if arm_types.is_empty() {
        return Type::Void;
    }

    let mut result = arm_types[0].clone();
    for ty in arm_types.iter().skip(1) {
        let mut unify_errors = vec![];
        match unify_types(&result, ty, span, &mut unify_errors) {
            Some(unified) => result = unified,
            None => {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::MatchArmTypeMismatch {
                        expected: result.clone(),
                        found: ty.clone(),
                    },
                ));
            }
        }
    }
    result
}
