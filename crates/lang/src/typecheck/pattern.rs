use std::collections::{HashMap, HashSet};

use internment::Intern;

use super::{
    composite::validate_field_names,
    error::{Diagnostic, DiagnosticKind},
    expr::type_from_lit,
    infer::{build_subst, subst_type},
    types::{EnumDef, TypeChecker},
};
use crate::{
    ast::{FloatSuffix, Ident, Lit, Pattern, PatternNode, Type, VariantKind},
    span::Span,
};

fn type_from_pattern_lit(lit: &Lit) -> Type {
    match lit {
        Lit::Int(_) => Type::Int,
        Lit::Float { suffix, .. } => match suffix {
            Some(FloatSuffix::F) | None => Type::Float,
            Some(FloatSuffix::D) => Type::Double,
        },
        Lit::Bool(_) => Type::Bool,
        Lit::String(_) => Type::String,
        Lit::Nil => Type::Void,
    }
}

fn check_ident_pattern(
    span: Span,
    name: Ident,
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> bool {
    if in_match && let Some(const_def) = type_checker.get_const(name) {
        let const_ty = const_def.ty.clone();
        let const_val = const_def.value.clone();
        if const_ty != *value_ty && !value_ty.is_infer() {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::InvalidLiteralPattern {
                    expected: value_ty.clone(),
                    found: const_ty,
                },
            ));
        }
        let span_key = (span.start, span.end);
        type_checker
            .const_pattern_values
            .insert(span_key, const_val);
        return false;
    }
    type_checker.set_var(name, value_ty.clone(), mutable);
    true
}

fn check_tuple_pattern(
    pattern: &PatternNode,
    subpatterns: &[PatternNode],
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let Some(elem_types) = value_ty.tuple_element_types() else {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::NonTupleInTuplePattern {
                found: value_ty.clone(),
                pattern_arity: subpatterns.len(),
            },
        ));
        return;
    };

    let same_arity = subpatterns.len() == elem_types.len();
    if !same_arity {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::TuplePatternArityMismatch {
                expected: elem_types.len(),
                found: subpatterns.len(),
            },
        ));
        return;
    }

    for (subpat, elem_ty) in subpatterns.iter().zip(elem_types.iter()) {
        check_pattern_inner(
            subpat,
            elem_ty,
            mutable,
            in_match,
            None,
            type_checker,
            errors,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_named_tuple_pattern(
    pattern: &PatternNode,
    elems: &[(Ident, PatternNode)],
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let Some(elem_types) = value_ty.tuple_element_types() else {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::NonTupleInTuplePattern {
                found: value_ty.clone(),
                pattern_arity: elems.len(),
            },
        ));
        return;
    };

    let value_labels: Option<Vec<Ident>> = match value_ty {
        Type::NamedTuple(fields) => Some(fields.iter().map(|(name, _)| *name).collect()),
        _ => None,
    };

    let same_arity = elems.len() == elem_types.len();
    if !same_arity {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::TuplePatternArityMismatch {
                expected: elem_types.len(),
                found: elems.len(),
            },
        ));
        return;
    }

    let Some(labels) = value_labels else {
        errors.push(
            Diagnostic::new(pattern.span, DiagnosticKind::NamedPatternOnPositionalTuple)
                .with_help("use positional pattern `(a, b, ...)` instead"),
        );
        return;
    };

    for ((pat_label, _), ty_label) in elems.iter().zip(labels.iter()) {
        if *pat_label != *ty_label {
            errors.push(Diagnostic::new(
                pattern.span,
                DiagnosticKind::TuplePatternLabelMismatch {
                    expected: *ty_label,
                    found: *pat_label,
                },
            ));
        }
    }

    for ((_, subpat), elem_ty) in elems.iter().zip(elem_types.iter()) {
        check_pattern_inner(
            subpat,
            elem_ty,
            mutable,
            in_match,
            None,
            type_checker,
            errors,
        );
    }
}

fn check_literal_pattern(span: Span, lit: &Lit, value_ty: &Type, errors: &mut Vec<Diagnostic>) {
    let is_valid = matches!(lit, Lit::Int(_) | Lit::Bool(_) | Lit::String(_));
    if !is_valid {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::InvalidLiteralPattern {
                expected: value_ty.clone(),
                found: type_from_pattern_lit(lit),
            },
        ));
        return;
    }
    let lit_ty = type_from_lit(lit);
    if lit_ty != *value_ty && !value_ty.is_infer() {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::InvalidLiteralPattern {
                expected: value_ty.clone(),
                found: lit_ty,
            },
        ));
    }
}

fn check_range_pattern(
    span: Span,
    start: Option<&Lit>,
    end: Option<&Lit>,
    inclusive: bool,
    value_ty: &Type,
    errors: &mut Vec<Diagnostic>,
) {
    match (start, end) {
        (Some(start_lit), Some(end_lit)) => {
            let start_ty = type_from_pattern_lit(start_lit);
            let end_ty = type_from_pattern_lit(end_lit);

            if start_ty != end_ty {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::RangePatternBoundTypeMismatch {
                        start: start_ty,
                        end: end_ty,
                    },
                ));
                return;
            }

            if !start_ty.is_num() {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::NonNumericRangePattern { found: start_ty },
                ));
                return;
            }

            if start_ty != *value_ty && !value_ty.is_infer() {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::InvalidLiteralPattern {
                        expected: value_ty.clone(),
                        found: start_ty,
                    },
                ));
                return;
            }

            match (start_lit, end_lit) {
                (Lit::Int(s), Lit::Int(e)) => {
                    let empty = if inclusive { s > e } else { s >= e };
                    if empty {
                        errors.push(Diagnostic::new(span, DiagnosticKind::EmptyRangePattern));
                    }
                }
                (Lit::Float { value: s, .. }, Lit::Float { value: e, .. }) => {
                    let empty = if inclusive { s > e } else { s >= e };
                    if empty {
                        errors.push(Diagnostic::new(span, DiagnosticKind::EmptyRangePattern));
                    }
                }
                _ => {}
            }
        }
        (Some(bound_lit), None) | (None, Some(bound_lit)) => {
            let bound_ty = type_from_pattern_lit(bound_lit);

            if !bound_ty.is_num() {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::NonNumericRangePattern { found: bound_ty },
                ));
                return;
            }

            if bound_ty != *value_ty && !value_ty.is_infer() {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::InvalidLiteralPattern {
                        expected: value_ty.clone(),
                        found: bound_ty,
                    },
                ));
            }
        }
        (None, None) => {}
    }
}

fn check_optional_pattern(
    pattern: &PatternNode,
    inner: &PatternNode,
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    if !value_ty.is_option() && !value_ty.is_infer() {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::OptionalPatternOnNonOptional {
                found: value_ty.clone(),
            },
        ));
        return;
    }

    if matches!(&inner.node, Pattern::Optional(_)) {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::NestedOptionalPattern,
        ));
        return;
    }

    let inner_ty = value_ty.option_inner().cloned().unwrap_or(Type::Infer);
    check_pattern_inner(
        inner,
        &inner_ty,
        mutable,
        in_match,
        None,
        type_checker,
        errors,
    );
}

#[allow(clippy::too_many_arguments)]
fn check_or_pattern(
    alternatives: &[PatternNode],
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    match_ctx: &mut Option<(&EnumDef, &mut HashSet<Ident>, &mut bool)>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let mut first_bindings: Option<Vec<(Ident, Type, bool)>> = None;

    for alt in alternatives {
        let inner_ctx = match_ctx
            .as_mut()
            .map(|(def, cov, wc)| (*def, &mut **cov, &mut **wc));
        type_checker.push_scope();
        check_pattern_inner(
            alt,
            value_ty,
            mutable,
            in_match,
            inner_ctx,
            type_checker,
            errors,
        );
        let scope_bindings = type_checker.collect_current_scope_bindings();
        type_checker.pop_scope();
        match &first_bindings {
            None => first_bindings = Some(scope_bindings),
            Some(first) => {
                validate_or_bindings(first, &scope_bindings, alt.span, errors);
            }
        }
    }

    if let Some(bindings) = first_bindings {
        for (name, ty, is_mutable) in bindings {
            type_checker.set_var(name, ty, is_mutable);
        }
    }
}

pub(super) fn check_pattern(
    pattern: &PatternNode,
    value_ty: &Type,
    mutable: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    check_pattern_inner(
        pattern,
        value_ty,
        mutable,
        false,
        None,
        type_checker,
        errors,
    );
}

pub(super) fn check_pattern_in_match(
    pattern: &PatternNode,
    value_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    check_pattern_inner(pattern, value_ty, false, true, None, type_checker, errors);
}

pub(super) fn pattern_has_var_binding(pattern: &PatternNode) -> bool {
    match &pattern.node {
        Pattern::VarIdent(_) => true,
        Pattern::Tuple(pats) | Pattern::Or(pats) => pats.iter().any(pattern_has_var_binding),
        Pattern::NamedTuple(fields) => fields.iter().any(|(_, p)| pattern_has_var_binding(p)),
        Pattern::EnumTuple { fields, .. } | Pattern::InferredEnumTuple { fields, .. } => {
            fields.iter().any(pattern_has_var_binding)
        }
        Pattern::EnumStruct { fields, .. }
        | Pattern::InferredEnumStruct { fields, .. }
        | Pattern::Struct { fields, .. } => fields.iter().any(|(_, p)| pattern_has_var_binding(p)),
        Pattern::Optional(inner) => pattern_has_var_binding(inner),
        Pattern::Ident(_)
        | Pattern::Wildcard
        | Pattern::EnumUnit { .. }
        | Pattern::InferredEnumUnit { .. }
        | Pattern::Range { .. }
        | Pattern::Lit(_)
        | Pattern::Rest
        | Pattern::Nil => false,
    }
}

#[allow(clippy::match_same_arms)] // refutable arms are scattered with complex recursive arms in between; regrouping hurts readability
pub(super) fn is_refutable(pattern: &Pattern, value_ty: &Type, type_checker: &TypeChecker) -> bool {
    match pattern {
        Pattern::Wildcard | Pattern::Ident(_) | Pattern::VarIdent(_) => false,
        Pattern::Lit(_) => true,
        Pattern::Tuple(subs) => {
            let Some(elem_types) = value_ty.tuple_element_types() else {
                return true;
            };
            if subs.len() != elem_types.len() {
                return true;
            }
            subs.iter()
                .zip(elem_types.iter())
                .any(|(sub, ty)| is_refutable(&sub.node, ty, type_checker))
        }
        Pattern::EnumUnit { qualifier, .. } => {
            let Some(enum_def) = type_checker.get_enum(*qualifier) else {
                return true;
            };
            enum_def.variants.len() > 1
        }
        Pattern::EnumTuple {
            qualifier, fields, ..
        } => {
            let Some(enum_def) = type_checker.get_enum(*qualifier) else {
                return true;
            };
            if enum_def.variants.len() > 1 {
                return true;
            }
            let Type::Enum { type_args, .. } = value_ty else {
                return true;
            };
            let enum_def = enum_def.clone();
            let Some(variant_def) = enum_def.variants.first() else {
                return true;
            };
            let VariantKind::Tuple(expected_types) = &variant_def.kind else {
                return true;
            };
            if fields.len() != expected_types.len() {
                return true;
            }
            let subst = build_subst(&enum_def.type_params, type_args);
            fields
                .iter()
                .zip(expected_types.iter())
                .any(|(subpat, expected_ty)| {
                    let resolved_ty = subst_type(expected_ty, &subst, &HashMap::new());
                    is_refutable(&subpat.node, &resolved_ty, type_checker)
                })
        }
        Pattern::EnumStruct {
            qualifier, fields, ..
        } => {
            let Some(enum_def) = type_checker.get_enum(*qualifier) else {
                return true;
            };
            if enum_def.variants.len() > 1 {
                return true;
            }
            let Type::Enum { type_args, .. } = value_ty else {
                return true;
            };
            let enum_def = enum_def.clone();
            let Some(variant_def) = enum_def.variants.first() else {
                return true;
            };
            let VariantKind::Struct(expected_fields) = &variant_def.kind else {
                return true;
            };
            let subst = build_subst(&enum_def.type_params, type_args);
            fields.iter().any(|(name, subpat)| {
                let Some(field_def) = expected_fields.iter().find(|f| f.name == *name) else {
                    return true;
                };
                let resolved_ty = subst_type(&field_def.ty, &subst, &HashMap::new());
                is_refutable(&subpat.node, &resolved_ty, type_checker)
            })
        }
        Pattern::InferredEnumUnit { .. } => {
            let Type::Enum { name, .. } = value_ty else {
                return true;
            };
            let Some(enum_def) = type_checker.get_enum(*name) else {
                return true;
            };
            enum_def.variants.len() > 1
        }
        Pattern::InferredEnumTuple { fields, .. } => {
            let Type::Enum { name, type_args } = value_ty else {
                return true;
            };
            let Some(enum_def) = type_checker.get_enum(*name) else {
                return true;
            };
            if enum_def.variants.len() > 1 {
                return true;
            }
            let enum_def = enum_def.clone();
            let Some(variant_def) = enum_def.variants.first() else {
                return true;
            };
            let VariantKind::Tuple(expected_types) = &variant_def.kind else {
                return true;
            };
            if fields.len() != expected_types.len() {
                return true;
            }
            let subst = build_subst(&enum_def.type_params, type_args);
            fields
                .iter()
                .zip(expected_types.iter())
                .any(|(subpat, expected_ty)| {
                    let resolved_ty = subst_type(expected_ty, &subst, &HashMap::new());
                    is_refutable(&subpat.node, &resolved_ty, type_checker)
                })
        }
        Pattern::InferredEnumStruct { fields, .. } => {
            let Type::Enum { name, type_args } = value_ty else {
                return true;
            };
            let Some(enum_def) = type_checker.get_enum(*name) else {
                return true;
            };
            if enum_def.variants.len() > 1 {
                return true;
            }
            let enum_def = enum_def.clone();
            let Some(variant_def) = enum_def.variants.first() else {
                return true;
            };
            let VariantKind::Struct(expected_fields) = &variant_def.kind else {
                return true;
            };
            let subst = build_subst(&enum_def.type_params, type_args);
            fields.iter().any(|(name, subpat)| {
                let Some(field_def) = expected_fields.iter().find(|f| f.name == *name) else {
                    return true;
                };
                let resolved_ty = subst_type(&field_def.ty, &subst, &HashMap::new());
                is_refutable(&subpat.node, &resolved_ty, type_checker)
            })
        }
        Pattern::Nil => true,
        Pattern::Optional(_) => true,
        Pattern::Or(subs) => subs
            .iter()
            .all(|s| is_refutable(&s.node, value_ty, type_checker)),
        Pattern::NamedTuple(_) | Pattern::Struct { .. } | Pattern::Range { .. } | Pattern::Rest => {
            true
        }
    }
}

pub(super) fn check_match_pattern(
    pattern: &PatternNode,
    scrutinee_ty: &Type,
    enum_def: &EnumDef,
    covered_variants: &mut HashSet<Ident>,
    has_wildcard: &mut bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    check_pattern_inner(
        pattern,
        scrutinee_ty,
        false,
        true,
        Some((enum_def, covered_variants, has_wildcard)),
        type_checker,
        errors,
    );
}

fn validate_or_bindings(
    first: &[(Ident, Type, bool)],
    current: &[(Ident, Type, bool)],
    span: Span,
    errors: &mut Vec<Diagnostic>,
) {
    let first_names: HashSet<Ident> = first.iter().map(|(name, _, _)| *name).collect();
    let current_names: HashSet<Ident> = current.iter().map(|(name, _, _)| *name).collect();

    if first_names != current_names {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::OrPatternBindingMismatch,
        ));
        return;
    }

    for (name, ty, _) in current {
        let Some((_, expected_ty, _)) = first.iter().find(|(n, _, _)| n == name) else {
            continue;
        };
        if ty != expected_ty {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::OrPatternTypeMismatch {
                    name: *name,
                    expected: expected_ty.clone(),
                    found: ty.clone(),
                },
            ));
        }
    }
}

fn check_pattern_inner(
    pattern: &PatternNode,
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    mut match_ctx: Option<(&EnumDef, &mut HashSet<Ident>, &mut bool)>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    match &pattern.node {
        Pattern::Ident(name) => {
            let is_binding = check_ident_pattern(
                pattern.span,
                *name,
                value_ty,
                mutable,
                in_match,
                type_checker,
                errors,
            );
            if is_binding && let Some((_, _, has_wildcard)) = &mut match_ctx {
                **has_wildcard = true;
            }
        }
        Pattern::Wildcard => {
            if let Some((_, _, has_wildcard)) = match_ctx {
                *has_wildcard = true;
            }
        }
        Pattern::Tuple(subpatterns) => {
            check_tuple_pattern(
                pattern,
                subpatterns,
                value_ty,
                mutable,
                in_match,
                type_checker,
                errors,
            );
        }
        Pattern::NamedTuple(elems) => {
            check_named_tuple_pattern(
                pattern,
                elems,
                value_ty,
                mutable,
                in_match,
                type_checker,
                errors,
            );
        }
        Pattern::Struct { name, fields } => {
            check_struct_destructure_pattern(
                pattern,
                *name,
                fields,
                value_ty,
                mutable,
                in_match,
                type_checker,
                errors,
            );
        }
        Pattern::EnumUnit { qualifier, variant } => {
            check_enum_pattern(
                pattern,
                *qualifier,
                *variant,
                &[],
                value_ty,
                mutable,
                in_match,
                match_ctx,
                type_checker,
                errors,
            );
        }
        Pattern::EnumTuple {
            qualifier,
            variant,
            fields,
        } => {
            check_enum_pattern(
                pattern,
                *qualifier,
                *variant,
                fields,
                value_ty,
                mutable,
                in_match,
                match_ctx,
                type_checker,
                errors,
            );
        }
        Pattern::EnumStruct {
            qualifier,
            variant,
            fields,
            has_rest,
        } => {
            check_enum_struct_pattern(
                pattern,
                *qualifier,
                *variant,
                fields,
                *has_rest,
                value_ty,
                mutable,
                in_match,
                match_ctx,
                type_checker,
                errors,
            );
        }
        Pattern::InferredEnumUnit { variant } => {
            let Some(enum_name) =
                resolve_inferred_enum_name(pattern.span, *variant, value_ty, errors)
            else {
                return;
            };
            check_enum_pattern(
                pattern,
                enum_name,
                *variant,
                &[],
                value_ty,
                mutable,
                in_match,
                match_ctx,
                type_checker,
                errors,
            );
        }
        Pattern::InferredEnumTuple { variant, fields } => {
            let Some(enum_name) =
                resolve_inferred_enum_name(pattern.span, *variant, value_ty, errors)
            else {
                return;
            };
            check_enum_pattern(
                pattern,
                enum_name,
                *variant,
                fields,
                value_ty,
                mutable,
                in_match,
                match_ctx,
                type_checker,
                errors,
            );
        }
        Pattern::InferredEnumStruct {
            variant,
            fields,
            has_rest,
        } => {
            let Some(enum_name) =
                resolve_inferred_enum_name(pattern.span, *variant, value_ty, errors)
            else {
                return;
            };
            check_enum_struct_pattern(
                pattern,
                enum_name,
                *variant,
                fields,
                *has_rest,
                value_ty,
                mutable,
                in_match,
                match_ctx,
                type_checker,
                errors,
            );
        }
        Pattern::Lit(lit) => {
            check_literal_pattern(pattern.span, lit, value_ty, errors);
        }
        Pattern::VarIdent(name) => {
            type_checker.set_var(*name, value_ty.clone(), true);
            if let Some((_, _, has_wildcard)) = match_ctx {
                *has_wildcard = true;
            }
        }
        Pattern::Rest => {
            // no-op, parent checks placement
        }
        Pattern::Nil => {
            if !value_ty.is_option() && !value_ty.is_infer() {
                errors.push(Diagnostic::new(
                    pattern.span,
                    DiagnosticKind::NilPatternOnNonOptional {
                        found: value_ty.clone(),
                    },
                ));
                return;
            }
            if let Some((_, covered_variants, _)) = match_ctx {
                let none_ident = Ident(Intern::new("None".to_string()));
                covered_variants.insert(none_ident);
            }
        }
        Pattern::Range {
            start,
            end,
            inclusive,
        } => {
            check_range_pattern(
                pattern.span,
                start.as_ref(),
                end.as_ref(),
                *inclusive,
                value_ty,
                errors,
            );
        }
        Pattern::Optional(inner) => {
            if let Some((_, covered_variants, _)) = &mut match_ctx {
                let some_ident = Ident(Intern::new("Some".to_string()));
                covered_variants.insert(some_ident);
            }
            check_optional_pattern(
                pattern,
                inner,
                value_ty,
                mutable,
                in_match,
                type_checker,
                errors,
            );
        }
        Pattern::Or(alternatives) => {
            check_or_pattern(
                alternatives,
                value_ty,
                mutable,
                in_match,
                &mut match_ctx,
                type_checker,
                errors,
            );
        }
    }
}

fn resolve_inferred_enum_name(
    span: Span,
    variant: Ident,
    value_ty: &Type,
    errors: &mut Vec<Diagnostic>,
) -> Option<Ident> {
    let Type::Enum { name, .. } = value_ty else {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::CannotInferEnumVariant { variant },
        ));
        return None;
    };
    Some(*name)
}

fn check_enum_preamble<'a>(
    pattern: &PatternNode,
    qualifier: Ident,
    variant_name: Ident,
    value_ty: &'a Type,
    type_checker: &TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<(&'a Vec<Type>, EnumDef)> {
    let Type::Enum {
        name: enum_name,
        type_args,
    } = value_ty
    else {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::MismatchedTypes {
                expected: value_ty.clone(),
                found: Type::Enum {
                    name: qualifier,
                    type_args: vec![],
                },
            },
        ));
        return None;
    };

    if qualifier != *enum_name {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::MatchPatternEnumMismatch {
                expected_enum: *enum_name,
                pattern_enum: qualifier,
            },
        ));
        return None;
    }

    let Some(enum_def) = type_checker.get_enum(qualifier) else {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::UnknownEnum { name: qualifier },
        ));
        return None;
    };
    let enum_def = enum_def.clone();

    if enum_def.variants.iter().all(|v| v.name != variant_name) {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::UnknownEnumVariant {
                enum_name: qualifier,
                variant_name,
            },
        ));
        return None;
    }

    Some((type_args, enum_def))
}

#[allow(clippy::too_many_arguments)]
fn check_enum_pattern(
    pattern: &PatternNode,
    qualifier: Ident,
    variant_name: Ident,
    fields: &[PatternNode],
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    match_ctx: Option<(&EnumDef, &mut HashSet<Ident>, &mut bool)>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let Some((type_args, enum_def)) = check_enum_preamble(
        pattern,
        qualifier,
        variant_name,
        value_ty,
        type_checker,
        errors,
    ) else {
        return;
    };

    if let Some((_, covered_variants, _)) = match_ctx {
        covered_variants.insert(variant_name);
    }

    let variant_def = enum_def
        .variants
        .iter()
        .find(|v| v.name == variant_name)
        .unwrap();

    match &variant_def.kind {
        VariantKind::Unit => {
            if !fields.is_empty() {
                errors.push(Diagnostic::new(
                    pattern.span,
                    DiagnosticKind::EnumVariantNotTuple {
                        enum_name: qualifier,
                        variant_name,
                    },
                ));
            }
        }
        VariantKind::Tuple(expected_types) => {
            if fields.len() != expected_types.len() {
                errors.push(Diagnostic::new(
                    pattern.span,
                    DiagnosticKind::EnumVariantArityMismatch {
                        enum_name: qualifier,
                        variant_name,
                        expected: expected_types.len(),
                        found: fields.len(),
                    },
                ));
                return;
            }

            let subst = build_subst(&enum_def.type_params, type_args);

            for (subpat, expected_ty) in fields.iter().zip(expected_types.iter()) {
                let resolved_ty =
                    type_checker.resolve_type(&subst_type(expected_ty, &subst, &HashMap::new()));
                check_pattern_inner(
                    subpat,
                    &resolved_ty,
                    mutable,
                    in_match,
                    None,
                    type_checker,
                    errors,
                );
            }
        }
        VariantKind::Struct(_) => {
            errors.push(Diagnostic::new(
                pattern.span,
                DiagnosticKind::EnumVariantNotTuple {
                    enum_name: qualifier,
                    variant_name,
                },
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_enum_struct_pattern(
    pattern: &PatternNode,
    qualifier: Ident,
    variant_name: Ident,
    fields: &[(Ident, PatternNode)],
    has_rest: bool,
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    match_ctx: Option<(&EnumDef, &mut HashSet<Ident>, &mut bool)>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let Some((type_args, enum_def)) = check_enum_preamble(
        pattern,
        qualifier,
        variant_name,
        value_ty,
        type_checker,
        errors,
    ) else {
        return;
    };

    if let Some((_, covered_variants, _)) = match_ctx {
        covered_variants.insert(variant_name);
    }

    let variant_def = enum_def
        .variants
        .iter()
        .find(|v| v.name == variant_name)
        .unwrap();

    let VariantKind::Struct(expected_fields) = &variant_def.kind else {
        errors.push(Diagnostic::new(
            pattern.span,
            DiagnosticKind::EnumVariantNotStruct {
                enum_name: qualifier,
                variant_name,
            },
        ));
        return;
    };

    let subst = build_subst(&enum_def.type_params, type_args);

    let provided: Vec<(Ident, _)> = fields.iter().map(|(n, _)| (*n, pattern.span)).collect();
    let matched = validate_field_names(
        &provided,
        pattern.span,
        expected_fields,
        has_rest,
        |field| DiagnosticKind::EnumVariantDuplicateField {
            enum_name: qualifier,
            variant_name,
            field,
        },
        |field| DiagnosticKind::EnumVariantUnknownField {
            enum_name: qualifier,
            variant_name,
            field,
        },
        |field| DiagnosticKind::EnumVariantMissingField {
            enum_name: qualifier,
            variant_name,
            field,
        },
        None,
        errors,
    );

    for ((_, subpat), matched_def) in fields.iter().zip(matched.iter()) {
        let Some(expected_field) = matched_def else {
            continue;
        };
        let resolved_ty = subst_type(&expected_field.ty, &subst, &HashMap::new());
        check_pattern_inner(
            subpat,
            &resolved_ty,
            mutable,
            in_match,
            None,
            type_checker,
            errors,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_struct_destructure_pattern(
    pattern: &PatternNode,
    type_name: Ident,
    fields: &[(Ident, PatternNode)],
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    // try native struct first
    if let Some(struct_def) = type_checker.get_struct(type_name).cloned() {
        let matches_type = value_ty
            .as_aggregate()
            .is_some_and(|agg| agg.name == type_name);
        if !matches_type {
            let expected = struct_def.make_type(type_name, vec![]);
            errors.push(Diagnostic::new(
                pattern.span,
                DiagnosticKind::MismatchedTypes {
                    expected,
                    found: value_ty.clone(),
                },
            ));
            return;
        }

        let provided: Vec<(Ident, Span)> = fields.iter().map(|(n, _)| (*n, pattern.span)).collect();
        let matched = validate_field_names(
            &provided,
            pattern.span,
            &struct_def.fields,
            true,
            |field| DiagnosticKind::StructDestructureDuplicateField { type_name, field },
            |field| DiagnosticKind::StructDestructureUnknownField { type_name, field },
            |_| unreachable!(),
            None,
            errors,
        );

        let subst = value_ty
            .as_aggregate()
            .map(|agg| build_subst(&struct_def.type_params, agg.type_args))
            .unwrap_or_default();

        for ((_, subpat), matched_def) in fields.iter().zip(matched.iter()) {
            let Some(field_def) = matched_def else {
                continue;
            };
            let resolved_ty = if subst.is_empty() {
                field_def.ty.clone()
            } else {
                subst_type(&field_def.ty, &subst, &HashMap::new())
            };
            check_pattern_inner(
                subpat,
                &resolved_ty,
                mutable,
                in_match,
                None,
                type_checker,
                errors,
            );
        }
        return;
    }

    // try extern type
    if let Some(extern_def) = type_checker.get_extern_type(type_name).cloned() {
        let matches_type = matches!(value_ty, Type::Extern { name } if *name == type_name);
        if !matches_type {
            errors.push(Diagnostic::new(
                pattern.span,
                DiagnosticKind::MismatchedTypes {
                    expected: Type::Extern { name: type_name },
                    found: value_ty.clone(),
                },
            ));
            return;
        }

        let expected_fields = extern_def.as_struct_fields();
        let provided: Vec<(Ident, Span)> = fields.iter().map(|(n, _)| (*n, pattern.span)).collect();
        let matched = validate_field_names(
            &provided,
            pattern.span,
            &expected_fields,
            true,
            |field| DiagnosticKind::StructDestructureDuplicateField { type_name, field },
            |field| DiagnosticKind::StructDestructureUnknownField { type_name, field },
            |_| unreachable!(),
            None,
            errors,
        );

        for ((_, subpat), matched_def) in fields.iter().zip(matched.iter()) {
            let Some(field_def) = matched_def else {
                continue;
            };
            check_pattern_inner(
                subpat,
                &field_def.ty,
                mutable,
                in_match,
                None,
                type_checker,
                errors,
            );
        }
        return;
    }

    errors.push(Diagnostic::new(
        pattern.span,
        DiagnosticKind::UnknownStruct { name: type_name },
    ));
}
