use std::collections::HashSet;

use crate::ast::{Ident, Lit, Pattern, PatternNode, Type, VariantKind};

use super::{
    composite::validate_field_names,
    error::{TypeErr, TypeErrKind},
    expr::type_from_lit,
    infer::{build_subst, subst_type},
    types::{EnumDef, TypeChecker},
};

pub(super) fn check_pattern(
    pattern: &PatternNode,
    value_ty: &Type,
    mutable: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    check_pattern_inner(pattern, value_ty, mutable, false, None, type_checker, errors);
}

pub(super) fn check_pattern_in_match(
    pattern: &PatternNode,
    value_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    check_pattern_inner(pattern, value_ty, false, true, None, type_checker, errors);
}

pub(super) fn is_refutable(pattern: &Pattern, value_ty: &Type, type_checker: &TypeChecker) -> bool {
    match pattern {
        Pattern::Wildcard => false,
        Pattern::Ident(_) => false,
        Pattern::VarIdent(_) => false,
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
            fields.iter().zip(expected_types.iter()).any(|(subpat, expected_ty)| {
                let resolved_ty = subst_type(expected_ty, &subst);
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
                let resolved_ty = subst_type(&field_def.ty, &subst);
                is_refutable(&subpat.node, &resolved_ty, type_checker)
            })
        }
        _ => true,
    }
}

pub(super) fn check_match_pattern(
    pattern: &PatternNode,
    scrutinee_ty: &Type,
    enum_def: &EnumDef,
    covered_variants: &mut HashSet<Ident>,
    has_wildcard: &mut bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
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

fn check_pattern_inner(
    pattern: &PatternNode,
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    match_ctx: Option<(&EnumDef, &mut HashSet<Ident>, &mut bool)>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    match &pattern.node {
        Pattern::Ident(name) => {
            if in_match {
                if let Some(const_def) = type_checker.get_const(*name) {
                    let const_ty = const_def.ty.clone();
                    let const_val = const_def.value.clone();
                    if const_ty != *value_ty && !value_ty.is_infer() {
                        errors.push(TypeErr::new(
                            pattern.span,
                            TypeErrKind::InvalidLiteralPattern {
                                expected: value_ty.clone(),
                                found: const_ty,
                            },
                        ));
                    }
                    let span_key = (pattern.span.start, pattern.span.end);
                    type_checker.const_pattern_values.insert(span_key, const_val);
                    return;
                }
            }
            type_checker.set_var(*name, value_ty.clone(), mutable);
            if let Some((_, _, has_wildcard)) = match_ctx {
                *has_wildcard = true;
            }
        }
        Pattern::Wildcard => {
            if let Some((_, _, has_wildcard)) = match_ctx {
                *has_wildcard = true;
            }
        }
        Pattern::Tuple(subpatterns) => {
            let Some(elem_types) = value_ty.tuple_element_types() else {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::NonTupleInTuplePattern {
                        found: value_ty.clone(),
                        pattern_arity: subpatterns.len(),
                    },
                ));
                return;
            };

            let same_arity = subpatterns.len() == elem_types.len();
            if !same_arity {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::TuplePatternArityMismatch {
                        expected: elem_types.len(),
                        found: subpatterns.len(),
                    },
                ));
                return;
            }

            for (subpat, elem_ty) in subpatterns.iter().zip(elem_types.iter()) {
                check_pattern_inner(subpat, elem_ty, mutable, in_match, None, type_checker, errors);
            }
        }
        Pattern::NamedTuple(elems) => {
            let Some(elem_types) = value_ty.tuple_element_types() else {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::NonTupleInTuplePattern {
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
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::TuplePatternArityMismatch {
                        expected: elem_types.len(),
                        found: elems.len(),
                    },
                ));
                return;
            }

            let Some(labels) = value_labels else {
                errors.push(
                    TypeErr::new(pattern.span, TypeErrKind::NamedPatternOnPositionalTuple)
                        .with_help("use positional pattern `(a, b, ...)` instead"),
                );
                return;
            };

            for ((pat_label, _), ty_label) in elems.iter().zip(labels.iter()) {
                if *pat_label != *ty_label {
                    errors.push(TypeErr::new(
                        pattern.span,
                        TypeErrKind::TuplePatternLabelMismatch {
                            expected: *ty_label,
                            found: *pat_label,
                        },
                    ));
                }
            }

            for ((_, subpat), elem_ty) in elems.iter().zip(elem_types.iter()) {
                check_pattern_inner(subpat, elem_ty, mutable, in_match, None, type_checker, errors);
            }
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
        Pattern::Lit(lit) => {
            let lit_ty = type_from_lit(lit);
            let is_valid_pattern_lit = matches!(lit, Lit::Int(_) | Lit::Bool(_) | Lit::String(_));
            if !is_valid_pattern_lit || (lit_ty != *value_ty && !value_ty.is_infer()) {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::InvalidLiteralPattern {
                        expected: value_ty.clone(),
                        found: lit_ty,
                    },
                ));
            }
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
    }
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
    errors: &mut Vec<TypeErr>,
) {
    let Type::Enum {
        name: enum_name,
        type_args,
    } = value_ty
    else {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::MismatchedTypes {
                expected: value_ty.clone(),
                found: Type::Enum {
                    name: qualifier,
                    type_args: vec![],
                },
            },
        ));
        return;
    };

    if qualifier != *enum_name {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::MatchPatternEnumMismatch {
                expected_enum: *enum_name,
                pattern_enum: qualifier,
            },
        ));
        return;
    }

    let Some(enum_def) = type_checker.get_enum(qualifier) else {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::UnknownEnum { name: qualifier },
        ));
        return;
    };
    let enum_def = enum_def.clone();

    let Some(variant_def) = enum_def.variants.iter().find(|v| v.name == variant_name) else {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::UnknownEnumVariant {
                enum_name: qualifier,
                variant_name,
            },
        ));
        return;
    };

    if let Some((_, covered_variants, _)) = match_ctx {
        covered_variants.insert(variant_name);
    }

    match &variant_def.kind {
        VariantKind::Unit => {
            if !fields.is_empty() {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::EnumVariantNotTuple {
                        enum_name: qualifier,
                        variant_name,
                    },
                ));
            }
        }
        VariantKind::Tuple(expected_types) => {
            if fields.len() != expected_types.len() {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::EnumVariantArityMismatch {
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
                let resolved_ty = subst_type(expected_ty, &subst);
                check_pattern_inner(subpat, &resolved_ty, mutable, in_match, None, type_checker, errors);
            }
        }
        VariantKind::Struct(_) => {
            errors.push(TypeErr::new(
                pattern.span,
                TypeErrKind::EnumVariantNotTuple {
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
    errors: &mut Vec<TypeErr>,
) {
    let Type::Enum {
        name: enum_name,
        type_args,
    } = value_ty
    else {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::MismatchedTypes {
                expected: value_ty.clone(),
                found: Type::Enum {
                    name: qualifier,
                    type_args: vec![],
                },
            },
        ));
        return;
    };

    if qualifier != *enum_name {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::MatchPatternEnumMismatch {
                expected_enum: *enum_name,
                pattern_enum: qualifier,
            },
        ));
        return;
    }

    let Some(enum_def) = type_checker.get_enum(qualifier) else {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::UnknownEnum { name: qualifier },
        ));
        return;
    };
    let enum_def = enum_def.clone();

    let Some(variant_def) = enum_def.variants.iter().find(|v| v.name == variant_name) else {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::UnknownEnumVariant {
                enum_name: qualifier,
                variant_name,
            },
        ));
        return;
    };

    if let Some((_, covered_variants, _)) = match_ctx {
        covered_variants.insert(variant_name);
    }

    let VariantKind::Struct(expected_fields) = &variant_def.kind else {
        errors.push(TypeErr::new(
            pattern.span,
            TypeErrKind::EnumVariantNotStruct {
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
        |field| TypeErrKind::EnumVariantDuplicateField {
            enum_name: qualifier,
            variant_name,
            field,
        },
        |field| TypeErrKind::EnumVariantUnknownField {
            enum_name: qualifier,
            variant_name,
            field,
        },
        |field| TypeErrKind::EnumVariantMissingField {
            enum_name: qualifier,
            variant_name,
            field,
        },
        errors,
    );

    for ((_, subpat), matched_def) in fields.iter().zip(matched.iter()) {
        let Some(expected_field) = matched_def else {
            continue;
        };
        let resolved_ty = subst_type(&expected_field.ty, &subst);
        check_pattern_inner(subpat, &resolved_ty, mutable, in_match, None, type_checker, errors);
    }
}

fn check_struct_destructure_pattern(
    pattern: &PatternNode,
    type_name: Ident,
    fields: &[(Ident, PatternNode)],
    value_ty: &Type,
    mutable: bool,
    in_match: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    // try native struct first
    if let Some(struct_def) = type_checker.get_struct(type_name).cloned() {
        let matches_type = matches!(
            value_ty,
            Type::Struct { name, .. } | Type::DataRef { name, .. } if *name == type_name
        );
        if !matches_type {
            let expected = struct_def.make_type(type_name, vec![]);
            errors.push(TypeErr::new(
                pattern.span,
                TypeErrKind::MismatchedTypes {
                    expected,
                    found: value_ty.clone(),
                },
            ));
            return;
        }

        let mut seen = HashSet::new();
        for (field_name, subpat) in fields {
            if !seen.insert(*field_name) {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::StructDestructureDuplicateField {
                        type_name,
                        field: *field_name,
                    },
                ));
                continue;
            }
            let Some(field_def) = struct_def.fields.iter().find(|f| f.name == *field_name) else {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::StructDestructureUnknownField {
                        type_name,
                        field: *field_name,
                    },
                ));
                continue;
            };
            let resolved_ty = match value_ty {
                Type::Struct { type_args, .. } | Type::DataRef { type_args, .. } => {
                    let subst = build_subst(&struct_def.type_params, type_args);
                    subst_type(&field_def.ty, &subst)
                }
                _ => field_def.ty.clone(),
            };
            check_pattern_inner(subpat, &resolved_ty, mutable, in_match, None, type_checker, errors);
        }
        return;
    }

    // try extern type
    if let Some(extern_def) = type_checker.get_extern_type(type_name).cloned() {
        let matches_type = matches!(value_ty, Type::Extern { name } if *name == type_name);
        if !matches_type {
            errors.push(TypeErr::new(
                pattern.span,
                TypeErrKind::MismatchedTypes {
                    expected: Type::Extern { name: type_name },
                    found: value_ty.clone(),
                },
            ));
            return;
        }

        let mut seen = HashSet::new();
        for (field_name, subpat) in fields {
            if !seen.insert(*field_name) {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::StructDestructureDuplicateField {
                        type_name,
                        field: *field_name,
                    },
                ));
                continue;
            }
            let Some(field_def) = extern_def.fields.get(field_name) else {
                errors.push(TypeErr::new(
                    pattern.span,
                    TypeErrKind::StructDestructureUnknownField {
                        type_name,
                        field: *field_name,
                    },
                ));
                continue;
            };
            check_pattern_inner(subpat, &field_def.ty, mutable, in_match, None, type_checker, errors);
        }
        return;
    }

    errors.push(TypeErr::new(
        pattern.span,
        TypeErrKind::UnknownStruct { name: type_name },
    ));
}
