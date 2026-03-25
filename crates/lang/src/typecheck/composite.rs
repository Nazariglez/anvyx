use crate::{
    ast::{
        ArrayFillNode, ArrayLen, ArrayLiteralNode, ExprKind, ExprNode, Ident, Lit, MapLiteralNode,
        RangeNode, StructField, StructLiteralNode, TupleIndexNode, Type, TypeParam, VariantKind,
    },
    span::Span,
};
use std::collections::HashSet;

use super::{
    constraint::TypeRef,
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    infer::{build_param_ref, constrain_slots_from_type, create_inference_slots},
    range::{range_inclusive_type, range_type},
    types::{ExternTypeDef, InferenceSlots, TypeChecker, validate_map_key_type},
};

pub(super) fn check_tuple(
    elements: &[ExprNode],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let element_types: Vec<Type> = elements
        .iter()
        .map(|el| check_expr(el, type_checker, errors, None))
        .collect();
    Type::Tuple(element_types)
}

pub(super) fn check_named_tuple(
    elements: &[(Ident, ExprNode)],
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let mut seen_labels = HashSet::new();
    let mut fields = Vec::with_capacity(elements.len());

    for (label, expr) in elements {
        let ty = check_expr(expr, type_checker, errors, None);

        let inserted = seen_labels.insert(*label);
        if !inserted {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::DuplicateTupleLabel { label: *label },
            ));
        }

        fields.push((*label, ty));
    }

    Type::NamedTuple(fields)
}

pub(super) fn validate_field_names<'a>(
    provided: &[(Ident, Span)],
    container_span: Span,
    expected: &'a [StructField],
    on_duplicate: impl Fn(Ident) -> TypeErrKind,
    on_unknown: impl Fn(Ident) -> TypeErrKind,
    on_missing: impl Fn(Ident) -> TypeErrKind,
    errors: &mut Vec<TypeErr>,
) -> Vec<Option<&'a StructField>> {
    let mut seen = HashSet::new();
    let mut matched_names = HashSet::new();

    let results = provided
        .iter()
        .map(|(name, span)| {
            if !seen.insert(*name) {
                errors.push(TypeErr::new(*span, on_duplicate(*name)));
                return None;
            }
            match expected.iter().find(|f| f.name == *name) {
                None => {
                    errors.push(TypeErr::new(*span, on_unknown(*name)));
                    None
                }
                Some(def) => {
                    matched_names.insert(*name);
                    Some(def)
                }
            }
        })
        .collect();

    for field in expected {
        if !matched_names.contains(&field.name) {
            errors.push(TypeErr::new(container_span, on_missing(field.name)));
        }
    }

    results
}

fn constrain_fields_and_extract_type_args(
    fields: &[(Ident, ExprNode)],
    matched: &[Option<&StructField>],
    type_params: &[TypeParam],
    slots: &InferenceSlots,
    is_generic: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Vec<Type> {
    for ((_, field_expr), matched_def) in fields.iter().zip(matched.iter()) {
        let Some(expected) = matched_def else {
            continue;
        };
        let field_ref = TypeRef::Expr(field_expr.node.id);
        let expected_ref = if is_generic {
            if let Some((_, field_ty)) = type_checker.get_type(field_expr.node.id) {
                let field_ty = field_ty.clone();
                constrain_slots_from_type(
                    &expected.ty, &field_ty, slots, field_expr.span, type_checker, errors,
                );
            }
            build_param_ref(&expected.ty, slots, type_checker)
        } else {
            let resolved = type_checker.resolve_type(&expected.ty);
            TypeRef::concrete(&resolved)
        };
        type_checker.constrain_assignable(field_expr.span, field_ref, expected_ref, errors);
    }

    if is_generic {
        type_params
            .iter()
            .map(|param| {
                let slot_name = slots.get(&param.id).expect("slot exists");
                type_checker
                    .get_var(*slot_name)
                    .map(|info| info.ty.clone())
                    .unwrap_or(Type::Infer)
            })
            .collect()
    } else {
        vec![]
    }
}

pub(super) fn check_struct_lit(
    lit_node: &StructLiteralNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let lit = &lit_node.node;

    // check if this is an enum struct variant
    if let Some(enum_name) = lit.qualifier {
        return check_enum_struct_variant(lit_node, enum_name, lit.name, type_checker, errors);
    }

    let struct_name = lit.name;

    let Some(struct_def) = type_checker.get_struct(struct_name).cloned() else {
        if let Some(extern_def) = type_checker.get_extern_type(struct_name).cloned() {
            return check_extern_init_lit(lit_node, struct_name, &extern_def, type_checker, errors);
        }
        errors.push(TypeErr::new(
            lit_node.span,
            TypeErrKind::UnknownStruct { name: struct_name },
        ));
        return Type::Infer;
    };

    let is_generic = !struct_def.type_params.is_empty();
    let slots = if is_generic {
        let call_id = type_checker.next_call_id();
        create_inference_slots(&struct_def.type_params, type_checker, call_id)
    } else {
        Default::default()
    };

    // typecheck all field expressions before name validation
    for (_, field_expr) in &lit.fields {
        check_expr(field_expr, type_checker, errors, None);
    }

    let provided: Vec<(Ident, Span)> = lit.fields.iter().map(|(n, e)| (*n, e.span)).collect();
    let matched = validate_field_names(
        &provided,
        lit_node.span,
        &struct_def.fields,
        |field| TypeErrKind::StructDuplicateField { struct_name, field },
        |field| TypeErrKind::StructUnknownField { struct_name, field },
        |field| TypeErrKind::StructMissingField { struct_name, field },
        errors,
    );

    let type_args = constrain_fields_and_extract_type_args(
        &lit.fields,
        &matched,
        &struct_def.type_params,
        &slots,
        is_generic,
        type_checker,
        errors,
    );

    Type::Struct {
        name: struct_name,
        type_args,
    }
}

fn check_extern_init_lit(
    lit_node: &StructLiteralNode,
    type_name: Ident,
    extern_def: &ExternTypeDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let lit = &lit_node.node;

    if !extern_def.has_init {
        errors.push(TypeErr::new(
            lit_node.span,
            TypeErrKind::ExternInitNoInit { type_name },
        ));
        return Type::Infer;
    }

    for (_, field_expr) in &lit.fields {
        check_expr(field_expr, type_checker, errors, None);
    }

    let expected_fields: Vec<StructField> = extern_def
        .field_order
        .iter()
        .map(|name| {
            let def = &extern_def.fields[name];
            StructField { name: *name, ty: def.ty.clone() }
        })
        .collect();

    let provided: Vec<(Ident, Span)> = lit.fields.iter().map(|(n, e)| (*n, e.span)).collect();
    let matched = validate_field_names(
        &provided,
        lit_node.span,
        &expected_fields,
        |field| TypeErrKind::ExternInitDuplicateField { type_name, field },
        |field| TypeErrKind::ExternInitUnknownField { type_name, field },
        |field| TypeErrKind::ExternInitMissingField { type_name, field },
        errors,
    );

    for ((_, field_expr), matched_def) in lit.fields.iter().zip(matched.iter()) {
        let Some(expected) = matched_def else { continue };
        let field_ref = TypeRef::Expr(field_expr.node.id);
        let resolved = type_checker.resolve_type(&expected.ty);
        let expected_ref = TypeRef::concrete(&resolved);
        type_checker.constrain_assignable(field_expr.span, field_ref, expected_ref, errors);
    }

    Type::Extern { name: type_name }
}

pub(super) fn check_tuple_index(
    index_node: &TupleIndexNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &index_node.node;
    let target_ty = check_expr(&node.target, type_checker, errors, None);
    let index = node.index;

    let Some(element_types) = target_ty.tuple_element_types() else {
        if matches!(target_ty, Type::Infer) {
            return Type::Infer;
        }

        errors.push(TypeErr::new(
            index_node.span,
            TypeErrKind::TupleIndexOnNonTuple {
                found: target_ty.clone(),
                index,
            },
        ));
        return Type::Infer;
    };

    let len = element_types.len();
    let is_in_bounds = (index as usize) < len;
    if is_in_bounds {
        element_types[index as usize].clone()
    } else {
        errors.push(TypeErr::new(
            index_node.span,
            TypeErrKind::TupleIndexOutOfBounds {
                tuple_type: target_ty.clone(),
                index,
                len,
            },
        ));
        Type::Infer
    }
}

pub(super) fn check_range(
    range: &RangeNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let start_expr = range.node.start.as_ref();
    let end_expr = range.node.end.as_ref();

    let start_ty = check_expr(start_expr, type_checker, errors, None);
    let _ = check_expr(end_expr, type_checker, errors, None);

    let start_ref = TypeRef::Expr(start_expr.node.id);
    let end_ref = TypeRef::Expr(end_expr.node.id);
    type_checker.constrain_equal(range.span, start_ref, end_ref, errors);

    let elem_ty = type_checker
        .get_type(start_expr.node.id)
        .map(|(_, ty)| ty.clone())
        .unwrap_or(start_ty);

    if range.node.inclusive {
        range_inclusive_type(elem_ty)
    } else {
        range_type(elem_ty)
    }
}

pub(super) fn check_array_literal(
    lit: &ArrayLiteralNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let elements = &lit.node.elements;
    if elements.is_empty() {
        return Type::Array {
            elem: Type::Infer.boxed(),
            len: ArrayLen::Fixed(0),
        };
    }

    let mut elem_types = vec![];

    for elem in elements {
        let ty = check_expr(elem, type_checker, errors, None);
        elem_types.push(ty);
    }

    for i in 1..elements.len() {
        let left_ref = TypeRef::Expr(elements[i - 1].node.id);
        let right_ref = TypeRef::Expr(elements[i].node.id);
        type_checker.constrain_equal(lit.span, left_ref, right_ref, errors);
    }

    let elem_ty = type_checker
        .get_type(elements[0].node.id)
        .map(|(_, ty)| ty.clone())
        .unwrap_or_else(|| elem_types[0].clone());

    Type::Array {
        elem: elem_ty.boxed(),
        len: ArrayLen::Fixed(elements.len()),
    }
}

pub(super) fn is_all_nil_array_literal(expr: &ExprNode) -> bool {
    if let ExprKind::ArrayLiteral(lit) = &expr.node.kind {
        let elements = &lit.node.elements;
        let has_elements = !elements.is_empty();
        if has_elements {
            return elements
                .iter()
                .all(|e| matches!(&e.node.kind, ExprKind::Lit(Lit::Nil)));
        }
    }

    false
}

pub(super) fn is_empty_map_literal(expr: &ExprNode) -> bool {
    if let ExprKind::MapLiteral(lit) = &expr.node.kind {
        return lit.node.entries.is_empty();
    }
    false
}

pub(super) fn check_array_fill(
    fill: &ArrayFillNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let value_ty = check_expr(&fill.node.value, type_checker, errors, None);
    check_expr(&fill.node.len, type_checker, errors, None);

    let len_expr = &fill.node.len;
    match &len_expr.node.kind {
        ExprKind::Lit(Lit::Int(n)) if *n >= 0 => Type::Array {
            elem: value_ty.boxed(),
            len: ArrayLen::Fixed(*n as usize),
        },
        _ => {
            errors.push(
                TypeErr::new(len_expr.span, TypeErrKind::ArrayFillLengthNotLiteral).with_help(
                    "the length in `[expr; len]` must be a compile-time integer literal",
                ),
            );
            Type::Array {
                elem: value_ty.boxed(),
                len: ArrayLen::Infer,
            }
        }
    }
}

pub(super) fn check_map_literal(
    lit: &MapLiteralNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let entries = &lit.node.entries;

    if entries.is_empty() {
        return Type::Map {
            key: Type::Infer.boxed(),
            value: Type::Infer.boxed(),
        };
    }

    let mut key_types = vec![];
    let mut value_types = vec![];
    let mut seen_literal_keys: Vec<&Lit> = vec![];

    for (key_expr, value_expr) in entries {
        // check for duplicate literal keys
        if let ExprKind::Lit(lit_key) = &key_expr.node.kind {
            let is_duplicate = seen_literal_keys.contains(&lit_key);
            if is_duplicate {
                errors.push(TypeErr::new(key_expr.span, TypeErrKind::MapDuplicateKey));
            } else {
                seen_literal_keys.push(lit_key);
            }
        }

        let key_ty = check_expr(key_expr, type_checker, errors, None);
        let value_ty = check_expr(value_expr, type_checker, errors, None);
        key_types.push(key_ty);
        value_types.push(value_ty);
    }

    // unify all key types
    for i in 1..entries.len() {
        let left_ref = TypeRef::Expr(entries[i - 1].0.node.id);
        let right_ref = TypeRef::Expr(entries[i].0.node.id);
        type_checker.constrain_equal(lit.span, left_ref, right_ref, errors);
    }

    // unify all value types
    for i in 1..entries.len() {
        let left_ref = TypeRef::Expr(entries[i - 1].1.node.id);
        let right_ref = TypeRef::Expr(entries[i].1.node.id);
        type_checker.constrain_equal(lit.span, left_ref, right_ref, errors);
    }

    let key_ty = type_checker
        .get_type(entries[0].0.node.id)
        .map(|(_, ty)| ty.clone())
        .unwrap_or_else(|| key_types[0].clone());

    let value_ty = type_checker
        .get_type(entries[0].1.node.id)
        .map(|(_, ty)| ty.clone())
        .unwrap_or_else(|| value_types[0].clone());

    // validate key type is keyable
    validate_map_key_type(lit.span, &key_ty, type_checker, errors);

    Type::Map {
        key: key_ty.boxed(),
        value: value_ty.boxed(),
    }
}

fn check_enum_struct_variant(
    lit_node: &StructLiteralNode,
    enum_name: Ident,
    variant_name: Ident,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let lit = &lit_node.node;

    let Some(enum_def) = type_checker.get_enum(enum_name).cloned() else {
        errors.push(TypeErr::new(
            lit_node.span,
            TypeErrKind::UnknownEnum { name: enum_name },
        ));
        return Type::Infer;
    };

    let Some(variant) = enum_def.variants.iter().find(|v| v.name == variant_name) else {
        errors.push(TypeErr::new(
            lit_node.span,
            TypeErrKind::UnknownEnumVariant {
                enum_name,
                variant_name,
            },
        ));
        return Type::Infer;
    };

    let VariantKind::Struct(expected_fields) = &variant.kind else {
        errors.push(TypeErr::new(
            lit_node.span,
            TypeErrKind::EnumVariantNotStruct {
                enum_name,
                variant_name,
            },
        ));
        return Type::Infer;
    };

    let is_generic = !enum_def.type_params.is_empty();
    let slots = if is_generic {
        let call_id = type_checker.next_call_id();
        create_inference_slots(&enum_def.type_params, type_checker, call_id)
    } else {
        Default::default()
    };

    // typecheck all field expressions before name validation
    for (_, field_expr) in &lit.fields {
        check_expr(field_expr, type_checker, errors, None);
    }

    let provided: Vec<(Ident, Span)> = lit.fields.iter().map(|(n, e)| (*n, e.span)).collect();
    let matched = validate_field_names(
        &provided,
        lit_node.span,
        expected_fields,
        |field| TypeErrKind::EnumVariantDuplicateField {
            enum_name,
            variant_name,
            field,
        },
        |field| TypeErrKind::EnumVariantUnknownField {
            enum_name,
            variant_name,
            field,
        },
        |field| TypeErrKind::EnumVariantMissingField {
            enum_name,
            variant_name,
            field,
        },
        errors,
    );

    let type_args = constrain_fields_and_extract_type_args(
        &lit.fields,
        &matched,
        &enum_def.type_params,
        &slots,
        is_generic,
        type_checker,
        errors,
    );

    Type::Enum {
        name: enum_name,
        type_args,
    }
}
