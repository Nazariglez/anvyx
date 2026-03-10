use crate::{
    ast::{
        ArrayFillNode, ArrayLen, ArrayLiteralNode, ExprKind, ExprNode, FieldAccessNode, Ident,
        IndexNode, Lit, MapLiteralNode, RangeNode, StructLiteralNode, TupleIndexNode, Type,
        VariantKind,
    },
    span::Span,
};
use std::collections::{HashMap, HashSet};

use super::{
    constraint::TypeRef,
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    infer::{create_inference_slots, subst_type, type_to_ref_with_inference},
    range::{range_inclusive_type, range_type},
    types::{EnumDef, TypeChecker, indexable_element_type, is_keyable},
};

pub(super) fn check_tuple(
    elements: &[ExprNode],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let element_types: Vec<Type> = elements
        .iter()
        .map(|el| check_expr(el, type_checker, errors))
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
        let ty = check_expr(expr, type_checker, errors);

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

    // otherwise it's a regular struct literal
    let struct_name = lit.name;

    let Some(struct_def) = type_checker.get_struct(struct_name).cloned() else {
        errors.push(TypeErr::new(
            lit_node.span,
            TypeErrKind::UnknownStruct { name: struct_name },
        ));
        return Type::Infer;
    };

    let is_generic = !struct_def.type_params.is_empty();
    let slots = is_generic
        .then(|| {
            let call_id = type_checker.next_call_id();
            create_inference_slots(&struct_def.type_params, type_checker, call_id)
        })
        .unwrap_or_default();

    let mut seen_fields = HashSet::new();
    let mut provided_fields = HashMap::new();

    for (field_name, field_expr) in &lit.fields {
        let field_ty = check_expr(field_expr, type_checker, errors);

        let is_new = seen_fields.insert(*field_name);
        if !is_new {
            errors.push(TypeErr::new(
                field_expr.span,
                TypeErrKind::StructDuplicateField {
                    struct_name,
                    field: *field_name,
                },
            ));
            continue;
        }

        let expected_field = struct_def.fields.iter().find(|f| f.name == *field_name);
        let Some(expected) = expected_field else {
            errors.push(TypeErr::new(
                field_expr.span,
                TypeErrKind::StructUnknownField {
                    struct_name,
                    field: *field_name,
                },
            ));
            continue;
        };

        let field_ref = TypeRef::Expr(field_expr.node.id);
        let expected_ref = if is_generic {
            type_to_ref_with_inference(&expected.ty, &slots)
        } else {
            let resolved = type_checker.resolve_type(&expected.ty);
            TypeRef::Concrete(resolved)
        };
        type_checker.constrain_assignable(field_expr.span, field_ref, expected_ref, errors);

        provided_fields.insert(*field_name, field_ty);
    }

    for struct_field in &struct_def.fields {
        let contains_field = provided_fields.contains_key(&struct_field.name);
        if !contains_field {
            errors.push(TypeErr::new(
                lit_node.span,
                TypeErrKind::StructMissingField {
                    struct_name,
                    field: struct_field.name,
                },
            ));
        }
    }

    let type_args = is_generic
        .then(|| {
            struct_def
                .type_params
                .iter()
                .map(|param| {
                    let slot_name = slots.get(&param.id).expect("slot exists");
                    type_checker
                        .get_var(*slot_name)
                        .cloned()
                        .unwrap_or(Type::Infer)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Type::Struct {
        name: struct_name,
        type_args,
    }
}

pub(super) fn check_tuple_index(
    index_node: &TupleIndexNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &index_node.node;
    let target_ty = check_expr(&node.target, type_checker, errors);
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

pub(super) fn try_check_enum_unit_variant(
    field_node: &FieldAccessNode,
    type_checker: &TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Option<Type> {
    let node = &field_node.node;

    let ExprKind::Ident(type_name) = &node.target.node.kind else {
        return None;
    };

    let Some(enum_def) = type_checker.get_enum(*type_name).cloned() else {
        return None;
    };

    let variant_name = node.field;
    let variant = enum_def.variants.iter().find(|v| v.name == variant_name);

    let Some(variant) = variant else {
        errors.push(TypeErr::new(
            field_node.span,
            TypeErrKind::UnknownEnumVariant {
                enum_name: *type_name,
                variant_name,
            },
        ));
        return Some(Type::Infer);
    };

    match &variant.kind {
        VariantKind::Unit => {
            let type_args = build_enum_type_args(&enum_def, type_checker);
            Some(Type::Enum {
                name: *type_name,
                type_args,
            })
        }
        VariantKind::Tuple(_) => {
            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::EnumVariantNotUnit {
                    enum_name: *type_name,
                    variant_name,
                },
            ));
            Some(Type::Infer)
        }
        VariantKind::Struct(_) => {
            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::EnumVariantNotUnit {
                    enum_name: *type_name,
                    variant_name,
                },
            ));
            Some(Type::Infer)
        }
    }
}

fn build_enum_type_args(enum_def: &EnumDef, _type_checker: &TypeChecker) -> Vec<Type> {
    if enum_def.type_params.is_empty() {
        vec![]
    } else {
        enum_def.type_params.iter().map(|_| Type::Infer).collect()
    }
}

pub(super) fn check_field_access(
    field_node: &FieldAccessNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    // check if this is an enum unit variant construction first
    if let Some(enum_ty) = try_check_enum_unit_variant(field_node, type_checker, errors) {
        return enum_ty;
    }

    let node = &field_node.node;
    let target_ty = check_expr(&node.target, type_checker, errors);
    let field = node.field;

    match &target_ty {
        Type::NamedTuple(fields) => {
            for (label, ty) in fields {
                if *label == field {
                    return ty.clone();
                }
            }
            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::NoSuchFieldOnTuple {
                    field,
                    tuple_type: target_ty.clone(),
                },
            ));
            Type::Infer
        }
        Type::Struct {
            name: struct_name,
            type_args,
        } => {
            let Some(struct_def) = type_checker.get_struct(*struct_name).cloned() else {
                errors.push(TypeErr::new(
                    field_node.span,
                    TypeErrKind::UnknownStruct { name: *struct_name },
                ));
                return Type::Infer;
            };

            let subst = struct_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect::<HashMap<_, _>>();

            for struct_field in &struct_def.fields {
                if struct_field.name == field {
                    let field_ty = subst_type(&struct_field.ty, &subst);
                    return type_checker.resolve_type(&field_ty);
                }
            }

            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::StructUnknownField {
                    struct_name: *struct_name,
                    field,
                },
            ));
            Type::Infer
        }
        Type::Infer => Type::Infer,
        _ => {
            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::FieldAccessOnNonNamedTuple {
                    field,
                    found: target_ty.clone(),
                },
            ));
            Type::Infer
        }
    }
}

pub(super) fn check_range(
    range: &RangeNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let start_expr = range.node.start.as_ref();
    let end_expr = range.node.end.as_ref();

    let start_ty = check_expr(start_expr, type_checker, errors);
    let _ = check_expr(end_expr, type_checker, errors);

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
        let ty = check_expr(elem, type_checker, errors);
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
    let value_ty = check_expr(&fill.node.value, type_checker, errors);
    check_expr(&fill.node.len, type_checker, errors);

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

pub(super) fn check_index(
    index_node: &IndexNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &index_node.node;
    let target_ty = check_expr(&node.target, type_checker, errors);
    let index_ty = check_expr(&node.index, type_checker, errors);

    // handle map indexing
    if let Type::Map { key, value } = &target_ty {
        // check key type matches
        let key_ref = TypeRef::Expr(node.index.node.id);
        let expected_ref = TypeRef::Concrete((**key).clone());
        type_checker.constrain_equal(node.index.span, key_ref, expected_ref, errors);

        // return V?
        return Type::Optional(value.clone());
    }

    // arrays requires int index
    let index_is_int = matches!(index_ty, Type::Int | Type::Infer);
    if !index_is_int {
        errors.push(TypeErr::new(
            node.index.span,
            TypeErrKind::IndexNotInt { found: index_ty },
        ));
    }

    match indexable_element_type(&target_ty) {
        Some(elem_ty) => elem_ty,
        None => {
            errors.push(TypeErr::new(
                node.target.span,
                TypeErrKind::IndexOnNonArray { found: target_ty },
            ));
            Type::Infer
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
            let is_duplicate = seen_literal_keys.iter().any(|k| *k == lit_key);
            if is_duplicate {
                errors.push(TypeErr::new(key_expr.span, TypeErrKind::MapDuplicateKey));
            } else {
                seen_literal_keys.push(lit_key);
            }
        }

        let key_ty = check_expr(key_expr, type_checker, errors);
        let value_ty = check_expr(value_expr, type_checker, errors);
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
    let is_type_infer = matches!(key_ty, Type::Infer);
    if !is_type_infer && !is_keyable(&key_ty) {
        let is_optional = matches!(key_ty, Type::Optional(_));
        if is_optional {
            errors.push(TypeErr::new(
                lit.span,
                TypeErrKind::MapOptionalKeyNotAllowed {
                    found: key_ty.clone(),
                },
            ));
        } else {
            errors.push(TypeErr::new(
                lit.span,
                TypeErrKind::MapKeyNotKeyable {
                    found: key_ty.clone(),
                },
            ));
        }
    }

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

    let variant = enum_def.variants.iter().find(|v| v.name == variant_name);

    let Some(variant) = variant else {
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
    let slots = is_generic
        .then(|| {
            let call_id = type_checker.next_call_id();
            create_inference_slots(&enum_def.type_params, type_checker, call_id)
        })
        .unwrap_or_default();

    let mut seen_fields = HashSet::new();
    let mut provided_fields = HashMap::new();

    for (field_name, field_expr) in &lit.fields {
        let field_ty = check_expr(field_expr, type_checker, errors);

        let is_new = seen_fields.insert(*field_name);
        if !is_new {
            errors.push(TypeErr::new(
                field_expr.span,
                TypeErrKind::EnumVariantDuplicateField {
                    enum_name,
                    variant_name,
                    field: *field_name,
                },
            ));
            continue;
        }

        let expected_field = expected_fields.iter().find(|f| f.name == *field_name);
        let Some(expected) = expected_field else {
            errors.push(TypeErr::new(
                field_expr.span,
                TypeErrKind::EnumVariantUnknownField {
                    enum_name,
                    variant_name,
                    field: *field_name,
                },
            ));
            continue;
        };

        let field_ref = TypeRef::Expr(field_expr.node.id);
        let expected_ref = if is_generic {
            type_to_ref_with_inference(&expected.ty, &slots)
        } else {
            let resolved = type_checker.resolve_type(&expected.ty);
            TypeRef::Concrete(resolved)
        };
        type_checker.constrain_assignable(field_expr.span, field_ref, expected_ref, errors);

        provided_fields.insert(*field_name, field_ty);
    }

    for expected_field in expected_fields {
        let contains_field = provided_fields.contains_key(&expected_field.name);
        if !contains_field {
            errors.push(TypeErr::new(
                lit_node.span,
                TypeErrKind::EnumVariantMissingField {
                    enum_name,
                    variant_name,
                    field: expected_field.name,
                },
            ));
        }
    }

    let type_args = is_generic
        .then(|| {
            enum_def
                .type_params
                .iter()
                .map(|param| {
                    let slot_name = slots.get(&param.id).expect("slot exists");
                    type_checker
                        .get_var(*slot_name)
                        .cloned()
                        .unwrap_or(Type::Infer)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Type::Enum {
        name: enum_name,
        type_args,
    }
}
