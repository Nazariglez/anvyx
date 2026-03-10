use crate::{
    ast::{ArrayLen, Type},
    span::Span,
};

use super::{
    constraint::TypeRef,
    error::{TypeErr, TypeErrKind},
    types::TypeChecker,
};

pub(super) fn contains_infer(ty: &Type) -> bool {
    match ty {
        Type::Infer => true,
        Type::Optional(inner) => contains_infer(inner),
        Type::Func { params, ret } => params.iter().any(contains_infer) || contains_infer(ret),
        Type::Tuple(elems) => elems.iter().any(contains_infer),
        Type::NamedTuple(fields) => fields.iter().any(|(_, t)| contains_infer(t)),
        Type::Struct { type_args, .. } => type_args.iter().any(contains_infer),
        Type::Array { elem, .. } => contains_infer(elem),
        Type::ArrayView { elem } => contains_infer(elem),
        _ => false,
    }
}

/// Checks if 'from' is assignable to 'to'
pub(super) fn is_assignable(from: &Type, to: &Type) -> bool {
    use Type::*;

    // same type is always assignable
    let is_same_type = from == to;
    if is_same_type {
        return true;
    }

    // if either side is Infer, we need to unify them
    let needs_inference = from.is_infer() || to.is_infer();
    if needs_inference {
        return true;
    }

    match (from, to) {
        // optional types needs to check the inner types
        (Optional(inner_from), Optional(inner_to)) => is_assignable(inner_from, inner_to),

        // T to T? is assignable
        (from_ty, Optional(inner_to)) if !matches!(from_ty, Optional(_)) => {
            is_assignable(from_ty, inner_to)
        }

        // function types needs to check the signature (params + return type)
        (
            Func {
                params: params_from,
                ret: ret_from,
            },
            Func {
                params: params_to,
                ret: ret_to,
            },
        ) => {
            params_from.len() == params_to.len()
                && params_from
                    .iter()
                    .zip(params_to.iter())
                    .all(|(pf, pt)| is_assignable(pf, pt))
                && is_assignable(ret_from, ret_to)
        }

        // T? to T is not assignable, the value must be unwrapped first
        (Optional(_), non_opt) if !matches!(non_opt, Optional(_)) => false,

        // array T[N] to T[M] requires the len and the element type to be assignable
        (
            Array {
                elem: elem_from,
                len: len_from,
            },
            Array {
                elem: elem_to,
                len: len_to,
            },
        ) => {
            let len_ok = match (len_from, len_to) {
                (ArrayLen::Fixed(n), ArrayLen::Fixed(m)) => n == m,
                (ArrayLen::Infer, _) | (_, ArrayLen::Infer) => true,
            };
            len_ok && is_assignable(elem_from, elem_to)
        }

        (List { elem: elem_from }, List { elem: elem_to }) => is_assignable(elem_from, elem_to),

        (
            Map {
                key: key_from,
                value: value_from,
            },
            Map {
                key: key_to,
                value: value_to,
            },
        ) => is_assignable(key_from, key_to) && is_assignable(value_from, value_to),

        // view [T; ..] to [U; ..] if T is assignable to U
        (ArrayView { elem: elem_from }, ArrayView { elem: elem_to }) => {
            is_assignable(elem_from, elem_to)
        }

        // array [T; N] to View [U; ..] if T is assignable to U
        (
            Array {
                elem: elem_from, ..
            },
            ArrayView { elem: elem_to },
        ) => is_assignable(elem_from, elem_to),

        // list [T] to View [U; ..] if T is assignable to U
        (List { elem: elem_from }, ArrayView { elem: elem_to }) => {
            is_assignable(elem_from, elem_to)
        }

        // anything else is just not assignable
        _ => false,
    }
}

/// Unifies two types returning the unified type if successful
pub(super) fn unify_types(
    left: &Type,
    right: &Type,
    span: Span,
    errors: &mut Vec<TypeErr>,
) -> Option<Type> {
    use Type::*;

    // same type, no need to unify
    if left == right {
        return Some(left.clone());
    }

    match (left, right) {
        // if either side is Infer we use the concrete side
        (Infer, t) | (t, Infer) => Some(t.clone()),

        // optional types needs to unify the inner types
        (Optional(l), Optional(r)) => {
            unify_types(l, r, span, errors).map(|inner| Optional(Box::new(inner)))
        }

        (Optional(inner), other) if !other.is_optional() => {
            unify_types(inner.as_ref(), other, span, errors)
                .map(|inner_ty| Optional(Box::new(inner_ty)))
        }
        (other, Optional(inner)) if !other.is_optional() => {
            unify_types(other, inner.as_ref(), span, errors)
                .map(|inner_ty| Optional(Box::new(inner_ty)))
        }

        // function types needs to unify the params and return type
        (
            Func {
                params: lp,
                ret: lr,
            },
            Func {
                params: rp,
                ret: rr,
            },
        ) => {
            if lp.len() != rp.len() {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::MismatchedTypes {
                        expected: left.clone(),
                        found: right.clone(),
                    },
                ));
                return None;
            }

            let mut new_params = Vec::with_capacity(lp.len());
            for (lpi, rpi) in lp.iter().zip(rp.iter()) {
                unify_types(lpi, rpi, span, errors).map(|p| new_params.push(p))?;
            }

            unify_types(lr, rr, span, errors).map(|new_ret| Func {
                params: new_params,
                ret: Box::new(new_ret),
            })
        }

        (
            Struct {
                name: ln,
                type_args: la,
            },
            Struct {
                name: rn,
                type_args: ra,
            },
        ) => {
            if ln != rn || la.len() != ra.len() {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::MismatchedTypes {
                        expected: left.clone(),
                        found: right.clone(),
                    },
                ));
                return None;
            }

            let unified_args = la
                .iter()
                .zip(ra.iter())
                .map(|(l_arg, r_arg)| unify_types(l_arg, r_arg, span, errors).unwrap())
                .collect();

            Some(Struct {
                name: *ln,
                type_args: unified_args,
            })
        }

        // arrays unify if the len matches and the element types can unify
        // Fixed(N) only unifies with Fixed(N) or Infer
        (Array { elem: le, len: ll }, Array { elem: re, len: rl }) => {
            let unified_len = match (ll, rl) {
                (ArrayLen::Fixed(a), ArrayLen::Fixed(b)) if a == b => ArrayLen::Fixed(*a),
                (ArrayLen::Fixed(a), ArrayLen::Infer) | (ArrayLen::Infer, ArrayLen::Fixed(a)) => {
                    ArrayLen::Fixed(*a)
                }
                (ArrayLen::Infer, ArrayLen::Infer) => ArrayLen::Infer,
                _ => {
                    errors.push(TypeErr::new(
                        span,
                        TypeErrKind::MismatchedTypes {
                            expected: left.clone(),
                            found: right.clone(),
                        },
                    ));
                    return None;
                }
            };
            unify_types(le, re, span, errors).map(|elem| Array {
                elem: elem.boxed(),
                len: unified_len,
            })
        }

        // lists unify if element types can unify
        (List { elem: le }, List { elem: re }) => {
            unify_types(le, re, span, errors).map(|elem| List { elem: elem.boxed() })
        }

        // maps unify if key and value types can unify
        (Map { key: lk, value: lv }, Map { key: rk, value: rv }) => {
            let unified_key = unify_types(lk, rk, span, errors)?;
            let unified_value = unify_types(lv, rv, span, errors)?;
            Some(Map {
                key: unified_key.boxed(),
                value: unified_value.boxed(),
            })
        }

        // views unify if element types can unify
        (ArrayView { elem: le }, ArrayView { elem: re }) => {
            unify_types(le, re, span, errors).map(|elem| ArrayView { elem: elem.boxed() })
        }

        // mismatched types report an error
        (l, r) => {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::MismatchedTypes {
                    expected: l.clone(),
                    found: r.clone(),
                },
            ));
            None
        }
    }
}

pub(super) fn unify_equal(
    tcx: &mut TypeChecker,
    span: Span,
    left: &TypeRef,
    right: &TypeRef,
    errors: &mut Vec<TypeErr>,
) -> bool {
    let (Some(lt), Some(rt)) = (tcx.get_type_ref(left), tcx.get_type_ref(right)) else {
        return false;
    };

    match unify_types(&lt, &rt, span, errors) {
        Some(new_ty) => {
            tcx.set_type_ref(left, new_ty.clone(), span);
            tcx.set_type_ref(right, new_ty, span);
            true
        }
        None => false,
    }
}
