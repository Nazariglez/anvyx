use super::{
    constraint::TypeRef,
    error::{Diagnostic, DiagnosticKind},
    types::TypeChecker,
    visit::{map_type_structure, type_any, walk_type_structure},
};
use crate::{
    ast::{ArrayLen, Type},
    span::Span,
};

pub(super) fn contains_infer(ty: &Type) -> bool {
    type_any(ty, &mut |t| matches!(t, Type::Infer))
}

/// Checks if 'from' is assignable to 'to'
pub(super) fn is_assignable(from: &Type, to: &Type) -> bool {
    use Type::{Array, ArrayView, Extern};

    // same type is always assignable
    if from == to {
        return true;
    }

    // any type accepts any concrete type
    if matches!(from, Type::Any) || matches!(to, Type::Any) {
        return true;
    }

    // if either side is Infer, we need to unify them
    if from.is_infer() || to.is_infer() {
        return true;
    }

    // T -> T? is assignable
    if to.is_option() && !from.is_option() {
        let inner = to.option_inner().expect("is_option guarantees inner");
        return is_assignable(from, inner);
    }

    // T? -> T is not assignable, must unwrap first
    if from.is_option() && !to.is_option() {
        return false;
    }

    match (from, to) {
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
                // infer and unresolved length params match any concrete size, specialization checks the actual value later
                (ArrayLen::Infer, _)
                | (_, ArrayLen::Infer)
                | (ArrayLen::Fixed(_), ArrayLen::Param(_) | ArrayLen::Named(_))
                | (ArrayLen::Param(_) | ArrayLen::Named(_), ArrayLen::Fixed(_)) => true,
                (ArrayLen::Param(a), ArrayLen::Param(b)) => a == b,
                _ => false,
            };
            len_ok && is_assignable(elem_from, elem_to)
        }

        // Array[T; N] | [T; ..] | [T] -> [U; ..] if T is assignable to U
        (
            Array {
                elem: elem_from, ..
            }
            | ArrayView { elem: elem_from }
            | Type::List { elem: elem_from },
            ArrayView { elem: elem_to },
        ) => is_assignable(elem_from, elem_to),

        // opaque extern types are only assignable to the exact same extern type
        (Extern { name: ln }, Extern { name: rn }) => ln == rn,

        // structural cases (List, Map, Func, Struct, DataRef, Enum, Tuple, NamedTuple)
        _ => walk_type_structure(from, to, &mut |f, t| is_assignable(f, t)),
    }
}

/// Unifies two types returning the unified type if successful
pub(super) fn unify_types(
    left: &Type,
    right: &Type,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) -> Option<Type> {
    use Type::{Array, Extern, Infer};

    // same type, no need to unify
    if left == right {
        return Some(left.clone());
    }

    // if either side is Infer we use the concrete side
    if let (Infer, t) | (t, Infer) = (left, right) {
        return Some(t.clone());
    }

    // T with T? unifies to T?
    if right.is_option() && !left.is_option() {
        let inner = right.option_inner().expect("is_option guarantees inner");
        return unify_types(left, inner, span, errors).map(Type::option_of);
    }
    if left.is_option() && !right.is_option() {
        let inner = left.option_inner().expect("is_option guarantees inner");
        return unify_types(inner, right, span, errors).map(Type::option_of);
    }

    // arrays unify if the len matches and the element types can unify
    // Fixed(N) only unifies with Fixed(N) or Infer
    if let (Array { elem: le, len: ll }, Array { elem: re, len: rl }) = (left, right) {
        let unified_len = match (ll, rl) {
            (ArrayLen::Fixed(a), ArrayLen::Fixed(b)) if a == b => ArrayLen::Fixed(*a),
            (ArrayLen::Fixed(a), ArrayLen::Infer) | (ArrayLen::Infer, ArrayLen::Fixed(a)) => {
                ArrayLen::Fixed(*a)
            }
            (ArrayLen::Infer, ArrayLen::Infer) => ArrayLen::Infer,
            (ArrayLen::Param(a), ArrayLen::Param(b)) if a == b => ArrayLen::Param(*a),
            _ => {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::MismatchedTypes {
                        expected: left.clone(),
                        found: right.clone(),
                    },
                ));
                return None;
            }
        };
        return unify_types(le, re, span, errors).map(|elem| Array {
            elem: elem.boxed(),
            len: unified_len,
        });
    }

    // extern types unify only if they have the same name
    if let (Extern { name: ln }, Extern { name: rn }) = (left, right) {
        if ln == rn {
            return Some(Extern { name: *ln });
        }
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::MismatchedTypes {
                expected: left.clone(),
                found: right.clone(),
            },
        ));
        return None;
    }

    // structural cases (List, Map, ArrayView, Func, Struct, DataRef, Enum, Tuple, NamedTuple)
    if let Some(unified) =
        map_type_structure(left, right, &mut |l, r| unify_types(l, r, span, errors))
    {
        return Some(unified);
    }

    errors.push(Diagnostic::new(
        span,
        DiagnosticKind::MismatchedTypes {
            expected: left.clone(),
            found: right.clone(),
        },
    ));
    None
}

pub(super) fn unify_equal(
    tcx: &mut TypeChecker,
    span: Span,
    left: &TypeRef,
    right: &TypeRef,
    errors: &mut Vec<Diagnostic>,
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
