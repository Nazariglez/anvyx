use super::{
    constraint::TypeRef,
    error::{Diagnostic, DiagnosticKind},
    types::TypeChecker,
    visit::type_any,
};
use crate::{
    ast::{ArrayLen, FuncParam, Ident, Type},
    span::Span,
};

pub(super) fn contains_infer(ty: &Type) -> bool {
    type_any(ty, &mut |t| matches!(t, Type::Infer))
}

fn named_type_args_assignable(ln: Ident, rn: Ident, la: &[Type], ra: &[Type]) -> bool {
    ln == rn && la.len() == ra.len() && la.iter().zip(ra.iter()).all(|(a, b)| is_assignable(a, b))
}

/// Checks if 'from' is assignable to 'to'
pub(super) fn is_assignable(from: &Type, to: &Type) -> bool {
    use Type::{Array, ArrayView, DataRef, Enum, Extern, Func, List, Map, Struct};

    // same type is always assignable
    let is_same_type = from == to;
    if is_same_type {
        return true;
    }

    // any type accepts any concrete type
    let has_any = matches!(from, Type::Any) || matches!(to, Type::Any);
    if has_any {
        return true;
    }

    // if either side is Infer, we need to unify them
    let needs_inference = from.is_infer() || to.is_infer();
    if needs_inference {
        return true;
    }

    match (from, to) {
        // optional types needs to unify the inner types
        (from_ty, to_ty) if from_ty.is_option() && to_ty.is_option() => {
            let inner_from = from_ty.option_inner().expect("is_option guarantees inner");
            let inner_to = to_ty.option_inner().expect("is_option guarantees inner");
            is_assignable(inner_from, inner_to)
        }

        // T -> T? is assignable
        (from_ty, to_ty) if to_ty.is_option() && !from_ty.is_option() => {
            let inner = to_ty.option_inner().expect("is_option guarantees inner");
            is_assignable(from_ty, inner)
        }

        // T? -> T is not assignable, must unwrap first
        (from_ty, to_ty) if from_ty.is_option() && !to_ty.is_option() => false,

        // enums, structs, and datarefs are assignable if same variant, same name, and all type_args assignable
        (from, to)
            if matches!(from, Enum { .. } | Struct { .. } | DataRef { .. })
                && std::mem::discriminant(from) == std::mem::discriminant(to) =>
        {
            let lhs = match from {
                Enum { name, type_args }
                | Struct { name, type_args }
                | DataRef { name, type_args } => (*name, type_args.as_slice()),
                _ => unreachable!(),
            };
            let rhs = match to {
                Enum { name, type_args }
                | Struct { name, type_args }
                | DataRef { name, type_args } => (*name, type_args.as_slice()),
                _ => unreachable!(),
            };
            named_type_args_assignable(lhs.0, rhs.0, lhs.1, rhs.1)
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
                    .all(|(pf, pt)| is_assignable(&pf.ty, &pt.ty))
                && is_assignable(ret_from, ret_to)
        }

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

        (List { elem: elem_from }, List { elem: elem_to })
        | (
            Array {
                elem: elem_from, ..
            },
            ArrayView { elem: elem_to },
        ) => is_assignable(elem_from, elem_to),

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
        (ArrayView { elem: elem_from } | List { elem: elem_from }, ArrayView { elem: elem_to }) => {
            is_assignable(elem_from, elem_to)
        }

        // array [T; N] to View [U; ..] if T is assignable to U

        // opaque extern types are only assignable to the exact same extern type
        (Extern { name: ln }, Extern { name: rn }) => ln == rn,

        // anything else is just not assignable
        _ => false,
    }
}

fn unify_named_type_args(
    ln: Ident,
    la: &[Type],
    rn: Ident,
    ra: &[Type],
    left: &Type,
    right: &Type,
    span: Span,
    errors: &mut Vec<Diagnostic>,
    make_type: impl Fn(Ident, Vec<Type>) -> Type,
) -> Option<Type> {
    if ln != rn || la.len() != ra.len() {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::MismatchedTypes {
                expected: left.clone(),
                found: right.clone(),
            },
        ));
        return None;
    }
    let unified_args: Option<Vec<Type>> = la
        .iter()
        .zip(ra.iter())
        .map(|(l_arg, r_arg)| unify_types(l_arg, r_arg, span, errors))
        .collect();
    unified_args.map(|type_args| make_type(ln, type_args))
}

/// Unifies two types returning the unified type if successful
pub(super) fn unify_types(
    left: &Type,
    right: &Type,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) -> Option<Type> {
    use Type::{Array, ArrayView, Enum, Extern, Func, Infer, List, Map, Struct};

    // same type, no need to unify
    if left == right {
        return Some(left.clone());
    }

    match (left, right) {
        // if either side is Infer we use the concrete side
        (Infer, t) | (t, Infer) => Some(t.clone()),

        // A? B? unify the inner types
        (l, r) if l.is_option() && r.is_option() => {
            let li = l.option_inner().expect("is_option guarantees inner");
            let ri = r.option_inner().expect("is_option guarantees inner");
            unify_types(li, ri, span, errors).map(Type::option_of)
        }

        // enum types of same name unify if all type_args can unify
        (
            Enum {
                name: ln,
                type_args: la,
            },
            Enum {
                name: rn,
                type_args: ra,
            },
        ) => unify_named_type_args(
            *ln,
            la,
            *rn,
            ra,
            left,
            right,
            span,
            errors,
            |name, type_args| Enum { name, type_args },
        ),

        // T with T? unifies to T?
        (other, opt) if opt.is_option() && !other.is_option() => {
            let inner = opt.option_inner().expect("is_option guarantees inner");
            unify_types(other, inner, span, errors).map(Type::option_of)
        }
        (opt, other) if opt.is_option() && !other.is_option() => {
            let inner = opt.option_inner().expect("is_option guarantees inner");
            unify_types(inner, other, span, errors).map(Type::option_of)
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
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::MismatchedTypes {
                        expected: left.clone(),
                        found: right.clone(),
                    },
                ));
                return None;
            }

            let mut new_params = Vec::with_capacity(lp.len());
            for (lpi, rpi) in lp.iter().zip(rp.iter()) {
                let unified_ty = unify_types(&lpi.ty, &rpi.ty, span, errors)?;
                new_params.push(FuncParam {
                    ty: unified_ty,
                    mutable: lpi.mutable,
                });
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
        ) => unify_named_type_args(
            *ln,
            la,
            *rn,
            ra,
            left,
            right,
            span,
            errors,
            |name, type_args| Struct { name, type_args },
        ),

        // arrays unify if the len matches and the element types can unify
        // Fixed(N) only unifies with Fixed(N) or Infer
        (Array { elem: le, len: ll }, Array { elem: re, len: rl }) => {
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

        // extern types unify only if they have the same name
        (Extern { name: ln }, Extern { name: rn }) => {
            if ln == rn {
                Some(Extern { name: *ln })
            } else {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::MismatchedTypes {
                        expected: left.clone(),
                        found: right.clone(),
                    },
                ));
                None
            }
        }

        // mismatched types report an error
        (l, r) => {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::MismatchedTypes {
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
