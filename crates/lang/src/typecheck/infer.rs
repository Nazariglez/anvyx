use crate::{
    ast::{ExprNode, Func, Ident, Type, TypeParam, TypeVarId},
    span::Span,
};
use internment::Intern;
use std::collections::HashMap;

use super::{
    constraint::{TypeRef, resolve_constraints},
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    types::{InferenceSlots, TypeChecker},
    unify::contains_infer,
};

/// Builds a fucntion type from the AST node
pub(super) fn type_from_fn(func: &Func) -> Type {
    Type::Func {
        params: func.params.iter().map(|param| param.ty.clone()).collect(),
        ret: Box::new(func.ret.clone()),
    }
}

/// Substitutes type variables in a type with concrete types from the substitution map
/// fn(T) -> T where "T = int" then fn(int) -> int
pub fn subst_type(ty: &Type, subst: &HashMap<TypeVarId, Type>) -> Type {
    use Type::*;
    match ty {
        Var(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Func { params, ret } => Func {
            params: params.iter().map(|p| subst_type(p, subst)).collect(),
            ret: subst_type(ret, subst).boxed(),
        },
        Tuple(elems) => Tuple(elems.iter().map(|e| subst_type(e, subst)).collect()),
        NamedTuple(fields) => NamedTuple(
            fields
                .iter()
                .map(|(n, t)| (*n, subst_type(t, subst)))
                .collect(),
        ),
        Struct { name, type_args } => Struct {
            name: *name,
            type_args: type_args.iter().map(|a| subst_type(a, subst)).collect(),
        },
        Enum { name, type_args } => Enum {
            name: *name,
            type_args: type_args.iter().map(|a| subst_type(a, subst)).collect(),
        },
        Array { elem, len } => Array {
            elem: subst_type(elem, subst).boxed(),
            len: *len,
        },
        ArrayView { elem } => ArrayView {
            elem: subst_type(elem, subst).boxed(),
        },
        List { elem } => List {
            elem: subst_type(elem, subst).boxed(),
        },
        Map { key, value } => Map {
            key: subst_type(key, subst).boxed(),
            value: subst_type(value, subst).boxed(),
        },
        _ => ty.clone(),
    }
}

/// Instantiates a generic function type with explicit type arguments
pub(super) fn instantiate_func_type(
    type_params: &[TypeParam],
    template: &Type,
    type_args: &[Type],
    span: Span,
    errors: &mut Vec<TypeErr>,
) -> Option<Type> {
    let same_param_count = type_params.len() == type_args.len();
    if !same_param_count {
        errors.push(TypeErr::new(
            span,
            TypeErrKind::GenericArgNumMismatch {
                expected: type_params.len(),
                found: type_args.len(),
            },
        ));
        return None;
    }

    let subst = type_params
        .iter()
        .zip(type_args.iter())
        .map(|(param, arg)| (param.id, arg.clone()))
        .collect();

    Some(subst_type(template, &subst))
}

/// Creates inference slots for a generic function call
/// for each type parameter, this creates a synthetic variable
/// in the type checker initialized to Type::Infer
pub(super) fn create_inference_slots(
    type_params: &[TypeParam],
    type_checker: &mut TypeChecker,
    call_id: usize,
) -> InferenceSlots {
    type_params
        .iter()
        .map(|param| {
            // lets use as name _infer_<name>_<id>_<call_id>
            let name = format!("__infer_{}_{}_{}", param.name, param.id.0, call_id);
            let infer_var_name = Ident(Intern::new(name));

            // initialize the inference slot
            type_checker.set_var(infer_var_name, Type::Infer, true);

            // map the type variable id to its synthetic variable name
            (param.id, infer_var_name)
        })
        .collect()
}

/// Converts 'Type::Var' references to 'TypeRef' for constraints
pub(super) fn type_to_ref_with_inference(ty: &Type, slots: &InferenceSlots) -> TypeRef {
    match ty {
        // use the synthetic variable ref for type variables
        Type::Var(id) => slots
            .get(id)
            .cloned()
            .map(TypeRef::Var)
            .unwrap_or_else(|| TypeRef::Concrete(ty.clone())),

        // anything else just use concrete
        _ => TypeRef::Concrete(substitute_vars_with_infer(ty, slots)),
    }
}

/// Substitutes 'Type::Var' with 'Type::Infer' for nested positions in compound types
/// imagine infer T from fn my_fn(T?) -> T then my_fn(10) and we should infer int from T?
pub(super) fn substitute_vars_with_infer(ty: &Type, slots: &InferenceSlots) -> Type {
    match ty {
        Type::Var(id) if slots.contains_key(id) => Type::Infer,
        Type::Func { params, ret } => Type::Func {
            params: params
                .iter()
                .map(|p| substitute_vars_with_infer(p, slots))
                .collect(),
            ret: substitute_vars_with_infer(ret, slots).boxed(),
        },
        Type::Tuple(elems) => Type::Tuple(
            elems
                .iter()
                .map(|e| substitute_vars_with_infer(e, slots))
                .collect(),
        ),
        Type::NamedTuple(fields) => Type::NamedTuple(
            fields
                .iter()
                .map(|(n, t)| (*n, substitute_vars_with_infer(t, slots)))
                .collect(),
        ),
        Type::Struct { name, type_args } => Type::Struct {
            name: *name,
            type_args: type_args
                .iter()
                .map(|a| substitute_vars_with_infer(a, slots))
                .collect(),
        },
        Type::Enum { name, type_args } => Type::Enum {
            name: *name,
            type_args: type_args
                .iter()
                .map(|a| substitute_vars_with_infer(a, slots))
                .collect(),
        },
        Type::Array { elem, len } => Type::Array {
            elem: substitute_vars_with_infer(elem, slots).boxed(),
            len: *len,
        },
        Type::ArrayView { elem } => Type::ArrayView {
            elem: substitute_vars_with_infer(elem, slots).boxed(),
        },
        Type::List { elem } => Type::List {
            elem: substitute_vars_with_infer(elem, slots).boxed(),
        },
        Type::Map { key, value } => Type::Map {
            key: substitute_vars_with_infer(key, slots).boxed(),
            value: substitute_vars_with_infer(value, slots).boxed(),
        },
        _ => ty.clone(),
    }
}

pub(super) fn infer_type_args_from_call(
    call_span: Span,
    type_params: &[TypeParam],
    param_template_types: &[Type],
    args: &[ExprNode],
    expected_on_mismatch: Type,
    ret_template: &Type,
    expected_ret: Option<&Type>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Option<Vec<Type>> {
    let call_id = type_checker.next_call_id();
    let slots = create_inference_slots(type_params, type_checker, call_id);

    if args.len() != param_template_types.len() {
        errors.push(TypeErr::new(
            call_span,
            TypeErrKind::MismatchedTypes {
                expected: expected_on_mismatch,
                found: Type::Func {
                    params: vec![Type::Infer; args.len()],
                    ret: Box::new(Type::Infer),
                },
            },
        ));
        return None;
    }

    for (arg_expr, param_ty) in args.iter().zip(param_template_types.iter()) {
        check_expr(arg_expr, type_checker, errors, None);
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let param_ref = type_to_ref_with_inference(param_ty, &slots);
        type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
    }

    if let Some(expected) = expected_ret {
        constrain_type_vars_from_expected(
            ret_template,
            expected,
            &slots,
            call_span,
            type_checker,
            errors,
        );
    }

    resolve_constraints(type_checker, errors);

    let mut inferred_type_args = Vec::with_capacity(type_params.len());
    let mut inference_failed = false;
    for param in type_params {
        let slot_ident = slots.get(&param.id);
        let slot_info = slot_ident.and_then(|si| type_checker.get_var(*si));
        let slot_ty = slot_info.map(|info| &info.ty);

        let ty = slot_ty
            .filter(|ty| !contains_infer(ty))
            .cloned()
            .unwrap_or_else(|| {
                inference_failed = true;
                Type::Infer
            });

        inferred_type_args.push(ty);
    }

    if inference_failed {
        errors.push(TypeErr::new(call_span, TypeErrKind::UnresolvedInfer));
        return None;
    }

    Some(inferred_type_args)
}

fn constrain_type_vars_from_expected(
    template: &Type,
    expected: &Type,
    slots: &InferenceSlots,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    match (template, expected) {
        (Type::Var(id), _) => {
            if let Some(slot_name) = slots.get(id) {
                let slot_ref = TypeRef::Var(*slot_name);
                let expected_ref = TypeRef::Concrete(expected.clone());
                type_checker.constrain_assignable(span, slot_ref, expected_ref, errors);
            }
        }
        (Type::Map { key: tk, value: tv }, Type::Map { key: ek, value: ev }) => {
            constrain_type_vars_from_expected(tk, ek, slots, span, type_checker, errors);
            constrain_type_vars_from_expected(tv, ev, slots, span, type_checker, errors);
        }
        (Type::List { elem: te }, Type::List { elem: ee }) => {
            constrain_type_vars_from_expected(te, ee, slots, span, type_checker, errors);
        }
        (Type::Array { elem: te, .. }, Type::Array { elem: ee, .. }) => {
            constrain_type_vars_from_expected(te, ee, slots, span, type_checker, errors);
        }
        (Type::ArrayView { elem: te }, Type::ArrayView { elem: ee }) => {
            constrain_type_vars_from_expected(te, ee, slots, span, type_checker, errors);
        }
        (
            Type::Func {
                params: tp,
                ret: tr,
            },
            Type::Func {
                params: ep,
                ret: er,
            },
        ) => {
            for (t, e) in tp.iter().zip(ep.iter()) {
                constrain_type_vars_from_expected(t, e, slots, span, type_checker, errors);
            }
            constrain_type_vars_from_expected(tr, er, slots, span, type_checker, errors);
        }
        (Type::Tuple(te), Type::Tuple(ee)) => {
            for (t, e) in te.iter().zip(ee.iter()) {
                constrain_type_vars_from_expected(t, e, slots, span, type_checker, errors);
            }
        }
        (Type::NamedTuple(tf), Type::NamedTuple(ef)) => {
            for ((_, t), (_, e)) in tf.iter().zip(ef.iter()) {
                constrain_type_vars_from_expected(t, e, slots, span, type_checker, errors);
            }
        }
        (
            Type::Struct {
                name: tn,
                type_args: ta,
            },
            Type::Struct {
                name: en,
                type_args: ea,
            },
        ) if tn == en => {
            for (t, e) in ta.iter().zip(ea.iter()) {
                constrain_type_vars_from_expected(t, e, slots, span, type_checker, errors);
            }
        }
        (
            Type::Enum {
                name: tn,
                type_args: ta,
            },
            Type::Enum {
                name: en,
                type_args: ea,
            },
        ) if tn == en => {
            for (t, e) in ta.iter().zip(ea.iter()) {
                constrain_type_vars_from_expected(t, e, slots, span, type_checker, errors);
            }
        }
        _ => {}
    }
}
