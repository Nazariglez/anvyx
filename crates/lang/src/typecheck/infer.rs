use crate::{
    ast::{ExprNode, Func, Ident, Type, TypeParam, TypeVarId},
    span::Span,
};
use internment::Intern;
use std::collections::HashMap;

use super::{
    constraint::{resolve_constraints, TypeRef},
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
pub(super) fn subst_type(ty: &Type, subst: &HashMap<TypeVarId, Type>) -> Type {
    use Type::*;
    match ty {
        Var(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Optional(inner) => Optional(subst_type(inner, subst).boxed()),
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
        Array { elem, len } => Array {
            elem: subst_type(elem, subst).boxed(),
            len: *len,
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
            type_checker.set_var(infer_var_name, Type::Infer);

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
        Type::Optional(inner) => Type::Optional(substitute_vars_with_infer(inner, slots).boxed()),
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
        Type::Array { elem, len } => Type::Array {
            elem: substitute_vars_with_infer(elem, slots).boxed(),
            len: *len,
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
        check_expr(arg_expr, type_checker, errors);
        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let param_ref = type_to_ref_with_inference(param_ty, &slots);
        type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
    }

    resolve_constraints(type_checker, errors);

    let mut inferred_type_args = Vec::with_capacity(type_params.len());
    let mut inference_failed = false;
    for param in type_params {
        let slot_var = slots
            .get(&param.id)
            .and_then(|slot_ident| type_checker.get_var(*slot_ident));

        let ty = slot_var
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
