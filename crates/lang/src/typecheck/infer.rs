use std::collections::HashMap;

use internment::Intern;

use super::{
    constraint::{TypeRef, resolve_constraints},
    error::{Diagnostic, DiagnosticKind},
    expr::check_expr,
    types::{InferenceSlots, TypeChecker},
    unify::contains_infer,
    visit::{fold_type, walk_type_structure},
};
use crate::{
    ast::{
        ArrayLen, ConstParam, ConstParamId, ExprNode, Func, FuncParam, Ident, Mutability, Type,
        TypeParam, TypeVarId,
    },
    span::Span,
};

pub(super) type ConstInferenceSlots = HashMap<ConstParamId, Option<usize>>;

/// Builds a function type from the AST node
pub(super) fn type_from_fn(func: &Func) -> Type {
    Type::Func {
        params: func
            .params
            .iter()
            .map(|param| {
                FuncParam::new(
                    param.ty.clone(),
                    matches!(param.mutability, Mutability::Mutable),
                )
            })
            .collect(),
        ret: Box::new(func.ret.clone()),
    }
}

/// Substitutes type variables in a type with concrete types from the substitution map
/// fn(T) -> T where "T = int" then fn(int) -> int
pub fn subst_type(
    ty: &Type,
    subst: &HashMap<TypeVarId, Type>,
    const_subst: &HashMap<ConstParamId, usize>,
) -> Type {
    if subst.is_empty() && const_subst.is_empty() {
        return ty.clone();
    }
    fold_type(ty, &mut |t| match t {
        Type::Var(id) => subst.get(&id).cloned().unwrap_or(Type::Var(id)),
        Type::Array { elem, len } => {
            let new_len = match &len {
                ArrayLen::Param(id) => const_subst.get(id).copied().map_or(len, ArrayLen::Fixed),
                _ => len,
            };
            Type::Array { elem, len: new_len }
        }
        other => other,
    })
}

pub(super) fn build_subst(
    type_params: &[TypeParam],
    type_args: &[Type],
) -> HashMap<TypeVarId, Type> {
    type_params
        .iter()
        .zip(type_args.iter())
        .map(|(param, arg)| (param.id, arg.clone()))
        .collect()
}

pub fn build_const_subst(
    const_params: &[ConstParam],
    const_args: &[usize],
) -> HashMap<ConstParamId, usize> {
    const_params
        .iter()
        .zip(const_args.iter())
        .map(|(param, &arg)| (param.id, arg))
        .collect()
}

#[cfg(test)]
pub(super) fn instantiate_func_type(
    type_params: &[TypeParam],
    template: &Type,
    type_args: &[Type],
    span: Span,
    errors: &mut Vec<Diagnostic>,
) -> Option<Type> {
    let same_param_count = type_params.len() == type_args.len();
    if !same_param_count {
        errors.push(Diagnostic::new(
            span,
            DiagnosticKind::GenericArgNumMismatch {
                expected: type_params.len(),
                found: type_args.len(),
            },
        ));
        return None;
    }

    let subst = build_subst(type_params, type_args);

    Some(subst_type(template, &subst, &HashMap::new()))
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

pub(super) fn create_const_inference_slots(const_params: &[ConstParam]) -> ConstInferenceSlots {
    const_params.iter().map(|param| (param.id, None)).collect()
}

pub(super) fn build_param_ref(
    ty: &Type,
    slots: &InferenceSlots,
    const_slots: &ConstInferenceSlots,
    type_checker: &TypeChecker,
) -> TypeRef {
    if let Type::Var(id) = ty {
        slots
            .get(id)
            .copied()
            .map_or_else(|| TypeRef::concrete(ty), TypeRef::Var)
    } else {
        let subst: HashMap<TypeVarId, Type> = slots
            .iter()
            .filter_map(|(var_id, slot_name)| {
                let slot_ty = type_checker.get_var(*slot_name)?.ty.clone();
                Some((*var_id, slot_ty))
            })
            .collect();
        let const_subst: HashMap<ConstParamId, usize> = const_slots
            .iter()
            .filter_map(|(id, val)| Some((*id, (*val)?)))
            .collect();
        let resolved = subst_type(ty, &subst, &const_subst);
        TypeRef::concrete(&resolved)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_type_args_from_call(
    call_span: Span,
    type_params: &[TypeParam],
    const_params: &[ConstParam],
    param_template_types: &[Type],
    args: &[ExprNode],
    expected_on_mismatch: Type,
    ret_template: &Type,
    expected_ret: Option<&Type>,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<(Vec<Type>, Vec<usize>)> {
    let call_id = type_checker.next_call_id();
    let slots = create_inference_slots(type_params, type_checker, call_id);
    let mut const_slots = create_const_inference_slots(const_params);

    if args.len() != param_template_types.len() {
        errors.push(Diagnostic::new(
            call_span,
            DiagnosticKind::MismatchedTypes {
                expected: expected_on_mismatch,
                found: Type::Func {
                    params: vec![FuncParam::immut(Type::Infer); args.len()],
                    ret: Box::new(Type::Infer),
                },
            },
        ));
        return None;
    }

    for (arg_expr, param_ty) in args.iter().zip(param_template_types.iter()) {
        check_expr(arg_expr, type_checker, errors, None);

        let resolved_param_ty = type_checker.resolve_type(param_ty);
        if let Some((_, arg_ty)) = type_checker.get_type(arg_expr.node.id) {
            let arg_ty = arg_ty.clone();
            constrain_slots_from_type(
                &resolved_param_ty,
                &arg_ty,
                &slots,
                &mut const_slots,
                arg_expr.span,
                type_checker,
                errors,
            );
        }

        let arg_ref = TypeRef::Expr(arg_expr.node.id);
        let param_ref = build_param_ref(&resolved_param_ty, &slots, &const_slots, type_checker);
        type_checker.constrain_assignable(arg_expr.span, arg_ref, param_ref, errors);
    }

    if let Some(expected) = expected_ret {
        let resolved_ret_template = type_checker.resolve_type(ret_template);
        constrain_slots_from_type(
            &resolved_ret_template,
            expected,
            &slots,
            &mut const_slots,
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
        errors.push(Diagnostic::new(call_span, DiagnosticKind::UnresolvedInfer));
        return None;
    }

    let mut const_inference_failed = false;
    let mut inferred_const_args = Vec::with_capacity(const_params.len());
    for param in const_params {
        if let Some(Some(value)) = const_slots.get(&param.id) {
            inferred_const_args.push(*value);
        } else {
            const_inference_failed = true;
            inferred_const_args.push(0);
        }
    }

    if const_inference_failed {
        errors.push(Diagnostic::new(call_span, DiagnosticKind::UnresolvedInfer));
        return None;
    }

    Some((inferred_type_args, inferred_const_args))
}

pub(super) fn constrain_slots_from_type(
    template: &Type,
    expected: &Type,
    slots: &InferenceSlots,
    const_slots: &mut ConstInferenceSlots,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    match (template, expected) {
        (Type::Var(id), _) => {
            if let Some(slot_name) = slots.get(id) {
                let slot_ref = TypeRef::Var(*slot_name);
                let expected_ref = TypeRef::concrete(expected);
                type_checker.constrain_equal(span, slot_ref, expected_ref, errors);
            }
        }
        (Type::Array { elem: te, len: tl }, Type::Array { elem: ee, len: el }) => {
            constrain_slots_from_type(te, ee, slots, const_slots, span, type_checker, errors);
            if let (ArrayLen::Param(id), ArrayLen::Fixed(n)) = (tl, el) {
                match const_slots.get(id) {
                    Some(Some(existing)) if *existing != *n => {
                        errors.push(Diagnostic::new(
                            span,
                            DiagnosticKind::ConflictingConstInference {
                                first: *existing,
                                second: *n,
                            },
                        ));
                    }
                    Some(Some(_)) => {}
                    _ => {
                        const_slots.insert(*id, Some(*n));
                    }
                }
            }
        }
        // cross-shape coercions slice accepts array and list
        (Type::Slice { elem: te }, Type::Array { elem: ee, .. } | Type::List { elem: ee }) => {
            constrain_slots_from_type(te, ee, slots, const_slots, span, type_checker, errors);
        }
        // func needs custom error reporting for mutability mismatches
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
                if t.mutable != e.mutable {
                    errors.push(Diagnostic::new(
                        span,
                        DiagnosticKind::MismatchedTypes {
                            expected: Type::Func {
                                params: tp.clone(),
                                ret: tr.clone(),
                            },
                            found: Type::Func {
                                params: ep.clone(),
                                ret: er.clone(),
                            },
                        },
                    ));
                    return;
                }
                constrain_slots_from_type(
                    &t.ty,
                    &e.ty,
                    slots,
                    const_slots,
                    span,
                    type_checker,
                    errors,
                );
            }
            constrain_slots_from_type(tr, er, slots, const_slots, span, type_checker, errors);
        }
        // structural cases (List, Map, Slice-Slice, Struct, DataRef, Enum, Tuple, NamedTuple)
        _ => {
            walk_type_structure(template, expected, &mut |t, e| {
                constrain_slots_from_type(t, e, slots, const_slots, span, type_checker, errors);
                true
            });
        }
    }
}

pub(super) fn infer_const_args_from_checked_args(
    param_templates: &[Type],
    args: &[ExprNode],
    const_params: &[ConstParam],
    type_checker: &TypeChecker,
) -> Vec<usize> {
    if const_params.is_empty() {
        return vec![];
    }
    let mut const_slots = create_const_inference_slots(const_params);
    for (arg_expr, template) in args.iter().zip(param_templates.iter()) {
        if let Some((_, arg_ty)) = type_checker.get_type(arg_expr.node.id) {
            collect_const_from_types(template, arg_ty, &mut const_slots);
        }
    }
    const_params
        .iter()
        .map(|p| const_slots.get(&p.id).and_then(|v| *v).unwrap_or(0))
        .collect()
}

fn collect_const_from_types(template: &Type, concrete: &Type, slots: &mut ConstInferenceSlots) {
    if let (Type::Array { elem: te, len: tl }, Type::Array { elem: ee, len: el }) =
        (template, concrete)
    {
        if let (ArrayLen::Param(id), ArrayLen::Fixed(n)) = (tl, el) {
            slots.insert(*id, Some(*n));
        }
        collect_const_from_types(te, ee, slots);
        return;
    }

    walk_type_structure(template, concrete, &mut |t, c| {
        collect_const_from_types(t, c, slots);
        true
    });
}

pub fn resolve_type_param_names(ty: &Type, type_params: &[TypeParam]) -> Type {
    fold_type(ty, &mut |t| match t {
        Type::UnresolvedName(name) => {
            if let Some(param) = type_params.iter().find(|p| p.name == name) {
                Type::Var(param.id)
            } else {
                Type::UnresolvedName(name)
            }
        }
        other => other,
    })
}
