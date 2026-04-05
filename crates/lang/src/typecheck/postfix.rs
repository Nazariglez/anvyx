use std::collections::HashMap;

use internment::Intern;

use super::{
    call::{check_instantiated_body, *},
    error::{Diagnostic, DiagnosticKind},
    expr::check_expr,
    infer::{build_const_subst, build_subst, resolve_type_param_names, subst_type},
    stmt::extend_base_key,
    types::{
        ExtendEntry, ExtendMethodDef, ExtendSpecKey, GenericExtendTemplate, InstantiationContext,
        PostfixNodeRef, TypeChecker, type_field_on_base, type_index_on_base, unwrap_opt_typ,
    },
    visit::fold_type,
};
use crate::{
    ast::{
        ArrayLen, CallNode, ConstParam, ConstParamId, ExprId, ExprKind, ExprNode, FuncParam, Ident,
        Mutability, Type, TypeParam,
    },
    backend_names,
    span::Span,
};

pub(super) fn collect_postfix_chain(expr: &ExprNode) -> (&ExprNode, Vec<PostfixNodeRef<'_>>) {
    let mut chain = vec![];
    let mut current = expr;

    loop {
        match &current.node.kind {
            ExprKind::Field(field_node) => {
                chain.push(PostfixNodeRef::Field {
                    expr_id: current.node.id,
                    node: field_node,
                });
                current = field_node.node.target.as_ref();
            }
            ExprKind::Index(index_node) => {
                chain.push(PostfixNodeRef::Index {
                    expr_id: current.node.id,
                    node: index_node,
                });
                current = index_node.node.target.as_ref();
            }
            ExprKind::Call(call_node) => {
                chain.push(PostfixNodeRef::Call {
                    expr_id: current.node.id,
                    node: call_node,
                });
                current = call_node.node.func.as_ref();
            }
            _ => break,
        }
    }

    chain.reverse();
    (current, chain)
}

pub(super) fn check_postfix_chain(
    expr_node: &ExprNode,
    base: &ExprNode,
    chain: &[PostfixNodeRef<'_>],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Type {
    // handle type name postfixes without looking them up as value variables
    if let Some(outcome) = try_type_name_dispatch(base, chain, type_checker, errors, expected) {
        let (ty, steps_consumed, safe) = outcome;
        let current_ty = if safe {
            Type::option_of(ty.clone())
        } else {
            ty.clone()
        };
        // set types for all consumed ops
        for op in &chain[..steps_consumed] {
            set_op_type(op, current_ty.clone(), type_checker);
        }
        // if all steps were consumed, we are done
        if steps_consumed == chain.len() {
            type_checker.set_type(expr_node.node.id, current_ty.clone(), expr_node.span);
            return current_ty;
        }
        // otherwise continue the chain from where we left off
        return continue_postfix_chain(
            expr_node,
            base,
            &chain[steps_consumed..],
            current_ty,
            safe,
            type_checker,
            errors,
            expected,
        );
    }

    let current_ty = check_expr(base, type_checker, errors, None);
    continue_postfix_chain(
        expr_node,
        base,
        chain,
        current_ty,
        false,
        type_checker,
        errors,
        expected,
    )
}

#[allow(clippy::too_many_arguments)]
fn continue_postfix_chain(
    expr_node: &ExprNode,
    base: &ExprNode,
    chain: &[PostfixNodeRef<'_>],
    initial_ty: Type,
    initial_optional: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Type {
    let mut current_ty = initial_ty;
    let mut chain_is_optional = initial_optional;
    let mut i = 0;

    while i < chain.len() {
        if let Some(method_result) = handle_method_call_if_applicable(
            chain,
            i,
            &current_ty,
            chain_is_optional,
            base,
            type_checker,
            errors,
        ) {
            match method_result {
                MethodCallOutcome::Handled {
                    ty,
                    chain_optional,
                    next_index,
                    call_expr,
                    call_span,
                } => {
                    current_ty = ty;
                    chain_is_optional = chain_optional;
                    type_checker.set_type(call_expr, current_ty.clone(), call_span);
                    i = next_index;
                    continue;
                }
                MethodCallOutcome::Abort => {
                    type_checker.set_type(expr_node.node.id, Type::Infer, expr_node.span);
                    return Type::Infer;
                }
                MethodCallOutcome::NotMethod => {
                    // fallthrough normal handling
                }
            }
        }

        let op = &chain[i];
        let op_safe = op.safe();
        let mut base_ty = if chain_is_optional {
            unwrap_opt_typ(&current_ty).clone()
        } else {
            current_ty.clone()
        };

        if op_safe {
            if let Some(inner) = current_ty.option_inner() {
                base_ty = inner.clone();
            } else {
                errors.push(
                    Diagnostic::new(
                        op.span(),
                        DiagnosticKind::OptionalChainingOnNonOpt {
                            found: current_ty.clone(),
                        },
                    )
                    .with_help("remove the `?` or make the base type optional"),
                );
                mark_remaining_ops_infer(chain, i, type_checker);
                type_checker.set_type(expr_node.node.id, Type::Infer, expr_node.span);
                return Type::Infer;
            }
        }

        let is_last = i + 1 == chain.len();
        let op_expected = if is_last { expected } else { None };
        let op_result_inner = apply_postfix_op(op, &base_ty, type_checker, errors, op_expected);
        // map indexing already returns an optional value, so dont wrap it twice
        let map_index = matches!(
            (op, &base_ty),
            (PostfixNodeRef::Index { .. }, Type::Map { .. })
        ) && !matches!(op_result_inner, Type::Infer);
        if op_safe || chain_is_optional {
            current_ty = if map_index {
                op_result_inner.clone()
            } else {
                Type::option_of(op_result_inner.clone())
            };
            chain_is_optional = true;
        } else {
            current_ty = op_result_inner.clone();
        }

        set_op_type(op, current_ty.clone(), type_checker);
        i += 1;
    }

    type_checker.set_type(expr_node.node.id, current_ty.clone(), expr_node.span);
    current_ty
}

fn is_named_type(ty: &Type) -> bool {
    ty.is_aggregate() || matches!(ty, Type::Enum { .. })
}

fn named_type_parts(ty: &Type) -> (Ident, &[Type]) {
    if let Some(agg) = ty.as_aggregate() {
        return (agg.name, agg.type_args);
    }
    match ty {
        Type::Enum { name, type_args } => (*name, type_args.as_slice()),
        _ => unreachable!("called named_type_parts on non-named type"),
    }
}

/// Check whether "receiver" matches "pattern", binding type params and fixed array lengths along the way
fn match_type_pattern(
    receiver: &Type,
    pattern: &Type,
    type_params: &[TypeParam],
    bindings: &mut HashMap<Ident, Type>,
    const_params: &[ConstParam],
    const_bindings: &mut HashMap<ConstParamId, usize>,
) -> bool {
    // pattern is a type variable, bind or check consistency
    if let Type::UnresolvedName(name) = pattern {
        if type_params.iter().any(|p| p.name == *name) {
            if let Some(existing) = bindings.get(name) {
                return existing == receiver;
            }
            bindings.insert(*name, receiver.clone());
            return true;
        }
        // unresolved name that is not a declared type param, treat as mismatch
        return false;
    }

    match (receiver, pattern) {
        // leaf types
        (Type::Int, Type::Int)
        | (Type::Float, Type::Float)
        | (Type::Double, Type::Double)
        | (Type::Bool, Type::Bool)
        | (Type::String, Type::String)
        | (Type::Void, Type::Void)
        | (Type::Any, Type::Any) => true,

        (Type::Extern { name: n1 }, Type::Extern { name: n2 }) => n1 == n2,

        // match named types by name and type args only, since the concrete variant may not be resolved yet
        (r, p) if is_named_type(r) && is_named_type(p) => {
            let (n1, a1) = named_type_parts(r);
            let (n2, a2) = named_type_parts(p);
            n1 == n2
                && a1.len() == a2.len()
                && a1.iter().zip(a2.iter()).all(|(rv, pv)| {
                    match_type_pattern(rv, pv, type_params, bindings, const_params, const_bindings)
                })
        }

        (Type::List { elem: e1 }, Type::List { elem: e2 }) => {
            match_type_pattern(e1, e2, type_params, bindings, const_params, const_bindings)
        }

        (Type::Map { key: k1, value: v1 }, Type::Map { key: k2, value: v2 }) => {
            match_type_pattern(k1, k2, type_params, bindings, const_params, const_bindings)
                && match_type_pattern(v1, v2, type_params, bindings, const_params, const_bindings)
        }

        (Type::Tuple(f1), Type::Tuple(f2)) => {
            f1.len() == f2.len()
                && f1.iter().zip(f2.iter()).all(|(r, p)| {
                    match_type_pattern(r, p, type_params, bindings, const_params, const_bindings)
                })
        }

        (Type::Array { elem: e1, len: l1 }, Type::Array { elem: e2, len: l2 }) => {
            let len_ok = match (l1, l2) {
                (ArrayLen::Fixed(n), ArrayLen::Param(id))
                    if const_params.iter().any(|p| p.id == *id) =>
                {
                    if let Some(&existing) = const_bindings.get(id) {
                        existing == *n
                    } else {
                        const_bindings.insert(*id, *n);
                        true
                    }
                }
                _ => l1 == l2,
            };
            len_ok
                && match_type_pattern(e1, e2, type_params, bindings, const_params, const_bindings)
        }

        _ => false,
    }
}

enum Specificity {
    MoreSpecific,
    LessSpecific,
    Equal,
    Incomparable,
}

fn substitute_fresh(pattern: &Type, type_params: &[TypeParam]) -> Type {
    fold_type(pattern, &mut |ty| match ty {
        Type::UnresolvedName(name) if type_params.iter().any(|p| p.name == name) => Type::Extern {
            name: Ident(Intern::new(format!("__Fresh_{name}"))),
        },
        other => other,
    })
}

fn compare_specificity(
    a_target: &Type,
    a_params: &[TypeParam],
    a_const_params: &[ConstParam],
    b_target: &Type,
    b_params: &[TypeParam],
    b_const_params: &[ConstParam],
) -> Specificity {
    let a_concrete = substitute_fresh(a_target, a_params);
    let b_concrete = substitute_fresh(b_target, b_params);

    let mut a_bindings = HashMap::new();
    let a_fits_b = match_type_pattern(
        &a_concrete,
        b_target,
        b_params,
        &mut a_bindings,
        b_const_params,
        &mut HashMap::new(),
    );

    let mut b_bindings = HashMap::new();
    let b_fits_a = match_type_pattern(
        &b_concrete,
        a_target,
        a_params,
        &mut b_bindings,
        a_const_params,
        &mut HashMap::new(),
    );

    match (a_fits_b, b_fits_a) {
        (true, true) => Specificity::Equal,
        (true, false) => Specificity::MoreSpecific,
        (false, true) => Specificity::LessSpecific,
        (false, false) => Specificity::Incomparable,
    }
}

#[allow(clippy::too_many_arguments)]
fn try_specialize_extend(
    receiver_ty: &Type,
    type_args: &[Type],
    const_args: &[usize],
    template: &GenericExtendTemplate,
    method_name: Ident,
    call_node: &CallNode,
    receiver: Option<&ExprNode>,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let Some(base_name) = extend_base_key(receiver_ty) else {
        return Type::Infer;
    };

    let cache_key = ExtendSpecKey {
        base_name,
        method_name,
        type_args: type_args.to_vec(),
        const_args: const_args.to_vec(),
        target_type: template.target_type.clone(),
    };

    let context_name = format!(
        "extend {}<{}>",
        base_name,
        type_args
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );

    let subst = build_subst(&template.type_params, type_args);
    let const_subst = build_const_subst(&template.const_params, const_args);
    let method = &template.method.node;
    let type_params = &template.type_params;

    let specialized_params: Vec<Type> = method
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i == 0 {
                receiver_ty.clone()
            } else {
                let resolved =
                    resolve_type_param_names(&type_checker.resolve_type(&p.ty), type_params);
                subst_type(&resolved, &subst, &const_subst)
            }
        })
        .collect();

    let ret_ty = if let Some(cached) = type_checker.extend_spec_cache.get(&cache_key).cloned() {
        report_cached_spec_error(
            &cached,
            &context_name,
            &template.type_params,
            type_args,
            span,
            errors,
        );
        let mangled =
            backend_names::encode_extend_specialization_name(&cache_key, &template.source_module);
        type_checker
            .extend_call_targets
            .insert(call_node.node.func.node.id, mangled);
        type_checker
            .store_extend_ref_mask(call_node.node.func.node.id, &template.method.node.params);
        cached.ret_ty
    } else {
        let raw_ret =
            resolve_type_param_names(&type_checker.resolve_type(&method.ret), type_params);
        let specialized_ret = subst_type(&raw_ret, &subst, &const_subst);

        let mut body_params: Vec<(Ident, Type, bool)> = method
            .params
            .iter()
            .zip(specialized_params.iter())
            .map(|(p, ty)| {
                (
                    p.name,
                    ty.clone(),
                    matches!(p.mutability, Mutability::Mutable),
                )
            })
            .collect();
        for param in &template.const_params {
            body_params.push((param.name, Type::Int, false));
        }

        let owned_module_env = if template.source_module.is_empty() {
            None
        } else {
            type_checker
                .module_check_contexts
                .get(&template.source_module)
                .cloned()
        };

        let ictx = InstantiationContext {
            module_env: owned_module_env.as_ref(),
            params: body_params,
            ret_ty: specialized_ret.clone(),
            method_ctx: None,
        };
        let mut body_errors = vec![];
        let spec_result =
            check_instantiated_body(&ictx, &method.body, span, type_checker, &mut body_errors)
                .into_spec_result();

        type_checker
            .extend_spec_cache
            .insert(cache_key.clone(), spec_result);

        report_instantiation_errors(
            body_errors,
            &context_name,
            &template.type_params,
            type_args,
            span,
            errors,
        );

        let mangled =
            backend_names::encode_extend_specialization_name(&cache_key, &template.source_module);
        type_checker
            .extend_call_targets
            .insert(call_node.node.func.node.id, mangled);
        type_checker
            .store_extend_ref_mask(call_node.node.func.node.id, &template.method.node.params);

        specialized_ret
    };

    check_call_signature(
        call_node.span,
        &specialized_params[1..],
        specialized_params.len() - 1,
        &ret_ty,
        &call_node.node.args,
        type_checker,
        errors,
    );
    check_extend_var_params(
        &template.method.node.params,
        receiver,
        method_name,
        call_node,
        type_checker,
        errors,
    );

    ret_ty
}

fn check_extend_var_params(
    params: &[crate::ast::Param],
    receiver: Option<&ExprNode>,
    method_name: Ident,
    call_node: &CallNode,
    type_checker: &TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    check_var_param_args(
        params[1..].iter().map(|p| (p.name, p.mutability)),
        &call_node.node.args,
        type_checker,
        errors,
    );
    if params
        .first()
        .is_some_and(|p| p.mutability == Mutability::Mutable)
        && let Some(recv) = receiver
    {
        let type_label = Ident(Intern::new(params[0].ty.to_string()));
        check_receiver_mutability(recv, type_label, method_name, type_checker, errors);
        check_var_self_aliasing(
            recv,
            params[1..].iter().map(|p| (p.name, p.mutability)),
            &call_node.node.args,
            errors,
        );
    }
}

fn check_extend_call_impl(
    def: &ExtendMethodDef,
    call_node: &CallNode,
    receiver: Option<&ExprNode>,
    method_name: Ident,
    skip_receiver: bool,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    let param_start = usize::from(skip_receiver);
    let param_types: Vec<Type> = def.params[param_start..]
        .iter()
        .map(|p| p.ty.clone())
        .collect();
    type_checker
        .extend_call_targets
        .insert(call_node.node.func.node.id, def.internal_name);
    type_checker.store_extend_ref_mask(call_node.node.func.node.id, &def.params);
    let result = check_call_signature(
        call_node.span,
        &param_types,
        param_types.len(),
        &def.ret,
        &call_node.node.args,
        type_checker,
        errors,
    );
    if skip_receiver {
        check_extend_var_params(
            &def.params,
            receiver,
            method_name,
            call_node,
            type_checker,
            errors,
        );
    } else {
        check_var_param_args(
            def.params.iter().map(|p| (p.name, p.mutability)),
            &call_node.node.args,
            type_checker,
            errors,
        );
    }
    result
}

fn check_extend_call(
    def: &ExtendMethodDef,
    call_node: &CallNode,
    receiver: Option<&ExprNode>,
    method_name: Ident,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    check_extend_call_impl(
        def,
        call_node,
        receiver,
        method_name,
        true,
        type_checker,
        errors,
    )
}

fn check_extend_qualified_call(
    def: &ExtendMethodDef,
    call_node: &CallNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    check_extend_call_impl(
        def,
        call_node,
        None,
        Ident(Intern::new(String::new())),
        false,
        type_checker,
        errors,
    )
}

/// Look up extend methods for `receiver_ty.method_name(...)`. Returns Some(ty) if found,
/// None if no extend entry exists. Reports AmbiguousExtendMethod if multiple modules define it.
fn resolve_extend_method(
    receiver_ty: &Type,
    method_name: Ident,
    call_node: &CallNode,
    receiver: Option<&ExprNode>,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<Type> {
    let entries = type_checker
        .get_extend_methods(receiver_ty, method_name)
        .to_vec();

    // Dedup by source_module — multiple entries from the same module count as one candidate
    let mut unique: Vec<&ExtendEntry> = vec![];
    let mut seen: Vec<&Vec<String>> = vec![];
    for entry in &entries {
        if !seen.contains(&&entry.source_module) {
            seen.push(&entry.source_module);
            unique.push(entry);
        }
    }

    match unique.as_slice() {
        [] => try_resolve_generic_extend(
            receiver_ty,
            method_name,
            call_node,
            receiver,
            span,
            type_checker,
            errors,
        ),
        [entry] => Some(check_extend_call(
            &entry.def,
            call_node,
            receiver,
            method_name,
            type_checker,
            errors,
        )),
        _ => {
            let candidates = unique.iter().map(|e| e.binding).collect();
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::AmbiguousExtendMethod {
                    ty: receiver_ty.clone(),
                    method: method_name,
                    candidates,
                },
            ));
            Some(Type::Infer)
        }
    }
}

fn try_resolve_generic_extend(
    receiver_ty: &Type,
    method_name: Ident,
    call_node: &CallNode,
    receiver: Option<&ExprNode>,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<Type> {
    let base_key = extend_base_key(receiver_ty)?;

    let key = (base_key, method_name);
    let templates = type_checker.ctx.generic_extend_templates.get(&key)?.clone();

    // try matching each templates target_type pattern against the receiver
    let mut matches: Vec<(GenericExtendTemplate, Vec<Type>, Vec<usize>)> = vec![];
    for template in &templates {
        let mut bindings = HashMap::new();
        let mut const_bindings = HashMap::new();
        if match_type_pattern(
            receiver_ty,
            &template.target_type,
            &template.type_params,
            &mut bindings,
            &template.const_params,
            &mut const_bindings,
        ) {
            let type_args: Vec<Type> = template
                .type_params
                .iter()
                .map(|p| bindings.get(&p.name).cloned().unwrap_or(Type::Infer))
                .collect();
            let const_args: Vec<usize> = template
                .const_params
                .iter()
                .map(|p| const_bindings.get(&p.id).copied().unwrap_or(0))
                .collect();
            matches.push((template.clone(), type_args, const_args));
        }
    }

    if matches.is_empty() {
        return None;
    }
    if matches.len() == 1 {
        let (template, type_args, const_args) = matches.remove(0);
        return Some(try_specialize_extend(
            receiver_ty,
            &type_args,
            &const_args,
            &template,
            method_name,
            call_node,
            receiver,
            span,
            type_checker,
            errors,
        ));
    }

    // find the most specific match via tournament
    let mut best_idx = 0;
    for i in 1..matches.len() {
        match compare_specificity(
            &matches[best_idx].0.target_type,
            &matches[best_idx].0.type_params,
            &matches[best_idx].0.const_params,
            &matches[i].0.target_type,
            &matches[i].0.type_params,
            &matches[i].0.const_params,
        ) {
            Specificity::LessSpecific | Specificity::Equal => best_idx = i,
            Specificity::MoreSpecific => {}
            Specificity::Incomparable => {
                let candidates = matches.iter().map(|(t, _, _)| t.binding).collect();
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::AmbiguousExtendMethod {
                        ty: receiver_ty.clone(),
                        method: method_name,
                        candidates,
                    },
                ));
                return Some(Type::Infer);
            }
        }
    }

    // pairwise verification, tournament may miss transitive incomparability
    for i in 0..matches.len() {
        if i == best_idx {
            continue;
        }
        match compare_specificity(
            &matches[best_idx].0.target_type,
            &matches[best_idx].0.type_params,
            &matches[best_idx].0.const_params,
            &matches[i].0.target_type,
            &matches[i].0.type_params,
            &matches[i].0.const_params,
        ) {
            Specificity::MoreSpecific => {}
            Specificity::Equal => {
                // if two matches are equally specific only cross-module ties are ambiguous
                if matches[best_idx].0.source_module != matches[i].0.source_module {
                    let candidates = matches.iter().map(|(t, _, _)| t.binding).collect();
                    errors.push(Diagnostic::new(
                        span,
                        DiagnosticKind::AmbiguousExtendMethod {
                            ty: receiver_ty.clone(),
                            method: method_name,
                            candidates,
                        },
                    ));
                    return Some(Type::Infer);
                }
            }
            _ => {
                let candidates = matches.iter().map(|(t, _, _)| t.binding).collect();
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::AmbiguousExtendMethod {
                        ty: receiver_ty.clone(),
                        method: method_name,
                        candidates,
                    },
                ));
                return Some(Type::Infer);
            }
        }
    }

    let (template, type_args, const_args) = matches.swap_remove(best_idx);
    Some(try_specialize_extend(
        receiver_ty,
        &type_args,
        &const_args,
        &template,
        method_name,
        call_node,
        receiver,
        span,
        type_checker,
        errors,
    ))
}

pub(super) enum MethodCallOutcome {
    NotMethod,
    Handled {
        ty: Type,
        chain_optional: bool,
        next_index: usize,
        call_expr: ExprId,
        call_span: Span,
    },
    Abort,
}

fn handled_method_result(
    method_ret: Type,
    op_safe: bool,
    chain_is_optional: bool,
    index: usize,
    call_op: PostfixNodeRef<'_>,
) -> MethodCallOutcome {
    let mut result_ty = method_ret;
    let mut chain_optional = chain_is_optional;
    if op_safe || chain_is_optional {
        chain_optional = true;
        result_ty = Type::option_of(result_ty);
    }
    MethodCallOutcome::Handled {
        ty: result_ty,
        chain_optional,
        next_index: index + 2,
        call_expr: call_op.expr_id(),
        call_span: call_op.span(),
    }
}

fn handled_infer(
    op_safe: bool,
    chain_is_optional: bool,
    index: usize,
    call_op: PostfixNodeRef<'_>,
) -> MethodCallOutcome {
    MethodCallOutcome::Handled {
        ty: Type::Infer,
        chain_optional: chain_is_optional || op_safe,
        next_index: index + 2,
        call_expr: call_op.expr_id(),
        call_span: call_op.span(),
    }
}

#[allow(clippy::too_many_arguments)]
fn try_struct_method(
    struct_name: Ident,
    struct_type_args: &[Type],
    detection_ty: &Type,
    field_node: &crate::ast::FieldAccessNode,
    call_node: &CallNode,
    call_op: PostfixNodeRef<'_>,
    op_safe: bool,
    chain_is_optional: bool,
    index: usize,
    chain: &[PostfixNodeRef<'_>],
    base: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<MethodCallOutcome> {
    let Some(struct_def) = type_checker.get_struct(struct_name).cloned() else {
        errors.push(Diagnostic::new(
            field_node.span,
            DiagnosticKind::UnknownStruct { name: struct_name },
        ));
        return Some(handled_infer(op_safe, chain_is_optional, index, call_op));
    };

    let method_name = field_node.node.field;
    let is_fn_field = struct_def
        .fields
        .iter()
        .any(|f| f.name == method_name && f.ty.is_func());
    if is_fn_field {
        return None;
    }

    let has_prior_call = chain[..index]
        .iter()
        .any(|op| matches!(op, PostfixNodeRef::Call { .. }));
    let effective_receiver = if has_prior_call { None } else { Some(base) };

    let method_ret = if struct_def.methods.contains_key(&method_name) {
        check_instance_method_call(
            call_node,
            struct_name,
            method_name,
            struct_type_args,
            &struct_def,
            effective_receiver,
            type_checker,
            errors,
        )
    } else if let Some(extend_ret) = resolve_extend_method(
        detection_ty,
        method_name,
        call_node,
        effective_receiver,
        field_node.span,
        type_checker,
        errors,
    ) {
        extend_ret
    } else {
        check_instance_method_call(
            call_node,
            struct_name,
            method_name,
            struct_type_args,
            &struct_def,
            effective_receiver,
            type_checker,
            errors,
        )
    };

    Some(handled_method_result(
        method_ret,
        op_safe,
        chain_is_optional,
        index,
        call_op,
    ))
}

#[allow(clippy::too_many_arguments)]
fn try_extern_method(
    extern_name: Ident,
    detection_ty: &Type,
    field_node: &crate::ast::FieldAccessNode,
    call_node: &CallNode,
    call_op: PostfixNodeRef<'_>,
    op_safe: bool,
    chain_is_optional: bool,
    index: usize,
    base: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<MethodCallOutcome> {
    let extern_def = type_checker.get_extern_type(extern_name).cloned()?;

    let method_name = field_node.node.field;
    let Some(method) = extern_def.methods.get(&method_name) else {
        if extern_def.statics.contains_key(&method_name) {
            errors.push(Diagnostic::new(
                call_node.span,
                DiagnosticKind::StaticMethodOnValue {
                    struct_name: extern_name,
                    method: method_name,
                },
            ));
        } else if let Some(extend_ret) = resolve_extend_method(
            detection_ty,
            method_name,
            call_node,
            Some(base),
            field_node.span,
            type_checker,
            errors,
        ) {
            return Some(handled_method_result(
                extend_ret,
                op_safe,
                chain_is_optional,
                index,
                call_op,
            ));
        } else {
            errors.push(Diagnostic::new(
                field_node.span,
                DiagnosticKind::ExternUnknownMethod {
                    type_name: extern_name,
                    method: method_name,
                },
            ));
        }
        return Some(handled_infer(op_safe, chain_is_optional, index, call_op));
    };

    let method_ret = check_extern_instance_method_call(
        call_node,
        extern_name,
        method_name,
        method,
        Some(base),
        type_checker,
        errors,
    );

    Some(handled_method_result(
        method_ret,
        op_safe,
        chain_is_optional,
        index,
        call_op,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_builtin_or_extend(
    target_ty: &Type,
    method_name: Ident,
    call_node: &CallNode,
    base: Option<&ExprNode>,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<Type> {
    match target_ty {
        Type::List { elem } => {
            // extend-first so "extend [int]" methods shadow builtins when defined
            if let Some(ty) = resolve_extend_method(
                target_ty,
                method_name,
                call_node,
                base,
                span,
                type_checker,
                errors,
            ) {
                return Some(ty);
            }
            Some(check_list_method(
                call_node,
                base.unwrap_or(call_node.node.func.as_ref()),
                method_name,
                elem.as_ref(),
                type_checker,
                errors,
            )?)
        }
        Type::Map { key, value } => {
            // extend-first so "extend {K: V}" methods shadow builtins when defined
            if let Some(ty) = resolve_extend_method(
                target_ty,
                method_name,
                call_node,
                base,
                span,
                type_checker,
                errors,
            ) {
                return Some(ty);
            }
            Some(check_map_method(
                call_node,
                base.unwrap_or(call_node.node.func.as_ref()),
                method_name,
                key.as_ref(),
                value.as_ref(),
                type_checker,
                errors,
            )?)
        }
        Type::Array { .. }
        | Type::Tuple { .. }
        | Type::Float
        | Type::Double
        | Type::Int
        | Type::Bool
        | Type::String
        | Type::Enum { .. } => resolve_extend_method(
            target_ty,
            method_name,
            call_node,
            base,
            span,
            type_checker,
            errors,
        ),
        _ => None,
    }
}

fn emit_unknown_method(
    ty: &Type,
    method_name: Ident,
    span: Span,
    op_safe: bool,
    chain_is_optional: bool,
    index: usize,
    call_op: PostfixNodeRef<'_>,
    errors: &mut Vec<Diagnostic>,
) -> MethodCallOutcome {
    let type_ident = Ident(Intern::new(format!("{ty}")));
    errors.push(Diagnostic::new(
        span,
        DiagnosticKind::UnknownMethod {
            kind: "type",
            struct_name: type_ident,
            method: method_name,
        },
    ));
    handled_infer(op_safe, chain_is_optional, index, call_op)
}

fn try_builtin_or_extend_method(
    detection_ty: &Type,
    field_node: &crate::ast::FieldAccessNode,
    call_op: PostfixNodeRef<'_>,
    op_safe: bool,
    chain_is_optional: bool,
    index: usize,
    base: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<MethodCallOutcome> {
    let PostfixNodeRef::Call {
        node: call_node, ..
    } = call_op
    else {
        unreachable!()
    };
    let method_name = field_node.node.field;

    if let Some(ret) = resolve_builtin_or_extend(
        detection_ty,
        method_name,
        call_node,
        Some(base),
        field_node.span,
        type_checker,
        errors,
    ) {
        return Some(handled_method_result(
            ret,
            op_safe,
            chain_is_optional,
            index,
            call_op,
        ));
    }

    // emit UnknownMethod for types where a method was expected but not found
    match detection_ty {
        Type::Array { .. }
        | Type::Float
        | Type::Double
        | Type::Int
        | Type::Bool
        | Type::String
        | Type::Enum { .. } => Some(emit_unknown_method(
            detection_ty,
            method_name,
            field_node.span,
            op_safe,
            chain_is_optional,
            index,
            call_op,
            errors,
        )),
        _ => None,
    }
}

fn handle_method_call_if_applicable(
    chain: &[PostfixNodeRef<'_>],
    index: usize,
    current_ty: &Type,
    chain_is_optional: bool,
    base: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
) -> Option<MethodCallOutcome> {
    if index + 1 >= chain.len() {
        return None;
    }

    let PostfixNodeRef::Field {
        node: field_node, ..
    } = chain[index]
    else {
        return None;
    };
    let field_op = chain[index];

    let PostfixNodeRef::Call {
        node: call_node, ..
    } = chain[index + 1]
    else {
        return None;
    };
    let call_op = chain[index + 1];

    if call_node.node.func.node.id != field_op.expr_id() {
        return Some(MethodCallOutcome::NotMethod);
    }

    let detection_ty = unwrap_opt_typ(current_ty);
    let op_safe = field_op.safe() || call_op.safe();

    if op_safe {
        let error_span = if field_op.safe() {
            field_node.span
        } else {
            call_op.span()
        };
        if !current_ty.is_option() {
            errors.push(
                Diagnostic::new(
                    error_span,
                    DiagnosticKind::OptionalChainingOnNonOpt {
                        found: current_ty.clone(),
                    },
                )
                .with_help("remove the `?` or make the base type optional"),
            );
            mark_remaining_ops_infer(chain, index, type_checker);
            return Some(MethodCallOutcome::Abort);
        }
    }

    // check "extend T?"" methods before unwrapping so they can be called on "T?" directly
    if !op_safe && current_ty.is_option() {
        let method_name = field_node.node.field;
        if let Some(extend_ret) = resolve_extend_method(
            current_ty,
            method_name,
            call_node,
            Some(base),
            field_node.span,
            type_checker,
            errors,
        ) {
            return Some(handled_method_result(
                extend_ret,
                op_safe,
                chain_is_optional,
                index,
                call_op,
            ));
        }
    }

    // struct and dataref method dispatch
    if let Some(agg) = detection_ty.as_aggregate() {
        return match try_struct_method(
            agg.name,
            agg.type_args,
            detection_ty,
            field_node,
            call_node,
            call_op,
            op_safe,
            chain_is_optional,
            index,
            chain,
            base,
            type_checker,
            errors,
        ) {
            Some(outcome) => Some(outcome),
            None => Some(MethodCallOutcome::NotMethod),
        };
    }

    // extern type method dispatch
    if let Type::Extern { name } = detection_ty
        && let Some(outcome) = try_extern_method(
            *name,
            detection_ty,
            field_node,
            call_node,
            call_op,
            op_safe,
            chain_is_optional,
            index,
            base,
            type_checker,
            errors,
        )
    {
        return Some(outcome);
    }

    // built-in list/map and extend methods on primitives/enums
    match try_builtin_or_extend_method(
        detection_ty,
        field_node,
        call_op,
        op_safe,
        chain_is_optional,
        index,
        base,
        type_checker,
        errors,
    ) {
        Some(outcome) => Some(outcome),
        None => Some(MethodCallOutcome::NotMethod),
    }
}

fn mark_remaining_ops_infer(
    chain: &[PostfixNodeRef<'_>],
    start_idx: usize,
    type_checker: &mut TypeChecker,
) {
    for op in &chain[start_idx..] {
        set_op_type(op, Type::Infer, type_checker);
    }
}

fn set_op_type(op: &PostfixNodeRef<'_>, ty: Type, type_checker: &mut TypeChecker) {
    type_checker.set_type(op.expr_id(), ty, op.span());
}

#[allow(clippy::too_many_arguments)]
fn try_module_dispatch(
    type_name: Ident,
    module_def: &super::types::ModuleDef,
    chain: &[PostfixNodeRef<'_>],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Option<(Type, usize, bool)> {
    let [
        PostfixNodeRef::Field {
            expr_id: field_expr_id,
            node: field_node,
        },
        rest @ ..,
    ] = chain
    else {
        return None;
    };

    let member_name = field_node.node.field;
    let op_safe = field_node.node.safe;

    // nested facade.submodule.func()
    if let Some(sub_def) = module_def.re_exported_modules.get(&member_name).cloned() {
        if let [
            PostfixNodeRef::Field {
                node: sub_field_node,
                ..
            },
            sub_rest @ ..,
        ] = rest
        {
            let sub_member = sub_field_node.node.field;
            let sub_op_safe = op_safe || sub_field_node.node.safe;
            let call_follows = matches!(sub_rest.first(), Some(PostfixNodeRef::Call { .. }));

            if call_follows {
                let call_op = sub_rest[0];
                let PostfixNodeRef::Call {
                    node: call_node, ..
                } = call_op
                else {
                    unreachable!()
                };
                let sub_op_safe = sub_op_safe || call_op.safe();
                let ty = check_module_func_call(
                    call_node,
                    member_name,
                    sub_member,
                    &sub_def,
                    type_checker,
                    errors,
                    expected,
                );
                return Some((ty, 3, sub_op_safe));
            }
        }
        let err_kind = DiagnosticKind::UnknownModuleMember {
            module: type_name,
            member: member_name,
        };
        errors.push(Diagnostic::new(field_node.span, err_kind));
        return Some((Type::Infer, 1, op_safe));
    }

    let call_follows = matches!(rest.first(), Some(PostfixNodeRef::Call { .. }));

    if call_follows {
        let call_op = rest[0];
        let PostfixNodeRef::Call {
            node: call_node, ..
        } = call_op
        else {
            unreachable!()
        };
        let op_safe = op_safe || call_op.safe();

        let is_regular_func = module_def.funcs.contains_key(&member_name)
            || module_def.generic_func_templates.contains_key(&member_name);

        if is_regular_func {
            let ty = check_module_func_call(
                call_node,
                type_name,
                member_name,
                module_def,
                type_checker,
                errors,
                expected,
            );
            return Some((ty, 2, op_safe));
        }

        let extend_entries: Vec<_> = module_def
            .extend_methods
            .iter()
            .filter(|e| e.name == member_name)
            .collect();

        if !extend_entries.is_empty() {
            let first_arg_ty = call_node.node.args.first().map_or(Type::Infer, |arg| {
                check_expr(arg, type_checker, errors, None)
            });
            let def = extend_entries
                .iter()
                .find(|e| e.ty == first_arg_ty)
                .map(|e| &e.def);
            if let Some(def) = def {
                let ty = check_extend_qualified_call(def, call_node, type_checker, errors);
                return Some((ty, 2, op_safe));
            }
        }

        // try generic extend methods from this module
        let generic_entries: Vec<_> = module_def
            .generic_extend_methods
            .iter()
            .filter(|e| e.method_name == member_name)
            .collect();

        if !generic_entries.is_empty() {
            let first_arg_ty = call_node.node.args.first().map_or(Type::Infer, |arg| {
                check_expr(arg, type_checker, errors, None)
            });
            let first_arg_expr = call_node.node.args.first();

            let mut matches: Vec<(GenericExtendTemplate, Vec<Type>, Vec<usize>)> = vec![];
            for entry in &generic_entries {
                let mut bindings = HashMap::new();
                let mut const_bindings = HashMap::new();
                if match_type_pattern(
                    &first_arg_ty,
                    &entry.target_type,
                    &entry.type_params,
                    &mut bindings,
                    &entry.const_params,
                    &mut const_bindings,
                ) {
                    let type_args: Vec<Type> = entry
                        .type_params
                        .iter()
                        .map(|p| bindings.get(&p.name).cloned().unwrap_or(Type::Infer))
                        .collect();
                    let const_args: Vec<usize> = entry
                        .const_params
                        .iter()
                        .map(|p| const_bindings.get(&p.id).copied().unwrap_or(0))
                        .collect();

                    // rebuild a temporary GenericExtendTemplate from the module entry
                    let base_key = extend_base_key(&first_arg_ty);
                    let stored = base_key.and_then(|k| {
                        type_checker
                            .ctx
                            .generic_extend_templates
                            .get(&(k, member_name))
                            .and_then(|templates| {
                                templates
                                    .iter()
                                    .find(|t| t.target_type == entry.target_type)
                                    .cloned()
                            })
                    });
                    if let Some(template) = stored {
                        matches.push((template, type_args, const_args));
                    }
                }
            }

            if matches.len() == 1 {
                let (template, type_args, const_args) = matches.remove(0);
                let ty = try_specialize_extend(
                    &first_arg_ty,
                    &type_args,
                    &const_args,
                    &template,
                    member_name,
                    call_node,
                    first_arg_expr,
                    field_node.span,
                    type_checker,
                    errors,
                );
                return Some((ty, 2, op_safe));
            } else if matches.len() > 1 {
                // specificity tournament
                let mut best_idx = 0;
                for i in 1..matches.len() {
                    match compare_specificity(
                        &matches[best_idx].0.target_type,
                        &matches[best_idx].0.type_params,
                        &matches[best_idx].0.const_params,
                        &matches[i].0.target_type,
                        &matches[i].0.type_params,
                        &matches[i].0.const_params,
                    ) {
                        Specificity::LessSpecific | Specificity::Equal => best_idx = i,
                        Specificity::MoreSpecific => {}
                        Specificity::Incomparable => {
                            let candidates = matches.iter().map(|(t, _, _)| t.binding).collect();
                            errors.push(Diagnostic::new(
                                field_node.span,
                                DiagnosticKind::AmbiguousExtendMethod {
                                    ty: first_arg_ty.clone(),
                                    method: member_name,
                                    candidates,
                                },
                            ));
                            return Some((Type::Infer, 2, op_safe));
                        }
                    }
                }
                let (template, type_args, const_args) = matches.swap_remove(best_idx);
                let ty = try_specialize_extend(
                    &first_arg_ty,
                    &type_args,
                    &const_args,
                    &template,
                    member_name,
                    call_node,
                    first_arg_expr,
                    field_node.span,
                    type_checker,
                    errors,
                );
                return Some((ty, 2, op_safe));
            }
        }

        let ty = check_module_func_call(
            call_node,
            type_name,
            member_name,
            module_def,
            type_checker,
            errors,
            expected,
        );
        return Some((ty, 2, op_safe));
    }

    // module.MemberName without a call — check for const access
    if let Some(const_def) = module_def.const_defs.get(&member_name) {
        type_checker
            .const_values
            .insert(*field_expr_id, const_def.value.clone());
        return Some((const_def.ty.clone(), 1, op_safe));
    }

    let err_kind = if module_def.all_names.contains(&member_name) {
        DiagnosticKind::PrivateModuleMember {
            module: type_name,
            member: member_name,
        }
    } else {
        DiagnosticKind::UnknownModuleMember {
            module: type_name,
            member: member_name,
        }
    };
    errors.push(Diagnostic::new(field_node.span, err_kind));
    Some((Type::Infer, 1, op_safe))
}

fn try_enum_or_static_dispatch(
    type_name: Ident,
    chain: &[PostfixNodeRef<'_>],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Option<(Type, usize, bool)> {
    let [
        PostfixNodeRef::Field {
            node: field_node, ..
        },
        rest @ ..,
    ] = chain
    else {
        return None;
    };

    let call_follows = matches!(rest.first(), Some(PostfixNodeRef::Call { .. }));

    if !call_follows {
        // enum unit variant: EnumName.Variant
        let enum_def = type_checker.get_enum(type_name).cloned()?;
        let ty = resolve_enum_unit_variant(type_name, field_node, &enum_def, errors);
        let safe = field_node.node.safe;
        return Some((ty, 1, safe));
    }

    // Field+Call: enum tuple variant or static method
    let call_op = rest[0];
    let PostfixNodeRef::Call {
        node: call_node, ..
    } = call_op
    else {
        unreachable!()
    };
    let op_safe = field_node.node.safe || call_op.safe();

    if type_checker.get_enum(type_name).is_some() {
        let ty = check_call(call_node, type_checker, errors, expected);
        return Some((ty, 2, op_safe));
    }

    if let Some(struct_def) = type_checker.get_struct(type_name).cloned() {
        let ty = check_static_method_call(
            call_node,
            type_name,
            field_node.node.field,
            &struct_def,
            type_checker,
            errors,
        );
        return Some((ty, 2, op_safe));
    }

    if let Some(extern_def) = type_checker.get_extern_type(type_name).cloned() {
        let ty = check_extern_static_method_call(
            call_node,
            type_name,
            field_node.node.field,
            &extern_def,
            type_checker,
            errors,
        );
        return Some((ty, 2, op_safe));
    }

    None
}

fn try_type_name_dispatch(
    base: &ExprNode,
    chain: &[PostfixNodeRef<'_>],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Option<(Type, usize, bool)> {
    let ExprKind::Ident(type_name) = &base.node.kind else {
        return None;
    };

    // module qualified access
    if let Some(module_def) = type_checker.get_module(*type_name).cloned() {
        return try_module_dispatch(
            *type_name,
            &module_def,
            chain,
            type_checker,
            errors,
            expected,
        );
    }

    // enum variant or static method dispatch
    try_enum_or_static_dispatch(*type_name, chain, type_checker, errors, expected)
}

fn resolve_enum_unit_variant(
    enum_name: Ident,
    field_node: &crate::ast::FieldAccessNode,
    enum_def: &super::types::EnumDef,
    errors: &mut Vec<Diagnostic>,
) -> Type {
    use crate::ast::VariantKind;
    let variant_name = field_node.node.field;
    let variant = enum_def.variants.iter().find(|v| v.name == variant_name);

    let Some(variant) = variant else {
        errors.push(Diagnostic::new(
            field_node.span,
            DiagnosticKind::UnknownEnumVariant {
                enum_name,
                variant_name,
            },
        ));
        return Type::Infer;
    };

    enum_def.check_deprecation(enum_name, variant, field_node.span, errors);

    match &variant.kind {
        VariantKind::Unit => {
            let type_args = if enum_def.type_params.is_empty() {
                vec![]
            } else {
                enum_def.type_params.iter().map(|_| Type::Infer).collect()
            };
            Type::Enum {
                name: enum_name,
                type_args,
            }
        }
        VariantKind::Tuple(_) | VariantKind::Struct(_) => {
            errors.push(Diagnostic::new(
                field_node.span,
                DiagnosticKind::EnumVariantNotUnit {
                    enum_name,
                    variant_name,
                },
            ));
            Type::Infer
        }
    }
}

fn check_closure_var_params(
    params: &[FuncParam],
    call_node: &CallNode,
    type_checker: &TypeChecker,
    errors: &mut Vec<Diagnostic>,
) {
    let var_params = params.iter().enumerate().map(|(i, fp)| {
        let name = Ident(Intern::new(format!("arg{i}")));
        let mutability = if fp.mutable {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        };
        (name, mutability)
    });
    check_var_param_args(var_params, &call_node.node.args, type_checker, errors);
}

fn apply_postfix_op(
    op: &PostfixNodeRef<'_>,
    base_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<Diagnostic>,
    expected: Option<&Type>,
) -> Type {
    match op {
        PostfixNodeRef::Field {
            node: field_node, ..
        } => type_field_on_base(
            base_ty,
            field_node.node.field,
            field_node.span,
            type_checker,
            errors,
        ),
        PostfixNodeRef::Index {
            node: index_node, ..
        } => {
            let index_ty = check_expr(&index_node.node.index, type_checker, errors, None);
            let result = type_index_on_base(
                base_ty,
                &index_ty,
                index_node.node.index.node.id,
                index_node.span,
                index_node.node.index.span,
                type_checker,
                errors,
            );
            // map indexing always returns an optional value, the chain handles the wrapping
            // for safe operators, and we wrap here for the non-safe case
            if matches!(base_ty, Type::Map { .. }) && !matches!(result, Type::Infer) {
                Type::option_of(result)
            } else {
                result
            }
        }
        PostfixNodeRef::Call {
            node: call_node, ..
        } => {
            if let ExprKind::Ident(name) = &call_node.node.func.node.kind {
                if let Some(reason) = type_checker.func_deprecated.get(name).cloned() {
                    errors.push(Diagnostic::new(
                        call_node.span,
                        DiagnosticKind::DeprecatedUsage {
                            kind: "function",
                            name: *name,
                            reason,
                        },
                    ));
                }
                let has_type_params = type_checker
                    .func_type_params
                    .get(name)
                    .is_some_and(|p| !p.is_empty());
                let has_const_params = type_checker
                    .func_const_params
                    .get(name)
                    .is_some_and(|cp| !cp.is_empty());
                let is_generic = has_type_params || has_const_params;
                let has_type_args = !call_node.node.type_args.is_empty();
                // delegate generic and explicit type ar calls to check_call
                if is_generic || has_type_args {
                    return check_call(call_node, type_checker, errors, expected);
                }
                // for plain ident calls use base_ty and check var-params separately
                let defaults = type_checker.func_param_defaults(*name);
                let required_count = match base_ty {
                    Type::Func { params, .. } => required_param_count(defaults, params.len()),
                    _ => 0,
                };
                let result =
                    type_call_on_base(base_ty, call_node, required_count, type_checker, errors);
                if let Some(param_info) = type_checker.func_param_info.get(name).cloned() {
                    check_var_param_args(param_info, &call_node.node.args, type_checker, errors);
                } else if let Type::Func { params, .. } = base_ty {
                    check_closure_var_params(params, call_node, type_checker, errors);
                }
                return result;
            }
            let required_count = match base_ty {
                Type::Func { params, .. } => params.len(),
                _ => 0,
            };
            let result =
                type_call_on_base(base_ty, call_node, required_count, type_checker, errors);
            if let Type::Func { params, .. } = base_ty {
                check_closure_var_params(params, call_node, type_checker, errors);
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use internment::Intern;

    use super::*;
    use crate::ast::{TypeParam, TypeVarId};

    fn ident(s: &str) -> Ident {
        Ident(Intern::new(s.to_string()))
    }

    fn tp(name: &str, id: u32) -> TypeParam {
        TypeParam {
            name: ident(name),
            id: TypeVarId(id),
        }
    }

    fn unresolved(name: &str) -> Type {
        Type::UnresolvedName(ident(name))
    }

    #[test]
    fn specificity_list_vs_nested_list() {
        let list_t = Type::List {
            elem: Box::new(unresolved("T")),
        };
        let nested_t = Type::List {
            elem: Box::new(Type::List {
                elem: Box::new(unresolved("T")),
            }),
        };
        let params = vec![tp("T", 0)];

        // [T] vs [[T]]: [T] is less specific, [[T]] is more specific
        assert!(matches!(
            compare_specificity(&list_t, &params, &[], &nested_t, &params, &[]),
            Specificity::LessSpecific
        ));
        assert!(matches!(
            compare_specificity(&nested_t, &params, &[], &list_t, &params, &[]),
            Specificity::MoreSpecific
        ));
    }

    #[test]
    fn specificity_equal_patterns() {
        let list_t = Type::List {
            elem: Box::new(unresolved("T")),
        };
        let list_u = Type::List {
            elem: Box::new(unresolved("U")),
        };
        let t_params = vec![tp("T", 0)];
        let u_params = vec![tp("U", 1)];

        // [T] vs [U] — structurally identical, just different param names
        assert!(matches!(
            compare_specificity(&list_t, &t_params, &[], &list_u, &u_params, &[]),
            Specificity::Equal
        ));
    }

    #[test]
    fn specificity_incomparable() {
        let tuple_t_int = Type::Tuple(vec![unresolved("T"), Type::Int]);
        let tuple_str_t = Type::Tuple(vec![Type::String, unresolved("T")]);
        let params = vec![tp("T", 0)];

        // (T, int) vs (string, T) — incomparable
        assert!(matches!(
            compare_specificity(&tuple_t_int, &params, &[], &tuple_str_t, &params, &[]),
            Specificity::Incomparable
        ));
    }

    #[test]
    fn specificity_option_generic_vs_option_list() {
        let opt_t = Type::Enum {
            name: ident("Option"),
            type_args: vec![unresolved("T")],
        };
        let opt_list_t = Type::Enum {
            name: ident("Option"),
            type_args: vec![Type::List {
                elem: Box::new(unresolved("T")),
            }],
        };
        let params = vec![tp("T", 0)];

        // Option<T> is less specific than Option<[T]>
        assert!(matches!(
            compare_specificity(&opt_t, &params, &[], &opt_list_t, &params, &[]),
            Specificity::LessSpecific
        ));
    }
}
