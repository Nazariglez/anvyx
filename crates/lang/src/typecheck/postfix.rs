use crate::{
    ast::{CallNode, ExprId, ExprKind, ExprNode, Ident, Mutability, Type},
    span::Span,
};
use internment::Intern;
use std::collections::HashMap;

use super::{
    call::{
        check_call, check_call_signature, check_extern_instance_method_call,
        check_extern_static_method_call, check_instance_method_call, check_list_method,
        check_map_method, check_module_func_call, check_var_param_args, report_cached_spec_error,
        report_instantiation_errors, required_param_count, type_call_on_base,
    },
    decl::check_body_common,
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    infer::{build_subst, resolve_type_param_names, subst_type},
    types::{
        ExtendEntry, ExtendMethodDef, ExtendSpecKey, GenericExtendTemplate, PostfixNodeRef,
        SpecializationResult, TypeChecker, type_field_on_base, type_index_on_base, unwrap_opt_typ,
    },
};

pub(super) fn collect_postfix_chain<'a>(
    expr: &'a ExprNode,
) -> (&'a ExprNode, Vec<PostfixNodeRef<'a>>) {
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
    errors: &mut Vec<TypeErr>,
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
    errors: &mut Vec<TypeErr>,
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
            match current_ty.option_inner() {
                Some(inner) => {
                    base_ty = inner.clone();
                }
                None => {
                    errors.push(
                        TypeErr::new(
                            op.span(),
                            TypeErrKind::OptionalChainingOnNonOpt {
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

#[allow(clippy::too_many_arguments)]
fn try_specialize_extend(
    receiver_ty: &Type,
    type_args: &[Type],
    template: &GenericExtendTemplate,
    method_name: Ident,
    call_node: &CallNode,
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let base_name = match receiver_ty {
        Type::Struct { name, .. } | Type::Enum { name, .. } | Type::DataRef { name, .. } => *name,
        _ => return Type::Infer,
    };

    let cache_key = ExtendSpecKey {
        base_name,
        method_name,
        type_args: type_args.to_vec(),
    };

    let context_name = format!(
        "extend {}<{}>",
        base_name,
        type_args
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    if let Some(cached) = type_checker.extend_spec_cache.get(&cache_key).cloned() {
        report_cached_spec_error(
            &cached,
            &context_name,
            &template.type_params,
            type_args,
            span,
            errors,
        );
        let mangled = cache_key.mangle(&template.source_module);
        type_checker
            .extend_call_targets
            .insert(call_node.node.func.node.id, mangled);
        let specialized_params: Vec<Type> = {
            let subst = build_subst(&template.type_params, type_args);
            template
                .method
                .node
                .params
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if i == 0 {
                        receiver_ty.clone()
                    } else {
                        let resolved = resolve_type_param_names(
                            &type_checker.resolve_type(&p.ty),
                            &template.type_params,
                        );
                        subst_type(&resolved, &subst)
                    }
                })
                .collect()
        };
        check_call_signature(
            call_node.span,
            &specialized_params[1..],
            specialized_params.len() - 1,
            &cached.ret_ty.clone(),
            &call_node.node.args,
            type_checker,
            errors,
        );
        return cached.ret_ty;
    }

    let subst = build_subst(&template.type_params, type_args);
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
                subst_type(&resolved, &subst)
            }
        })
        .collect();
    let raw_ret = resolve_type_param_names(&type_checker.resolve_type(&method.ret), type_params);
    let specialized_ret = subst_type(&raw_ret, &subst);

    let prev_snapshot = type_checker.spec_type_snapshot.take();
    type_checker.spec_type_snapshot = Some(HashMap::new());

    let module_scope = if template.source_module.is_empty() {
        None
    } else {
        type_checker
            .resolved_module_defs
            .get(&template.source_module)
            .cloned()
    };
    if let Some(ref module_def) = module_scope {
        type_checker.push_scope();
        for (name, ty) in &module_def.funcs {
            type_checker.set_var(*name, ty.clone(), false);
        }
    }

    let body_params: Vec<(Ident, Type, bool)> = method
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

    let mut body_errors = vec![];
    check_body_common(
        &body_params,
        &method.body,
        &specialized_ret,
        span,
        type_checker,
        &mut body_errors,
    );

    if module_scope.is_some() {
        type_checker.pop_scope();
    }

    let body_types = type_checker.spec_type_snapshot.take().unwrap_or_default();
    type_checker.spec_type_snapshot = prev_snapshot;

    let cached_err = body_errors.first().map(|err| (err.span, err.kind.clone()));
    type_checker.extend_spec_cache.insert(
        cache_key.clone(),
        SpecializationResult {
            ret_ty: specialized_ret.clone(),
            err: cached_err,
            body_types,
        },
    );

    report_instantiation_errors(
        body_errors,
        &context_name,
        &template.type_params,
        type_args,
        span,
        errors,
    );

    let mangled = cache_key.mangle(&template.source_module);
    type_checker
        .extend_call_targets
        .insert(call_node.node.func.node.id, mangled);

    check_call_signature(
        call_node.span,
        &specialized_params[1..],
        specialized_params.len() - 1,
        &specialized_ret,
        &call_node.node.args,
        type_checker,
        errors,
    );

    specialized_ret
}

/// Type-check an extend method call in receiver position: `receiver.method(args...)`.
/// Uses params[1..] — the self param is consumed by the receiver, not in args.
fn check_extend_call(
    def: &ExtendMethodDef,
    call_node: &CallNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let param_types: Vec<Type> = def.params[1..].iter().map(|p| p.ty.clone()).collect();
    type_checker
        .extend_call_targets
        .insert(call_node.node.func.node.id, def.internal_name);
    check_call_signature(
        call_node.span,
        &param_types,
        param_types.len(),
        &def.ret,
        &call_node.node.args,
        type_checker,
        errors,
    )
}

/// Type-check an extend method in qualified call position: `module.method(receiver, args...)`.
/// Uses params[0..] — the receiver is explicitly the first argument.
fn check_extend_qualified_call(
    def: &ExtendMethodDef,
    call_node: &CallNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let param_types: Vec<Type> = def.params.iter().map(|p| p.ty.clone()).collect();
    type_checker
        .extend_call_targets
        .insert(call_node.node.func.node.id, def.internal_name);
    check_call_signature(
        call_node.span,
        &param_types,
        param_types.len(),
        &def.ret,
        &call_node.node.args,
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
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
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
            span,
            type_checker,
            errors,
        ),
        [entry] => Some(check_extend_call(
            &entry.def,
            call_node,
            type_checker,
            errors,
        )),
        _ => {
            let candidates = unique.iter().map(|e| e.binding).collect();
            errors.push(TypeErr::new(
                span,
                TypeErrKind::AmbiguousExtendMethod {
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
    span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Option<Type> {
    let (base_name, type_args) = match receiver_ty {
        Type::Struct { name, type_args } if !type_args.is_empty() => (*name, type_args.as_slice()),
        Type::Enum { name, type_args } if !type_args.is_empty() => (*name, type_args.as_slice()),
        Type::DataRef { name, type_args } if !type_args.is_empty() => (*name, type_args.as_slice()),
        _ => return None,
    };

    let key = (base_name, method_name);
    let templates = type_checker.generic_extend_templates.get(&key)?.clone();

    let mut unique: Vec<&GenericExtendTemplate> = vec![];
    let mut seen: Vec<&Vec<String>> = vec![];
    for tmpl in &templates {
        if !seen.contains(&&tmpl.source_module) {
            seen.push(&tmpl.source_module);
            unique.push(tmpl);
        }
    }

    match unique.as_slice() {
        [] => None,
        [template] => {
            let template = (*template).clone();
            Some(try_specialize_extend(
                receiver_ty,
                type_args,
                &template,
                method_name,
                call_node,
                span,
                type_checker,
                errors,
            ))
        }
        _ => {
            let candidates = unique.iter().map(|t| t.binding).collect();
            errors.push(TypeErr::new(
                span,
                TypeErrKind::AmbiguousExtendMethod {
                    ty: receiver_ty.clone(),
                    method: method_name,
                    candidates,
                },
            ));
            Some(Type::Infer)
        }
    }
}

pub(super) enum MethodCallOutcome {
    NotMethod,
    Handled {
        ty: Type,
        chain_optional: bool,
        next_index: usize,
        call_expr: ExprId,
        call_span: crate::span::Span,
    },
    Abort,
}

fn handle_method_call_if_applicable(
    chain: &[PostfixNodeRef<'_>],
    index: usize,
    current_ty: &Type,
    chain_is_optional: bool,
    base: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Option<MethodCallOutcome> {
    if index + 1 >= chain.len() {
        return None;
    }

    let field_op = match chain[index] {
        PostfixNodeRef::Field { .. } => chain[index],
        _ => return None,
    };
    let call_op = match chain[index + 1] {
        PostfixNodeRef::Call { .. } => chain[index + 1],
        _ => return None,
    };

    let call_node = match call_op {
        PostfixNodeRef::Call { node, .. } => node,
        _ => unreachable!(),
    };
    let field_node = match field_op {
        PostfixNodeRef::Field { node, .. } => node,
        _ => unreachable!(),
    };

    if call_node.node.func.node.id != field_op.expr_id() {
        return Some(MethodCallOutcome::NotMethod);
    }

    let detection_ty = unwrap_opt_typ(current_ty);
    let struct_info = match &detection_ty {
        Type::Struct { name, type_args } | Type::DataRef { name, type_args } => {
            Some((*name, type_args.clone()))
        }
        _ => None,
    };

    let op_safe = field_op.safe() || call_op.safe();
    if op_safe {
        let error_span = if field_op.safe() {
            field_node.span
        } else {
            call_op.span()
        };

        if !current_ty.is_option() {
            errors.push(
                TypeErr::new(
                    error_span,
                    TypeErrKind::OptionalChainingOnNonOpt {
                        found: current_ty.clone(),
                    },
                )
                .with_help("remove the `?` or make the base type optional"),
            );
            mark_remaining_ops_infer(chain, index, type_checker);
            return Some(MethodCallOutcome::Abort);
        }
    }

    if let Some((struct_name, struct_type_args)) = struct_info {
        let Some(struct_def) = type_checker.get_struct(struct_name).cloned() else {
            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::UnknownStruct { name: struct_name },
            ));
            return Some(MethodCallOutcome::Handled {
                ty: Type::Infer,
                chain_optional: chain_is_optional || op_safe,
                next_index: index + 2,
                call_expr: call_op.expr_id(),
                call_span: call_op.span(),
            });
        };

        let method_name = field_node.node.field;
        let method_ret = if struct_def.methods.contains_key(&method_name) {
            // Native method exists — instance or static, let existing path handle it
            check_instance_method_call(
                call_node,
                struct_name,
                method_name,
                &struct_type_args,
                &struct_def,
                Some(base),
                type_checker,
                errors,
            )
        } else if let Some(extend_ret) = resolve_extend_method(
            detection_ty,
            method_name,
            call_node,
            field_node.span,
            type_checker,
            errors,
        ) {
            extend_ret
        } else {
            // Not found — reuse existing path to emit UnknownMethod
            check_instance_method_call(
                call_node,
                struct_name,
                method_name,
                &struct_type_args,
                &struct_def,
                Some(base),
                type_checker,
                errors,
            )
        };

        let mut result_ty = method_ret;
        let mut chain_optional = chain_is_optional;
        if op_safe || chain_is_optional {
            chain_optional = true;
            result_ty = Type::option_of(result_ty);
        }

        return Some(MethodCallOutcome::Handled {
            ty: result_ty,
            chain_optional,
            next_index: index + 2,
            call_expr: call_op.expr_id(),
            call_span: call_op.span(),
        });
    }

    if let Type::Extern { name } = &detection_ty
        && let Some(extern_def) = type_checker.get_extern_type(*name).cloned()
    {
        let method_name = field_node.node.field;
        let Some(method) = extern_def.methods.get(&method_name) else {
            if extern_def.statics.contains_key(&method_name) {
                errors.push(TypeErr::new(
                    call_node.span,
                    TypeErrKind::StaticMethodOnValue {
                        struct_name: *name,
                        method: method_name,
                    },
                ));
            } else if let Some(extend_ret) = resolve_extend_method(
                detection_ty,
                method_name,
                call_node,
                field_node.span,
                type_checker,
                errors,
            ) {
                let mut result_ty = extend_ret;
                let mut chain_optional = chain_is_optional;
                if op_safe || chain_is_optional {
                    chain_optional = true;
                    result_ty = Type::option_of(result_ty);
                }
                return Some(MethodCallOutcome::Handled {
                    ty: result_ty,
                    chain_optional,
                    next_index: index + 2,
                    call_expr: call_op.expr_id(),
                    call_span: call_op.span(),
                });
            } else {
                errors.push(TypeErr::new(
                    field_node.span,
                    TypeErrKind::ExternUnknownMethod {
                        type_name: *name,
                        method: method_name,
                    },
                ));
            }
            return Some(MethodCallOutcome::Handled {
                ty: Type::Infer,
                chain_optional: chain_is_optional || op_safe,
                next_index: index + 2,
                call_expr: call_op.expr_id(),
                call_span: call_op.span(),
            });
        };

        let method_ret = check_extern_instance_method_call(
            call_node,
            *name,
            method_name,
            method,
            Some(base),
            type_checker,
            errors,
        );

        let mut result_ty = method_ret;
        let mut chain_optional = chain_is_optional;
        if op_safe || chain_is_optional {
            chain_optional = true;
            result_ty = Type::option_of(result_ty);
        }

        return Some(MethodCallOutcome::Handled {
            ty: result_ty,
            chain_optional,
            next_index: index + 2,
            call_expr: call_op.expr_id(),
            call_span: call_op.span(),
        });
    }

    let method_target = field_node.node.target.as_ref();
    let method_name = field_node.node.field;
    let method_ret = match &detection_ty {
        Type::List { elem } => check_list_method(
            call_node,
            method_target,
            method_name,
            elem.as_ref(),
            type_checker,
            errors,
        ),
        Type::Map { key, value } => check_map_method(
            call_node,
            method_target,
            method_name,
            key.as_ref(),
            value.as_ref(),
            type_checker,
            errors,
        ),
        Type::Float | Type::Double | Type::Int | Type::Bool | Type::String | Type::Enum { .. } => {
            if let Some(extend_ret) = resolve_extend_method(
                detection_ty,
                method_name,
                call_node,
                field_node.span,
                type_checker,
                errors,
            ) {
                let mut result_ty = extend_ret;
                let mut chain_optional = chain_is_optional;
                if op_safe || chain_is_optional {
                    chain_optional = true;
                    result_ty = Type::option_of(result_ty);
                }
                return Some(MethodCallOutcome::Handled {
                    ty: result_ty,
                    chain_optional,
                    next_index: index + 2,
                    call_expr: call_op.expr_id(),
                    call_span: call_op.span(),
                });
            }
            // No native methods and no extend method found — emit a clear error
            let type_ident = Ident(Intern::new(format!("{detection_ty}")));
            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::UnknownMethod {
                    struct_name: type_ident,
                    method: method_name,
                },
            ));
            return Some(MethodCallOutcome::Handled {
                ty: Type::Infer,
                chain_optional: chain_is_optional || op_safe,
                next_index: index + 2,
                call_expr: call_op.expr_id(),
                call_span: call_op.span(),
            });
        }
        _ => return Some(MethodCallOutcome::NotMethod),
    };

    let mut result_ty = method_ret;
    let mut chain_optional = chain_is_optional;
    if op_safe || chain_is_optional {
        chain_optional = true;
        result_ty = Type::option_of(result_ty);
    }

    Some(MethodCallOutcome::Handled {
        ty: result_ty,
        chain_optional,
        next_index: index + 2,
        call_expr: call_op.expr_id(),
        call_span: call_op.span(),
    })
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

/// For chains that start with a type-name identifier (enum or struct), dispatches
/// without evaluating the identifier as a value variable. Returns the result type,
/// the number of chain steps consumed, and whether any safe operator was used.
fn try_type_name_dispatch(
    base: &ExprNode,
    chain: &[PostfixNodeRef<'_>],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
    expected: Option<&Type>,
) -> Option<(Type, usize, bool)> {
    let ExprKind::Ident(type_name) = &base.node.kind else {
        return None;
    };

    // handle module qualified access before enum/struct dispatch
    if let Some(module_def) = type_checker.get_module(*type_name).cloned() {
        if let [
            PostfixNodeRef::Field {
                expr_id: field_expr_id,
                node: field_node,
            },
            rest @ ..,
        ] = chain
        {
            let member_name = field_node.node.field;
            let op_safe = field_node.node.safe;

            // nested facade.submodule.func(), submodule is a re-exported module
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
                    let call_follows =
                        matches!(sub_rest.first(), Some(PostfixNodeRef::Call { .. }));

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
                // facade.submodule without further chaining, not a valid expression
                let err_kind = TypeErrKind::UnknownModuleMember {
                    module: *type_name,
                    member: member_name,
                };
                errors.push(TypeErr::new(field_node.span, err_kind));
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

                // Regular functions take precedence over extend methods
                let is_regular_func = module_def.funcs.contains_key(&member_name)
                    || module_def.generic_func_templates.contains_key(&member_name);

                if is_regular_func {
                    let ty = check_module_func_call(
                        call_node,
                        *type_name,
                        member_name,
                        &module_def,
                        type_checker,
                        errors,
                        expected,
                    );
                    return Some((ty, 2, op_safe));
                }

                // Try extend method: find entry matching member_name
                let extend_entries: Vec<_> = module_def
                    .extend_methods
                    .iter()
                    .filter(|e| e.name == member_name)
                    .collect();

                if !extend_entries.is_empty() {
                    // Disambiguate by type-checking first arg if multiple entries
                    let def = if extend_entries.len() == 1 {
                        &extend_entries[0].def
                    } else {
                        let first_arg_ty = call_node
                            .node
                            .args
                            .first()
                            .map(|arg| check_expr(arg, type_checker, errors, None))
                            .unwrap_or(Type::Infer);
                        extend_entries
                            .iter()
                            .find(|e| e.ty == first_arg_ty)
                            .map(|e| &e.def)
                            .unwrap_or(&extend_entries[0].def)
                    };
                    let ty = check_extend_qualified_call(def, call_node, type_checker, errors);
                    return Some((ty, 2, op_safe));
                }

                // Not a regular func or extend method — emit unknown member error
                let ty = check_module_func_call(
                    call_node,
                    *type_name,
                    member_name,
                    &module_def,
                    type_checker,
                    errors,
                    expected,
                );
                return Some((ty, 2, op_safe));
            } else {
                // module.MemberName without a call — check for const access
                if let Some(const_def) = module_def.const_defs.get(&member_name) {
                    type_checker
                        .const_values
                        .insert(*field_expr_id, const_def.value.clone());
                    return Some((const_def.ty.clone(), 1, op_safe));
                }

                let err_kind = if module_def.all_names.contains(&member_name) {
                    TypeErrKind::PrivateModuleMember {
                        module: *type_name,
                        member: member_name,
                    }
                } else {
                    TypeErrKind::UnknownModuleMember {
                        module: *type_name,
                        member: member_name,
                    }
                };
                errors.push(TypeErr::new(field_node.span, err_kind));
                return Some((Type::Infer, 1, op_safe));
            }
        }
        return None;
    }

    // single Field step: may be an enum unit variant access
    if let [
        PostfixNodeRef::Field {
            node: field_node, ..
        },
        rest @ ..,
    ] = chain
    {
        let call_follows = matches!(rest.first(), Some(PostfixNodeRef::Call { .. }));

        if !call_follows {
            // enum unit variant: EnumName.Variant
            if let Some(enum_def) = type_checker.get_enum(*type_name).cloned() {
                let ty = resolve_enum_unit_variant(*type_name, field_node, &enum_def, errors);
                let safe = field_node.node.safe;
                return Some((ty, 1, safe));
            }
            return None;
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

        if let Some(enum_def) = type_checker.get_enum(*type_name).cloned() {
            let ty = check_call(call_node, type_checker, errors, expected);
            let _ = enum_def;
            return Some((ty, 2, op_safe));
        }

        if let Some(struct_def) = type_checker.get_struct(*type_name).cloned() {
            let ty = check_static_method_call_outer(
                call_node,
                *type_name,
                field_node.node.field,
                &struct_def,
                type_checker,
                errors,
            );
            return Some((ty, 2, op_safe));
        }

        if let Some(extern_def) = type_checker.get_extern_type(*type_name).cloned() {
            let ty = check_extern_static_method_call(
                call_node,
                *type_name,
                field_node.node.field,
                &extern_def,
                type_checker,
                errors,
            );
            return Some((ty, 2, op_safe));
        }
    }

    None
}

fn resolve_enum_unit_variant(
    enum_name: crate::ast::Ident,
    field_node: &crate::ast::FieldAccessNode,
    enum_def: &super::types::EnumDef,
    errors: &mut Vec<TypeErr>,
) -> Type {
    use crate::ast::VariantKind;
    let variant_name = field_node.node.field;
    let variant = enum_def.variants.iter().find(|v| v.name == variant_name);

    let Some(variant) = variant else {
        errors.push(TypeErr::new(
            field_node.span,
            TypeErrKind::UnknownEnumVariant {
                enum_name,
                variant_name,
            },
        ));
        return Type::Infer;
    };

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
            errors.push(TypeErr::new(
                field_node.span,
                TypeErrKind::EnumVariantNotUnit {
                    enum_name,
                    variant_name,
                },
            ));
            Type::Infer
        }
    }
}

fn check_static_method_call_outer(
    call_node: &crate::ast::CallNode,
    struct_name: crate::ast::Ident,
    method_name: crate::ast::Ident,
    struct_def: &super::types::StructDef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    super::call::check_static_method_call(
        call_node,
        struct_name,
        method_name,
        struct_def,
        type_checker,
        errors,
    )
}

fn apply_postfix_op(
    op: &PostfixNodeRef<'_>,
    base_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
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
                let is_generic = type_checker
                    .func_type_params
                    .get(name)
                    .is_some_and(|p| !p.is_empty());
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
                }
                return result;
            }
            let required_count = match base_ty {
                Type::Func { params, .. } => params.len(),
                _ => 0,
            };
            type_call_on_base(base_ty, call_node, required_count, type_checker, errors)
        }
    }
}
