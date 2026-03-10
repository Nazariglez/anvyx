use crate::ast::{ExprId, ExprKind, ExprNode, Type};

use super::{
    call::{check_instance_method_call, type_call_on_base},
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    types::{type_field_on_base, type_index_on_base, PostfixNodeRef, TypeChecker, unwrap_opt_typ},
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

pub(super) fn chain_has_safe_op(chain: &[PostfixNodeRef<'_>]) -> bool {
    chain.iter().any(|op| op.safe())
}

pub(super) fn check_postfix_chain(
    expr_node: &ExprNode,
    base: &ExprNode,
    chain: &[PostfixNodeRef<'_>],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let mut current_ty = check_expr(base, type_checker, errors);
    let mut chain_is_optional = false;
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
            match &current_ty {
                Type::Optional(inner) => {
                    base_ty = (**inner).clone();
                }
                _ => {
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

        let op_result_inner = apply_postfix_op(op, &base_ty, type_checker, errors);
        if op_safe || chain_is_optional {
            current_ty = Type::Optional(Box::new(op_result_inner.clone()));
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
        Type::Struct { name, type_args } => Some((*name, type_args.clone())),
        _ => None,
    };

    let Some((struct_name, struct_type_args)) = struct_info else {
        return Some(MethodCallOutcome::NotMethod);
    };

    let op_safe = field_op.safe() || call_op.safe();
    if op_safe {
        let error_span = if field_op.safe() {
            field_node.span
        } else {
            call_op.span()
        };

        match current_ty {
            Type::Optional(inner) => {
                let _ = inner;
            }
            _ => {
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
    }

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

    let method_ret = check_instance_method_call(
        call_node,
        struct_name,
        field_node.node.field,
        &struct_type_args,
        &struct_def,
        Some(base),
        type_checker,
        errors,
    );

    let mut result_ty = method_ret;
    let mut chain_optional = chain_is_optional;
    if op_safe || chain_is_optional {
        chain_optional = true;
        result_ty = Type::Optional(result_ty.boxed());
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

fn apply_postfix_op(
    op: &PostfixNodeRef<'_>,
    base_ty: &Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
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
            let index_ty = check_expr(&index_node.node.index, type_checker, errors);
            type_index_on_base(
                base_ty,
                &index_ty,
                index_node.span,
                index_node.node.index.span,
                errors,
            )
        }
        PostfixNodeRef::Call {
            node: call_node, ..
        } => {
            // ror calls we need to check arguments and compute return type
            type_call_on_base(base_ty, call_node, type_checker, errors)
        }
    }
}
