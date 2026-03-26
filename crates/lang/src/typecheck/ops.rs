use crate::ast::{
    AssignNode, AssignOp, BinaryNode, BinaryOp, ExprKind, MethodReceiver, Type, UnaryNode, UnaryOp,
};

use super::{
    constraint::TypeRef,
    error::{TypeErr, TypeErrKind},
    expr::{check_expr, root_ident},
    types::{TypeChecker, equatable_reason, is_equatable},
    unify::unify_types,
};

pub(super) fn check_binary(
    bin: &BinaryNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    use BinaryOp::*;

    let node = &bin.node;
    let left_ty = check_expr(&node.left, type_checker, errors, None);
    let right_ty = check_expr(&node.right, type_checker, errors, None);
    let same_ty = left_ty == right_ty;

    match node.op {
        // string concat and string + primitive coercion
        Add if left_ty.is_str() || right_ty.is_str() => {
            let both_str = left_ty.is_str() && same_ty;
            let left_str_right_prim = left_ty.is_str() && right_ty.is_stringable_primitive();
            let right_str_left_prim = right_ty.is_str() && left_ty.is_stringable_primitive();
            let is_valid = both_str || left_str_right_prim || right_str_left_prim;
            if is_valid {
                Type::String
            } else {
                errors.push(TypeErr::new(
                    bin.span,
                    TypeErrKind::MismatchedTypes {
                        expected: left_ty.clone(),
                        found: right_ty.clone(),
                    },
                ));
                Type::Infer
            }
        }

        // numeric ops
        Add | Sub | Mul | Div | Rem => {
            if left_ty.is_num() && same_ty {
                left_ty
            } else if let Some(result) =
                resolve_extern_binary_op(node.op, &left_ty, &right_ty, type_checker)
            {
                match result {
                    ExternOpResult::Found(ret) => ret,
                    ExternOpResult::Ambiguous => {
                        errors.push(TypeErr::new(
                            bin.span,
                            TypeErrKind::AmbiguousOperator {
                                op: node.op.to_string(),
                                left: left_ty.clone(),
                                right: right_ty.clone(),
                            },
                        ));
                        Type::Infer
                    }
                }
            } else {
                errors.push(TypeErr::new(
                    bin.span,
                    TypeErrKind::MismatchedTypes {
                        expected: left_ty.clone(),
                        found: right_ty.clone(),
                    },
                ));
                Type::Infer
            }
        }

        // equal ops must be the same type and both sides must be equatable
        Eq | NotEq => {
            let eq_ty = if same_ty {
                Some(left_ty.clone())
            } else {
                let both_optional = left_ty.is_option() && right_ty.is_option();
                let unified = both_optional
                    .then(|| unify_types(&left_ty, &right_ty, bin.span, &mut vec![]))
                    .flatten();
                if let Some(ref ty) = unified {
                    type_checker.set_type(node.left.node.id, ty.clone(), bin.span);
                    type_checker.set_type(node.right.node.id, ty.clone(), bin.span);
                } else {
                    errors.push(TypeErr::new(
                        bin.span,
                        TypeErrKind::MismatchedTypes {
                            expected: left_ty.clone(),
                            found: right_ty.clone(),
                        },
                    ));
                }
                unified
            };

            if let Some(ref ty) = eq_ty {
                if !ty.is_infer() {
                    let has_extern_eq = matches!(ty, Type::Extern { name }
                        if type_checker
                            .get_extern_type(*name)
                            .map_or(false, |def| def.operators.iter().any(|o| o.op == BinaryOp::Eq)));
                    if !has_extern_eq && !is_equatable(ty, type_checker) {
                        let mut err =
                            TypeErr::new(bin.span, TypeErrKind::NotEquatable { ty: ty.clone() });
                        if let Some(reason) = equatable_reason(ty, type_checker) {
                            err.notes.push(reason);
                        }
                        errors.push(err);
                    }
                }
            }
            Type::Bool
        }

        // comparison ops must be numeric or string
        LessThan | GreaterThan | LessThanEq | GreaterThanEq => {
            let is_comparable = (left_ty.is_num() || left_ty.is_str()) && same_ty;
            if is_comparable {
                Type::Bool
            } else {
                errors.push(TypeErr::new(
                    bin.span,
                    TypeErrKind::MismatchedTypes {
                        expected: left_ty.clone(),
                        found: right_ty.clone(),
                    },
                ));
                Type::Infer
            }
        }

        // logical ops must be bool
        And | Or | Xor => {
            if left_ty.is_bool() && same_ty {
                Type::Bool
            } else {
                let wrong_ty = if !left_ty.is_bool() {
                    left_ty
                } else {
                    right_ty
                };
                errors.push(TypeErr::new(
                    bin.span,
                    TypeErrKind::InvalidOperand {
                        op: node.op.to_string(),
                        operand_type: wrong_ty,
                    },
                ));
                Type::Infer
            }
        }

        Coalesce => check_coalesce(bin, left_ty, right_ty, type_checker, errors),
    }
}

fn check_coalesce(
    bin: &BinaryNode,
    left_ty: Type,
    right_ty: Type,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &bin.node;

    // left must be optional
    let Some(left_inner_ty) = left_ty.option_inner().cloned() else {
        errors.push(TypeErr::new(
            bin.span,
            TypeErrKind::InvalidOperand {
                op: node.op.to_string(),
                operand_type: left_ty,
            },
        ));
        return Type::Infer;
    };

    let right_ref = TypeRef::Expr(node.right.node.id);

    // if right is optional too then we're chaining optionals
    if right_ty.is_option() {
        let right_inner_ty = right_ty.option_inner().cloned().unwrap_or(Type::Infer);
        // constrain the inner types if both are optional
        let left_inner_ref = TypeRef::concrete(&left_inner_ty);
        let right_inner_ref = TypeRef::concrete(&right_inner_ty);
        type_checker.constrain_equal(bin.span, left_inner_ref, right_inner_ref, errors);

        // get the unified inner type
        let unified_inner = type_checker
            .get_type_ref(&right_ref)
            .and_then(|t| t.option_inner().cloned())
            .unwrap_or(left_inner_ty.clone());

        // set the left expression's type to the unified inner type
        let ty = Type::option_of(unified_inner);
        type_checker.set_type(node.left.node.id, ty.clone(), bin.span);

        return ty;
    }

    // if right side is not optional then we're unwrapping or returning the right side
    let left_inner_ref = TypeRef::concrete(&left_inner_ty);
    type_checker.constrain_equal(bin.span, left_inner_ref, right_ref.clone(), errors);

    // get the unified inner type
    let unified_inner = type_checker
        .get_type_ref(&right_ref)
        .unwrap_or(left_inner_ty);

    // set the left expression's type to the unified inner type
    type_checker.set_type(
        node.left.node.id,
        Type::option_of(unified_inner.clone()),
        bin.span,
    );

    unified_inner
}

enum ExternOpResult {
    Found(Type),
    Ambiguous,
}

fn resolve_extern_binary_op(
    op: BinaryOp,
    left_ty: &Type,
    right_ty: &Type,
    type_checker: &TypeChecker,
) -> Option<ExternOpResult> {
    let left_match = if let Type::Extern { name } = left_ty {
        type_checker
            .get_extern_type(*name)
            .and_then(|def| {
                def.operators
                    .iter()
                    .find(|o| o.op == op && !o.self_on_right && o.other_ty == *right_ty)
                    .map(|o| o.ret.clone())
            })
    } else {
        None
    };

    let right_match = if let Type::Extern { name } = right_ty {
        type_checker
            .get_extern_type(*name)
            .and_then(|def| {
                def.operators
                    .iter()
                    .find(|o| o.op == op && o.self_on_right && o.other_ty == *left_ty)
                    .map(|o| o.ret.clone())
            })
    } else {
        None
    };

    match (left_match, right_match) {
        (Some(_), Some(_)) => Some(ExternOpResult::Ambiguous),
        (Some(ret), None) => Some(ExternOpResult::Found(ret)),
        (None, Some(ret)) => Some(ExternOpResult::Found(ret)),
        (None, None) => None,
    }
}

pub(super) fn check_unary(
    unary: &UnaryNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &unary.node;
    let expr_ty = check_expr(&node.expr, type_checker, errors, None);

    match node.op {
        UnaryOp::Neg if expr_ty.is_num() => expr_ty,
        UnaryOp::Not if expr_ty.is_bool() => Type::Bool,
        UnaryOp::Neg => {
            if let Type::Extern { name } = &expr_ty {
                if let Some(def) = type_checker.get_extern_type(*name) {
                    if let Some(op_def) = def.unary_operators.iter().find(|o| o.op == UnaryOp::Neg)
                    {
                        return op_def.ret.clone();
                    }
                }
            }
            errors.push(TypeErr::new(
                unary.span,
                TypeErrKind::InvalidOperand {
                    op: node.op.to_string(),
                    operand_type: expr_ty.clone(),
                },
            ));
            Type::Infer
        }
        _ => {
            errors.push(TypeErr::new(
                unary.span,
                TypeErrKind::InvalidOperand {
                    op: node.op.to_string(),
                    operand_type: expr_ty.clone(),
                },
            ));
            Type::Infer
        }
    }
}

fn immutable_assignment_error(assign: &AssignNode, type_checker: &TypeChecker) -> Option<TypeErr> {
    let root = root_ident(&assign.node.target)?;
    let info = type_checker.get_var(root)?;
    if info.mutable {
        return None;
    }

    Some(
        TypeErr::new(assign.span, TypeErrKind::ImmutableAssignment { name: root })
            .with_help("declare with 'var' to allow mutation"),
    )
}

fn readonly_self_mutation_error(
    assign: &AssignNode,
    type_checker: &TypeChecker,
) -> Option<TypeErr> {
    let method_ctx = type_checker.current_method()?;

    if !matches!(method_ctx.receiver, Some(MethodReceiver::Value)) {
        return None;
    }

    let ExprKind::Field(field_node) = &assign.node.target.node.kind else {
        return None;
    };

    let ExprKind::Ident(ident) = &field_node.node.target.node.kind else {
        return None;
    };

    if ident.0.as_ref() != "self" {
        return None;
    }

    Some(TypeErr::new(
        assign.span,
        TypeErrKind::ReadonlySelfMutation {
            struct_name: method_ctx.struct_name,
            field: field_node.node.field,
        },
    ))
}

pub(super) fn check_assign(
    assign: &AssignNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let maybe_error = readonly_self_mutation_error(assign, type_checker);
    if let Some(error) = maybe_error {
        errors.push(error);
        return Type::Infer;
    }

    if let Some(name) = root_ident(&assign.node.target) {
        if type_checker.const_defs.contains_key(&name) {
            errors.push(TypeErr::new(
                assign.span,
                TypeErrKind::ConstAssignment { name },
            ));
            return Type::Infer;
        }
    }

    if let Some(error) = immutable_assignment_error(assign, type_checker) {
        errors.push(error);
        return Type::Infer;
    }

    let node = &assign.node;
    let target_ty = check_expr(&node.target, type_checker, errors, None);
    let expected = if target_ty != Type::Infer { Some(&target_ty) } else { None };
    check_expr(&node.value, type_checker, errors, expected);

    let target_ref = TypeRef::Expr(node.target.node.id);
    let value_ref = TypeRef::Expr(node.value.node.id);

    match node.op {
        AssignOp::Assign => check_assign_op(assign, target_ref, value_ref, type_checker, errors),
        AssignOp::AddAssign | AssignOp::SubAssign | AssignOp::MulAssign | AssignOp::DivAssign => {
            check_compound_assign_op(assign, target_ref, value_ref, type_checker, errors)
        }
    }
}

fn check_assign_op(
    assign: &AssignNode,
    target_ref: TypeRef,
    value_ref: TypeRef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    type_checker.constrain_assignable(assign.span, value_ref, target_ref.clone(), errors);
    Type::Void
}

fn check_compound_assign_op(
    assign: &AssignNode,
    target_ref: TypeRef,
    value_ref: TypeRef,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let target_ty = type_checker
        .get_type_ref(&target_ref)
        .unwrap_or(Type::Infer);
    let value_ty = type_checker.get_type_ref(&value_ref).unwrap_or(Type::Infer);

    let is_add_assign = assign.node.op == AssignOp::AddAssign;
    let is_str_concat = is_add_assign
        && target_ty.is_str()
        && (value_ty.is_str() || value_ty.is_stringable_primitive());

    if !is_str_concat {
        type_checker.constrain_equal(assign.span, target_ref.clone(), value_ref, errors);
    }

    let target_ty = type_checker
        .get_type_ref(&target_ref)
        .unwrap_or(Type::Infer);
    let is_valid =
        target_ty.is_num() || target_ty.is_infer() || (is_add_assign && target_ty.is_str());
    if !is_valid {
        errors.push(TypeErr::new(
            assign.span,
            TypeErrKind::InvalidOperand {
                op: assign.node.op.to_string(),
                operand_type: target_ty.clone(),
            },
        ));
    }

    Type::Void
}
