use crate::ast::{
    AssignNode, AssignOp, BinaryNode, BinaryOp, ExprKind, ExprNode, Ident, MethodReceiver, Type,
    UnaryNode, UnaryOp,
};

use super::{
    constraint::TypeRef,
    error::{TypeErr, TypeErrKind},
    expr::check_expr,
    types::TypeChecker,
};

pub(super) fn check_binary(
    bin: &BinaryNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    use BinaryOp::*;

    let node = &bin.node;
    let left_ty = check_expr(&node.left, type_checker, errors);
    let right_ty = check_expr(&node.right, type_checker, errors);
    let same_ty = left_ty == right_ty;

    match node.op {
        // numeric ops
        Add | Sub | Mul | Div | Rem => {
            if left_ty.is_num() && same_ty {
                left_ty
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

        // equal ops must be the same type
        Eq | NotEq => {
            if !same_ty {
                errors.push(TypeErr::new(
                    bin.span,
                    TypeErrKind::MismatchedTypes {
                        expected: left_ty.clone(),
                        found: right_ty.clone(),
                    },
                ));
            }
            Type::Bool
        }

        // comparison ops must be numeric
        LessThan | GreaterThan | LessThanEq | GreaterThanEq => {
            if left_ty.is_num() && same_ty {
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
    let Type::Optional(left_inner) = left_ty.clone() else {
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
    let left_inner_ty = *left_inner;

    // if right is optional too then we're chaining optionals
    if let Type::Optional(right_inner) = right_ty.clone() {
        // constrain the inner types if both are optional
        let left_inner_ref = TypeRef::Concrete(left_inner_ty.clone());
        let right_inner_ref = TypeRef::Concrete(*right_inner.clone());
        type_checker.constrain_equal(bin.span, left_inner_ref, right_inner_ref, errors);

        // get the unified inner type
        let unified_inner = type_checker
            .get_type_ref(&right_ref)
            .and_then(|t| {
                if let Type::Optional(inner) = t {
                    Some(*inner)
                } else {
                    None
                }
            })
            .unwrap_or(left_inner_ty.clone());

        // set the left expression's type to the unified inner type
        let ty = Type::Optional(Box::new(unified_inner));
        type_checker.set_type(node.left.node.id, ty.clone(), bin.span);

        return ty;
    }

    // if right side is not optional then we're unwrapping or returning the right side
    let left_inner_ref = TypeRef::Concrete(left_inner_ty.clone());
    type_checker.constrain_equal(bin.span, left_inner_ref, right_ref.clone(), errors);

    // get the unified inner type
    let unified_inner = type_checker
        .get_type_ref(&right_ref)
        .unwrap_or(left_inner_ty);

    // set the left expression's type to the unified inner type
    type_checker.set_type(
        node.left.node.id,
        Type::Optional(Box::new(unified_inner.clone())),
        bin.span,
    );

    unified_inner
}

pub(super) fn check_unary(
    unary: &UnaryNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let node = &unary.node;
    let expr_ty = check_expr(&node.expr, type_checker, errors);

    match node.op {
        UnaryOp::Neg if expr_ty.is_num() => expr_ty,
        UnaryOp::Not if expr_ty.is_bool() => Type::Bool,
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

fn root_ident(expr: &ExprNode) -> Option<Ident> {
    match &expr.node.kind {
        ExprKind::Ident(name) => Some(*name),
        ExprKind::Field(field) => root_ident(&field.node.target),
        ExprKind::Index(index) => root_ident(&index.node.target),
        _ => None,
    }
}

fn immutable_assignment_error(
    assign: &AssignNode,
    type_checker: &TypeChecker,
) -> Option<TypeErr> {
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

    if let Some(error) = immutable_assignment_error(assign, type_checker) {
        errors.push(error);
        return Type::Infer;
    }

    let node = &assign.node;
    check_expr(&node.target, type_checker, errors);
    check_expr(&node.value, type_checker, errors);

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
    type_checker.constrain_equal(assign.span, target_ref.clone(), value_ref, errors);

    let target_ty = type_checker
        .get_type_ref(&target_ref)
        .unwrap_or(Type::Infer);

    let is_numeric = target_ty.is_num() || target_ty.is_infer();
    if !is_numeric {
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
