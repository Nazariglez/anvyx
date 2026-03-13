use crate::ast::{CastNode, ExprKind, ExprNode, Ident, Lit, StringPart, Type};

use super::{
    composite::{
        check_array_fill, check_array_literal, check_map_literal, check_named_tuple, check_range,
        check_struct_lit, check_tuple, check_tuple_index,
    },
    control::{check_if, check_match},
    error::{TypeErr, TypeErrKind},
    ops::{check_assign, check_binary, check_unary},
    postfix::{check_postfix_chain, collect_postfix_chain},
    stmt::check_block_expr,
    types::TypeChecker,
};

pub(super) fn check_expr(
    expr_node: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    // route all postfix expressions through the shared chain checker
    if matches!(
        expr_node.node.kind,
        ExprKind::Field(_) | ExprKind::Index(_) | ExprKind::Call(_)
    ) {
        let (base, chain) = collect_postfix_chain(expr_node);
        return check_postfix_chain(expr_node, base, &chain, type_checker, errors);
    }

    let expr = &expr_node.node;
    let ty = match &expr.kind {
        ExprKind::Ident(ident) => match type_checker.get_var(*ident) {
            Some(info) => info.ty.clone(),
            None => {
                errors.push(TypeErr::new(
                    expr_node.span,
                    TypeErrKind::UnknownVariable { name: *ident },
                ));
                Type::Infer
            }
        },
        ExprKind::Block(spanned) => {
            let (block_ty, _) = check_block_expr(spanned, type_checker, errors);
            block_ty
        }
        ExprKind::Lit(lit) => type_from_lit(lit),
        ExprKind::Binary(bin) => check_binary(bin, type_checker, errors),
        ExprKind::Unary(unary) => check_unary(unary, type_checker, errors),
        ExprKind::Assign(assign) => check_assign(assign, type_checker, errors),
        ExprKind::If(if_node) => check_if(if_node, type_checker, errors),
        ExprKind::Tuple(elements) => check_tuple(elements, type_checker, errors),
        ExprKind::NamedTuple(elements) => {
            check_named_tuple(elements, expr_node.span, type_checker, errors)
        }
        ExprKind::TupleIndex(index_node) => check_tuple_index(index_node, type_checker, errors),
        ExprKind::StructLiteral(lit_node) => check_struct_lit(lit_node, type_checker, errors),
        ExprKind::Range(range_node) => check_range(range_node, type_checker, errors),
        ExprKind::ArrayLiteral(lit_node) => check_array_literal(lit_node, type_checker, errors),
        ExprKind::ArrayFill(fill_node) => check_array_fill(fill_node, type_checker, errors),
        ExprKind::MapLiteral(lit_node) => check_map_literal(lit_node, type_checker, errors),
        ExprKind::Match(match_node) => check_match(match_node, type_checker, errors),
        ExprKind::StringInterp(parts) => check_string_interp(parts, type_checker, errors),
        ExprKind::Cast(cast_node) => check_cast(cast_node, type_checker, errors),
        ExprKind::Field(_) | ExprKind::Index(_) | ExprKind::Call(_) => {
            unreachable!("postfix expressions should be routed through check_postfix_chain")
        }
    };

    type_checker.set_type(expr_node.node.id, ty.clone(), expr_node.span);
    ty
}

pub(super) fn root_ident(expr: &ExprNode) -> Option<Ident> {
    match &expr.node.kind {
        ExprKind::Ident(name) => Some(*name),
        ExprKind::Field(field) => root_ident(&field.node.target),
        ExprKind::Index(index) => root_ident(&index.node.target),
        _ => None,
    }
}

pub(super) fn type_from_lit(lit: &Lit) -> Type {
    match lit {
        Lit::Int(_) => Type::Int,
        Lit::Float(_) => Type::Float,
        Lit::Bool(_) => Type::Bool,
        Lit::String(_) => Type::String,
        Lit::Nil => Type::option_of(Type::Infer),
    }
}

fn check_cast(
    cast_node: &CastNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    let from_ty = check_expr(&cast_node.node.expr, type_checker, errors);
    let to_ty = &cast_node.node.target;

    let valid = match (&from_ty, to_ty) {
        (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,
        _ => from_ty == *to_ty,
    };

    if !valid {
        errors.push(TypeErr::new(
            cast_node.span,
            TypeErrKind::InvalidCast {
                from: from_ty,
                to: to_ty.clone(),
            },
        ));
    }

    to_ty.clone()
}

fn check_string_interp(
    parts: &[StringPart],
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    for part in parts {
        let StringPart::Expr(expr_node) = part else {
            continue;
        };
        let expr_ty = check_expr(expr_node, type_checker, errors);
        let is_valid = expr_ty.is_str()
            || expr_ty.is_stringable_primitive()
            || expr_ty.is_infer();
        if !is_valid {
            errors.push(TypeErr::new(
                expr_node.span,
                TypeErrKind::MismatchedTypes {
                    expected: Type::String,
                    found: expr_ty,
                },
            ));
        }
    }
    Type::String
}
