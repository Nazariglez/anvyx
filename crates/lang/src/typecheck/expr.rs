use crate::ast::{ExprKind, ExprNode, Lit, Type};

use super::{
    call::check_call,
    composite::{
        check_array_fill, check_array_literal, check_field_access, check_index,
        check_map_literal, check_named_tuple, check_range, check_struct_lit, check_tuple,
        check_tuple_index,
    },
    control::{check_if, check_match},
    error::{TypeErr, TypeErrKind},
    ops::{check_assign, check_binary, check_unary},
    postfix::{chain_has_safe_op, check_postfix_chain, collect_postfix_chain},
    stmt::check_block_expr,
    types::TypeChecker,
};

pub(super) fn check_expr(
    expr_node: &ExprNode,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    // for postfix expressions (field, index, call) check if there is optional chaining
    if matches!(
        expr_node.node.kind,
        ExprKind::Field(_) | ExprKind::Index(_) | ExprKind::Call(_)
    ) {
        let (base, chain) = collect_postfix_chain(expr_node);
        if chain_has_safe_op(&chain) {
            return check_postfix_chain(expr_node, base, &chain, type_checker, errors);
        }
    }

    let expr = &expr_node.node;
    let ty = match &expr.kind {
        ExprKind::Ident(ident) => match type_checker.get_var(*ident) {
            Some(ty) => ty.clone(),
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
        ExprKind::Call(call) => check_call(call, type_checker, errors),
        ExprKind::Binary(bin) => check_binary(bin, type_checker, errors),
        ExprKind::Unary(unary) => check_unary(unary, type_checker, errors),
        ExprKind::Assign(assign) => check_assign(assign, type_checker, errors),
        ExprKind::If(if_node) => check_if(if_node, type_checker, errors),
        ExprKind::Tuple(elements) => check_tuple(elements, type_checker, errors),
        ExprKind::NamedTuple(elements) => {
            check_named_tuple(elements, expr_node.span, type_checker, errors)
        }
        ExprKind::TupleIndex(index_node) => check_tuple_index(index_node, type_checker, errors),
        ExprKind::Field(field_node) => check_field_access(field_node, type_checker, errors),
        ExprKind::StructLiteral(lit_node) => check_struct_lit(lit_node, type_checker, errors),
        ExprKind::Range(range_node) => check_range(range_node, type_checker, errors),
        ExprKind::ArrayLiteral(lit_node) => check_array_literal(lit_node, type_checker, errors),
        ExprKind::ArrayFill(fill_node) => check_array_fill(fill_node, type_checker, errors),
        ExprKind::MapLiteral(lit_node) => check_map_literal(lit_node, type_checker, errors),
        ExprKind::Index(index_node) => check_index(index_node, type_checker, errors),
        ExprKind::Match(match_node) => check_match(match_node, type_checker, errors),
    };

    type_checker.set_type(expr_node.node.id, ty.clone(), expr_node.span);
    ty
}

pub(super) fn type_from_lit(lit: &Lit) -> Type {
    match lit {
        Lit::Int(_) => Type::Int,
        Lit::Float(_) => Type::Float,
        Lit::Bool(_) => Type::Bool,
        Lit::String(_) => Type::String,
        Lit::Nil => Type::Optional(Box::new(Type::Infer)),
    }
}
