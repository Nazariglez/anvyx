use crate::{
    ast,
    lexer::{Op, Token},
    span::{Span, Spanned},
};
use chumsky::prelude::*;

use super::{AnvParser, BoxedParser};

pub(super) fn infix_left<'src>(
    lower: impl AnvParser<'src, ast::ExprNode>,
    op: impl AnvParser<'src, ast::BinaryOp>,
) -> BoxedParser<'src, ast::ExprNode> {
    let op_rhs = op.then(lower.clone());
    lower
        .foldl_with(op_rhs.repeated(), |left, (op, right), e| {
            let span = Span::new(left.span.start, right.span.end);
            let bin_node = Spanned::new(
                ast::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                span,
            );

            let expr_id = e.state().new_expr_id();
            let expr = ast::Expr::new(ast::ExprKind::Binary(bin_node), expr_id);
            Spanned::new(expr, span)
        })
        .boxed()
}

pub(super) fn mul_div_op<'src>() -> BoxedParser<'src, ast::BinaryOp> {
    select! {
        (Token::Op(Op::Mul), _) => ast::BinaryOp::Mul,
        (Token::Op(Op::Div), _) => ast::BinaryOp::Div,
        (Token::Op(Op::Rem), _) => ast::BinaryOp::Rem,
    }
    .labelled("multiplicative op")
    .as_context()
    .boxed()
}

pub(super) fn add_sub_op<'src>() -> BoxedParser<'src, ast::BinaryOp> {
    select! {
        (Token::Op(Op::Add), _) => ast::BinaryOp::Add,
        (Token::Op(Op::Sub), _) => ast::BinaryOp::Sub,
    }
    .labelled("additive op")
    .as_context()
    .boxed()
}

pub(super) fn cmp_op<'src>() -> BoxedParser<'src, ast::BinaryOp> {
    select! {
        (Token::Op(Op::LessThan), _) => ast::BinaryOp::LessThan,
        (Token::Op(Op::GreaterThan), _) => ast::BinaryOp::GreaterThan,
        (Token::Op(Op::LessThanEq), _) => ast::BinaryOp::LessThanEq,
        (Token::Op(Op::GreaterThanEq), _) => ast::BinaryOp::GreaterThanEq,
    }
    .labelled("comparison op")
    .as_context()
    .boxed()
}

pub(super) fn eq_op<'src>() -> BoxedParser<'src, ast::BinaryOp> {
    select! {
        (Token::Op(Op::Eq), _) => ast::BinaryOp::Eq,
        (Token::Op(Op::NotEq), _) => ast::BinaryOp::NotEq,
    }
    .labelled("equality op")
    .as_context()
    .boxed()
}

pub(super) fn and_op<'src>() -> BoxedParser<'src, ast::BinaryOp> {
    select! {
        (Token::Op(Op::And), _) => ast::BinaryOp::And,
    }
    .labelled("logical and op")
    .as_context()
    .boxed()
}

pub(super) fn coalesce_op<'src>() -> BoxedParser<'src, ast::BinaryOp> {
    select! {
        (Token::Op(Op::Coalesce), _) => ast::BinaryOp::Coalesce,
    }
    .labelled("coalesce op")
    .as_context()
    .boxed()
}

pub(super) fn or_op<'src>() -> BoxedParser<'src, ast::BinaryOp> {
    select! {
        (Token::Op(Op::Or), _) => ast::BinaryOp::Or,
    }
    .labelled("logical or op")
    .as_context()
    .boxed()
}

pub(super) fn assign_op<'src>() -> BoxedParser<'src, ast::AssignOp> {
    select! {
        (Token::Op(Op::Assign), _) => ast::AssignOp::Assign,
        (Token::Op(Op::AddAssign), _) => ast::AssignOp::AddAssign,
        (Token::Op(Op::SubAssign), _) => ast::AssignOp::SubAssign,
        (Token::Op(Op::MulAssign), _) => ast::AssignOp::MulAssign,
        (Token::Op(Op::DivAssign), _) => ast::AssignOp::DivAssign,
    }
    .labelled("assign op")
    .as_context()
    .boxed()
}
