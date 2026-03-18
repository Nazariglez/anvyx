use crate::{
    ast,
    lexer::{Keyword, Op, Token},
    span::{Span, Spanned},
};
use chumsky::prelude::*;

use super::common::block_stmt;
use super::decl::function;
use super::expr::{cond_expression, expression};
use super::pattern::pattern;
use super::types::type_ident;
use super::{AnvParser, BoxedParser};

pub(super) fn statement<'src>() -> BoxedParser<'src, ast::StmtNode> {
    recursive(|stmt| {
        let expr = expression(stmt.clone());
        let func = function(stmt.clone());
        let bind = binding(stmt.clone());
        let ret = return_stmt(stmt.clone());
        let while_s = while_stmt(stmt.clone(), expr.clone());
        let for_s = for_stmt(stmt.clone(), expr.clone());
        let break_s = break_stmt();
        let continue_s = continue_stmt();

        let at_stmt_start = select! {
            (Token::Keyword(Keyword::Let), _) => (),
            (Token::Keyword(Keyword::Var), _) => (),
            (Token::Keyword(Keyword::Return), _) => (),
            (Token::Keyword(Keyword::Fn), _) => (),
            (Token::Keyword(Keyword::Pub), _) => (),
            (Token::Keyword(Keyword::If), _) => (),
            (Token::Keyword(Keyword::Match), _) => (),
            (Token::Keyword(Keyword::Struct), _) => (),
            (Token::Keyword(Keyword::While), _) => (),
            (Token::Keyword(Keyword::For), _) => (),
            (Token::Keyword(Keyword::Break), _) => (),
            (Token::Keyword(Keyword::Continue), _) => (),
        }
        .rewind();

        let at_assign_start = select! { (Token::Ident(_), _) => () }
            .then(select! {
                (Token::Op(Op::Assign), _) => (),
                (Token::Op(Op::AddAssign), _) => (),
                (Token::Op(Op::SubAssign), _) => (),
                (Token::Op(Op::MulAssign), _) => (),
                (Token::Op(Op::DivAssign), _) => (),
            })
            .to(())
            .rewind();

        let expr_stmt = expr
            .then_ignore(
                select! { (Token::Semicolon, _) => () }
                    .or(at_stmt_start)
                    .or(at_assign_start),
            )
            .map(|expr_node| {
                let span = expr_node.span;
                Spanned::new(ast::Stmt::Expr(expr_node), span)
            });

        choice((
            func.map(|func_node| {
                let span = func_node.span;
                Spanned::new(ast::Stmt::Func(func_node), span)
            }),
            bind.map(|bind_node| {
                let span = bind_node.span;
                Spanned::new(ast::Stmt::Binding(bind_node), span)
            }),
            ret,
            while_s,
            for_s,
            break_s,
            continue_s,
            expr_stmt,
        ))
    })
    .labelled("statement")
    .as_context()
    .boxed()
}

fn binding<'src>(stmt: impl AnvParser<'src, ast::StmtNode>) -> BoxedParser<'src, ast::BindingNode> {
    let mutability = select! {
        (Token::Keyword(Keyword::Let), _) => ast::Mutability::Immutable,
        (Token::Keyword(Keyword::Var), _) => ast::Mutability::Mutable,
    };

    mutability
        .then(pattern())
        .then(
            select! {
                (Token::Colon, _) => (),
            }
            .ignore_then(type_ident())
            .or_not(),
        )
        .then_ignore(select! {
            (Token::Op(Op::Assign), _) => (),
        })
        .then(expression(stmt))
        .then_ignore(select! {
            (Token::Semicolon, _) => (),
        })
        .map_with(|(((mutability, pat), ty), value), e| {
            let s = e.span();
            Spanned::new(
                ast::Binding {
                    pattern: pat,
                    ty,
                    mutability,
                    value,
                },
                Span::new(s.start, s.end),
            )
        })
        .boxed()
}

fn while_stmt<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
    expr: impl AnvParser<'src, ast::ExprNode>,
) -> BoxedParser<'src, ast::StmtNode> {
    let cond_expr = cond_expression();

    select! {
        (Token::Keyword(Keyword::While), _) => (),
    }
    .ignore_then(cond_expr)
    .then(block_stmt(stmt, expr))
    .map_with(|(cond, body), e| {
        let s = e.span();
        let span = Span::new(s.start, s.end);
        let while_node = Spanned::new(ast::While { cond, body }, span);
        Spanned::new(ast::Stmt::While(while_node), span)
    })
    .labelled("while statement")
    .as_context()
    .boxed()
}

// rev key is only valid on for loops
fn contextual_rev<'src>() -> BoxedParser<'src, bool> {
    select! {
        (Token::Ident(ident), _) if ident.0.as_ref() == "rev" => true,
    }
    .or_not()
    .map(|o| o.unwrap_or(false))
    .boxed()
}

// step key is only valid on for loops
fn contextual_step<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, Option<ast::ExprNode>> {
    select! {
        (Token::Ident(ident), _) if ident.0.as_ref() == "step" => (),
    }
    .ignore_then(expression(stmt))
    .or_not()
    .boxed()
}

fn for_stmt<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
    expr: impl AnvParser<'src, ast::ExprNode>,
) -> BoxedParser<'src, ast::StmtNode> {
    select! {
        (Token::Keyword(Keyword::For), _) => (),
    }
    .ignore_then(pattern())
    .then_ignore(select! {
        (Token::Keyword(Keyword::In), _) => (),
    })
    .then(contextual_rev())
    .then(expression(stmt.clone()))
    .then(contextual_step(stmt.clone()))
    .then(block_stmt(stmt, expr))
    .map_with(|((((pat, reversed), iterable), step), body), e| {
        let s = e.span();
        let span = Span::new(s.start, s.end);
        let for_node = Spanned::new(
            ast::For {
                pattern: pat,
                iterable,
                step,
                reversed,
                body,
            },
            span,
        );
        Spanned::new(ast::Stmt::For(for_node), span)
    })
    .labelled("for statement")
    .as_context()
    .boxed()
}

fn break_stmt<'src>() -> BoxedParser<'src, ast::StmtNode> {
    select! {
        (Token::Keyword(Keyword::Break), _) => (),
    }
    .then_ignore(select! {
        (Token::Semicolon, _) => (),
    })
    .map_with(|(), e| {
        let s: chumsky::span::SimpleSpan<usize> = e.span();
        let span = Span::new(s.start, s.end);
        Spanned::new(ast::Stmt::Break, span)
    })
    .labelled("break statement")
    .as_context()
    .boxed()
}

fn continue_stmt<'src>() -> BoxedParser<'src, ast::StmtNode> {
    select! {
        (Token::Keyword(Keyword::Continue), _) => (),
    }
    .then_ignore(select! {
        (Token::Semicolon, _) => (),
    })
    .map_with(|(), e| {
        let s: chumsky::span::SimpleSpan<usize> = e.span();
        let span = Span::new(s.start, s.end);
        Spanned::new(ast::Stmt::Continue, span)
    })
    .labelled("continue statement")
    .as_context()
    .boxed()
}

fn return_stmt<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, ast::StmtNode> {
    select! {
        (Token::Keyword(Keyword::Return), _) => (),
    }
    .ignore_then(expression(stmt).or_not())
    .then_ignore(select! {
        (Token::Semicolon, _) => (),
    })
    .map_with(|value_opt, e| {
        let s = e.span();
        let span = Span::new(s.start, s.end);
        let ret = ast::Return { value: value_opt };
        Spanned::new(ast::Stmt::Return(Spanned::new(ret, span)), span)
    })
    .labelled("return")
    .as_context()
    .boxed()
}
