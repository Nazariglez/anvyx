use crate::{
    ast,
    lexer::{Delimiter, FloatSuffix, Keyword, LitToken, Op, Token},
    span::{Span, Spanned},
};
use chumsky::prelude::*;
use internment::Intern;

use super::types::{param_type_ident, type_ident};
use super::{AnvParser, BoxedParser};

pub(super) fn identifier<'src>() -> BoxedParser<'src, ast::Ident> {
    select! {
        (Token::Ident(ident), _) => ident,
    }
    .labelled("identifier")
    .as_context()
    .boxed()
}

pub(super) fn keyword_as_ident<'src>() -> BoxedParser<'src, ast::Ident> {
    select! {
        (Token::Keyword(kw), _) => ast::Ident(Intern::new(kw.to_string()))
    }
    .labelled("identifier")
    .boxed()
}

pub(super) fn field_name_ident<'src>() -> BoxedParser<'src, ast::Ident> {
    choice((identifier(), keyword_as_ident())).boxed()
}

pub(super) fn literal<'src>() -> BoxedParser<'src, ast::Lit> {
    select! {
        (Token::Literal(lit), _) => match lit {
            LitToken::Number(n) => ast::Lit::Int(n),
            LitToken::Float(s, suffix) => {
                let ast_suffix = suffix.map(|s| match s {
                    FloatSuffix::F => ast::FloatSuffix::F,
                    FloatSuffix::D => ast::FloatSuffix::D,
                });
                let value = s.as_ref().parse::<f64>().unwrap_or(0.0);
                ast::Lit::Float { value, suffix: ast_suffix }
            }
            LitToken::String(s) => ast::Lit::String(s.to_string()),
        },
        (Token::Keyword(Keyword::True), _) => ast::Lit::Bool(true),
        (Token::Keyword(Keyword::False), _) => ast::Lit::Bool(false),
        (Token::Keyword(Keyword::Nil), _) => ast::Lit::Nil,
    }
    .labelled("literal")
    .as_context()
    .boxed()
}

pub(super) fn params<'src>() -> BoxedParser<'src, Vec<ast::Param>> {
    select! {
        (Token::Open(Delimiter::Parent), _) => (),
    }
    .ignore_then(
        param()
            .separated_by(select! {
                (Token::Comma, _) => (),
            })
            .collect::<Vec<_>>()
            .or_not()
            .map(|opt| opt.unwrap_or_default()),
    )
    .then_ignore(select! {
        (Token::Close(Delimiter::Parent), _) => (),
    })
    .boxed()
}

pub(super) fn param<'src>() -> BoxedParser<'src, ast::Param> {
    let var_kw = select! {
        (Token::Keyword(Keyword::Var), _) => (),
    }
    .or_not()
    .map(|opt| match opt {
        Some(()) => ast::Mutability::Mutable,
        None => ast::Mutability::Immutable,
    });

    var_kw
        .then(identifier())
        .then_ignore(select! {
            (Token::Colon, _) => (),
        })
        .then(param_type_ident())
        .map(|((mutability, name), ty)| ast::Param {
            mutability,
            name,
            ty,
        })
        .labelled("parameter")
        .as_context()
        .boxed()
}

pub(super) fn return_type<'src>() -> BoxedParser<'src, Option<ast::Type>> {
    select! {
        (Token::Op(Op::ThinArrow), _) => (),
    }
    .ignore_then(type_ident())
    .or_not()
    .labelled("return type")
    .as_context()
    .boxed()
}

pub(super) fn block_stmt<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
    tail_expr: impl AnvParser<'src, ast::ExprNode>,
) -> BoxedParser<'src, ast::BlockNode> {
    select! {
        (Token::Open(Delimiter::Brace), _) => (),
    }
    .ignore_then(stmt.repeated().collect::<Vec<_>>())
    .then(tail_expr.or_not())
    .then_ignore(select! {
        (Token::Close(Delimiter::Brace), _) => (),
    })
    .map_with(|(stmts, tail), e| {
        let s = e.span();
        Spanned::new(ast::Block { stmts, tail: tail.map(Box::new) }, Span::new(s.start, s.end))
    })
    .labelled("block")
    .as_context()
    .boxed()
}

// Shared helper for validating tuple shapes (used by expr and types)
pub(super) enum TupleShapeResult<T> {
    Empty,
    OneTupleError(T),
    Grouped(T),
    Tuple(Vec<T>),
    UnexpectedComma,
}

pub(super) fn validate_tuple_shape_raw<T>(
    first: Option<T>,
    mut rest: Vec<T>,
    trailing_comma: bool,
) -> TupleShapeResult<T> {
    match (first, rest.len(), trailing_comma) {
        (None, 0, _) => TupleShapeResult::Empty,
        (Some(single), 0, true) => TupleShapeResult::OneTupleError(single),
        (Some(single), 0, false) => TupleShapeResult::Grouped(single),
        (Some(first), _, _) => {
            rest.insert(0, first);
            TupleShapeResult::Tuple(rest)
        }
        (None, _, _) => TupleShapeResult::UnexpectedComma,
    }
}
