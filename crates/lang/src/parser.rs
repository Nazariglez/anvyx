use crate::ast;
use crate::lexer::{Delimiter, Keyword, LitToken, Op, Token};
use chumsky::extra;
use chumsky::prelude::*;

pub fn parse_ast(tokens: &[Token]) -> Result<ast::Program, String> {
    parser().parse(tokens).into_result().map_err(|errors| {
        errors
            .into_iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    })
}

fn parser<'src>()
-> impl Parser<'src, &'src [Token], ast::Program, extra::Full<Simple<'src, Token>, (), ()>> + 'src {
    statement()
        .repeated()
        .collect::<Vec<_>>()
        .map(|stmts| ast::Program { stmts })
}

fn statement<'src>()
-> impl Parser<'src, &'src [Token], ast::Stmt, extra::Full<Simple<'src, Token>, (), ()>> + 'src {
    recursive(|stmt| {
        let expr = expression(stmt.clone());
        choice((
            function(stmt.clone()).map(ast::Stmt::Func),
            expr.map(ast::Stmt::Expr),
        ))
        .boxed()
    })
}

fn function<'src>(
    stmt: impl Parser<'src, &'src [Token], ast::Stmt, extra::Full<Simple<'src, Token>, (), ()>> + Clone,
) -> impl Parser<'src, &'src [Token], ast::Func, extra::Full<Simple<'src, Token>, (), ()>> {
    select! {
        Token::Keyword(Keyword::Fn) => (),
    }
    .ignore_then(term_ident())
    .then(params())
    .then(return_type())
    .then(block_stmt(stmt))
    .map(|(((name, params), ret), body)| ast::Func {
        name,
        visibility: ast::Visibility::Private,
        params,
        ret: ret.unwrap_or(ast::Type::Void),
        body,
    })
}

fn block_stmt<'src>(
    stmt: impl Parser<'src, &'src [Token], ast::Stmt, extra::Full<Simple<'src, Token>, (), ()>> + Clone,
) -> impl Parser<'src, &'src [Token], ast::Block, extra::Full<Simple<'src, Token>, (), ()>> {
    select! {
        Token::Open(Delimiter::Brace) => (),
    }
    .ignore_then(stmt.repeated().collect::<Vec<_>>())
    .then_ignore(select! {
        Token::Close(Delimiter::Brace) => (),
    })
    .map(|stmts| ast::Block { stmts })
}

fn expression<'src>(
    stmt: impl Parser<'src, &'src [Token], ast::Stmt, extra::Full<Simple<'src, Token>, (), ()>> + Clone,
) -> impl Parser<'src, &'src [Token], ast::Expr, extra::Full<Simple<'src, Token>, (), ()>> {
    choice((
        literal().map(ast::Expr::Lit),
        identifier().map(ast::Expr::Ident),
        block_stmt(stmt).map(ast::Expr::Block),
    ))
}

fn identifier<'src>()
-> impl Parser<'src, &'src [Token], ast::Ident, extra::Full<Simple<'src, Token>, (), ()>> {
    term_ident()
}

fn term_ident<'src>()
-> impl Parser<'src, &'src [Token], ast::Ident, extra::Full<Simple<'src, Token>, (), ()>> {
    select! {
        Token::TermIdent(ident) | Token::TypeIdent(ident) => ident,
    }
}

fn literal<'src>()
-> impl Parser<'src, &'src [Token], ast::Lit, extra::Full<Simple<'src, Token>, (), ()>> {
    select! {
        Token::Literal(lit) => match lit {
            LitToken::Number(n) => ast::Lit::Int(n),
            LitToken::Float(s) => {
                s.parse::<f64>()
                    .map(ast::Lit::Float)
                    .unwrap_or(ast::Lit::Float(0.0))
            }
            LitToken::String(s) => ast::Lit::String(s.to_string()),
        },
        Token::Keyword(Keyword::True) => ast::Lit::Bool(true),
        Token::Keyword(Keyword::False) => ast::Lit::Bool(false),
        Token::Keyword(Keyword::Nil) => ast::Lit::Nil,
    }
}

fn params<'src>()
-> impl Parser<'src, &'src [Token], Vec<ast::Param>, extra::Full<Simple<'src, Token>, (), ()>> {
    select! {
        Token::Open(Delimiter::Parent) => (),
    }
    .ignore_then(param().repeated().collect::<Vec<_>>())
    .then_ignore(select! {
        Token::Close(Delimiter::Parent) => (),
    })
}

fn param<'src>()
-> impl Parser<'src, &'src [Token], ast::Param, extra::Full<Simple<'src, Token>, (), ()>> {
    term_ident()
        .then(type_())
        .map(|(name, ty)| ast::Param { name, ty })
}

fn return_type<'src>()
-> impl Parser<'src, &'src [Token], Option<ast::Type>, extra::Full<Simple<'src, Token>, (), ()>> {
    select! {
        Token::Op(Op::ThinArrow) => (),
    }
    .ignore_then(type_())
    .or_not()
}

fn type_<'src>()
-> impl Parser<'src, &'src [Token], ast::Type, extra::Full<Simple<'src, Token>, (), ()>> {
    select! {
        Token::Keyword(Keyword::Int) => ast::Type::Int,
        Token::Keyword(Keyword::Float) => ast::Type::Float,
        Token::Keyword(Keyword::Bool) => ast::Type::Bool,
        Token::Keyword(Keyword::String) => ast::Type::String,
        Token::Keyword(Keyword::Void) => ast::Type::Void,
    }
}
