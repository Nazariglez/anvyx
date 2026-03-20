use chumsky::{Parser, error::Rich, extra, input::InputRef, prelude::*};
use chumsky::input::Cursor;
use internment::Intern;
use std::fmt::Display;

use crate::{ast, span::Span};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Keyword(Keyword),
    Open(Delimiter),
    Close(Delimiter),
    Ident(ast::Ident),
    Literal(LitToken),
    Op(Op),
    Interp(InterpToken),
    Colon,
    Semicolon,
    Comma,
    Question,
    Dot,
    Range,
    RangeEq,
}

pub type SpannedToken = (Token, Span);

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Keyword(keyword) => write!(f, "{}", keyword),
            Token::Ident(ident) => write!(f, "{}", ident),
            Token::Literal(lit_token) => write!(f, "{}", lit_token),
            Token::Open(Delimiter::Parent) => write!(f, "("),
            Token::Open(Delimiter::Brace) => write!(f, "{{"),
            Token::Open(Delimiter::Bracket) => write!(f, "["),
            Token::Close(Delimiter::Parent) => write!(f, ")"),
            Token::Close(Delimiter::Brace) => write!(f, "}}"),
            Token::Close(Delimiter::Bracket) => write!(f, "]"),
            Token::Op(op) => write!(f, "{}", op),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Comma => write!(f, ","),
            Token::Question => write!(f, "?"),
            Token::Dot => write!(f, "."),
            Token::Range => write!(f, ".."),
            Token::RangeEq => write!(f, "..="),
            Token::Interp(interp) => write!(f, "{}", interp),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LitToken {
    Number(i64),
    Float(Intern<String>),
    String(Intern<String>),
}

impl Display for LitToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LitToken::Number(n) => write!(f, "{}", n),
            LitToken::Float(s) => write!(f, "{}", s),
            LitToken::String(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterpToken {
    Start,
    Text(Intern<String>),
    ExprStart,
    ExprEnd,
    End,
}

impl Display for InterpToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpToken::Start => write!(f, "\"..."),
            InterpToken::Text(s) => write!(f, "{}", s),
            InterpToken::ExprStart => write!(f, "{{"),
            InterpToken::ExprEnd => write!(f, "}}"),
            InterpToken::End => write!(f, "...\""),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Delimiter {
    Parent,
    Brace,
    Bracket,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Keyword {
    Int,
    Float,
    Bool,
    String,
    Void,
    Any,
    Nil,
    True,
    False,
    Fn,
    Return,
    Let,
    Var,
    If,
    Else,
    While,
    For,
    In,
    Break,
    Continue,
    Match,
    Pub,
    Struct,
    Enum,
    As,
    Extern,
    Type,
    Import,
}

impl Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Keyword::Int => write!(f, "int"),
            Keyword::Float => write!(f, "float"),
            Keyword::Bool => write!(f, "bool"),
            Keyword::String => write!(f, "string"),
            Keyword::Void => write!(f, "void"),
            Keyword::Any => write!(f, "any"),
            Keyword::Nil => write!(f, "nil"),
            Keyword::True => write!(f, "true"),
            Keyword::False => write!(f, "false"),
            Keyword::Fn => write!(f, "fn"),
            Keyword::Return => write!(f, "return"),
            Keyword::Let => write!(f, "let"),
            Keyword::Var => write!(f, "var"),
            Keyword::If => write!(f, "if"),
            Keyword::Else => write!(f, "else"),
            Keyword::While => write!(f, "while"),
            Keyword::For => write!(f, "for"),
            Keyword::In => write!(f, "in"),
            Keyword::Break => write!(f, "break"),
            Keyword::Continue => write!(f, "continue"),
            Keyword::Match => write!(f, "match"),
            Keyword::Pub => write!(f, "pub"),
            Keyword::Struct => write!(f, "struct"),
            Keyword::Enum => write!(f, "enum"),
            Keyword::As => write!(f, "as"),
            Keyword::Extern => write!(f, "extern"),
            Keyword::Type => write!(f, "type"),
            Keyword::Import => write!(f, "import"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    NotEq,
    LessThan,
    GreaterThan,
    LessThanEq,
    GreaterThanEq,
    And,
    Or,
    Not,
    Coalesce,
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ThinArrow,
    FatArrow,
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add => write!(f, "+"),
            Op::Sub => write!(f, "-"),
            Op::Mul => write!(f, "*"),
            Op::Div => write!(f, "/"),
            Op::Rem => write!(f, "%"),
            Op::Eq => write!(f, "=="),
            Op::NotEq => write!(f, "!="),
            Op::LessThan => write!(f, "<"),
            Op::GreaterThan => write!(f, ">"),
            Op::LessThanEq => write!(f, "<="),
            Op::GreaterThanEq => write!(f, ">="),
            Op::And => write!(f, "&&"),
            Op::Or => write!(f, "||"),
            Op::Not => write!(f, "!"),
            Op::Coalesce => write!(f, "??"),
            Op::Assign => write!(f, "="),
            Op::AddAssign => write!(f, "+="),
            Op::SubAssign => write!(f, "-="),
            Op::MulAssign => write!(f, "*="),
            Op::DivAssign => write!(f, "/="),
            Op::ThinArrow => write!(f, "->"),
            Op::FatArrow => write!(f, "=>"),
        }
    }
}

pub fn tokenize(program: &str) -> Result<Vec<SpannedToken>, Vec<Rich<'_, char>>> {
    lexer().parse(program).into_result()
}

type Extra<'src> = extra::Full<Rich<'src, char>, (), ()>;
type LexErr<'src> = Rich<'src, char>;

fn lexer<'src>() -> impl Parser<'src, &'src str, Vec<SpannedToken>, Extra<'src>> {
    choice((
        line_comment().to(vec![]),
        string_literal(),
        token().map(|t| vec![t]),
    ))
    .padded()
    .repeated()
    .collect::<Vec<_>>()
    .map(|items| items.into_iter().flatten().collect())
    .then_ignore(end())
}

fn token<'src>() -> impl Parser<'src, &'src str, SpannedToken, Extra<'src>> {
    choice((delimiter(), literal(), ident(), op(), punctuation())).map_with(|tok, e| {
        let span = e.span();
        (
            tok,
            Span {
                start: span.start,
                end: span.end,
            },
        )
    })
}

fn scan_escape<'src, 'p>(
    input: &mut InputRef<'src, 'p, &'src str, Extra<'src>>,
    char_cursor: &Cursor<'src, 'p, &'src str>,
    text_buf: &mut String,
) -> Result<(), LexErr<'src>> {
    match input.next() {
        Some('n') => text_buf.push('\n'),
        Some('t') => text_buf.push('\t'),
        Some('r') => text_buf.push('\r'),
        Some('\\') => text_buf.push('\\'),
        Some('"') => text_buf.push('"'),
        Some('{') => text_buf.push('{'),
        _ => {
            return Err(Rich::custom(
                input.span_since(char_cursor),
                "Unexpected character",
            ));
        }
    }
    Ok(())
}

fn scan_interp_body<'src, 'p>(
    input: &mut InputRef<'src, 'p, &'src str, Extra<'src>>,
    str_open: &Cursor<'src, 'p, &'src str>,
) -> Result<String, LexErr<'src>> {
    let mut expr_str = String::new();
    let mut depth = 1usize;
    loop {
        match input.next() {
            None => {
                return Err(Rich::custom(
                    input.span_since(str_open),
                    "unterminated string interpolation",
                ));
            }
            Some('{') => {
                depth += 1;
                expr_str.push('{');
            }
            Some('}') => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                expr_str.push('}');
            }
            Some(c) => expr_str.push(c),
        }
    }
    Ok(expr_str)
}

fn tokenize_interp_expr<'src, 'p>(
    input: &mut InputRef<'src, 'p, &'src str, Extra<'src>>,
    str_open: &Cursor<'src, 'p, &'src str>,
    expr_str: &str,
    expr_start_offset: usize,
) -> Result<Vec<SpannedToken>, LexErr<'src>> {
    match tokenize(expr_str) {
        Ok(expr_tokens) => Ok(expr_tokens
            .into_iter()
            .map(|(tok, span)| {
                (
                    tok,
                    Span {
                        start: span.start + expr_start_offset,
                        end: span.end + expr_start_offset,
                    },
                )
            })
            .collect()),
        Err(_) => Err(Rich::custom(
            input.span_since(str_open),
            "invalid expression in string interpolation",
        )),
    }
}

fn string_literal<'src>() -> impl Parser<'src, &'src str, Vec<SpannedToken>, Extra<'src>> {
    custom(|input: &mut InputRef<'src, '_, &'src str, Extra<'src>>| {
        let str_open = input.cursor();
        match input.peek() {
            Some('"') => input.skip(),
            _ => return Err(Rich::custom(input.span_since(&str_open), "expected string literal")),
        }

        let mut tokens: Vec<SpannedToken> = vec![];
        let mut text_buf = String::new();
        let mut text_src_start = input.span_since(&str_open).end;
        let mut is_interpolated = false;

        loop {
            let char_cursor = input.cursor();
            match input.next() {
                None => return Err(Rich::custom(input.span_since(&str_open), "unterminated string literal")),
                Some('"') => break,
                Some('\\') => scan_escape(input, &char_cursor, &mut text_buf)?,
                Some('{') => {
                    is_interpolated = true;
                    let after_brace = input.span_since(&str_open);
                    let brace_pos = after_brace.end - 1;

                    if !text_buf.is_empty() {
                        tokens.push((
                            Token::Interp(InterpToken::Text(Intern::new(std::mem::take(&mut text_buf)))),
                            Span { start: text_src_start, end: brace_pos },
                        ));
                    }
                    tokens.push((Token::Interp(InterpToken::ExprStart), Span { start: brace_pos, end: brace_pos + 1 }));

                    let expr_start_offset = after_brace.end;
                    let expr_str = scan_interp_body(input, &str_open)?;

                    let after_close = input.span_since(&str_open);
                    let close_pos = after_close.end - 1;

                    let expr_tokens = tokenize_interp_expr(input, &str_open, &expr_str, expr_start_offset)?;
                    tokens.extend(expr_tokens);
                    tokens.push((Token::Interp(InterpToken::ExprEnd), Span { start: close_pos, end: close_pos + 1 }));

                    text_src_start = after_close.end;
                }
                Some(c) => text_buf.push(c),
            }
        }

        let full = input.span_since(&str_open);
        let full_span = Span { start: full.start, end: full.end };

        if is_interpolated {
            if !text_buf.is_empty() {
                let text_end = full_span.end - 1;
                tokens.push((
                    Token::Interp(InterpToken::Text(Intern::new(text_buf))),
                    Span { start: text_src_start, end: text_end },
                ));
            }
            let mut result = vec![(Token::Interp(InterpToken::Start), Span { start: full_span.start, end: full_span.start + 1 })];
            result.extend(tokens);
            result.push((Token::Interp(InterpToken::End), Span { start: full_span.end - 1, end: full_span.end }));
            Ok(result)
        } else {
            Ok(vec![(Token::Literal(LitToken::String(Intern::new(text_buf))), full_span)])
        }
    })
}

fn delimiter<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    choice((open_delimiter(), close_delimiter()))
}

fn open_delimiter<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    choice((
        just("(").to(Delimiter::Parent),
        just("{").to(Delimiter::Brace),
        just("[").to(Delimiter::Bracket),
    ))
    .map(Token::Open)
}

fn close_delimiter<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    choice((
        just(")").to(Delimiter::Parent),
        just("}").to(Delimiter::Brace),
        just("]").to(Delimiter::Bracket),
    ))
    .map(Token::Close)
}

fn literal<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    choice((lit_float(), lit_integer())).map(Token::Literal)
}

fn lit_integer<'src>() -> impl Parser<'src, &'src str, LitToken, Extra<'src>> {
    text::int(10)
        .map(|s: &str| s.parse().unwrap())
        .map(LitToken::Number)
}

fn lit_float<'src>() -> impl Parser<'src, &'src str, LitToken, Extra<'src>> {
    text::int(10)
        .then(just('.'))
        .then(text::digits(10))
        .to_slice()
        .map(|s: &str| Intern::new(s.to_string()))
        .map(LitToken::Float)
}

fn ident<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    text::ident().map(|s: &str| match s {
        "struct" => Token::Keyword(Keyword::Struct),
        "enum" => Token::Keyword(Keyword::Enum),
        "string" => Token::Keyword(Keyword::String),
        "pub" => Token::Keyword(Keyword::Pub),
        "let" => Token::Keyword(Keyword::Let),
        "var" => Token::Keyword(Keyword::Var),
        "if" => Token::Keyword(Keyword::If),
        "else" => Token::Keyword(Keyword::Else),
        "while" => Token::Keyword(Keyword::While),
        "for" => Token::Keyword(Keyword::For),
        "in" => Token::Keyword(Keyword::In),
        "break" => Token::Keyword(Keyword::Break),
        "continue" => Token::Keyword(Keyword::Continue),
        "match" => Token::Keyword(Keyword::Match),
        "fn" => Token::Keyword(Keyword::Fn),
        "return" => Token::Keyword(Keyword::Return),
        "int" => Token::Keyword(Keyword::Int),
        "float" => Token::Keyword(Keyword::Float),
        "bool" => Token::Keyword(Keyword::Bool),
        "void" => Token::Keyword(Keyword::Void),
        "any" => Token::Keyword(Keyword::Any),
        "nil" => Token::Keyword(Keyword::Nil),
        "true" => Token::Keyword(Keyword::True),
        "false" => Token::Keyword(Keyword::False),
        "as" => Token::Keyword(Keyword::As),
        "extern" => Token::Keyword(Keyword::Extern),
        "type" => Token::Keyword(Keyword::Type),
        "import" => Token::Keyword(Keyword::Import),
        _ => {
            let ident = ast::Ident(Intern::new(s.to_string()));
            Token::Ident(ident)
        }
    })
}

fn op<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    choice((
        // complex op
        just("??").to(Op::Coalesce),
        just("==").to(Op::Eq),
        just("!=").to(Op::NotEq),
        just("<=").to(Op::LessThanEq),
        just(">=").to(Op::GreaterThanEq),
        just("&&").to(Op::And),
        just("||").to(Op::Or),
        just("+=").to(Op::AddAssign),
        just("-=").to(Op::SubAssign),
        just("*=").to(Op::MulAssign),
        just("/=").to(Op::DivAssign),
        just("->").to(Op::ThinArrow),
        just("=>").to(Op::FatArrow),
        // simple op
        just("+").to(Op::Add),
        just("-").to(Op::Sub),
        just("*").to(Op::Mul),
        just("/").to(Op::Div),
        just("%").to(Op::Rem),
        just("<").to(Op::LessThan),
        just(">").to(Op::GreaterThan),
        just("!").to(Op::Not),
        just("=").to(Op::Assign),
    ))
    .map(Token::Op)
}

fn line_comment<'src>() -> impl Parser<'src, &'src str, (), Extra<'src>> {
    just("//").ignore_then(none_of("\n").repeated()).ignored()
}

fn punctuation<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    choice((
        just("..=").to(Token::RangeEq),
        just("..").to(Token::Range),
        just(":").to(Token::Colon),
        just(";").to(Token::Semicolon),
        just(",").to(Token::Comma),
        just("?").to(Token::Question),
        just(".").to(Token::Dot),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize_string(src: &str) -> Result<String, ()> {
        let tokens = tokenize(src).map_err(|_| ())?;
        match tokens.into_iter().next() {
            Some((Token::Literal(LitToken::String(s)), _)) => Ok(s.to_string()),
            _ => Err(()),
        }
    }

    fn tokenize_tokens(src: &str) -> Vec<Token> {
        tokenize(src)
            .unwrap_or_else(|_| panic!("tokenize failed for: {src}"))
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }

    fn ident_tok(name: &str) -> Token {
        Token::Ident(ast::Ident(Intern::new(name.to_string())))
    }

    fn str_text(s: &str) -> Token {
        Token::Interp(InterpToken::Text(Intern::new(s.to_string())))
    }

    #[test]
    fn test_string_literal_basic() {
        assert_eq!(tokenize_string(r#""hello""#).unwrap(), "hello");
    }

    #[test]
    fn test_string_escape_newline() {
        assert_eq!(tokenize_string(r#""hello\nworld""#).unwrap(), "hello\nworld");
    }

    #[test]
    fn test_string_escape_tab() {
        assert_eq!(tokenize_string(r#""col1\tcol2""#).unwrap(), "col1\tcol2");
    }

    #[test]
    fn test_string_escape_carriage_return() {
        assert_eq!(tokenize_string(r#""line\r""#).unwrap(), "line\r");
    }

    #[test]
    fn test_string_escape_backslash() {
        assert_eq!(tokenize_string(r#""path\\to""#).unwrap(), "path\\to");
    }

    #[test]
    fn test_string_escape_quote() {
        assert_eq!(tokenize_string(r#""say \"hi\"""#).unwrap(), r#"say "hi""#);
    }

    #[test]
    fn test_string_escape_brace() {
        assert_eq!(tokenize_string(r#""\{""#).unwrap(), "{");
    }

    #[test]
    fn test_string_multiple_escapes() {
        assert_eq!(tokenize_string(r#""a\nb\tc\\d""#).unwrap(), "a\nb\tc\\d");
    }

    #[test]
    fn test_string_empty() {
        assert_eq!(tokenize_string(r#""""#).unwrap(), "");
    }

    #[test]
    fn test_string_invalid_escape_err() {
        assert!(tokenize(r#""hello\z""#).is_err());
    }

    #[test]
    fn test_interp_string_only_expr() {
        let tokens = tokenize_tokens(r#""{x}""#);
        assert_eq!(tokens, vec![
            Token::Interp(InterpToken::Start),
            Token::Interp(InterpToken::ExprStart),
            ident_tok("x"),
            Token::Interp(InterpToken::ExprEnd),
            Token::Interp(InterpToken::End),
        ]);
    }

    #[test]
    fn test_interp_string_single_var() {
        let tokens = tokenize_tokens(r#""HP: {hp}""#);
        assert_eq!(tokens, vec![
            Token::Interp(InterpToken::Start),
            str_text("HP: "),
            Token::Interp(InterpToken::ExprStart),
            ident_tok("hp"),
            Token::Interp(InterpToken::ExprEnd),
            Token::Interp(InterpToken::End),
        ]);
    }

    #[test]
    fn test_interp_string_expression() {
        let tokens = tokenize_tokens(r#""a {x + y} b""#);
        assert_eq!(tokens, vec![
            Token::Interp(InterpToken::Start),
            str_text("a "),
            Token::Interp(InterpToken::ExprStart),
            ident_tok("x"),
            Token::Op(Op::Add),
            ident_tok("y"),
            Token::Interp(InterpToken::ExprEnd),
            str_text(" b"),
            Token::Interp(InterpToken::End),
        ]);
    }

    #[test]
    fn test_interp_string_multiple_parts() {
        let tokens = tokenize_tokens(r#""{a} and {b}""#);
        assert_eq!(tokens, vec![
            Token::Interp(InterpToken::Start),
            Token::Interp(InterpToken::ExprStart),
            ident_tok("a"),
            Token::Interp(InterpToken::ExprEnd),
            str_text(" and "),
            Token::Interp(InterpToken::ExprStart),
            ident_tok("b"),
            Token::Interp(InterpToken::ExprEnd),
            Token::Interp(InterpToken::End),
        ]);
    }

    #[test]
    fn test_interp_string_adjacent() {
        let tokens = tokenize_tokens(r#""{a}{b}""#);
        assert_eq!(tokens, vec![
            Token::Interp(InterpToken::Start),
            Token::Interp(InterpToken::ExprStart),
            ident_tok("a"),
            Token::Interp(InterpToken::ExprEnd),
            Token::Interp(InterpToken::ExprStart),
            ident_tok("b"),
            Token::Interp(InterpToken::ExprEnd),
            Token::Interp(InterpToken::End),
        ]);
    }

    #[test]
    fn test_interp_string_text_only_still_plain() {
        let tokens = tokenize_tokens(r#""just text""#);
        assert_eq!(tokens, vec![
            Token::Literal(LitToken::String(Intern::new("just text".to_string()))),
        ]);
    }

    #[test]
    fn test_interp_string_escaped_brace_no_interp() {
        // \{ is an escape that produces a literal `{` — no interpolation
        let tokens = tokenize_tokens(r#""\{not_interp}""#);
        assert_eq!(tokens, vec![
            Token::Literal(LitToken::String(Intern::new("{not_interp}".to_string()))),
        ]);
    }

    #[test]
    fn test_interp_string_unterminated_expr_err() {
        assert!(tokenize(r#""hello {oops""#).is_err());
    }
}
