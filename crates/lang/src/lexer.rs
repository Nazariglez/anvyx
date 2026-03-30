use chumsky::input::Cursor;
use chumsky::{Parser, error::Rich, extra, input::InputRef, prelude::*};
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
    At,
    DocComment(Intern<String>),
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
            Token::At => write!(f, "@"),
            Token::Interp(interp) => write!(f, "{}", interp),
            Token::DocComment(s) => write!(f, "/// {}", s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FloatSuffix {
    F,
    D,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LitToken {
    Number(i64),
    Float(Intern<String>, Option<FloatSuffix>),
    String(Intern<String>),
}

impl Display for LitToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LitToken::Number(n) => write!(f, "{}", n),
            LitToken::Float(s, _) => write!(f, "{}", s),
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
    Double,
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
    Const,
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
    Extend,
    DataRef,
}

impl Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Keyword::Int => write!(f, "int"),
            Keyword::Float => write!(f, "float"),
            Keyword::Double => write!(f, "double"),
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
            Keyword::Const => write!(f, "const"),
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
            Keyword::Extend => write!(f, "extend"),
            Keyword::DataRef => write!(f, "dataref"),
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
    Pipe,
    Caret,
    CaretAssign,
    Tilde,
    BitAnd,
    BitAndAssign,
    BitOrAssign,
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
            Op::Pipe => write!(f, "|"),
            Op::Caret => write!(f, "^"),
            Op::CaretAssign => write!(f, "^="),
            Op::Tilde => write!(f, "~"),
            Op::BitAnd => write!(f, "&"),
            Op::BitAndAssign => write!(f, "&="),
            Op::BitOrAssign => write!(f, "|="),
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
        doc_comment(),
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
            _ => {
                return Err(Rich::custom(
                    input.span_since(&str_open),
                    "expected string literal",
                ));
            }
        }

        let mut tokens: Vec<SpannedToken> = vec![];
        let mut text_buf = String::new();
        let mut text_src_start = input.span_since(&str_open).end;
        let mut is_interpolated = false;

        loop {
            let char_cursor = input.cursor();
            match input.next() {
                None => {
                    return Err(Rich::custom(
                        input.span_since(&str_open),
                        "unterminated string literal",
                    ));
                }
                Some('"') => break,
                Some('\\') => scan_escape(input, &char_cursor, &mut text_buf)?,
                Some('{') => {
                    is_interpolated = true;
                    let after_brace = input.span_since(&str_open);
                    let brace_pos = after_brace.end - 1;

                    if !text_buf.is_empty() {
                        tokens.push((
                            Token::Interp(InterpToken::Text(Intern::new(std::mem::take(
                                &mut text_buf,
                            )))),
                            Span {
                                start: text_src_start,
                                end: brace_pos,
                            },
                        ));
                    }
                    tokens.push((
                        Token::Interp(InterpToken::ExprStart),
                        Span {
                            start: brace_pos,
                            end: brace_pos + 1,
                        },
                    ));

                    let expr_start_offset = after_brace.end;
                    let expr_str = scan_interp_body(input, &str_open)?;

                    let after_close = input.span_since(&str_open);
                    let close_pos = after_close.end - 1;

                    let expr_tokens =
                        tokenize_interp_expr(input, &str_open, &expr_str, expr_start_offset)?;
                    tokens.extend(expr_tokens);
                    tokens.push((
                        Token::Interp(InterpToken::ExprEnd),
                        Span {
                            start: close_pos,
                            end: close_pos + 1,
                        },
                    ));

                    text_src_start = after_close.end;
                }
                Some(c) => text_buf.push(c),
            }
        }

        let full = input.span_since(&str_open);
        let full_span = Span {
            start: full.start,
            end: full.end,
        };

        if is_interpolated {
            if !text_buf.is_empty() {
                let text_end = full_span.end - 1;
                tokens.push((
                    Token::Interp(InterpToken::Text(Intern::new(text_buf))),
                    Span {
                        start: text_src_start,
                        end: text_end,
                    },
                ));
            }
            let mut result = vec![(
                Token::Interp(InterpToken::Start),
                Span {
                    start: full_span.start,
                    end: full_span.start + 1,
                },
            )];
            result.extend(tokens);
            result.push((
                Token::Interp(InterpToken::End),
                Span {
                    start: full_span.end - 1,
                    end: full_span.end,
                },
            ));
            Ok(result)
        } else {
            Ok(vec![(
                Token::Literal(LitToken::String(Intern::new(text_buf))),
                full_span,
            )])
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

fn validate_numeric_underscores(s: &str) -> bool {
    !s.contains("__")
        && !s.contains("_.")
        && !s.contains("._")
        && !s.contains("_+")
        && !s.contains("+_")
        && !s.contains("_-")
        && !s.contains("-_")
        && !s.ends_with('_')
}

fn strip_underscores(s: &str) -> String {
    s.chars().filter(|c| *c != '_').collect()
}

fn try_consume_exponent<'src, 'p>(
    input: &mut InputRef<'src, 'p, &'src str, Extra<'src>>,
    buf: &mut String,
) {
    if !matches!(input.peek(), Some('e' | 'E')) {
        return;
    }

    let checkpoint = input.save();
    let e = input.next().unwrap();
    let mut exp_buf = String::new();
    exp_buf.push(e);

    if matches!(input.peek(), Some('+' | '-')) {
        exp_buf.push(input.next().unwrap());
    }

    let mut has_digit = false;
    loop {
        match input.peek() {
            Some(c) if c.is_ascii_digit() => {
                has_digit = true;
                exp_buf.push(input.next().unwrap());
            }
            Some('_') => {
                exp_buf.push(input.next().unwrap());
            }
            _ => break,
        }
    }

    if !has_digit {
        input.rewind(checkpoint);
        return;
    }

    buf.push_str(&exp_buf);
}

fn float_suffix<'src>() -> impl Parser<'src, &'src str, FloatSuffix, Extra<'src>> {
    one_of("fd")
        .then_ignore(
            any::<&'src str, Extra<'src>>()
                .filter(|c: &char| c.is_alphanumeric() || *c == '_')
                .rewind()
                .not(),
        )
        .map(|c: char| {
            if c == 'f' {
                FloatSuffix::F
            } else {
                FloatSuffix::D
            }
        })
}

fn literal<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    choice((lit_float(), lit_int_suffixed(), lit_integer())).map(Token::Literal)
}

fn is_digit_for_radix(c: char, radix: u32) -> bool {
    match radix {
        2 => c == '0' || c == '1',
        8 => matches!(c, '0'..='7'),
        16 => c.is_ascii_hexdigit(),
        _ => unreachable!(),
    }
}

fn parse_prefixed_digits<'src, 'p>(
    input: &mut InputRef<'src, 'p, &'src str, Extra<'src>>,
    start: &Cursor<'src, 'p, &'src str>,
    radix: u32,
) -> Result<LitToken, LexErr<'src>> {
    let base_name = match radix {
        2 => "binary",
        8 => "octal",
        16 => "hexadecimal",
        _ => unreachable!(),
    };

    let mut buf = String::new();

    loop {
        match input.peek() {
            Some(c) if is_digit_for_radix(c, radix) || c == '_' => {
                buf.push(c);
                input.skip();
            }
            _ => break,
        }
    }

    let has_digits = buf.chars().any(|c| c != '_');
    let next_is_alnum = matches!(input.peek(), Some(c) if c.is_ascii_alphanumeric());

    if !has_digits && next_is_alnum {
        return Err(Rich::custom(
            input.span_since(start),
            format!("invalid digit in {base_name} literal"),
        ));
    }

    if !has_digits {
        return Err(Rich::custom(
            input.span_since(start),
            format!("expected digits after {base_name} prefix"),
        ));
    }

    if next_is_alnum {
        return Err(Rich::custom(
            input.span_since(start),
            format!("invalid digit in {base_name} literal"),
        ));
    }

    if !validate_numeric_underscores(&buf) {
        return Err(Rich::custom(
            input.span_since(start),
            "invalid underscore placement in numeric literal",
        ));
    }

    let cleaned = strip_underscores(&buf);
    i64::from_str_radix(&cleaned, radix)
        .map(LitToken::Number)
        .map_err(|_| Rich::custom(input.span_since(start), "integer literal overflow"))
}

fn lit_integer<'src>() -> impl Parser<'src, &'src str, LitToken, Extra<'src>> {
    custom(|input: &mut InputRef<'src, '_, &'src str, Extra<'src>>| {
        let start = input.cursor();
        let mut buf = String::new();

        match input.next() {
            Some(c) if c.is_ascii_digit() => {
                if c == '0' {
                    let radix = match input.peek() {
                        Some('x' | 'X') => Some(16u32),
                        Some('b' | 'B') => Some(2),
                        Some('o' | 'O') => Some(8),
                        _ => None,
                    };

                    if let Some(radix) = radix {
                        input.skip(); // consume the prefix letter
                        return parse_prefixed_digits(input, &start, radix);
                    }
                }
                buf.push(c);
            }
            _ => {
                return Err(Rich::custom(
                    input.span_since(&start),
                    "expected integer literal",
                ));
            }
        }

        loop {
            match input.peek() {
                Some(c) if c.is_ascii_digit() || c == '_' => {
                    buf.push(c);
                    input.skip();
                }
                _ => break,
            }
        }

        if !validate_numeric_underscores(&buf) {
            return Err(Rich::custom(
                input.span_since(&start),
                "invalid underscore placement in numeric literal",
            ));
        }

        let cleaned = strip_underscores(&buf);
        cleaned
            .parse::<i64>()
            .map(LitToken::Number)
            .map_err(|_| Rich::custom(input.span_since(&start), "integer literal overflow"))
    })
}

fn lit_float<'src>() -> impl Parser<'src, &'src str, LitToken, Extra<'src>> {
    custom(|input: &mut InputRef<'src, '_, &'src str, Extra<'src>>| {
        let start = input.cursor();
        let mut buf = String::new();

        match input.next() {
            Some(c) if c.is_ascii_digit() => buf.push(c),
            _ => {
                return Err(Rich::custom(
                    input.span_since(&start),
                    "expected float literal",
                ));
            }
        }
        loop {
            match input.peek() {
                Some(c) if c.is_ascii_digit() || c == '_' => {
                    buf.push(c);
                    input.skip();
                }
                _ => break,
            }
        }

        match input.next() {
            Some('.') => buf.push('.'),
            _ => {
                return Err(Rich::custom(
                    input.span_since(&start),
                    "expected '.' in float literal",
                ));
            }
        }

        match input.next() {
            Some(c) if c.is_ascii_digit() || c == '_' => buf.push(c),
            _ => {
                return Err(Rich::custom(
                    input.span_since(&start),
                    "expected digits after '.' in float literal",
                ));
            }
        }
        loop {
            match input.peek() {
                Some(c) if c.is_ascii_digit() || c == '_' => {
                    buf.push(c);
                    input.skip();
                }
                _ => break,
            }
        }

        try_consume_exponent(input, &mut buf);

        if !validate_numeric_underscores(&buf) {
            return Err(Rich::custom(
                input.span_since(&start),
                "invalid underscore placement in numeric literal",
            ));
        }

        Ok(strip_underscores(&buf))
    })
    .then(float_suffix().or_not())
    .map(|(s, suffix)| LitToken::Float(Intern::new(s), suffix))
}

fn lit_int_suffixed<'src>() -> impl Parser<'src, &'src str, LitToken, Extra<'src>> {
    custom(|input: &mut InputRef<'src, '_, &'src str, Extra<'src>>| {
        let start = input.cursor();
        let mut buf = String::new();

        match input.next() {
            Some(c) if c.is_ascii_digit() => buf.push(c),
            _ => return Err(Rich::custom(input.span_since(&start), "expected integer")),
        }
        loop {
            match input.peek() {
                Some(c) if c.is_ascii_digit() || c == '_' => {
                    buf.push(c);
                    input.skip();
                }
                _ => break,
            }
        }

        try_consume_exponent(input, &mut buf);

        if !validate_numeric_underscores(&buf) {
            return Err(Rich::custom(
                input.span_since(&start),
                "invalid underscore placement in numeric literal",
            ));
        }

        Ok(strip_underscores(&buf))
    })
    .then(float_suffix())
    .map(|(s, suffix)| LitToken::Float(Intern::new(s), Some(suffix)))
}

fn ident<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    text::ident().map(|s: &str| match s {
        "struct" => Token::Keyword(Keyword::Struct),
        "enum" => Token::Keyword(Keyword::Enum),
        "string" => Token::Keyword(Keyword::String),
        "pub" => Token::Keyword(Keyword::Pub),
        "let" => Token::Keyword(Keyword::Let),
        "var" => Token::Keyword(Keyword::Var),
        "const" => Token::Keyword(Keyword::Const),
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
        "double" => Token::Keyword(Keyword::Double),
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
        "extend" => Token::Keyword(Keyword::Extend),
        "dataref" => Token::Keyword(Keyword::DataRef),
        _ => {
            let ident = ast::Ident(Intern::new(s.to_string()));
            Token::Ident(ident)
        }
    })
}

fn op<'src>() -> impl Parser<'src, &'src str, Token, Extra<'src>> {
    let complex = choice((
        just("??").to(Op::Coalesce),
        just("==").to(Op::Eq),
        just("!=").to(Op::NotEq),
        just("<=").to(Op::LessThanEq),
        just(">=").to(Op::GreaterThanEq),
        just("&=").to(Op::BitAndAssign),
        just("|=").to(Op::BitOrAssign),
        just("&&").to(Op::And),
        just("||").to(Op::Or),
        just("+=").to(Op::AddAssign),
        just("-=").to(Op::SubAssign),
        just("*=").to(Op::MulAssign),
        just("/=").to(Op::DivAssign),
        just("->").to(Op::ThinArrow),
        just("=>").to(Op::FatArrow),
        just("^=").to(Op::CaretAssign),
    ));
    let simple = choice((
        just("+").to(Op::Add),
        just("-").to(Op::Sub),
        just("*").to(Op::Mul),
        just("/").to(Op::Div),
        just("%").to(Op::Rem),
        just("<").to(Op::LessThan),
        just(">").to(Op::GreaterThan),
        just("!").to(Op::Not),
        just("=").to(Op::Assign),
        just("|").to(Op::Pipe),
        just("^").to(Op::Caret),
        just("~").to(Op::Tilde),
        just("&").to(Op::BitAnd),
    ));
    complex.or(simple).map(Token::Op)
}

fn doc_comment<'src>() -> impl Parser<'src, &'src str, Vec<SpannedToken>, Extra<'src>> {
    just("///")
        .then_ignore(just("/").rewind().not())
        .ignore_then(none_of("\n").repeated().collect::<String>())
        .map_with(|content, e| -> Vec<SpannedToken> {
            let span: chumsky::span::SimpleSpan<usize> = e.span();
            let stripped = content.strip_prefix(' ').unwrap_or(&content).to_string();
            vec![(
                Token::DocComment(Intern::new(stripped)),
                Span {
                    start: span.start,
                    end: span.end,
                },
            )]
        })
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
        just("@").to(Token::At),
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
        assert_eq!(
            tokenize_string(r#""hello\nworld""#).unwrap(),
            "hello\nworld"
        );
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
        assert_eq!(
            tokens,
            vec![
                Token::Interp(InterpToken::Start),
                Token::Interp(InterpToken::ExprStart),
                ident_tok("x"),
                Token::Interp(InterpToken::ExprEnd),
                Token::Interp(InterpToken::End),
            ]
        );
    }

    #[test]
    fn test_interp_string_single_var() {
        let tokens = tokenize_tokens(r#""HP: {hp}""#);
        assert_eq!(
            tokens,
            vec![
                Token::Interp(InterpToken::Start),
                str_text("HP: "),
                Token::Interp(InterpToken::ExprStart),
                ident_tok("hp"),
                Token::Interp(InterpToken::ExprEnd),
                Token::Interp(InterpToken::End),
            ]
        );
    }

    #[test]
    fn test_interp_string_expression() {
        let tokens = tokenize_tokens(r#""a {x + y} b""#);
        assert_eq!(
            tokens,
            vec![
                Token::Interp(InterpToken::Start),
                str_text("a "),
                Token::Interp(InterpToken::ExprStart),
                ident_tok("x"),
                Token::Op(Op::Add),
                ident_tok("y"),
                Token::Interp(InterpToken::ExprEnd),
                str_text(" b"),
                Token::Interp(InterpToken::End),
            ]
        );
    }

    #[test]
    fn test_interp_string_multiple_parts() {
        let tokens = tokenize_tokens(r#""{a} and {b}""#);
        assert_eq!(
            tokens,
            vec![
                Token::Interp(InterpToken::Start),
                Token::Interp(InterpToken::ExprStart),
                ident_tok("a"),
                Token::Interp(InterpToken::ExprEnd),
                str_text(" and "),
                Token::Interp(InterpToken::ExprStart),
                ident_tok("b"),
                Token::Interp(InterpToken::ExprEnd),
                Token::Interp(InterpToken::End),
            ]
        );
    }

    #[test]
    fn test_interp_string_adjacent() {
        let tokens = tokenize_tokens(r#""{a}{b}""#);
        assert_eq!(
            tokens,
            vec![
                Token::Interp(InterpToken::Start),
                Token::Interp(InterpToken::ExprStart),
                ident_tok("a"),
                Token::Interp(InterpToken::ExprEnd),
                Token::Interp(InterpToken::ExprStart),
                ident_tok("b"),
                Token::Interp(InterpToken::ExprEnd),
                Token::Interp(InterpToken::End),
            ]
        );
    }

    #[test]
    fn test_interp_string_text_only_still_plain() {
        let tokens = tokenize_tokens(r#""just text""#);
        assert_eq!(
            tokens,
            vec![Token::Literal(LitToken::String(Intern::new(
                "just text".to_string()
            ))),]
        );
    }

    #[test]
    fn test_interp_string_escaped_brace_no_interp() {
        // \{ is an escape that produces a literal `{` — no interpolation
        let tokens = tokenize_tokens(r#""\{not_interp}""#);
        assert_eq!(
            tokens,
            vec![Token::Literal(LitToken::String(Intern::new(
                "{not_interp}".to_string()
            ))),]
        );
    }

    #[test]
    fn test_interp_string_unterminated_expr_err() {
        assert!(tokenize(r#""hello {oops""#).is_err());
    }

    fn tokenize_lit(src: &str) -> Result<LitToken, ()> {
        let tokens = tokenize(src).map_err(|_| ())?;
        match tokens.into_iter().next() {
            Some((Token::Literal(lit), _)) => Ok(lit),
            _ => Err(()),
        }
    }

    fn all_tokens(src: &str) -> Result<Vec<Token>, ()> {
        tokenize(src)
            .map_err(|_| ())
            .map(|ts| ts.into_iter().map(|(t, _)| t).collect())
    }

    #[test]
    fn test_integer_with_underscores() {
        assert_eq!(tokenize_lit("1_000").unwrap(), LitToken::Number(1000));
    }

    #[test]
    fn test_integer_with_many_underscores() {
        assert_eq!(
            tokenize_lit("1_000_000").unwrap(),
            LitToken::Number(1_000_000)
        );
    }

    #[test]
    fn test_float_basic() {
        assert_eq!(
            tokenize_lit("3.14").unwrap(),
            LitToken::Float(Intern::new("3.14".to_string()), None)
        );
    }

    #[test]
    fn test_float_with_underscores_frac() {
        assert_eq!(
            tokenize_lit("3.141_592").unwrap(),
            LitToken::Float(Intern::new("3.141592".to_string()), None)
        );
    }

    #[test]
    fn test_float_with_underscores_int() {
        assert_eq!(
            tokenize_lit("1_000.5").unwrap(),
            LitToken::Float(Intern::new("1000.5".to_string()), None)
        );
    }

    #[test]
    fn test_float_suffix_f() {
        assert_eq!(
            tokenize_lit("1.5f").unwrap(),
            LitToken::Float(Intern::new("1.5".to_string()), Some(FloatSuffix::F))
        );
    }

    #[test]
    fn test_float_suffix_d() {
        assert_eq!(
            tokenize_lit("1.5d").unwrap(),
            LitToken::Float(Intern::new("1.5".to_string()), Some(FloatSuffix::D))
        );
    }

    #[test]
    fn test_int_suffixed_f() {
        assert_eq!(
            tokenize_lit("42f").unwrap(),
            LitToken::Float(Intern::new("42".to_string()), Some(FloatSuffix::F))
        );
    }

    #[test]
    fn test_int_suffixed_d() {
        assert_eq!(
            tokenize_lit("42d").unwrap(),
            LitToken::Float(Intern::new("42".to_string()), Some(FloatSuffix::D))
        );
    }

    #[test]
    fn test_int_suffixed_with_underscores() {
        assert_eq!(
            tokenize_lit("1_000f").unwrap(),
            LitToken::Float(Intern::new("1000".to_string()), Some(FloatSuffix::F))
        );
    }

    #[test]
    fn test_float_underscores_and_suffix() {
        assert_eq!(
            tokenize_lit("1_000.5d").unwrap(),
            LitToken::Float(Intern::new("1000.5".to_string()), Some(FloatSuffix::D))
        );
    }

    #[test]
    fn test_consecutive_underscores_err() {
        assert!(tokenize("1__000").is_err());
    }

    #[test]
    fn test_trailing_underscore_err() {
        assert!(tokenize("1_").is_err());
    }

    #[test]
    fn test_underscore_before_dot_err() {
        assert!(tokenize("1_.5").is_err());
    }

    #[test]
    fn test_underscore_after_dot_no_float() {
        // 1._5 lexes as int `1` + dot + ident `_5` — not a float literal
        let tokens = all_tokens("1._5").unwrap();
        assert_eq!(tokens[0], Token::Literal(LitToken::Number(1)));
        assert_eq!(tokens[1], Token::Dot);
    }

    #[test]
    fn test_underscore_before_suffix_float_err() {
        assert!(tokenize("1.5_f").is_err());
    }

    #[test]
    fn test_underscore_before_suffix_int_err() {
        assert!(tokenize("42_d").is_err());
    }

    #[test]
    fn test_uppercase_suffix_f_no_suffix() {
        // 1.5F lexes as float `1.5` + ident `F` — no suffix captured
        let tokens = all_tokens("1.5F").unwrap();
        assert_eq!(
            tokens[0],
            Token::Literal(LitToken::Float(Intern::new("1.5".to_string()), None))
        );
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_uppercase_suffix_d_no_suffix() {
        // 1.5D lexes as float `1.5` + ident `D` — no suffix captured
        let tokens = all_tokens("1.5D").unwrap();
        assert_eq!(
            tokens[0],
            Token::Literal(LitToken::Float(Intern::new("1.5".to_string()), None))
        );
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_suffix_not_captured_when_followed_by_ident() {
        // 42floor: integer 42 + ident floor, suffix NOT captured
        let tokens = all_tokens("42floor").unwrap();
        assert_eq!(tokens[0], Token::Literal(LitToken::Number(42)));
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_float_suffix_not_captured_when_followed_by_ident() {
        // 1.5floor: float 1.5 (no suffix) + ident floor
        let tokens = all_tokens("1.5floor").unwrap();
        assert_eq!(
            tokens[0],
            Token::Literal(LitToken::Float(Intern::new("1.5".to_string()), None))
        );
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_float_scientific() {
        assert_eq!(
            tokenize_lit("1.5e3").unwrap(),
            LitToken::Float(Intern::new("1.5e3".to_string()), None)
        );
    }

    #[test]
    fn test_float_scientific_negative_exp() {
        assert_eq!(
            tokenize_lit("1.5e-3").unwrap(),
            LitToken::Float(Intern::new("1.5e-3".to_string()), None)
        );
    }

    #[test]
    fn test_float_scientific_positive_exp() {
        assert_eq!(
            tokenize_lit("1.5e+3").unwrap(),
            LitToken::Float(Intern::new("1.5e+3".to_string()), None)
        );
    }

    #[test]
    fn test_float_scientific_uppercase() {
        assert_eq!(
            tokenize_lit("1.5E3").unwrap(),
            LitToken::Float(Intern::new("1.5E3".to_string()), None)
        );
    }

    #[test]
    fn test_float_scientific_suffix_f() {
        assert_eq!(
            tokenize_lit("1.5e3f").unwrap(),
            LitToken::Float(Intern::new("1.5e3".to_string()), Some(FloatSuffix::F))
        );
    }

    #[test]
    fn test_float_scientific_negative_suffix_d() {
        assert_eq!(
            tokenize_lit("1.5e-3d").unwrap(),
            LitToken::Float(Intern::new("1.5e-3".to_string()), Some(FloatSuffix::D))
        );
    }

    #[test]
    fn test_float_scientific_underscore_in_exp() {
        assert_eq!(
            tokenize_lit("1.5e1_0").unwrap(),
            LitToken::Float(Intern::new("1.5e10".to_string()), None)
        );
    }

    #[test]
    fn test_float_scientific_underscore_before_e() {
        assert_eq!(
            tokenize_lit("1_000.5e3").unwrap(),
            LitToken::Float(Intern::new("1000.5e3".to_string()), None)
        );
    }

    #[test]
    fn test_int_suffixed_scientific_f() {
        assert_eq!(
            tokenize_lit("2e3f").unwrap(),
            LitToken::Float(Intern::new("2e3".to_string()), Some(FloatSuffix::F))
        );
    }

    #[test]
    fn test_int_suffixed_scientific_d() {
        assert_eq!(
            tokenize_lit("2e3d").unwrap(),
            LitToken::Float(Intern::new("2e3".to_string()), Some(FloatSuffix::D))
        );
    }

    #[test]
    fn test_int_suffixed_scientific_negative_exp() {
        assert_eq!(
            tokenize_lit("5e-2f").unwrap(),
            LitToken::Float(Intern::new("5e-2".to_string()), Some(FloatSuffix::F))
        );
    }

    #[test]
    fn test_float_e_not_consumed_no_digit() {
        // 1.5e should lex as float 1.5 + ident e
        let tokens = all_tokens("1.5e").unwrap();
        assert_eq!(
            tokens[0],
            Token::Literal(LitToken::Float(Intern::new("1.5".to_string()), None))
        );
        assert_eq!(tokens[1], ident_tok("e"));
    }

    #[test]
    fn test_float_e_not_consumed_sign_no_digit() {
        // 1.5e- should lex as float 1.5 + ident e + op -
        let tokens = all_tokens("1.5e-").unwrap();
        assert_eq!(
            tokens[0],
            Token::Literal(LitToken::Float(Intern::new("1.5".to_string()), None))
        );
        assert_eq!(tokens[1], ident_tok("e"));
        assert_eq!(tokens[2], Token::Op(Op::Sub));
    }

    #[test]
    fn test_float_e_followed_by_ident() {
        // 1.5efoo should lex as float 1.5 + ident efoo
        let tokens = all_tokens("1.5efoo").unwrap();
        assert_eq!(
            tokens[0],
            Token::Literal(LitToken::Float(Intern::new("1.5".to_string()), None))
        );
        assert_eq!(tokens[1], ident_tok("efoo"));
    }

    // --- Hex integer literals ---

    #[test]
    fn test_hex_basic() {
        assert_eq!(tokenize_lit("0xFF").unwrap(), LitToken::Number(255));
    }

    #[test]
    fn test_hex_uppercase_prefix() {
        assert_eq!(tokenize_lit("0XFF").unwrap(), LitToken::Number(255));
    }

    #[test]
    fn test_hex_with_underscores() {
        assert_eq!(tokenize_lit("0xFF_FF").unwrap(), LitToken::Number(65535));
    }

    #[test]
    fn test_hex_lowercase_digits() {
        assert_eq!(tokenize_lit("0xab").unwrap(), LitToken::Number(171));
    }

    #[test]
    fn test_hex_mixed_case_digits() {
        assert_eq!(tokenize_lit("0xaBcD").unwrap(), LitToken::Number(43981));
    }

    #[test]
    fn test_hex_zero() {
        assert_eq!(tokenize_lit("0x0").unwrap(), LitToken::Number(0));
    }

    #[test]
    fn test_hex_leading_underscore() {
        assert_eq!(tokenize_lit("0x_FF").unwrap(), LitToken::Number(255));
    }

    // --- Binary integer literals ---

    #[test]
    fn test_binary_basic() {
        assert_eq!(tokenize_lit("0b1010").unwrap(), LitToken::Number(10));
    }

    #[test]
    fn test_binary_uppercase_prefix() {
        assert_eq!(tokenize_lit("0B1010").unwrap(), LitToken::Number(10));
    }

    #[test]
    fn test_binary_with_underscores() {
        assert_eq!(tokenize_lit("0b1010_1010").unwrap(), LitToken::Number(170));
    }

    #[test]
    fn test_binary_zero() {
        assert_eq!(tokenize_lit("0b0").unwrap(), LitToken::Number(0));
    }

    // --- Octal integer literals ---

    #[test]
    fn test_octal_basic() {
        assert_eq!(tokenize_lit("0o77").unwrap(), LitToken::Number(63));
    }

    #[test]
    fn test_octal_uppercase_prefix() {
        assert_eq!(tokenize_lit("0O77").unwrap(), LitToken::Number(63));
    }

    #[test]
    fn test_octal_with_underscores() {
        assert_eq!(tokenize_lit("0o7_7_7").unwrap(), LitToken::Number(511));
    }

    #[test]
    fn test_octal_zero() {
        assert_eq!(tokenize_lit("0o0").unwrap(), LitToken::Number(0));
    }

    // --- Prefixed literal error cases ---

    #[test]
    fn test_hex_invalid_digit_err() {
        assert!(tokenize("0xZZ").is_err());
    }

    #[test]
    fn test_binary_invalid_digit_err() {
        assert!(tokenize("0b12").is_err());
    }

    #[test]
    fn test_octal_invalid_digit_err() {
        assert!(tokenize("0o89").is_err());
    }

    #[test]
    fn test_hex_empty_body_err() {
        assert!(tokenize("0x ").is_err());
    }

    #[test]
    fn test_binary_empty_body_err() {
        assert!(tokenize("0b ").is_err());
    }

    #[test]
    fn test_octal_empty_body_err() {
        assert!(tokenize("0o ").is_err());
    }

    #[test]
    fn test_hex_consecutive_underscores_err() {
        assert!(tokenize("0xFF__FF").is_err());
    }

    #[test]
    fn test_hex_trailing_underscore_err() {
        assert!(tokenize("0xFF_").is_err());
    }
}
