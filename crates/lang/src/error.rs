use crate::lexer::{SpannedToken, Token};
use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::error::{Rich, RichPattern};

pub fn report_parse_errors(src: &str, tokens: &[SpannedToken], errors: Vec<Rich<SpannedToken>>) {
    for e in errors {
        let token_span = e.span();

        let start_byte = tokens
            .get(token_span.start)
            .map(|(_, span)| span.start)
            .unwrap_or(0);
        let end_byte = tokens
            .get(token_span.end)
            .map(|(_, span)| span.end.saturating_sub(1))
            .unwrap_or(start_byte);

        let byte_range = start_byte..end_byte;

        let last_context = last_ctx(&e)
            .map(|s| format!("while parsing a {}", s))
            .unwrap_or_default();
        let (msg_title, msg_body) = if let Some((found_token, _)) = e.found() {
            let token_desc = describe_token(found_token);
            (format!("Unexpected token {}", last_context), token_desc)
        } else {
            (
                format!("Unexpected end of input {}", last_context),
                "end of file".to_string(),
            )
        };

        let report = Report::build(ReportKind::Error, byte_range.clone())
            .with_message(msg_title)
            .with_label(
                Label::new(byte_range.clone())
                    .with_color(Color::Red)
                    .with_message(msg_body),
            );
        let _ = report.finish().print(Source::from(src));
    }
}

fn last_ctx(ctx: &Rich<'_, SpannedToken>) -> Option<String> {
    ctx.contexts()
        .filter_map(|(pat, span)| match pat {
            RichPattern::Label(s) => Some((s.as_ref(), span)),
            _ => None,
        })
        .last()
        .map(|(s, _)| s.to_string())
}

fn describe_token(token: &Token) -> String {
    match token {
        Token::Keyword(keyword) => format!("'{}' keyword", keyword),
        Token::Open(..) | Token::Close(..) => format!("'{}' delimiter", token),
        Token::Ident(_) => "identifier".to_string(),
        Token::Literal(lit) => format!("'{}' literal", lit),
        Token::Op(op) => format!("'{}' operator", op),
    }
}
