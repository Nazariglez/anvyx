use crate::{
    ast,
    lexer::{Delimiter, Keyword, Token},
    span::{Span, Spanned},
};
use chumsky::{error::Rich, prelude::*};

use super::common::{TupleShapeResult, identifier, literal, validate_tuple_shape_raw};
use super::{AnvParser, BoxedParser};

pub(super) fn pattern<'src>() -> BoxedParser<'src, ast::PatternNode> {
    recursive(|pat| {
        let ident_or_wildcard = identifier().map_with(|ident, e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            if ident.0.as_ref() == "_" {
                Spanned::new(ast::Pattern::Wildcard, span)
            } else {
                Spanned::new(ast::Pattern::Ident(ident), span)
            }
        });

        let rest_pat = select! {
            (Token::Range, s) => Spanned::new(ast::Pattern::Rest, s)
        };

        let var_pat = select! {
            (Token::Keyword(Keyword::Var), _) => ()
        }
        .ignore_then(identifier())
        .map_with(|name, e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            Spanned::new(ast::Pattern::VarIdent(name), span)
        });

        let lit_pat = literal().map_with(|lit, e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            Spanned::new(ast::Pattern::Lit(lit), span)
        });

        let tuple_pat = tuple_pattern(pat.clone());
        let enum_pat = enum_pattern(pat.clone());
        let struct_pat = struct_pattern(pat);

        choice((
            rest_pat,
            var_pat,
            lit_pat,
            enum_pat,
            struct_pat,
            tuple_pat,
            ident_or_wildcard,
        ))
    })
    .labelled("pattern")
    .as_context()
    .boxed()
}

fn struct_pattern<'src>(
    pat: impl AnvParser<'src, ast::PatternNode>,
) -> BoxedParser<'src, ast::PatternNode> {
    let comma = select! { (Token::Comma, _) => () };
    let colon = select! { (Token::Colon, _) => () };
    let open_brace = select! { (Token::Open(Delimiter::Brace), _) => () };
    let close_brace = select! { (Token::Close(Delimiter::Brace), _) => () };

    let field_with_pattern = identifier()
        .then_ignore(colon)
        .then(pat.clone())
        .map(|(name, p)| (name, p));

    let field_shorthand = identifier().map_with(|name, e| {
        let s = e.span();
        let span = Span::new(s.start, s.end);
        (name, Spanned::new(ast::Pattern::Ident(name), span))
    });

    let field = choice((field_with_pattern, field_shorthand));

    identifier()
        .then(
            open_brace
                .ignore_then(
                    field
                        .separated_by(comma)
                        .allow_trailing()
                        .collect::<Vec<_>>(),
                )
                .then_ignore(close_brace),
        )
        .map_with(|(name, fields), e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            Spanned::new(ast::Pattern::Struct { name, fields }, span)
        })
        .labelled("struct pattern")
        .as_context()
        .boxed()
}

#[derive(Clone)]
enum EnumPatternKind {
    Unit,
    Tuple(Vec<ast::PatternNode>),
    Struct(Vec<(ast::Ident, ast::PatternNode)>, bool),
}

fn enum_pattern<'src>(
    pat: impl AnvParser<'src, ast::PatternNode>,
) -> BoxedParser<'src, ast::PatternNode> {
    let dot = select! { (Token::Dot, _) => () };

    let qualified_name = identifier().then_ignore(dot).then(identifier());

    qualified_name
        .then(choice((
            enum_tuple_payload(pat.clone()),
            enum_struct_payload(pat),
            empty().to(EnumPatternKind::Unit),
        )))
        .map_with(|((qualifier, variant), kind), e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            let pattern = match kind {
                EnumPatternKind::Unit => ast::Pattern::EnumUnit { qualifier, variant },
                EnumPatternKind::Tuple(fields) => ast::Pattern::EnumTuple {
                    qualifier,
                    variant,
                    fields,
                },
                EnumPatternKind::Struct(fields, has_rest) => ast::Pattern::EnumStruct {
                    qualifier,
                    variant,
                    fields,
                    has_rest,
                },
            };
            Spanned::new(pattern, span)
        })
        .labelled("enum pattern")
        .as_context()
        .boxed()
}

fn enum_tuple_payload<'src>(
    pat: impl AnvParser<'src, ast::PatternNode>,
) -> BoxedParser<'src, EnumPatternKind> {
    let comma = select! { (Token::Comma, _) => () };
    let open_paren = select! { (Token::Open(Delimiter::Parent), _) => () };
    let close_paren = select! { (Token::Close(Delimiter::Parent), _) => () };

    open_paren
        .ignore_then(pat.separated_by(comma).allow_trailing().collect::<Vec<_>>())
        .then_ignore(close_paren)
        .map(EnumPatternKind::Tuple)
        .boxed()
}

fn enum_struct_payload<'src>(
    pat: impl AnvParser<'src, ast::PatternNode>,
) -> BoxedParser<'src, EnumPatternKind> {
    let comma = select! { (Token::Comma, _) => () };
    let colon = select! { (Token::Colon, _) => () };
    let open_brace = select! { (Token::Open(Delimiter::Brace), _) => () };
    let close_brace = select! { (Token::Close(Delimiter::Brace), _) => () };
    let rest = select! { (Token::Range, _) => () };

    let field_with_pattern = identifier()
        .then_ignore(colon)
        .then(pat.clone())
        .map(|(name, p)| (name, p));

    let field_shorthand = identifier().map_with(|name, e| {
        let s = e.span();
        let span = Span::new(s.start, s.end);
        (name, Spanned::new(ast::Pattern::Ident(name), span))
    });

    let field = choice((field_with_pattern, field_shorthand));

    open_brace
        .ignore_then(
            field
                .separated_by(comma)
                .allow_trailing()
                .collect::<Vec<_>>(),
        )
        .then(rest.or_not())
        .then_ignore(close_brace)
        .map(|(fields, rest_tok)| EnumPatternKind::Struct(fields, rest_tok.is_some()))
        .boxed()
}

enum TuplePatternElem {
    Pos(ast::PatternNode),
    Labeled(ast::Ident, ast::PatternNode),
}

fn tuple_pattern<'src>(
    pat: impl AnvParser<'src, ast::PatternNode>,
) -> BoxedParser<'src, ast::PatternNode> {
    let comma = select! { (Token::Comma, _) => () };
    let open_paren = select! { (Token::Open(Delimiter::Parent), _) => () };
    let close_paren = select! { (Token::Close(Delimiter::Parent), _) => () };
    let colon = select! { (Token::Colon, _) => () };

    let labeled_elem = identifier()
        .then_ignore(colon)
        .then(pat.clone())
        .map(|(name, p)| TuplePatternElem::Labeled(name, p));

    let pos_elem = pat.map(TuplePatternElem::Pos);
    let elem = choice((labeled_elem, pos_elem));

    let first_elem = elem.clone();
    let rest_elems = comma.ignore_then(elem).repeated().collect::<Vec<_>>();

    open_paren
        .ignore_then(first_elem.or_not())
        .then(rest_elems)
        .then(comma.or_not())
        .then_ignore(close_paren)
        .validate(|((first, rest), trailing_comma), e, emitter| {
            let s = e.span();
            let span = Span::new(s.start, s.end);

            match validate_tuple_shape_raw(first, rest, trailing_comma.is_some()) {
                TupleShapeResult::Empty => {
                    emitter.emit(Rich::custom(s, "empty tuple patterns are not supported"));
                    Spanned::new(ast::Pattern::Wildcard, span)
                }
                TupleShapeResult::OneTupleError(elem) => {
                    emitter.emit(Rich::custom(s, "1-tuple patterns are not supported"));
                    match elem {
                        TuplePatternElem::Pos(p) => p,
                        TuplePatternElem::Labeled(_, p) => p,
                    }
                }
                TupleShapeResult::Grouped(elem) => match elem {
                    TuplePatternElem::Pos(p) => p,
                    TuplePatternElem::Labeled(_, p) => p,
                },
                TupleShapeResult::Tuple(elems) => {
                    let all_pos = elems.iter().all(|e| matches!(e, TuplePatternElem::Pos(_)));
                    let all_labeled = elems
                        .iter()
                        .all(|e| matches!(e, TuplePatternElem::Labeled(_, _)));

                    if all_pos {
                        let pats: Vec<ast::PatternNode> = elems
                            .into_iter()
                            .map(|e| match e {
                                TuplePatternElem::Pos(p) => p,
                                TuplePatternElem::Labeled(_, p) => p,
                            })
                            .collect();
                        return Spanned::new(ast::Pattern::Tuple(pats), span);
                    }

                    if all_labeled {
                        let fields: Vec<(ast::Ident, ast::PatternNode)> = elems
                            .into_iter()
                            .map(|e| match e {
                                TuplePatternElem::Labeled(name, p) => (name, p),
                                TuplePatternElem::Pos(_) => unreachable!(),
                            })
                            .collect();
                        return Spanned::new(ast::Pattern::NamedTuple(fields), span);
                    }

                    emitter.emit(Rich::custom(
                        s,
                        "cannot mix labeled and unlabeled elements in tuple pattern",
                    ));
                    Spanned::new(ast::Pattern::Wildcard, span)
                }
                TupleShapeResult::UnexpectedComma => {
                    emitter.emit(Rich::custom(s, "unexpected comma in pattern"));
                    Spanned::new(ast::Pattern::Wildcard, span)
                }
            }
        })
        .labelled("tuple pattern")
        .as_context()
        .boxed()
}
