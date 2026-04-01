use crate::{
    ast::{self, Type},
    lexer::{Delimiter, Keyword, LitToken, Op, Token},
};
use chumsky::{error::Rich, prelude::*};

use super::common::{TupleShapeResult, identifier, validate_tuple_shape_raw};
use super::{AnvParser, BoxedParser};

#[derive(Clone)]
enum TypeSuffix {
    Optional,
}

pub(super) fn type_ident<'src>() -> BoxedParser<'src, ast::Type> {
    type_ident_inner(false)
}

pub(super) fn param_type_ident<'src>() -> BoxedParser<'src, ast::Type> {
    type_ident_inner(true)
}

fn type_ident_inner<'src>(allow_view: bool) -> BoxedParser<'src, ast::Type> {
    recursive(move |type_parser| {
        let builtin_typ = select! {
            (Token::Keyword(Keyword::Int), _) => ast::Type::Int,
            (Token::Keyword(Keyword::Float), _) => ast::Type::Float,
            (Token::Keyword(Keyword::Double), _) => ast::Type::Double,
            (Token::Keyword(Keyword::Bool), _) => ast::Type::Bool,
            (Token::Keyword(Keyword::String), _) => ast::Type::String,
            (Token::Keyword(Keyword::Void), _) => ast::Type::Void,
            (Token::Keyword(Keyword::Any), _) => ast::Type::Any,
        };

        let type_args = select! { (Token::Op(Op::LessThan), _) => () }
            .ignore_then(
                type_parser
                    .clone()
                    .separated_by(select! { (Token::Comma, _) => () })
                    .allow_trailing()
                    .collect::<Vec<_>>(),
            )
            .then_ignore(select! { (Token::Op(Op::GreaterThan), _) => () });

        let type_name_ref = identifier()
            .then(type_args.or_not())
            .map(|(name, args)| match args {
                Some(type_args) => ast::Type::Struct { name, type_args },
                None => ast::Type::UnresolvedName(name),
            });

        let paren_type = paren_or_tuple_type(type_parser.clone());

        let open_paren = select! { (Token::Open(Delimiter::Parent), _) => () };
        let close_paren = select! { (Token::Close(Delimiter::Parent), _) => () };
        let comma = select! { (Token::Comma, _) => () };
        let arrow = select! { (Token::Op(Op::ThinArrow), _) => () };
        let open_bracket = select! { (Token::Open(Delimiter::Bracket), _) => () };
        let close_bracket = select! { (Token::Close(Delimiter::Bracket), _) => () };
        let semicolon = select! { (Token::Semicolon, _) => () };
        let colon = select! { (Token::Colon, _) => () };

        let param_type_parser: BoxedParser<'src, ast::Type> = if allow_view {
            type_parser.clone().boxed()
        } else {
            param_type_ident()
        };

        let array_len_fixed =
            select! { (Token::Literal(LitToken::Number(n)), _) => ast::ArrayLen::Fixed(n as usize) };
        let array_len_ident = identifier().map(|ident| {
            if ident.0.as_ref() == "_" {
                ast::ArrayLen::Infer
            } else {
                ast::ArrayLen::Named(ident)
            }
        });
        let array_len = choice((array_len_fixed, array_len_ident));

        let map_type = open_bracket
            .ignore_then(type_parser.clone())
            .then_ignore(colon)
            .then(type_parser.clone())
            .then_ignore(close_bracket)
            .map(|(key, value)| ast::Type::Map {
                key: key.boxed(),
                value: value.boxed(),
            });

        let list_type = open_bracket
            .ignore_then(type_parser.clone())
            .then_ignore(close_bracket)
            .map(|elem| ast::Type::List { elem: elem.boxed() });

        let array_type = open_bracket
            .ignore_then(type_parser.clone())
            .then_ignore(semicolon)
            .then(array_len)
            .then_ignore(close_bracket)
            .map(|(elem, len)| ast::Type::Array {
                elem: elem.boxed(),
                len,
            });

        let view_type = open_bracket
            .ignore_then(type_parser.clone())
            .then_ignore(semicolon)
            .then_ignore(select! { (Token::Range, _) => () })
            .then_ignore(close_bracket)
            .try_map(move |elem, span| {
                if allow_view {
                    Ok(ast::Type::ArrayView { elem: elem.boxed() })
                } else {
                    Err(Rich::custom(
                        span,
                        "view types are only allowed in function parameters",
                    ))
                }
            });

        let bracketed_type = choice((view_type, choice((array_type, choice((map_type, list_type))))));

        let var_kw = select! { (Token::Keyword(Keyword::Var), _) => () }
            .or_not()
            .map(|opt| opt.is_some());

        let fn_param = var_kw
            .then(param_type_parser)
            .map(|(mutable, ty)| ast::FuncParam::new(ty, mutable));

        let fn_type = select! { (Token::Keyword(Keyword::Fn), _) => () }
            .ignore_then(
                open_paren
                    .ignore_then(
                        fn_param
                            .separated_by(comma)
                            .allow_trailing()
                            .collect::<Vec<_>>()
                            .or_not()
                            .map(Option::unwrap_or_default),
                    )
                    .then_ignore(close_paren),
            )
            .then_ignore(arrow)
            .then(type_parser.clone())
            .map(|(params, ret)| ast::Type::Func {
                params,
                ret: ret.boxed(),
            });

        let primary_type = choice((builtin_typ, type_name_ref, paren_type, bracketed_type, fn_type));
        let optional_suffix = select! { (Token::Question, _) => TypeSuffix::Optional };
        let type_suffix = optional_suffix;

        primary_type
            .then(type_suffix.repeated().collect::<Vec<_>>())
            .map(|(base, suffixes)| {
                suffixes.into_iter().fold(base, |ty, sfx| match sfx {
                    TypeSuffix::Optional => Type::option_of(ty),
                })
            })
    })
    .labelled("type")
    .as_context()
    .boxed()
}

enum TupleTypeElem {
    Pos(ast::Type),
    Labeled(ast::Ident, ast::Type),
}

fn paren_or_tuple_type<'src>(
    type_parser: impl AnvParser<'src, ast::Type>,
) -> BoxedParser<'src, ast::Type> {
    let comma = select! { (Token::Comma, _) => () };
    let open_paren = select! { (Token::Open(Delimiter::Parent), _) => () };
    let close_paren = select! { (Token::Close(Delimiter::Parent), _) => () };
    let colon = select! { (Token::Colon, _) => () };

    let labeled_elem = identifier()
        .then_ignore(colon)
        .then(type_parser.clone())
        .map(|(name, ty)| TupleTypeElem::Labeled(name, ty));

    let pos_elem = type_parser.map(TupleTypeElem::Pos);
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

            match validate_tuple_shape_raw(first, rest, trailing_comma.is_some()) {
                TupleShapeResult::Empty => {
                    emitter.emit(Rich::custom(s, "empty tuples are not supported"));
                    ast::Type::Void
                }
                TupleShapeResult::OneTupleError(elem) => {
                    emitter.emit(Rich::custom(s, "1-tuples are not supported"));
                    match elem {
                        TupleTypeElem::Pos(ty) => ty,
                        TupleTypeElem::Labeled(_, ty) => ty,
                    }
                }
                TupleShapeResult::Grouped(elem) => match elem {
                    TupleTypeElem::Pos(ty) => ty,
                    TupleTypeElem::Labeled(_, ty) => ty,
                },
                TupleShapeResult::Tuple(elems) => {
                    let all_pos = elems.iter().all(|e| matches!(e, TupleTypeElem::Pos(_)));
                    let all_labeled = elems
                        .iter()
                        .all(|e| matches!(e, TupleTypeElem::Labeled(_, _)));

                    if all_pos {
                        let types: Vec<ast::Type> = elems
                            .into_iter()
                            .map(|e| match e {
                                TupleTypeElem::Pos(ty) => ty,
                                TupleTypeElem::Labeled(_, ty) => ty,
                            })
                            .collect();
                        return ast::Type::Tuple(types);
                    }

                    if all_labeled {
                        let fields: Vec<(ast::Ident, ast::Type)> = elems
                            .into_iter()
                            .map(|e| match e {
                                TupleTypeElem::Labeled(name, ty) => (name, ty),
                                TupleTypeElem::Pos(_) => unreachable!(),
                            })
                            .collect();
                        return ast::Type::NamedTuple(fields);
                    }

                    emitter.emit(Rich::custom(
                        s,
                        "cannot mix labeled and unlabeled elements in tuple type",
                    ));
                    ast::Type::Void
                }
                TupleShapeResult::UnexpectedComma => {
                    emitter.emit(Rich::custom(s, "unexpected comma"));
                    ast::Type::Void
                }
            }
        })
        .labelled("tuple or grouped type")
        .as_context()
        .boxed()
}
