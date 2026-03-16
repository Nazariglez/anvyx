use std::collections::HashMap;

use crate::{
    ast,
    lexer::{Delimiter, Keyword, Op, Token},
    span::{Span, Spanned},
};
use chumsky::{error::Rich, prelude::*};

use super::common::{block_stmt, identifier, param, params, return_type};
use super::expr::expression;
use super::types::type_ident;
use super::{AnvParser, BoxedParser};

fn type_params<'src>() -> BoxedParser<'src, Vec<ast::TypeParam>> {
    select! {
        (Token::Op(Op::LessThan), _) => (),
    }
    .ignore_then(
        identifier()
            .map_with(|name, e| {
                let id = e.state().new_type_var_id();
                ast::TypeParam { name, id }
            })
            .separated_by(select! {
                (Token::Comma, _) => (),
            })
            .allow_trailing()
            .collect::<Vec<_>>(),
    )
    .then_ignore(select! {
        (Token::Op(Op::GreaterThan), _) => (),
    })
    .or_not()
    .map(|opt| opt.unwrap_or_default())
    .labelled("type parameters")
    .as_context()
    .boxed()
}

pub(super) fn import_declaration<'src>() -> BoxedParser<'src, ast::StmtNode> {
    let import_kw = select! { (Token::Keyword(Keyword::Import), _) => () };
    let dot = select! { (Token::Dot, _) => () };
    let semicolon = select! { (Token::Semicolon, _) => () };
    let as_kw = select! { (Token::Keyword(Keyword::As), _) => () };
    let open_brace = select! { (Token::Open(Delimiter::Brace), _) => () };
    let close_brace = select! { (Token::Close(Delimiter::Brace), _) => () };
    let star = select! { (Token::Op(Op::Mul), _) => () };
    let comma = select! { (Token::Comma, _) => () };

    let import_path = identifier()
        .then(dot.ignore_then(identifier()).repeated().collect::<Vec<_>>())
        .map(|(first, mut rest)| {
            rest.insert(0, first);
            rest
        });

    let import_item = identifier()
        .then(as_kw.ignore_then(identifier()).or_not())
        .map(|(name, alias)| ast::ImportItem { name, alias });

    let selective_items = import_item
        .separated_by(comma.clone())
        .allow_trailing()
        .at_least(1)
        .collect::<Vec<_>>();

    let import_tail = choice((
        as_kw
            .ignore_then(identifier())
            .then_ignore(semicolon.clone())
            .map(ast::ImportKind::ModuleAs),
        open_brace
            .ignore_then(choice((
                star.to(ast::ImportKind::Wildcard),
                selective_items.map(ast::ImportKind::Selective),
            )))
            .then_ignore(close_brace)
            .then_ignore(semicolon.clone()),
        semicolon.to(ast::ImportKind::Module),
    ));

    import_kw
        .ignore_then(import_path)
        .then(import_tail)
        .map_with(|(path, kind), e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            let node = Spanned::new(ast::Import { path, kind }, span);
            Spanned::new(ast::Stmt::Import(node), span)
        })
        .labelled("import declaration")
        .as_context()
        .boxed()
}

pub(super) fn extern_declaration<'src>() -> BoxedParser<'src, ast::StmtNode> {
    select! { (Token::Keyword(Keyword::Extern), _) => () }
        .ignore_then(choice((
            // extern fn
            select! { (Token::Keyword(Keyword::Fn), _) => () }
                .ignore_then(identifier())
                .then(params())
                .then(return_type())
                .map_with(|((name, params), ret), e| {
                    let s = e.span();
                    let resolved_ret = ret.unwrap_or(ast::Type::Void);
                    let node = Spanned::new(
                        ast::ExternFunc {
                            name,
                            params,
                            ret: resolved_ret,
                        },
                        Span::new(s.start, s.end),
                    );
                    let span = node.span;
                    Spanned::new(ast::Stmt::ExternFunc(node), span)
                }),
            // extern type
            select! { (Token::Keyword(Keyword::Type), _) => () }
                .ignore_then(identifier())
                .map_with(|name, e| {
                    let s = e.span();
                    let node = Spanned::new(ast::ExternType { name }, Span::new(s.start, s.end));
                    let span = node.span;
                    Spanned::new(ast::Stmt::ExternType(node), span)
                }),
        )))
        .labelled("extern declaration")
        .as_context()
        .boxed()
}

fn visibility<'src>() -> BoxedParser<'src, ast::Visibility> {
    select! {
        (Token::Keyword(Keyword::Pub), _) => ast::Visibility::Public,
    }
    .or_not()
    .map(|v| v.unwrap_or(ast::Visibility::Private))
    .boxed()
}

pub(super) fn function<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, ast::FuncNode> {
    let tail_expr = expression(stmt.clone());
    visibility()
        .then_ignore(select! {
            (Token::Keyword(Keyword::Fn), _) => (),
        })
        .then(identifier())
        .then(type_params())
        .then(params())
        .then(return_type())
        .then(block_stmt(stmt, tail_expr))
        .map_with(|(((((vis, name), type_params), params), ret), body), e| {
            let s = e.span();
            let type_param_map: HashMap<ast::Ident, ast::TypeVarId> =
                type_params.iter().map(|tp| (tp.name, tp.id)).collect();

            let resolved_params = params
                .into_iter()
                .map(|p| {
                    let ty = resolve_type_params(&p.ty, &type_param_map);
                    ast::Param {
                        mutability: p.mutability,
                        name: p.name,
                        ty,
                    }
                })
                .collect();

            let resolved_ret = match ret {
                Some(ty) => resolve_type_params(&ty, &type_param_map),
                None => ast::Type::Void,
            };

            Spanned::new(
                ast::Func {
                    name,
                    visibility: vis,
                    type_params,
                    params: resolved_params,
                    ret: resolved_ret,
                    body,
                },
                Span::new(s.start, s.end),
            )
        })
        .labelled("function")
        .as_context()
        .boxed()
}

fn struct_field<'src>() -> BoxedParser<'src, ast::StructField> {
    identifier()
        .then_ignore(select! {
            (Token::Colon, _) => (),
        })
        .then(type_ident())
        .map(|(name, ty)| ast::StructField { name, ty })
        .labelled("struct field")
        .as_context()
        .boxed()
}

enum StructMember {
    Field(ast::StructField),
    Method(ast::Method),
}

fn struct_method<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, ast::Method> {
    let tail_expr = expression(stmt.clone());
    select! {
        (Token::Keyword(Keyword::Fn), _) => (),
    }
    .ignore_then(identifier())
    .then(type_params())
    .then(method_params())
    .then(return_type())
    .then(block_stmt(stmt, tail_expr))
    .map_with(
        |((((name, method_type_params), (receiver, params)), ret), body), e| {
            let s = e.span();

            let type_param_map = method_type_params
                .iter()
                .map(|tp| (tp.name, tp.id))
                .collect();

            let resolved_params = params
                .into_iter()
                .map(|p| {
                    let ty = resolve_type_params(&p.ty, &type_param_map);
                    ast::Param {
                        mutability: p.mutability,
                        name: p.name,
                        ty,
                    }
                })
                .collect();

            let resolved_ret = match ret {
                Some(ty) => resolve_type_params(&ty, &type_param_map),
                None => ast::Type::Void,
            };

            ast::Method {
                name,
                visibility: ast::Visibility::Private,
                type_params: method_type_params,
                receiver,
                params: resolved_params,
                ret: resolved_ret,
                body: Spanned::new(body.node, Span::new(s.start, s.end)),
            }
        },
    )
    .labelled("method")
    .as_context()
    .boxed()
}

fn method_params<'src>() -> BoxedParser<'src, (Option<ast::MethodReceiver>, Vec<ast::Param>)> {
    select! {
        (Token::Open(Delimiter::Parent), _) => (),
    }
    .ignore_then(
        method_param_list()
            .or_not()
            .map(|opt| opt.unwrap_or_default()),
    )
    .then_ignore(select! {
        (Token::Close(Delimiter::Parent), _) => (),
    })
    .boxed()
}

fn self_param<'src>() -> BoxedParser<'src, ast::MethodReceiver> {
    let var_kw = select! {
        (Token::Keyword(Keyword::Var), _) => (),
    }
    .or_not();

    var_kw
        .then(identifier().try_map(|ident, span| {
            if ident.0.as_ref() == "self" {
                Ok(())
            } else {
                Err(Rich::custom(span, "expected 'self'"))
            }
        }))
        .then_ignore(
            select! { (Token::Colon, _) => () }
                .ignore_then(type_ident())
                .or_not(),
        )
        .map(|(var_opt, _)| match var_opt {
            Some(()) => ast::MethodReceiver::Var,
            None => ast::MethodReceiver::Value,
        })
        .boxed()
}

fn method_param_list<'src>() -> BoxedParser<'src, (Option<ast::MethodReceiver>, Vec<ast::Param>)> {
    let regular_params = param()
        .separated_by(select! { (Token::Comma, _) => () })
        .collect::<Vec<_>>();

    choice((
        self_param()
            .then(
                select! { (Token::Comma, _) => () }
                    .ignore_then(regular_params.clone())
                    .or_not()
                    .map(|opt| opt.unwrap_or_default()),
            )
            .map(|(receiver, params)| (Some(receiver), params)),
        regular_params.map(|params| (None, params)),
    ))
    .boxed()
}

fn struct_member<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, StructMember> {
    choice((
        struct_method(stmt).map(StructMember::Method),
        struct_field().map(StructMember::Field),
    ))
    .boxed()
}

pub(super) fn struct_declaration<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, ast::StructDeclNode> {
    visibility()
        .then_ignore(select! {
            (Token::Keyword(Keyword::Struct), _) => (),
        })
        .then(identifier())
        .then(type_params())
        .then(
        select! {
            (Token::Open(Delimiter::Brace), _) => (),
        }
        .ignore_then(
            struct_member(stmt)
                .separated_by(select! { (Token::Comma, _) => () })
                .allow_trailing()
                .collect::<Vec<_>>(),
        )
        .then_ignore(select! {
            (Token::Close(Delimiter::Brace), _) => (),
        }),
    )
    .map_with(|(((vis, name), type_params), members), e| {
        let s = e.span();

        let struct_type_param_map: HashMap<ast::Ident, ast::TypeVarId> =
            type_params.iter().map(|tp| (tp.name, tp.id)).collect();

        let self_type = ast::Type::Struct {
            name,
            type_args: type_params.iter().map(|tp| ast::Type::Var(tp.id)).collect(),
        };

        let mut fields = vec![];
        let mut methods = vec![];

        for member in members {
            match member {
                StructMember::Field(f) => {
                    let ty = resolve_type_params_with_self(
                        &f.ty,
                        &struct_type_param_map,
                        Some(&self_type),
                    );
                    fields.push(ast::StructField { name: f.name, ty });
                }
                StructMember::Method(m) => {
                    let mut combined_type_param_map = struct_type_param_map.clone();
                    for tp in &m.type_params {
                        combined_type_param_map.insert(tp.name, tp.id);
                    }

                    let resolved_params = m
                        .params
                        .iter()
                        .map(|p| ast::Param {
                            mutability: p.mutability,
                            name: p.name,
                            ty: resolve_type_params_with_self(
                                &p.ty,
                                &combined_type_param_map,
                                Some(&self_type),
                            ),
                        })
                        .collect();

                    let resolved_ret = resolve_type_params_with_self(
                        &m.ret,
                        &combined_type_param_map,
                        Some(&self_type),
                    );
                    methods.push(ast::Method {
                        name: m.name,
                        visibility: m.visibility,
                        type_params: m.type_params,
                        receiver: m.receiver,
                        params: resolved_params,
                        ret: resolved_ret,
                        body: m.body,
                    });
                }
            }
        }

        Spanned::new(
            ast::StructDecl {
                name,
                visibility: vis,
                type_params,
                fields,
                methods,
            },
            Span::new(s.start, s.end),
        )
    })
    .labelled("struct declaration")
    .as_context()
    .boxed()
}

fn enum_variant_tuple_payload<'src>() -> BoxedParser<'src, ast::VariantKind> {
    select! { (Token::Open(Delimiter::Parent), _) => () }
        .ignore_then(
            type_ident()
                .separated_by(select! { (Token::Comma, _) => () })
                .allow_trailing()
                .collect::<Vec<_>>(),
        )
        .then_ignore(select! { (Token::Close(Delimiter::Parent), _) => () })
        .map(ast::VariantKind::Tuple)
        .boxed()
}

fn enum_variant_struct_payload<'src>() -> BoxedParser<'src, ast::VariantKind> {
    select! { (Token::Open(Delimiter::Brace), _) => () }
        .ignore_then(
            struct_field()
                .separated_by(select! { (Token::Comma, _) => () })
                .allow_trailing()
                .collect::<Vec<_>>(),
        )
        .then_ignore(select! { (Token::Close(Delimiter::Brace), _) => () })
        .map(ast::VariantKind::Struct)
        .boxed()
}

fn enum_variant<'src>() -> BoxedParser<'src, ast::EnumVariant> {
    identifier()
        .then(choice((
            enum_variant_tuple_payload(),
            enum_variant_struct_payload(),
            empty().to(ast::VariantKind::Unit),
        )))
        .map(|(name, kind)| ast::EnumVariant { name, kind })
        .labelled("enum variant")
        .as_context()
        .boxed()
}

pub(super) fn enum_declaration<'src>() -> BoxedParser<'src, ast::EnumDeclNode> {
    visibility()
        .then_ignore(select! { (Token::Keyword(Keyword::Enum), _) => () })
        .then(identifier())
        .then(type_params())
        .then(
            select! { (Token::Open(Delimiter::Brace), _) => () }
                .ignore_then(
                    enum_variant()
                        .separated_by(select! { (Token::Comma, _) => () })
                        .allow_trailing()
                        .collect::<Vec<_>>(),
                )
                .then_ignore(select! { (Token::Close(Delimiter::Brace), _) => () }),
        )
        .map_with(|(((vis, name), type_params), variants), e| {
            let s = e.span();

            let type_param_map: HashMap<ast::Ident, ast::TypeVarId> =
                type_params.iter().map(|tp| (tp.name, tp.id)).collect();

            let resolved_variants = variants
                .into_iter()
                .map(|v| {
                    let resolved_kind = match v.kind {
                        ast::VariantKind::Unit => ast::VariantKind::Unit,
                        ast::VariantKind::Tuple(types) => {
                            let resolved = types
                                .iter()
                                .map(|ty| resolve_type_params(ty, &type_param_map))
                                .collect();
                            ast::VariantKind::Tuple(resolved)
                        }
                        ast::VariantKind::Struct(fields) => {
                            let resolved = fields
                                .iter()
                                .map(|f| ast::StructField {
                                    name: f.name,
                                    ty: resolve_type_params(&f.ty, &type_param_map),
                                })
                                .collect();
                            ast::VariantKind::Struct(resolved)
                        }
                    };
                    ast::EnumVariant {
                        name: v.name,
                        kind: resolved_kind,
                    }
                })
                .collect();

            Spanned::new(
                ast::EnumDecl {
                    name,
                    visibility: vis,
                    type_params,
                    variants: resolved_variants,
                },
                Span::new(s.start, s.end),
            )
        })
        .labelled("enum declaration")
        .as_context()
        .boxed()
}

fn resolve_type_params_with_self(
    ty: &ast::Type,
    type_param_map: &HashMap<ast::Ident, ast::TypeVarId>,
    self_type: Option<&ast::Type>,
) -> ast::Type {
    use ast::Type::*;
    match ty {
        UnresolvedName(ident) => {
            if let Some(id) = type_param_map.get(ident) {
                return Var(*id);
            }
            if let Some(st) = self_type
                && ident.0.as_ref() == "Self"
            {
                return st.clone();
            }
            ty.clone()
        }

        Enum { name, type_args } => Enum {
            name: *name,
            type_args: type_args
                .iter()
                .map(|a| resolve_type_params_with_self(a, type_param_map, self_type))
                .collect(),
        },

        Func { params, ret } => {
            let resolved_params = params
                .iter()
                .map(|p| resolve_type_params_with_self(p, type_param_map, self_type))
                .collect::<Vec<_>>();
            Func {
                params: resolved_params,
                ret: resolve_type_params_with_self(ret, type_param_map, self_type).boxed(),
            }
        }

        Tuple(elements) => {
            let resolved_elements = elements
                .iter()
                .map(|el| resolve_type_params_with_self(el, type_param_map, self_type))
                .collect::<Vec<_>>();
            Tuple(resolved_elements)
        }

        NamedTuple(fields) => {
            let resolved_fields: Vec<_> = fields
                .iter()
                .map(|(name, ty)| {
                    (
                        *name,
                        resolve_type_params_with_self(ty, type_param_map, self_type),
                    )
                })
                .collect();
            NamedTuple(resolved_fields)
        }

        Struct { name, type_args } => {
            let resolved_args = type_args
                .iter()
                .map(|arg| resolve_type_params_with_self(arg, type_param_map, self_type))
                .collect::<Vec<_>>();
            Struct {
                name: *name,
                type_args: resolved_args,
            }
        }

        Array { elem, len } => Array {
            elem: resolve_type_params_with_self(elem, type_param_map, self_type).boxed(),
            len: *len,
        },

        ArrayView { elem } => ArrayView {
            elem: resolve_type_params_with_self(elem, type_param_map, self_type).boxed(),
        },

        List { elem } => List {
            elem: resolve_type_params_with_self(elem, type_param_map, self_type).boxed(),
        },

        Map { key, value } => Map {
            key: resolve_type_params_with_self(key, type_param_map, self_type).boxed(),
            value: resolve_type_params_with_self(value, type_param_map, self_type).boxed(),
        },

        _ => ty.clone(),
    }
}

fn resolve_type_params(
    ty: &ast::Type,
    type_param_map: &HashMap<ast::Ident, ast::TypeVarId>,
) -> ast::Type {
    resolve_type_params_with_self(ty, type_param_map, None)
}
