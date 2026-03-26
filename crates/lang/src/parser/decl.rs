use std::collections::HashMap;

use crate::{
    ast,
    lexer::{Delimiter, Keyword, Op, Token},
    span::{Span, Spanned},
};
use chumsky::{error::Rich, prelude::*};

use super::common::{block_stmt, field_name_ident, identifier, param, params, return_type};
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

    visibility()
        .then_ignore(import_kw)
        .then(import_path)
        .then(import_tail)
        .map_with(|((visibility, path), kind), e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            let node = Spanned::new(
                ast::Import {
                    visibility,
                    path,
                    kind,
                },
                span,
            );
            Spanned::new(ast::Stmt::Import(node), span)
        })
        .labelled("import declaration")
        .as_context()
        .boxed()
}

pub(super) fn extern_declaration<'src>() -> BoxedParser<'src, ast::StmtNode> {
    let semicolon = select! { (Token::Semicolon, _) => () };

    select! { (Token::Keyword(Keyword::Extern), _) => () }
        .ignore_then(choice((
            // extern fn ... ;
            select! { (Token::Keyword(Keyword::Fn), _) => () }
                .ignore_then(identifier())
                .then(params())
                .then(return_type())
                .then_ignore(semicolon)
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
            extern_type_declaration(),
        )))
        .labelled("extern declaration")
        .as_context()
        .boxed()
}

fn extern_type_declaration<'src>() -> BoxedParser<'src, ast::StmtNode> {
    let semicolon = select! { (Token::Semicolon, _) => () };

    select! { (Token::Keyword(Keyword::Type), _) => () }
        .ignore_then(identifier())
        .then(choice((
            extern_type_body().map(Some),
            semicolon.map(|_| None),
        )))
        .map_with(|(name, body), e| {
            let s = e.span();
            let (members, has_init) = body.unwrap_or((vec![], false));
            let self_type = ast::Type::Extern { name };
            let empty_map = HashMap::new();
            let resolved_members = resolve_extern_members(members, &empty_map, &self_type);
            let node = Spanned::new(
                ast::ExternType {
                    name,
                    has_init,
                    members: resolved_members,
                },
                Span::new(s.start, s.end),
            );
            let span = node.span;
            Spanned::new(ast::Stmt::ExternType(node), span)
        })
        .boxed()
}

fn resolve_extern_members(
    members: Vec<ast::ExternTypeMember>,
    type_param_map: &HashMap<ast::Ident, ast::TypeVarId>,
    self_type: &ast::Type,
) -> Vec<ast::ExternTypeMember> {
    members
        .into_iter()
        .map(|member| match member {
            ast::ExternTypeMember::Field { name, ty, computed } => ast::ExternTypeMember::Field {
                name,
                ty: resolve_type_params_with_self(&ty, type_param_map, Some(self_type)),
                computed,
            },
            ast::ExternTypeMember::Method {
                name,
                receiver,
                params,
                ret,
            } => ast::ExternTypeMember::Method {
                name,
                receiver,
                params: params
                    .iter()
                    .map(|p| ast::Param {
                        mutability: p.mutability,
                        name: p.name,
                        ty: resolve_type_params_with_self(&p.ty, type_param_map, Some(self_type)),
                    })
                    .collect(),
                ret: resolve_type_params_with_self(&ret, type_param_map, Some(self_type)),
            },
            ast::ExternTypeMember::StaticMethod { name, params, ret } => {
                ast::ExternTypeMember::StaticMethod {
                    name,
                    params: params
                        .iter()
                        .map(|p| ast::Param {
                            mutability: p.mutability,
                            name: p.name,
                            ty: resolve_type_params_with_self(
                                &p.ty,
                                type_param_map,
                                Some(self_type),
                            ),
                        })
                        .collect(),
                    ret: resolve_type_params_with_self(&ret, type_param_map, Some(self_type)),
                }
            }
            ast::ExternTypeMember::Operator { op, other_ty, ret, self_on_right } => {
                ast::ExternTypeMember::Operator {
                    op,
                    other_ty: resolve_type_params_with_self(&other_ty, type_param_map, Some(self_type)),
                    ret: resolve_type_params_with_self(&ret, type_param_map, Some(self_type)),
                    self_on_right,
                }
            }
            ast::ExternTypeMember::UnaryOperator { op, ret } => {
                ast::ExternTypeMember::UnaryOperator {
                    op,
                    ret: resolve_type_params_with_self(&ret, type_param_map, Some(self_type)),
                }
            }
        })
        .collect()
}

fn extern_type_body<'src>() -> BoxedParser<'src, (Vec<ast::ExternTypeMember>, bool)> {
    enum BodyItem {
        Member(ast::ExternTypeMember),
        Init,
    }

    let semicolon = select! { (Token::Semicolon, _) => () };
    let init_item = select! { (Token::Ident(ident), _) if ident.0.as_ref() == "init" => () }
        .then_ignore(semicolon.clone())
        .map(|_| BodyItem::Init);
    let member_item = extern_type_member().map(BodyItem::Member);

    select! { (Token::Open(Delimiter::Brace), _) => () }
        .ignore_then(choice((init_item, member_item)).repeated().collect::<Vec<_>>())
        .then_ignore(select! { (Token::Close(Delimiter::Brace), _) => () })
        .validate(|items, extra, emitter| {
            let mut members = vec![];
            let mut init_count = 0usize;
            for item in items {
                match item {
                    BodyItem::Member(m) => members.push(m),
                    BodyItem::Init => {
                        init_count += 1;
                        if init_count > 1 {
                            emitter.emit(Rich::custom(extra.span(), "duplicate 'init' in extern type body"));
                        }
                    }
                }
            }
            (members, init_count > 0)
        })
        .boxed()
}

fn extern_type_member<'src>() -> BoxedParser<'src, ast::ExternTypeMember> {
    let semicolon = select! { (Token::Semicolon, _) => () };

    choice((
        extern_type_op_member().then_ignore(semicolon.clone()),
        extern_type_method_member().then_ignore(semicolon.clone()),
        extern_type_field_member().then_ignore(semicolon),
    ))
    .boxed()
}

fn is_self_type(ty: &ast::Type) -> bool {
    matches!(ty, ast::Type::UnresolvedName(ident) if ident.0.as_ref() == "Self")
}

fn extern_type_op_member<'src>() -> BoxedParser<'src, ast::ExternTypeMember> {
    let op_kw = select! { (Token::Ident(ident), _) if ident.0.as_ref() == "op" => () };
    let arrow = select! { (Token::Op(Op::ThinArrow), _) => () };

    let binary_op_tok = select! {
        (Token::Op(Op::Add), _) => ast::BinaryOp::Add,
        (Token::Op(Op::Sub), _) => ast::BinaryOp::Sub,
        (Token::Op(Op::Mul), _) => ast::BinaryOp::Mul,
        (Token::Op(Op::Div), _) => ast::BinaryOp::Div,
        (Token::Op(Op::Rem), _) => ast::BinaryOp::Rem,
        (Token::Op(Op::Eq), _) => ast::BinaryOp::Eq,
    };

    let unary = select! { (Token::Op(Op::Sub), _) => () }
        .then_ignore(type_ident())
        .then_ignore(arrow.clone())
        .then(type_ident())
        .map(|((), ret)| ast::ExternTypeMember::UnaryOperator {
            op: ast::UnaryOp::Neg,
            ret,
        });

    let binary = type_ident()
        .then(binary_op_tok)
        .then(type_ident())
        .then_ignore(arrow)
        .then(type_ident())
        .validate(|(((lhs, op), rhs), ret), extra, emitter| {
            let (other_ty, self_on_right) = if is_self_type(&lhs) {
                (rhs, false)
            } else if is_self_type(&rhs) {
                (lhs, true)
            } else {
                emitter.emit(Rich::custom(extra.span(), "one operand must be 'Self'"));
                (lhs, false)
            };
            ast::ExternTypeMember::Operator { op, other_ty, ret, self_on_right }
        });

    op_kw.ignore_then(choice((unary, binary))).boxed()
}

fn extern_type_field_member<'src>() -> BoxedParser<'src, ast::ExternTypeMember> {
    identifier()
        .then_ignore(select! { (Token::Colon, _) => () })
        .then(type_ident())
        .map(|(name, ty)| ast::ExternTypeMember::Field { name, ty, computed: false })
        .boxed()
}

fn extern_type_method_member<'src>() -> BoxedParser<'src, ast::ExternTypeMember> {
    select! { (Token::Keyword(Keyword::Fn), _) => () }
        .ignore_then(field_name_ident())
        .then(method_params())
        .then(return_type())
        .map(|((name, (receiver, params)), ret)| {
            let ret = ret.unwrap_or(ast::Type::Void);
            match receiver {
                Some(recv) => ast::ExternTypeMember::Method {
                    name,
                    receiver: recv,
                    params,
                    ret,
                },
                None => ast::ExternTypeMember::StaticMethod { name, params, ret },
            }
        })
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
                struct_field()
                    .separated_by(select! { (Token::Comma, _) => () })
                    .allow_trailing()
                    .collect::<Vec<_>>(),
            )
            .then(
                struct_method(stmt)
                    .repeated()
                    .collect::<Vec<_>>(),
            )
            .then_ignore(select! {
                (Token::Close(Delimiter::Brace), _) => (),
            }),
        )
        .map_with(|(((vis, name), type_params), (raw_fields, raw_methods)), e| {
            let s = e.span();

            let struct_type_param_map: HashMap<ast::Ident, ast::TypeVarId> =
                type_params.iter().map(|tp| (tp.name, tp.id)).collect();

            let self_type = ast::Type::Struct {
                name,
                type_args: type_params.iter().map(|tp| ast::Type::Var(tp.id)).collect(),
            };

            let fields = raw_fields
                .into_iter()
                .map(|f| {
                    let ty = resolve_type_params_with_self(
                        &f.ty,
                        &struct_type_param_map,
                        Some(&self_type),
                    );
                    ast::StructField { name: f.name, ty }
                })
                .collect();

            let methods = raw_methods
                .into_iter()
                .map(|m| {
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

                    ast::Method {
                        name: m.name,
                        visibility: m.visibility,
                        type_params: m.type_params,
                        receiver: m.receiver,
                        params: resolved_params,
                        ret: resolved_ret,
                        body: m.body,
                    }
                })
                .collect();

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

fn extend_method<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, ast::ExtendMethodNode> {
    let tail_expr = expression(stmt.clone());
    select! {
        (Token::Keyword(Keyword::Fn), _) => (),
    }
    .ignore_then(field_name_ident())
    .then(method_params())
    .then(return_type())
    .then(block_stmt(stmt, tail_expr))
    .map_with(|(((name, (receiver, params)), ret), body), e| {
        let s = e.span();
        let self_param = receiver.map(|r| ast::Param {
            mutability: match r {
                ast::MethodReceiver::Var => ast::Mutability::Mutable,
                ast::MethodReceiver::Value => ast::Mutability::Immutable,
            },
            name: ast::Ident(internment::Intern::new("self".to_string())),
            ty: ast::Type::Infer,
        });
        let all_params: Vec<ast::Param> = self_param.into_iter().chain(params).collect();
        let ret_ty = ret.unwrap_or(ast::Type::Void);
        Spanned::new(
            ast::ExtendMethod {
                name,
                params: all_params,
                ret: ret_ty,
                body: Spanned::new(body.node, Span::new(s.start, s.end)),
            },
            Span::new(s.start, s.end),
        )
    })
    .labelled("extend method")
    .as_context()
    .boxed()
}

pub(super) fn extend_declaration<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, ast::ExtendDeclNode> {
    visibility()
        .then_ignore(select! {
            (Token::Keyword(Keyword::Extend), _) => (),
        })
        .then(type_ident())
        .then(
            select! { (Token::Open(Delimiter::Brace), _) => () }
                .ignore_then(
                    extend_method(stmt)
                        .repeated()
                        .collect::<Vec<_>>(),
                )
                .then_ignore(select! { (Token::Close(Delimiter::Brace), _) => () }),
        )
        .map_with(|((vis, ty), methods), e| {
            let s = e.span();
            Spanned::new(
                ast::ExtendDecl {
                    visibility: vis,
                    ty,
                    methods,
                },
                Span::new(s.start, s.end),
            )
        })
        .labelled("extend declaration")
        .as_context()
        .boxed()
}

pub(super) fn const_decl<'src>(
    stmt: impl AnvParser<'src, ast::StmtNode>,
) -> BoxedParser<'src, ast::StmtNode> {
    visibility()
        .then_ignore(select! {
            (Token::Keyword(Keyword::Const), _) => (),
        })
        .then(identifier())
        .then(
            select! { (Token::Colon, _) => () }
                .ignore_then(type_ident())
                .or_not(),
        )
        .then_ignore(select! { (Token::Op(Op::Assign), _) => () })
        .then(expression(stmt))
        .then_ignore(select! { (Token::Semicolon, _) => () })
        .map_with(|(((vis, name), ty), value), e| {
            let s = e.span();
            let span = Span::new(s.start, s.end);
            let node = Spanned::new(
                ast::ConstDecl {
                    name,
                    ty,
                    value,
                    visibility: vis,
                },
                span,
            );
            Spanned::new(ast::Stmt::Const(node), span)
        })
        .labelled("const declaration")
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
