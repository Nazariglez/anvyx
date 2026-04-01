use super::{
    FuncLower, LowerCtx, LowerError, alloc_and_bind, emit_counter_increment, lower_block,
    lower_expr, register_named_local,
};

fn destructure_map_entry_tuple(
    subs: &[ast::PatternNode],
    entry_ty: &Type,
    entry_local: hir::LocalId,
    types: &[&Type; 2],
    span: Span,
    fc: &mut FuncLower,
    body_stmts: &mut Vec<hir::Stmt>,
) -> Result<(), LowerError> {
    for (k, sub) in subs.iter().enumerate() {
        match &sub.node {
            Pattern::Ident(name) => {
                let local_id = register_named_local(fc, *name, types[k].clone());
                body_stmts.push(hir::Stmt {
                    span,
                    kind: hir::StmtKind::Let {
                        local: local_id,
                        init: hir::Expr::new(
                            types[k].clone(),
                            span,
                            hir::ExprKind::TupleIndex {
                                tuple: Box::new(hir::Expr::local(
                                    entry_ty.clone(),
                                    span,
                                    entry_local,
                                )),
                                index: k as u16,
                            },
                        ),
                    },
                });
            }
            Pattern::Wildcard => {}
            _ => return Err(LowerError::UnsupportedPattern { span }),
        }
    }
    Ok(())
}

fn lower_for_user_body(
    for_node: &ast::ForNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    body_stmts: &mut Vec<hir::Stmt>,
) -> Result<(), LowerError> {
    let old = fc.enter_loop_defer();
    let user_body = lower_block(&for_node.node.body, ctx, fc, false, &Type::Void)?;
    fc.leave_loop_defer(old);
    body_stmts.extend(user_body.stmts);
    Ok(())
}
use crate::{
    ast::{self, BinaryOp, Pattern, Type},
    hir,
    span::Span,
};

enum IterableKind {
    Range,
    Sequence(Type),
    Map { key_ty: Type, value_ty: Type },
}

fn classify_iterable(ty: &Type) -> Option<IterableKind> {
    match ty {
        Type::Struct { name, type_args } => {
            let name_str = name.0.as_ref();
            let is_range =
                name_str == "Range" || name_str == "RangeInclusive" || name_str == "RangeFrom";
            if is_range && type_args.len() == 1 {
                Some(IterableKind::Range)
            } else {
                None
            }
        }
        Type::Array { elem, .. } | Type::List { elem } | Type::ArrayView { elem } => {
            Some(IterableKind::Sequence(*elem.clone()))
        }
        Type::Map { key, value } => Some(IterableKind::Map {
            key_ty: *key.clone(),
            value_ty: *value.clone(),
        }),
        _ => None,
    }
}

pub(super) fn lower_for(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Stmt>, LowerError> {
    let iterable_ty = ctx.expr_type(for_node.node.iterable.node.id, span)?;
    match classify_iterable(&iterable_ty) {
        Some(IterableKind::Range) => lower_for_range(for_node, span, ctx, fc, out),
        Some(IterableKind::Sequence(elem_ty)) => {
            lower_for_sequence(for_node, span, ctx, fc, out, &elem_ty)
        }
        Some(IterableKind::Map { key_ty, value_ty }) => {
            lower_for_map(for_node, span, ctx, fc, out, &key_ty, &value_ty)
        }
        None => Err(LowerError::UnsupportedStmtKind {
            span,
            kind: "for loop over unsupported iterable type".to_string(),
        }),
    }
}

fn is_inclusive_range_type(ty: &Type) -> bool {
    matches!(ty, Type::Struct { name, .. } if name.0.as_ref() == "RangeInclusive")
}

fn is_range_from_type(ty: &Type) -> bool {
    matches!(ty, Type::Struct { name, .. } if name.0.as_ref() == "RangeFrom")
}

fn lower_for_range(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
) -> Result<Option<hir::Stmt>, LowerError> {
    let mark = fc.enter_scope();

    let iterable_ty = ctx.expr_type(for_node.node.iterable.node.id, span)?;
    if is_range_from_type(&iterable_ty) {
        return lower_for_range_from(for_node, span, ctx, fc, out, mark, &iterable_ty);
    }

    let (start_expr, end_expr, inclusive, item_ty);

    if let ast::ExprKind::Range(range_node) = &for_node.node.iterable.node.kind
        && let ast::Range::Bounded {
            start,
            end,
            inclusive: inc,
        } = &range_node.node
    {
        item_ty = ctx.expr_type(start.node.id, span)?;
        start_expr = lower_expr(start, ctx, fc, out)?;
        end_expr = lower_expr(end, ctx, fc, out)?;
        inclusive = *inc;
    } else {
        inclusive = is_inclusive_range_type(&iterable_ty);
        let Type::Struct { type_args, .. } = &iterable_ty else {
            return Err(LowerError::UnsupportedStmtKind {
                span,
                kind: "for loop over non-range iterable".to_string(),
            });
        };
        item_ty = type_args[0].clone();
        let range_expr = lower_expr(&for_node.node.iterable, ctx, fc, out)?;
        let range_local = alloc_and_bind(fc, span, out, iterable_ty.clone(), range_expr);
        let range_local_expr = hir::Expr::local(iterable_ty, span, range_local);
        start_expr = hir::Expr::new(
            item_ty.clone(),
            span,
            hir::ExprKind::FieldGet {
                object: Box::new(range_local_expr.clone()),
                index: 0,
            },
        );
        end_expr = hir::Expr::new(
            item_ty.clone(),
            span,
            hir::ExprKind::FieldGet {
                object: Box::new(range_local_expr),
                index: 1,
            },
        );
    }

    let step_expr = match &for_node.node.step {
        Some(s) => lower_expr(s, ctx, fc, out)?,
        None => hir::Expr::new(item_ty.clone(), span, hir::ExprKind::Int(1)),
    };

    let reversed = for_node.node.reversed;

    let (bound_local, i_init, cmp_op, inc_op) = if reversed {
        let start_local = alloc_and_bind(fc, span, out, item_ty.clone(), start_expr);
        let i_init = if inclusive {
            end_expr
        } else {
            hir::Expr::binary(
                item_ty.clone(),
                span,
                BinaryOp::Sub,
                end_expr,
                hir::Expr::new(item_ty.clone(), span, hir::ExprKind::Int(1)),
            )
        };
        (start_local, i_init, BinaryOp::GreaterThanEq, BinaryOp::Sub)
    } else {
        let end_local = alloc_and_bind(fc, span, out, item_ty.clone(), end_expr);
        let cmp_op = if inclusive {
            BinaryOp::LessThanEq
        } else {
            BinaryOp::LessThan
        };
        (end_local, start_expr, cmp_op, BinaryOp::Add)
    };

    let step_local = alloc_and_bind(fc, span, out, item_ty.clone(), step_expr);
    let i_local = alloc_and_bind(fc, span, out, item_ty.clone(), i_init);

    let cond = hir::Expr::binary(
        Type::Bool,
        span,
        cmp_op,
        hir::Expr::local(item_ty.clone(), span, i_local),
        hir::Expr::local(item_ty.clone(), span, bound_local),
    );

    let body_stmts = lower_for_body(
        for_node, span, ctx, fc, &item_ty, i_local, step_local, inc_op,
    )?;

    fc.leave_scope(mark);
    Ok(Some(hir::Stmt {
        span,
        kind: hir::StmtKind::While {
            cond,
            body: hir::Block { stmts: body_stmts },
        },
    }))
}

fn lower_for_range_from(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
    mark: usize,
    iterable_ty: &Type,
) -> Result<Option<hir::Stmt>, LowerError> {
    let Type::Struct { type_args, .. } = iterable_ty else {
        return Err(LowerError::UnsupportedStmtKind {
            span,
            kind: "for loop over non-range iterable".to_string(),
        });
    };
    let item_ty = type_args[0].clone();

    let start_expr = if let ast::ExprKind::Range(range_node) = &for_node.node.iterable.node.kind
        && let ast::Range::From { start } = &range_node.node
    {
        lower_expr(start, ctx, fc, out)?
    } else {
        let range_expr = lower_expr(&for_node.node.iterable, ctx, fc, out)?;
        let range_local = alloc_and_bind(fc, span, out, iterable_ty.clone(), range_expr);
        hir::Expr::new(
            item_ty.clone(),
            span,
            hir::ExprKind::FieldGet {
                object: Box::new(hir::Expr::local(iterable_ty.clone(), span, range_local)),
                index: 0,
            },
        )
    };

    let step_expr = match &for_node.node.step {
        Some(s) => lower_expr(s, ctx, fc, out)?,
        None => hir::Expr::new(item_ty.clone(), span, hir::ExprKind::Int(1)),
    };

    let step_local = alloc_and_bind(fc, span, out, item_ty.clone(), step_expr);
    let i_local = alloc_and_bind(fc, span, out, item_ty.clone(), start_expr);

    let cond = hir::Expr::new(Type::Bool, span, hir::ExprKind::Bool(true));

    let body_stmts = lower_for_body(
        for_node,
        span,
        ctx,
        fc,
        &item_ty,
        i_local,
        step_local,
        BinaryOp::Add,
    )?;

    fc.leave_scope(mark);
    Ok(Some(hir::Stmt {
        span,
        kind: hir::StmtKind::While {
            cond,
            body: hir::Block { stmts: body_stmts },
        },
    }))
}

fn lower_for_sequence(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
    elem_ty: &Type,
) -> Result<Option<hir::Stmt>, LowerError> {
    let mark = fc.enter_scope();

    let xs_expr = lower_expr(&for_node.node.iterable, ctx, fc, out)?;
    let xs_ty = xs_expr.ty.clone();

    let xs_local = alloc_and_bind(fc, span, out, xs_ty.clone(), xs_expr);

    let len_expr = hir::Expr::new(
        Type::Int,
        span,
        hir::ExprKind::CollectionLen {
            collection: Box::new(hir::Expr::local(xs_ty.clone(), span, xs_local)),
        },
    );
    let len_local = alloc_and_bind(fc, span, out, Type::Int, len_expr);

    let step_expr = match &for_node.node.step {
        Some(s) => lower_expr(s, ctx, fc, out)?,
        None => hir::Expr::int_lit(span, 1),
    };
    let step_local = alloc_and_bind(fc, span, out, Type::Int, step_expr);

    let reversed = for_node.node.reversed;

    let i_init = if reversed {
        hir::Expr::binary(
            Type::Int,
            span,
            BinaryOp::Sub,
            hir::Expr::local(Type::Int, span, len_local),
            hir::Expr::int_lit(span, 1),
        )
    } else {
        hir::Expr::int_lit(span, 0)
    };
    let i_local = alloc_and_bind(fc, span, out, Type::Int, i_init);

    let cond = if reversed {
        hir::Expr::binary(
            Type::Bool,
            span,
            BinaryOp::GreaterThanEq,
            hir::Expr::local(Type::Int, span, i_local),
            hir::Expr::int_lit(span, 0),
        )
    } else {
        hir::Expr::binary(
            Type::Bool,
            span,
            BinaryOp::LessThan,
            hir::Expr::local(Type::Int, span, i_local),
            hir::Expr::local(Type::Int, span, len_local),
        )
    };

    let inc_op = if reversed {
        BinaryOp::Sub
    } else {
        BinaryOp::Add
    };

    let body_stmts = lower_for_seq_body(
        for_node, span, ctx, fc, elem_ty, xs_local, &xs_ty, i_local, step_local, inc_op,
    )?;

    fc.leave_scope(mark);
    Ok(Some(hir::Stmt {
        span,
        kind: hir::StmtKind::While {
            cond,
            body: hir::Block { stmts: body_stmts },
        },
    }))
}

#[allow(clippy::too_many_arguments)]
fn lower_for_seq_body(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    elem_ty: &Type,
    xs_local: hir::LocalId,
    xs_ty: &Type,
    i_local: hir::LocalId,
    step_local: hir::LocalId,
    inc_op: BinaryOp,
) -> Result<Vec<hir::Stmt>, LowerError> {
    let mut body_stmts = vec![];

    let index_get_expr = || {
        hir::Expr::new(
            elem_ty.clone(),
            span,
            hir::ExprKind::IndexGet {
                target: Box::new(hir::Expr::local(xs_ty.clone(), span, xs_local)),
                index: Box::new(hir::Expr::local(Type::Int, span, i_local)),
            },
        )
    };

    match &for_node.node.pattern.node {
        Pattern::Ident(name) => {
            let local_id = register_named_local(fc, *name, elem_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: local_id,
                    init: index_get_expr(),
                },
            });
        }
        Pattern::Wildcard => {}
        Pattern::Tuple(subs) if subs.len() == 2 => {
            let is_2_tuple = elem_ty
                .tuple_element_types()
                .is_some_and(|elems| elems.len() == 2);

            if is_2_tuple {
                let tuple_elems = elem_ty.tuple_element_types().unwrap();

                let el_local =
                    alloc_and_bind(fc, span, &mut body_stmts, elem_ty.clone(), index_get_expr());

                for (k, sub) in subs.iter().enumerate() {
                    if let Pattern::Ident(name) = &sub.node {
                        let local_id = register_named_local(fc, *name, tuple_elems[k].clone());
                        body_stmts.push(hir::Stmt {
                            span,
                            kind: hir::StmtKind::Let {
                                local: local_id,
                                init: hir::Expr::new(
                                    tuple_elems[k].clone(),
                                    span,
                                    hir::ExprKind::TupleIndex {
                                        tuple: Box::new(hir::Expr::local(
                                            elem_ty.clone(),
                                            span,
                                            el_local,
                                        )),
                                        index: k as u16,
                                    },
                                ),
                            },
                        });
                    }
                }
            } else {
                if let Pattern::Ident(name) = &subs[0].node {
                    let local_id = register_named_local(fc, *name, Type::Int);
                    body_stmts.push(hir::Stmt {
                        span,
                        kind: hir::StmtKind::Let {
                            local: local_id,
                            init: hir::Expr::local(Type::Int, span, i_local),
                        },
                    });
                }

                if let Pattern::Ident(name) = &subs[1].node {
                    let local_id = register_named_local(fc, *name, elem_ty.clone());
                    body_stmts.push(hir::Stmt {
                        span,
                        kind: hir::StmtKind::Let {
                            local: local_id,
                            init: index_get_expr(),
                        },
                    });
                }
            }
        }
        _ => return Err(LowerError::UnsupportedPattern { span }),
    }

    emit_counter_increment(
        &mut body_stmts,
        span,
        i_local,
        &Type::Int,
        hir::Expr::local(Type::Int, span, step_local),
        inc_op,
    );

    lower_for_user_body(for_node, ctx, fc, &mut body_stmts)?;

    Ok(body_stmts)
}

fn lower_for_map(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    out: &mut Vec<hir::Stmt>,
    key_ty: &Type,
    value_ty: &Type,
) -> Result<Option<hir::Stmt>, LowerError> {
    let mark = fc.enter_scope();

    let m_expr = lower_expr(&for_node.node.iterable, ctx, fc, out)?;
    let m_ty = m_expr.ty.clone();

    let m_local = alloc_and_bind(fc, span, out, m_ty.clone(), m_expr);

    let len_expr = hir::Expr::new(
        Type::Int,
        span,
        hir::ExprKind::MapLen {
            map: Box::new(hir::Expr::local(m_ty.clone(), span, m_local)),
        },
    );
    let len_local = alloc_and_bind(fc, span, out, Type::Int, len_expr);

    let i_local = alloc_and_bind(fc, span, out, Type::Int, hir::Expr::int_lit(span, 0));

    let cond = hir::Expr::binary(
        Type::Bool,
        span,
        BinaryOp::LessThan,
        hir::Expr::local(Type::Int, span, i_local),
        hir::Expr::local(Type::Int, span, len_local),
    );

    let entry_ty = Type::Tuple(vec![key_ty.clone(), value_ty.clone()]);
    let body_stmts = lower_for_map_body(
        for_node, span, ctx, fc, key_ty, value_ty, &entry_ty, m_local, &m_ty, i_local,
    )?;

    fc.leave_scope(mark);
    Ok(Some(hir::Stmt {
        span,
        kind: hir::StmtKind::While {
            cond,
            body: hir::Block { stmts: body_stmts },
        },
    }))
}

#[allow(clippy::too_many_arguments)]
fn lower_for_map_body(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    key_ty: &Type,
    value_ty: &Type,
    entry_ty: &Type,
    m_local: hir::LocalId,
    m_ty: &Type,
    i_local: hir::LocalId,
) -> Result<Vec<hir::Stmt>, LowerError> {
    let mut body_stmts = vec![];

    let entry_at_expr = || {
        hir::Expr::new(
            entry_ty.clone(),
            span,
            hir::ExprKind::MapEntryAt {
                map: Box::new(hir::Expr::local(m_ty.clone(), span, m_local)),
                index: Box::new(hir::Expr::local(Type::Int, span, i_local)),
            },
        )
    };

    match &for_node.node.pattern.node {
        Pattern::Ident(name) => {
            let local_id = register_named_local(fc, *name, entry_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: local_id,
                    init: entry_at_expr(),
                },
            });
        }
        Pattern::Wildcard => {}
        Pattern::Tuple(subs) if subs.len() == 2 => {
            let is_enumerate = matches!(&subs[1].node, Pattern::Tuple(inner_subs) if {
                value_ty.tuple_element_types()
                    .is_none_or(|t2_elems| t2_elems.len() != inner_subs.len())
            });

            let types = [key_ty, value_ty];
            if is_enumerate {
                if let Pattern::Ident(name) = &subs[0].node {
                    let local_id = register_named_local(fc, *name, Type::Int);
                    body_stmts.push(hir::Stmt {
                        span,
                        kind: hir::StmtKind::Let {
                            local: local_id,
                            init: hir::Expr::local(Type::Int, span, i_local),
                        },
                    });
                }

                let entry_local =
                    alloc_and_bind(fc, span, &mut body_stmts, entry_ty.clone(), entry_at_expr());

                let Pattern::Tuple(inner_subs) = &subs[1].node else {
                    return Err(LowerError::UnsupportedPattern { span });
                };

                destructure_map_entry_tuple(
                    inner_subs,
                    entry_ty,
                    entry_local,
                    &types,
                    span,
                    fc,
                    &mut body_stmts,
                )?;
            } else {
                let entry_local =
                    alloc_and_bind(fc, span, &mut body_stmts, entry_ty.clone(), entry_at_expr());

                destructure_map_entry_tuple(
                    subs,
                    entry_ty,
                    entry_local,
                    &types,
                    span,
                    fc,
                    &mut body_stmts,
                )?;
            }
        }
        _ => return Err(LowerError::UnsupportedPattern { span }),
    }

    emit_counter_increment(
        &mut body_stmts,
        span,
        i_local,
        &Type::Int,
        hir::Expr::int_lit(span, 1),
        BinaryOp::Add,
    );

    lower_for_user_body(for_node, ctx, fc, &mut body_stmts)?;

    Ok(body_stmts)
}

#[allow(clippy::too_many_arguments)]
fn lower_for_body(
    for_node: &ast::ForNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    item_ty: &Type,
    i_local: hir::LocalId,
    step_local: hir::LocalId,
    inc_op: BinaryOp,
) -> Result<Vec<hir::Stmt>, LowerError> {
    let mut body_stmts = vec![];

    match &for_node.node.pattern.node {
        Pattern::Ident(name) => {
            let local_id = register_named_local(fc, *name, item_ty.clone());
            body_stmts.push(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: local_id,
                    init: hir::Expr::local(item_ty.clone(), span, i_local),
                },
            });
        }
        Pattern::Wildcard => {}
        _ => return Err(LowerError::UnsupportedPattern { span }),
    }

    emit_counter_increment(
        &mut body_stmts,
        span,
        i_local,
        item_ty,
        hir::Expr::local(item_ty.clone(), span, step_local),
        inc_op,
    );

    lower_for_user_body(for_node, ctx, fc, &mut body_stmts)?;

    Ok(body_stmts)
}
