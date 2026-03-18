use std::collections::HashMap;
use std::fmt;

use crate::ast::{
    self, AssignOp, BinaryOp, ExprId, Ident, Lit, Pattern, Stmt, StringPart, Type, TypeVarId,
};
use crate::builtin::Builtin;
use crate::hir;
use crate::span::Span;
use crate::typecheck::{TypeChecker, subst_type};
use internment::Intern;

#[derive(Debug)]
pub enum LowerError {
    UnsupportedStmtKind { span: Span, kind: String },
    UnsupportedExprKind { span: Span, kind: String },
    UnsupportedPattern { span: Span },
    UnsupportedAssign { span: Span, detail: String },
    UnknownLocal { name: Ident, span: Span },
    UnknownFunc { name: Ident, span: Span },
    MissingExprType { span: Span },
    NonDirectCall { span: Span },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedStmtKind { span, kind } => write!(
                f,
                "unsupported statement kind in HIR v1: '{kind}' at offset {}",
                span.start
            ),
            Self::UnsupportedExprKind { span, kind } => write!(
                f,
                "unsupported expression kind in HIR v1: '{kind}' at offset {}",
                span.start
            ),
            Self::UnsupportedPattern { span } => write!(
                f,
                "only simple identifier patterns are supported in HIR v1 (at offset {})",
                span.start
            ),
            Self::UnsupportedAssign { span, detail } => write!(
                f,
                "unsupported assignment in HIR v1: {detail} (at offset {})",
                span.start
            ),
            Self::UnknownLocal { name, span } => write!(
                f,
                "unknown local variable '{name}' at offset {}",
                span.start
            ),
            Self::UnknownFunc { name, span } => {
                write!(f, "unknown function '{name}' at offset {}", span.start)
            }
            Self::MissingExprType { span } => write!(
                f,
                "expression at offset {} has no resolved type (compiler bug)",
                span.start
            ),
            Self::NonDirectCall { span } => write!(
                f,
                "only direct named function calls are supported in HIR v1 (at offset {})",
                span.start
            ),
        }
    }
}

fn mangle_generic_name(name: Ident, type_args: &[Type]) -> Ident {
    let suffix = type_args
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join("$");
    Ident(Intern::new(format!("{name}${suffix}")))
}

struct LowerCtx<'a> {
    tcx: &'a TypeChecker,
    funcs: HashMap<Ident, hir::FuncId>,
    externs: HashMap<Ident, hir::ExternId>,
    type_overrides: Option<&'a HashMap<ExprId, (Span, Type)>>,
    struct_type_ids: HashMap<Ident, u32>,
    enum_type_ids: HashMap<Ident, u32>,
}

struct FuncLower {
    locals: Vec<hir::Local>,
    local_map: HashMap<Ident, hir::LocalId>,
}

pub fn lower_program(
    ast: &ast::Program,
    tcx: &TypeChecker,
    module_list: &[(Vec<String>, Vec<ast::StmtNode>)],
) -> Result<hir::Program, LowerError> {
    let mut struct_type_ids: HashMap<Ident, u32> = HashMap::new();
    let mut next_type_id = 0u32;
    for name in tcx.struct_names() {
        struct_type_ids.entry(name).or_insert_with(|| {
            let id = next_type_id;
            next_type_id += 1;
            id
        });
    }
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            if let ast::Stmt::Struct(s) = &stmt_node.node {
                struct_type_ids.entry(s.node.name).or_insert_with(|| {
                    let id = next_type_id;
                    next_type_id += 1;
                    id
                });
            }
        }
    }

    let mut enum_type_ids: HashMap<Ident, u32> = HashMap::new();
    for name in tcx.enum_names() {
        enum_type_ids.entry(name).or_insert_with(|| {
            let id = next_type_id;
            next_type_id += 1;
            id
        });
    }
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            if let ast::Stmt::Enum(e) = &stmt_node.node {
                enum_type_ids.entry(e.node.name).or_insert_with(|| {
                    let id = next_type_id;
                    next_type_id += 1;
                    id
                });
            }
        }
    }

    let mut ctx = LowerCtx {
        tcx,
        funcs: HashMap::new(),
        externs: HashMap::new(),
        type_overrides: None,
        struct_type_ids,
        enum_type_ids,
    };

    let mut func_nodes: Vec<&ast::FuncNode> = vec![];
    let mut next_func_id = 0u32;
    let mut next_extern_id = 0u32;
    let mut extern_decls: Vec<hir::ExternDecl> = vec![];

    // collect functions from imported modules first so qualified and selective
    // calls can resolve to the correct FuncId
    // modules arrive in DFS post-order (deepest dependencies first), so aliases
    // from re-export stmts (pub import inner { original as renamed }) are
    // processed after the original function is already registered.
    for (_path, stmts) in module_list {
        for stmt_node in stmts {
            match &stmt_node.node {
                Stmt::Func(func_node) => {
                    if func_node.node.type_params.is_empty()
                        && !ctx.funcs.contains_key(&func_node.node.name)
                    {
                        let id = hir::FuncId(next_func_id);
                        next_func_id += 1;
                        ctx.funcs.insert(func_node.node.name, id);
                        func_nodes.push(func_node);
                    }
                }
                Stmt::ExternFunc(extern_node) => {
                    if !ctx.externs.contains_key(&extern_node.node.name) {
                        let id = hir::ExternId(next_extern_id);
                        next_extern_id += 1;
                        ctx.externs.insert(extern_node.node.name, id);
                        extern_decls.push(hir::ExternDecl {
                            id,
                            name: extern_node.node.name,
                            params: extern_node
                                .node
                                .params
                                .iter()
                                .map(|p| p.ty.clone())
                                .collect(),
                            ret: extern_node.node.ret.clone(),
                        });
                    }
                }
                Stmt::Import(import_node) => {
                    // pub import inner { original as renamed }, register the alias
                    // so qualified or selective callers can resolve renamed -> FuncId
                    if let ast::ImportKind::Selective(items) = &import_node.node.kind {
                        for item in items {
                            let Some(alias) = item.alias else {
                                continue;
                            };
                            if let Some(&func_id) = ctx.funcs.get(&item.name) {
                                ctx.funcs.insert(alias, func_id);
                            } else if let Some(&extern_id) = ctx.externs.get(&item.name) {
                                ctx.externs.insert(alias, extern_id);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // collect top-level declarations (first pass)
    for stmt_node in &ast.stmts {
        match &stmt_node.node {
            Stmt::Func(func_node) => {
                // skip generic function templates, HIR is monomorphic
                if !func_node.node.type_params.is_empty() {
                    continue;
                }
                let id = hir::FuncId(next_func_id);
                next_func_id += 1;
                ctx.funcs.insert(func_node.node.name, id);
                func_nodes.push(func_node);
            }
            Stmt::ExternFunc(extern_node) => {
                let id = hir::ExternId(next_extern_id);
                next_extern_id += 1;
                ctx.externs.insert(extern_node.node.name, id);
                extern_decls.push(hir::ExternDecl {
                    id,
                    name: extern_node.node.name,
                    params: extern_node
                        .node
                        .params
                        .iter()
                        .map(|p| p.ty.clone())
                        .collect(),
                    ret: extern_node.node.ret.clone(),
                });
            }
            Stmt::ExternType(_) => {}
            Stmt::Import(import_node) => {
                // for selective imports register aliases so they resolve to the
                // same FuncId/ExternId as the original name
                if let ast::ImportKind::Selective(items) = &import_node.node.kind {
                    for item in items {
                        let Some(alias) = item.alias else {
                            continue;
                        };
                        if let Some(&func_id) = ctx.funcs.get(&item.name) {
                            ctx.funcs.insert(alias, func_id);
                        } else if let Some(&extern_id) = ctx.externs.get(&item.name) {
                            ctx.externs.insert(alias, extern_id);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut spec_registrations: Vec<(Ident, crate::typecheck::SpecializationKey)> = vec![];
    for spec_key in ctx.tcx.specializations().keys() {
        let spec_result = &ctx.tcx.specializations()[spec_key];
        if spec_result.err.is_some() {
            continue;
        }
        if ctx.tcx.generic_template(spec_key.func_name).is_none() {
            continue;
        }
        let mangled = mangle_generic_name(spec_key.func_name, &spec_key.type_args);
        if ctx.funcs.contains_key(&mangled) {
            continue;
        }
        let id = hir::FuncId(next_func_id);
        next_func_id += 1;
        ctx.funcs.insert(mangled, id);
        spec_registrations.push((mangled, spec_key.clone()));
    }

    // lower non-generic function bodies (FuncIds 0..N assigned in passes above)
    let mut funcs = vec![];
    for func_node in func_nodes {
        funcs.push(lower_func(func_node, &ctx)?);
    }

    for (mangled, spec_key) in &spec_registrations {
        let spec_result = &ctx.tcx.specializations()[spec_key];
        let template = ctx
            .tcx
            .generic_template(spec_key.func_name)
            .expect("template must exist as checked during pre-registration");

        let &id = ctx
            .funcs
            .get(mangled)
            .expect("mangled name was just registered");

        let type_params = &template.node.type_params;
        let subst: HashMap<TypeVarId, Type> = type_params
            .iter()
            .zip(spec_key.type_args.iter())
            .map(|(param, arg)| (param.id, arg.clone()))
            .collect();

        let specialized_params: Vec<Type> = template
            .node
            .params
            .iter()
            .map(|p| subst_type(&p.ty, &subst))
            .collect();
        let specialized_ret = subst_type(&template.node.ret, &subst);

        let spec_ctx = LowerCtx {
            tcx: ctx.tcx,
            funcs: ctx.funcs.clone(),
            externs: ctx.externs.clone(),
            type_overrides: Some(&spec_result.body_types),
            struct_type_ids: ctx.struct_type_ids.clone(),
            enum_type_ids: ctx.enum_type_ids.clone(),
        };

        let mut fc = FuncLower {
            locals: vec![],
            local_map: HashMap::new(),
        };

        for (param, ty) in template.node.params.iter().zip(specialized_params.iter()) {
            let local_id = hir::LocalId(fc.locals.len() as u32);
            fc.locals.push(hir::Local {
                name: Some(param.name),
                ty: ty.clone(),
            });
            fc.local_map.insert(param.name, local_id);
        }
        let params_len = fc.locals.len() as u32;

        let body = lower_block(
            &template.node.body,
            &spec_ctx,
            &mut fc,
            true,
            &specialized_ret,
        )?;

        funcs.push(hir::Func {
            id,
            name: *mangled,
            locals: fc.locals,
            params_len,
            ret: specialized_ret,
            body,
            span: template.span,
        });
    }

    Ok(hir::Program {
        funcs,
        externs: extern_decls,
    })
}

fn lower_func(func_node: &ast::FuncNode, ctx: &LowerCtx) -> Result<hir::Func, LowerError> {
    let func = &func_node.node;
    let id = *ctx
        .funcs
        .get(&func.name)
        .expect("func id must exist after pass 1");

    let mut fc = FuncLower {
        locals: vec![],
        local_map: HashMap::new(),
    };

    // register parameters as locals first
    for param in &func.params {
        let local_id = hir::LocalId(fc.locals.len() as u32);
        fc.locals.push(hir::Local {
            name: Some(param.name),
            ty: param.ty.clone(),
        });
        fc.local_map.insert(param.name, local_id);
    }
    let params_len = fc.locals.len() as u32;

    let body = lower_block(&func.body, ctx, &mut fc, true, &func.ret)?;

    Ok(hir::Func {
        id,
        name: func.name,
        locals: fc.locals,
        params_len,
        ret: func.ret.clone(),
        body,
        span: func_node.span,
    })
}

fn lower_block(
    block: &ast::BlockNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
) -> Result<hir::Block, LowerError> {
    let saved_local_map = if !is_func_body {
        Some(fc.local_map.clone())
    } else {
        None
    };

    let mut stmts = vec![];

    for stmt_node in &block.node.stmts {
        if let Some(hir_stmt) = lower_stmt(stmt_node, ctx, fc)? {
            stmts.push(hir_stmt);
        }
    }

    if let Some(tail_expr) = &block.node.tail {
        let span = tail_expr.span;
        match &tail_expr.node.kind {
            ast::ExprKind::If(if_node) => {
                stmts.push(lower_if(if_node, span, ctx, fc, is_func_body, ret_ty)?);
            }
            ast::ExprKind::Assign(assign_node) => {
                stmts.push(lower_assign(assign_node, span, ctx, fc)?);
            }
            ast::ExprKind::Match(match_node) => {
                stmts.push(lower_match_stmts(
                    match_node,
                    span,
                    ctx,
                    fc,
                    is_func_body,
                    ret_ty,
                )?);
            }
            _ => {
                let hir_expr = lower_expr(tail_expr, ctx, fc)?;
                let kind = if is_func_body && !ret_ty.is_void() {
                    hir::StmtKind::Return(Some(hir_expr))
                } else {
                    hir::StmtKind::Expr(hir_expr)
                };
                stmts.push(hir::Stmt { span, kind });
            }
        }
    }

    if let Some(saved) = saved_local_map {
        fc.local_map = saved;
    }

    Ok(hir::Block { stmts })
}

fn lower_stmt(
    stmt_node: &ast::StmtNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
) -> Result<Option<hir::Stmt>, LowerError> {
    let span = stmt_node.span;

    match &stmt_node.node {
        Stmt::Binding(binding_node) => {
            let binding = &binding_node.node;

            let name = match &binding.pattern.node {
                Pattern::Ident(name) => *name,
                _ => {
                    return Err(LowerError::UnsupportedPattern {
                        span: binding.pattern.span,
                    });
                }
            };

            // after constraint solving, this is always a fully-resolved monomorphic type.
            let ty = {
                let (_, ty) = ctx
                    .tcx
                    .get_type(binding.value.node.id)
                    .ok_or(LowerError::MissingExprType { span })?;
                ty.clone()
            };

            let local_id = hir::LocalId(fc.locals.len() as u32);

            // lower the init before inserting into local_map to prevent `let x = x` from
            // accidentally resolving to the new local.
            let init = lower_expr(&binding.value, ctx, fc)?;

            fc.locals.push(hir::Local {
                name: Some(name),
                ty,
            });
            fc.local_map.insert(name, local_id);

            Ok(Some(hir::Stmt {
                span,
                kind: hir::StmtKind::Let {
                    local: local_id,
                    init,
                },
            }))
        }

        Stmt::Expr(expr_node) => match &expr_node.node.kind {
            ast::ExprKind::If(if_node) => {
                Ok(Some(lower_if(if_node, span, ctx, fc, false, &Type::Void)?))
            }
            ast::ExprKind::Assign(assign_node) => {
                Ok(Some(lower_assign(assign_node, span, ctx, fc)?))
            }
            ast::ExprKind::Match(match_node) => Ok(Some(lower_match_stmts(
                match_node,
                span,
                ctx,
                fc,
                false,
                &Type::Void,
            )?)),
            _ => {
                let hir_expr = lower_expr(expr_node, ctx, fc)?;
                Ok(Some(hir::Stmt {
                    span,
                    kind: hir::StmtKind::Expr(hir_expr),
                }))
            }
        },

        Stmt::Return(return_node) => {
            let value = match &return_node.node.value {
                Some(expr) => Some(lower_expr(expr, ctx, fc)?),
                None => None,
            };
            Ok(Some(hir::Stmt {
                span,
                kind: hir::StmtKind::Return(value),
            }))
        }

        Stmt::While(while_node) => {
            let cond = lower_expr(&while_node.node.cond, ctx, fc)?;
            let body = lower_block(&while_node.node.body, ctx, fc, false, &Type::Void)?;
            Ok(Some(hir::Stmt {
                span,
                kind: hir::StmtKind::While { cond, body },
            }))
        }

        Stmt::Break => Ok(Some(hir::Stmt {
            span,
            kind: hir::StmtKind::Break,
        })),

        Stmt::Continue => Ok(Some(hir::Stmt {
            span,
            kind: hir::StmtKind::Continue,
        })),

        Stmt::For(_) => Err(LowerError::UnsupportedStmtKind {
            span,
            kind: "for loop".to_string(),
        }),

        Stmt::ExternFunc(_) => Ok(None),
        Stmt::ExternType(_) => Ok(None),
        Stmt::Import(_) => Ok(None),

        Stmt::Func(_) => Err(LowerError::UnsupportedStmtKind {
            span,
            kind: "nested function".to_string(),
        }),

        Stmt::Struct(_) => Ok(None),

        Stmt::Enum(_) => Ok(None),
    }
}

fn lower_if(
    if_node: &ast::IfNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
) -> Result<hir::Stmt, LowerError> {
    let cond = lower_expr(&if_node.node.cond, ctx, fc)?;
    let then_block = lower_block(&if_node.node.then_block, ctx, fc, is_func_body, ret_ty)?;
    let else_block = match &if_node.node.else_block {
        Some(b) => Some(lower_block(b, ctx, fc, is_func_body, ret_ty)?),
        None => None,
    };
    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::If {
            cond,
            then_block,
            else_block,
        },
    })
}

fn lower_assign(
    assign_node: &ast::AssignNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
) -> Result<hir::Stmt, LowerError> {
    if assign_node.node.op != AssignOp::Assign {
        return Err(LowerError::UnsupportedAssign {
            span,
            detail: format!(
                "compound assignment '{}' is not supported in HIR v1",
                assign_node.node.op
            ),
        });
    }

    match &assign_node.node.target.node.kind {
        ast::ExprKind::Ident(name) => {
            let local_id = *fc
                .local_map
                .get(name)
                .ok_or(LowerError::UnknownLocal { name: *name, span })?;
            let value = lower_expr(&assign_node.node.value, ctx, fc)?;
            Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::Assign {
                    local: local_id,
                    value,
                },
            })
        }

        ast::ExprKind::Field(field_access) => {
            let field_name = field_access.node.field;

            let root_name = match &field_access.node.target.node.kind {
                ast::ExprKind::Ident(n) => *n,
                _ => {
                    return Err(LowerError::UnsupportedAssign {
                        span,
                        detail: "nested field assignment is not yet supported".to_string(),
                    });
                }
            };

            let local_id = *fc
                .local_map
                .get(&root_name)
                .ok_or(LowerError::UnknownLocal {
                    name: root_name,
                    span,
                })?;

            let target_ty = {
                let (_, ty) = ctx
                    .type_overrides
                    .and_then(|overrides| overrides.get(&field_access.node.target.node.id))
                    .or_else(|| ctx.tcx.get_type(field_access.node.target.node.id))
                    .ok_or(LowerError::MissingExprType { span })?;
                ty.clone()
            };

            let field_index = match &target_ty {
                Type::Struct { name, .. } => ctx
                    .tcx
                    .struct_field_index(*name, field_name)
                    .ok_or_else(|| LowerError::UnsupportedAssign {
                        span,
                        detail: format!("unknown field '{field_name}' on struct '{name}'"),
                    })? as u16,
                other => {
                    return Err(LowerError::UnsupportedAssign {
                        span,
                        detail: format!("field assignment on unsupported type '{other}'"),
                    });
                }
            };

            let value = lower_expr(&assign_node.node.value, ctx, fc)?;
            Ok(hir::Stmt {
                span,
                kind: hir::StmtKind::SetField {
                    object: local_id,
                    field_index,
                    value,
                },
            })
        }

        _ => Err(LowerError::UnsupportedAssign {
            span,
            detail: "assignment target must be a plain local variable or field access".to_string(),
        }),
    }
}

fn lower_arm_body(
    body_expr: &ast::ExprNode,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
) -> Result<hir::Block, LowerError> {
    if let ast::ExprKind::Block(block_node) = &body_expr.node.kind {
        lower_block(block_node, ctx, fc, is_func_body, ret_ty)
    } else {
        let span = body_expr.span;
        let hir_expr = lower_expr(body_expr, ctx, fc)?;
        let kind = if is_func_body && !ret_ty.is_void() {
            hir::StmtKind::Return(Some(hir_expr))
        } else {
            hir::StmtKind::Expr(hir_expr)
        };
        Ok(hir::Block {
            stmts: vec![hir::Stmt { span, kind }],
        })
    }
}

fn lower_match_stmts(
    match_node: &ast::MatchNode,
    span: Span,
    ctx: &LowerCtx,
    fc: &mut FuncLower,
    is_func_body: bool,
    ret_ty: &Type,
) -> Result<hir::Stmt, LowerError> {
    let scrutinee_expr = lower_expr(&match_node.node.scrutinee, ctx, fc)?;
    let scrutinee_ty = scrutinee_expr.ty.clone();

    let scrutinee_local = hir::LocalId(fc.locals.len() as u32);
    fc.locals.push(hir::Local {
        name: None,
        ty: scrutinee_ty.clone(),
    });

    let (enum_name, type_args) = match &scrutinee_ty {
        Type::Enum { name, type_args } => (*name, type_args.clone()),
        other => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: format!("match on non-enum type '{other}'"),
            });
        }
    };

    let mut arms: Vec<hir::MatchArm> = vec![];
    let mut else_body: Option<hir::MatchElse> = None;

    for arm in &match_node.node.arms {
        let saved_local_map = fc.local_map.clone();

        match &arm.node.pattern.node {
            Pattern::Wildcard => {
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                else_body = Some(hir::MatchElse {
                    binding: None,
                    body,
                });
            }

            Pattern::Ident(name) => {
                let binding_local = hir::LocalId(fc.locals.len() as u32);
                fc.locals.push(hir::Local {
                    name: Some(*name),
                    ty: scrutinee_ty.clone(),
                });
                fc.local_map.insert(*name, binding_local);
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                else_body = Some(hir::MatchElse {
                    binding: Some(binding_local),
                    body,
                });
            }

            Pattern::EnumUnit {
                qualifier: _,
                variant,
            } => {
                let variant_idx =
                    ctx.tcx
                        .enum_variant_index(enum_name, *variant)
                        .ok_or_else(|| LowerError::UnsupportedExprKind {
                            span,
                            kind: format!("unknown variant '{variant}' on enum '{enum_name}'"),
                        })?;
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings: vec![],
                    body,
                });
            }

            Pattern::EnumTuple {
                qualifier: _,
                variant,
                fields: subpatterns,
            } => {
                let variant_idx =
                    ctx.tcx
                        .enum_variant_index(enum_name, *variant)
                        .ok_or_else(|| LowerError::UnsupportedExprKind {
                            span,
                            kind: format!("unknown variant '{variant}' on enum '{enum_name}'"),
                        })?;
                let field_types = ctx
                    .tcx
                    .enum_variant_field_types(enum_name, *variant, &type_args)
                    .unwrap_or_default();

                let mut bindings = vec![];
                for (field_idx, subpat) in subpatterns.iter().enumerate() {
                    if let Pattern::Ident(binding_name) = &subpat.node {
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        let local = hir::LocalId(fc.locals.len() as u32);
                        fc.locals.push(hir::Local {
                            name: Some(*binding_name),
                            ty: field_ty,
                        });
                        fc.local_map.insert(*binding_name, local);
                        bindings.push(hir::MatchBinding {
                            field_index: field_idx as u16,
                            local,
                        });
                    }
                }
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings,
                    body,
                });
            }

            Pattern::EnumStruct {
                qualifier: _,
                variant,
                fields: field_patterns,
            } => {
                let variant_idx =
                    ctx.tcx
                        .enum_variant_index(enum_name, *variant)
                        .ok_or_else(|| LowerError::UnsupportedExprKind {
                            span,
                            kind: format!("unknown variant '{variant}' on enum '{enum_name}'"),
                        })?;
                let field_names = ctx
                    .tcx
                    .enum_variant_field_names(enum_name, *variant)
                    .unwrap_or_default();
                let field_types = ctx
                    .tcx
                    .enum_variant_field_types(enum_name, *variant, &type_args)
                    .unwrap_or_default();

                let mut bindings = vec![];
                for (pat_field_name, subpat) in field_patterns {
                    if let Pattern::Ident(binding_name) = &subpat.node {
                        let field_idx = field_names
                            .iter()
                            .position(|n| n == pat_field_name)
                            .unwrap_or(0);
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(Type::Void);
                        let local = hir::LocalId(fc.locals.len() as u32);
                        fc.locals.push(hir::Local {
                            name: Some(*binding_name),
                            ty: field_ty,
                        });
                        fc.local_map.insert(*binding_name, local);
                        bindings.push(hir::MatchBinding {
                            field_index: field_idx as u16,
                            local,
                        });
                    }
                }
                let body = lower_arm_body(&arm.node.body, ctx, fc, is_func_body, ret_ty)?;
                arms.push(hir::MatchArm {
                    variant: variant_idx,
                    bindings,
                    body,
                });
            }

            other => {
                return Err(LowerError::UnsupportedExprKind {
                    span,
                    kind: format!(
                        "unsupported match pattern '{}'",
                        format!("{other:?}").split('(').next().unwrap_or("?")
                    ),
                });
            }
        }

        fc.local_map = saved_local_map;
    }

    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::Match {
            scrutinee_init: Box::new(scrutinee_expr),
            scrutinee: scrutinee_local,
            arms,
            else_body,
        },
    })
}

fn lower_string_interp(
    parts: &[StringPart],
    span: Span,
    ty: &Type,
    ctx: &LowerCtx,
    fc: &FuncLower,
) -> Result<hir::ExprKind, LowerError> {
    let mut hir_parts: Vec<hir::Expr> = vec![];

    for part in parts {
        let expr = match part {
            StringPart::Text(s) => hir::Expr {
                ty: Type::String,
                span,
                kind: hir::ExprKind::String(s.clone()),
            },
            StringPart::Expr(e) => lower_expr(e, ctx, fc)?,
        };
        hir_parts.push(expr);
    }

    match hir_parts.len() {
        0 => Ok(hir::ExprKind::String(String::new())),
        1 => Ok(hir_parts.remove(0).kind),
        _ => {
            let mut iter = hir_parts.into_iter();
            let first = iter.next().unwrap();
            let folded = iter.fold(first, |acc, rhs| hir::Expr {
                ty: ty.clone(),
                span,
                kind: hir::ExprKind::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(acc),
                    rhs: Box::new(rhs),
                },
            });
            Ok(folded.kind)
        }
    }
}

fn lower_expr(
    ast_expr: &ast::ExprNode,
    ctx: &LowerCtx,
    fc: &FuncLower,
) -> Result<hir::Expr, LowerError> {
    let span = ast_expr.span;
    let ty = {
        let (_, ty) = ctx
            .type_overrides
            .and_then(|overrides| overrides.get(&ast_expr.node.id))
            .or_else(|| ctx.tcx.get_type(ast_expr.node.id))
            .ok_or(LowerError::MissingExprType { span })?;
        ty.clone()
    };

    let kind = match &ast_expr.node.kind {
        ast::ExprKind::Ident(name) => {
            let local_id = *fc
                .local_map
                .get(name)
                .ok_or(LowerError::UnknownLocal { name: *name, span })?;
            hir::ExprKind::Local(local_id)
        }

        ast::ExprKind::Lit(lit) => match lit {
            Lit::Int(v) => hir::ExprKind::Int(*v),
            Lit::Float(v) => hir::ExprKind::Float(*v),
            Lit::Bool(v) => hir::ExprKind::Bool(*v),
            Lit::String(v) => hir::ExprKind::String(v.clone()),
            Lit::Nil => hir::ExprKind::Nil,
        },

        ast::ExprKind::Unary(u) => {
            let inner = lower_expr(&u.node.expr, ctx, fc)?;
            hir::ExprKind::Unary {
                op: u.node.op,
                expr: Box::new(inner),
            }
        }

        ast::ExprKind::Binary(b) => {
            // coalesce is sugar that must be desugared before HIR
            if b.node.op == BinaryOp::Coalesce {
                return Err(LowerError::UnsupportedExprKind {
                    span,
                    kind: "coalesce operator (??)".to_string(),
                });
            }
            let lhs = lower_expr(&b.node.left, ctx, fc)?;
            let rhs = lower_expr(&b.node.right, ctx, fc)?;
            hir::ExprKind::Binary {
                op: b.node.op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }
        }

        ast::ExprKind::StringInterp(parts) => lower_string_interp(parts, span, &ty, ctx, fc)?,

        ast::ExprKind::Call(c) => {
            if let Type::Enum {
                name: enum_name, ..
            } = &ty
            {
                if let ast::ExprKind::Field(field) = &c.node.func.node.kind {
                    if let ast::ExprKind::Ident(_) = &field.node.target.node.kind {
                        let enum_name = *enum_name;
                        let variant_name = field.node.field;
                        let type_id = *ctx.enum_type_ids.get(&enum_name).ok_or_else(|| {
                            LowerError::UnsupportedExprKind {
                                span,
                                kind: format!("unknown enum '{enum_name}'"),
                            }
                        })?;
                        let variant = ctx
                            .tcx
                            .enum_variant_index(enum_name, variant_name)
                            .ok_or_else(|| LowerError::UnsupportedExprKind {
                                span,
                                kind: format!(
                                    "unknown variant '{variant_name}' on enum '{enum_name}'"
                                ),
                            })?;
                        let fields = c
                            .node
                            .args
                            .iter()
                            .map(|arg| lower_expr(arg, ctx, fc))
                            .collect::<Result<Vec<_>, _>>()?;
                        return Ok(hir::Expr {
                            ty,
                            span,
                            kind: hir::ExprKind::EnumLiteral {
                                type_id,
                                variant,
                                fields,
                            },
                        });
                    }
                }
            }

            let callee_name = match &c.node.func.node.kind {
                ast::ExprKind::Ident(name) => *name,
                ast::ExprKind::Field(field) => {
                    // module qualified call module.func(args), resolve to the function name
                    if let ast::ExprKind::Ident(module_name) = &field.node.target.node.kind {
                        if ctx.tcx.is_module_name(*module_name) {
                            field.node.field
                        } else {
                            return Err(LowerError::NonDirectCall { span });
                        }
                    } else if let ast::ExprKind::Field(inner_field) = &field.node.target.node.kind {
                        // nested facade.submodule.func(args), two levels of field access
                        if let ast::ExprKind::Ident(outer_module) =
                            &inner_field.node.target.node.kind
                        {
                            if ctx.tcx.is_module_name(*outer_module) {
                                field.node.field
                            } else {
                                return Err(LowerError::NonDirectCall { span });
                            }
                        } else {
                            return Err(LowerError::NonDirectCall { span });
                        }
                    } else {
                        return Err(LowerError::NonDirectCall { span });
                    }
                }
                _ => return Err(LowerError::NonDirectCall { span }),
            };
            let args = c
                .node
                .args
                .iter()
                .map(|arg| lower_expr(arg, ctx, fc))
                .collect::<Result<Vec<_>, _>>()?;

            // builtins take precedence over user functions and externs of the same name
            if let Some(builtin) = Builtin::from_name(callee_name.0.as_ref()) {
                hir::ExprKind::CallBuiltin { builtin, args }
            } else if let Some(&func_id) = ctx.funcs.get(&callee_name) {
                hir::ExprKind::Call {
                    func: func_id,
                    args,
                }
            } else if let Some(&extern_id) = ctx.externs.get(&callee_name) {
                hir::ExprKind::CallExtern { extern_id, args }
            } else if let Some((func_name, type_args)) = ctx.tcx.call_type_args(c.node.func.node.id)
            {
                let mangled = mangle_generic_name(*func_name, type_args);
                let &func_id = ctx.funcs.get(&mangled).ok_or(LowerError::UnknownFunc {
                    name: callee_name,
                    span,
                })?;
                hir::ExprKind::Call {
                    func: func_id,
                    args,
                }
            } else {
                return Err(LowerError::UnknownFunc {
                    name: callee_name,
                    span,
                });
            }
        }

        ast::ExprKind::StructLiteral(lit) => {
            // enum struct variant Event.Move { dx: 5, dy: 10 }
            if let Some(enum_name) = lit.node.qualifier {
                let variant_name = lit.node.name;
                let type_id = *ctx.enum_type_ids.get(&enum_name).ok_or_else(|| {
                    LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown enum '{enum_name}'"),
                    }
                })?;
                let variant = ctx
                    .tcx
                    .enum_variant_index(enum_name, variant_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown variant '{variant_name}' on enum '{enum_name}'"),
                    })?;
                let field_names = ctx
                    .tcx
                    .enum_variant_field_names(enum_name, variant_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!(
                            "variant '{variant_name}' on enum '{enum_name}' is not a struct variant"
                        ),
                    })?;
                let provided: HashMap<Ident, &ast::ExprNode> = lit
                    .node
                    .fields
                    .iter()
                    .map(|(name, expr)| (*name, expr))
                    .collect();
                let fields = field_names
                    .iter()
                    .map(|name| {
                        let expr = provided
                            .get(name)
                            .expect("typechecker ensures all declared fields are provided");
                        lower_expr(expr, ctx, fc)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(hir::Expr {
                    ty,
                    span,
                    kind: hir::ExprKind::EnumLiteral {
                        type_id,
                        variant,
                        fields,
                    },
                });
            }

            let struct_name = lit.node.name;
            let type_id = *ctx.struct_type_ids.get(&struct_name).ok_or_else(|| {
                LowerError::UnsupportedExprKind {
                    span,
                    kind: format!("unknown struct '{struct_name}'"),
                }
            })?;

            let field_names = ctx.tcx.struct_field_names(struct_name).ok_or_else(|| {
                LowerError::UnsupportedExprKind {
                    span,
                    kind: format!("unknown struct '{struct_name}'"),
                }
            })?;

            // build a lookup from field name -> provided expr
            let provided: HashMap<Ident, &ast::ExprNode> = lit
                .node
                .fields
                .iter()
                .map(|(name, expr)| (*name, expr))
                .collect();

            // lower fields in declaration order
            let fields = field_names
                .iter()
                .map(|name| {
                    let expr = provided
                        .get(name)
                        .expect("typechecker ensures all declared fields are provided");
                    lower_expr(expr, ctx, fc)
                })
                .collect::<Result<Vec<_>, _>>()?;

            hir::ExprKind::StructLiteral { type_id, fields }
        }

        ast::ExprKind::Tuple(elements) => {
            let lowered = elements
                .iter()
                .map(|el| lower_expr(el, ctx, fc))
                .collect::<Result<Vec<_>, _>>()?;
            hir::ExprKind::TupleLiteral { elements: lowered }
        }

        ast::ExprKind::Field(field_access) => {
            let target = &field_access.node.target;
            let field_name = field_access.node.field;

            // unit enum variant Color.Red (result type is Enum)
            if let Type::Enum {
                name: enum_name, ..
            } = &ty
            {
                let enum_name = *enum_name;
                let type_id = *ctx.enum_type_ids.get(&enum_name).ok_or_else(|| {
                    LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown enum '{enum_name}'"),
                    }
                })?;
                let variant = ctx
                    .tcx
                    .enum_variant_index(enum_name, field_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown variant '{field_name}' on enum '{enum_name}'"),
                    })?;
                return Ok(hir::Expr {
                    ty,
                    span,
                    kind: hir::ExprKind::EnumLiteral {
                        type_id,
                        variant,
                        fields: vec![],
                    },
                });
            }

            let target_ty = {
                let (_, ty) = ctx
                    .type_overrides
                    .and_then(|overrides| overrides.get(&target.node.id))
                    .or_else(|| ctx.tcx.get_type(target.node.id))
                    .ok_or(LowerError::MissingExprType { span })?;
                ty.clone()
            };

            let index = match &target_ty {
                Type::Struct { name, .. } => ctx
                    .tcx
                    .struct_field_index(*name, field_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown field '{field_name}' on struct '{name}'"),
                    })? as u16,
                Type::NamedTuple(fields) => fields
                    .iter()
                    .position(|(label, _)| *label == field_name)
                    .ok_or_else(|| LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("unknown field '{field_name}' on named tuple"),
                    })? as u16,
                other => {
                    return Err(LowerError::UnsupportedExprKind {
                        span,
                        kind: format!("field access on unsupported type '{other}'"),
                    });
                }
            };

            let object = lower_expr(target, ctx, fc)?;
            hir::ExprKind::FieldGet {
                object: Box::new(object),
                index,
            }
        }

        ast::ExprKind::TupleIndex(tuple_idx) => {
            let tuple = lower_expr(&tuple_idx.node.target, ctx, fc)?;
            hir::ExprKind::TupleIndex {
                tuple: Box::new(tuple),
                index: tuple_idx.node.index as u16,
            }
        }

        other => {
            return Err(LowerError::UnsupportedExprKind {
                span,
                kind: format!("{other:?}")
                    .split('(')
                    .next()
                    .unwrap_or("Unknown")
                    .to_string(),
            });
        }
    };

    Ok(hir::Expr { ty, span, kind })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Type;
    use crate::builtin::Builtin;
    use crate::hir;
    use crate::hir::{ExprKind, LocalId, StmtKind};
    use crate::test_helpers::TestCtx;

    fn lower_ok(source: &str) -> hir::Program {
        TestCtx::lower_ok(source)
    }

    fn lower_err(source: &str) -> LowerError {
        TestCtx::lower_err(source)
    }

    fn find_main(prog: &hir::Program) -> &hir::Func {
        prog.funcs
            .iter()
            .find(|f| f.name.to_string() == "main")
            .expect("main function not found")
    }

    #[test]
    fn empty_main() {
        let prog = lower_ok("fn main() {}");
        let main = find_main(&prog);
        assert_eq!(main.params_len, 0);
        assert_eq!(main.locals.len(), 0);
        assert_eq!(main.body.stmts.len(), 0);
        assert_eq!(main.ret, Type::Void);
    }

    #[test]
    fn let_binding_int() {
        let prog = lower_ok("fn main() { let x = 42; }");
        let main = find_main(&prog);
        assert_eq!(main.locals.len(), 1);
        assert_eq!(main.locals[0].name.unwrap().to_string(), "x");
        assert_eq!(main.locals[0].ty, Type::Int);
        assert_eq!(main.body.stmts.len(), 1);
        let StmtKind::Let {
            local: LocalId(0),
            init,
        } = &main.body.stmts[0].kind
        else {
            panic!("expected Let stmt")
        };
        assert!(matches!(init.kind, ExprKind::Int(42)));
    }

    #[test]
    fn let_binding_binary() {
        let prog = lower_ok("fn main() { let x = 1 + 2; }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let stmt")
        };
        assert!(matches!(
            init.kind,
            ExprKind::Binary {
                op: crate::ast::BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn let_binding_unary() {
        let prog = lower_ok("fn main() { let x = -1; }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let stmt")
        };
        assert!(matches!(
            init.kind,
            ExprKind::Unary {
                op: crate::ast::UnaryOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn explicit_return_with_value() {
        let prog = lower_ok("fn main() -> int { return 1; }");
        let main = find_main(&prog);
        assert!(matches!(main.body.stmts[0].kind, StmtKind::Return(Some(_))));
    }

    #[test]
    fn explicit_return_void() {
        let prog = lower_ok("fn main() { return; }");
        let main = find_main(&prog);
        assert!(matches!(main.body.stmts[0].kind, StmtKind::Return(None)));
    }

    #[test]
    fn implicit_return_from_if_expr() {
        let prog = lower_ok("fn foo() -> int { if true { 1 } else { 2 } }");
        let foo = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "foo")
            .unwrap();
        // The if is the tail of the function body → StmtKind::If
        let StmtKind::If {
            then_block,
            else_block,
            ..
        } = &foo.body.stmts[0].kind
        else {
            panic!("expected If stmt")
        };
        // Both branches must end with Return(Some(...))
        assert!(matches!(
            then_block.stmts[0].kind,
            StmtKind::Return(Some(_))
        ));
        let else_stmts = &else_block.as_ref().unwrap().stmts;
        assert!(matches!(else_stmts[0].kind, StmtKind::Return(Some(_))));
    }

    #[test]
    fn implicit_return_from_nested_if_expr() {
        let prog = lower_ok("fn foo() -> int { if true { if false { 1 } else { 2 } } else { 3 } }");
        let foo = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "foo")
            .unwrap();
        let StmtKind::If {
            then_block,
            else_block,
            ..
        } = &foo.body.stmts[0].kind
        else {
            panic!("expected outer If")
        };
        // outer else branch must return
        let else_stmts = &else_block.as_ref().unwrap().stmts;
        assert!(matches!(else_stmts[0].kind, StmtKind::Return(Some(_))));
        // inner if in then_branch, both inner branches must return
        let StmtKind::If {
            then_block: inner_then,
            else_block: inner_else,
            ..
        } = &then_block.stmts[0].kind
        else {
            panic!("expected inner If")
        };
        assert!(matches!(
            inner_then.stmts[0].kind,
            StmtKind::Return(Some(_))
        ));
        let inner_else_stmts = &inner_else.as_ref().unwrap().stmts;
        assert!(matches!(
            inner_else_stmts[0].kind,
            StmtKind::Return(Some(_))
        ));
    }

    #[test]
    fn if_without_else() {
        let prog = lower_ok("fn main() { let x = true; if x {} }");
        let main = find_main(&prog);
        // stmts[0] = Let(x), stmts[1] = If (promoted from tail)
        assert_eq!(main.body.stmts.len(), 2);
        assert!(matches!(
            main.body.stmts[1].kind,
            StmtKind::If {
                else_block: None,
                ..
            }
        ));
    }

    #[test]
    fn if_with_else() {
        let prog = lower_ok("fn main() { let x = true; if x {} else {} }");
        let main = find_main(&prog);
        assert_eq!(main.body.stmts.len(), 2);
        assert!(matches!(
            main.body.stmts[1].kind,
            StmtKind::If {
                else_block: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn if_cond_uses_local() {
        let prog = lower_ok("fn main() { let x = true; if x {} }");
        let main = find_main(&prog);
        let StmtKind::If { cond, .. } = &main.body.stmts[1].kind else {
            panic!("expected If stmt")
        };
        assert!(matches!(cond.kind, ExprKind::Local(LocalId(0))));
    }

    #[test]
    fn while_with_break() {
        let prog = lower_ok("fn main() { while true { break; } }");
        let main = find_main(&prog);
        let StmtKind::While { body, .. } = &main.body.stmts[0].kind else {
            panic!("expected While stmt")
        };
        assert!(matches!(body.stmts[0].kind, StmtKind::Break));
    }

    #[test]
    fn while_with_continue() {
        let prog = lower_ok("fn main() { while true { continue; } }");
        let main = find_main(&prog);
        let StmtKind::While { body, .. } = &main.body.stmts[0].kind else {
            panic!("expected While stmt")
        };
        assert!(matches!(body.stmts[0].kind, StmtKind::Continue));
    }

    #[test]
    fn while_cond_is_bool_literal() {
        let prog = lower_ok("fn main() { while true { break; } }");
        let main = find_main(&prog);
        let StmtKind::While { cond, .. } = &main.body.stmts[0].kind else {
            panic!("expected While stmt")
        };
        assert!(matches!(cond.kind, ExprKind::Bool(true)));
    }

    #[test]
    fn direct_function_call() {
        let prog = lower_ok("fn foo() {} fn main() { foo(); }");
        let foo = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "foo")
            .expect("foo");
        let main = find_main(&prog);
        let foo_id = foo.id;
        let StmtKind::Expr(call_expr) = &main.body.stmts[0].kind else {
            panic!("expected Expr stmt")
        };
        assert!(matches!(call_expr.kind, ExprKind::Call { func, .. } if func == foo_id));
    }

    #[test]
    fn function_call_args_are_lowered() {
        let prog =
            lower_ok("fn add(a: int, b: int) -> int { return a + b; } fn main() { add(1, 2); }");
        let add = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "add")
            .expect("add");
        let main = find_main(&prog);

        assert_eq!(add.params_len, 2);
        assert_eq!(add.locals[0].name.unwrap().to_string(), "a");
        assert_eq!(add.locals[1].name.unwrap().to_string(), "b");

        let StmtKind::Expr(call_expr) = &main.body.stmts[0].kind else {
            panic!("expected Expr stmt")
        };
        let ExprKind::Call { args, .. } = &call_expr.kind else {
            panic!("expected Call expr")
        };
        assert_eq!(args.len(), 2);
        assert!(matches!(args[0].kind, ExprKind::Int(1)));
        assert!(matches!(args[1].kind, ExprKind::Int(2)));
    }

    #[test]
    fn call_builtin_println() {
        let prog = lower_ok(r#"fn main() { println("hi"); }"#);
        let main = find_main(&prog);
        let StmtKind::Expr(expr) = &main.body.stmts[0].kind else {
            panic!("expected Expr stmt")
        };
        assert!(matches!(
            expr.kind,
            ExprKind::CallBuiltin {
                builtin: Builtin::Println,
                ..
            }
        ));
    }

    #[test]
    fn call_builtin_assert() {
        let prog = lower_ok("fn main() { assert(true); }");
        let main = find_main(&prog);
        let StmtKind::Expr(expr) = &main.body.stmts[0].kind else {
            panic!("expected Expr stmt")
        };
        assert!(matches!(
            expr.kind,
            ExprKind::CallBuiltin {
                builtin: Builtin::Assert,
                ..
            }
        ));
    }

    #[test]
    fn call_builtin_assert_msg() {
        let prog = lower_ok(r#"fn main() { assert_msg(true, "ok"); }"#);
        let main = find_main(&prog);
        let StmtKind::Expr(expr) = &main.body.stmts[0].kind else {
            panic!("expected Expr stmt")
        };
        assert!(matches!(
            expr.kind,
            ExprKind::CallBuiltin {
                builtin: Builtin::AssertMsg,
                ..
            }
        ));
    }

    #[test]
    fn variable_reference_resolves_to_local_id() {
        let prog = lower_ok("fn main() { let x = 1; let y = x; }");
        let main = find_main(&prog);
        // x is LocalId(0), y is LocalId(1)
        let StmtKind::Let { init, .. } = &main.body.stmts[1].kind else {
            panic!("expected Let stmt")
        };
        assert!(matches!(init.kind, ExprKind::Local(LocalId(0))));
    }

    #[test]
    fn assignment_emits_assign_stmt() {
        let prog = lower_ok("fn main() { var x = 1; x = 2; }");
        let main = find_main(&prog);
        assert!(matches!(
            main.body.stmts[1].kind,
            StmtKind::Assign {
                local: LocalId(0),
                ..
            }
        ));
    }

    #[test]
    fn multiple_functions_have_distinct_ids() {
        let prog = lower_ok("fn foo() {} fn bar() {} fn main() {}");
        assert_eq!(prog.funcs.len(), 3);
        let foo = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "foo")
            .expect("foo");
        let bar = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "bar")
            .expect("bar");
        let main = find_main(&prog);
        assert_ne!(foo.id, bar.id);
        assert_ne!(bar.id, main.id);
        assert_ne!(foo.id, main.id);
    }

    #[test]
    fn cross_function_call_resolves_id() {
        let prog = lower_ok("fn foo() {} fn bar() { foo(); } fn main() { bar(); }");
        let foo = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "foo")
            .expect("foo");
        let bar = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "bar")
            .expect("bar");
        let main = find_main(&prog);
        let foo_id = foo.id;
        let bar_id = bar.id;

        let StmtKind::Expr(bar_call) = &bar.body.stmts[0].kind else {
            panic!()
        };
        assert!(matches!(bar_call.kind, ExprKind::Call { func, .. } if func == foo_id));

        let StmtKind::Expr(main_call) = &main.body.stmts[0].kind else {
            panic!()
        };
        assert!(matches!(main_call.kind, ExprKind::Call { func, .. } if func == bar_id));
    }

    #[test]
    fn tail_expr_becomes_implicit_return() {
        let prog = lower_ok("fn answer() -> int { 42 } fn main() { answer(); }");
        let answer = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "answer")
            .expect("answer");
        assert_eq!(answer.body.stmts.len(), 1);
        let StmtKind::Return(Some(expr)) = &answer.body.stmts[0].kind else {
            panic!("expected Return stmt")
        };
        assert!(matches!(expr.kind, ExprKind::Int(42)));
    }

    #[test]
    fn void_tail_expr_becomes_expr_stmt() {
        // A tail expression in a void function becomes Expr, not Return
        let prog = lower_ok(r#"fn main() { println("hi") }"#);
        let main = find_main(&prog);
        assert_eq!(main.body.stmts.len(), 1);
        assert!(matches!(main.body.stmts[0].kind, StmtKind::Expr(_)));
    }

    #[test]
    fn params_have_correct_locals() {
        let prog = lower_ok(
            "fn greet(name: string, count: int) -> bool { return count > 0; } fn main() {}",
        );
        let greet = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "greet")
            .expect("greet");
        assert_eq!(greet.params_len, 2);
        assert_eq!(greet.locals.len(), 2);
        assert_eq!(greet.locals[0].name.unwrap().to_string(), "name");
        assert_eq!(greet.locals[0].ty, Type::String);
        assert_eq!(greet.locals[1].name.unwrap().to_string(), "count");
        assert_eq!(greet.locals[1].ty, Type::Int);
    }

    #[test]
    fn inner_block_locals_do_not_leak_to_outer_scope() {
        let prog = lower_ok("fn main() { while true { let inner = 1; break; } let outer = 2; }");
        let main = find_main(&prog);
        assert_eq!(main.locals.len(), 2);
        assert_eq!(main.locals[0].name.unwrap().to_string(), "inner");
        assert_eq!(main.locals[1].name.unwrap().to_string(), "outer");
    }

    #[test]
    fn if_in_stmts_position_is_promoted() {
        let prog = lower_ok("fn main() { if true {} let x = 1; }");
        let main = find_main(&prog);
        assert!(matches!(main.body.stmts[0].kind, StmtKind::If { .. }));
    }

    #[test]
    fn rejects_for_loop() {
        let err = lower_err("fn main() { for n in 0..10 {} }");
        assert!(matches!(err, LowerError::UnsupportedStmtKind { .. }));
    }

    #[test]
    fn rejects_array_literal() {
        let err = lower_err("fn main() { let x = [1, 2, 3]; }");
        assert!(matches!(err, LowerError::UnsupportedExprKind { .. }));
    }

    #[test]
    fn rejects_map_literal() {
        let err = lower_err(r#"fn main() { let x = ["a": 1]; }"#);
        assert!(matches!(err, LowerError::UnsupportedExprKind { .. }));
    }

    #[test]
    fn tuple_literal_lowers_to_hir() {
        let prog = lower_ok("fn main() { let t = (1, 2); }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let stmt")
        };
        assert!(
            matches!(init.kind, ExprKind::TupleLiteral { .. }),
            "expected TupleLiteral, got {:?}",
            init.kind
        );
    }

    #[test]
    fn tuple_index_lowers_to_hir() {
        let prog = lower_ok("fn main() { let t = (10, 20); let v = t.0; }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[1].kind else {
            panic!("expected Let stmt")
        };
        assert!(
            matches!(init.kind, ExprKind::TupleIndex { index: 0, .. }),
            "expected TupleIndex(0), got {:?}",
            init.kind
        );
    }

    #[test]
    fn rejects_range() {
        let err = lower_err("fn main() { let x = 0..10; }");
        assert!(matches!(err, LowerError::UnsupportedExprKind { .. }));
    }

    #[test]
    fn string_interp_with_var() {
        let prog = lower_ok(r#"fn main() { let n = 1; let s = "n = {n}"; }"#);
        let main = find_main(&prog);
        // s = "n = " + n  → Binary(Add, String("n = "), Local(n))
        let StmtKind::Let { init, .. } = &main.body.stmts[1].kind else {
            panic!("expected Let stmt for s")
        };
        assert!(matches!(
            init.kind,
            ExprKind::Binary {
                op: crate::ast::BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn string_interp_single_expr_only() {
        let prog = lower_ok(r#"fn main() { let x = "hi"; let s = "{x}"; }"#);
        let main = find_main(&prog);
        // single Expr part -> just the local, no wrapper
        let StmtKind::Let { init, .. } = &main.body.stmts[1].kind else {
            panic!("expected Let stmt for s")
        };
        assert!(matches!(init.kind, ExprKind::Local(_)));
    }

    #[test]
    fn string_interp_multiple_parts() {
        let prog = lower_ok(r#"fn main() { let x = 1; let y = 2; let s = "a {x} b {y}"; }"#);
        let main = find_main(&prog);
        // "a {x} b {y}" → (("a " + x) + " b ") + y
        let StmtKind::Let { init, .. } = &main.body.stmts[2].kind else {
            panic!("expected Let stmt for s")
        };
        // outermost node is Add
        assert!(matches!(
            init.kind,
            ExprKind::Binary {
                op: crate::ast::BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn struct_literal_lowers_to_hir() {
        let prog =
            lower_ok("struct Point { x: int, y: int } fn main() { let p = Point { x: 1, y: 2 }; }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let stmt")
        };
        assert!(
            matches!(init.kind, ExprKind::StructLiteral { .. }),
            "expected StructLiteral, got {:?}",
            init.kind
        );
    }

    #[test]
    fn struct_literal_fields_in_declaration_order() {
        // fields provided in reversed order, lowering must reorder to declaration order
        let prog =
            lower_ok("struct Pair { a: int, b: int } fn main() { let p = Pair { b: 2, a: 1 }; }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let stmt")
        };
        let ExprKind::StructLiteral { fields, .. } = &init.kind else {
            panic!("expected StructLiteral")
        };
        // declaration order: a=0, b=1; provided b=2,a=1 -> fields[0]=Int(1), fields[1]=Int(2)
        assert!(matches!(fields[0].kind, ExprKind::Int(1)));
        assert!(matches!(fields[1].kind, ExprKind::Int(2)));
    }

    #[test]
    fn struct_literal_has_correct_type_id() {
        let prog = lower_ok(
            "struct A { x: int } struct B { y: int } fn main() { let a = A { x: 1 }; let b = B { y: 2 }; }",
        );
        let main = find_main(&prog);
        let StmtKind::Let { init: init_a, .. } = &main.body.stmts[0].kind else {
            panic!()
        };
        let StmtKind::Let { init: init_b, .. } = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::StructLiteral { type_id: id_a, .. } = &init_a.kind else {
            panic!()
        };
        let ExprKind::StructLiteral { type_id: id_b, .. } = &init_b.kind else {
            panic!()
        };
        assert_ne!(id_a, id_b, "different structs must have different type_ids");
    }

    #[test]
    fn field_get_lowers_to_hir() {
        let prog = lower_ok(
            "struct Point { x: int, y: int } fn main() { let p = Point { x: 5, y: 10 }; let v = p.x; }",
        );
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[1].kind else {
            panic!("expected Let stmt")
        };
        assert!(
            matches!(init.kind, ExprKind::FieldGet { index: 0, .. }),
            "expected FieldGet(index=0), got {:?}",
            init.kind
        );
    }

    #[test]
    fn set_field_lowers_to_hir() {
        let prog = lower_ok(
            "struct Point { x: int, y: int } fn main() { var p = Point { x: 1, y: 2 }; p.x = 99; }",
        );
        let main = find_main(&prog);
        assert!(
            matches!(
                main.body.stmts[1].kind,
                StmtKind::SetField { field_index: 0, .. }
            ),
            "expected SetField, got {:?}",
            main.body.stmts[1].kind
        );
    }

    #[test]
    fn rejects_coalesce() {
        let err = lower_err("fn main() { var x: int? = nil; let y = x ?? 0; }");
        assert!(matches!(err, LowerError::UnsupportedExprKind { .. }));
    }

    #[test]
    fn rejects_compound_assignment() {
        let err = lower_err("fn main() { var x = 1; x += 1; }");
        assert!(matches!(err, LowerError::UnsupportedAssign { .. }));
    }

    #[test]
    fn rejects_non_ident_pattern_in_let() {
        // Tuple destructuring is not supported in HIR v1
        let err = lower_err("fn main() { let (a, b) = (1, 2); }");
        assert!(matches!(err, LowerError::UnsupportedPattern { .. }));
    }

    #[test]
    fn lowers_match_expr() {
        let prog = lower_ok(
            "fn main() { var x: int? = nil; match x { Option.Some(v) => {}, Option.None => {}, } }",
        );
        let main = find_main(&prog);
        // let x, then the match stmt
        assert_eq!(main.body.stmts.len(), 2);
        assert!(matches!(main.body.stmts[1].kind, StmtKind::Match { .. }));
    }

    // ---- extern fn lowering tests ----

    #[test]
    fn extern_fn_emits_call_extern_node() {
        let prog =
            lower_ok("extern fn add(a: int, b: int) -> int\nfn main() { let x = add(1, 2); }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let stmt");
        };
        assert!(
            matches!(init.kind, ExprKind::CallExtern { .. }),
            "expected CallExtern, got {:?}",
            init.kind
        );
    }

    #[test]
    fn extern_fn_decl_is_in_hir_program() {
        let prog = lower_ok("extern fn tick()\nextern fn add(a: int, b: int) -> int\nfn main() {}");
        assert_eq!(prog.externs.len(), 2);
        assert_eq!(prog.externs[0].name.to_string(), "tick");
        assert_eq!(prog.externs[1].name.to_string(), "add");
        assert_eq!(prog.externs[1].params, vec![Type::Int, Type::Int]);
        assert_eq!(prog.externs[1].ret, Type::Int);
    }

    #[test]
    fn extern_fn_call_extern_has_correct_id() {
        let prog =
            lower_ok("extern fn add(a: int, b: int) -> int\nfn main() { let x = add(1, 2); }");
        assert_eq!(prog.externs[0].id, hir::ExternId(0));
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let");
        };
        let ExprKind::CallExtern { extern_id, args } = &init.kind else {
            panic!("expected CallExtern");
        };
        assert_eq!(*extern_id, hir::ExternId(0));
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn extern_type_flows_through_hir() {
        let prog = lower_ok(
            "extern type Sprite\nextern fn create() -> Sprite\nfn main() { let s = create(); }",
        );
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let stmt");
        };
        let ExprKind::CallExtern { .. } = &init.kind else {
            panic!("expected CallExtern, got {:?}", init.kind);
        };
        let Type::Extern { name } = &init.ty else {
            panic!("expected Type::Extern, got {:?}", init.ty);
        };
        assert_eq!(name.to_string(), "Sprite");
    }

    // ---- enum lowering tests ----

    #[test]
    fn lowers_unit_enum_variant() {
        let prog = lower_ok("enum Color { Red, Green, Blue } fn main() { let c = Color.Red; }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let");
        };
        assert!(
            matches!(init.kind, ExprKind::EnumLiteral { variant: 0, .. }),
            "expected EnumLiteral variant=0, got {:?}",
            init.kind
        );
    }

    #[test]
    fn lowers_tuple_enum_variant() {
        let prog =
            lower_ok("enum Msg { Ping(int), Move(int, int) } fn main() { let m = Msg.Ping(42); }");
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let");
        };
        let ExprKind::EnumLiteral {
            variant, fields, ..
        } = &init.kind
        else {
            panic!("expected EnumLiteral, got {:?}", init.kind);
        };
        assert_eq!(*variant, 0);
        assert_eq!(fields.len(), 1);
    }

    #[test]
    fn lowers_struct_enum_variant() {
        let prog = lower_ok(
            "enum Ev { Move { dx: int, dy: int } } fn main() { let e = Ev.Move { dx: 5, dy: 10 }; }",
        );
        let main = find_main(&prog);
        let StmtKind::Let { init, .. } = &main.body.stmts[0].kind else {
            panic!("expected Let");
        };
        let ExprKind::EnumLiteral {
            variant, fields, ..
        } = &init.kind
        else {
            panic!("expected EnumLiteral, got {:?}", init.kind);
        };
        assert_eq!(*variant, 0);
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn match_arms_have_correct_variant_indices() {
        let prog = lower_ok(
            "enum Color { Red, Green, Blue } fn main() { let c = Color.Green; match c { Color.Red => {}, Color.Green => {}, Color.Blue => {}, } }",
        );
        let main = find_main(&prog);
        let StmtKind::Match { arms, .. } = &main.body.stmts[1].kind else {
            panic!("expected Match stmt");
        };
        assert_eq!(arms[0].variant, 0); // Red
        assert_eq!(arms[1].variant, 1); // Green
        assert_eq!(arms[2].variant, 2); // Blue
    }

    #[test]
    fn match_wildcard_becomes_else_body() {
        let prog = lower_ok(
            "enum Color { Red, Green } fn main() { let c = Color.Red; match c { Color.Red => {}, _ => {}, } }",
        );
        let main = find_main(&prog);
        let StmtKind::Match {
            arms, else_body, ..
        } = &main.body.stmts[1].kind
        else {
            panic!("expected Match stmt");
        };
        assert_eq!(arms.len(), 1);
        assert!(else_body.is_some());
        assert!(else_body.as_ref().unwrap().binding.is_none());
    }

    #[test]
    fn match_tuple_arm_has_bindings() {
        let prog = lower_ok(
            "enum Msg { Ping(int) } fn main() { let m = Msg.Ping(42); match m { Msg.Ping(v) => {}, } }",
        );
        let main = find_main(&prog);
        let StmtKind::Match { arms, .. } = &main.body.stmts[1].kind else {
            panic!("expected Match stmt");
        };
        assert_eq!(arms[0].bindings.len(), 1);
        assert_eq!(arms[0].bindings[0].field_index, 0);
    }
}
