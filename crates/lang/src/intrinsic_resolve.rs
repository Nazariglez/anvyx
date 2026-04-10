use std::{collections::HashSet, mem};

use crate::{
    ast,
    intrinsic::{
        self, CompilationContext, CompileTimeValueKind, DiagnosticKind, IntrinsicDiagnostic,
        IntrinsicDiagnosticLevel, IntrinsicError, IntrinsicKind, SourceLocationInfo,
    },
    span::Spanned,
};

struct ResolveCtx<'a> {
    ctx: &'a CompilationContext,
    errors: Vec<IntrinsicError>,
    intrinsic_bools: HashSet<ast::ExprId>,
    diagnostics: Vec<IntrinsicDiagnostic>,
    source_loc: Option<&'a SourceLocationInfo>,
}

pub struct IntrinsicResolveResult {
    pub errors: Vec<IntrinsicError>,
    pub diagnostics: Vec<IntrinsicDiagnostic>,
}

impl IntrinsicResolveResult {
    pub fn has_error_diagnostic(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| matches!(d.level, IntrinsicDiagnosticLevel::Error))
    }
}

pub fn resolve_intrinsics(
    program: &mut ast::Program,
    ctx: &CompilationContext,
    source_loc: Option<&SourceLocationInfo>,
) -> IntrinsicResolveResult {
    let mut rcx = ResolveCtx {
        ctx,
        errors: vec![],
        intrinsic_bools: HashSet::new(),
        diagnostics: vec![],
        source_loc,
    };
    for stmt in &mut program.stmts {
        resolve_stmt(stmt, &mut rcx, None);
    }
    IntrinsicResolveResult {
        errors: rcx.errors,
        diagnostics: rcx.diagnostics,
    }
}

fn resolve_stmt(stmt: &mut ast::StmtNode, rcx: &mut ResolveCtx<'_>, fn_name: Option<&str>) {
    match &mut stmt.node {
        ast::Stmt::Expr(e) => resolve_expr(e, rcx, fn_name),
        ast::Stmt::Binding(b) => resolve_expr(&mut b.node.value, rcx, fn_name),
        ast::Stmt::LetElse(le) => {
            resolve_expr(&mut le.node.value, rcx, fn_name);
            resolve_block(&mut le.node.else_block, rcx, fn_name);
        }
        ast::Stmt::Return(r) => {
            if let Some(v) = &mut r.node.value {
                resolve_expr(v, rcx, fn_name);
            }
        }
        ast::Stmt::While(w) => {
            resolve_expr(&mut w.node.cond, rcx, fn_name);
            resolve_block(&mut w.node.body, rcx, fn_name);
        }
        ast::Stmt::WhileLet(wl) => {
            resolve_expr(&mut wl.node.value, rcx, fn_name);
            resolve_block(&mut wl.node.body, rcx, fn_name);
        }
        ast::Stmt::For(f) => {
            resolve_expr(&mut f.node.iterable, rcx, fn_name);
            if let Some(step) = &mut f.node.step {
                resolve_expr(step, rcx, fn_name);
            }
            resolve_block(&mut f.node.body, rcx, fn_name);
        }
        ast::Stmt::Defer(d) => match &mut d.node.body {
            ast::DeferBody::Expr(e) => resolve_expr(e, rcx, fn_name),
            ast::DeferBody::Block(b) => resolve_block(b, rcx, fn_name),
        },
        ast::Stmt::Func(f) => {
            let func_name = f.node.name.0.as_ref();
            resolve_params(&mut f.node.params, rcx, Some(func_name));
            resolve_block(&mut f.node.body, rcx, Some(func_name));
        }
        ast::Stmt::Extend(ext) => {
            let type_name = ext.node.ty.to_string();
            for method in &mut ext.node.methods {
                let method_fn_name = format!("{type_name}.{}", method.node.name);
                resolve_params(&mut method.node.params, rcx, Some(&method_fn_name));
                resolve_block(&mut method.node.body, rcx, Some(&method_fn_name));
            }
            for cast_from in &mut ext.node.cast_froms {
                let cast_fn_name = format!("{type_name}.from");
                if let Some(default) = &mut cast_from.node.param.default {
                    resolve_expr(default, rcx, Some(&cast_fn_name));
                }
                resolve_block(&mut cast_from.node.body, rcx, Some(&cast_fn_name));
            }
        }
        ast::Stmt::Aggregate(s) => {
            resolve_struct_fields(&mut s.node.fields, rcx, fn_name);
            let struct_name = s.node.name.to_string();
            for method in &mut s.node.methods {
                let method_fn_name = format!("{struct_name}.{}", method.name);
                resolve_params(&mut method.params, rcx, Some(&method_fn_name));
                resolve_block(&mut method.body, rcx, Some(&method_fn_name));
            }
        }
        ast::Stmt::Enum(e) => {
            for variant in &mut e.node.variants {
                if let ast::VariantKind::Struct(fields) = &mut variant.kind {
                    resolve_struct_fields(fields, rcx, fn_name);
                }
            }
        }
        ast::Stmt::Const(c) => resolve_expr(&mut c.node.value, rcx, fn_name),
        ast::Stmt::Import(_)
        | ast::Stmt::ExternFunc(_)
        | ast::Stmt::ExternType(_)
        | ast::Stmt::Break
        | ast::Stmt::Continue => {}
    }
}

fn resolve_expr(expr: &mut ast::ExprNode, rcx: &mut ResolveCtx<'_>, fn_name: Option<&str>) {
    match &mut expr.node.kind {
        ast::ExprKind::IntrinsicCall(node) => {
            let name: &str = node.node.name.0.as_ref();
            let Some(def) = intrinsic::lookup(name) else {
                rcx.errors.push(IntrinsicError::UnknownIntrinsic {
                    name: name.to_string(),
                    span: expr.span,
                });
                return;
            };

            let expected_count = match def.arg {
                intrinsic::ArgShape::None => 0,
                intrinsic::ArgShape::Ident | intrinsic::ArgShape::StringLit => 1,
            };
            if node.node.args.len() != expected_count {
                rcx.errors.push(IntrinsicError::WrongArgCount {
                    name: name.to_string(),
                    expected: expected_count,
                    found: node.node.args.len(),
                    span: expr.span,
                });
                return;
            }

            match def.kind {
                IntrinsicKind::CompileTimeValue(ctv) => {
                    let call_span = expr.span;
                    match ctv {
                        CompileTimeValueKind::File => {
                            let path = rcx
                                .source_loc
                                .map(|s| s.file_path.clone())
                                .unwrap_or_default();
                            expr.node.kind = ast::ExprKind::Lit(ast::Lit::String(path));
                        }
                        CompileTimeValueKind::Line => {
                            let line = rcx
                                .source_loc
                                .and_then(|s| s.line_for_token.get(call_span.start))
                                .copied()
                                .unwrap_or(0);
                            expr.node.kind = ast::ExprKind::Lit(ast::Lit::Int(line));
                        }
                        CompileTimeValueKind::Func => {
                            let func = fn_name.unwrap_or_default().to_string();
                            expr.node.kind = ast::ExprKind::Lit(ast::Lit::String(func));
                        }
                    }
                }
                IntrinsicKind::Predicate => {
                    let ast::ExprKind::Ident(ident) = &node.node.args[0].node.kind else {
                        rcx.errors.push(IntrinsicError::ArgNotIdent {
                            name: name.to_string(),
                            span: expr.span,
                        });
                        return;
                    };
                    let arg_str = ident.0.as_ref();
                    match intrinsic::evaluate_predicate(name, arg_str, rcx.ctx, expr.span) {
                        Ok(value) => {
                            expr.node.kind = ast::ExprKind::Lit(ast::Lit::Bool(value));
                            rcx.intrinsic_bools.insert(expr.node.id);
                        }
                        Err(e) => rcx.errors.push(e),
                    }
                }
                IntrinsicKind::Diagnostic(diag_kind) => {
                    let ast::ExprKind::Lit(ast::Lit::String(msg)) = &node.node.args[0].node.kind
                    else {
                        rcx.errors.push(IntrinsicError::ArgNotStringLiteral {
                            name: name.to_string(),
                            span: expr.span,
                        });
                        return;
                    };
                    let msg = msg.clone();
                    let call_span = expr.span;
                    let level = match diag_kind {
                        DiagnosticKind::Warn => IntrinsicDiagnosticLevel::Warning,
                        DiagnosticKind::Error => IntrinsicDiagnosticLevel::Error,
                        DiagnosticKind::Log => IntrinsicDiagnosticLevel::Note,
                    };
                    rcx.diagnostics.push(IntrinsicDiagnostic {
                        level,
                        message: msg,
                        span: call_span,
                    });
                    expr.node.kind = empty_block(call_span);
                }
            }
        }

        ast::ExprKind::Binary(node) => {
            resolve_expr(&mut node.node.left, rcx, fn_name);
            resolve_expr(&mut node.node.right, rcx, fn_name);
        }
        ast::ExprKind::Unary(node) => {
            resolve_expr(&mut node.node.expr, rcx, fn_name);
        }
        ast::ExprKind::Call(node) => {
            resolve_expr(&mut node.node.func, rcx, fn_name);
            for arg in &mut node.node.args {
                resolve_expr(arg, rcx, fn_name);
            }
        }
        ast::ExprKind::If(node) => {
            resolve_expr(&mut node.node.cond, rcx, fn_name);
        }
        ast::ExprKind::IfLet(node) => {
            resolve_expr(&mut node.node.value, rcx, fn_name);
            resolve_block(&mut node.node.then_block, rcx, fn_name);
            if let Some(else_block) = &mut node.node.else_block {
                resolve_block(else_block, rcx, fn_name);
            }
        }
        ast::ExprKind::Block(block) => {
            resolve_block(block, rcx, fn_name);
        }
        ast::ExprKind::Assign(node) => {
            resolve_expr(&mut node.node.target, rcx, fn_name);
            resolve_expr(&mut node.node.value, rcx, fn_name);
        }
        ast::ExprKind::Tuple(exprs) => {
            for e in exprs {
                resolve_expr(e, rcx, fn_name);
            }
        }
        ast::ExprKind::NamedTuple(fields) => {
            for (_, e) in fields {
                resolve_expr(e, rcx, fn_name);
            }
        }
        ast::ExprKind::StructLiteral(node) => {
            for (_, e) in &mut node.node.fields {
                resolve_expr(e, rcx, fn_name);
            }
        }
        ast::ExprKind::ArrayLiteral(node) => {
            for e in &mut node.node.elements {
                resolve_expr(e, rcx, fn_name);
            }
        }
        ast::ExprKind::ArrayFill(node) => {
            resolve_expr(&mut node.node.value, rcx, fn_name);
            resolve_expr(&mut node.node.len, rcx, fn_name);
        }
        ast::ExprKind::MapLiteral(node) => {
            for (k, v) in &mut node.node.entries {
                resolve_expr(k, rcx, fn_name);
                resolve_expr(v, rcx, fn_name);
            }
        }
        ast::ExprKind::Index(node) => {
            resolve_expr(&mut node.node.target, rcx, fn_name);
            resolve_expr(&mut node.node.index, rcx, fn_name);
        }
        ast::ExprKind::Range(node) => match &mut node.node {
            ast::Range::Bounded { start, end, .. } => {
                resolve_expr(start, rcx, fn_name);
                resolve_expr(end, rcx, fn_name);
            }
            ast::Range::From { start } => resolve_expr(start, rcx, fn_name),
            ast::Range::To { end, .. } => resolve_expr(end, rcx, fn_name),
        },
        ast::ExprKind::Match(node) => {
            resolve_expr(&mut node.node.scrutinee, rcx, fn_name);
            for arm in &mut node.node.arms {
                resolve_expr(&mut arm.node.body, rcx, fn_name);
            }
        }
        ast::ExprKind::StringInterp(parts) => {
            for part in parts {
                if let ast::StringPart::Expr(e, _) = part {
                    resolve_expr(e, rcx, fn_name);
                }
            }
        }
        ast::ExprKind::Cast(node) => {
            resolve_expr(&mut node.node.expr, rcx, fn_name);
        }
        ast::ExprKind::Lambda(node) => {
            resolve_expr(&mut node.node.body, rcx, fn_name);
        }
        ast::ExprKind::InferredEnum(node) => match &mut node.node.args {
            ast::InferredEnumArgs::Tuple(v) => {
                for e in v {
                    resolve_expr(e, rcx, fn_name);
                }
            }
            ast::InferredEnumArgs::Struct(v) => {
                for (_, e) in v {
                    resolve_expr(e, rcx, fn_name);
                }
            }
            ast::InferredEnumArgs::Unit => {}
        },
        ast::ExprKind::Field(node) => {
            resolve_expr(&mut node.node.target, rcx, fn_name);
        }
        ast::ExprKind::TupleIndex(node) => {
            resolve_expr(&mut node.node.target, rcx, fn_name);
        }
        ast::ExprKind::Lit(_) | ast::ExprKind::Ident(_) => {}
    }

    try_fold_bool(expr, &mut rcx.intrinsic_bools);
    try_prune_if(expr, rcx, fn_name);
}

/// Turns boolean expressions like "!true" or "true && false" into one value
/// and remembers when it came from an intrinsic.
fn try_fold_bool(expr: &mut ast::ExprNode, intrinsic_bools: &mut HashSet<ast::ExprId>) {
    enum Action {
        SetLit { value: bool, intrinsic_origin: bool },
        TakeRight,
        NoFold,
    }

    let action = match &expr.node.kind {
        ast::ExprKind::Unary(node) if node.node.op == ast::UnaryOp::Not => {
            match &node.node.expr.node.kind {
                ast::ExprKind::Lit(ast::Lit::Bool(v)) => {
                    let is_intrinsic = intrinsic_bools.contains(&node.node.expr.node.id);
                    Action::SetLit {
                        value: !v,
                        intrinsic_origin: is_intrinsic,
                    }
                }
                _ => Action::NoFold,
            }
        }
        ast::ExprKind::Binary(node)
            if node.node.op == ast::BinaryOp::And || node.node.op == ast::BinaryOp::Or =>
        {
            let left = match &node.node.left.node.kind {
                ast::ExprKind::Lit(ast::Lit::Bool(v)) => Some(*v),
                _ => None,
            };
            let right = match &node.node.right.node.kind {
                ast::ExprKind::Lit(ast::Lit::Bool(v)) => Some(*v),
                _ => None,
            };
            let left_intrinsic = intrinsic_bools.contains(&node.node.left.node.id);
            let right_intrinsic = intrinsic_bools.contains(&node.node.right.node.id);
            let op = node.node.op;
            match (left, right, op) {
                (Some(l), Some(r), ast::BinaryOp::And) => Action::SetLit {
                    value: l && r,
                    intrinsic_origin: left_intrinsic && right_intrinsic,
                },
                (Some(l), Some(r), ast::BinaryOp::Or) => Action::SetLit {
                    value: l || r,
                    intrinsic_origin: left_intrinsic && right_intrinsic,
                },
                (Some(false), _, ast::BinaryOp::And) => Action::SetLit {
                    value: false,
                    intrinsic_origin: left_intrinsic,
                },
                (Some(true), _, ast::BinaryOp::Or) => Action::SetLit {
                    value: true,
                    intrinsic_origin: left_intrinsic,
                },
                (Some(true), None, ast::BinaryOp::And) | (Some(false), None, ast::BinaryOp::Or) => {
                    Action::TakeRight
                }
                _ => Action::NoFold,
            }
        }
        _ => Action::NoFold,
    };

    match action {
        Action::SetLit {
            value,
            intrinsic_origin: is_intrinsic_origin,
        } => {
            expr.node.kind = ast::ExprKind::Lit(ast::Lit::Bool(value));
            if is_intrinsic_origin {
                intrinsic_bools.insert(expr.node.id);
            }
        }
        Action::TakeRight => {
            let old = mem::replace(
                &mut expr.node.kind,
                ast::ExprKind::Lit(ast::Lit::Bool(false)),
            );
            if let ast::ExprKind::Binary(bin) = old {
                let right = *bin.node.right;
                expr.node = right.node;
                expr.span = right.span;
            }
        }
        Action::NoFold => {}
    }
}

/// If an intrinsic turns an "if" condition into boolean, keep only the branch that can run
fn try_prune_if(expr: &mut ast::ExprNode, rcx: &mut ResolveCtx<'_>, fn_name: Option<&str>) {
    let cond_value = match &expr.node.kind {
        ast::ExprKind::If(node) => match &node.node.cond.node.kind {
            ast::ExprKind::Lit(ast::Lit::Bool(v))
                if rcx.intrinsic_bools.contains(&node.node.cond.node.id) =>
            {
                Some(*v)
            }
            _ => None,
        },
        _ => return,
    };

    match cond_value {
        Some(take_then) => {
            let old = mem::replace(
                &mut expr.node.kind,
                ast::ExprKind::Lit(ast::Lit::Bool(false)),
            );
            let ast::ExprKind::If(if_node) = old else {
                unreachable!()
            };
            let if_data = if_node.into_node();
            let chosen = if take_then {
                Some(if_data.then_block)
            } else {
                if_data.else_block
            };
            match chosen {
                Some(mut block) => {
                    resolve_block(&mut block, rcx, fn_name);
                    expr.node.kind = ast::ExprKind::Block(block);
                }
                None => {
                    expr.node.kind = empty_block(expr.span);
                }
            }
        }
        None => {
            let ast::ExprKind::If(node) = &mut expr.node.kind else {
                unreachable!()
            };
            resolve_block(&mut node.node.then_block, rcx, fn_name);
            if let Some(else_block) = &mut node.node.else_block {
                resolve_block(else_block, rcx, fn_name);
            }
        }
    }
}

fn resolve_params(params: &mut [ast::Param], rcx: &mut ResolveCtx<'_>, fn_name: Option<&str>) {
    for param in params {
        if let Some(default) = &mut param.default {
            resolve_expr(default, rcx, fn_name);
        }
    }
}

fn resolve_struct_fields(
    fields: &mut [ast::StructField],
    rcx: &mut ResolveCtx<'_>,
    fn_name: Option<&str>,
) {
    for field in fields {
        if let Some(default) = &mut field.default {
            resolve_expr(default, rcx, fn_name);
        }
    }
}

fn empty_block(span: crate::span::Span) -> ast::ExprKind {
    ast::ExprKind::Block(Spanned::new(
        ast::Block {
            stmts: vec![],
            tail: None,
        },
        span,
    ))
}

fn resolve_block(block: &mut ast::BlockNode, rcx: &mut ResolveCtx<'_>, fn_name: Option<&str>) {
    for stmt in &mut block.node.stmts {
        resolve_stmt(stmt, rcx, fn_name);
    }
    if let Some(tail) = &mut block.node.tail {
        resolve_expr(tail, rcx, fn_name);
    }
}
