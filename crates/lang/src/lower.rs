use std::collections::HashMap;
use std::fmt;

use crate::ast::{self, AssignOp, BinaryOp, Ident, Lit, Pattern, Stmt, StringPart, Type};
use crate::builtin::Builtin;
use crate::hir;
use crate::span::Span;
use crate::typecheck::TypeChecker;

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

struct LowerCtx<'a> {
    tcx: &'a TypeChecker,
    funcs: HashMap<Ident, hir::FuncId>,
    externs: HashMap<Ident, hir::ExternId>,
}

struct FuncLower {
    locals: Vec<hir::Local>,
    local_map: HashMap<Ident, hir::LocalId>,
}

pub fn lower_program(ast: &ast::Program, tcx: &TypeChecker) -> Result<hir::Program, LowerError> {
    let mut ctx = LowerCtx {
        tcx,
        funcs: HashMap::new(),
        externs: HashMap::new(),
    };

    let mut func_nodes: Vec<&ast::FuncNode> = vec![];
    let mut next_func_id = 0u32;
    let mut next_extern_id = 0u32;
    let mut extern_decls: Vec<hir::ExternDecl> = vec![];

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
                // register aliases so they resolve to the same FuncId/ExternId as the original
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

    // lower each function body
    let mut funcs = vec![];
    for func_node in func_nodes {
        funcs.push(lower_func(func_node, &ctx)?);
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

        Stmt::Struct(_) => Err(LowerError::UnsupportedStmtKind {
            span,
            kind: "struct declaration".to_string(),
        }),

        Stmt::Enum(_) => Err(LowerError::UnsupportedStmtKind {
            span,
            kind: "enum declaration".to_string(),
        }),
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

    let name = match &assign_node.node.target.node.kind {
        ast::ExprKind::Ident(name) => *name,
        _ => {
            return Err(LowerError::UnsupportedAssign {
                span,
                detail: "assignment target must be a plain local variable".to_string(),
            });
        }
    };

    let local_id = *fc
        .local_map
        .get(&name)
        .ok_or(LowerError::UnknownLocal { name, span })?;

    let value = lower_expr(&assign_node.node.value, ctx, fc)?;

    Ok(hir::Stmt {
        span,
        kind: hir::StmtKind::Assign {
            local: local_id,
            value,
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
            .tcx
            .get_type(ast_expr.node.id)
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
            } else {
                return Err(LowerError::UnknownFunc {
                    name: callee_name,
                    span,
                });
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
    fn rejects_tuple() {
        let err = lower_err("fn main() { let x = (1, 2); }");
        assert!(matches!(err, LowerError::UnsupportedExprKind { .. }));
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
    fn rejects_struct_literal() {
        let err = lower_err(
            "struct Point { x: int, y: int } fn main() { let p = Point { x: 1, y: 2 }; }",
        );
        assert!(matches!(err, LowerError::UnsupportedExprKind { .. }));
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
    fn rejects_match_expr() {
        let err = lower_err(
            "fn main() { var x: int? = nil; match x { Option.Some(v) => {}, Option.None => {}, } }",
        );
        assert!(matches!(
            err,
            LowerError::UnsupportedStmtKind { .. } | LowerError::UnsupportedExprKind { .. }
        ));
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
}
