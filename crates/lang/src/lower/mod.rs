use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    fmt,
};

use crate::{
    ast::{self, ExprId, Ident, Type},
    hir,
    span::Span,
    typecheck::TypecheckResult,
};

mod assign;
mod block;
mod coerce;
mod expr;
mod for_loop;
mod helpers;
mod match_stmt;
mod ownership;
mod program;

use assign::lower_assign;
use block::{lower_block, lower_block_to_target, lower_string_interp};
use expr::lower_expr;
use for_loop::lower_for;
use helpers::*;
use match_stmt::{
    alloc_write_through, lower_if_let, lower_let_else, lower_match_stmts, lower_while_let,
};
pub use ownership::analyze_ownership;
pub use program::lower_program;

#[derive(Debug)]
pub enum LowerError {
    UnsupportedStmtKind { span: Span, kind: String },
    UnsupportedExprKind { span: Span, kind: String },
    UnsupportedPattern { span: Span },
    UnsupportedAssign { span: Span, detail: String },
    UnknownLocal { name: Ident, span: Span },
    UnknownFunc { name: Ident, span: Span },
    MissingExprType { span: Span },
    MissingBindingType { span: Span },
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
            Self::MissingBindingType { span } => write!(
                f,
                "binding at offset {} has no resolved type (compiler bug)",
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

pub(super) struct SharedCtx<'a> {
    pub(super) tcx: &'a TypecheckResult,
    pub(super) funcs: HashMap<Ident, hir::FuncId>,
    pub(super) externs: HashMap<Ident, hir::ExternId>,
    pub(super) struct_type_ids: HashMap<Ident, u32>,
    pub(super) enum_type_ids: HashMap<Ident, u32>,
    pub(super) next_func_id: Cell<u32>,
    pub(super) lambda_funcs: RefCell<Vec<hir::Func>>,
    pub(super) func_asts: HashMap<Ident, ast::FuncNode>,
}

impl SharedCtx<'_> {
    pub(super) fn get_func_ast(&self, name: Ident) -> Option<&ast::FuncNode> {
        self.func_asts.get(&name)
    }

    pub(super) fn alloc_func_id(&self) -> hir::FuncId {
        let id = self.next_func_id.get();
        self.next_func_id.set(id + 1);
        hir::FuncId(id)
    }

    pub(super) fn register_lambda_func(&self, func: hir::Func) {
        self.lambda_funcs.borrow_mut().push(func);
    }
}

pub(super) struct LowerCtx<'a> {
    pub(super) shared: &'a SharedCtx<'a>,
    pub(super) type_overrides: Option<&'a HashMap<ExprId, (Span, Type)>>,
    pub(super) binding_type_overrides: Option<&'a HashMap<ExprId, Type>>,
}

impl LowerCtx<'_> {
    pub(super) fn expr_type(&self, id: ExprId, span: Span) -> Result<Type, LowerError> {
        let result = self
            .type_overrides
            .and_then(|overrides| overrides.get(&id))
            .or_else(|| self.shared.tcx.get_type(id));
        let (_, ty) = result.ok_or(LowerError::MissingExprType { span })?;
        Ok(ty.clone())
    }

    pub(super) fn binding_type(&self, id: ExprId, span: Span) -> Result<Type, LowerError> {
        let ty = self
            .binding_type_overrides
            .and_then(|overrides| overrides.get(&id))
            .or_else(|| self.shared.tcx.binding_type(id))
            .ok_or(LowerError::MissingBindingType { span })?;
        Ok(ty.clone())
    }
}

pub(super) struct FuncLower {
    pub(super) locals: Vec<hir::Local>,
    pub(super) local_map: HashMap<Ident, hir::LocalId>,
    scope_log: Vec<(Ident, Option<hir::LocalId>)>,
    pub(super) defer_stack: Vec<Vec<Vec<hir::Stmt>>>,
    pub(super) loop_defer_depth: Option<usize>,
}

impl FuncLower {
    pub(super) fn new() -> Self {
        Self {
            locals: vec![],
            local_map: HashMap::new(),
            scope_log: vec![],
            defer_stack: vec![],
            loop_defer_depth: None,
        }
    }

    pub(super) fn enter_scope(&mut self) -> usize {
        self.scope_log.len()
    }

    pub(super) fn leave_scope(&mut self, mark: usize) {
        while self.scope_log.len() > mark {
            let (name, prev) = self.scope_log.pop().unwrap();
            match prev {
                Some(old_id) => {
                    self.local_map.insert(name, old_id);
                }
                None => {
                    self.local_map.remove(&name);
                }
            }
        }
    }

    pub(super) fn bind_local(&mut self, name: Ident, id: hir::LocalId) {
        let prev = self.local_map.insert(name, id);
        self.scope_log.push((name, prev));
    }

    pub(super) fn push_defer_scope(&mut self) {
        self.defer_stack.push(vec![]);
    }

    pub(super) fn pop_defer_scope(&mut self) -> Vec<Vec<hir::Stmt>> {
        self.defer_stack.pop().expect("defer scope underflow")
    }

    pub(super) fn add_defer(&mut self, stmts: Vec<hir::Stmt>) {
        if let Some(scope) = self.defer_stack.last_mut() {
            scope.push(stmts);
        }
    }

    pub(super) fn defers_from_depth(&self, from_depth: usize) -> Vec<hir::Stmt> {
        self.defer_stack[from_depth..]
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev().flat_map(|d| d.iter().cloned()))
            .collect()
    }

    pub(super) fn all_active_defers(&self) -> Vec<hir::Stmt> {
        self.defers_from_depth(0)
    }

    pub(super) fn has_defers_from_depth(&self, from_depth: usize) -> bool {
        self.defer_stack[from_depth..]
            .iter()
            .any(|scope| !scope.is_empty())
    }

    pub(super) fn has_any_defers(&self) -> bool {
        self.has_defers_from_depth(0)
    }

    pub(super) fn enter_loop_defer(&mut self) -> Option<usize> {
        let old = self.loop_defer_depth;
        self.loop_defer_depth = Some(self.defer_stack.len());
        old
    }

    pub(super) fn leave_loop_defer(&mut self, old: Option<usize>) {
        self.loop_defer_depth = old;
    }
}

pub(super) fn flush_defer_scope(fc: &mut FuncLower, stmts: &mut Vec<hir::Stmt>) {
    let defers = fc.pop_defer_scope();
    if !defers.is_empty() {
        let ends_with_exit = stmts.last().is_some_and(|s| {
            matches!(
                s.kind,
                hir::StmtKind::Return(_) | hir::StmtKind::Break | hir::StmtKind::Continue
            )
        });
        if !ends_with_exit {
            for defer_group in defers.into_iter().rev() {
                stmts.extend(defer_group);
            }
        }
    }
}

pub(super) fn emit_deferred_return(
    fc: &mut FuncLower,
    span: Span,
    out: &mut Vec<hir::Stmt>,
    hir_expr: hir::Expr,
) -> hir::Stmt {
    let expr_ty = hir_expr.ty.clone();
    let temp = alloc_and_bind(fc, span, out, expr_ty.clone(), hir_expr);
    out.extend(fc.all_active_defers());
    hir::Stmt {
        span,
        kind: hir::StmtKind::Return(Some(hir::Expr::local(expr_ty, span, temp))),
    }
}

#[cfg(test)]
mod tests;
