use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    fmt,
};

use crate::{
    ast::{self, ExprId, Ident, Type},
    hir,
    span::Span,
    typecheck::TypeChecker,
};

mod assign;
mod block;
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

pub(super) struct SharedCtx<'a> {
    pub(super) tcx: &'a TypeChecker,
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
}

impl LowerCtx<'_> {
    pub(super) fn expr_type(&self, id: ExprId, span: Span) -> Result<Type, LowerError> {
        let (_, ty) = self
            .type_overrides
            .and_then(|overrides| overrides.get(&id))
            .or_else(|| self.shared.tcx.get_type(id))
            .ok_or(LowerError::MissingExprType { span })?;
        Ok(ty.clone())
    }
}

pub(super) struct FuncLower {
    pub(super) locals: Vec<hir::Local>,
    pub(super) local_map: HashMap<Ident, hir::LocalId>,
    scope_log: Vec<(Ident, Option<hir::LocalId>)>,
}

impl FuncLower {
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
}

#[cfg(test)]
mod tests;
