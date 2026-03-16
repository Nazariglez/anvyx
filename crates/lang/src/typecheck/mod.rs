mod call;
mod composite;
mod constraint;
mod control;
mod decl;
mod error;
mod expr;
mod infer;
mod ops;
mod pattern;
mod postfix;
mod range;
mod stmt;
mod types;
mod unify;

#[cfg(test)]
mod tests;

pub use error::{TypeErr, TypeErrKind};
pub use types::TypeChecker;

use crate::ast::{Program, StmtNode};
use crate::builtin::Builtin;
use constraint::resolve_constraints;
use stmt::{build_module_def_with_reexports, check_block_stmts};
use unify::contains_infer;

fn register_builtins(type_checker: &mut TypeChecker) {
    type_checker.push_scope();
    for builtin in Builtin::all() {
        type_checker.set_var(builtin.ident(), builtin.func_type(), false);
    }
}

pub fn check_program(program: &Program) -> Result<TypeChecker, Vec<TypeErr>> {
    check_program_with_modules(program, &[])
}

pub fn check_program_with_modules(
    program: &Program,
    module_list: &[(Vec<String>, Vec<StmtNode>)],
) -> Result<TypeChecker, Vec<TypeErr>> {
    let mut type_checker = TypeChecker::default();
    let mut errors = vec![];

    register_builtins(&mut type_checker);

    // pre-load module stmts in DFS post-order (deepest dependencies first)
    for (path, stmts) in module_list {
        type_checker
            .resolved_module_stmts
            .insert(path.clone(), stmts.clone());
    }

    for (path, stmts) in module_list {
        let module_def = build_module_def_with_reexports(stmts, &type_checker);
        type_checker
            .resolved_module_defs
            .insert(path.clone(), module_def);
    }

    // first pass we collect the types from the ast
    // we don't need the type of the file scope blocks
    let _ = check_block_stmts(&program.stmts, None, &mut type_checker, &mut errors);

    if !errors.is_empty() {
        return Err(errors);
    }

    // second pass we infer the types from the constraints
    resolve_constraints(&mut type_checker, &mut errors);

    // at this point there should be no remaining unresolved types
    // so if there are any we add an error
    for (_expr_id, (span, ty)) in type_checker.types() {
        if contains_infer(ty) {
            errors.push(TypeErr::new(*span, TypeErrKind::UnresolvedInfer));
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(type_checker)
}
