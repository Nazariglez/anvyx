mod annotations;
mod call;
mod composite;
mod const_eval;
mod constraint;
mod control;
mod cyclicity;
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
mod visit;

#[cfg(test)]
mod tests;

pub use const_eval::ConstValue;
use const_eval::evaluate_and_export_consts;
use constraint::resolve_constraints;
pub use error::{Diagnostic, DiagnosticKind, Severity};
pub use infer::{build_const_subst, resolve_type_param_names, subst_type};
use internment::Intern;
use stmt::{build_module_def_with_reexports, check_block_stmts};
use types::{ExtendEntry, GenericExtendTemplate, TypeChecker};
pub use types::{
    ExtendSpecKey, ExternTypeDef, FieldDefault, MethodSpecKey, SpecializationKey, TypecheckResult,
};
use unify::contains_infer;
pub use visit::{map_type_structure, walk_type_structure};

use crate::{
    ast::{Ident, Program, Stmt, StmtNode, Visibility},
    builtin::Builtin,
};

fn register_builtins(type_checker: &mut TypeChecker) {
    type_checker.push_scope();
    for builtin in Builtin::all() {
        type_checker.set_var(builtin.ident(), builtin.func_type(), false);
    }
}

pub fn check_program_with_modules(
    program: &Program,
    module_list: &[(Vec<String>, Vec<StmtNode>)],
    auto_use_modules: &[Vec<String>],
) -> Result<TypecheckResult, Vec<Diagnostic>> {
    let mut type_checker = TypeChecker::default();
    let mut errors = vec![];

    register_builtins(&mut type_checker);

    // pre register types from the main program (prelude types like Option, Range)
    // so they're available during module def building
    for stmt in &program.stmts {
        match &stmt.node {
            Stmt::Enum(node) => {
                type_checker
                    .ctx
                    .enum_defs
                    .insert(node.node.name, types::EnumDef::from_ast(&node.node));
            }
            Stmt::Struct(node) => {
                type_checker.ctx.struct_defs.insert(
                    node.node.name,
                    types::StructDef::from_ast(&node.node, false),
                );
            }
            Stmt::DataRef(node) => {
                type_checker
                    .ctx
                    .struct_defs
                    .insert(node.node.name, types::StructDef::from_ast(&node.node, true));
            }
            _ => {}
        }
    }

    // pre-load module stmts in DFS post-order (deepest dependencies first)
    for (path, stmts) in module_list {
        type_checker
            .resolved_module_stmts
            .insert(path.clone(), stmts.clone());
    }

    for (path, stmts) in module_list {
        let (mut module_def, reexport_errors) =
            build_module_def_with_reexports(stmts, &type_checker, path);
        errors.extend(reexport_errors);

        let (const_defs, const_errors) =
            evaluate_and_export_consts(stmts, &type_checker.resolved_module_defs);
        errors.extend(const_errors);

        for stmt in stmts {
            let Stmt::Func(node) = &stmt.node else {
                continue;
            };
            let func = &node.node;
            if func.visibility != Visibility::Public {
                continue;
            }
            if !func.params.iter().any(|p| p.default.is_some()) {
                continue;
            }
            let defaults: Vec<Option<ConstValue>> = func
                .params
                .iter()
                .map(|p| {
                    p.default
                        .as_ref()
                        .and_then(|expr| const_eval::eval_const_expr(expr, &const_defs).ok())
                })
                .collect();
            module_def.func_param_defaults.insert(func.name, defaults);
        }

        for (name, def) in const_defs {
            if def.visibility == Visibility::Public {
                module_def.const_defs.insert(name, def);
            }
        }

        type_checker
            .resolved_module_defs
            .insert(path.clone(), module_def);
    }

    // auto activate extend methods from core modules (no import required)
    for path in auto_use_modules {
        if let Some(module_def) = type_checker.resolved_module_defs.get(path).cloned() {
            let binding = Ident(Intern::new(String::new()));
            for entry in &module_def.extend_methods {
                let key = (entry.ty.clone(), entry.name);
                type_checker
                    .ctx
                    .extend_defs
                    .entry(key)
                    .or_default()
                    .push(ExtendEntry {
                        source_module: path.clone(),
                        binding,
                        def: entry.def.clone(),
                    });
            }
            for entry in &module_def.generic_extend_methods {
                let key = (entry.base_name, entry.method_name);
                type_checker
                    .ctx
                    .generic_extend_templates
                    .entry(key)
                    .or_default()
                    .push(GenericExtendTemplate {
                        type_params: entry.type_params.clone(),
                        const_params: entry.const_params.clone(),
                        target_type: entry.target_type.clone(),
                        method: entry.method.clone(),
                        source_module: path.clone(),
                        binding,
                    });
            }
        }
    }

    let prelude_ctx = type_checker.ctx.clone();

    flush_errors(&mut errors, &mut type_checker)?;

    // we typecheck modules first so the main pass can specialize against fully populated imported-module contexts
    for (path, stmts) in module_list {
        type_checker.ctx = prelude_ctx.clone();
        type_checker.ctx.module_path = Some(path.clone());
        let _ = check_block_stmts(stmts, None, &mut type_checker, &mut errors, None);
        type_checker
            .module_check_contexts
            .insert(path.clone(), type_checker.ctx.clone());
    }

    flush_errors(&mut errors, &mut type_checker)?;

    // this pass only records AST types, we do not care about file-scope block types here
    type_checker.ctx = prelude_ctx.clone();
    let _ = check_block_stmts(&program.stmts, None, &mut type_checker, &mut errors, None);

    flush_errors(&mut errors, &mut type_checker)?;

    for module_def in type_checker.resolved_module_defs.values() {
        for (name, def) in &module_def.extern_types {
            type_checker
                .ctx
                .extern_type_defs
                .entry(*name)
                .or_insert_with(|| def.clone());
        }
    }

    flush_errors(&mut errors, &mut type_checker)?;

    // second pass we infer the types from the constraints
    resolve_constraints(&mut type_checker, &mut errors);

    // at this point there should be no remaining unresolved types
    // so if there are any we add an error
    for (_expr_id, (span, ty)) in type_checker.types() {
        if contains_infer(ty) {
            errors.push(Diagnostic::new(*span, DiagnosticKind::UnresolvedInfer));
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    type_checker.cycle_capable_types =
        cyclicity::analyze_cyclicity(&type_checker.ctx.struct_defs, &type_checker.ctx.enum_defs);

    Ok(type_checker.into_result())
}

fn flush_errors(
    errors: &mut Vec<Diagnostic>,
    type_checker: &mut TypeChecker,
) -> Result<(), Vec<Diagnostic>> {
    let (errs, warns): (Vec<_>, Vec<_>) = errors
        .drain(..)
        .partition(|d| d.kind.severity() == Severity::Error);
    type_checker.warnings.extend(warns);
    if errs.is_empty() { Ok(()) } else { Err(errs) }
}
