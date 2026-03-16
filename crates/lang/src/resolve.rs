use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::ast::{Stmt, StmtNode};
use crate::span::Span;

#[derive(Debug)]
pub enum ImportError {
    FileNotFound { path: String, span: Span },
    ParseError { file_path: String },
    CircularImport { path: String, span: Span },
}

pub struct ModuleSource {
    pub path_key: Vec<String>,
    pub stmts: Vec<StmtNode>,
}

pub struct ResolveResult {
    pub modules: Vec<ModuleSource>,
}

pub(crate) fn resolve_imports(
    stmts: &[StmtNode],
    project_root: &Path,
) -> Result<ResolveResult, Vec<ImportError>> {
    let mut modules = vec![];
    let mut resolving: HashSet<Vec<String>> = HashSet::new();
    let mut resolved: HashSet<Vec<String>> = HashSet::new();
    let mut errors = vec![];

    collect_imports(
        stmts,
        project_root,
        &mut resolving,
        &mut resolved,
        &mut modules,
        &mut errors,
    );

    if errors.is_empty() {
        Ok(ResolveResult { modules })
    } else {
        Err(errors)
    }
}

fn collect_imports(
    stmts: &[StmtNode],
    project_root: &Path,
    resolving: &mut HashSet<Vec<String>>,
    resolved: &mut HashSet<Vec<String>>,
    modules: &mut Vec<ModuleSource>,
    errors: &mut Vec<ImportError>,
) {
    for stmt in stmts {
        let Stmt::Import(import_node) = &stmt.node else {
            continue;
        };

        let import = &import_node.node;
        let path_key: Vec<String> = import.path.iter().map(|id| id.to_string()).collect();

        if path_key.first().map(|s| s.as_str()) == Some("std") {
            continue;
        }

        // already fully resolved, deduplicate
        if resolved.contains(&path_key) {
            continue;
        }

        // currently on the DFS stack, this is a real cycle
        if resolving.contains(&path_key) {
            errors.push(ImportError::CircularImport {
                path: path_key.join("."),
                span: import_node.span,
            });
            continue;
        }

        resolving.insert(path_key.clone());

        let file_path = build_file_path(project_root, &path_key);

        let source = match std::fs::read_to_string(&file_path) {
            Ok(s) => s,
            Err(_) => {
                errors.push(ImportError::FileNotFound {
                    path: file_path.display().to_string(),
                    span: import_node.span,
                });
                resolving.remove(&path_key);
                continue;
            }
        };

        let file_path_str = file_path.display().to_string();
        let (module_ast, _tokens) = match crate::parse_source(&source, &file_path_str) {
            Ok(r) => r,
            Err(_) => {
                errors.push(ImportError::ParseError {
                    file_path: file_path_str,
                });
                resolving.remove(&path_key);
                continue;
            }
        };

        // recursively resolve imports declared inside this module
        collect_imports(
            &module_ast.stmts,
            project_root,
            resolving,
            resolved,
            modules,
            errors,
        );

        resolving.remove(&path_key);
        resolved.insert(path_key.clone());

        modules.push(ModuleSource {
            path_key,
            stmts: module_ast.stmts,
        });
    }
}

fn build_file_path(project_root: &Path, path_key: &[String]) -> PathBuf {
    let mut p = project_root.to_path_buf();
    for (i, segment) in path_key.iter().enumerate() {
        if i < path_key.len() - 1 {
            p.push(segment);
        } else {
            p.push(format!("{}.anv", segment));
        }
    }
    p
}
