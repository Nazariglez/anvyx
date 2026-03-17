use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::StdModuleSource;
use crate::ast::{Stmt, StmtNode};
use crate::span::Span;

#[derive(Debug)]
pub enum ImportError {
    FileNotFound { path: String, span: Span },
    ParseError { file_path: String },
    CircularImport { path: String, span: Span },
    UnknownStdModule { name: String, span: Span },
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
    extern_names: &HashSet<String>,
    std_modules: &HashMap<String, StdModuleSource>,
) -> Result<ResolveResult, Vec<ImportError>> {
    let mut modules = vec![];
    let mut resolving: HashSet<Vec<String>> = HashSet::new();
    let mut resolved: HashSet<Vec<String>> = HashSet::new();
    let mut errors = vec![];

    collect_imports(
        stmts,
        project_root,
        extern_names,
        std_modules,
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
    extern_names: &HashSet<String>,
    std_modules: &HashMap<String, StdModuleSource>,
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
            if resolved.contains(&path_key) {
                continue;
            }

            let module_name = match path_key.get(1) {
                Some(name) => name.as_str(),
                None => {
                    errors.push(ImportError::UnknownStdModule {
                        name: "std".to_string(),
                        span: import_node.span,
                    });
                    continue;
                }
            };

            let Some(source) = std_modules.get(module_name) else {
                errors.push(ImportError::UnknownStdModule {
                    name: module_name.to_string(),
                    span: import_node.span,
                });
                continue;
            };

            let file_label = format!("<std.{module_name}>");
            let (module_ast, _) = match crate::parse_source(&source.anv_source, &file_label) {
                Ok(r) => r,
                Err(_) => {
                    errors.push(ImportError::ParseError {
                        file_path: file_label,
                    });
                    continue;
                }
            };

            resolved.insert(path_key.clone());
            modules.push(ModuleSource {
                path_key,
                stmts: module_ast.stmts,
            });
            continue;
        }

        // extern provider names are resolved separately from metadata, not as local files
        let is_extern_provider = path_key
            .first()
            .is_some_and(|s| extern_names.contains(s.as_str()));
        if is_extern_provider {
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
            extern_names,
            std_modules,
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
