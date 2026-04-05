use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use crate::{
    StdModuleSource,
    ast::{Stmt, StmtNode},
    span::Span,
};

#[derive(Debug)]
pub enum ImportError {
    FileNotFound { path: String, span: Span },
    ParseError { file_path: String },
    UnknownStdModule { name: String, span: Span },
}

pub struct ModuleSource {
    pub path_key: Vec<String>,
    pub stmts: Vec<StmtNode>,
}

pub struct ResolveResult {
    pub module_groups: Vec<Vec<ModuleSource>>,
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
        let module_groups = topological_sort_sccs(modules);
        Ok(ResolveResult { module_groups })
    } else {
        Err(errors)
    }
}

#[allow(clippy::too_many_arguments)]
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
        let path_key: Vec<String> = import.path.iter().map(ToString::to_string).collect();

        if path_key.first().map(String::as_str) == Some("std") {
            if resolved.contains(&path_key) {
                continue;
            }

            let Some(module_name) = path_key.get(1) else {
                errors.push(ImportError::UnknownStdModule {
                    name: "std".to_string(),
                    span: import_node.span,
                });
                continue;
            };
            let module_name = module_name.as_str();

            let Some(source) = std_modules.get(module_name) else {
                errors.push(ImportError::UnknownStdModule {
                    name: module_name.to_string(),
                    span: import_node.span,
                });
                continue;
            };

            let file_label = format!("<std.{module_name}>");
            let Ok((module_ast, _)) = crate::parse_source(&source.anv_source, &file_label) else {
                errors.push(ImportError::ParseError {
                    file_path: file_label,
                });
                continue;
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

        if resolving.contains(&path_key) {
            continue;
        }

        resolving.insert(path_key.clone());

        let file_path = build_file_path(project_root, &path_key);

        let Ok(source) = std::fs::read_to_string(&file_path) else {
            errors.push(ImportError::FileNotFound {
                path: file_path.display().to_string(),
                span: import_node.span,
            });
            resolving.remove(&path_key);
            continue;
        };

        let file_path_str = file_path.display().to_string();
        let Ok((module_ast, _tokens)) = crate::parse_source(&source, &file_path_str) else {
            errors.push(ImportError::ParseError {
                file_path: file_path_str,
            });
            resolving.remove(&path_key);
            continue;
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
    if let Some((last, dirs)) = path_key.split_last() {
        for segment in dirs {
            p.push(segment);
        }
        p.push(format!("{last}.anv"));
    }
    p
}

fn topological_sort_sccs(modules: Vec<ModuleSource>) -> Vec<Vec<ModuleSource>> {
    if modules.is_empty() {
        return vec![];
    }

    let index: HashMap<Vec<String>, usize> = modules
        .iter()
        .enumerate()
        .map(|(i, m)| (m.path_key.clone(), i))
        .collect();

    let mut adj: Vec<Vec<usize>> = vec![vec![]; modules.len()];
    for (i, module) in modules.iter().enumerate() {
        for stmt in &module.stmts {
            let Stmt::Import(import_node) = &stmt.node else {
                continue;
            };
            let target: Vec<String> = import_node
                .node
                .path
                .iter()
                .map(ToString::to_string)
                .collect();
            if let Some(&j) = index.get(&target) {
                adj[i].push(j);
            }
        }
    }

    let sccs = tarjan_scc(&adj);

    let mut slots: Vec<Option<ModuleSource>> = modules.into_iter().map(Some).collect();
    sccs.into_iter()
        .map(|scc| scc.into_iter().map(|i| slots[i].take().unwrap()).collect())
        .collect()
}

fn tarjan_scc(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut index_counter = 0u32;
    let mut stack: Vec<usize> = vec![];
    let mut on_stack = vec![false; n];
    let mut indices = vec![u32::MAX; n];
    let mut lowlinks = vec![0u32; n];
    let mut sccs: Vec<Vec<usize>> = vec![];

    for i in 0..n {
        if indices[i] == u32::MAX {
            tarjan_visit(
                i,
                adj,
                &mut index_counter,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut lowlinks,
                &mut sccs,
            );
        }
    }

    sccs
}

fn tarjan_visit(
    v: usize,
    adj: &[Vec<usize>],
    index_counter: &mut u32,
    stack: &mut Vec<usize>,
    on_stack: &mut [bool],
    indices: &mut [u32],
    lowlinks: &mut [u32],
    sccs: &mut Vec<Vec<usize>>,
) {
    indices[v] = *index_counter;
    lowlinks[v] = *index_counter;
    *index_counter += 1;
    stack.push(v);
    on_stack[v] = true;

    for &w in &adj[v] {
        if indices[w] == u32::MAX {
            tarjan_visit(
                w,
                adj,
                index_counter,
                stack,
                on_stack,
                indices,
                lowlinks,
                sccs,
            );
            lowlinks[v] = lowlinks[v].min(lowlinks[w]);
        } else if on_stack[w] {
            lowlinks[v] = lowlinks[v].min(indices[w]);
        }
    }

    if lowlinks[v] == indices[v] {
        let mut scc = vec![];
        loop {
            let w = stack.pop().unwrap();
            on_stack[w] = false;
            scc.push(w);
            if w == v {
                break;
            }
        }
        sccs.push(scc);
    }
}
