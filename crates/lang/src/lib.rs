mod ast;
mod builtin;
mod error;
mod hir;
mod lexer;
mod lower;
mod parser;
mod resolve;
mod span;
mod typecheck;
mod vm;

pub mod metadata;

pub use anvyx_macros::{export_fn, provider};
pub use metadata::{ExternDecl, ExternFuncMeta, ExternProviderMeta, exports_to_json, parse_provider_json};
pub use vm::{ExternHandler, RuntimeError, Value};

#[cfg(test)]
mod test_helpers;

pub(crate) const CORE_PRELUDE: &str = include_str!("../core/prelude.anv");

pub(crate) fn parse_source(
    source: &str,
    file_path: &str,
) -> Result<(ast::Program, Vec<lexer::SpannedToken>), String> {
    let tokens = match lexer::tokenize(source) {
        Ok(tokens) => tokens,
        Err(errors) => {
            error::report_lexer_errors(source, file_path, errors);
            return Err("Failed to tokenize program".to_string());
        }
    };

    let ast = match parser::parse_ast(&tokens) {
        Ok(ast) => ast,
        Err(errors) => {
            error::report_parse_errors(source, file_path, &tokens, errors);
            return Err("Failed to parse program".to_string());
        }
    };

    Ok((ast, tokens))
}

type AnalyzeResult = Result<
    (
        ast::Program,
        typecheck::TypeChecker,
        Vec<lexer::SpannedToken>,
        Vec<(Vec<String>, Vec<ast::StmtNode>)>,
    ),
    String,
>;

fn analyze_with_extern_meta(
    program: &str,
    file_path: &str,
    extern_metadata: &std::collections::HashMap<String, String>,
) -> AnalyzeResult {
    use std::collections::HashSet;

    let (prelude_ast, _) = parse_source(CORE_PRELUDE, "<prelude>")
        .map_err(|_| "Failed to parse prelude (internal error)".to_string())?;

    let (user_ast, user_tokens) = parse_source(program, file_path)?;

    // parse extern metadata JSON and extract provider names for resolve skip list
    let mut parsed_providers = std::collections::HashMap::new();
    for (name, json) in extern_metadata {
        let meta = metadata::parse_provider_json(json)
            .map_err(|e| format!("Failed to parse metadata for extern '{name}': {e}"))?;
        parsed_providers.insert(name.clone(), meta);
    }
    let extern_names: HashSet<String> = parsed_providers.keys().cloned().collect();

    // resolve local file imports, skipping extern provider names
    let mut module_list = if file_path.starts_with('<') {
        vec![]
    } else {
        let project_root = {
            let p = std::path::Path::new(file_path);
            p.parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .to_path_buf()
        };

        match resolve::resolve_imports(&user_ast.stmts, &project_root, &extern_names) {
            Ok(result) => result
                .modules
                .into_iter()
                .map(|module| (module.path_key, module.stmts))
                .collect(),
            Err(errors) => {
                error::report_import_errors(program, file_path, &user_tokens, &errors);
                return Err("Failed to resolve imports".to_string());
            }
        }
    };

    // convert extern metadata into synthetic ExternFunc stmts and append to module_list
    for (name, meta) in &parsed_providers {
        let stmts = metadata::metadata_to_extern_stmts(meta)
            .map_err(|e| format!("Failed to create extern stmts for '{name}': {e}"))?;
        module_list.push((vec![name.clone()], stmts));
    }

    let mut combined_stmts = prelude_ast.stmts;
    combined_stmts.extend(user_ast.stmts);
    let combined = ast::Program { stmts: combined_stmts };

    let tcx = match typecheck::check_program_with_modules(&combined, &module_list) {
        Ok(tcx) => tcx,
        Err(errors) => {
            error::report_typecheck_errors(program, file_path, &user_tokens, errors);
            return Err("Failed to typecheck program".to_string());
        }
    };

    Ok((combined, tcx, user_tokens, module_list))
}

pub fn generate_ast(program: &str, file_path: &str) -> Result<ast::Program, String> {
    generate_ast_with_externs(program, file_path, &std::collections::HashMap::new())
}

pub fn generate_ast_with_externs(
    program: &str,
    file_path: &str,
    extern_metadata: &std::collections::HashMap<String, String>,
) -> Result<ast::Program, String> {
    let (ast, _, _, _) = analyze_with_extern_meta(program, file_path, extern_metadata)?;
    Ok(ast)
}

pub(crate) fn generate_hir(program: &str, file_path: &str) -> Result<hir::Program, String> {
    generate_hir_with_externs(program, file_path, &std::collections::HashMap::new())
}

pub(crate) fn generate_hir_with_externs(
    program: &str,
    file_path: &str,
    extern_metadata: &std::collections::HashMap<String, String>,
) -> Result<hir::Program, String> {
    let (ast, tcx, _, module_list) = analyze_with_extern_meta(program, file_path, extern_metadata)?;
    lower::lower_program(&ast, &tcx, &module_list).map_err(|e| format!("Lowering error: {e}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Vm,
    Transpiler,
}

impl Backend {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "vm" => Ok(Self::Vm),
            "transpiler" => Ok(Self::Transpiler),
            _ => Err(format!(
                "Unknown backend: '{s}'. Expected 'vm' or 'transpiler'"
            )),
        }
    }
}

pub fn run_program(program: &str, file_path: &str, backend: Backend) -> Result<String, String> {
    run_program_with_externs(
        program,
        file_path,
        backend,
        std::collections::HashMap::new(),
        std::collections::HashMap::new(),
    )
}

pub fn run_program_with_externs(
    program: &str,
    file_path: &str,
    backend: Backend,
    externs: std::collections::HashMap<String, ExternHandler>,
    extern_metadata: std::collections::HashMap<String, String>,
) -> Result<String, String> {
    let hir = generate_hir_with_externs(program, file_path, &extern_metadata)?;
    match backend {
        Backend::Vm => vm::run_with_externs(&hir, externs),
        Backend::Transpiler => Err("Transpiler backend is not yet implemented".to_string()),
    }
}

#[cfg(test)]
mod extern_import_tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_metadata() -> HashMap<String, String> {
        let json = r#"{"types":[],"functions":[
            {"name":"add","params":[["a","int"],["b","int"]],"ret":"int"},
            {"name":"greet","params":[["name","string"]],"ret":"string"}
        ]}"#;
        let mut m = HashMap::new();
        m.insert("my_extern".to_string(), json.to_string());
        m
    }

    #[test]
    fn selective_import_typechecks() {
        let src = "import my_extern { add };\nfn main() { let x = add(3, 4); }";
        let result = analyze_with_extern_meta(src, "<test>", &sample_metadata());
        assert!(result.is_ok(), "unexpected error: {:?}", result.err());
    }

    #[test]
    fn qualified_import_typechecks() {
        let src = "import my_extern;\nfn main() { let x = my_extern.add(3, 4); }";
        let result = analyze_with_extern_meta(src, "<test>", &sample_metadata());
        assert!(result.is_ok(), "unexpected error: {:?}", result.err());
    }

    #[test]
    fn wildcard_import_typechecks() {
        let src = "import my_extern { * };\nfn main() { let x = add(3, 4); }";
        let result = analyze_with_extern_meta(src, "<test>", &sample_metadata());
        assert!(result.is_ok(), "unexpected error: {:?}", result.err());
    }

    #[test]
    fn alias_import_typechecks() {
        let src = "import my_extern { add as plus };\nfn main() { let x = plus(3, 4); }";
        let result = analyze_with_extern_meta(src, "<test>", &sample_metadata());
        assert!(result.is_ok(), "unexpected error: {:?}", result.err());
    }

    #[test]
    fn module_alias_import_typechecks() {
        let src = "import my_extern as ext;\nfn main() { let x = ext.add(3, 4); }";
        let result = analyze_with_extern_meta(src, "<test>", &sample_metadata());
        assert!(result.is_ok(), "unexpected error: {:?}", result.err());
    }

    #[test]
    fn unknown_member_error() {
        let src = "import my_extern { nonexistent };\nfn main() {}";
        let result = analyze_with_extern_meta(src, "<test>", &sample_metadata());
        assert!(result.is_err());
    }

    #[test]
    fn wrong_arg_type_error() {
        let src = "import my_extern { add };\nfn main() { let x = add(true, 4); }";
        let result = analyze_with_extern_meta(src, "<test>", &sample_metadata());
        assert!(result.is_err());
    }

    #[test]
    fn hir_contains_extern_decls() {
        let src = "import my_extern { add, greet };\nfn main() { let x = add(1, 2); }";
        let hir = generate_hir_with_externs(src, "<test>", &sample_metadata());
        assert!(hir.is_ok(), "unexpected error: {:?}", hir.err());
        let program = hir.unwrap();
        assert!(!program.externs.is_empty(), "expected extern decls in HIR");
    }

    #[test]
    fn no_extern_meta_still_works() {
        let src = "fn main() { let x = 1; }";
        let result = analyze_with_extern_meta(src, "<test>", &HashMap::new());
        assert!(result.is_ok());
    }
}
