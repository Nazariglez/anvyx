mod ast;
mod builtin;
mod error;
mod hir;
mod lexer;
mod lower;
mod parser;
mod span;
mod typecheck;
mod vm;

pub use anvyx_macros::export_fn;
pub use anvyx_macros::provider;
pub use vm::ExternHandler;
pub use vm::RuntimeError;
pub use vm::Value;

#[cfg(test)]
mod test_helpers;

pub(crate) const CORE_PRELUDE: &str = include_str!("../core/prelude.anv");

fn parse_source(
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

fn analyze(
    program: &str,
    file_path: &str,
) -> Result<
    (
        ast::Program,
        typecheck::TypeChecker,
        Vec<lexer::SpannedToken>,
    ),
    String,
> {
    let (prelude_ast, _) = parse_source(CORE_PRELUDE, "<prelude>")
        .map_err(|_| "Failed to parse prelude (internal error)".to_string())?;

    let (user_ast, user_tokens) = parse_source(program, file_path)?;

    let mut combined_stmts = prelude_ast.stmts;
    combined_stmts.extend(user_ast.stmts);
    let combined = ast::Program {
        stmts: combined_stmts,
    };

    let tcx = match typecheck::check_program(&combined) {
        Ok(tcx) => tcx,
        Err(errors) => {
            error::report_typecheck_errors(program, file_path, &user_tokens, errors);
            return Err("Failed to typecheck program".to_string());
        }
    };

    Ok((combined, tcx, user_tokens))
}

pub fn generate_ast(program: &str, file_path: &str) -> Result<ast::Program, String> {
    let (ast, _, _) = analyze(program, file_path)?;
    Ok(ast)
}

pub(crate) fn generate_hir(program: &str, file_path: &str) -> Result<hir::Program, String> {
    let (ast, tcx, _) = analyze(program, file_path)?;
    lower::lower_program(&ast, &tcx).map_err(|e| format!("Lowering error: {e}"))
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
    run_program_with_externs(program, file_path, backend, std::collections::HashMap::new())
}

pub fn run_program_with_externs(
    program: &str,
    file_path: &str,
    backend: Backend,
    externs: std::collections::HashMap<String, ExternHandler>,
) -> Result<String, String> {
    let hir = generate_hir(program, file_path)?;
    match backend {
        Backend::Vm => vm::run_with_externs(&hir, externs),
        Backend::Transpiler => Err("Transpiler backend is not yet implemented".to_string()),
    }
}
