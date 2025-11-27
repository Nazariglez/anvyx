mod ast;
mod error;
mod lexer;
mod parser;
mod span;
mod typecheck;

pub fn generate_ast(program: &str, file_path: &str) -> Result<ast::Program, String> {
    let tokens = match lexer::tokenize(program) {
        Ok(tokens) => tokens,
        Err(errors) => {
            error::report_lexer_errors(program, file_path, errors);
            return Err("Failed to tokenize program".to_string());
        }
    };

    let ast = match parser::parse_ast(&tokens) {
        Ok(ast) => ast,
        Err(errors) => {
            error::report_parse_errors(program, file_path, &tokens, errors);
            return Err("Failed to parse program".to_string());
        }
    };

    if let Err(errors) = typecheck::check_program(&ast) {
        error::report_typecheck_errors(program, file_path, &tokens, errors);
        return Err("Failed to typecheck program".to_string());
    }

    Ok(ast)
}

pub fn run_program(program: &str, file_path: &str) -> Result<String, String> {
    let _ast = generate_ast(program, file_path)?;
    Ok("output".to_string())
}
