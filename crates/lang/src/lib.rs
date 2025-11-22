mod ast;
mod error;
mod lexer;
mod parser;
mod span;

pub fn generate_ast(program: &str) -> Result<ast::Program, String> {
    let tokens = match lexer::tokenize(program) {
        Ok(tokens) => tokens,
        Err(errors) => {
            error::report_lexer_errors(program, errors);
            return Err("Failed to tokenize program".to_string());
        }
    };
    match parser::parse_ast(&tokens) {
        Ok(ast) => Ok(ast),
        Err(errors) => {
            error::report_parse_errors(program, &tokens, errors);
            Err("Failed to parse program".to_string())
        }
    }
}

pub fn run_program(program: &str) -> Result<String, String> {
    let _ast = generate_ast(program)?;
    Ok("output".to_string())
}
