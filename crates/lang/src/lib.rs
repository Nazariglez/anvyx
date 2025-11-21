mod ast;
mod error;
mod lexer;
mod parser;
mod span;

pub fn run_program(program: &str) -> Result<ast::Program, String> {
    let tokens = lexer::tokenize(program)?;
    match parser::parse_ast(&tokens) {
        Ok(ast) => Ok(ast),
        Err(errors) => {
            error::report_parse_errors(program, &tokens, errors);
            Err("Failed to parse program".to_string())
        }
    }
}
