mod ast;
mod lexer;
mod parser;

pub fn run_program(program: &str) -> Result<ast::Program, String> {
    let tokens = lexer::tokenize(program)?;
    println!("tokens: {tokens:?}");
    let ast = parser::parse_ast(&tokens)?;
    println!("ast: {ast:?}");
    Ok(ast)
}
