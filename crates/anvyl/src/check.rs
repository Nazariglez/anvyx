use anvyl_lang::generate_ast;
use std::{fs, path::Path};

pub fn cmd(file: &Path) -> Result<(), String> {
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {}", e))?;
    let _ast = generate_ast(&program)?;
    println!("File checked successfully");
    Ok(())
}
