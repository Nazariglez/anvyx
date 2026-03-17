use std::collections::HashMap;
use std::{fs, path::Path};

pub fn cmd(file: &Path) -> Result<(), String> {
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {}", e))?;
    let file_path = file.to_string_lossy().to_string();
    let _ast = anvyx_lang::generate_ast(&program, &file_path)?;
    println!("File checked successfully");
    Ok(())
}

pub fn cmd_with_externs(file: &Path, extern_meta: &HashMap<String, String>) -> Result<(), String> {
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {e}"))?;
    let file_path = file.to_string_lossy().to_string();
    let _ast = anvyx_lang::generate_ast_with_externs(&program, &file_path, extern_meta)?;
    println!("File checked successfully");
    Ok(())
}
