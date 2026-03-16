use anvyx_lang::{Backend, run_program};
use std::fs;
use std::path::Path;

pub fn cmd(file: &Path, backend: &str) -> Result<(), String> {
    let backend = Backend::from_str(backend)?;
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {e}"))?;
    let file_path = file.to_string_lossy().to_string();
    let output = run_program(&program, &file_path, backend)?;
    print!("{output}");
    Ok(())
}
