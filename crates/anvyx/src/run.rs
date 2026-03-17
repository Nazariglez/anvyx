use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anvyx_lang::{Backend, run_program_with_std};

use crate::std_support::collect_std;

pub fn cmd(file: &Path, backend: &str) -> Result<(), String> {
    let backend = Backend::from_str(backend)?;
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {e}"))?;
    let file_path = file.to_string_lossy().to_string();
    let (std_sources, std_handlers) = collect_std();
    let output = run_program_with_std(
        &program,
        &file_path,
        backend,
        std_handlers,
        HashMap::new(),
        std_sources,
    )?;
    print!("{output}");
    Ok(())
}
