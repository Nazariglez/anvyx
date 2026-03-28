use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anvyx_lang::{Backend, run_program_with_std};

use crate::std_support::collect_std;

pub fn cmd(file: &Path, backend: &str) -> Result<(), String> {
    let backend = Backend::from_str(backend)?;
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {e}"))?;
    let file_path = file.to_string_lossy().to_string();
    let (std_sources, mut handlers) = collect_std();

    let core_mods = anvyx_core::core_modules();
    let core_source: String = core_mods.iter().map(|m| m.full_anv_source()).collect::<Vec<_>>().join("\n");
    for m in &core_mods {
        handlers.extend((m.handlers)());
    }

    let output = run_program_with_std(
        &program,
        &file_path,
        &core_source,
        backend,
        handlers,
        HashMap::new(),
        std_sources,
    )?;
    print!("{output}");
    Ok(())
}
