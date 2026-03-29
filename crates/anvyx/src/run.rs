use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anvyx_lang::{Backend, CoreSource, run_program_with_std};

use crate::std_support::{collect_core, collect_std};

pub fn cmd(file: &Path, backend: &str) -> Result<(), String> {
    let backend = backend.parse::<Backend>()?;
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {e}"))?;
    let file_path = file.to_string_lossy().to_string();
    let (std_sources, mut handlers) = collect_std();
    let (core_prelude, core_sources, core_handlers) = collect_core();
    handlers.extend(core_handlers);

    let core = CoreSource {
        prelude: core_prelude,
        modules: core_sources,
    };
    let output = run_program_with_std(
        &program,
        &file_path,
        backend,
        handlers,
        HashMap::new(),
        std_sources,
        core,
    )?;
    print!("{output}");
    Ok(())
}
