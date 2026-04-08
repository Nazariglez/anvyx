use std::{collections::HashMap, fs, path::Path};

use anvyx_lang::LintConfig;

use crate::std_support::{collect_core, collect_std};

pub fn cmd(
    file: &Path,
    extern_meta: &HashMap<String, String>,
    lint: LintConfig,
) -> Result<(), String> {
    let program = fs::read_to_string(file).map_err(|e| format!("Failed to read file: {e}"))?;
    let file_path = file.to_string_lossy().to_string();
    let (std_sources, _) = collect_std();
    let (core_prelude, core_sources, _) = collect_core();

    let _ast = anvyx_lang::generate_ast_with_std(
        &program,
        &file_path,
        &core_prelude,
        extern_meta,
        &std_sources,
        &core_sources,
        lint,
    )?;
    Ok(())
}
