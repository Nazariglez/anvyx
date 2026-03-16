use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    project: Project,
}

#[derive(Deserialize)]
struct Project {
    entry: String,
}

pub fn resolve_entry(file: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(path) = file {
        return Ok(path.to_path_buf());
    }

    let manifest_path = Path::new("anvyx.toml");
    if !manifest_path.exists() {
        return Err(
            "No file provided and no anvyx.toml found in the current directory".to_string(),
        );
    }

    let contents = fs::read_to_string(manifest_path)
        .map_err(|e| format!("Failed to read anvyx.toml: {e}"))?;
    let manifest: Manifest = toml::from_str(&contents)
        .map_err(|e| format!("Failed to parse anvyx.toml: {e}"))?;

    Ok(PathBuf::from(manifest.project.entry))
}
