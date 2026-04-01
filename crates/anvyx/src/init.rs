use std::fs;
use std::path::Path;

const MAIN_ANV: &str = r#"fn main() {
    println("Hello, Anvyx!");
}
"#;

pub fn cmd(name: Option<&str>) -> Result<(), String> {
    let (target_dir, project_name) = if let Some(n) = name {
        (Path::new(n).to_path_buf(), n.to_string())
    } else {
        let cwd =
            std::env::current_dir().map_err(|e| format!("Failed to get current directory: {e}"))?;
        let project_name = cwd
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("project")
            .to_string();
        (cwd, project_name)
    };

    if target_dir.exists() {
        let is_non_empty = target_dir
            .read_dir()
            .map_err(|e| format!("Failed to read directory: {e}"))?
            .next()
            .is_some();
        if is_non_empty {
            return Err(format!("directory '{}' is not empty", target_dir.display()));
        }
    } else {
        fs::create_dir_all(&target_dir).map_err(|e| format!("Failed to create directory: {e}"))?;
    }

    let src_dir = target_dir.join("src");
    fs::create_dir_all(&src_dir).map_err(|e| format!("Failed to create src/ directory: {e}"))?;

    let manifest = format!("[project]\nname = \"{project_name}\"\nentry = \"src/main.anv\"\n");
    fs::write(target_dir.join("anvyx.toml"), manifest)
        .map_err(|e| format!("Failed to write anvyx.toml: {e}"))?;

    fs::write(src_dir.join("main.anv"), MAIN_ANV)
        .map_err(|e| format!("Failed to write src/main.anv: {e}"))?;

    crate::progress::status("Created", &format!("project '{project_name}'"));
    Ok(())
}
