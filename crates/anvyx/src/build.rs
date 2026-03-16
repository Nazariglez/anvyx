use std::fs;
use std::path::{Path, PathBuf};

use crate::manifest::Manifest;

pub fn generate_runner_crate(project_root: &Path, manifest: &Manifest) -> Result<PathBuf, String> {
    let runner_dir = project_root.join("build/runner");
    let src_dir = runner_dir.join("src");

    fs::create_dir_all(&src_dir)
        .map_err(|e| format!("Failed to create runner crate directory: {e}"))?;

    let cargo_toml = generate_cargo_toml(project_root, manifest);
    fs::write(runner_dir.join("Cargo.toml"), cargo_toml)
        .map_err(|e| format!("Failed to write runner Cargo.toml: {e}"))?;

    let main_rs = generate_main_rs(manifest);
    fs::write(src_dir.join("main.rs"), main_rs)
        .map_err(|e| format!("Failed to write runner src/main.rs: {e}"))?;

    Ok(runner_dir)
}

pub fn runner_binary_path(project_root: &Path) -> PathBuf {
    project_root.join("build/runner/target/release/anvyx-runner")
}

fn generate_cargo_toml(project_root: &Path, manifest: &Manifest) -> String {
    // TODO: use `anvyx-lang = "x.y.z"` once published to crates.io
    // for now, compute the path to crates/lang relative to build/runner/
    let lang_path = resolve_relative_path(project_root, "crates/lang");

    let mut deps = format!("anvyx-lang = {{ path = \"{lang_path}\" }}\n");

    let mut entries: Vec<(&String, &crate::manifest::ExternEntry)> =
        manifest.externs.iter().collect();
    entries.sort_by_key(|(name, _)| *name);

    for (name, entry) in entries {
        let extern_path = resolve_relative_path(project_root, &entry.path);
        deps.push_str(&format!("{name} = {{ path = \"{extern_path}\" }}\n"));
    }

    format!(
        "[package]\nname = \"anvyx-runner\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\n{deps}"
    )
}

fn generate_main_rs(manifest: &Manifest) -> String {
    let mut extend_lines = String::new();
    let mut entries: Vec<&String> = manifest.externs.keys().collect();
    entries.sort();

    for name in entries {
        extend_lines.push_str(&format!("    externs.extend({name}::anvyx_externs());\n"));
    }

    format!(
        r#"use std::collections::HashMap;
use std::env;
use std::fs;

fn main() {{
    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];
    let backend = args.get(2).map(|s| s.as_str()).unwrap_or("vm");

    let source = fs::read_to_string(file_path)
        .unwrap_or_else(|e| {{ eprintln!("Failed to read file: {{e}}"); std::process::exit(1); }});

    let backend = anvyx_lang::Backend::from_str(backend)
        .unwrap_or_else(|e| {{ eprintln!("{{e}}"); std::process::exit(1); }});

    let mut externs: HashMap<String, anvyx_lang::ExternHandler> = HashMap::new();
{extend_lines}
    match anvyx_lang::run_program_with_externs(&source, file_path, backend, externs) {{
        Ok(output) => print!("{{output}}"),
        Err(e) => {{ eprintln!("{{e}}"); std::process::exit(1); }}
    }}
}}
"#
    )
}

fn resolve_relative_path(project_root: &Path, rel_to_root: &str) -> String {
    let abs = project_root.join(rel_to_root);
    // build/runner/ is 2 levels below project_root, so prefix with ../../
    let from_runner = format!("../../{rel_to_root}");

    // if the absolute path exists, prefer it so Cargo can verify it; otherwise
    // fall back to the relative form (covers tests using temp dirs).
    if abs.exists() {
        abs.to_string_lossy().into_owned()
    } else {
        from_runner
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::manifest::{ExternEntry, Manifest, Project};

    fn manifest_no_externs() -> Manifest {
        Manifest {
            project: Project {
                name: None,
                entry: "src/main.anv".into(),
            },
            externs: HashMap::new(),
        }
    }

    fn manifest_one_extern() -> Manifest {
        let mut externs = HashMap::new();
        externs.insert(
            "engine".into(),
            ExternEntry {
                path: "my_externs/engine".into(),
            },
        );
        Manifest {
            project: Project {
                name: None,
                entry: "src/main.anv".into(),
            },
            externs,
        }
    }

    fn manifest_two_externs() -> Manifest {
        let mut externs = HashMap::new();
        externs.insert(
            "engine".into(),
            ExternEntry {
                path: "my_externs/engine".into(),
            },
        );
        externs.insert(
            "audio".into(),
            ExternEntry {
                path: "my_externs/audio".into(),
            },
        );
        Manifest {
            project: Project {
                name: None,
                entry: "src/main.anv".into(),
            },
            externs,
        }
    }

    #[test]
    fn cargo_toml_no_externs() {
        let root = Path::new("/fake/project");
        let output = generate_cargo_toml(root, &manifest_no_externs());

        assert!(output.contains("[package]"));
        assert!(output.contains("name = \"anvyx-runner\""));
        assert!(output.contains("anvyx-lang"));
        assert!(!output.contains("engine"));
        assert!(!output.contains("audio"));
    }

    #[test]
    fn cargo_toml_one_extern() {
        let root = Path::new("/fake/project");
        let output = generate_cargo_toml(root, &manifest_one_extern());

        assert!(output.contains("engine"));
        assert!(output.contains("my_externs/engine"));
    }

    #[test]
    fn cargo_toml_multiple_externs() {
        let root = Path::new("/fake/project");
        let output = generate_cargo_toml(root, &manifest_two_externs());

        assert!(output.contains("engine"));
        assert!(output.contains("my_externs/engine"));
        assert!(output.contains("audio"));
        assert!(output.contains("my_externs/audio"));
    }

    #[test]
    fn main_rs_no_externs() {
        let output = generate_main_rs(&manifest_no_externs());

        assert!(output.contains("run_program_with_externs"));
        assert!(!output.contains("extend"));
    }

    #[test]
    fn main_rs_one_extern() {
        let output = generate_main_rs(&manifest_one_extern());

        assert!(output.contains("externs.extend(engine::anvyx_externs());"));
    }

    #[test]
    fn main_rs_multiple_externs() {
        let output = generate_main_rs(&manifest_two_externs());

        assert!(output.contains("externs.extend(engine::anvyx_externs());"));
        assert!(output.contains("externs.extend(audio::anvyx_externs());"));
    }

    #[test]
    fn generate_runner_crate_creates_files() {
        let tmp = std::env::temp_dir().join(format!("anvyx-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);

        let manifest = manifest_one_extern();
        let runner_dir = generate_runner_crate(&tmp, &manifest).unwrap();

        assert!(runner_dir.join("Cargo.toml").exists());
        assert!(runner_dir.join("src/main.rs").exists());

        let cargo_content = fs::read_to_string(runner_dir.join("Cargo.toml")).unwrap();
        assert!(cargo_content.contains("anvyx-runner"));
        assert!(cargo_content.contains("engine"));

        let main_content = fs::read_to_string(runner_dir.join("src/main.rs")).unwrap();
        assert!(main_content.contains("run_program_with_externs"));
        assert!(main_content.contains("engine::anvyx_externs"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn integration_manifest_to_runner_crate() {
        let toml_src = r#"
            [project]
            name = "my_game"
            entry = "src/main.anv"

            [externs.engine]
            path = "my_externs/engine"

            [externs.audio]
            path = "my_externs/audio"
        "#;

        let manifest: Manifest = toml::from_str(toml_src).unwrap();

        assert!(manifest.has_externs());
        assert_eq!(manifest.externs.len(), 2);

        let tmp =
            std::env::temp_dir().join(format!("anvyx-integration-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);

        let runner_dir = generate_runner_crate(&tmp, &manifest).unwrap();

        let cargo_content = fs::read_to_string(runner_dir.join("Cargo.toml")).unwrap();
        assert!(cargo_content.contains("engine"));
        assert!(cargo_content.contains("my_externs/engine"));
        assert!(cargo_content.contains("audio"));
        assert!(cargo_content.contains("my_externs/audio"));

        let main_content = fs::read_to_string(runner_dir.join("src/main.rs")).unwrap();
        assert!(main_content.contains("externs.extend(audio::anvyx_externs());"));
        assert!(main_content.contains("externs.extend(engine::anvyx_externs());"));

        let _ = fs::remove_dir_all(&tmp);
    }
}
