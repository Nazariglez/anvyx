use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use crate::manifest::Manifest;

const ANVYX_CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");

fn lang_crate_path() -> String {
    Path::new(ANVYX_CRATE_DIR)
        .parent()
        .unwrap()
        .join("lang")
        .to_string_lossy()
        .into_owned()
}

pub fn generate_runner_crate(project_root: &Path, manifest: &Manifest) -> Result<PathBuf, String> {
    for (name, entry) in &manifest.externs {
        let resolved = project_root.join(&entry.path);
        if !resolved.exists() {
            return Err(format!(
                "Extern provider '{name}' not found at path: {}",
                resolved.display()
            ));
        }
    }

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

pub fn execute_runner(project_root: &Path, entry_path: &Path, backend: &str) -> Result<(), String> {
    let binary = runner_binary_path(project_root);
    if !binary.exists() {
        return Err(format!("Runner binary not found at {}", binary.display()));
    }

    let metadata_dir = project_root.join("build/metadata");

    let status = process::Command::new(&binary)
        .arg(entry_path)
        .arg(backend)
        .arg(&metadata_dir)
        .status()
        .map_err(|e| format!("Failed to execute runner binary: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        let code = status.code().unwrap_or(1);
        Err(format!("Runner exited with code {code}"))
    }
}

pub fn build_runner(runner_dir: &Path) -> Result<(), String> {
    let manifest_path = runner_dir.join("Cargo.toml");
    let target_dir = runner_dir.join("target");
    let output = process::Command::new("cargo")
        .args(["build", "--release", "--manifest-path"])
        .arg(&manifest_path)
        .arg("--target-dir")
        .arg(&target_dir)
        // unset CARGO_TARGET_DIR so the explicit --target-dir takes effect.
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .map_err(|e| format!("Failed to run cargo build: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format_build_error(&stderr))
    }
}

pub fn extract_metadata(project_root: &Path) -> Result<(), String> {
    let binary = runner_binary_path(project_root);
    if !binary.exists() {
        return Err(format!("Runner binary not found at {}", binary.display()));
    }

    let output_dir = project_root.join("build/metadata");

    let status = process::Command::new(&binary)
        .arg("--metadata")
        .arg(&output_dir)
        .status()
        .map_err(|e| format!("Failed to run metadata extraction: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        let code = status.code().unwrap_or(1);
        Err(format!("Metadata extraction exited with code {code}"))
    }
}

fn format_build_error(stderr: &str) -> String {
    let mut msg = format!("Failed to build runner crate:\n{stderr}");
    if stderr.contains("anvyx_externs") {
        msg.push_str(
            "\nHint: each extern provider must export a \
            `pub fn anvyx_externs() -> HashMap<String, ExternHandler>` function",
        );
    }
    msg
}

fn generate_cargo_toml(project_root: &Path, manifest: &Manifest) -> String {
    // TODO: use `anvyx-lang = "x.y.z"` once published to crates.io
    let lang_path = lang_crate_path();

    let mut deps = format!("anvyx-lang = {{ path = \"{lang_path}\" }}\n");

    let mut entries: Vec<(&String, &crate::manifest::ExternEntry)> =
        manifest.externs.iter().collect();
    entries.sort_by_key(|(name, _)| *name);

    for (name, entry) in entries {
        let extern_path = resolve_relative_path(project_root, &entry.path);
        deps.push_str(&format!("{name} = {{ path = \"{extern_path}\" }}\n"));
    }

    format!(
        "[package]\nname = \"anvyx-runner\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[workspace]\n\n[dependencies]\n{deps}"
    )
}

fn generate_main_rs(manifest: &Manifest) -> String {
    let mut entries: Vec<&String> = manifest.externs.keys().collect();
    entries.sort();

    let mut extend_lines = String::new();
    for name in &entries {
        extend_lines.push_str(&format!("    externs.extend({name}::anvyx_externs());\n"));
    }

    let metadata_lines = generate_metadata_lines(&entries);
    let metadata_read_lines = generate_metadata_read_lines(&entries);

    format!(
        r#"use std::collections::HashMap;
use std::env;
use std::fs;

fn main() {{
    let args: Vec<String> = env::args().collect();

    if args.get(1).map(|s| s.as_str()) == Some("--metadata") {{
        let output_dir = args.get(2).unwrap_or_else(|| {{
            eprintln!("--metadata requires an output directory argument");
            std::process::exit(1);
        }});
        fs::create_dir_all(output_dir)
            .unwrap_or_else(|e| {{ eprintln!("Failed to create metadata dir: {{e}}"); std::process::exit(1); }});
{metadata_lines}
        return;
    }}

    let file_path = &args[1];
    let backend = args.get(2).map(|s| s.as_str()).unwrap_or("vm");
    let metadata_dir = args.get(3).map(|s| s.as_str()).unwrap_or("build/metadata");

    let mut extern_metadata: HashMap<String, String> = HashMap::new();
{metadata_read_lines}
    let source = fs::read_to_string(file_path)
        .unwrap_or_else(|e| {{ eprintln!("Failed to read file: {{e}}"); std::process::exit(1); }});

    let backend = anvyx_lang::Backend::from_str(backend)
        .unwrap_or_else(|e| {{ eprintln!("{{e}}"); std::process::exit(1); }});

    let mut externs: HashMap<String, anvyx_lang::ExternHandler> = HashMap::new();
{extend_lines}
    match anvyx_lang::run_program_with_externs(&source, file_path, backend, externs, extern_metadata) {{
        Ok(output) => print!("{{output}}"),
        Err(e) => {{ eprintln!("{{e}}"); std::process::exit(1); }}
    }}
}}
"#
    )
}

fn generate_metadata_lines(sorted_names: &[&String]) -> String {
    let mut lines = String::new();
    for name in sorted_names {
        lines.push_str(&format!(
            "        {{\n\
             \x20           let json = anvyx_lang::exports_to_json({name}::ANVYX_EXPORTS);\n\
             \x20           fs::write(format!(\"{{output_dir}}/{name}.json\"), json)\n\
             \x20               .unwrap_or_else(|e| {{ eprintln!(\"Failed to write metadata for '{name}': {{e}}\"); std::process::exit(1); }});\n\
             \x20       }}\n"
        ));
    }
    lines
}

fn generate_metadata_read_lines(sorted_names: &[&String]) -> String {
    let mut lines = String::new();
    for name in sorted_names {
        lines.push_str(&format!(
            "    {{\n\
             \x20       let json = fs::read_to_string(format!(\"{{metadata_dir}}/{name}.json\"))\n\
             \x20           .unwrap_or_else(|e| {{ eprintln!(\"Failed to read metadata for '{name}': {{e}}\"); std::process::exit(1); }});\n\
             \x20       extern_metadata.insert(\"{name}\".to_string(), json);\n\
             \x20   }}\n"
        ));
    }
    lines
}

pub fn read_metadata(
    project_root: &Path,
    manifest: &Manifest,
) -> Result<std::collections::HashMap<String, String>, String> {
    let metadata_dir = project_root.join("build/metadata");
    let mut result = std::collections::HashMap::new();
    for name in manifest.externs.keys() {
        let path = metadata_dir.join(format!("{name}.json"));
        let json = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read metadata for extern '{name}': {e}"))?;
        result.insert(name.clone(), json);
    }
    Ok(result)
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
        assert!(output.contains("[workspace]"));
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
        assert!(output.contains("extern_metadata"));
        assert!(output.contains("metadata_dir"));
        assert!(output.contains("engine.json"));
    }

    #[test]
    fn main_rs_multiple_externs() {
        let output = generate_main_rs(&manifest_two_externs());

        assert!(output.contains("externs.extend(engine::anvyx_externs());"));
        assert!(output.contains("externs.extend(audio::anvyx_externs());"));
        assert!(output.contains("engine.json"));
        assert!(output.contains("audio.json"));
    }

    #[test]
    fn main_rs_passes_extern_metadata_to_run() {
        let output = generate_main_rs(&manifest_one_extern());

        assert!(output.contains("extern_metadata"));
        assert!(output.contains("run_program_with_externs"));
        // the call should include both externs and extern_metadata
        let run_call = output.find("run_program_with_externs").unwrap();
        let call_snippet = &output[run_call..run_call + 120];
        assert!(call_snippet.contains("extern_metadata"), "run call should pass extern_metadata");
    }

    #[test]
    fn main_rs_reads_metadata_dir_from_arg() {
        let output = generate_main_rs(&manifest_one_extern());
        assert!(output.contains("args.get(3)"));
        assert!(output.contains("metadata_dir"));
    }

    #[test]
    fn generate_runner_crate_creates_files() {
        let tmp = std::env::temp_dir().join(format!("anvyx-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("my_externs/engine")).unwrap();

        let manifest = manifest_one_extern();
        let runner_dir = generate_runner_crate(&tmp, &manifest).unwrap();

        assert!(runner_dir.join("Cargo.toml").exists());
        assert!(runner_dir.join("src/main.rs").exists());

        let cargo_content = fs::read_to_string(runner_dir.join("Cargo.toml")).unwrap();
        assert!(cargo_content.contains("anvyx-runner"));
        assert!(cargo_content.contains("[workspace]"));
        assert!(cargo_content.contains("engine"));

        let main_content = fs::read_to_string(runner_dir.join("src/main.rs")).unwrap();
        assert!(main_content.contains("run_program_with_externs"));
        assert!(main_content.contains("engine::anvyx_externs"));
        assert!(main_content.contains("extern_metadata"));
        assert!(main_content.contains("metadata_dir"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn generate_runner_validates_extern_paths() {
        let tmp = std::env::temp_dir().join(format!("anvyx-validate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let mut externs = HashMap::new();
        externs.insert(
            "missing_crate".into(),
            ExternEntry {
                path: "does_not_exist".into(),
            },
        );
        let manifest = Manifest {
            project: Project {
                name: None,
                entry: "src/main.anv".into(),
            },
            externs,
        };

        let result = generate_runner_crate(&tmp, &manifest);

        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("not found at path"), "unexpected error: {msg}");
        assert!(msg.contains("missing_crate"), "unexpected error: {msg}");

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn build_runner_invalid_crate_returns_error() {
        let tmp = std::env::temp_dir().join(format!("anvyx-build-err-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        // invalid cargo.toml, missing [package] section
        fs::write(tmp.join("Cargo.toml"), "[invalid_section]\nfoo = \"bar\"\n").unwrap();

        let result = build_runner(&tmp);

        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("Failed to build runner crate"),
            "unexpected error: {msg}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn execute_runner_missing_binary_returns_error() {
        let tmp = std::env::temp_dir().join(format!("anvyx-exec-err-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let result = execute_runner(&tmp, Path::new("src/main.anv"), "vm");

        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("Runner binary not found"),
            "unexpected error: {msg}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn format_build_error_no_hint_when_no_anvyx_externs() {
        let msg = format_build_error("error[E0425]: cannot find function `something_else`");
        assert!(msg.contains("Failed to build runner crate"));
        assert!(!msg.contains("Hint:"));
    }

    #[test]
    fn format_build_error_adds_hint_when_anvyx_externs_mentioned() {
        let msg = format_build_error("error[E0425]: cannot find function `anvyx_externs`");
        assert!(msg.contains("Failed to build runner crate"));
        assert!(msg.contains("Hint:"));
        assert!(msg.contains("anvyx_externs()"));
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

        let tmp = std::env::temp_dir().join(format!("anvyx-integration-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("my_externs/engine")).unwrap();
        fs::create_dir_all(tmp.join("my_externs/audio")).unwrap();

        let runner_dir = generate_runner_crate(&tmp, &manifest).unwrap();

        let cargo_content = fs::read_to_string(runner_dir.join("Cargo.toml")).unwrap();
        assert!(cargo_content.contains("[workspace]"));
        assert!(cargo_content.contains("engine"));
        assert!(cargo_content.contains("my_externs/engine"));
        assert!(cargo_content.contains("audio"));
        assert!(cargo_content.contains("my_externs/audio"));

        let main_content = fs::read_to_string(runner_dir.join("src/main.rs")).unwrap();
        assert!(main_content.contains("externs.extend(audio::anvyx_externs());"));
        assert!(main_content.contains("externs.extend(engine::anvyx_externs());"));
        assert!(main_content.contains("extern_metadata"));
        assert!(main_content.contains("metadata_dir"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn read_metadata_missing_file_returns_error() {
        let tmp =
            std::env::temp_dir().join(format!("anvyx-meta-read-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let result = read_metadata(&tmp, &manifest_one_extern());

        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("Failed to read metadata for extern 'engine'"),
            "unexpected error: {msg}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn read_metadata_reads_json_files() {
        let tmp =
            std::env::temp_dir().join(format!("anvyx-meta-ok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let metadata_dir = tmp.join("build/metadata");
        fs::create_dir_all(&metadata_dir).unwrap();

        let json = r#"{"types":[],"functions":[]}"#;
        fs::write(metadata_dir.join("engine.json"), json).unwrap();

        let result = read_metadata(&tmp, &manifest_one_extern()).unwrap();
        assert_eq!(result.get("engine").map(|s| s.as_str()), Some(json));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn main_rs_metadata_mode_present() {
        let output = generate_main_rs(&manifest_one_extern());
        assert!(output.contains("--metadata"));
        assert!(output.contains("engine::ANVYX_EXPORTS"));
        assert!(output.contains("anvyx_lang::exports_to_json"));
        assert!(output.contains("engine.json"));
    }

    #[test]
    fn main_rs_metadata_mode_multiple_externs() {
        let output = generate_main_rs(&manifest_two_externs());
        assert!(output.contains("--metadata"));
        assert!(output.contains("engine::ANVYX_EXPORTS"));
        assert!(output.contains("audio::ANVYX_EXPORTS"));
        assert!(output.contains("engine.json"));
        assert!(output.contains("audio.json"));
    }

    #[test]
    fn main_rs_metadata_mode_no_externs() {
        // With no externs, the --metadata block is still present but writes nothing
        let output = generate_main_rs(&manifest_no_externs());
        assert!(output.contains("--metadata"));
        assert!(!output.contains("ANVYX_EXPORTS"));
    }

    #[test]
    fn main_rs_metadata_and_run_both_present() {
        let output = generate_main_rs(&manifest_one_extern());
        // both metadata extraction and normal run paths exist
        assert!(output.contains("--metadata"));
        assert!(output.contains("run_program_with_externs"));
    }

    #[test]
    fn extract_metadata_missing_binary_returns_error() {
        let tmp = std::env::temp_dir().join(format!("anvyx-meta-err-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let result = extract_metadata(&tmp);

        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("Runner binary not found"),
            "unexpected error: {msg}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn generate_runner_crate_includes_metadata_mode() {
        let tmp = std::env::temp_dir().join(format!("anvyx-meta-gen-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("my_externs/engine")).unwrap();

        let runner_dir = generate_runner_crate(&tmp, &manifest_one_extern()).unwrap();
        let main_content = fs::read_to_string(runner_dir.join("src/main.rs")).unwrap();

        assert!(main_content.contains("--metadata"));
        assert!(main_content.contains("engine::ANVYX_EXPORTS"));
        assert!(main_content.contains("engine.json"));

        let _ = fs::remove_dir_all(&tmp);
    }
}
