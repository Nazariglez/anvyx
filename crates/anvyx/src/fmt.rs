use std::{
    fs,
    io::{self, Read as _},
    path::{Path, PathBuf},
};

use walkdir::WalkDir;

use crate::progress;

pub fn cmd(path: Option<PathBuf>, check: bool, stdin: bool) -> Result<(), String> {
    if stdin {
        return format_stdin();
    }

    let target = match path {
        Some(p) => p,
        None => {
            std::env::current_dir().map_err(|e| format!("Failed to get current directory: {e}"))?
        }
    };

    let process_files = |buff: &[PathBuf]| -> Result<(), String> {
        if check {
            check_files(buff)
        } else {
            format_files(buff)
        }
    };

    let is_file = target.is_file();
    if is_file {
        return process_files(&[target]);
    }

    let is_dir = target.is_dir();
    if is_dir {
        let files = collect_anv_files(&target)?;
        if files.is_empty() {
            progress::status("Formatted", "no .anv files found");
            return Ok(());
        }
        return process_files(&files);
    }

    Err(format!("Path does not exist: {}", target.display()))
}

fn format_stdin() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("Failed to read stdin: {e}"))?;
    let formatted = anvyx_fmt::format_source(&input).map_err(|e| format!("{e}"))?;
    print!("{formatted}");
    Ok(())
}

fn collect_anv_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = vec![];
    for entry in WalkDir::new(dir)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
    {
        let entry = entry.map_err(|e| format!("Failed to walk directory: {e}"))?;
        if entry.file_type().is_file()
            && let Some(ext) = entry.path().extension()
            && ext == "anv"
        {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

fn is_hidden(entry: &walkdir::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|s| s.starts_with('.'))
}

fn format_files(files: &[PathBuf]) -> Result<(), String> {
    let mut changed = 0;
    let mut errors = 0;

    for file in files {
        let source = fs::read_to_string(file)
            .map_err(|e| format!("Failed to read {}: {e}", file.display()))?;
        match anvyx_fmt::format_source(&source) {
            Ok(formatted) => {
                if formatted != source {
                    fs::write(file, &formatted)
                        .map_err(|e| format!("Failed to write {}: {e}", file.display()))?;
                    progress::status("Formatted", &file.display().to_string());
                    changed += 1;
                }
            }
            Err(e) => {
                eprintln!("Warning: skipping {} ({})", file.display(), e);
                errors += 1;
            }
        }
    }

    let total = files.len() - errors;
    if changed == 0 {
        progress::status("Finished", &format!("{total} file(s) already formatted"));
    } else {
        progress::status(
            "Finished",
            &format!("{changed} file(s) formatted, {} unchanged", total - changed),
        );
    }

    Ok(())
}

fn check_files(files: &[PathBuf]) -> Result<(), String> {
    let mut unformatted: Vec<&PathBuf> = vec![];
    let mut errors = 0;

    for file in files {
        let source = fs::read_to_string(file)
            .map_err(|e| format!("Failed to read {}: {e}", file.display()))?;
        match anvyx_fmt::format_source(&source) {
            Ok(formatted) => {
                if formatted != source {
                    unformatted.push(file);
                }
            }
            Err(e) => {
                eprintln!("Warning: skipping {} ({})", file.display(), e);
                errors += 1;
            }
        }
    }

    if unformatted.is_empty() {
        let total = files.len() - errors;
        progress::status("Finished", &format!("{total} file(s) correctly formatted"));
        return Ok(());
    }

    for file in &unformatted {
        eprintln!("  needs formatting: {}", file.display());
    }

    Err(format!("{} file(s) need formatting", unformatted.len()))
}
