mod build;
mod check;
mod init;
mod manifest;
mod run;
mod std_support;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_PKG_NAME"),
    version = env!("CARGO_PKG_VERSION"),
    about = env!("CARGO_PKG_DESCRIPTION")
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(about = "Run an Anvyx program")]
    Run {
        file: Option<PathBuf>,
        #[arg(long, default_value = "vm")]
        backend: String,
    },
    #[command(about = "Check an Anvyx file")]
    Check { file: Option<PathBuf> },
    #[command(about = "Create a new Anvyx project")]
    Init { name: Option<String> },
    #[command(about = "Build an Anvyx project for distribution")]
    Build,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run { file, backend } => {
            let manifest = manifest::parse_manifest()?;
            let path = match file {
                Some(f) => f,
                None => {
                    let m = manifest
                        .as_ref()
                        .ok_or("No file provided and no anvyx.toml found in the current directory")?;
                    PathBuf::from(&m.project.entry)
                }
            };

            let has_externs = manifest.as_ref().is_some_and(|m| m.has_externs());
            if has_externs {
                let cwd = std::env::current_dir()
                    .map_err(|e| format!("Failed to get current directory: {e}"))?;
                let m = manifest.as_ref().unwrap();
                let runner_dir = build::generate_runner_crate(&cwd, m)?;
                build::build_runner(&runner_dir)?;
                build::extract_metadata(&cwd)?;
                build::execute_runner(&cwd, &path, &backend)?;
            } else {
                run::cmd(&path, &backend)?;
            }
        }
        Command::Check { file } => {
            let manifest = manifest::parse_manifest()?;
            let path = match file {
                Some(f) => f,
                None => {
                    let m = manifest
                        .as_ref()
                        .ok_or("No file provided and no anvyx.toml found in the current directory")?;
                    PathBuf::from(&m.project.entry)
                }
            };

            let has_externs = manifest.as_ref().is_some_and(|m| m.has_externs());
            if has_externs {
                let cwd = std::env::current_dir()
                    .map_err(|e| format!("Failed to get current directory: {e}"))?;
                let m = manifest.as_ref().unwrap();
                let runner_dir = build::generate_runner_crate(&cwd, m)?;
                build::build_runner(&runner_dir)?;
                build::extract_metadata(&cwd)?;
                let extern_meta = build::read_metadata(&cwd, m)?;
                check::cmd_with_externs(&path, &extern_meta)?;
            } else {
                check::cmd(&path)?;
            }
        }
        Command::Init { name } => {
            init::cmd(name.as_deref())?;
        }
        Command::Build => {
            let manifest = manifest::parse_manifest()?
                .ok_or("anvyx build requires an anvyx.toml manifest")?;
            let cwd = std::env::current_dir()
                .map_err(|e| format!("Failed to get current directory: {e}"))?;
            let project_name = build::resolve_project_name(&manifest, &cwd);

            if manifest.has_externs() {
                let runner_dir = build::generate_runner_crate(&cwd, &manifest)?;
                build::build_runner(&runner_dir)?;
                build::extract_metadata(&cwd)?;

                let runner_dir = build::generate_build_runner_crate(&cwd, &manifest)?;
                build::build_runner(&runner_dir)?;
            } else {
                let runner_dir = build::generate_build_runner_crate(&cwd, &manifest)?;
                build::build_runner(&runner_dir)?;
            }

            let dist_dir = build::assemble_dist(&cwd, &project_name)?;
            build::bundle_sources(&cwd, &dist_dir, &manifest)?;
            println!("Build complete: {}", dist_dir.display());
        }
    }

    Ok(())
}
