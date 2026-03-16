mod build;
mod check;
mod init;
mod manifest;
mod run;

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
                build::execute_runner(&cwd, &path, &backend)?;
            } else {
                run::cmd(&path, &backend)?;
            }
        }
        Command::Check { file } => {
            let path = manifest::resolve_entry(file.as_deref())?;
            check::cmd(&path)?;
        }
        Command::Init { name } => {
            init::cmd(name.as_deref())?;
        }
    }

    Ok(())
}
