mod build;
mod check;
mod init;
mod manifest;
mod progress;
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

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        progress::error(&e);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Run { file, backend } => {
            let manifest = manifest::parse_manifest()?;
            let path = match file {
                Some(f) => f,
                None => {
                    let m = manifest.as_ref().ok_or(
                        "No file provided and no anvyx.toml found in the current directory",
                    )?;
                    PathBuf::from(&m.project.entry)
                }
            };

            let has_externs = manifest.as_ref().is_some_and(|m| m.has_externs());
            if has_externs {
                let cwd = std::env::current_dir()
                    .map_err(|e| format!("Failed to get current directory: {e}"))?;
                let m = manifest.as_ref().unwrap();

                for name in m.externs.keys() {
                    progress::status("Loading", &format!("extern {name}..."));
                }
                let runner_dir = build::generate_runner_crate(&cwd, m)?;

                let spinner = progress::start_spinner("Compiling", "externs...");
                build::build_runner(&runner_dir)?;
                progress::finish_spinner(&spinner);

                progress::status("Resolving", "extern types...");
                build::extract_metadata(&cwd)?;

                progress::status("Checking", &format!("{}...", path.display()));
                progress::status("Running", &format!("{}...", path.display()));
                build::execute_runner(&cwd, &path, &backend)?;
            } else {
                progress::status("Checking", &format!("{}...", path.display()));
                progress::status("Running", &format!("{}...", path.display()));
                run::cmd(&path, &backend)?;
            }
        }
        Command::Check { file } => {
            let manifest = manifest::parse_manifest()?;
            let path = match file {
                Some(f) => f,
                None => {
                    let m = manifest.as_ref().ok_or(
                        "No file provided and no anvyx.toml found in the current directory",
                    )?;
                    PathBuf::from(&m.project.entry)
                }
            };

            let has_externs = manifest.as_ref().is_some_and(|m| m.has_externs());
            if has_externs {
                let cwd = std::env::current_dir()
                    .map_err(|e| format!("Failed to get current directory: {e}"))?;
                let m = manifest.as_ref().unwrap();

                for name in m.externs.keys() {
                    progress::status("Loading", &format!("extern {name}..."));
                }
                let runner_dir = build::generate_runner_crate(&cwd, m)?;

                let spinner = progress::start_spinner("Compiling", "externs...");
                build::build_runner(&runner_dir)?;
                progress::finish_spinner(&spinner);

                progress::status("Resolving", "extern types...");
                build::extract_metadata(&cwd)?;

                progress::status("Checking", &format!("{}...", path.display()));
                let extern_meta = build::read_metadata(&cwd, m)?;
                check::cmd_with_externs(&path, &extern_meta)?;
            } else {
                progress::status("Checking", &format!("{}...", path.display()));
                check::cmd(&path)?;
            }
            progress::status(
                "Finished",
                &format!("{} checked successfully", path.display()),
            );
        }
        Command::Init { name } => {
            init::cmd(name.as_deref())?;
        }
        Command::Build => {
            let manifest =
                manifest::parse_manifest()?.ok_or("anvyx build requires an anvyx.toml manifest")?;
            let cwd = std::env::current_dir()
                .map_err(|e| format!("Failed to get current directory: {e}"))?;
            let project_name = build::resolve_project_name(&manifest, &cwd);

            if manifest.has_externs() {
                for name in manifest.externs.keys() {
                    progress::status("Loading", &format!("extern {name}..."));
                }
                let runner_dir = build::generate_runner_crate(&cwd, &manifest)?;

                let spinner = progress::start_spinner("Compiling", "externs...");
                build::build_runner(&runner_dir)?;
                progress::finish_spinner(&spinner);

                progress::status("Resolving", "extern types...");
                build::extract_metadata(&cwd)?;

                let runner_dir = build::generate_build_runner_crate(&cwd, &manifest)?;

                let spinner = progress::start_spinner("Bundling", "distribution...");
                build::build_runner(&runner_dir)?;
                progress::finish_spinner(&spinner);
            } else {
                let runner_dir = build::generate_build_runner_crate(&cwd, &manifest)?;

                let spinner = progress::start_spinner("Bundling", "distribution...");
                build::build_runner(&runner_dir)?;
                progress::finish_spinner(&spinner);
            }

            progress::status("Assembling", "distribution...");
            let dist_dir = build::assemble_dist(&cwd, &project_name)?;
            build::bundle_sources(&cwd, &dist_dir, &manifest)?;
            progress::status("Finished", &format!("{}", dist_dir.display()));
        }
    }

    Ok(())
}
