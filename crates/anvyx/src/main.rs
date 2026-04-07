mod build;
mod check;
mod clean;
mod init;
mod manifest;
mod progress;
mod run;
mod std_support;

use std::{collections::HashMap, path::PathBuf};

use clap::{Parser, Subcommand};

use crate::manifest::Manifest;

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
        #[arg(long)]
        release: bool,
    },
    #[command(about = "Check an Anvyx file")]
    Check { file: Option<PathBuf> },
    #[command(about = "Create a new Anvyx project")]
    Init { name: Option<String> },
    #[command(about = "Build an Anvyx project for distribution")]
    Build {
        #[arg(long)]
        release: bool,
    },
    #[command(about = "Remove build cache")]
    Clean,
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
        Command::Run {
            file,
            backend,
            release,
        } => {
            let manifest = manifest::parse_manifest()?;
            let path = resolve_entry(file, manifest.as_ref())?;

            let has_externs = manifest.as_ref().is_some_and(Manifest::has_externs);
            if has_externs {
                let manifest = manifest.as_ref().unwrap();
                let ctx = prepare_externs(manifest)?;

                progress::status("Checking", &format!("{}...", path.display()));
                progress::status("Running", &format!("{}...", path.display()));
                build::execute_runner(&ctx.cwd, &path, &backend, release)?;
            } else {
                progress::status("Checking", &format!("{}...", path.display()));
                progress::status("Running", &format!("{}...", path.display()));
                run::cmd(&path, &backend, release)?;
            }
        }
        Command::Check { file } => {
            let manifest = manifest::parse_manifest()?;
            let path = resolve_entry(file, manifest.as_ref())?;

            let has_externs = manifest.as_ref().is_some_and(Manifest::has_externs);
            let extern_meta = if has_externs {
                let manifest = manifest.as_ref().unwrap();
                let ctx = prepare_externs(manifest)?;
                ctx.metadata
            } else {
                HashMap::new()
            };

            progress::status("Checking", &format!("{}...", path.display()));
            check::cmd(&path, &extern_meta)?;
            progress::status(
                "Finished",
                &format!("{} checked successfully", path.display()),
            );
        }
        Command::Init { name } => {
            init::cmd(name.as_deref())?;
        }
        Command::Clean => {
            clean::cmd()?;
        }
        Command::Build { release } => {
            let manifest =
                manifest::parse_manifest()?.ok_or("anvyx build requires an anvyx.toml manifest")?;
            let cwd = std::env::current_dir()
                .map_err(|e| format!("Failed to get current directory: {e}"))?;
            let project_name = build::resolve_project_name(&manifest, &cwd);

            if manifest.has_externs() {
                prepare_externs(&manifest)?;
            }

            let runner_dir = build::generate_build_runner_crate(&cwd, &manifest, release)?;

            let spinner = progress::start_spinner("Bundling", "distribution...");
            build::build_runner(&runner_dir)?;
            progress::finish_spinner(&spinner);

            progress::status("Assembling", "distribution...");
            let dist_dir = build::assemble_dist(&cwd, &project_name)?;
            build::bundle_sources(&cwd, &dist_dir, &manifest)?;
            progress::status("Finished", &format!("{}", dist_dir.display()));
        }
    }

    Ok(())
}

fn resolve_entry(file: Option<PathBuf>, manifest: Option<&Manifest>) -> Result<PathBuf, String> {
    if let Some(f) = file {
        Ok(f)
    } else {
        let m =
            manifest.ok_or("No file provided and no anvyx.toml found in the current directory")?;
        Ok(PathBuf::from(&m.project.entry))
    }
}

struct ExternContext {
    cwd: PathBuf,
    metadata: HashMap<String, String>,
}

fn prepare_externs(manifest: &Manifest) -> Result<ExternContext, String> {
    let cwd =
        std::env::current_dir().map_err(|e| format!("Failed to get current directory: {e}"))?;

    for name in manifest.externs.keys() {
        progress::status("Loading", &format!("extern {name}..."));
    }
    let runner_dir = build::generate_runner_crate(&cwd, manifest)?;

    let spinner = progress::start_spinner("Compiling", "externs...");
    build::build_runner(&runner_dir)?;
    progress::finish_spinner(&spinner);

    progress::status("Resolving", "extern types...");
    build::extract_metadata(&cwd)?;

    let metadata = build::read_metadata(&cwd, manifest)?;
    Ok(ExternContext { cwd, metadata })
}
