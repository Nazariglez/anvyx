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
            let path = manifest::resolve_entry(file.as_deref())?;
            run::cmd(&path, &backend)?;
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
