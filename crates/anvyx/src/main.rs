mod check;
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
        file: PathBuf,
        #[arg(long, default_value = "vm")]
        backend: String,
    },
    #[command(about = "Check an Anvyx file")]
    Check { file: PathBuf },
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run { file, backend } => {
            run::cmd(file.as_path(), &backend)?;
        }
        Command::Check { file } => {
            check::cmd(file.as_path())?;
        }
    }

    Ok(())
}
