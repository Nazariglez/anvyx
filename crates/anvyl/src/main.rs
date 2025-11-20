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
    #[command(about = "Run an Anvyl program")]
    Run { file: PathBuf },
    #[command(about = "Check an Anvyl file")]
    Check { file: PathBuf },
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run { file } => {
            run::cmd(file.as_path())?;
        }
        Command::Check { file } => {
            check::cmd(file.as_path())?;
        }
    }

    Ok(())
}
