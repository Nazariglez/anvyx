use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendArg {
    Vm,
    Rust,
    Both,
}

impl BackendArg {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "vm" => Ok(Self::Vm),
            "rust" => Ok(Self::Rust),
            "both" => Ok(Self::Both),
            _ => Err(format!(
                "Unknown backend: '{s}'. Expected 'vm', 'rust', or 'both'"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vm => "vm",
            Self::Rust => "rust",
            Self::Both => "both",
        }
    }
}

#[derive(Debug)]
pub struct RunnerArgs {
    pub paths: Vec<PathBuf>,
    pub timeout_ms: u64,
    pub quiet: bool,
    pub release: bool,
    pub backend: BackendArg,
}

pub const USAGE: &str = "\
Usage: test-runner <PATH>... [OPTIONS]

Arguments:
  <PATH>...  One or more test files or directories

Options:
  --backend <vm|rust|both>  Backend to test (default: vm)
  --timeout <ms>            Test timeout in milliseconds (default: 2000)
  --quiet                   Suppress individual test output
  --release                 Build in release mode";

impl RunnerArgs {
    pub fn new() -> Result<Self, String> {
        let args = std::env::args().collect::<Vec<String>>();
        let paths = parse_paths(&args)?;
        let quiet = parse_quiet(&args);
        let release = parse_release(&args);
        let timeout_ms = parse_timeout(&args);
        let backend = parse_backend(&args)?;
        Ok(Self {
            paths,
            timeout_ms,
            quiet,
            release,
            backend,
        })
    }
}

fn parse_quiet(args: &[String]) -> bool {
    args.iter().any(|arg| arg.as_str() == "--quiet")
}

fn parse_release(args: &[String]) -> bool {
    args.iter().any(|arg| arg.as_str() == "--release")
}

fn parse_timeout(args: &[String]) -> u64 {
    args.iter()
        .position(|arg| arg.as_str() == "--timeout")
        .and_then(|i| args.get(i + 1))
        .map_or(2000, |arg| arg.parse::<u64>().unwrap())
}

fn parse_paths(args: &[String]) -> Result<Vec<PathBuf>, String> {
    let paths: Vec<PathBuf> = args[1..]
        .iter()
        .take_while(|arg| !arg.starts_with("--"))
        .map(PathBuf::from)
        .collect();

    if paths.is_empty() {
        return Err("Provide one or more directories or files as arguments".to_string());
    }

    for path in &paths {
        if !path.is_file() && !path.is_dir() {
            return Err(format!("Path not found: {}", path.display()));
        }
    }

    Ok(paths)
}

fn parse_backend(args: &[String]) -> Result<BackendArg, String> {
    args.iter()
        .position(|arg| arg.as_str() == "--backend")
        .and_then(|i| args.get(i + 1))
        .map_or(Ok(BackendArg::Vm), |s| BackendArg::from_str(s))
}
