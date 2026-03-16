use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendArg {
    Vm,
    Transpiler,
    Both,
}

impl BackendArg {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "vm" => Ok(Self::Vm),
            "transpiler" => Ok(Self::Transpiler),
            "both" => Ok(Self::Both),
            _ => Err(format!(
                "Unknown backend: '{s}'. Expected 'vm', 'transpiler', or 'both'"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vm => "vm",
            Self::Transpiler => "transpiler",
            Self::Both => "both",
        }
    }
}

#[derive(Debug)]
pub struct RunnerArgs {
    pub root: PathBuf,
    pub timeout_ms: u64,
    pub quiet: bool,
    pub release: bool,
    pub file: Option<PathBuf>,
    pub backend: BackendArg,
}

impl RunnerArgs {
    pub fn new() -> Result<Self, String> {
        let args = std::env::args().collect::<Vec<String>>();
        let (root, file) = parse_root_file(&args)?;
        let quiet = parse_quiet(&args);
        let release = parse_release(&args);
        let timeout_ms = parse_timeout(&args);
        let backend = parse_backend(&args)?;
        Ok(Self {
            root,
            timeout_ms,
            quiet,
            release,
            file,
            backend,
        })
    }
}

fn parse_quiet(args: &[String]) -> bool {
    args.iter().find(|arg| arg.as_str() == "--quiet").is_some()
}

fn parse_release(args: &[String]) -> bool {
    args.iter()
        .find(|arg| arg.as_str() == "--release")
        .is_some()
}

fn parse_timeout(args: &[String]) -> u64 {
    args.iter()
        .position(|arg| arg.as_str() == "--timeout")
        .and_then(|i| args.get(i + 1))
        .map(|arg| arg.parse::<u64>().unwrap())
        .unwrap_or(2000)
}

fn parse_root_file(args: &[String]) -> Result<(PathBuf, Option<PathBuf>), String> {
    if args.len() == 1 || args[1].starts_with("--") {
        return Err("Provide a directory or a file as first argument".to_string());
    }

    let dir = std::path::Path::new(&args[1]);
    if dir.is_dir() {
        return Ok((dir.to_path_buf(), None));
    }

    if dir.is_file() {
        return Ok((dir.parent().unwrap().to_path_buf(), Some(dir.to_path_buf())));
    }

    Err("Provide a directory or a file as first argument".to_string())
}

fn parse_backend(args: &[String]) -> Result<BackendArg, String> {
    args.iter()
        .position(|arg| arg.as_str() == "--backend")
        .and_then(|i| args.get(i + 1))
        .map(|s| BackendArg::from_str(s))
        .unwrap_or(Ok(BackendArg::Vm))
}
