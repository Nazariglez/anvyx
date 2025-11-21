mod args;
mod directives;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{path::PathBuf, time::Instant};
use walkdir::WalkDir;

const EXT: &str = "anv";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const BLUE: &str = "\x1b[34m";
const RESET: &str = "\x1b[0m";

fn main() {
    let args = args::RunnerArgs::new().unwrap();

    let start_time = Instant::now();
    let files = if let Some(file) = args.file {
        vec![file]
    } else {
        WalkDir::new(&args.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_type().is_file()
                    && entry.path().extension().and_then(|s| s.to_str()) == Some(EXT)
            })
            .map(|entry| entry.path().to_path_buf())
            .collect()
    };

    println!("");
    println!("--- Testing {} files ---", files.len());
    println!("");

    let results = files
        .par_iter()
        .filter_map(|file| match test_file(file) {
            Ok(res) => Some((file.clone(), res)),
            Err(e) => Some((
                file.clone(),
                TestResult::Fail {
                    message: format!("Test runner error: {e}"),
                },
            )),
        })
        .collect::<Vec<_>>();

    let mut summary = Summary::default();
    for (file, result) in results {
        summary.add(file, result, args.quiet);
    }
    summary.print_summary(start_time, files.len());
    println!("");
}

fn test_file(file: &PathBuf) -> Result<TestResult, String> {
    let src = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let directives = directives::Directives::new(&src);
    if directives.skip.is_some() {
        return Ok(TestResult::Skip {
            message: directives.skip.unwrap(),
        });
    }
    // TODO: parse the file and test it
    // Ok(TestResult::Pass)
    Err("Not implemented".to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedResult {
    Success,
    Error,
    Timeout,
}

impl ExpectedResult {
    fn from_str(s: &str) -> Self {
        match s {
            "success" => Self::Success,
            "error" => Self::Error,
            "timeout" => Self::Timeout,
            _ => panic!("Invalid expected result: {}", s),
        }
    }
}

impl std::fmt::Display for ExpectedResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
enum TestResult {
    Pass,
    Fail { message: String },
    Timeout,
    Skip { message: String },
}

#[derive(Debug, Default)]
struct Summary {
    passed: usize,
    failed: usize,
    skipped: usize,
    timed_out: usize,

    failures: Vec<(PathBuf, String)>,
    timeouts: Vec<PathBuf>,
    skips: Vec<(PathBuf, String)>,
}

impl Summary {
    fn add(&mut self, file: PathBuf, result: TestResult, quiet: bool) {
        match result {
            TestResult::Pass => {
                self.passed += 1;
                pass_msg(&file, quiet);
            }
            TestResult::Fail { message } => {
                self.failed += 1;
                fail_msg(&file, quiet);
                self.failures.push((file, message));
            }
            TestResult::Timeout => {
                self.timed_out += 1;
                timeout_msg(&file, quiet);
                self.timeouts.push(file);
            }
            TestResult::Skip { message } => {
                self.skipped += 1;
                skip_msg(&file, quiet);
                self.skips.push((file, message));
            }
        }
    }

    fn print_summary(&self, start_time: Instant, total: usize) {
        println!("");
        println!("--- Test Results ---");
        println!("");

        if self.skipped > 0 {
            println!("{}Skipped:{} {}", YELLOW, RESET, self.skipped);
            self.skips.iter().for_each(|(f, m)| {
                println!("{YELLOW}  - {}:{RESET}", f.display());
                println!("    * {m}");
            });
            println!("");
        }

        if self.timed_out > 0 {
            println!("{}Timed out:{} {}", BLUE, RESET, self.timed_out);
            self.timeouts
                .iter()
                .for_each(|f| println!("{BLUE}  - {}{RESET}", f.display()));
            println!("");
        }

        if self.failed > 0 {
            println!("{}Failed:{} {}", RED, RESET, self.failed);
            self.failures.iter().for_each(|(f, m)| {
                println!("{RED}  - {}:{RESET}", f.display());
                println!("    * {m}");
            });
            println!("");
        }

        println!("{GREEN}Passed:{RESET} {} of {total}", self.passed);
        println!("");
        println!(
            "{CYAN}Total time:{RESET} {:.2}s",
            start_time.elapsed().as_secs_f64()
        );
    }
}

fn pass_msg(file: &PathBuf, quiet: bool) {
    if quiet {
        return;
    }
    println!("{GREEN}[PASS]{RESET} {}", file.display());
}

fn fail_msg(file: &PathBuf, quiet: bool) {
    if quiet {
        return;
    }
    println!("{RED}[FAIL]{RESET} {}", file.display());
}

fn timeout_msg(file: &PathBuf, quiet: bool) {
    if quiet {
        return;
    }
    println!("{BLUE}[TIMEOUT]{RESET} {}", file.display());
}

fn skip_msg(file: &PathBuf, quiet: bool) {
    if quiet {
        return;
    }
    println!("{YELLOW}[SKIP]{RESET} {}", file.display());
}
