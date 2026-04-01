use std::time::Duration;

use console::style;
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};

const VERB_WIDTH: usize = 12;

pub fn status(verb: &str, message: &str) {
    let padded = format!("{verb:>VERB_WIDTH$}");
    eprintln!("{} {}", style(padded).green().bold(), message);
}

pub fn error(message: &str) {
    let padded = format!("{:>width$}", "error:", width = VERB_WIDTH);
    eprintln!("{} {}", style(padded).red().bold(), message);
}

pub fn start_spinner(verb: &str, message: &str) -> ProgressBar {
    let padded = format!("{verb:>VERB_WIDTH$}");
    let text = format!("{} {}", style(padded).green().bold(), message);
    let pb = ProgressBar::new_spinner().with_finish(ProgressFinish::AndClear);
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{msg} {spinner}")
            .expect("valid template"),
    );
    pb.set_message(text);
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

pub fn finish_spinner(pb: &ProgressBar) {
    pb.finish_and_clear();
}
