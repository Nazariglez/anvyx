use std::{
    sync::LazyLock,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anvyx_lang::{export_fn, provider};

use super::StdModule;

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

pub fn init() {
    // this is to ensure the start time is set before any other code runs
    // rust doesn't have a way to initialize globals like this, only lazy initializations
    // so we need to do it here
    let _ = *START;
}

#[export_fn]
pub fn elapsed() -> f64 {
    START.elapsed().as_secs_f64()
}

#[export_fn]
pub fn elapsed_ms() -> i64 {
    START.elapsed().as_millis() as i64
}

#[export_fn]
pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[export_fn]
pub fn sleep(ms: i64) {
    if ms > 0 {
        std::thread::sleep(Duration::from_millis(ms as u64));
    }
}

#[export_fn]
pub fn sleep_secs(secs: f64) {
    if secs > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(secs));
    }
}

provider!(elapsed, elapsed_ms, now, sleep, sleep_secs);

pub fn module() -> StdModule {
    StdModule {
        name: "time",
        anv_source: include_str!("./time.anv"),
        exports: anvyx_exports,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
        init: Some(init),
    }
}
