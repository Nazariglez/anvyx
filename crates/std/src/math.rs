use anvyx_lang::{export_fn, provider};

use super::StdModule;

#[export_fn]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

#[export_fn]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

#[export_fn]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[export_fn]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

#[export_fn]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

#[export_fn]
pub fn round(x: f64) -> f64 {
    x.round()
}

#[export_fn]
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

#[export_fn]
pub fn log(x: f64) -> f64 {
    x.ln()
}

provider!(sin, cos, sqrt, floor, ceil, round, pow, log);

pub fn module() -> StdModule {
    StdModule {
        name: "math",
        anv_source: include_str!("./math.anv"),
        exports: ANVYX_EXPORTS,
        handlers: anvyx_externs,
    }
}
