use anvyx_lang::{export_fn, provider};

use super::StdModule;

/// Computes the sine of x.
#[export_fn]
pub fn sin(x: f32) -> f32 {
    x.sin()
}

#[export_fn]
pub fn cos(x: f32) -> f32 {
    x.cos()
}

#[export_fn]
pub fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

#[export_fn]
pub fn floor(x: f32) -> f32 {
    x.floor()
}

#[export_fn]
pub fn ceil(x: f32) -> f32 {
    x.ceil()
}

#[export_fn]
pub fn round(x: f32) -> f32 {
    x.round()
}

#[export_fn]
pub fn pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

#[export_fn]
pub fn log(x: f32) -> f32 {
    x.ln()
}

provider!(sin, cos, sqrt, floor, ceil, round, pow, log);

pub fn module() -> StdModule {
    StdModule {
        name: "math",
        anv_source: include_str!("./math.anv"),
        exports: ANVYX_EXPORTS,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
        init: None,
    }
}
