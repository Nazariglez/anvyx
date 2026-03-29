use anvyx_lang::{StdModule, export_fn, provider};

// --- Trigonometric ---

#[export_fn]
pub fn double_sin(x: f64) -> f64 {
    x.sin()
}

#[export_fn]
pub fn double_cos(x: f64) -> f64 {
    x.cos()
}

#[export_fn]
pub fn double_tan(x: f64) -> f64 {
    x.tan()
}

#[export_fn]
pub fn double_asin(x: f64) -> f64 {
    x.asin()
}

#[export_fn]
pub fn double_acos(x: f64) -> f64 {
    x.acos()
}

#[export_fn]
pub fn double_atan(x: f64) -> f64 {
    x.atan()
}

#[export_fn]
pub fn double_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

// --- Rounding ---

#[export_fn]
pub fn double_floor(x: f64) -> f64 {
    x.floor()
}

#[export_fn]
pub fn double_ceil(x: f64) -> f64 {
    x.ceil()
}

#[export_fn]
pub fn double_round(x: f64) -> f64 {
    x.round()
}

#[export_fn]
pub fn double_trunc(x: f64) -> f64 {
    x.trunc()
}

// --- Roots, powers, exponentials ---

#[export_fn]
pub fn double_sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[export_fn]
pub fn double_cbrt(x: f64) -> f64 {
    x.cbrt()
}

#[export_fn]
pub fn double_pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

#[export_fn]
pub fn double_exp(x: f64) -> f64 {
    x.exp()
}

#[export_fn]
pub fn double_ln(x: f64) -> f64 {
    x.ln()
}

// --- Comparison ---

#[export_fn]
pub fn double_abs(x: f64) -> f64 {
    x.abs()
}

#[export_fn]
pub fn double_min(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[export_fn]
pub fn double_max(a: f64, b: f64) -> f64 {
    a.max(b)
}

#[export_fn]
pub fn double_clamp(val: f64, lo: f64, hi: f64) -> f64 {
    val.clamp(lo, hi)
}

// --- Interpolation ---

#[export_fn]
pub fn double_lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// --- Conversion ---

#[export_fn]
pub fn double_to_radians(deg: f64) -> f64 {
    deg.to_radians()
}

#[export_fn]
pub fn double_to_degrees(rad: f64) -> f64 {
    rad.to_degrees()
}

provider!(
    double_sin,
    double_cos,
    double_tan,
    double_asin,
    double_acos,
    double_atan,
    double_atan2,
    double_floor,
    double_ceil,
    double_round,
    double_trunc,
    double_sqrt,
    double_cbrt,
    double_pow,
    double_exp,
    double_ln,
    double_abs,
    double_min,
    double_max,
    double_clamp,
    double_lerp,
    double_to_radians,
    double_to_degrees,
);

pub fn module() -> StdModule {
    StdModule {
        name: "core_double",
        anv_source: include_str!("./double.anv"),
        exports: ANVYX_EXPORTS,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
        init: None,
    }
}
