use anvyx_lang::{StdModule, export_fn, provider};

// --- Trigonometric ---

#[export_fn]
pub fn float_sin(x: f32) -> f32 {
    x.sin()
}

#[export_fn]
pub fn float_cos(x: f32) -> f32 {
    x.cos()
}

#[export_fn]
pub fn float_tan(x: f32) -> f32 {
    x.tan()
}

#[export_fn]
pub fn float_asin(x: f32) -> f32 {
    x.asin()
}

#[export_fn]
pub fn float_acos(x: f32) -> f32 {
    x.acos()
}

#[export_fn]
pub fn float_atan(x: f32) -> f32 {
    x.atan()
}

#[export_fn]
pub fn float_atan2(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

// --- Rounding ---

#[export_fn]
pub fn float_floor(x: f32) -> f32 {
    x.floor()
}

#[export_fn]
pub fn float_ceil(x: f32) -> f32 {
    x.ceil()
}

#[export_fn]
pub fn float_round(x: f32) -> f32 {
    x.round()
}

#[export_fn]
pub fn float_trunc(x: f32) -> f32 {
    x.trunc()
}

// --- Roots, powers, exponentials ---

#[export_fn]
pub fn float_sqrt(x: f32) -> f32 {
    x.sqrt()
}

#[export_fn]
pub fn float_cbrt(x: f32) -> f32 {
    x.cbrt()
}

#[export_fn]
pub fn float_pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

#[export_fn]
pub fn float_exp(x: f32) -> f32 {
    x.exp()
}

#[export_fn]
pub fn float_ln(x: f32) -> f32 {
    x.ln()
}

// --- Comparison ---

#[export_fn]
pub fn float_abs(x: f32) -> f32 {
    x.abs()
}

#[export_fn]
pub fn float_min(a: f32, b: f32) -> f32 {
    a.min(b)
}

#[export_fn]
pub fn float_max(a: f32, b: f32) -> f32 {
    a.max(b)
}

#[export_fn]
pub fn float_clamp(val: f32, lo: f32, hi: f32) -> f32 {
    val.clamp(lo, hi)
}

// --- Interpolation ---

#[export_fn]
pub fn float_lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// --- Conversion ---

#[export_fn]
pub fn float_to_radians(deg: f32) -> f32 {
    deg.to_radians()
}

#[export_fn]
pub fn float_to_degrees(rad: f32) -> f32 {
    rad.to_degrees()
}

provider!(
    float_sin,
    float_cos,
    float_tan,
    float_asin,
    float_acos,
    float_atan,
    float_atan2,
    float_floor,
    float_ceil,
    float_round,
    float_trunc,
    float_sqrt,
    float_cbrt,
    float_pow,
    float_exp,
    float_ln,
    float_abs,
    float_min,
    float_max,
    float_clamp,
    float_lerp,
    float_to_radians,
    float_to_degrees,
);

pub fn module() -> StdModule {
    StdModule {
        name: "core_float",
        anv_source: include_str!("./float.anv"),
        exports: anvyx_exports,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
        init: None,
    }
}
