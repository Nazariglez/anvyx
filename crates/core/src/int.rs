use anvyx_lang::{StdModule, export_fn, provider};

#[export_fn]
pub fn int_abs(x: i64) -> i64 {
    x.abs()
}

#[export_fn]
pub fn int_min(a: i64, b: i64) -> i64 {
    a.min(b)
}

#[export_fn]
pub fn int_max(a: i64, b: i64) -> i64 {
    a.max(b)
}

#[export_fn]
pub fn int_clamp(val: i64, lo: i64, hi: i64) -> i64 {
    val.clamp(lo, hi)
}

provider!(int_abs, int_min, int_max, int_clamp);

pub fn module() -> StdModule {
    StdModule {
        name: "core_int",
        anv_source: include_str!("./int.anv"),
        exports: anvyx_exports,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
        init: None,
    }
}
