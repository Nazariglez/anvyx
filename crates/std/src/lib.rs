use std::collections::HashMap;

use anvyx_lang::ExternHandler;

pub struct StdModule {
    pub name: &'static str,
    pub anv_source: &'static str,
    pub handlers: fn() -> HashMap<String, ExternHandler>,
}

pub fn std_modules() -> Vec<StdModule> {
    vec![math::module()]
}

mod math;
