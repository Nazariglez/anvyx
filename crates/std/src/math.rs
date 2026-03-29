use std::collections::HashMap;

use anvyx_lang::StdModule;

pub fn module() -> StdModule {
    StdModule {
        name: "math",
        anv_source: include_str!("./math.anv"),
        exports: &[],
        type_exports: || vec![],
        handlers: || HashMap::new(),
        init: None,
    }
}
