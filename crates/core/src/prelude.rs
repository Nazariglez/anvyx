use std::collections::HashMap;

use anvyx_lang::StdModule;

pub fn module() -> StdModule {
    StdModule {
        name: "prelude",
        anv_source: include_str!("./prelude.anv"),
        exports: &[],
        type_exports: || vec![],
        handlers: || HashMap::new(),
        init: None,
    }
}
