use std::collections::HashMap;

use anvyx_lang::StdModule;

pub fn module() -> StdModule {
    StdModule {
        name: "core_option",
        anv_source: include_str!("./option.anv"),
        exports: || vec![],
        type_exports: || vec![],
        handlers: || HashMap::new(),
        init: None,
    }
}
