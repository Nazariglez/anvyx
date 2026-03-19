use std::collections::HashMap;

use super::StdModule;

pub fn module() -> StdModule {
    StdModule {
        name: "maps",
        anv_source: include_str!("./maps.anv"),
        exports: &[],
        handlers: || HashMap::new(),
    }
}
