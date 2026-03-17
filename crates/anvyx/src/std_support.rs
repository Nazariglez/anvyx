use std::collections::HashMap;

use anvyx_lang::{ExternHandler, StdModuleSource};

pub fn collect_std() -> (HashMap<String, StdModuleSource>, HashMap<String, ExternHandler>) {
    let modules = anvyx_std::std_modules();
    let mut sources = HashMap::new();
    let mut handlers = HashMap::new();

    for module in modules {
        sources.insert(
            module.name.to_string(),
            StdModuleSource {
                anv_source: module.anv_source.to_string(),
            },
        );
        handlers.extend((module.handlers)());
    }

    (sources, handlers)
}
