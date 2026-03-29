use std::collections::HashMap;

use anvyx_lang::{ExternHandler, StdModuleSource};

pub fn collect_std() -> (
    HashMap<String, StdModuleSource>,
    HashMap<String, ExternHandler>,
) {
    let modules = anvyx_std::std_modules();
    anvyx_std::init_std_modules(&modules);
    let mut sources = HashMap::new();
    let mut handlers = HashMap::new();

    for module in modules {
        sources.insert(
            module.name.to_string(),
            StdModuleSource {
                anv_source: module.full_anv_source(),
            },
        );
        handlers.extend((module.handlers)());
    }

    (sources, handlers)
}

pub fn collect_core() -> (
    String,
    HashMap<String, StdModuleSource>,
    HashMap<String, ExternHandler>,
) {
    let (prelude_mods, method_mods) = anvyx_core::split_core_modules();
    let mut prelude = String::new();
    let mut sources = HashMap::new();
    let mut handlers = HashMap::new();

    for m in prelude_mods {
        prelude.push_str(&m.full_anv_source());
        prelude.push('\n');
        handlers.extend((m.handlers)());
    }

    for m in method_mods {
        sources.insert(
            m.name.to_string(),
            StdModuleSource {
                anv_source: m.full_anv_source(),
            },
        );
        handlers.extend((m.handlers)());
    }

    (prelude, sources, handlers)
}
