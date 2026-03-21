use anvyx_lang::{MapStorage, ManagedRc, Value, export_fn, provider};

use super::StdModule;

#[export_fn(name = "ordered_map")]
pub fn make_ordered_map() -> Value {
    Value::Map(ManagedRc::new(MapStorage::new_ordered()))
}

provider!(make_ordered_map);

pub fn module() -> StdModule {
    StdModule {
        name: "maps",
        anv_source: include_str!("./maps.anv"),
        exports: ANVYX_EXPORTS,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
    }
}
