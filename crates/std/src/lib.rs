pub use anvyx_lang::{StdModule, init_std_modules};

mod linalg;
mod maps;
mod math;
mod mem;
mod time;

pub fn std_modules() -> Vec<StdModule> {
    vec![
        math::module(),
        maps::module(),
        linalg::module(),
        mem::module(),
        time::module(),
    ]
}
