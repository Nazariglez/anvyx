mod prelude;
mod string;

use anvyx_lang::StdModule;

pub fn core_modules() -> Vec<StdModule> {
    vec![prelude::module(), string::module()]
}
