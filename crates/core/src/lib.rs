mod option;
mod range;
mod string;

use anvyx_lang::StdModule;

pub fn core_modules() -> Vec<StdModule> {
    vec![option::module(), range::module(), string::module()]
}
