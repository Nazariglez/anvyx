mod double;
mod float;
mod int;
mod option;
mod range;
mod string;

use anvyx_lang::StdModule;

pub fn core_modules() -> Vec<StdModule> {
    vec![
        option::module(),
        range::module(),
        string::module(),
        float::module(),
        double::module(),
        int::module(),
    ]
}

/// Splits core modules into (prelude, method-providing).
/// Prelude modules (no exports) are prepended to the AST.
/// Method-providing modules (has exports) go through the module system.
pub fn split_core_modules() -> (Vec<StdModule>, Vec<StdModule>) {
    core_modules()
        .into_iter()
        .partition(|m| (m.exports)().is_empty())
}
