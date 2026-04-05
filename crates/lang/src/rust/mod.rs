mod emit;
mod pipeline;
mod stringify;
pub(crate) mod validate;

use crate::{RustBackendConfig, hir};

pub fn run(program: &hir::Program, config: &RustBackendConfig) -> Result<String, String> {
    let plan = validate::validate(program)?;
    let source = emit::emit(program, &plan)?;
    pipeline::compile_and_run(&source, config)
}
