use std::collections::HashMap;

use anvyx_lang::{ExternDecl, ExternHandler};

pub struct StdModule {
    pub name: &'static str,
    pub anv_source: &'static str,
    pub exports: &'static [ExternDecl],
    pub handlers: fn() -> HashMap<String, ExternHandler>,
}

impl StdModule {
    pub fn full_anv_source(&self) -> String {
        let mut out = String::new();
        for decl in self.exports {
            out.push_str("extern fn ");
            out.push_str(decl.name);
            out.push('(');
            for (i, (pname, pty)) in decl.params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(pname);
                out.push_str(": ");
                out.push_str(pty);
            }
            out.push(')');
            if decl.ret != "void" {
                out.push_str(" -> ");
                out.push_str(decl.ret);
            }
            out.push('\n');
        }
        if !self.exports.is_empty() && !self.anv_source.is_empty() {
            out.push('\n');
        }
        out.push_str(self.anv_source);
        out
    }
}

pub fn std_modules() -> Vec<StdModule> {
    vec![math::module()]
}

mod math;
