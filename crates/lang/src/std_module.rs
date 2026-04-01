use std::collections::HashMap;

use crate::{ExternDecl, ExternHandler, ExternTypeDecl};

fn emit_doc(out: &mut String, doc: Option<&str>) {
    if let Some(doc) = doc {
        for line in doc.lines() {
            if line.is_empty() {
                out.push_str("///\n");
            } else {
                out.push_str("/// ");
                out.push_str(line);
                out.push('\n');
            }
        }
    }
}

pub struct StdModule {
    pub name: &'static str,
    pub anv_source: &'static str,
    pub exports: &'static [ExternDecl],
    pub type_exports: fn() -> Vec<ExternTypeDecl>,
    pub handlers: fn() -> HashMap<String, ExternHandler>,
    pub init: Option<fn()>,
}

impl StdModule {
    pub fn full_anv_source(&self) -> String {
        let mut out = String::new();
        for ty in (self.type_exports)() {
            let has_members = ty.has_init
                || !ty.fields.is_empty()
                || !ty.methods.is_empty()
                || !ty.statics.is_empty()
                || !ty.operators.is_empty();
            emit_doc(&mut out, ty.doc);
            if has_members {
                out.push_str("extern type ");
                out.push_str(ty.name);
                out.push_str(" {\n");
                if ty.has_init {
                    out.push_str("    init;\n");
                }
                for field in &ty.fields {
                    out.push_str("    ");
                    out.push_str(field.name);
                    out.push_str(": ");
                    out.push_str(field.ty);
                    out.push_str(";\n");
                }
                for method in &ty.methods {
                    emit_doc(&mut out, method.doc);
                    out.push_str("    fn ");
                    out.push_str(method.name);
                    out.push('(');
                    let receiver_str = match method.receiver {
                        "var" => "var self",
                        _ => "self",
                    };
                    out.push_str(receiver_str);
                    for (pname, pty) in method.params {
                        out.push_str(", ");
                        out.push_str(pname);
                        out.push_str(": ");
                        out.push_str(pty);
                    }
                    out.push(')');
                    if method.ret != "void" {
                        out.push_str(" -> ");
                        out.push_str(method.ret);
                    }
                    out.push_str(";\n");
                }
                for s in &ty.statics {
                    emit_doc(&mut out, s.doc);
                    out.push_str("    fn ");
                    out.push_str(s.name);
                    out.push('(');
                    for (i, (pname, pty)) in s.params.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(pname);
                        out.push_str(": ");
                        out.push_str(pty);
                    }
                    out.push(')');
                    if s.ret != "void" {
                        out.push_str(" -> ");
                        out.push_str(s.ret);
                    }
                    out.push_str(";\n");
                }
                for op in &ty.operators {
                    let sym = match op.op {
                        "Add" => "+",
                        "Sub" | "Neg" => "-",
                        "Mul" => "*",
                        "Div" => "/",
                        "Rem" => "%",
                        "Eq" => "==",
                        other => panic!("unknown operator: {other}"),
                    };
                    let as_self =
                        |s: &'static str| -> &'static str { if s == ty.name { "Self" } else { s } };
                    let ret_str = as_self(op.ret);
                    match (op.rhs, op.lhs) {
                        (None, None) => {
                            // unary
                            out.push_str("    op -Self -> ");
                            out.push_str(ret_str);
                            out.push_str(";\n");
                        }
                        (Some(rhs), None) => {
                            // Self op rhs -> ret
                            out.push_str("    op Self ");
                            out.push_str(sym);
                            out.push(' ');
                            out.push_str(as_self(rhs));
                            out.push_str(" -> ");
                            out.push_str(ret_str);
                            out.push_str(";\n");
                        }
                        (None, Some(lhs)) => {
                            // lhs op Self -> ret
                            out.push_str("    op ");
                            out.push_str(as_self(lhs));
                            out.push(' ');
                            out.push_str(sym);
                            out.push_str(" Self -> ");
                            out.push_str(ret_str);
                            out.push_str(";\n");
                        }
                        (Some(_), Some(_)) => {
                            panic!("ExternOpDecl cannot have both rhs and lhs set");
                        }
                    }
                }
                out.push_str("}\n");
            } else {
                out.push_str("extern type ");
                out.push_str(ty.name);
                out.push_str(";\n");
            }
        }
        for decl in self.exports {
            emit_doc(&mut out, decl.doc);
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
            out.push_str(";\n");
        }
        if !self.exports.is_empty() && !self.anv_source.is_empty() {
            out.push('\n');
        }
        out.push_str(self.anv_source);
        out
    }
}

pub fn init_std_modules(modules: &[StdModule]) {
    for module in modules {
        if let Some(init) = module.init {
            init();
        }
    }
}
