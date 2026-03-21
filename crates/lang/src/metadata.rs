#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternDecl {
    pub name: &'static str,
    pub params: &'static [(&'static str, &'static str)],
    pub ret: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternTypeDecl {
    pub name: &'static str,
    pub has_init: bool,
    pub fields: Vec<ExternFieldDecl>,
    pub methods: Vec<ExternMethodDecl>,
    pub statics: Vec<ExternStaticMethodDecl>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternTypeDeclConst {
    pub name: &'static str,
    pub has_init: bool,
    pub fields: &'static [ExternFieldDecl],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternFieldDecl {
    pub name: &'static str,
    pub ty: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternMethodDecl {
    pub name: &'static str,
    pub receiver: &'static str,
    pub params: &'static [(&'static str, &'static str)],
    pub ret: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternStaticMethodDecl {
    pub name: &'static str,
    pub params: &'static [(&'static str, &'static str)],
    pub ret: &'static str,
}

pub fn exports_to_json(decls: &[ExternDecl], type_decls: &[ExternTypeDecl]) -> String {
    let mut out = String::from("{\"types\":[");
    for (i, ty) in type_decls.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        push_escaped(&mut out, ty.name);
        out.push('"');
        if ty.has_init {
            out.push_str(",\"init\":true");
        }
        out.push_str(",\"fields\":[");
        for (j, field) in ty.fields.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str("{\"name\":\"");
            push_escaped(&mut out, field.name);
            out.push_str("\",\"type\":\"");
            push_escaped(&mut out, field.ty);
            out.push_str("\"}");
        }
        out.push_str("],\"methods\":[");
        for (j, method) in ty.methods.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str("{\"name\":\"");
            push_escaped(&mut out, method.name);
            out.push_str("\",\"receiver\":\"");
            push_escaped(&mut out, method.receiver);
            out.push_str("\",\"params\":");
            push_params(&mut out, method.params);
            out.push_str(",\"ret\":\"");
            push_escaped(&mut out, method.ret);
            out.push_str("\"}");
        }
        out.push_str("],\"statics\":[");
        for (j, s) in ty.statics.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str("{\"name\":\"");
            push_escaped(&mut out, s.name);
            out.push_str("\",\"params\":");
            push_params(&mut out, s.params);
            out.push_str(",\"ret\":\"");
            push_escaped(&mut out, s.ret);
            out.push_str("\"}");
        }
        out.push_str("]}");
    }
    out.push_str("],\"functions\":[");
    for (i, decl) in decls.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        push_escaped(&mut out, decl.name);
        out.push_str("\",\"params\":");
        push_params(&mut out, decl.params);
        out.push_str(",\"ret\":\"");
        push_escaped(&mut out, decl.ret);
        out.push_str("\"}");
    }
    out.push_str("]}");
    out
}

fn push_params(out: &mut String, params: &[(&str, &str)]) {
    out.push('[');
    for (j, (pname, pty)) in params.iter().enumerate() {
        if j > 0 {
            out.push(',');
        }
        out.push_str("[\"");
        push_escaped(out, pname);
        out.push_str("\",\"");
        push_escaped(out, pty);
        out.push_str("\"]");
    }
    out.push(']');
}

fn push_escaped(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            ch => out.push(ch),
        }
    }
}

#[derive(Debug)]
pub struct ExternFuncMeta {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub ret: String,
}

#[derive(Debug)]
pub struct ExternTypeMeta {
    pub name: String,
    pub has_init: bool,
    pub fields: Vec<ExternFieldMeta>,
    pub methods: Vec<ExternMethodMeta>,
    pub statics: Vec<ExternStaticMethodMeta>,
}

#[derive(Debug)]
pub struct ExternFieldMeta {
    pub name: String,
    pub ty: String,
}

#[derive(Debug)]
pub struct ExternMethodMeta {
    pub name: String,
    pub receiver: String,
    pub params: Vec<(String, String)>,
    pub ret: String,
}

#[derive(Debug)]
pub struct ExternStaticMethodMeta {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub ret: String,
}

#[derive(Debug)]
pub struct ExternProviderMeta {
    pub types: Vec<ExternTypeMeta>,
    pub functions: Vec<ExternFuncMeta>,
}

pub fn parse_provider_json(json: &str) -> Result<ExternProviderMeta, String> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("Invalid metadata JSON: {e}"))?;

    let empty_arr = vec![];
    let types = match v["types"].as_array() {
        Some(arr) => {
            let mut type_metas = vec![];
            for ty in arr {
                let name = ty["name"]
                    .as_str()
                    .ok_or("Type entry missing 'name'")?
                    .to_string();
                let has_init = ty["init"].as_bool().unwrap_or(false);

                let fields = ty["fields"]
                    .as_array()
                    .unwrap_or(&empty_arr)
                    .iter()
                    .map(|f| {
                        Ok(ExternFieldMeta {
                            name: f["name"].as_str().ok_or("Field missing 'name'")?.to_string(),
                            ty: f["type"].as_str().ok_or("Field missing 'type'")?.to_string(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;

                let methods = ty["methods"]
                    .as_array()
                    .unwrap_or(&empty_arr)
                    .iter()
                    .map(|m| {
                        Ok(ExternMethodMeta {
                            name: m["name"].as_str().ok_or("Method missing 'name'")?.to_string(),
                            receiver: m["receiver"].as_str().ok_or("Method missing 'receiver'")?.to_string(),
                            params: parse_params(m["params"].as_array().unwrap_or(&empty_arr))?,
                            ret: m["ret"].as_str().ok_or("Method missing 'ret'")?.to_string(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;

                let statics = ty["statics"]
                    .as_array()
                    .unwrap_or(&empty_arr)
                    .iter()
                    .map(|s| {
                        Ok(ExternStaticMethodMeta {
                            name: s["name"].as_str().ok_or("Static method missing 'name'")?.to_string(),
                            params: parse_params(s["params"].as_array().unwrap_or(&empty_arr))?,
                            ret: s["ret"].as_str().ok_or("Static method missing 'ret'")?.to_string(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;

                type_metas.push(ExternTypeMeta { name, has_init, fields, methods, statics });
            }
            type_metas
        }
        None => vec![],
    };

    let functions = v["functions"]
        .as_array()
        .ok_or("Metadata JSON missing 'functions' array")?;

    let mut funcs = vec![];
    for func in functions {
        let name = func["name"]
            .as_str()
            .ok_or("Function entry missing 'name'")?
            .to_string();

        let params = parse_params(
            func["params"]
                .as_array()
                .ok_or("Function entry missing 'params'")?,
        )?;

        let ret = func["ret"]
            .as_str()
            .ok_or("Function entry missing 'ret'")?
            .to_string();

        funcs.push(ExternFuncMeta { name, params, ret });
    }

    Ok(ExternProviderMeta { types, functions: funcs })
}

fn parse_params(arr: &[serde_json::Value]) -> Result<Vec<(String, String)>, String> {
    arr.iter()
        .map(|param| {
            let pair = param.as_array().ok_or("Param entry is not an array")?;
            if pair.len() != 2 {
                return Err(format!("Param entry has {} elements, expected 2", pair.len()));
            }
            let pname = pair[0].as_str().ok_or("Param name is not a string")?.to_string();
            let pty = pair[1].as_str().ok_or("Param type is not a string")?.to_string();
            Ok((pname, pty))
        })
        .collect()
}

pub(crate) fn anvyx_type_from_str(s: &str) -> Result<crate::ast::Type, String> {
    match s {
        "int" => Ok(crate::ast::Type::Int),
        "float" => Ok(crate::ast::Type::Float),
        "bool" => Ok(crate::ast::Type::Bool),
        "string" => Ok(crate::ast::Type::String),
        "void" => Ok(crate::ast::Type::Void),
        "any" => Ok(crate::ast::Type::Any),
        _ => crate::parser::parse_type_str(s)
            .map_err(|_| format!("Invalid Anvyx type in extern metadata: '{s}'")),
    }
}

pub(crate) fn metadata_to_extern_stmts(
    meta: &ExternProviderMeta,
) -> Result<Vec<crate::ast::StmtNode>, String> {
    use crate::ast::{
        ExternFunc, ExternType, ExternTypeMember, Ident, MethodReceiver, Stmt,
    };
    use crate::span::{Span, Spanned};
    use internment::Intern;

    let span = Span::new(0, 0);
    let mut stmts = vec![];

    for ty in &meta.types {
        let name = Ident(Intern::new(ty.name.clone()));
        let mut members = vec![];

        for field in &ty.fields {
            members.push(ExternTypeMember::Field {
                name: Ident(Intern::new(field.name.clone())),
                ty: anvyx_type_from_str(&field.ty)?,
            });
        }

        for method in &ty.methods {
            let receiver = match method.receiver.as_str() {
                "var" => MethodReceiver::Var,
                _ => MethodReceiver::Value,
            };
            let params = params_to_ast(&method.params)?;
            let ret = anvyx_type_from_str(&method.ret)?;
            members.push(ExternTypeMember::Method {
                name: Ident(Intern::new(method.name.clone())),
                receiver,
                params,
                ret,
            });
        }

        for static_m in &ty.statics {
            let params = params_to_ast(&static_m.params)?;
            let ret = anvyx_type_from_str(&static_m.ret)?;
            members.push(ExternTypeMember::StaticMethod {
                name: Ident(Intern::new(static_m.name.clone())),
                params,
                ret,
            });
        }

        let node = Spanned::new(ExternType { name, has_init: ty.has_init, members }, span);
        stmts.push(Spanned::new(Stmt::ExternType(node), span));
    }

    for func in &meta.functions {
        let name = Ident(Intern::new(func.name.clone()));
        let params = params_to_ast(&func.params)?;
        let ret = anvyx_type_from_str(&func.ret)?;

        let node = Spanned::new(ExternFunc { name, params, ret }, span);
        stmts.push(Spanned::new(Stmt::ExternFunc(node), span));
    }

    Ok(stmts)
}

fn params_to_ast(
    params: &[(String, String)],
) -> Result<Vec<crate::ast::Param>, String> {
    use crate::ast::{Ident, Mutability, Param};
    use internment::Intern;

    params
        .iter()
        .map(|(pname, pty)| {
            Ok(Param {
                mutability: Mutability::Immutable,
                name: Ident(Intern::new(pname.clone())),
                ty: anvyx_type_from_str(pty)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_decls() {
        assert_eq!(exports_to_json(&[], &[]), "{\"types\":[],\"functions\":[]}");
    }

    #[test]
    fn single_function_with_params() {
        let decl = ExternDecl {
            name: "add",
            params: &[("a", "int"), ("b", "int")],
            ret: "int",
        };
        let json = exports_to_json(&[decl], &[]);
        assert_eq!(
            json,
            "{\"types\":[],\"functions\":[{\"name\":\"add\",\"params\":[[\"a\",\"int\"],[\"b\",\"int\"]],\"ret\":\"int\"}]}"
        );
    }

    #[test]
    fn multiple_functions() {
        let decls = [
            ExternDecl { name: "add", params: &[("a", "int"), ("b", "int")], ret: "int" },
            ExternDecl { name: "greet", params: &[("name", "string")], ret: "string" },
        ];
        let json = exports_to_json(&decls, &[]);
        assert!(json.contains("\"name\":\"add\""));
        assert!(json.contains("\"name\":\"greet\""));
        assert!(json.starts_with("{\"types\":[],\"functions\":["));
        assert!(json.ends_with("]}"));
    }

    #[test]
    fn void_return() {
        let decl = ExternDecl { name: "noop", params: &[], ret: "void" };
        let json = exports_to_json(&[decl], &[]);
        assert!(json.contains("\"ret\":\"void\""));
        assert!(json.contains("\"params\":[]"));
    }

    #[test]
    fn all_type_variants() {
        let decls = [
            ExternDecl { name: "f_int", params: &[("x", "int")], ret: "int" },
            ExternDecl { name: "f_float", params: &[("x", "float")], ret: "float" },
            ExternDecl { name: "f_bool", params: &[("x", "bool")], ret: "bool" },
            ExternDecl { name: "f_string", params: &[("x", "string")], ret: "string" },
        ];
        let json = exports_to_json(&decls, &[]);
        assert!(json.contains("\"int\""));
        assert!(json.contains("\"float\""));
        assert!(json.contains("\"bool\""));
        assert!(json.contains("\"string\""));
    }

    #[test]
    fn no_params() {
        let decl = ExternDecl { name: "get_val", params: &[], ret: "int" };
        let json = exports_to_json(&[decl], &[]);
        assert!(json.contains("\"params\":[]"));
        assert!(json.contains("\"ret\":\"int\""));
    }

    #[test]
    fn parse_empty_functions() {
        let json = "{\"types\":[],\"functions\":[]}";
        let result = parse_provider_json(json).unwrap();
        assert!(result.functions.is_empty());
    }

    #[test]
    fn parse_invalid_json() {
        let result = parse_provider_json("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_name_field() {
        let json = r#"{"types":[],"functions":[{"params":[],"ret":"int"}]}"#;
        let result = parse_provider_json(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing 'name'"));
    }

    #[test]
    fn parse_valid_json_round_trip() {
        let decls = [
            ExternDecl { name: "add", params: &[("a", "int"), ("b", "int")], ret: "int" },
            ExternDecl { name: "greet", params: &[("name", "string")], ret: "string" },
        ];
        let json = exports_to_json(&decls, &[]);
        let meta = parse_provider_json(&json).unwrap();
        assert_eq!(meta.functions.len(), 2);
        assert_eq!(meta.functions[0].name, "add");
        assert_eq!(meta.functions[0].params, vec![("a".to_string(), "int".to_string()), ("b".to_string(), "int".to_string())]);
        assert_eq!(meta.functions[0].ret, "int");
        assert_eq!(meta.functions[1].name, "greet");
        assert_eq!(meta.functions[1].params, vec![("name".to_string(), "string".to_string())]);
        assert_eq!(meta.functions[1].ret, "string");
    }

    #[test]
    fn anvyx_type_from_str_valid() {
        assert!(matches!(anvyx_type_from_str("int"), Ok(crate::ast::Type::Int)));
        assert!(matches!(anvyx_type_from_str("float"), Ok(crate::ast::Type::Float)));
        assert!(matches!(anvyx_type_from_str("bool"), Ok(crate::ast::Type::Bool)));
        assert!(matches!(anvyx_type_from_str("string"), Ok(crate::ast::Type::String)));
        assert!(matches!(anvyx_type_from_str("void"), Ok(crate::ast::Type::Void)));
    }

    #[test]
    fn anvyx_type_from_str_any() {
        assert!(matches!(anvyx_type_from_str("any"), Ok(crate::ast::Type::Any)));
    }

    #[test]
    fn anvyx_type_from_str_list() {
        let ty = anvyx_type_from_str("[int]").unwrap();
        assert!(matches!(ty, crate::ast::Type::List { .. }));
    }

    #[test]
    fn anvyx_type_from_str_map() {
        let ty = anvyx_type_from_str("[string: int]").unwrap();
        assert!(matches!(ty, crate::ast::Type::Map { .. }));
    }

    #[test]
    fn anvyx_type_from_str_optional() {
        let ty = anvyx_type_from_str("int?").unwrap();
        assert!(matches!(ty, crate::ast::Type::Enum { .. }));
    }

    #[test]
    fn anvyx_type_from_str_tuple() {
        let ty = anvyx_type_from_str("(int, string)").unwrap();
        assert!(matches!(ty, crate::ast::Type::Tuple(_)));
    }

    #[test]
    fn anvyx_type_from_str_nested() {
        let ty = anvyx_type_from_str("[string: [int]]").unwrap();
        assert!(matches!(ty, crate::ast::Type::Map { .. }));
    }

    #[test]
    fn anvyx_type_from_str_unknown() {
        let result = anvyx_type_from_str("@#$");
        assert!(result.is_err());
    }

    #[test]
    fn metadata_to_extern_stmts_basic() {
        use crate::ast::{Mutability, Stmt};

        let meta = ExternProviderMeta {
            types: vec![],
            functions: vec![
                ExternFuncMeta {
                    name: "add".to_string(),
                    params: vec![
                        ("a".to_string(), "int".to_string()),
                        ("b".to_string(), "int".to_string()),
                    ],
                    ret: "int".to_string(),
                },
            ],
        };

        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        assert_eq!(stmts.len(), 1);
        let Stmt::ExternFunc(node) = &stmts[0].node else {
            panic!("expected ExternFunc");
        };
        assert_eq!(node.node.name.to_string(), "add");
        assert_eq!(node.node.params.len(), 2);
        assert_eq!(node.node.params[0].mutability, Mutability::Immutable);
        assert!(matches!(node.node.params[0].ty, crate::ast::Type::Int));
        assert!(matches!(node.node.ret, crate::ast::Type::Int));
    }

    #[test]
    fn metadata_round_trip_composite_types() {
        let meta = ExternProviderMeta {
            types: vec![],
            functions: vec![ExternFuncMeta {
                name: "make_list".to_string(),
                params: vec![("size".to_string(), "int".to_string())],
                ret: "[int]".to_string(),
            }],
        };
        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        assert_eq!(stmts.len(), 1);
        let crate::ast::Stmt::ExternFunc(node) = &stmts[0].node else {
            panic!("expected ExternFunc");
        };
        assert!(matches!(node.node.ret, crate::ast::Type::List { .. }));
    }

    #[test]
    fn exports_to_json_with_types() {
        let type_decls = [
            ExternTypeDecl { name: "Sprite", has_init: false, fields: vec![], methods: vec![], statics: vec![] },
            ExternTypeDecl { name: "Texture", has_init: false, fields: vec![], methods: vec![], statics: vec![] },
        ];
        let json = exports_to_json(&[], &type_decls);
        assert!(json.contains("\"name\":\"Sprite\""));
        assert!(json.contains("\"name\":\"Texture\""));
        assert!(json.contains("\"fields\":[]"));
        assert!(json.contains("\"methods\":[]"));
        assert!(json.contains("\"statics\":[]"));
        assert!(json.starts_with("{\"types\":[{"));
        assert!(json.ends_with("]}"));
    }

    #[test]
    fn exports_to_json_with_types_and_functions() {
        let type_decls = [ExternTypeDecl { name: "Sprite", has_init: false, fields: vec![], methods: vec![], statics: vec![] }];
        let func_decls = [ExternDecl { name: "create", params: &[], ret: "Sprite" }];
        let json = exports_to_json(&func_decls, &type_decls);
        assert!(json.contains("\"name\":\"Sprite\""));
        assert!(json.contains("\"name\":\"create\""));
    }

    #[test]
    fn parse_provider_json_with_types() {
        let json = r#"{"types":[{"name":"Sprite","fields":[],"methods":[],"statics":[]},{"name":"Texture","fields":[],"methods":[],"statics":[]}],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 2);
        assert_eq!(meta.types[0].name, "Sprite");
        assert!(meta.types[0].fields.is_empty());
        assert!(meta.types[0].methods.is_empty());
        assert!(meta.types[0].statics.is_empty());
        assert_eq!(meta.types[1].name, "Texture");
        assert!(meta.functions.is_empty());
    }

    #[test]
    fn parse_provider_json_missing_types_backwards_compat() {
        let json = r#"{"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert!(meta.types.is_empty());
        assert!(meta.functions.is_empty());
    }

    #[test]
    fn metadata_to_extern_stmts_with_types() {
        use crate::ast::Stmt;

        let meta = ExternProviderMeta {
            types: vec![ExternTypeMeta {
                name: "Sprite".to_string(),
                has_init: false,
                fields: vec![],
                methods: vec![],
                statics: vec![],
            }],
            functions: vec![ExternFuncMeta {
                name: "create".to_string(),
                params: vec![],
                ret: "int".to_string(),
            }],
        };

        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        assert_eq!(stmts.len(), 2);
        let Stmt::ExternType(ty_node) = &stmts[0].node else {
            panic!("expected ExternType as first stmt");
        };
        assert_eq!(ty_node.node.name.to_string(), "Sprite");
        let Stmt::ExternFunc(fn_node) = &stmts[1].node else {
            panic!("expected ExternFunc as second stmt");
        };
        assert_eq!(fn_node.node.name.to_string(), "create");
    }

    #[test]
    fn round_trip_with_types() {
        use crate::ast::Stmt;

        let type_decls = [ExternTypeDecl { name: "Sprite", has_init: false, fields: vec![], methods: vec![], statics: vec![] }];
        let func_decls = [ExternDecl { name: "create", params: &[("x", "int")], ret: "int" }];
        let json = exports_to_json(&func_decls, &type_decls);
        let meta = parse_provider_json(&json).unwrap();

        assert_eq!(meta.types.len(), 1);
        assert_eq!(meta.types[0].name, "Sprite");
        assert_eq!(meta.functions.len(), 1);
        assert_eq!(meta.functions[0].name, "create");

        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        assert_eq!(stmts.len(), 2);
        assert!(matches!(stmts[0].node, Stmt::ExternType(_)));
        assert!(matches!(stmts[1].node, Stmt::ExternFunc(_)));
    }

    fn rich_type() -> ExternTypeDecl {
        ExternTypeDecl {
            name: "Point",
            has_init: false,
            fields: vec![
                ExternFieldDecl { name: "x", ty: "float" },
                ExternFieldDecl { name: "y", ty: "float" },
            ],
            methods: vec![ExternMethodDecl {
                name: "move_by",
                receiver: "var",
                params: &[("dx", "float"), ("dy", "float")],
                ret: "void",
            }],
            statics: vec![ExternStaticMethodDecl {
                name: "new",
                params: &[("x", "float"), ("y", "float")],
                ret: "Point",
            }],
        }
    }

    #[test]
    fn exports_to_json_rich_type() {
        let json = exports_to_json(&[], &[rich_type()]);
        assert!(json.contains("\"name\":\"Point\""));
        assert!(json.contains("\"name\":\"x\",\"type\":\"float\""));
        assert!(json.contains("\"name\":\"y\",\"type\":\"float\""));
        assert!(json.contains("\"name\":\"move_by\",\"receiver\":\"var\""));
        assert!(json.contains("\"name\":\"new\""));
        assert!(json.contains("\"ret\":\"Point\""));
    }

    #[test]
    fn parse_provider_json_rich_type() {
        let json = r#"{"types":[{"name":"Point","fields":[{"name":"x","type":"float"},{"name":"y","type":"float"}],"methods":[{"name":"move_by","receiver":"var","params":[["dx","float"],["dy","float"]],"ret":"void"}],"statics":[{"name":"new","params":[["x","float"],["y","float"]],"ret":"Point"}]}],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 1);
        let ty = &meta.types[0];
        assert_eq!(ty.name, "Point");
        assert_eq!(ty.fields.len(), 2);
        assert_eq!(ty.fields[0].name, "x");
        assert_eq!(ty.fields[0].ty, "float");
        assert_eq!(ty.fields[1].name, "y");
        assert_eq!(ty.methods.len(), 1);
        assert_eq!(ty.methods[0].name, "move_by");
        assert_eq!(ty.methods[0].receiver, "var");
        assert_eq!(ty.methods[0].params.len(), 2);
        assert_eq!(ty.methods[0].ret, "void");
        assert_eq!(ty.statics.len(), 1);
        assert_eq!(ty.statics[0].name, "new");
        assert_eq!(ty.statics[0].params.len(), 2);
        assert_eq!(ty.statics[0].ret, "Point");
    }

    #[test]
    fn round_trip_rich_types() {
        let json = exports_to_json(&[], &[rich_type()]);
        let meta = parse_provider_json(&json).unwrap();
        assert_eq!(meta.types.len(), 1);
        let ty = &meta.types[0];
        assert_eq!(ty.name, "Point");
        assert_eq!(ty.fields.len(), 2);
        assert_eq!(ty.fields[0].name, "x");
        assert_eq!(ty.fields[1].name, "y");
        assert_eq!(ty.methods.len(), 1);
        assert_eq!(ty.methods[0].name, "move_by");
        assert_eq!(ty.methods[0].receiver, "var");
        assert_eq!(ty.statics.len(), 1);
        assert_eq!(ty.statics[0].name, "new");
        assert_eq!(ty.statics[0].ret, "Point");
    }

    #[test]
    fn metadata_to_extern_stmts_with_members() {
        use crate::ast::{ExternTypeMember, MethodReceiver, Stmt};

        let meta = ExternProviderMeta {
            types: vec![ExternTypeMeta {
                name: "Point".to_string(),
                has_init: false,
                fields: vec![
                    ExternFieldMeta { name: "x".to_string(), ty: "float".to_string() },
                    ExternFieldMeta { name: "y".to_string(), ty: "float".to_string() },
                ],
                methods: vec![ExternMethodMeta {
                    name: "move_by".to_string(),
                    receiver: "var".to_string(),
                    params: vec![
                        ("dx".to_string(), "float".to_string()),
                        ("dy".to_string(), "float".to_string()),
                    ],
                    ret: "void".to_string(),
                }],
                statics: vec![ExternStaticMethodMeta {
                    name: "new".to_string(),
                    params: vec![
                        ("x".to_string(), "float".to_string()),
                        ("y".to_string(), "float".to_string()),
                    ],
                    ret: "Point".to_string(),
                }],
            }],
            functions: vec![],
        };

        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        assert_eq!(stmts.len(), 1);
        let Stmt::ExternType(ty_node) = &stmts[0].node else {
            panic!("expected ExternType");
        };
        assert_eq!(ty_node.node.name.to_string(), "Point");
        assert_eq!(ty_node.node.members.len(), 4);

        assert!(matches!(&ty_node.node.members[0], ExternTypeMember::Field { name, .. } if name.to_string() == "x"));
        assert!(matches!(&ty_node.node.members[1], ExternTypeMember::Field { name, .. } if name.to_string() == "y"));
        assert!(matches!(&ty_node.node.members[2], ExternTypeMember::Method { name, receiver, params, .. }
            if name.to_string() == "move_by" && *receiver == MethodReceiver::Var && params.len() == 2));
        assert!(matches!(&ty_node.node.members[3], ExternTypeMember::StaticMethod { name, params, .. }
            if name.to_string() == "new" && params.len() == 2));
    }

    #[test]
    fn metadata_to_extern_stmts_receiver_mapping() {
        use crate::ast::{ExternTypeMember, MethodReceiver, Stmt};

        let make_meta = |receiver: &str| ExternProviderMeta {
            types: vec![ExternTypeMeta {
                name: "T".to_string(),
                has_init: false,
                fields: vec![],
                methods: vec![ExternMethodMeta {
                    name: "m".to_string(),
                    receiver: receiver.to_string(),
                    params: vec![],
                    ret: "void".to_string(),
                }],
                statics: vec![],
            }],
            functions: vec![],
        };

        let stmts = metadata_to_extern_stmts(&make_meta("self")).unwrap();
        let Stmt::ExternType(ty) = &stmts[0].node else { panic!() };
        assert!(matches!(&ty.node.members[0], ExternTypeMember::Method { receiver, .. } if *receiver == MethodReceiver::Value));

        let stmts = metadata_to_extern_stmts(&make_meta("var")).unwrap();
        let Stmt::ExternType(ty) = &stmts[0].node else { panic!() };
        assert!(matches!(&ty.node.members[0], ExternTypeMember::Method { receiver, .. } if *receiver == MethodReceiver::Var));
    }

    #[test]
    fn parse_provider_json_type_missing_optional_arrays() {
        let json = r#"{"types":[{"name":"Empty"}],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 1);
        assert_eq!(meta.types[0].name, "Empty");
        assert!(meta.types[0].fields.is_empty());
        assert!(meta.types[0].methods.is_empty());
        assert!(meta.types[0].statics.is_empty());
    }

    #[test]
    fn exports_to_json_has_init() {
        let type_decls = [ExternTypeDecl {
            name: "Pt",
            has_init: true,
            fields: vec![],
            methods: vec![],
            statics: vec![],
        }];
        let json = exports_to_json(&[], &type_decls);
        assert!(json.contains("\"init\":true"));
    }

    #[test]
    fn exports_to_json_no_init() {
        let type_decls = [ExternTypeDecl {
            name: "Pt",
            has_init: false,
            fields: vec![],
            methods: vec![],
            statics: vec![],
        }];
        let json = exports_to_json(&[], &type_decls);
        assert!(!json.contains("\"init\""));
    }

    #[test]
    fn parse_provider_json_has_init() {
        let json = r#"{"types":[{"name":"Pt","init":true}],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 1);
        assert!(meta.types[0].has_init);
    }

    #[test]
    fn parse_provider_json_missing_init_defaults_false() {
        let json = r#"{"types":[{"name":"Pt"}],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 1);
        assert!(!meta.types[0].has_init);
    }

    #[test]
    fn metadata_to_extern_stmts_has_init() {
        use crate::ast::Stmt;

        let meta = ExternProviderMeta {
            types: vec![ExternTypeMeta {
                name: "Pt".to_string(),
                has_init: true,
                fields: vec![],
                methods: vec![],
                statics: vec![],
            }],
            functions: vec![],
        };
        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        let Stmt::ExternType(ty) = &stmts[0].node else { panic!("expected ExternType") };
        assert!(ty.node.has_init);
    }
}
