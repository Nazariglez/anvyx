#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternDecl {
    pub name: &'static str,
    pub params: &'static [(&'static str, &'static str)],
    pub ret: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternTypeDecl {
    pub name: &'static str,
}

pub fn exports_to_json(decls: &[ExternDecl], type_decls: &[ExternTypeDecl]) -> String {
    let mut out = String::from("{\"types\":[");
    for (i, ty) in type_decls.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        push_escaped(&mut out, ty.name);
        out.push('"');
    }
    out.push_str("],\"functions\":[");
    for (i, decl) in decls.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        push_escaped(&mut out, decl.name);
        out.push_str("\",\"params\":[");
        for (j, (pname, pty)) in decl.params.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push('[');
            out.push('"');
            push_escaped(&mut out, pname);
            out.push_str("\",\"");
            push_escaped(&mut out, pty);
            out.push_str("\"]");
        }
        out.push_str("],\"ret\":\"");
        push_escaped(&mut out, decl.ret);
        out.push_str("\"}");
    }
    out.push_str("]}");
    out
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
}

#[derive(Debug)]
pub struct ExternProviderMeta {
    pub types: Vec<ExternTypeMeta>,
    pub functions: Vec<ExternFuncMeta>,
}

pub fn parse_provider_json(json: &str) -> Result<ExternProviderMeta, String> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("Invalid metadata JSON: {e}"))?;

    let types = match v["types"].as_array() {
        Some(arr) => {
            let mut type_metas = vec![];
            for ty in arr {
                let name = ty.as_str()
                    .ok_or("Type entry is not a string")?
                    .to_string();
                type_metas.push(ExternTypeMeta { name });
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

        let params_arr = func["params"]
            .as_array()
            .ok_or("Function entry missing 'params'")?;

        let mut params = vec![];
        for param in params_arr {
            let arr = param.as_array().ok_or("Param entry is not an array")?;
            if arr.len() != 2 {
                return Err(format!("Param entry has {} elements, expected 2", arr.len()));
            }
            let pname = arr[0].as_str().ok_or("Param name is not a string")?.to_string();
            let pty = arr[1].as_str().ok_or("Param type is not a string")?.to_string();
            params.push((pname, pty));
        }

        let ret = func["ret"]
            .as_str()
            .ok_or("Function entry missing 'ret'")?
            .to_string();

        funcs.push(ExternFuncMeta { name, params, ret });
    }

    Ok(ExternProviderMeta { types, functions: funcs })
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
    use crate::ast::{ExternFunc, ExternType, Ident, Mutability, Param, Stmt};
    use crate::span::{Span, Spanned};
    use internment::Intern;

    let span = Span::new(0, 0);
    let mut stmts = vec![];

    for ty in &meta.types {
        let name = Ident(Intern::new(ty.name.clone()));
        let node = Spanned::new(ExternType { name }, span);
        stmts.push(Spanned::new(Stmt::ExternType(node), span));
    }

    for func in &meta.functions {
        let name = Ident(Intern::new(func.name.clone()));
        let params = func
            .params
            .iter()
            .map(|(pname, pty)| {
                Ok(Param {
                    mutability: Mutability::Immutable,
                    name: Ident(Intern::new(pname.clone())),
                    ty: anvyx_type_from_str(pty)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let ret = anvyx_type_from_str(&func.ret)?;

        let node = Spanned::new(ExternFunc { name, params, ret }, span);
        stmts.push(Spanned::new(Stmt::ExternFunc(node), span));
    }

    Ok(stmts)
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
            ExternTypeDecl { name: "Sprite" },
            ExternTypeDecl { name: "Texture" },
        ];
        let json = exports_to_json(&[], &type_decls);
        assert_eq!(json, "{\"types\":[\"Sprite\",\"Texture\"],\"functions\":[]}");
    }

    #[test]
    fn exports_to_json_with_types_and_functions() {
        let type_decls = [ExternTypeDecl { name: "Sprite" }];
        let func_decls = [ExternDecl { name: "create", params: &[], ret: "Sprite" }];
        let json = exports_to_json(&func_decls, &type_decls);
        assert!(json.contains("\"types\":[\"Sprite\"]"));
        assert!(json.contains("\"name\":\"create\""));
    }

    #[test]
    fn parse_provider_json_with_types() {
        let json = r#"{"types":["Sprite","Texture"],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 2);
        assert_eq!(meta.types[0].name, "Sprite");
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
            types: vec![ExternTypeMeta { name: "Sprite".to_string() }],
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

        let type_decls = [ExternTypeDecl { name: "Sprite" }];
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
}
