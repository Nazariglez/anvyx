use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExternDecl {
    pub name: &'static str,
    pub params: &'static [(&'static str, &'static str)],
    pub ret: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExternTypeDecl {
    pub name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<&'static str>,
    pub has_init: bool,
    pub fields: Vec<ExternFieldDecl>,
    pub methods: Vec<ExternMethodDecl>,
    pub statics: Vec<ExternStaticMethodDecl>,
    pub operators: Vec<ExternOpDecl>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternTypeDeclConst {
    pub name: &'static str,
    pub doc: Option<&'static str>,
    pub has_init: bool,
    pub fields: &'static [ExternFieldDecl],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExternFieldDecl {
    pub name: &'static str,
    pub ty: &'static str,
    pub computed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExternMethodDecl {
    pub name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<&'static str>,
    pub receiver: &'static str,
    pub params: &'static [(&'static str, &'static str)],
    pub ret: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExternStaticMethodDecl {
    pub name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<&'static str>,
    pub params: &'static [(&'static str, &'static str)],
    pub ret: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExternOpDecl {
    pub op: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs: Option<&'static str>,
    pub ret: &'static str,
}

#[derive(Serialize)]
struct ExportsJson<'a> {
    types: &'a [ExternTypeDecl],
    functions: &'a [ExternDecl],
}

pub fn exports_to_json(decls: &[ExternDecl], type_decls: &[ExternTypeDecl]) -> String {
    let wrapper = ExportsJson {
        types: type_decls,
        functions: decls,
    };
    serde_json::to_string(&wrapper).expect("metadata serialization cannot fail")
}

#[derive(Debug, Deserialize)]
pub struct ExternFuncMeta {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub ret: String,
    #[serde(default)]
    pub doc: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ExternTypeMeta {
    pub name: String,
    #[serde(default)]
    pub doc: Option<String>,
    #[serde(default)]
    pub has_init: bool,
    #[serde(default)]
    pub fields: Vec<ExternFieldMeta>,
    #[serde(default)]
    pub methods: Vec<ExternMethodMeta>,
    #[serde(default)]
    pub statics: Vec<ExternStaticMethodMeta>,
    #[serde(default)]
    pub operators: Vec<ExternOpMeta>,
}

#[derive(Debug, Deserialize)]
pub struct ExternFieldMeta {
    pub name: String,
    pub ty: String,
    #[serde(default)]
    pub computed: bool,
}

#[derive(Debug, Deserialize)]
pub struct ExternMethodMeta {
    pub name: String,
    #[serde(default)]
    pub doc: Option<String>,
    pub receiver: String,
    pub params: Vec<(String, String)>,
    pub ret: String,
}

#[derive(Debug, Deserialize)]
pub struct ExternStaticMethodMeta {
    pub name: String,
    #[serde(default)]
    pub doc: Option<String>,
    pub params: Vec<(String, String)>,
    pub ret: String,
}

#[derive(Debug, Deserialize)]
pub struct ExternOpMeta {
    pub op: String,
    #[serde(default)]
    pub rhs: Option<String>,
    #[serde(default)]
    pub lhs: Option<String>,
    pub ret: String,
}

#[derive(Debug, Deserialize)]
pub struct ExternProviderMeta {
    #[serde(default)]
    pub types: Vec<ExternTypeMeta>,
    pub functions: Vec<ExternFuncMeta>,
}

pub fn parse_provider_json(json: &str) -> Result<ExternProviderMeta, String> {
    serde_json::from_str(json).map_err(|e| format!("Invalid metadata JSON: {e}"))
}

pub(crate) fn anvyx_type_from_str(s: &str) -> Result<crate::ast::Type, String> {
    match s {
        "int" => Ok(crate::ast::Type::Int),
        "float" => Ok(crate::ast::Type::Float),
        "double" => Ok(crate::ast::Type::Double),
        "bool" => Ok(crate::ast::Type::Bool),
        "string" => Ok(crate::ast::Type::String),
        "void" => Ok(crate::ast::Type::Void),
        "any" => Ok(crate::ast::Type::Any),
        _ => crate::parser::parse_type_str(s)
            .map_err(|_| format!("Invalid Anvyx type in extern metadata: '{s}'")),
    }
}

fn op_str_to_binary_op(s: &str) -> Result<crate::ast::BinaryOp, String> {
    use crate::ast::BinaryOp;
    match s {
        "Add" => Ok(BinaryOp::Add),
        "Sub" => Ok(BinaryOp::Sub),
        "Mul" => Ok(BinaryOp::Mul),
        "Div" => Ok(BinaryOp::Div),
        "Rem" => Ok(BinaryOp::Rem),
        "Eq" => Ok(BinaryOp::Eq),
        other => Err(format!("unknown binary operator: {other}")),
    }
}

pub(crate) fn metadata_to_extern_stmts(
    meta: &ExternProviderMeta,
) -> Result<Vec<crate::ast::StmtNode>, String> {
    use crate::ast::{
        ExternFunc, ExternType, ExternTypeMember, Ident, MethodReceiver, Stmt, UnaryOp,
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
                computed: field.computed,
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
                doc: method.doc.clone(),
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
                doc: static_m.doc.clone(),
                name: Ident(Intern::new(static_m.name.clone())),
                params,
                ret,
            });
        }

        for op in &ty.operators {
            let ret = anvyx_type_from_str(&op.ret)?;
            match (op.rhs.as_deref(), op.lhs.as_deref()) {
                (None, None) => {
                    let unary_op = match op.op.as_str() {
                        "Neg" => UnaryOp::Neg,
                        other => return Err(format!("unknown unary operator: {other}")),
                    };
                    members.push(ExternTypeMember::UnaryOperator { op: unary_op, ret });
                }
                (Some(rhs_str), None) => {
                    let bin_op = op_str_to_binary_op(&op.op)?;
                    let other_ty = anvyx_type_from_str(rhs_str)?;
                    members.push(ExternTypeMember::Operator {
                        op: bin_op,
                        other_ty,
                        ret,
                        self_on_right: false,
                    });
                }
                (None, Some(lhs_str)) => {
                    let bin_op = op_str_to_binary_op(&op.op)?;
                    let other_ty = anvyx_type_from_str(lhs_str)?;
                    members.push(ExternTypeMember::Operator {
                        op: bin_op,
                        other_ty,
                        ret,
                        self_on_right: true,
                    });
                }
                (Some(_), Some(_)) => {
                    return Err("operator cannot have both 'rhs' and 'lhs'".to_string());
                }
            }
        }

        let node = Spanned::new(
            ExternType {
                doc: ty.doc.clone(),
                name,
                has_init: ty.has_init,
                members,
            },
            span,
        );
        stmts.push(Spanned::new(Stmt::ExternType(node), span));
    }

    for func in &meta.functions {
        let name = Ident(Intern::new(func.name.clone()));
        let params = params_to_ast(&func.params)?;
        let ret = anvyx_type_from_str(&func.ret)?;

        let node = Spanned::new(
            ExternFunc {
                doc: func.doc.clone(),
                name,
                params,
                ret,
            },
            span,
        );
        stmts.push(Spanned::new(Stmt::ExternFunc(node), span));
    }

    Ok(stmts)
}

fn params_to_ast(params: &[(String, String)]) -> Result<Vec<crate::ast::Param>, String> {
    use crate::ast::{Ident, Mutability, Param};
    use internment::Intern;

    params
        .iter()
        .map(|(pname, pty)| {
            Ok(Param {
                mutability: Mutability::Immutable,
                name: Ident(Intern::new(pname.clone())),
                ty: anvyx_type_from_str(pty)?,
                default: None,
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
            doc: None,
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
            ExternDecl {
                name: "add",
                params: &[("a", "int"), ("b", "int")],
                ret: "int",
                doc: None,
            },
            ExternDecl {
                name: "greet",
                params: &[("name", "string")],
                ret: "string",
                doc: None,
            },
        ];
        let json = exports_to_json(&decls, &[]);
        assert!(json.contains("\"name\":\"add\""));
        assert!(json.contains("\"name\":\"greet\""));
        assert!(json.starts_with("{\"types\":[],\"functions\":["));
        assert!(json.ends_with("]}"));
    }

    #[test]
    fn void_return() {
        let decl = ExternDecl {
            name: "noop",
            params: &[],
            ret: "void",
            doc: None,
        };
        let json = exports_to_json(&[decl], &[]);
        assert!(json.contains("\"ret\":\"void\""));
        assert!(json.contains("\"params\":[]"));
    }

    #[test]
    fn all_type_variants() {
        let decls = [
            ExternDecl {
                name: "f_int",
                params: &[("x", "int")],
                ret: "int",
                doc: None,
            },
            ExternDecl {
                name: "f_float",
                params: &[("x", "float")],
                ret: "float",
                doc: None,
            },
            ExternDecl {
                name: "f_bool",
                params: &[("x", "bool")],
                ret: "bool",
                doc: None,
            },
            ExternDecl {
                name: "f_string",
                params: &[("x", "string")],
                ret: "string",
                doc: None,
            },
        ];
        let json = exports_to_json(&decls, &[]);
        assert!(json.contains("\"int\""));
        assert!(json.contains("\"float\""));
        assert!(json.contains("\"bool\""));
        assert!(json.contains("\"string\""));
    }

    #[test]
    fn no_params() {
        let decl = ExternDecl {
            name: "get_val",
            params: &[],
            ret: "int",
            doc: None,
        };
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
        assert!(result.unwrap_err().contains("missing field `name`"));
    }

    #[test]
    fn parse_valid_json_round_trip() {
        let decls = [
            ExternDecl {
                name: "add",
                params: &[("a", "int"), ("b", "int")],
                ret: "int",
                doc: None,
            },
            ExternDecl {
                name: "greet",
                params: &[("name", "string")],
                ret: "string",
                doc: None,
            },
        ];
        let json = exports_to_json(&decls, &[]);
        let meta = parse_provider_json(&json).unwrap();
        assert_eq!(meta.functions.len(), 2);
        assert_eq!(meta.functions[0].name, "add");
        assert_eq!(
            meta.functions[0].params,
            vec![
                ("a".to_string(), "int".to_string()),
                ("b".to_string(), "int".to_string())
            ]
        );
        assert_eq!(meta.functions[0].ret, "int");
        assert_eq!(meta.functions[1].name, "greet");
        assert_eq!(
            meta.functions[1].params,
            vec![("name".to_string(), "string".to_string())]
        );
        assert_eq!(meta.functions[1].ret, "string");
    }

    #[test]
    fn anvyx_type_from_str_valid() {
        assert!(matches!(
            anvyx_type_from_str("int"),
            Ok(crate::ast::Type::Int)
        ));
        assert!(matches!(
            anvyx_type_from_str("float"),
            Ok(crate::ast::Type::Float)
        ));
        assert!(matches!(
            anvyx_type_from_str("bool"),
            Ok(crate::ast::Type::Bool)
        ));
        assert!(matches!(
            anvyx_type_from_str("string"),
            Ok(crate::ast::Type::String)
        ));
        assert!(matches!(
            anvyx_type_from_str("void"),
            Ok(crate::ast::Type::Void)
        ));
    }

    #[test]
    fn anvyx_type_from_str_any() {
        assert!(matches!(
            anvyx_type_from_str("any"),
            Ok(crate::ast::Type::Any)
        ));
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
            functions: vec![ExternFuncMeta {
                name: "add".to_string(),
                params: vec![
                    ("a".to_string(), "int".to_string()),
                    ("b".to_string(), "int".to_string()),
                ],
                ret: "int".to_string(),
                doc: None,
            }],
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
                doc: None,
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
            ExternTypeDecl {
                name: "Sprite",
                doc: None,
                has_init: false,
                fields: vec![],
                methods: vec![],
                statics: vec![],
                operators: vec![],
            },
            ExternTypeDecl {
                name: "Texture",
                doc: None,
                has_init: false,
                fields: vec![],
                methods: vec![],
                statics: vec![],
                operators: vec![],
            },
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
        let type_decls = [ExternTypeDecl {
            name: "Sprite",
            doc: None,
            has_init: false,
            fields: vec![],
            methods: vec![],
            statics: vec![],
            operators: vec![],
        }];
        let func_decls = [ExternDecl {
            name: "create",
            params: &[],
            ret: "Sprite",
            doc: None,
        }];
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
                doc: None,
                has_init: false,
                fields: vec![],
                methods: vec![],
                statics: vec![],
                operators: vec![],
            }],
            functions: vec![ExternFuncMeta {
                name: "create".to_string(),
                params: vec![],
                ret: "int".to_string(),
                doc: None,
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

        let type_decls = [ExternTypeDecl {
            name: "Sprite",
            doc: None,
            has_init: false,
            fields: vec![],
            methods: vec![],
            statics: vec![],
            operators: vec![],
        }];
        let func_decls = [ExternDecl {
            name: "create",
            params: &[("x", "int")],
            ret: "int",
            doc: None,
        }];
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
            doc: None,
            has_init: false,
            fields: vec![
                ExternFieldDecl {
                    name: "x",
                    ty: "float",
                    computed: false,
                },
                ExternFieldDecl {
                    name: "y",
                    ty: "float",
                    computed: false,
                },
            ],
            methods: vec![ExternMethodDecl {
                name: "move_by",
                doc: None,
                receiver: "var",
                params: &[("dx", "float"), ("dy", "float")],
                ret: "void",
            }],
            statics: vec![ExternStaticMethodDecl {
                name: "new",
                doc: None,
                params: &[("x", "float"), ("y", "float")],
                ret: "Point",
            }],
            operators: vec![],
        }
    }

    #[test]
    fn exports_to_json_rich_type() {
        let json = exports_to_json(&[], &[rich_type()]);
        assert!(json.contains("\"name\":\"Point\""));
        assert!(json.contains("\"name\":\"x\",\"ty\":\"float\""));
        assert!(json.contains("\"name\":\"y\",\"ty\":\"float\""));
        assert!(json.contains("\"name\":\"move_by\",\"receiver\":\"var\""));
        assert!(json.contains("\"name\":\"new\""));
        assert!(json.contains("\"ret\":\"Point\""));
    }

    #[test]
    fn parse_provider_json_rich_type() {
        let json = r#"{"types":[{"name":"Point","fields":[{"name":"x","ty":"float"},{"name":"y","ty":"float"}],"methods":[{"name":"move_by","receiver":"var","params":[["dx","float"],["dy","float"]],"ret":"void"}],"statics":[{"name":"new","params":[["x","float"],["y","float"]],"ret":"Point"}]}],"functions":[]}"#;
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
                doc: None,
                has_init: false,
                fields: vec![
                    ExternFieldMeta {
                        name: "x".to_string(),
                        ty: "float".to_string(),
                        computed: false,
                    },
                    ExternFieldMeta {
                        name: "y".to_string(),
                        ty: "float".to_string(),
                        computed: false,
                    },
                ],
                methods: vec![ExternMethodMeta {
                    name: "move_by".to_string(),
                    doc: None,
                    receiver: "var".to_string(),
                    params: vec![
                        ("dx".to_string(), "float".to_string()),
                        ("dy".to_string(), "float".to_string()),
                    ],
                    ret: "void".to_string(),
                }],
                statics: vec![ExternStaticMethodMeta {
                    name: "new".to_string(),
                    doc: None,
                    params: vec![
                        ("x".to_string(), "float".to_string()),
                        ("y".to_string(), "float".to_string()),
                    ],
                    ret: "Point".to_string(),
                }],
                operators: vec![],
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

        assert!(
            matches!(&ty_node.node.members[0], ExternTypeMember::Field { name, .. } if name.to_string() == "x")
        );
        assert!(
            matches!(&ty_node.node.members[1], ExternTypeMember::Field { name, .. } if name.to_string() == "y")
        );
        assert!(
            matches!(&ty_node.node.members[2], ExternTypeMember::Method { name, receiver, params, .. }
            if name.to_string() == "move_by" && *receiver == MethodReceiver::Var && params.len() == 2)
        );
        assert!(
            matches!(&ty_node.node.members[3], ExternTypeMember::StaticMethod { name, params, .. }
            if name.to_string() == "new" && params.len() == 2)
        );
    }

    #[test]
    fn metadata_to_extern_stmts_receiver_mapping() {
        use crate::ast::{ExternTypeMember, MethodReceiver, Stmt};

        let make_meta = |receiver: &str| ExternProviderMeta {
            types: vec![ExternTypeMeta {
                name: "T".to_string(),
                doc: None,
                has_init: false,
                fields: vec![],
                methods: vec![ExternMethodMeta {
                    name: "m".to_string(),
                    doc: None,
                    receiver: receiver.to_string(),
                    params: vec![],
                    ret: "void".to_string(),
                }],
                statics: vec![],
                operators: vec![],
            }],
            functions: vec![],
        };

        let stmts = metadata_to_extern_stmts(&make_meta("self")).unwrap();
        let Stmt::ExternType(ty) = &stmts[0].node else {
            panic!()
        };
        assert!(
            matches!(&ty.node.members[0], ExternTypeMember::Method { receiver, .. } if *receiver == MethodReceiver::Value)
        );

        let stmts = metadata_to_extern_stmts(&make_meta("var")).unwrap();
        let Stmt::ExternType(ty) = &stmts[0].node else {
            panic!()
        };
        assert!(
            matches!(&ty.node.members[0], ExternTypeMember::Method { receiver, .. } if *receiver == MethodReceiver::Var)
        );
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
            doc: None,
            has_init: true,
            fields: vec![],
            methods: vec![],
            statics: vec![],
            operators: vec![],
        }];
        let json = exports_to_json(&[], &type_decls);
        assert!(json.contains("\"has_init\":true"));
    }

    #[test]
    fn exports_to_json_no_init() {
        let type_decls = [ExternTypeDecl {
            name: "Pt",
            doc: None,
            has_init: false,
            fields: vec![],
            methods: vec![],
            statics: vec![],
            operators: vec![],
        }];
        let json = exports_to_json(&[], &type_decls);
        assert!(json.contains("\"has_init\":false"));
    }

    #[test]
    fn parse_provider_json_has_init() {
        let json = r#"{"types":[{"name":"Pt","has_init":true}],"functions":[]}"#;
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
                doc: None,
                has_init: true,
                fields: vec![],
                methods: vec![],
                statics: vec![],
                operators: vec![],
            }],
            functions: vec![],
        };
        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        let Stmt::ExternType(ty) = &stmts[0].node else {
            panic!("expected ExternType")
        };
        assert!(ty.node.has_init);
    }

    #[test]
    fn exports_to_json_with_operators() {
        let type_decls = [ExternTypeDecl {
            name: "Vec2",
            doc: None,
            has_init: false,
            fields: vec![],
            methods: vec![],
            statics: vec![],
            operators: vec![
                ExternOpDecl {
                    op: "Add",
                    rhs: Some("Vec2"),
                    lhs: None,
                    ret: "Vec2",
                },
                ExternOpDecl {
                    op: "Mul",
                    rhs: Some("float"),
                    lhs: None,
                    ret: "Vec2",
                },
                ExternOpDecl {
                    op: "Mul",
                    rhs: None,
                    lhs: Some("float"),
                    ret: "Vec2",
                },
                ExternOpDecl {
                    op: "Neg",
                    rhs: None,
                    lhs: None,
                    ret: "Vec2",
                },
                ExternOpDecl {
                    op: "Eq",
                    rhs: Some("Vec2"),
                    lhs: None,
                    ret: "bool",
                },
            ],
        }];
        let json = exports_to_json(&[], &type_decls);
        assert!(json.contains("\"operators\":["));
        assert!(json.contains("\"op\":\"Add\""));
        assert!(json.contains("\"rhs\":\"Vec2\""));
        assert!(json.contains("\"op\":\"Neg\""));
        assert!(json.contains("\"op\":\"Eq\""));
        assert!(json.contains("\"ret\":\"bool\""));
        // lhs-only entry should not have rhs key
        assert!(json.contains("\"lhs\":\"float\""));
    }

    #[test]
    fn parse_provider_json_with_operators() {
        let json = r#"{"types":[{"name":"Vec2","operators":[{"op":"Add","rhs":"Vec2","ret":"Vec2"},{"op":"Neg","ret":"Vec2"},{"op":"Mul","lhs":"float","ret":"Vec2"}]}],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 1);
        let ty = &meta.types[0];
        assert_eq!(ty.operators.len(), 3);
        assert_eq!(ty.operators[0].op, "Add");
        assert_eq!(ty.operators[0].rhs, Some("Vec2".to_string()));
        assert!(ty.operators[0].lhs.is_none());
        assert_eq!(ty.operators[0].ret, "Vec2");
        assert_eq!(ty.operators[1].op, "Neg");
        assert!(ty.operators[1].rhs.is_none());
        assert!(ty.operators[1].lhs.is_none());
        assert_eq!(ty.operators[2].op, "Mul");
        assert!(ty.operators[2].rhs.is_none());
        assert_eq!(ty.operators[2].lhs, Some("float".to_string()));
    }

    #[test]
    fn parse_provider_json_missing_operators_defaults_empty() {
        let json = r#"{"types":[{"name":"Pt"}],"functions":[]}"#;
        let meta = parse_provider_json(json).unwrap();
        assert_eq!(meta.types.len(), 1);
        assert!(meta.types[0].operators.is_empty());
    }

    #[test]
    fn round_trip_operators() {
        let type_decls = [ExternTypeDecl {
            name: "Vec2",
            doc: None,
            has_init: false,
            fields: vec![],
            methods: vec![],
            statics: vec![],
            operators: vec![
                ExternOpDecl {
                    op: "Add",
                    rhs: Some("Vec2"),
                    lhs: None,
                    ret: "Vec2",
                },
                ExternOpDecl {
                    op: "Neg",
                    rhs: None,
                    lhs: None,
                    ret: "Vec2",
                },
                ExternOpDecl {
                    op: "Eq",
                    rhs: Some("Vec2"),
                    lhs: None,
                    ret: "bool",
                },
            ],
        }];
        let json = exports_to_json(&[], &type_decls);
        let meta = parse_provider_json(&json).unwrap();
        assert_eq!(meta.types.len(), 1);
        let ty = &meta.types[0];
        assert_eq!(ty.operators.len(), 3);
        assert_eq!(ty.operators[0].op, "Add");
        assert_eq!(ty.operators[0].rhs, Some("Vec2".to_string()));
        assert!(ty.operators[0].lhs.is_none());
        assert_eq!(ty.operators[1].op, "Neg");
        assert!(ty.operators[1].rhs.is_none());
        assert!(ty.operators[1].lhs.is_none());
        assert_eq!(ty.operators[2].op, "Eq");
        assert_eq!(ty.operators[2].ret, "bool");
    }

    #[test]
    fn metadata_to_extern_stmts_with_operators() {
        use crate::ast::{BinaryOp, ExternTypeMember, Stmt, Type, UnaryOp};

        let meta = ExternProviderMeta {
            types: vec![ExternTypeMeta {
                name: "Vec2".to_string(),
                doc: None,
                has_init: false,
                fields: vec![],
                methods: vec![],
                statics: vec![],
                operators: vec![
                    ExternOpMeta {
                        op: "Add".to_string(),
                        rhs: Some("Vec2".to_string()),
                        lhs: None,
                        ret: "Vec2".to_string(),
                    },
                    ExternOpMeta {
                        op: "Neg".to_string(),
                        rhs: None,
                        lhs: None,
                        ret: "Vec2".to_string(),
                    },
                    ExternOpMeta {
                        op: "Eq".to_string(),
                        rhs: Some("Vec2".to_string()),
                        lhs: None,
                        ret: "bool".to_string(),
                    },
                ],
            }],
            functions: vec![],
        };

        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        assert_eq!(stmts.len(), 1);
        let Stmt::ExternType(ty_node) = &stmts[0].node else {
            panic!("expected ExternType")
        };
        assert_eq!(ty_node.node.members.len(), 3);

        assert!(matches!(
            &ty_node.node.members[0],
            ExternTypeMember::Operator {
                op: BinaryOp::Add,
                self_on_right: false,
                ..
            }
        ));
        assert!(matches!(
            &ty_node.node.members[1],
            ExternTypeMember::UnaryOperator {
                op: UnaryOp::Neg,
                ..
            }
        ));
        assert!(matches!(
            &ty_node.node.members[2],
            ExternTypeMember::Operator {
                op: BinaryOp::Eq,
                ret: Type::Bool,
                self_on_right: false,
                ..
            }
        ));
    }

    #[test]
    fn metadata_to_extern_stmts_operator_rhs_and_lhs() {
        use crate::ast::{BinaryOp, ExternTypeMember, Stmt};

        let meta = ExternProviderMeta {
            types: vec![ExternTypeMeta {
                name: "Vec2".to_string(),
                doc: None,
                has_init: false,
                fields: vec![],
                methods: vec![],
                statics: vec![],
                operators: vec![
                    ExternOpMeta {
                        op: "Mul".to_string(),
                        rhs: Some("float".to_string()),
                        lhs: None,
                        ret: "Vec2".to_string(),
                    },
                    ExternOpMeta {
                        op: "Mul".to_string(),
                        rhs: None,
                        lhs: Some("float".to_string()),
                        ret: "Vec2".to_string(),
                    },
                ],
            }],
            functions: vec![],
        };

        let stmts = metadata_to_extern_stmts(&meta).unwrap();
        let Stmt::ExternType(ty_node) = &stmts[0].node else {
            panic!()
        };
        assert_eq!(ty_node.node.members.len(), 2);

        assert!(matches!(
            &ty_node.node.members[0],
            ExternTypeMember::Operator {
                op: BinaryOp::Mul,
                self_on_right: false,
                ..
            }
        ));
        assert!(matches!(
            &ty_node.node.members[1],
            ExternTypeMember::Operator {
                op: BinaryOp::Mul,
                self_on_right: true,
                ..
            }
        ));
    }
}
