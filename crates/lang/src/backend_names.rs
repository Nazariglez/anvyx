use internment::Intern;

use crate::{
    ast::{ArrayLen, Ident, OPTION_ENUM_NAME, Type},
    typecheck::ExtendSpecKey,
};

pub fn encode_type(ty: &Type) -> String {
    match ty {
        Type::Int => "i64".to_string(),
        Type::Float => "f32".to_string(),
        Type::Double => "f64".to_string(),
        Type::Bool => "bool".to_string(),
        Type::String => "String".to_string(),
        Type::Void => "void".to_string(),
        Type::Any => "Any".to_string(),
        Type::Infer => "Infer".to_string(),
        Type::Var(id) => format!("V{}", id.0),
        Type::UnresolvedName(ident) => ident.to_string(),
        Type::Enum { name, type_args } if name.0.as_ref() == OPTION_ENUM_NAME => {
            let inner = type_args.first().map_or("Infer".to_string(), encode_type);
            format!("Opt_{inner}")
        }
        Type::Enum { name, type_args }
        | Type::Struct { name, type_args }
        | Type::DataRef { name, type_args } => encode_named_with_args(name.to_string(), type_args),
        Type::Tuple(elems) => {
            let n = elems.len();
            let parts: Vec<_> = elems.iter().map(encode_type).collect();
            format!("Tup{}_{}", n, parts.join("_"))
        }
        Type::NamedTuple(fields) => {
            let n = fields.len();
            let parts: Vec<_> = fields.iter().map(|(_, ty)| encode_type(ty)).collect();
            format!("Tup{}_{}", n, parts.join("_"))
        }
        Type::List { elem } => format!("List_{}", encode_type(elem)),
        Type::Array { elem, len } => {
            let elem_enc = encode_type(elem);
            let len_enc = match len {
                ArrayLen::Fixed(n) => n.to_string(),
                ArrayLen::Infer => "I".to_string(),
                ArrayLen::Named(i) => i.to_string(),
                ArrayLen::Param(id) => format!("p{}", id.0),
            };
            format!("Arr_{elem_enc}_{len_enc}")
        }
        Type::Map { key, value } => {
            format!("Map_{}_{}", encode_type(key), encode_type(value))
        }
        Type::ArrayView { elem } => format!("ArrView_{}", encode_type(elem)),
        Type::Extern { name } => format!("Ext_{name}"),
        Type::Func { .. } => "Fn".to_string(),
    }
}

fn encode_named_with_args(name: String, type_args: &[Type]) -> String {
    if type_args.is_empty() {
        name
    } else {
        let parts: Vec<_> = type_args.iter().map(encode_type).collect();
        format!("{}_{}", name, parts.join("_"))
    }
}

pub fn encode_generic_suffix(type_args: &[Type], const_args: &[usize]) -> String {
    let mut parts: Vec<String> = type_args.iter().map(encode_type).collect();
    for c in const_args {
        parts.push(format!("c{c}"));
    }
    parts.join("_")
}

pub fn encode_specialization_name(base: Ident, type_args: &[Type], const_args: &[usize]) -> Ident {
    let suffix = encode_generic_suffix(type_args, const_args);
    Ident(Intern::new(format!("{base}${suffix}")))
}

pub fn encode_method_specialization_name(
    owner: Ident,
    method: Ident,
    type_args: &[Type],
    const_args: &[usize],
) -> Ident {
    if type_args.is_empty() && const_args.is_empty() {
        return Ident(Intern::new(format!("{owner}::{method}")));
    }
    let suffix = encode_generic_suffix(type_args, const_args);
    Ident(Intern::new(format!("{owner}::{method}${suffix}")))
}

pub fn encode_extend_name(module: &str, target_ty: &Type, method: Ident) -> Ident {
    let type_enc = encode_type(target_ty);
    Ident(Intern::new(format!(
        "__extend::{module}::{type_enc}::{method}"
    )))
}

pub fn encode_cast_name(module: &str, target_ty: &Type, source_ty: &Type) -> Ident {
    let target_enc = encode_type(target_ty);
    let source_enc = encode_type(source_ty);
    Ident(Intern::new(format!(
        "__cast::{module}::{target_enc}::from_{source_enc}"
    )))
}

pub fn encode_extend_specialization_name(key: &ExtendSpecKey, source_module: &[String]) -> Ident {
    let module_part = if source_module.is_empty() {
        String::new()
    } else {
        source_module.join("::")
    };
    let type_enc = encode_type(&key.target_type);
    let args_enc = encode_generic_suffix(&key.type_args, &key.const_args);
    let spec_part = if args_enc.is_empty() {
        type_enc
    } else {
        format!("{type_enc}${args_enc}")
    };
    Ident(Intern::new(format!(
        "__extend::{module_part}::{spec_part}::{}",
        key.method_name
    )))
}

pub fn mangle_for_rust(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ':' if chars.peek() == Some(&':') => {
                chars.next();
                out.push_str("__");
            }
            '$' => out.push_str("_S"),
            c if c.is_ascii_alphanumeric() || c == '_' => out.push(c),
            _ => out.push('_'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ArrayLen, ConstParamId, TypeVarId};

    fn ident(s: &str) -> Ident {
        Ident(Intern::new(s.to_string()))
    }

    #[test]
    fn primitives() {
        assert_eq!(encode_type(&Type::Int), "i64");
        assert_eq!(encode_type(&Type::Float), "f32");
        assert_eq!(encode_type(&Type::Double), "f64");
        assert_eq!(encode_type(&Type::Bool), "bool");
        assert_eq!(encode_type(&Type::String), "String");
        assert_eq!(encode_type(&Type::Void), "void");
        assert_eq!(encode_type(&Type::Any), "Any");
        assert_eq!(encode_type(&Type::Infer), "Infer");
    }

    #[test]
    fn type_var() {
        assert_eq!(encode_type(&Type::Var(TypeVarId(3))), "V3");
    }

    #[test]
    fn option_type() {
        let opt_int = Type::Enum {
            name: ident("Option"),
            type_args: vec![Type::Int],
        };
        assert_eq!(encode_type(&opt_int), "Opt_i64");
    }

    #[test]
    fn option_nested() {
        let inner = Type::Tuple(vec![Type::Int, Type::String]);
        let opt = Type::Enum {
            name: ident("Option"),
            type_args: vec![inner],
        };
        assert_eq!(encode_type(&opt), "Opt_Tup2_i64_String");
    }

    #[test]
    fn tuple_type() {
        let t = Type::Tuple(vec![Type::String, Type::Int]);
        assert_eq!(encode_type(&t), "Tup2_String_i64");
    }

    #[test]
    fn generic_struct() {
        let t = Type::Struct {
            name: ident("Pair"),
            type_args: vec![Type::Int],
        };
        assert_eq!(encode_type(&t), "Pair_i64");
    }

    #[test]
    fn plain_enum() {
        let t = Type::Enum {
            name: ident("Color"),
            type_args: vec![],
        };
        assert_eq!(encode_type(&t), "Color");
    }

    #[test]
    fn list_type() {
        let t = Type::List {
            elem: Box::new(Type::Int),
        };
        assert_eq!(encode_type(&t), "List_i64");
    }

    #[test]
    fn array_fixed() {
        let t = Type::Array {
            elem: Box::new(Type::Int),
            len: ArrayLen::Fixed(4),
        };
        assert_eq!(encode_type(&t), "Arr_i64_4");
    }

    #[test]
    fn array_infer() {
        let t = Type::Array {
            elem: Box::new(Type::Int),
            len: ArrayLen::Infer,
        };
        assert_eq!(encode_type(&t), "Arr_i64_I");
    }

    #[test]
    fn array_param() {
        let t = Type::Array {
            elem: Box::new(Type::Int),
            len: ArrayLen::Param(ConstParamId(2)),
        };
        assert_eq!(encode_type(&t), "Arr_i64_p2");
    }

    #[test]
    fn map_type() {
        let t = Type::Map {
            key: Box::new(Type::String),
            value: Box::new(Type::Int),
        };
        assert_eq!(encode_type(&t), "Map_String_i64");
    }

    #[test]
    fn specialization_name() {
        let name = encode_specialization_name(ident("foo"), &[Type::Int], &[]);
        assert_eq!(name.to_string(), "foo$i64");
    }

    #[test]
    fn method_spec_name_with_args() {
        let name =
            encode_method_specialization_name(ident("Vec"), ident("push"), &[Type::Int], &[]);
        assert_eq!(name.to_string(), "Vec::push$i64");
    }

    #[test]
    fn method_spec_name_no_args() {
        let name = encode_method_specialization_name(ident("Foo"), ident("bar"), &[], &[]);
        assert_eq!(name.to_string(), "Foo::bar");
    }

    #[test]
    fn mangle_rust_plain() {
        assert_eq!(mangle_for_rust("main"), "main");
    }

    #[test]
    fn mangle_rust_colons() {
        assert_eq!(mangle_for_rust("Foo::bar"), "Foo__bar");
    }

    #[test]
    fn mangle_rust_dollar() {
        assert_eq!(mangle_for_rust("foo$i64"), "foo_Si64");
    }

    #[test]
    fn mangle_rust_extend() {
        assert_eq!(
            mangle_for_rust("__extend::mod::Opt_i64::unwrap"),
            "__extend__mod__Opt_i64__unwrap"
        );
    }

    #[test]
    fn cast_name_encoding() {
        let target = Type::Struct {
            name: ident("Vec2"),
            type_args: vec![],
        };
        let name = encode_cast_name("main", &target, &Type::Float);
        assert_eq!(name.to_string(), "__cast::main::Vec2::from_f32");
    }

    #[test]
    fn encoded_types_are_identifier_safe() {
        let types = vec![
            Type::Int,
            Type::Float,
            Type::Double,
            Type::Bool,
            Type::String,
            Type::Void,
            Type::Enum {
                name: ident("Option"),
                type_args: vec![Type::Int],
            },
            Type::Tuple(vec![Type::Int, Type::String]),
            Type::List {
                elem: Box::new(Type::Int),
            },
            Type::Map {
                key: Box::new(Type::String),
                value: Box::new(Type::Int),
            },
        ];
        for ty in &types {
            let enc = encode_type(ty);
            assert!(
                enc.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "encoded type contains invalid chars: {enc:?} (from {ty:?})"
            );
        }
    }
}
