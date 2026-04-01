use crate::ast::{FuncParam, Type};

pub fn fold_type(ty: &Type, f: &mut impl FnMut(Type) -> Type) -> Type {
    let folded = match ty {
        Type::Func { params, ret } => Type::Func {
            params: params
                .iter()
                .map(|p| FuncParam::new(fold_type(&p.ty, f), p.mutable))
                .collect(),
            ret: Box::new(fold_type(ret, f)),
        },
        Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| fold_type(e, f)).collect()),
        Type::NamedTuple(fields) => {
            Type::NamedTuple(fields.iter().map(|(n, t)| (*n, fold_type(t, f))).collect())
        }
        Type::Struct { name, type_args } => Type::Struct {
            name: *name,
            type_args: type_args.iter().map(|a| fold_type(a, f)).collect(),
        },
        Type::DataRef { name, type_args } => Type::DataRef {
            name: *name,
            type_args: type_args.iter().map(|a| fold_type(a, f)).collect(),
        },
        Type::Enum { name, type_args } => Type::Enum {
            name: *name,
            type_args: type_args.iter().map(|a| fold_type(a, f)).collect(),
        },
        Type::List { elem } => Type::List {
            elem: Box::new(fold_type(elem, f)),
        },
        Type::Array { elem, len } => Type::Array {
            elem: Box::new(fold_type(elem, f)),
            len: *len,
        },
        Type::ArrayView { elem } => Type::ArrayView {
            elem: Box::new(fold_type(elem, f)),
        },
        Type::Map { key, value } => Type::Map {
            key: Box::new(fold_type(key, f)),
            value: Box::new(fold_type(value, f)),
        },
        // leaves: Infer, Any, Int, Float, Double, Bool, String, Void, Var, UnresolvedName, Extern
        // TODO: should be explicit here to avoid forgetting to add new leaves?
        _ => ty.clone(),
    };
    f(folded)
}

pub fn type_any(ty: &Type, pred: &mut impl FnMut(&Type) -> bool) -> bool {
    if pred(ty) {
        return true;
    }
    match ty {
        Type::Func { params, ret } => {
            params.iter().any(|p| type_any(&p.ty, pred)) || type_any(ret, pred)
        }
        Type::Tuple(elems) => elems.iter().any(|e| type_any(e, pred)),
        Type::NamedTuple(fields) => fields.iter().any(|(_, t)| type_any(t, pred)),
        Type::Struct { type_args, .. }
        | Type::DataRef { type_args, .. }
        | Type::Enum { type_args, .. } => type_args.iter().any(|a| type_any(a, pred)),
        Type::List { elem } | Type::Array { elem, .. } | Type::ArrayView { elem } => {
            type_any(elem, pred)
        }
        Type::Map { key, value } => type_any(key, pred) || type_any(value, pred),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use internment::Intern;

    use super::{fold_type, type_any};
    use crate::ast::{ArrayLen, FuncParam, Ident, Type, TypeVarId};

    fn ident(s: &str) -> Ident {
        Ident(Intern::new(s.to_string()))
    }

    fn var(id: u32) -> Type {
        Type::Var(TypeVarId(id))
    }

    // ---- fold_type tests ----

    #[test]
    fn fold_type_leaf_passthrough() {
        let result = fold_type(&Type::Int, &mut |t| t);
        assert_eq!(result, Type::Int);
    }

    #[test]
    fn fold_type_leaf_transform() {
        let result = fold_type(&Type::Int, &mut |t| match t {
            Type::Int => Type::Bool,
            other => other,
        });
        assert_eq!(result, Type::Bool);
    }

    #[test]
    fn fold_type_substitutes_var() {
        let ty = Type::Func {
            params: vec![FuncParam::new(var(0), false)],
            ret: Box::new(var(0)),
        };
        let result = fold_type(&ty, &mut |t| match t {
            Type::Var(TypeVarId(0)) => Type::Int,
            other => other,
        });
        assert_eq!(
            result,
            Type::Func {
                params: vec![FuncParam::new(Type::Int, false)],
                ret: Box::new(Type::Int),
            }
        );
    }

    #[test]
    fn fold_type_nested_struct() {
        let ty = Type::Struct {
            name: ident("Foo"),
            type_args: vec![Type::List {
                elem: Box::new(var(1)),
            }],
        };
        let result = fold_type(&ty, &mut |t| match t {
            Type::Var(TypeVarId(1)) => Type::String,
            other => other,
        });
        assert_eq!(
            result,
            Type::Struct {
                name: ident("Foo"),
                type_args: vec![Type::List {
                    elem: Box::new(Type::String),
                }],
            }
        );
    }

    #[test]
    fn fold_type_map() {
        let ty = Type::Map {
            key: Box::new(var(0)),
            value: Box::new(var(1)),
        };
        let result = fold_type(&ty, &mut |t| match t {
            Type::Var(TypeVarId(0)) => Type::Int,
            Type::Var(TypeVarId(1)) => Type::String,
            other => other,
        });
        assert_eq!(
            result,
            Type::Map {
                key: Box::new(Type::Int),
                value: Box::new(Type::String),
            }
        );
    }

    #[test]
    fn fold_type_named_tuple() {
        let ty = Type::NamedTuple(vec![(ident("x"), var(0)), (ident("y"), var(1))]);
        let result = fold_type(&ty, &mut |t| match t {
            Type::Var(TypeVarId(0)) => Type::Int,
            Type::Var(TypeVarId(1)) => Type::Bool,
            other => other,
        });
        assert_eq!(
            result,
            Type::NamedTuple(vec![(ident("x"), Type::Int), (ident("y"), Type::Bool)])
        );
    }

    #[test]
    fn fold_type_array_preserves_len() {
        let ty = Type::Array {
            elem: Box::new(var(0)),
            len: ArrayLen::Fixed(5),
        };
        let result = fold_type(&ty, &mut |t| match t {
            Type::Var(TypeVarId(0)) => Type::Float,
            other => other,
        });
        assert_eq!(
            result,
            Type::Array {
                elem: Box::new(Type::Float),
                len: ArrayLen::Fixed(5),
            }
        );
    }

    // ---- type_any tests ----

    #[test]
    fn type_any_leaf_match() {
        assert!(type_any(&Type::Infer, &mut |t| matches!(t, Type::Infer)));
        assert!(!type_any(&Type::Int, &mut |t| matches!(t, Type::Infer)));
    }

    #[test]
    fn type_any_nested_in_list() {
        let ty = Type::List {
            elem: Box::new(Type::Infer),
        };
        assert!(type_any(&ty, &mut |t| matches!(t, Type::Infer)));
    }

    #[test]
    fn type_any_nested_in_func() {
        let ty = Type::Func {
            params: vec![FuncParam::new(Type::Int, false)],
            ret: Box::new(Type::Infer),
        };
        assert!(type_any(&ty, &mut |t| matches!(t, Type::Infer)));
    }

    #[test]
    fn type_any_no_match() {
        let ty = Type::Map {
            key: Box::new(Type::String),
            value: Box::new(Type::Int),
        };
        assert!(!type_any(&ty, &mut |t| matches!(t, Type::Infer)));
    }

    #[test]
    fn type_any_enum_type_args() {
        let ty = Type::Enum {
            name: ident("Option"),
            type_args: vec![Type::Infer],
        };
        assert!(type_any(&ty, &mut |t| matches!(t, Type::Infer)));
    }

    #[test]
    fn type_any_deeply_nested() {
        let ty = Type::Map {
            key: Box::new(Type::String),
            value: Box::new(Type::List {
                elem: Box::new(Type::Array {
                    elem: Box::new(Type::Infer),
                    len: ArrayLen::Fixed(3),
                }),
            }),
        };
        assert!(type_any(&ty, &mut |t| matches!(t, Type::Infer)));
    }
}
