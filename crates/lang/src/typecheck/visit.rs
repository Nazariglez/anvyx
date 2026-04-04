use crate::ast::{FuncParam, Ident, Type};

fn walk_type_pairs<F>(left: &[Type], right: &[Type], relate: &mut F) -> bool
where
    F: FnMut(&Type, &Type) -> bool,
{
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| relate(left, right))
}

fn walk_named_type_args<F>(
    left_name: Ident,
    left_args: &[Type],
    right_name: Ident,
    right_args: &[Type],
    relate: &mut F,
) -> bool
where
    F: FnMut(&Type, &Type) -> bool,
{
    left_name == right_name && walk_type_pairs(left_args, right_args, relate)
}

fn map_named_type_args<F>(
    left_name: Ident,
    left_args: &[Type],
    right_name: Ident,
    right_args: &[Type],
    map: &mut F,
    make: impl Fn(Ident, Vec<Type>) -> Type,
) -> Option<Type>
where
    F: FnMut(&Type, &Type) -> Option<Type>,
{
    if left_name != right_name || left_args.len() != right_args.len() {
        return None;
    }
    let type_args: Option<Vec<Type>> = left_args
        .iter()
        .zip(right_args.iter())
        .map(|(l, r)| map(l, r))
        .collect();
    Some(make(left_name, type_args?))
}

fn walk_func_params<F>(left: &[FuncParam], right: &[FuncParam], relate: &mut F) -> bool
where
    F: FnMut(&Type, &Type) -> bool,
{
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.mutable == right.mutable && relate(&left.ty, &right.ty))
}

fn walk_named_tuple_fields<F>(
    left: &[(Ident, Type)],
    right: &[(Ident, Type)],
    relate: &mut F,
) -> bool
where
    F: FnMut(&Type, &Type) -> bool,
{
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((left_name, left_ty), (right_name, right_ty))| {
                left_name == right_name && relate(left_ty, right_ty)
            })
}

pub fn walk_type_structure<F>(left: &Type, right: &Type, relate: &mut F) -> bool
where
    F: FnMut(&Type, &Type) -> bool,
{
    match (left, right) {
        (Type::List { elem: left }, Type::List { elem: right })
        | (Type::Array { elem: left, .. }, Type::Array { elem: right, .. })
        | (Type::ArrayView { elem: left }, Type::ArrayView { elem: right }) => relate(left, right),
        (
            Type::Map {
                key: left_key,
                value: left_value,
            },
            Type::Map {
                key: right_key,
                value: right_value,
            },
        ) => relate(left_key, right_key) && relate(left_value, right_value),
        (
            Type::Func {
                params: left_params,
                ret: left_ret,
            },
            Type::Func {
                params: right_params,
                ret: right_ret,
            },
        ) => walk_func_params(left_params, right_params, relate) && relate(left_ret, right_ret),
        (
            Type::Struct {
                name: left_name,
                type_args: left_args,
            },
            Type::Struct {
                name: right_name,
                type_args: right_args,
            },
        )
        | (
            Type::DataRef {
                name: left_name,
                type_args: left_args,
            },
            Type::DataRef {
                name: right_name,
                type_args: right_args,
            },
        )
        | (
            Type::Enum {
                name: left_name,
                type_args: left_args,
            },
            Type::Enum {
                name: right_name,
                type_args: right_args,
            },
        ) => walk_named_type_args(*left_name, left_args, *right_name, right_args, relate),
        (Type::Tuple(left_elems), Type::Tuple(right_elems)) => {
            walk_type_pairs(left_elems, right_elems, relate)
        }
        (Type::NamedTuple(left_fields), Type::NamedTuple(right_fields)) => {
            walk_named_tuple_fields(left_fields, right_fields, relate)
        }
        _ => false,
    }
}

pub fn map_type_structure<F>(left: &Type, right: &Type, map: &mut F) -> Option<Type>
where
    F: FnMut(&Type, &Type) -> Option<Type>,
{
    match (left, right) {
        (Type::List { elem: left }, Type::List { elem: right }) => Some(Type::List {
            elem: map(left, right)?.boxed(),
        }),
        (Type::ArrayView { elem: left }, Type::ArrayView { elem: right }) => {
            Some(Type::ArrayView {
                elem: map(left, right)?.boxed(),
            })
        }
        (
            Type::Map {
                key: left_key,
                value: left_value,
            },
            Type::Map {
                key: right_key,
                value: right_value,
            },
        ) => {
            let key = map(left_key, right_key)?;
            let value = map(left_value, right_value)?;
            Some(Type::Map {
                key: key.boxed(),
                value: value.boxed(),
            })
        }
        (
            Type::Func {
                params: left_params,
                ret: left_ret,
            },
            Type::Func {
                params: right_params,
                ret: right_ret,
            },
        ) => {
            if left_params.len() != right_params.len() {
                return None;
            }
            let new_params: Option<Vec<FuncParam>> = left_params
                .iter()
                .zip(right_params.iter())
                .map(|(l, r)| {
                    if l.mutable != r.mutable {
                        return None;
                    }
                    map(&l.ty, &r.ty).map(|ty| FuncParam::new(ty, l.mutable))
                })
                .collect();
            let ret = map(left_ret, right_ret)?;
            Some(Type::Func {
                params: new_params?,
                ret: ret.boxed(),
            })
        }
        (
            Type::Struct {
                name: left_name,
                type_args: left_args,
            },
            Type::Struct {
                name: right_name,
                type_args: right_args,
            },
        ) => map_named_type_args(
            *left_name,
            left_args,
            *right_name,
            right_args,
            map,
            |n, a| Type::Struct {
                name: n,
                type_args: a,
            },
        ),
        (
            Type::DataRef {
                name: left_name,
                type_args: left_args,
            },
            Type::DataRef {
                name: right_name,
                type_args: right_args,
            },
        ) => map_named_type_args(
            *left_name,
            left_args,
            *right_name,
            right_args,
            map,
            |n, a| Type::DataRef {
                name: n,
                type_args: a,
            },
        ),
        (
            Type::Enum {
                name: left_name,
                type_args: left_args,
            },
            Type::Enum {
                name: right_name,
                type_args: right_args,
            },
        ) => map_named_type_args(
            *left_name,
            left_args,
            *right_name,
            right_args,
            map,
            |n, a| Type::Enum {
                name: n,
                type_args: a,
            },
        ),
        (Type::Tuple(left_elems), Type::Tuple(right_elems))
            if left_elems.len() == right_elems.len() =>
        {
            let elems: Option<Vec<Type>> = left_elems
                .iter()
                .zip(right_elems.iter())
                .map(|(l, r)| map(l, r))
                .collect();
            Some(Type::Tuple(elems?))
        }
        (Type::NamedTuple(left_fields), Type::NamedTuple(right_fields))
            if left_fields.len() == right_fields.len()
                && left_fields
                    .iter()
                    .zip(right_fields.iter())
                    .all(|((ln, _), (rn, _))| ln == rn) =>
        {
            let fields: Option<Vec<(Ident, Type)>> = left_fields
                .iter()
                .zip(right_fields.iter())
                .map(|((name, l), (_, r))| map(l, r).map(|ty| (*name, ty)))
                .collect();
            Some(Type::NamedTuple(fields?))
        }
        _ => None,
    }
}

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
        Type::Infer
        | Type::Any
        | Type::Int
        | Type::Float
        | Type::Double
        | Type::Bool
        | Type::String
        | Type::Void
        | Type::Var(_)
        | Type::UnresolvedName(_)
        | Type::Extern { .. } => ty.clone(),
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

    use super::{fold_type, type_any, walk_type_structure};
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

    // ---- walk_type_structure tests ----

    #[test]
    fn walk_type_structure_pairs_nested_children() {
        let left = Type::Map {
            key: Box::new(Type::List {
                elem: Box::new(Type::Int),
            }),
            value: Box::new(Type::ArrayView {
                elem: Box::new(Type::Bool),
            }),
        };
        let right = Type::Map {
            key: Box::new(Type::List {
                elem: Box::new(Type::String),
            }),
            value: Box::new(Type::ArrayView {
                elem: Box::new(Type::Float),
            }),
        };
        let mut visited = vec![];

        let walked = walk_type_structure(&left, &right, &mut |left, right| {
            visited.push((left.clone(), right.clone()));
            true
        });

        assert!(walked);
        assert_eq!(
            visited,
            vec![
                (
                    Type::List {
                        elem: Box::new(Type::Int)
                    },
                    Type::List {
                        elem: Box::new(Type::String)
                    }
                ),
                (
                    Type::ArrayView {
                        elem: Box::new(Type::Bool)
                    },
                    Type::ArrayView {
                        elem: Box::new(Type::Float)
                    }
                )
            ]
        );
    }

    #[test]
    fn walk_type_structure_rejects_func_mutability_mismatch() {
        let left = Type::Func {
            params: vec![FuncParam::new(Type::Int, true)],
            ret: Box::new(Type::Void),
        };
        let right = Type::Func {
            params: vec![FuncParam::new(Type::Int, false)],
            ret: Box::new(Type::Void),
        };

        assert!(!walk_type_structure(&left, &right, &mut |_, _| true));
    }

    #[test]
    fn walk_type_structure_rejects_func_arity_mismatch() {
        let left = Type::Func {
            params: vec![FuncParam::new(Type::Int, false)],
            ret: Box::new(Type::Void),
        };
        let right = Type::Func {
            params: vec![
                FuncParam::new(Type::Int, false),
                FuncParam::new(Type::Bool, false),
            ],
            ret: Box::new(Type::Void),
        };

        assert!(!walk_type_structure(&left, &right, &mut |_, _| true));
    }

    #[test]
    fn walk_type_structure_rejects_named_aggregate_name_and_arity_mismatch() {
        let left_name = Type::Struct {
            name: ident("Foo"),
            type_args: vec![Type::Int],
        };
        let right_name = Type::Struct {
            name: ident("Bar"),
            type_args: vec![Type::Int],
        };
        let right_arity = Type::Struct {
            name: ident("Foo"),
            type_args: vec![Type::Int, Type::Bool],
        };

        assert!(!walk_type_structure(
            &left_name,
            &right_name,
            &mut |_, _| true
        ));
        assert!(!walk_type_structure(
            &left_name,
            &right_arity,
            &mut |_, _| true
        ));
    }

    #[test]
    fn walk_type_structure_checks_named_tuple_field_order() {
        let left = Type::NamedTuple(vec![(ident("x"), Type::Int), (ident("y"), Type::Bool)]);
        let right = Type::NamedTuple(vec![(ident("x"), Type::String), (ident("y"), Type::Float)]);
        let wrong_order =
            Type::NamedTuple(vec![(ident("y"), Type::String), (ident("x"), Type::Float)]);
        let mut visited = 0;

        let walked = walk_type_structure(&left, &right, &mut |_, _| {
            visited += 1;
            true
        });

        assert!(walked);
        assert_eq!(visited, 2);
        assert!(!walk_type_structure(&left, &wrong_order, &mut |_, _| true));
    }

    #[test]
    fn walk_type_structure_rejects_tuple_length_mismatch() {
        let left = Type::Tuple(vec![Type::Int]);
        let right = Type::Tuple(vec![Type::String, Type::Bool]);

        assert!(!walk_type_structure(&left, &right, &mut |_, _| true));
    }

    #[test]
    fn walk_type_structure_returns_false_for_leaf_types() {
        assert!(!walk_type_structure(
            &Type::Extern { name: ident("Foo") },
            &Type::Extern { name: ident("Foo") },
            &mut |_, _| true
        ));
        assert!(!walk_type_structure(&Type::Int, &Type::Int, &mut |_, _| {
            true
        }));
        assert!(!walk_type_structure(
            &Type::Bool,
            &Type::String,
            &mut |_, _| true
        ));
    }

    // ---- map_type_structure tests ----

    #[test]
    fn map_type_structure_maps_list_element() {
        use super::map_type_structure;

        let left = Type::List {
            elem: Box::new(Type::Int),
        };
        let right = Type::List {
            elem: Box::new(Type::Bool),
        };

        let result = map_type_structure(&left, &right, &mut |l, r| {
            assert_eq!(l, &Type::Int);
            assert_eq!(r, &Type::Bool);
            Some(Type::String)
        });

        assert_eq!(
            result,
            Some(Type::List {
                elem: Box::new(Type::String)
            })
        );
    }

    #[test]
    fn map_type_structure_maps_map_key_and_value() {
        use super::map_type_structure;

        let left = Type::Map {
            key: Box::new(Type::Int),
            value: Box::new(Type::Bool),
        };
        let right = Type::Map {
            key: Box::new(Type::String),
            value: Box::new(Type::Float),
        };
        let mut calls = 0;

        let result = map_type_structure(&left, &right, &mut |_, _| {
            calls += 1;
            Some(Type::Void)
        });

        assert!(result.is_some());
        assert_eq!(calls, 2);
    }

    #[test]
    fn map_type_structure_rejects_func_mutability_mismatch() {
        use super::map_type_structure;

        let left = Type::Func {
            params: vec![FuncParam::new(Type::Int, true)],
            ret: Box::new(Type::Void),
        };
        let right = Type::Func {
            params: vec![FuncParam::new(Type::Int, false)],
            ret: Box::new(Type::Void),
        };

        assert_eq!(
            map_type_structure(&left, &right, &mut |_, _| Some(Type::Void)),
            None
        );
    }

    #[test]
    fn map_type_structure_rejects_func_arity_mismatch() {
        use super::map_type_structure;

        let left = Type::Func {
            params: vec![FuncParam::new(Type::Int, false)],
            ret: Box::new(Type::Void),
        };
        let right = Type::Func {
            params: vec![
                FuncParam::new(Type::Int, false),
                FuncParam::new(Type::Bool, false),
            ],
            ret: Box::new(Type::Void),
        };

        assert_eq!(
            map_type_structure(&left, &right, &mut |_, _| Some(Type::Void)),
            None
        );
    }

    #[test]
    fn map_type_structure_rejects_named_aggregate_name_mismatch() {
        use super::map_type_structure;

        let left = Type::Struct {
            name: ident("Foo"),
            type_args: vec![Type::Int],
        };
        let right = Type::Struct {
            name: ident("Bar"),
            type_args: vec![Type::Int],
        };

        assert_eq!(
            map_type_structure(&left, &right, &mut |_, _| Some(Type::Void)),
            None
        );
    }

    #[test]
    fn map_type_structure_maps_struct_type_args() {
        use super::map_type_structure;

        let left = Type::Struct {
            name: ident("Box"),
            type_args: vec![Type::Int],
        };
        let right = Type::Struct {
            name: ident("Box"),
            type_args: vec![Type::Bool],
        };

        let result = map_type_structure(&left, &right, &mut |_, _| Some(Type::String));

        assert_eq!(
            result,
            Some(Type::Struct {
                name: ident("Box"),
                type_args: vec![Type::String]
            })
        );
    }

    #[test]
    fn map_type_structure_returns_none_for_array_pairs() {
        use super::map_type_structure;

        let left = Type::Array {
            elem: Box::new(Type::Int),
            len: ArrayLen::Fixed(3),
        };
        let right = Type::Array {
            elem: Box::new(Type::Int),
            len: ArrayLen::Fixed(3),
        };

        assert_eq!(
            map_type_structure(&left, &right, &mut |_, _| Some(Type::Void)),
            None
        );
    }

    #[test]
    fn map_type_structure_returns_none_for_leaf_types() {
        use super::map_type_structure;

        assert_eq!(
            map_type_structure(&Type::Int, &Type::Int, &mut |_, _| Some(Type::Void)),
            None
        );
        assert_eq!(
            map_type_structure(
                &Type::Extern { name: ident("Foo") },
                &Type::Extern { name: ident("Foo") },
                &mut |_, _| Some(Type::Void),
            ),
            None
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
