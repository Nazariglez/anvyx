use std::collections::HashMap;

use super::helpers::{dummy_span, opt_type, type_param, type_var};
use crate::{
    ast::{FuncParam, Type, TypeVarId},
    typecheck::{
        error::DiagnosticKind,
        infer::{instantiate_func_type, subst_type},
    },
};

// ---- subst_type tests ----

#[test]
fn test_subst_type_simple() {
    // substitute T -> int in T
    let t_var = type_var(0);
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);

    let result = subst_type(&t_var, &subst, &HashMap::new());
    assert_eq!(result, Type::Int);
}

#[test]
fn test_subst_type_optional() {
    // substitute T -> int in T?
    let t_var = type_var(0);
    let opt_t = opt_type(t_var);
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);

    let result = subst_type(&opt_t, &subst, &HashMap::new());
    assert_eq!(result, opt_type(Type::Int));
}

#[test]
fn test_subst_type_func() {
    // substitute T -> int, U -> bool in fn(T) -> U
    let t_var = type_var(0);
    let u_var = type_var(1);
    let func_ty = Type::Func {
        params: vec![FuncParam::immut(t_var)],
        ret: Box::new(u_var),
    };
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);
    subst.insert(TypeVarId(1), Type::Bool);

    let result = subst_type(&func_ty, &subst, &HashMap::new());
    assert_eq!(
        result,
        Type::Func {
            params: vec![FuncParam::immut(Type::Int)],
            ret: Box::new(Type::Bool),
        }
    );
}

#[test]
fn test_subst_type_repeated_var() {
    // substitute T -> int in fn(T, T) -> T
    let t_var = type_var(0);
    let func_ty = Type::Func {
        params: vec![
            FuncParam::immut(t_var.clone()),
            FuncParam::immut(t_var.clone()),
        ],
        ret: Box::new(t_var),
    };
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);

    let result = subst_type(&func_ty, &subst, &HashMap::new());
    assert_eq!(
        result,
        Type::Func {
            params: vec![FuncParam::immut(Type::Int), FuncParam::immut(Type::Int)],
            ret: Box::new(Type::Int),
        }
    );
}

#[test]
fn test_subst_type_no_change_for_concrete() {
    // substitute T -> int in int (no change)
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);

    let result = subst_type(&Type::Bool, &subst, &HashMap::new());
    assert_eq!(result, Type::Bool);
}

// ---- instantiate_func_type tests ----

#[test]
fn test_instantiate_identity() {
    let span = dummy_span();
    let mut errors = vec![];

    // fn identity<T>(x: T) -> T instantiated with <int> yields fn(int) -> int
    let type_params = vec![type_param("T", 0)];
    let t_var = type_var(0);
    let template = Type::Func {
        params: vec![FuncParam::immut(t_var.clone())],
        ret: Box::new(t_var),
    };
    let type_args = vec![Type::Int];

    let result = instantiate_func_type(&type_params, &template, &type_args, span, &mut errors);
    assert!(errors.is_empty());
    assert_eq!(
        result,
        Some(Type::Func {
            params: vec![FuncParam::immut(Type::Int)],
            ret: Box::new(Type::Int),
        })
    );
}

#[test]
fn test_instantiate_two_params() {
    let span = dummy_span();
    let mut errors = vec![];

    // fn pair<T, U>(a: T, b: U) -> T? instantiated with <int, bool> yields fn(int, bool) -> int?
    let type_params = vec![type_param("T", 0), type_param("U", 1)];
    let t_var = type_var(0);
    let u_var = type_var(1);
    let template = Type::Func {
        params: vec![FuncParam::immut(t_var.clone()), FuncParam::immut(u_var)],
        ret: Box::new(opt_type(t_var)),
    };
    let type_args = vec![Type::Int, Type::Bool];

    let result = instantiate_func_type(&type_params, &template, &type_args, span, &mut errors);
    assert!(errors.is_empty());
    assert_eq!(
        result,
        Some(Type::Func {
            params: vec![FuncParam::immut(Type::Int), FuncParam::immut(Type::Bool)],
            ret: Box::new(opt_type(Type::Int)),
        })
    );
}

#[test]
fn test_instantiate_arity_mismatch_too_few() {
    let span = dummy_span();
    let mut errors = vec![];

    // fn<T, U> instantiated with <int> (too few)
    let type_params = vec![type_param("T", 0), type_param("U", 1)];
    let template = Type::Func {
        params: vec![FuncParam::immut(type_var(0)), FuncParam::immut(type_var(1))],
        ret: Box::new(Type::Void),
    };
    let type_args = vec![Type::Int];

    let result = instantiate_func_type(&type_params, &template, &type_args, span, &mut errors);
    assert_eq!(result, None);
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        &errors[0].kind,
        DiagnosticKind::GenericArgNumMismatch {
            expected: 2,
            found: 1
        }
    ));
}

#[test]
fn test_instantiate_arity_mismatch_too_many() {
    let span = dummy_span();
    let mut errors = vec![];

    // fn<T> instantiated with <int, bool, string> (too many)
    let type_params = vec![type_param("T", 0)];
    let template = Type::Func {
        params: vec![FuncParam::immut(type_var(0))],
        ret: Box::new(Type::Void),
    };
    let type_args = vec![Type::Int, Type::Bool, Type::String];

    let result = instantiate_func_type(&type_params, &template, &type_args, span, &mut errors);
    assert_eq!(result, None);
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        &errors[0].kind,
        DiagnosticKind::GenericArgNumMismatch {
            expected: 1,
            found: 3
        }
    ));
}

// ---- subst_type: List / Map / ArrayView ----

#[test]
fn test_subst_type_list() {
    // substitute T -> int in [T], expect [int]
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);

    let list_t = Type::List {
        elem: Box::new(type_var(0)),
    };
    let result = subst_type(&list_t, &subst, &HashMap::new());
    assert_eq!(
        result,
        Type::List {
            elem: Box::new(Type::Int)
        }
    );
}

#[test]
fn test_subst_type_map() {
    // substitute T -> int, U -> string in [T: U], expect [int: string]
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);
    subst.insert(TypeVarId(1), Type::String);

    let map_tu = Type::Map {
        key: Box::new(type_var(0)),
        value: Box::new(type_var(1)),
    };
    let result = subst_type(&map_tu, &subst, &HashMap::new());
    assert_eq!(
        result,
        Type::Map {
            key: Box::new(Type::Int),
            value: Box::new(Type::String),
        }
    );
}

#[test]
fn test_subst_type_array_view() {
    // substitute T -> int in [T; ..], expect [int; ..]
    let mut subst = HashMap::new();
    subst.insert(TypeVarId(0), Type::Int);

    let view_t = Type::ArrayView {
        elem: Box::new(type_var(0)),
    };
    let result = subst_type(&view_t, &subst, &HashMap::new());
    assert_eq!(
        result,
        Type::ArrayView {
            elem: Box::new(Type::Int)
        }
    );
}
