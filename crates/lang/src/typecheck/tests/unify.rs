use super::helpers::{dummy_span, opt_type, type_var};
use crate::ast::Type;
use crate::typecheck::error::TypeErrKind;
use crate::typecheck::unify::{contains_infer, is_assignable, unify_types};

#[test]
fn test_unify_primitives() {
    let span = dummy_span();
    let mut errors = vec![];

    // int unifies with int
    let result = unify_types(&Type::Int, &Type::Int, span, &mut errors);
    assert_eq!(result, Some(Type::Int));
    assert_eq!(errors.len(), 0);

    // float unifies with float
    let result = unify_types(&Type::Float, &Type::Float, span, &mut errors);
    assert_eq!(result, Some(Type::Float));
    assert_eq!(errors.len(), 0);

    // bool unifies with bool
    let result = unify_types(&Type::Bool, &Type::Bool, span, &mut errors);
    assert_eq!(result, Some(Type::Bool));
    assert_eq!(errors.len(), 0);

    // string unifies with string
    let result = unify_types(&Type::String, &Type::String, span, &mut errors);
    assert_eq!(result, Some(Type::String));
    assert_eq!(errors.len(), 0);

    // void unifies with void
    let result = unify_types(&Type::Void, &Type::Void, span, &mut errors);
    assert_eq!(result, Some(Type::Void));
    assert_eq!(errors.len(), 0);
}

#[test]
fn test_unify_infer_with_concrete() {
    let span = dummy_span();
    let mut errors = vec![];

    // infer unifies with int (both directions)
    let result = unify_types(&Type::Infer, &Type::Int, span, &mut errors);
    assert_eq!(result, Some(Type::Int));
    assert_eq!(errors.len(), 0);

    let result = unify_types(&Type::Int, &Type::Infer, span, &mut errors);
    assert_eq!(result, Some(Type::Int));
    assert_eq!(errors.len(), 0);

    // infer unifies with optional(int)
    let result = unify_types(&Type::Infer, &opt_type(Type::Int), span, &mut errors);
    assert_eq!(result, Some(opt_type(Type::Int)));
    assert_eq!(errors.len(), 0);
}

#[test]
fn test_unify_optional() {
    let span = dummy_span();
    let mut errors = vec![];

    // int? unifies with int?
    let result = unify_types(
        &opt_type(Type::Int),
        &opt_type(Type::Int),
        span,
        &mut errors,
    );
    assert_eq!(result, Some(opt_type(Type::Int)));
    assert_eq!(errors.len(), 0);

    // infer? unifies with string?
    let result = unify_types(
        &opt_type(Type::Infer),
        &opt_type(Type::String),
        span,
        &mut errors,
    );
    assert_eq!(result, Some(opt_type(Type::String)));
    assert_eq!(errors.len(), 0);
}

#[test]
fn test_unify_non_optional_with_optional() {
    let span = dummy_span();
    let mut errors = vec![];

    let result = unify_types(&Type::Int, &opt_type(Type::Infer), span, &mut errors);
    assert_eq!(result, Some(opt_type(Type::Int)));
    assert!(errors.is_empty());

    let result = unify_types(&opt_type(Type::String), &Type::String, span, &mut errors);
    assert_eq!(result, Some(opt_type(Type::String)));
    assert!(errors.is_empty());
}

#[test]
fn test_unify_non_optional_with_optional_mismatch() {
    let span = dummy_span();
    let mut errors = vec![];

    let result = unify_types(&Type::Int, &opt_type(Type::String), span, &mut errors);
    assert_eq!(result, None);
    assert!(!errors.is_empty());
}

#[test]
fn test_unify_function_types() {
    let span = dummy_span();
    let mut errors = vec![];

    // fn(int, bool) -> float unifies with identical signature
    let func_type = Type::Func {
        params: vec![Type::Int, Type::Bool],
        ret: Box::new(Type::Float),
    };
    let result = unify_types(&func_type, &func_type, span, &mut errors);
    assert_eq!(result, Some(func_type.clone()));
    assert_eq!(errors.len(), 0);

    // parameter length mismatch produces error
    let func1 = Type::Func {
        params: vec![Type::Int],
        ret: Box::new(Type::Void),
    };
    let func2 = Type::Func {
        params: vec![Type::Int, Type::Bool],
        ret: Box::new(Type::Void),
    };
    let result = unify_types(&func1, &func2, span, &mut errors);
    assert_eq!(result, None);
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        &errors[0].kind,
        TypeErrKind::MismatchedTypes { .. }
    ));
}

#[test]
fn test_unify_mismatched_types() {
    let span = dummy_span();
    let mut errors = vec![];

    // int vs bool produces error
    let result = unify_types(&Type::Int, &Type::Bool, span, &mut errors);
    assert_eq!(result, None);
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        &errors[0].kind,
        TypeErrKind::MismatchedTypes { expected, found }
        if *expected == Type::Int && *found == Type::Bool
    ));

    // optional vs non-optional now unifies to optional
    errors.clear();
    let result = unify_types(&opt_type(Type::Int), &Type::Int, span, &mut errors);
    assert_eq!(result, Some(opt_type(Type::Int)));
    assert!(errors.is_empty());
}

// ---- unification tests for type variables ----

#[test]
fn test_unify_same_type_var() {
    let span = dummy_span();
    let mut errors = vec![];

    // T unifies with T (same variable)
    let t = type_var(0);
    let result = unify_types(&t, &t, span, &mut errors);
    assert_eq!(result, Some(t.clone()));
    assert!(errors.is_empty());
}

#[test]
fn test_unify_different_type_vars_error() {
    let span = dummy_span();
    let mut errors = vec![];

    // T and U are different type variables
    let t = type_var(0);
    let u = type_var(1);
    let result = unify_types(&t, &u, span, &mut errors);
    assert_eq!(result, None);
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        &errors[0].kind,
        TypeErrKind::MismatchedTypes { expected, found }
        if *expected == t && *found == u
    ));
}

#[test]
fn test_unify_type_var_with_concrete_error() {
    let span = dummy_span();
    let mut errors = vec![];

    // T and int are different types
    let t = type_var(0);
    let result = unify_types(&t, &Type::Int, span, &mut errors);
    assert_eq!(result, None);
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        &errors[0].kind,
        TypeErrKind::MismatchedTypes { expected, found }
        if *expected == t && *found == Type::Int
    ));

    // int and T are different types
    errors.clear();
    let result = unify_types(&Type::Int, &t, span, &mut errors);
    assert_eq!(result, None);
    assert_eq!(errors.len(), 1);
    assert!(matches!(
        &errors[0].kind,
        TypeErrKind::MismatchedTypes { expected, found }
        if *expected == Type::Int && *found == t
    ));
}

#[test]
fn test_unify_func_with_same_type_vars() {
    let span = dummy_span();
    let mut errors = vec![];

    // fn(T) -> T unifies with itself
    let t = type_var(0);
    let func_type = Type::Func {
        params: vec![t.clone()],
        ret: Box::new(t.clone()),
    };
    let result = unify_types(&func_type, &func_type, span, &mut errors);
    assert_eq!(result, Some(func_type.clone()));
    assert!(errors.is_empty());
}

#[test]
fn test_unify_func_with_different_type_vars_error() {
    let span = dummy_span();
    let mut errors = vec![];

    // fn(T) -> T and fn(U) -> U are different functions
    let t = type_var(0);
    let u = type_var(1);
    let func_t = Type::Func {
        params: vec![t.clone()],
        ret: Box::new(t.clone()),
    };
    let func_u = Type::Func {
        params: vec![u.clone()],
        ret: Box::new(u.clone()),
    };
    let result = unify_types(&func_t, &func_u, span, &mut errors);
    assert_eq!(result, None);
    assert!(!errors.is_empty());
}

#[test]
fn test_unify_optional_with_same_type_var() {
    let span = dummy_span();
    let mut errors = vec![];

    // T? unifies with T?
    let t = type_var(0);
    let opt_t = opt_type(t.clone());
    let result = unify_types(&opt_t, &opt_t, span, &mut errors);
    assert_eq!(result, Some(opt_t.clone()));
    assert!(errors.is_empty());
}

#[test]
fn test_unify_optional_with_different_type_vars_error() {
    let span = dummy_span();
    let mut errors = vec![];

    // T? and U? are different optional types
    let t = type_var(0);
    let u = type_var(1);
    let opt_t = opt_type(t.clone());
    let opt_u = opt_type(u.clone());
    let result = unify_types(&opt_t, &opt_u, span, &mut errors);
    assert_eq!(result, None);
    assert!(!errors.is_empty());
}

// ---- assignability tests for type variables ----

#[test]
fn test_assignable_same_type_var() {
    // T is assignable to T
    let t = type_var(0);
    assert!(is_assignable(&t, &t));
}

#[test]
fn test_assignable_different_type_vars() {
    // T is NOT assignable to U
    let t = type_var(0);
    let u = type_var(1);
    assert!(!is_assignable(&t, &u));
}

#[test]
fn test_assignable_type_var_to_concrete() {
    // T is NOT assignable to int
    let t = type_var(0);
    assert!(!is_assignable(&t, &Type::Int));
}

#[test]
fn test_assignable_concrete_to_type_var() {
    // int is NOT assignable to T
    let t = type_var(0);
    assert!(!is_assignable(&Type::Int, &t));
}

#[test]
fn test_assignable_func_with_same_type_vars() {
    // fn(T) -> U is assignable to fn(T) -> U
    let t = type_var(0);
    let u = type_var(1);
    let func_type = Type::Func {
        params: vec![t.clone()],
        ret: Box::new(u.clone()),
    };
    assert!(is_assignable(&func_type, &func_type));
}

#[test]
fn test_assignable_func_with_different_type_vars() {
    // fn(T) -> T is NOT assignable to fn(U) -> U
    let t = type_var(0);
    let u = type_var(1);
    let func_t = Type::Func {
        params: vec![t.clone()],
        ret: Box::new(t.clone()),
    };
    let func_u = Type::Func {
        params: vec![u.clone()],
        ret: Box::new(u.clone()),
    };
    assert!(!is_assignable(&func_t, &func_u));
}

#[test]
fn test_assignable_optional_same_type_var() {
    // T? is assignable to T?
    let t = type_var(0);
    let opt_t = opt_type(t.clone());
    assert!(is_assignable(&opt_t, &opt_t));
}

#[test]
fn test_assignable_optional_different_type_vars() {
    // T? is NOT assignable to U?
    let t = type_var(0);
    let u = type_var(1);
    let opt_t = opt_type(t.clone());
    let opt_u = opt_type(u.clone());
    assert!(!is_assignable(&opt_t, &opt_u));
}

// ---- contains_infer tests for type variables ----

#[test]
fn test_contains_infer_type_var_is_false() {
    // type variables are not considered as containing inference
    let t = type_var(0);
    assert!(!contains_infer(&t));
}

#[test]
fn test_contains_infer_optional_type_var_is_false() {
    // T? does not contain infer
    let t = type_var(0);
    let opt_t = opt_type(t);
    assert!(!contains_infer(&opt_t));
}

#[test]
fn test_contains_infer_func_with_type_var_is_false() {
    // fn(T) -> U does not contain infer
    let t = type_var(0);
    let u = type_var(1);
    let func_type = Type::Func {
        params: vec![t],
        ret: Box::new(u),
    };
    assert!(!contains_infer(&func_type));
}

#[test]
fn test_contains_infer_optional_infer() {
    // infer? returns true
    let opt_infer = opt_type(Type::Infer);
    assert!(contains_infer(&opt_infer));
}

#[test]
fn test_contains_infer_func_with_infer() {
    // fn(Infer) -> int returns true
    let func_type = Type::Func {
        params: vec![Type::Infer],
        ret: Box::new(Type::Int),
    };
    assert!(contains_infer(&func_type));
}
