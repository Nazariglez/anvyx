use super::helpers::{parse_param_type, parse_type};
use crate::ast;

#[test]
fn array_type_fixed_len_parses() {
    let ty = parse_type("[int; 3]");
    match ty {
        ast::Type::Array { elem, len } => {
            assert_eq!(*elem, ast::Type::Int);
            assert_eq!(len, ast::ArrayLen::Fixed(3));
        }
        other => panic!("expected array type, found {other:?}"),
    }
}

#[test]
fn list_type_parses() {
    let ty = parse_type("[string]");
    match ty {
        ast::Type::List { elem } => {
            assert_eq!(*elem, ast::Type::String);
        }
        other => panic!("expected list type, found {other:?}"),
    }
}

#[test]
fn array_type_infer_len_parses() {
    let ty = parse_type("[float; _]");
    match ty {
        ast::Type::Array { elem, len } => {
            assert_eq!(*elem, ast::Type::Float);
            assert_eq!(len, ast::ArrayLen::Infer);
        }
        other => panic!("expected infer-length array type, found {other:?}"),
    }
}

#[test]
fn array_type_can_be_optional() {
    let ty = parse_type("[int; 3]?");
    assert!(ty.is_option(), "expected optional array type, found {ty:?}");
    let inner = ty.option_inner().expect("is_option guarantees inner");
    match inner {
        ast::Type::Array { elem, len } => {
            assert_eq!(**elem, ast::Type::Int);
            assert_eq!(*len, ast::ArrayLen::Fixed(3));
        }
        other => panic!("expected inner array type, found {other:?}"),
    }
}

#[test]
fn array_type_with_struct_elem_parses() {
    let ty = parse_type("[MyStruct; 5]");
    match ty {
        ast::Type::Array { elem, len } => {
            assert_eq!(len, ast::ArrayLen::Fixed(5));
            match *elem {
                ast::Type::UnresolvedName(name) => {
                    assert_eq!(name.0.as_ref(), "MyStruct");
                }
                other => panic!("expected unresolved name, found {other:?}"),
            }
        }
        other => panic!("expected array type, found {other:?}"),
    }
}

#[test]
fn map_type_parses() {
    let ty = parse_type("[string: int]");
    match ty {
        ast::Type::Map { key, value } => {
            assert_eq!(*key, ast::Type::String);
            assert_eq!(*value, ast::Type::Int);
        }
        other => panic!("expected map type, found {other:?}"),
    }
}

#[test]
fn function_type_parses() {
    let ty = parse_type("fn(int, string) -> bool");
    match ty {
        ast::Type::Func { params, ret } => {
            assert_eq!(params, vec![ast::Type::Int, ast::Type::String]);
            assert_eq!(*ret, ast::Type::Bool);
        }
        other => panic!("expected function type, found {other:?}"),
    }
}

#[test]
fn optional_function_type_parses() {
    let ty = parse_type("(fn(float) -> int)?");
    assert!(
        ty.is_option(),
        "expected optional function type, found {ty:?}"
    );
    let inner = ty.option_inner().expect("is_option guarantees inner");
    match inner {
        ast::Type::Func { params, ret } => {
            assert_eq!(*params, vec![ast::Type::Float]);
            assert_eq!(**ret, ast::Type::Int);
        }
        other => panic!("expected function type inside optional, found {other:?}"),
    }
}

#[test]
fn nested_array_type_parses() {
    let ty = parse_type("[[int; 3]; 2]");
    match ty {
        ast::Type::Array { elem, len } => {
            assert_eq!(len, ast::ArrayLen::Fixed(2));
            match *elem {
                ast::Type::Array {
                    elem: inner_elem,
                    len: inner_len,
                } => {
                    assert_eq!(*inner_elem, ast::Type::Int);
                    assert_eq!(inner_len, ast::ArrayLen::Fixed(3));
                }
                other => panic!("expected inner array type, found {other:?}"),
            }
        }
        other => panic!("expected nested array type, found {other:?}"),
    }
}

#[test]
fn type_nested_optional_parses() {
    let ty = parse_type("(int?)?");
    assert!(ty.is_option(), "expected optional type, found {ty:?}");
    let inner = ty.option_inner().expect("is_option guarantees inner");
    assert!(
        inner.is_option(),
        "expected Optional(Optional(Int)), found {inner:?}"
    );
    let inner2 = inner.option_inner().expect("is_option guarantees inner");
    assert_eq!(*inner2, ast::Type::Int);
}

#[test]
fn type_optional_array_infer_parses() {
    let ty = parse_type("[int?; _]");
    match ty {
        ast::Type::Array { ref elem, len } => {
            assert_eq!(len, ast::ArrayLen::Infer);
            assert!(elem.is_option(), "expected Optional(Int), found {elem:?}");
            let inner = elem.option_inner().expect("is_option guarantees inner");
            assert_eq!(*inner, ast::Type::Int);
        }
        other => panic!("expected Array(Optional(Int), Infer), found {other:?}"),
    }
}

#[test]
fn type_optional_list_parses() {
    let ty = parse_type("[int?]");
    match ty {
        ast::Type::List { ref elem } => {
            assert!(elem.is_option(), "expected Optional(Int), found {elem:?}");
            let inner = elem.option_inner().expect("is_option guarantees inner");
            assert_eq!(*inner, ast::Type::Int);
        }
        other => panic!("expected List(Optional(Int)), found {other:?}"),
    }
}

#[test]
fn type_list_optional_parses() {
    let ty = parse_type("[int]?");
    assert!(ty.is_option(), "expected Optional(List(Int)), found {ty:?}");
    let inner = ty.option_inner().expect("is_option guarantees inner");
    match inner {
        ast::Type::List { elem } => {
            assert_eq!(**elem, ast::Type::Int);
        }
        other => panic!("expected List(Int), found {other:?}"),
    }
}

#[test]
fn type_optional_array_fixed_parses() {
    let ty = parse_type("[int?; 3]");
    match ty {
        ast::Type::Array { ref elem, len } => {
            assert_eq!(len, ast::ArrayLen::Fixed(3));
            assert!(elem.is_option(), "expected Optional(Int), found {elem:?}");
            let inner = elem.option_inner().expect("is_option guarantees inner");
            assert_eq!(*inner, ast::Type::Int);
        }
        other => panic!("expected Array(Optional(Int), Fixed(3)), found {other:?}"),
    }
}

#[test]
fn view_type_int_parses() {
    let ty = parse_param_type("[int; ..]");
    match ty {
        ast::Type::ArrayView { elem } => {
            assert_eq!(*elem, ast::Type::Int);
        }
        other => panic!("expected View(Int), found {other:?}"),
    }
}

#[test]
fn view_type_float_parses() {
    let ty = parse_param_type("[float; ..]");
    match ty {
        ast::Type::ArrayView { elem } => {
            assert_eq!(*elem, ast::Type::Float);
        }
        other => panic!("expected View(Float), found {other:?}"),
    }
}

#[test]
fn view_type_array_parses() {
    let ty = parse_param_type("[[int; 3]; ..]");
    match ty {
        ast::Type::ArrayView { elem } => match *elem {
            ast::Type::Array { elem: inner, len } => {
                assert_eq!(*inner, ast::Type::Int);
                assert_eq!(len, ast::ArrayLen::Fixed(3));
            }
            other => panic!("expected Array(Int, Fixed(3)), found {other:?}"),
        },
        other => panic!("expected View(Array(...)), found {other:?}"),
    }
}

#[test]
fn view_type_list_parses() {
    let ty = parse_param_type("[[string]; ..]");
    match ty {
        ast::Type::ArrayView { elem } => match *elem {
            ast::Type::List { elem: inner } => {
                assert_eq!(*inner, ast::Type::String);
            }
            other => panic!("expected List(String), found {other:?}"),
        },
        other => panic!("expected View(List(...)), found {other:?}"),
    }
}

#[test]
fn view_type_optional_parses() {
    let ty = parse_param_type("[int; ..]?");
    assert!(ty.is_option(), "expected Optional(View(Int)), found {ty:?}");
    let inner = ty.option_inner().expect("is_option guarantees inner");
    match inner {
        ast::Type::ArrayView { elem } => {
            assert_eq!(**elem, ast::Type::Int);
        }
        other => panic!("expected View(Int), found {other:?}"),
    }
}
