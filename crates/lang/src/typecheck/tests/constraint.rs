use super::helpers::{
    array_literal, assert_expr_type, call_expr, dummy_ident, expr_stmt, fn_decl, generic_method,
    get_expr_id, ident_expr, let_binding, lit_int, lit_string, map_literal_expr, method, program,
    reset_expr_ids, return_stmt, run_err, run_ok, struct_decl, struct_literal_expr,
};
use crate::ast::{MethodReceiver, Type, TypeParam, TypeVarId};
use crate::typecheck::error::TypeErrKind;

// ---- list constrain_assignable tests ----

#[test]
fn test_constrain_list_passthrough_ok() {
    reset_expr_ids();

    // fn get_ints(xs: [int]) -> [int] { return xs; }
    // let a: [int] = [1, 2, 3];
    // let b: [int] = get_ints(a);
    let list_int = Type::List {
        elem: Type::Int.boxed(),
    };
    let get_ints = fn_decl(
        "get_ints",
        vec![("xs", list_int.clone())],
        list_int.clone(),
        vec![return_stmt(Some(ident_expr("xs")))],
    );

    let arr = array_literal(vec![lit_int(1), lit_int(2), lit_int(3)]);
    let a_binding = let_binding("a", Some(list_int.clone()), arr);
    let call = call_expr(ident_expr("get_ints"), vec![ident_expr("a")]);
    let call_id = get_expr_id(&call);
    let b_binding = let_binding("b", Some(list_int.clone()), call);

    let prog = program(vec![get_ints, a_binding, b_binding]);
    let tcx = run_ok(prog);
    assert_expr_type(&tcx, call_id, list_int);
}

#[test]
fn test_constrain_list_annotation_then_assign_ok() {
    reset_expr_ids();

    // let a: [int] = [1, 2, 3];
    // let b: [int] = a;
    let list_int = Type::List {
        elem: Type::Int.boxed(),
    };

    let arr = array_literal(vec![lit_int(1), lit_int(2), lit_int(3)]);
    let a_binding = let_binding("a", Some(list_int.clone()), arr);

    let b_val = ident_expr("a");
    let b_id = get_expr_id(&b_val);
    let b_binding = let_binding("b", Some(list_int.clone()), b_val);

    let prog = program(vec![a_binding, b_binding]);
    let tcx = run_ok(prog);
    assert_expr_type(&tcx, b_id, list_int);
}

#[test]
fn test_constrain_list_type_mismatch_errors() {
    reset_expr_ids();

    // fn make_strings() -> [string] { ... }
    // let x: [int] = make_strings(); // ERROR
    let make_strings = fn_decl(
        "make_strings",
        vec![],
        Type::List {
            elem: Type::String.boxed(),
        },
        vec![return_stmt(Some(ident_expr("make_strings")))],
    );

    let call = call_expr(ident_expr("make_strings"), vec![]);
    let binding = let_binding(
        "x",
        Some(Type::List {
            elem: Type::Int.boxed(),
        }),
        call,
    );

    let prog = program(vec![make_strings, binding]);
    let errors = run_err(prog);
    assert!(!errors.is_empty());
    assert!(
        errors
            .iter()
            .any(|e| matches!(&e.kind, TypeErrKind::MismatchedTypes { .. })),
        "Expected MismatchedTypes, got: {:?}",
        errors
    );
}

// ---- map constrain_assignable tests ----

#[test]
fn test_constrain_map_infer_flows_to_annotated() {
    reset_expr_ids();

    // fn make_map() -> [int: string] { ... }
    // let x: [int: string] = make_map();
    let map_ty = Type::Map {
        key: Type::Int.boxed(),
        value: Type::String.boxed(),
    };
    let make_map = fn_decl(
        "make_map",
        vec![],
        map_ty.clone(),
        vec![return_stmt(Some(map_literal_expr(vec![(
            lit_int(1),
            lit_string("a"),
        )])))],
    );

    let call = call_expr(ident_expr("make_map"), vec![]);
    let call_id = get_expr_id(&call);
    let binding = let_binding("x", Some(map_ty.clone()), call);

    let prog = program(vec![make_map, binding]);
    let tcx = run_ok(prog);
    assert_expr_type(&tcx, call_id, map_ty);
}

#[test]
fn test_constrain_map_type_mismatch_errors() {
    reset_expr_ids();

    // fn make_map() -> [int: string] { ... }
    // let x: [int: int] = make_map(); // ERROR
    let make_map = fn_decl(
        "make_map",
        vec![],
        Type::Map {
            key: Type::Int.boxed(),
            value: Type::String.boxed(),
        },
        vec![return_stmt(Some(map_literal_expr(vec![])))],
    );

    let call = call_expr(ident_expr("make_map"), vec![]);
    let binding = let_binding(
        "x",
        Some(Type::Map {
            key: Type::Int.boxed(),
            value: Type::Int.boxed(),
        }),
        call,
    );

    let prog = program(vec![make_map, binding]);
    let errors = run_err(prog);
    assert!(!errors.is_empty());
    assert!(
        errors
            .iter()
            .any(|e| matches!(&e.kind, TypeErrKind::MismatchedTypes { .. })),
        "Expected MismatchedTypes, got: {:?}",
        errors
    );
}

// ---- struct constrain_assignable tests ----

#[test]
fn test_constrain_struct_passthrough_ok() {
    reset_expr_ids();

    // struct Point { x: int, y: int }
    // fn make_point(x: int, y: int) -> Point { Point { x: x, y: y } }
    // let p: Point = make_point(1, 2);
    let point_ty = Type::Struct {
        name: dummy_ident("Point"),
        type_args: vec![],
    };
    let point_decl = struct_decl("Point", vec![("x", Type::Int), ("y", Type::Int)], vec![]);
    let make_point = fn_decl(
        "make_point",
        vec![("px", Type::Int), ("py", Type::Int)],
        point_ty.clone(),
        vec![return_stmt(Some(struct_literal_expr(
            "Point",
            vec![("x", ident_expr("px")), ("y", ident_expr("py"))],
        )))],
    );

    let call = call_expr(ident_expr("make_point"), vec![lit_int(1), lit_int(2)]);
    let call_id = get_expr_id(&call);
    let binding = let_binding("p", Some(point_ty.clone()), call);

    let prog = program(vec![point_decl, make_point, binding]);
    let tcx = run_ok(prog);
    assert_expr_type(&tcx, call_id, point_ty);
}

#[test]
fn test_constrain_generic_struct_name_mismatch_errors() {
    reset_expr_ids();

    // struct Foo { x: int }
    // struct Bar { x: int }
    // fn make_foo() -> Foo { Foo { x: 1 } }
    // let b: Bar = make_foo(); // ERROR
    let foo_decl = struct_decl("Foo", vec![("x", Type::Int)], vec![]);
    let bar_decl = struct_decl("Bar", vec![("x", Type::Int)], vec![]);

    let make_foo = fn_decl(
        "make_foo",
        vec![],
        Type::Struct {
            name: dummy_ident("Foo"),
            type_args: vec![],
        },
        vec![return_stmt(Some(struct_literal_expr(
            "Foo",
            vec![("x", lit_int(1))],
        )))],
    );

    let call = call_expr(ident_expr("make_foo"), vec![]);
    let binding = let_binding(
        "b",
        Some(Type::Struct {
            name: dummy_ident("Bar"),
            type_args: vec![],
        }),
        call,
    );

    let prog = program(vec![foo_decl, bar_decl, make_foo, binding]);
    let errors = run_err(prog);
    assert!(!errors.is_empty());
    assert!(
        errors
            .iter()
            .any(|e| matches!(&e.kind, TypeErrKind::MismatchedTypes { .. })),
        "Expected MismatchedTypes, got: {:?}",
        errors
    );
}

// ---- list literal annotation flowing through binding ----

#[test]
fn test_constrain_list_literal_annotated_binding() {
    reset_expr_ids();

    // let x: [string] = ["hello", "world"];
    let arr = array_literal(vec![lit_string("hello"), lit_string("world")]);
    let arr_id = get_expr_id(&arr);
    let binding = let_binding(
        "x",
        Some(Type::List {
            elem: Type::String.boxed(),
        }),
        arr,
    );

    let prog = program(vec![binding]);
    let tcx = run_ok(prog);
    assert_expr_type(
        &tcx,
        arr_id,
        Type::List {
            elem: Type::String.boxed(),
        },
    );
}

#[test]
fn test_constrain_map_literal_annotated_binding() {
    reset_expr_ids();

    // let m: [int: string] = [1: "a"];
    let map = map_literal_expr(vec![(lit_int(1), lit_string("a"))]);
    let map_id = get_expr_id(&map);
    let binding = let_binding(
        "m",
        Some(Type::Map {
            key: Type::Int.boxed(),
            value: Type::String.boxed(),
        }),
        map,
    );

    let prog = program(vec![binding]);
    let tcx = run_ok(prog);
    assert_expr_type(
        &tcx,
        map_id,
        Type::Map {
            key: Type::Int.boxed(),
            value: Type::String.boxed(),
        },
    );
}

// ---- generic method rejection ----

#[test]
fn test_generic_method_on_struct_errors() {
    reset_expr_ids();

    // struct Foo {}
    // with fn bar<T>(self, x: T) -> T { ... }
    // Should produce GenericMethodNotSupported error.
    let t_id = TypeVarId(0);
    let t_type = Type::Var(t_id);
    let gm = generic_method(
        "bar",
        vec![TypeParam {
            name: dummy_ident("T"),
            id: t_id,
        }],
        Some(MethodReceiver::Value),
        vec![("x", t_type.clone())],
        t_type,
        vec![return_stmt(Some(ident_expr("x")))],
    );
    let foo_decl = struct_decl("Foo", vec![], vec![gm]);

    let prog = program(vec![foo_decl]);
    let errors = run_err(prog);
    assert!(!errors.is_empty());
    assert!(
        errors
            .iter()
            .any(|e| matches!(&e.kind, TypeErrKind::GenericMethodNotSupported { .. })),
        "Expected GenericMethodNotSupported, got: {:?}",
        errors
    );
}

#[test]
fn test_non_generic_method_on_struct_ok() {
    reset_expr_ids();

    // struct Foo {}
    // with fn get_zero(self) -> int { 0 }
    // Should succeed — non-generic methods are fine.
    let m = method(
        "get_zero",
        Some(MethodReceiver::Value),
        vec![],
        Type::Int,
        vec![expr_stmt(lit_int(0))],
    );
    let foo_decl = struct_decl("Foo", vec![], vec![m]);

    let prog = program(vec![foo_decl]);
    let _ = run_ok(prog);
}
