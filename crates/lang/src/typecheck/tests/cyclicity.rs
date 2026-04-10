use super::helpers::*;
use crate::{
    ast::{Type, VariantKind},
    typecheck::error::DiagnosticKind,
};

fn struct_type(name: &str) -> Type {
    Type::Struct {
        name: dummy_ident(name),
        type_args: vec![],
        origin: None,
    }
}

fn enum_type(name: &str) -> Type {
    Type::Enum {
        name: dummy_ident(name),
        type_args: vec![],
        origin: None,
    }
}

#[test]
fn self_referencing_struct_detected() {
    reset_expr_ids();
    let prog = program(vec![
        struct_decl("Node", vec![("child", struct_type("Node"))], vec![]),
        fn_decl("main", vec![], Type::Void, vec![]),
    ]);
    let errors = run_err(prog);
    assert!(errors.iter().any(|e| matches!(
        &e.kind,
        DiagnosticKind::InfiniteSizeType { type_name, .. } if type_name.to_string() == "Node"
    )));
}

#[test]
fn mutual_cycle_detected() {
    reset_expr_ids();
    let prog = program(vec![
        struct_decl("A", vec![("b", struct_type("B"))], vec![]),
        struct_decl("B", vec![("a", struct_type("A"))], vec![]),
        fn_decl("main", vec![], Type::Void, vec![]),
    ]);
    let errors = run_err(prog);
    let cycle_errors: Vec<_> = errors
        .iter()
        .filter(|e| matches!(&e.kind, DiagnosticKind::InfiniteSizeType { .. }))
        .collect();
    assert!(
        cycle_errors.len() >= 2,
        "Expected errors for both A and B, got {cycle_errors:?}"
    );
}

#[test]
fn no_cycle_linear_structs() {
    reset_expr_ids();
    let prog = program(vec![
        struct_decl("A", vec![("x", Type::Int)], vec![]),
        struct_decl("B", vec![("a", struct_type("A"))], vec![]),
        fn_decl("main", vec![], Type::Void, vec![]),
    ]);
    run_ok(prog);
}

#[test]
fn list_breaks_cycle() {
    reset_expr_ids();
    let prog = program(vec![
        struct_decl(
            "Tree",
            vec![(
                "children",
                Type::List {
                    elem: Box::new(struct_type("Tree")),
                },
            )],
            vec![],
        ),
        fn_decl("main", vec![], Type::Void, vec![]),
    ]);
    run_ok(prog);
}

#[test]
fn map_breaks_cycle() {
    reset_expr_ids();
    let prog = program(vec![
        struct_decl(
            "Registry",
            vec![(
                "entries",
                Type::Map {
                    key: Box::new(Type::String),
                    value: Box::new(struct_type("Registry")),
                },
            )],
            vec![],
        ),
        fn_decl("main", vec![], Type::Void, vec![]),
    ]);
    run_ok(prog);
}

#[test]
fn enum_variant_cycle_detected() {
    reset_expr_ids();
    let prog = program(vec![
        struct_decl("Wrapper", vec![("inner", enum_type("Payload"))], vec![]),
        enum_decl(
            "Payload",
            vec![("Data", VariantKind::Tuple(vec![struct_type("Wrapper")]))],
        ),
        fn_decl("main", vec![], Type::Void, vec![]),
    ]);
    let errors = run_err(prog);
    assert!(
        errors
            .iter()
            .any(|e| matches!(&e.kind, DiagnosticKind::InfiniteSizeType { .. }))
    );
}
