use super::helpers::{parse_program, parse_program_err};
use crate::ast::{self, ImportKind};

fn ident_str(ident: &ast::Ident) -> &str {
    ident.0.as_ref()
}

fn first_import(src: &str) -> ast::Import {
    let prog = parse_program(src);
    let ast::Stmt::Import(node) = &prog.stmts[0].node else {
        panic!("expected Import statement, found {:?}", prog.stmts[0].node);
    };
    node.node.clone()
}

// --- qualified module imports ---

#[test]
fn import_single_segment_parses() {
    let imp = first_import("import foo;");
    assert_eq!(imp.path.len(), 1);
    assert_eq!(ident_str(&imp.path[0]), "foo");
    assert_eq!(imp.kind, ImportKind::Module);
}

#[test]
fn import_two_segment_path_parses() {
    let imp = first_import("import foo.bar;");
    assert_eq!(imp.path.len(), 2);
    assert_eq!(ident_str(&imp.path[0]), "foo");
    assert_eq!(ident_str(&imp.path[1]), "bar");
    assert_eq!(imp.kind, ImportKind::Module);
}

#[test]
fn import_three_segment_path_parses() {
    let imp = first_import("import foo.bar.baz;");
    assert_eq!(imp.path.len(), 3);
    assert_eq!(ident_str(&imp.path[0]), "foo");
    assert_eq!(ident_str(&imp.path[1]), "bar");
    assert_eq!(ident_str(&imp.path[2]), "baz");
    assert_eq!(imp.kind, ImportKind::Module);
}

// --- module aliases ---

#[test]
fn import_single_segment_with_alias_parses() {
    let imp = first_import("import foo as f;");
    assert_eq!(imp.path.len(), 1);
    assert_eq!(ident_str(&imp.path[0]), "foo");
    let ImportKind::ModuleAs(alias) = &imp.kind else {
        panic!("expected ModuleAs, found {:?}", imp.kind);
    };
    assert_eq!(ident_str(alias), "f");
}

#[test]
fn import_multi_segment_with_alias_parses() {
    let imp = first_import("import foo.bar as b;");
    assert_eq!(imp.path.len(), 2);
    assert_eq!(ident_str(&imp.path[0]), "foo");
    assert_eq!(ident_str(&imp.path[1]), "bar");
    let ImportKind::ModuleAs(alias) = &imp.kind else {
        panic!("expected ModuleAs, found {:?}", imp.kind);
    };
    assert_eq!(ident_str(alias), "b");
}

// --- selective imports ---

#[test]
fn import_selective_single_item_parses() {
    let imp = first_import("import foo { bar };");
    let ImportKind::Selective(items) = &imp.kind else {
        panic!("expected Selective, found {:?}", imp.kind);
    };
    assert_eq!(items.len(), 1);
    assert_eq!(ident_str(&items[0].name), "bar");
    assert!(items[0].alias.is_none());
}

#[test]
fn import_selective_multiple_items_parses() {
    let imp = first_import("import foo { bar, baz };");
    let ImportKind::Selective(items) = &imp.kind else {
        panic!("expected Selective, found {:?}", imp.kind);
    };
    assert_eq!(items.len(), 2);
    assert_eq!(ident_str(&items[0].name), "bar");
    assert_eq!(ident_str(&items[1].name), "baz");
}

#[test]
fn import_selective_trailing_comma_parses() {
    let imp = first_import("import foo { bar, baz, };");
    let ImportKind::Selective(items) = &imp.kind else {
        panic!("expected Selective, found {:?}", imp.kind);
    };
    assert_eq!(items.len(), 2);
}

#[test]
fn import_selective_item_with_alias_parses() {
    let imp = first_import("import foo { bar as b };");
    let ImportKind::Selective(items) = &imp.kind else {
        panic!("expected Selective, found {:?}", imp.kind);
    };
    assert_eq!(items.len(), 1);
    assert_eq!(ident_str(&items[0].name), "bar");
    let alias = items[0].alias.as_ref().expect("expected alias");
    assert_eq!(ident_str(alias), "b");
}

#[test]
fn import_selective_mixed_alias_and_plain_parses() {
    let imp = first_import("import foo { bar, baz as z };");
    let ImportKind::Selective(items) = &imp.kind else {
        panic!("expected Selective, found {:?}", imp.kind);
    };
    assert_eq!(items.len(), 2);
    assert_eq!(ident_str(&items[0].name), "bar");
    assert!(items[0].alias.is_none());
    assert_eq!(ident_str(&items[1].name), "baz");
    assert_eq!(ident_str(items[1].alias.as_ref().unwrap()), "z");
}

// --- wildcard imports ---

#[test]
fn import_wildcard_parses() {
    let imp = first_import("import foo { * };");
    assert_eq!(imp.path.len(), 1);
    assert_eq!(ident_str(&imp.path[0]), "foo");
    assert_eq!(imp.kind, ImportKind::Wildcard);
}

// --- multiple imports in one file ---

#[test]
fn multiple_imports_parse() {
    let prog = parse_program(
        "import foo;\nimport bar.baz;\nimport qux { * };\nfn main() {}",
    );
    assert_eq!(prog.stmts.len(), 4);
    assert!(matches!(prog.stmts[0].node, ast::Stmt::Import(_)));
    assert!(matches!(prog.stmts[1].node, ast::Stmt::Import(_)));
    assert!(matches!(prog.stmts[2].node, ast::Stmt::Import(_)));
    assert!(matches!(prog.stmts[3].node, ast::Stmt::Func(_)));
}

// --- parse errors ---

#[test]
fn import_missing_path_is_error() {
    parse_program_err("import;");
}

#[test]
fn import_trailing_dot_is_error() {
    parse_program_err("import foo.;");
}

#[test]
fn import_empty_braces_is_error() {
    parse_program_err("import foo { };");
}
