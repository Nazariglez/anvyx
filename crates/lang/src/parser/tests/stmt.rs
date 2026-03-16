use super::helpers::parse_program;
use crate::ast::{self, MethodReceiver, Mutability};

#[test]
fn while_with_binary_cond_parses() {
    let prog = parse_program("fn main() { while x < 3 {} }");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let body_stmts = &func_node.node.body.node.stmts;
    assert_eq!(body_stmts.len(), 1);
    let ast::Stmt::While(while_node) = &body_stmts[0].node else {
        panic!("expected While");
    };
    let cond = &while_node.node.cond;
    match &cond.node.kind {
        ast::ExprKind::Binary(bin) => {
            assert_eq!(bin.node.op, ast::BinaryOp::LessThan);
        }
        other => panic!("expected Binary cond, found {other:?}"),
    }
}

#[test]
fn while_with_ident_cond_parses() {
    let prog = parse_program("fn main() { while x {} }");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let body_stmts = &func_node.node.body.node.stmts;
    assert_eq!(body_stmts.len(), 1);
    let ast::Stmt::While(while_node) = &body_stmts[0].node else {
        panic!("expected While");
    };
    let cond = &while_node.node.cond;
    match &cond.node.kind {
        ast::ExprKind::Ident(ident) => {
            assert_eq!(ident.0.as_ref(), "x");
        }
        other => panic!("expected Ident cond, found {other:?}"),
    }
}

#[test]
fn if_with_ident_cond_parses() {
    let prog = parse_program("fn main() { if x {} }");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let body_stmts = &func_node.node.body.node.stmts;
    assert_eq!(body_stmts.len(), 1);
    let ast::Stmt::Expr(expr_node) = &body_stmts[0].node else {
        panic!("expected Expr stmt");
    };
    let ast::ExprKind::If(if_node) = &expr_node.node.kind else {
        panic!("expected If expr");
    };
    let cond = &if_node.node.cond;
    match &cond.node.kind {
        ast::ExprKind::Ident(ident) => {
            assert_eq!(ident.0.as_ref(), "x");
        }
        other => panic!("expected Ident cond, found {other:?}"),
    }
}

#[test]
fn while_with_inner_break_and_assign_parses() {
    let src = r#"
        fn main() {
            var i: int = 0;
            while true {
                if i == 3 {
                    break;
                }
                i = i + 1;
            }
        }
    "#;
    let prog = parse_program(src);
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let body_stmts = &func_node.node.body.node.stmts;
    assert_eq!(body_stmts.len(), 2);

    let ast::Stmt::Binding(_) = &body_stmts[0].node else {
        panic!("expected Binding stmt");
    };

    let ast::Stmt::While(while_node) = &body_stmts[1].node else {
        panic!("expected While stmt");
    };
    let while_body = &while_node.node.body.node.stmts;
    assert_eq!(while_body.len(), 2);

    let ast::Stmt::Expr(if_expr_node) = &while_body[0].node else {
        panic!("expected Expr stmt for if");
    };
    assert!(matches!(&if_expr_node.node.kind, ast::ExprKind::If(_)));

    let ast::Stmt::Expr(assign_expr_node) = &while_body[1].node else {
        panic!("expected Expr stmt for assignment");
    };
    assert!(matches!(
        &assign_expr_node.node.kind,
        ast::ExprKind::Assign(_)
    ));
}

#[test]
fn for_parses_basic_range() {
    let prog = parse_program("fn main() { for n in 0..10 {} }");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let body_stmts = &func_node.node.body.node.stmts;
    assert_eq!(body_stmts.len(), 1);

    let ast::Stmt::For(for_node) = &body_stmts[0].node else {
        panic!("expected For stmt");
    };
    let for_inner = &for_node.node;

    let ast::Pattern::Ident(ident) = &for_inner.pattern.node else {
        panic!("expected Ident pattern");
    };
    assert_eq!(ident.0.as_ref(), "n");

    assert!(!for_inner.reversed);
    assert!(for_inner.step.is_none());

    let ast::ExprKind::Range(range_node) = &for_inner.iterable.node.kind else {
        panic!("expected Range iterable");
    };
    assert!(!range_node.node.inclusive);
}

#[test]
fn for_parses_rev_and_step() {
    let prog = parse_program("fn main() { for n in rev 0..10 step 2 {} }");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let body_stmts = &func_node.node.body.node.stmts;
    assert_eq!(body_stmts.len(), 1);

    let ast::Stmt::For(for_node) = &body_stmts[0].node else {
        panic!("expected For stmt");
    };
    let for_inner = &for_node.node;

    assert!(for_inner.reversed);
    assert!(for_inner.step.is_some());

    let step_expr = for_inner.step.as_ref().unwrap();
    let ast::ExprKind::Lit(ast::Lit::Int(2)) = &step_expr.node.kind else {
        panic!("expected Int(2) step");
    };
}

#[test]
fn for_parses_inclusive_range() {
    let prog = parse_program("fn main() { for n in 0..=10 {} }");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let body_stmts = &func_node.node.body.node.stmts;
    assert_eq!(body_stmts.len(), 1);

    let ast::Stmt::For(for_node) = &body_stmts[0].node else {
        panic!("expected For stmt");
    };
    let for_inner = &for_node.node;

    let ast::ExprKind::Range(range_node) = &for_inner.iterable.node.kind else {
        panic!("expected Range iterable");
    };
    assert!(range_node.node.inclusive);
}

#[test]
fn test_var_param_parses() {
    let prog = parse_program("fn f(var x: int) {}");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let params = &func_node.node.params;
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].mutability, Mutability::Mutable);
    assert_eq!(params[0].name.0.as_ref(), "x");
}

#[test]
fn test_mixed_params_parse() {
    let prog = parse_program("fn f(a: int, var b: int) {}");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Func(func_node) = &prog.stmts[0].node else {
        panic!("expected Func");
    };
    let params = &func_node.node.params;
    assert_eq!(params.len(), 2);
    assert_eq!(params[0].mutability, Mutability::Immutable);
    assert_eq!(params[0].name.0.as_ref(), "a");
    assert_eq!(params[1].mutability, Mutability::Mutable);
    assert_eq!(params[1].name.0.as_ref(), "b");
}

#[test]
fn test_var_self_parses() {
    let prog = parse_program("struct S { fn m(var self) {} }");
    assert_eq!(prog.stmts.len(), 1);
    let ast::Stmt::Struct(struct_node) = &prog.stmts[0].node else {
        panic!("expected Struct");
    };
    let methods = &struct_node.node.methods;
    assert_eq!(methods.len(), 1);
    assert_eq!(methods[0].receiver, Some(MethodReceiver::Var));
    assert_eq!(methods[0].name.0.as_ref(), "m");
}
