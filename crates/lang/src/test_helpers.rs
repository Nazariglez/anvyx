use internment::Intern;

use crate::{
    Profile, ast,
    ast::{BinaryOp, Ident, Type},
    hir::{Block, Expr, Func, FuncId, Local, LocalId, Program, Stmt, StmtKind},
    lower,
    lower::LowerError,
    span::Span,
    typecheck, vm,
};

const TEST_CORE_PRELUDE: &str = concat!(
    include_str!("../../core/src/option.anv"),
    "\n",
    include_str!("../../core/src/range.anv"),
);
use std::collections::HashMap;

// ---- pipeline helpers ----

pub(crate) fn generate_hir(source: &str, file_path: &str) -> Result<Program, String> {
    crate::generate_hir_with_std(
        source,
        file_path,
        TEST_CORE_PRELUDE,
        &HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
    )
}

pub(crate) struct TestCtx;

impl TestCtx {
    #[track_caller]
    pub(crate) fn lower_ok(source: &str) -> Program {
        let (ast, tcx) = Self::pipeline(source);
        lower::lower_program(&ast, &tcx, &[]).expect("lowering should succeed")
    }

    #[track_caller]
    pub(crate) fn lower_err(source: &str) -> LowerError {
        let (ast, tcx) = Self::pipeline(source);
        lower::lower_program(&ast, &tcx, &[]).expect_err("lowering should fail")
    }

    #[track_caller]
    pub(crate) fn vm_ok(source: &str) -> String {
        let hir = generate_hir(source, "<test>").expect("generate_hir failed");
        vm::run_with_externs(&hir, HashMap::new(), Profile::default()).expect("vm run failed")
    }

    #[track_caller]
    pub(crate) fn vm_err(source: &str) -> String {
        let hir = generate_hir(source, "<test>").expect("generate_hir failed");
        vm::run_with_externs(&hir, HashMap::new(), Profile::default())
            .expect_err("expected vm error")
    }

    fn pipeline(source: &str) -> (ast::Program, typecheck::TypecheckResult) {
        let prelude_tokens =
            crate::lexer::tokenize(TEST_CORE_PRELUDE).expect("prelude must tokenize");
        let prelude_ast = crate::parser::parse_ast(&prelude_tokens).expect("prelude must parse");

        let user_tokens = crate::lexer::tokenize(source).expect("source must tokenize");
        let user_ast = crate::parser::parse_ast(&user_tokens).expect("source must parse");

        let mut stmts = prelude_ast.stmts;
        stmts.extend(user_ast.stmts);
        let combined = ast::Program { stmts };

        let tcx = typecheck::check_program_with_modules(&combined, &[], &[], &[])
            .expect("source must typecheck");
        (combined, tcx)
    }
}

// ---- shared span / ident helpers ----

pub(crate) fn dummy_span() -> Span {
    Span::new(0, 0)
}

pub(crate) fn dummy_ident(name: &str) -> Ident {
    Ident(Intern::new(name.to_string()))
}

// ---- HIR expression builders ----

pub(crate) fn hir_int_expr(v: i64) -> Expr {
    Expr::int_lit(dummy_span(), v)
}

pub(crate) fn hir_bool_expr(v: bool) -> Expr {
    Expr::bool_lit(dummy_span(), v)
}

pub(crate) fn hir_local_expr(id: u32) -> Expr {
    Expr::local(Type::Int, dummy_span(), LocalId(id))
}

pub(crate) fn hir_binary_expr(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::binary(Type::Int, dummy_span(), op, lhs, rhs)
}

// ---- HIR statement / function / program builders ----

pub(crate) fn hir_stmt(kind: StmtKind) -> Stmt {
    Stmt {
        span: dummy_span(),
        kind,
    }
}

pub(crate) fn hir_simple_func(
    name: &str,
    stmts: Vec<Stmt>,
    locals: Vec<Local>,
    params_len: u32,
    ret: Type,
) -> Func {
    Func {
        id: FuncId(0),
        name: dummy_ident(name),
        locals,
        params_len,
        ret,
        body: Block { stmts },
        span: dummy_span(),
    }
}

pub(crate) fn hir_main_func(stmts: Vec<Stmt>) -> Func {
    hir_simple_func("main", stmts, vec![], 0, Type::Void)
}

pub(crate) fn hir_program(func: Func) -> Program {
    Program {
        funcs: vec![func],
        externs: vec![],
        aggregate_meta: vec![],
        enum_meta: vec![],
    }
}
