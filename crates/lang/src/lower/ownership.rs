use std::collections::HashSet;

use crate::hir::{self, Block, Expr, ExprKind, LocalId, Ownership, Stmt, StmtKind};

pub fn analyze_ownership(program: &mut hir::Program) {
    for func in &mut program.funcs {
        analyze_func(func);
    }
}

struct LivenessCtx {
    seen: HashSet<LocalId>,
    reassigned: HashSet<LocalId>,
    ref_locals: HashSet<LocalId>,
}

fn analyze_func(func: &mut hir::Func) {
    let reassigned = collect_reassigned(&func.body);
    let ref_locals = func
        .locals
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_ref)
        .map(|(i, _)| LocalId(i as u32))
        .collect();
    let mut ctx = LivenessCtx {
        seen: HashSet::new(),
        reassigned,
        ref_locals,
    };
    analyze_block(&mut func.body, &mut ctx);
}

fn collect_reassigned(block: &Block) -> HashSet<LocalId> {
    let mut set = HashSet::new();
    collect_reassigned_block(block, &mut set);
    set
}

fn collect_reassigned_block(block: &Block, set: &mut HashSet<LocalId>) {
    for stmt in &block.stmts {
        collect_reassigned_stmt(stmt, set);
    }
}

fn collect_reassigned_stmt(stmt: &Stmt, set: &mut HashSet<LocalId>) {
    match &stmt.kind {
        StmtKind::Assign { local, .. } => {
            set.insert(*local);
        }
        StmtKind::SetField { object, .. } => {
            set.insert(*object);
        }
        StmtKind::SetIndex { object, .. } => {
            set.insert(*object);
        }
        StmtKind::If {
            then_block,
            else_block,
            ..
        } => {
            collect_reassigned_block(then_block, set);
            if let Some(b) = else_block {
                collect_reassigned_block(b, set);
            }
        }
        StmtKind::While { body, .. } => {
            collect_reassigned_block(body, set);
        }
        StmtKind::Match {
            arms, else_body, ..
        } => {
            for arm in arms {
                collect_reassigned_block(&arm.body, set);
            }
            if let Some(eb) = else_body {
                collect_reassigned_block(&eb.body, set);
            }
        }
        StmtKind::Expr(e) => {
            collect_reassigned_expr(e, set);
        }
        StmtKind::Let { init, .. } | StmtKind::Return(Some(init)) => {
            collect_reassigned_expr(init, set);
        }
        StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue => {}
    }
}

fn collect_reassigned_expr(expr: &Expr, set: &mut HashSet<LocalId>) {
    match &expr.kind {
        ExprKind::CollectionMut { object, .. } => {
            set.insert(*object);
        }
        ExprKind::SortBy { collection, .. } => {
            set.insert(*collection);
        }
        ExprKind::Call { args, .. }
        | ExprKind::CallBuiltin { args, .. }
        | ExprKind::CallExtern { args, .. } => {
            for a in args {
                collect_reassigned_expr(a, set);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_reassigned_expr(lhs, set);
            collect_reassigned_expr(rhs, set);
        }
        ExprKind::Unary { expr, .. }
        | ExprKind::Cast(expr)
        | ExprKind::ToString(expr)
        | ExprKind::Format(expr, _)
        | ExprKind::CollectionLen { collection: expr }
        | ExprKind::MapLen { map: expr }
        | ExprKind::ArrayFill { value: expr, .. }
        | ExprKind::ListFill { value: expr, .. }
        | ExprKind::UnwrapOptional(expr) => {
            collect_reassigned_expr(expr, set);
        }
        ExprKind::FieldGet { object, .. } | ExprKind::TupleIndex { tuple: object, .. } => {
            collect_reassigned_expr(object, set);
        }
        ExprKind::IndexGet { target, index } | ExprKind::MapEntryAt { map: target, index } => {
            collect_reassigned_expr(target, set);
            collect_reassigned_expr(index, set);
        }
        ExprKind::Slice {
            target, start, end, ..
        } => {
            collect_reassigned_expr(target, set);
            collect_reassigned_expr(start, set);
            collect_reassigned_expr(end, set);
        }
        ExprKind::StructLiteral { fields, .. }
        | ExprKind::DataRefLiteral { fields, .. }
        | ExprKind::EnumLiteral { fields, .. } => {
            for f in fields {
                collect_reassigned_expr(f, set);
            }
        }
        ExprKind::TupleLiteral { elements }
        | ExprKind::ArrayLiteral { elements }
        | ExprKind::ListLiteral { elements } => {
            for e in elements {
                collect_reassigned_expr(e, set);
            }
        }
        ExprKind::MapLiteral { entries } => {
            for (k, v) in entries {
                collect_reassigned_expr(k, set);
                collect_reassigned_expr(v, set);
            }
        }
        ExprKind::CreateClosure { captures, .. } => {
            for c in captures {
                collect_reassigned_expr(c, set);
            }
        }
        ExprKind::CallClosure { callee, args, .. } => {
            collect_reassigned_expr(callee, set);
            for a in args {
                collect_reassigned_expr(a, set);
            }
        }
        ExprKind::Local(_)
        | ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Double(_)
        | ExprKind::Bool(_)
        | ExprKind::String(_)
        | ExprKind::Nil => {}
    }
}

// Backward liveness walk

fn analyze_block(block: &mut Block, ctx: &mut LivenessCtx) {
    for stmt in block.stmts.iter_mut().rev() {
        analyze_stmt(stmt, ctx);
    }
}

fn analyze_stmt(stmt: &mut Stmt, ctx: &mut LivenessCtx) {
    match &mut stmt.kind {
        StmtKind::Expr(e) => analyze_expr(e, ctx),
        StmtKind::Return(Some(e)) => analyze_expr(e, ctx),
        StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Let { init, .. } => analyze_expr(init, ctx),
        StmtKind::Assign { local, value } => {
            analyze_expr(value, ctx);
            // counts as a use so earlier reads of this local stay borrow
            ctx.seen.insert(*local);
        }
        StmtKind::SetField { object, value, .. } => {
            analyze_expr(value, ctx);
            ctx.seen.insert(*object);
        }
        StmtKind::SetIndex {
            object,
            index,
            value,
        } => {
            analyze_expr(value, ctx);
            let index = index.as_mut();
            analyze_expr(index, ctx);
            ctx.seen.insert(*object);
        }
        StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            let before = ctx.seen.clone();

            let mut then_ctx = LivenessCtx {
                seen: before.clone(),
                reassigned: ctx.reassigned.clone(),
                ref_locals: ctx.ref_locals.clone(),
            };
            analyze_block(then_block, &mut then_ctx);

            let mut else_ctx = LivenessCtx {
                seen: before.clone(),
                reassigned: ctx.reassigned.clone(),
                ref_locals: ctx.ref_locals.clone(),
            };
            if let Some(eb) = else_block {
                analyze_block(eb, &mut else_ctx);
            }

            ctx.seen = then_ctx.seen;
            for id in else_ctx.seen {
                ctx.seen.insert(id);
            }

            analyze_expr(cond, ctx);
        }
        StmtKind::While { cond, body } => {
            // pre mark all locals referenced in the loop so they stay borrow inside.
            let mut pre_seen = ctx.seen.clone();
            collect_locals_in_block(body, &mut pre_seen);
            collect_locals_in_expr(cond, &mut pre_seen);

            let mut loop_ctx = LivenessCtx {
                seen: pre_seen,
                reassigned: ctx.reassigned.clone(),
                ref_locals: ctx.ref_locals.clone(),
            };
            analyze_block(body, &mut loop_ctx);
            analyze_expr(cond, &mut loop_ctx);

            for id in loop_ctx.seen {
                ctx.seen.insert(id);
            }
        }
        StmtKind::Match {
            scrutinee_init,
            scrutinee,
            write_through: _,
            arms,
            else_body,
        } => {
            let before = ctx.seen.clone();

            let mut merged: HashSet<LocalId> = before.clone();

            for arm in arms.iter_mut() {
                let mut arm_ctx = LivenessCtx {
                    seen: before.clone(),
                    reassigned: ctx.reassigned.clone(),
                    ref_locals: ctx.ref_locals.clone(),
                };
                analyze_block(&mut arm.body, &mut arm_ctx);
                for id in &arm_ctx.seen {
                    merged.insert(*id);
                }
            }

            if let Some(eb) = else_body {
                let mut else_ctx = LivenessCtx {
                    seen: before.clone(),
                    reassigned: ctx.reassigned.clone(),
                    ref_locals: ctx.ref_locals.clone(),
                };
                analyze_block(&mut eb.body, &mut else_ctx);
                for id in &else_ctx.seen {
                    merged.insert(*id);
                }
            }

            // the scrutinee local is implicitly read by every arm
            merged.insert(*scrutinee);

            ctx.seen = merged;
            analyze_expr(scrutinee_init, ctx);
        }
    }
}

fn collect_locals_in_block(block: &Block, set: &mut HashSet<LocalId>) {
    for stmt in &block.stmts {
        collect_locals_in_stmt(stmt, set);
    }
}

fn collect_locals_in_stmt(stmt: &Stmt, set: &mut HashSet<LocalId>) {
    match &stmt.kind {
        StmtKind::Expr(e) | StmtKind::Return(Some(e)) | StmtKind::Let { init: e, .. } => {
            collect_locals_in_expr(e, set);
        }
        StmtKind::Assign { local, value } => {
            set.insert(*local);
            collect_locals_in_expr(value, set);
        }
        StmtKind::SetField { object, value, .. } => {
            set.insert(*object);
            collect_locals_in_expr(value, set);
        }
        StmtKind::SetIndex {
            object,
            index,
            value,
        } => {
            set.insert(*object);
            collect_locals_in_expr(index, set);
            collect_locals_in_expr(value, set);
        }
        StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            collect_locals_in_expr(cond, set);
            collect_locals_in_block(then_block, set);
            if let Some(b) = else_block {
                collect_locals_in_block(b, set);
            }
        }
        StmtKind::While { cond, body } => {
            collect_locals_in_expr(cond, set);
            collect_locals_in_block(body, set);
        }
        StmtKind::Match {
            scrutinee_init,
            scrutinee,
            write_through,
            arms,
            else_body,
        } => {
            collect_locals_in_expr(scrutinee_init, set);
            set.insert(*scrutinee);
            if let Some(wt) = write_through {
                set.insert(wt.ref_local);
            }
            for arm in arms {
                collect_locals_in_block(&arm.body, set);
            }
            if let Some(eb) = else_body {
                collect_locals_in_block(&eb.body, set);
            }
        }
        StmtKind::Return(None) | StmtKind::Break | StmtKind::Continue => {}
    }
}

fn collect_locals_in_expr(expr: &Expr, set: &mut HashSet<LocalId>) {
    match &expr.kind {
        ExprKind::Local(id) => {
            set.insert(*id);
        }
        ExprKind::CollectionMut { object, args, .. } => {
            set.insert(*object);
            for a in args {
                collect_locals_in_expr(a, set);
            }
        }
        ExprKind::SortBy { collection, .. } => {
            set.insert(*collection);
        }
        ExprKind::Call { args, .. }
        | ExprKind::CallBuiltin { args, .. }
        | ExprKind::CallExtern { args, .. } => {
            for a in args {
                collect_locals_in_expr(a, set);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            collect_locals_in_expr(lhs, set);
            collect_locals_in_expr(rhs, set);
        }
        ExprKind::Unary { expr, .. }
        | ExprKind::Cast(expr)
        | ExprKind::ToString(expr)
        | ExprKind::Format(expr, _)
        | ExprKind::CollectionLen { collection: expr }
        | ExprKind::MapLen { map: expr }
        | ExprKind::ArrayFill { value: expr, .. }
        | ExprKind::ListFill { value: expr, .. }
        | ExprKind::UnwrapOptional(expr) => {
            collect_locals_in_expr(expr, set);
        }
        ExprKind::FieldGet { object, .. } | ExprKind::TupleIndex { tuple: object, .. } => {
            collect_locals_in_expr(object, set);
        }
        ExprKind::IndexGet { target, index } | ExprKind::MapEntryAt { map: target, index } => {
            collect_locals_in_expr(target, set);
            collect_locals_in_expr(index, set);
        }
        ExprKind::Slice {
            target, start, end, ..
        } => {
            collect_locals_in_expr(target, set);
            collect_locals_in_expr(start, set);
            collect_locals_in_expr(end, set);
        }
        ExprKind::StructLiteral { fields, .. }
        | ExprKind::DataRefLiteral { fields, .. }
        | ExprKind::EnumLiteral { fields, .. } => {
            for f in fields {
                collect_locals_in_expr(f, set);
            }
        }
        ExprKind::TupleLiteral { elements }
        | ExprKind::ArrayLiteral { elements }
        | ExprKind::ListLiteral { elements } => {
            for e in elements {
                collect_locals_in_expr(e, set);
            }
        }
        ExprKind::MapLiteral { entries } => {
            for (k, v) in entries {
                collect_locals_in_expr(k, set);
                collect_locals_in_expr(v, set);
            }
        }
        ExprKind::CreateClosure { captures, .. } => {
            for c in captures {
                collect_locals_in_expr(c, set);
            }
        }
        ExprKind::CallClosure { callee, args, .. } => {
            collect_locals_in_expr(callee, set);
            for a in args {
                collect_locals_in_expr(a, set);
            }
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Double(_)
        | ExprKind::Bool(_)
        | ExprKind::String(_)
        | ExprKind::Nil => {}
    }
}

fn analyze_expr(expr: &mut Expr, ctx: &mut LivenessCtx) {
    // Recurse right-to-left, then set ownership on this node.
    match &mut expr.kind {
        ExprKind::Local(id) => {
            let id = *id;
            if ctx.ref_locals.contains(&id)
                || ctx.reassigned.contains(&id)
                || ctx.seen.contains(&id)
            {
                expr.ownership = Ownership::Borrow;
            } else {
                expr.ownership = Ownership::Move;
                ctx.seen.insert(id);
            }
            return;
        }
        ExprKind::CollectionMut { object, args, .. } => {
            for a in args.iter_mut().rev() {
                analyze_expr(a, ctx);
            }
            // object is read and mutated, so already in reassigned
            ctx.seen.insert(*object);
            expr.ownership = Ownership::Own;
            return;
        }
        ExprKind::SortBy { collection, .. } => {
            ctx.seen.insert(*collection);
            expr.ownership = Ownership::Own;
            return;
        }
        ExprKind::Call { args, .. }
        | ExprKind::CallBuiltin { args, .. }
        | ExprKind::CallExtern { args, .. } => {
            for a in args.iter_mut().rev() {
                analyze_expr(a, ctx);
            }
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            analyze_expr(rhs, ctx);
            analyze_expr(lhs, ctx);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Cast(inner)
        | ExprKind::ToString(inner)
        | ExprKind::Format(inner, _)
        | ExprKind::CollectionLen { collection: inner }
        | ExprKind::MapLen { map: inner }
        | ExprKind::ArrayFill { value: inner, .. }
        | ExprKind::ListFill { value: inner, .. }
        | ExprKind::UnwrapOptional(inner) => {
            analyze_expr(inner, ctx);
        }
        ExprKind::FieldGet { object, .. } | ExprKind::TupleIndex { tuple: object, .. } => {
            analyze_expr(object, ctx);
            expr.ownership = Ownership::Borrow;
            return;
        }
        ExprKind::IndexGet { target, index } | ExprKind::MapEntryAt { map: target, index } => {
            analyze_expr(index, ctx);
            analyze_expr(target, ctx);
            expr.ownership = Ownership::Borrow;
            return;
        }
        ExprKind::Slice {
            target, start, end, ..
        } => {
            analyze_expr(end, ctx);
            analyze_expr(start, ctx);
            analyze_expr(target, ctx);
            expr.ownership = Ownership::Borrow;
            return;
        }
        ExprKind::StructLiteral { fields, .. }
        | ExprKind::DataRefLiteral { fields, .. }
        | ExprKind::EnumLiteral { fields, .. } => {
            for f in fields.iter_mut().rev() {
                analyze_expr(f, ctx);
            }
        }
        ExprKind::TupleLiteral { elements }
        | ExprKind::ArrayLiteral { elements }
        | ExprKind::ListLiteral { elements } => {
            for e in elements.iter_mut().rev() {
                analyze_expr(e, ctx);
            }
        }
        ExprKind::MapLiteral { entries } => {
            for (k, v) in entries.iter_mut().rev() {
                analyze_expr(v, ctx);
                analyze_expr(k, ctx);
            }
        }
        ExprKind::CreateClosure { captures, .. } => {
            for c in captures.iter_mut().rev() {
                analyze_expr(c, ctx);
            }
        }
        ExprKind::CallClosure { callee, args, .. } => {
            for a in args.iter_mut().rev() {
                analyze_expr(a, ctx);
            }
            analyze_expr(callee, ctx);
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Double(_)
        | ExprKind::Bool(_)
        | ExprKind::String(_)
        | ExprKind::Nil => {}
    }
    // Non-early-return paths default to Own.
    expr.ownership = Ownership::Own;
}

#[cfg(test)]
mod tests {
    use crate::hir::{ExprKind, Ownership, StmtKind};
    use crate::test_helpers::TestCtx;

    fn lower(source: &str) -> crate::hir::Program {
        TestCtx::lower_ok(source)
    }

    fn find_main(prog: &crate::hir::Program) -> &crate::hir::Func {
        prog.funcs
            .iter()
            .find(|f| f.name.to_string() == "main")
            .expect("main not found")
    }

    #[test]
    fn single_use_is_move() {
        let prog = lower("fn main() { let x = 42; println(x); }");
        let main = find_main(&prog);
        // The println call's arg is Local(x), should be Move (last and only use)
        let StmtKind::Expr(call) = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args, .. } = &call.kind else {
            panic!()
        };
        assert_eq!(args[0].ownership, Ownership::Move);
    }

    #[test]
    fn second_use_is_move_first_is_borrow() {
        let prog = lower("fn main() { let x = 42; println(x); println(x); }");
        let main = find_main(&prog);
        let StmtKind::Expr(call1) = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args: args1, .. } = &call1.kind else {
            panic!()
        };
        let StmtKind::Expr(call2) = &main.body.stmts[2].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args: args2, .. } = &call2.kind else {
            panic!()
        };
        assert_eq!(args1[0].ownership, Ownership::Borrow);
        assert_eq!(args2[0].ownership, Ownership::Move);
    }

    #[test]
    fn use_before_if_is_borrow_when_used_inside() {
        let prog = lower("fn main() { let x = 42; println(x); if true { println(x); } }");
        let main = find_main(&prog);
        // First println (before if) should be Borrow because x is used inside if
        let StmtKind::Expr(call) = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args, .. } = &call.kind else {
            panic!()
        };
        assert_eq!(args[0].ownership, Ownership::Borrow);
    }

    #[test]
    fn reassigned_local_is_always_borrow() {
        let prog = lower("fn main() { var x = 1; println(x); x = 2; println(x); }");
        let main = find_main(&prog);
        let StmtKind::Expr(call1) = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args: args1, .. } = &call1.kind else {
            panic!()
        };
        let StmtKind::Expr(call2) = &main.body.stmts[3].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args: args2, .. } = &call2.kind else {
            panic!()
        };
        assert_eq!(args1[0].ownership, Ownership::Borrow);
        assert_eq!(args2[0].ownership, Ownership::Borrow);
    }

    #[test]
    fn return_local_is_move() {
        let prog = lower("fn foo() -> int { let x = 42; return x; } fn main() { foo(); }");
        let foo = prog
            .funcs
            .iter()
            .find(|f| f.name.to_string() == "foo")
            .unwrap();
        let StmtKind::Return(Some(ret_expr)) = &foo.body.stmts[1].kind else {
            panic!()
        };
        assert_eq!(ret_expr.ownership, Ownership::Move);
    }

    #[test]
    fn var_local_reassigned_later_is_borrow() {
        // var x is used, then reassigned — both uses must be Borrow.
        let prog = lower(r#"fn main() { var x = "a"; println(x); x = "b"; println(x); }"#);
        let main = find_main(&prog);
        // stmt[1]: println(x) — x is in reassigned set → Borrow
        let StmtKind::Expr(call1) = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args: args1, .. } = &call1.kind else {
            panic!()
        };
        assert_eq!(args1[0].ownership, Ownership::Borrow);
        // stmt[3]: println(x) — x is still in reassigned set → Borrow
        let StmtKind::Expr(call2) = &main.body.stmts[3].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args: args2, .. } = &call2.kind else {
            panic!()
        };
        assert_eq!(args2[0].ownership, Ownership::Borrow);
    }

    #[test]
    fn local_used_in_loop_is_borrow() {
        let prog = lower(
            r#"fn main() { let x = "hi"; var i = 0; while i < 3 { println(x); i = i + 1; } }"#,
        );
        let main = find_main(&prog);
        // x is captured by the loop pre-scan → Borrow inside the loop body
        let StmtKind::While { body, .. } = &main.body.stmts[2].kind else {
            panic!()
        };
        let StmtKind::Expr(call) = &body.stmts[0].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args, .. } = &call.kind else {
            panic!()
        };
        assert_eq!(args[0].ownership, Ownership::Borrow);
    }

    #[test]
    fn local_used_in_one_match_arm_is_borrow_before() {
        let prog = lower(
            r#"fn main() {
                let x = "val";
                let y = 1;
                println(x);
                match y {
                    1 => { println(x); },
                    _ => {},
                }
            }"#,
        );
        let main = find_main(&prog);
        // stmt[2]: println(x) before the match — x is used in arm 1 → merged seen includes x
        // → walking backward, x is in ctx.seen when stmt[2] is processed → Borrow
        let StmtKind::Expr(call) = &main.body.stmts[2].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args, .. } = &call.kind else {
            panic!()
        };
        assert_eq!(args[0].ownership, Ownership::Borrow);
    }

    #[test]
    fn nested_call_inner_arg_is_move() {
        let prog = lower(
            "fn bar(x: int) -> int { x } fn foo(x: int) -> int { x } fn main() { let x = 1; foo(bar(x)); }",
        );
        let main = find_main(&prog);
        // stmt[1]: foo(bar(x)) — x is only used once (inside bar), so it's Move
        let StmtKind::Expr(outer_call) = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::Call {
            args: outer_args, ..
        } = &outer_call.kind
        else {
            panic!()
        };
        let ExprKind::Call {
            args: inner_args, ..
        } = &outer_args[0].kind
        else {
            panic!()
        };
        assert_eq!(inner_args[0].ownership, Ownership::Move);
    }

    #[test]
    fn field_get_is_borrow() {
        let prog = lower(
            "struct Foo { value: int } fn main() { let f = Foo { value: 42 }; println(f.value); }",
        );
        let main = find_main(&prog);
        let StmtKind::Expr(call) = &main.body.stmts[1].kind else {
            panic!()
        };
        let ExprKind::CallBuiltin { args, .. } = &call.kind else {
            panic!()
        };
        assert_eq!(args[0].ownership, Ownership::Borrow);
    }
}
