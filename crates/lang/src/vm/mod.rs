mod builtins;
mod bytecode;
mod compiler;
mod runtime;
mod value;

use crate::hir;
use std::collections::HashMap;

pub use runtime::ExternHandler;
pub use value::RuntimeError;
pub use value::Value;

pub fn run(hir_prog: &hir::Program) -> Result<String, String> {
    run_with_externs(hir_prog, HashMap::new())
}

pub fn run_with_externs(
    hir_prog: &hir::Program,
    externs: HashMap<String, ExternHandler>,
) -> Result<String, String> {
    let compiled = compiler::compile(hir_prog).map_err(|e| format!("Compile error: {e}"))?;
    let mut vm = runtime::VM::new(&compiled);

    for (name, handler) in externs {
        let idx = compiled.extern_names.iter().position(|n| n == &name);
        match idx {
            Some(i) => vm.register_extern(i, handler),
            None => {
                return Err(format!(
                    "Registered extern '{name}' was not declared in the program"
                ));
            }
        }
    }

    vm.run().map_err(|e| format!("Runtime error: {e}"))?;
    Ok(vm.stdout)
}

#[cfg(test)]
mod tests {
    use super::{ExternHandler, Value, run_with_externs};
    use crate::test_helpers::TestCtx;
    use std::collections::HashMap;

    fn vm_ok(source: &str) -> String {
        TestCtx::vm_ok(source)
    }

    fn vm_err(source: &str) -> String {
        TestCtx::vm_err(source)
    }

    fn vm_ok_with_externs(source: &str, externs: HashMap<String, ExternHandler>) -> String {
        let hir = crate::generate_hir(source, "<test>").expect("generate_hir failed");
        run_with_externs(&hir, externs).expect("vm run failed")
    }

    fn vm_err_with_externs(source: &str, externs: HashMap<String, ExternHandler>) -> String {
        let hir = crate::generate_hir(source, "<test>").expect("generate_hir failed");
        run_with_externs(&hir, externs).expect_err("expected vm error")
    }

    #[test]
    fn empty_main() {
        let out = vm_ok("fn main() {}");
        assert!(out.is_empty());
    }

    #[test]
    fn let_binding() {
        let out = vm_ok("fn main() { let x = 42; }");
        assert!(out.is_empty());
    }

    #[test]
    fn return_int() {
        let out = vm_ok("fn main() -> int { 42 }");
        assert!(out.is_empty());
    }

    #[test]
    fn println_hello() {
        let out = vm_ok(r#"fn main() { println("hello"); }"#);
        assert_eq!(out, "hello\n");
    }

    #[test]
    fn println_multiple_lines() {
        let out = vm_ok(r#"fn main() { println("a"); println("b"); }"#);
        assert_eq!(out, "a\nb\n");
    }

    #[test]
    fn assert_true_succeeds() {
        let out = vm_ok("fn main() { assert(true); }");
        assert!(out.is_empty());
    }

    #[test]
    fn assert_false_fails() {
        let err = vm_err("fn main() { assert(false); }");
        assert!(err.contains("assertion failed"));
    }

    #[test]
    fn assert_msg_false_fails_with_message() {
        let err = vm_err(r#"fn main() { assert_msg(false, "oops"); }"#);
        assert!(err.contains("oops"));
    }

    #[test]
    fn if_true_branch() {
        let out = vm_ok(r#"fn main() { if true { println("yes"); } }"#);
        assert_eq!(out, "yes\n");
    }

    #[test]
    fn if_false_skips_branch() {
        let out = vm_ok(r#"fn main() { if false { println("no"); } }"#);
        assert!(out.is_empty());
    }

    #[test]
    fn if_else() {
        let out = vm_ok(r#"fn main() { if false { println("a"); } else { println("b"); } }"#);
        assert_eq!(out, "b\n");
    }

    #[test]
    fn while_false_does_not_execute_body() {
        let out = vm_ok(r#"fn main() { while false { println("x"); } }"#);
        assert!(out.is_empty());
    }

    #[test]
    fn while_with_break() {
        let out = vm_ok(
            r#"
            fn main() {
                var i = 0;
                while true {
                    i = i + 1;
                    if i == 3 { break; }
                }
                println("done");
            }
        "#,
        );
        assert_eq!(out, "done\n");
    }

    #[test]
    fn while_with_continue() {
        let out = vm_ok(
            r#"
            fn main() {
                var i = 0;
                var sum = 0;
                while i < 5 {
                    i = i + 1;
                    if i == 3 { continue; }
                    sum = sum + i;
                }
                println("ok");
            }
        "#,
        );
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn call_empty_function() {
        let out = vm_ok("fn foo() {} fn main() { foo(); }");
        assert!(out.is_empty());
    }

    #[test]
    fn call_function_with_return_value() {
        let out = vm_ok(
            r#"
            fn answer() -> int { 42 }
            fn main() { let x = answer(); println("ok"); }
        "#,
        );
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn call_function_with_args() {
        let out = vm_ok(
            r#"
            fn add(a: int, b: int) -> int { a + b }
            fn main() { add(1, 2); }
        "#,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn arithmetic_ops() {
        let out = vm_ok(
            r#"
            fn main() {
                let a = 10;
                let b = 3;
                let add = a + b;
                let sub = a - b;
                let mul = a * b;
                let div = a / b;
                let rem = a % b;
                println("ok");
            }
        "#,
        );
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn string_concat() {
        let out = vm_ok(
            r#"
            fn main() {
                let s = "hello" + " " + "world";
                println(s);
            }
        "#,
        );
        assert_eq!(out, "hello world\n");
    }

    #[test]
    fn string_interp_int() {
        let out = vm_ok(r#"fn main() { let x = 42; println("val: {x}"); }"#);
        assert_eq!(out, "val: 42\n");
    }

    #[test]
    fn string_interp_multiple_values() {
        let out = vm_ok(r#"fn main() { let a = "hi"; let b = 3; println("{a} {b}"); }"#);
        assert_eq!(out, "hi 3\n");
    }

    #[test]
    fn nested_function_calls() {
        let out = vm_ok(
            r#"
            fn double(x: int) -> int { x * 2 }
            fn quadruple(x: int) -> int { double(double(x)) }
            fn main() {
                quadruple(3);
                println("ok");
            }
        "#,
        );
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn if_expr_returns_value() {
        let out = vm_ok(
            r#"
            fn choose(x: bool) -> int { if x { 1 } else { 2 } }
            fn main() {
                let a = choose(true);
                let b = choose(false);
                if a == 1 {
                    println("ok");
                }
                if b == 2 {
                    println("ok");
                }
            }
        "#,
        );
        assert_eq!(out, "ok\nok\n");
    }

    #[test]
    fn fib_recursive() {
        let out = vm_ok(
            r#"
            fn fib(n: int) -> int {
                if n <= 1 {
                    n
                } else {
                    fib(n - 1) + fib(n - 2)
                }
            }
            fn main() {
                let result = fib(10);
                if result == 55 {
                    println("fib ok");
                } else {
                    println("fib failed");
                }
            }
        "#,
        );
        assert_eq!(out, "fib ok\n");
    }

    #[test]
    fn bool_operations() {
        let out = vm_ok(
            r#"
            fn main() {
                let t = true;
                let f = false;
                let a = t && f;
                let b = t || f;
                let c = !t;
                println("ok");
            }
        "#,
        );
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn comparison_ops() {
        let out = vm_ok(
            r#"
            fn main() {
                let x = 5;
                let eq = x == 5;
                let ne = x != 4;
                let lt = x < 10;
                let gt = x > 1;
                let le = x <= 5;
                let ge = x >= 5;
                println("ok");
            }
        "#,
        );
        assert_eq!(out, "ok\n");
    }

    // ---- extern fn vm ----

    #[test]
    fn extern_fn_returns_value() {
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "add".to_string(),
            Box::new(|args| {
                let Value::Int(a) = args[0] else {
                    panic!("expected int")
                };
                let Value::Int(b) = args[1] else {
                    panic!("expected int")
                };
                Ok(Value::Int(a + b))
            }),
        );
        let src = r#"
            extern fn add(a: int, b: int) -> int
            fn main() {
                let result = add(3, 4);
                assert(result == 7);
                println("ok");
            }
        "#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_fn_zero_args() {
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "get_answer".to_string(),
            Box::new(|_args| Ok(Value::Int(42))),
        );
        let src = r#"
            extern fn get_answer() -> int
            fn main() {
                let x = get_answer();
                assert(x == 42);
                println("ok");
            }
        "#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_fn_void_with_side_effect() {
        use std::sync::{Arc, Mutex};
        let counter = Arc::new(Mutex::new(0i64));
        let counter_clone = counter.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "increment".to_string(),
            Box::new(move |_args| {
                *counter_clone.lock().unwrap() += 1;
                Ok(Value::Nil)
            }),
        );
        let src = r#"
            extern fn increment()
            fn main() {
                increment();
                increment();
                println("ok");
            }
        "#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[test]
    fn extern_fn_missing_handler_errors() {
        let externs: HashMap<String, ExternHandler> = HashMap::new();
        let src = r#"
            extern fn add(a: int, b: int) -> int
            fn main() { let x = add(1, 2); }
        "#;
        let err = vm_err_with_externs(src, externs);
        assert!(
            err.contains("missing extern") && err.contains("add"),
            "got: {err}"
        );
    }

    #[test]
    fn extern_fn_multiple_externs() {
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "double".to_string(),
            Box::new(|args| {
                let Value::Int(n) = args[0] else {
                    panic!("expected int")
                };
                Ok(Value::Int(n * 2))
            }),
        );
        externs.insert(
            "negate".to_string(),
            Box::new(|args| {
                let Value::Int(n) = args[0] else {
                    panic!("expected int")
                };
                Ok(Value::Int(-n))
            }),
        );
        let src = r#"
            extern fn double(n: int) -> int
            extern fn negate(n: int) -> int
            fn main() {
                let x = double(5);
                let y = negate(x);
                assert(y == -10);
                println("ok");
            }
        "#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_handle_round_trip() {
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_handle".to_string(),
            Box::new(|_args| Ok(Value::ExternHandle(42))),
        );
        externs.insert(
            "use_handle".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(id) = args[0] else {
                    panic!("expected ExternHandle");
                };
                assert_eq!(id, 42);
                Ok(Value::Nil)
            }),
        );
        let src = "
extern type Foo
extern fn make_handle() -> Foo
extern fn use_handle(h: Foo)
fn main() {
    let h = make_handle();
    use_handle(h);
    println(\"ok\");
}";
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_handle_display() {
        assert_eq!(Value::ExternHandle(99).to_string(), "<extern:99>");
    }
}
