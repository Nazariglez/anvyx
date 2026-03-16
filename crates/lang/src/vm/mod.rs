mod builtins;
mod bytecode;
mod compiler;
mod runtime;
mod value;

use crate::hir;

pub fn run(hir_prog: &hir::Program) -> Result<String, String> {
    let compiled = compiler::compile(hir_prog).map_err(|e| format!("Compile error: {e}"))?;
    let mut vm = runtime::VM::new(&compiled);
    vm.run().map_err(|e| format!("Runtime error: {e}"))?;
    Ok(vm.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vm_ok(source: &str) -> String {
        let hir = crate::generate_hir(source, "<test>").expect("generate_hir failed");
        run(&hir).expect("vm run failed")
    }

    fn vm_err(source: &str) -> String {
        let hir = crate::generate_hir(source, "<test>").expect("generate_hir failed");
        run(&hir).expect_err("expected vm error")
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
        let out = vm_ok(r#"
            fn main() {
                var i = 0;
                while true {
                    i = i + 1;
                    if i == 3 { break; }
                }
                println("done");
            }
        "#);
        assert_eq!(out, "done\n");
    }

    #[test]
    fn while_with_continue() {
        let out = vm_ok(r#"
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
        "#);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn call_empty_function() {
        let out = vm_ok("fn foo() {} fn main() { foo(); }");
        assert!(out.is_empty());
    }

    #[test]
    fn call_function_with_return_value() {
        let out = vm_ok(r#"
            fn answer() -> int { 42 }
            fn main() { let x = answer(); println("ok"); }
        "#);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn call_function_with_args() {
        let out = vm_ok(r#"
            fn add(a: int, b: int) -> int { a + b }
            fn main() { add(1, 2); }
        "#);
        assert!(out.is_empty());
    }

    #[test]
    fn arithmetic_ops() {
        let out = vm_ok(r#"
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
        "#);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn string_concat() {
        let out = vm_ok(r#"
            fn main() {
                let s = "hello" + " " + "world";
                println(s);
            }
        "#);
        assert_eq!(out, "hello world\n");
    }

    #[test]
    fn nested_function_calls() {
        let out = vm_ok(r#"
            fn double(x: int) -> int { x * 2 }
            fn quadruple(x: int) -> int { double(double(x)) }
            fn main() {
                quadruple(3);
                println("ok");
            }
        "#);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn bool_operations() {
        let out = vm_ok(r#"
            fn main() {
                let t = true;
                let f = false;
                let a = t && f;
                let b = t || f;
                let c = !t;
                println("ok");
            }
        "#);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn comparison_ops() {
        let out = vm_ok(r#"
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
        "#);
        assert_eq!(out, "ok\n");
    }
}

