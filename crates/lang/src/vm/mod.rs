mod builtins;
mod bytecode;
mod compiler;
pub(crate) mod cycle_collector;
pub mod extern_type;
pub mod handle_store;
pub mod managed_rc;
pub mod meta;
mod runtime;
mod value;

use std::{collections::HashMap, fmt::Write};

pub use extern_type::{AnvyxConvert, AnvyxExternType, extern_handle};
pub use handle_store::HandleStore;
pub use managed_rc::ManagedRc;
pub use runtime::ExternHandler;
pub use value::{
    DisplayDetect, DisplayDetectFallback, EnumData, ExternHandleData, MapStorage, RuntimeError,
    StructData, Value,
};

use crate::hir;

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

    // we need to take the stdout before dropping the VM because the VM drops the stack
    let stdout = std::mem::take(&mut vm.stdout);
    drop(vm);

    // free any remaining suspects
    cycle_collector::collect_cycles();

    // check for memory leaks and output the details if any are found
    // the user should be able to see this if we do thing right and the cycle detector works
    let live = managed_rc::managed_alloc_count();
    if live > 0 {
        let mut msg = format!(
            "memory leak: {live} managed object(s) still alive after final cycle collection"
        );
        let mut details = managed_rc::managed_alloc_details();
        details.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in details {
            let _ = write!(msg, "\n  {name}: {count}");
        }
        return Err(msg);
    }

    Ok(stdout)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        ExternHandler, Value, managed_rc::ManagedRc, run_with_externs, value::ExternHandleData,
    };
    use crate::test_helpers::TestCtx;

    fn noop_drop(_id: u64) {}

    fn extern_handle(id: u64) -> Value {
        Value::ExternHandle(ManagedRc::new(ExternHandleData {
            id,
            drop_fn: noop_drop,
            type_name: "Extern",
            to_string_fn: |_| "<Extern>".to_string(),
        }))
    }

    fn vm_ok(source: &str) -> String {
        TestCtx::vm_ok(source)
    }

    fn vm_err(source: &str) -> String {
        TestCtx::vm_err(source)
    }

    fn vm_ok_with_externs(source: &str, externs: HashMap<String, ExternHandler>) -> String {
        let hir = crate::test_helpers::generate_hir(source, "<test>").expect("generate_hir failed");
        run_with_externs(&hir, externs).expect("vm run failed")
    }

    fn vm_err_with_externs(source: &str, externs: HashMap<String, ExternHandler>) -> String {
        let hir = crate::test_helpers::generate_hir(source, "<test>").expect("generate_hir failed");
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
        let out = vm_ok(r#"fn main() { let x = 42; println(f"val: {x}"); }"#);
        assert_eq!(out, "val: 42\n");
    }

    #[test]
    fn string_interp_multiple_values() {
        let out = vm_ok(r#"fn main() { let a = "hi"; let b = 3; println(f"{a} {b}"); }"#);
        assert_eq!(out, "hi 3\n");
    }

    #[test]
    fn nested_function_calls() {
        let out = vm_ok(
            r#"
            fn twice(x: int) -> int { x * 2 }
            fn quadruple(x: int) -> int { twice(twice(x)) }
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
            extern fn add(a: int, b: int) -> int;
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
            extern fn get_answer() -> int;
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
            extern fn increment();
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
            extern fn add(a: int, b: int) -> int;
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
            "twice".to_string(),
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
            extern fn twice(n: int) -> int;
            extern fn negate(n: int) -> int;
            fn main() {
                let x = twice(5);
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
            Box::new(|_args| Ok(extern_handle(42))),
        );
        externs.insert(
            "use_handle".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(ref data) = args[0] else {
                    panic!("expected ExternHandle");
                };
                let id = data.id;
                assert_eq!(id, 42);
                Ok(Value::Nil)
            }),
        );
        let src = "
extern type Foo;
extern fn make_handle() -> Foo;
extern fn use_handle(h: Foo);
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
        assert_eq!(extern_handle(99).to_string(), "<extern:99>");
    }

    #[test]
    fn extern_type_field_get() {
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_point".to_string(),
            Box::new(|_args| Ok(extern_handle(1))),
        );
        externs.insert(
            "Point::__get_x".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(ref data) = args[0] else {
                    panic!("expected ExternHandle");
                };
                let id = data.id;
                assert_eq!(id, 1);
                Ok(Value::Float(3.5_f32))
            }),
        );
        let src = r#"
extern type Point {
    x: float;
}
extern fn make_point() -> Point;
fn main() {
    let p = make_point();
    let v = p.x;
    assert(v == 3.5);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_type_field_set() {
        use std::sync::{Arc, Mutex};
        let captured = Arc::new(Mutex::new(0.0f64));
        let captured_clone = captured.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_point".to_string(),
            Box::new(|_args| Ok(extern_handle(1))),
        );
        externs.insert(
            "Point::__set_x".to_string(),
            Box::new(move |args| {
                let Value::ExternHandle(ref data) = args[0] else {
                    panic!("expected ExternHandle");
                };
                let id = data.id;
                assert_eq!(id, 1);
                let Value::Float(val) = args[1] else {
                    panic!("expected Float");
                };
                *captured_clone.lock().unwrap() = val as f64;
                Ok(Value::Nil)
            }),
        );
        let src = r#"
extern type Point {
    x: float;
}
extern fn make_point() -> Point;
fn main() {
    var p = make_point();
    p.x = 42.0;
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        assert_eq!(*captured.lock().unwrap(), 42.0);
    }

    #[test]
    fn extern_type_instance_method() {
        use std::sync::{Arc, Mutex};
        let captured = Arc::new(Mutex::new((0u64, 0.0f64, 0.0f64)));
        let captured_clone = captured.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_point".to_string(),
            Box::new(|_args| Ok(extern_handle(7))),
        );
        externs.insert(
            "Point::move_by".to_string(),
            Box::new(move |args| {
                let Value::ExternHandle(ref data) = args[0] else {
                    panic!("expected ExternHandle");
                };
                let id = data.id;
                let Value::Float(dx) = args[1] else {
                    panic!("expected Float dx");
                };
                let Value::Float(dy) = args[2] else {
                    panic!("expected Float dy");
                };
                *captured_clone.lock().unwrap() = (id, dx as f64, dy as f64);
                Ok(Value::Nil)
            }),
        );
        let src = r#"
extern type Point {
    fn move_by(var self, dx: float, dy: float);
}
extern fn make_point() -> Point;
fn main() {
    var p = make_point();
    p.move_by(5.0, -3.0);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        let vals = *captured.lock().unwrap();
        assert_eq!(vals, (7, 5.0, -3.0));
    }

    #[test]
    fn extern_type_method_returns_value() {
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_point".to_string(),
            Box::new(|_args| Ok(extern_handle(1))),
        );
        externs.insert(
            "Point::length".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle");
                };
                Ok(Value::Float(5.0_f32))
            }),
        );
        let src = r#"
extern type Point {
    fn length(self) -> float;
}
extern fn make_point() -> Point;
fn main() {
    let p = make_point();
    let d = p.length();
    assert(d == 5.0);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_type_combined() {
        use std::sync::{Arc, Mutex};
        let set_val = Arc::new(Mutex::new(0.0f64));
        let set_val_clone = set_val.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_point".to_string(),
            Box::new(|_args| Ok(extern_handle(1))),
        );
        externs.insert(
            "Point::__get_x".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle");
                };
                Ok(Value::Float(10.0_f32))
            }),
        );
        externs.insert(
            "Point::__set_x".to_string(),
            Box::new(move |args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle");
                };
                let Value::Float(val) = args[1] else {
                    panic!("expected Float");
                };
                *set_val_clone.lock().unwrap() = val as f64;
                Ok(Value::Nil)
            }),
        );
        externs.insert(
            "Point::length".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle");
                };
                Ok(Value::Float(5.0_f32))
            }),
        );
        let src = r#"
extern type Point {
    x: float;
    fn length(self) -> float;
}
extern fn make_point() -> Point;
fn main() {
    var p = make_point();
    let v = p.x;
    assert(v == 10.0);
    let d = p.length();
    assert(d == 5.0);
    p.x = 99.0;
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        assert_eq!(*set_val.lock().unwrap(), 99.0);
    }

    #[test]
    fn extern_type_static_method() {
        use std::sync::{Arc, Mutex};
        let called_args = Arc::new(Mutex::new(vec![]));
        let called_args_clone = called_args.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "Point::new".to_string(),
            Box::new(move |args| {
                *called_args_clone.lock().unwrap() = args.to_vec();
                Ok(extern_handle(42))
            }),
        );
        let src = r#"
extern type Point {
    fn new(x: float, y: float) -> Self;
}
fn main() {
    let p = Point.new(1.0, 2.0);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        let args = called_args.lock().unwrap();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], Value::Float(1.0_f32));
        assert_eq!(args[1], Value::Float(2.0_f32));
    }

    #[test]
    fn extern_type_static_with_instance() {
        use std::sync::{Arc, Mutex};
        let move_args = Arc::new(Mutex::new((0u64, 0.0f64, 0.0f64)));
        let move_args_clone = move_args.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "Point::new".to_string(),
            Box::new(|_args| Ok(extern_handle(7))),
        );
        externs.insert(
            "Point::__get_x".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle");
                };
                Ok(Value::Float(3.0_f32))
            }),
        );
        externs.insert(
            "Point::move_by".to_string(),
            Box::new(move |args| {
                let Value::ExternHandle(ref data) = args[0] else {
                    panic!("expected ExternHandle");
                };
                let id = data.id;
                let Value::Float(dx) = args[1] else {
                    panic!("expected Float dx");
                };
                let Value::Float(dy) = args[2] else {
                    panic!("expected Float dy");
                };
                *move_args_clone.lock().unwrap() = (id, dx as f64, dy as f64);
                Ok(Value::Nil)
            }),
        );
        let src = r#"
extern type Point {
    x: float;
    fn new(x: float, y: float) -> Self;
    fn move_by(var self, dx: float, dy: float);
}
fn main() {
    var p = Point.new(1.0, 2.0);
    let v = p.x;
    assert(v == 3.0);
    p.move_by(5.0, -3.0);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        let vals = *move_args.lock().unwrap();
        assert_eq!(vals, (7, 5.0, -3.0));
    }

    #[test]
    fn extern_type_static_void() {
        use std::sync::{Arc, Mutex};
        let called = Arc::new(Mutex::new(false));
        let called_clone = called.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "Logger::reset".to_string(),
            Box::new(move |_args| {
                *called_clone.lock().unwrap() = true;
                Ok(Value::Nil)
            }),
        );
        let src = r#"
extern type Logger {
    fn reset();
}
fn main() {
    Logger.reset();
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        assert!(*called.lock().unwrap());
    }

    #[test]
    fn extern_type_struct_literal_init() {
        use std::sync::{Arc, Mutex};
        let init_args = Arc::new(Mutex::new(vec![]));
        let init_args_clone = init_args.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "Point::__init__".to_string(),
            Box::new(move |args| {
                *init_args_clone.lock().unwrap() = args.to_vec();
                Ok(extern_handle(10))
            }),
        );
        externs.insert(
            "Point::__get_x".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle")
                };
                Ok(Value::Float(1.0_f32))
            }),
        );
        externs.insert(
            "Point::__get_y".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle")
                };
                Ok(Value::Float(2.0_f32))
            }),
        );
        let src = r#"
extern type Point { init; x: float; y: float; }
fn main() {
    let p = Point { x: 1.0, y: 2.0 };
    assert(p.x == 1.0);
    assert(p.y == 2.0);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
        let args = init_args.lock().unwrap();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], Value::Float(1.0_f32));
        assert_eq!(args[1], Value::Float(2.0_f32));
    }

    #[test]
    fn extern_type_destructure() {
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert("make_point".to_string(), Box::new(|_| Ok(extern_handle(5))));
        externs.insert(
            "Point::__get_x".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle")
                };
                Ok(Value::Float(7.5_f32))
            }),
        );
        externs.insert(
            "Point::__get_y".to_string(),
            Box::new(|args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle")
                };
                Ok(Value::Float(3.0_f32))
            }),
        );
        let src = r#"
extern type Point { x: float; y: float; }
extern fn make_point() -> Point;
fn main() {
    let p = make_point();
    let Point { x, y } = p;
    assert(x == 7.5);
    assert(y == 3.0);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_type_init_destructure_round_trip() {
        use std::sync::{Arc, Mutex};
        let state = Arc::new(Mutex::new((0.0f64, 0.0f64)));
        let state_init = state.clone();
        let state_move = state.clone();
        let state_get_x = state.clone();
        let state_get_y = state.clone();
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "Point::__init__".to_string(),
            Box::new(move |args| {
                let Value::Float(x) = args[0] else {
                    panic!("expected Float")
                };
                let Value::Float(y) = args[1] else {
                    panic!("expected Float")
                };
                *state_init.lock().unwrap() = (x as f64, y as f64);
                Ok(extern_handle(1))
            }),
        );
        externs.insert(
            "Point::move_by".to_string(),
            Box::new(move |args| {
                let Value::ExternHandle(_) = args[0] else {
                    panic!("expected ExternHandle")
                };
                let Value::Float(dx) = args[1] else {
                    panic!("expected Float")
                };
                let Value::Float(dy) = args[2] else {
                    panic!("expected Float")
                };
                let mut s = state_move.lock().unwrap();
                s.0 += dx as f64;
                s.1 += dy as f64;
                Ok(Value::Nil)
            }),
        );
        externs.insert(
            "Point::__get_x".to_string(),
            Box::new(move |_| Ok(Value::Float(state_get_x.lock().unwrap().0 as f32))),
        );
        externs.insert(
            "Point::__get_y".to_string(),
            Box::new(move |_| Ok(Value::Float(state_get_y.lock().unwrap().1 as f32))),
        );
        let src = r#"
extern type Point {
    init;
    x: float;
    y: float;
    fn move_by(var self, dx: float, dy: float);
}
fn main() {
    var p = Point { x: 10.0, y: 20.0 };
    p.move_by(5.0, -3.0);
    let Point { x, y } = p;
    assert(x == 15.0);
    assert(y == 17.0);
    println("ok");
}
"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "ok\n");
    }

    #[test]
    fn extern_handle_to_string_custom() {
        fn custom_fmt(_id: u64) -> String {
            "MyWidget".to_string()
        }
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_widget".to_string(),
            Box::new(|_args| {
                Ok(Value::ExternHandle(ManagedRc::new(ExternHandleData {
                    id: 1,
                    drop_fn: noop_drop,
                    type_name: "Widget",
                    to_string_fn: custom_fmt,
                })))
            }),
        );
        let src = r#"
extern type Widget;
extern fn make_widget() -> Widget;
fn main() {
    let w = make_widget();
    println(w);
}"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "MyWidget\n");
    }

    #[test]
    fn extern_handle_to_string_fallback() {
        fn fallback_fmt(_id: u64) -> String {
            "<Window>".to_string()
        }
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_window".to_string(),
            Box::new(|_args| {
                Ok(Value::ExternHandle(ManagedRc::new(ExternHandleData {
                    id: 1,
                    drop_fn: noop_drop,
                    type_name: "Window",
                    to_string_fn: fallback_fmt,
                })))
            }),
        );
        let src = r#"
extern type Window;
extern fn make_window() -> Window;
fn main() {
    let w = make_window();
    println(w);
}"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "<Window>\n");
    }

    #[test]
    fn ownership_rc_reduction_benchmark() {
        use super::managed_rc::{rc_dec_count, rc_inc_count, reset_rc_counts};

        let source = r#"
            struct Point { x: int, y: int }

            fn make_point(x: int, y: int) -> Point {
                Point { x: x, y: y }
            }

            fn sum_point(p: Point) -> int {
                p.x + p.y
            }

            fn transform(p: Point) -> Point {
                let x = p.x * 2;
                let y = p.y + 1;
                make_point(x, y)
            }

            fn main() {
                var total = 0;
                var i = 0;
                while i < 100 {
                    let p = make_point(i, i + 1);
                    let q = transform(p);
                    total = total + sum_point(q);
                    i = i + 1;
                }
                println(total);
            }
        "#;

        reset_rc_counts();
        let out = vm_ok(source);
        let incs = rc_inc_count();
        let decs = rc_dec_count();

        eprintln!("RC increments: {incs}, RC decrements: {decs}");

        assert!(!out.is_empty());
    }

    #[test]
    fn extern_handle_to_string_in_list() {
        fn item_fmt(_id: u64) -> String {
            "Item".to_string()
        }
        let mut externs: HashMap<String, ExternHandler> = HashMap::new();
        externs.insert(
            "make_item".to_string(),
            Box::new(|_args| {
                Ok(Value::ExternHandle(ManagedRc::new(ExternHandleData {
                    id: 1,
                    drop_fn: noop_drop,
                    type_name: "Item",
                    to_string_fn: item_fmt,
                })))
            }),
        );
        let src = r#"
extern type Item;
extern fn make_item() -> Item;
fn main() {
    let items = [make_item(), make_item()];
    println(items);
}"#;
        let out = vm_ok_with_externs(src, externs);
        assert_eq!(out, "[Item, Item]\n");
    }
}
