use anvyx_lang::{ExternDecl, Value, export_fn};
use std::rc::Rc;

#[export_fn]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[test]
fn export_fn_generates_companion() {
    let (name, handler) = __anvyx_export_add();
    assert_eq!(name, "add");
    let result = handler(vec![Value::Int(3), Value::Int(4)]).unwrap();
    assert_eq!(result, Value::Int(7));
}

#[export_fn(name = "custom_name")]
fn my_fn(x: i64) -> i64 {
    x * 2
}

#[test]
fn export_fn_name_override() {
    let (name, _) = __anvyx_export_my_fn();
    assert_eq!(name, "custom_name");
}

#[export_fn]
fn greet(name: String) -> String {
    format!("hi {name}")
}

#[test]
fn export_fn_string_params() {
    let (_, handler) = __anvyx_export_greet();
    let result = handler(vec![Value::String(Rc::from("world"))]).unwrap();
    assert_eq!(result, Value::String(Rc::from("hi world")));
}

#[export_fn]
fn noop() {}

#[test]
fn export_fn_void_return() {
    let (_, handler) = __anvyx_export_noop();
    let result = handler(vec![]).unwrap();
    assert_eq!(result, Value::Nil);
}

#[test]
fn export_fn_wrong_type_returns_error() {
    let (_, handler) = __anvyx_export_add();
    let result = handler(vec![Value::Bool(true), Value::Int(1)]);
    assert!(result.is_err());
}

#[export_fn]
fn scale(x: f64, factor: f64) -> f64 {
    x * factor
}

#[test]
fn export_fn_float_params() {
    let (name, handler) = __anvyx_export_scale();
    assert_eq!(name, "scale");
    let result = handler(vec![Value::Float(2.5), Value::Float(4.0)]).unwrap();
    assert_eq!(result, Value::Float(10.0));
}

#[export_fn]
fn toggle(flag: bool) -> bool {
    !flag
}

#[test]
fn export_fn_bool_params() {
    let (name, handler) = __anvyx_export_toggle();
    assert_eq!(name, "toggle");
    let result = handler(vec![Value::Bool(true)]).unwrap();
    assert_eq!(result, Value::Bool(false));
}

// -- provider! tests --

mod math_mod {
    use anvyx_lang::export_fn;

    #[export_fn]
    pub fn double(x: i64) -> i64 {
        x * 2
    }

    #[export_fn(name = "triple")]
    pub fn triple_val(x: i64) -> i64 {
        x * 3
    }
}

mod greet_mod {
    use anvyx_lang::export_fn;

    #[export_fn]
    pub fn hello(name: String) -> String {
        format!("hello {name}")
    }
}

// provider! with module-qualified paths
anvyx_lang::provider!(math_mod::double, math_mod::triple_val, greet_mod::hello);

#[test]
fn provider_generates_anvyx_externs() {
    let externs = anvyx_externs();
    assert_eq!(externs.len(), 3);
    assert!(externs.contains_key("double"));
    assert!(externs.contains_key("triple"));
    assert!(externs.contains_key("hello"));
}

#[test]
fn provider_handlers_work_correctly() {
    let externs = anvyx_externs();

    let result = externs["double"](vec![Value::Int(5)]).unwrap();
    assert_eq!(result, Value::Int(10));

    let result = externs["triple"](vec![Value::Int(4)]).unwrap();
    assert_eq!(result, Value::Int(12));

    let result = externs["hello"](vec![Value::String(Rc::from("world"))]).unwrap();
    assert_eq!(result, Value::String(Rc::from("hello world")));
}

// provider! with bare idents (no module prefix) — functions defined at the same scope
mod flat_mod {
    use anvyx_lang::export_fn;

    #[export_fn]
    pub fn inc(x: i64) -> i64 {
        x + 1
    }

    #[export_fn]
    pub fn dec(x: i64) -> i64 {
        x - 1
    }

    anvyx_lang::provider!(inc, dec);
}

#[test]
fn provider_bare_ident() {
    let externs = flat_mod::anvyx_externs();
    assert_eq!(externs.len(), 2);

    let result = externs["inc"](vec![Value::Int(5)]).unwrap();
    assert_eq!(result, Value::Int(6));

    let result = externs["dec"](vec![Value::Int(5)]).unwrap();
    assert_eq!(result, Value::Int(4));
}

#[test]
fn provider_trailing_comma() {
    // Ensure trailing comma is accepted
    let externs = flat_mod::anvyx_externs();
    assert!(externs.contains_key("inc"));
}

// -- ExternDecl metadata tests --

#[test]
fn export_fn_generates_decl_const() {
    let decl: ExternDecl = __ANVYX_DECL_ADD;
    assert_eq!(decl.name, "add");
    assert_eq!(decl.params, &[("a", "int"), ("b", "int")]);
    assert_eq!(decl.ret, "int");
}

#[test]
fn export_fn_name_override_in_decl() {
    let decl: ExternDecl = __ANVYX_DECL_MY_FN;
    assert_eq!(decl.name, "custom_name");
    assert_eq!(decl.params, &[("x", "int")]);
    assert_eq!(decl.ret, "int");
}

#[test]
fn export_fn_string_param_decl() {
    let decl: ExternDecl = __ANVYX_DECL_GREET;
    assert_eq!(decl.name, "greet");
    assert_eq!(decl.params, &[("name", "string")]);
    assert_eq!(decl.ret, "string");
}

#[test]
fn export_fn_void_return_decl() {
    let decl: ExternDecl = __ANVYX_DECL_NOOP;
    assert_eq!(decl.name, "noop");
    assert_eq!(decl.params, &[]);
    assert_eq!(decl.ret, "void");
}

#[test]
fn export_fn_float_param_decl() {
    let decl: ExternDecl = __ANVYX_DECL_SCALE;
    assert_eq!(decl.name, "scale");
    assert_eq!(decl.params, &[("x", "float"), ("factor", "float")]);
    assert_eq!(decl.ret, "float");
}

#[test]
fn export_fn_bool_param_decl() {
    let decl: ExternDecl = __ANVYX_DECL_TOGGLE;
    assert_eq!(decl.name, "toggle");
    assert_eq!(decl.params, &[("flag", "bool")]);
    assert_eq!(decl.ret, "bool");
}

#[test]
fn provider_generates_anvyx_exports() {
    assert_eq!(ANVYX_EXPORTS.len(), 3);
    let names: Vec<&str> = ANVYX_EXPORTS.iter().map(|d| d.name).collect();
    assert!(names.contains(&"double"));
    assert!(names.contains(&"triple"));
    assert!(names.contains(&"hello"));
}

#[test]
fn provider_exports_name_override() {
    let triple = ANVYX_EXPORTS.iter().find(|d| d.name == "triple").unwrap();
    assert_eq!(triple.params, &[("x", "int")]);
    assert_eq!(triple.ret, "int");
}

#[test]
fn provider_exports_correct_types() {
    let double = ANVYX_EXPORTS.iter().find(|d| d.name == "double").unwrap();
    assert_eq!(double.params, &[("x", "int")]);
    assert_eq!(double.ret, "int");

    let hello = ANVYX_EXPORTS.iter().find(|d| d.name == "hello").unwrap();
    assert_eq!(hello.params, &[("name", "string")]);
    assert_eq!(hello.ret, "string");
}

#[test]
fn provider_bare_ident_exports() {
    assert_eq!(flat_mod::ANVYX_EXPORTS.len(), 2);
    let names: Vec<&str> = flat_mod::ANVYX_EXPORTS.iter().map(|d| d.name).collect();
    assert!(names.contains(&"inc"));
    assert!(names.contains(&"dec"));

    let inc = flat_mod::ANVYX_EXPORTS.iter().find(|d| d.name == "inc").unwrap();
    assert_eq!(inc.params, &[("x", "int")]);
    assert_eq!(inc.ret, "int");
}
