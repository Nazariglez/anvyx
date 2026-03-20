use anvyx_lang::{ExternDecl, ExternTypeDecl, ManagedRc, Value, export_fn, export_type};

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
    let result = handler(vec![Value::String(ManagedRc::new("world".to_string()))]).unwrap();
    assert_eq!(result, Value::String(ManagedRc::new("hi world".to_string())));
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

    let result = externs["hello"](vec![Value::String(ManagedRc::new("world".to_string()))]).unwrap();
    assert_eq!(result, Value::String(ManagedRc::new("hello world".to_string())));
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

// -- Value passthrough tests --

#[export_fn]
fn identity(v: Value) -> Value {
    v
}

#[test]
fn export_fn_value_passthrough() {
    let (name, handler) = __anvyx_export_identity();
    assert_eq!(name, "identity");
    let result = handler(vec![Value::Int(42)]).unwrap();
    assert_eq!(result, Value::Int(42));
    let result = handler(vec![Value::Bool(true)]).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn export_fn_value_passthrough_decl() {
    let decl: ExternDecl = __ANVYX_DECL_IDENTITY;
    assert_eq!(decl.name, "identity");
    assert_eq!(decl.params, &[("v", "any")]);
    assert_eq!(decl.ret, "any");
}

// -- ret annotation tests --

#[export_fn(ret = "[K: V]")]
fn make_map() -> Value {
    Value::Nil
}

#[test]
fn export_fn_ret_annotation_decl() {
    let decl: ExternDecl = __ANVYX_DECL_MAKE_MAP;
    assert_eq!(decl.name, "make_map");
    assert_eq!(decl.params, &[]);
    assert_eq!(decl.ret, "[K: V]");
}

#[test]
fn export_fn_ret_annotation_handler() {
    let (_, handler) = __anvyx_export_make_map();
    let result = handler(vec![]).unwrap();
    assert_eq!(result, Value::Nil);
}

// -- params annotation tests --

#[export_fn(params(list = "[T]"), ret = "int")]
fn list_length(list: Value) -> i64 {
    match list {
        Value::List(l) => l.len() as i64,
        _ => 0,
    }
}

#[test]
fn export_fn_params_annotation_decl() {
    let decl: ExternDecl = __ANVYX_DECL_LIST_LENGTH;
    assert_eq!(decl.name, "list_length");
    assert_eq!(decl.params, &[("list", "[T]")]);
    assert_eq!(decl.ret, "int");
}

#[test]
fn export_fn_params_annotation_handler() {
    let (_, handler) = __anvyx_export_list_length();
    let list = Value::List(ManagedRc::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));
    let result = handler(vec![list]).unwrap();
    assert_eq!(result, Value::Int(3));
}

// -- combined name + ret annotations --

#[export_fn(name = "ordered", ret = "[K: V]")]
fn create_ordered_map() -> Value {
    Value::Nil
}

#[test]
fn export_fn_name_and_ret_combined() {
    let decl: ExternDecl = __ANVYX_DECL_CREATE_ORDERED_MAP;
    assert_eq!(decl.name, "ordered");
    assert_eq!(decl.ret, "[K: V]");
}

// -- mixed Value and primitive params --

#[export_fn(params(data = "[T]"))]
fn mixed_params(x: i64, data: Value) -> bool {
    x > 0 && matches!(data, Value::List(_))
}

#[test]
fn export_fn_mixed_value_and_primitive_decl() {
    let decl: ExternDecl = __ANVYX_DECL_MIXED_PARAMS;
    assert_eq!(decl.params, &[("x", "int"), ("data", "[T]")]);
    assert_eq!(decl.ret, "bool");
}

#[test]
fn export_fn_mixed_value_and_primitive_handler() {
    let (_, handler) = __anvyx_export_mixed_params();
    let list = Value::List(ManagedRc::new(vec![]));
    let result = handler(vec![Value::Int(1), list]).unwrap();
    assert_eq!(result, Value::Bool(true));
}

// -- #[export_type] tests --

#[export_type(name = "Sprite")]
pub struct SpriteData {
    x: f64,
    y: f64,
}

#[test]
fn export_type_generates_decl_const() {
    let decl: ExternTypeDecl = __ANVYX_TYPE_DECL_SPRITEDATA;
    assert_eq!(decl.name, "Sprite");
}

#[test]
fn export_type_generates_store() {
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let mut store = s.borrow_mut();
        let id = store.insert(SpriteData { x: 1.0, y: 2.0 });
        assert_eq!(store.get(id).unwrap().x, 1.0);
        store.remove(id).unwrap();
    });
}

#[export_type(name = "Handle")]
pub struct OpaqueHandle;

#[test]
fn export_type_unit_struct() {
    let decl: ExternTypeDecl = __ANVYX_TYPE_DECL_OPAQUEHANDLE;
    assert_eq!(decl.name, "Handle");
}

#[export_type(name = "Color")]
pub struct Color(pub u8, pub u8, pub u8);

#[test]
fn export_type_tuple_struct() {
    let decl: ExternTypeDecl = __ANVYX_TYPE_DECL_COLOR;
    assert_eq!(decl.name, "Color");
    __ANVYX_STORE_COLOR.with(|s| {
        let mut store = s.borrow_mut();
        let id = store.insert(Color(255, 0, 128));
        assert_eq!(store.get(id).unwrap().0, 255);
    });
}

// -- provider! with types: tests --

mod typed_provider {
    use anvyx_lang::{export_fn, export_type};

    #[export_type(name = "Widget")]
    pub struct Widget {
        pub val: i64,
    }

    #[export_fn]
    pub fn make_val(x: i64) -> i64 {
        x + 1
    }

    anvyx_lang::provider!(types: [Widget], make_val);
}

#[test]
fn provider_types_populates_type_exports() {
    assert_eq!(typed_provider::ANVYX_TYPE_EXPORTS.len(), 1);
    assert_eq!(typed_provider::ANVYX_TYPE_EXPORTS[0].name, "Widget");
}

#[test]
fn provider_types_still_has_fn_exports() {
    assert_eq!(typed_provider::ANVYX_EXPORTS.len(), 1);
    assert_eq!(typed_provider::ANVYX_EXPORTS[0].name, "make_val");
}

#[test]
fn provider_types_handlers_work() {
    let externs = typed_provider::anvyx_externs();
    assert_eq!(externs.len(), 1);
    let result = externs["make_val"](vec![Value::Int(5)]).unwrap();
    assert_eq!(result, Value::Int(6));
}

mod types_only {
    use anvyx_lang::export_type;

    #[export_type(name = "Node")]
    pub struct Node {
        pub id: i64,
    }

    anvyx_lang::provider!(types: [Node]);
}

#[test]
fn provider_types_only() {
    assert_eq!(types_only::ANVYX_TYPE_EXPORTS.len(), 1);
    assert_eq!(types_only::ANVYX_TYPE_EXPORTS[0].name, "Node");
    assert!(types_only::ANVYX_EXPORTS.is_empty());
    assert!(types_only::anvyx_externs().is_empty());
}

mod qualified_type {
    pub mod inner {
        use anvyx_lang::export_type;

        #[export_type(name = "Inner")]
        pub struct InnerType {
            pub data: i64,
        }
    }

    use anvyx_lang::export_fn;

    #[export_fn]
    pub fn get_data() -> i64 {
        42
    }

    anvyx_lang::provider!(types: [inner::InnerType], get_data);
}

#[test]
fn provider_module_qualified_type() {
    assert_eq!(qualified_type::ANVYX_TYPE_EXPORTS.len(), 1);
    assert_eq!(qualified_type::ANVYX_TYPE_EXPORTS[0].name, "Inner");
}

mod multi_types {
    use anvyx_lang::export_type;

    #[export_type(name = "Texture")]
    pub struct Tex {
        pub w: i64,
    }

    #[export_type(name = "Shader")]
    pub struct Shd {
        pub id: i64,
    }

    anvyx_lang::provider!(types: [Tex, Shd]);
}

#[test]
fn provider_multiple_types() {
    assert_eq!(multi_types::ANVYX_TYPE_EXPORTS.len(), 2);
    let names: Vec<&str> = multi_types::ANVYX_TYPE_EXPORTS.iter().map(|d| d.name).collect();
    assert!(names.contains(&"Texture"));
    assert!(names.contains(&"Shader"));
}

#[test]
fn provider_no_types_backward_compat() {
    assert!(ANVYX_TYPE_EXPORTS.is_empty());
}

// -- #[export_fn] with extern types --

#[export_fn]
pub fn create_sprite(x: f64, y: f64) -> SpriteData {
    SpriteData { x, y }
}

#[export_fn]
pub fn sprite_x(s: &SpriteData) -> f64 {
    s.x
}

#[export_fn]
pub fn set_sprite_x(s: &mut SpriteData, x: f64) {
    s.x = x;
}

#[export_fn]
pub fn destroy_sprite(s: SpriteData) {
    let _ = s;
}

#[export_fn]
pub fn move_sprite(s: &mut SpriteData, dx: f64, dy: f64) {
    s.x += dx;
    s.y += dy;
}

// Step 5: extern type return

#[test]
fn export_fn_extern_type_return() {
    let (_, handler) = __anvyx_export_create_sprite();
    let result = handler(vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
    let Value::ExternHandle(id) = result else {
        panic!("expected ExternHandle");
    };
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let store = s.borrow();
        let sprite = store.get(id).unwrap();
        assert_eq!(sprite.x, 1.0);
        assert_eq!(sprite.y, 2.0);
    });
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id).unwrap();
    });
}

#[test]
fn export_fn_extern_type_return_decl() {
    let decl: ExternDecl = __ANVYX_DECL_CREATE_SPRITE;
    assert_eq!(decl.name, "create_sprite");
    assert_eq!(decl.params, &[("x", "float"), ("y", "float")]);
    assert_eq!(decl.ret, "Sprite");
}

// Step 6: extern ref param (&T)

#[test]
fn export_fn_extern_ref_param() {
    let id = __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().insert(SpriteData { x: 5.0, y: 10.0 })
    });
    let (_, handler) = __anvyx_export_sprite_x();
    let result = handler(vec![Value::ExternHandle(id)]).unwrap();
    assert_eq!(result, Value::Float(5.0));
    // Sprite still in store (not consumed)
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        assert!(s.borrow().get(id).is_ok());
    });
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id).unwrap();
    });
}

#[test]
fn export_fn_extern_ref_param_decl() {
    let decl: ExternDecl = __ANVYX_DECL_SPRITE_X;
    assert_eq!(decl.name, "sprite_x");
    assert_eq!(decl.params, &[("s", "Sprite")]);
    assert_eq!(decl.ret, "float");
}

// Step 7: extern mutable ref param (&mut T)

#[test]
fn export_fn_extern_mut_ref_param() {
    let id = __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().insert(SpriteData { x: 1.0, y: 2.0 })
    });
    let (_, handler) = __anvyx_export_set_sprite_x();
    handler(vec![Value::ExternHandle(id), Value::Float(99.0)]).unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let store = s.borrow();
        let sprite = store.get(id).unwrap();
        assert_eq!(sprite.x, 99.0);
        assert_eq!(sprite.y, 2.0);
    });
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id).unwrap();
    });
}

#[test]
fn export_fn_extern_mut_ref_param_decl() {
    let decl: ExternDecl = __ANVYX_DECL_SET_SPRITE_X;
    assert_eq!(decl.name, "set_sprite_x");
    assert_eq!(decl.params, &[("s", "Sprite"), ("x", "float")]);
    assert_eq!(decl.ret, "void");
}

// Step 8: extern owned param (consuming T)

#[test]
fn export_fn_extern_owned_param() {
    let id = __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().insert(SpriteData { x: 1.0, y: 2.0 })
    });
    let (_, handler) = __anvyx_export_destroy_sprite();
    handler(vec![Value::ExternHandle(id)]).unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        assert!(s.borrow().get(id).is_err());
    });
}

#[test]
fn export_fn_extern_owned_param_decl() {
    let decl: ExternDecl = __ANVYX_DECL_DESTROY_SPRITE;
    assert_eq!(decl.name, "destroy_sprite");
    assert_eq!(decl.params, &[("s", "Sprite")]);
    assert_eq!(decl.ret, "void");
}

// Step 9: mixed + error + round-trip

#[test]
fn export_fn_mixed_extern_and_primitive() {
    let id = __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().insert(SpriteData { x: 1.0, y: 2.0 })
    });
    let (_, handler) = __anvyx_export_move_sprite();
    handler(vec![Value::ExternHandle(id), Value::Float(10.0), Value::Float(20.0)]).unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let store = s.borrow();
        let sprite = store.get(id).unwrap();
        assert_eq!(sprite.x, 11.0);
        assert_eq!(sprite.y, 22.0);
    });
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id).unwrap();
    });
}

#[test]
fn export_fn_mixed_extern_and_primitive_decl() {
    let decl: ExternDecl = __ANVYX_DECL_MOVE_SPRITE;
    assert_eq!(decl.name, "move_sprite");
    assert_eq!(decl.params, &[("s", "Sprite"), ("dx", "float"), ("dy", "float")]);
    assert_eq!(decl.ret, "void");
}

#[test]
fn export_fn_extern_ref_invalid_handle() {
    let (_, handler) = __anvyx_export_sprite_x();
    let result = handler(vec![Value::ExternHandle(99999)]);
    assert!(result.is_err());
}

#[test]
fn export_fn_extern_ref_wrong_type() {
    let (_, handler) = __anvyx_export_sprite_x();
    let result = handler(vec![Value::Int(42)]);
    assert!(result.is_err());
}

#[test]
fn export_fn_extern_type_full_round_trip() {
    let (_, create) = __anvyx_export_create_sprite();
    let (_, get_x) = __anvyx_export_sprite_x();
    let (_, set_x) = __anvyx_export_set_sprite_x();
    let (_, destroy) = __anvyx_export_destroy_sprite();

    // Create
    let result = create(vec![Value::Float(10.0), Value::Float(20.0)]).unwrap();
    let Value::ExternHandle(id) = result else {
        panic!("expected ExternHandle");
    };

    // Read
    let x = get_x(vec![Value::ExternHandle(id)]).unwrap();
    assert_eq!(x, Value::Float(10.0));

    // Mutate
    set_x(vec![Value::ExternHandle(id), Value::Float(99.0)]).unwrap();

    // Read again
    let x = get_x(vec![Value::ExternHandle(id)]).unwrap();
    assert_eq!(x, Value::Float(99.0));

    // Destroy
    destroy(vec![Value::ExternHandle(id)]).unwrap();

    // Verify gone
    let result = get_x(vec![Value::ExternHandle(id)]);
    assert!(result.is_err());
}

// Step 10: provider integration with extern types

mod extern_type_provider {
    use anvyx_lang::{export_fn, export_type};

    #[export_type(name = "Widget")]
    pub struct Widget {
        pub val: i64,
    }

    #[export_fn]
    pub fn create_widget(val: i64) -> Widget {
        Widget { val }
    }

    #[export_fn]
    pub fn widget_val(w: &Widget) -> i64 {
        w.val
    }

    #[export_fn]
    pub fn destroy_widget(w: Widget) {
        let _ = w;
    }

    anvyx_lang::provider!(types: [Widget], create_widget, widget_val, destroy_widget);
}

#[test]
fn provider_extern_type_decls() {
    assert_eq!(extern_type_provider::ANVYX_TYPE_EXPORTS.len(), 1);
    assert_eq!(extern_type_provider::ANVYX_TYPE_EXPORTS[0].name, "Widget");

    assert_eq!(extern_type_provider::ANVYX_EXPORTS.len(), 3);

    let create = extern_type_provider::ANVYX_EXPORTS
        .iter()
        .find(|d| d.name == "create_widget")
        .unwrap();
    assert_eq!(create.params, &[("val", "int")]);
    assert_eq!(create.ret, "Widget");

    let get_val = extern_type_provider::ANVYX_EXPORTS
        .iter()
        .find(|d| d.name == "widget_val")
        .unwrap();
    assert_eq!(get_val.params, &[("w", "Widget")]);
    assert_eq!(get_val.ret, "int");

    let destroy = extern_type_provider::ANVYX_EXPORTS
        .iter()
        .find(|d| d.name == "destroy_widget")
        .unwrap();
    assert_eq!(destroy.params, &[("w", "Widget")]);
    assert_eq!(destroy.ret, "void");
}

#[test]
fn provider_extern_type_handlers_work() {
    let externs = extern_type_provider::anvyx_externs();

    // Create
    let result = externs["create_widget"](vec![Value::Int(42)]).unwrap();
    let Value::ExternHandle(id) = result else {
        panic!("expected ExternHandle");
    };

    // Read
    let val = externs["widget_val"](vec![Value::ExternHandle(id)]).unwrap();
    assert_eq!(val, Value::Int(42));

    // Destroy
    externs["destroy_widget"](vec![Value::ExternHandle(id)]).unwrap();

    // Verify gone
    let result = externs["widget_val"](vec![Value::ExternHandle(id)]);
    assert!(result.is_err());
}
