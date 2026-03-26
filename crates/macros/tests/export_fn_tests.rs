use anvyx_lang::{
    AnvyxExternType, ExternDecl, ExternHandleData, ExternTypeDeclConst, ManagedRc, Value,
    export_fn, export_type,
};

fn noop_drop(_id: u64) {}

fn extern_handle(id: u64) -> Value {
    Value::ExternHandle(ManagedRc::new(ExternHandleData {
        id,
        drop_fn: noop_drop,
        type_name: "Extern",
        to_string_fn: |_| "<Extern>".to_string(),
    }))
}

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
    assert_eq!(
        result,
        Value::String(ManagedRc::new("hi world".to_string()))
    );
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
fn scale(x: f32, factor: f32) -> f32 {
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

    let result =
        externs["hello"](vec![Value::String(ManagedRc::new("world".to_string()))]).unwrap();
    assert_eq!(
        result,
        Value::String(ManagedRc::new("hello world".to_string()))
    );
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

    let inc = flat_mod::ANVYX_EXPORTS
        .iter()
        .find(|d| d.name == "inc")
        .unwrap();
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
    let list = Value::List(ManagedRc::new(vec![
        Value::Int(1),
        Value::Int(2),
        Value::Int(3),
    ]));
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
    x: f32,
    y: f32,
}

#[test]
fn export_type_generates_decl_const() {
    let decl: ExternTypeDeclConst = __ANVYX_TYPE_DECL_SPRITEDATA;
    assert_eq!(decl.name, "Sprite");
}

#[test]
fn export_type_generates_store() {
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let mut store = s.borrow_mut();
        let id = store.insert(SpriteData { x: 1.0, y: 2.0 });
        assert_eq!(store.borrow(id).unwrap().x, 1.0);
        store.remove(id).unwrap();
    });
}

#[export_type(name = "Handle")]
pub struct OpaqueHandle;

#[test]
fn export_type_unit_struct() {
    let decl: ExternTypeDeclConst = __ANVYX_TYPE_DECL_OPAQUEHANDLE;
    assert_eq!(decl.name, "Handle");
}

#[export_type(name = "Color")]
pub struct Color(pub u8, pub u8, pub u8);

#[test]
fn export_type_tuple_struct() {
    let decl: ExternTypeDeclConst = __ANVYX_TYPE_DECL_COLOR;
    assert_eq!(decl.name, "Color");
    __ANVYX_STORE_COLOR.with(|s| {
        let mut store = s.borrow_mut();
        let id = store.insert(Color(255, 0, 128));
        assert_eq!(store.borrow(id).unwrap().0, 255);
    });
}

// -- to_string_fn display tests --

#[export_type(name = "FmtVec2")]
pub struct TestVec2 {
    pub x: f32,
    pub y: f32,
}

impl std::fmt::Display for TestVec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Vec2({}, {})", self.x as i64, self.y as i64)
    }
}

#[test]
fn export_type_with_display_to_string() {
    let id = TestVec2::with_store(|s| s.borrow_mut().insert(TestVec2 { x: 1.0, y: 2.0 }));
    let result = TestVec2::to_display(id);
    assert_eq!(result, "Vec2(1, 2)");
    TestVec2::cleanup(id);
}

#[test]
fn export_type_without_display_to_string() {
    let id = OpaqueHandle::with_store(|s| s.borrow_mut().insert(OpaqueHandle));
    let result = OpaqueHandle::to_display(id);
    assert_eq!(result, "<Handle>");
    OpaqueHandle::cleanup(id);
}

// -- provider! with types: tests --

mod typed_provider {
    use anvyx_lang::{export_fn, export_methods, export_type};

    #[export_type(name = "Widget")]
    pub struct Widget {
        pub val: i64,
    }

    #[export_methods]
    impl Widget {}

    #[export_fn]
    pub fn make_val(x: i64) -> i64 {
        x + 1
    }

    anvyx_lang::provider!(types: [Widget], make_val);
}

#[test]
fn provider_types_populates_type_exports() {
    let types = typed_provider::anvyx_type_exports();
    assert_eq!(types.len(), 1);
    assert_eq!(types[0].name, "Widget");
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
    use anvyx_lang::{export_methods, export_type};

    #[export_type(name = "Node")]
    pub struct Node {
        pub id: i64,
    }

    #[export_methods]
    impl Node {}

    anvyx_lang::provider!(types: [Node]);
}

#[test]
fn provider_types_only() {
    let types = types_only::anvyx_type_exports();
    assert_eq!(types.len(), 1);
    assert_eq!(types[0].name, "Node");
    assert!(types_only::ANVYX_EXPORTS.is_empty());
    assert!(types_only::anvyx_externs().is_empty());
}

mod qualified_type {
    pub mod inner {
        use anvyx_lang::{export_methods, export_type};

        #[export_type(name = "Inner")]
        pub struct InnerType {
            pub data: i64,
        }

        #[export_methods]
        impl InnerType {}
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
    let types = qualified_type::anvyx_type_exports();
    assert_eq!(types.len(), 1);
    assert_eq!(types[0].name, "Inner");
}

mod multi_types {
    use anvyx_lang::{export_methods, export_type};

    #[export_type(name = "Texture")]
    pub struct Tex {
        pub w: i64,
    }

    #[export_methods]
    impl Tex {}

    #[export_type(name = "Shader")]
    pub struct Shd {
        pub id: i64,
    }

    #[export_methods]
    impl Shd {}

    anvyx_lang::provider!(types: [Tex, Shd]);
}

#[test]
fn provider_multiple_types() {
    let types = multi_types::anvyx_type_exports();
    assert_eq!(types.len(), 2);
    let names: Vec<&str> = types.iter().map(|d| d.name).collect();
    assert!(names.contains(&"Texture"));
    assert!(names.contains(&"Shader"));
}

#[test]
fn provider_no_types_backward_compat() {
    assert!(anvyx_type_exports().is_empty());
}

// -- #[export_fn] with extern types --

#[export_fn]
pub fn create_sprite(x: f32, y: f32) -> SpriteData {
    SpriteData { x, y }
}

#[export_fn]
pub fn sprite_x(s: &SpriteData) -> f32 {
    s.x
}

#[export_fn]
pub fn set_sprite_x(s: &mut SpriteData, x: f32) {
    s.x = x;
}

#[export_fn]
pub fn destroy_sprite(s: SpriteData) {
    let _ = s;
}

#[export_fn]
pub fn move_sprite(s: &mut SpriteData, dx: f32, dy: f32) {
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
        let sprite = store.borrow(id.id).unwrap();
        assert_eq!(sprite.x, 1.0);
        assert_eq!(sprite.y, 2.0);
    });
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id.id).unwrap();
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
    let id =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 5.0, y: 10.0 }));
    let (_, handler) = __anvyx_export_sprite_x();
    let result = handler(vec![extern_handle(id)]).unwrap();
    assert_eq!(result, Value::Float(5.0));
    // Sprite still in store (not consumed)
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        assert!(s.borrow().borrow(id).is_ok());
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
    let id =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 1.0, y: 2.0 }));
    let (_, handler) = __anvyx_export_set_sprite_x();
    handler(vec![extern_handle(id), Value::Float(99.0)]).unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let store = s.borrow();
        let sprite = store.borrow(id).unwrap();
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
    let id =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 1.0, y: 2.0 }));
    let (_, handler) = __anvyx_export_destroy_sprite();
    handler(vec![extern_handle(id)]).unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        assert!(s.borrow().borrow(id).is_err());
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
    let id =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 1.0, y: 2.0 }));
    let (_, handler) = __anvyx_export_move_sprite();
    handler(vec![
        extern_handle(id),
        Value::Float(10.0),
        Value::Float(20.0),
    ])
    .unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let store = s.borrow();
        let sprite = store.borrow(id).unwrap();
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
    assert_eq!(
        decl.params,
        &[("s", "Sprite"), ("dx", "float"), ("dy", "float")]
    );
    assert_eq!(decl.ret, "void");
}

#[test]
fn export_fn_extern_ref_invalid_handle() {
    let (_, handler) = __anvyx_export_sprite_x();
    let result = handler(vec![extern_handle(99999)]);
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
    let x = get_x(vec![Value::ExternHandle(id.clone())]).unwrap();
    assert_eq!(x, Value::Float(10.0));

    // Mutate
    set_x(vec![Value::ExternHandle(id.clone()), Value::Float(99.0)]).unwrap();

    // Read again
    let x = get_x(vec![Value::ExternHandle(id.clone())]).unwrap();
    assert_eq!(x, Value::Float(99.0));

    // Destroy
    destroy(vec![Value::ExternHandle(id.clone())]).unwrap();

    // Verify gone
    let result = get_x(vec![Value::ExternHandle(id)]);
    assert!(result.is_err());
}

// Step 10: provider integration with extern types

mod extern_type_provider {
    use anvyx_lang::{export_fn, export_methods, export_type};

    #[export_type(name = "Widget")]
    pub struct Widget {
        pub val: i64,
    }

    #[export_methods]
    impl Widget {}

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
    let types = extern_type_provider::anvyx_type_exports();
    assert_eq!(types.len(), 1);
    assert_eq!(types[0].name, "Widget");

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
    let val = externs["widget_val"](vec![Value::ExternHandle(id.clone())]).unwrap();
    assert_eq!(val, Value::Int(42));

    // Destroy
    externs["destroy_widget"](vec![Value::ExternHandle(id.clone())]).unwrap();

    // Verify gone
    let result = externs["widget_val"](vec![Value::ExternHandle(id)]);
    assert!(result.is_err());
}

// -- same-store multi-borrow tests --

#[export_fn]
pub fn distance_sprites(a: &SpriteData, b: &SpriteData) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

#[export_fn]
pub fn move_sprite_towards(s: &mut SpriteData, target: &SpriteData) {
    s.x = target.x;
    s.y = target.y;
}

#[test]
fn export_fn_same_store_two_immutable_refs() {
    let id_a =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 0.0, y: 0.0 }));
    let id_b =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 3.0, y: 4.0 }));
    let (_, handler) = __anvyx_export_distance_sprites();
    let result = handler(vec![extern_handle(id_a), extern_handle(id_b)]).unwrap();
    assert_eq!(result, Value::Float(5.0));
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id_a).unwrap();
        s.borrow_mut().remove(id_b).unwrap();
    });
}

#[test]
fn export_fn_same_store_mut_and_immut_refs() {
    let id_a =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 0.0, y: 0.0 }));
    let id_b =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 10.0, y: 20.0 }));
    let (_, handler) = __anvyx_export_move_sprite_towards();
    handler(vec![extern_handle(id_a), extern_handle(id_b)]).unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        let store = s.borrow();
        let sprite = store.borrow(id_a).unwrap();
        assert_eq!(sprite.x, 10.0);
        assert_eq!(sprite.y, 20.0);
    });
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id_a).unwrap();
        s.borrow_mut().remove(id_b).unwrap();
    });
}

#[test]
fn export_fn_same_store_same_handle_mut_fails() {
    let id =
        __ANVYX_STORE_SPRITEDATA.with(|s| s.borrow_mut().insert(SpriteData { x: 1.0, y: 2.0 }));
    let (_, handler) = __anvyx_export_move_sprite_towards();
    let result = handler(vec![extern_handle(id), extern_handle(id)]);
    assert!(result.is_err());
    __ANVYX_STORE_SPRITEDATA.with(|s| {
        s.borrow_mut().remove(id).unwrap();
    });
}

// -- cleanup integration tests --

#[test]
fn cleanup_loop_no_leak() {
    let (_, create) = __anvyx_export_create_sprite();
    for i in 0..100 {
        let _ = create(vec![Value::Float(i as f32), Value::Float(0.0_f32)]).unwrap();
    }
    __ANVYX_STORE_SPRITEDATA.with(|s| assert_eq!(s.borrow().len(), 0));
}

#[test]
fn cleanup_explicit_destroy_then_drop_no_panic() {
    let (_, create) = __anvyx_export_create_sprite();
    let (_, destroy) = __anvyx_export_destroy_sprite();

    let result = create(vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
    let Value::ExternHandle(handle) = result else {
        panic!("expected ExternHandle")
    };
    let lingering = handle.clone();

    destroy(vec![Value::ExternHandle(handle)]).unwrap();
    __ANVYX_STORE_SPRITEDATA.with(|s| assert_eq!(s.borrow().len(), 0));

    drop(lingering);
    // drop_fn fires again -> store.remove returns Err -> silently ignored, no panic
    __ANVYX_STORE_SPRITEDATA.with(|s| assert_eq!(s.borrow().len(), 0));
}

// -- #[export_methods] tests --

mod method_tests {
    use super::extern_handle;
    use anvyx_lang::{
        ExternMethodDecl, ExternStaticMethodDecl, Value, export_methods, export_type,
    };

    #[export_type(name = "Vec2")]
    pub struct Vec2 {
        pub x: f32,
        pub y: f32,
    }

    #[export_type(name = "Color")]
    pub struct Color {
        pub r: f32,
        pub g: f32,
        pub b: f32,
    }

    #[export_methods]
    impl Vec2 {
        pub fn new(x: f32, y: f32) -> Vec2 {
            Vec2 { x, y }
        }
        pub fn zero() -> Vec2 {
            Vec2 { x: 0.0, y: 0.0 }
        }
        pub fn get_x(&self) -> f32 {
            self.x
        }
        pub fn get_y(&self) -> f32 {
            self.y
        }
        pub fn length(&self) -> f32 {
            (self.x * self.x + self.y * self.y).sqrt()
        }
        pub fn move_by(&mut self, dx: f32, dy: f32) {
            self.x += dx;
            self.y += dy;
        }
        pub fn set_x(&mut self, x: f32) {
            self.x = x;
        }
        pub fn scaled(&self, factor: f32) -> Vec2 {
            Vec2 {
                x: self.x * factor,
                y: self.y * factor,
            }
        }
        pub fn negated(&self) -> Vec2 {
            Vec2 {
                x: -self.x,
                y: -self.y,
            }
        }
        pub fn dot(&self, other: &Vec2) -> f32 {
            self.x * other.x + self.y * other.y
        }
        pub fn add_assign(&mut self, other: &Vec2) {
            self.x += other.x;
            self.y += other.y;
        }
        pub fn color_brightness(&self, c: &Color) -> f32 {
            self.length() * (c.r + c.g + c.b) / 3.0
        }
        pub fn tinted_scale(&self, c: &Color) -> Vec2 {
            let factor = (c.r + c.g + c.b) / 3.0;
            Vec2 {
                x: self.x * factor,
                y: self.y * factor,
            }
        }
        pub fn apply_color(&self, c: Color) -> f32 {
            self.x * c.r + self.y * c.g
        }
        pub fn with_value(&self, v: Value) -> Value {
            match v {
                Value::Float(f) => Value::Float(f + self.x as f32),
                other => other,
            }
        }
        pub fn from_color(c: &Color) -> Vec2 {
            Vec2 { x: c.r, y: c.g }
        }
        pub fn from_color_owned(c: Color) -> Vec2 {
            Vec2 { x: c.r, y: c.g }
        }
    }

    #[test]
    fn export_methods_static_handler_key() {
        let (key, _) = __anvyx_method_Vec2_new();
        assert_eq!(key, "Vec2::new");
    }

    #[test]
    fn export_methods_static_handler_creates_handle() {
        let (_, handler) = __anvyx_method_Vec2_new();
        let result = handler(vec![Value::Float(3.0), Value::Float(4.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(id.id).unwrap();
            assert_eq!(v.x, 3.0);
            assert_eq!(v.y, 4.0);
        });
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id.id).unwrap();
        });
    }

    #[test]
    fn export_methods_static_no_params() {
        let (_, handler) = __anvyx_method_Vec2_zero();
        let result = handler(vec![]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(id.id).unwrap();
            assert_eq!(v.x, 0.0);
            assert_eq!(v.y, 0.0);
        });
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id.id).unwrap();
        });
    }

    #[test]
    fn export_methods_borrow_handler_primitive() {
        let id = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 7.0, y: 8.0 }));
        let (_, handler) = __anvyx_method_Vec2_get_x();
        let result = handler(vec![extern_handle(id)]).unwrap();
        assert_eq!(result, Value::Float(7.0));
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
        });
    }

    #[test]
    fn export_methods_borrow_handler_length() {
        let id = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 3.0, y: 4.0 }));
        let (_, handler) = __anvyx_method_Vec2_length();
        let result = handler(vec![extern_handle(id)]).unwrap();
        assert_eq!(result, Value::Float(5.0));
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
        });
    }

    #[test]
    fn export_methods_mut_handler() {
        let id = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 1.0, y: 2.0 }));
        let (_, handler) = __anvyx_method_Vec2_move_by();
        handler(vec![
            extern_handle(id),
            Value::Float(10.0),
            Value::Float(20.0),
        ])
        .unwrap();
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(id).unwrap();
            assert_eq!(v.x, 11.0);
            assert_eq!(v.y, 22.0);
        });
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
        });
    }

    #[test]
    fn export_methods_borrow_returns_self() {
        let id = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 2.0, y: 3.0 }));
        let (_, handler) = __anvyx_method_Vec2_scaled();
        let result = handler(vec![extern_handle(id), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(new_id) = result else {
            panic!("expected ExternHandle")
        };
        assert_ne!(id, new_id.id);
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_id.id).unwrap();
            assert_eq!(v.x, 4.0);
            assert_eq!(v.y, 6.0);
        });
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
            s.borrow_mut().remove(new_id.id).unwrap();
        });
    }

    #[test]
    fn export_methods_borrow_returns_self_no_params() {
        let id = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 5.0, y: 6.0 }));
        let (_, handler) = __anvyx_method_Vec2_negated();
        let result = handler(vec![extern_handle(id)]).unwrap();
        let Value::ExternHandle(new_id) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_id.id).unwrap();
            assert_eq!(v.x, -5.0);
            assert_eq!(v.y, -6.0);
        });
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
            s.borrow_mut().remove(new_id.id).unwrap();
        });
    }

    #[test]
    fn export_methods_invalid_handle() {
        let (_, handler) = __anvyx_method_Vec2_get_x();
        let result = handler(vec![extern_handle(99999)]);
        assert!(result.is_err());
    }

    #[test]
    fn export_methods_wrong_type_for_self() {
        let (_, handler) = __anvyx_method_Vec2_get_x();
        let result = handler(vec![Value::Int(42)]);
        assert!(result.is_err());
    }

    #[test]
    fn export_methods_method_metadata() {
        let methods: &[ExternMethodDecl] = __ANVYX_METHODS_DECL_VEC2;
        let names: Vec<&str> = methods.iter().map(|m| m.name).collect();
        assert!(names.contains(&"get_x"));
        assert!(names.contains(&"get_y"));
        assert!(names.contains(&"length"));
        assert!(names.contains(&"move_by"));
        assert!(names.contains(&"set_x"));
        assert!(names.contains(&"scaled"));
        assert!(names.contains(&"negated"));

        let get_x = methods.iter().find(|m| m.name == "get_x").unwrap();
        assert_eq!(get_x.receiver, "self");
        assert_eq!(get_x.params, &[]);
        assert_eq!(get_x.ret, "float");

        let move_by = methods.iter().find(|m| m.name == "move_by").unwrap();
        assert_eq!(move_by.receiver, "var");
        assert_eq!(move_by.params, &[("dx", "float"), ("dy", "float")]);
        assert_eq!(move_by.ret, "void");

        let scaled = methods.iter().find(|m| m.name == "scaled").unwrap();
        assert_eq!(scaled.receiver, "self");
        assert_eq!(scaled.params, &[("factor", "float")]);
        assert_eq!(scaled.ret, "Vec2");
    }

    #[test]
    fn export_methods_static_metadata() {
        let statics: &[ExternStaticMethodDecl] = __ANVYX_STATICS_DECL_VEC2;
        let names: Vec<&str> = statics.iter().map(|s| s.name).collect();
        assert!(names.contains(&"new"));
        assert!(names.contains(&"zero"));

        let new_m = statics.iter().find(|s| s.name == "new").unwrap();
        assert_eq!(new_m.params, &[("x", "float"), ("y", "float")]);
        assert_eq!(new_m.ret, "Vec2");

        let zero_m = statics.iter().find(|s| s.name == "zero").unwrap();
        assert_eq!(zero_m.params, &[]);
        assert_eq!(zero_m.ret, "Vec2");
    }

    #[test]
    fn export_methods_convenience_function() {
        let handlers = __anvyx_methods_Vec2();
        let keys: Vec<&str> = handlers.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"Vec2::new"));
        assert!(keys.contains(&"Vec2::zero"));
        assert!(keys.contains(&"Vec2::get_x"));
        assert!(keys.contains(&"Vec2::get_y"));
        assert!(keys.contains(&"Vec2::length"));
        assert!(keys.contains(&"Vec2::move_by"));
        assert!(keys.contains(&"Vec2::set_x"));
        assert!(keys.contains(&"Vec2::scaled"));
        assert!(keys.contains(&"Vec2::negated"));
    }

    #[test]
    fn export_methods_full_round_trip() {
        let handler_map: std::collections::HashMap<&str, _> =
            __anvyx_methods_Vec2().into_iter().collect();

        let result = handler_map["Vec2::new"](vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle")
        };

        let x = handler_map["Vec2::get_x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(1.0));

        handler_map["Vec2::set_x"](vec![Value::ExternHandle(id.clone()), Value::Float(99.0)])
            .unwrap();

        let x = handler_map["Vec2::get_x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(99.0));

        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id.id).unwrap();
        });
    }

    #[test]
    fn export_methods_same_store_borrow() {
        let id_a = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 1.0, y: 2.0 }));
        let id_b = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 3.0, y: 4.0 }));
        let (_, handler) = __anvyx_method_Vec2_dot();
        let result = handler(vec![extern_handle(id_a), extern_handle(id_b)]).unwrap();
        assert_eq!(result, Value::Float(1.0 * 3.0 + 2.0 * 4.0));
        __ANVYX_STORE_VEC2.with(|s| {
            assert!(s.borrow().borrow(id_a).is_ok());
            assert!(s.borrow().borrow(id_b).is_ok());
            s.borrow_mut().remove(id_a).unwrap();
            s.borrow_mut().remove(id_b).unwrap();
        });
    }

    #[test]
    fn export_methods_same_store_mut_and_ref() {
        let id_a = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 1.0, y: 2.0 }));
        let id_b = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 10.0, y: 20.0 }));
        let (_, handler) = __anvyx_method_Vec2_add_assign();
        handler(vec![extern_handle(id_a), extern_handle(id_b)]).unwrap();
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(id_a).unwrap();
            assert_eq!(v.x, 11.0);
            assert_eq!(v.y, 22.0);
        });
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id_a).unwrap();
            s.borrow_mut().remove(id_b).unwrap();
        });
    }

    #[test]
    fn export_methods_same_store_same_handle_fails() {
        let id = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 1.0, y: 2.0 }));
        let (_, handler) = __anvyx_method_Vec2_add_assign();
        let result = handler(vec![extern_handle(id), extern_handle(id)]);
        assert!(result.is_err());
        __ANVYX_STORE_VEC2.with(|s| {
            assert!(s.borrow().borrow(id).is_ok());
            s.borrow_mut().remove(id).unwrap();
        });
    }

    #[test]
    fn export_methods_cross_store_borrow() {
        let id_v = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 3.0, y: 4.0 }));
        let id_c = __ANVYX_STORE_COLOR.with(|s| {
            s.borrow_mut().insert(Color {
                r: 0.5,
                g: 0.5,
                b: 0.5,
            })
        });
        let (_, handler) = __anvyx_method_Vec2_color_brightness();
        let result = handler(vec![extern_handle(id_v), extern_handle(id_c)]).unwrap();
        // length(3,4) = 5.0, (0.5+0.5+0.5)/3 = 0.5, product = 2.5
        assert_eq!(result, Value::Float(2.5));
        __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().remove(id_v).unwrap());
        __ANVYX_STORE_COLOR.with(|s| s.borrow_mut().remove(id_c).unwrap());
    }

    #[test]
    fn export_methods_cross_store_returns_self() {
        let id_v = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 2.0, y: 3.0 }));
        let id_c = __ANVYX_STORE_COLOR.with(|s| {
            s.borrow_mut().insert(Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            })
        });
        let (_, handler) = __anvyx_method_Vec2_tinted_scale();
        let result = handler(vec![extern_handle(id_v), extern_handle(id_c)]).unwrap();
        let Value::ExternHandle(new_id) = result else {
            panic!("expected ExternHandle");
        };
        assert_ne!(id_v, new_id.id);
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_id.id).unwrap();
            assert_eq!(v.x, 2.0);
            assert_eq!(v.y, 3.0);
        });
        __ANVYX_STORE_VEC2.with(|s| {
            s.borrow_mut().remove(id_v).unwrap();
            s.borrow_mut().remove(new_id.id).unwrap();
        });
        __ANVYX_STORE_COLOR.with(|s| s.borrow_mut().remove(id_c).unwrap());
    }

    #[test]
    fn export_methods_owned_param() {
        let id_v = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 2.0, y: 3.0 }));
        let id_c = __ANVYX_STORE_COLOR.with(|s| {
            s.borrow_mut().insert(Color {
                r: 0.5,
                g: 1.0,
                b: 0.0,
            })
        });
        let (_, handler) = __anvyx_method_Vec2_apply_color();
        let result = handler(vec![extern_handle(id_v), extern_handle(id_c)]).unwrap();
        // 2.0*0.5 + 3.0*1.0 = 4.0
        assert_eq!(result, Value::Float(4.0));
        __ANVYX_STORE_COLOR.with(|s| assert!(s.borrow().borrow(id_c).is_err()));
        __ANVYX_STORE_VEC2.with(|s| {
            assert!(s.borrow().borrow(id_v).is_ok());
            s.borrow_mut().remove(id_v).unwrap();
        });
    }

    #[test]
    fn export_methods_value_passthrough() {
        let id = __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().insert(Vec2 { x: 10.0, y: 0.0 }));
        let (_, handler) = __anvyx_method_Vec2_with_value();
        let result = handler(vec![extern_handle(id), Value::Float(5.0)]).unwrap();
        assert_eq!(result, Value::Float(15.0));
        __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().remove(id).unwrap());
    }

    #[test]
    fn export_methods_static_with_extern_ref() {
        let id_c = __ANVYX_STORE_COLOR.with(|s| {
            s.borrow_mut().insert(Color {
                r: 3.0,
                g: 4.0,
                b: 5.0,
            })
        });
        let (_, handler) = __anvyx_method_Vec2_from_color();
        let result = handler(vec![extern_handle(id_c)]).unwrap();
        let Value::ExternHandle(new_id) = result else {
            panic!("expected ExternHandle");
        };
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_id.id).unwrap();
            assert_eq!(v.x, 3.0);
            assert_eq!(v.y, 4.0);
        });
        __ANVYX_STORE_COLOR.with(|s| assert!(s.borrow().borrow(id_c).is_ok()));
        __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().remove(new_id.id).unwrap());
        __ANVYX_STORE_COLOR.with(|s| s.borrow_mut().remove(id_c).unwrap());
    }

    #[test]
    fn export_methods_static_with_owned() {
        let id_c = __ANVYX_STORE_COLOR.with(|s| {
            s.borrow_mut().insert(Color {
                r: 7.0,
                g: 8.0,
                b: 9.0,
            })
        });
        let (_, handler) = __anvyx_method_Vec2_from_color_owned();
        let result = handler(vec![extern_handle(id_c)]).unwrap();
        let Value::ExternHandle(new_id) = result else {
            panic!("expected ExternHandle");
        };
        __ANVYX_STORE_VEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_id.id).unwrap();
            assert_eq!(v.x, 7.0);
            assert_eq!(v.y, 8.0);
        });
        __ANVYX_STORE_COLOR.with(|s| assert!(s.borrow().borrow(id_c).is_err()));
        __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().remove(new_id.id).unwrap());
    }

    #[test]
    fn export_methods_metadata_cross_type() {
        let methods: &[ExternMethodDecl] = __ANVYX_METHODS_DECL_VEC2;

        let dot = methods.iter().find(|m| m.name == "dot").unwrap();
        assert_eq!(dot.params, &[("other", "Vec2")]);
        assert_eq!(dot.ret, "float");

        let cb = methods
            .iter()
            .find(|m| m.name == "color_brightness")
            .unwrap();
        assert_eq!(cb.params, &[("c", "Color")]);
        assert_eq!(cb.ret, "float");

        let ac = methods.iter().find(|m| m.name == "apply_color").unwrap();
        assert_eq!(ac.params, &[("c", "Color")]);

        let wv = methods.iter().find(|m| m.name == "with_value").unwrap();
        assert_eq!(wv.params, &[("v", "any")]);

        let statics: &[ExternStaticMethodDecl] = __ANVYX_STATICS_DECL_VEC2;

        let fc = statics.iter().find(|s| s.name == "from_color").unwrap();
        assert_eq!(fc.params, &[("c", "Color")]);
        assert_eq!(fc.ret, "Vec2");
    }
}

// -- provider! + #[export_methods] integration tests --

mod methods_provider {
    use anvyx_lang::{Value, export_fn, export_methods, export_type, exports_to_json};

    #[export_type(name = "Vec2")]
    pub struct Vec2 {
        pub x: f32,
        pub y: f32,
    }

    #[export_methods]
    impl Vec2 {
        pub fn new(x: f32, y: f32) -> Vec2 {
            Vec2 { x, y }
        }
        pub fn x(&self) -> f32 {
            self.x
        }
        pub fn set_x(&mut self, x: f32) {
            self.x = x;
        }
    }

    #[export_fn]
    pub fn add_floats(a: f32, b: f32) -> f32 {
        a + b
    }

    anvyx_lang::provider!(types: [Vec2], add_floats);

    #[test]
    fn provider_methods_in_externs() {
        let externs = anvyx_externs();
        assert_eq!(externs.len(), 4);
        assert!(externs.contains_key("Vec2::new"));
        assert!(externs.contains_key("Vec2::x"));
        assert!(externs.contains_key("Vec2::set_x"));
        assert!(externs.contains_key("add_floats"));
    }

    #[test]
    fn provider_method_handlers_work() {
        let externs = anvyx_externs();
        let result = externs["Vec2::new"](vec![Value::Float(3.0), Value::Float(4.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle")
        };
        let x = externs["Vec2::x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(3.0));
        externs["Vec2::set_x"](vec![Value::ExternHandle(id.clone()), Value::Float(99.0)]).unwrap();
        let x = externs["Vec2::x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(99.0));
        __ANVYX_STORE_VEC2.with(|s| s.borrow_mut().remove(id.id).unwrap());
    }

    #[test]
    fn provider_type_exports_have_methods() {
        let types = anvyx_type_exports();
        assert_eq!(types.len(), 1);
        let ty = &types[0];
        assert_eq!(ty.name, "Vec2");
        assert_eq!(ty.methods.len(), 2);
        assert_eq!(ty.statics.len(), 1);

        let x_method = ty.methods.iter().find(|m| m.name == "x").unwrap();
        assert_eq!(x_method.receiver, "self");
        assert_eq!(x_method.params, &[]);
        assert_eq!(x_method.ret, "float");

        let set_x = ty.methods.iter().find(|m| m.name == "set_x").unwrap();
        assert_eq!(set_x.receiver, "var");

        let new_static = ty.statics.iter().find(|s| s.name == "new").unwrap();
        assert_eq!(new_static.params, &[("x", "float"), ("y", "float")]);
        assert_eq!(new_static.ret, "Vec2");
    }

    #[test]
    fn provider_metadata_json_includes_methods() {
        let json = exports_to_json(ANVYX_EXPORTS, &anvyx_type_exports());
        assert!(json.contains("\"methods\":["));
        assert!(json.contains("\"statics\":["));
        assert!(json.contains("\"receiver\":\"self\""));
        assert!(json.contains("\"name\":\"new\""));
    }
}

// -- #[field] annotation + field handlers + provider integration tests --

mod field_provider {
    use anvyx_lang::{Value, export_methods, export_type, exports_to_json};

    #[export_type(name = "Pos")]
    pub struct Pos {
        #[field]
        pub x: f32,
        #[field]
        pub y: f32,
        pub internal: f32, // not exported — no #[field]
    }

    #[export_methods]
    impl Pos {
        pub fn new(x: f32, y: f32) -> Pos {
            Pos {
                x,
                y,
                internal: 0.0,
            }
        }
    }

    anvyx_lang::provider!(types: [Pos]);

    #[test]
    fn field_handler_count() {
        let externs = anvyx_externs();
        // 1 static method (Pos::new) + 4 field handlers (get_x, set_x, get_y, set_y)
        assert_eq!(externs.len(), 5);
    }

    #[test]
    fn field_handler_keys() {
        let externs = anvyx_externs();
        assert!(externs.contains_key("Pos::__get_x"));
        assert!(externs.contains_key("Pos::__set_x"));
        assert!(externs.contains_key("Pos::__get_y"));
        assert!(externs.contains_key("Pos::__set_y"));
        assert!(externs.contains_key("Pos::new"));
        // internal field should NOT have handlers
        assert!(!externs.contains_key("Pos::__get_internal"));
    }

    #[test]
    fn field_getter_returns_correct_value() {
        let externs = anvyx_externs();
        let result = externs["Pos::new"](vec![Value::Float(3.5), Value::Float(7.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle")
        };
        let x = externs["Pos::__get_x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(3.5));
        let y = externs["Pos::__get_y"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(y, Value::Float(7.0));
    }

    #[test]
    fn field_setter_modifies_value() {
        let externs = anvyx_externs();
        let result = externs["Pos::new"](vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle")
        };

        externs["Pos::__set_x"](vec![Value::ExternHandle(id.clone()), Value::Float(99.0)]).unwrap();
        let x = externs["Pos::__get_x"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(x, Value::Float(99.0));
    }

    #[test]
    fn field_metadata_in_type_exports() {
        let types = anvyx_type_exports();
        assert_eq!(types.len(), 1);
        let ty = &types[0];
        assert_eq!(ty.name, "Pos");
        assert_eq!(ty.fields.len(), 2);
        assert_eq!(ty.fields[0].name, "x");
        assert_eq!(ty.fields[0].ty, "float");
        assert_eq!(ty.fields[1].name, "y");
        assert_eq!(ty.fields[1].ty, "float");
    }

    #[test]
    fn field_metadata_json() {
        let json = exports_to_json(ANVYX_EXPORTS, &anvyx_type_exports());
        assert!(json.contains("\"fields\":["));
        assert!(json.contains("\"name\":\"x\""));
        assert!(json.contains("\"ty\":\"float\""));
    }
}

mod getter_setter_tests {
    use anvyx_lang::{Value, export_methods, export_type};

    #[export_type(name = "Vec2")]
    pub struct GsVec2(f32, f32);

    #[export_methods(name = "Vec2")]
    impl GsVec2 {
        pub fn new(x: f32, y: f32) -> GsVec2 {
            GsVec2(x, y)
        }
        #[getter]
        pub fn x(&self) -> f32 {
            self.0
        }
        #[setter]
        pub fn set_x(&mut self, v: f32) {
            self.0 = v;
        }
        #[getter]
        pub fn y(&self) -> f32 {
            self.1
        }
        #[setter]
        pub fn set_y(&mut self, v: f32) {
            self.1 = v;
        }
        pub fn length(&self) -> f32 {
            (self.0 * self.0 + self.1 * self.1).sqrt()
        }
    }

    anvyx_lang::provider!(types: [GsVec2]);

    #[test]
    fn getter_setter_handlers_registered() {
        let externs = anvyx_externs();
        assert!(externs.contains_key("Vec2::__get_x"));
        assert!(externs.contains_key("Vec2::__set_x"));
        assert!(externs.contains_key("Vec2::__get_y"));
        assert!(externs.contains_key("Vec2::__set_y"));
        assert!(externs.contains_key("Vec2::new"));
        assert!(externs.contains_key("Vec2::length"));
        assert_eq!(externs.len(), 6);
    }

    #[test]
    fn getter_setter_type_exports_fields() {
        let types = anvyx_type_exports();
        assert_eq!(types.len(), 1);
        let ty = &types[0];
        assert_eq!(ty.fields.len(), 2);
        assert_eq!(ty.fields[0].name, "x");
        assert_eq!(ty.fields[0].ty, "float");
        assert_eq!(ty.fields[1].name, "y");
        assert_eq!(ty.fields[1].ty, "float");
    }

    #[test]
    fn getter_setter_methods_exclude_getters_setters() {
        let types = anvyx_type_exports();
        let ty = &types[0];
        assert_eq!(ty.methods.len(), 1);
        assert_eq!(ty.methods[0].name, "length");
    }

    #[test]
    fn getter_returns_correct_value() {
        let externs = anvyx_externs();
        let handle = externs["Vec2::new"](vec![Value::Float(3.0), Value::Float(4.0)]).unwrap();
        let Value::ExternHandle(ref ehd) = handle else {
            panic!("expected ExternHandle")
        };
        let x = externs["Vec2::__get_x"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(x, Value::Float(3.0));
        let y = externs["Vec2::__get_y"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(y, Value::Float(4.0));
    }

    #[test]
    fn setter_modifies_value() {
        let externs = anvyx_externs();
        let handle = externs["Vec2::new"](vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(ref ehd) = handle else {
            panic!("expected ExternHandle")
        };
        externs["Vec2::__set_x"](vec![Value::ExternHandle(ehd.clone()), Value::Float(99.0)])
            .unwrap();
        let x = externs["Vec2::__get_x"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(x, Value::Float(99.0));
    }
}

mod getter_setter_with_field_tests {
    use anvyx_lang::{Value, export_methods, export_type};

    #[export_type(name = "Sprite")]
    pub struct GsSprite {
        #[field]
        pub x: f32,
        #[field]
        pub y: f32,
        scale: f32,
    }

    #[export_methods(name = "Sprite")]
    impl GsSprite {
        pub fn new(x: f32, y: f32) -> GsSprite {
            GsSprite { x, y, scale: 1.0 }
        }
        #[getter]
        pub fn scale(&self) -> f32 {
            self.scale as f32
        }
        #[setter]
        pub fn set_scale(&mut self, v: f32) {
            self.scale = v as f32;
        }
    }

    anvyx_lang::provider!(types: [GsSprite]);

    #[test]
    fn combined_field_and_getter_fields() {
        let types = anvyx_type_exports();
        assert_eq!(types.len(), 1);
        let ty = &types[0];
        // x, y from #[field]; scale from #[getter]
        assert_eq!(ty.fields.len(), 3);
        assert_eq!(ty.fields[0].name, "x");
        assert_eq!(ty.fields[1].name, "y");
        assert_eq!(ty.fields[2].name, "scale");
        assert_eq!(ty.fields[2].ty, "float");
    }

    #[test]
    fn getter_setter_scale_works() {
        let externs = anvyx_externs();
        let handle = externs["Sprite::new"](vec![Value::Float(0.0), Value::Float(0.0)]).unwrap();
        let Value::ExternHandle(ref ehd) = handle else {
            panic!("expected ExternHandle")
        };
        let scale = externs["Sprite::__get_scale"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(scale, Value::Float(1.0));
        externs["Sprite::__set_scale"](vec![Value::ExternHandle(ehd.clone()), Value::Float(2.5)])
            .unwrap();
        let scale2 =
            externs["Sprite::__get_scale"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(scale2, Value::Float(2.5));
    }
}

mod getter_setter_name_override_tests {
    use anvyx_lang::{Value, export_methods, export_type};

    #[export_type(name = "Pt")]
    pub struct SomePoint {
        #[field]
        pub x: f32,
    }

    #[export_methods(name = "Pt")]
    impl SomePoint {
        pub fn new(x: f32) -> SomePoint {
            SomePoint { x }
        }
        #[getter]
        pub fn mag(&self) -> f32 {
            self.x.abs()
        }
        #[setter]
        pub fn set_mag(&mut self, v: f32) {
            self.x = v;
        }
    }

    anvyx_lang::provider!(types: [SomePoint]);

    #[test]
    fn name_override_uses_pt_prefix() {
        let externs = anvyx_externs();
        assert!(externs.contains_key("Pt::new"));
        assert!(externs.contains_key("Pt::__get_mag"));
        assert!(externs.contains_key("Pt::__set_mag"));
        assert!(externs.contains_key("Pt::__get_x"));
        assert!(externs.contains_key("Pt::__set_x"));
        assert!(!externs.contains_key("SomePoint::new"));
    }

    #[test]
    fn name_override_getter_works() {
        let externs = anvyx_externs();
        let handle = externs["Pt::new"](vec![Value::Float(-5.0)]).unwrap();
        let Value::ExternHandle(ref ehd) = handle else {
            panic!("expected ExternHandle")
        };
        let mag = externs["Pt::__get_mag"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(mag, Value::Float(5.0));
    }
}

mod init_explicit_tests {
    use anvyx_lang::{Value, export_methods, export_type};

    #[export_type(name = "Vec2")]
    pub struct InitVec2(f32, f32);

    #[export_methods(name = "Vec2")]
    impl InitVec2 {
        #[init]
        pub fn create(x: f32, y: f32) -> InitVec2 {
            InitVec2(x, y)
        }
        #[getter]
        pub fn x(&self) -> f32 {
            self.0
        }
        #[setter]
        pub fn set_x(&mut self, v: f32) {
            self.0 = v;
        }
        #[getter]
        pub fn y(&self) -> f32 {
            self.1
        }
        #[setter]
        pub fn set_y(&mut self, v: f32) {
            self.1 = v;
        }
        pub fn length(&self) -> f32 {
            (self.0 * self.0 + self.1 * self.1).sqrt()
        }
    }

    anvyx_lang::provider!(types: [InitVec2]);

    #[test]
    fn init_handler_registered() {
        let externs = anvyx_externs();
        assert!(externs.contains_key("Vec2::__init__"));
    }

    #[test]
    fn init_method_not_in_statics() {
        let types = anvyx_type_exports();
        assert_eq!(types[0].statics.len(), 0);
    }

    #[test]
    fn init_has_init_flag() {
        let types = anvyx_type_exports();
        assert!(types[0].has_init);
    }

    #[test]
    fn init_handler_creates_handle() {
        let externs = anvyx_externs();
        let handle = externs["Vec2::__init__"](vec![Value::Float(3.0), Value::Float(4.0)]).unwrap();
        let Value::ExternHandle(ref ehd) = handle else {
            panic!("expected ExternHandle")
        };
        let x = externs["Vec2::__get_x"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(x, Value::Float(3.0));
        let y = externs["Vec2::__get_y"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(y, Value::Float(4.0));
    }

    #[test]
    fn init_regular_methods_still_work() {
        let externs = anvyx_externs();
        assert!(externs.contains_key("Vec2::length"));
    }
}

mod init_auto_tests {
    use anvyx_lang::{Value, export_methods, export_type};

    #[export_type(name = "AutoPos")]
    pub struct AutoPos {
        #[field]
        pub x: f32,
        #[field]
        pub y: f32,
    }

    #[export_methods(name = "AutoPos")]
    impl AutoPos {
        pub fn new(x: f32, y: f32) -> AutoPos {
            AutoPos { x, y }
        }
    }

    anvyx_lang::provider!(types: [AutoPos]);

    #[test]
    fn auto_init_handler_registered() {
        let externs = anvyx_externs();
        assert!(externs.contains_key("AutoPos::__init__"));
    }

    #[test]
    fn auto_init_new_still_exists() {
        let externs = anvyx_externs();
        assert!(externs.contains_key("AutoPos::new"));
    }

    #[test]
    fn auto_init_has_init_flag() {
        let types = anvyx_type_exports();
        assert!(types[0].has_init);
    }

    #[test]
    fn auto_init_handler_works() {
        let externs = anvyx_externs();
        let handle =
            externs["AutoPos::__init__"](vec![Value::Float(1.5), Value::Float(2.5)]).unwrap();
        let Value::ExternHandle(ref ehd) = handle else {
            panic!("expected ExternHandle")
        };
        let x = externs["AutoPos::__get_x"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(x, Value::Float(1.5));
        let y = externs["AutoPos::__get_y"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(y, Value::Float(2.5));
    }
}

mod init_no_auto_tests {
    use anvyx_lang::{Value, export_methods, export_type};

    #[export_type(name = "Obj")]
    pub struct NoAutoObj {
        #[field]
        pub x: f32,
        internal: f32,
    }

    #[export_methods(name = "Obj")]
    impl NoAutoObj {
        pub fn new(x: f32) -> NoAutoObj {
            NoAutoObj { x, internal: 0.0 }
        }
    }

    anvyx_lang::provider!(types: [NoAutoObj]);

    #[test]
    fn no_auto_init_handler() {
        let externs = anvyx_externs();
        assert!(!externs.contains_key("Obj::__init__"));
    }

    #[test]
    fn no_auto_init_flag() {
        let types = anvyx_type_exports();
        assert!(!types[0].has_init);
    }

    #[test]
    fn no_auto_new_still_works() {
        let externs = anvyx_externs();
        let handle = externs["Obj::new"](vec![Value::Float(7.0)]).unwrap();
        let Value::ExternHandle(ref ehd) = handle else {
            panic!("expected ExternHandle")
        };
        let x = externs["Obj::__get_x"](vec![Value::ExternHandle(ehd.clone())]).unwrap();
        assert_eq!(x, Value::Float(7.0));
    }
}

// -- #[op(...)] annotation tests --

mod op_tests {
    use super::extern_handle;
    use anvyx_lang::{ExternOpDecl, Value, export_methods, export_type};

    #[export_type(name = "Vec2")]
    pub struct OpVec2 {
        pub x: f32,
        pub y: f32,
    }

    #[export_methods(name = "Vec2")]
    impl OpVec2 {
        #[init]
        pub fn create(x: f32, y: f32) -> OpVec2 {
            OpVec2 { x, y }
        }

        #[getter]
        pub fn x(&self) -> f32 {
            self.x
        }
        #[setter]
        pub fn set_x(&mut self, v: f32) {
            self.x = v;
        }
        #[getter]
        pub fn y(&self) -> f32 {
            self.y
        }
        #[setter]
        pub fn set_y(&mut self, v: f32) {
            self.y = v;
        }

        #[op(Self + Self)]
        pub fn add(&self, other: &OpVec2) -> OpVec2 {
            OpVec2 {
                x: self.x + other.x,
                y: self.y + other.y,
            }
        }

        #[op(Self - Self)]
        pub fn sub(&self, other: &OpVec2) -> OpVec2 {
            OpVec2 {
                x: self.x - other.x,
                y: self.y - other.y,
            }
        }

        #[op(Self * float)]
        pub fn mul_scalar(&self, s: f32) -> OpVec2 {
            OpVec2 {
                x: self.x * s,
                y: self.y * s,
            }
        }

        #[op(float * Self)]
        pub fn scalar_mul(&self, s: f32) -> OpVec2 {
            OpVec2 {
                x: self.x * s,
                y: self.y * s,
            }
        }

        #[op(-Self)]
        pub fn neg(&self) -> OpVec2 {
            OpVec2 {
                x: -self.x,
                y: -self.y,
            }
        }

        #[op(Self == Self)]
        pub fn eq(&self, other: &OpVec2) -> bool {
            self.x == other.x && self.y == other.y
        }
    }

    #[test]
    fn op_add_generates_correct_key() {
        let (key, _) = __anvyx_method_OpVec2___op_add__Vec2();
        assert_eq!(key, "Vec2::__op_add__Vec2");
    }

    #[test]
    fn op_add_handler_works() {
        let id_a =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 1.0, y: 2.0 }));
        let id_b =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 3.0, y: 4.0 }));
        let (_, handler) = __anvyx_method_OpVec2___op_add__Vec2();
        let result = handler(vec![extern_handle(id_a), extern_handle(id_b)]).unwrap();
        let Value::ExternHandle(new_handle) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_OPVEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_handle.id).unwrap();
            assert_eq!(v.x, 4.0);
            assert_eq!(v.y, 6.0);
        });
        __ANVYX_STORE_OPVEC2.with(|s| {
            s.borrow_mut().remove(id_a).unwrap();
            s.borrow_mut().remove(id_b).unwrap();
        });
    }

    #[test]
    fn op_sub_handler_works() {
        let id_a =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 5.0, y: 7.0 }));
        let id_b =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 2.0, y: 3.0 }));
        let (_, handler) = __anvyx_method_OpVec2___op_sub__Vec2();
        let result = handler(vec![extern_handle(id_a), extern_handle(id_b)]).unwrap();
        let Value::ExternHandle(new_handle) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_OPVEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_handle.id).unwrap();
            assert_eq!(v.x, 3.0);
            assert_eq!(v.y, 4.0);
        });
        __ANVYX_STORE_OPVEC2.with(|s| {
            s.borrow_mut().remove(id_a).unwrap();
            s.borrow_mut().remove(id_b).unwrap();
        });
    }

    #[test]
    fn op_mul_scalar_generates_correct_key() {
        let (key, _) = __anvyx_method_OpVec2___op_mul__float();
        assert_eq!(key, "Vec2::__op_mul__float");
    }

    #[test]
    fn op_mul_scalar_handler_works() {
        let id =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 2.0, y: 3.0 }));
        let (_, handler) = __anvyx_method_OpVec2___op_mul__float();
        let result = handler(vec![extern_handle(id), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(new_handle) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_OPVEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_handle.id).unwrap();
            assert_eq!(v.x, 4.0);
            assert_eq!(v.y, 6.0);
        });
        __ANVYX_STORE_OPVEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
        });
    }

    #[test]
    fn op_scalar_mul_generates_correct_key() {
        let (key, _) = __anvyx_method_OpVec2___op_rmul__float();
        assert_eq!(key, "Vec2::__op_rmul__float");
    }

    #[test]
    fn op_scalar_mul_handler_swaps_args() {
        let id =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 2.0, y: 3.0 }));
        let (_, handler) = __anvyx_method_OpVec2___op_rmul__float();
        // right-dispatch: float first, Vec2 handle second
        let result = handler(vec![Value::Float(2.0), extern_handle(id)]).unwrap();
        let Value::ExternHandle(new_handle) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_OPVEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_handle.id).unwrap();
            assert_eq!(v.x, 4.0);
            assert_eq!(v.y, 6.0);
        });
        __ANVYX_STORE_OPVEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
        });
    }

    #[test]
    fn op_neg_generates_correct_key() {
        let (key, _) = __anvyx_method_OpVec2___op_neg();
        assert_eq!(key, "Vec2::__op_neg");
    }

    #[test]
    fn op_neg_handler_works() {
        let id =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 1.0, y: -2.0 }));
        let (_, handler) = __anvyx_method_OpVec2___op_neg();
        let result = handler(vec![extern_handle(id)]).unwrap();
        let Value::ExternHandle(new_handle) = result else {
            panic!("expected ExternHandle")
        };
        __ANVYX_STORE_OPVEC2.with(|s| {
            let store = s.borrow();
            let v = store.borrow(new_handle.id).unwrap();
            assert_eq!(v.x, -1.0);
            assert_eq!(v.y, 2.0);
        });
        __ANVYX_STORE_OPVEC2.with(|s| {
            s.borrow_mut().remove(id).unwrap();
        });
    }

    #[test]
    fn op_eq_generates_correct_key() {
        let (key, _) = __anvyx_method_OpVec2___op_eq__Vec2();
        assert_eq!(key, "Vec2::__op_eq__Vec2");
    }

    #[test]
    fn op_eq_handler_returns_bool() {
        let id_a =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 1.0, y: 2.0 }));
        let id_b =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 1.0, y: 2.0 }));
        let id_c =
            __ANVYX_STORE_OPVEC2.with(|s| s.borrow_mut().insert(OpVec2 { x: 9.0, y: 9.0 }));
        let (_, handler) = __anvyx_method_OpVec2___op_eq__Vec2();
        let result_eq = handler(vec![extern_handle(id_a), extern_handle(id_b)]).unwrap();
        assert_eq!(result_eq, Value::Bool(true));
        let result_ne = handler(vec![extern_handle(id_a), extern_handle(id_c)]).unwrap();
        assert_eq!(result_ne, Value::Bool(false));
        __ANVYX_STORE_OPVEC2.with(|s| {
            s.borrow_mut().remove(id_a).unwrap();
            s.borrow_mut().remove(id_b).unwrap();
            s.borrow_mut().remove(id_c).unwrap();
        });
    }

    #[test]
    fn op_handlers_registered_in_companion() {
        let handlers = __anvyx_methods_OpVec2();
        let keys: Vec<&str> = handlers.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"Vec2::__op_add__Vec2"));
        assert!(keys.contains(&"Vec2::__op_sub__Vec2"));
        assert!(keys.contains(&"Vec2::__op_mul__float"));
        assert!(keys.contains(&"Vec2::__op_rmul__float"));
        assert!(keys.contains(&"Vec2::__op_neg"));
        assert!(keys.contains(&"Vec2::__op_eq__Vec2"));
    }

    #[test]
    fn op_metadata_const_has_all_entries() {
        let ops: &[ExternOpDecl] = __ANVYX_OPS_DECL_OPVEC2;
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn op_metadata_binary_left_dispatch() {
        let ops: &[ExternOpDecl] = __ANVYX_OPS_DECL_OPVEC2;
        let add = ops.iter().find(|o| o.op == "Add").unwrap();
        assert_eq!(add.rhs, Some("Vec2"));
        assert!(add.lhs.is_none());
        assert_eq!(add.ret, "Vec2");
    }

    #[test]
    fn op_metadata_binary_right_dispatch() {
        let ops: &[ExternOpDecl] = __ANVYX_OPS_DECL_OPVEC2;
        // float * Self
        let rmul = ops.iter().find(|o| o.op == "Mul" && o.lhs.is_some()).unwrap();
        assert!(rmul.rhs.is_none());
        assert_eq!(rmul.lhs, Some("float"));
        assert_eq!(rmul.ret, "Vec2");
    }

    #[test]
    fn op_metadata_binary_left_float() {
        let ops: &[ExternOpDecl] = __ANVYX_OPS_DECL_OPVEC2;
        // Self * float
        let mul = ops.iter().find(|o| o.op == "Mul" && o.rhs.is_some()).unwrap();
        assert_eq!(mul.rhs, Some("float"));
        assert!(mul.lhs.is_none());
    }

    #[test]
    fn op_metadata_unary() {
        let ops: &[ExternOpDecl] = __ANVYX_OPS_DECL_OPVEC2;
        let neg = ops.iter().find(|o| o.op == "Neg").unwrap();
        assert!(neg.rhs.is_none());
        assert!(neg.lhs.is_none());
        assert_eq!(neg.ret, "Vec2");
    }

    #[test]
    fn op_metadata_eq_returns_bool() {
        let ops: &[ExternOpDecl] = __ANVYX_OPS_DECL_OPVEC2;
        let eq = ops.iter().find(|o| o.op == "Eq").unwrap();
        assert_eq!(eq.rhs, Some("Vec2"));
        assert!(eq.lhs.is_none());
        assert_eq!(eq.ret, "bool");
    }
}
