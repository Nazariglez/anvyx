use std::cell::RefCell;

use super::{
    handle_store::HandleStore,
    managed_rc::ManagedRc,
    value::{ExternHandleData, RuntimeError, Value},
};

pub trait AnvyxExternType: Sized + 'static {
    const TYPE_NAME: &'static str;
    fn with_store<R>(f: impl FnOnce(&RefCell<HandleStore<Self>>) -> R) -> R;
    fn cleanup(id: u64);
    fn to_display(id: u64) -> String;
}

pub fn extern_handle<T: AnvyxExternType>(value: T) -> Value {
    let id = T::with_store(|s| s.borrow_mut().insert(value));
    Value::ExternHandle(ManagedRc::new(ExternHandleData {
        id,
        drop_fn: T::cleanup,
        type_name: T::TYPE_NAME,
        to_string_fn: T::to_display,
    }))
}

pub trait AnvyxConvert: Sized {
    const ANVYX_TYPE: &'static str;
    const ANVYX_OPTION_TYPE: &'static str;
    fn into_anvyx(self) -> Value;
    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError>;
}

impl AnvyxConvert for i64 {
    const ANVYX_TYPE: &'static str = "int";
    const ANVYX_OPTION_TYPE: &'static str = "Option<int>";
    fn into_anvyx(self) -> Value {
        Value::Int(self)
    }
    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError> {
        match v {
            Value::Int(n) => Ok(*n),
            _ => Err(RuntimeError::new("expected int")),
        }
    }
}

impl AnvyxConvert for f32 {
    const ANVYX_TYPE: &'static str = "float";
    const ANVYX_OPTION_TYPE: &'static str = "Option<float>";
    fn into_anvyx(self) -> Value {
        Value::Float(self)
    }
    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError> {
        match v {
            Value::Float(n) => Ok(*n),
            _ => Err(RuntimeError::new("expected float")),
        }
    }
}

impl AnvyxConvert for f64 {
    const ANVYX_TYPE: &'static str = "double";
    const ANVYX_OPTION_TYPE: &'static str = "Option<double>";
    fn into_anvyx(self) -> Value {
        Value::Double(self)
    }
    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError> {
        match v {
            Value::Double(n) => Ok(*n),
            _ => Err(RuntimeError::new("expected double")),
        }
    }
}

impl AnvyxConvert for bool {
    const ANVYX_TYPE: &'static str = "bool";
    const ANVYX_OPTION_TYPE: &'static str = "Option<bool>";
    fn into_anvyx(self) -> Value {
        Value::Bool(self)
    }
    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError> {
        match v {
            Value::Bool(b) => Ok(*b),
            _ => Err(RuntimeError::new("expected bool")),
        }
    }
}

impl AnvyxConvert for String {
    const ANVYX_TYPE: &'static str = "string";
    const ANVYX_OPTION_TYPE: &'static str = "Option<string>";
    fn into_anvyx(self) -> Value {
        Value::String(ManagedRc::new(self))
    }
    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError> {
        match v {
            Value::String(s) => Ok((**s).clone()),
            _ => Err(RuntimeError::new("expected string")),
        }
    }
}

impl AnvyxConvert for Value {
    const ANVYX_TYPE: &'static str = "any";
    const ANVYX_OPTION_TYPE: &'static str = "Option<any>";
    fn into_anvyx(self) -> Value {
        self
    }
    fn from_anvyx(v: &Value) -> Result<Self, RuntimeError> {
        Ok(v.clone())
    }
}
