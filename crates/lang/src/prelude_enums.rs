use crate::vm::{EnumData, ManagedRc, Value};

pub const OPTION_TYPE_ID: u32 = 0;

pub fn option_some(value: Value) -> Value {
    Value::Enum(ManagedRc::new(EnumData {
        type_id: OPTION_TYPE_ID,
        variant: 1,
        fields: vec![value],
    }))
}

pub fn option_none() -> Value {
    Value::Enum(ManagedRc::new(EnumData {
        type_id: OPTION_TYPE_ID,
        variant: 0,
        fields: vec![],
    }))
}
