use std::collections::HashMap;

use anvyx_lang::{ExternHandler, Value};

pub fn anvyx_externs() -> HashMap<String, ExternHandler> {
    let mut m = HashMap::new();
    m.insert(
        "add".into(),
        Box::new(|args: Vec<Value>| {
            let Value::Int(a) = &args[0] else {
                unreachable!()
            };
            let Value::Int(b) = &args[1] else {
                unreachable!()
            };
            Ok(Value::Int(a + b))
        }) as ExternHandler,
    );
    m
}
