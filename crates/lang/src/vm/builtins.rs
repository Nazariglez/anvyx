use std::fmt::Write;

use super::value::{RuntimeError, Value};
use crate::builtin::Builtin;

pub fn call_builtin(
    builtin: Builtin,
    args: &[Value],
    stdout: &mut String,
) -> Result<Value, RuntimeError> {
    match builtin {
        Builtin::Println => {
            writeln!(stdout, "{}", args[0]).unwrap();
            Ok(Value::Nil)
        }

        Builtin::Assert => match &args[0] {
            Value::Bool(true) => Ok(Value::Nil),
            _ => Err(RuntimeError::new("assertion failed")),
        },

        Builtin::AssertMsg => match &args[0] {
            Value::Bool(true) => Ok(Value::Nil),
            _ => Err(RuntimeError::new(format!("assertion failed: {}", args[1]))),
        },
    }
}
