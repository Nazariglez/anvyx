use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(Rc<str>),
    Nil,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Nil => write!(f, "nil"),
        }
    }
}

/// Shared error type for all VM-layer failures
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeError {
    pub message: String,
}

impl RuntimeError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Bool(_) => "bool",
        Value::String(_) => "string",
        Value::Nil => "nil",
    }
}

pub fn value_add(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        // string concatenation, either side may be a non-string
        (Value::String(a), b) => Ok(Value::String(Rc::from(format!("{a}{b}").as_str()))),
        (a, Value::String(b)) => Ok(Value::String(Rc::from(format!("{a}{b}").as_str()))),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot add {} and {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_sub(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot subtract {} from {}",
            type_name(&b),
            type_name(&a)
        ))),
    }
}

pub fn value_mul(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot multiply {} and {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_div(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => {
            if b == 0 {
                Err(RuntimeError::new("division by zero"))
            } else {
                Ok(Value::Int(a / b))
            }
        }
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot divide {} by {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_rem(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => {
            if b == 0 {
                Err(RuntimeError::new("remainder by zero"))
            } else {
                Ok(Value::Int(a % b))
            }
        }
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot compute remainder of {} and {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_negate(v: Value) -> Result<Value, RuntimeError> {
    match v {
        Value::Int(n) => Ok(Value::Int(-n)),
        Value::Float(f) => Ok(Value::Float(-f)),
        other => Err(RuntimeError::new(format!(
            "type error: cannot negate {}",
            type_name(&other)
        ))),
    }
}

pub fn value_not(v: Value) -> Result<Value, RuntimeError> {
    match v {
        Value::Bool(b) => Ok(Value::Bool(!b)),
        other => Err(RuntimeError::new(format!(
            "type error: logical not requires bool, got {}",
            type_name(&other)
        ))),
    }
}

pub fn value_eq(lhs: &Value, rhs: &Value) -> Value {
    Value::Bool(lhs == rhs)
}

pub fn value_neq(lhs: &Value, rhs: &Value) -> Value {
    Value::Bool(lhs != rhs)
}

pub fn value_lt(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a < b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot compare {} < {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_gt(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a > b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot compare {} > {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_lte(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a <= b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot compare {} <= {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_gte(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
        (Value::String(a), Value::String(b)) => Ok(Value::Bool(a >= b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: cannot compare {} >= {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_and(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a && b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: logical and requires bool, got {} and {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_or(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a || b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: logical or requires bool, got {} and {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

pub fn value_xor(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a ^ b)),
        (a, b) => Err(RuntimeError::new(format!(
            "type error: logical xor requires bool, got {} and {}",
            type_name(&a),
            type_name(&b)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(text: &str) -> Value {
        Value::String(Rc::from(text))
    }

    #[test]
    fn display_int() {
        assert_eq!(Value::Int(42).to_string(), "42");
    }

    #[test]
    fn display_float() {
        assert_eq!(Value::Float(3.14).to_string(), "3.14");
        assert_eq!(Value::Float(42.0).to_string(), "42");
    }

    #[test]
    fn display_bool() {
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Bool(false).to_string(), "false");
    }

    #[test]
    fn display_string() {
        assert_eq!(s("hello").to_string(), "hello");
    }

    #[test]
    fn display_nil() {
        assert_eq!(Value::Nil.to_string(), "nil");
    }

    #[test]
    fn add_int() {
        assert_eq!(value_add(Value::Int(1), Value::Int(2)), Ok(Value::Int(3)));
    }

    #[test]
    fn add_float() {
        assert_eq!(
            value_add(Value::Float(1.0), Value::Float(2.0)),
            Ok(Value::Float(3.0))
        );
    }

    #[test]
    fn add_string_concat() {
        assert_eq!(value_add(s("a"), s("b")), Ok(s("ab")));
    }

    #[test]
    fn add_string_int_concat() {
        assert_eq!(value_add(s("n="), Value::Int(3)), Ok(s("n=3")));
    }

    #[test]
    fn add_int_string_concat() {
        assert_eq!(value_add(Value::Int(3), s("!")), Ok(s("3!")));
    }

    #[test]
    fn add_float_string_concat() {
        assert_eq!(value_add(Value::Float(1.5), s("x")), Ok(s("1.5x")));
    }

    #[test]
    fn add_type_mismatch_error() {
        assert!(value_add(Value::Int(1), Value::Bool(true)).is_err());
    }

    #[test]
    fn sub_int() {
        assert_eq!(value_sub(Value::Int(5), Value::Int(3)), Ok(Value::Int(2)));
    }

    #[test]
    fn mul_int() {
        assert_eq!(value_mul(Value::Int(3), Value::Int(4)), Ok(Value::Int(12)));
    }

    #[test]
    fn div_int() {
        assert_eq!(value_div(Value::Int(10), Value::Int(3)), Ok(Value::Int(3)));
    }

    #[test]
    fn div_by_zero_error() {
        assert!(value_div(Value::Int(1), Value::Int(0)).is_err());
    }

    #[test]
    fn rem_int() {
        assert_eq!(value_rem(Value::Int(10), Value::Int(3)), Ok(Value::Int(1)));
    }

    #[test]
    fn negate_int() {
        assert_eq!(value_negate(Value::Int(5)), Ok(Value::Int(-5)));
    }

    #[test]
    fn negate_float() {
        assert_eq!(value_negate(Value::Float(1.5)), Ok(Value::Float(-1.5)));
    }

    #[test]
    fn negate_type_error() {
        assert!(value_negate(Value::Bool(true)).is_err());
    }

    #[test]
    fn not_bool() {
        assert_eq!(value_not(Value::Bool(true)), Ok(Value::Bool(false)));
        assert_eq!(value_not(Value::Bool(false)), Ok(Value::Bool(true)));
    }

    #[test]
    fn not_type_error() {
        assert!(value_not(Value::Int(1)).is_err());
    }

    #[test]
    fn eq_same_type() {
        assert_eq!(value_eq(&Value::Int(1), &Value::Int(1)), Value::Bool(true));
        assert_eq!(value_eq(&Value::Int(1), &Value::Int(2)), Value::Bool(false));
    }

    #[test]
    fn eq_different_types_is_false() {
        assert_eq!(
            value_eq(&Value::Int(1), &Value::Bool(true)),
            Value::Bool(false)
        );
    }

    #[test]
    fn neq_same_type() {
        assert_eq!(value_neq(&Value::Int(1), &Value::Int(2)), Value::Bool(true));
    }

    #[test]
    fn lt_int() {
        assert_eq!(
            value_lt(Value::Int(1), Value::Int(2)),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            value_lt(Value::Int(2), Value::Int(1)),
            Ok(Value::Bool(false))
        );
    }

    #[test]
    fn gt_float() {
        assert_eq!(
            value_gt(Value::Float(2.0), Value::Float(1.0)),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn lte_string() {
        assert_eq!(value_lte(s("abc"), s("abc")), Ok(Value::Bool(true)));
        assert_eq!(value_lte(s("abc"), s("abd")), Ok(Value::Bool(true)));
    }

    #[test]
    fn gte_int() {
        assert_eq!(
            value_gte(Value::Int(3), Value::Int(3)),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn comparison_type_error() {
        assert!(value_lt(Value::Int(1), Value::Float(1.0)).is_err());
    }

    #[test]
    fn and_bools() {
        assert_eq!(
            value_and(Value::Bool(true), Value::Bool(false)),
            Ok(Value::Bool(false))
        );
        assert_eq!(
            value_and(Value::Bool(true), Value::Bool(true)),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn or_bools() {
        assert_eq!(
            value_or(Value::Bool(false), Value::Bool(true)),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            value_or(Value::Bool(false), Value::Bool(false)),
            Ok(Value::Bool(false))
        );
    }

    #[test]
    fn xor_bools() {
        assert_eq!(
            value_xor(Value::Bool(true), Value::Bool(true)),
            Ok(Value::Bool(false))
        );
        assert_eq!(
            value_xor(Value::Bool(true), Value::Bool(false)),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn logical_type_error() {
        assert!(value_and(Value::Int(1), Value::Bool(true)).is_err());
    }
}
