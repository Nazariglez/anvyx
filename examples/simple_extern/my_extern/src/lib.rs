use anvyx_lang::export_fn;

#[export_fn]
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[export_fn(name = "greet")]
pub fn greet_user(name: String) -> String {
    format!("Hello, {name}!")
}

anvyx_lang::provider!(add, greet_user);

#[cfg(test)]
mod tests {
    use super::*;
    use anvyx_lang::{ManagedRc, Value};

    #[test]
    fn anvyx_externs_contains_all() {
        let externs = anvyx_externs();
        assert_eq!(externs.len(), 2);
        assert!(externs.contains_key("add"));
        assert!(externs.contains_key("greet"));
    }

    #[test]
    fn add_handler_works() {
        let externs = anvyx_externs();
        let result = externs["add"](vec![Value::Int(3), Value::Int(4)]).unwrap();
        assert_eq!(result, Value::Int(7));
    }

    #[test]
    fn greet_handler_works() {
        let externs = anvyx_externs();
        let result = externs["greet"](vec![Value::String(ManagedRc::new("Anvyx".to_string()))]).unwrap();
        assert_eq!(result, Value::String(ManagedRc::new("Hello, Anvyx!".to_string())));
    }
}
