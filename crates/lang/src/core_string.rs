use std::collections::HashMap;

use crate::vm::{ExternHandler, ManagedRc, RuntimeError, Value};

fn str_arg(args: &[Value], idx: usize, name: &str) -> Result<ManagedRc<String>, RuntimeError> {
    match args.get(idx) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(RuntimeError::new(format!(
            "{name}: expected string at arg {idx}, got {other}"
        ))),
        None => Err(RuntimeError::new(format!(
            "{name}: missing arg {idx}"
        ))),
    }
}

fn int_arg(args: &[Value], idx: usize, name: &str) -> Result<i64, RuntimeError> {
    match args.get(idx) {
        Some(Value::Int(n)) => Ok(*n),
        Some(other) => Err(RuntimeError::new(format!(
            "{name}: expected int at arg {idx}, got {other}"
        ))),
        None => Err(RuntimeError::new(format!(
            "{name}: missing arg {idx}"
        ))),
    }
}

pub(crate) fn core_handlers() -> HashMap<String, ExternHandler> {
    let mut map: HashMap<String, ExternHandler> = HashMap::new();

    map.insert("__str_len".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_len")?;
        Ok(Value::Int(s.chars().count() as i64))
    }));

    map.insert("__str_contains".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_contains")?;
        let sub = str_arg(&args, 1, "__str_contains")?;
        Ok(Value::Bool(s.contains(sub.as_str())))
    }));

    map.insert("__str_starts_with".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_starts_with")?;
        let prefix = str_arg(&args, 1, "__str_starts_with")?;
        Ok(Value::Bool(s.starts_with(prefix.as_str())))
    }));

    map.insert("__str_ends_with".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_ends_with")?;
        let suffix = str_arg(&args, 1, "__str_ends_with")?;
        Ok(Value::Bool(s.ends_with(suffix.as_str())))
    }));

    map.insert("__str_find".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_find")?;
        let sub = str_arg(&args, 1, "__str_find")?;
        let result = s.find(sub.as_str()).map(|i| i as i64).unwrap_or(-1);
        Ok(Value::Int(result))
    }));

    map.insert("__str_to_upper".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_to_upper")?;
        Ok(Value::String(ManagedRc::new(s.to_uppercase())))
    }));

    map.insert("__str_to_lower".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_to_lower")?;
        Ok(Value::String(ManagedRc::new(s.to_lowercase())))
    }));

    map.insert("__str_trim".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_trim")?;
        Ok(Value::String(ManagedRc::new(s.trim().to_string())))
    }));

    map.insert("__str_trim_start".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_trim_start")?;
        Ok(Value::String(ManagedRc::new(s.trim_start().to_string())))
    }));

    map.insert("__str_trim_end".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_trim_end")?;
        Ok(Value::String(ManagedRc::new(s.trim_end().to_string())))
    }));

    map.insert("__str_substring".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_substring")?;
        let start = int_arg(&args, 1, "__str_substring")?;
        let len = int_arg(&args, 2, "__str_substring")?;
        if start < 0 || len < 0 {
            return Err(RuntimeError::new(
                "__str_substring: start and len must be non-negative",
            ));
        }
        let start = start as usize;
        let len = len as usize;
        let char_count = s.chars().count();
        if start > char_count || start + len > char_count {
            return Err(RuntimeError::new(format!(
                "__str_substring: index out of bounds (start={start}, len={len}, char_count={char_count})"
            )));
        }
        let result: String = s.chars().skip(start).take(len).collect();
        Ok(Value::String(ManagedRc::new(result)))
    }));

    map.insert("__str_char_at".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_char_at")?;
        let index = int_arg(&args, 1, "__str_char_at")?;
        if index < 0 {
            return Err(RuntimeError::new(
                "__str_char_at: index must be non-negative",
            ));
        }
        match s.chars().nth(index as usize) {
            Some(c) => Ok(Value::String(ManagedRc::<String>::new(c.to_string()))),
            None => Err(RuntimeError::new(format!(
                "__str_char_at: index {index} out of bounds"
            ))),
        }
    }));

    map.insert("__str_split".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_split")?;
        let sep = str_arg(&args, 1, "__str_split")?;
        let parts: Vec<Value> = s
            .split(sep.as_str())
            .map(|part: &str| Value::String(ManagedRc::new(part.to_string())))
            .collect();
        Ok(Value::List(ManagedRc::new(parts)))
    }));

    map.insert("__str_replace".to_string(), Box::new(|args| {
        let s = str_arg(&args, 0, "__str_replace")?;
        let from = str_arg(&args, 1, "__str_replace")?;
        let to = str_arg(&args, 2, "__str_replace")?;
        Ok(Value::String(ManagedRc::new(s.replace(from.as_str(), to.as_str()))))
    }));

    map
}
