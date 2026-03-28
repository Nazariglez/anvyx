use std::collections::HashMap;

use anvyx_lang::{ExternDecl, ExternHandler, ManagedRc, RuntimeError, StdModule, Value};
use anvyx_macros::export_fn;

fn str_arg(args: &[Value], idx: usize, name: &str) -> Result<ManagedRc<String>, RuntimeError> {
    match args.get(idx) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(RuntimeError::new(format!(
            "{name}: expected string at arg {idx}, got {other}"
        ))),
        None => Err(RuntimeError::new(format!("{name}: missing arg {idx}"))),
    }
}

fn int_arg(args: &[Value], idx: usize, name: &str) -> Result<i64, RuntimeError> {
    match args.get(idx) {
        Some(Value::Int(n)) => Ok(*n),
        Some(other) => Err(RuntimeError::new(format!(
            "{name}: expected int at arg {idx}, got {other}"
        ))),
        None => Err(RuntimeError::new(format!("{name}: missing arg {idx}"))),
    }
}

#[export_fn]
pub fn str_len(s: String) -> i64 {
    s.chars().count() as i64
}

#[export_fn]
pub fn str_contains(s: String, sub: String) -> bool {
    s.contains(sub.as_str())
}

#[export_fn]
pub fn str_starts_with(s: String, prefix: String) -> bool {
    s.starts_with(prefix.as_str())
}

#[export_fn]
pub fn str_ends_with(s: String, suffix: String) -> bool {
    s.ends_with(suffix.as_str())
}

#[export_fn]
pub fn str_find(s: String, sub: String) -> i64 {
    s.find(sub.as_str()).map(|i| i as i64).unwrap_or(-1)
}

#[export_fn]
pub fn str_to_upper(s: String) -> String {
    s.to_uppercase()
}

#[export_fn]
pub fn str_to_lower(s: String) -> String {
    s.to_lowercase()
}

#[export_fn]
pub fn str_trim(s: String) -> String {
    s.trim().to_string()
}

#[export_fn]
pub fn str_trim_start(s: String) -> String {
    s.trim_start().to_string()
}

#[export_fn]
pub fn str_trim_end(s: String) -> String {
    s.trim_end().to_string()
}

#[export_fn]
pub fn str_replace(s: String, from: String, to: String) -> String {
    s.replace(from.as_str(), to.as_str())
}

const DECL_STR_SPLIT: ExternDecl = ExternDecl {
    name: "str_split",
    params: &[("s", "string"), ("sep", "string")],
    ret: "[string]",
};

const DECL_STR_SUBSTRING: ExternDecl = ExternDecl {
    name: "str_substring",
    params: &[("s", "string"), ("start", "int"), ("len", "int")],
    ret: "string",
};

const DECL_STR_CHAR_AT: ExternDecl = ExternDecl {
    name: "str_char_at",
    params: &[("s", "string"), ("index", "int")],
    ret: "string",
};

pub const EXPORTS: &[ExternDecl] = &[
    __ANVYX_DECL_STR_LEN,
    __ANVYX_DECL_STR_CONTAINS,
    __ANVYX_DECL_STR_STARTS_WITH,
    __ANVYX_DECL_STR_ENDS_WITH,
    __ANVYX_DECL_STR_FIND,
    __ANVYX_DECL_STR_TO_UPPER,
    __ANVYX_DECL_STR_TO_LOWER,
    __ANVYX_DECL_STR_TRIM,
    __ANVYX_DECL_STR_TRIM_START,
    __ANVYX_DECL_STR_TRIM_END,
    __ANVYX_DECL_STR_REPLACE,
    DECL_STR_SPLIT,
    DECL_STR_SUBSTRING,
    DECL_STR_CHAR_AT,
];

pub fn handlers() -> HashMap<String, ExternHandler> {
    let mut m = HashMap::new();

    let (name, handler) = __anvyx_export_str_len();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_contains();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_starts_with();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_ends_with();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_find();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_to_upper();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_to_lower();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_trim();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_trim_start();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_trim_end();
    m.insert(name.to_string(), handler);
    let (name, handler) = __anvyx_export_str_replace();
    m.insert(name.to_string(), handler);

    m.insert(
        "str_split".to_string(),
        Box::new(|args| {
            let s = str_arg(&args, 0, "str_split")?;
            let sep = str_arg(&args, 1, "str_split")?;
            let parts: Vec<Value> = s
                .split(sep.as_str())
                .map(|part| Value::String(ManagedRc::new(part.to_string())))
                .collect();
            Ok(Value::List(ManagedRc::new(parts)))
        }),
    );

    m.insert(
        "str_substring".to_string(),
        Box::new(|args| {
            let s = str_arg(&args, 0, "str_substring")?;
            let start = int_arg(&args, 1, "str_substring")?;
            let len = int_arg(&args, 2, "str_substring")?;
            if start < 0 || len < 0 {
                return Err(RuntimeError::new(
                    "str_substring: start and len must be non-negative",
                ));
            }
            let start = start as usize;
            let len = len as usize;
            let char_count = s.chars().count();
            if start > char_count || start + len > char_count {
                return Err(RuntimeError::new(format!(
                    "str_substring: index out of bounds (start={start}, len={len}, char_count={char_count})"
                )));
            }
            let result: String = s.chars().skip(start).take(len).collect();
            Ok(Value::String(ManagedRc::new(result)))
        }),
    );

    m.insert(
        "str_char_at".to_string(),
        Box::new(|args| {
            let s = str_arg(&args, 0, "str_char_at")?;
            let index = int_arg(&args, 1, "str_char_at")?;
            if index < 0 {
                return Err(RuntimeError::new(
                    "str_char_at: index must be non-negative",
                ));
            }
            match s.chars().nth(index as usize) {
                Some(c) => Ok(Value::String(ManagedRc::new(c.to_string()))),
                None => Err(RuntimeError::new(format!(
                    "str_char_at: index {index} out of bounds"
                ))),
            }
        }),
    );

    m
}

pub fn module() -> StdModule {
    StdModule {
        name: "core_string",
        anv_source: include_str!("./string.anv"),
        exports: EXPORTS,
        type_exports: || vec![],
        handlers,
        init: None,
    }
}
