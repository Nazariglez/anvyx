use anvyx_lang::{ManagedRc, StdModule, Value, export_fn, provider};

#[export_fn]
pub fn str_len(s: &str) -> i64 {
    s.chars().count() as i64
}

#[export_fn]
pub fn str_contains(s: &str, sub: &str) -> bool {
    s.contains(sub)
}

#[export_fn]
pub fn str_starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

#[export_fn]
pub fn str_ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

#[export_fn]
pub fn str_find(s: &str, sub: &str) -> i64 {
    s.find(sub)
        .map_or(-1, |byte_pos| s[..byte_pos].chars().count() as i64)
}

#[export_fn]
pub fn str_to_upper(s: &str) -> String {
    s.to_uppercase()
}

#[export_fn]
pub fn str_to_lower(s: &str) -> String {
    s.to_lowercase()
}

#[export_fn]
pub fn str_trim(s: &str) -> String {
    s.trim().to_string()
}

#[export_fn]
pub fn str_trim_start(s: &str) -> String {
    s.trim_start().to_string()
}

#[export_fn]
pub fn str_trim_end(s: &str) -> String {
    s.trim_end().to_string()
}

#[export_fn]
pub fn str_replace(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

#[export_fn(ret = "[string]")]
pub fn str_split(s: &str, sep: &str) -> Value {
    let parts: Vec<Value> = s
        .split(sep)
        .map(|part| Value::String(ManagedRc::new(part.to_string())))
        .collect();
    Value::List(ManagedRc::new(parts))
}

#[export_fn]
pub fn str_substring(s: &str, start: i64, len: i64) -> Option<String> {
    if start < 0 || len < 0 {
        return None;
    }
    let start = start as usize;
    let len = len as usize;
    let chars: Vec<char> = s.chars().collect();
    if start + len > chars.len() {
        return None;
    }
    Some(chars[start..start + len].iter().collect())
}

#[export_fn]
pub fn str_char_at(s: &str, index: i64) -> Option<String> {
    if index < 0 {
        return None;
    }
    s.chars().nth(index as usize).map(|c| c.to_string())
}

provider!(
    str_len,
    str_contains,
    str_starts_with,
    str_ends_with,
    str_find,
    str_to_upper,
    str_to_lower,
    str_trim,
    str_trim_start,
    str_trim_end,
    str_replace,
    str_split,
    str_substring,
    str_char_at,
);

pub fn module() -> StdModule {
    StdModule {
        name: "core_string",
        anv_source: include_str!("./string.anv"),
        exports: anvyx_exports,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
        init: None,
    }
}
