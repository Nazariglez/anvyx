use anvyx_lang::{ManagedRc, StdModule, Value, export_fn, provider};

#[export_fn]
pub fn str_len(s: String) -> i64 {
    let s = s.into_boxed_str();
    s.chars().count() as i64
}

#[export_fn]
pub fn str_contains(s: String, sub: String) -> bool {
    let s = s.into_boxed_str();
    let sub = sub.into_boxed_str();
    s.contains(sub.as_ref())
}

#[export_fn]
pub fn str_starts_with(s: String, prefix: String) -> bool {
    let s = s.into_boxed_str();
    let prefix = prefix.into_boxed_str();
    s.starts_with(prefix.as_ref())
}

#[export_fn]
pub fn str_ends_with(s: String, suffix: String) -> bool {
    let s = s.into_boxed_str();
    let suffix = suffix.into_boxed_str();
    s.ends_with(suffix.as_ref())
}

#[export_fn]
pub fn str_find(s: String, sub: String) -> i64 {
    let s = s.into_boxed_str();
    let sub = sub.into_boxed_str();
    s.find(sub.as_ref())
        .map_or(-1, |byte_pos| s[..byte_pos].chars().count() as i64)
}

#[export_fn]
pub fn str_to_upper(s: String) -> String {
    s.into_boxed_str().to_uppercase()
}

#[export_fn]
pub fn str_to_lower(s: String) -> String {
    s.into_boxed_str().to_lowercase()
}

#[export_fn]
pub fn str_trim(s: String) -> String {
    s.into_boxed_str().trim().to_string()
}

#[export_fn]
pub fn str_trim_start(s: String) -> String {
    s.into_boxed_str().trim_start().to_string()
}

#[export_fn]
pub fn str_trim_end(s: String) -> String {
    s.into_boxed_str().trim_end().to_string()
}

#[export_fn]
pub fn str_replace(s: String, from: String, to: String) -> String {
    let s = s.into_boxed_str();
    let from = from.into_boxed_str();
    let to = to.into_boxed_str();
    s.replace(from.as_ref(), to.as_ref())
}

#[export_fn(ret = "[string]")]
pub fn str_split(s: String, sep: String) -> Value {
    let s = s.into_boxed_str();
    let sep = sep.into_boxed_str();
    let parts: Vec<Value> = s
        .split(sep.as_ref())
        .map(|part| Value::String(ManagedRc::new(part.to_string())))
        .collect();
    Value::List(ManagedRc::new(parts))
}

#[export_fn]
pub fn str_substring(s: String, start: i64, len: i64) -> Option<String> {
    let s = s.into_boxed_str();
    if start < 0 || len < 0 {
        return None;
    }
    let start = start as usize;
    let len = len as usize;
    let char_count = s.chars().count();
    if start > char_count || start + len > char_count {
        return None;
    }
    Some(s.chars().skip(start).take(len).collect())
}

#[export_fn]
pub fn str_char_at(s: String, index: i64) -> Option<String> {
    let s = s.into_boxed_str();
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
        exports: ANVYX_EXPORTS,
        type_exports: anvyx_type_exports,
        handlers: anvyx_externs,
        init: None,
    }
}
