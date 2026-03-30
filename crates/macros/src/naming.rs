use proc_macro2::Ident;
use quote::format_ident;

/// `__ANVYX_DECL_{UPPER}`
pub fn fn_decl_ident(fn_name: &str) -> Ident {
    format_ident!("__ANVYX_DECL_{}", fn_name.to_uppercase())
}

/// `__anvyx_export_{name}`
pub fn fn_companion_ident(fn_name: &Ident) -> Ident {
    format_ident!("__anvyx_export_{}", fn_name)
}

/// `__ANVYX_TYPE_DECL_{UPPER}`
pub fn type_decl_ident(name: &str) -> Ident {
    format_ident!("__ANVYX_TYPE_DECL_{}", name.to_uppercase())
}

/// `__ANVYX_STORE_{UPPER}`
pub fn type_store_ident(name: &str) -> Ident {
    format_ident!("__ANVYX_STORE_{}", name.to_uppercase())
}

/// `__ANVYX_METHODS_DECL_{UPPER}`
pub fn methods_decl_ident(name: &str) -> Ident {
    format_ident!("__ANVYX_METHODS_DECL_{}", name.to_uppercase())
}

/// `__ANVYX_STATICS_DECL_{UPPER}`
pub fn statics_decl_ident(name: &str) -> Ident {
    format_ident!("__ANVYX_STATICS_DECL_{}", name.to_uppercase())
}

/// `__ANVYX_OPS_DECL_{UPPER}`
pub fn ops_decl_ident(name: &str) -> Ident {
    format_ident!("__ANVYX_OPS_DECL_{}", name.to_uppercase())
}

/// `__ANVYX_HAS_INIT_{UPPER}`
pub fn has_init_ident(name: &str) -> Ident {
    format_ident!("__ANVYX_HAS_INIT_{}", name.to_uppercase())
}

/// `__anvyx_methods_{name}`
pub fn methods_fn_ident(name: &Ident) -> Ident {
    format_ident!("__anvyx_methods_{}", name)
}

/// `__anvyx_fields_{name}`
pub fn fields_fn_ident(name: &Ident) -> Ident {
    format_ident!("__anvyx_fields_{}", name)
}

/// `__anvyx_getter_fields_{name}`
pub fn getter_fields_fn_ident(name: &Ident) -> Ident {
    format_ident!("__anvyx_getter_fields_{}", name)
}

/// `__anvyx_init_fields_{name}`
pub fn init_fields_fn_ident(name: &Ident) -> Ident {
    format_ident!("__anvyx_init_fields_{}", name)
}
