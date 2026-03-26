use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ReturnType, Type};

pub struct TypeMapping {
    pub extract_variant: TokenStream,
    pub convert_extracted: TokenStream,
    pub wrap_result: TokenStream,
    pub anvyx_type: &'static str,
}

pub struct ExternTypeInfo {
    pub store_ident: syn::Ident,
    pub decl_ident: syn::Ident,
    pub cleanup_fn_ident: syn::Ident,
    pub to_string_fn_ident: syn::Ident,
}

pub enum ParamMode {
    Primitive(TypeMapping),
    ValuePassthrough,
    ExternRef(ExternTypeInfo),
    ExternMutRef(ExternTypeInfo),
    ExternOwned(ExternTypeInfo),
}

pub enum ReturnMode {
    Void,
    Primitive(TypeMapping),
    ValuePassthrough,
    ExternOwned(ExternTypeInfo),
}

pub fn is_value_passthrough(ty: &Type) -> bool {
    match ty {
        Type::Path(path) if path.qself.is_none() => {
            path.path.get_ident().is_some_and(|id| id == "Value")
        }
        _ => false,
    }
}

pub fn map_type(ty: &Type) -> Option<TypeMapping> {
    let ident = match ty {
        Type::Path(path) if path.qself.is_none() => path.path.get_ident()?,
        _ => return None,
    };

    match ident.to_string().as_str() {
        "i64" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::Int(p) },
            convert_extracted: quote! { p },
            wrap_result: quote! { anvyx_lang::Value::Int(result) },
            anvyx_type: "int",
        }),
        "f32" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::Float(p) },
            convert_extracted: quote! { p },
            wrap_result: quote! { anvyx_lang::Value::Float(result) },
            anvyx_type: "float",
        }),
        "f64" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::Double(p) },
            convert_extracted: quote! { p },
            wrap_result: quote! { anvyx_lang::Value::Double(result) },
            anvyx_type: "double",
        }),
        "bool" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::Bool(p) },
            convert_extracted: quote! { p },
            wrap_result: quote! { anvyx_lang::Value::Bool(result) },
            anvyx_type: "bool",
        }),
        "String" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::String(p) },
            convert_extracted: quote! { (*p).clone() },
            wrap_result: quote! {
                anvyx_lang::Value::String(anvyx_lang::ManagedRc::new(result))
            },
            anvyx_type: "string",
        }),
        _ => None,
    }
}

fn extern_type_info(ident: &syn::Ident) -> ExternTypeInfo {
    let name_upper = ident.to_string().to_uppercase();
    ExternTypeInfo {
        store_ident: format_ident!("__ANVYX_STORE_{}", name_upper),
        decl_ident: format_ident!("__ANVYX_TYPE_DECL_{}", name_upper),
        cleanup_fn_ident: format_ident!("__anvyx_cleanup_{}", ident),
        to_string_fn_ident: format_ident!("__anvyx_to_string_{}", ident),
    }
}

pub fn classify_param(ty: &Type) -> Option<ParamMode> {
    if is_value_passthrough(ty) {
        return Some(ParamMode::ValuePassthrough);
    }

    if let Type::Reference(ref_type) = ty {
        if let Type::Path(path) = &*ref_type.elem {
            if path.qself.is_none() {
                if let Some(ident) = path.path.get_ident() {
                    if map_type(&ref_type.elem).is_none() && !is_value_passthrough(&ref_type.elem)
                    {
                        let info = extern_type_info(ident);
                        return Some(if ref_type.mutability.is_some() {
                            ParamMode::ExternMutRef(info)
                        } else {
                            ParamMode::ExternRef(info)
                        });
                    }
                }
            }
        }
        return None;
    }

    if let Some(mapping) = map_type(ty) {
        return Some(ParamMode::Primitive(mapping));
    }

    if let Type::Path(path) = ty {
        if path.qself.is_none() {
            if let Some(ident) = path.path.get_ident() {
                return Some(ParamMode::ExternOwned(extern_type_info(ident)));
            }
        }
    }

    None
}

pub fn classify_return(output: &ReturnType) -> Option<ReturnMode> {
    match output {
        ReturnType::Default => Some(ReturnMode::Void),
        ReturnType::Type(_, ty) => {
            let is_unit = matches!(ty.as_ref(), Type::Tuple(t) if t.elems.is_empty());
            if is_unit {
                return Some(ReturnMode::Void);
            }
            if is_value_passthrough(ty) {
                return Some(ReturnMode::ValuePassthrough);
            }
            if let Some(mapping) = map_type(ty) {
                return Some(ReturnMode::Primitive(mapping));
            }
            if let Type::Path(path) = ty.as_ref() {
                if path.qself.is_none() {
                    if let Some(ident) = path.path.get_ident() {
                        return Some(ReturnMode::ExternOwned(extern_type_info(ident)));
                    }
                }
            }
            None
        }
    }
}
