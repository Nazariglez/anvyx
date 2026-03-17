use proc_macro2::TokenStream;
use quote::quote;
use syn::Type;

pub struct TypeMapping {
    pub extract_variant: TokenStream,
    pub convert_extracted: TokenStream,
    pub wrap_result: TokenStream,
    pub anvyx_type: &'static str,
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
        "f64" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::Float(p) },
            convert_extracted: quote! { p },
            wrap_result: quote! { anvyx_lang::Value::Float(result) },
            anvyx_type: "float",
        }),
        "bool" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::Bool(p) },
            convert_extracted: quote! { p },
            wrap_result: quote! { anvyx_lang::Value::Bool(result) },
            anvyx_type: "bool",
        }),
        "String" => Some(TypeMapping {
            extract_variant: quote! { anvyx_lang::Value::String(p) },
            convert_extracted: quote! { p.to_string() },
            wrap_result: quote! {
                anvyx_lang::Value::String(::std::rc::Rc::from(result.as_str()))
            },
            anvyx_type: "string",
        }),
        _ => None,
    }
}
