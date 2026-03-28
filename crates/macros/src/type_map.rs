use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{GenericArgument, PathArguments, ReturnType, Type};

pub struct TypeMapping {
    pub extract_variant: TokenStream,
    pub convert_extracted: TokenStream,
    pub wrap_result: TokenStream,
    pub anvyx_type: &'static str,
}

pub struct ExternTypeInfo {
    pub type_ident: syn::Ident,
    pub decl_ident: syn::Ident,
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

pub enum ReturnWrapper {
    None,
    Fallible,
    AnvyxOption,
}

pub struct ClassifiedReturn {
    pub mode: ReturnMode,
    pub wrapper: ReturnWrapper,
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
        type_ident: ident.clone(),
        decl_ident: format_ident!("__ANVYX_TYPE_DECL_{}", name_upper),
    }
}

pub fn classify_param(ty: &Type) -> Option<ParamMode> {
    if is_value_passthrough(ty) {
        return Some(ParamMode::ValuePassthrough);
    }

    if let Type::Reference(ref_type) = ty {
        if let Type::Path(path) = &*ref_type.elem
            && path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
            && map_type(&ref_type.elem).is_none()
            && !is_value_passthrough(&ref_type.elem)
        {
            let info = extern_type_info(ident);
            return Some(if ref_type.mutability.is_some() {
                ParamMode::ExternMutRef(info)
            } else {
                ParamMode::ExternRef(info)
            });
        }
        return None;
    }

    if let Some(mapping) = map_type(ty) {
        return Some(ParamMode::Primitive(mapping));
    }

    if let Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(ident) = path.path.get_ident()
    {
        return Some(ParamMode::ExternOwned(extern_type_info(ident)));
    }

    None
}

fn classify_inner_type(ty: &Type) -> Option<ReturnMode> {
    let is_unit = matches!(ty, Type::Tuple(t) if t.elems.is_empty());
    if is_unit {
        return Some(ReturnMode::Void);
    }
    if is_value_passthrough(ty) {
        return Some(ReturnMode::ValuePassthrough);
    }
    if let Some(mapping) = map_type(ty) {
        return Some(ReturnMode::Primitive(mapping));
    }
    if let Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(ident) = path.path.get_ident()
    {
        return Some(ReturnMode::ExternOwned(extern_type_info(ident)));
    }
    None
}

pub fn classify_return(output: &ReturnType) -> Option<ClassifiedReturn> {
    match output {
        ReturnType::Default => Some(ClassifiedReturn {
            mode: ReturnMode::Void,
            wrapper: ReturnWrapper::None,
        }),
        ReturnType::Type(_, ty) => {
            if let Type::Path(path) = ty.as_ref()
                && path.qself.is_none()
                && path.path.segments.len() == 1
            {
                let seg = &path.path.segments[0];

                if seg.ident == "Option"
                    && let PathArguments::AngleBracketed(args) = &seg.arguments
                    && args.args.len() == 1
                    && let GenericArgument::Type(inner_ty) = &args.args[0]
                {
                    let inner_mode = classify_inner_type(inner_ty)?;
                    return Some(ClassifiedReturn {
                        mode: inner_mode,
                        wrapper: ReturnWrapper::AnvyxOption,
                    });
                }

                if seg.ident == "Result"
                    && let PathArguments::AngleBracketed(args) = &seg.arguments
                    && args.args.len() == 2
                    && let GenericArgument::Type(ok_ty) = &args.args[0]
                    && let GenericArgument::Type(err_ty) = &args.args[1]
                {
                    let is_runtime_error = matches!(err_ty, Type::Path(p)
                        if p.path.segments.last().is_some_and(|s| s.ident == "RuntimeError"));
                    if is_runtime_error {
                        let inner_mode = classify_inner_type(ok_ty)?;
                        return Some(ClassifiedReturn {
                            mode: inner_mode,
                            wrapper: ReturnWrapper::Fallible,
                        });
                    }
                }
            }

            let mode = classify_inner_type(ty)?;
            Some(ClassifiedReturn {
                mode,
                wrapper: ReturnWrapper::None,
            })
        }
    }
}
