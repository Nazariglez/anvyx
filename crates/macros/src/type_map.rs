use proc_macro2::TokenStream;
use quote::quote;
use syn::{GenericArgument, PathArguments, ReturnType, Type};

pub struct ExternTypeInfo {
    pub type_ident: syn::Ident,
    pub decl_ident: syn::Ident,
}

pub enum ParamMode {
    Owned(TokenStream),
    StrRef,
    ExternRef(ExternTypeInfo),
    ExternMutRef(ExternTypeInfo),
}

pub enum ReturnMode {
    Void,
    Valued(TokenStream),
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

fn extern_type_info(ident: &syn::Ident) -> ExternTypeInfo {
    ExternTypeInfo {
        type_ident: ident.clone(),
        decl_ident: crate::naming::type_decl_ident(&ident.to_string()),
    }
}

pub fn classify_param(ty: &Type) -> Option<ParamMode> {
    if let Type::Reference(ref_type) = ty {
        if let Type::Path(path) = &*ref_type.elem
            && path.qself.is_none()
            && let Some(ident) = path.path.get_ident()
        {
            if ident == "str" && ref_type.mutability.is_none() {
                return Some(ParamMode::StrRef);
            }
            let info = extern_type_info(ident);
            return Some(if ref_type.mutability.is_some() {
                ParamMode::ExternMutRef(info)
            } else {
                ParamMode::ExternRef(info)
            });
        }
        return None;
    }

    Some(ParamMode::Owned(quote! { #ty }))
}

fn classify_inner_type(ty: &Type) -> ReturnMode {
    let is_unit = matches!(ty, Type::Tuple(t) if t.elems.is_empty());
    if is_unit {
        ReturnMode::Void
    } else {
        ReturnMode::Valued(quote! { #ty })
    }
}

pub fn classify_return(output: &ReturnType) -> ClassifiedReturn {
    match output {
        ReturnType::Default => ClassifiedReturn {
            mode: ReturnMode::Void,
            wrapper: ReturnWrapper::None,
        },
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
                    let inner_mode = classify_inner_type(inner_ty);
                    return ClassifiedReturn {
                        mode: inner_mode,
                        wrapper: ReturnWrapper::AnvyxOption,
                    };
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
                        let inner_mode = classify_inner_type(ok_ty);
                        return ClassifiedReturn {
                            mode: inner_mode,
                            wrapper: ReturnWrapper::Fallible,
                        };
                    }
                }
            }

            let mode = classify_inner_type(ty);
            ClassifiedReturn {
                mode,
                wrapper: ReturnWrapper::None,
            }
        }
    }
}
