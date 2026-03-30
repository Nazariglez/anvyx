use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ReturnType, Type};

use crate::type_map::{
    ClassifiedReturn, ExternTypeInfo, ParamMode, ReturnMode, ReturnWrapper, classify_param,
};

pub fn extract_doc(attrs: &[syn::Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc")
                && let syn::Meta::NameValue(nv) = &attr.meta
                && let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
            {
                return Some(s.value());
            }
            None
        })
        .collect();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n").trim().to_string())
    }
}

pub struct BorrowParam {
    pub param_name: Option<syn::Ident>,
    pub type_ident: syn::Ident,
    pub handle_ident: syn::Ident,
    pub guard_ident: syn::Ident,
    pub is_mut: bool,
}

pub struct SelfBorrow {
    pub extraction: TokenStream,
    pub borrow_param: BorrowParam,
}

pub struct ExtractedParams {
    pub extractions: Vec<TokenStream>,
    pub param_names: Vec<syn::Ident>,
    pub borrow_params: Vec<BorrowParam>,
    pub anvyx_types: Vec<TokenStream>,
}

pub fn build_self_borrow(
    type_ident: &syn::Ident,
    is_mut: bool,
    handler_key: &str,
    arg_idx: usize,
) -> SelfBorrow {
    let extraction = quote! {
        let anvyx_lang::Value::ExternHandle(ref __ehd_self) = args[#arg_idx] else {
            return Err(anvyx_lang::RuntimeError::new(format!(
                "expected extern handle for self in extern method '{}'",
                #handler_key
            )));
        };
        let __handle_self = __ehd_self.id;
    };
    SelfBorrow {
        extraction,
        borrow_param: BorrowParam {
            param_name: None,
            type_ident: type_ident.clone(),
            handle_ident: format_ident!("__handle_self"),
            guard_ident: format_ident!("__guard_self"),
            is_mut,
        },
    }
}

pub fn extract_params(
    params: &[&syn::FnArg],
    arg_offset: usize,
    handler_label: &str,
    type_ident: Option<&syn::Ident>,
) -> syn::Result<ExtractedParams> {
    let mut extractions = vec![];
    let mut param_names = vec![];
    let mut borrow_params = vec![];
    let mut anvyx_types = vec![];

    for (i, arg) in params.iter().enumerate() {
        let syn::FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(arg, "expected typed parameter"));
        };

        let param_name = match &*pat_type.pat {
            syn::Pat::Ident(pi) => pi.ident.clone(),
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "only simple parameter names are supported",
                ));
            }
        };

        let param_name_str = param_name.to_string();
        let arg_idx = arg_offset + i;

        let resolved_ty = match type_ident {
            Some(ti) => resolve_self_in_type(&pat_type.ty, ti),
            None => (*pat_type.ty).clone(),
        };

        let mode = classify_param(&resolved_ty)
            .ok_or_else(|| syn::Error::new_spanned(&pat_type.ty, "unsupported parameter type"))?;

        let (info, is_mut) = match mode {
            ParamMode::Owned(ty) => {
                extractions.push(quote! {
                    let #param_name = <#ty as anvyx_lang::AnvyxConvert>::from_anvyx(&args[#arg_idx])?;
                });
                anvyx_types.push(
                    quote! { (#param_name_str, <#ty as anvyx_lang::AnvyxConvert>::ANVYX_TYPE) },
                );
                param_names.push(param_name);
                continue;
            }
            ParamMode::ExternRef(info) => (info, false),
            ParamMode::ExternMutRef(info) => (info, true),
        };

        let ExternTypeInfo {
            type_ident: ref_type_ident,
            decl_ident,
        } = info;

        let handle_ident = format_ident!("__handle_{}", i);
        extractions.push(quote! {
            let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#arg_idx] else {
                return Err(anvyx_lang::RuntimeError::new(format!(
                    "expected extern handle for parameter '{}' in {}",
                    #param_name_str, #handler_label
                )));
            };
            let #handle_ident = __ehd.id;
        });

        anvyx_types.push(quote! { (#param_name_str, #decl_ident.name) });

        let guard_ident = format_ident!("__guard_{}", i);
        borrow_params.push(BorrowParam {
            param_name: Some(param_name.clone()),
            type_ident: ref_type_ident,
            handle_ident,
            guard_ident,
            is_mut,
        });

        param_names.push(param_name);
    }

    Ok(ExtractedParams {
        extractions,
        param_names,
        borrow_params,
        anvyx_types,
    })
}

pub fn ret_anvyx_type_str(classified: &ClassifiedReturn) -> TokenStream {
    match &classified.wrapper {
        ReturnWrapper::None | ReturnWrapper::Fallible => match &classified.mode {
            ReturnMode::Void => quote! { "void" },
            ReturnMode::Valued(ty) => quote! { <#ty as anvyx_lang::AnvyxConvert>::ANVYX_TYPE },
        },
        ReturnWrapper::AnvyxOption => match &classified.mode {
            ReturnMode::Void => quote! { "void" },
            ReturnMode::Valued(ty) => {
                quote! { <#ty as anvyx_lang::AnvyxConvert>::ANVYX_OPTION_TYPE }
            }
        },
    }
}

fn wrap_raw_value(value_expr: &TokenStream, classified: &ClassifiedReturn) -> TokenStream {
    match (&classified.wrapper, &classified.mode) {
        (_, ReturnMode::Void) => quote! { Ok(anvyx_lang::Value::Nil) },
        (ReturnWrapper::AnvyxOption, ReturnMode::Valued(_)) => quote! {
            match #value_expr {
                Some(result) => Ok(anvyx_lang::option_some(
                    anvyx_lang::AnvyxConvert::into_anvyx(result)
                )),
                None => Ok(anvyx_lang::option_none()),
            }
        },
        (_, ReturnMode::Valued(_)) => quote! {
            Ok(anvyx_lang::AnvyxConvert::into_anvyx(#value_expr))
        },
    }
}

pub fn build_flat_call(call: &TokenStream, classified: &ClassifiedReturn) -> TokenStream {
    let is_fallible = matches!(classified.wrapper, ReturnWrapper::Fallible);
    match &classified.mode {
        ReturnMode::Void => {
            let exec = if is_fallible {
                quote! { #call?; }
            } else {
                quote! { #call; }
            };
            let wrap = wrap_raw_value(&quote! {}, classified);
            quote! { #exec #wrap }
        }
        ReturnMode::Valued(_) => {
            let exec = if is_fallible {
                quote! { let result = #call?; }
            } else {
                quote! { let result = #call; }
            };
            let wrap = wrap_raw_value(&quote! { result }, classified);
            quote! { #exec #wrap }
        }
    }
}

pub fn build_call_with_borrows(
    call: &TokenStream,
    classified: &ClassifiedReturn,
    handler_label: &str,
    borrow_params: &[BorrowParam],
) -> TokenStream {
    if borrow_params.is_empty() {
        return build_flat_call(call, classified);
    }

    let is_void = matches!(classified.mode, ReturnMode::Void);
    let innermost = match &classified.wrapper {
        ReturnWrapper::None | ReturnWrapper::AnvyxOption => {
            if is_void {
                quote! { #call; Ok(()) }
            } else {
                quote! { Ok(#call) }
            }
        }
        ReturnWrapper::Fallible => {
            quote! { #call }
        }
    };

    struct StoreGroup<'a> {
        type_ident: &'a syn::Ident,
        params: Vec<&'a BorrowParam>,
    }

    let mut groups: Vec<StoreGroup> = vec![];
    for bp in borrow_params {
        match groups.iter_mut().find(|g| g.type_ident == &bp.type_ident) {
            Some(group) => group.params.push(bp),
            None => groups.push(StoreGroup {
                type_ident: &bp.type_ident,
                params: vec![bp],
            }),
        }
    }

    let mut current = innermost;
    for group in groups.iter().rev() {
        let type_ident = group.type_ident;

        let borrow_stmts: Vec<TokenStream> = group
            .params
            .iter()
            .map(|bp| {
                let handle = &bp.handle_ident;
                let guard = &bp.guard_ident;

                if let Some(param_name) = &bp.param_name {
                    let param_str = param_name.to_string();
                    if bp.is_mut {
                        quote! {
                            let mut #guard = __borrow.borrow_mut(#handle).map_err(|e| {
                                anvyx_lang::RuntimeError::new(format!(
                                    "invalid handle for parameter '{}' in {}: {}",
                                    #param_str, #handler_label, e.message
                                ))
                            })?;
                            let #param_name = &mut *#guard;
                        }
                    } else {
                        quote! {
                            let #guard = __borrow.borrow(#handle).map_err(|e| {
                                anvyx_lang::RuntimeError::new(format!(
                                    "invalid handle for parameter '{}' in {}: {}",
                                    #param_str, #handler_label, e.message
                                ))
                            })?;
                            let #param_name = &*#guard;
                        }
                    }
                } else if bp.is_mut {
                    quote! {
                        let mut #guard = __borrow.borrow_mut(#handle).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle for self in {}: {}",
                                #handler_label, e.message
                            ))
                        })?;
                    }
                } else {
                    quote! {
                        let #guard = __borrow.borrow(#handle).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle for self in {}: {}",
                                #handler_label, e.message
                            ))
                        })?;
                    }
                }
            })
            .collect();

        current = quote! {
            <#type_ident as anvyx_lang::AnvyxExternType>::with_store(|__store| {
                let __borrow = __store.borrow();
                #(#borrow_stmts)*
                #current
            })
        };
    }

    match &classified.mode {
        ReturnMode::Void => {
            let wrap = wrap_raw_value(&quote! {}, classified);
            quote! { #current?; #wrap }
        }
        ReturnMode::Valued(_) => {
            let wrap = wrap_raw_value(&quote! { result }, classified);
            quote! { let result = #current?; #wrap }
        }
    }
}

pub fn resolve_self_in_return(output: &ReturnType, type_ident: &syn::Ident) -> ReturnType {
    match output {
        ReturnType::Default => ReturnType::Default,
        ReturnType::Type(arrow, ty) => {
            let resolved = resolve_self_in_type(ty, type_ident);
            ReturnType::Type(*arrow, Box::new(resolved))
        }
    }
}

pub fn resolve_self_in_type(ty: &Type, type_ident: &syn::Ident) -> Type {
    match ty {
        Type::Path(path) if path.qself.is_none() => {
            if let Some(ident) = path.path.get_ident()
                && ident == "Self"
            {
                return Type::Path(syn::TypePath {
                    qself: None,
                    path: type_ident.clone().into(),
                });
            }
            ty.clone()
        }
        Type::Reference(ref_type) => {
            let resolved_elem = resolve_self_in_type(&ref_type.elem, type_ident);
            Type::Reference(syn::TypeReference {
                and_token: ref_type.and_token,
                lifetime: ref_type.lifetime.clone(),
                mutability: ref_type.mutability,
                elem: Box::new(resolved_elem),
            })
        }
        _ => ty.clone(),
    }
}
