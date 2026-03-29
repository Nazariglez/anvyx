use proc_macro2::TokenStream;
use quote::quote;

use crate::type_map::{ClassifiedReturn, ReturnMode, ReturnWrapper};

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

pub fn ret_anvyx_type_str(classified: &ClassifiedReturn) -> Result<TokenStream, &'static str> {
    match &classified.wrapper {
        ReturnWrapper::None | ReturnWrapper::Fallible => Ok(match &classified.mode {
            ReturnMode::Void => quote! { "void" },
            ReturnMode::Primitive(m) => {
                let s = m.anvyx_type;
                quote! { #s }
            }
            ReturnMode::ValuePassthrough => quote! { "any" },
            ReturnMode::ExternOwned(info) => {
                let di = &info.decl_ident;
                quote! { #di.name }
            }
        }),
        ReturnWrapper::AnvyxOption => match &classified.mode {
            ReturnMode::Primitive(m) => {
                let option_type = format!("Option<{}>", m.anvyx_type);
                Ok(quote! { #option_type })
            }
            ReturnMode::Void => Err("Option<()> is not supported"),
            ReturnMode::ValuePassthrough => Err("Option<Value> requires a `ret` override"),
            ReturnMode::ExternOwned(_) => Err("Option<ExternType> requires a `ret` override"),
        },
    }
}

pub fn build_flat_call(call: &TokenStream, classified: &ClassifiedReturn) -> TokenStream {
    match &classified.wrapper {
        ReturnWrapper::None => build_flat_call_inner(call, &classified.mode),
        ReturnWrapper::Fallible => build_flat_call_fallible(call, &classified.mode),
        ReturnWrapper::AnvyxOption => build_flat_call_option(call, &classified.mode),
    }
}

fn build_flat_call_inner(call: &TokenStream, mode: &ReturnMode) -> TokenStream {
    match mode {
        ReturnMode::Void => quote! {
            #call;
            Ok(anvyx_lang::Value::Nil)
        },
        ReturnMode::Primitive(m) => {
            let wrap = &m.wrap_result;
            quote! {
                let result = #call;
                Ok(#wrap)
            }
        }
        ReturnMode::ValuePassthrough => quote! {
            let result = #call;
            Ok(result)
        },
        ReturnMode::ExternOwned(info) => {
            let ty = &info.type_ident;
            quote! {
                let result = #call;
                Ok(anvyx_lang::extern_handle::<#ty>(result))
            }
        }
    }
}

fn build_flat_call_fallible(call: &TokenStream, mode: &ReturnMode) -> TokenStream {
    match mode {
        ReturnMode::Void => quote! {
            #call?;
            Ok(anvyx_lang::Value::Nil)
        },
        ReturnMode::Primitive(m) => {
            let wrap = &m.wrap_result;
            quote! {
                let result = #call?;
                Ok(#wrap)
            }
        }
        ReturnMode::ValuePassthrough => quote! {
            let result = #call?;
            Ok(result)
        },
        ReturnMode::ExternOwned(info) => {
            let ty = &info.type_ident;
            quote! {
                let result = #call?;
                Ok(anvyx_lang::extern_handle::<#ty>(result))
            }
        }
    }
}

fn build_flat_call_option(call: &TokenStream, mode: &ReturnMode) -> TokenStream {
    match mode {
        ReturnMode::Void => quote! {
            #call;
            Ok(anvyx_lang::Value::Nil)
        },
        ReturnMode::Primitive(m) => {
            let wrap = &m.wrap_result;
            quote! {
                match #call {
                    Some(result) => Ok(anvyx_lang::option_some(#wrap)),
                    None => Ok(anvyx_lang::option_none()),
                }
            }
        }
        ReturnMode::ValuePassthrough => quote! {
            match #call {
                Some(result) => Ok(anvyx_lang::option_some(result)),
                None => Ok(anvyx_lang::option_none()),
            }
        },
        ReturnMode::ExternOwned(info) => {
            let ty = &info.type_ident;
            quote! {
                match #call {
                    Some(result) => Ok(anvyx_lang::option_some(anvyx_lang::extern_handle::<#ty>(result))),
                    None => Ok(anvyx_lang::option_none()),
                }
            }
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

    match &classified.wrapper {
        ReturnWrapper::None | ReturnWrapper::Fallible => match &classified.mode {
            ReturnMode::Void => quote! {
                #current?;
                Ok(anvyx_lang::Value::Nil)
            },
            ReturnMode::Primitive(m) => {
                let wrap = &m.wrap_result;
                quote! {
                    let result = #current?;
                    Ok(#wrap)
                }
            }
            ReturnMode::ValuePassthrough => quote! {
                let result = #current?;
                Ok(result)
            },
            ReturnMode::ExternOwned(info) => {
                let ty = &info.type_ident;
                quote! {
                    let result = #current?;
                    Ok(anvyx_lang::extern_handle::<#ty>(result))
                }
            }
        },
        ReturnWrapper::AnvyxOption => match &classified.mode {
            ReturnMode::Void => quote! {
                #current?;
                Ok(anvyx_lang::Value::Nil)
            },
            ReturnMode::Primitive(m) => {
                let wrap = &m.wrap_result;
                quote! {
                    let __opt = #current?;
                    match __opt {
                        Some(result) => Ok(anvyx_lang::option_some(#wrap)),
                        None => Ok(anvyx_lang::option_none()),
                    }
                }
            }
            ReturnMode::ValuePassthrough => quote! {
                let __opt = #current?;
                match __opt {
                    Some(result) => Ok(anvyx_lang::option_some(result)),
                    None => Ok(anvyx_lang::option_none()),
                }
            },
            ReturnMode::ExternOwned(info) => {
                let ty = &info.type_ident;
                quote! {
                    let __opt = #current?;
                    match __opt {
                        Some(result) => Ok(anvyx_lang::option_some(anvyx_lang::extern_handle::<#ty>(result))),
                        None => Ok(anvyx_lang::option_none()),
                    }
                }
            }
        },
    }
}
