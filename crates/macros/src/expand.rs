use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, ItemFn, LitStr, Pat, Result, ReturnType, Token,
    parse::{Parse, ParseStream},
};

use crate::type_map::{ExternTypeInfo, ParamMode, classify_param, classify_return};

struct ExportFnArgs {
    name: Option<String>,
    ret: Option<String>,
    params: HashMap<String, String>,
}

impl Parse for ExportFnArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut name = None;
        let mut ret = None;
        let mut params = HashMap::new();

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "name" => {
                    let _eq: Token![=] = input.parse()?;
                    let lit: LitStr = input.parse()?;
                    name = Some(lit.value());
                }
                "ret" => {
                    let _eq: Token![=] = input.parse()?;
                    let lit: LitStr = input.parse()?;
                    ret = Some(lit.value());
                }
                "params" => {
                    let content;
                    syn::parenthesized!(content in input);
                    while !content.is_empty() {
                        let param_name: syn::Ident = content.parse()?;
                        let _eq: Token![=] = content.parse()?;
                        let lit: LitStr = content.parse()?;
                        params.insert(param_name.to_string(), lit.value());
                        if !content.is_empty() {
                            let _comma: Token![,] = content.parse()?;
                        }
                    }
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `name`, `ret`, or `params`",
                    ));
                }
            }
            if !input.is_empty() {
                let _comma: Token![,] = input.parse()?;
            }
        }

        Ok(Self { name, ret, params })
    }
}

pub fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    match do_expand(attr, item.clone()) {
        Ok(ts) => ts,
        Err(e) => {
            let err = e.to_compile_error();
            quote! { #err #item }
        }
    }
}

fn do_expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: ExportFnArgs = syn::parse2(attr)?;
    let func: ItemFn = syn::parse2(item)?;

    let fn_ident = &func.sig.ident;
    let export_name = args.name.unwrap_or_else(|| fn_ident.to_string());
    let companion_ident = format_ident!("__anvyx_export_{}", fn_ident);
    let decl_upper = fn_ident.to_string().to_uppercase();
    let decl_ident = format_ident!("__ANVYX_DECL_{}", decl_upper);

    let mut extractions = vec![];
    let mut param_names = vec![];
    let mut param_tuples = vec![];
    let mut borrow_params: Vec<crate::codegen::BorrowParam> = vec![];

    for (i, arg) in func.sig.inputs.iter().enumerate() {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "self parameters are not supported in #[export_fn]",
            ));
        };

        let param_name = match &*pat_type.pat {
            Pat::Ident(pi) => pi.ident.clone(),
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "only simple parameter names are supported in #[export_fn]",
                ));
            }
        };

        let param_name_str = param_name.to_string();
        let handle_ident = format_ident!("__handle_{}", i);

        let mode = classify_param(&pat_type.ty).ok_or_else(|| {
            syn::Error::new_spanned(&pat_type.ty, "unsupported type in #[export_fn]")
        })?;

        match mode {
            ParamMode::ValuePassthrough => {
                let anvyx_type = args
                    .params
                    .get(&param_name_str)
                    .cloned()
                    .unwrap_or_else(|| "any".to_string());

                extractions.push(quote! {
                    let #param_name = args[#i].clone();
                });
                param_tuples.push(quote! { (#param_name_str, #anvyx_type) });
            }
            ParamMode::Primitive(mapping) => {
                let extract_variant = &mapping.extract_variant;
                let convert_extracted = &mapping.convert_extracted;
                let anvyx_type_str = args
                    .params
                    .get(&param_name_str)
                    .cloned()
                    .unwrap_or_else(|| mapping.anvyx_type.to_string());

                extractions.push(quote! {
                    let #extract_variant = args[#i].clone() else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected correct type for parameter '{}' in extern fn '{}'",
                            #param_name_str, #export_name
                        )));
                    };
                    let #param_name = #convert_extracted;
                });
                param_tuples.push(quote! { (#param_name_str, #anvyx_type_str) });
            }
            ParamMode::ExternOwned(info) => {
                let ty = &info.type_ident;
                let type_decl_ident = &info.decl_ident;

                extractions.push(quote! {
                    let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#i] else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected extern handle for parameter '{}' in extern fn '{}'",
                            #param_name_str, #export_name
                        )));
                    };
                    let #handle_ident = __ehd.id;
                    let #param_name = <#ty as anvyx_lang::AnvyxExternType>::with_store(|__s| __s.borrow_mut().remove(#handle_ident))?;
                });
                param_tuples.push(quote! { (#param_name_str, #type_decl_ident.name) });
            }
            ParamMode::ExternRef(info) => {
                let ExternTypeInfo {
                    type_ident,
                    decl_ident,
                } = info;

                extractions.push(quote! {
                    let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#i] else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected extern handle for parameter '{}' in extern fn '{}'",
                            #param_name_str, #export_name
                        )));
                    };
                    let #handle_ident = __ehd.id;
                });
                param_tuples.push(quote! { (#param_name_str, #decl_ident.name) });
                let guard_ident = format_ident!("__guard_{}", i);
                borrow_params.push(crate::codegen::BorrowParam {
                    param_name: Some(param_name.clone()),
                    type_ident,
                    handle_ident,
                    guard_ident,
                    is_mut: false,
                });
            }
            ParamMode::ExternMutRef(info) => {
                let ExternTypeInfo {
                    type_ident,
                    decl_ident,
                } = info;

                extractions.push(quote! {
                    let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#i] else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected extern handle for parameter '{}' in extern fn '{}'",
                            #param_name_str, #export_name
                        )));
                    };
                    let #handle_ident = __ehd.id;
                });
                param_tuples.push(quote! { (#param_name_str, #decl_ident.name) });
                let guard_ident = format_ident!("__guard_{}", i);
                borrow_params.push(crate::codegen::BorrowParam {
                    param_name: Some(param_name.clone()),
                    type_ident,
                    handle_ident,
                    guard_ident,
                    is_mut: true,
                });
            }
        }

        param_names.push(param_name);
    }

    let ret_type_ts = ret_anvyx_type(&func.sig.output, &export_name, &args.ret)?;
    let call_body = build_call_body(
        &func.sig.output,
        fn_ident,
        &param_names,
        &export_name,
        &borrow_params,
    )?;

    let doc_token = match crate::codegen::extract_doc(&func.attrs) {
        Some(s) => quote! { Some(#s) },
        None => quote! { None },
    };

    Ok(quote! {
        #func

        pub const #decl_ident: anvyx_lang::ExternDecl = anvyx_lang::ExternDecl {
            name: #export_name,
            params: &[#(#param_tuples),*],
            ret: #ret_type_ts,
            doc: #doc_token,
        };

        pub fn #companion_ident() -> (&'static str, anvyx_lang::ExternHandler) {
            (#export_name, Box::new(|args: Vec<anvyx_lang::Value>| {
                #(#extractions)*
                #call_body
            }))
        }
    })
}

fn ret_anvyx_type(
    output: &ReturnType,
    export_name: &str,
    ret_override: &Option<String>,
) -> syn::Result<TokenStream> {
    if let Some(ret_str) = ret_override {
        return Ok(quote! { #ret_str });
    }

    let mode = classify_return(output).ok_or_else(|| {
        syn::Error::new_spanned(
            output,
            format!(
                "unsupported return type in #[export_fn] '{export_name}': \
                 only i64, f64, bool, String, Value, and exported types are supported"
            ),
        )
    })?;

    crate::codegen::ret_anvyx_type_str(&mode).map_err(|msg| syn::Error::new_spanned(output, msg))
}

fn build_call_body(
    output: &ReturnType,
    fn_ident: &syn::Ident,
    param_names: &[syn::Ident],
    export_name: &str,
    borrow_params: &[crate::codegen::BorrowParam],
) -> syn::Result<TokenStream> {
    let call = quote! { #fn_ident(#(#param_names),*) };

    let mode = classify_return(output).ok_or_else(|| {
        syn::Error::new_spanned(
            output,
            format!(
                "unsupported return type in #[export_fn] '{export_name}': \
                 only i64, f64, bool, String, Value, and exported types are supported"
            ),
        )
    })?;

    let handler_label = format!("extern fn '{export_name}'");
    Ok(crate::codegen::build_call_with_borrows(
        &call,
        &mode,
        &handler_label,
        borrow_params,
    ))
}
