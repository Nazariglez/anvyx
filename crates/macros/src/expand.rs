use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    FnArg, ItemFn, LitStr, Result, Token,
    parse::{Parse, ParseStream},
};

use crate::type_map::classify_return;

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
    let companion_ident = crate::naming::fn_companion_ident(fn_ident);
    let decl_ident = crate::naming::fn_decl_ident(&fn_ident.to_string());

    // reject self parameters because they are not valid in free functions
    for arg in &func.sig.inputs {
        if let FnArg::Receiver(_) = arg {
            return Err(syn::Error::new_spanned(
                arg,
                "self parameters are not supported in #[export_fn]",
            ));
        }
    }

    // extract all params via codegen helper
    let handler_label = format!("extern fn '{export_name}'");
    let params: Vec<_> = func.sig.inputs.iter().collect();
    let mut extracted = crate::codegen::extract_params(&params, 0, &handler_label, None)?;

    // Apply param type overrides from #[export_fn(params(...))]
    for (i, name) in extracted.param_names.iter().enumerate() {
        if let Some(override_str) = args.params.get(&name.to_string()) {
            let name_str = name.to_string();
            extracted.anvyx_types[i] = quote! { (#name_str, #override_str) };
        }
    }

    // classify return type once because it is used for both decl and handler body
    let classified = classify_return(&func.sig.output).ok_or_else(|| {
        syn::Error::new_spanned(
            &func.sig.output,
            format!(
                "unsupported return type in #[export_fn] '{export_name}': \
                 only i64, f64, bool, String, Value, and exported types are supported"
            ),
        )
    })?;

    let ret_type_ts = match &args.ret {
        Some(ret_str) => quote! { #ret_str },
        None => crate::codegen::ret_anvyx_type_str(&classified),
    };

    let param_names = &extracted.param_names;
    let call = quote! { #fn_ident(#(#param_names),*) };
    let call_body = crate::codegen::build_call_with_borrows(
        &call,
        &classified,
        &handler_label,
        &extracted.borrow_params,
    );

    let extractions = &extracted.extractions;
    let param_tuples = &extracted.anvyx_types;

    let doc_token = if let Some(s) = crate::codegen::extract_doc(&func.attrs) {
        quote! { Some(#s) }
    } else {
        quote! { None }
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
