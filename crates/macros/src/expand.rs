use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, ItemFn, LitStr, Pat, Result, ReturnType, Token, Type,
    parse::{Parse, ParseStream},
};

use crate::type_map::map_type;

struct ExportFnArgs {
    name: Option<String>,
}

impl Parse for ExportFnArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.is_empty() {
            return Ok(Self { name: None });
        }
        let key: syn::Ident = input.parse()?;
        if key != "name" {
            return Err(syn::Error::new(key.span(), "expected `name = \"...\"`"));
        }
        let _eq: Token![=] = input.parse()?;
        let lit: LitStr = input.parse()?;
        Ok(Self {
            name: Some(lit.value()),
        })
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

        let mapping = map_type(&pat_type.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &pat_type.ty,
                "unsupported type in #[export_fn]: only i64, f64, bool, and String are supported",
            )
        })?;

        let extract_variant = &mapping.extract_variant;
        let convert_extracted = &mapping.convert_extracted;
        let param_name_str = param_name.to_string();
        let anvyx_type_str = mapping.anvyx_type;

        extractions.push(quote! {
            let #extract_variant = args[#i].clone() else {
                return Err(anvyx_lang::RuntimeError::new(format!(
                    "expected correct type for parameter '{}' in extern fn '{}'",
                    #param_name_str, #export_name
                )));
            };
            let #param_name = #convert_extracted;
        });

        param_names.push(param_name);
        param_tuples.push(quote! { (#param_name_str, #anvyx_type_str) });
    }

    let ret_type_str = ret_anvyx_type(&func.sig.output, &export_name)?;
    let call_and_wrap = build_return(&func.sig.output, fn_ident, &param_names, &export_name)?;

    Ok(quote! {
        #func

        pub const #decl_ident: anvyx_lang::ExternDecl = anvyx_lang::ExternDecl {
            name: #export_name,
            params: &[#(#param_tuples),*],
            ret: #ret_type_str,
        };

        pub fn #companion_ident() -> (&'static str, anvyx_lang::ExternHandler) {
            (#export_name, Box::new(|args: Vec<anvyx_lang::Value>| {
                #(#extractions)*
                #call_and_wrap
            }))
        }
    })
}

fn ret_anvyx_type(output: &ReturnType, export_name: &str) -> syn::Result<&'static str> {
    match output {
        ReturnType::Default => Ok("void"),
        ReturnType::Type(_, ty) => {
            let is_unit = matches!(ty.as_ref(), Type::Tuple(t) if t.elems.is_empty());
            if is_unit {
                return Ok("void");
            }
            let mapping = map_type(ty).ok_or_else(|| {
                syn::Error::new_spanned(
                    ty,
                    format!(
                        "unsupported return type in #[export_fn] '{export_name}': \
                        only i64, f64, bool, and String are supported"
                    ),
                )
            })?;
            Ok(mapping.anvyx_type)
        }
    }
}

fn build_return(
    output: &ReturnType,
    fn_ident: &syn::Ident,
    param_names: &[syn::Ident],
    export_name: &str,
) -> syn::Result<TokenStream> {
    let call = quote! { #fn_ident(#(#param_names),*) };

    match output {
        ReturnType::Default => Ok(quote! {
            #call;
            Ok(anvyx_lang::Value::Nil)
        }),
        ReturnType::Type(_, ty) => {
            let is_unit = matches!(ty.as_ref(), Type::Tuple(t) if t.elems.is_empty());
            if is_unit {
                return Ok(quote! {
                    #call;
                    Ok(anvyx_lang::Value::Nil)
                });
            }
            let mapping = map_type(ty).ok_or_else(|| {
                syn::Error::new_spanned(
                    ty,
                    format!(
                        "unsupported return type in #[export_fn] '{export_name}': \
                        only i64, f64, bool, and String are supported"
                    ),
                )
            })?;
            let wrap_result = &mapping.wrap_result;
            Ok(quote! {
                let result = #call;
                Ok(#wrap_result)
            })
        }
    }
}
