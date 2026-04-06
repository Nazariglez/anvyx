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
    let fallback = item.clone();
    let result = do_expand(attr, item);
    crate::util::expand_or_error(&fallback, result)
}

fn do_expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
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

    // auto derive anvyx strings for AnvyxFn callback params
    for (i, arg) in func.sig.inputs.iter().enumerate() {
        let has_param_override = args
            .params
            .contains_key(&extracted.param_names[i].to_string());
        if has_param_override {
            continue;
        }

        if let FnArg::Typed(pat_type) = arg
            && is_anvyx_fn_type(&pat_type.ty)
        {
            match derive_anvyx_fn_type(&pat_type.ty) {
                Some(anvyx_str) => {
                    let name_str = extracted.param_names[i].to_string();
                    extracted.anvyx_types[i] = quote! { (#name_str, #anvyx_str) };
                }
                None => {
                    let name = &extracted.param_names[i];
                    return Err(syn::Error::new_spanned(
                        &pat_type.ty,
                        format!(
                            "callback parameter `{name}` has non-primitive type arguments that \
                                 cannot be auto-derived; add an explicit annotation: \
                                 #[export_fn(params({name} = \"fn(...) -> ...\"))]"
                        ),
                    ));
                }
            }
        }
    }

    // classify return type once because it is used for both decl and handler body
    let classified = classify_return(&func.sig.output);

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

        #[allow(non_snake_case)]
        pub fn #decl_ident() -> anvyx_lang::ExternDecl {
            anvyx_lang::ExternDecl {
                name: #export_name,
                params: vec![#(#param_tuples),*],
                ret: #ret_type_ts,
                doc: #doc_token,
            }
        }

        pub fn #companion_ident() -> (&'static str, anvyx_lang::ExternHandler) {
            (#export_name, Box::new(|args: Vec<anvyx_lang::Value>| {
                #(#extractions)*
                #call_body
            }))
        }
    })
}

fn rust_type_to_anvyx(ty: &syn::Type) -> Option<String> {
    if let syn::Type::Tuple(t) = ty {
        if t.elems.is_empty() {
            return Some("void".into());
        }
        return None;
    }
    let syn::Type::Path(path) = ty else {
        return None;
    };

    // simple idents, primitives and string
    if let Some(ident) = path.path.get_ident() {
        return match ident.to_string().as_str() {
            "i64" => Some("int".into()),
            "f32" => Some("float".into()),
            "f64" => Some("double".into()),
            "bool" => Some("bool".into()),
            "String" => Some("string".into()),
            _ => None,
        };
    }

    // extract inner ident name from ExternHandle<T>
    if path.path.segments.len() == 1 {
        let seg = &path.path.segments[0];
        if seg.ident == "ExternHandle"
            && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
            && args.args.len() == 1
            && let syn::GenericArgument::Type(syn::Type::Path(inner_path)) = &args.args[0]
            && let Some(inner_ident) = inner_path.path.get_ident()
        {
            return Some(inner_ident.to_string());
        }
    }

    None
}

fn is_anvyx_fn_type(ty: &syn::Type) -> bool {
    let syn::Type::Path(path) = ty else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "AnvyxFn")
}

fn derive_anvyx_fn_type(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let seg = path.path.segments.last()?;
    if seg.ident != "AnvyxFn" {
        return None;
    }

    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    if args.args.len() != 2 {
        return None;
    }

    let syn::GenericArgument::Type(tuple_ty) = &args.args[0] else {
        return None;
    };
    let syn::GenericArgument::Type(ret_ty) = &args.args[1] else {
        return None;
    };

    let param_strs: Vec<String> = match tuple_ty {
        syn::Type::Tuple(t) => {
            let strs: Option<Vec<_>> = t.elems.iter().map(rust_type_to_anvyx).collect();
            strs?
        }
        _ => return None,
    };

    let ret_str = rust_type_to_anvyx(ret_ty)?;
    Some(format!("fn({}) -> {ret_str}", param_strs.join(", ")))
}
