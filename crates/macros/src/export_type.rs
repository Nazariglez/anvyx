use crate::type_map::map_type;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    ItemStruct, LitStr, Result, Token,
    parse::{Parse, ParseStream},
};

struct ExportTypeArgs {
    name: String,
}

impl Parse for ExportTypeArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let key: syn::Ident = input.parse()?;
        if key != "name" {
            return Err(syn::Error::new(key.span(), "expected `name`"));
        }
        let _eq: Token![=] = input.parse()?;
        let lit: LitStr = input.parse()?;
        if !input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "unexpected tokens after name",
            ));
        }
        Ok(Self { name: lit.value() })
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

fn do_expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let args: ExportTypeArgs = syn::parse2(attr)?;
    let item_struct: ItemStruct = syn::parse2(item)?;

    let struct_ident = &item_struct.ident;
    let name_upper = struct_ident.to_string().to_uppercase();
    let decl_ident = format_ident!("__ANVYX_TYPE_DECL_{}", name_upper);
    let store_ident = format_ident!("__ANVYX_STORE_{}", name_upper);
    let cleanup_fn_ident = format_ident!("__anvyx_cleanup_{}", struct_ident);
    let anvyx_name = &args.name;

    let mut cleaned_struct = item_struct.clone();
    if let syn::Fields::Named(ref mut fields) = cleaned_struct.fields {
        for field in &mut fields.named {
            field.attrs.retain(|a| !a.path().is_ident("field"));
        }
    }

    let field_infos: Vec<_> = match &item_struct.fields {
        syn::Fields::Named(fields) => fields
            .named
            .iter()
            .filter(|f| f.attrs.iter().any(|a| a.path().is_ident("field")))
            .map(|f| {
                let ident = f.ident.as_ref().unwrap();
                let mapping = map_type(&f.ty).ok_or_else(|| {
                    syn::Error::new_spanned(
                        &f.ty,
                        "unsupported type for #[field]; only f64, i64, bool, and String are supported",
                    )
                })?;
                Ok((ident.clone(), mapping))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => vec![],
    };

    let total_named_fields = match &item_struct.fields {
        syn::Fields::Named(f) => f.named.len(),
        _ => 0,
    };
    let auto_init = total_named_fields > 0 && field_infos.len() == total_named_fields;

    let field_decls: Vec<_> = field_infos
        .iter()
        .map(|(ident, mapping)| {
            let name_str = ident.to_string();
            let ty_str = mapping.anvyx_type;
            quote! { anvyx_lang::ExternFieldDecl { name: #name_str, ty: #ty_str } }
        })
        .collect();

    let mut getter_setter_fns = vec![];
    let mut companion_entries = vec![];

    for (field_ident, mapping) in &field_infos {
        let field_str = field_ident.to_string();
        let get_key = format!("{}::__get_{}", anvyx_name, field_str);
        let set_key = format!("{}::__set_{}", anvyx_name, field_str);
        let get_fn_ident = format_ident!("__anvyx_field_get_{}_{}", struct_ident, field_ident);
        let set_fn_ident = format_ident!("__anvyx_field_set_{}_{}", struct_ident, field_ident);

        let wrap_result = &mapping.wrap_result;
        let extract_variant = &mapping.extract_variant;
        let convert_extracted = &mapping.convert_extracted;

        getter_setter_fns.push(quote! {
            #[allow(non_snake_case)]
            fn #get_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#get_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    let anvyx_lang::Value::ExternHandle(ref __ehd) = args[0] else {
                        return Err(anvyx_lang::RuntimeError::new(
                            format!("expected ExternHandle for '{}'", #get_key)
                        ));
                    };
                    let handle = __ehd.id;
                    #store_ident.with(|__store| {
                        let __borrow = __store.borrow();
                        let __guard = __borrow.borrow(handle).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle in '{}': {}", #get_key, e.message
                            ))
                        })?;
                        let result = __guard.#field_ident.clone();
                        Ok(#wrap_result)
                    })
                }))
            }

            #[allow(non_snake_case)]
            fn #set_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#set_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    let anvyx_lang::Value::ExternHandle(ref __ehd) = args[0] else {
                        return Err(anvyx_lang::RuntimeError::new(
                            format!("expected ExternHandle for '{}'", #set_key)
                        ));
                    };
                    let handle = __ehd.id;
                    let #extract_variant = args[1].clone() else {
                        return Err(anvyx_lang::RuntimeError::new(
                            format!("invalid argument type for '{}'", #set_key)
                        ));
                    };
                    let val = #convert_extracted;
                    #store_ident.with(|__store| {
                        let __borrow = __store.borrow();
                        let mut __guard = __borrow.borrow_mut(handle).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle in '{}': {}", #set_key, e.message
                            ))
                        })?;
                        __guard.#field_ident = val;
                        Ok(anvyx_lang::Value::Nil)
                    })
                }))
            }
        });

        companion_entries.push(quote! { #get_fn_ident() });
        companion_entries.push(quote! { #set_fn_ident() });
    }

    if auto_init {
        let init_key = format!("{}::__init__", anvyx_name);
        let init_fn_ident = format_ident!("__anvyx_auto_init_{}", struct_ident);

        let mut init_extractions = vec![];
        let mut init_field_assigns = vec![];
        for (i, (field_ident, mapping)) in field_infos.iter().enumerate() {
            let extract_variant = &mapping.extract_variant;
            let convert_extracted = &mapping.convert_extracted;
            let val_ident = format_ident!("__val_{}", i);
            init_extractions.push(quote! {
                let #extract_variant = args[#i].clone() else {
                    return Err(anvyx_lang::RuntimeError::new(
                        format!("invalid argument type for '{}'", #init_key)
                    ));
                };
                let #val_ident = #convert_extracted;
            });
            init_field_assigns.push(quote! { #field_ident: #val_ident });
        }

        getter_setter_fns.push(quote! {
            #[allow(non_snake_case)]
            fn #init_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#init_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    #(#init_extractions)*
                    let instance = #struct_ident { #(#init_field_assigns),* };
                    let id = #store_ident.with(|__s| __s.borrow_mut().insert(instance));
                    Ok(anvyx_lang::Value::ExternHandle(anvyx_lang::ManagedRc::new(
                        anvyx_lang::ExternHandleData { id, drop_fn: #cleanup_fn_ident }
                    )))
                }))
            }
        });
        companion_entries.push(quote! { #init_fn_ident() });
    }

    let companion_fn_ident = format_ident!("__anvyx_fields_{}", struct_ident);

    Ok(quote! {
        #cleaned_struct

        pub const #decl_ident: anvyx_lang::ExternTypeDeclConst =
            anvyx_lang::ExternTypeDeclConst {
                name: #anvyx_name,
                has_init: #auto_init,
                fields: &[#(#field_decls),*],
            };

        ::std::thread_local! {
            static #store_ident: ::std::cell::RefCell<anvyx_lang::HandleStore<#struct_ident>>
                = ::std::cell::RefCell::new(anvyx_lang::HandleStore::new());
        }

        #[allow(non_snake_case)]
        fn #cleanup_fn_ident(id: u64) {
            #store_ident.with(|s| {
                if let Ok(mut store) = s.try_borrow_mut() {
                    let _ = store.remove(id);
                }
            });
        }

        #(#getter_setter_fns)*

        #[allow(non_snake_case)]
        pub fn #companion_fn_ident() -> Vec<(&'static str, anvyx_lang::ExternHandler)> {
            vec![#(#companion_entries),*]
        }
    })
}
