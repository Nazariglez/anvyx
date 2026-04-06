use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    ItemStruct, LitStr, Result, Token,
    parse::{Parse, ParseStream},
};

use crate::{
    codegen,
    type_map::{ClassifiedReturn, ReturnMode, ReturnWrapper},
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
    let fallback = item.clone();
    let result = do_expand(attr, item);
    crate::util::expand_or_error(&fallback, result)
}

fn do_expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let item_struct: ItemStruct = syn::parse2(item)?;
    let struct_ident = &item_struct.ident;

    let anvyx_name = if attr.is_empty() {
        struct_ident.to_string()
    } else {
        let args: ExportTypeArgs = syn::parse2(attr)?;
        args.name
    };

    let option_name = format!("Option<{anvyx_name}>");

    let struct_name = struct_ident.to_string();
    let decl_ident = crate::naming::type_decl_ident(&struct_name);
    let store_ident = crate::naming::type_store_ident(&struct_name);

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
            .map(|f| (f.ident.as_ref().unwrap().clone(), f.ty.clone()))
            .collect(),
        _ => vec![],
    };

    let total_named_fields = match &item_struct.fields {
        syn::Fields::Named(f) => f.named.len(),
        _ => 0,
    };
    let auto_init = total_named_fields > 0 && field_infos.len() == total_named_fields;

    let field_decls: Vec<_> = field_infos
        .iter()
        .map(|(ident, ty)| {
            let name_str = ident.to_string();
            quote! { anvyx_lang::ExternFieldDecl { name: #name_str, ty: <#ty as anvyx_lang::AnvyxConvert>::anvyx_type(), computed: false } }
        })
        .collect();

    let mut getter_setter_fns = vec![];
    let mut companion_entries = vec![];

    for (field_ident, field_ty) in &field_infos {
        let field_str = field_ident.to_string();
        let get_key = format!("{anvyx_name}::__get_{field_str}");
        let set_key = format!("{anvyx_name}::__set_{field_str}");
        let get_fn_ident = format_ident!("__anvyx_field_get_{}_{}", struct_ident, field_ident);
        let set_fn_ident = format_ident!("__anvyx_field_set_{}_{}", struct_ident, field_ident);

        // getter
        let get_sb = codegen::build_self_borrow(struct_ident, false, &get_key, 0);
        let get_self_extract = &get_sb.extraction;
        let get_call = quote! { __guard_self.#field_ident.clone() };
        let get_classified = ClassifiedReturn {
            mode: ReturnMode::Valued(quote! { #field_ty }),
            wrapper: ReturnWrapper::None,
        };
        let get_label = format!("extern method '{get_key}'");
        let get_body = codegen::build_call_with_borrows(
            &get_call,
            &get_classified,
            &get_label,
            &[get_sb.borrow_param],
        );

        // setter
        let set_sb = codegen::build_self_borrow(struct_ident, true, &set_key, 0);
        let set_self_extract = &set_sb.extraction;
        let set_call = quote! { __guard_self.#field_ident = val };
        let set_classified = ClassifiedReturn {
            mode: ReturnMode::Void,
            wrapper: ReturnWrapper::None,
        };
        let set_label = format!("extern method '{set_key}'");
        let set_body = codegen::build_call_with_borrows(
            &set_call,
            &set_classified,
            &set_label,
            &[set_sb.borrow_param],
        );

        getter_setter_fns.push(quote! {
            #[allow(non_snake_case)]
            fn #get_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#get_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    #get_self_extract
                    #get_body
                }))
            }

            #[allow(non_snake_case)]
            fn #set_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#set_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    #set_self_extract
                    let val = <#field_ty as anvyx_lang::AnvyxConvert>::from_anvyx(&args[1])?;
                    #set_body
                }))
            }
        });

        companion_entries.push(quote! { #get_fn_ident() });
        companion_entries.push(quote! { #set_fn_ident() });
    }

    if auto_init {
        let init_key = format!("{anvyx_name}::__init__");
        let init_fn_ident = format_ident!("__anvyx_auto_init_{}", struct_ident);

        let mut init_extractions = vec![];
        let mut init_field_assigns = vec![];
        for (i, (field_ident, field_ty)) in field_infos.iter().enumerate() {
            let val_ident = format_ident!("__val_{}", i);
            init_extractions.push(quote! {
                let #val_ident = <#field_ty as anvyx_lang::AnvyxConvert>::from_anvyx(&args[#i])?;
            });
            init_field_assigns.push(quote! { #field_ident: #val_ident });
        }

        let init_call = quote! { #struct_ident { #(#init_field_assigns),* } };
        let init_classified = ClassifiedReturn {
            mode: ReturnMode::Valued(quote! { #struct_ident }),
            wrapper: ReturnWrapper::None,
        };
        let init_body = codegen::build_flat_call(&init_call, &init_classified);

        getter_setter_fns.push(quote! {
            #[allow(non_snake_case)]
            fn #init_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#init_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    #(#init_extractions)*
                    #init_body
                }))
            }
        });
        companion_entries.push(quote! { #init_fn_ident() });
    }

    let companion_fn_ident = crate::naming::fields_fn_ident(struct_ident);

    let type_doc_token = if let Some(s) = codegen::extract_doc(&item_struct.attrs) {
        quote! { Some(#s) }
    } else {
        quote! { None }
    };

    Ok(quote! {
        #cleaned_struct

        #[allow(non_snake_case)]
        pub fn #decl_ident() -> anvyx_lang::ExternTypeDeclConst {
            anvyx_lang::ExternTypeDeclConst {
                name: #anvyx_name,
                doc: #type_doc_token,
                has_init: #auto_init,
                fields: vec![#(#field_decls),*],
            }
        }

        ::std::thread_local! {
            static #store_ident: ::std::cell::RefCell<anvyx_lang::HandleStore<#struct_ident>>
                = ::std::cell::RefCell::new(anvyx_lang::HandleStore::new());
        }

        impl anvyx_lang::AnvyxExternType for #struct_ident {
            const TYPE_NAME: &'static str = #anvyx_name;

            fn with_store<R>(f: impl FnOnce(&::std::cell::RefCell<anvyx_lang::HandleStore<Self>>) -> R) -> R {
                #store_ident.with(f)
            }

            fn cleanup(id: u64) {
                #store_ident.with(|s| {
                    if let Ok(mut store) = s.try_borrow_mut() {
                        let _ = store.remove(id);
                    }
                });
            }

            fn to_display(id: u64) -> String {
                use anvyx_lang::DisplayDetectFallback as _;
                #store_ident.with(|__store| {
                    let Ok(__borrow) = __store.try_borrow() else {
                        return format!("<{}>", Self::TYPE_NAME);
                    };
                    match __borrow.borrow(id) {
                        Ok(__guard) => anvyx_lang::DisplayDetect(&*__guard).anvyx_display(Self::TYPE_NAME),
                        Err(_) => format!("<{}>", Self::TYPE_NAME),
                    }
                })
            }
        }

        impl anvyx_lang::AnvyxConvert for #struct_ident {
            fn anvyx_type() -> &'static str { #anvyx_name }
            fn anvyx_option_type() -> &'static str { #option_name }

            fn into_anvyx(self) -> anvyx_lang::Value {
                anvyx_lang::extern_handle(self)
            }

            fn from_anvyx(v: &anvyx_lang::Value) -> Result<Self, anvyx_lang::RuntimeError> {
                let anvyx_lang::Value::ExternHandle(ehd) = v else {
                    return Err(anvyx_lang::RuntimeError::new(
                        format!("expected {}", #anvyx_name)
                    ));
                };
                let id = ehd.id;
                <#struct_ident as anvyx_lang::AnvyxExternType>::with_store(|s| s.borrow().clone_value(id))
            }
        }

        #(#getter_setter_fns)*

        #[allow(non_snake_case)]
        pub fn #companion_fn_ident() -> Vec<(&'static str, anvyx_lang::ExternHandler)> {
            vec![#(#companion_entries),*]
        }
    })
}
