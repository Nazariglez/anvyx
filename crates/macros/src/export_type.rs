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
            return Err(syn::Error::new(input.span(), "unexpected tokens after name"));
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
    let anvyx_name = &args.name;

    Ok(quote! {
        #item_struct

        pub const #decl_ident: anvyx_lang::ExternTypeDecl =
            anvyx_lang::ExternTypeDecl { name: #anvyx_name };

        ::std::thread_local! {
            static #store_ident: ::std::cell::RefCell<anvyx_lang::HandleStore<#struct_ident>>
                = ::std::cell::RefCell::new(anvyx_lang::HandleStore::new());
        }
    })
}
