use proc_macro2::TokenStream;
use quote::quote;

pub(crate) fn expand_or_error(item: &TokenStream, result: syn::Result<TokenStream>) -> TokenStream {
    match result {
        Ok(ts) => ts,
        Err(e) => {
            let err = e.to_compile_error();
            quote! { #err #item }
        }
    }
}
