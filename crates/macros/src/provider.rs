use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Path, Token,
};

struct ProviderArgs {
    paths: Punctuated<Path, Token![,]>,
}

impl Parse for ProviderArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            paths: Punctuated::parse_terminated(input)?,
        })
    }
}

pub fn expand(input: TokenStream) -> TokenStream {
    match do_expand(input) {
        Ok(ts) => ts,
        Err(e) => e.to_compile_error(),
    }
}

fn do_expand(input: TokenStream) -> syn::Result<TokenStream> {
    let args: ProviderArgs = syn::parse2(input)?;

    let mut inserts = vec![];
    let mut decl_refs = vec![];

    for path in &args.paths {
        let segments = &path.segments;
        if segments.is_empty() {
            return Err(syn::Error::new_spanned(path, "expected a function path"));
        }

        let fn_name = &segments.last().unwrap().ident;
        let companion_ident = format_ident!("__anvyx_export_{}", fn_name);
        let decl_upper = fn_name.to_string().to_uppercase();
        let decl_ident = format_ident!("__ANVYX_DECL_{}", decl_upper);

        let (companion_call, decl_ref) = if segments.len() == 1 {
            (
                quote! { #companion_ident() },
                quote! { #decl_ident },
            )
        } else {
            let prefix: Vec<_> = segments.iter().take(segments.len() - 1).collect();
            (
                quote! { #(#prefix)::*::#companion_ident() },
                quote! { #(#prefix)::*::#decl_ident },
            )
        };

        inserts.push(quote! {
            let (name, handler) = #companion_call;
            m.insert(name.into(), handler);
        });

        decl_refs.push(decl_ref);
    }

    Ok(quote! {
        pub const ANVYX_EXPORTS: &[anvyx_lang::ExternDecl] = &[#(#decl_refs),*];

        pub fn anvyx_externs() -> ::std::collections::HashMap<String, anvyx_lang::ExternHandler> {
            let mut m = ::std::collections::HashMap::new();
            #(#inserts)*
            m
        }
    })
}
