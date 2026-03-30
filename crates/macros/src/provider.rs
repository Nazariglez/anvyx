use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Path, PathSegment, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

fn qualify(prefix: &[&PathSegment], ident: &syn::Ident) -> TokenStream {
    if prefix.is_empty() {
        quote! { #ident }
    } else {
        quote! { #(#prefix)::*::#ident }
    }
}

struct ProviderArgs {
    type_paths: Vec<Path>,
    fn_paths: Punctuated<Path, Token![,]>,
}

impl Parse for ProviderArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut type_paths = vec![];

        if input.peek(syn::Ident) {
            let fork = input.fork();
            let ident: syn::Ident = fork.parse()?;
            if ident == "types" && fork.peek(Token![:]) && !fork.peek(Token![::]) {
                // Commit: consume from real input
                let _ident: syn::Ident = input.parse()?;
                let _colon: Token![:] = input.parse()?;
                let content;
                syn::bracketed!(content in input);
                let types: Punctuated<Path, Token![,]> = Punctuated::parse_terminated(&content)?;
                type_paths = types.into_iter().collect();
                if !input.is_empty() {
                    let _comma: Token![,] = input.parse()?;
                }
            }
        }

        let fn_paths = Punctuated::parse_terminated(input)?;
        Ok(Self {
            type_paths,
            fn_paths,
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

    for path in &args.fn_paths {
        let segments = &path.segments;
        if segments.is_empty() {
            return Err(syn::Error::new_spanned(path, "expected a function path"));
        }

        let fn_name = &segments.last().unwrap().ident;
        let companion_ident = crate::naming::fn_companion_ident(fn_name);
        let decl_ident = crate::naming::fn_decl_ident(&fn_name.to_string());

        let prefix: Vec<_> = segments.iter().take(segments.len() - 1).collect();
        let companion_path = qualify(&prefix, &companion_ident);
        let decl_path = qualify(&prefix, &decl_ident);

        inserts.push(quote! {
            let (name, handler) = #companion_path();
            m.insert(name.into(), handler);
        });

        decl_refs.push(decl_path);
    }

    let mut type_decl_refs = vec![];
    let mut method_inserts = vec![];
    let mut field_inserts = vec![];

    for path in &args.type_paths {
        let segments = &path.segments;
        if segments.is_empty() {
            return Err(syn::Error::new_spanned(path, "expected a type path"));
        }

        let type_name = &segments.last().unwrap().ident;
        let type_name_str = type_name.to_string();

        let prefix: Vec<_> = segments.iter().take(segments.len() - 1).collect();

        let type_decl = qualify(&prefix, &crate::naming::type_decl_ident(&type_name_str));
        let methods_decl = qualify(&prefix, &crate::naming::methods_decl_ident(&type_name_str));
        let statics_decl = qualify(&prefix, &crate::naming::statics_decl_ident(&type_name_str));
        let ops_decl = qualify(&prefix, &crate::naming::ops_decl_ident(&type_name_str));
        let has_init = qualify(&prefix, &crate::naming::has_init_ident(&type_name_str));
        let companion_fn = qualify(&prefix, &crate::naming::methods_fn_ident(type_name));
        let fields_fn = qualify(&prefix, &crate::naming::fields_fn_ident(type_name));
        let getter_fields_fn = qualify(&prefix, &crate::naming::getter_fields_fn_ident(type_name));
        let init_fields_fn = qualify(&prefix, &crate::naming::init_fields_fn_ident(type_name));

        type_decl_refs.push(quote! {
            anvyx_lang::ExternTypeDecl {
                name: #type_decl.name,
                doc: #type_decl.doc,
                has_init: #type_decl.has_init || #has_init,
                fields: {
                    let init_fields = #init_fields_fn();
                    if !init_fields.is_empty() {
                        let init_names: ::std::collections::HashSet<&str> =
                            init_fields.iter().map(|fd| fd.name).collect();
                        let mut f = init_fields;
                        for fd in #type_decl.fields.iter() {
                            if !init_names.contains(fd.name) {
                                f.push(anvyx_lang::ExternFieldDecl {
                                    name: fd.name, ty: fd.ty, computed: true
                                });
                            }
                        }
                        for fd in #getter_fields_fn().into_iter() {
                            if !init_names.contains(fd.name) {
                                f.push(fd);
                            }
                        }
                        f
                    } else {
                        let mut f = #type_decl.fields.to_vec();
                        f.extend(#getter_fields_fn());
                        f
                    }
                },
                methods: #methods_decl.to_vec(),
                statics: #statics_decl.to_vec(),
                operators: #ops_decl.to_vec(),
            }
        });

        method_inserts.push(quote! {
            for (name, handler) in #companion_fn() {
                m.insert(name.into(), handler);
            }
        });

        field_inserts.push(quote! {
            for (name, handler) in #fields_fn() {
                m.insert(name.into(), handler);
            }
        });
    }

    Ok(quote! {
        pub const ANVYX_EXPORTS: &[anvyx_lang::ExternDecl] = &[#(#decl_refs),*];

        pub fn anvyx_type_exports() -> Vec<anvyx_lang::ExternTypeDecl> {
            vec![#(#type_decl_refs),*]
        }

        pub fn anvyx_externs() -> ::std::collections::HashMap<String, anvyx_lang::ExternHandler> {
            let mut m = ::std::collections::HashMap::new();
            #(#inserts)*
            #(#field_inserts)*
            #(#method_inserts)*
            m
        }
    })
}
