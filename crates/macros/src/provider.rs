use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Path, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

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
        let companion_ident = format_ident!("__anvyx_export_{}", fn_name);
        let decl_upper = fn_name.to_string().to_uppercase();
        let decl_ident = format_ident!("__ANVYX_DECL_{}", decl_upper);

        let (companion_call, decl_ref) = if segments.len() == 1 {
            (quote! { #companion_ident() }, quote! { #decl_ident })
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

    let mut type_decl_refs = vec![];
    let mut method_inserts = vec![];
    let mut field_inserts = vec![];

    for path in &args.type_paths {
        let segments = &path.segments;
        if segments.is_empty() {
            return Err(syn::Error::new_spanned(path, "expected a type path"));
        }

        let type_name = &segments.last().unwrap().ident;
        let name_upper = type_name.to_string().to_uppercase();
        let type_decl_ident = format_ident!("__ANVYX_TYPE_DECL_{}", name_upper);
        let methods_decl_ident = format_ident!("__ANVYX_METHODS_DECL_{}", name_upper);
        let statics_decl_ident = format_ident!("__ANVYX_STATICS_DECL_{}", name_upper);
        let ops_decl_ident = format_ident!("__ANVYX_OPS_DECL_{}", name_upper);
        let companion_fn_ident = format_ident!("__anvyx_methods_{}", type_name);
        let fields_fn_ident = format_ident!("__anvyx_fields_{}", type_name);
        let getter_fields_fn_ident = format_ident!("__anvyx_getter_fields_{}", type_name);
        let init_fields_fn_ident = format_ident!("__anvyx_init_fields_{}", type_name);
        let has_init_ident = format_ident!("__ANVYX_HAS_INIT_{}", name_upper);

        if segments.len() == 1 {
            type_decl_refs.push(quote! {
                anvyx_lang::ExternTypeDecl {
                    name: #type_decl_ident.name,
                    doc: #type_decl_ident.doc,
                    has_init: #type_decl_ident.has_init || #has_init_ident,
                    fields: {
                        let init_fields = #init_fields_fn_ident();
                        if !init_fields.is_empty() {
                            let init_names: ::std::collections::HashSet<&str> =
                                init_fields.iter().map(|fd| fd.name).collect();
                            let mut f = init_fields;
                            for fd in #type_decl_ident.fields.iter() {
                                if !init_names.contains(fd.name) {
                                    f.push(anvyx_lang::ExternFieldDecl {
                                        name: fd.name, ty: fd.ty, computed: true
                                    });
                                }
                            }
                            for fd in #getter_fields_fn_ident().into_iter() {
                                if !init_names.contains(fd.name) {
                                    f.push(fd);
                                }
                            }
                            f
                        } else {
                            let mut f = #type_decl_ident.fields.to_vec();
                            f.extend(#getter_fields_fn_ident());
                            f
                        }
                    },
                    methods: #methods_decl_ident.to_vec(),
                    statics: #statics_decl_ident.to_vec(),
                    operators: #ops_decl_ident.to_vec(),
                }
            });
            method_inserts.push(quote! {
                for (name, handler) in #companion_fn_ident() {
                    m.insert(name.into(), handler);
                }
            });
            field_inserts.push(quote! {
                for (name, handler) in #fields_fn_ident() {
                    m.insert(name.into(), handler);
                }
            });
        } else {
            let prefix: Vec<_> = segments.iter().take(segments.len() - 1).collect();
            type_decl_refs.push(quote! {
                anvyx_lang::ExternTypeDecl {
                    name: #(#prefix)::*::#type_decl_ident.name,
                    doc: #(#prefix)::*::#type_decl_ident.doc,
                    has_init: #(#prefix)::*::#type_decl_ident.has_init || #(#prefix)::*::#has_init_ident,
                    fields: {
                        let init_fields = #(#prefix)::*::#init_fields_fn_ident();
                        if !init_fields.is_empty() {
                            let init_names: ::std::collections::HashSet<&str> =
                                init_fields.iter().map(|fd| fd.name).collect();
                            let mut f = init_fields;
                            for fd in #(#prefix)::*::#type_decl_ident.fields.iter() {
                                if !init_names.contains(fd.name) {
                                    f.push(anvyx_lang::ExternFieldDecl {
                                        name: fd.name, ty: fd.ty, computed: true
                                    });
                                }
                            }
                            for fd in #(#prefix)::*::#getter_fields_fn_ident().into_iter() {
                                if !init_names.contains(fd.name) {
                                    f.push(fd);
                                }
                            }
                            f
                        } else {
                            let mut f = #(#prefix)::*::#type_decl_ident.fields.to_vec();
                            f.extend(#(#prefix)::*::#getter_fields_fn_ident());
                            f
                        }
                    },
                    methods: #(#prefix)::*::#methods_decl_ident.to_vec(),
                    statics: #(#prefix)::*::#statics_decl_ident.to_vec(),
                    operators: #(#prefix)::*::#ops_decl_ident.to_vec(),
                }
            });
            method_inserts.push(quote! {
                for (name, handler) in #(#prefix)::*::#companion_fn_ident() {
                    m.insert(name.into(), handler);
                }
            });
            field_inserts.push(quote! {
                for (name, handler) in #(#prefix)::*::#fields_fn_ident() {
                    m.insert(name.into(), handler);
                }
            });
        }
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
