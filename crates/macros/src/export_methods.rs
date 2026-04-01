use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, ImplItem, ItemImpl, LitStr, Pat, Token, Type,
    parse::{Parse, ParseStream},
};

use crate::type_map::{
    ClassifiedReturn, ParamMode, ReturnMode, ReturnWrapper, classify_param, classify_return,
};

struct ExportMethodsArgs {
    name_override: Option<String>,
}

impl Parse for ExportMethodsArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self {
                name_override: None,
            });
        }
        let key: syn::Ident = input.parse()?;
        if key != "name" {
            return Err(syn::Error::new(key.span(), "expected `name = \"...\"`"));
        }
        let _eq: Token![=] = input.parse()?;
        let lit: LitStr = input.parse()?;
        if !input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "unexpected tokens after name",
            ));
        }
        Ok(Self {
            name_override: Some(lit.value()),
        })
    }
}

#[derive(Copy, Clone)]
enum MethodKind {
    Static,
    Borrowing,
    Mutating,
}

enum OpKind {
    Binary {
        op: &'static str,
        lhs: String,
        rhs: String,
    },
    Unary {
        op: &'static str,
    },
}

struct OpAnnotation {
    kind: OpKind,
    self_on_right: bool,
}

fn parse_op_annotation(tokens: TokenStream) -> syn::Result<OpAnnotation> {
    use proc_macro2::TokenTree;

    let tts: Vec<TokenTree> = tokens.into_iter().collect();
    let err = |msg: &str| syn::Error::new(proc_macro2::Span::call_site(), msg);

    match tts.len() {
        2 => {
            let is_neg = matches!(&tts[0], TokenTree::Punct(p) if p.as_char() == '-');
            let is_self = matches!(&tts[1], TokenTree::Ident(i) if i == "Self");
            if is_neg && is_self {
                return Ok(OpAnnotation {
                    kind: OpKind::Unary { op: "neg" },
                    self_on_right: false,
                });
            }
            Err(err("invalid unary #[op(...)] syntax; expected `-Self`"))
        }
        3 => {
            let lhs = match &tts[0] {
                TokenTree::Ident(i) => i.to_string(),
                _ => return Err(err("expected identifier for left operand")),
            };
            let op = match &tts[1] {
                TokenTree::Punct(p) => match p.as_char() {
                    '+' => "add",
                    '-' => "sub",
                    '*' => "mul",
                    '/' => "div",
                    '%' => "rem",
                    c => return Err(err(&format!("unsupported operator '{c}'"))),
                },
                _ => return Err(err("expected operator")),
            };
            let rhs = match &tts[2] {
                TokenTree::Ident(i) => i.to_string(),
                _ => return Err(err("expected identifier for right operand")),
            };
            let lhs_is_self = lhs == "Self";
            let rhs_is_self = rhs == "Self";
            if !lhs_is_self && !rhs_is_self {
                return Err(err("at least one operand must be `Self`"));
            }
            Ok(OpAnnotation {
                kind: OpKind::Binary { op, lhs, rhs },
                self_on_right: !lhs_is_self && rhs_is_self,
            })
        }
        4 => {
            let lhs = match &tts[0] {
                TokenTree::Ident(i) => i.to_string(),
                _ => return Err(err("expected identifier for left operand")),
            };
            let c1 = match &tts[1] {
                TokenTree::Punct(p) => p.as_char(),
                _ => return Err(err("expected two-char operator")),
            };
            let c2 = match &tts[2] {
                TokenTree::Punct(p) => p.as_char(),
                _ => return Err(err("expected two-char operator")),
            };
            let rhs = match &tts[3] {
                TokenTree::Ident(i) => i.to_string(),
                _ => return Err(err("expected identifier for right operand")),
            };
            match (c1, c2) {
                ('=', '=') => {
                    let lhs_is_self = lhs == "Self";
                    let rhs_is_self = rhs == "Self";
                    if !lhs_is_self && !rhs_is_self {
                        return Err(err("at least one operand must be `Self`"));
                    }
                    Ok(OpAnnotation {
                        kind: OpKind::Binary { op: "eq", lhs, rhs },
                        self_on_right: !lhs_is_self && rhs_is_self,
                    })
                }
                ('!', '=') => Err(err(
                    "`!=` is auto-derived from `==` and cannot be declared separately",
                )),
                _ => Err(err("unsupported two-char operator; expected `==`")),
            }
        }
        _ => Err(err("invalid #[op(...)] syntax")),
    }
}

struct GetterInfo {
    field_name: String,
    method_ident: syn::Ident,
    classified: ClassifiedReturn,
}

struct SetterInfo {
    field_name: String,
    method_ident: syn::Ident,
    param_ident: syn::Ident,
    param_ty: TokenStream,
}

struct OpInfo {
    op_cap: &'static str,
    other_type: String,
    is_unary: bool,
    op_sym: &'static str,
}

struct HandlerOutput {
    handler_fn: TokenStream,
    handler_call: TokenStream,
}

struct InitResult {
    handler: HandlerOutput,
    field_decls: Vec<TokenStream>,
}

struct OpResult {
    handler: HandlerOutput,
    op_decl: TokenStream,
    dup_key: (&'static str, String, bool),
    op_sym: &'static str,
    is_unary: bool,
}

enum MethodDeclKind {
    Instance(TokenStream),
    Static(TokenStream),
}

struct MethodResult {
    handler: HandlerOutput,
    decl: MethodDeclKind,
}

pub fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_for_error = item.clone();
    match do_expand(attr, item) {
        Ok(ts) => ts,
        Err(e) => {
            let err = e.to_compile_error();
            quote! { #err #item_for_error }
        }
    }
}

fn do_expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: ExportMethodsArgs = syn::parse2(attr)?;
    let impl_block: ItemImpl = syn::parse2(item)?;

    let rust_type_ident = match &*impl_block.self_ty {
        Type::Path(type_path) if type_path.qself.is_none() => {
            type_path.path.get_ident().cloned().ok_or_else(|| {
                syn::Error::new_spanned(
                    &impl_block.self_ty,
                    "#[export_methods] requires a simple type path (e.g., `impl Point { ... }`)",
                )
            })?
        }
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "#[export_methods] requires a simple type path (e.g., `impl Point { ... }`)",
            ));
        }
    };

    let anvyx_name_str = args
        .name_override
        .unwrap_or_else(|| rust_type_ident.to_string());
    let rust_type_str = rust_type_ident.to_string();
    let methods_decl_ident = crate::naming::methods_decl_ident(&rust_type_str);
    let statics_decl_ident = crate::naming::statics_decl_ident(&rust_type_str);
    let ops_decl_ident = crate::naming::ops_decl_ident(&rust_type_str);
    let companion_fn_ident = crate::naming::methods_fn_ident(&rust_type_ident);

    let mut method_handler_fns = vec![];
    let mut method_handler_calls = vec![];
    let mut method_decls = vec![];
    let mut static_decls = vec![];
    let mut op_decls: Vec<TokenStream> = vec![];
    let mut seen_ops: Vec<(&'static str, String, bool)> = vec![];
    let mut getters: Vec<GetterInfo> = vec![];
    let mut found_init = false;
    let mut init_field_decls: Vec<TokenStream> = vec![];
    let mut setters: Vec<SetterInfo> = vec![];

    for impl_item in &impl_block.items {
        let ImplItem::Fn(method) = impl_item else {
            continue;
        };

        let has_getter = method.attrs.iter().any(|a| a.path().is_ident("getter"));
        let has_setter = method.attrs.iter().any(|a| a.path().is_ident("setter"));
        let has_init_attr = method.attrs.iter().any(|a| a.path().is_ident("init"));
        let has_op = method.attrs.iter().any(|a| a.path().is_ident("op"));

        if has_init_attr {
            if found_init {
                return Err(syn::Error::new_spanned(
                    &method.sig,
                    "only one #[init] method is allowed per type",
                ));
            }
            let result = process_init(method, &anvyx_name_str, &rust_type_ident, &rust_type_str)?;
            method_handler_fns.push(result.handler.handler_fn);
            method_handler_calls.push(result.handler.handler_call);
            init_field_decls = result.field_decls;
            found_init = true;
            continue;
        }

        if has_getter {
            getters.push(process_getter(method, &rust_type_ident)?);
            continue;
        }

        if has_setter {
            setters.push(process_setter(method, &rust_type_ident)?);
            continue;
        }

        if has_op {
            let op_attr = method
                .attrs
                .iter()
                .find(|a| a.path().is_ident("op"))
                .unwrap();
            let result = process_op(method, &anvyx_name_str, &rust_type_ident, &rust_type_str)?;
            if seen_ops.contains(&result.dup_key) {
                let sym = result.op_sym;
                let msg = if result.is_unary {
                    format!("duplicate operator `{sym}Self`")
                } else {
                    let self_on_right = result.dup_key.2;
                    let other_type = &result.dup_key.1;
                    let lhs_display = if self_on_right {
                        other_type.as_str()
                    } else {
                        anvyx_name_str.as_str()
                    };
                    let rhs_display = if self_on_right {
                        anvyx_name_str.as_str()
                    } else {
                        other_type.as_str()
                    };
                    format!("duplicate operator `{sym}` for `({lhs_display}, {rhs_display})`")
                };
                return Err(syn::Error::new_spanned(op_attr, msg));
            }
            seen_ops.push(result.dup_key);
            method_handler_fns.push(result.handler.handler_fn);
            method_handler_calls.push(result.handler.handler_call);
            op_decls.push(result.op_decl);
            continue;
        }

        let result = process_method(method, &anvyx_name_str, &rust_type_ident, &rust_type_str)?;
        method_handler_fns.push(result.handler.handler_fn);
        method_handler_calls.push(result.handler.handler_call);
        match result.decl {
            MethodDeclKind::Instance(d) => method_decls.push(d),
            MethodDeclKind::Static(d) => static_decls.push(d),
        }
    }

    // validate getter/setter pairs
    for g in &getters {
        let has_setter = setters.iter().any(|s| s.field_name == g.field_name);
        if !has_setter {
            return Err(syn::Error::new_spanned(
                &g.method_ident,
                format!(
                    "getter `{}` has no matching setter; add a `#[setter]` to expose it as a field, \
                     or remove `#[getter]` and keep it as a regular method instead",
                    g.field_name
                ),
            ));
        }
    }
    for s in &setters {
        let matching_getter = getters.iter().find(|g| g.field_name == s.field_name);
        match matching_getter {
            None => {
                return Err(syn::Error::new_spanned(
                    &s.method_ident,
                    format!(
                        "setter `set_{}` has no matching getter `{}`; a getter is required for every setter",
                        s.field_name, s.field_name
                    ),
                ));
            }
            Some(g) => {
                let getter_ty_str = match &g.classified.mode {
                    ReturnMode::Valued(ty) => ty.to_string(),
                    ReturnMode::Void => unreachable!(),
                };
                let types_match = matches!(g.classified.wrapper, ReturnWrapper::None)
                    && getter_ty_str == s.param_ty.to_string();
                if !types_match {
                    return Err(syn::Error::new_spanned(
                        &s.method_ident,
                        format!(
                            "setter `set_{}` type does not match getter `{}` return type",
                            s.field_name, s.field_name,
                        ),
                    ));
                }
            }
        }
    }

    // generate getter/setter handlers
    let mut getter_field_decls = vec![];
    for g in &getters {
        let method_ident = &g.method_ident;
        let field_name = &g.field_name;
        let get_key = format!("{anvyx_name_str}::__get_{field_name}");
        let get_fn_ident = format_ident!("__anvyx_method_{}_get_{}", rust_type_str, field_name);

        let sb = crate::codegen::build_self_borrow(&rust_type_ident, false, &get_key, 0);
        let self_extract = &sb.extraction;
        let call = quote! { __guard_self.#method_ident() };
        let call_body = build_handler_body(&call, &g.classified, &get_key, &[sb.borrow_param]);

        method_handler_fns.push(quote! {
            pub fn #get_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#get_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    #self_extract
                    #call_body
                }))
            }
        });
        method_handler_calls.push(quote! { #get_fn_ident() });

        let anvyx_type_tokens = crate::codegen::ret_anvyx_type_str(&g.classified);
        getter_field_decls.push(quote! {
            anvyx_lang::ExternFieldDecl { name: #field_name, ty: #anvyx_type_tokens, computed: true }
        });
    }
    for s in &setters {
        let method_ident = &s.method_ident;
        let field_name = &s.field_name;
        let param_ident = &s.param_ident;
        let param_ty = &s.param_ty;
        let set_key = format!("{anvyx_name_str}::__set_{field_name}");
        let set_fn_ident = format_ident!("__anvyx_method_{}_set_{}", rust_type_str, field_name);

        let sb = crate::codegen::build_self_borrow(&rust_type_ident, true, &set_key, 0);
        let self_extract = &sb.extraction;
        let extraction = quote! {
            let #param_ident = <#param_ty as anvyx_lang::AnvyxConvert>::from_anvyx(&args[1])?;
        };
        let call = quote! { __guard_self.#method_ident(#param_ident) };
        let void_return = ClassifiedReturn {
            mode: ReturnMode::Void,
            wrapper: ReturnWrapper::None,
        };
        let call_body = build_handler_body(&call, &void_return, &set_key, &[sb.borrow_param]);

        method_handler_fns.push(quote! {
            pub fn #set_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#set_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    #self_extract
                    #extraction
                    #call_body
                }))
            }
        });
        method_handler_calls.push(quote! { #set_fn_ident() });
    }

    let getter_fields_fn_ident = crate::naming::getter_fields_fn_ident(&rust_type_ident);
    let init_fields_fn_ident = crate::naming::init_fields_fn_ident(&rust_type_ident);
    let has_init_ident = crate::naming::has_init_ident(&rust_type_str);

    let mut cleaned_impl = impl_block.clone();
    for item in &mut cleaned_impl.items {
        if let ImplItem::Fn(method) = item {
            method.attrs.retain(|a| {
                !a.path().is_ident("getter")
                    && !a.path().is_ident("setter")
                    && !a.path().is_ident("init")
                    && !a.path().is_ident("op")
            });
        }
    }

    Ok(quote! {
        #cleaned_impl

        #(#method_handler_fns)*

        pub const #methods_decl_ident: &[anvyx_lang::ExternMethodDecl] = &[
            #(#method_decls),*
        ];

        pub const #statics_decl_ident: &[anvyx_lang::ExternStaticMethodDecl] = &[
            #(#static_decls),*
        ];

        #[allow(non_snake_case)]
        pub fn #companion_fn_ident() -> Vec<(&'static str, anvyx_lang::ExternHandler)> {
            vec![#(#method_handler_calls),*]
        }

        #[allow(non_snake_case)]
        pub fn #getter_fields_fn_ident() -> Vec<anvyx_lang::ExternFieldDecl> {
            vec![#(#getter_field_decls),*]
        }

        #[allow(non_snake_case)]
        pub fn #init_fields_fn_ident() -> Vec<anvyx_lang::ExternFieldDecl> {
            vec![#(#init_field_decls),*]
        }

        pub const #has_init_ident: bool = #found_init;

        pub const #ops_decl_ident: &[anvyx_lang::ExternOpDecl] = &[
            #(#op_decls),*
        ];
    })
}

fn process_init(
    method: &syn::ImplItemFn,
    anvyx_name_str: &str,
    rust_type_ident: &syn::Ident,
    rust_type_str: &str,
) -> syn::Result<InitResult> {
    let kind = classify_method_kind(&method.sig)?;
    if !matches!(kind, MethodKind::Static) {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[init] method must be a static method (no self parameter)",
        ));
    }
    let method_ident = &method.sig.ident;
    let handler_key = format!("{anvyx_name_str}::__init__");
    let handler_fn_ident = format_ident!("__anvyx_method_{}___init__", rust_type_str);

    let resolved_output =
        crate::codegen::resolve_self_in_return(&method.sig.output, rust_type_ident);
    let ret_mode = classify_return(&resolved_output);
    let returns_self =
        matches!(&ret_mode.mode, ReturnMode::Valued(ty) if *rust_type_ident == *ty.to_string());
    if !returns_self {
        return Err(syn::Error::new_spanned(
            &method.sig.output,
            "#[init] method must return Self",
        ));
    }
    if matches!(ret_mode.wrapper, ReturnWrapper::AnvyxOption) {
        return Err(syn::Error::new_spanned(
            &method.sig.output,
            "#[init] does not support Option return type",
        ));
    }

    let non_self_params: Vec<_> = method
        .sig
        .inputs
        .iter()
        .filter(|arg| matches!(arg, FnArg::Typed(_)))
        .collect();

    let mut param_extractions = vec![];
    let mut param_names = vec![];
    let mut field_decls = vec![];

    for (i, arg) in non_self_params.iter().enumerate() {
        let FnArg::Typed(pat_type) = arg else {
            unreachable!()
        };
        let param_name = match &*pat_type.pat {
            Pat::Ident(pi) => pi.ident.clone(),
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "only simple parameter names are supported in #[init]",
                ));
            }
        };
        let param_name_str = param_name.to_string();
        let resolved_ty = crate::codegen::resolve_self_in_type(&pat_type.ty, rust_type_ident);
        let mode = classify_param(&resolved_ty)
            .ok_or_else(|| syn::Error::new_spanned(&pat_type.ty, "unsupported type in #[init]"))?;
        match mode {
            ParamMode::Owned(ty) => {
                param_extractions.push(quote! {
                    let #param_name = <#ty as anvyx_lang::AnvyxConvert>::from_anvyx(&args[#i])?;
                });
                field_decls.push(quote! {
                    anvyx_lang::ExternFieldDecl {
                        name: #param_name_str,
                        ty: <#ty as anvyx_lang::AnvyxConvert>::ANVYX_TYPE,
                        computed: false,
                    }
                });
            }
            ParamMode::ExternRef(_) | ParamMode::ExternMutRef(_) => {
                return Err(syn::Error::new_spanned(
                    &pat_type.ty,
                    "#[init] does not support reference parameters; pass by value instead",
                ));
            }
        }
        param_names.push(param_name);
    }

    let call = quote! { #rust_type_ident::#method_ident(#(#param_names),*) };
    let handler_body = crate::codegen::build_flat_call(&call, &ret_mode);

    let handler_fn = quote! {
        pub fn #handler_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
            (#handler_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                #(#param_extractions)*
                #handler_body
            }))
        }
    };
    let handler_call = quote! { #handler_fn_ident() };

    Ok(InitResult {
        handler: HandlerOutput {
            handler_fn,
            handler_call,
        },
        field_decls,
    })
}

fn process_getter(
    method: &syn::ImplItemFn,
    rust_type_ident: &syn::Ident,
) -> syn::Result<GetterInfo> {
    let method_ident = method.sig.ident.clone();
    let kind = classify_method_kind(&method.sig)?;
    if !matches!(kind, MethodKind::Borrowing) {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[getter] method must take `&self`",
        ));
    }
    let non_self_count = method
        .sig
        .inputs
        .iter()
        .filter(|a| matches!(a, FnArg::Typed(_)))
        .count();
    if non_self_count != 0 {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[getter] method must have no parameters besides `&self`",
        ));
    }
    let resolved_output =
        crate::codegen::resolve_self_in_return(&method.sig.output, rust_type_ident);
    let ret_mode = classify_return(&resolved_output);
    if matches!(ret_mode.mode, ReturnMode::Void) {
        return Err(syn::Error::new_spanned(
            &method.sig.output,
            "#[getter] method must have a non-void return type",
        ));
    }
    Ok(GetterInfo {
        field_name: method_ident.to_string(),
        method_ident,
        classified: ret_mode,
    })
}

fn process_setter(
    method: &syn::ImplItemFn,
    rust_type_ident: &syn::Ident,
) -> syn::Result<SetterInfo> {
    let method_ident = method.sig.ident.clone();
    let method_name_str = method_ident.to_string();
    if !method_name_str.starts_with("set_") {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[setter] method name must start with `set_`",
        ));
    }
    let field_name = method_name_str["set_".len()..].to_string();
    let kind = classify_method_kind(&method.sig)?;
    if !matches!(kind, MethodKind::Mutating) {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[setter] method must take `&mut self`",
        ));
    }
    let non_self_params: Vec<_> = method
        .sig
        .inputs
        .iter()
        .filter(|a| matches!(a, FnArg::Typed(_)))
        .collect();
    if non_self_params.len() != 1 {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[setter] method must have exactly one parameter besides `&mut self`",
        ));
    }
    let FnArg::Typed(pat_type) = non_self_params[0] else {
        unreachable!()
    };
    let param_ident = match &*pat_type.pat {
        Pat::Ident(pi) => pi.ident.clone(),
        other => {
            return Err(syn::Error::new_spanned(
                other,
                "only simple parameter names are supported",
            ));
        }
    };
    let resolved_ty = crate::codegen::resolve_self_in_type(&pat_type.ty, rust_type_ident);
    let param_mode = classify_param(&resolved_ty)
        .ok_or_else(|| syn::Error::new_spanned(&pat_type.ty, "unsupported type in #[setter]"))?;
    let ParamMode::Owned(param_ty) = param_mode else {
        return Err(syn::Error::new_spanned(
            &pat_type.ty,
            "#[setter] does not support reference parameters; pass by value instead",
        ));
    };
    Ok(SetterInfo {
        field_name,
        method_ident,
        param_ident,
        param_ty,
    })
}

fn process_op(
    method: &syn::ImplItemFn,
    anvyx_name_str: &str,
    rust_type_ident: &syn::Ident,
    rust_type_str: &str,
) -> syn::Result<OpResult> {
    let op_attr = method
        .attrs
        .iter()
        .find(|a| a.path().is_ident("op"))
        .unwrap();
    let op_tokens = match &op_attr.meta {
        syn::Meta::List(list) => list.tokens.clone(),
        _ => {
            return Err(syn::Error::new_spanned(op_attr, "expected #[op(...)]"));
        }
    };
    let op_ann = parse_op_annotation(op_tokens)?;

    let kind = classify_method_kind(&method.sig)?;
    if !matches!(kind, MethodKind::Borrowing) {
        return Err(syn::Error::new_spanned(
            &method.sig,
            "#[op] method must take `&self`",
        ));
    }

    let method_ident = &method.sig.ident;

    let non_self_params: Vec<_> = method
        .sig
        .inputs
        .iter()
        .filter(|arg| matches!(arg, FnArg::Typed(_)))
        .collect();

    let self_on_right = op_ann.self_on_right;
    let (handler_key, ident_component, op_info) = match op_ann.kind {
        OpKind::Binary { op, lhs, rhs } => {
            if non_self_params.len() != 1 {
                return Err(syn::Error::new_spanned(
                    &method.sig,
                    "#[op] binary operator method must have exactly one non-self parameter",
                ));
            }
            let op_cap = op_to_capitalized(op);
            let op_sym = op_to_symbol(op);
            if self_on_right {
                let lhs_r = if lhs == "Self" {
                    anvyx_name_str.to_owned()
                } else {
                    lhs
                };
                (
                    format!("{anvyx_name_str}::__op_r{op}__{lhs_r}"),
                    format!("__op_r{op}__{lhs_r}"),
                    OpInfo {
                        op_cap,
                        other_type: lhs_r,
                        is_unary: false,
                        op_sym,
                    },
                )
            } else {
                let rhs_r = if rhs == "Self" {
                    anvyx_name_str.to_owned()
                } else {
                    rhs
                };
                (
                    format!("{anvyx_name_str}::__op_{op}__{rhs_r}"),
                    format!("__op_{op}__{rhs_r}"),
                    OpInfo {
                        op_cap,
                        other_type: rhs_r,
                        is_unary: false,
                        op_sym,
                    },
                )
            }
        }
        OpKind::Unary { op } => {
            if !non_self_params.is_empty() {
                return Err(syn::Error::new_spanned(
                    &method.sig,
                    "#[op] unary operator method must have no non-self parameters",
                ));
            }
            let op_cap = op_to_capitalized(op);
            let op_sym = op_to_symbol(op);
            (
                format!("{anvyx_name_str}::__op_{op}"),
                format!("__op_{op}"),
                OpInfo {
                    op_cap,
                    other_type: String::new(),
                    is_unary: true,
                    op_sym,
                },
            )
        }
    };

    let dup_key = (op_info.op_cap, op_info.other_type.clone(), self_on_right);

    let handler_fn_ident = format_ident!("__anvyx_method_{}_{}", rust_type_str, ident_component);

    let resolved_output =
        crate::codegen::resolve_self_in_return(&method.sig.output, rust_type_ident);
    let ret_mode = classify_return(&resolved_output);

    if op_info.op_cap == "Eq" {
        let is_bool = matches!(&ret_mode.mode, ReturnMode::Valued(ty) if ty.to_string() == "bool");
        if !is_bool {
            return Err(syn::Error::new_spanned(
                &method.sig.output,
                "#[op(Self == T)] must return bool",
            ));
        }
    }

    let ret_anvyx = crate::codegen::ret_anvyx_type_str(&ret_mode);
    let op_cap_str = op_info.op_cap;
    let other_type_str = &op_info.other_type;
    let op_decl = if op_info.is_unary {
        quote! {
            anvyx_lang::ExternOpDecl { op: #op_cap_str, rhs: None, lhs: None, ret: #ret_anvyx }
        }
    } else if self_on_right {
        quote! {
            anvyx_lang::ExternOpDecl { op: #op_cap_str, rhs: None, lhs: Some(#other_type_str), ret: #ret_anvyx }
        }
    } else {
        quote! {
            anvyx_lang::ExternOpDecl { op: #op_cap_str, rhs: Some(#other_type_str), lhs: None, ret: #ret_anvyx }
        }
    };

    let self_arg_idx: usize = usize::from(self_on_right);
    let other_arg_idx: usize = usize::from(!self_on_right);

    let sb = crate::codegen::build_self_borrow(rust_type_ident, false, &handler_key, self_arg_idx);

    let handler_label = format!("extern method '{handler_key}'");
    let extracted = crate::codegen::extract_params(
        &non_self_params,
        other_arg_idx,
        &handler_label,
        Some(rust_type_ident),
    )?;

    let param_names = extracted.param_names.as_slice();
    let param_extractions = extracted.extractions.as_slice();
    let mut all_borrows = vec![sb.borrow_param];
    all_borrows.extend(extracted.borrow_params);
    let call = quote! { __guard_self.#method_ident(#(#param_names),*) };
    let call_body = build_handler_body(&call, &ret_mode, &handler_key, &all_borrows);
    let self_extract = &sb.extraction;

    let handler_body = quote! {
        #self_extract
        #(#param_extractions)*
        #call_body
    };

    let handler_fn = quote! {
        pub fn #handler_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
            (#handler_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                #handler_body
            }))
        }
    };
    let handler_call = quote! { #handler_fn_ident() };

    Ok(OpResult {
        handler: HandlerOutput {
            handler_fn,
            handler_call,
        },
        op_decl,
        dup_key,
        op_sym: op_info.op_sym,
        is_unary: op_info.is_unary,
    })
}

fn process_method(
    method: &syn::ImplItemFn,
    anvyx_name_str: &str,
    rust_type_ident: &syn::Ident,
    rust_type_str: &str,
) -> syn::Result<MethodResult> {
    let method_ident = &method.sig.ident;
    let method_name_str = method_ident.to_string();
    let handler_key = format!("{anvyx_name_str}::{method_name_str}");
    let handler_fn_ident = format_ident!("__anvyx_method_{}_{}", rust_type_str, method_name_str);

    let kind = classify_method_kind(&method.sig)?;

    let resolved_output =
        crate::codegen::resolve_self_in_return(&method.sig.output, rust_type_ident);
    let ret_mode = classify_return(&resolved_output);

    let self_offset = match kind {
        MethodKind::Static => 0,
        _ => 1,
    };

    let non_self_params: Vec<_> = method
        .sig
        .inputs
        .iter()
        .filter(|arg| matches!(arg, FnArg::Typed(_)))
        .collect();

    let handler_label = format!("extern method '{handler_key}'");
    let extracted = crate::codegen::extract_params(
        &non_self_params,
        self_offset,
        &handler_label,
        Some(rust_type_ident),
    )?;

    let handler_body = {
        let param_names = extracted.param_names.as_slice();
        let extractions = extracted.extractions.as_slice();
        match kind {
            MethodKind::Static => {
                let call = quote! { #rust_type_ident::#method_ident(#(#param_names),*) };
                let call_body =
                    build_handler_body(&call, &ret_mode, &handler_key, &extracted.borrow_params);
                quote! {
                    #(#extractions)*
                    #call_body
                }
            }
            MethodKind::Borrowing | MethodKind::Mutating => {
                let is_mut = matches!(kind, MethodKind::Mutating);
                let sb =
                    crate::codegen::build_self_borrow(rust_type_ident, is_mut, &handler_key, 0);
                let mut all_borrows = vec![sb.borrow_param];
                all_borrows.extend(extracted.borrow_params);
                let call = quote! { __guard_self.#method_ident(#(#param_names),*) };
                let call_body = build_handler_body(&call, &ret_mode, &handler_key, &all_borrows);
                let self_extract = &sb.extraction;
                quote! {
                    #self_extract
                    #(#extractions)*
                    #call_body
                }
            }
        }
    };

    let handler_fn = quote! {
        pub fn #handler_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
            (#handler_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                #handler_body
            }))
        }
    };
    let handler_call = quote! { #handler_fn_ident() };

    let ret_anvyx_str = crate::codegen::ret_anvyx_type_str(&ret_mode);
    let method_doc_token = if let Some(s) = crate::codegen::extract_doc(&method.attrs) {
        quote! { Some(#s) }
    } else {
        quote! { None }
    };
    let param_anvyx_types = extracted.anvyx_types.as_slice();

    let decl = match kind {
        MethodKind::Static => MethodDeclKind::Static(quote! {
            anvyx_lang::ExternStaticMethodDecl {
                name: #method_name_str,
                doc: #method_doc_token,
                params: &[#(#param_anvyx_types),*],
                ret: #ret_anvyx_str,
            }
        }),
        MethodKind::Borrowing => MethodDeclKind::Instance(quote! {
            anvyx_lang::ExternMethodDecl {
                name: #method_name_str,
                doc: #method_doc_token,
                receiver: "self",
                params: &[#(#param_anvyx_types),*],
                ret: #ret_anvyx_str,
            }
        }),
        MethodKind::Mutating => MethodDeclKind::Instance(quote! {
            anvyx_lang::ExternMethodDecl {
                name: #method_name_str,
                doc: #method_doc_token,
                receiver: "var",
                params: &[#(#param_anvyx_types),*],
                ret: #ret_anvyx_str,
            }
        }),
    };

    Ok(MethodResult {
        handler: HandlerOutput {
            handler_fn,
            handler_call,
        },
        decl,
    })
}

fn classify_method_kind(sig: &syn::Signature) -> syn::Result<MethodKind> {
    match sig.inputs.first() {
        None | Some(FnArg::Typed(_)) => Ok(MethodKind::Static),
        Some(FnArg::Receiver(recv)) => {
            if recv.reference.is_none() {
                return Err(syn::Error::new_spanned(
                    recv,
                    "consuming self is not supported in #[export_methods]; use &self or &mut self",
                ));
            }
            if recv.mutability.is_some() {
                Ok(MethodKind::Mutating)
            } else {
                Ok(MethodKind::Borrowing)
            }
        }
    }
}

fn build_handler_body(
    call: &TokenStream,
    classified: &ClassifiedReturn,
    handler_key: &str,
    borrow_params: &[crate::codegen::BorrowParam],
) -> TokenStream {
    let handler_label = format!("extern method '{handler_key}'");
    crate::codegen::build_call_with_borrows(call, classified, &handler_label, borrow_params)
}

fn op_to_capitalized(op: &str) -> &'static str {
    match op {
        "add" => "Add",
        "sub" => "Sub",
        "mul" => "Mul",
        "div" => "Div",
        "rem" => "Rem",
        "eq" => "Eq",
        "neg" => "Neg",
        _ => unreachable!("unknown op: {op}"),
    }
}

fn op_to_symbol(op: &str) -> &'static str {
    match op {
        "add" => "+",
        "sub" | "neg" => "-",
        "mul" => "*",
        "div" => "/",
        "rem" => "%",
        "eq" => "==",
        _ => unreachable!("unknown op: {op}"),
    }
}
