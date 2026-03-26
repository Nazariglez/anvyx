use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{FnArg, ImplItem, ItemImpl, LitStr, Pat, ReturnType, Token, Type};

use crate::type_map::{ExternTypeInfo, ParamMode, ReturnMode, classify_param, classify_return};

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

struct BorrowParam {
    param_name: Option<syn::Ident>,
    type_ident: syn::Ident,
    handle_ident: syn::Ident,
    guard_ident: syn::Ident,
    is_mut: bool,
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
    let type_upper = rust_type_str.to_uppercase();
    let methods_decl_ident = format_ident!("__ANVYX_METHODS_DECL_{}", type_upper);
    let statics_decl_ident = format_ident!("__ANVYX_STATICS_DECL_{}", type_upper);
    let ops_decl_ident = format_ident!("__ANVYX_OPS_DECL_{}", type_upper);
    let type_decl_ident = format_ident!("__ANVYX_TYPE_DECL_{}", type_upper);
    let companion_fn_ident = format_ident!("__anvyx_methods_{}", rust_type_ident);

    struct GetterInfo {
        field_name: String,
        method_ident: syn::Ident,
        anvyx_type_str: &'static str,
        wrap_result: TokenStream,
    }

    struct SetterInfo {
        field_name: String,
        method_ident: syn::Ident,
        param_ident: syn::Ident,
        anvyx_type_str: &'static str,
        extract_variant: TokenStream,
        convert_extracted: TokenStream,
    }

    struct OpInfo {
        op_cap: &'static str,
        other_type: String,
        is_unary: bool,
        op_sym: &'static str,
    }

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
            let kind = classify_method_kind(&method.sig)?;
            if !matches!(kind, MethodKind::Static) {
                return Err(syn::Error::new_spanned(
                    &method.sig,
                    "#[init] method must be a static method (no self parameter)",
                ));
            }
            let method_ident = &method.sig.ident;
            let method_name_str = method_ident.to_string();
            let handler_key = format!("{}::__init__", anvyx_name_str);
            let handler_fn_ident = format_ident!("__anvyx_method_{}___init__", rust_type_str);

            let resolved_output = resolve_self_in_return(&method.sig.output, &rust_type_ident);
            let ret_mode = classify_return(&resolved_output).ok_or_else(|| {
                syn::Error::new_spanned(&method.sig.output, "#[init] method must return Self")
            })?;
            let returns_self = matches!(&ret_mode, ReturnMode::ExternOwned(info) if info.type_ident == rust_type_ident);
            if !returns_self {
                return Err(syn::Error::new_spanned(
                    &method.sig.output,
                    "#[init] method must return Self",
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
                let resolved_ty = resolve_self_in_type(&pat_type.ty, &rust_type_ident);
                let mode = classify_param(&resolved_ty).ok_or_else(|| {
                    syn::Error::new_spanned(&pat_type.ty, "unsupported type in #[init]")
                })?;
                match mode {
                    ParamMode::Primitive(mapping) => {
                        let extract_variant = &mapping.extract_variant;
                        let convert_extracted = &mapping.convert_extracted;
                        param_extractions.push(quote! {
                            let #extract_variant = args[#i].clone() else {
                                return Err(anvyx_lang::RuntimeError::new(format!(
                                    "expected correct type for parameter '{}' in extern method '{}'",
                                    #param_name_str, #handler_key
                                )));
                            };
                            let #param_name = #convert_extracted;
                        });
                        let anvyx_type_str = mapping.anvyx_type;
                        init_field_decls.push(quote! {
                            anvyx_lang::ExternFieldDecl { name: #param_name_str, ty: #anvyx_type_str, computed: false }
                        });
                    }
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &pat_type.ty,
                            "#[init] parameters must be primitive types (f64, i64, bool, or String)",
                        ));
                    }
                }
                param_names.push(param_name);
            }

            let call = quote! { #rust_type_ident::#method_ident(#(#param_names),*) };
            let handler_body = build_flat_call(&call, &ret_mode);
            let _ = method_name_str; // suppress unused warning

            method_handler_fns.push(quote! {
                pub fn #handler_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                    (#handler_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                        #(#param_extractions)*
                        #handler_body
                    }))
                }
            });
            method_handler_calls.push(quote! { #handler_fn_ident() });
            found_init = true;
            continue;
        }

        if has_getter {
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
            let resolved_output = resolve_self_in_return(&method.sig.output, &rust_type_ident);
            let ret_mode = classify_return(&resolved_output).ok_or_else(|| {
                syn::Error::new_spanned(&method.sig.output, "unsupported return type in #[getter]")
            })?;
            if matches!(ret_mode, ReturnMode::Void) {
                return Err(syn::Error::new_spanned(
                    &method.sig.output,
                    "#[getter] method must have a non-void return type",
                ));
            }
            let (anvyx_type_str, wrap_result) = match ret_mode {
                ReturnMode::Primitive(m) => (m.anvyx_type, m.wrap_result),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &method.sig.output,
                        "#[getter] return type must be a primitive (f64, i64, bool, or String)",
                    ));
                }
            };
            getters.push(GetterInfo {
                field_name: method_ident.to_string(),
                method_ident,
                anvyx_type_str,
                wrap_result,
            });
            continue;
        }

        if has_setter {
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
            let resolved_ty = resolve_self_in_type(&pat_type.ty, &rust_type_ident);
            let param_mode = classify_param(&resolved_ty).ok_or_else(|| {
                syn::Error::new_spanned(&pat_type.ty, "unsupported type in #[setter]")
            })?;
            let (anvyx_type_str, extract_variant, convert_extracted) = match param_mode {
                ParamMode::Primitive(m) => (m.anvyx_type, m.extract_variant, m.convert_extracted),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &pat_type.ty,
                        "#[setter] parameter type must be a primitive (f64, i64, bool, or String)",
                    ));
                }
            };
            setters.push(SetterInfo {
                field_name,
                method_ident,
                param_ident,
                anvyx_type_str,
                extract_variant,
                convert_extracted,
            });
            continue;
        }

        if has_op {
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
                            anvyx_name_str.clone()
                        } else {
                            lhs
                        };
                        (
                            format!("{}::__op_r{}__{}", anvyx_name_str, op, lhs_r),
                            format!("__op_r{}__{}", op, lhs_r),
                            OpInfo {
                                op_cap,
                                other_type: lhs_r,
                                is_unary: false,
                                op_sym,
                            },
                        )
                    } else {
                        let rhs_r = if rhs == "Self" {
                            anvyx_name_str.clone()
                        } else {
                            rhs
                        };
                        (
                            format!("{}::__op_{}__{}", anvyx_name_str, op, rhs_r),
                            format!("__op_{}__{}", op, rhs_r),
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
                        format!("{}::__op_{}", anvyx_name_str, op),
                        format!("__op_{}", op),
                        OpInfo {
                            op_cap,
                            other_type: String::new(),
                            is_unary: true,
                            op_sym,
                        },
                    )
                }
            };

            // duplicate op detection
            let dup_key = (op_info.op_cap, op_info.other_type.clone(), self_on_right);
            if seen_ops.contains(&dup_key) {
                let sym = op_info.op_sym;
                let msg = if op_info.is_unary {
                    format!("duplicate operator `{sym}Self`")
                } else {
                    let lhs_display = if self_on_right {
                        op_info.other_type.as_str()
                    } else {
                        anvyx_name_str.as_str()
                    };
                    let rhs_display = if self_on_right {
                        anvyx_name_str.as_str()
                    } else {
                        op_info.other_type.as_str()
                    };
                    format!("duplicate operator `{sym}` for `({lhs_display}, {rhs_display})`")
                };
                return Err(syn::Error::new_spanned(op_attr, msg));
            }
            seen_ops.push(dup_key);

            let handler_fn_ident =
                format_ident!("__anvyx_method_{}_{}", rust_type_str, ident_component);

            let resolved_output = resolve_self_in_return(&method.sig.output, &rust_type_ident);
            let ret_mode = classify_return(&resolved_output).ok_or_else(|| {
                syn::Error::new_spanned(&method.sig.output, "unsupported return type in #[op]")
            })?;

            // == must return bool
            if op_info.op_cap == "Eq" {
                let is_bool =
                    matches!(&ret_mode, ReturnMode::Primitive(m) if m.anvyx_type == "bool");
                if !is_bool {
                    return Err(syn::Error::new_spanned(
                        &method.sig.output,
                        "#[op(Self == T)] must return bool",
                    ));
                }
            }

            // collect op metadata for __ANVYX_OPS_DECL_ const
            let returns_self = matches!(&ret_mode, ReturnMode::ExternOwned(info) if info.type_ident == rust_type_ident);
            let ret_anvyx = build_ret_anvyx_str(&ret_mode, returns_self, &type_decl_ident);
            let op_cap_str = op_info.op_cap;
            let other_type_str = &op_info.other_type;
            if op_info.is_unary {
                op_decls.push(quote! {
                    anvyx_lang::ExternOpDecl { op: #op_cap_str, rhs: None, lhs: None, ret: #ret_anvyx }
                });
            } else if self_on_right {
                op_decls.push(quote! {
                    anvyx_lang::ExternOpDecl { op: #op_cap_str, rhs: None, lhs: Some(#other_type_str), ret: #ret_anvyx }
                });
            } else {
                op_decls.push(quote! {
                    anvyx_lang::ExternOpDecl { op: #op_cap_str, rhs: Some(#other_type_str), lhs: None, ret: #ret_anvyx }
                });
            }

            let self_arg_idx: usize = if self_on_right { 1 } else { 0 };
            let other_arg_idx: usize = if self_on_right { 0 } else { 1 };

            let self_extract = quote! {
                let anvyx_lang::Value::ExternHandle(ref __ehd_self) = args[#self_arg_idx] else {
                    return Err(anvyx_lang::RuntimeError::new(format!(
                        "expected extern handle for self in extern method '{}'",
                        #handler_key
                    )));
                };
                let __handle_self = __ehd_self.id;
            };

            let self_bp = BorrowParam {
                param_name: None,
                type_ident: rust_type_ident.clone(),
                handle_ident: format_ident!("__handle_self"),
                guard_ident: format_ident!("__guard_self"),
                is_mut: false,
            };

            let mut param_extractions = vec![];
            let mut param_names = vec![];
            let mut borrow_params: Vec<BorrowParam> = vec![];

            for (i, arg) in non_self_params.iter().enumerate() {
                let FnArg::Typed(pat_type) = arg else {
                    unreachable!()
                };
                let param_name = match &*pat_type.pat {
                    Pat::Ident(pi) => pi.ident.clone(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "only simple parameter names are supported in #[op]",
                        ));
                    }
                };
                let param_name_str = param_name.to_string();
                let arg_idx = other_arg_idx + i;

                let resolved_ty = resolve_self_in_type(&pat_type.ty, &rust_type_ident);
                let mode = classify_param(&resolved_ty).ok_or_else(|| {
                    syn::Error::new_spanned(&pat_type.ty, "unsupported type in #[op]")
                })?;

                match mode {
                    ParamMode::Primitive(mapping) => {
                        let extract_variant = &mapping.extract_variant;
                        let convert_extracted = &mapping.convert_extracted;
                        param_extractions.push(quote! {
                            let #extract_variant = args[#arg_idx].clone() else {
                                return Err(anvyx_lang::RuntimeError::new(format!(
                                    "expected correct type for parameter '{}' in extern method '{}'",
                                    #param_name_str, #handler_key
                                )));
                            };
                            let #param_name = #convert_extracted;
                        });
                    }
                    ParamMode::ValuePassthrough => {
                        param_extractions.push(quote! {
                            let #param_name = args[#arg_idx].clone();
                        });
                    }
                    ParamMode::ExternOwned(info) => {
                        let ty = &info.type_ident;
                        let handle_ident = format_ident!("__handle_{}", i);
                        param_extractions.push(quote! {
                            let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#arg_idx] else {
                                return Err(anvyx_lang::RuntimeError::new(format!(
                                    "expected extern handle for parameter '{}' in extern method '{}'",
                                    #param_name_str, #handler_key
                                )));
                            };
                            let #handle_ident = __ehd.id;
                            let #param_name = <#ty as anvyx_lang::AnvyxExternType>::with_store(|__s| __s.borrow_mut().remove(#handle_ident))?;
                        });
                    }
                    ParamMode::ExternRef(info) => {
                        let handle_ident = format_ident!("__handle_{}", i);
                        param_extractions.push(quote! {
                            let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#arg_idx] else {
                                return Err(anvyx_lang::RuntimeError::new(format!(
                                    "expected extern handle for parameter '{}' in extern method '{}'",
                                    #param_name_str, #handler_key
                                )));
                            };
                            let #handle_ident = __ehd.id;
                        });
                        let guard_ident = format_ident!("__guard_{}", i);
                        borrow_params.push(BorrowParam {
                            param_name: Some(param_name.clone()),
                            type_ident: info.type_ident,
                            handle_ident,
                            guard_ident,
                            is_mut: false,
                        });
                    }
                    ParamMode::ExternMutRef(info) => {
                        let handle_ident = format_ident!("__handle_{}", i);
                        param_extractions.push(quote! {
                            let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#arg_idx] else {
                                return Err(anvyx_lang::RuntimeError::new(format!(
                                    "expected extern handle for parameter '{}' in extern method '{}'",
                                    #param_name_str, #handler_key
                                )));
                            };
                            let #handle_ident = __ehd.id;
                        });
                        let guard_ident = format_ident!("__guard_{}", i);
                        borrow_params.push(BorrowParam {
                            param_name: Some(param_name.clone()),
                            type_ident: info.type_ident,
                            handle_ident,
                            guard_ident,
                            is_mut: true,
                        });
                    }
                }

                param_names.push(param_name);
            }

            let mut all_borrows = vec![self_bp];
            all_borrows.extend(borrow_params);
            let call = quote! { __guard_self.#method_ident(#(#param_names),*) };
            let call_body = build_handler_body(&call, &ret_mode, &handler_key, &all_borrows);

            let handler_body = quote! {
                #self_extract
                #(#param_extractions)*
                #call_body
            };

            method_handler_fns.push(quote! {
                pub fn #handler_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                    (#handler_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                        #handler_body
                    }))
                }
            });
            method_handler_calls.push(quote! { #handler_fn_ident() });

            continue;
        }

        let method_ident = &method.sig.ident;
        let method_name_str = method_ident.to_string();
        let handler_key = format!("{}::{}", anvyx_name_str, method_name_str);
        let handler_fn_ident =
            format_ident!("__anvyx_method_{}_{}", rust_type_str, method_name_str);

        let kind = classify_method_kind(&method.sig)?;

        let resolved_output = resolve_self_in_return(&method.sig.output, &rust_type_ident);

        let ret_mode = classify_return(&resolved_output).ok_or_else(|| {
            syn::Error::new_spanned(
                &method.sig.output,
                "unsupported return type in #[export_methods]",
            )
        })?;

        let returns_self = matches!(&ret_mode, ReturnMode::ExternOwned(info) if info.type_ident == rust_type_ident);

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

        let mut param_extractions = vec![];
        let mut param_names = vec![];
        let mut param_anvyx_types = vec![];
        let mut borrow_params: Vec<BorrowParam> = vec![];

        for (i, arg) in non_self_params.iter().enumerate() {
            let FnArg::Typed(pat_type) = arg else {
                unreachable!()
            };
            let param_name = match &*pat_type.pat {
                Pat::Ident(pi) => pi.ident.clone(),
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "only simple parameter names are supported in #[export_methods]",
                    ));
                }
            };

            let param_name_str = param_name.to_string();
            let arg_idx = self_offset + i;

            let resolved_ty = resolve_self_in_type(&pat_type.ty, &rust_type_ident);
            let mode = classify_param(&resolved_ty).ok_or_else(|| {
                syn::Error::new_spanned(&pat_type.ty, "unsupported type in #[export_methods]")
            })?;

            match mode {
                ParamMode::Primitive(mapping) => {
                    let extract_variant = &mapping.extract_variant;
                    let convert_extracted = &mapping.convert_extracted;
                    let anvyx_type_str = mapping.anvyx_type;
                    param_extractions.push(quote! {
                        let #extract_variant = args[#arg_idx].clone() else {
                            return Err(anvyx_lang::RuntimeError::new(format!(
                                "expected correct type for parameter '{}' in extern method '{}'",
                                #param_name_str, #handler_key
                            )));
                        };
                        let #param_name = #convert_extracted;
                    });
                    param_anvyx_types.push(quote! { (#param_name_str, #anvyx_type_str) });
                }
                ParamMode::ValuePassthrough => {
                    param_extractions.push(quote! {
                        let #param_name = args[#arg_idx].clone();
                    });
                    param_anvyx_types.push(quote! { (#param_name_str, "any") });
                }
                ParamMode::ExternOwned(info) => {
                    let ty = &info.type_ident;
                    let decl = &info.decl_ident;
                    let handle_ident = format_ident!("__handle_{}", i);
                    param_extractions.push(quote! {
                        let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#arg_idx] else {
                            return Err(anvyx_lang::RuntimeError::new(format!(
                                "expected extern handle for parameter '{}' in extern method '{}'",
                                #param_name_str, #handler_key
                            )));
                        };
                        let #handle_ident = __ehd.id;
                        let #param_name = <#ty as anvyx_lang::AnvyxExternType>::with_store(|__s| __s.borrow_mut().remove(#handle_ident))?;
                    });
                    param_anvyx_types.push(quote! { (#param_name_str, #decl.name) });
                }
                ParamMode::ExternRef(info) => {
                    let ExternTypeInfo { type_ident: ref_type, decl_ident: ref_decl } = info;
                    let handle_ident = format_ident!("__handle_{}", i);
                    param_extractions.push(quote! {
                        let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#arg_idx] else {
                            return Err(anvyx_lang::RuntimeError::new(format!(
                                "expected extern handle for parameter '{}' in extern method '{}'",
                                #param_name_str, #handler_key
                            )));
                        };
                        let #handle_ident = __ehd.id;
                    });
                    param_anvyx_types.push(quote! { (#param_name_str, #ref_decl.name) });
                    let guard_ident = format_ident!("__guard_{}", i);
                    borrow_params.push(BorrowParam {
                        param_name: Some(param_name.clone()),
                        type_ident: ref_type,
                        handle_ident,
                        guard_ident,
                        is_mut: false,
                    });
                }
                ParamMode::ExternMutRef(info) => {
                    let ExternTypeInfo { type_ident: ref_type, decl_ident: ref_decl } = info;
                    let handle_ident = format_ident!("__handle_{}", i);
                    param_extractions.push(quote! {
                        let anvyx_lang::Value::ExternHandle(ref __ehd) = args[#arg_idx] else {
                            return Err(anvyx_lang::RuntimeError::new(format!(
                                "expected extern handle for parameter '{}' in extern method '{}'",
                                #param_name_str, #handler_key
                            )));
                        };
                        let #handle_ident = __ehd.id;
                    });
                    param_anvyx_types.push(quote! { (#param_name_str, #ref_decl.name) });
                    let guard_ident = format_ident!("__guard_{}", i);
                    borrow_params.push(BorrowParam {
                        param_name: Some(param_name.clone()),
                        type_ident: ref_type,
                        handle_ident,
                        guard_ident,
                        is_mut: true,
                    });
                }
            }

            param_names.push(param_name);
        }

        let handler_body = match kind {
            MethodKind::Static => {
                let call = quote! { #rust_type_ident::#method_ident(#(#param_names),*) };
                let call_body = build_handler_body(&call, &ret_mode, &handler_key, &borrow_params);
                quote! {
                    #(#param_extractions)*
                    #call_body
                }
            }
            MethodKind::Borrowing => {
                let self_extract = quote! {
                    let anvyx_lang::Value::ExternHandle(ref __ehd_self) = args[0] else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected extern handle for self in extern method '{}'",
                            #handler_key
                        )));
                    };
                    let __handle_self = __ehd_self.id;
                };
                let self_bp = BorrowParam {
                    param_name: None,
                    type_ident: rust_type_ident.clone(),
                    handle_ident: format_ident!("__handle_self"),
                    guard_ident: format_ident!("__guard_self"),
                    is_mut: false,
                };
                let mut all_borrows = vec![self_bp];
                all_borrows.extend(borrow_params);
                let call = quote! { __guard_self.#method_ident(#(#param_names),*) };
                let call_body = build_handler_body(&call, &ret_mode, &handler_key, &all_borrows);
                quote! {
                    #self_extract
                    #(#param_extractions)*
                    #call_body
                }
            }
            MethodKind::Mutating => {
                let self_extract = quote! {
                    let anvyx_lang::Value::ExternHandle(ref __ehd_self) = args[0] else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected extern handle for self in extern method '{}'",
                            #handler_key
                        )));
                    };
                    let __handle_self = __ehd_self.id;
                };
                let self_bp = BorrowParam {
                    param_name: None,
                    type_ident: rust_type_ident.clone(),
                    handle_ident: format_ident!("__handle_self"),
                    guard_ident: format_ident!("__guard_self"),
                    is_mut: true,
                };
                let mut all_borrows = vec![self_bp];
                all_borrows.extend(borrow_params);
                let call = quote! { __guard_self.#method_ident(#(#param_names),*) };
                let call_body = build_handler_body(&call, &ret_mode, &handler_key, &all_borrows);
                quote! {
                    #self_extract
                    #(#param_extractions)*
                    #call_body
                }
            }
        };

        method_handler_fns.push(quote! {
            pub fn #handler_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#handler_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    #handler_body
                }))
            }
        });
        method_handler_calls.push(quote! { #handler_fn_ident() });

        let ret_anvyx_str = build_ret_anvyx_str(&ret_mode, returns_self, &type_decl_ident);

        match classify_method_kind(&method.sig)? {
            MethodKind::Static => {
                static_decls.push(quote! {
                    anvyx_lang::ExternStaticMethodDecl {
                        name: #method_name_str,
                        params: &[#(#param_anvyx_types),*],
                        ret: #ret_anvyx_str,
                    }
                });
            }
            MethodKind::Borrowing => {
                method_decls.push(quote! {
                    anvyx_lang::ExternMethodDecl {
                        name: #method_name_str,
                        receiver: "self",
                        params: &[#(#param_anvyx_types),*],
                        ret: #ret_anvyx_str,
                    }
                });
            }
            MethodKind::Mutating => {
                method_decls.push(quote! {
                    anvyx_lang::ExternMethodDecl {
                        name: #method_name_str,
                        receiver: "var",
                        params: &[#(#param_anvyx_types),*],
                        ret: #ret_anvyx_str,
                    }
                });
            }
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
            Some(g) if g.anvyx_type_str != s.anvyx_type_str => {
                return Err(syn::Error::new_spanned(
                    &s.method_ident,
                    format!(
                        "setter `set_{}` parameter type `{}` does not match getter `{}` return type `{}`",
                        s.field_name, s.anvyx_type_str, s.field_name, g.anvyx_type_str,
                    ),
                ));
            }
            _ => {}
        }
    }

    // generate getter/setter handlers
    let mut getter_field_decls = vec![];
    for g in &getters {
        let method_ident = &g.method_ident;
        let field_name = &g.field_name;
        let get_key = format!("{}::__get_{}", anvyx_name_str, field_name);
        let get_fn_ident = format_ident!("__anvyx_method_{}_get_{}", rust_type_str, field_name);
        let wrap_result = &g.wrap_result;

        method_handler_fns.push(quote! {
            pub fn #get_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#get_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    let anvyx_lang::Value::ExternHandle(ref __ehd_self) = args[0] else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected extern handle for self in extern method '{}'",
                            #get_key
                        )));
                    };
                    let __handle_self = __ehd_self.id;
                    <#rust_type_ident as anvyx_lang::AnvyxExternType>::with_store(|__store| {
                        let __borrow = __store.borrow();
                        let __guard_self = __borrow.borrow(__handle_self).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle for self in extern method '{}': {}",
                                #get_key, e.message
                            ))
                        })?;
                        let result = __guard_self.#method_ident();
                        Ok(#wrap_result)
                    })
                }))
            }
        });
        method_handler_calls.push(quote! { #get_fn_ident() });

        let anvyx_type = g.anvyx_type_str;
        getter_field_decls.push(quote! {
            anvyx_lang::ExternFieldDecl { name: #field_name, ty: #anvyx_type, computed: true }
        });
    }
    for s in &setters {
        let method_ident = &s.method_ident;
        let field_name = &s.field_name;
        let param_ident = &s.param_ident;
        let set_key = format!("{}::__set_{}", anvyx_name_str, field_name);
        let set_fn_ident = format_ident!("__anvyx_method_{}_set_{}", rust_type_str, field_name);
        let extract_variant = &s.extract_variant;
        let convert_extracted = &s.convert_extracted;

        method_handler_fns.push(quote! {
            pub fn #set_fn_ident() -> (&'static str, anvyx_lang::ExternHandler) {
                (#set_key, Box::new(|args: Vec<anvyx_lang::Value>| {
                    let anvyx_lang::Value::ExternHandle(ref __ehd_self) = args[0] else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "expected extern handle for self in extern method '{}'",
                            #set_key
                        )));
                    };
                    let __handle_self = __ehd_self.id;
                    let #extract_variant = args[1].clone() else {
                        return Err(anvyx_lang::RuntimeError::new(format!(
                            "invalid argument type for '{}'",
                            #set_key
                        )));
                    };
                    let #param_ident = #convert_extracted;
                    <#rust_type_ident as anvyx_lang::AnvyxExternType>::with_store(|__store| {
                        let __borrow = __store.borrow();
                        let mut __guard_self = __borrow.borrow_mut(__handle_self).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle for self in extern method '{}': {}",
                                #set_key, e.message
                            ))
                        })?;
                        __guard_self.#method_ident(#param_ident);
                        Ok(anvyx_lang::Value::Nil)
                    })
                }))
            }
        });
        method_handler_calls.push(quote! { #set_fn_ident() });
    }

    let getter_fields_fn_ident = format_ident!("__anvyx_getter_fields_{}", rust_type_ident);
    let init_fields_fn_ident = format_ident!("__anvyx_init_fields_{}", rust_type_ident);
    let has_init_ident = format_ident!("__ANVYX_HAS_INIT_{}", type_upper);

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

fn resolve_self_in_return(output: &ReturnType, type_ident: &syn::Ident) -> ReturnType {
    match output {
        ReturnType::Default => ReturnType::Default,
        ReturnType::Type(arrow, ty) => {
            let resolved = resolve_self_in_type(ty, type_ident);
            ReturnType::Type(*arrow, Box::new(resolved))
        }
    }
}

fn resolve_self_in_type(ty: &Type, type_ident: &syn::Ident) -> Type {
    match ty {
        Type::Path(path) if path.qself.is_none() => {
            if let Some(ident) = path.path.get_ident() {
                if ident == "Self" {
                    return Type::Path(syn::TypePath {
                        qself: None,
                        path: type_ident.clone().into(),
                    });
                }
            }
            ty.clone()
        }
        Type::Reference(ref_type) => {
            let resolved_elem = resolve_self_in_type(&ref_type.elem, type_ident);
            Type::Reference(syn::TypeReference {
                and_token: ref_type.and_token,
                lifetime: ref_type.lifetime.clone(),
                mutability: ref_type.mutability,
                elem: Box::new(resolved_elem),
            })
        }
        _ => ty.clone(),
    }
}

fn build_ret_anvyx_str(
    mode: &ReturnMode,
    returns_self: bool,
    type_decl_ident: &syn::Ident,
) -> TokenStream {
    match mode {
        ReturnMode::Void => quote! { "void" },
        ReturnMode::Primitive(m) => {
            let s = m.anvyx_type;
            quote! { #s }
        }
        ReturnMode::ExternOwned(_) if returns_self => {
            quote! { #type_decl_ident.name }
        }
        ReturnMode::ExternOwned(info) => {
            let di = &info.decl_ident;
            quote! { #di.name }
        }
        ReturnMode::ValuePassthrough => quote! { "any" },
    }
}

fn build_flat_call(call: &TokenStream, mode: &ReturnMode) -> TokenStream {
    match mode {
        ReturnMode::Void => quote! {
            #call;
            Ok(anvyx_lang::Value::Nil)
        },
        ReturnMode::Primitive(m) => {
            let wrap = &m.wrap_result;
            quote! {
                let result = #call;
                Ok(#wrap)
            }
        }
        ReturnMode::ValuePassthrough => quote! {
            let result = #call;
            Ok(result)
        },
        ReturnMode::ExternOwned(info) => {
            let ty = &info.type_ident;
            quote! {
                let result = #call;
                Ok(anvyx_lang::extern_handle::<#ty>(result))
            }
        }
    }
}

fn build_handler_body(
    call: &TokenStream,
    ret_mode: &ReturnMode,
    handler_key: &str,
    borrow_params: &[BorrowParam],
) -> TokenStream {
    if borrow_params.is_empty() {
        return build_flat_call(call, ret_mode);
    }

    let is_void = matches!(ret_mode, ReturnMode::Void);
    let innermost = if is_void {
        quote! { #call; Ok(()) }
    } else {
        quote! { Ok(#call) }
    };

    struct StoreGroup<'a> {
        type_ident: &'a syn::Ident,
        params: Vec<&'a BorrowParam>,
    }

    let mut groups: Vec<StoreGroup> = vec![];
    for bp in borrow_params {
        match groups.iter_mut().find(|g| g.type_ident == &bp.type_ident) {
            Some(group) => group.params.push(bp),
            None => groups.push(StoreGroup {
                type_ident: &bp.type_ident,
                params: vec![bp],
            }),
        }
    }

    let mut current = innermost;
    for group in groups.iter().rev() {
        let type_ident = group.type_ident;

        let borrow_stmts: Vec<TokenStream> = group
            .params
            .iter()
            .map(|bp| {
                let handle = &bp.handle_ident;
                let guard = &bp.guard_ident;

                if let Some(param_name) = &bp.param_name {
                    let param_str = param_name.to_string();
                    if bp.is_mut {
                        quote! {
                            let mut #guard = __borrow.borrow_mut(#handle).map_err(|e| {
                                anvyx_lang::RuntimeError::new(format!(
                                    "invalid handle for parameter '{}' in extern method '{}': {}",
                                    #param_str, #handler_key, e.message
                                ))
                            })?;
                            let #param_name = &mut *#guard;
                        }
                    } else {
                        quote! {
                            let #guard = __borrow.borrow(#handle).map_err(|e| {
                                anvyx_lang::RuntimeError::new(format!(
                                    "invalid handle for parameter '{}' in extern method '{}': {}",
                                    #param_str, #handler_key, e.message
                                ))
                            })?;
                            let #param_name = &*#guard;
                        }
                    }
                } else if bp.is_mut {
                    quote! {
                        let mut #guard = __borrow.borrow_mut(#handle).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle for self in extern method '{}': {}",
                                #handler_key, e.message
                            ))
                        })?;
                    }
                } else {
                    quote! {
                        let #guard = __borrow.borrow(#handle).map_err(|e| {
                            anvyx_lang::RuntimeError::new(format!(
                                "invalid handle for self in extern method '{}': {}",
                                #handler_key, e.message
                            ))
                        })?;
                    }
                }
            })
            .collect();

        current = quote! {
            <#type_ident as anvyx_lang::AnvyxExternType>::with_store(|__store| {
                let __borrow = __store.borrow();
                #(#borrow_stmts)*
                #current
            })
        };
    }

    match ret_mode {
        ReturnMode::Void => quote! {
            #current?;
            Ok(anvyx_lang::Value::Nil)
        },
        ReturnMode::Primitive(m) => {
            let wrap = &m.wrap_result;
            quote! {
                let result = #current?;
                Ok(#wrap)
            }
        }
        ReturnMode::ValuePassthrough => quote! {
            let result = #current?;
            Ok(result)
        },
        ReturnMode::ExternOwned(info) => {
            let ty = &info.type_ident;
            quote! {
                let result = #current?;
                Ok(anvyx_lang::extern_handle::<#ty>(result))
            }
        }
    }
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
        "sub" => "-",
        "mul" => "*",
        "div" => "/",
        "rem" => "%",
        "eq" => "==",
        "neg" => "-",
        _ => unreachable!("unknown op: {op}"),
    }
}
