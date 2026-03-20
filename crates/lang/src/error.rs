use std::ops::Range;

use crate::{
    lexer::{SpannedToken, Token},
    resolve::ImportError,
    typecheck::{TypeErr, TypeErrKind},
};
use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::error::{Rich, RichPattern, RichReason};

pub fn report_lexer_errors(src: &str, file_path: &str, errors: Vec<Rich<'_, char>>) {
    for e in errors {
        let span = e.span();
        let byte_range = span.start..span.end;

        let last_context = last_ctx(&e)
            .map(|s| format!("while lexing a {}", s))
            .unwrap_or_default();

        let custom_msg = match e.reason() {
            RichReason::Custom(msg) => Some(msg.to_string()),
            _ => None,
        };

        let (msg_title, msg_body) = if let Some(msg) = custom_msg {
            (msg, String::new())
        } else if let Some(found_char) = e.found() {
            (
                format!("Unexpected character {}", last_context),
                format!("'{}'", found_char),
            )
        } else {
            (
                format!("Unexpected end of input {}", last_context),
                "end of file".to_string(),
            )
        };

        emit_report(
            src,
            file_path,
            (byte_range, msg_title, msg_body),
            vec![],
            vec![],
            None,
        );
    }
}

pub fn report_parse_errors(
    src: &str,
    file_path: &str,
    tokens: &[SpannedToken],
    errors: Vec<Rich<SpannedToken>>,
) {
    for e in errors {
        let token_span = e.span();

        let byte_range = token_span_to_byte_range(tokens, token_span.start..token_span.end);

        let custom_msg = match e.reason() {
            RichReason::Custom(msg) => Some(msg.to_string()),
            _ => None,
        };

        let last_context = last_ctx(&e)
            .map(|s| format!("while parsing a {}", s))
            .unwrap_or_default();

        let (msg_title, msg_body) = if let Some(msg) = custom_msg {
            (msg, String::new())
        } else if let Some((found_token, _)) = e.found() {
            let token_desc = describe_token(found_token);
            (format!("Unexpected token {}", last_context), token_desc)
        } else {
            (
                format!("Unexpected end of input {}", last_context),
                "end of file".to_string(),
            )
        };

        emit_report(
            src,
            file_path,
            (byte_range, msg_title, msg_body),
            vec![],
            vec![],
            None,
        );
    }
}

pub fn report_typecheck_errors(
    src: &str,
    file_path: &str,
    tokens: &[SpannedToken],
    errors: Vec<TypeErr>,
) {
    for e in errors {
        let byte_range = token_span_to_byte_range(tokens, e.span.start..e.span.end);
        let (title, body) = format_type_error(&e.kind);

        let secondary: Vec<_> = e
            .secondary
            .iter()
            .map(|(span, msg)| {
                let range = token_span_to_byte_range(tokens, span.start..span.end);
                (range, msg.clone())
            })
            .collect();

        emit_report(
            src,
            file_path,
            (byte_range, title, body),
            secondary,
            e.notes.clone(),
            e.help.clone(),
        );
    }
}

pub fn report_import_errors(
    src: &str,
    file_path: &str,
    tokens: &[SpannedToken],
    errors: &[ImportError],
) {
    for e in errors {
        match e {
            ImportError::FileNotFound { path, span } => {
                let byte_range = token_span_to_byte_range(tokens, span.start..span.end);
                emit_report(
                    src,
                    file_path,
                    (
                        byte_range,
                        format!("Cannot find module file '{path}'"),
                        "import path cannot be resolved to a file".to_string(),
                    ),
                    vec![],
                    vec![],
                    None,
                );
            }
            ImportError::ParseError { file_path: imported_path } => {
                emit_report(
                    src,
                    file_path,
                    (
                        0..0,
                        format!("Failed to parse imported module '{imported_path}'"),
                        "the imported file contains errors".to_string(),
                    ),
                    vec![],
                    vec![],
                    None,
                );
            }
            ImportError::CircularImport { path, span } => {
                let byte_range = token_span_to_byte_range(tokens, span.start..span.end);
                emit_report(
                    src,
                    file_path,
                    (
                        byte_range,
                        format!("Circular import detected for '{path}'"),
                        "this import creates a cycle".to_string(),
                    ),
                    vec![],
                    vec![],
                    None,
                );
            }
            ImportError::UnknownStdModule { name, span } => {
                let byte_range = token_span_to_byte_range(tokens, span.start..span.end);
                emit_report(
                    src,
                    file_path,
                    (
                        byte_range,
                        format!("Unknown standard library module 'std.{name}'"),
                        "no std module with this name exists".to_string(),
                    ),
                    vec![],
                    vec![],
                    None,
                );
            }
        }
    }
}

fn format_type_error(kind: &TypeErrKind) -> (String, String) {
    match kind {
        TypeErrKind::UnknownVariable { name } => (
            format!("Unknown variable '{name}'"),
            "This variable is not in scope".to_string(),
        ),
        TypeErrKind::UnknownFunction { name } => (
            format!("Unknown function '{name}'"),
            "This function is not defined in this scope".to_string(),
        ),
        TypeErrKind::MismatchedTypes { expected, found } => (
            "Mismatched types".to_string(),
            format!("expected '{expected}', found '{found}'"),
        ),
        TypeErrKind::InvalidOperand { op, operand_type } => (
            "Invalid operand type".to_string(),
            format!("operator '{op}' cannot be applied to '{operand_type}'"),
        ),
        TypeErrKind::NotAFunction { expr_type } => (
            "Not a function".to_string(),
            format!("expression of type '{expr_type}' is not callable"),
        ),
        TypeErrKind::UnresolvedInfer => (
            "Could not infer type".to_string(),
            "type inference could not resolve this expression".to_string(),
        ),
        TypeErrKind::GenericArgNumMismatch { expected, found } => (
            "Wrong number of type arguments".to_string(),
            format!("expected {expected} type argument(s), found {found}"),
        ),
        TypeErrKind::NotGenericFunction => (
            "Function is not generic".to_string(),
            "type arguments cannot be provided for a non-generic function".to_string(),
        ),
        TypeErrKind::IfConditionNotBool { found } => (
            "Condition of if expression must be bool".to_string(),
            format!("found '{found}'"),
        ),
        TypeErrKind::IfMissingElse => (
            "if expression used as value must have an else branch".to_string(),
            "add an else branch to use this if expression as a value".to_string(),
        ),
        TypeErrKind::WhileConditionNotBool { found } => (
            "Condition of while must be bool".to_string(),
            format!("found '{found}'"),
        ),
        TypeErrKind::BreakOutsideLoop => (
            "break outside of loop".to_string(),
            "break can only be used inside a loop".to_string(),
        ),
        TypeErrKind::ContinueOutsideLoop => (
            "continue outside of loop".to_string(),
            "continue can only be used inside a loop".to_string(),
        ),
        TypeErrKind::TupleIndexOnNonTuple { found, index } => (
            format!("cannot index non-tuple type with .{index}"),
            format!("expression has type '{found}', which is not a tuple"),
        ),
        TypeErrKind::TupleIndexOutOfBounds {
            tuple_type,
            index,
            len,
        } => (
            format!("tuple index {index} is out of bounds"),
            format!(
                "tuple type '{tuple_type}' has {len} elements (indices 0..{})",
                len - 1
            ),
        ),
        TypeErrKind::TuplePatternArityMismatch { expected, found } => (
            "tuple pattern arity mismatch".to_string(),
            format!("expected {expected} elements, found {found}"),
        ),
        TypeErrKind::NonTupleInTuplePattern {
            found,
            pattern_arity,
        } => (
            format!("cannot destructure non-tuple type with {pattern_arity}-element pattern"),
            format!("expression has type '{found}', which is not a tuple"),
        ),
        TypeErrKind::TuplePatternLabelMismatch { expected, found } => (
            "tuple pattern label mismatch".to_string(),
            format!("expected label '{expected}', found '{found}'"),
        ),
        TypeErrKind::NamedPatternOnPositionalTuple => (
            "cannot use named tuple pattern on positional tuple".to_string(),
            "use positional pattern '(a, b, ...)' instead".to_string(),
        ),
        TypeErrKind::DuplicateTupleLabel { label } => (
            format!("duplicate field '{label}' in named tuple"),
            "each field label must be unique".to_string(),
        ),
        TypeErrKind::NoSuchFieldOnTuple { field, tuple_type } => (
            format!("tuple has no field '{field}'"),
            format!("type '{tuple_type}' does not contain a field named '{field}'"),
        ),
        TypeErrKind::FieldAccessOnNonNamedTuple { field, found } => (
            format!("cannot access field '{field}' on non-named tuple"),
            format!("type '{found}' has no named fields; use '.0', '.1', ... instead"),
        ),
        TypeErrKind::UnknownStruct { name } => (
            format!("Unknown struct '{name}'"),
            "This struct is not defined in this scope".to_string(),
        ),
        TypeErrKind::StructMissingField { struct_name, field } => (
            format!("Missing field '{field}' in struct literal"),
            format!("struct '{struct_name}' requires field '{field}'"),
        ),
        TypeErrKind::StructUnknownField { struct_name, field } => (
            format!("Unknown field '{field}' for struct '{struct_name}'"),
            format!("struct '{struct_name}' has no field named '{field}'"),
        ),
        TypeErrKind::StructDuplicateField { struct_name, field } => (
            format!("Duplicate field '{field}' in struct literal"),
            format!("field '{field}' is specified more than once in '{struct_name}'"),
        ),
        TypeErrKind::UnknownMethod {
            struct_name,
            method,
        } => (
            format!("Unknown method '{method}' for struct '{struct_name}'"),
            format!("struct '{struct_name}' has no method named '{method}'"),
        ),
        TypeErrKind::StaticMethodOnValue {
            struct_name,
            method,
        } => (
            format!("Cannot call static method '{method}' on a value"),
            format!("'{method}' is a static method; call it as '{struct_name}.{method}(...)'"),
        ),
        TypeErrKind::InstanceMethodOnType {
            struct_name,
            method,
        } => (
            format!("Cannot call instance method '{method}' on a type"),
            format!("'{method}' requires an instance of '{struct_name}'; create one first"),
        ),
        TypeErrKind::ReadonlySelfMutation { struct_name, field } => (
            format!("Cannot assign to field '{field}' in readonly method"),
            format!(
                "'self' is readonly in this method of '{struct_name}'; use 'var self' for mutating methods"
            ),
        ),
        TypeErrKind::ForIterableNotSupported { found } => (
            "type is not iterable".to_string(),
            format!("found '{found}'; expected a range, array, list, view, or map"),
        ),
        TypeErrKind::ForStepNotInt { item_ty, step_ty } => (
            "step is only supported for integer ranges".to_string(),
            format!("item type is '{item_ty}', step type is '{step_ty}'; both must be 'int'"),
        ),
        TypeErrKind::ForMapStepNotAllowed => (
            "step is not supported for map iteration".to_string(),
            "maps do not have a meaningful index stride; remove the 'step' clause".to_string(),
        ),
        TypeErrKind::ForMapRevNotAllowed => (
            "rev is not supported for map iteration".to_string(),
            "map iteration order is unspecified; remove the 'rev' keyword".to_string(),
        ),
        TypeErrKind::ArrayAllNilAmbiguous => (
            "cannot infer element type for all-nil array literal".to_string(),
            "try adding a type annotation, e.g. `var a: [int?; _] = [nil, nil];`".to_string(),
        ),
        TypeErrKind::ArrayFillLengthNotLiteral => (
            "array fill length must be an integer literal".to_string(),
            "the length in `[expr; len]` must be a compile-time integer literal".to_string(),
        ),
        TypeErrKind::IndexOnNonArray { found } => (
            "cannot index non-array type".to_string(),
            format!("expression has type '{found}', which is not an array or list"),
        ),
        TypeErrKind::IndexNotInt { found } => (
            "index must be an integer".to_string(),
            format!("expected 'int', found '{found}'"),
        ),
        TypeErrKind::OptionalChainingOnNonOpt { found } => (
            "optional chaining requires an optional base type".to_string(),
            format!(
                "found '{found}', which is not optional; remove the '?' or make the base type optional"
            ),
        ),

        TypeErrKind::UnknownEnum { name } => (
            format!("Unknown enum '{name}'"),
            "This enum is not defined in this scope".to_string(),
        ),
        TypeErrKind::UnknownEnumVariant {
            enum_name,
            variant_name,
        } => (
            format!("Unknown variant '{variant_name}' for enum '{enum_name}'"),
            format!("enum '{enum_name}' has no variant named '{variant_name}'"),
        ),
        TypeErrKind::EnumVariantArityMismatch {
            enum_name,
            variant_name,
            expected,
            found,
        } => (
            format!("Wrong number of arguments for variant '{enum_name}.{variant_name}'"),
            format!("expected {expected} argument(s), found {found}"),
        ),
        TypeErrKind::EnumVariantNotTuple {
            enum_name,
            variant_name,
        } => (
            format!("'{enum_name}.{variant_name}' is not a tuple variant"),
            "cannot use function call syntax on this variant".to_string(),
        ),
        TypeErrKind::EnumVariantNotStruct {
            enum_name,
            variant_name,
        } => (
            format!("'{enum_name}.{variant_name}' is not a struct variant"),
            "cannot use struct literal syntax on this variant".to_string(),
        ),
        TypeErrKind::EnumVariantNotUnit {
            enum_name,
            variant_name,
        } => (
            format!("'{enum_name}.{variant_name}' is not a unit variant"),
            "this variant requires arguments".to_string(),
        ),
        TypeErrKind::EnumVariantMissingField {
            enum_name,
            variant_name,
            field,
        } => (
            format!("Missing field '{field}' in variant '{enum_name}.{variant_name}'"),
            format!("variant '{variant_name}' requires field '{field}'"),
        ),
        TypeErrKind::EnumVariantUnknownField {
            enum_name,
            variant_name,
            field,
        } => (
            format!("Unknown field '{field}' for variant '{enum_name}.{variant_name}'"),
            format!("variant '{variant_name}' has no field named '{field}'"),
        ),
        TypeErrKind::EnumVariantDuplicateField {
            enum_name,
            variant_name,
            field,
        } => (
            format!("Duplicate field '{field}' in variant '{enum_name}.{variant_name}'"),
            format!("field '{field}' is specified more than once"),
        ),

        TypeErrKind::MatchScrutineeNotEnum { found } => (
            "match scrutinee must be an enum type".to_string(),
            format!("found '{found}'"),
        ),
        TypeErrKind::NonExhaustiveMatch { missing } => {
            let missing_str = missing
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            (
                "non-exhaustive match".to_string(),
                format!("missing variants: {missing_str}"),
            )
        }
        TypeErrKind::MatchArmTypeMismatch { expected, found } => (
            "match arm type mismatch".to_string(),
            format!("expected '{expected}', found '{found}'"),
        ),
        TypeErrKind::MatchPatternEnumMismatch {
            expected_enum,
            pattern_enum,
        } => (
            "pattern does not match scrutinee enum".to_string(),
            format!("scrutinee is '{expected_enum}', but pattern uses '{pattern_enum}'"),
        ),

        TypeErrKind::ImmutableAssignment { name } => (
            format!("Cannot assign to immutable variable '{name}'"),
            format!("'{name}' is declared with 'let'; use 'var' to allow mutation"),
        ),
        TypeErrKind::VarParamNotLvalue { param } => (
            format!("Cannot pass non-lvalue to 'var' parameter '{param}'"),
            format!("'var' parameter '{param}' requires a variable, not a literal or expression"),
        ),
        TypeErrKind::VarParamImmutableBinding { param, binding } => (
            format!("Cannot pass immutable binding '{binding}' to 'var' parameter '{param}'"),
            format!("'{binding}' is declared with 'let'; use 'var' to allow mutation"),
        ),
        TypeErrKind::MutatingMethodOnImmutable {
            struct_name,
            method,
        } => (
            format!("Cannot call mutating method '{method}' on immutable binding"),
            format!(
                "'{method}' on '{struct_name}' requires 'var self'; declare the binding with 'var'"
            ),
        ),

        TypeErrKind::MapEmptyLiteralNoContext => (
            "cannot infer key/value types of empty map".to_string(),
            "add a type annotation, e.g. `let m: [string: int] = [:];`".to_string(),
        ),
        TypeErrKind::MapKeyFloat => (
            "float cannot be used as map key".to_string(),
            "float keys are unreliable due to NaN and precision; use int or string instead"
                .to_string(),
        ),
        TypeErrKind::MapKeyNotKeyable { found } => (
            "type cannot be used as map key".to_string(),
            format!("'{found}' is not keyable; map keys must be int, bool, string, or structs/enums/tuples whose fields are all keyable"),
        ),
        TypeErrKind::MapOptionalKeyNotAllowed { found } => (
            "optional type cannot be used as map key".to_string(),
            format!("'{found}' is optional; map keys cannot be optional"),
        ),
        TypeErrKind::MapDuplicateKey => (
            "duplicate key in map literal".to_string(),
            "this key appears more than once".to_string(),
        ),
        TypeErrKind::InvalidCast { from, to } => (
            "Invalid cast".to_string(),
            format!("cannot cast '{from}' to '{to}'"),
        ),
        TypeErrKind::MethodTypeParamShadowsStruct { struct_name, method, param } => (
            "method type parameter shadows struct type parameter".to_string(),
            format!("method '{method}' on struct '{struct_name}' declares type parameter '{param}' which shadows a struct type parameter"),
        ),
        TypeErrKind::NotEquatable { ty } => (
            "type is not equatable".to_string(),
            format!("type '{ty}' does not support '==' or '!='"),
        ),
        TypeErrKind::UnknownModuleMember { module, member } => (
            format!("Unknown member '{member}' in module '{module}'"),
            format!("module '{module}' does not have a member named '{member}'"),
        ),
        TypeErrKind::PrivateModuleMember { module, member } => (
            format!("'{member}' is private in module '{module}'"),
            format!("'{member}' exists in module '{module}' but is not marked `pub`"),
        ),
        TypeErrKind::ReExportCollision { name, first_source, second_source } => (
            format!("symbol '{name}' re-exported by both '{first_source}' and '{second_source}'"),
            format!("'{name}' is introduced by 'pub import' from '{first_source}' and also from '{second_source}'"),
        ),
        TypeErrKind::AnyTypeNotAllowed => (
            "'any' type is only allowed in extern function declarations".to_string(),
            "'any' cannot be used in user-written type annotations".to_string(),
        ),
    }
}

fn token_span_to_byte_range(tokens: &[SpannedToken], span: Range<usize>) -> Range<usize> {
    let start_byte = tokens.get(span.start).map(|(_, s)| s.start).unwrap_or(0);
    let last_tok_idx = span.end.saturating_sub(1);
    let end_byte = tokens
        .get(last_tok_idx)
        .map(|(_, s)| s.end)
        .unwrap_or(start_byte);

    start_byte..end_byte
}

fn last_ctx<T>(ctx: &Rich<'_, T>) -> Option<String> {
    ctx.contexts()
        .filter_map(|(pat, span)| match pat {
            RichPattern::Label(s) => Some((s.as_ref(), span)),
            _ => None,
        })
        .last()
        .map(|(s, _)| s.to_string())
}

fn emit_report(
    src: &str,
    file_path: &str,
    primary: (Range<usize>, String, String),
    secondary: Vec<(Range<usize>, String)>,
    notes: Vec<String>,
    help: Option<String>,
) {
    let (range, title, label) = primary;

    let mut report = Report::build(ReportKind::Error, (file_path, range.clone()))
        .with_message(title)
        .with_label(
            Label::new((file_path, range))
                .with_color(Color::Red)
                .with_message(label),
        );

    for (sec_range, sec_msg) in secondary {
        report = report.with_label(
            Label::new((file_path, sec_range))
                .with_color(Color::Blue)
                .with_message(sec_msg),
        );
    }

    for note in notes {
        report = report.with_note(note);
    }

    if let Some(h) = help {
        report = report.with_help(h);
    }

    let _ = report.finish().print((file_path, Source::from(src)));
}

fn describe_token(token: &Token) -> String {
    match token {
        Token::Keyword(keyword) => format!("'{}' keyword", keyword),
        Token::Open(..) | Token::Close(..) => format!("'{}' delimiter", token),
        Token::Ident(_) => "identifier".to_string(),
        Token::Literal(lit) => format!("'{}' literal", lit),
        Token::Op(op) => format!("'{}' operator", op),
        _ => format!("'{}' token", token),
    }
}
