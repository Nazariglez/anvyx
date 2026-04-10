use std::ops::Range;

use chumsky::error::{Rich, RichPattern, RichReason};

use crate::{
    conditional::CondError,
    diagnostic::{self, Severity},
    intrinsic::{IntrinsicDiagnostic, IntrinsicDiagnosticLevel, IntrinsicError},
    lexer::{SpannedToken, Token},
    resolve::ImportError,
    span::Span,
    typecheck::{Diagnostic, DiagnosticKind},
};

fn build_unexpected_msg(
    custom_msg: Option<String>,
    found_desc: Option<String>,
    last_context: &str,
    unexpected_label: &str,
) -> (String, String) {
    if let Some(msg) = custom_msg {
        (msg, String::new())
    } else if let Some(desc) = found_desc {
        (
            format!("Unexpected {unexpected_label} {last_context}"),
            desc,
        )
    } else {
        (
            format!("Unexpected end of input {last_context}"),
            "end of file".to_string(),
        )
    }
}

pub fn collect_lexer_diagnostics(
    file_path: &str,
    errors: Vec<Rich<'_, char>>,
) -> Vec<diagnostic::Diagnostic> {
    errors
        .into_iter()
        .map(|e| {
            let span = e.span();
            let last_context = last_ctx(&e)
                .map(|s| format!("while lexing a {s}"))
                .unwrap_or_default();
            let custom_msg = match e.reason() {
                RichReason::Custom(msg) => Some(msg.clone()),
                RichReason::ExpectedFound { .. } => None,
            };
            let found_desc = e.found().map(|c| format!("'{c}'"));
            let (title, body) =
                build_unexpected_msg(custom_msg, found_desc, &last_context, "character");
            make_plain_diagnostic(
                file_path,
                Span::new(span.start, span.end),
                Severity::Error,
                title,
                body,
            )
        })
        .collect()
}

pub fn collect_parse_diagnostics(
    file_path: &str,
    tokens: &[SpannedToken],
    errors: Vec<Rich<SpannedToken>>,
) -> Vec<diagnostic::Diagnostic> {
    errors
        .into_iter()
        .map(|e| {
            let token_span = e.span();
            let custom_msg = match e.reason() {
                RichReason::Custom(msg) => Some(msg.clone()),
                RichReason::ExpectedFound { .. } => None,
            };
            let last_context = last_ctx(&e)
                .map(|s| format!("while parsing a {s}"))
                .unwrap_or_default();
            let found_desc = e.found().map(|(tok, _)| describe_token(tok));
            let (title, body) =
                build_unexpected_msg(custom_msg, found_desc, &last_context, "token");
            make_plain_diagnostic(
                file_path,
                token_span_to_span(tokens, token_span.start..token_span.end),
                Severity::Error,
                title,
                body,
            )
        })
        .collect()
}

pub fn collect_typecheck_diagnostics(
    file_path: &str,
    tokens: &[SpannedToken],
    errors: Vec<Diagnostic>,
) -> Vec<diagnostic::Diagnostic> {
    errors
        .into_iter()
        .map(|e| {
            let (title, body) = format_diagnostic(&e.kind);
            let severity = e.kind.severity();
            let related = e
                .secondary
                .into_iter()
                .map(|(span, msg)| diagnostic::DiagnosticLabel {
                    file: file_path.to_string(),
                    span: token_span_to_span(tokens, span.start..span.end),
                    message: msg,
                })
                .collect();
            make_simple_diagnostic(
                file_path,
                token_span_to_span(tokens, e.span.start..e.span.end),
                severity,
                title,
                body,
                related,
                e.notes,
                e.help,
            )
        })
        .collect()
}

pub fn collect_import_diagnostics(
    file_path: &str,
    tokens: &[SpannedToken],
    errors: &[ImportError],
) -> Vec<diagnostic::Diagnostic> {
    errors
        .iter()
        .map(|e| match e {
            ImportError::FileNotFound { path, span } => make_plain_diagnostic(
                file_path,
                token_span_to_span(tokens, span.start..span.end),
                Severity::Error,
                format!("Cannot find module file '{path}'"),
                "import path cannot be resolved to a file".to_string(),
            ),
            ImportError::UnknownStdModule { name, span } => make_plain_diagnostic(
                file_path,
                token_span_to_span(tokens, span.start..span.end),
                Severity::Error,
                format!("Unknown standard library module 'std.{name}'"),
                "no std module with this name exists".to_string(),
            ),
        })
        .collect()
}

pub fn collect_conditional_diagnostics(
    file_path: &str,
    errors: &[CondError],
) -> Vec<diagnostic::Diagnostic> {
    errors
        .iter()
        .map(|err| {
            let span = err.span();
            let help = match err {
                CondError::UnknownPredicate { .. } => {
                    Some("valid predicates are: profile, os, arch, feature".to_string())
                }
                CondError::UnknownPredicateArg {
                    valid, predicate, ..
                } => {
                    let valid_list = valid.join(", ");
                    Some(format!("valid values for {predicate}() are: {valid_list}"))
                }
                CondError::ElifAfterElse { .. } => {
                    Some("#elif must appear before #else".to_string())
                }
                _ => None,
            };
            make_simple_diagnostic(
                file_path,
                Span::new(span.start, span.end),
                Severity::Error,
                err.to_string(),
                String::new(),
                vec![],
                vec![],
                help,
            )
        })
        .collect()
}

pub fn collect_intrinsic_diagnostics(
    file_path: &str,
    tokens: &[SpannedToken],
    diagnostics: &[IntrinsicDiagnostic],
) -> Vec<diagnostic::Diagnostic> {
    diagnostics
        .iter()
        .map(|diag| {
            let (severity, prefix) = match diag.level {
                IntrinsicDiagnosticLevel::Error => (Severity::Error, "#error"),
                IntrinsicDiagnosticLevel::Warning => (Severity::Warning, "#warning"),
                IntrinsicDiagnosticLevel::Note => (Severity::Note, "#log"),
            };
            make_plain_diagnostic(
                file_path,
                token_span_to_span(tokens, diag.span.start..diag.span.end),
                severity,
                format!("{prefix}: {}", diag.message),
                String::new(),
            )
        })
        .collect()
}

pub fn collect_intrinsic_errors(
    file_path: &str,
    tokens: &[SpannedToken],
    errors: &[IntrinsicError],
) -> Vec<diagnostic::Diagnostic> {
    errors
        .iter()
        .map(|err| {
            let span = err.span();
            make_plain_diagnostic(
                file_path,
                token_span_to_span(tokens, span.start..span.end),
                Severity::Error,
                err.to_string(),
                String::new(),
            )
        })
        .collect()
}

fn make_plain_diagnostic(
    file_path: &str,
    span: Span,
    severity: Severity,
    title: String,
    body: String,
) -> diagnostic::Diagnostic {
    make_simple_diagnostic(file_path, span, severity, title, body, vec![], vec![], None)
}

fn make_simple_diagnostic(
    file_path: &str,
    span: Span,
    severity: Severity,
    title: String,
    body: String,
    related: Vec<diagnostic::DiagnosticLabel>,
    notes: Vec<String>,
    help: Option<String>,
) -> diagnostic::Diagnostic {
    diagnostic::Diagnostic {
        severity,
        message: title,
        primary: diagnostic::DiagnosticLabel {
            file: file_path.to_string(),
            span,
            message: body,
        },
        related,
        notes,
        help,
    }
}

fn format_diagnostic(kind: &DiagnosticKind) -> (String, String) {
    match kind {
        DiagnosticKind::UnknownVariable { name } => (
            format!("Unknown variable '{name}'"),
            "This variable is not in scope".to_string(),
        ),
        DiagnosticKind::UnknownFunction { name } => (
            format!("Unknown function '{name}'"),
            "This function is not defined in this scope".to_string(),
        ),
        DiagnosticKind::MismatchedTypes { expected, found } => (
            "Mismatched types".to_string(),
            format!("expected '{expected}', found '{found}'"),
        ),
        DiagnosticKind::InvalidOperand { op, operand_type } => (
            "Invalid operand type".to_string(),
            format!("operator '{op}' cannot be applied to '{operand_type}'"),
        ),
        DiagnosticKind::NotAFunction { expr_type } => (
            "Not a function".to_string(),
            format!("expression of type '{expr_type}' is not callable"),
        ),
        DiagnosticKind::UnresolvedInfer => (
            "Could not infer type".to_string(),
            "type inference could not resolve this expression".to_string(),
        ),
        DiagnosticKind::GenericArgNumMismatch { expected, found } => (
            "Wrong number of type arguments".to_string(),
            format!("expected {expected} type argument(s), found {found}"),
        ),
        DiagnosticKind::ConflictingConstInference { first, second } => (
            "Conflicting const inference".to_string(),
            format!("const parameter inferred as both {first} and {second}"),
        ),
        DiagnosticKind::NotGenericFunction => (
            "Function is not generic".to_string(),
            "type arguments cannot be provided for a non-generic function".to_string(),
        ),
        DiagnosticKind::IfConditionNotBool { found } => (
            "Condition of if expression must be bool".to_string(),
            format!("found '{found}'"),
        ),
        DiagnosticKind::IfMissingElse => (
            "if expression used as value must have an else branch".to_string(),
            "add an else branch to use this if expression as a value".to_string(),
        ),
        DiagnosticKind::WhileConditionNotBool { found } => (
            "Condition of while must be bool".to_string(),
            format!("found '{found}'"),
        ),
        DiagnosticKind::BreakOutsideLoop => (
            "break outside of loop".to_string(),
            "break can only be used inside a loop".to_string(),
        ),
        DiagnosticKind::ContinueOutsideLoop => (
            "continue outside of loop".to_string(),
            "continue can only be used inside a loop".to_string(),
        ),
        DiagnosticKind::TupleIndexOnNonTuple { found, index } => (
            format!("cannot index non-tuple type with .{index}"),
            format!("expression has type '{found}', which is not a tuple"),
        ),
        DiagnosticKind::TupleIndexOutOfBounds {
            tuple_type,
            index,
            len,
        } => (
            format!("tuple index {index} is out of bounds"),
            format!(
                "tuple type '{tuple_type}' has {len} elements (indices 0..{})",
                len.saturating_sub(1)
            ),
        ),
        DiagnosticKind::TuplePatternArityMismatch { expected, found } => (
            "tuple pattern arity mismatch".to_string(),
            format!("expected {expected} elements, found {found}"),
        ),
        DiagnosticKind::NonTupleInTuplePattern {
            found,
            pattern_arity,
        } => (
            format!("cannot destructure non-tuple type with {pattern_arity}-element pattern"),
            format!("expression has type '{found}', which is not a tuple"),
        ),
        DiagnosticKind::TuplePatternLabelMismatch { expected, found } => (
            "tuple pattern label mismatch".to_string(),
            format!("expected label '{expected}', found '{found}'"),
        ),
        DiagnosticKind::NamedPatternOnPositionalTuple => (
            "cannot use named tuple pattern on positional tuple".to_string(),
            "use positional pattern '(a, b, ...)' instead".to_string(),
        ),
        DiagnosticKind::DuplicateTupleLabel { label } => (
            format!("duplicate field '{label}' in named tuple"),
            "each field label must be unique".to_string(),
        ),
        DiagnosticKind::NoSuchFieldOnTuple { field, tuple_type } => (
            format!("tuple has no field '{field}'"),
            format!("type '{tuple_type}' does not contain a field named '{field}'"),
        ),
        DiagnosticKind::FieldAccessOnNonNamedTuple { field, found } => (
            format!("cannot access field '{field}' on non-named tuple"),
            format!("type '{found}' has no named fields; use '.0', '.1', ... instead"),
        ),
        DiagnosticKind::UnknownStruct { name } => (
            format!("Unknown struct '{name}'"),
            "This struct is not defined in this scope".to_string(),
        ),
        DiagnosticKind::StructMissingField { kind, struct_name, field } => (
            format!("Missing field '{field}' in {kind} literal"),
            format!("{kind} '{struct_name}' requires field '{field}'"),
        ),
        DiagnosticKind::StructUnknownField { kind, struct_name, field } => (
            format!("Unknown field '{field}' for {kind} '{struct_name}'"),
            format!("{kind} '{struct_name}' has no field named '{field}'"),
        ),
        DiagnosticKind::ExternUnknownField { type_name, field } => (
            format!("Unknown field '{field}' on extern type '{type_name}'"),
            format!("extern type '{type_name}' has no field named '{field}'"),
        ),
        DiagnosticKind::ExternUnknownMethod { type_name, method } => (
            format!("Unknown method '{method}' on extern type '{type_name}'"),
            format!("extern type '{type_name}' has no method named '{method}'"),
        ),
        DiagnosticKind::ExternInitNoInit { type_name } => (
            format!("Type '{type_name}' does not support struct literal construction"),
            format!("use a static method like '{type_name}.new(...)' instead"),
        ),
        DiagnosticKind::ExternInitMissingField { type_name, field } => (
            format!("Missing field '{field}' in '{type_name}' literal"),
            format!("type '{type_name}' requires field '{field}'"),
        ),
        DiagnosticKind::ExternInitUnknownField { type_name, field } => (
            format!("Unknown field '{field}' for type '{type_name}'"),
            format!("type '{type_name}' has no field named '{field}'"),
        ),
        DiagnosticKind::ExternInitDuplicateField { type_name, field } => (
            format!("Duplicate field '{field}' in '{type_name}' literal"),
            format!("field '{field}' is specified more than once"),
        ),
        DiagnosticKind::StructDestructureUnknownField { type_name, field } => (
            format!("Unknown field '{field}' in destructure of '{type_name}'"),
            format!("type '{type_name}' has no field named '{field}'"),
        ),
        DiagnosticKind::StructDestructureDuplicateField { type_name, field } => (
            format!("Duplicate field '{field}' in destructure of '{type_name}'"),
            format!("field '{field}' is specified more than once"),
        ),
        DiagnosticKind::StructDuplicateField { kind, struct_name, field } => (
            format!("Duplicate field '{field}' in {kind} literal"),
            format!("field '{field}' is specified more than once in '{struct_name}'"),
        ),
        DiagnosticKind::FieldDefaultNotConst { kind, struct_name, field } => (
            "default field value must be a constant expression".to_string(),
            format!("field '{field}' on {kind} '{struct_name}' has a non-constant default"),
        ),
        DiagnosticKind::FieldDefaultTypeMismatch { kind, struct_name, field, expected, found } => (
            format!("default value type mismatch for field '{field}'"),
            format!("field '{field}' on {kind} '{struct_name}': expected '{expected}', found '{found}'"),
        ),
        DiagnosticKind::FieldDefaultOnGenericType { kind, struct_name, field } => (
            "default values are not allowed on fields with generic types".to_string(),
            format!("field '{field}' on {kind} '{struct_name}' has a generic type"),
        ),
        DiagnosticKind::UnknownMethod {
            kind,
            struct_name,
            method,
        } => (
            format!("Unknown method '{method}' for {kind} '{struct_name}'"),
            format!("{kind} '{struct_name}' has no method named '{method}'"),
        ),
        DiagnosticKind::StaticMethodOnValue {
            struct_name,
            method,
        } => (
            format!("Cannot call static method '{method}' on a value"),
            format!("'{method}' is a static method; call it as '{struct_name}.{method}(...)'"),
        ),
        DiagnosticKind::InstanceMethodOnType {
            struct_name,
            method,
        } => (
            format!("Cannot call instance method '{method}' on a type"),
            format!("'{method}' requires an instance of '{struct_name}'; create one first"),
        ),
        DiagnosticKind::ReadonlySelfMutation { struct_name, field } => (
            format!("Cannot assign to field '{field}' in readonly method"),
            format!(
                "'self' is readonly in this method of '{struct_name}'; use 'var self' for mutating methods"
            ),
        ),
        DiagnosticKind::InvalidToStringSignature { kind, struct_name, reason } => (
            format!("Invalid 'to_string' method on {kind} '{struct_name}'"),
            reason.clone(),
        ),
        DiagnosticKind::ForIterableNotSupported { found } => (
            "type is not iterable".to_string(),
            format!("found '{found}'; expected a range, array, list, slice, or map"),
        ),
        DiagnosticKind::ForStepNotInt { item_ty, step_ty } => (
            "step is only supported for integer ranges".to_string(),
            format!("item type is '{item_ty}', step type is '{step_ty}'; both must be 'int'"),
        ),
        DiagnosticKind::ForMapStepNotAllowed => (
            "step is not supported for map iteration".to_string(),
            "maps do not have a meaningful index stride; remove the 'step' clause".to_string(),
        ),
        DiagnosticKind::ForMapRevNotAllowed => (
            "rev is not supported for map iteration".to_string(),
            "map iteration order is unspecified; remove the 'rev' keyword".to_string(),
        ),
        DiagnosticKind::ForRangeFromRevNotAllowed => (
            "cannot reverse an open-ended range (start..)".to_string(),
            "open-ended ranges have no end to start from; remove the 'rev' keyword".to_string(),
        ),
        DiagnosticKind::ArrayAllNilAmbiguous => (
            "cannot infer element type for all-nil array literal".to_string(),
            "try adding a type annotation, e.g. `var a: [int?; _] = [nil, nil];`".to_string(),
        ),
        DiagnosticKind::ArrayFillLengthNotLiteral => (
            "array fill length must be a compile-time constant".to_string(),
            "the length in `[expr; len]` must be an integer literal or a const with an integer value".to_string(),
        ),
        DiagnosticKind::IndexOnNonArray { found } => (
            "cannot index non-array type".to_string(),
            format!("expression has type '{found}', which is not an array or list"),
        ),
        DiagnosticKind::IndexNotInt { found } => (
            "index must be an integer".to_string(),
            format!("expected 'int', found '{found}'"),
        ),
        DiagnosticKind::RangeIndexNotInt { found } => (
            "range index bounds must be integers".to_string(),
            format!("range element type is '{found}', expected 'int'"),
        ),
        DiagnosticKind::RangeIndexOnMap => (
            "maps do not support range indexing".to_string(),
            "range slicing is only supported on arrays, lists, and slices".to_string(),
        ),
        DiagnosticKind::OptionalChainingOnNonOpt { found } => (
            "optional chaining requires an optional base type".to_string(),
            format!(
                "found '{found}', which is not optional; remove the '?' or make the base type optional"
            ),
        ),

        DiagnosticKind::UnknownEnum { name } => (
            format!("Unknown enum '{name}'"),
            "This enum is not defined in this scope".to_string(),
        ),
        DiagnosticKind::UnknownEnumVariant {
            enum_name,
            variant_name,
        } => (
            format!("Unknown variant '{variant_name}' for enum '{enum_name}'"),
            format!("enum '{enum_name}' has no variant named '{variant_name}'"),
        ),
        DiagnosticKind::EnumVariantArityMismatch {
            enum_name,
            variant_name,
            expected,
            found,
        } => (
            format!("Wrong number of arguments for variant '{enum_name}.{variant_name}'"),
            format!("expected {expected} argument(s), found {found}"),
        ),
        DiagnosticKind::EnumVariantNotTuple {
            enum_name,
            variant_name,
        } => (
            format!("'{enum_name}.{variant_name}' is not a tuple variant"),
            "cannot use function call syntax on this variant".to_string(),
        ),
        DiagnosticKind::EnumVariantNotStruct {
            enum_name,
            variant_name,
        } => (
            format!("'{enum_name}.{variant_name}' is not a struct variant"),
            "cannot use struct literal syntax on this variant".to_string(),
        ),
        DiagnosticKind::EnumVariantNotUnit {
            enum_name,
            variant_name,
        } => (
            format!("'{enum_name}.{variant_name}' is not a unit variant"),
            "this variant requires arguments".to_string(),
        ),
        DiagnosticKind::EnumVariantMissingField {
            enum_name,
            variant_name,
            field,
        } => (
            format!("Missing field '{field}' in variant '{enum_name}.{variant_name}'"),
            format!("variant '{variant_name}' requires field '{field}'"),
        ),
        DiagnosticKind::EnumVariantUnknownField {
            enum_name,
            variant_name,
            field,
        } => (
            format!("Unknown field '{field}' for variant '{enum_name}.{variant_name}'"),
            format!("variant '{variant_name}' has no field named '{field}'"),
        ),
        DiagnosticKind::EnumVariantDuplicateField {
            enum_name,
            variant_name,
            field,
        } => (
            format!("Duplicate field '{field}' in variant '{enum_name}.{variant_name}'"),
            format!("field '{field}' is specified more than once"),
        ),

        DiagnosticKind::UnsupportedMatchScrutinee { found } => (
            "unsupported match scrutinee type".to_string(),
            format!("type '{found}' cannot be used as a match scrutinee"),
        ),
        DiagnosticKind::InvalidLiteralPattern { expected, found } => (
            "invalid literal pattern".to_string(),
            format!("expected '{expected}', found '{found}'"),
        ),
        DiagnosticKind::NonExhaustiveMatchNoCatchAll => (
            "non-exhaustive match".to_string(),
            "match on this type requires a catch-all (`_` or variable binding) pattern"
                .to_string(),
        ),
        DiagnosticKind::NonExhaustiveMatch { missing } => {
            let missing_str = missing
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            (
                "non-exhaustive match".to_string(),
                format!("missing variants: {missing_str}"),
            )
        }
        DiagnosticKind::MatchArmTypeMismatch { expected, found } => (
            "match arm type mismatch".to_string(),
            format!("expected '{expected}', found '{found}'"),
        ),
        DiagnosticKind::MatchPatternEnumMismatch {
            expected_enum,
            pattern_enum,
        } => (
            "pattern does not match scrutinee enum".to_string(),
            format!("scrutinee is '{expected_enum}', but pattern uses '{pattern_enum}'"),
        ),

        DiagnosticKind::NilPatternOnNonOptional { found } => (
            "'nil' pattern requires an optional type".to_string(),
            format!("found '{found}', which is not optional"),
        ),

        DiagnosticKind::OptionalPatternOnNonOptional { found } => (
            "'?' pattern requires an optional type".to_string(),
            format!("found '{found}', which is not optional"),
        ),

        DiagnosticKind::NestedOptionalPattern => (
            "nested '?' patterns are not supported".to_string(),
            "use 'if let' chains for nested optionals".to_string(),
        ),

        DiagnosticKind::NonNumericRangePattern { found } => (
            "non-numeric range pattern".to_string(),
            format!("range patterns require numeric types (int, float, double), found '{found}'"),
        ),
        DiagnosticKind::RangePatternBoundTypeMismatch { start, end } => (
            "range pattern type mismatch".to_string(),
            format!("range bounds must have the same type, found '{start}' and '{end}'"),
        ),
        DiagnosticKind::EmptyRangePattern => (
            "empty range pattern".to_string(),
            "range start must be less than end".to_string(),
        ),

        DiagnosticKind::OrPatternBindingMismatch => (
            "or-pattern binding mismatch".to_string(),
            "or-pattern alternatives must bind the same variables".to_string(),
        ),
        DiagnosticKind::OrPatternTypeMismatch { name, expected, found } => (
            "or-pattern type mismatch".to_string(),
            format!("or-pattern binding '{name}' has type '{found}' in one alternative but '{expected}' in another"),
        ),

        DiagnosticKind::ImmutableAssignment { name } => (
            format!("Cannot assign to immutable variable '{name}'"),
            format!("'{name}' is declared with 'let'; use 'var' to allow mutation"),
        ),
        DiagnosticKind::VarParamNotLvalue { param } => (
            format!("Cannot pass non-lvalue to 'var' parameter '{param}'"),
            format!("'var' parameter '{param}' requires a variable, not a literal or expression"),
        ),
        DiagnosticKind::VarParamImmutableBinding { param, binding } => (
            format!("Cannot pass immutable binding '{binding}' to 'var' parameter '{param}'"),
            format!("'{binding}' is declared with 'let'; use 'var' to allow mutation"),
        ),
        DiagnosticKind::VarParamAliasing { param_a, param_b, binding } => (
            format!("Cannot pass same variable '{binding}' to 'var' parameters '{param_a}' and '{param_b}'"),
            "same variable would create aliased mutable references".to_string(),
        ),
        DiagnosticKind::MutatingMethodOnImmutable {
            struct_name,
            method,
        } => (
            format!("Cannot call mutating method '{method}' on immutable binding"),
            format!(
                "'{method}' on '{struct_name}' requires 'var self'; declare the binding with 'var'"
            ),
        ),

        DiagnosticKind::MapEmptyLiteralNoContext => (
            "cannot infer key/value types of empty map".to_string(),
            "add a type annotation, e.g. `let m: [string: int] = [:];`".to_string(),
        ),
        DiagnosticKind::MapKeyFloat => (
            "floating-point type cannot be used as map key".to_string(),
            "float/double keys are unreliable due to NaN and precision; use int or string instead"
                .to_string(),
        ),
        DiagnosticKind::MapKeyNotKeyable { found } => (
            "type cannot be used as map key".to_string(),
            format!("'{found}' is not keyable; map keys must be int, bool, string, or structs/enums/tuples whose fields are all keyable"),
        ),
        DiagnosticKind::MapOptionalKeyNotAllowed { found } => (
            "optional type cannot be used as map key".to_string(),
            format!("'{found}' is optional; map keys cannot be optional"),
        ),
        DiagnosticKind::MapDuplicateKey => (
            "duplicate key in map literal".to_string(),
            "this key appears more than once".to_string(),
        ),
        DiagnosticKind::InvalidCast { from, to } => (
            "Invalid cast".to_string(),
            format!("cannot cast '{from}' to '{to}'"),
        ),
        DiagnosticKind::MethodTypeParamShadowsStruct { kind, struct_name, method, param } => (
            format!("method type parameter shadows {kind} type parameter"),
            format!("method '{method}' on {kind} '{struct_name}' declares type parameter '{param}' which shadows a {kind} type parameter"),
        ),
        DiagnosticKind::NotEquatable { ty } => (
            "type is not equatable".to_string(),
            format!("type '{ty}' does not support '==' or '!='"),
        ),
        DiagnosticKind::UnknownModuleMember { module, member } => (
            format!("Unknown member '{member}' in module '{module}'"),
            format!("module '{module}' does not have a member named '{member}'"),
        ),
        DiagnosticKind::PrivateModuleMember { module, member } => (
            format!("'{member}' is private in module '{module}'"),
            format!("'{member}' exists in module '{module}' but is not marked `pub`"),
        ),
        DiagnosticKind::ReExportCollision { name, first_source, second_source } => (
            format!("symbol '{name}' re-exported by both '{first_source}' and '{second_source}'"),
            format!("'{name}' is introduced by 'pub import' from '{first_source}' and also from '{second_source}'"),
        ),
        DiagnosticKind::AnyTypeNotAllowed => (
            "'any' type is only allowed in extern function declarations".to_string(),
            "'any' cannot be used in user-written type annotations".to_string(),
        ),
        DiagnosticKind::AmbiguousOperator { op, left, right } => (
            "ambiguous operator".to_string(),
            format!("operator '{op}' is declared by both '{left}' and '{right}'"),
        ),
        DiagnosticKind::LetElseMustDiverge => (
            "let-else block must diverge".to_string(),
            "the else block of a let-else must always return, break, or continue".to_string(),
        ),
        DiagnosticKind::LetElseIrrefutable => (
            "irrefutable pattern in let-else".to_string(),
            "this pattern always matches; use a plain 'let' binding instead".to_string(),
        ),
        DiagnosticKind::NotConstantExpression => (
            "not a constant expression".to_string(),
            "only literals, const references, and simple arithmetic are allowed in const expressions".to_string(),
        ),
        DiagnosticKind::CircularConstDependency { name } => (
            format!("circular const dependency for '{name}'"),
            "this constant depends on itself, directly or indirectly".to_string(),
        ),
        DiagnosticKind::ConstDivisionByZero => (
            "division by zero in constant expression".to_string(),
            "integer division by zero is not allowed at compile time".to_string(),
        ),
        DiagnosticKind::ConstIntegerOverflow => (
            "integer overflow in constant expression".to_string(),
            "arithmetic result exceeds the range of 'int'".to_string(),
        ),
        DiagnosticKind::ConstTypeMismatch { expected, got } => (
            "constant type mismatch".to_string(),
            format!("expected '{expected}', found '{got}'"),
        ),
        DiagnosticKind::ConstAssignment { name } => (
            format!("cannot assign to constant '{name}'"),
            format!("'{name}' is declared as 'const' and cannot be reassigned"),
        ),
        DiagnosticKind::DuplicateConst { name } => (
            format!("constant '{name}' is already declared"),
            format!("'{name}' is already declared in this scope"),
        ),
        DiagnosticKind::DuplicateTypeDefinition { name } => (
            format!("type '{name}' is already defined"),
            format!("'{name}' is already defined in this scope"),
        ),
        DiagnosticKind::ImportNameConflict { name, existing } => (
            format!("imported name '{name}' conflicts with {existing}"),
            format!("'{name}' is already bound as {existing} in this scope"),
        ),
        DiagnosticKind::ModuleBindingConflict { name } => (
            format!("module binding '{name}' is already in use"),
            format!("'{name}' is already bound as a module in this scope"),
        ),
        DiagnosticKind::AmbiguousType { name, first_module, second_module } => (
            format!("type '{name}' is ambiguous — found in '{first_module}' and '{second_module}'"),
            "add an explicit import to disambiguate".to_string(),
        ),
        DiagnosticKind::AmbiguousExtendMethod { ty, method, candidates } => {
            let mods = candidates
                .iter()
                .map(|c| format!("'{c}'"))
                .collect::<Vec<_>>()
                .join(", ");
            (
                format!("ambiguous method '{method}' on '{ty}'"),
                format!("found in modules: {mods}; use module.method(val) to disambiguate"),
            )
        }
        DiagnosticKind::ExtendMethodConflict { ty, method } => (
            format!("cannot extend '{ty}' with method '{method}'"),
            "a method with this name already exists on the type".to_string(),
        ),
        DiagnosticKind::DuplicateExtendMethod { ty, method } => (
            format!("duplicate extend method '{method}' on '{ty}'"),
            format!("'{method}' is already defined in an extend block for '{ty}' in this module"),
        ),
        DiagnosticKind::ExtendMethodMissingSelf { method } => (
            format!("extend method '{method}' must have 'self' or 'var self' as its first parameter"),
            "extend methods require a receiver".to_string(),
        ),
        DiagnosticKind::ExtendSelfTypeAnnotation { method } => (
            format!("extend method '{method}': 'self' must not have a type annotation"),
            "the type of self is determined by the extend block".to_string(),
        ),
        DiagnosticKind::ExtendUnsupportedType { ty } => (
            format!("cannot extend type '{ty}'"),
            "only concrete types (int, float, double, bool, string, structs, enums, extern types, options, lists, arrays, maps, tuples) can be extended".to_string(),
        ),
        DiagnosticKind::ExtendTypeParamCountMismatch { ty_name, expected, found } => (
            format!("extend type '{ty_name}' expects {expected} type parameter{}, but {found} {} provided",
                if *expected == 1 { "" } else { "s" },
                if *found == 1 { "was" } else { "were" }),
            format!("'{ty_name}' has {expected} type parameter{}", if *expected == 1 { "" } else { "s" }),
        ),
        DiagnosticKind::ExtendTypeParamsOnNonGeneric { ty_name } => (
            format!("type '{ty_name}' is not generic, but type parameters were provided on extend"),
            "remove the type parameters from the extend head".to_string(),
        ),
        DiagnosticKind::ExtendUndeclaredTypeParam { name } => (
            format!("undeclared type parameter '{name}' used in extend target type"),
            "type parameters used in the extend target must be declared in the extend<...> head".to_string(),
        ),
        DiagnosticKind::ExtendUnusedTypeParam { param_name } => (
            format!("unused type parameter '{param_name}' in extend declaration"),
            "every declared type parameter must appear in the target type".to_string(),
        ),
        DiagnosticKind::DuplicateCastFrom { source_ty, target_ty } => (
            format!("duplicate cast from '{source_ty}' to '{target_ty}'"),
            format!("a 'cast from({source_ty})' is already defined for '{target_ty}' in this module"),
        ),
        DiagnosticKind::CastFromReturnTypeMismatch { expected, found } => (
            format!("cast from return type mismatch: expected '{expected}', found '{found}'"),
            format!("the return type must match the extend target type '{expected}'"),
        ),
        DiagnosticKind::CastFromSelfConversion { ty } => (
            format!("pointless cast from '{ty}' to itself"),
            "converting a type to itself has no effect; 'expr as T' is already a no-op when types match".to_string(),
        ),
        DiagnosticKind::RequiredParamAfterOptional { func, param } => (
            format!("required parameter '{param}' cannot follow a parameter with a default value"),
            format!("in function '{func}': reorder so all required parameters come first"),
        ),
        DiagnosticKind::ParamDefaultNotConst { func, param } => (
            "default parameter value must be a constant expression".to_string(),
            format!("parameter '{param}' in function '{func}' has a non-constant default"),
        ),
        DiagnosticKind::ParamDefaultTypeMismatch { func, param, expected, found } => (
            format!("default value type mismatch for parameter '{param}'"),
            format!("in function '{func}': expected '{expected}', found '{found}'"),
        ),
        DiagnosticKind::ParamDefaultOnGenericType { func, param } => (
            "default values are not allowed on parameters with generic types".to_string(),
            format!("parameter '{param}' in function '{func}' has a generic type"),
        ),
        DiagnosticKind::TooFewArguments { expected, found } => (
            format!("too few arguments: expected at least {expected}, found {found}"),
            "add the missing required arguments".to_string(),
        ),
        DiagnosticKind::TooManyArguments { expected, found } => (
            format!("too many arguments: expected at most {expected}, found {found}"),
            "remove the extra arguments".to_string(),
        ),
        DiagnosticKind::CannotInferLambdaParam { name } => (
            format!("cannot infer type for lambda parameter '{name}'"),
            "add an explicit type annotation or provide a type context".to_string(),
        ),
        DiagnosticKind::MutateCapturedVar { name } => (
            format!("cannot mutate captured variable '{name}'"),
            "closures capture by value; the captured variable cannot be reassigned".to_string(),
        ),
        DiagnosticKind::LambdaParamCountMismatch { expected, found } => (
            "lambda parameter count mismatch".to_string(),
            format!("expected {expected} parameter(s), found {found}"),
        ),
        DiagnosticKind::MutableParamRequiresVarTarget { name } => (
            format!("'|var {name}|' requires the target to be declared with 'var'"),
            "declare the collection with 'var' to allow in-place mutation".to_string(),
        ),
        DiagnosticKind::MutableFnParamRequiresVarTarget => (
            "function with 'var' parameter requires the target to be declared with 'var'".to_string(),
            "declare the collection with 'var' to allow in-place mutation".to_string(),
        ),
        DiagnosticKind::VarPatternOnImmutable => (
            "var binding in pattern requires a mutable (var) scrutinee".to_string(),
            "declare the scrutinee with 'var' to allow write-through".to_string(),
        ),
        DiagnosticKind::UnknownAnnotation { name } => (
            format!("unknown annotation `@{name}`"),
            "unknown annotation".to_string(),
        ),
        DiagnosticKind::InvalidAnnotationTarget { name, target, .. } => (
            format!("`@{name}` is not valid on {target} declarations"),
            format!("`@{name}` cannot be applied here"),
        ),
        DiagnosticKind::DuplicateAnnotation { name } => (
            format!("duplicate annotation `@{name}`"),
            "annotation already applied".to_string(),
        ),
        DiagnosticKind::InvalidAnnotationArgs { name, message } => (
            format!("invalid arguments for `@{name}`"),
            message.clone(),
        ),
        DiagnosticKind::InvalidFormatSpec { reason } => (
            "invalid format specifier".to_string(),
            reason.clone(),
        ),
        DiagnosticKind::CannotInferEnumVariant { variant } => (
            format!("cannot infer enum type for '.{variant}'"),
            format!("use fully qualified 'EnumName.{variant}'"),
        ),
        DiagnosticKind::BareCatchAllOnOptional { pattern_name } => (
            "'if let' on optional type requires an unwrapping pattern".to_string(),
            format!("'{pattern_name}' matches any value, including nil; use '{pattern_name}?' to unwrap"),
        ),
        DiagnosticKind::DeprecatedUsage { kind, name, reason } => {
            let title = format!("use of deprecated {kind} '{name}'");
            let label = reason.as_deref().unwrap_or("deprecated").to_string();
            (title, label)
        }
        DiagnosticKind::InternalAccess { kind, name, type_name, reason, .. } => {
            let title = format!("accessing internal {kind} '{name}' of type '{type_name}'");
            let label = reason
                .as_deref()
                .unwrap_or("marked @internal and may change without notice")
                .to_string();
            (title, label)
        }
        DiagnosticKind::InternalOnToString => (
            "`@internal` cannot be applied to `to_string` methods".to_string(),
            "`to_string` is implicitly public for printing and formatting".to_string(),
        ),
        DiagnosticKind::ReturnInDefer => (
            "return inside defer".to_string(),
            "'return' is not allowed inside a defer body".to_string(),
        ),
        DiagnosticKind::BreakInDefer => (
            "break inside defer".to_string(),
            "'break' is not allowed inside a defer body".to_string(),
        ),
        DiagnosticKind::ContinueInDefer => (
            "continue inside defer".to_string(),
            "'continue' is not allowed inside a defer body".to_string(),
        ),
        DiagnosticKind::InfiniteSizeType { type_name, cycle_field, cycle_target } => (
            format!("type '{type_name}' has infinite size"),
            format!("field '{cycle_field}' of type '{cycle_target}' creates a value-type cycle"),
        ),
    }
}

fn token_span_to_span(tokens: &[SpannedToken], span: Range<usize>) -> Span {
    let start_byte = tokens.get(span.start).map_or(0, |(_, s)| s.start);
    let last_tok_idx = span.end.saturating_sub(1);
    let end_byte = tokens.get(last_tok_idx).map_or(start_byte, |(_, s)| s.end);
    Span::new(start_byte, end_byte)
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

fn describe_token(token: &Token) -> String {
    match token {
        Token::Keyword(keyword) => format!("'{keyword}' keyword"),
        Token::Open(..) | Token::Close(..) => format!("'{token}' delimiter"),
        Token::Ident(_) => "identifier".to_string(),
        Token::Literal(lit) => format!("'{lit}' literal"),
        Token::Op(op) => format!("'{op}' operator"),
        _ => format!("'{token}' token"),
    }
}
