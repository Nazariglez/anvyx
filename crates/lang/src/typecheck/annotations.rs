use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::LazyLock,
};

use super::{
    error::{Diagnostic, DiagnosticKind},
    types::Deprecated,
};
use crate::{
    ast::{AnnotationArgs, AnnotationNode, Ident, Lit},
    span::Span,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum AnnotationTarget {
    Func,
    Struct,
    DataRef,
    Enum,
    Field,
    Variant,
}

impl fmt::Display for AnnotationTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Func => write!(f, "function"),
            Self::Struct => write!(f, "struct"),
            Self::DataRef => write!(f, "dataref"),
            Self::Enum => write!(f, "enum"),
            Self::Field => write!(f, "field"),
            Self::Variant => write!(f, "variant"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AnnotationArgSchema {
    NoArgs,
    OptionalPositionalString,
}

struct AnnotationDef {
    targets: &'static [AnnotationTarget],
    args: AnnotationArgSchema,
}

static KNOWN_ANNOTATIONS: LazyLock<HashMap<&'static str, AnnotationDef>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "test",
        AnnotationDef {
            targets: &[AnnotationTarget::Func],
            args: AnnotationArgSchema::NoArgs,
        },
    );
    map.insert(
        "deprecated",
        AnnotationDef {
            targets: &[
                AnnotationTarget::Func,
                AnnotationTarget::Struct,
                AnnotationTarget::DataRef,
                AnnotationTarget::Enum,
                AnnotationTarget::Field,
                AnnotationTarget::Variant,
            ],
            args: AnnotationArgSchema::OptionalPositionalString,
        },
    );
    map
});

pub(super) fn validate_annotations(
    annotations: &[AnnotationNode],
    target: AnnotationTarget,
    errors: &mut Vec<Diagnostic>,
) {
    let registry = &*KNOWN_ANNOTATIONS;
    let mut seen = HashSet::new();

    for annotation in annotations {
        let name = &annotation.node.name;
        let name_str = name.to_string();

        let Some(def) = registry.get(name_str.as_str()) else {
            errors.push(Diagnostic::new(
                annotation.span,
                DiagnosticKind::UnknownAnnotation { name: *name },
            ));
            continue;
        };

        if !def.targets.contains(&target) {
            let valid_targets = def
                .targets
                .iter()
                .map(|t| format!("{t}s"))
                .collect::<Vec<_>>()
                .join(", ");
            errors.push(
                Diagnostic::new(
                    annotation.span,
                    DiagnosticKind::InvalidAnnotationTarget {
                        name: *name,
                        target: format!("{target}"),
                        valid_targets: valid_targets.clone(),
                    },
                )
                .with_help(format!("`@{name}` can only be applied to {valid_targets}")),
            );
            continue;
        }

        if seen.contains(&name_str) {
            errors.push(Diagnostic::new(
                annotation.span,
                DiagnosticKind::DuplicateAnnotation { name: *name },
            ));
            continue;
        }
        seen.insert(name_str);

        validate_annotation_args(
            &annotation.node.args,
            *name,
            def.args,
            annotation.span,
            errors,
        );
    }
}

pub(super) fn extract_deprecated(annotations: &[AnnotationNode]) -> Deprecated {
    annotations
        .iter()
        .find(|a| a.node.name.to_string() == "deprecated")
        .map_or(Deprecated::No, |a| match &a.node.args {
            AnnotationArgs::Positional(Lit::String(s)) => Deprecated::Yes(Some(s.clone())),
            _ => Deprecated::Yes(None),
        })
}

fn validate_annotation_args(
    args: &AnnotationArgs,
    name: Ident,
    schema: AnnotationArgSchema,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) {
    match schema {
        AnnotationArgSchema::NoArgs => {
            if !matches!(args, AnnotationArgs::None) {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::InvalidAnnotationArgs {
                        name,
                        message: "this annotation does not accept arguments".to_string(),
                    },
                ));
            }
        }
        AnnotationArgSchema::OptionalPositionalString => match args {
            AnnotationArgs::None | AnnotationArgs::Positional(Lit::String(_)) => {}
            _ => {
                errors.push(Diagnostic::new(
                    span,
                    DiagnosticKind::InvalidAnnotationArgs {
                        name,
                        message: "expected no arguments or a string argument".to_string(),
                    },
                ));
            }
        },
    }
}
