use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::LazyLock,
};

use super::{
    error::{Diagnostic, DiagnosticKind},
    lint::LintLevel,
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
    Const,
    ExternFunc,
    ExternType,
    InlineMethod,
    ExtendMethod,
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
            Self::Const => write!(f, "const"),
            Self::ExternFunc => write!(f, "extern function"),
            Self::ExternType => write!(f, "extern type"),
            Self::InlineMethod => write!(f, "inline method"),
            Self::ExtendMethod => write!(f, "extend method"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum KnownAnnotationKind {
    Test,
    Deprecated,
    Internal,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum AnnotationArgSchema {
    NoArgs,
    OptionalPositionalString,
}

pub(super) struct AnnotationSpec {
    pub kind: KnownAnnotationKind,
    pub targets: &'static [AnnotationTarget],
    pub args: AnnotationArgSchema,
}

#[derive(Debug, Clone)]
pub(super) enum AppliedAnnotation {
    Test,
    Deprecated { reason: Option<String> },
    Internal { reason: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeprecationInfo<'a> {
    NotDeprecated,
    Deprecated(Option<&'a str>),
}

#[derive(Debug, Clone, Default)]
pub(super) struct AppliedAnnotations {
    items: Vec<AppliedAnnotation>,
}

impl AppliedAnnotations {
    fn deprecation(&self) -> DeprecationInfo<'_> {
        self.items
            .iter()
            .find_map(|a| match a {
                AppliedAnnotation::Deprecated { reason } => {
                    Some(DeprecationInfo::Deprecated(reason.as_deref()))
                }
                AppliedAnnotation::Test | AppliedAnnotation::Internal { .. } => None,
            })
            .unwrap_or(DeprecationInfo::NotDeprecated)
    }

    pub(super) fn has_internal(&self) -> bool {
        self.items
            .iter()
            .any(|a| matches!(a, AppliedAnnotation::Internal { .. }))
    }

    pub(super) fn check_deprecation(
        &self,
        span: Span,
        kind: &'static str,
        name: Ident,
        errors: &mut Vec<Diagnostic>,
    ) {
        if let DeprecationInfo::Deprecated(reason) = self.deprecation() {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::DeprecatedUsage {
                    kind,
                    name,
                    reason: reason.map(str::to_string),
                },
            ));
        }
    }

    pub(super) fn check_internal(
        &self,
        span: Span,
        kind: &'static str,
        name: Ident,
        type_name: Ident,
        cross_module: bool,
        lint_level: LintLevel,
        errors: &mut Vec<Diagnostic>,
    ) {
        if !cross_module || lint_level == LintLevel::Allow {
            return;
        }
        let reason = self.items.iter().find_map(|a| match a {
            AppliedAnnotation::Internal { reason } => Some(reason.as_deref()),
            _ => None,
        });
        if let Some(reason) = reason {
            errors.push(Diagnostic::new(
                span,
                DiagnosticKind::InternalAccess {
                    kind,
                    name,
                    type_name,
                    reason: reason.map(str::to_string),
                    level: lint_level,
                },
            ));
        }
    }
}

static KNOWN_ANNOTATIONS: LazyLock<HashMap<&'static str, AnnotationSpec>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "test",
        AnnotationSpec {
            kind: KnownAnnotationKind::Test,
            targets: &[AnnotationTarget::Func],
            args: AnnotationArgSchema::NoArgs,
        },
    );
    map.insert(
        "deprecated",
        AnnotationSpec {
            kind: KnownAnnotationKind::Deprecated,
            targets: &[
                AnnotationTarget::Func,
                AnnotationTarget::Struct,
                AnnotationTarget::DataRef,
                AnnotationTarget::Enum,
                AnnotationTarget::Field,
                AnnotationTarget::Variant,
                AnnotationTarget::Const,
                AnnotationTarget::ExternFunc,
                AnnotationTarget::ExternType,
                AnnotationTarget::InlineMethod,
                AnnotationTarget::ExtendMethod,
            ],
            args: AnnotationArgSchema::OptionalPositionalString,
        },
    );
    map.insert(
        "internal",
        AnnotationSpec {
            kind: KnownAnnotationKind::Internal,
            targets: &[AnnotationTarget::Field, AnnotationTarget::InlineMethod],
            args: AnnotationArgSchema::OptionalPositionalString,
        },
    );
    map
});

fn format_valid_targets(targets: &[AnnotationTarget]) -> String {
    targets
        .iter()
        .map(|t| format!("{t}s"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) fn normalize_annotations(
    annotations: &[AnnotationNode],
    target: AnnotationTarget,
    errors: &mut Vec<Diagnostic>,
) -> AppliedAnnotations {
    let registry = &*KNOWN_ANNOTATIONS;
    let mut seen = HashSet::new();
    let mut items = vec![];

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
            let valid_targets = format_valid_targets(def.targets);
            let help = format!("`@{name}` can only be applied to {valid_targets}");
            errors.push(
                Diagnostic::new(
                    annotation.span,
                    DiagnosticKind::InvalidAnnotationTarget {
                        name: *name,
                        target: format!("{target}"),
                        valid_targets,
                    },
                )
                .with_help(help),
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

        if !validate_annotation_args(
            &annotation.node.args,
            *name,
            def.args,
            annotation.span,
            errors,
        ) {
            continue;
        }

        let reason = match &annotation.node.args {
            AnnotationArgs::Positional(Lit::String(s)) => Some(s.clone()),
            _ => None,
        };
        let applied = match def.kind {
            KnownAnnotationKind::Test => AppliedAnnotation::Test,
            KnownAnnotationKind::Deprecated => AppliedAnnotation::Deprecated { reason },
            KnownAnnotationKind::Internal => AppliedAnnotation::Internal { reason },
        };
        items.push(applied);
    }

    AppliedAnnotations { items }
}

fn validate_annotation_args(
    args: &AnnotationArgs,
    name: Ident,
    schema: AnnotationArgSchema,
    span: Span,
    errors: &mut Vec<Diagnostic>,
) -> bool {
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
                return false;
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
                return false;
            }
        },
    }
    true
}
