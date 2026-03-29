use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ast::{AnnotationArgs, AnnotationNode, Ident};
use crate::span::Span;

use super::error::{TypeErr, TypeErrKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum AnnotationTarget {
    Func,
    Struct,
    Enum,
    Field,
    Variant,
}

impl fmt::Display for AnnotationTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Func => write!(f, "function"),
            Self::Struct => write!(f, "struct"),
            Self::Enum => write!(f, "enum"),
            Self::Field => write!(f, "field"),
            Self::Variant => write!(f, "variant"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AnnotationArgSchema {
    NoArgs,
}

struct AnnotationDef {
    targets: &'static [AnnotationTarget],
    args: AnnotationArgSchema,
}

fn known_annotations() -> HashMap<&'static str, AnnotationDef> {
    let mut map = HashMap::new();
    map.insert(
        "test",
        AnnotationDef {
            targets: &[AnnotationTarget::Func],
            args: AnnotationArgSchema::NoArgs,
        },
    );
    map
}

pub(super) fn validate_annotations(
    annotations: &[AnnotationNode],
    target: AnnotationTarget,
    errors: &mut Vec<TypeErr>,
) {
    let registry = known_annotations();
    let mut seen = HashSet::new();

    for annotation in annotations {
        let name = &annotation.node.name;
        let name_str = name.to_string();

        let Some(def) = registry.get(name_str.as_str()) else {
            errors.push(TypeErr::new(
                annotation.span,
                TypeErrKind::UnknownAnnotation { name: *name },
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
                TypeErr::new(
                    annotation.span,
                    TypeErrKind::InvalidAnnotationTarget {
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
            errors.push(TypeErr::new(
                annotation.span,
                TypeErrKind::DuplicateAnnotation { name: *name },
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

fn validate_annotation_args(
    args: &AnnotationArgs,
    name: Ident,
    schema: AnnotationArgSchema,
    span: Span,
    errors: &mut Vec<TypeErr>,
) {
    match schema {
        AnnotationArgSchema::NoArgs => {
            if !matches!(args, AnnotationArgs::None) {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::InvalidAnnotationArgs {
                        name,
                        message: "this annotation does not accept arguments".to_string(),
                    },
                ));
            }
        }
    }
}
