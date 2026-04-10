use crate::{Profile, lexer::SpannedToken, span::Span};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicKind {
    Predicate,
    Diagnostic(DiagnosticKind),
    CompileTimeValue(CompileTimeValueKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticKind {
    Warn,
    Error,
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileTimeValueKind {
    File,
    Line,
    Func,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgShape {
    None,
    Ident,
    StringLit,
}

pub struct IntrinsicDef {
    pub name: &'static str,
    pub kind: IntrinsicKind,
    pub arg: ArgShape,
}

const REGISTRY: &[IntrinsicDef] = &[
    IntrinsicDef {
        name: "profile",
        kind: IntrinsicKind::Predicate,
        arg: ArgShape::Ident,
    },
    IntrinsicDef {
        name: "os",
        kind: IntrinsicKind::Predicate,
        arg: ArgShape::Ident,
    },
    IntrinsicDef {
        name: "arch",
        kind: IntrinsicKind::Predicate,
        arg: ArgShape::Ident,
    },
    IntrinsicDef {
        name: "feature",
        kind: IntrinsicKind::Predicate,
        arg: ArgShape::Ident,
    },
    IntrinsicDef {
        name: "warning",
        kind: IntrinsicKind::Diagnostic(DiagnosticKind::Warn),
        arg: ArgShape::StringLit,
    },
    IntrinsicDef {
        name: "error",
        kind: IntrinsicKind::Diagnostic(DiagnosticKind::Error),
        arg: ArgShape::StringLit,
    },
    IntrinsicDef {
        name: "log",
        kind: IntrinsicKind::Diagnostic(DiagnosticKind::Log),
        arg: ArgShape::StringLit,
    },
    IntrinsicDef {
        name: "file",
        kind: IntrinsicKind::CompileTimeValue(CompileTimeValueKind::File),
        arg: ArgShape::None,
    },
    IntrinsicDef {
        name: "line",
        kind: IntrinsicKind::CompileTimeValue(CompileTimeValueKind::Line),
        arg: ArgShape::None,
    },
    IntrinsicDef {
        name: "function",
        kind: IntrinsicKind::CompileTimeValue(CompileTimeValueKind::Func),
        arg: ArgShape::None,
    },
];

pub fn lookup(name: &str) -> Option<&'static IntrinsicDef> {
    REGISTRY.iter().find(|d| d.name == name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOs {
    MacOs,
    Linux,
    Windows,
    Wasm,
    Ios,
    Android,
}

impl std::str::FromStr for TargetOs {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "macos" => Ok(Self::MacOs),
            "linux" => Ok(Self::Linux),
            "windows" => Ok(Self::Windows),
            "wasm" => Ok(Self::Wasm),
            "ios" => Ok(Self::Ios),
            "android" => Ok(Self::Android),
            _ => Err(()),
        }
    }
}

impl TargetOs {
    pub const ALL: &[&str] = &["macos", "linux", "windows", "wasm", "ios", "android"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::MacOs => "macos",
            Self::Linux => "linux",
            Self::Windows => "windows",
            Self::Wasm => "wasm",
            Self::Ios => "ios",
            Self::Android => "android",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    X86_64,
    Aarch64,
}

impl std::str::FromStr for TargetArch {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "x86_64" => Ok(Self::X86_64),
            "aarch64" => Ok(Self::Aarch64),
            _ => Err(()),
        }
    }
}

impl TargetArch {
    pub const ALL: &[&str] = &["x86_64", "aarch64"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompilationContext {
    pub profile: Profile,
    pub os: TargetOs,
    pub arch: TargetArch,
    pub features: Vec<String>,
}

impl CompilationContext {
    pub fn from_host(profile: Profile) -> Self {
        Self {
            profile,
            os: Self::detect_os(),
            arch: Self::detect_arch(),
            features: vec![],
        }
    }

    fn detect_os() -> TargetOs {
        if cfg!(target_os = "macos") {
            TargetOs::MacOs
        } else if cfg!(target_os = "linux") {
            TargetOs::Linux
        } else if cfg!(target_os = "windows") {
            TargetOs::Windows
        } else if cfg!(target_arch = "wasm32") {
            TargetOs::Wasm
        } else {
            TargetOs::Linux
        }
    }

    fn detect_arch() -> TargetArch {
        if cfg!(target_arch = "x86_64") {
            TargetArch::X86_64
        } else if cfg!(target_arch = "aarch64") {
            TargetArch::Aarch64
        } else {
            TargetArch::X86_64
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum IntrinsicError {
    UnknownIntrinsic {
        name: String,
        span: Span,
    },
    UnknownValue {
        predicate: String,
        value: String,
        valid: Vec<String>,
        span: Span,
    },
    WrongArgCount {
        name: String,
        expected: usize,
        found: usize,
        span: Span,
    },
    ArgNotIdent {
        name: String,
        span: Span,
    },
    ArgNotStringLiteral {
        name: String,
        span: Span,
    },
}

impl IntrinsicError {
    pub fn span(&self) -> Span {
        match self {
            Self::UnknownIntrinsic { span, .. }
            | Self::UnknownValue { span, .. }
            | Self::WrongArgCount { span, .. }
            | Self::ArgNotIdent { span, .. }
            | Self::ArgNotStringLiteral { span, .. } => *span,
        }
    }
}

impl std::fmt::Display for IntrinsicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownIntrinsic { name, .. } => {
                write!(f, "unknown intrinsic '#{name}'")
            }
            Self::UnknownValue {
                predicate,
                value,
                valid,
                ..
            } => {
                let valid_list = valid.join("' or '");
                write!(f, "unknown {predicate} '{value}'; expected '{valid_list}'")
            }
            Self::WrongArgCount {
                name,
                expected,
                found,
                ..
            } => {
                write!(f, "#{name} expects {expected} argument(s), found {found}")
            }
            Self::ArgNotIdent { name, .. } => {
                write!(f, "#{name} argument must be an identifier")
            }
            Self::ArgNotStringLiteral { name, .. } => {
                write!(f, "#{name} argument must be a string literal")
            }
        }
    }
}

pub struct SourceLocationInfo {
    pub file_path: String,
    pub line_for_token: Vec<i64>,
}

impl SourceLocationInfo {
    pub fn new(file_path: String, source: &str, tokens: &[SpannedToken]) -> Self {
        let mut line_starts = vec![0usize];
        for (i, byte) in source.as_bytes().iter().enumerate() {
            if *byte == b'\n' {
                line_starts.push(i + 1);
            }
        }

        let line_for_token = tokens
            .iter()
            .map(|(_, span)| {
                let byte_offset = span.start;
                let line = match line_starts.binary_search(&byte_offset) {
                    Ok(idx) => idx + 1,
                    Err(idx) => idx,
                };
                line as i64
            })
            .collect();

        Self {
            file_path,
            line_for_token,
        }
    }
}

pub fn evaluate_predicate(
    name: &str,
    arg: &str,
    ctx: &CompilationContext,
    span: Span,
) -> Result<bool, IntrinsicError> {
    let (valid, actual): (&[&str], &str) = match name {
        "profile" => (Profile::ALL, ctx.profile.as_str()),
        "os" => (TargetOs::ALL, ctx.os.as_str()),
        "arch" => (TargetArch::ALL, ctx.arch.as_str()),
        "feature" => return Ok(ctx.features.iter().any(|f| f == arg)),
        _ => {
            return Err(IntrinsicError::UnknownIntrinsic {
                name: name.to_string(),
                span,
            });
        }
    };
    if !valid.contains(&arg) {
        return Err(IntrinsicError::UnknownValue {
            predicate: name.to_string(),
            value: arg.to_string(),
            valid: valid.iter().map(ToString::to_string).collect(),
            span,
        });
    }
    Ok(actual == arg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicDiagnosticLevel {
    Warning,
    Error,
    Note,
}

#[derive(Debug, Clone)]
pub struct IntrinsicDiagnostic {
    pub level: IntrinsicDiagnosticLevel,
    pub message: String,
    pub span: Span,
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMMY: Span = Span { start: 0, end: 0 };

    fn ctx_macos() -> CompilationContext {
        CompilationContext {
            profile: Profile::Debug,
            os: TargetOs::MacOs,
            arch: TargetArch::X86_64,
            features: vec![],
        }
    }

    fn ctx_with_features(features: Vec<String>) -> CompilationContext {
        CompilationContext {
            profile: Profile::Debug,
            os: TargetOs::Linux,
            arch: TargetArch::X86_64,
            features,
        }
    }

    #[test]
    fn os_matching() {
        let ctx = ctx_macos();
        assert_eq!(evaluate_predicate("os", "macos", &ctx, DUMMY), Ok(true));
        assert_eq!(evaluate_predicate("os", "linux", &ctx, DUMMY), Ok(false));
    }

    #[test]
    fn os_unknown_value() {
        let ctx = ctx_macos();
        let err = evaluate_predicate("os", "haiku", &ctx, DUMMY).unwrap_err();
        assert!(matches!(err, IntrinsicError::UnknownValue { .. }));
        assert!(err.to_string().contains("haiku"));
    }

    #[test]
    fn arch_matching() {
        let ctx = CompilationContext {
            profile: Profile::Debug,
            os: TargetOs::Linux,
            arch: TargetArch::X86_64,
            features: vec![],
        };
        assert_eq!(evaluate_predicate("arch", "x86_64", &ctx, DUMMY), Ok(true));
        assert_eq!(
            evaluate_predicate("arch", "aarch64", &ctx, DUMMY),
            Ok(false)
        );
    }

    #[test]
    fn arch_unknown_value() {
        let ctx = ctx_macos();
        let err = evaluate_predicate("arch", "mips", &ctx, DUMMY).unwrap_err();
        assert!(matches!(err, IntrinsicError::UnknownValue { .. }));
        assert!(err.to_string().contains("mips"));
    }

    #[test]
    fn feature_present() {
        let ctx = ctx_with_features(vec!["ecs".to_string()]);
        assert_eq!(evaluate_predicate("feature", "ecs", &ctx, DUMMY), Ok(true));
    }

    #[test]
    fn feature_absent() {
        let ctx = ctx_with_features(vec![]);
        assert_eq!(evaluate_predicate("feature", "ecs", &ctx, DUMMY), Ok(false));
    }

    #[test]
    fn target_os_round_trip() {
        use std::str::FromStr;
        for s in ["macos", "linux", "windows", "wasm", "ios", "android"] {
            let os = TargetOs::from_str(s).expect("should parse");
            assert_eq!(os.as_str(), s);
        }
        assert!(TargetOs::from_str("haiku").is_err());
    }

    #[test]
    fn target_arch_round_trip() {
        use std::str::FromStr;
        for s in ["x86_64", "aarch64"] {
            let arch = TargetArch::from_str(s).expect("should parse");
            assert_eq!(arch.as_str(), s);
        }
        assert!(TargetArch::from_str("mips").is_err());
    }

    #[test]
    fn from_host_does_not_panic() {
        let ctx = CompilationContext::from_host(Profile::Debug);
        assert_eq!(ctx.profile, Profile::Debug);
        assert!(ctx.features.is_empty());
    }

    #[test]
    fn lookup_diagnostic_intrinsics() {
        for name in ["warning", "error", "log"] {
            let def = lookup(name).expect("should find diagnostic intrinsic");
            assert!(matches!(def.kind, IntrinsicKind::Diagnostic(_)));
            assert_eq!(def.arg, ArgShape::StringLit);
        }
    }

    #[test]
    fn arg_not_string_literal_display() {
        let err = IntrinsicError::ArgNotStringLiteral {
            name: "warning".to_string(),
            span: Span::new(0, 0),
        };
        assert!(err.to_string().contains("string literal"));
    }

    #[test]
    fn lookup_compile_time_value_intrinsics() {
        for name in ["file", "line", "function"] {
            let def = lookup(name).expect("should find compile-time value intrinsic");
            assert!(matches!(def.kind, IntrinsicKind::CompileTimeValue(_)));
            assert_eq!(def.arg, ArgShape::None);
        }
    }

    #[test]
    fn source_location_info_line_numbers() {
        use crate::lexer::{self, Token};
        let source = "fn main() {\n    println(1);\n}\n";
        let tokens = lexer::tokenize(source).unwrap();
        let info = SourceLocationInfo::new("test.anv".to_string(), source, &tokens);
        assert_eq!(info.file_path, "test.anv");
        // "fn" token is on line 1
        assert_eq!(info.line_for_token[0], 1);
        // Find "println" token — should be on line 2
        let println_idx = tokens
            .iter()
            .position(|(t, _)| matches!(t, Token::Ident(id) if id.0.as_ref() == "println"))
            .unwrap();
        assert_eq!(info.line_for_token[println_idx], 2);
    }
}
