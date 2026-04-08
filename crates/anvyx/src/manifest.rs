use std::{collections::HashMap, fs, path::Path};

use anvyx_lang::LintConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub project: Project,
    #[serde(default)]
    pub externs: HashMap<String, ExternEntry>,
    #[serde(default)]
    pub lint: LintConfig,
}

impl Manifest {
    pub fn has_externs(&self) -> bool {
        !self.externs.is_empty()
    }
}

#[derive(Debug, Deserialize)]
pub struct Project {
    pub name: Option<String>,
    pub entry: String,
}

#[derive(Debug, Deserialize)]
pub struct ExternEntry {
    pub path: String,
}

pub fn parse_manifest() -> Result<Option<Manifest>, String> {
    let manifest_path = Path::new("anvyx.toml");
    if !manifest_path.exists() {
        return Ok(None);
    }

    let contents =
        fs::read_to_string(manifest_path).map_err(|e| format!("Failed to read anvyx.toml: {e}"))?;
    let manifest: Manifest =
        toml::from_str(&contents).map_err(|e| format!("Failed to parse anvyx.toml: {e}"))?;

    Ok(Some(manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml: &str) -> Result<Manifest, String> {
        toml::from_str(toml).map_err(|e| format!("Failed to parse: {e}"))
    }

    #[test]
    fn parse_manifest_no_externs() {
        let manifest = parse(
            r#"
            [project]
            entry = "src/main.anv"
            "#,
        )
        .unwrap();

        assert!(manifest.externs.is_empty());
        assert!(!manifest.has_externs());
        assert_eq!(manifest.project.entry, "src/main.anv");
    }

    #[test]
    fn parse_manifest_with_one_extern() {
        let manifest = parse(
            r#"
            [project]
            entry = "src/main.anv"

            [externs.engine]
            path = "my_externs/engine"
            "#,
        )
        .unwrap();

        assert!(manifest.has_externs());
        assert_eq!(manifest.externs.len(), 1);
        assert_eq!(manifest.externs["engine"].path, "my_externs/engine");
    }

    #[test]
    fn parse_manifest_with_multiple_externs() {
        let manifest = parse(
            r#"
            [project]
            entry = "src/main.anv"

            [externs.engine]
            path = "my_externs/engine"

            [externs.audio]
            path = "my_externs/audio"
            "#,
        )
        .unwrap();

        assert!(manifest.has_externs());
        assert_eq!(manifest.externs.len(), 2);
        assert_eq!(manifest.externs["engine"].path, "my_externs/engine");
        assert_eq!(manifest.externs["audio"].path, "my_externs/audio");
    }

    #[test]
    fn parse_manifest_missing_project_errors() {
        let result = parse(
            r#"
            [externs.engine]
            path = "my_externs/engine"
            "#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn parse_manifest_with_optional_name() {
        let with_name = parse(
            r#"
            [project]
            name = "my_game"
            entry = "src/main.anv"
            "#,
        )
        .unwrap();

        assert_eq!(with_name.project.name.as_deref(), Some("my_game"));

        let without_name = parse(
            r#"
            [project]
            entry = "src/main.anv"
            "#,
        )
        .unwrap();

        assert!(without_name.project.name.is_none());
    }

    #[test]
    fn parse_manifest_lint_error() {
        use anvyx_lang::LintLevel;
        let manifest = parse(
            r#"
            [project]
            entry = "src/main.anv"

            [lint]
            internal_access = "error"
            "#,
        )
        .unwrap();

        assert_eq!(manifest.lint.internal_access, LintLevel::Error);
    }

    #[test]
    fn parse_manifest_lint_allow() {
        use anvyx_lang::LintLevel;
        let manifest = parse(
            r#"
            [project]
            entry = "src/main.anv"

            [lint]
            internal_access = "allow"
            "#,
        )
        .unwrap();

        assert_eq!(manifest.lint.internal_access, LintLevel::Allow);
    }

    #[test]
    fn parse_manifest_lint_default() {
        use anvyx_lang::LintLevel;
        let manifest = parse(
            r#"
            [project]
            entry = "src/main.anv"
            "#,
        )
        .unwrap();

        assert_eq!(manifest.lint.internal_access, LintLevel::Warn);
    }
}
