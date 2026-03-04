//! Structured validation report for programmatic consumption.
//!
//! [`generate_report`] converts a [`ValidationContext`] into a
//! [`ValidationReport`] that can be serialized to JSON for the CLI,
//! TUI, or MCP server.
//!
//! # Examples
//!
//! ```
//! use cobre_io::validation::{ErrorKind, ValidationContext};
//! use cobre_io::{generate_report};
//!
//! let mut ctx = ValidationContext::new();
//! ctx.add_error(ErrorKind::FileNotFound, "system/hydros.json", None::<&str>, "file missing");
//! ctx.add_warning(ErrorKind::UnusedEntity, "system/thermals.json", Some("T1"), "inactive");
//!
//! let report = generate_report(&ctx);
//! assert_eq!(report.error_count, 1);
//! assert_eq!(report.warning_count, 1);
//!
//! let json = report.to_json().unwrap();
//! assert!(json.contains("error_count"));
//! ```

use serde::Serialize;

use crate::validation::ValidationContext;
use crate::LoadError;

// ── ReportEntry ───────────────────────────────────────────────────────────────

/// A single diagnostic entry in a [`ValidationReport`].
///
/// The `kind` field holds the `Debug` representation of the [`ErrorKind`]
/// variant (e.g., `"FileNotFound"`), making it suitable for programmatic
/// inspection by the CLI, TUI, or MCP server.
#[derive(Debug, Clone, Serialize)]
pub struct ReportEntry {
    /// The `ErrorKind` variant name as a string (e.g., `"FileNotFound"`).
    pub kind: String,
    /// Path to the file where the problem was detected, as a string.
    pub file: String,
    /// Optional identifier of the entity involved (e.g., `"hydro_042"`).
    pub entity: Option<String>,
    /// Human-readable description of the problem.
    pub message: String,
}

// ── ValidationReport ─────────────────────────────────────────────────────────

/// A structured summary of all validation diagnostics collected by a
/// [`ValidationContext`].
///
/// Produced by [`generate_report`] and serializable to JSON via [`to_json`].
///
/// [`to_json`]: ValidationReport::to_json
#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    /// Total number of error-severity diagnostics.
    pub error_count: usize,
    /// Total number of warning-severity diagnostics.
    pub warning_count: usize,
    /// All error-severity diagnostics.
    pub errors: Vec<ReportEntry>,
    /// All warning-severity diagnostics.
    pub warnings: Vec<ReportEntry>,
}

impl ValidationReport {
    /// Serialize this report to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::ParseError`] with path `"<report>"` if `serde_json`
    /// fails to serialize the report. This should not occur in practice given
    /// the types used in [`ValidationReport`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_io::validation::{ErrorKind, ValidationContext};
    /// use cobre_io::generate_report;
    ///
    /// let ctx = ValidationContext::new();
    /// let report = generate_report(&ctx);
    /// let json = report.to_json().unwrap();
    /// assert!(json.contains("error_count"));
    /// assert!(json.contains("errors"));
    /// assert!(json.contains("warnings"));
    /// ```
    pub fn to_json(&self) -> Result<String, LoadError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| LoadError::parse("<report>", format!("JSON serialization failed: {e}")))
    }
}

// ── generate_report ───────────────────────────────────────────────────────────

/// Convert a [`ValidationContext`] into a [`ValidationReport`].
///
/// Reads all diagnostics from `ctx` and maps them to [`ReportEntry`] values,
/// separated by severity. The `ctx` is not consumed so the caller can still
/// call [`ValidationContext::into_result`] afterward.
///
/// # Examples
///
/// ```
/// use cobre_io::validation::{ErrorKind, ValidationContext};
/// use cobre_io::generate_report;
///
/// let mut ctx = ValidationContext::new();
/// ctx.add_error(ErrorKind::FileNotFound, "system/hydros.json", None::<&str>, "missing");
/// ctx.add_error(ErrorKind::ParseError, "stages.json", None::<&str>, "malformed");
/// ctx.add_warning(ErrorKind::UnusedEntity, "system/thermals.json", Some("T1"), "inactive");
///
/// let report = generate_report(&ctx);
/// assert_eq!(report.error_count, 2);
/// assert_eq!(report.warning_count, 1);
/// ```
#[must_use]
pub fn generate_report(ctx: &ValidationContext) -> ValidationReport {
    let errors: Vec<ReportEntry> = ctx
        .errors()
        .into_iter()
        .map(|entry| ReportEntry {
            kind: format!("{:?}", entry.kind),
            file: entry.file.display().to_string(),
            entity: entry.entity.clone(),
            message: entry.message.clone(),
        })
        .collect();

    let warnings: Vec<ReportEntry> = ctx
        .warnings()
        .into_iter()
        .map(|entry| ReportEntry {
            kind: format!("{:?}", entry.kind),
            file: entry.file.display().to_string(),
            entity: entry.entity.clone(),
            message: entry.message.clone(),
        })
        .collect();

    let error_count = errors.len();
    let warning_count = warnings.len();

    ValidationReport {
        error_count,
        warning_count,
        errors,
        warnings,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::validation::{ErrorKind, ValidationContext};

    fn make_context_with_errors_and_warnings() -> ValidationContext {
        let mut ctx = ValidationContext::new();
        ctx.add_error(
            ErrorKind::FileNotFound,
            "system/hydros.json",
            None::<&str>,
            "required file is missing",
        );
        ctx.add_error(
            ErrorKind::ParseError,
            "stages.json",
            Some("stage_001"),
            "malformed JSON at line 42",
        );
        ctx.add_warning(
            ErrorKind::UnusedEntity,
            "system/thermals.json",
            Some("T1"),
            "max_generation=0 for all stages",
        );
        ctx
    }

    #[test]
    fn test_generate_report_errors_and_warnings() {
        let ctx = make_context_with_errors_and_warnings();
        let report = generate_report(&ctx);

        assert_eq!(report.error_count, 2);
        assert_eq!(report.warning_count, 1);
        assert_eq!(report.errors.len(), 2);
        assert_eq!(report.warnings.len(), 1);
    }

    #[test]
    fn test_generate_report_empty_context() {
        let ctx = ValidationContext::new();
        let report = generate_report(&ctx);

        assert_eq!(report.error_count, 0);
        assert_eq!(report.warning_count, 0);
        assert!(report.errors.is_empty());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn test_report_to_json_valid() {
        let ctx = make_context_with_errors_and_warnings();
        let report = generate_report(&ctx);
        let json = report.to_json().unwrap();

        assert!(json.contains("\"error_count\""));
        assert!(json.contains("\"errors\""));
        assert!(json.contains("\"warnings\""));
        assert!(json.contains("\"warning_count\""));
    }

    #[test]
    fn test_report_entry_fields() {
        let mut ctx = ValidationContext::new();
        ctx.add_error(
            ErrorKind::FileNotFound,
            "system/hydros.json",
            Some("hydro_042"),
            "required file is missing",
        );
        let report = generate_report(&ctx);

        assert_eq!(report.errors.len(), 1);
        let entry = &report.errors[0];

        assert_eq!(entry.kind, "FileNotFound");
        assert!(entry.file.contains("system/hydros.json"));
        assert_eq!(entry.entity.as_deref(), Some("hydro_042"));
        assert_eq!(entry.message, "required file is missing");
    }

    #[test]
    fn test_generate_report_does_not_consume_context() {
        let mut ctx = ValidationContext::new();
        ctx.add_error(
            ErrorKind::FileNotFound,
            "system/hydros.json",
            None::<&str>,
            "missing",
        );

        let report = generate_report(&ctx);
        assert_eq!(report.error_count, 1);

        assert!(ctx.into_result().is_err());
    }
}
