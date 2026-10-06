//! Validation infrastructure for the cobre-io loading pipeline.
//!
//! This module provides the [`ValidationContext`] collector that every validation layer
//! reports into, the [`ErrorKind`] and [`Severity`] enums, and the [`rules`] table. A layer
//! adds a diagnostic only through [`ValidationContext::emit`], for a rule of
//! [`rules::RULES`], and that rule supplies the diagnostic's kind and severity.
//!
//! ## Design
//!
//! Validation in Cobre collects **all** errors before failing rather than stopping on the
//! first problem.  This lets users see and fix every issue in a single iteration.
//!
//! ```
//! use cobre_io::validation::{ValidationContext, rules::RULES};
//!
//! let missing_file = RULES
//!     .iter()
//!     .find(|rule| rule.id == "structural.2")
//!     .expect("structural.2 is listed");
//!
//! let mut ctx = ValidationContext::new();
//! ctx.emit(
//!     missing_file,
//!     "system/hydros.json",
//!     None::<&str>,
//!     "required file is missing",
//! );
//! assert!(ctx.has_errors());
//! assert!(ctx.into_result().is_err());
//! ```

pub mod dimensional;
pub mod productivity_resolution;
pub mod referential;
pub mod rules;
pub mod scalar_parameters;
pub mod schema;
pub mod semantic;
pub mod structural;

use std::path::PathBuf;

use crate::LoadError;

use self::rules::ValidationRule;

// ── Severity ─────────────────────────────────────────────────────────────────

/// Diagnostic severity attached to every [`ValidationEntry`].
///
/// `Error` entries prevent execution; `Warning` entries are reported but do not
/// block the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A validation failure that prevents execution.
    Error,
    /// A non-fatal observation that is reported but does not block execution.
    Warning,
}

// ── ErrorKind ────────────────────────────────────────────────────────────────

/// Categorises the kind of validation problem found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Required file is missing from the case directory.
    FileNotFound,
    /// File exists but cannot be parsed (invalid JSON syntax, unreadable Parquet header).
    ParseError,
    /// File parses successfully but does not conform to its expected schema
    /// (missing required field, wrong type, value out of valid range).
    SchemaViolation,
    /// A cross-entity foreign-key reference points to a non-existent entity.
    InvalidReference,
    /// Two entities in the same registry share the same ID.
    DuplicateId,
    /// A field value falls outside its valid range or violates a value constraint.
    InvalidValue,
    /// A directed graph (e.g., hydro cascade) contains a cycle.
    CycleDetected,
    /// A cross-file coverage check fails (e.g., missing inflow params for a hydro).
    DimensionMismatch,
    /// A domain-specific business rule is violated.
    BusinessRuleViolation,
    /// A stored policy loaded to seed training, or to run simulation only, cannot be used with the current case.
    WarmStartIncompatible,
    /// A stored checkpoint loaded to resume training cannot be used with the current case.
    ResumeIncompatible,
    /// A feature that is used in the input files is not yet implemented.
    NotImplemented,
    /// An entity is defined but appears to be inactive (warning only).
    UnusedEntity,
    /// A statistical quality concern in the input model (warning only).
    ModelQuality,
    /// A valid construct whose semantics are ambiguous or stage-dependent
    /// (warning only); surfaces the `thermal_generation` / `anticipated_decision`
    /// stage-drift case.
    SemanticAmbiguity,
}

// ── ValidationEntry ──────────────────────────────────────────────────────────

/// A single diagnostic emitted during validation.
///
/// Each entry records the source file, the affected entity (if any), the kind of
/// problem, and a human-readable message.
#[derive(Debug, Clone)]
pub struct ValidationEntry {
    /// Severity of this diagnostic.
    pub severity: Severity,
    /// Categorised kind of the problem.
    pub kind: ErrorKind,
    /// Path to the file in which the problem was detected.
    pub file: PathBuf,
    /// Optional identifier of the entity involved (e.g., `"hydro_042"`).
    pub entity: Option<String>,
    /// Human-readable description of the problem.
    pub message: String,
}

// ── ValidationContext ─────────────────────────────────────────────────────────

/// Collects all validation diagnostics emitted during the loading pipeline.
///
/// Pass a `&mut ValidationContext` to each validation function.  After all layers
/// have run, call [`into_result`] to convert the collected diagnostics into a
/// `Result<(), LoadError>`.
///
/// [`into_result`]: ValidationContext::into_result
///
/// # Examples
///
/// ```
/// use cobre_io::validation::ValidationContext;
///
/// let mut ctx = ValidationContext::new();
/// assert!(!ctx.has_errors());
/// assert!(ctx.into_result().is_ok());
/// ```
#[derive(Debug, Default)]
pub struct ValidationContext {
    entries: Vec<ValidationEntry>,
}

impl ValidationContext {
    /// Creates an empty validation context with no diagnostics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a diagnostic for `rule`, with the rule's own kind and severity.
    pub fn emit(
        &mut self,
        rule: &ValidationRule,
        file: impl Into<PathBuf>,
        entity: Option<impl Into<String>>,
        message: impl Into<String>,
    ) {
        self.entries.push(ValidationEntry {
            severity: rule.severity,
            kind: rule.kind,
            file: file.into(),
            entity: entity.map(Into::into),
            message: message.into(),
        });
    }

    /// Returns `true` if any error-severity diagnostics have been collected.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.entries.iter().any(|e| e.severity == Severity::Error)
    }

    /// Returns the number of error-severity diagnostics without allocating.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.severity == Severity::Error)
            .count()
    }

    /// Returns all error-severity [`ValidationEntry`] items.
    #[must_use]
    pub fn errors(&self) -> Vec<&ValidationEntry> {
        self.entries
            .iter()
            .filter(|e| e.severity == Severity::Error)
            .collect()
    }

    /// Returns a slice of all warning-severity [`ValidationEntry`] items.
    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationEntry> {
        self.entries
            .iter()
            .filter(|e| e.severity == Severity::Warning)
            .collect()
    }

    /// Converts the collected diagnostics into a `Result`.
    ///
    /// Warnings are not surfaced by this method — inspect [`warnings()`] before
    /// calling `into_result()`.
    ///
    /// [`warnings()`]: ValidationContext::warnings
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::ConstraintError`] if any error-severity diagnostics
    /// were collected.  The `description` field contains all error messages
    /// joined by newlines, formatted as `[ErrorKind] file (entity): message`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_io::validation::{ValidationContext, rules::RULES};
    ///
    /// let stub_term = RULES
    ///     .iter()
    ///     .find(|rule| rule.id == "referential.11")
    ///     .expect("referential.11 is listed");
    ///
    /// let mut ctx = ValidationContext::new();
    /// ctx.emit(
    ///     stub_term,
    ///     "constraints/generic_constraints.json",
    ///     Some("GenericConstraint 1 term[0]"),
    ///     "GenericConstraint 1 term[0] references Contract 3 which is a stub entity with no LP effect",
    /// );
    /// assert!(ctx.into_result().is_ok());
    /// ```
    pub fn into_result(self) -> Result<(), LoadError> {
        let error_messages: Vec<String> = self
            .entries
            .iter()
            .filter(|e| e.severity == Severity::Error)
            .map(|e| {
                let file = e.file.display();
                if let Some(entity) = &e.entity {
                    format!("[{:?}] {file} ({entity}): {}", e.kind, e.message)
                } else {
                    format!("[{:?}] {file}: {}", e.kind, e.message)
                }
            })
            .collect();

        if error_messages.is_empty() {
            return Ok(());
        }
        Err(LoadError::ConstraintError {
            description: error_messages.join("\n"),
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_context_empty() {
        let ctx = ValidationContext::new();
        assert!(!ctx.has_errors(), "new context should have no errors");
        assert!(
            ctx.errors().is_empty(),
            "new context should have empty errors list"
        );
        assert!(
            ctx.warnings().is_empty(),
            "new context should have empty warnings list"
        );
        assert!(
            ctx.into_result().is_ok(),
            "empty context should produce Ok result"
        );
    }

    #[test]
    fn test_context_errors_collected() {
        let mut ctx = ValidationContext::new();
        ctx.emit(
            &rules::STRUCTURAL_REQUIRED_FILE_MISSING,
            "system/hydros.json",
            None::<&str>,
            "file missing",
        );
        ctx.emit(
            &rules::SCHEMA_FILE_UNPARSABLE,
            "stages.json",
            None::<&str>,
            "malformed JSON",
        );
        ctx.emit(
            &rules::SCHEMA_FILE_NONCONFORMING,
            "system/buses.json",
            Some("bus_42"),
            "missing field bus_id",
        );
        assert!(
            ctx.has_errors(),
            "context with 3 errors should report has_errors=true"
        );
        assert_eq!(
            ctx.errors().len(),
            3,
            "errors() should return exactly 3 entries"
        );
    }

    #[test]
    fn test_context_warnings_not_errors() {
        let mut ctx = ValidationContext::new();
        ctx.emit(
            &rules::REFERENTIAL_GENERIC_TERM_STUB_CONTRACT,
            "system/thermals.json",
            Some("thermal_old"),
            "max_generation=0 for all stages",
        );
        ctx.emit(
            &rules::SEMANTIC_SOBOL_OPENING_COUNT,
            "scenarios/inflow_seasonal_stats.parquet",
            None::<&str>,
            "residual bias detected",
        );
        assert!(
            !ctx.has_errors(),
            "context with only warnings should report has_errors=false"
        );
        assert_eq!(
            ctx.warnings().len(),
            2,
            "warnings() should return exactly 2 entries"
        );
        assert!(
            ctx.errors().is_empty(),
            "errors() should be empty when only warnings exist"
        );
    }

    #[test]
    fn test_context_into_result_with_errors() {
        let mut ctx = ValidationContext::new();
        ctx.emit(
            &rules::STRUCTURAL_REQUIRED_FILE_MISSING,
            "system/hydros.json",
            None::<&str>,
            "required file is missing",
        );
        let result = ctx.into_result();
        assert!(result.is_err(), "context with errors should produce Err");
        let err = result.unwrap_err();
        let display = err.to_string();
        assert!(
            display.contains("required file is missing"),
            "error description should contain the original message, got: {display}"
        );
    }

    #[test]
    fn test_context_into_result_warnings_only_is_ok() {
        let mut ctx = ValidationContext::new();
        ctx.emit(
            &rules::REFERENTIAL_GENERIC_TERM_STUB_CONTRACT,
            "system/thermals.json",
            Some("T1"),
            "inactive thermal",
        );
        assert!(
            ctx.into_result().is_ok(),
            "context with only warnings should produce Ok"
        );
    }

    #[test]
    fn test_context_into_result_multiple_errors_joined() {
        let mut ctx = ValidationContext::new();
        ctx.emit(
            &rules::STRUCTURAL_REQUIRED_FILE_MISSING,
            "system/hydros.json",
            None::<&str>,
            "file alpha missing",
        );
        ctx.emit(
            &rules::STRUCTURAL_REQUIRED_FILE_MISSING,
            "system/buses.json",
            None::<&str>,
            "file beta missing",
        );
        let result = ctx.into_result();
        assert!(result.is_err());
        let description = result.unwrap_err().to_string();
        assert!(
            description.contains("file alpha missing"),
            "description should contain first error, got: {description}"
        );
        assert!(
            description.contains("file beta missing"),
            "description should contain second error, got: {description}"
        );
    }
}
