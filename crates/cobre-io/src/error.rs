//! Error types for the `cobre-io` loading pipeline.
//!
//! [`LoadError`] is the primary error type returned by [`crate::load_case`] and every
//! internal parsing function. Each variant carries enough context for the caller to
//! produce a diagnostic message without re-reading the input files.

use std::path::{Path, PathBuf};

/// Errors that can occur during case loading.
///
/// Variants are ordered by the pipeline phase in which they typically occur:
/// I/O read → parse → schema validation → cross-reference validation → semantic
/// constraint validation → warm-start policy compatibility.
///
/// # Examples
///
/// ```
/// use cobre_io::LoadError;
/// use std::path::PathBuf;
///
/// let err = LoadError::SchemaError {
///     path: PathBuf::from("system/hydros.json"),
///     field: "bus_id".to_string(),
///     message: "required field is missing".to_string(),
/// };
/// assert!(err.to_string().contains("bus_id"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Filesystem read failure (file not found, permission denied, I/O error).
    #[error("I/O error reading {path}: {source}")]
    IoError {
        /// Path to the file that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// JSON or Parquet parsing failure (malformed content, encoding error).
    #[error("parse error in {path}: {message}")]
    ParseError {
        /// Path to the file that failed to parse.
        path: PathBuf,
        /// Human-readable description of the parse failure.
        message: String,
    },

    /// Schema validation failure (missing required field, wrong type, value out of range).
    #[error("schema error in {path}, field {field}: {message}")]
    SchemaError {
        /// Path to the file containing the invalid entry.
        path: PathBuf,
        /// Dot-separated field path within the JSON object (e.g., `"hydros[3].bus_id"`).
        field: String,
        /// Human-readable description of the schema violation.
        message: String,
    },

    /// Cross-reference validation failure (dangling entity ID, broken foreign key).
    #[error(
        "cross-reference error: {source_entity} in {source_file} references \
         non-existent {target_entity} in {target_collection}"
    )]
    CrossReferenceError {
        /// Path to the file that contains the dangling reference.
        source_file: PathBuf,
        /// String identifier of the entity that holds the broken reference
        /// (e.g., `"Hydro 'H1'"`).
        source_entity: String,
        /// Name of the collection that was expected to contain `target_entity`
        /// (e.g., `"bus registry"`).
        target_collection: String,
        /// String identifier of the entity that could not be found
        /// (e.g., `"BUS_99"`).
        target_entity: String,
    },

    /// Semantic constraint violation (acyclic cascade, complete coverage, consistency).
    #[error("constraint violation: {description}")]
    ConstraintError {
        /// Human-readable description of the violated constraint.
        description: String,
    },

    /// Warm-start policy is structurally incompatible with the current system.
    ///
    /// See SS7.1 in `input-loading-pipeline.md` for the four compatibility checks.
    #[error(
        "policy incompatible: {check} mismatch — policy has {policy_value}, \
         system has {system_value}"
    )]
    PolicyIncompatible {
        /// Name of the failing compatibility check (e.g., `"hydro count"`).
        check: String,
        /// Value recorded in the policy file.
        policy_value: String,
        /// Value present in the current system.
        system_value: String,
    },
}

impl LoadError {
    /// Construct an [`LoadError::IoError`] wrapping an [`std::io::Error`] with path context.
    ///
    /// Prefer this helper over constructing the variant directly to ensure consistent
    /// path handling. Do **not** implement `From<std::io::Error>` — that conversion loses
    /// the path context required for diagnostic messages.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_io::LoadError;
    /// use std::io;
    ///
    /// let io_err = io::Error::new(io::ErrorKind::NotFound, "no such file");
    /// let err = LoadError::io("system/hydros.json", io_err);
    /// assert!(err.to_string().contains("system/hydros.json"));
    /// ```
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self::IoError {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    /// Construct a [`LoadError::ParseError`] with path context and a message.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_io::LoadError;
    ///
    /// let err = LoadError::parse("stages.json", "unexpected end of input");
    /// assert!(err.to_string().contains("stages.json"));
    /// assert!(err.to_string().contains("unexpected end of input"));
    /// ```
    pub fn parse(path: impl AsRef<Path>, message: impl Into<String>) -> Self {
        Self::ParseError {
            path: path.as_ref().to_path_buf(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_load_error_io_display() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "no such file or directory");
        let err = LoadError::io("system/hydros.json", io_err);
        let display = err.to_string();
        assert!(
            display.contains("system/hydros.json"),
            "display should contain path, got: {display}"
        );
        assert!(
            display.contains("no such file or directory"),
            "display should contain source message, got: {display}"
        );
    }

    #[test]
    fn test_load_error_parse_display() {
        let err = LoadError::parse("stages.json", "unexpected end of input");
        let display = err.to_string();
        assert!(
            display.contains("stages.json"),
            "display should contain path, got: {display}"
        );
        assert!(
            display.contains("unexpected end of input"),
            "display should contain message, got: {display}"
        );
    }

    #[test]
    fn test_load_error_schema_display() {
        let err = LoadError::SchemaError {
            path: PathBuf::from("system/hydros.json"),
            field: "bus_id".to_string(),
            message: "required field is missing".to_string(),
        };
        let display = err.to_string();
        assert!(
            display.contains("system/hydros.json"),
            "display should contain path, got: {display}"
        );
        assert!(
            display.contains("bus_id"),
            "display should contain field name, got: {display}"
        );
        assert!(
            display.contains("required field is missing"),
            "display should contain message, got: {display}"
        );
    }

    #[test]
    fn test_load_error_cross_reference_display() {
        let err = LoadError::CrossReferenceError {
            source_file: PathBuf::from("system/hydros.json"),
            source_entity: "Hydro 'H1'".to_string(),
            target_collection: "bus registry".to_string(),
            target_entity: "BUS_99".to_string(),
        };
        let display = err.to_string();
        assert!(
            display.contains("Hydro 'H1'"),
            "display should contain source_entity, got: {display}"
        );
        assert!(
            display.contains("system/hydros.json"),
            "display should contain source_file, got: {display}"
        );
        assert!(
            display.contains("BUS_99"),
            "display should contain target_entity, got: {display}"
        );
        assert!(
            display.contains("bus registry"),
            "display should contain target_collection, got: {display}"
        );
    }

    #[test]
    fn test_load_error_is_std_error() {
        let err = LoadError::ConstraintError {
            description: "hydro cascade contains a cycle".to_string(),
        };
        // Verify LoadError can be used as &dyn std::error::Error by calling source() on it.
        let dyn_err: &dyn std::error::Error = &err;
        // ConstraintError has no source, so source() returns None.
        assert!(dyn_err.source().is_none());
        assert!(err.to_string().contains("hydro cascade contains a cycle"));
    }

    #[test]
    fn test_load_error_io_helper() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let err = LoadError::io("config.json", io_err);
        // Verify the helper constructs the IoError variant correctly
        assert!(matches!(err, LoadError::IoError { .. }));
        let display = err.to_string();
        assert!(display.contains("config.json"));
        assert!(display.contains("permission denied"));
    }
}
