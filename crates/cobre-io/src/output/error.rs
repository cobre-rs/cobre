//! Error types for the `cobre-io` output writing pipeline.
//!
//! [`OutputError`] is the primary error type returned by all output writer functions.
//! Each variant carries enough context for the caller to produce a diagnostic message
//! without re-reading the output files.

use std::path::{Path, PathBuf};

/// Errors that can occur during output writing operations.
///
/// # Examples
///
/// ```
/// use cobre_io::OutputError;
/// use std::path::PathBuf;
///
/// let err = OutputError::SchemaError {
///     file: "convergence.parquet".to_string(),
///     column: "iteration".to_string(),
///     message: "column type mismatch".to_string(),
/// };
/// assert!(err.to_string().contains("convergence.parquet"));
/// assert!(err.to_string().contains("iteration"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    /// Filesystem write failure.
    #[error("I/O error writing {path}: {source}")]
    IoError {
        /// Path to the file that could not be written.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// Arrow/Parquet encoding failure.
    #[error("serialization error for entity {entity}: {message}")]
    SerializationError {
        /// Name of the entity collection (e.g., `"hydros"`).
        entity: String,
        /// Human-readable error description.
        message: String,
    },

    /// Parquet schema validation failure.
    #[error("schema error in {file}, column {column}: {message}")]
    SchemaError {
        /// Name of the Parquet file being validated.
        file: String,
        /// Name of the column that failed validation.
        column: String,
        /// Human-readable error description.
        message: String,
    },

    /// Manifest construction failure.
    #[error("manifest error for {manifest_type}: {message}")]
    ManifestError {
        /// Type of manifest being constructed.
        manifest_type: String,
        /// Human-readable error description.
        message: String,
    },
}

impl OutputError {
    /// Construct an [`OutputError::IoError`] from a path and source error.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_io::OutputError;
    /// use std::io;
    ///
    /// let io_err = io::Error::new(io::ErrorKind::NotFound, "no such file");
    /// let err = OutputError::io("simulation/costs/data.parquet", io_err);
    /// assert!(err.to_string().contains("simulation/costs/data.parquet"));
    /// ```
    pub fn io(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self::IoError {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    /// Construct an [`OutputError::SerializationError`] with entity context and a message.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_io::OutputError;
    ///
    /// let err = OutputError::serialization("hydros", "unsupported field type");
    /// assert!(err.to_string().contains("hydros"));
    /// assert!(err.to_string().contains("unsupported field type"));
    /// ```
    pub fn serialization(entity: impl Into<String>, message: impl Into<String>) -> Self {
        Self::SerializationError {
            entity: entity.into(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io;

    fn assert_send_sync_static<E: std::error::Error + Send + Sync + 'static>() {}

    #[test]
    fn display_io_error_contains_path_and_source() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "no such file or directory");
        let err = OutputError::io("simulation/costs/data.parquet", io_err);
        let display = err.to_string();
        assert!(
            display.contains("simulation/costs/data.parquet"),
            "display should contain path, got: {display}"
        );
        assert!(
            display.contains("no such file or directory"),
            "display should contain source message, got: {display}"
        );
    }

    #[test]
    fn display_serialization_error_contains_entity_and_message() {
        let err = OutputError::serialization("hydros", "unsupported field type");
        let display = err.to_string();
        assert!(
            display.contains("hydros"),
            "display should contain entity, got: {display}"
        );
        assert!(
            display.contains("unsupported field type"),
            "display should contain message, got: {display}"
        );
    }

    #[test]
    fn display_schema_error_contains_file_and_column() {
        let err = OutputError::SchemaError {
            file: "convergence.parquet".to_string(),
            column: "iteration".to_string(),
            message: "column type mismatch".to_string(),
        };
        let display = err.to_string();
        assert!(
            display.contains("convergence.parquet"),
            "display should contain file name, got: {display}"
        );
        assert!(
            display.contains("iteration"),
            "display should contain column name, got: {display}"
        );
        assert!(
            display.contains("column type mismatch"),
            "display should contain message, got: {display}"
        );
    }

    #[test]
    fn display_manifest_error_contains_type_and_message() {
        let err = OutputError::ManifestError {
            manifest_type: "simulation".to_string(),
            message: "failed to serialize partition list".to_string(),
        };
        let display = err.to_string();
        assert!(
            display.contains("simulation"),
            "display should contain manifest_type, got: {display}"
        );
        assert!(
            display.contains("failed to serialize partition list"),
            "display should contain message, got: {display}"
        );
    }

    #[test]
    fn output_error_is_send_sync_static() {
        assert_send_sync_static::<OutputError>();
    }

    #[test]
    fn output_error_satisfies_std_error_trait() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let variants: Vec<OutputError> = vec![
            OutputError::io("output/data.parquet", io_err),
            OutputError::serialization("thermals", "batch size mismatch"),
            OutputError::SchemaError {
                file: "costs.parquet".to_string(),
                column: "cost".to_string(),
                message: "expected Float64".to_string(),
            },
            OutputError::ManifestError {
                manifest_type: "policy".to_string(),
                message: "missing required field".to_string(),
            },
        ];
        for err in &variants {
            let _: &dyn std::error::Error = err;
        }
    }

    #[test]
    fn io_helper_constructs_correct_variant() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "permission denied");
        let err = OutputError::io("output/costs.parquet", io_err);
        assert!(
            matches!(err, OutputError::IoError { .. }),
            "io() helper must construct IoError variant"
        );
        let display = err.to_string();
        assert!(display.contains("output/costs.parquet"));
        assert!(display.contains("permission denied"));
    }

    #[test]
    fn serialization_helper_constructs_correct_variant() {
        let err = OutputError::serialization("lines", "record batch has wrong schema");
        assert!(
            matches!(err, OutputError::SerializationError { .. }),
            "serialization() helper must construct SerializationError variant"
        );
        let display = err.to_string();
        assert!(display.contains("lines"));
        assert!(display.contains("record batch has wrong schema"));
    }

    #[test]
    fn all_variants_debug_non_empty() {
        let io_err = io::Error::other("disk full");
        let variants: Vec<OutputError> = vec![
            OutputError::io("output/data.parquet", io_err),
            OutputError::serialization("buses", "null value in non-nullable column"),
            OutputError::SchemaError {
                file: "flows.parquet".to_string(),
                column: "flow_mw".to_string(),
                message: "expected Float64, got Int32".to_string(),
            },
            OutputError::ManifestError {
                manifest_type: "simulation".to_string(),
                message: "partition list is empty".to_string(),
            },
        ];
        for err in &variants {
            assert!(
                !format!("{err:?}").is_empty(),
                "Debug output must not be empty for variant: {err}"
            );
        }
    }
}
