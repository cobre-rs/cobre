//! JSON Schema generation for all user-facing input types.
//!
//! This module is only compiled when the `schema` feature is enabled.
//! It provides [`generate_schemas`], which returns JSON Schema documents for
//! every case directory input file that users author by hand.
//!
//! # Usage
//!
//! ```rust
//! use cobre_io::schema::generate_schemas;
//!
//! let schemas = generate_schemas().expect("schema generation must not fail");
//! assert!(!schemas.is_empty());
//! for (filename, value) in &schemas {
//!     println!("{filename}: {} top-level keys", value.as_object().map_or(0, |o| o.len()));
//! }
//! ```

use crate::{
    config::Config,
    constraints::generic::RawGenericConstraintsFile,
    extensions::{
        production_models::RawProductionModelFile, scalar_parameters::ScalarParametersFile,
    },
    initial_conditions::RawInitialConditions,
    penalties::RawPenalties,
    post_study_stages::RawPostStudyStagesFile,
    scenarios::{
        correlation::RawCorrelationFile, load_factors::RawLoadFactorsFile,
        non_controllable_factors::RawNcsFactorsFile,
    },
    stages::RawStagesFile,
    system::{
        buses::RawBusFile, energy_contracts::RawContractFile, hydros::RawHydroFile,
        lines::RawLineFile, non_controllable::RawNcsFile, pumping_stations::RawPumpingFile,
        thermals::RawThermalFile,
    },
};

use std::io;
use std::path::{Path, PathBuf};

use serde_json::{Error, Value};

/// Generate JSON Schema documents for all user-facing case directory input files.
///
/// Returns a list of `(filename, schema_value)` pairs, where `filename` is the
/// conventional name of the generated schema file (e.g. `"config.schema.json"`)
/// and `schema_value` is the JSON Schema as a [`serde_json::Value`].
///
/// Covers all user-facing case directory inputs: configuration, system entities,
/// stages, penalties, constraints, scenarios, initial conditions, post-study
/// stages, and extensions.
///
/// # Errors
///
/// Returns [`serde_json::Error`] if any generated schema fails to serialize to
/// a [`serde_json::Value`]. In practice this should not occur because
/// `schemars` produces schema types that are always serializable, but the
/// error type is propagated for correctness.
///
/// # Examples
///
/// ```rust
/// use cobre_io::schema::generate_schemas;
///
/// let schemas = generate_schemas().expect("schema generation must not fail");
/// assert!(schemas.len() >= 17);
/// let config_schema = schemas.iter().find(|(name, _)| name == "config.schema.json");
/// assert!(config_schema.is_some());
/// ```
pub fn generate_schemas() -> Result<Vec<(String, Value)>, Error> {
    let pairs: Vec<(&str, schemars::Schema)> = vec![
        ("config.schema.json", schemars::schema_for!(Config)),
        ("buses.schema.json", schemars::schema_for!(RawBusFile)),
        ("hydros.schema.json", schemars::schema_for!(RawHydroFile)),
        (
            "thermals.schema.json",
            schemars::schema_for!(RawThermalFile),
        ),
        ("lines.schema.json", schemars::schema_for!(RawLineFile)),
        (
            "energy_contracts.schema.json",
            schemars::schema_for!(RawContractFile),
        ),
        (
            "non_controllable_sources.schema.json",
            schemars::schema_for!(RawNcsFile),
        ),
        (
            "pumping_stations.schema.json",
            schemars::schema_for!(RawPumpingFile),
        ),
        ("stages.schema.json", schemars::schema_for!(RawStagesFile)),
        ("penalties.schema.json", schemars::schema_for!(RawPenalties)),
        (
            "generic_constraints.schema.json",
            schemars::schema_for!(RawGenericConstraintsFile),
        ),
        (
            "load_factors.schema.json",
            schemars::schema_for!(RawLoadFactorsFile),
        ),
        (
            "non_controllable_factors.schema.json",
            schemars::schema_for!(RawNcsFactorsFile),
        ),
        (
            "correlation.schema.json",
            schemars::schema_for!(RawCorrelationFile),
        ),
        (
            "initial_conditions.schema.json",
            schemars::schema_for!(RawInitialConditions),
        ),
        (
            "post_study_stages.schema.json",
            schemars::schema_for!(RawPostStudyStagesFile),
        ),
        (
            "production_models.schema.json",
            schemars::schema_for!(RawProductionModelFile),
        ),
        (
            "generic_parameters.schema.json",
            schemars::schema_for!(ScalarParametersFile),
        ),
    ];

    pairs
        .into_iter()
        .map(|(name, schema)| {
            let value = serde_json::to_value(schema)?;
            Ok((name.to_string(), value))
        })
        .collect()
}

/// Errors from [`export_schemas`], distinguishing a [`generate_schemas`] failure
/// from a filesystem or per-file serialization failure on the write path, so a
/// caller can route each to a different error class.
#[derive(Debug, thiserror::Error)]
pub enum SchemaExportError {
    /// [`generate_schemas`] failed to produce a schema value.
    #[error("schema generation failed: {0}")]
    Generation(#[source] Error),

    /// A generated schema value could not be serialized to JSON text.
    #[error("serialization error for schema {filename}: {source}")]
    Serialization {
        /// Name of the schema file being serialized.
        filename: String,
        /// Underlying serialization error.
        source: Error,
    },

    /// The output directory could not be created or a schema file could not be written.
    #[error("I/O error exporting schema to {path}: {source}")]
    Io {
        /// Path to the directory or file involved in the failure.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
}

impl SchemaExportError {
    /// Construct a [`SchemaExportError::Io`] with path context.
    pub fn io(path: impl AsRef<Path>, source: io::Error) -> Self {
        Self::Io {
            path: path.as_ref().to_path_buf(),
            source,
        }
    }
}

/// Generate JSON Schema documents and write them to `output_dir`, creating it
/// if it does not exist. Returns the number of files written.
///
/// # Errors
///
/// Returns [`SchemaExportError::Generation`] if schema generation fails, or
/// [`SchemaExportError::Io`] if the output directory cannot be created, a
/// schema fails to serialize, or a file write fails.
pub fn export_schemas(output_dir: &Path) -> Result<usize, SchemaExportError> {
    std::fs::create_dir_all(output_dir)
        .map_err(|source| SchemaExportError::io(output_dir, source))?;

    let schemas = generate_schemas().map_err(SchemaExportError::Generation)?;
    let count = schemas.len();

    for (filename, value) in schemas {
        let dest = output_dir.join(&filename);
        let content = serde_json::to_string_pretty(&value).map_err(|source| {
            SchemaExportError::Serialization {
                filename: filename.clone(),
                source,
            }
        })?;
        std::fs::write(&dest, content).map_err(|source| SchemaExportError::io(&dest, source))?;
    }

    Ok(count)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_schemas_returns_expected_count() {
        let schemas = generate_schemas().unwrap();
        assert!(
            schemas.len() >= 17,
            "expected at least 17 schema entries, got {}",
            schemas.len()
        );
    }

    #[test]
    fn test_all_schema_filenames_and_values_non_empty() {
        let schemas = generate_schemas().unwrap();
        for (name, value) in &schemas {
            assert!(!name.is_empty(), "schema filename must not be empty");
            assert!(!value.is_null(), "schema value must not be null for {name}");
        }
    }

    #[test]
    fn test_all_schemas_are_objects() {
        let schemas = generate_schemas().unwrap();
        for (name, value) in &schemas {
            assert!(
                value.is_object(),
                "schema for {name} must be a JSON object, got: {value}"
            );
        }
    }

    #[test]
    fn test_all_schemas_have_structure_keys() {
        let schemas = generate_schemas().unwrap();
        for (name, value) in &schemas {
            let obj = value.as_object().unwrap_or_else(|| {
                panic!("schema for {name} is not an object");
            });
            // schemars v1 may hoist definitions and reference them; at minimum
            // a non-trivial schema always has one of these structural keys.
            assert!(
                obj.contains_key("properties")
                    || obj.contains_key("oneOf")
                    || obj.contains_key("anyOf")
                    || obj.contains_key("$defs"),
                "schema for {name} has no expected structural keys (properties/oneOf/anyOf/$defs)"
            );
        }
    }

    #[test]
    fn test_config_schema_contains_expected_fields() {
        let schemas = generate_schemas().unwrap();
        let (_, config_schema) = schemas
            .iter()
            .find(|(name, _)| name == "config.schema.json")
            .unwrap_or_else(|| panic!("config.schema.json not found in schemas"));

        let props = config_schema
            .pointer("/properties")
            .unwrap_or_else(|| panic!("config schema has no /properties"));

        let obj = props.as_object().unwrap_or_else(|| {
            panic!("config schema /properties is not an object");
        });

        for expected_field in &["training", "simulation", "exports"] {
            assert!(
                obj.contains_key(*expected_field),
                "config schema /properties should contain '{expected_field}'"
            );
        }
    }

    #[test]
    fn test_buses_schema_contains_buses_array() {
        let schemas = generate_schemas().unwrap();
        let (_, buses_schema) = schemas
            .iter()
            .find(|(name, _)| name == "buses.schema.json")
            .unwrap_or_else(|| panic!("buses.schema.json not found in schemas"));

        let props = buses_schema
            .pointer("/properties")
            .unwrap_or_else(|| panic!("buses schema has no /properties"));

        let obj = props.as_object().unwrap_or_else(|| {
            panic!("buses schema /properties is not an object");
        });

        assert!(
            obj.contains_key("buses"),
            "buses schema /properties should contain 'buses'"
        );
    }

    #[test]
    fn test_all_expected_schema_filenames_present() {
        let schemas = generate_schemas().unwrap();
        let names: Vec<&str> = schemas.iter().map(|(n, _)| n.as_str()).collect();

        let expected = [
            "config.schema.json",
            "buses.schema.json",
            "hydros.schema.json",
            "thermals.schema.json",
            "lines.schema.json",
            "energy_contracts.schema.json",
            "non_controllable_sources.schema.json",
            "pumping_stations.schema.json",
            "stages.schema.json",
            "penalties.schema.json",
            "generic_constraints.schema.json",
            "load_factors.schema.json",
            "non_controllable_factors.schema.json",
        ];

        for name in &expected {
            assert!(
                names.contains(name),
                "expected schema '{name}' not found; got: {names:?}"
            );
        }
    }

    #[test]
    fn test_export_schemas_writes_all_files_as_valid_json() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = dir.path();

        let count = export_schemas(output_dir).unwrap();

        let entries: Vec<_> = std::fs::read_dir(output_dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(count, entries.len());
        assert_eq!(count, generate_schemas().unwrap().len());

        for entry in &entries {
            let content = std::fs::read_to_string(entry).unwrap();
            let parsed: Value = serde_json::from_str(&content).unwrap();
            assert!(parsed.is_object(), "schema for {entry:?} must be an object");
        }
    }

    #[test]
    fn test_export_schemas_creates_missing_nested_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested").join("schemas");

        let count = export_schemas(&nested).unwrap();

        assert!(nested.is_dir());
        assert!(count > 0);
    }

    fn rustdoc_escapes(text: &str) -> Vec<String> {
        let mut found = Vec::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\'
                && let Some(escaped) = chars.next_if(char::is_ascii_punctuation)
            {
                found.push(format!("\\{escaped}"));
            }
        }
        found
    }

    fn collect_descriptions(value: &Value, pointer: &str, out: &mut Vec<(String, String)>) {
        match value {
            Value::Object(map) => {
                if let Some(Value::String(text)) = map.get("description") {
                    out.push((pointer.to_owned(), text.clone()));
                }
                for (key, child) in map {
                    collect_descriptions(child, &format!("{pointer}/{key}"), out);
                }
            }
            Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    collect_descriptions(child, &format!("{pointer}/{index}"), out);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn exported_schema_descriptions_carry_no_rustdoc_escapes() {
        let mut offences = Vec::new();
        for (name, schema) in generate_schemas().unwrap() {
            let mut descriptions = Vec::new();
            collect_descriptions(&schema, "", &mut descriptions);
            for (pointer, text) in descriptions {
                let escapes = rustdoc_escapes(&text);
                if !escapes.is_empty() {
                    offences.push(format!("{name} {pointer}: {}", escapes.join(" ")));
                }
            }
        }
        assert!(
            offences.is_empty(),
            "schema descriptions carry rustdoc escape artifacts:\n{}",
            offences.join("\n")
        );
    }

    #[test]
    fn rustdoc_escape_scan_flags_backslash_punctuation_only() {
        assert_eq!(rustdoc_escapes(r"Power \[MW\]."), ["\\[", "\\]"]);
        assert_eq!(rustdoc_escapes(r"a\_b \* c"), ["\\_", "\\*"]);
        for clean in [
            r"Window $\tau$",
            "see [`Type`]",
            "Power (MW).",
            "in [-1.0, 1.0]",
        ] {
            assert!(rustdoc_escapes(clean).is_empty(), "{clean:?} was flagged");
        }
    }
}
