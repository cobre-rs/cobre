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
    fn stages_schema_does_not_accept_scenario_source() {
        let schemas = generate_schemas().unwrap();
        let (_, stages_schema) = schemas
            .iter()
            .find(|(name, _)| name == "stages.schema.json")
            .unwrap_or_else(|| panic!("stages.schema.json not found in schemas"));

        assert_eq!(stages_schema.pointer("/properties/scenario_source"), None);
        assert_eq!(
            stages_schema.pointer("/additionalProperties"),
            Some(&Value::Bool(false))
        );
        for expected_field in [
            "policy_graph",
            "stages",
            "pre_study_stages",
            "season_definitions",
        ] {
            assert!(
                stages_schema
                    .pointer(&format!("/properties/{expected_field}"))
                    .is_some(),
                "stages schema /properties should contain '{expected_field}'"
            );
        }
    }

    #[test]
    fn config_schema_requires_training_selection_and_stopping_rules() {
        let schemas = generate_schemas().unwrap();
        let (_, config_schema) = schemas
            .iter()
            .find(|(name, _)| name == "config.schema.json")
            .unwrap_or_else(|| panic!("config.schema.json not found in schemas"));

        let required = config_schema
            .pointer("/$defs/TrainingConfig/required")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("TrainingConfig has no required list"));
        for key in ["selection", "stopping_rules"] {
            assert!(
                required.iter().any(|entry| entry == key),
                "TrainingConfig should require '{key}', got: {required:?}"
            );
        }

        assert_eq!(
            config_schema.pointer("/$defs/TrainingConfig/properties/stopping_rules/type"),
            Some(&Value::String("array".to_string()))
        );

        let selection = config_schema
            .pointer("/$defs/TrainingConfig/properties/selection")
            .unwrap_or_else(|| panic!("TrainingConfig has no selection property"));
        assert_eq!(selection.get("anyOf"), None);
        assert_eq!(selection.get("default"), None);
        assert_eq!(
            selection
                .get("oneOf")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn config_schema_requires_an_iteration_limit_stopping_rule() {
        let schemas = generate_schemas().unwrap();
        let (_, config_schema) = schemas
            .iter()
            .find(|(name, _)| name == "config.schema.json")
            .unwrap_or_else(|| panic!("config.schema.json not found in schemas"));

        assert_eq!(
            config_schema.pointer("/$defs/TrainingConfig/properties/stopping_rules/contains"),
            Some(&serde_json::json!({
                "required": ["type"],
                "properties": {"type": {"const": "iteration_limit"}}
            }))
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

    const IMPLEMENTATION_MARKERS_ANY_CASE: [&str; 11] = [
        "intermediate type",
        "intermediate serde",
        "intermediate enum",
        "intermediate untagged",
        "intermediate representation",
        "serde",
        "deserializ",
        "re-export",
        "untagged",
        "internally tagged",
        "internally-tagged",
    ];
    const IMPLEMENTATION_MARKERS: [&str; 16] = [
        "#[",
        "deny_unknown_fields",
        "::",
        "`None`",
        "`Some(",
        "Option<",
        "Vec<",
        "HashMap<",
        "<f64>",
        "<u32>",
        "`f64`",
        "`u32`",
        "`i32`",
        "`usize`",
        "```\n",
        "```rust",
    ];

    fn rustdoc_links(text: &str) -> Vec<String> {
        let mut found = Vec::new();
        let mut rest = text;
        while let Some(open) = rest.find("[`") {
            let Some(close) = rest[open + 2..].find("`]") else {
                break;
            };
            let end = open + 2 + close + 2;
            if !matches!(rest[end..].chars().next(), Some('(' | '[')) {
                found.push(rest[open..end].to_owned());
            }
            rest = &rest[end..];
        }
        found
    }

    fn raw_latex(text: &str) -> Vec<String> {
        let mut found = Vec::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\'
                && let Some(letter) = chars.next_if(char::is_ascii_alphabetic)
            {
                found.push(format!("\\{letter}"));
            }
        }
        let mut rest = text;
        while let Some(open) = rest.find('$') {
            let Some(close) = rest[open + 1..].find('$') else {
                break;
            };
            let inner = &rest[open + 1..open + 1 + close];
            let is_latex = inner
                .chars()
                .next()
                .is_some_and(|first| first.is_ascii_alphabetic() || first == '\\')
                && !inner.contains(char::is_whitespace)
                && inner.contains(['_', '^', '{', '\\']);
            if is_latex {
                found.push(format!("${inner}$"));
                rest = &rest[open + close + 2..];
            } else {
                rest = &rest[open + 1 + close..];
            }
        }
        found
    }

    fn implementation_wording(text: &str) -> Vec<&'static str> {
        let lower = text.to_lowercase();
        IMPLEMENTATION_MARKERS_ANY_CASE
            .into_iter()
            .filter(|marker| lower.contains(marker))
            .chain(
                IMPLEMENTATION_MARKERS
                    .into_iter()
                    .filter(|marker| text.contains(marker)),
            )
            .collect()
    }

    fn case_author_artifacts(text: &str) -> Vec<String> {
        let mut found = rustdoc_escapes(text);
        found.extend(rustdoc_links(text));
        found.extend(raw_latex(text));
        found.extend(implementation_wording(text).into_iter().map(str::to_owned));
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
    fn exported_schema_descriptions_are_written_for_case_authors() {
        let mut offences = Vec::new();
        for (name, schema) in generate_schemas().unwrap() {
            let mut descriptions = Vec::new();
            collect_descriptions(&schema, "", &mut descriptions);
            for (pointer, text) in descriptions {
                let artifacts = case_author_artifacts(&text);
                if !artifacts.is_empty() {
                    offences.push(format!("{name} {pointer}: {}", artifacts.join(" ")));
                }
            }
        }
        assert!(
            offences.is_empty(),
            "schema descriptions carry artifacts not written for case authors:\n{}",
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

    #[test]
    fn case_author_scan_flags_rust_artifacts_and_spares_author_text() {
        assert_eq!(rustdoc_links("see [`Self::Pacf`] path"), ["[`Self::Pacf`]"]);
        assert!(rustdoc_links("a [`x`](https://example.org) link").is_empty());
        assert_eq!(raw_latex(r"Window size $\tau$."), [r"\t", r"$\tau$"]);
        assert_eq!(raw_latex("Maximum count $k_{max}$."), ["$k_{max}$"]);

        let flagged: [(&str, &[&str]); 17] = [
            (
                "Top-level intermediate type for `hydros.json`.",
                &["intermediate type"],
            ),
            (
                "Intermediate serde type for `config.json`.",
                &["intermediate serde", "serde"],
            ),
            (
                "Raw intermediate enum for contract direction.",
                &["intermediate enum"],
            ),
            (
                "Intermediate untagged union for `risk_measure`.",
                &["intermediate untagged", "untagged"],
            ),
            (
                "Per-entry intermediate representation.",
                &["intermediate representation"],
            ),
            (
                "Private — only used during deserialization. Not re-exported.",
                &["deserializ", "re-export"],
            ),
            (
                "Untagged with per-variant `deny_unknown_fields`.",
                &["untagged", "deny_unknown_fields"],
            ),
            ("Internally tagged on `method`.", &["internally tagged"]),
            ("An internally-tagged union.", &["internally-tagged"]),
            ("Uses `#[serde(tag = \"model\")]`.", &["serde", "#["]),
            (
                "`cobre_core::AnticipatedConfig` keeps a plain derive.",
                &["::"],
            ),
            (
                "Defaults to `None`; `Some(n)` caps it.",
                &["`None`", "`Some("],
            ),
            (
                "Fields are `Option<f64>` in a `Vec<i32>` keyed by `HashMap<K, V>`.",
                &["Option<", "Vec<", "HashMap<"],
            ),
            (
                "Shape `{ \"tolerance_deg\": <f64>, \"n_samples\": <u32> }`.",
                &["<f64>", "<u32>"],
            ),
            (
                "Wraps the `i32` id, not a `usize` index; `f64` and `u32` fields.",
                &["`i32`", "`usize`", "`f64`", "`u32`"],
            ),
            ("# Examples\n\n```\nlet x = 1;\n```", &["```\n"]),
            ("```rust\nlet x = 1;\n```", &["```rust"]),
        ];
        for (text, markers) in flagged {
            let found = implementation_wording(text);
            for marker in markers {
                assert!(found.contains(marker), "{text:?} missed {marker:?}");
            }
        }

        let spared = [
            "Power (MW).",
            "Cost ($/`MWh`) and ($/hm³).",
            "Penalty ($/(m³/s·h)).",
            "in [-1.0, 1.0]",
            "Method: `\"none\"` or `\"truncation\"`.",
            "An array such as `[1940, 1953, 1971]`.",
            "Must be symmetric: `|m[i][j] - m[j][i]| <= 1e-10`.",
            "```json\n{}\n```",
            "Intermediate stages are allowed.",
            "Between $10 and $20.",
            "The `method` key selects the scheduler.",
        ];
        for text in spared {
            let artifacts = case_author_artifacts(text);
            assert!(artifacts.is_empty(), "{text:?} was flagged: {artifacts:?}");
        }
    }

    #[test]
    fn hydro_penalty_descriptions_state_the_priced_unit() {
        const FLOW: &str = "($/(m³/s·h))";
        const STORAGE: &str = "($/hm³)";
        const ENERGY: &str = "($/`MWh`)";
        let units: [(&str, &str); 16] = [
            ("spillage_cost", FLOW),
            ("turbined_cost", FLOW),
            ("diversion_cost", FLOW),
            ("storage_violation_below_cost", STORAGE),
            ("filling_target_violation_cost", STORAGE),
            ("turbined_violation_below_cost", FLOW),
            ("outflow_violation_below_cost", FLOW),
            ("outflow_violation_above_cost", FLOW),
            ("generation_violation_below_cost", ENERGY),
            ("evaporation_violation_cost", FLOW),
            ("water_withdrawal_violation_cost", FLOW),
            ("water_withdrawal_violation_pos_cost", FLOW),
            ("water_withdrawal_violation_neg_cost", FLOW),
            ("evaporation_violation_pos_cost", FLOW),
            ("evaporation_violation_neg_cost", FLOW),
            ("inflow_nonnegativity_cost", FLOW),
        ];
        let schemas = generate_schemas().unwrap();
        let properties = |file: &str, def: &str| {
            let (_, schema) = schemas
                .iter()
                .find(|(name, _)| name == file)
                .unwrap_or_else(|| panic!("{file} not found in schemas"));
            schema
                .pointer(&format!("/$defs/{def}/properties"))
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("{file} has no /$defs/{def}/properties"))
                .clone()
        };

        let mut offences = Vec::new();
        for (file, def) in [
            ("penalties.schema.json", "RawHydroPenalties"),
            ("hydros.schema.json", "RawHydroPenaltyOverrides"),
        ] {
            for (key, property) in properties(file, def) {
                let description = property.get("description").and_then(Value::as_str);
                match units.iter().find(|(unit_key, _)| *unit_key == key) {
                    None => offences.push(format!("{file} {def}.{key}: has no unit in the table")),
                    Some((_, unit)) if !description.is_some_and(|text| text.contains(unit)) => {
                        offences.push(format!(
                            "{file} {def}.{key}: {} lacks {unit}",
                            description.unwrap_or("no description")
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
        assert!(
            offences.is_empty(),
            "hydro penalty descriptions do not state the priced unit:\n{}",
            offences.join("\n")
        );

        let mut hydro_section: Vec<String> =
            properties("penalties.schema.json", "RawHydroPenalties")
                .keys()
                .cloned()
                .collect();
        hydro_section.sort_unstable();
        let mut table: Vec<&str> = units.iter().map(|(key, _)| *key).collect();
        table.sort_unstable();
        assert_eq!(hydro_section, table);
    }

    fn schema_named<'a>(schemas: &'a [(String, Value)], name: &str) -> &'a Value {
        let Some((_, schema)) = schemas.iter().find(|(file, _)| file == name) else {
            panic!("{name} not found in schemas");
        };
        schema
    }

    #[test]
    fn entity_penalty_override_descriptions_cite_penalties_json_keys() {
        let schemas = generate_schemas().unwrap();
        let penalties = schema_named(&schemas, "penalties.schema.json");
        for (file, def, property, section, key) in [
            (
                "non_controllable_sources.schema.json",
                "RawNcs",
                "curtailment_cost",
                "non_controllable_source",
                "curtailment_cost",
            ),
            (
                "lines.schema.json",
                "RawLine",
                "exchange_cost",
                "line",
                "exchange_cost",
            ),
            (
                "buses.schema.json",
                "RawBus",
                "deficit_segments",
                "bus",
                "deficit_segments",
            ),
        ] {
            let pointer = format!("/$defs/{def}/properties/{property}/description");
            let description = schema_named(&schemas, file)
                .pointer(&pointer)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("{file} has no {pointer}"));
            let cited = format!("`{section}.{key}`");
            assert!(
                description.contains(&cited) && description.contains("`penalties.json`"),
                "{file} {pointer} does not cite {cited} in `penalties.json`: {description:?}"
            );

            let reference = penalties
                .pointer(&format!("/properties/{section}/$ref"))
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    panic!("penalties.schema.json has no /properties/{section}/$ref")
                });
            let section_properties = reference
                .strip_prefix('#')
                .and_then(|def_pointer| penalties.pointer(def_pointer))
                .and_then(|section_def| section_def.get("properties"))
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("penalties.schema.json {reference} has no properties"));
            assert!(
                section_properties.contains_key(key),
                "penalties.schema.json {reference}/properties has no {key}"
            );
        }
    }

    #[test]
    fn hydro_penalty_override_fields_are_fields_of_the_penalties_hydro_section() {
        let schemas = generate_schemas().unwrap();
        let penalties = schema_named(&schemas, "penalties.schema.json");
        let hydros = schema_named(&schemas, "hydros.schema.json");

        let reference = penalties
            .pointer("/properties/hydro/$ref")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("penalties.schema.json has no /properties/hydro/$ref"));
        let hydro_section = reference
            .strip_prefix('#')
            .and_then(|def_pointer| penalties.pointer(def_pointer))
            .and_then(|section_def| section_def.get("properties"))
            .and_then(Value::as_object)
            .unwrap_or_else(|| panic!("penalties.schema.json {reference} has no properties"));
        let overrides = hydros
            .pointer("/$defs/RawHydroPenaltyOverrides/properties")
            .and_then(Value::as_object)
            .unwrap_or_else(|| {
                panic!("hydros.schema.json has no /$defs/RawHydroPenaltyOverrides/properties")
            });
        let foreign: Vec<&String> = overrides
            .keys()
            .filter(|key| !hydro_section.contains_key(*key))
            .collect();
        assert!(
            foreign.is_empty(),
            "hydros.schema.json /$defs/RawHydroPenaltyOverrides/properties names keys absent \
             from penalties.schema.json {reference}/properties: {foreign:?}"
        );

        let pointer = "/$defs/RawHydro/properties/penalties/description";
        let description = hydros
            .pointer(pointer)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("hydros.schema.json has no {pointer}"));
        assert!(
            description.contains("`hydro`") && description.contains("`penalties.json`"),
            "hydros.schema.json {pointer} does not cite the `hydro` section of \
             `penalties.json`: {description:?}"
        );
    }

    #[test]
    fn generic_parameters_schema_enumerates_every_parameter_kind() {
        let schemas = generate_schemas().unwrap();
        let parameters = schema_named(&schemas, "generic_parameters.schema.json");
        let pointer = "/$defs/ScalarParameterJsonEntry/properties/kind/$ref";
        let reference = parameters
            .pointer(pointer)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("generic_parameters.schema.json has no {pointer}"));
        let kind_def = reference
            .strip_prefix('#')
            .and_then(|def_pointer| parameters.pointer(def_pointer))
            .unwrap_or_else(|| panic!("generic_parameters.schema.json has no {reference}"));
        let listed: Vec<&Value> = match (kind_def.get("oneOf"), kind_def.get("enum")) {
            (Some(Value::Array(variants)), _) => {
                variants.iter().map(|variant| &variant["const"]).collect()
            }
            (_, Some(Value::Array(values))) => values.iter().collect(),
            _ => panic!("{reference} has neither a oneOf nor an enum array: {kind_def}"),
        };
        let mut kinds: Vec<&str> = listed
            .into_iter()
            .map(|kind| {
                kind.as_str()
                    .unwrap_or_else(|| panic!("{reference} lists a non-string kind: {kind}"))
            })
            .collect();
        kinds.sort_unstable();
        assert_eq!(
            kinds,
            [
                "computed",
                "constant",
                "per_stage",
                "per_stage_block",
                "seasonal"
            ]
        );
    }
}
