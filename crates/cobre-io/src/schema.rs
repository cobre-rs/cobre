//! JSON Schema generation for all user-facing input types.
//!
//! This module is only compiled when the `schema` feature is enabled.
//! It provides [`generate_schemas`], which returns JSON Schema documents for
//! every case directory input file that users author by hand.
//!
//! # Usage
//!
//! ```rust
//! # #[cfg(feature = "schema")]
//! # {
//! use cobre_io::schema::generate_schemas;
//!
//! let schemas = generate_schemas().expect("schema generation must not fail");
//! assert!(!schemas.is_empty());
//! for (filename, value) in &schemas {
//!     println!("{filename}: {} top-level keys", value.as_object().map_or(0, |o| o.len()));
//! }
//! # }
//! ```

use crate::{
    config::Config,
    penalties::RawPenalties,
    stages::RawStagesFile,
    system::{
        buses::RawBusFile, energy_contracts::RawContractFile, hydros::RawHydroFile,
        lines::RawLineFile, non_controllable::RawNcsFile, pumping_stations::RawPumpingFile,
        thermals::RawThermalFile,
    },
};

/// Generate JSON Schema documents for all user-facing case directory input files.
///
/// Returns a list of `(filename, schema_value)` pairs, where `filename` is the
/// conventional name of the generated schema file (e.g. `"config.schema.json"`)
/// and `schema_value` is the JSON Schema as a [`serde_json::Value`].
///
/// The returned `Vec` contains one entry per top-level input file:
///
/// | Filename                              | Describes                         |
/// | ------------------------------------- | ---------------------------------- |
/// | `config.schema.json`                  | `config.json`                      |
/// | `buses.schema.json`                   | `system/buses.json`                |
/// | `hydros.schema.json`                  | `system/hydros.json`               |
/// | `thermals.schema.json`                | `system/thermals.json`             |
/// | `lines.schema.json`                   | `system/lines.json`                |
/// | `energy_contracts.schema.json`        | `system/energy_contracts.json`     |
/// | `non_controllable_sources.schema.json`| `system/non_controllable_sources.json` |
/// | `pumping_stations.schema.json`        | `system/pumping_stations.json`     |
/// | `stages.schema.json`                  | `stages.json`                      |
/// | `penalties.schema.json`               | `penalties.json`                   |
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
/// assert!(schemas.len() >= 10);
/// let config_schema = schemas.iter().find(|(name, _)| name == "config.schema.json");
/// assert!(config_schema.is_some());
/// ```
pub fn generate_schemas() -> Result<Vec<(String, serde_json::Value)>, serde_json::Error> {
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
    ];

    pairs
        .into_iter()
        .map(|(name, schema)| {
            let value = serde_json::to_value(schema)?;
            Ok((name.to_string(), value))
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    /// `generate_schemas()` returns at least 10 entries (one per input file).
    #[test]
    fn test_generate_schemas_returns_expected_count() {
        let schemas = generate_schemas().unwrap();
        assert!(
            schemas.len() >= 10,
            "expected at least 10 schema entries, got {}",
            schemas.len()
        );
    }

    /// Every returned schema has a non-empty filename and a non-null JSON value.
    #[test]
    fn test_all_schema_filenames_and_values_non_empty() {
        let schemas = generate_schemas().unwrap();
        for (name, value) in &schemas {
            assert!(!name.is_empty(), "schema filename must not be empty");
            assert!(!value.is_null(), "schema value must not be null for {name}");
        }
    }

    /// Every schema is a JSON object (not an array or scalar).
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

    /// Every schema contains either a `"properties"` key (for object schemas)
    /// or a `"oneOf"` / `"anyOf"` key (for enum schemas).
    #[test]
    fn test_all_schemas_have_structure_keys() {
        let schemas = generate_schemas().unwrap();
        for (name, value) in &schemas {
            let obj = value.as_object().unwrap_or_else(|| {
                panic!("schema for {name} is not an object");
            });
            let has_properties = obj.contains_key("properties");
            let has_one_of = obj.contains_key("oneOf");
            let has_any_of = obj.contains_key("anyOf");
            let has_defs = obj.contains_key("$defs");
            // schemars v1 may hoist definitions and reference them; at minimum
            // a non-trivial schema always has one of these structural keys.
            assert!(
                has_properties || has_one_of || has_any_of || has_defs,
                "schema for {name} has no expected structural keys (properties/oneOf/anyOf/$defs)"
            );
        }
    }

    /// The `config.schema.json` entry contains expected top-level field names.
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

    /// The `buses.schema.json` entry contains a `buses` array property.
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

    /// All 10 expected schema filenames are present in the output.
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
        ];

        for name in &expected {
            assert!(
                names.contains(name),
                "expected schema '{name}' not found; got: {names:?}"
            );
        }
    }
}
