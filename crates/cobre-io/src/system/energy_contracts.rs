//! Parsing for `system/energy_contracts.json` — energy contract entity registry.
//!
//! [`parse_energy_contracts`] reads the file from the case directory and
//! returns a fully-validated, sorted `Vec<EnergyContract>`.
//!
//! This file is **optional** — callers should use the
//! [`super::load_energy_contracts`] wrapper which accepts `Option<&Path>`
//! and returns `Ok(Vec::new())` when the file is absent.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://cobre-rs.github.io/cobre/schemas/energy_contracts.schema.json",
//!   "contracts": [
//!     {
//!       "id": 0,
//!       "name": "Importação Argentina",
//!       "bus_id": 5,
//!       "type": "import",
//!       "price_per_mwh": 200.0,
//!       "limits": { "min_mw": 0.0, "max_mw": 1000.0 }
//!     },
//!     {
//!       "id": 1,
//!       "name": "Exportação Uruguai",
//!       "bus_id": 6,
//!       "type": "export",
//!       "entry_stage_id": 1,
//!       "exit_stage_id": 60,
//!       "price_per_mwh": -150.0,
//!       "limits": { "min_mw": 0.0, "max_mw": 500.0 }
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two contracts share the same `id`.
//! 2. `limits.min_mw >= 0.0` and `limits.max_mw >= 0.0`.
//! 3. `limits.max_mw >= limits.min_mw`.
//!
//! Note: `price_per_mwh` is NOT validated as positive — it may be negative
//! for export contracts where the system receives revenue.
//!
//! Unknown values for the `type` field are caught as a serde `ParseError`
//! (classified as [`LoadError::SchemaError`] via error message inspection).
//!
//! Cross-reference validation (e.g., checking that `bus_id` exists in the bus
//! registry) is deferred to Layer 3 (Epic 06).

use cobre_core::{
    EntityId,
    entities::{ContractType, EnergyContract},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

/// Top-level intermediate type for `energy_contracts.json` (serde only, not re-exported).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawContractFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,
    /// Array of contract entries.
    contracts: Vec<RawContract>,
}

/// Intermediate type for a single energy contract entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawContract {
    /// Contract identifier. Must be unique within the file.
    id: i32,
    /// Human-readable contract name.
    name: String,
    /// Bus at which the contracted power is injected or withdrawn.
    bus_id: i32,
    /// Direction of energy flow. Uses `#[serde(rename = "type")]` since `type`
    /// is a Rust keyword.
    #[serde(rename = "type")]
    contract_type: RawContractType,
    /// Stage index when the contract enters service. Absent or null = always active.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the contract expires. Absent or null = never expires.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Contract price per `MWh`. May be negative for export revenue [$/`MWh`].
    price_per_mwh: f64,
    /// Nested limits object with min and max MW bounds.
    limits: RawContractLimits,
}

/// Raw intermediate enum for contract direction.
///
/// Uses `#[serde(rename_all = "snake_case")]` to map JSON `"import"`/`"export"`
/// to Rust `Import`/`Export` variants. The core `ContractType` enum does not
/// carry `rename_all`, so we use this intermediate.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawContractType {
    /// External energy flows into the modeled system.
    Import,
    /// System energy flows out to an external entity.
    Export,
}

/// Intermediate type for the nested limits sub-object.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawContractLimits {
    /// Minimum contracted power [MW].
    min_mw: f64,
    /// Maximum contracted power [MW].
    max_mw: f64,
}

/// Load and validate `system/energy_contracts.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to
/// `Vec<EnergyContract>`. The result is sorted by `id` ascending to satisfy
/// declaration-order invariance.
///
/// Note: `price_per_mwh` is not validated as positive — export contracts
/// legitimately carry negative prices (revenue).
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// | --------------------------------------------- | -------------------------- |
/// | File not found / read failure                 | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field | [`LoadError::ParseError`]  |
/// | Unknown `type` value in a contract            | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the contracts array     | [`LoadError::SchemaError`] |
/// | Negative `limits.min_mw` or `limits.max_mw`  | [`LoadError::SchemaError`] |
/// | `limits.max_mw < limits.min_mw`               | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_energy_contracts;
/// use std::path::Path;
///
/// let contracts = parse_energy_contracts(
///     Path::new("case/system/energy_contracts.json"),
/// ).unwrap();
/// ```
pub fn parse_energy_contracts(path: &Path) -> Result<Vec<EnergyContract>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawContractFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_contracts(&raw, path)?;

    Ok(convert_contracts(raw))
}

/// Validate all invariants on the raw deserialized contract data.
fn validate_raw_contracts(raw: &RawContractFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_contract_ids(&raw.contracts, path)?;
    for (i, contract) in raw.contracts.iter().enumerate() {
        validate_contract_limits(&contract.limits, i, path)?;
    }
    Ok(())
}

/// Check that no two contracts share the same `id`.
fn validate_no_duplicate_contract_ids(
    contracts: &[RawContract],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, contract) in contracts.iter().enumerate() {
        if !seen.insert(contract.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("contracts[{i}].id"),
                message: format!("duplicate id {} in contracts array", contract.id),
            });
        }
    }
    Ok(())
}

/// Validate limits for contract at `contract_index`.
///
/// Checks: `min_mw >= 0.0`, `max_mw >= 0.0`, `max_mw >= min_mw`.
fn validate_contract_limits(
    limits: &RawContractLimits,
    contract_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if limits.min_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("contracts[{contract_index}].limits.min_mw"),
            message: format!("limits.min_mw must be >= 0.0, got {}", limits.min_mw),
        });
    }
    if limits.max_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("contracts[{contract_index}].limits.max_mw"),
            message: format!("limits.max_mw must be >= 0.0, got {}", limits.max_mw),
        });
    }
    if limits.max_mw < limits.min_mw {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("contracts[{contract_index}].limits.max_mw"),
            message: format!(
                "limits.max_mw ({}) must be >= limits.min_mw ({})",
                limits.max_mw, limits.min_mw
            ),
        });
    }
    Ok(())
}

/// Convert validated raw contract data into `Vec<EnergyContract>`, sorted by
/// `id` ascending.
fn convert_contracts(raw: RawContractFile) -> Vec<EnergyContract> {
    let mut contracts: Vec<EnergyContract> = raw
        .contracts
        .into_iter()
        .map(|raw_contract| {
            let contract_type = match raw_contract.contract_type {
                RawContractType::Import => ContractType::Import,
                RawContractType::Export => ContractType::Export,
            };

            EnergyContract {
                id: EntityId(raw_contract.id),
                name: raw_contract.name,
                bus_id: EntityId(raw_contract.bus_id),
                contract_type,
                entry_stage_id: raw_contract.entry_stage_id,
                exit_stage_id: raw_contract.exit_stage_id,
                price_per_mwh: raw_contract.price_per_mwh,
                // Flatten nested limits object into flat fields.
                min_mw: raw_contract.limits.min_mw,
                max_mw: raw_contract.limits.max_mw,
            }
        })
        .collect();

    // Sort by id ascending to satisfy declaration-order invariance.
    contracts.sort_by_key(|c| c.id.0);
    contracts
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write a string to a temp file and return the file handle (keeps it alive).
    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // ── AC: valid contracts with import and export types ───────────────────────

    /// Given a valid `energy_contracts.json` with one import and one export
    /// contract, `parse_energy_contracts` returns `Ok(vec)` where the import has
    /// `contract_type: ContractType::Import` and the export has
    /// `contract_type: ContractType::Export`.
    #[test]
    fn test_parse_valid_contracts() {
        let json = r#"{
          "$schema": "https://cobre-rs.github.io/cobre/schemas/energy_contracts.schema.json",
          "contracts": [
            {
              "id": 0,
              "name": "Importação Argentina",
              "bus_id": 5,
              "type": "import",
              "price_per_mwh": 200.0,
              "limits": { "min_mw": 0.0, "max_mw": 1000.0 }
            },
            {
              "id": 1,
              "name": "Exportação Uruguai",
              "bus_id": 6,
              "type": "export",
              "entry_stage_id": 1,
              "exit_stage_id": 60,
              "price_per_mwh": -150.0,
              "limits": { "min_mw": 0.0, "max_mw": 500.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let contracts = parse_energy_contracts(f.path()).unwrap();

        assert_eq!(contracts.len(), 2);

        // Contract 0: import
        assert_eq!(contracts[0].id, EntityId(0));
        assert_eq!(contracts[0].name, "Importação Argentina");
        assert_eq!(contracts[0].bus_id, EntityId(5));
        assert_eq!(contracts[0].contract_type, ContractType::Import);
        assert_eq!(contracts[0].entry_stage_id, None);
        assert_eq!(contracts[0].exit_stage_id, None);
        assert!((contracts[0].price_per_mwh - 200.0).abs() < f64::EPSILON);
        assert!((contracts[0].min_mw - 0.0).abs() < f64::EPSILON);
        assert!((contracts[0].max_mw - 1000.0).abs() < f64::EPSILON);

        // Contract 1: export with negative price (revenue) and stage bounds
        assert_eq!(contracts[1].id, EntityId(1));
        assert_eq!(contracts[1].name, "Exportação Uruguai");
        assert_eq!(contracts[1].contract_type, ContractType::Export);
        assert_eq!(contracts[1].entry_stage_id, Some(1));
        assert_eq!(contracts[1].exit_stage_id, Some(60));
        assert!((contracts[1].price_per_mwh - (-150.0)).abs() < f64::EPSILON);
        assert!((contracts[1].min_mw - 0.0).abs() < f64::EPSILON);
        assert!((contracts[1].max_mw - 500.0).abs() < f64::EPSILON);
    }

    // ── AC: unknown contract type → ParseError ────────────────────────────────

    /// Given a contract with unknown `type` value, `parse_energy_contracts` returns
    /// `Err(LoadError::ParseError)` (serde cannot deserialize the unknown variant).
    #[test]
    fn test_unknown_contract_type() {
        let json = r#"{
          "contracts": [
            {
              "id": 0, "name": "Bad", "bus_id": 0,
              "type": "unknown_value",
              "price_per_mwh": 100.0,
              "limits": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_energy_contracts(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for unknown contract type, got: {err:?}"
        );
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `energy_contracts.json` with duplicate `id` values,
    /// `parse_energy_contracts` returns `Err(LoadError::SchemaError)`.
    #[test]
    fn test_duplicate_contract_id() {
        let json = r#"{
          "contracts": [
            {
              "id": 4, "name": "Alpha", "bus_id": 0,
              "type": "import",
              "price_per_mwh": 100.0,
              "limits": { "min_mw": 0.0, "max_mw": 100.0 }
            },
            {
              "id": 4, "name": "Beta", "bus_id": 1,
              "type": "export",
              "price_per_mwh": -50.0,
              "limits": { "min_mw": 0.0, "max_mw": 200.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_energy_contracts(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("contracts[1].id"),
                    "field should contain 'contracts[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: negative limits → SchemaError ─────────────────────────────────────

    /// Negative `limits.min_mw` → `SchemaError`.
    #[test]
    fn test_negative_limits_min_mw() {
        let json = r#"{
          "contracts": [
            {
              "id": 0, "name": "Bad", "bus_id": 0,
              "type": "import",
              "price_per_mwh": 100.0,
              "limits": { "min_mw": -10.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_energy_contracts(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("limits.min_mw"),
                    "field should contain 'limits.min_mw', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `limits.max_mw < limits.min_mw` → `SchemaError`.
    #[test]
    fn test_max_mw_less_than_min_mw() {
        let json = r#"{
          "contracts": [
            {
              "id": 0, "name": "Bad", "bus_id": 0,
              "type": "import",
              "price_per_mwh": 100.0,
              "limits": { "min_mw": 500.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_energy_contracts(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("limits.max_mw"),
                    "field should contain 'limits.max_mw', got: {field}"
                );
                assert!(
                    message.contains("min_mw"),
                    "message should mention min_mw, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: negative price_per_mwh is valid for export contracts ─────────────

    /// Export contracts with negative `price_per_mwh` are valid (export revenue).
    #[test]
    fn test_negative_price_per_mwh_is_valid_for_export() {
        let json = r#"{
          "contracts": [
            {
              "id": 0, "name": "Export Revenue", "bus_id": 0,
              "type": "export",
              "price_per_mwh": -500.0,
              "limits": { "min_mw": 0.0, "max_mw": 200.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_energy_contracts(f.path());
        assert!(
            result.is_ok(),
            "negative price_per_mwh should be valid for export, got: {result:?}"
        );
        let contracts = result.unwrap();
        assert!((contracts[0].price_per_mwh - (-500.0)).abs() < f64::EPSILON);
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given contracts in reverse ID order in JSON, `parse_energy_contracts`
    /// returns a `Vec` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "contracts": [
            {
              "id": 0, "name": "Import A", "bus_id": 0,
              "type": "import",
              "price_per_mwh": 100.0,
              "limits": { "min_mw": 0.0, "max_mw": 100.0 }
            },
            {
              "id": 1, "name": "Export B", "bus_id": 1,
              "type": "export",
              "price_per_mwh": -50.0,
              "limits": { "min_mw": 0.0, "max_mw": 200.0 }
            }
          ]
        }"#;
        let json_reversed = r#"{
          "contracts": [
            {
              "id": 1, "name": "Export B", "bus_id": 1,
              "type": "export",
              "price_per_mwh": -50.0,
              "limits": { "min_mw": 0.0, "max_mw": 200.0 }
            },
            {
              "id": 0, "name": "Import A", "bus_id": 0,
              "type": "import",
              "price_per_mwh": 100.0,
              "limits": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;

        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let contracts1 = parse_energy_contracts(f1.path()).unwrap();
        let contracts2 = parse_energy_contracts(f2.path()).unwrap();

        assert_eq!(
            contracts1, contracts2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(contracts1[0].id, EntityId(0));
        assert_eq!(contracts1[1].id, EntityId(1));
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_energy_contracts` returns
    /// `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/energy_contracts.json");
        let err = parse_energy_contracts(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_energy_contracts` returns
    /// `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"contracts": [not valid json}}"#);
        let err = parse_energy_contracts(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `contracts` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_contracts_array() {
        let json = r#"{ "contracts": [] }"#;
        let f = write_json(json);
        let contracts = parse_energy_contracts(f.path()).unwrap();
        assert!(contracts.is_empty());
    }

    /// `min_mw == max_mw` (degenerate range) is valid.
    #[test]
    fn test_min_equals_max_mw_is_valid() {
        let json = r#"{
          "contracts": [
            {
              "id": 0, "name": "Fixed", "bus_id": 0,
              "type": "import",
              "price_per_mwh": 100.0,
              "limits": { "min_mw": 100.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_energy_contracts(f.path());
        assert!(
            result.is_ok(),
            "min_mw == max_mw should be valid, got: {result:?}"
        );
    }

    /// Zero price is valid for import contracts.
    #[test]
    fn test_zero_price_is_valid() {
        let json = r#"{
          "contracts": [
            {
              "id": 0, "name": "Free Import", "bus_id": 0,
              "type": "import",
              "price_per_mwh": 0.0,
              "limits": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_energy_contracts(f.path());
        assert!(
            result.is_ok(),
            "zero price should be valid, got: {result:?}"
        );
    }
}
