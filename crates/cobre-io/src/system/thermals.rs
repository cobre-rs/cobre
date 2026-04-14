//! Parsing for `system/thermals.json` — thermal plant entity registry.
//!
//! [`parse_thermals`] reads `system/thermals.json` from the case directory and
//! returns a fully-validated, sorted `Vec<Thermal>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/thermals.schema.json",
//!   "thermals": [
//!     {
//!       "id": 0,
//!       "name": "Angra 1",
//!       "bus_id": 2,
//!       "cost_per_mwh": 12.0,
//!       "generation": { "min_mw": 0.0, "max_mw": 600.0 }
//!     },
//!     {
//!       "id": 1,
//!       "name": "Pecém I",
//!       "bus_id": 3,
//!       "entry_stage_id": 1,
//!       "exit_stage_id": 120,
//!       "cost_per_mwh": 120.0,
//!       "generation": { "min_mw": 100.0, "max_mw": 360.0 },
//!       "gnl_config": { "lag_stages": 2 }
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two thermals share the same `id`.
//! 2. `cost_per_mwh` must be ≥ 0.0.
//! 3. `min_generation_mw` and `max_generation_mw` must be ≥ 0.0.
//! 4. `max_generation_mw >= min_generation_mw`.
//!
//! GNL config is parsed but NOT rejected at this layer — GNL rejection is a
//! semantic validation concern for Epic 06. Cross-reference validation (checking
//! that `bus_id` exists) is also deferred to Layer 3 (Epic 06).

use cobre_core::{
    EntityId,
    entities::{GnlConfig, Thermal},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

/// Top-level intermediate type for `thermals.json` (serde only, not re-exported).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawThermalFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,
    /// Array of thermal plant entries.
    thermals: Vec<RawThermal>,
}

/// Intermediate type for a single thermal plant entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawThermal {
    /// Thermal plant identifier. Must be unique within the file.
    id: i32,
    /// Human-readable plant name.
    name: String,
    /// Bus to which this plant's generation is injected.
    bus_id: i32,
    /// Stage index when the plant enters service. Absent or null = always exists.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. Absent or null = never.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Marginal cost of generation [$/`MWh`]. Must be ≥ 0.0.
    cost_per_mwh: f64,
    /// Generation bounds.
    generation: RawThermalGeneration,
    /// GNL dispatch anticipation configuration. Absent = no lag.
    #[serde(default)]
    gnl_config: Option<RawGnlConfig>,
}

/// Intermediate type for the generation bounds sub-object.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawThermalGeneration {
    /// Minimum generation (minimum stable load) [MW].
    min_mw: f64,
    /// Maximum generation (installed capacity) [MW].
    max_mw: f64,
}

/// Intermediate type for GNL configuration.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawGnlConfig {
    /// Number of stages of dispatch anticipation.
    lag_stages: i32,
}

/// Load and validate `system/thermals.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to `Vec<Thermal>`.
/// The result is sorted by `id` ascending to satisfy declaration-order invariance.
///
/// GNL config is parsed but NOT rejected at this layer. Cross-reference
/// validation (e.g., `bus_id` existence in the bus registry) is deferred to
/// Layer 3 (Epic 06).
///
/// # Errors
///
/// | Condition                                           | Error variant              |
/// | --------------------------------------------------- | -------------------------- |
/// | File not found / read failure                       | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field       | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the thermals array            | [`LoadError::SchemaError`] |
/// | Negative `cost_per_mwh`                             | [`LoadError::SchemaError`] |
/// | Negative `min_generation_mw` or `max_generation_mw` | [`LoadError::SchemaError`] |
/// | `max_generation_mw < min_generation_mw`             | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_thermals;
/// use std::path::Path;
///
/// let thermals = parse_thermals(Path::new("case/system/thermals.json")).unwrap();
/// assert!(!thermals.is_empty());
/// ```
pub fn parse_thermals(path: &Path) -> Result<Vec<Thermal>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawThermalFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_thermals(&raw, path)?;

    Ok(convert_thermals(raw))
}

/// Validate all invariants on the raw deserialized thermal data.
fn validate_raw_thermals(raw: &RawThermalFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_thermal_ids(&raw.thermals, path)?;
    for (i, thermal) in raw.thermals.iter().enumerate() {
        validate_cost_per_mwh(thermal.cost_per_mwh, i, path)?;
        validate_generation_bounds(&thermal.generation, i, path)?;
    }
    Ok(())
}

/// Check that no two thermals share the same `id`.
fn validate_no_duplicate_thermal_ids(
    thermals: &[RawThermal],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, thermal) in thermals.iter().enumerate() {
        if !seen.insert(thermal.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermals[{i}].id"),
                message: format!("duplicate id {} in thermals array", thermal.id),
            });
        }
    }
    Ok(())
}

/// Validate `cost_per_mwh` for thermal at `thermal_index`.
///
/// Checks: `cost_per_mwh >= 0.0`.
fn validate_cost_per_mwh(
    cost_per_mwh: f64,
    thermal_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if cost_per_mwh < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].cost_per_mwh"),
            message: format!("cost_per_mwh must be >= 0.0, got {cost_per_mwh}"),
        });
    }
    Ok(())
}

/// Validate generation bounds for thermal at `thermal_index`.
///
/// Checks: `min_mw >= 0.0`, `max_mw >= 0.0`, `max_mw >= min_mw`.
fn validate_generation_bounds(
    bounds: &RawThermalGeneration,
    thermal_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if bounds.min_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].generation.min_mw"),
            message: format!("min_mw must be >= 0.0, got {}", bounds.min_mw),
        });
    }
    if bounds.max_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].generation.max_mw"),
            message: format!("max_mw must be >= 0.0, got {}", bounds.max_mw),
        });
    }
    if bounds.max_mw < bounds.min_mw {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].generation.max_mw"),
            message: format!(
                "max_mw ({}) must be >= min_mw ({})",
                bounds.max_mw, bounds.min_mw
            ),
        });
    }
    Ok(())
}

/// Convert validated raw thermal data into `Vec<Thermal>`, sorted by `id` ascending.
fn convert_thermals(raw: RawThermalFile) -> Vec<Thermal> {
    let mut thermals: Vec<Thermal> = raw
        .thermals
        .into_iter()
        .map(|raw_thermal| {
            let gnl_config: Option<GnlConfig> = raw_thermal.gnl_config.map(|g| GnlConfig {
                lag_stages: g.lag_stages,
            });

            Thermal {
                id: EntityId(raw_thermal.id),
                name: raw_thermal.name,
                bus_id: EntityId(raw_thermal.bus_id),
                entry_stage_id: raw_thermal.entry_stage_id,
                exit_stage_id: raw_thermal.exit_stage_id,
                cost_per_mwh: raw_thermal.cost_per_mwh,
                min_generation_mw: raw_thermal.generation.min_mw,
                max_generation_mw: raw_thermal.generation.max_mw,
                gnl_config,
            }
        })
        .collect();

    // Sort by id ascending to satisfy declaration-order invariance.
    thermals.sort_by_key(|t| t.id.0);
    thermals
}

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

    /// Canonical valid `thermals.json` with 2 thermals: one with GNL (id=1),
    /// one without (id=0).
    const VALID_JSON: &str = r#"{
      "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/thermals.schema.json",
      "thermals": [
        {
          "id": 0,
          "name": "Angra 1",
          "bus_id": 2,
          "cost_per_mwh": 12.0,
          "generation": { "min_mw": 0.0, "max_mw": 600.0 }
        },
        {
          "id": 1,
          "name": "Pecém I",
          "bus_id": 3,
          "entry_stage_id": 1,
          "exit_stage_id": 120,
          "cost_per_mwh": 120.0,
          "generation": { "min_mw": 100.0, "max_mw": 360.0 },
          "gnl_config": { "lag_stages": 2 }
        }
      ]
    }"#;

    // ── AC: valid thermals with and without GNL ────────────────────────────────

    /// Given a valid `thermals.json` with 2 thermals (one with GNL config, one
    /// without), `parse_thermals` returns `Ok(vec)` with 2 `Thermal` entries
    /// sorted by `id`; the GNL thermal has `gnl_config: Some(GnlConfig { lag_stages: 2 })`.
    #[test]
    fn test_parse_valid_thermals() {
        let f = write_json(VALID_JSON);
        let thermals = parse_thermals(f.path()).unwrap();

        assert_eq!(thermals.len(), 2);

        // Thermal 0: no GNL, no stage bounds
        assert_eq!(thermals[0].id, EntityId(0));
        assert_eq!(thermals[0].name, "Angra 1");
        assert_eq!(thermals[0].bus_id, EntityId(2));
        assert_eq!(thermals[0].entry_stage_id, None);
        assert_eq!(thermals[0].exit_stage_id, None);
        assert!(
            (thermals[0].cost_per_mwh - 12.0).abs() < f64::EPSILON,
            "cost_per_mwh: expected 12.0, got {}",
            thermals[0].cost_per_mwh
        );
        assert!((thermals[0].min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((thermals[0].max_generation_mw - 600.0).abs() < f64::EPSILON);
        assert_eq!(thermals[0].gnl_config, None);

        // Thermal 1: has GNL config and stage bounds
        assert_eq!(thermals[1].id, EntityId(1));
        assert_eq!(thermals[1].name, "Pecém I");
        assert_eq!(thermals[1].bus_id, EntityId(3));
        assert_eq!(thermals[1].entry_stage_id, Some(1));
        assert_eq!(thermals[1].exit_stage_id, Some(120));
        assert!(
            (thermals[1].cost_per_mwh - 120.0).abs() < f64::EPSILON,
            "cost_per_mwh: expected 120.0, got {}",
            thermals[1].cost_per_mwh
        );
        assert!((thermals[1].min_generation_mw - 100.0).abs() < f64::EPSILON);
        assert!((thermals[1].max_generation_mw - 360.0).abs() < f64::EPSILON);
        assert_eq!(thermals[1].gnl_config, Some(GnlConfig { lag_stages: 2 }));
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `thermals.json` with duplicate `id` values, `parse_thermals` returns
    /// `Err(LoadError::SchemaError)` mentioning "duplicate".
    #[test]
    fn test_duplicate_thermal_id() {
        let json = r#"{
          "thermals": [
            {
              "id": 5, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            },
            {
              "id": 5, "name": "Beta", "bus_id": 0,
              "cost_per_mwh": 80.0,
              "generation": { "min_mw": 0.0, "max_mw": 200.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("thermals[1].id"),
                    "field should contain 'thermals[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: cost_per_mwh 75.0 round-trips through parser ───────────────────

    /// Given a `thermals.json` with `"cost_per_mwh": 75.0`, the parser
    /// produces `Thermal { cost_per_mwh: 75.0, .. }`.
    #[test]
    fn test_parse_cost_per_mwh_75() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 75.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert_eq!(thermals.len(), 1);
        assert!(
            (thermals[0].cost_per_mwh - 75.0).abs() < f64::EPSILON,
            "cost_per_mwh: expected 75.0, got {}",
            thermals[0].cost_per_mwh
        );
    }

    // ── AC: old array-based cost format fails to parse ──────────────────────

    /// Given `thermals.json` using the old array-based cost format (removed in
    /// this version), `parse_thermals` returns `Err(LoadError::ParseError)` —
    /// the old format is a breaking change with no backward compat.
    #[test]
    fn test_legacy_array_cost_format_fails() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for legacy array-based cost format, got: {err:?}"
        );
    }

    // ── AC: negative cost_per_mwh → SchemaError ──────────────────────────────

    /// Negative `cost_per_mwh` → `SchemaError`.
    #[test]
    fn test_negative_cost_per_mwh() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": -50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("cost_per_mwh"),
                    "field should contain 'cost_per_mwh', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: generation bound validation ───────────────────────────────────────

    /// Negative `min_mw` → `SchemaError`.
    #[test]
    fn test_negative_min_generation_mw() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": -10.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("min_mw"),
                    "field should contain 'min_mw', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `max_mw` → `SchemaError`.
    #[test]
    fn test_negative_max_generation_mw() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": -100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_mw"),
                    "field should contain 'max_mw', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `max_mw < min_mw` → `SchemaError`.
    #[test]
    fn test_max_mw_less_than_min_mw() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 200.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_mw"),
                    "field should contain 'max_mw', got: {field}"
                );
                assert!(
                    message.contains("min_mw"),
                    "message should mention min_mw, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given thermals in reverse ID order in JSON, `parse_thermals` returns a
    /// `Vec<Thermal>` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "thermals": [
            {
              "id": 0, "name": "Angra 1", "bus_id": 0,
              "cost_per_mwh": 12.0,
              "generation": { "min_mw": 0.0, "max_mw": 600.0 }
            },
            {
              "id": 1, "name": "Pecém I", "bus_id": 1,
              "cost_per_mwh": 120.0,
              "generation": { "min_mw": 0.0, "max_mw": 360.0 }
            }
          ]
        }"#;
        let json_reversed = r#"{
          "thermals": [
            {
              "id": 1, "name": "Pecém I", "bus_id": 1,
              "cost_per_mwh": 120.0,
              "generation": { "min_mw": 0.0, "max_mw": 360.0 }
            },
            {
              "id": 0, "name": "Angra 1", "bus_id": 0,
              "cost_per_mwh": 12.0,
              "generation": { "min_mw": 0.0, "max_mw": 600.0 }
            }
          ]
        }"#;

        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let thermals1 = parse_thermals(f1.path()).unwrap();
        let thermals2 = parse_thermals(f2.path()).unwrap();

        assert_eq!(
            thermals1, thermals2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(thermals1[0].id, EntityId(0));
        assert_eq!(thermals1[1].id, EntityId(1));
    }

    // ── AC: GNL thermals accepted without rejection ───────────────────────────

    /// GNL thermals are parsed and accepted at this layer (rejection is semantic
    /// validation in Epic 06).
    #[test]
    fn test_gnl_thermal_not_rejected() {
        let f = write_json(VALID_JSON);
        let thermals = parse_thermals(f.path()).unwrap();
        // Thermal 1 has GNL config: lag_stages = 2
        let gnl_thermal = thermals.iter().find(|t| t.id == EntityId(1)).unwrap();
        assert_eq!(gnl_thermal.gnl_config, Some(GnlConfig { lag_stages: 2 }));
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_thermals` returns `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/thermals.json");
        let err = parse_thermals(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_thermals` returns `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"thermals": [not valid json}}"#);
        let err = parse_thermals(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `thermals` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_thermals_array() {
        let json = r#"{ "thermals": [] }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert!(thermals.is_empty());
    }

    /// `entry_stage_id` and `exit_stage_id` default to `None` when absent.
    #[test]
    fn test_optional_stage_ids_default_to_none() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert_eq!(thermals[0].entry_stage_id, None);
        assert_eq!(thermals[0].exit_stage_id, None);
    }

    /// `min_mw == max_mw` (degenerate range) is valid.
    #[test]
    fn test_min_equals_max_is_valid() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 100.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_thermals(f.path());
        assert!(
            result.is_ok(),
            "min_mw == max_mw should be valid, got: {result:?}"
        );
    }

    /// Zero-cost thermal (`cost_per_mwh` = 0.0) is valid (non-negative check only).
    #[test]
    fn test_zero_cost_is_valid() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "cost_per_mwh": 0.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_thermals(f.path());
        assert!(
            result.is_ok(),
            "cost_per_mwh = 0.0 should be valid, got: {result:?}"
        );
    }
}
