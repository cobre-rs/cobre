//! Parsing for `system/pumping_stations.json` — pumping station entity registry.
//!
//! [`parse_pumping_stations`] reads the file from the case directory and
//! returns a fully-validated, sorted `Vec<PumpingStation>`.
//!
//! This file is **optional** — callers should use the
//! [`super::load_pumping_stations`] wrapper which accepts `Option<&Path>`
//! and returns `Ok(Vec::new())` when the file is absent.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://cobre-rs.github.io/cobre/schemas/pumping_stations.schema.json",
//!   "pumping_stations": [
//!     {
//!       "id": 0,
//!       "name": "Bombeamento Serra da Mesa",
//!       "bus_id": 10,
//!       "source_hydro_id": 3,
//!       "destination_hydro_id": 5,
//!       "consumption_mw_per_m3s": 0.5,
//!       "flow": { "min_m3s": 0.0, "max_m3s": 150.0 }
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two stations share the same `id`.
//! 2. `consumption_mw_per_m3s >= 0.0`.
//! 3. `flow.min_m3s >= 0.0` and `flow.max_m3s >= 0.0`.
//! 4. `flow.max_m3s >= flow.min_m3s`.
//!
//! Cross-reference validation (e.g., checking that `bus_id`, `source_hydro_id`,
//! `destination_hydro_id` exist in their respective registries) is deferred to
//! Layer 3 (Epic 06).

use cobre_core::{EntityId, entities::PumpingStation};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

/// Top-level intermediate type for `pumping_stations.json` (serde only, not re-exported).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawPumpingFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,
    /// Array of pumping station entries.
    pumping_stations: Vec<RawPumpingStation>,
}

/// Intermediate type for a single pumping station entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawPumpingStation {
    /// Station identifier. Must be unique within the file.
    id: i32,
    /// Human-readable station name.
    name: String,
    /// Bus from which electrical power is consumed.
    bus_id: i32,
    /// Hydro plant from whose reservoir water is extracted.
    source_hydro_id: i32,
    /// Hydro plant into whose reservoir water is injected.
    destination_hydro_id: i32,
    /// Stage index when the station enters service. Absent or null = always exists.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the station is decommissioned. Absent or null = never.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Power consumption rate per unit of pumped flow [MW/(m³/s)].
    consumption_mw_per_m3s: f64,
    /// Nested flow bounds object.
    flow: RawPumpingFlow,
}

/// Intermediate type for the nested flow bounds sub-object.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawPumpingFlow {
    /// Minimum pumped flow [m³/s].
    min_m3s: f64,
    /// Maximum pumped flow [m³/s].
    max_m3s: f64,
}

/// Load and validate `system/pumping_stations.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to
/// `Vec<PumpingStation>`. The result is sorted by `id` ascending to satisfy
/// declaration-order invariance.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// | --------------------------------------------- | -------------------------- |
/// | File not found / read failure                 | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the stations array      | [`LoadError::SchemaError`] |
/// | Negative `consumption_mw_per_m3s`             | [`LoadError::SchemaError`] |
/// | Negative `flow.min_m3s` or `flow.max_m3s`    | [`LoadError::SchemaError`] |
/// | `flow.max_m3s < flow.min_m3s`                 | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_pumping_stations;
/// use std::path::Path;
///
/// let stations = parse_pumping_stations(
///     Path::new("case/system/pumping_stations.json"),
/// ).unwrap();
/// ```
pub fn parse_pumping_stations(path: &Path) -> Result<Vec<PumpingStation>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawPumpingFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_pumping(&raw, path)?;

    Ok(convert_pumping(raw))
}

/// Validate all invariants on the raw deserialized pumping station data.
fn validate_raw_pumping(raw: &RawPumpingFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_pumping_ids(&raw.pumping_stations, path)?;
    for (i, station) in raw.pumping_stations.iter().enumerate() {
        validate_consumption(station.consumption_mw_per_m3s, i, path)?;
        validate_flow_bounds(&station.flow, i, path)?;
    }
    Ok(())
}

/// Check that no two stations share the same `id`.
fn validate_no_duplicate_pumping_ids(
    stations: &[RawPumpingStation],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, station) in stations.iter().enumerate() {
        if !seen.insert(station.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("pumping_stations[{i}].id"),
                message: format!("duplicate id {} in pumping_stations array", station.id),
            });
        }
    }
    Ok(())
}

/// Validate `consumption_mw_per_m3s` for station at `station_index`.
///
/// Checks: `consumption_mw_per_m3s >= 0.0`.
fn validate_consumption(
    consumption_mw_per_m3s: f64,
    station_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if consumption_mw_per_m3s < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("pumping_stations[{station_index}].consumption_mw_per_m3s"),
            message: format!("consumption_mw_per_m3s must be >= 0.0, got {consumption_mw_per_m3s}"),
        });
    }
    Ok(())
}

/// Validate flow bounds for station at `station_index`.
///
/// Checks: `min_m3s >= 0.0`, `max_m3s >= 0.0`, `max_m3s >= min_m3s`.
fn validate_flow_bounds(
    flow: &RawPumpingFlow,
    station_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if flow.min_m3s < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("pumping_stations[{station_index}].flow.min_m3s"),
            message: format!("flow.min_m3s must be >= 0.0, got {}", flow.min_m3s),
        });
    }
    if flow.max_m3s < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("pumping_stations[{station_index}].flow.max_m3s"),
            message: format!("flow.max_m3s must be >= 0.0, got {}", flow.max_m3s),
        });
    }
    if flow.max_m3s < flow.min_m3s {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("pumping_stations[{station_index}].flow.max_m3s"),
            message: format!(
                "flow.max_m3s ({}) must be >= flow.min_m3s ({})",
                flow.max_m3s, flow.min_m3s
            ),
        });
    }
    Ok(())
}

/// Convert validated raw pumping station data into `Vec<PumpingStation>`, sorted
/// by `id` ascending.
fn convert_pumping(raw: RawPumpingFile) -> Vec<PumpingStation> {
    let mut stations: Vec<PumpingStation> = raw
        .pumping_stations
        .into_iter()
        .map(|raw_station| PumpingStation {
            id: EntityId(raw_station.id),
            name: raw_station.name,
            bus_id: EntityId(raw_station.bus_id),
            source_hydro_id: EntityId(raw_station.source_hydro_id),
            destination_hydro_id: EntityId(raw_station.destination_hydro_id),
            entry_stage_id: raw_station.entry_stage_id,
            exit_stage_id: raw_station.exit_stage_id,
            consumption_mw_per_m3s: raw_station.consumption_mw_per_m3s,
            // Flatten nested flow object into flat fields.
            min_flow_m3s: raw_station.flow.min_m3s,
            max_flow_m3s: raw_station.flow.max_m3s,
        })
        .collect();

    // Sort by id ascending to satisfy declaration-order invariance.
    stations.sort_by_key(|s| s.id.0);
    stations
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

    // ── AC: valid pumping stations ─────────────────────────────────────────────

    /// Given a valid `pumping_stations.json` with 2 stations,
    /// `parse_pumping_stations` returns `Ok(vec)` with 2 entries sorted by `id`.
    #[test]
    fn test_parse_valid_pumping_stations() {
        let json = r#"{
          "$schema": "https://cobre-rs.github.io/cobre/schemas/pumping_stations.schema.json",
          "pumping_stations": [
            {
              "id": 0,
              "name": "Bombeamento Serra da Mesa",
              "bus_id": 10,
              "source_hydro_id": 3,
              "destination_hydro_id": 5,
              "consumption_mw_per_m3s": 0.5,
              "flow": { "min_m3s": 0.0, "max_m3s": 150.0 }
            },
            {
              "id": 1,
              "name": "Bombeamento Cana Brava",
              "bus_id": 11,
              "source_hydro_id": 4,
              "destination_hydro_id": 6,
              "entry_stage_id": 6,
              "exit_stage_id": 120,
              "consumption_mw_per_m3s": 0.8,
              "flow": { "min_m3s": 10.0, "max_m3s": 200.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let stations = parse_pumping_stations(f.path()).unwrap();

        assert_eq!(stations.len(), 2);

        // Station 0
        assert_eq!(stations[0].id, EntityId(0));
        assert_eq!(stations[0].name, "Bombeamento Serra da Mesa");
        assert_eq!(stations[0].bus_id, EntityId(10));
        assert_eq!(stations[0].source_hydro_id, EntityId(3));
        assert_eq!(stations[0].destination_hydro_id, EntityId(5));
        assert_eq!(stations[0].entry_stage_id, None);
        assert_eq!(stations[0].exit_stage_id, None);
        assert!((stations[0].consumption_mw_per_m3s - 0.5).abs() < f64::EPSILON);
        assert!((stations[0].min_flow_m3s - 0.0).abs() < f64::EPSILON);
        assert!((stations[0].max_flow_m3s - 150.0).abs() < f64::EPSILON);

        // Station 1: has stage bounds
        assert_eq!(stations[1].id, EntityId(1));
        assert_eq!(stations[1].entry_stage_id, Some(6));
        assert_eq!(stations[1].exit_stage_id, Some(120));
        assert!((stations[1].consumption_mw_per_m3s - 0.8).abs() < f64::EPSILON);
        assert!((stations[1].min_flow_m3s - 10.0).abs() < f64::EPSILON);
        assert!((stations[1].max_flow_m3s - 200.0).abs() < f64::EPSILON);
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `pumping_stations.json` with duplicate `id` values,
    /// `parse_pumping_stations` returns `Err(LoadError::SchemaError)`.
    #[test]
    fn test_duplicate_pumping_station_id() {
        let json = r#"{
          "pumping_stations": [
            {
              "id": 2, "name": "Alpha", "bus_id": 0,
              "source_hydro_id": 1, "destination_hydro_id": 2,
              "consumption_mw_per_m3s": 0.5,
              "flow": { "min_m3s": 0.0, "max_m3s": 100.0 }
            },
            {
              "id": 2, "name": "Beta", "bus_id": 1,
              "source_hydro_id": 2, "destination_hydro_id": 3,
              "consumption_mw_per_m3s": 0.6,
              "flow": { "min_m3s": 0.0, "max_m3s": 200.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_pumping_stations(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("pumping_stations[1].id"),
                    "field should contain 'pumping_stations[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: negative values → SchemaError ─────────────────────────────────────

    /// Negative `consumption_mw_per_m3s` → `SchemaError`.
    #[test]
    fn test_negative_consumption() {
        let json = r#"{
          "pumping_stations": [
            {
              "id": 0, "name": "Bad", "bus_id": 0,
              "source_hydro_id": 1, "destination_hydro_id": 2,
              "consumption_mw_per_m3s": -0.5,
              "flow": { "min_m3s": 0.0, "max_m3s": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_pumping_stations(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("consumption_mw_per_m3s"),
                    "field should contain 'consumption_mw_per_m3s', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `flow.min_m3s` → `SchemaError`.
    #[test]
    fn test_negative_flow_min() {
        let json = r#"{
          "pumping_stations": [
            {
              "id": 0, "name": "Bad", "bus_id": 0,
              "source_hydro_id": 1, "destination_hydro_id": 2,
              "consumption_mw_per_m3s": 0.5,
              "flow": { "min_m3s": -10.0, "max_m3s": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_pumping_stations(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("flow.min_m3s"),
                    "field should contain 'flow.min_m3s', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `flow.max_m3s < flow.min_m3s` → `SchemaError`.
    #[test]
    fn test_max_flow_less_than_min_flow() {
        let json = r#"{
          "pumping_stations": [
            {
              "id": 0, "name": "Bad", "bus_id": 0,
              "source_hydro_id": 1, "destination_hydro_id": 2,
              "consumption_mw_per_m3s": 0.5,
              "flow": { "min_m3s": 200.0, "max_m3s": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_pumping_stations(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("flow.max_m3s"),
                    "field should contain 'flow.max_m3s', got: {field}"
                );
                assert!(
                    message.contains("min_m3s"),
                    "message should mention min_m3s, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given stations in reverse ID order in JSON, `parse_pumping_stations`
    /// returns a `Vec` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "pumping_stations": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "source_hydro_id": 1, "destination_hydro_id": 2,
              "consumption_mw_per_m3s": 0.5,
              "flow": { "min_m3s": 0.0, "max_m3s": 100.0 }
            },
            {
              "id": 1, "name": "Beta", "bus_id": 1,
              "source_hydro_id": 2, "destination_hydro_id": 3,
              "consumption_mw_per_m3s": 0.8,
              "flow": { "min_m3s": 0.0, "max_m3s": 200.0 }
            }
          ]
        }"#;
        let json_reversed = r#"{
          "pumping_stations": [
            {
              "id": 1, "name": "Beta", "bus_id": 1,
              "source_hydro_id": 2, "destination_hydro_id": 3,
              "consumption_mw_per_m3s": 0.8,
              "flow": { "min_m3s": 0.0, "max_m3s": 200.0 }
            },
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "source_hydro_id": 1, "destination_hydro_id": 2,
              "consumption_mw_per_m3s": 0.5,
              "flow": { "min_m3s": 0.0, "max_m3s": 100.0 }
            }
          ]
        }"#;

        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let stations1 = parse_pumping_stations(f1.path()).unwrap();
        let stations2 = parse_pumping_stations(f2.path()).unwrap();

        assert_eq!(
            stations1, stations2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(stations1[0].id, EntityId(0));
        assert_eq!(stations1[1].id, EntityId(1));
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_pumping_stations` returns
    /// `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/pumping_stations.json");
        let err = parse_pumping_stations(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_pumping_stations` returns
    /// `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"pumping_stations": [not valid json}}"#);
        let err = parse_pumping_stations(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `pumping_stations` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_pumping_stations_array() {
        let json = r#"{ "pumping_stations": [] }"#;
        let f = write_json(json);
        let stations = parse_pumping_stations(f.path()).unwrap();
        assert!(stations.is_empty());
    }

    /// `min_m3s == max_m3s` (degenerate range) is valid.
    #[test]
    fn test_min_equals_max_flow_is_valid() {
        let json = r#"{
          "pumping_stations": [
            {
              "id": 0, "name": "Alpha", "bus_id": 0,
              "source_hydro_id": 1, "destination_hydro_id": 2,
              "consumption_mw_per_m3s": 0.5,
              "flow": { "min_m3s": 100.0, "max_m3s": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_pumping_stations(f.path());
        assert!(
            result.is_ok(),
            "min_m3s == max_m3s should be valid, got: {result:?}"
        );
    }
}
