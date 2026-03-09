//! Parsing for `system/buses.json` — bus entity registry.
//!
//! [`parse_buses`] reads `system/buses.json` from the case directory and
//! returns a fully-validated, sorted `Vec<Bus>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://cobre.dev/schemas/v2/buses.schema.json",
//!   "buses": [
//!     { "id": 0, "name": "South" },
//!     {
//!       "id": 1,
//!       "name": "North",
//!       "deficit_segments": [
//!         { "depth_mw": 500.0, "cost": 1000.0 },
//!         { "depth_mw": null, "cost": 5000.0 }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two buses share the same `id`.
//! 2. If entity-level `deficit_segments` are present:
//!    - all costs must be strictly positive (> 0.0).
//!    - the last segment must have `depth_mw: null` (unbounded).
//!    - costs must be monotonically increasing across segments.
//!
//! Penalty resolution uses `resolve_bus_deficit_segments` and
//! `resolve_bus_excess_cost` for the global → entity penalty cascade.
//! The bus-level `excess_cost` has no entity-level override in the JSON schema
//! (SS1 spec): it always comes from the global default.
//!
//! Cross-reference validation (e.g., checking that bus IDs are referenced by
//! thermals, lines, hydros) is deferred to Layer 3 (Epic 06).

use cobre_core::{
    entities::{Bus, DeficitSegment},
    penalty::{resolve_bus_deficit_segments, resolve_bus_excess_cost, GlobalPenaltyDefaults},
    EntityId,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

/// Top-level intermediate type for `buses.json` (serde only, not re-exported).
#[derive(Deserialize)]
struct RawBusFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,
    /// Array of bus entries.
    buses: Vec<RawBus>,
}

/// Intermediate type for a single bus entry.
#[derive(Deserialize)]
struct RawBus {
    /// Bus identifier. Must be unique within the file.
    id: i32,
    /// Human-readable bus name.
    name: String,
    /// Optional entity-level deficit segment overrides.
    /// When absent, the global defaults from `penalties.json` are used.
    deficit_segments: Option<Vec<RawDeficitSegment>>,
}

/// Intermediate type for a single deficit segment entry.
#[derive(Deserialize)]
struct RawDeficitSegment {
    /// MW depth of this segment. `null` means unbounded (last segment only).
    depth_mw: Option<f64>,
    /// Cost per `MWh` of deficit in this segment [$/`MWh`].
    cost: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `system/buses.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to `Vec<Bus>` using
/// the two-tier penalty resolution cascade (global → entity). The result is
/// sorted by `id` ascending to satisfy declaration-order invariance.
///
/// # Errors
///
/// | Condition                                          | Error variant              |
/// | -------------------------------------------------- | -------------------------- |
/// | File not found / read failure                      | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field      | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the buses array              | [`LoadError::SchemaError`] |
/// | Deficit segment cost ≤ 0.0                         | [`LoadError::SchemaError`] |
/// | Last deficit segment has `depth_mw` set (not null) | [`LoadError::SchemaError`] |
/// | Deficit segment costs not monotonically increasing | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_buses;
/// use cobre_core::penalty::GlobalPenaltyDefaults;
/// use std::path::Path;
///
/// # fn make_global() -> GlobalPenaltyDefaults { unimplemented!() }
/// let global = make_global();
/// let buses = parse_buses(Path::new("case/system/buses.json"), &global).unwrap();
/// assert!(!buses.is_empty());
/// ```
pub fn parse_buses(
    path: &Path,
    global_penalties: &GlobalPenaltyDefaults,
) -> Result<Vec<Bus>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawBusFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_buses(&raw, path)?;

    Ok(convert_buses(raw, global_penalties))
}

/// Validate all invariants on the raw deserialized bus data.
fn validate_raw_buses(raw: &RawBusFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_bus_ids(&raw.buses, path)?;
    for (i, bus) in raw.buses.iter().enumerate() {
        if let Some(segments) = &bus.deficit_segments {
            validate_deficit_segments(segments, i, path)?;
        }
    }
    Ok(())
}

/// Check that no two buses share the same `id`.
fn validate_no_duplicate_bus_ids(buses: &[RawBus], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, bus) in buses.iter().enumerate() {
        if !seen.insert(bus.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("buses[{i}].id"),
                message: format!("duplicate id {} in buses array", bus.id),
            });
        }
    }
    Ok(())
}

/// Validate entity-level deficit segments for bus `bus_index`.
///
/// Checks: all costs > 0, last segment has `depth_mw: null`, costs monotonically
/// increasing. Mirrors the global deficit segment validation in `penalties.rs`.
fn validate_deficit_segments(
    segments: &[RawDeficitSegment],
    bus_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if segments.is_empty() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("buses[{bus_index}].deficit_segments"),
            message: "deficit_segments must not be empty".to_string(),
        });
    }

    for (j, seg) in segments.iter().enumerate() {
        if seg.cost <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("buses[{bus_index}].deficit_segments[{j}].cost"),
                message: format!("penalty value must be > 0.0, got {}", seg.cost),
            });
        }
    }

    let last_idx = segments.len() - 1;
    if segments[last_idx].depth_mw.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("buses[{bus_index}].deficit_segments[{last_idx}].depth_mw"),
            message: "the last deficit segment must have depth_mw: null (uncapped final segment)"
                .to_string(),
        });
    }

    for j in 1..segments.len() {
        let prev_cost = segments[j - 1].cost;
        let curr_cost = segments[j].cost;
        if curr_cost <= prev_cost {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("buses[{bus_index}].deficit_segments[{j}].cost"),
                message: format!(
                    "deficit segment costs must be monotonically increasing: \
                     segment[{}].cost ({}) must be > segment[{}].cost ({})",
                    j,
                    curr_cost,
                    j - 1,
                    prev_cost,
                ),
            });
        }
    }

    Ok(())
}

/// Convert validated raw bus data into `Vec<Bus>`, sorted by `id` ascending.
fn convert_buses(raw: RawBusFile, global: &GlobalPenaltyDefaults) -> Vec<Bus> {
    let mut buses: Vec<Bus> = raw
        .buses
        .into_iter()
        .map(|raw_bus| {
            // Convert optional entity-level deficit segments to core types.
            let entity_segments: Option<Vec<DeficitSegment>> =
                raw_bus.deficit_segments.map(|segs| {
                    segs.into_iter()
                        .map(|s| DeficitSegment {
                            depth_mw: s.depth_mw,
                            cost_per_mwh: s.cost,
                        })
                        .collect()
                });

            // Two-tier penalty resolution: entity override wins, else global default.
            let deficit_segments = resolve_bus_deficit_segments(&entity_segments, global);
            // Excess cost has no entity-level override in SS1 spec — always global.
            let excess_cost = resolve_bus_excess_cost(None, global);

            Bus {
                id: EntityId(raw_bus.id),
                name: raw_bus.name,
                deficit_segments,
                excess_cost,
            }
        })
        .collect();

    // Sort by id ascending to satisfy declaration-order invariance.
    buses.sort_by_key(|b| b.id.0);
    buses
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use super::*;
    use cobre_core::entities::HydroPenalties;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write a string to a temp file and return the file handle (keeps it alive).
    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    /// Build a canonical `GlobalPenaltyDefaults` for test use.
    fn make_global() -> GlobalPenaltyDefaults {
        GlobalPenaltyDefaults {
            bus_deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(500.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            bus_excess_cost: 100.0,
            line_exchange_cost: 2.0,
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                fpha_turbined_cost: 0.05,
                diversion_cost: 0.1,
                storage_violation_below_cost: 10_000.0,
                filling_target_violation_cost: 50_000.0,
                turbined_violation_below_cost: 500.0,
                outflow_violation_below_cost: 500.0,
                outflow_violation_above_cost: 500.0,
                generation_violation_below_cost: 1_000.0,
                evaporation_violation_cost: 5_000.0,
                water_withdrawal_violation_cost: 1_000.0,
            },
            ncs_curtailment_cost: 0.005,
        }
    }

    /// Canonical valid `buses.json` with 2 buses: one with entity-level deficit
    /// overrides (id=1), one without (id=0).
    const VALID_JSON: &str = r#"{
      "$schema": "https://cobre.dev/schemas/v2/buses.schema.json",
      "buses": [
        { "id": 0, "name": "South" },
        {
          "id": 1,
          "name": "North",
          "deficit_segments": [
            { "depth_mw": 300.0, "cost": 2000.0 },
            { "depth_mw": null, "cost": 8000.0 }
          ]
        }
      ]
    }"#;

    // ── AC: valid buses with and without entity-level deficit overrides ────────

    /// Given a valid `buses.json` with 2 buses (one with entity-level deficit
    /// overrides, one without), `parse_buses` returns `Ok(vec)` with 2 `Bus`
    /// entries sorted by `id`; the bus with overrides has its entity-level
    /// segments; the bus without has global default segments.
    #[test]
    fn test_parse_valid_buses() {
        let f = write_json(VALID_JSON);
        let global = make_global();
        let buses = parse_buses(f.path(), &global).unwrap();

        assert_eq!(buses.len(), 2);

        // Bus 0: no entity-level override -> uses global defaults
        assert_eq!(buses[0].id, EntityId(0));
        assert_eq!(buses[0].name, "South");
        assert_eq!(buses[0].deficit_segments.len(), 2);
        assert_eq!(buses[0].deficit_segments[0].depth_mw, Some(500.0));
        assert!(
            (buses[0].deficit_segments[0].cost_per_mwh - 1000.0).abs() < f64::EPSILON,
            "bus 0 segment 0 cost: expected 1000.0, got {}",
            buses[0].deficit_segments[0].cost_per_mwh
        );
        assert!(buses[0].deficit_segments[1].depth_mw.is_none());
        assert!(
            (buses[0].deficit_segments[1].cost_per_mwh - 5000.0).abs() < f64::EPSILON,
            "bus 0 segment 1 cost: expected 5000.0, got {}",
            buses[0].deficit_segments[1].cost_per_mwh
        );
        assert!((buses[0].excess_cost - 100.0).abs() < f64::EPSILON);

        // Bus 1: has entity-level override -> uses entity segments
        assert_eq!(buses[1].id, EntityId(1));
        assert_eq!(buses[1].name, "North");
        assert_eq!(buses[1].deficit_segments.len(), 2);
        assert_eq!(buses[1].deficit_segments[0].depth_mw, Some(300.0));
        assert!(
            (buses[1].deficit_segments[0].cost_per_mwh - 2000.0).abs() < f64::EPSILON,
            "bus 1 segment 0 cost: expected 2000.0, got {}",
            buses[1].deficit_segments[0].cost_per_mwh
        );
        assert!(buses[1].deficit_segments[1].depth_mw.is_none());
        assert!(
            (buses[1].deficit_segments[1].cost_per_mwh - 8000.0).abs() < f64::EPSILON,
            "bus 1 segment 1 cost: expected 8000.0, got {}",
            buses[1].deficit_segments[1].cost_per_mwh
        );
        // Excess cost is always from global — no entity-level override in spec.
        assert!((buses[1].excess_cost - 100.0).abs() < f64::EPSILON);
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `buses.json` with duplicate `id` values, `parse_buses` returns
    /// `Err(LoadError::SchemaError)` with field containing `"buses[1].id"` and
    /// message containing `"duplicate"`.
    #[test]
    fn test_duplicate_bus_id() {
        let json = r#"{
          "buses": [
            { "id": 5, "name": "Alpha" },
            { "id": 5, "name": "Beta" }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_buses(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("buses[1].id"),
                    "field should contain 'buses[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given buses in reverse ID order in JSON, `parse_buses` returns a
    /// `Vec<Bus>` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "buses": [
            { "id": 0, "name": "South" },
            { "id": 1, "name": "North" }
          ]
        }"#;
        let json_reversed = r#"{
          "buses": [
            { "id": 1, "name": "North" },
            { "id": 0, "name": "South" }
          ]
        }"#;
        let global = make_global();

        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let buses1 = parse_buses(f1.path(), &global).unwrap();
        let buses2 = parse_buses(f2.path(), &global).unwrap();

        assert_eq!(
            buses1, buses2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(buses1[0].id, EntityId(0));
        assert_eq!(buses1[1].id, EntityId(1));
    }

    // ── AC: entity-level deficit segment validation ───────────────────────────

    /// Bus with entity-level deficit segment having cost ≤ 0 → `SchemaError`.
    #[test]
    fn test_entity_deficit_segment_negative_cost() {
        let json = r#"{
          "buses": [
            {
              "id": 0,
              "name": "Bad",
              "deficit_segments": [
                { "depth_mw": null, "cost": -100.0 }
              ]
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_buses(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("deficit_segments[0].cost"),
                    "field should contain deficit_segments[0].cost, got: {field}"
                );
                assert!(
                    message.contains("> 0.0"),
                    "message should mention positive constraint, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Bus with entity-level deficit: last segment not uncapped → `SchemaError`.
    #[test]
    fn test_entity_deficit_segment_last_not_uncapped() {
        let json = r#"{
          "buses": [
            {
              "id": 0,
              "name": "Bad",
              "deficit_segments": [
                { "depth_mw": 100.0, "cost": 1000.0 },
                { "depth_mw": 200.0, "cost": 5000.0 }
              ]
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_buses(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("depth_mw"),
                    "field should contain 'depth_mw', got: {field}"
                );
                assert!(
                    message.contains("uncapped final segment"),
                    "message should mention uncapped final segment, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Bus with entity-level deficit: non-monotonic costs → `SchemaError`.
    #[test]
    fn test_entity_deficit_segment_non_monotonic() {
        let json = r#"{
          "buses": [
            {
              "id": 0,
              "name": "Bad",
              "deficit_segments": [
                { "depth_mw": 100.0, "cost": 5000.0 },
                { "depth_mw": null, "cost": 1000.0 }
              ]
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_buses(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("deficit_segments[1].cost"),
                    "field should name the non-monotonic segment, got: {field}"
                );
                assert!(
                    message.contains("monotonically increasing"),
                    "message should mention monotonic ordering, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_buses` returns `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/buses.json");
        let global = make_global();
        let err = parse_buses(path, &global).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_buses` returns `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"buses": [not valid json}}"#);
        let global = make_global();
        let err = parse_buses(f.path(), &global).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `buses` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_buses_array() {
        let json = r#"{ "buses": [] }"#;
        let f = write_json(json);
        let global = make_global();
        let buses = parse_buses(f.path(), &global).unwrap();
        assert!(buses.is_empty());
    }

    /// Excess cost on bus always comes from global defaults (no entity override).
    #[test]
    fn test_excess_cost_always_from_global() {
        let json = r#"{
          "buses": [{ "id": 0, "name": "Alpha" }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let buses = parse_buses(f.path(), &global).unwrap();
        assert!(
            (buses[0].excess_cost - global.bus_excess_cost).abs() < f64::EPSILON,
            "excess_cost must always equal global default, got {}",
            buses[0].excess_cost
        );
    }
}
