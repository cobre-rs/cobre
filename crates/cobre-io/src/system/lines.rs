//! Parsing for `system/lines.json` — transmission line entity registry.
//!
//! [`parse_lines`] reads `system/lines.json` from the case directory and
//! returns a fully-validated, sorted `Vec<Line>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/lines.schema.json",
//!   "lines": [
//!     {
//!       "id": 0,
//!       "name": "SE-S",
//!       "source_bus_id": 0,
//!       "target_bus_id": 1,
//!       "capacity": { "direct_mw": 2500.0, "reverse_mw": 2000.0 },
//!       "exchange_cost": 1.5,
//!       "losses_percent": 0.5
//!     },
//!     {
//!       "id": 1,
//!       "name": "SE-NE",
//!       "source_bus_id": 0,
//!       "target_bus_id": 2,
//!       "entry_stage_id": 1,
//!       "exit_stage_id": 60,
//!       "capacity": { "direct_mw": 4000.0, "reverse_mw": 3500.0 }
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two lines share the same `id`.
//! 2. `direct_mw >= 0.0` and `reverse_mw >= 0.0`.
//! 3. `losses_percent >= 0.0` (defaults to 0.0 when absent).
//!
//! Cross-reference validation (e.g., checking that `source_bus_id` and `target_bus_id`
//! exist in the bus registry) is deferred to Layer 3.

use cobre_core::{
    EntityId,
    entities::Line,
    penalty::{GlobalPenaltyDefaults, resolve_line_exchange_cost},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

/// Top-level intermediate type for `lines.json` (serde only, not re-exported).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawLineFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,
    /// Array of line entries.
    lines: Vec<RawLine>,
}

/// Intermediate type for a single line entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawLine {
    /// Line identifier. Must be unique within the file.
    id: i32,
    /// Human-readable line name.
    name: String,
    /// Source bus identifier.
    source_bus_id: i32,
    /// Target bus identifier.
    target_bus_id: i32,
    /// Stage index when the line enters service. Absent or null = always exists.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the line is decommissioned. Absent or null = never.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Nested capacity object with direct and reverse MW limits.
    capacity: RawLineCapacity,
    /// Optional entity-level exchange cost override [$/`MWh`].
    /// When absent, falls back to global `line_exchange_cost`.
    #[serde(default)]
    exchange_cost: Option<f64>,
    /// Transmission losses as percentage. Defaults to 0.0 when absent.
    #[serde(default)]
    losses_percent: f64,
}

/// Intermediate type for the nested capacity sub-object.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawLineCapacity {
    /// Maximum flow from source to target [MW].
    direct_mw: f64,
    /// Maximum flow from target to source [MW].
    reverse_mw: f64,
}

/// Load and validate `system/lines.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to `Vec<Line>` using
/// the two-tier penalty resolution cascade (global → entity) for `exchange_cost`.
/// The result is sorted by `id` ascending to satisfy declaration-order invariance.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// | --------------------------------------------- | -------------------------- |
/// | File not found / read failure                 | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the lines array         | [`LoadError::SchemaError`] |
/// | Negative `direct_mw` or `reverse_mw`          | [`LoadError::SchemaError`] |
/// | Negative `losses_percent`                     | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_lines;
/// use cobre_core::penalty::GlobalPenaltyDefaults;
/// use std::path::Path;
///
/// # fn make_global() -> GlobalPenaltyDefaults { unimplemented!() }
/// let global = make_global();
/// let lines = parse_lines(Path::new("case/system/lines.json"), &global).unwrap();
/// assert!(!lines.is_empty());
/// ```
pub fn parse_lines(
    path: &Path,
    global_penalties: &GlobalPenaltyDefaults,
) -> Result<Vec<Line>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawLineFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_lines(&raw, path)?;

    Ok(convert_lines(raw, global_penalties))
}

/// Validate all invariants on the raw deserialized line data.
fn validate_raw_lines(raw: &RawLineFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_line_ids(&raw.lines, path)?;
    for (i, line) in raw.lines.iter().enumerate() {
        validate_line_capacity(&line.capacity, i, path)?;
        validate_losses_percent(line.losses_percent, i, path)?;
    }
    Ok(())
}

/// Check that no two lines share the same `id`.
fn validate_no_duplicate_line_ids(lines: &[RawLine], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, line) in lines.iter().enumerate() {
        if !seen.insert(line.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("lines[{i}].id"),
                message: format!("duplicate id {} in lines array", line.id),
            });
        }
    }
    Ok(())
}

/// Validate capacity values for line at `line_index`.
///
/// Checks: `direct_mw >= 0.0`, `reverse_mw >= 0.0`.
fn validate_line_capacity(
    capacity: &RawLineCapacity,
    line_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if capacity.direct_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("lines[{line_index}].capacity.direct_mw"),
            message: format!("direct_mw must be >= 0.0, got {}", capacity.direct_mw),
        });
    }
    if capacity.reverse_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("lines[{line_index}].capacity.reverse_mw"),
            message: format!("reverse_mw must be >= 0.0, got {}", capacity.reverse_mw),
        });
    }
    Ok(())
}

/// Validate `losses_percent` for line at `line_index`.
///
/// Checks: `losses_percent >= 0.0`.
fn validate_losses_percent(
    losses_percent: f64,
    line_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if losses_percent < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("lines[{line_index}].losses_percent"),
            message: format!("losses_percent must be >= 0.0, got {losses_percent}"),
        });
    }
    Ok(())
}

/// Convert validated raw line data into `Vec<Line>`, sorted by `id` ascending.
fn convert_lines(raw: RawLineFile, global: &GlobalPenaltyDefaults) -> Vec<Line> {
    let mut lines: Vec<Line> = raw
        .lines
        .into_iter()
        .map(|raw_line| {
            // Two-tier penalty resolution: entity override wins, else global default.
            let exchange_cost = resolve_line_exchange_cost(raw_line.exchange_cost, global);

            Line {
                id: EntityId(raw_line.id),
                name: raw_line.name,
                source_bus_id: EntityId(raw_line.source_bus_id),
                target_bus_id: EntityId(raw_line.target_bus_id),
                entry_stage_id: raw_line.entry_stage_id,
                exit_stage_id: raw_line.exit_stage_id,
                direct_capacity_mw: raw_line.capacity.direct_mw,
                reverse_capacity_mw: raw_line.capacity.reverse_mw,
                losses_percent: raw_line.losses_percent,
                exchange_cost,
            }
        })
        .collect();

    // Sort by id ascending to satisfy declaration-order invariance.
    lines.sort_by_key(|l| l.id.0);
    lines
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use super::*;
    use cobre_core::entities::{DeficitSegment, HydroPenalties};
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
                water_withdrawal_violation_pos_cost: 1_000.0,
                water_withdrawal_violation_neg_cost: 1_000.0,
                evaporation_violation_pos_cost: 5_000.0,
                evaporation_violation_neg_cost: 5_000.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            ncs_curtailment_cost: 0.005,
        }
    }

    // ── AC: valid lines with and without entity-level exchange_cost override ──

    /// Given a valid `lines.json` with 2 lines (one with entity-level `exchange_cost`
    /// override, one without), `parse_lines` returns `Ok(vec)` with 2 `Line` entries
    /// sorted by `id`; the line without override has `exchange_cost` equal to the
    /// global default.
    #[test]
    fn test_parse_valid_lines() {
        let json = r#"{
          "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/lines.schema.json",
          "lines": [
            {
              "id": 0,
              "name": "SE-S",
              "source_bus_id": 0,
              "target_bus_id": 1,
              "capacity": { "direct_mw": 2500.0, "reverse_mw": 2000.0 },
              "exchange_cost": 1.5,
              "losses_percent": 0.5
            },
            {
              "id": 1,
              "name": "SE-NE",
              "source_bus_id": 0,
              "target_bus_id": 2,
              "entry_stage_id": 1,
              "exit_stage_id": 60,
              "capacity": { "direct_mw": 4000.0, "reverse_mw": 3500.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let lines = parse_lines(f.path(), &global).unwrap();

        assert_eq!(lines.len(), 2);

        // Line 0: has entity-level exchange_cost override
        assert_eq!(lines[0].id, EntityId(0));
        assert_eq!(lines[0].name, "SE-S");
        assert_eq!(lines[0].source_bus_id, EntityId(0));
        assert_eq!(lines[0].target_bus_id, EntityId(1));
        assert_eq!(lines[0].entry_stage_id, None);
        assert_eq!(lines[0].exit_stage_id, None);
        assert!((lines[0].direct_capacity_mw - 2500.0).abs() < f64::EPSILON);
        assert!((lines[0].reverse_capacity_mw - 2000.0).abs() < f64::EPSILON);
        assert!((lines[0].losses_percent - 0.5).abs() < f64::EPSILON);
        assert!(
            (lines[0].exchange_cost - 1.5).abs() < f64::EPSILON,
            "expected entity-level exchange_cost 1.5, got {}",
            lines[0].exchange_cost
        );

        // Line 1: no entity-level override -> uses global default (2.0)
        assert_eq!(lines[1].id, EntityId(1));
        assert_eq!(lines[1].name, "SE-NE");
        assert_eq!(lines[1].entry_stage_id, Some(1));
        assert_eq!(lines[1].exit_stage_id, Some(60));
        assert!((lines[1].direct_capacity_mw - 4000.0).abs() < f64::EPSILON);
        assert!((lines[1].reverse_capacity_mw - 3500.0).abs() < f64::EPSILON);
        assert!((lines[1].losses_percent - 0.0).abs() < f64::EPSILON);
        assert!(
            (lines[1].exchange_cost - 2.0).abs() < f64::EPSILON,
            "expected global exchange_cost 2.0, got {}",
            lines[1].exchange_cost
        );
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `lines.json` with duplicate `id` values, `parse_lines` returns
    /// `Err(LoadError::SchemaError)` with field containing `"lines[1].id"` and
    /// message containing `"duplicate"`.
    #[test]
    fn test_duplicate_line_id() {
        let json = r#"{
          "lines": [
            {
              "id": 5, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": 100.0, "reverse_mw": 100.0 }
            },
            {
              "id": 5, "name": "Beta", "source_bus_id": 1, "target_bus_id": 2,
              "capacity": { "direct_mw": 200.0, "reverse_mw": 200.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_lines(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("lines[1].id"),
                    "field should contain 'lines[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: negative capacity → SchemaError ───────────────────────────────────

    /// Negative `direct_mw` → `SchemaError`.
    #[test]
    fn test_negative_direct_mw() {
        let json = r#"{
          "lines": [
            {
              "id": 0, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": -100.0, "reverse_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_lines(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("direct_mw"),
                    "field should contain 'direct_mw', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `reverse_mw` → `SchemaError`.
    #[test]
    fn test_negative_reverse_mw() {
        let json = r#"{
          "lines": [
            {
              "id": 0, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": 100.0, "reverse_mw": -50.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_lines(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reverse_mw"),
                    "field should contain 'reverse_mw', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `losses_percent` → `SchemaError`.
    #[test]
    fn test_negative_losses_percent() {
        let json = r#"{
          "lines": [
            {
              "id": 0, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": 100.0, "reverse_mw": 100.0 },
              "losses_percent": -1.0
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_lines(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("losses_percent"),
                    "field should contain 'losses_percent', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given lines in reverse ID order in JSON, `parse_lines` returns a
    /// `Vec<Line>` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "lines": [
            {
              "id": 0, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": 100.0, "reverse_mw": 100.0 }
            },
            {
              "id": 1, "name": "Beta", "source_bus_id": 1, "target_bus_id": 2,
              "capacity": { "direct_mw": 200.0, "reverse_mw": 200.0 }
            }
          ]
        }"#;
        let json_reversed = r#"{
          "lines": [
            {
              "id": 1, "name": "Beta", "source_bus_id": 1, "target_bus_id": 2,
              "capacity": { "direct_mw": 200.0, "reverse_mw": 200.0 }
            },
            {
              "id": 0, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": 100.0, "reverse_mw": 100.0 }
            }
          ]
        }"#;
        let global = make_global();

        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let lines1 = parse_lines(f1.path(), &global).unwrap();
        let lines2 = parse_lines(f2.path(), &global).unwrap();

        assert_eq!(
            lines1, lines2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(lines1[0].id, EntityId(0));
        assert_eq!(lines1[1].id, EntityId(1));
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_lines` returns `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/lines.json");
        let global = make_global();
        let err = parse_lines(path, &global).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_lines` returns `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"lines": [not valid json}}"#);
        let global = make_global();
        let err = parse_lines(f.path(), &global).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `lines` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_lines_array() {
        let json = r#"{ "lines": [] }"#;
        let f = write_json(json);
        let global = make_global();
        let lines = parse_lines(f.path(), &global).unwrap();
        assert!(lines.is_empty());
    }

    /// `losses_percent` defaults to 0.0 when absent.
    #[test]
    fn test_losses_percent_defaults_to_zero() {
        let json = r#"{
          "lines": [
            {
              "id": 0, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": 100.0, "reverse_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let lines = parse_lines(f.path(), &global).unwrap();
        assert!(
            (lines[0].losses_percent - 0.0).abs() < f64::EPSILON,
            "losses_percent should default to 0.0, got {}",
            lines[0].losses_percent
        );
    }

    /// Zero-capacity line is valid (degenerate but not invalid at schema level).
    #[test]
    fn test_zero_capacity_is_valid() {
        let json = r#"{
          "lines": [
            {
              "id": 0, "name": "Alpha", "source_bus_id": 0, "target_bus_id": 1,
              "capacity": { "direct_mw": 0.0, "reverse_mw": 0.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let result = parse_lines(f.path(), &global);
        assert!(
            result.is_ok(),
            "zero-capacity line should be valid, got: {result:?}"
        );
    }
}
