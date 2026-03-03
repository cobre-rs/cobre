//! Parsing for `penalties.json` — global penalty defaults.
//!
//! [`parse_penalties`] reads `penalties.json` from the case directory root and
//! returns a fully-validated [`GlobalPenaltyDefaults`].
//!
//! ## JSON structure
//!
//! The file has four top-level penalty sections (`bus`, `line`, `hydro`,
//! `non_controllable_source`) that map to flat fields on
//! [`GlobalPenaltyDefaults`]. Intermediate serde types bridge the nested JSON
//! schema to the flat Rust representation.
//!
//! ## Validation
//!
//! After deserializing, three invariants are checked before conversion:
//!
//! 1. Every scalar penalty value is strictly positive (> 0.0).
//! 2. The last deficit segment has `depth_mw: null` (unbounded final segment).
//! 3. Deficit segment costs are monotonically increasing.
//!
//! See the penalty system spec §3 for the full requirements.

use cobre_core::{
    entities::{DeficitSegment, HydroPenalties},
    penalty::GlobalPenaltyDefaults,
};
use serde::Deserialize;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `penalties.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
struct RawPenalties {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// File format version — informational, not validated.
    #[allow(dead_code)]
    version: Option<String>,

    /// Bus penalty defaults.
    bus: RawBusPenalties,

    /// Line penalty defaults.
    line: RawLinePenalties,

    /// Hydro penalty defaults.
    hydro: RawHydroPenalties,

    /// Non-controllable source penalty defaults.
    non_controllable_source: RawNcsPenalties,
}

/// Intermediate type for the `bus` section.
#[derive(Deserialize)]
struct RawBusPenalties {
    /// Piecewise-linear deficit cost segments.
    deficit_segments: Vec<RawDeficitSegment>,
    /// Excess generation cost \[$/`MWh`\].
    excess_cost: f64,
}

/// Intermediate type for one deficit segment entry.
#[derive(Deserialize)]
struct RawDeficitSegment {
    /// MW depth of this segment. `null` means the segment is unbounded (last segment).
    depth_mw: Option<f64>,
    /// Cost per `MWh` of deficit in this segment \[$/`MWh`\].
    cost: f64,
}

/// Intermediate type for the `line` section.
#[derive(Deserialize)]
struct RawLinePenalties {
    /// Exchange cost \[$/`MWh`\].
    exchange_cost: f64,
}

/// Intermediate type for the `hydro` section.
///
/// All fields end with `_cost` because these are penalty cost values. The
/// shared postfix is intentional and mirrors both the JSON schema and the
/// [`HydroPenalties`] struct field names.
#[allow(clippy::struct_field_names)]
#[derive(Deserialize)]
struct RawHydroPenalties {
    spillage_cost: f64,
    fpha_turbined_cost: f64,
    diversion_cost: f64,
    storage_violation_below_cost: f64,
    filling_target_violation_cost: f64,
    turbined_violation_below_cost: f64,
    outflow_violation_below_cost: f64,
    outflow_violation_above_cost: f64,
    generation_violation_below_cost: f64,
    evaporation_violation_cost: f64,
    water_withdrawal_violation_cost: f64,
}

/// Intermediate type for the `non_controllable_source` section.
#[derive(Deserialize)]
struct RawNcsPenalties {
    /// Curtailment cost \[$/`MWh`\].
    curtailment_cost: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `penalties.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types, then
/// performs post-deserialization validation before converting to
/// [`GlobalPenaltyDefaults`].
///
/// # Errors
///
/// | Condition                                      | Error variant              |
/// | ---------------------------------------------- | -------------------------- |
/// | File not found / read failure                  | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field  | [`LoadError::ParseError`]  |
/// | Penalty value ≤ 0.0                            | [`LoadError::SchemaError`] |
/// | Last deficit segment has `depth_mw` set        | [`LoadError::SchemaError`] |
/// | Deficit costs are not monotonically increasing | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::penalties::parse_penalties;
/// use std::path::Path;
///
/// let defaults = parse_penalties(Path::new("case/penalties.json")).unwrap();
/// assert!(defaults.bus_excess_cost > 0.0);
/// ```
pub fn parse_penalties(path: &Path) -> Result<GlobalPenaltyDefaults, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawPenalties =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    Ok(convert(raw))
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized data.
///
/// Called before conversion so that error messages can reference JSON field
/// paths rather than Rust field names.
fn validate_raw(raw: &RawPenalties, path: &Path) -> Result<(), LoadError> {
    validate_bus(raw, path)?;
    validate_line(raw, path)?;
    validate_hydro(raw, path)?;
    validate_ncs(raw, path)?;
    Ok(())
}

/// Check all bus penalty values and deficit segment invariants.
fn validate_bus(raw: &RawPenalties, path: &Path) -> Result<(), LoadError> {
    let bus = &raw.bus;

    if bus.excess_cost <= 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "bus.excess_cost".to_string(),
            message: format!("penalty value must be > 0.0, got {}", bus.excess_cost),
        });
    }

    if bus.deficit_segments.is_empty() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "bus.deficit_segments".to_string(),
            message: "deficit_segments must not be empty".to_string(),
        });
    }

    for (i, seg) in bus.deficit_segments.iter().enumerate() {
        if seg.cost <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("bus.deficit_segments[{i}].cost"),
                message: format!("penalty value must be > 0.0, got {}", seg.cost),
            });
        }
    }

    let last_idx = bus.deficit_segments.len() - 1;
    let last = &bus.deficit_segments[last_idx];
    if last.depth_mw.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("bus.deficit_segments[{last_idx}].depth_mw"),
            message: "the last deficit segment must have depth_mw: null (uncapped final segment)"
                .to_string(),
        });
    }

    for i in 1..bus.deficit_segments.len() {
        let prev_cost = bus.deficit_segments[i - 1].cost;
        let curr_cost = bus.deficit_segments[i].cost;
        if curr_cost <= prev_cost {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("bus.deficit_segments[{i}].cost"),
                message: format!(
                    "deficit segment costs must be monotonically increasing: \
                     segment[{}].cost ({}) must be > segment[{}].cost ({})",
                    i,
                    curr_cost,
                    i - 1,
                    prev_cost,
                ),
            });
        }
    }

    Ok(())
}

/// Check all line penalty values.
fn validate_line(raw: &RawPenalties, path: &Path) -> Result<(), LoadError> {
    if raw.line.exchange_cost <= 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "line.exchange_cost".to_string(),
            message: format!(
                "penalty value must be > 0.0, got {}",
                raw.line.exchange_cost
            ),
        });
    }
    Ok(())
}

/// Check all hydro penalty values.
fn validate_hydro(raw: &RawPenalties, path: &Path) -> Result<(), LoadError> {
    let h = &raw.hydro;

    // Build a list of (field_path, value) pairs for uniform checking.
    let fields: &[(&str, f64)] = &[
        ("hydro.spillage_cost", h.spillage_cost),
        ("hydro.fpha_turbined_cost", h.fpha_turbined_cost),
        ("hydro.diversion_cost", h.diversion_cost),
        (
            "hydro.storage_violation_below_cost",
            h.storage_violation_below_cost,
        ),
        (
            "hydro.filling_target_violation_cost",
            h.filling_target_violation_cost,
        ),
        (
            "hydro.turbined_violation_below_cost",
            h.turbined_violation_below_cost,
        ),
        (
            "hydro.outflow_violation_below_cost",
            h.outflow_violation_below_cost,
        ),
        (
            "hydro.outflow_violation_above_cost",
            h.outflow_violation_above_cost,
        ),
        (
            "hydro.generation_violation_below_cost",
            h.generation_violation_below_cost,
        ),
        (
            "hydro.evaporation_violation_cost",
            h.evaporation_violation_cost,
        ),
        (
            "hydro.water_withdrawal_violation_cost",
            h.water_withdrawal_violation_cost,
        ),
    ];

    for &(field, value) in fields {
        if value <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: field.to_string(),
                message: format!("penalty value must be > 0.0, got {value}"),
            });
        }
    }

    Ok(())
}

/// Check all non-controllable source penalty values.
fn validate_ncs(raw: &RawPenalties, path: &Path) -> Result<(), LoadError> {
    if raw.non_controllable_source.curtailment_cost <= 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "non_controllable_source.curtailment_cost".to_string(),
            message: format!(
                "penalty value must be > 0.0, got {}",
                raw.non_controllable_source.curtailment_cost
            ),
        });
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert validated raw data into [`GlobalPenaltyDefaults`].
///
/// Precondition: [`validate_raw`] has returned `Ok(())` for this data.
fn convert(raw: RawPenalties) -> GlobalPenaltyDefaults {
    let bus_deficit_segments = raw
        .bus
        .deficit_segments
        .into_iter()
        .map(|seg| DeficitSegment {
            depth_mw: seg.depth_mw,
            cost_per_mwh: seg.cost,
        })
        .collect();

    let hydro = HydroPenalties {
        spillage_cost: raw.hydro.spillage_cost,
        fpha_turbined_cost: raw.hydro.fpha_turbined_cost,
        diversion_cost: raw.hydro.diversion_cost,
        storage_violation_below_cost: raw.hydro.storage_violation_below_cost,
        filling_target_violation_cost: raw.hydro.filling_target_violation_cost,
        turbined_violation_below_cost: raw.hydro.turbined_violation_below_cost,
        outflow_violation_below_cost: raw.hydro.outflow_violation_below_cost,
        outflow_violation_above_cost: raw.hydro.outflow_violation_above_cost,
        generation_violation_below_cost: raw.hydro.generation_violation_below_cost,
        evaporation_violation_cost: raw.hydro.evaporation_violation_cost,
        water_withdrawal_violation_cost: raw.hydro.water_withdrawal_violation_cost,
    };

    GlobalPenaltyDefaults {
        bus_deficit_segments,
        bus_excess_cost: raw.bus.excess_cost,
        line_exchange_cost: raw.line.exchange_cost,
        hydro,
        ncs_curtailment_cost: raw.non_controllable_source.curtailment_cost,
    }
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

    /// Canonical valid `penalties.json` with 2 deficit segments.
    const VALID_JSON: &str = r#"{
      "$schema": "https://cobre.dev/schemas/v2/penalties.schema.json",
      "version": "1.0",
      "bus": {
        "deficit_segments": [
          { "depth_mw": 500.0, "cost": 1000.0 },
          { "depth_mw": null, "cost": 5000.0 }
        ],
        "excess_cost": 100.0
      },
      "line": {
        "exchange_cost": 2.0
      },
      "hydro": {
        "spillage_cost": 0.01,
        "fpha_turbined_cost": 0.05,
        "diversion_cost": 0.1,
        "storage_violation_below_cost": 10000.0,
        "filling_target_violation_cost": 50000.0,
        "turbined_violation_below_cost": 500.0,
        "outflow_violation_below_cost": 500.0,
        "outflow_violation_above_cost": 500.0,
        "generation_violation_below_cost": 1000.0,
        "evaporation_violation_cost": 5000.0,
        "water_withdrawal_violation_cost": 1000.0
      },
      "non_controllable_source": {
        "curtailment_cost": 0.005
      }
    }"#;

    // ── AC: parse valid penalties — all fields match ───────────────────────────

    /// Given a valid `penalties.json` with 2 deficit segments and all penalty
    /// fields, `parse_penalties` returns `Ok(defaults)` with correct field mapping.
    #[test]
    fn test_parse_valid_penalties() {
        let f = write_json(VALID_JSON);
        let defaults = parse_penalties(f.path()).unwrap();

        // Bus
        assert_eq!(defaults.bus_deficit_segments.len(), 2);
        assert_eq!(defaults.bus_deficit_segments[0].depth_mw, Some(500.0));
        assert!(
            (defaults.bus_deficit_segments[0].cost_per_mwh - 1000.0).abs() < f64::EPSILON,
            "expected 1000.0, got {}",
            defaults.bus_deficit_segments[0].cost_per_mwh
        );
        assert!(defaults.bus_deficit_segments[1].depth_mw.is_none());
        assert!(
            (defaults.bus_deficit_segments[1].cost_per_mwh - 5000.0).abs() < f64::EPSILON,
            "expected 5000.0, got {}",
            defaults.bus_deficit_segments[1].cost_per_mwh
        );
        assert!((defaults.bus_excess_cost - 100.0).abs() < f64::EPSILON);

        // Line
        assert!((defaults.line_exchange_cost - 2.0).abs() < f64::EPSILON);

        // Hydro
        assert!((defaults.hydro.spillage_cost - 0.01).abs() < f64::EPSILON);
        assert!((defaults.hydro.fpha_turbined_cost - 0.05).abs() < f64::EPSILON);
        assert!((defaults.hydro.diversion_cost - 0.1).abs() < f64::EPSILON);
        assert!((defaults.hydro.storage_violation_below_cost - 10_000.0).abs() < f64::EPSILON);
        assert!((defaults.hydro.filling_target_violation_cost - 50_000.0).abs() < f64::EPSILON);
        assert!((defaults.hydro.turbined_violation_below_cost - 500.0).abs() < f64::EPSILON);
        assert!((defaults.hydro.outflow_violation_below_cost - 500.0).abs() < f64::EPSILON);
        assert!((defaults.hydro.outflow_violation_above_cost - 500.0).abs() < f64::EPSILON);
        assert!((defaults.hydro.generation_violation_below_cost - 1000.0).abs() < f64::EPSILON);
        assert!((defaults.hydro.evaporation_violation_cost - 5000.0).abs() < f64::EPSILON);
        assert!((defaults.hydro.water_withdrawal_violation_cost - 1000.0).abs() < f64::EPSILON);

        // Non-controllable source
        assert!((defaults.ncs_curtailment_cost - 0.005).abs() < f64::EPSILON);
    }

    /// Given a valid `penalties.json`, when parsed and compared with a manually
    /// constructed `GlobalPenaltyDefaults`, all 15+ fields match.
    #[test]
    fn test_parse_penalties_matches_manual_construction() {
        let f = write_json(VALID_JSON);
        let defaults = parse_penalties(f.path()).unwrap();

        let expected = GlobalPenaltyDefaults {
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
                generation_violation_below_cost: 1000.0,
                evaporation_violation_cost: 5000.0,
                water_withdrawal_violation_cost: 1000.0,
            },
            ncs_curtailment_cost: 0.005,
        };

        assert_eq!(defaults, expected);
    }

    // ── AC: negative penalty values → SchemaError ─────────────────────────────

    /// Given a `penalties.json` with a negative penalty value, `parse_penalties`
    /// returns `Err(LoadError::SchemaError)` with the offending field name.
    #[test]
    fn test_parse_penalties_negative_value() {
        let json = r#"{
          "bus": {
            "deficit_segments": [{ "depth_mw": null, "cost": 1000.0 }],
            "excess_cost": -1.0
          },
          "line": { "exchange_cost": 2.0 },
          "hydro": {
            "spillage_cost": 0.01, "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1, "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
          },
          "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#;

        let f = write_json(json);
        let err = parse_penalties(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("excess_cost"),
                    "field should name excess_cost, got: {field}"
                );
                assert!(
                    message.contains("-1"),
                    "message should contain the offending value, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative cost in a deficit segment — `SchemaError` mentioning the segment field.
    #[test]
    fn test_parse_penalties_negative_deficit_segment_cost() {
        let json = r#"{
          "bus": {
            "deficit_segments": [
              { "depth_mw": 100.0, "cost": -500.0 },
              { "depth_mw": null, "cost": 5000.0 }
            ],
            "excess_cost": 100.0
          },
          "line": { "exchange_cost": 2.0 },
          "hydro": {
            "spillage_cost": 0.01, "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1, "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
          },
          "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#;

        let f = write_json(json);
        let err = parse_penalties(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("deficit_segments[0].cost"),
                    "field should name the offending segment, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: last deficit segment not uncapped → SchemaError ───────────────────

    /// Given a `penalties.json` where the last deficit segment has
    /// `depth_mw: 200.0` (not null), `parse_penalties` returns
    /// `Err(LoadError::SchemaError)` mentioning the uncapped final segment.
    #[test]
    fn test_parse_penalties_capped_last_segment() {
        let json = r#"{
          "bus": {
            "deficit_segments": [
              { "depth_mw": 500.0, "cost": 1000.0 },
              { "depth_mw": 200.0, "cost": 5000.0 }
            ],
            "excess_cost": 100.0
          },
          "line": { "exchange_cost": 2.0 },
          "hydro": {
            "spillage_cost": 0.01, "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1, "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
          },
          "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#;

        let f = write_json(json);
        let err = parse_penalties(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("deficit_segments[1].depth_mw"),
                    "field should name the last segment depth_mw, got: {field}"
                );
                assert!(
                    message.contains("uncapped final segment"),
                    "message should mention uncapped final segment, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given a valid `penalties.json` with last segment uncapped (`depth_mw: null`),
    /// `parse_penalties` returns `Ok`.
    #[test]
    fn test_parse_penalties_uncapped_last_segment() {
        let f = write_json(VALID_JSON);
        let result = parse_penalties(f.path());
        assert!(
            result.is_ok(),
            "expected Ok for valid uncapped last segment, got: {result:?}"
        );
    }

    // ── AC: non-monotonic deficit costs → SchemaError ─────────────────────────

    /// Given a `penalties.json` with non-monotonically increasing deficit segment
    /// costs, `parse_penalties` returns `Err(LoadError::SchemaError)`.
    #[test]
    fn test_parse_penalties_monotonic_deficit() {
        let json = r#"{
          "bus": {
            "deficit_segments": [
              { "depth_mw": 500.0, "cost": 5000.0 },
              { "depth_mw": 1000.0, "cost": 1000.0 },
              { "depth_mw": null, "cost": 3000.0 }
            ],
            "excess_cost": 100.0
          },
          "line": { "exchange_cost": 2.0 },
          "hydro": {
            "spillage_cost": 0.01, "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1, "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
          },
          "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#;

        let f = write_json(json);
        let err = parse_penalties(f.path()).unwrap_err();
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

    // ── AC: missing required field → ParseError ───────────────────────────────

    /// Given a `penalties.json` with a missing required field (`hydro` section
    /// omitted), `parse_penalties` returns `Err(LoadError::ParseError)`.
    #[test]
    fn test_parse_penalties_missing_field() {
        let json = r#"{
          "bus": {
            "deficit_segments": [{ "depth_mw": null, "cost": 1000.0 }],
            "excess_cost": 100.0
          },
          "line": { "exchange_cost": 2.0 },
          "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#;

        let f = write_json(json);
        let err = parse_penalties(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for missing hydro section, got: {err:?}"
        );
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_penalties` returns `Err(LoadError::IoError)`.
    #[test]
    fn test_parse_penalties_file_not_found() {
        let path = Path::new("/nonexistent/penalties.json");
        let err = parse_penalties(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── Additional validation edge cases ─────────────────────────────────────

    /// Negative hydro penalty value — `SchemaError` with the correct field.
    #[test]
    fn test_parse_penalties_negative_hydro_value() {
        let json = r#"{
          "bus": {
            "deficit_segments": [{ "depth_mw": null, "cost": 1000.0 }],
            "excess_cost": 100.0
          },
          "line": { "exchange_cost": 2.0 },
          "hydro": {
            "spillage_cost": -0.01,
            "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1, "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
          },
          "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#;

        let f = write_json(json);
        let err = parse_penalties(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("hydro.spillage_cost"),
                    "field should name hydro.spillage_cost, got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Three deficit segments — valid, with monotonically increasing costs and
    /// the last uncapped — deserializes correctly.
    #[test]
    fn test_parse_penalties_three_deficit_segments() {
        let json = r#"{
          "bus": {
            "deficit_segments": [
              { "depth_mw": 500.0, "cost": 1000.0 },
              { "depth_mw": 1000.0, "cost": 3000.0 },
              { "depth_mw": null, "cost": 5000.0 }
            ],
            "excess_cost": 100.0
          },
          "line": { "exchange_cost": 2.0 },
          "hydro": {
            "spillage_cost": 0.01, "fpha_turbined_cost": 0.05,
            "diversion_cost": 0.1, "storage_violation_below_cost": 10000.0,
            "filling_target_violation_cost": 50000.0,
            "turbined_violation_below_cost": 500.0,
            "outflow_violation_below_cost": 500.0,
            "outflow_violation_above_cost": 500.0,
            "generation_violation_below_cost": 1000.0,
            "evaporation_violation_cost": 5000.0,
            "water_withdrawal_violation_cost": 1000.0
          },
          "non_controllable_source": { "curtailment_cost": 0.005 }
        }"#;

        let f = write_json(json);
        let defaults = parse_penalties(f.path()).unwrap();
        assert_eq!(defaults.bus_deficit_segments.len(), 3);
        assert_eq!(defaults.bus_deficit_segments[0].depth_mw, Some(500.0));
        assert_eq!(defaults.bus_deficit_segments[1].depth_mw, Some(1000.0));
        assert!(defaults.bus_deficit_segments[2].depth_mw.is_none());
    }
}
