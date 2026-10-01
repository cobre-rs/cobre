//! Parsing for `system/non_controllable_sources.json` — intermittent source registry.
//!
//! [`parse_non_controllable_sources`] reads the file from the case directory and
//! returns a fully-validated, sorted `Vec<NonControllableSource>`.
//!
//! This file is **optional** — callers should use the
//! [`super::load_non_controllable_sources`] wrapper which accepts `Option<&Path>`
//! and returns `Ok(Vec::new())` when the file is absent.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/non_controllable_sources.schema.json",
//!   "non_controllable_sources": [
//!     {
//!       "id": 0,
//!       "name": "Eólica Caetité",
//!       "operational_start_date": "2024-01-01",
//!       "bus_id": 2,
//!       "max_generation_mw": 300.0,
//!       "curtailment_cost": 0.01
//!     },
//!     {
//!       "id": 1,
//!       "name": "Solar Pirapora",
//!       "operational_start_date": "2024-01-01",
//!       "bus_id": 3,
//!       "entry_stage_id": 12,
//!       "max_generation_mw": 400.0
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two sources share the same `id`.
//! 2. `max_generation_mw >= 0.0`.
//!
//! Cross-reference validation (e.g., checking that `bus_id` exists in the bus
//! registry) is deferred to Layer 3.

use cobre_core::{
    EntityId,
    entities::NonControllableSource,
    penalty::{GlobalPenaltyDefaults, resolve_ncs_curtailment_cost},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use super::parse_operational_start_date;
use crate::LoadError;

/// Default for [`RawNcs::allow_curtailment`] when omitted: sources are
/// curtailable unless the operator opts out.
fn default_allow_curtailment() -> bool {
    true
}

/// Top-level intermediate type for `non_controllable_sources.json` (serde only, not re-exported).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RawNcsFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,
    /// Array of non-controllable source entries.
    non_controllable_sources: Vec<RawNcs>,
}

/// Intermediate type for a single non-controllable source entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RawNcs {
    /// Source identifier. Must be unique within the file.
    id: i32,
    /// Human-readable source name.
    name: String,
    /// Date the entity enters service (ISO 8601 `YYYY-MM-DD`).
    operational_start_date: String,
    /// Bus to which this source's generation is injected.
    bus_id: i32,
    /// Stage index when the source enters service. Absent or null = always exists.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the source is decommissioned. Absent or null = never.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Maximum generation (installed capacity) \[MW\].
    max_generation_mw: f64,
    /// Whether the LP is allowed to curtail this source.
    ///
    /// `true` (default) — the LP may curtail dispatch within
    /// `[0, max_generation_mw × availability]` at `curtailment_cost`.
    ///
    /// `false` — must-run: dispatch is pinned to the realized availability
    /// every scenario (`col_lower = col_upper`). Use this for non-simulated
    /// aggregate generation (small hydro, biomass, wind, solar, distributed
    /// generation) that the source model pre-nets from load before the
    /// dispatch LP runs.
    #[serde(default = "default_allow_curtailment")]
    allow_curtailment: bool,
    /// Optional entity-level curtailment cost override \[$/`MWh`\].
    /// When absent, falls back to global `ncs_curtailment_cost`.
    #[serde(default)]
    curtailment_cost: Option<f64>,
}

/// Load and validate `system/non_controllable_sources.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to
/// `Vec<NonControllableSource>` using the two-tier penalty resolution cascade
/// (global → entity) for `curtailment_cost`. The result is sorted by `id`
/// ascending, so parser output is deterministic regardless of file row order
/// (declaration-order invariance); canonical order is
/// [`SystemBuilder::build`](cobre_core::SystemBuilder::build)'s to establish.
///
/// # Errors
///
/// | Condition                                          | Error variant              |
/// | -------------------------------------------------- | -------------------------- |
/// | File not found / read failure                      | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field      | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the sources array            | [`LoadError::SchemaError`] |
/// | Negative `max_generation_mw`                       | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_non_controllable_sources;
/// use cobre_core::penalty::GlobalPenaltyDefaults;
/// use std::path::Path;
///
/// # fn make_global() -> GlobalPenaltyDefaults { unimplemented!() }
/// let global = make_global();
/// let sources = parse_non_controllable_sources(
///     Path::new("case/system/non_controllable_sources.json"),
///     &global,
/// ).unwrap();
/// ```
pub fn parse_non_controllable_sources(
    path: &Path,
    global_penalties: &GlobalPenaltyDefaults,
) -> Result<Vec<NonControllableSource>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawNcsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_ncs(&raw, path)?;

    convert_ncs(raw, global_penalties, path)
}

fn validate_raw_ncs(raw: &RawNcsFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_ncs_ids(&raw.non_controllable_sources, path)?;
    for (i, ncs) in raw.non_controllable_sources.iter().enumerate() {
        validate_ncs_generation(ncs.max_generation_mw, i, path)?;
    }
    Ok(())
}

fn validate_no_duplicate_ncs_ids(sources: &[RawNcs], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, ncs) in sources.iter().enumerate() {
        if !seen.insert(ncs.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("non_controllable_sources[{i}].id"),
                message: format!("duplicate id {} in non_controllable_sources array", ncs.id),
            });
        }
    }
    Ok(())
}

fn validate_ncs_generation(
    max_generation_mw: f64,
    ncs_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if max_generation_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("non_controllable_sources[{ncs_index}].max_generation_mw"),
            message: format!("max_generation_mw must be >= 0.0, got {max_generation_mw}"),
        });
    }
    Ok(())
}

fn convert_ncs(
    raw: RawNcsFile,
    global: &GlobalPenaltyDefaults,
    path: &Path,
) -> Result<Vec<NonControllableSource>, LoadError> {
    let mut sources: Vec<NonControllableSource> = raw
        .non_controllable_sources
        .into_iter()
        .enumerate()
        .map(|(i, raw_ncs)| {
            let operational_start_date = parse_operational_start_date(
                &raw_ncs.operational_start_date,
                path,
                &format!("non_controllable_sources[{i}].operational_start_date"),
            )?;

            let curtailment_cost = resolve_ncs_curtailment_cost(raw_ncs.curtailment_cost, global);

            Ok(NonControllableSource {
                id: EntityId(raw_ncs.id),
                name: raw_ncs.name,
                operational_start_date,
                bus_id: EntityId(raw_ncs.bus_id),
                entry_stage_id: raw_ncs.entry_stage_id,
                exit_stage_id: raw_ncs.exit_stage_id,
                max_generation_mw: raw_ncs.max_generation_mw,
                allow_curtailment: raw_ncs.allow_curtailment,
                curtailment_cost,
            })
        })
        .collect::<Result<_, LoadError>>()?;

    // Sort by id: output must not depend on file row order (declaration-order
    // invariance).
    sources.sort_by_key(|s| s.id.0);
    Ok(sources)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::test_support::{make_global, write_json};

    // ── AC: valid NCS with and without entity-level curtailment_cost ──────────

    /// Given a valid `non_controllable_sources.json` with 2 sources (one with
    /// entity-level `curtailment_cost` override, one without), `parse_non_controllable_sources`
    /// returns `Ok(vec)` with 2 entries sorted by `id`; the source without override
    /// has `curtailment_cost` equal to the global default.
    #[test]
    fn test_parse_valid_ncs() {
        let json = r#"{
          "non_controllable_sources": [
            {
              "id": 0,
              "name": "Eólica Caetité",
              "operational_start_date": "2024-01-01",
              "bus_id": 2,
              "max_generation_mw": 300.0,
              "curtailment_cost": 0.01
            },
            {
              "id": 1,
              "name": "Solar Pirapora",
              "operational_start_date": "2024-01-01",
              "bus_id": 3,
              "entry_stage_id": 12,
              "max_generation_mw": 400.0
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let sources = parse_non_controllable_sources(f.path(), &global).unwrap();

        assert_eq!(sources.len(), 2);

        assert_eq!(sources[0].id, EntityId(0));
        assert_eq!(sources[0].name, "Eólica Caetité");
        assert_eq!(sources[0].bus_id, EntityId(2));
        assert_eq!(sources[0].entry_stage_id, None);
        assert_eq!(sources[0].exit_stage_id, None);
        assert!((sources[0].max_generation_mw - 300.0).abs() < f64::EPSILON);
        assert!(
            (sources[0].curtailment_cost - 0.01).abs() < f64::EPSILON,
            "expected entity curtailment_cost 0.01, got {}",
            sources[0].curtailment_cost
        );

        assert_eq!(sources[1].id, EntityId(1));
        assert_eq!(sources[1].name, "Solar Pirapora");
        assert_eq!(sources[1].entry_stage_id, Some(12));
        assert!((sources[1].max_generation_mw - 400.0).abs() < f64::EPSILON);
        assert!(
            (sources[1].curtailment_cost - 0.005).abs() < f64::EPSILON,
            "expected global curtailment_cost 0.005, got {}",
            sources[1].curtailment_cost
        );

        assert!(sources[0].allow_curtailment);
        assert!(sources[1].allow_curtailment);
    }

    /// `allow_curtailment` is parsed through to the resolved entity when
    /// supplied; `false` is the must-run case used for non-simulated
    /// aggregate generation.
    #[test]
    fn test_parse_allow_curtailment() {
        let json = r#"{
          "non_controllable_sources": [
            {
              "id": 0,
              "name": "PCH NE Aggregate",
              "operational_start_date": "2024-01-01",
              "bus_id": 0,
              "max_generation_mw": 120.0,
              "allow_curtailment": false
            },
            {
              "id": 1,
              "name": "Wind NE Partial",
              "operational_start_date": "2024-01-01",
              "bus_id": 0,
              "max_generation_mw": 300.0,
              "allow_curtailment": true
            }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let sources = parse_non_controllable_sources(f.path(), &global).unwrap();

        assert!(!sources[0].allow_curtailment, "must-run source");
        assert!(sources[1].allow_curtailment, "curtailable source");
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `non_controllable_sources.json` with duplicate `id` values,
    /// `parse_non_controllable_sources` returns `Err(LoadError::SchemaError)`.
    #[test]
    fn test_duplicate_ncs_id() {
        let json = r#"{
          "non_controllable_sources": [
            { "id": 3, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0, "max_generation_mw": 100.0 },
            { "id": 3, "name": "Beta", "operational_start_date": "2024-01-01",  "bus_id": 1, "max_generation_mw": 200.0 }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_non_controllable_sources(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("non_controllable_sources[1].id"),
                    "field should contain 'non_controllable_sources[1].id', got: {field}"
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

    /// Negative `max_generation_mw` → `SchemaError`.
    #[test]
    fn test_negative_max_generation_mw() {
        let json = r#"{
          "non_controllable_sources": [
            { "id": 0, "name": "Bad", "operational_start_date": "2024-01-01", "bus_id": 0, "max_generation_mw": -10.0 }
          ]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_non_controllable_sources(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_generation_mw"),
                    "field should contain 'max_generation_mw', got: {field}"
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

    /// Given sources in reverse ID order in JSON, `parse_non_controllable_sources`
    /// returns a `Vec` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "non_controllable_sources": [
            { "id": 0, "name": "Wind", "operational_start_date": "2024-01-01", "bus_id": 0, "max_generation_mw": 100.0 },
            { "id": 1, "name": "Solar", "operational_start_date": "2024-01-01", "bus_id": 1, "max_generation_mw": 200.0 }
          ]
        }"#;
        let json_reversed = r#"{
          "non_controllable_sources": [
            { "id": 1, "name": "Solar", "operational_start_date": "2024-01-01", "bus_id": 1, "max_generation_mw": 200.0 },
            { "id": 0, "name": "Wind", "operational_start_date": "2024-01-01",  "bus_id": 0, "max_generation_mw": 100.0 }
          ]
        }"#;
        let global = make_global();

        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let sources1 = parse_non_controllable_sources(f1.path(), &global).unwrap();
        let sources2 = parse_non_controllable_sources(f2.path(), &global).unwrap();

        assert_eq!(
            sources1, sources2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(sources1[0].id, EntityId(0));
        assert_eq!(sources1[1].id, EntityId(1));
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_non_controllable_sources` returns
    /// `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/non_controllable_sources.json");
        let global = make_global();
        let err = parse_non_controllable_sources(path, &global).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_non_controllable_sources` returns
    /// `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"non_controllable_sources": [not valid json}}"#);
        let global = make_global();
        let err = parse_non_controllable_sources(f.path(), &global).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `non_controllable_sources` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_ncs_array() {
        let json = r#"{ "non_controllable_sources": [] }"#;
        let f = write_json(json);
        let global = make_global();
        let sources = parse_non_controllable_sources(f.path(), &global).unwrap();
        assert!(sources.is_empty());
    }
}
