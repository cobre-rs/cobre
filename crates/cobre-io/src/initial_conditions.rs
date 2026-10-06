//! Parsing for `initial_conditions.json` — initial system state.
//!
//! [`parse_initial_conditions`] reads `initial_conditions.json` from the case
//! directory root and returns a fully-validated [`InitialConditions`].
//!
//! ## JSON structure
//!
//! The file contains two required top-level arrays and three optional arrays:
//!
//! - `storage` — initial reservoir volumes for operating hydros (hm³).
//! - `filling_storage` — initial reservoir volumes for filling hydros (hm³).
//! - `recent_observations` — the conditioning layer over the windowed
//!   `inflow_history` record ([`crate::scenarios::parse_inflow_history`]):
//!   observed inflow data for partial periods before the study start (m³/s
//!   per date range per hydro), shadowing the record day-wise wherever both
//!   cover the same date (the two never blend). Optional; defaults to an
//!   empty array when absent.
//! - `past_anticipated_commitments` — committed MW windows per anticipated
//!   thermal plant, each a self-describing `[start_date, end_date)` window
//!   carrying a `value_mw` rate held constant over the window. Optional;
//!   defaults to an empty array when no anticipated thermals are present.
//! - `past_defluences` — past release windows per arc (m³/s per date range),
//!   keyed by the upstream hydro whose release feeds the arc. Each entry is a
//!   self-describing `[start_date, end_date)` window on the pre-study calendar,
//!   mirroring `recent_observations`. Optional; defaults to an empty array when
//!   absent.
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/initial_conditions.schema.json",
//!   "storage": [
//!     { "hydro_id": 0, "value_hm3": 15000.0 },
//!     { "hydro_id": 1, "value_hm3": 8500.0 }
//!   ],
//!   "filling_storage": [{ "hydro_id": 10, "value_hm3": 200.0 }],
//!   "recent_observations": [
//!     { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": 500.0 },
//!     { "hydro_id": 0, "start_date": "2026-04-04", "end_date": "2026-04-11", "value_m3s": 480.0 }
//!   ],
//!   "past_anticipated_commitments": [
//!     { "thermal_id": 1, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 0.0 }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. Every `value_hm3` is non-negative (`>= 0.0`).
//! 2. No `hydro_id` appears more than once within `storage` or within
//!    `filling_storage` (no intra-array duplicates).
//! 3. No `hydro_id` appears in both `storage` and `filling_storage`
//!    (mutual exclusion).
//! 4. Every `start_date` and `end_date` in `recent_observations` parses as
//!    ISO 8601 (`YYYY-MM-DD`), and `end_date > start_date`.
//! 5. Every `value_m3s` in `recent_observations` is finite. A negative value is
//!    accepted — the quantity is incremental inflow, a difference between a
//!    plant's natural flow and its upstream plants' — and Layer 5a reports the
//!    count.
//! 6. For observations with the same `hydro_id`, date ranges do not overlap
//!    (adjacent ranges where `start == prev_end` are accepted).
//! 7. No `(thermal_id, start_date)` pair appears more than once in
//!    `past_anticipated_commitments`.
//! 8. Every `start_date` and `end_date` in `past_anticipated_commitments`
//!    parses as ISO 8601 (`YYYY-MM-DD`), `end_date > start_date`, every
//!    `value_mw` is finite, and windows for the same `thermal_id` do not overlap
//!    (adjacent ranges where `start == prev_end` are accepted). All four route
//!    through the shared windowed-record validator.
//! 9. Each thermal's windows tile the delivery stages it decides before the
//!    study at coverage `1.0` — no gap, no overlap — on both sides of the study
//!    horizon: the leading in-study stages and, when the plant's lead reaches
//!    past the horizon, the post-horizon stages it decides before the study. A
//!    window may not cover a stage the study itself decides or one past the
//!    plant's decision reach. Every in-study `value_mw` lies within that
//!    delivery stage's resolved generation bounds (the same bounds the
//!    anticipated-commitment decision column reads); a purely post-horizon
//!    `value_mw` lies within the plant's static
//!    `[min_generation_mw, max_generation_mw]` bounds instead. Either way, if
//!    the plant has a commissioning window, the value must mature inside it.
//!    All are enforced by the semantic validator (Layer 5a); the committed
//!    values are sunk cost and do not enter the study objective. See
//!    [`AnticipatedCommitmentHistory`] in `cobre-core` for the full contract.
//! 10. Every `start_date` and `end_date` in `past_defluences` parses as ISO 8601
//!     (`YYYY-MM-DD`), and `end_date > start_date`.
//! 11. Every `value_m3s` in `past_defluences` is finite and non-negative.
//! 12. For defluence windows with the same `hydro_id`, date ranges do not overlap
//!     (adjacent ranges where `start == prev_end` are accepted).
//!
//! Cross-reference validation (checking that hydro IDs exist in the hydro
//! registry) is deferred to Layer 3 (deferred). Storage bounds validation
//! (value within `[min_storage_hm3, max_storage_hm3]`) also requires the
//! hydro registry and is likewise deferred.

use chrono::NaiveDate;
use cobre_core::{
    AnticipatedCommitmentHistory, EntityId, HydroPastDefluence, HydroStorage, InitialConditions,
    RecentObservation,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;
use crate::windowed_history::{WindowedRecord, parse_iso_date, validate_windowed_records};

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Root object of `initial_conditions.json`.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawInitialConditions {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Initial reservoir volumes for operating hydros (hm³).
    storage: Vec<RawHydroStorage>,

    /// Initial reservoir volumes for filling hydros (hm³).
    /// A filling hydro may not also appear in `storage`.
    filling_storage: Vec<RawHydroStorage>,

    /// Observed inflow data for partial periods before the study start
    /// (m³/s per date range per hydro). Used to seed the lag accumulator when
    /// a study begins mid-season. Date ranges for the same hydro must not
    /// overlap; adjacent ranges (start == previous end) are accepted.
    /// Optional; defaults to empty.
    #[serde(default)]
    recent_observations: Vec<RawRecentObservation>,

    /// Past committed MW windows per anticipated thermal plant, each a
    /// self-describing `[start_date, end_date)` window carrying a `value_mw`
    /// rate. Present only when the study includes at least one anticipated
    /// thermal. Windows for the same thermal must not overlap; adjacent ranges
    /// (start == previous end) are accepted. Optional; defaults to empty.
    #[serde(default)]
    past_anticipated_commitments: Vec<RawAnticipatedCommitmentHistory>,

    /// Past defluence (release) windows per arc (m³/s per date range), keyed by
    /// the upstream hydro whose release feeds the arc. Each entry is a
    /// self-describing `[start_date, end_date)` window on the pre-study calendar.
    /// Windows for the same hydro must not overlap; adjacent ranges
    /// (start == previous end) are accepted. Optional; defaults to empty.
    #[serde(default)]
    past_defluences: Vec<RawHydroPastDefluence>,
}

/// Initial reservoir volume for one hydro plant, in hm³.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHydroStorage {
    /// Hydro plant identifier. Must be unique within its array.
    hydro_id: i32,
    /// Reservoir volume (hm³). Must be >= 0.0.
    value_hm3: f64,
}

/// Past defluence (release) for the arc fed by a single upstream hydro over a
/// specific date range.
///
/// Same window form as a `recent_observations` entry: a self-describing
/// `[start_date, end_date)`
/// window (ISO 8601, `end_date` exclusive and after `start_date`) carrying the
/// average release rate over the window. Multiple windows per hydro are allowed;
/// date ranges for the same hydro must not overlap, though adjacent ranges
/// (`start_date == previous end_date`) are accepted.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHydroPastDefluence {
    /// Upstream hydro plant identifier whose release feeds the arc.
    hydro_id: i32,
    /// Start of the release window (inclusive), as an ISO 8601 date
    /// (YYYY-MM-DD).
    start_date: String,
    /// End of the release window (exclusive), as an ISO 8601 date (YYYY-MM-DD).
    /// Must be after `start_date`.
    end_date: String,
    /// Average release rate over the window (m³/s). Must be finite and
    /// non-negative.
    value_m3s: f64,
}

/// Observed inflow for a single hydro plant over a specific date range.
///
/// Used to seed the lag accumulator when a study begins mid-season (before the
/// first lag-period boundary). Each entry covers one hydro over one
/// observation period. Multiple entries per hydro are allowed for rolling
/// revisions.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecentObservation {
    /// Hydro plant identifier. Must reference a hydro entity in the system.
    hydro_id: i32,
    /// Start of the observation period (inclusive), as an ISO 8601 date
    /// (YYYY-MM-DD).
    start_date: String,
    /// End of the observation period (exclusive), as an ISO 8601 date
    /// (YYYY-MM-DD). Must be after `start_date`.
    end_date: String,
    /// Average inflow observed during the period (m³/s). Must be finite;
    /// negative values are accepted (the quantity is incremental inflow).
    value_m3s: f64,
}

/// One past committed MW window for an anticipated thermal plant.
///
/// A self-describing `[start_date, end_date)` window (ISO 8601, `end_date`
/// exclusive and after `start_date`) carrying `value_mw`, the MW rate held
/// constant over the window. A plant's windows must tile the delivery stages it
/// decides before the study exactly — those may fall on either side of the
/// study horizon (leading in-study stages, or post-horizon stages the plant
/// decides ahead of the study), never a stage the study itself decides
/// (validated semantically). The values are sunk cost: they do not enter the
/// study objective. Each in-study `value_mw` must lie within that delivery
/// stage's resolved generation bounds; a purely post-horizon `value_mw` must
/// lie within the plant's static `[min_generation_mw, max_generation_mw]`
/// bounds instead. Either way, if the plant has a commissioning window, the
/// value must mature inside it (both validated semantically).
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawAnticipatedCommitmentHistory {
    /// Thermal plant identifier. Must reference an anticipated thermal.
    thermal_id: i32,
    /// Start of the commitment window (inclusive), as an ISO 8601 date
    /// (YYYY-MM-DD).
    start_date: String,
    /// End of the commitment window (exclusive), as an ISO 8601 date
    /// (YYYY-MM-DD). Must be after `start_date`.
    end_date: String,
    /// Committed MW rate over the window, held constant.
    value_mw: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `initial_conditions.json` from `path`.
///
/// # Errors
///
/// | Condition                                              | Error variant              |
/// | ------------------------------------------------------ | -------------------------- |
/// | File not found / read failure                          | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field          | [`LoadError::ParseError`]  |
/// | Negative `value_hm3`                                  | [`LoadError::SchemaError`] |
/// | Duplicate `hydro_id` within `storage`                 | [`LoadError::SchemaError`] |
/// | Duplicate `hydro_id` within `filling_storage`         | [`LoadError::SchemaError`] |
/// | `hydro_id` in both `storage` and `filling_storage`    | [`LoadError::SchemaError`] |
/// | Invalid date / non-finite value / overlap in `past_defluences` | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::initial_conditions::parse_initial_conditions;
/// use std::path::Path;
///
/// let ic = parse_initial_conditions(Path::new("case/initial_conditions.json")).unwrap();
/// assert_eq!(ic.storage.len(), 2);
/// ```
pub fn parse_initial_conditions(path: &Path) -> Result<InitialConditions, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawInitialConditions =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    Ok(convert(raw))
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized data.
///
/// Called before conversion so that error messages can reference JSON field
/// paths rather than Rust field names.
fn validate_raw(raw: &RawInitialConditions, path: &Path) -> Result<(), LoadError> {
    validate_non_negative(&raw.storage, "storage", path)?;
    validate_non_negative(&raw.filling_storage, "filling_storage", path)?;
    validate_no_duplicates(&raw.storage, "storage", path)?;
    validate_no_duplicates(&raw.filling_storage, "filling_storage", path)?;
    validate_mutual_exclusion(raw, path)?;

    let recent_observation_windows =
        parse_recent_observation_windows(&raw.recent_observations, path)?;
    validate_windowed_records(
        &recent_observation_windows,
        "recent_observations",
        "hydro_id",
        "value_m3s",
        path,
    )?;

    let anticipated_commitment_windows =
        parse_anticipated_commitment_windows(&raw.past_anticipated_commitments, path)?;
    validate_no_duplicate_commitment_windows(&anticipated_commitment_windows, path)?;
    validate_windowed_records(
        &anticipated_commitment_windows,
        "past_anticipated_commitments",
        "thermal_id",
        "value_mw",
        path,
    )?;

    let past_defluence_windows = parse_past_defluence_windows(&raw.past_defluences, path)?;
    validate_windowed_records(
        &past_defluence_windows,
        "past_defluences",
        "hydro_id",
        "value_m3s",
        path,
    )?;
    validate_past_defluences_non_negative(&raw.past_defluences, path)?;

    Ok(())
}

/// Parse `recent_observations` dates and build the shared windowed view for
/// [`validate_windowed_records`].
fn parse_recent_observation_windows(
    entries: &[RawRecentObservation],
    path: &Path,
) -> Result<Vec<WindowedRecord>, LoadError> {
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let start_date = parse_iso_date(
                &format!("recent_observations[{i}].start_date"),
                &entry.start_date,
                path,
            )?;
            let end_date = parse_iso_date(
                &format!("recent_observations[{i}].end_date"),
                &entry.end_date,
                path,
            )?;
            Ok(WindowedRecord {
                entity_id: entry.hydro_id,
                start_date,
                end_date,
                value: entry.value_m3s,
            })
        })
        .collect()
}

/// Parse `past_anticipated_commitments` dates and build the shared windowed
/// view for [`validate_windowed_records`].
fn parse_anticipated_commitment_windows(
    entries: &[RawAnticipatedCommitmentHistory],
    path: &Path,
) -> Result<Vec<WindowedRecord>, LoadError> {
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let start_date = parse_iso_date(
                &format!("past_anticipated_commitments[{i}].start_date"),
                &entry.start_date,
                path,
            )?;
            let end_date = parse_iso_date(
                &format!("past_anticipated_commitments[{i}].end_date"),
                &entry.end_date,
                path,
            )?;
            Ok(WindowedRecord {
                entity_id: entry.thermal_id,
                start_date,
                end_date,
                value: entry.value_mw,
            })
        })
        .collect()
}

/// Parse `past_defluences` dates and build the shared windowed view for
/// [`validate_windowed_records`].
fn parse_past_defluence_windows(
    entries: &[RawHydroPastDefluence],
    path: &Path,
) -> Result<Vec<WindowedRecord>, LoadError> {
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let start_date = parse_iso_date(
                &format!("past_defluences[{i}].start_date"),
                &entry.start_date,
                path,
            )?;
            let end_date = parse_iso_date(
                &format!("past_defluences[{i}].end_date"),
                &entry.end_date,
                path,
            )?;
            Ok(WindowedRecord {
                entity_id: entry.hydro_id,
                start_date,
                end_date,
                value: entry.value_m3s,
            })
        })
        .collect()
}

fn validate_non_negative(
    entries: &[RawHydroStorage],
    array_name: &str,
    path: &Path,
) -> Result<(), LoadError> {
    for (i, entry) in entries.iter().enumerate() {
        if entry.value_hm3 < 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{array_name}[{i}].value_hm3"),
                message: format!("value_hm3 must be >= 0.0, got {}", entry.value_hm3),
            });
        }
    }
    Ok(())
}

fn validate_no_duplicates(
    entries: &[RawHydroStorage],
    array_name: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        if !seen.insert(entry.hydro_id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{array_name}[{i}].hydro_id"),
                message: format!("duplicate hydro_id {} in {array_name}", entry.hydro_id),
            });
        }
    }
    Ok(())
}

fn validate_mutual_exclusion(raw: &RawInitialConditions, path: &Path) -> Result<(), LoadError> {
    let storage_ids: HashSet<i32> = raw.storage.iter().map(|e| e.hydro_id).collect();

    for (i, entry) in raw.filling_storage.iter().enumerate() {
        if storage_ids.contains(&entry.hydro_id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("filling_storage[{i}].hydro_id"),
                message: format!(
                    "hydro_id {} appears in both storage and filling_storage; \
                     a hydro must appear in exactly one of the two arrays",
                    entry.hydro_id
                ),
            });
        }
    }
    Ok(())
}

/// Reject two commitment windows sharing a `(thermal_id, start_date)` — the
/// windowed successor to the "duplicate `thermal_id`" rule. A thermal legitimately
/// carries multiple windows at distinct start dates; only a repeated start on
/// the same thermal is a duplicate. Date/overlap/finiteness are enforced by
/// [`validate_windowed_records`].
fn validate_no_duplicate_commitment_windows(
    windows: &[WindowedRecord],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<(i32, NaiveDate)> = HashSet::new();
    for (i, window) in windows.iter().enumerate() {
        if !seen.insert((window.entity_id, window.start_date)) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("past_anticipated_commitments[{i}].start_date"),
                message: format!(
                    "duplicate (thermal_id {}, start_date {}) in past_anticipated_commitments",
                    window.entity_id, window.start_date
                ),
            });
        }
    }
    Ok(())
}

/// Reject negative `value_m3s` — the sign policy that distinguishes
/// `past_defluences` from `recent_observations`/`inflow_history` (finiteness
/// is enforced by [`validate_windowed_records`]).
fn validate_past_defluences_non_negative(
    entries: &[RawHydroPastDefluence],
    path: &Path,
) -> Result<(), LoadError> {
    for (i, entry) in entries.iter().enumerate() {
        if entry.value_m3s < 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("past_defluences[{i}].value_m3s"),
                message: format!(
                    "past_defluences[{i}].value_m3s must be >= 0.0, got {}",
                    entry.value_m3s
                ),
            });
        }
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

fn parse_validated_date(date_str: &str) -> NaiveDate {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .unwrap_or_else(|_| unreachable!("date already validated"))
}

fn convert_storage(entries: Vec<RawHydroStorage>) -> Vec<HydroStorage> {
    let mut storage: Vec<HydroStorage> = entries
        .into_iter()
        .map(|e| HydroStorage {
            hydro_id: EntityId(e.hydro_id),
            value_hm3: e.value_hm3,
        })
        .collect();
    storage.sort_by_key(|e| e.hydro_id.0);
    storage
}

/// Convert validated raw data into [`InitialConditions`].
///
/// Precondition: [`validate_raw`] has returned `Ok(())` for this data.
/// All arrays are sorted by `hydro_id` to satisfy the declaration-order
/// invariance requirement.
fn convert(raw: RawInitialConditions) -> InitialConditions {
    let storage = convert_storage(raw.storage);
    let filling_storage = convert_storage(raw.filling_storage);

    let mut recent_observations: Vec<RecentObservation> = raw
        .recent_observations
        .into_iter()
        .map(|e| RecentObservation {
            hydro_id: EntityId(e.hydro_id),
            start_date: parse_validated_date(&e.start_date),
            end_date: parse_validated_date(&e.end_date),
            value_m3s: e.value_m3s,
        })
        .collect();
    recent_observations.sort_by_key(|e| (e.hydro_id.0, e.start_date));

    let mut past_anticipated_commitments: Vec<AnticipatedCommitmentHistory> = raw
        .past_anticipated_commitments
        .into_iter()
        .map(|e| AnticipatedCommitmentHistory {
            thermal_id: EntityId(e.thermal_id),
            start_date: parse_validated_date(&e.start_date),
            end_date: parse_validated_date(&e.end_date),
            value_mw: e.value_mw,
        })
        .collect();
    past_anticipated_commitments.sort_by_key(|e| (e.thermal_id.0, e.start_date));

    let mut past_defluences: Vec<HydroPastDefluence> = raw
        .past_defluences
        .into_iter()
        .map(|e| HydroPastDefluence {
            hydro_id: EntityId(e.hydro_id),
            start_date: parse_validated_date(&e.start_date),
            end_date: parse_validated_date(&e.end_date),
            value_m3s: e.value_m3s,
        })
        .collect();
    past_defluences.sort_by_key(|e| (e.hydro_id.0, e.start_date));

    InitialConditions {
        storage,
        filling_storage,
        past_anticipated_commitments,
        recent_observations,
        past_defluences,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::test_support::write_json;

    /// Canonical valid `initial_conditions.json` with 2 storage and 1 filling entry.
    const VALID_JSON: &str = r#"{
      "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/initial_conditions.schema.json",
      "storage": [
        { "hydro_id": 0, "value_hm3": 15000.0 },
        { "hydro_id": 1, "value_hm3": 8500.0 }
      ],
      "filling_storage": [
        { "hydro_id": 10, "value_hm3": 200.0 }
      ]
    }"#;

    // ── AC: parse valid initial conditions ────────────────────────────────────

    /// Given a valid `initial_conditions.json` with 2 storage entries and
    /// 1 filling entry, `parse_initial_conditions` returns `Ok(ic)` with
    /// correct field counts and entity IDs.
    #[test]
    fn test_parse_valid_initial_conditions() {
        let f = write_json(VALID_JSON);
        let ic = parse_initial_conditions(f.path()).unwrap();

        assert_eq!(ic.storage.len(), 2);
        assert_eq!(ic.filling_storage.len(), 1);

        assert_eq!(ic.storage[0].hydro_id, EntityId(0));
        assert!(
            (ic.storage[0].value_hm3 - 15_000.0).abs() < f64::EPSILON,
            "expected 15000.0, got {}",
            ic.storage[0].value_hm3
        );
        assert_eq!(ic.storage[1].hydro_id, EntityId(1));
        assert!(
            (ic.storage[1].value_hm3 - 8_500.0).abs() < f64::EPSILON,
            "expected 8500.0, got {}",
            ic.storage[1].value_hm3
        );

        assert_eq!(ic.filling_storage[0].hydro_id, EntityId(10));
        assert!(
            (ic.filling_storage[0].value_hm3 - 200.0).abs() < f64::EPSILON,
            "expected 200.0, got {}",
            ic.filling_storage[0].value_hm3
        );
    }

    // ── AC: empty arrays → Ok ─────────────────────────────────────────────────

    /// Given an `initial_conditions.json` with empty arrays, `parse_initial_conditions`
    /// returns `Ok(ic)` with empty storage and `filling_storage` vectors.
    #[test]
    fn test_parse_empty_arrays() {
        let json = r#"{ "storage": [], "filling_storage": [] }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert!(ic.storage.is_empty());
        assert!(ic.filling_storage.is_empty());
    }

    // ── AC: negative value_hm3 → SchemaError ─────────────────────────────────

    /// Given an `initial_conditions.json` with a negative `value_hm3` in
    /// `storage`, `parse_initial_conditions` returns `Err(LoadError::SchemaError)`
    /// with field containing `"value_hm3"`.
    #[test]
    fn test_negative_storage_value() {
        let json = r#"{
          "storage": [
            { "hydro_id": 0, "value_hm3": -1.0 }
          ],
          "filling_storage": []
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("value_hm3"),
                    "field should contain 'value_hm3', got: {field}"
                );
                assert!(
                    message.contains("value_hm3"),
                    "message should mention value_hm3, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given an `initial_conditions.json` with a negative `value_hm3` in
    /// `filling_storage`, `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` with field containing `"value_hm3"`.
    #[test]
    fn test_negative_filling_storage_value() {
        let json = r#"{
          "storage": [],
          "filling_storage": [
            { "hydro_id": 10, "value_hm3": -100.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("value_hm3"),
                    "field should contain 'value_hm3', got: {field}"
                );
                assert!(
                    message.contains("value_hm3"),
                    "message should mention value_hm3, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: duplicate hydro_id within storage → SchemaError ───────────────────

    /// Given an `initial_conditions.json` where the same `hydro_id` appears
    /// twice in `storage`, `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` mentioning "duplicate".
    #[test]
    fn test_duplicate_hydro_id_in_storage() {
        let json = r#"{
          "storage": [
            { "hydro_id": 5, "value_hm3": 1000.0 },
            { "hydro_id": 5, "value_hm3": 2000.0 }
          ],
          "filling_storage": []
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("storage"),
                    "field should mention 'storage', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should mention 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given an `initial_conditions.json` where the same `hydro_id` appears
    /// twice in `filling_storage`, `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` mentioning "duplicate".
    #[test]
    fn test_duplicate_hydro_id_in_filling_storage() {
        let json = r#"{
          "storage": [],
          "filling_storage": [
            { "hydro_id": 10, "value_hm3": 100.0 },
            { "hydro_id": 10, "value_hm3": 200.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("filling_storage"),
                    "field should mention 'filling_storage', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should mention 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: hydro_id in both lists → SchemaError ──────────────────────────────

    /// Given an `initial_conditions.json` where the same `hydro_id` appears in
    /// both `storage` and `filling_storage`, `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` mentioning mutual exclusion.
    #[test]
    fn test_hydro_id_in_both_lists() {
        let json = r#"{
          "storage": [
            { "hydro_id": 5, "value_hm3": 1000.0 }
          ],
          "filling_storage": [
            { "hydro_id": 5, "value_hm3": 100.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("filling_storage"),
                    "field should mention 'filling_storage', got: {field}"
                );
                assert!(
                    message.contains("storage") && message.contains("filling_storage"),
                    "message should mention both arrays for mutual exclusion, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_initial_conditions` returns
    /// `Err(LoadError::IoError)` with the matching path.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/initial_conditions.json");
        let err = parse_initial_conditions(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Zero storage value (exactly 0.0) is valid — the boundary is non-negative.
    #[test]
    fn test_zero_storage_value_is_valid() {
        let json = r#"{
          "storage": [
            { "hydro_id": 0, "value_hm3": 0.0 }
          ],
          "filling_storage": []
        }"#;
        let f = write_json(json);
        let result = parse_initial_conditions(f.path());
        assert!(
            result.is_ok(),
            "0.0 is non-negative and must be accepted, got: {result:?}"
        );
    }

    /// Filling storage value below dead volume is valid . Only non-negativity is
    /// checked here; bounds against dead volume are deferred to Layer 3.
    #[test]
    fn test_filling_storage_below_dead_volume_is_valid() {
        let json = r#"{
          "storage": [],
          "filling_storage": [
            { "hydro_id": 10, "value_hm3": 1.0 }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_initial_conditions(f.path());
        assert!(
            result.is_ok(),
            "filling storage values below dead volume are valid at this layer, got: {result:?}"
        );
    }

    /// Declaration-order invariance: arrays are sorted by `hydro_id` after loading.
    #[test]
    fn test_declaration_order_invariance() {
        let json_ordered = r#"{
          "storage": [
            { "hydro_id": 0, "value_hm3": 1000.0 },
            { "hydro_id": 1, "value_hm3": 2000.0 }
          ],
          "filling_storage": []
        }"#;
        let json_reversed = r#"{
          "storage": [
            { "hydro_id": 1, "value_hm3": 2000.0 },
            { "hydro_id": 0, "value_hm3": 1000.0 }
          ],
          "filling_storage": []
        }"#;

        let f1 = write_json(json_ordered);
        let f2 = write_json(json_reversed);
        let ic1 = parse_initial_conditions(f1.path()).unwrap();
        let ic2 = parse_initial_conditions(f2.path()).unwrap();

        assert_eq!(
            ic1, ic2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(ic1.storage[0].hydro_id, EntityId(0));
        assert_eq!(ic1.storage[1].hydro_id, EntityId(1));
    }

    /// Invalid JSON syntax → `ParseError`.
    #[test]
    fn test_invalid_json_syntax() {
        let f = write_json(r#"{"storage": [not valid json}}"#);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    /// Missing required field `storage` → `ParseError`.
    #[test]
    fn test_missing_required_field() {
        let json = r#"{ "filling_storage": [] }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for missing storage field, got: {err:?}"
        );
    }

    // ── AC: recent_observations absent → empty Vec (backward compat) ──────────

    /// Given a `initial_conditions.json` without a `recent_observations` key,
    /// `parse_initial_conditions` returns `Ok(ic)` with an empty
    /// `recent_observations` vec.
    #[test]
    fn test_recent_observations_absent_defaults_to_empty() {
        let f = write_json(VALID_JSON);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert!(
            ic.recent_observations.is_empty(),
            "absent recent_observations must default to empty vec"
        );
    }

    /// Given `"recent_observations": []`, `parse_initial_conditions` returns
    /// `Ok(ic)` with an empty `recent_observations` vec.
    #[test]
    fn test_recent_observations_empty_array() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": []
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert!(ic.recent_observations.is_empty());
    }

    // ── AC: valid recent_observations → parsed and sorted ─────────────────────

    /// Given two valid `recent_observations` entries for the same hydro with
    /// adjacent (non-overlapping) date ranges, `parse_initial_conditions` returns
    /// `Ok(ic)` with both entries, dates parsed as `NaiveDate`.
    #[test]
    fn test_recent_observations_valid_two_entries() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": 500.0 },
            { "hydro_id": 0, "start_date": "2026-04-04", "end_date": "2026-04-11", "value_m3s": 480.0 }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert_eq!(ic.recent_observations.len(), 2);
        assert_eq!(ic.recent_observations[0].hydro_id, EntityId(0));
        assert_eq!(
            ic.recent_observations[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 4, 1).unwrap()
        );
        assert_eq!(
            ic.recent_observations[0].end_date,
            chrono::NaiveDate::from_ymd_opt(2026, 4, 4).unwrap()
        );
        assert!((ic.recent_observations[0].value_m3s - 500.0).abs() < f64::EPSILON);
        assert!((ic.recent_observations[1].value_m3s - 480.0).abs() < f64::EPSILON);
    }

    // ── AC: invalid start_date format → SchemaError ───────────────────────────

    /// Given a `recent_observations` entry with an invalid `start_date` format
    /// (slash-separated), `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` with field containing `start_date`.
    #[test]
    fn test_recent_observations_invalid_start_date_format() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026/04/01", "end_date": "2026-04-04", "value_m3s": 500.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("start_date"),
                    "field should mention 'start_date', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given a `recent_observations` entry with an invalid `end_date` format,
    /// `parse_initial_conditions` returns `Err(LoadError::SchemaError)` with
    /// field containing `end_date`.
    #[test]
    fn test_recent_observations_invalid_end_date_format() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "not-a-date", "value_m3s": 500.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("end_date"),
                    "field should mention 'end_date', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: end_date == start_date → SchemaError ──────────────────────────────

    /// Given a `recent_observations` entry where `end_date == start_date`,
    /// `parse_initial_conditions` returns `Err(LoadError::SchemaError)` with
    /// `message` containing "`end_date` must be after `start_date`".
    #[test]
    fn test_recent_observations_end_date_equals_start_date() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-01", "value_m3s": 500.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("end_date must be after start_date"),
                    "message should contain 'end_date must be after start_date', got: {message}"
                );
                assert!(
                    message.contains("hydro_id 0"),
                    "message should name the entity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given a `recent_observations` entry where `end_date < start_date`,
    /// `parse_initial_conditions` returns `Err(LoadError::SchemaError)` with
    /// `message` containing "`end_date` must be after `start_date`".
    #[test]
    fn test_recent_observations_end_date_before_start_date() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-05", "end_date": "2026-04-01", "value_m3s": 500.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("end_date must be after start_date"),
                    "message should contain 'end_date must be after start_date', got: {message}"
                );
                assert!(
                    message.contains("hydro_id 0"),
                    "message should name the entity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: negative value_m3s loads (incremental inflow) ────────────────────

    #[test]
    fn test_recent_observations_negative_value_accepted() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": -1.0 }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert_eq!(ic.recent_observations.len(), 1);
        assert!((ic.recent_observations[0].value_m3s - (-1.0)).abs() < f64::EPSILON);
    }

    // ── AC: overlapping date ranges → SchemaError ─────────────────────────────

    /// Given two `recent_observations` entries for the same hydro with
    /// overlapping date ranges, `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` with `message` containing "overlapping".
    #[test]
    fn test_recent_observations_overlapping_ranges() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-05", "value_m3s": 500.0 },
            { "hydro_id": 0, "start_date": "2026-04-03", "end_date": "2026-04-10", "value_m3s": 480.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("overlapping"),
                    "message should contain 'overlapping', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: adjacent non-overlapping ranges → Ok ──────────────────────────────

    /// Given two `recent_observations` entries for the same hydro where
    /// `start == prev_end` (adjacent, exclusive-end convention), they are
    /// accepted.
    #[test]
    fn test_recent_observations_adjacent_ranges_are_valid() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": 500.0 },
            { "hydro_id": 0, "start_date": "2026-04-04", "end_date": "2026-04-11", "value_m3s": 480.0 }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_initial_conditions(f.path());
        assert!(
            result.is_ok(),
            "adjacent ranges (start == prev_end) must be accepted, got: {result:?}"
        );
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given `recent_observations` entries for `hydro_id`s [1, 0] in that order,
    /// the result is sorted by `(hydro_id, start_date)` with `hydro_id` 0 first.
    #[test]
    fn test_recent_observations_sorted_by_hydro_id_then_start_date() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 1, "start_date": "2026-04-01", "end_date": "2026-04-07", "value_m3s": 300.0 },
            { "hydro_id": 0, "start_date": "2026-04-04", "end_date": "2026-04-11", "value_m3s": 480.0 },
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": 500.0 }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert_eq!(ic.recent_observations.len(), 3);
        assert_eq!(ic.recent_observations[0].hydro_id, EntityId(0));
        assert_eq!(
            ic.recent_observations[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 4, 1).unwrap()
        );
        assert_eq!(ic.recent_observations[1].hydro_id, EntityId(0));
        assert_eq!(
            ic.recent_observations[1].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 4, 4).unwrap()
        );
        assert_eq!(ic.recent_observations[2].hydro_id, EntityId(1));
    }

    /// Given `recent_observations` entries declared in reverse vs forward order,
    /// the resulting `InitialConditions` values are equal (declaration-order
    /// invariance).
    #[test]
    fn test_recent_observations_declaration_order_invariance() {
        let json_forward = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": 500.0 },
            { "hydro_id": 0, "start_date": "2026-04-04", "end_date": "2026-04-11", "value_m3s": 480.0 }
          ]
        }"#;
        let json_reversed = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-04", "end_date": "2026-04-11", "value_m3s": 480.0 },
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": 500.0 }
          ]
        }"#;
        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let ic1 = parse_initial_conditions(f1.path()).unwrap();
        let ic2 = parse_initial_conditions(f2.path()).unwrap();
        assert_eq!(
            ic1, ic2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(
            ic1.recent_observations[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 4, 1).unwrap()
        );
    }

    // ── AC: past_anticipated_commitments ─────────────────────────────────────

    /// Given an `initial_conditions.json` with one `past_anticipated_commitments`
    /// entry, `parse_initial_conditions` returns the correct parsed value.
    #[test]
    fn test_parse_past_anticipated_commitments_present() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 1, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 120.0 },
            { "thermal_id": 1, "start_date": "2026-02-01", "end_date": "2026-03-01", "value_mw": 180.0 }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();

        assert_eq!(ic.past_anticipated_commitments.len(), 2);
        assert_eq!(ic.past_anticipated_commitments[0].thermal_id, EntityId(1));
        assert_eq!(
            ic.past_anticipated_commitments[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
        );
        assert!((ic.past_anticipated_commitments[0].value_mw - 120.0).abs() < f64::EPSILON);
        assert!((ic.past_anticipated_commitments[1].value_mw - 180.0).abs() < f64::EPSILON);
    }

    /// Given an `initial_conditions.json` with no `past_anticipated_commitments`
    /// key, `parse_initial_conditions` returns an empty vec.
    #[test]
    fn test_parse_past_anticipated_commitments_absent_defaults_empty() {
        let f = write_json(VALID_JSON);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert!(
            ic.past_anticipated_commitments.is_empty(),
            "absent past_anticipated_commitments must default to empty vec"
        );
    }

    /// Given two JSON inputs identical except for entry order in
    /// `past_anticipated_commitments`, the resulting vecs are identical
    /// (sorted by `thermal_id` ascending — declaration-order invariance).
    #[test]
    fn test_past_anticipated_commitments_declaration_order_invariance() {
        let json_forward = r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 1, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 120.0 },
            { "thermal_id": 1, "start_date": "2026-02-01", "end_date": "2026-03-01", "value_mw": 180.0 },
            { "thermal_id": 2, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 50.0 }
          ]
        }"#;
        let json_reversed = r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 2, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 50.0 },
            { "thermal_id": 1, "start_date": "2026-02-01", "end_date": "2026-03-01", "value_mw": 180.0 },
            { "thermal_id": 1, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 120.0 }
          ]
        }"#;
        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let ic1 = parse_initial_conditions(f1.path()).unwrap();
        let ic2 = parse_initial_conditions(f2.path()).unwrap();

        assert_eq!(
            ic1.past_anticipated_commitments, ic2.past_anticipated_commitments,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(ic1.past_anticipated_commitments[0].thermal_id, EntityId(1));
        assert_eq!(
            ic1.past_anticipated_commitments[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
        );
        assert_eq!(ic1.past_anticipated_commitments[1].thermal_id, EntityId(1));
        assert_eq!(
            ic1.past_anticipated_commitments[1].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 2, 1).unwrap()
        );
        assert_eq!(ic1.past_anticipated_commitments[2].thermal_id, EntityId(2));
    }

    /// Given two windows sharing `(thermal_id: 5, start_date)` in
    /// `past_anticipated_commitments`, `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` with field containing
    /// `"past_anticipated_commitments["` and message containing `"duplicate"`.
    #[test]
    fn test_duplicate_thermal_id_in_past_anticipated_commitments_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 5, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 100.0 },
            { "thermal_id": 5, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 200.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("past_anticipated_commitments["),
                    "field should contain 'past_anticipated_commitments[', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// A thermal legitimately carries several windows at distinct start dates;
    /// `parse_initial_conditions` accepts them (contrast with the shared
    /// `(thermal_id, start_date)` duplicate rule).
    #[test]
    fn test_multiple_windows_same_thermal_distinct_starts_accepted() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 3, "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 100.0 },
            { "thermal_id": 3, "start_date": "2026-02-01", "end_date": "2026-03-01", "value_mw": 200.0 }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert_eq!(ic.past_anticipated_commitments.len(), 2);
    }

    /// A commitment window whose `end_date <= start_date` is rejected by the
    /// shared windowed-record validator.
    #[test]
    fn test_commitment_window_end_before_start_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 7, "start_date": "2026-02-01", "end_date": "2026-01-01", "value_mw": 120.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("end_date must be after start_date"),
                    "message should contain 'end_date must be after start_date', got: {message}"
                );
                assert!(
                    message.contains("thermal_id 7"),
                    "message should name the entity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Two overlapping commitment windows for the same thermal are rejected by
    /// the shared windowed-record validator.
    #[test]
    fn test_commitment_windows_overlapping_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 7, "start_date": "2026-01-01", "end_date": "2026-02-15", "value_mw": 120.0 },
            { "thermal_id": 7, "start_date": "2026-02-01", "end_date": "2026-03-01", "value_mw": 130.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("overlapping"),
                    "message should contain 'overlapping', got: {message}"
                );
                assert!(
                    message.contains("thermal_id 7"),
                    "message should name thermal_id (the commitment surface's own entity \
                     label), got: {message}"
                );
                assert!(
                    !message.contains("hydro_id"),
                    "message must not borrow the hydro surfaces' entity label, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // Non-finite `value_mw` has no JSON-reachable regression here: serde_json
    // errors on an out-of-range float literal before deserialization ever
    // produces an f64. The commitment surface's finiteness-error labels are
    // covered directly against `validate_windowed_records` in
    // `windowed_history`'s test module, which builds a `WindowedRecord` with
    // `f64::NAN` instead of going through JSON text.

    // ── AC: past_defluences absent → empty; present → parsed and sorted ───────

    /// Given a `initial_conditions.json` without a `past_defluences` key,
    /// `parse_initial_conditions` returns `Ok(ic)` with an empty
    /// `past_defluences` vec.
    #[test]
    fn test_past_defluences_absent_defaults_to_empty() {
        let f = write_json(VALID_JSON);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert!(
            ic.past_defluences.is_empty(),
            "absent past_defluences must default to empty vec"
        );
    }

    /// Given two valid `past_defluences` windows for the same hydro with adjacent
    /// (non-overlapping) date ranges plus one for another hydro, the entries are
    /// parsed with dates as `NaiveDate` and sorted by `(hydro_id, start_date)`
    /// (declaration-order invariance).
    #[test]
    fn test_parse_valid_past_defluences_windows_sorted() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_defluences": [
            { "hydro_id": 1, "start_date": "2023-12-30", "end_date": "2024-01-01", "value_m3s": 200.0 },
            { "hydro_id": 0, "start_date": "2023-12-25", "end_date": "2023-12-28", "value_m3s": 600.0 },
            { "hydro_id": 0, "start_date": "2023-12-28", "end_date": "2024-01-01", "value_m3s": 500.0 }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert_eq!(ic.past_defluences.len(), 3);
        assert_eq!(ic.past_defluences[0].hydro_id, EntityId(0));
        assert_eq!(
            ic.past_defluences[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2023, 12, 25).unwrap()
        );
        assert_eq!(ic.past_defluences[1].hydro_id, EntityId(0));
        assert_eq!(
            ic.past_defluences[1].start_date,
            chrono::NaiveDate::from_ymd_opt(2023, 12, 28).unwrap()
        );
        assert_eq!(ic.past_defluences[2].hydro_id, EntityId(1));
        assert!((ic.past_defluences[2].value_m3s - 200.0).abs() < f64::EPSILON);
    }

    /// Given two `past_defluences` windows for the same hydro with overlapping
    /// date ranges, `parse_initial_conditions` returns `Err(LoadError::SchemaError)`
    /// with `message` containing "overlapping".
    #[test]
    fn test_past_defluences_overlapping_windows_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_defluences": [
            { "hydro_id": 0, "start_date": "2023-12-25", "end_date": "2023-12-30", "value_m3s": 500.0 },
            { "hydro_id": 0, "start_date": "2023-12-28", "end_date": "2024-01-01", "value_m3s": 480.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("overlapping"),
                    "message should contain 'overlapping', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given a `past_defluences` window where `end_date <= start_date`,
    /// `parse_initial_conditions` returns `Err(LoadError::SchemaError)` with
    /// `message` containing "`end_date` must be after `start_date`".
    #[test]
    fn test_past_defluences_end_before_start_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_defluences": [
            { "hydro_id": 0, "start_date": "2024-01-01", "end_date": "2023-12-30", "value_m3s": 500.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("end_date must be after start_date"),
                    "message should contain 'end_date must be after start_date', got: {message}"
                );
                assert!(
                    message.contains("hydro_id 0"),
                    "message should name the entity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given a `past_defluences` window with `value_m3s: -1.0`,
    /// `parse_initial_conditions` returns `Err(LoadError::SchemaError)` with
    /// field containing `"value_m3s"`.
    #[test]
    fn test_past_defluences_negative_value_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_defluences": [
            { "hydro_id": 0, "start_date": "2023-12-30", "end_date": "2024-01-01", "value_m3s": -1.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("value_m3s"),
                    "field should contain 'value_m3s', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: legacy past_inflows field → rejected ──────────────────────────────

    /// A legacy `initial_conditions.json` still carrying a `past_inflows` array
    /// is rejected by `deny_unknown_fields` rather than silently dropped.
    #[test]
    fn test_legacy_past_inflows_field_is_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 0, "values_m3s": [600.0, 500.0] }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("past_inflows"),
                    "message should name the unknown 'past_inflows' field, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    // ── AC: retired future_anticipated_deliveries field → rejected ────────────

    /// A stale `initial_conditions.json` still declaring a
    /// `future_anticipated_deliveries` array is rejected by `deny_unknown_fields`
    /// rather than silently dropped — the field's retirement is a breaking
    /// input-contract change, and the unknown-field message routes to
    /// `ParseError` (via serde), not `SchemaError`.
    #[test]
    fn test_retired_future_anticipated_deliveries_field_is_rejected() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "future_anticipated_deliveries": [
            { "thermal_id": 86, "delivery_start": "2026-09-05", "delivery_end": "2026-09-12", "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("future_anticipated_deliveries"),
                    "message should name the unknown 'future_anticipated_deliveries' field, \
                     got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }
}
