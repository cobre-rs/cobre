//! Parsing for `initial_conditions.json` — initial system state.
//!
//! [`parse_initial_conditions`] reads `initial_conditions.json` from the case
//! directory root and returns a fully-validated [`InitialConditions`].
//!
//! ## JSON structure
//!
//! The file contains two required top-level arrays and two optional arrays:
//!
//! - `storage` — initial reservoir volumes for operating hydros (hm³).
//! - `filling_storage` — initial reservoir volumes for filling hydros (hm³).
//! - `past_inflows` — past inflow values for PAR(p) lag initialization (m³/s),
//!   ordered from most recent (lag 1) to oldest (lag p). Optional; defaults to
//!   an empty array when absent.
//! - `recent_observations` — observed inflow data for partial periods before
//!   the study start (m³/s per date range per hydro). Optional; defaults to an
//!   empty array when absent.
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/initial_conditions.schema.json",
//!   "storage": [
//!     { "hydro_id": 0, "value_hm3": 15000.0 },
//!     { "hydro_id": 1, "value_hm3": 8500.0 }
//!   ],
//!   "filling_storage": [{ "hydro_id": 10, "value_hm3": 200.0 }],
//!   "past_inflows": [
//!     { "hydro_id": 0, "values_m3s": [600.0, 500.0] },
//!     { "hydro_id": 1, "values_m3s": [200.0, 100.0] }
//!   ],
//!   "recent_observations": [
//!     { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": 500.0 },
//!     { "hydro_id": 0, "start_date": "2026-04-04", "end_date": "2026-04-11", "value_m3s": 480.0 }
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
//! 4. No `hydro_id` appears more than once in `past_inflows`.
//! 5. Every value in `past_inflows[i].values_m3s` is finite and non-negative.
//! 6. Every `start_date` and `end_date` in `recent_observations` parses as
//!    ISO 8601 (`YYYY-MM-DD`), and `end_date > start_date`.
//! 7. Every `value_m3s` in `recent_observations` is finite and non-negative.
//! 8. For observations with the same `hydro_id`, date ranges do not overlap
//!    (adjacent ranges where `start == prev_end` are accepted).
//!
//! Cross-reference validation (checking that hydro IDs exist in the hydro
//! registry) is deferred to Layer 3 (deferred). Storage bounds validation
//! (value within `[min_storage_hm3, max_storage_hm3]`) also requires the
//! hydro registry and is likewise deferred.

use chrono::NaiveDate;
use cobre_core::{EntityId, HydroPastInflows, HydroStorage, InitialConditions, RecentObservation};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Initial reservoir storage conditions, past inflow values, and recent
/// observations for all hydro plants in the case.
///
/// Two arrays specify starting volumes at simulation time zero:
/// - `storage` — operating hydros (those participating in generation dispatch).
/// - `filling_storage` — filling hydros (reservoirs under construction or filling).
///
/// An optional array provides past inflow values for PAR(p) lag initialization:
/// - `past_inflows` — ordered from most recent (lag 1) to oldest (lag p).
///
/// An optional array provides observed inflow data for mid-season study starts:
/// - `recent_observations` — date-ranged observations that seed the lag accumulator.
///
/// A hydro may appear in at most one of the two storage arrays. Duplicate
/// `hydro_id` values within the same array are rejected. Cross-reference
/// validation (checking that IDs exist in the hydro registry) is deferred to
/// a later validation layer.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
pub(crate) struct RawInitialConditions {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Initial reservoir volumes for operating hydros [hm³].
    storage: Vec<RawHydroStorage>,

    /// Initial reservoir volumes for filling hydros [hm³].
    /// A filling hydro may not also appear in `storage`.
    filling_storage: Vec<RawHydroStorage>,

    /// Past inflow values for PAR(p) lag initialization [m³/s], one entry per
    /// hydro. For each hydro, `values_m3s[0]` is the most recent past inflow
    /// (lag 1) and `values_m3s[p-1]` is the oldest (lag p). Required when
    /// `inflow_lags` is enabled and the PAR order is > 0. Optional; defaults
    /// to empty.
    #[serde(default)]
    past_inflows: Vec<RawHydroPastInflows>,

    /// Observed inflow data for partial periods before the study start
    /// [m³/s per date range per hydro]. Used to seed the lag accumulator when
    /// a study begins mid-season. Date ranges for the same hydro must not
    /// overlap; adjacent ranges (start == previous end) are accepted.
    /// Optional; defaults to empty.
    #[serde(default)]
    recent_observations: Vec<RawRecentObservation>,
}

/// Initial reservoir volume for one hydro plant, in hm³.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawHydroStorage {
    /// Hydro plant identifier. Must be unique within its array.
    hydro_id: i32,
    /// Reservoir volume [hm³]. Must be >= 0.0.
    value_hm3: f64,
}

/// Past inflow values for PAR(p) lag initialization for one hydro plant.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawHydroPastInflows {
    /// Hydro plant identifier. Must be unique within `past_inflows`.
    hydro_id: i32,
    /// Past inflow values [m³/s], ordered from most recent (lag 1, index 0) to
    /// oldest (lag p, index p-1). Must have length >= the hydro's PAR order.
    values_m3s: Vec<f64>,
    /// Optional season IDs corresponding to each lag entry in `values_m3s`,
    /// one per entry. When present, length must equal `values_m3s.length`.
    /// Each value must reference a season ID defined in `season_definitions`.
    /// Absent from legacy JSON files (backward compatible).
    #[serde(default)]
    season_ids: Option<Vec<u32>>,
}

/// Observed inflow for a single hydro plant over a specific date range.
///
/// Used to seed the lag accumulator when a study begins mid-season (before the
/// first lag-period boundary). Each entry covers one hydro over one
/// observation period. Multiple entries per hydro are allowed for rolling
/// revisions.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawRecentObservation {
    /// Hydro plant identifier. Must reference a hydro entity in the system.
    hydro_id: i32,
    /// Start of the observation period (inclusive), as an ISO 8601 date
    /// (YYYY-MM-DD).
    start_date: String,
    /// End of the observation period (exclusive), as an ISO 8601 date
    /// (YYYY-MM-DD). Must be after `start_date`.
    end_date: String,
    /// Average inflow observed during the period [m³/s]. Must be finite and
    /// non-negative.
    value_m3s: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `initial_conditions.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types, then
/// performs post-deserialization validation before converting to
/// [`InitialConditions`].
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
/// | Duplicate `hydro_id` within `past_inflows`            | [`LoadError::SchemaError`] |
/// | Non-finite or negative value in `past_inflows`        | [`LoadError::SchemaError`] |
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
    validate_past_inflows_no_duplicates(&raw.past_inflows, path)?;
    validate_past_inflows_values(&raw.past_inflows, path)?;
    validate_past_inflows_season_ids(&raw.past_inflows, path)?;
    validate_recent_observations_dates(&raw.recent_observations, path)?;
    validate_recent_observations_values(&raw.recent_observations, path)?;
    validate_recent_observations_no_overlap(&raw.recent_observations, path)?;
    Ok(())
}

/// Check that all `value_hm3` entries in an array are non-negative.
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

/// Check that no `hydro_id` appears more than once within an array.
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

/// Check that no `hydro_id` appears in both `storage` and `filling_storage`.
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

/// Check that no `hydro_id` appears more than once in `past_inflows`.
fn validate_past_inflows_no_duplicates(
    entries: &[RawHydroPastInflows],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        if !seen.insert(entry.hydro_id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("past_inflows[{i}].hydro_id"),
                message: format!("duplicate hydro_id {} in past_inflows", entry.hydro_id),
            });
        }
    }
    Ok(())
}

/// Check that every value in `past_inflows[i].values_m3s` is finite and non-negative.
fn validate_past_inflows_values(
    entries: &[RawHydroPastInflows],
    path: &Path,
) -> Result<(), LoadError> {
    for (i, entry) in entries.iter().enumerate() {
        for (j, &v) in entry.values_m3s.iter().enumerate() {
            if !v.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("past_inflows[{i}].values_m3s[{j}]"),
                    message: format!(
                        "past_inflows[{i}].values_m3s[{j}] is not finite (got {v}); \
                         all inflow values must be finite numbers"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Check that when `season_ids` is present for a `past_inflows` entry, its
/// length equals `values_m3s.len()`.
fn validate_past_inflows_season_ids(
    entries: &[RawHydroPastInflows],
    path: &Path,
) -> Result<(), LoadError> {
    for (i, entry) in entries.iter().enumerate() {
        if let Some(season_ids) = &entry.season_ids {
            if season_ids.len() != entry.values_m3s.len() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("past_inflows[{i}].season_ids"),
                    message: format!(
                        "past_inflows[{i}].season_ids has {} element(s) but \
                         past_inflows[{i}].values_m3s has {} element(s); \
                         season_ids length must equal values_m3s length",
                        season_ids.len(),
                        entry.values_m3s.len()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Check that every `start_date` and `end_date` in `recent_observations` is a
/// valid ISO 8601 date (`YYYY-MM-DD`) and that `end_date > start_date`.
fn validate_recent_observations_dates(
    entries: &[RawRecentObservation],
    path: &Path,
) -> Result<(), LoadError> {
    for (i, entry) in entries.iter().enumerate() {
        let start = NaiveDate::parse_from_str(&entry.start_date, "%Y-%m-%d").map_err(|_| {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("recent_observations[{i}].start_date"),
                message: format!(
                    "recent_observations[{i}].start_date '{}' is not a valid ISO 8601 date \
                     (expected YYYY-MM-DD)",
                    entry.start_date
                ),
            }
        })?;
        let end = NaiveDate::parse_from_str(&entry.end_date, "%Y-%m-%d").map_err(|_| {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("recent_observations[{i}].end_date"),
                message: format!(
                    "recent_observations[{i}].end_date '{}' is not a valid ISO 8601 date \
                     (expected YYYY-MM-DD)",
                    entry.end_date
                ),
            }
        })?;
        if end <= start {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("recent_observations[{i}].end_date"),
                message: format!(
                    "recent_observations[{i}]: end_date must be after start_date \
                     (start_date={}, end_date={})",
                    entry.start_date, entry.end_date
                ),
            });
        }
    }
    Ok(())
}

/// Check that every `value_m3s` in `recent_observations` is finite and non-negative.
fn validate_recent_observations_values(
    entries: &[RawRecentObservation],
    path: &Path,
) -> Result<(), LoadError> {
    for (i, entry) in entries.iter().enumerate() {
        if !entry.value_m3s.is_finite() || entry.value_m3s < 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("recent_observations[{i}].value_m3s"),
                message: format!(
                    "recent_observations[{i}].value_m3s must be a finite non-negative number, \
                     got {}",
                    entry.value_m3s
                ),
            });
        }
    }
    Ok(())
}

/// Check that for observations with the same `hydro_id`, date ranges do not
/// overlap. Adjacent ranges where `start_date == previous end_date` are
/// accepted (exclusive-end convention).
///
/// Precondition: [`validate_recent_observations_dates`] has returned `Ok(())`
/// for these entries (dates are valid and `end > start`).
fn validate_recent_observations_no_overlap(
    entries: &[RawRecentObservation],
    path: &Path,
) -> Result<(), LoadError> {
    // Group entry indices by hydro_id, then sort each group by start_date and
    // check consecutive pairs.
    use std::collections::HashMap;

    let mut by_hydro: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, entry) in entries.iter().enumerate() {
        by_hydro.entry(entry.hydro_id).or_default().push(i);
    }

    for (hydro_id, mut indices) in by_hydro {
        // Sort indices by start_date (dates already validated as parseable).
        indices.sort_by_key(|&i| {
            NaiveDate::parse_from_str(&entries[i].start_date, "%Y-%m-%d")
                .unwrap_or_else(|_| unreachable!("start_date already validated"))
        });

        for window in indices.windows(2) {
            let (i_prev, i_curr) = (window[0], window[1]);
            let prev_end = NaiveDate::parse_from_str(&entries[i_prev].end_date, "%Y-%m-%d")
                .unwrap_or_else(|_| unreachable!("end_date already validated"));
            let curr_start = NaiveDate::parse_from_str(&entries[i_curr].start_date, "%Y-%m-%d")
                .unwrap_or_else(|_| unreachable!("start_date already validated"));

            if curr_start < prev_end {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("recent_observations[{i_curr}].start_date"),
                    message: format!(
                        "recent_observations: overlapping date ranges for hydro_id {hydro_id}: \
                         entry [{i_prev}] ends on {prev_end} but entry [{i_curr}] starts on \
                         {curr_start}"
                    ),
                });
            }
        }
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert validated raw data into [`InitialConditions`].
///
/// Precondition: [`validate_raw`] has returned `Ok(())` for this data.
/// All arrays are sorted by `hydro_id` to satisfy the declaration-order
/// invariance requirement.
fn convert(raw: RawInitialConditions) -> InitialConditions {
    let mut storage: Vec<HydroStorage> = raw
        .storage
        .into_iter()
        .map(|e| HydroStorage {
            hydro_id: EntityId(e.hydro_id),
            value_hm3: e.value_hm3,
        })
        .collect();
    storage.sort_by_key(|e| e.hydro_id.0);

    let mut filling_storage: Vec<HydroStorage> = raw
        .filling_storage
        .into_iter()
        .map(|e| HydroStorage {
            hydro_id: EntityId(e.hydro_id),
            value_hm3: e.value_hm3,
        })
        .collect();
    filling_storage.sort_by_key(|e| e.hydro_id.0);

    let mut past_inflows: Vec<HydroPastInflows> = raw
        .past_inflows
        .into_iter()
        .map(|e| HydroPastInflows {
            hydro_id: EntityId(e.hydro_id),
            values_m3s: e.values_m3s,
            season_ids: e.season_ids,
        })
        .collect();
    past_inflows.sort_by_key(|e| e.hydro_id.0);

    let mut recent_observations: Vec<RecentObservation> = raw
        .recent_observations
        .into_iter()
        .map(|e| RecentObservation {
            hydro_id: EntityId(e.hydro_id),
            // SAFETY: dates already validated by validate_recent_observations_dates
            start_date: NaiveDate::parse_from_str(&e.start_date, "%Y-%m-%d")
                .unwrap_or_else(|_| unreachable!("start_date already validated")),
            end_date: NaiveDate::parse_from_str(&e.end_date, "%Y-%m-%d")
                .unwrap_or_else(|_| unreachable!("end_date already validated")),
            value_m3s: e.value_m3s,
        })
        .collect();
    recent_observations.sort_by_key(|e| (e.hydro_id.0, e.start_date));

    InitialConditions {
        storage,
        filling_storage,
        past_inflows,
        recent_observations,
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

    /// Canonical valid `initial_conditions.json` with 2 storage and 1 filling entry.
    const VALID_JSON: &str = r#"{
      "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/initial_conditions.schema.json",
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
        assert!(
            ic.past_inflows.is_empty(),
            "past_inflows absent defaults to empty"
        );

        // Both storage entries present with correct IDs and values
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

        // Filling storage entry present
        assert_eq!(ic.filling_storage[0].hydro_id, EntityId(10));
        assert!(
            (ic.filling_storage[0].value_hm3 - 200.0).abs() < f64::EPSILON,
            "expected 200.0, got {}",
            ic.filling_storage[0].value_hm3
        );
    }

    /// Given a valid `initial_conditions.json` with `past_inflows`, the values
    /// are parsed correctly and sorted by `hydro_id`.
    #[test]
    fn test_parse_valid_past_inflows() {
        let json = r#"{
          "storage": [
            { "hydro_id": 0, "value_hm3": 1000.0 },
            { "hydro_id": 1, "value_hm3": 2000.0 }
          ],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 1, "values_m3s": [200.0, 100.0] },
            { "hydro_id": 0, "values_m3s": [600.0, 500.0] }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();

        assert_eq!(ic.past_inflows.len(), 2);
        // Sorted by hydro_id ascending
        assert_eq!(ic.past_inflows[0].hydro_id, EntityId(0));
        assert_eq!(ic.past_inflows[0].values_m3s, vec![600.0, 500.0]);
        assert_eq!(ic.past_inflows[1].hydro_id, EntityId(1));
        assert_eq!(ic.past_inflows[1].values_m3s, vec![200.0, 100.0]);
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
        assert!(ic.past_inflows.is_empty());
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

    // ── AC: past_inflows duplicate hydro_id → SchemaError ─────────────────────

    /// Given `past_inflows` with a duplicate `hydro_id`, `parse_initial_conditions`
    /// returns `Err(LoadError::SchemaError)` mentioning "duplicate" and "`past_inflows`".
    #[test]
    fn test_duplicate_hydro_id_in_past_inflows() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 3, "values_m3s": [100.0] },
            { "hydro_id": 3, "values_m3s": [200.0] }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("past_inflows"),
                    "field should mention 'past_inflows', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should mention 'duplicate', got: {message}"
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
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 0, "values_m3s": [600.0, 500.0] },
            { "hydro_id": 1, "values_m3s": [200.0, 100.0] }
          ]
        }"#;
        let json_reversed = r#"{
          "storage": [
            { "hydro_id": 1, "value_hm3": 2000.0 },
            { "hydro_id": 0, "value_hm3": 1000.0 }
          ],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 1, "values_m3s": [200.0, 100.0] },
            { "hydro_id": 0, "values_m3s": [600.0, 500.0] }
          ]
        }"#;

        let f1 = write_json(json_ordered);
        let f2 = write_json(json_reversed);
        let ic1 = parse_initial_conditions(f1.path()).unwrap();
        let ic2 = parse_initial_conditions(f2.path()).unwrap();

        assert_eq!(
            ic1, ic2,
            "results must be identical regardless of input ordering"
        );
        // Sorted by hydro_id ascending
        assert_eq!(ic1.storage[0].hydro_id, EntityId(0));
        assert_eq!(ic1.storage[1].hydro_id, EntityId(1));
        assert_eq!(ic1.past_inflows[0].hydro_id, EntityId(0));
        assert_eq!(ic1.past_inflows[1].hydro_id, EntityId(1));
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

    /// Zero value in `past_inflows.values_m3s` is valid (dry season).
    #[test]
    fn test_zero_past_inflow_value_is_valid() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 1, "values_m3s": [0.0, 50.0] }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_initial_conditions(f.path());
        assert!(
            result.is_ok(),
            "0.0 in past_inflows is valid (dry season), got: {result:?}"
        );
    }

    /// Empty `values_m3s` array is accepted — no lag initialization needed.
    #[test]
    fn test_empty_values_m3s_is_valid() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 1, "values_m3s": [] }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_initial_conditions(f.path());
        assert!(
            result.is_ok(),
            "empty values_m3s should be accepted, got: {result:?}"
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
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: negative value_m3s → SchemaError ─────────────────────────────────

    /// Given a `recent_observations` entry with `value_m3s: -1.0`,
    /// `parse_initial_conditions` returns `Err(LoadError::SchemaError)` with
    /// field containing `"value_m3s"`.
    #[test]
    fn test_recent_observations_negative_value() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "recent_observations": [
            { "hydro_id": 0, "start_date": "2026-04-01", "end_date": "2026-04-04", "value_m3s": -1.0 }
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
        // Sorted by (hydro_id, start_date): hydro 0 first with earlier start first
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

    // ── AC: past_inflows season_ids ───────────────────────────────────────────

    /// Given a `past_inflows` entry with matching `season_ids` and `values_m3s`
    /// lengths, `parse_initial_conditions` returns `Ok(ic)` with the `season_ids`
    /// preserved.
    #[test]
    fn test_parse_past_inflows_with_valid_season_ids() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 0, "values_m3s": [600.0, 500.0], "season_ids": [3, 2] }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert_eq!(ic.past_inflows.len(), 1);
        assert_eq!(ic.past_inflows[0].hydro_id, EntityId(0));
        assert_eq!(ic.past_inflows[0].values_m3s, vec![600.0, 500.0]);
        assert_eq!(ic.past_inflows[0].season_ids, Some(vec![3, 2]));
    }

    /// Given a `past_inflows` entry where `season_ids` has length 3 but
    /// `values_m3s` has length 2, `parse_initial_conditions` returns
    /// `Err(LoadError::SchemaError)` with field containing `season_ids`.
    #[test]
    fn test_parse_past_inflows_season_ids_length_mismatch() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 0, "values_m3s": [600.0, 500.0], "season_ids": [3, 2, 1] }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_initial_conditions(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("season_ids"),
                    "field should contain 'season_ids', got: {field}"
                );
                assert!(
                    field.contains("past_inflows[0]"),
                    "field should reference 'past_inflows[0]', got: {field}"
                );
                assert!(
                    message.contains("season_ids length must equal values_m3s length")
                        || message.contains('3')
                        || message.contains('2'),
                    "message should describe the mismatch, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given a `past_inflows` entry without a `season_ids` key (legacy JSON),
    /// `parse_initial_conditions` returns `Ok(ic)` with `season_ids == None`.
    #[test]
    fn test_parse_past_inflows_without_season_ids_backward_compat() {
        let json = r#"{
          "storage": [],
          "filling_storage": [],
          "past_inflows": [
            { "hydro_id": 0, "values_m3s": [600.0, 500.0] }
          ]
        }"#;
        let f = write_json(json);
        let ic = parse_initial_conditions(f.path()).unwrap();
        assert_eq!(ic.past_inflows.len(), 1);
        assert_eq!(
            ic.past_inflows[0].season_ids, None,
            "absent season_ids must deserialize as None"
        );
    }
}
