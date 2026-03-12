//! Parsing for `initial_conditions.json` — initial system state.
//!
//! [`parse_initial_conditions`] reads `initial_conditions.json` from the case
//! directory root and returns a fully-validated [`InitialConditions`].
//!
//! ## JSON structure
//!
//! The file contains two top-level arrays:
//!
//! - `storage` — initial reservoir volumes for operating hydros (hm³).
//! - `filling_storage` — initial reservoir volumes for filling hydros (hm³).
//!
//! ```json
//! {
//!   "$schema": "https://cobre-rs.github.io/cobre/schemas/initial_conditions.schema.json",
//!   "storage": [
//!     { "hydro_id": 0, "value_hm3": 15000.0 },
//!     { "hydro_id": 1, "value_hm3": 8500.0 }
//!   ],
//!   "filling_storage": [{ "hydro_id": 10, "value_hm3": 200.0 }]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, three invariants are checked before conversion:
//!
//! 1. Every `value_hm3` is non-negative (`>= 0.0`).
//! 2. No `hydro_id` appears more than once within `storage` or within
//!    `filling_storage` (no intra-array duplicates).
//! 3. No `hydro_id` appears in both `storage` and `filling_storage`
//!    (mutual exclusion).
//!
//! Cross-reference validation (checking that hydro IDs exist in the hydro
//! registry) is deferred to Layer 3 (Epic 06). Storage bounds validation
//! (value within `[min_storage_hm3, max_storage_hm3]`) also requires the
//! hydro registry and is likewise deferred.

use cobre_core::{EntityId, HydroStorage, InitialConditions};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `initial_conditions.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
struct RawInitialConditions {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Initial storage for operating hydros.
    storage: Vec<RawHydroStorage>,

    /// Initial storage for filling hydros.
    filling_storage: Vec<RawHydroStorage>,
}

/// Intermediate type for one hydro storage entry.
#[derive(Deserialize)]
struct RawHydroStorage {
    /// Hydro plant identifier.
    hydro_id: i32,
    /// Reservoir volume in hm³.
    value_hm3: f64,
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

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert validated raw data into [`InitialConditions`].
///
/// Precondition: [`validate_raw`] has returned `Ok(())` for this data.
/// Both arrays are sorted by `hydro_id` to satisfy the declaration-order
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

    InitialConditions {
        storage,
        filling_storage,
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
      "$schema": "https://cobre-rs.github.io/cobre/schemas/initial_conditions.schema.json",
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

    /// Filling storage value below dead volume is valid (ticket note: that is the
    /// point of filling hydros). Only non-negativity is checked here; bounds
    /// against dead volume are deferred to Layer 3.
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
        // Sorted by hydro_id ascending
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
}
