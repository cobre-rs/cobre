//! Parsing for `scenarios/load_factors.json` — per-bus-per-stage block-level
//! load scaling factors.
//!
//! [`parse_load_factors`] reads `scenarios/load_factors.json` and returns a
//! sorted `Vec<LoadFactorEntry>`. When the file is absent, the optional
//! `load_load_factors` wrapper in `scenarios/mod.rs` returns `Ok(Vec::new())`.
//!
//! ## JSON structure (spec SS4)
//!
//! ```json
//! {
//!   "$schema": "...",
//!   "load_factors": [
//!     {
//!       "bus_id": 0,
//!       "stage_id": 0,
//!       "block_factors": [
//!         { "block_id": 0, "factor": 0.85 },
//!         { "block_id": 1, "factor": 1.0 },
//!         { "block_id": 2, "factor": 1.15 }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! ## Output ordering
//!
//! Entries are sorted by `(bus_id, stage_id)` ascending. The `block_factors`
//! within each entry are sorted by `block_id` ascending.
//!
//! ## Validation
//!
//! After deserializing, the following constraints are checked before conversion:
//!
//! - No two entries share the same `(bus_id, stage_id)` pair (no duplicates).
//! - Every `factor` value must be strictly positive (> 0.0) and finite.
//!
//! Deferred validations (not performed here):
//!
//! - `bus_id` existence in the bus registry — Layer 3.
//! - `stage_id` existence in the stages registry — Layer 3.
//! - `block_id` contiguity and count matching the stage block count — Layer 3/5.

use cobre_core::EntityId;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `load_factors.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawLoadFactorsFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of load factor entries.
    load_factors: Vec<RawLoadFactorEntry>,
}

/// Intermediate type for a single load factor entry (one bus-stage pair).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct RawLoadFactorEntry {
    /// Bus identifier.
    bus_id: i32,
    /// Stage identifier.
    stage_id: i32,
    /// Per-block scaling factors for this bus-stage pair.
    block_factors: Vec<RawBlockFactor>,
}

/// Intermediate type for a single block factor.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct RawBlockFactor {
    /// Block identifier.
    block_id: i32,
    /// Scaling factor. Must be strictly positive (> 0.0) and finite.
    factor: f64,
}

// ── Public types ──────────────────────────────────────────────────────────────

/// A single block factor entry from `scenarios/load_factors.json`.
///
/// Pairs a block ID with its scaling factor.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::BlockFactor;
///
/// let bf = BlockFactor { block_id: 0, factor: 0.85 };
/// assert_eq!(bf.block_id, 0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BlockFactor {
    /// Block identifier (0-based).
    pub block_id: i32,
    /// Scaling factor applied to the stochastic load realization. Must be > 0.0.
    pub factor: f64,
}

/// A single load factor entry from `scenarios/load_factors.json`.
///
/// One entry per `(bus_id, stage_id)` pair, containing the per-block scaling
/// factors for load demand in that bus-stage combination.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::{BlockFactor, LoadFactorEntry};
/// use cobre_core::EntityId;
///
/// let entry = LoadFactorEntry {
///     bus_id: EntityId::from(0),
///     stage_id: 0,
///     block_factors: vec![
///         BlockFactor { block_id: 0, factor: 0.85 },
///         BlockFactor { block_id: 1, factor: 1.0 },
///         BlockFactor { block_id: 2, factor: 1.15 },
///     ],
/// };
/// assert_eq!(entry.bus_id, EntityId::from(0));
/// assert_eq!(entry.block_factors.len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LoadFactorEntry {
    /// Bus this entry applies to.
    pub bus_id: EntityId,
    /// Stage this entry applies to.
    pub stage_id: i32,
    /// Per-block scaling factors sorted by `block_id` ascending.
    pub block_factors: Vec<BlockFactor>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse `scenarios/load_factors.json` and return a sorted entry list.
///
/// Reads the JSON file at `path`, deserializes it through intermediate types,
/// validates per-entry and per-block constraints, then returns all entries
/// sorted by `(bus_id, stage_id)` ascending. The `block_factors` within each
/// entry are sorted by `block_id` ascending.
///
/// # Errors
///
/// | Condition                                             | Error variant              |
/// | ----------------------------------------------------- | -------------------------- |
/// | File not found / read failure                         | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field         | [`LoadError::ParseError`]  |
/// | `factor` not finite or `<= 0.0`                       | [`LoadError::SchemaError`] |
/// | Duplicate `(bus_id, stage_id)` pair                   | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_load_factors;
/// use std::path::Path;
///
/// let entries = parse_load_factors(Path::new("scenarios/load_factors.json"))
///     .expect("valid load factors file");
/// println!("loaded {} load factor entries", entries.len());
/// ```
pub fn parse_load_factors(path: &Path) -> Result<Vec<LoadFactorEntry>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawLoadFactorsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    Ok(convert(raw))
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized data.
///
/// Called before conversion so that error messages can reference JSON field
/// paths rather than Rust field names.
fn validate_raw(raw: &RawLoadFactorsFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_entries(&raw.load_factors, path)?;
    for (i, entry) in raw.load_factors.iter().enumerate() {
        validate_block_factors(&entry.block_factors, i, path)?;
    }
    Ok(())
}

/// Check that no two entries share the same `(bus_id, stage_id)` pair.
fn validate_no_duplicate_entries(
    entries: &[RawLoadFactorEntry],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<(i32, i32)> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        let key = (entry.bus_id, entry.stage_id);
        if !seen.insert(key) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("load_factors[{i}]"),
                message: format!(
                    "duplicate (bus_id={}, stage_id={}) in load_factors",
                    entry.bus_id, entry.stage_id
                ),
            });
        }
    }
    Ok(())
}

/// Validate all block factors for a single entry.
///
/// Each `factor` must be strictly positive (> 0.0) and finite.
fn validate_block_factors(
    block_factors: &[RawBlockFactor],
    entry_idx: usize,
    path: &Path,
) -> Result<(), LoadError> {
    for (j, bf) in block_factors.iter().enumerate() {
        if !bf.factor.is_finite() || bf.factor <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("load_factors[{entry_idx}].block_factors[{j}].factor"),
                message: format!("factor must be finite and > 0.0, got {}", bf.factor),
            });
        }
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert validated raw data into `Vec<LoadFactorEntry>`.
///
/// Precondition: [`validate_raw`] has returned `Ok(())` for this data.
/// Entries are sorted by `(bus_id, stage_id)` ascending. Block factors within
/// each entry are sorted by `block_id` ascending.
fn convert(raw: RawLoadFactorsFile) -> Vec<LoadFactorEntry> {
    let mut entries: Vec<LoadFactorEntry> = raw
        .load_factors
        .into_iter()
        .map(|e| {
            let mut block_factors: Vec<BlockFactor> = e
                .block_factors
                .into_iter()
                .map(|bf| BlockFactor {
                    block_id: bf.block_id,
                    factor: bf.factor,
                })
                .collect();
            block_factors.sort_by_key(|bf| bf.block_id);
            LoadFactorEntry {
                bus_id: EntityId::from(e.bus_id),
                stage_id: e.stage_id,
                block_factors,
            }
        })
        .collect();

    entries.sort_by(|a, b| {
        a.bus_id
            .0
            .cmp(&b.bus_id.0)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    entries
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Write a JSON string to a temp file and return the handle (keeps it alive).
    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    /// Canonical valid JSON for error-path tests (2 entries, 3 block factors each).
    const VALID_JSON: &str = r#"{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.85 },
        { "block_id": 1, "factor": 1.0 },
        { "block_id": 2, "factor": 1.15 }
      ]
    },
    {
      "bus_id": 1,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.90 },
        { "block_id": 1, "factor": 1.05 },
        { "block_id": 2, "factor": 1.20 }
      ]
    }
  ]
}"#;

    // ── AC: valid file with 2 entries, 3 block factors each ───────────────────

    /// Valid 2-entry file. Bus 0 and bus 1, both stage 0. Result sorted by bus_id.
    /// Block factors preserved in block_id order.
    #[test]
    fn test_valid_2_entries_sorted_and_block_factors_correct() {
        let tmp = write_json(VALID_JSON);
        let entries = parse_load_factors(tmp.path()).unwrap();

        assert_eq!(entries.len(), 2);

        assert_eq!(entries[0].bus_id, EntityId::from(0));
        assert_eq!(entries[0].stage_id, 0);
        assert_eq!(entries[0].block_factors.len(), 3);
        assert_eq!(entries[0].block_factors[0].block_id, 0);
        assert!((entries[0].block_factors[0].factor - 0.85).abs() < 1e-10);
        assert_eq!(entries[0].block_factors[1].block_id, 1);
        assert!((entries[0].block_factors[1].factor - 1.0).abs() < f64::EPSILON);
        assert_eq!(entries[0].block_factors[2].block_id, 2);
        assert!((entries[0].block_factors[2].factor - 1.15).abs() < 1e-10);
        assert_eq!(entries[1].bus_id, EntityId::from(1));
        assert_eq!(entries[1].stage_id, 0);
        assert_eq!(entries[1].block_factors.len(), 3);
    }

    // ── AC: sort order for entries out of order ───────────────────────────────

    /// Entries out of sort order — result is sorted by (bus_id, stage_id).
    #[test]
    fn test_entries_sorted_by_bus_stage() {
        let json = r#"{
  "load_factors": [
    {
      "bus_id": 1,
      "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1.0 }]
    },
    {
      "bus_id": 0,
      "stage_id": 1,
      "block_factors": [{ "block_id": 0, "factor": 0.9 }]
    },
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 0.8 }]
    }
  ]
}"#;
        let tmp = write_json(json);
        let entries = parse_load_factors(tmp.path()).unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].bus_id, EntityId::from(0));
        assert_eq!(entries[0].stage_id, 0);
        assert_eq!(entries[1].bus_id, EntityId::from(0));
        assert_eq!(entries[1].stage_id, 1);
        assert_eq!(entries[2].bus_id, EntityId::from(1));
        assert_eq!(entries[2].stage_id, 0);
    }

    // ── AC: block_factors sorted by block_id ─────────────────────────────────

    /// Block factors out of block_id order — result is sorted by block_id.
    #[test]
    fn test_block_factors_sorted_by_block_id() {
        let json = r#"{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 2, "factor": 1.15 },
        { "block_id": 0, "factor": 0.85 },
        { "block_id": 1, "factor": 1.0 }
      ]
    }
  ]
}"#;
        let tmp = write_json(json);
        let entries = parse_load_factors(tmp.path()).unwrap();

        assert_eq!(entries.len(), 1);
        let bfs = &entries[0].block_factors;
        assert_eq!(bfs[0].block_id, 0);
        assert_eq!(bfs[1].block_id, 1);
        assert_eq!(bfs[2].block_id, 2);
    }

    // ── AC: factor = 0.0 -> SchemaError ──────────────────────────────────────

    /// `factor = 0.0` is rejected (must be strictly positive).
    #[test]
    fn test_zero_factor_rejected() {
        let json = r#"{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 0.0 }]
    }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_load_factors(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("factor"),
                    "field should contain 'factor', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: factor negative -> SchemaError ───────────────────────────────────

    /// `factor = -0.5` is rejected (must be > 0.0).
    #[test]
    fn test_negative_factor_rejected() {
        let json = r#"{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": -0.5 }]
    }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_load_factors(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("factor"),
                    "field should contain 'factor', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: duplicate (bus_id, stage_id) -> SchemaError ──────────────────────

    /// Two entries with the same `(bus_id=0, stage_id=0)` -> SchemaError with
    /// field containing `"load_factors[1]"` and message containing `"duplicate"`.
    #[test]
    fn test_duplicate_bus_stage_rejected() {
        let json = r#"{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1.0 }]
    },
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1.1 }]
    }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_load_factors(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("load_factors[1]"),
                    "field should contain 'load_factors[1]', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: empty load_factors array -> Ok(vec![]) ────────────────────────────

    /// Empty `load_factors` array -> Ok(Vec::new()).
    #[test]
    fn test_empty_load_factors_returns_empty_vec() {
        let json = r#"{ "load_factors": [] }"#;
        let tmp = write_json(json);
        let entries = parse_load_factors(tmp.path()).unwrap();
        assert!(entries.is_empty());
    }

    // ── AC: missing block_factors field -> ParseError ─────────────────────────

    /// Entry missing `block_factors` field -> ParseError (required by serde).
    #[test]
    fn test_missing_block_factors_field_is_parse_error() {
        let json = r#"{
  "load_factors": [
    { "bus_id": 0, "stage_id": 0 }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_load_factors(tmp.path()).unwrap_err();

        match &err {
            LoadError::ParseError { .. } => {
                // Expected — serde missing field produces ParseError.
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    // ── AC: factor very small positive -> accepted ────────────────────────────

    /// A very small positive factor (just above 0.0) is accepted.
    #[test]
    fn test_very_small_positive_factor_accepted() {
        let json = r#"{
  "load_factors": [
    {
      "bus_id": 0,
      "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1e-300 }]
    }
  ]
}"#;
        let tmp = write_json(json);
        let entries = parse_load_factors(tmp.path()).unwrap();

        assert_eq!(entries.len(), 1);
        assert!(entries[0].block_factors[0].factor > 0.0);
    }

    // ── AC: declaration-order invariance ─────────────────────────────────────

    /// Reordering entries in the JSON array does not change the output ordering.
    #[test]
    fn test_declaration_order_invariance() {
        let json_fwd = r#"{
  "load_factors": [
    {
      "bus_id": 0, "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1.0 }]
    },
    {
      "bus_id": 1, "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1.1 }]
    }
  ]
}"#;
        let json_rev = r#"{
  "load_factors": [
    {
      "bus_id": 1, "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1.1 }]
    },
    {
      "bus_id": 0, "stage_id": 0,
      "block_factors": [{ "block_id": 0, "factor": 1.0 }]
    }
  ]
}"#;
        let tmp_fwd = write_json(json_fwd);
        let tmp_rev = write_json(json_rev);
        let entries_fwd = parse_load_factors(tmp_fwd.path()).unwrap();
        let entries_rev = parse_load_factors(tmp_rev.path()).unwrap();

        let keys_fwd: Vec<(i32, i32)> = entries_fwd
            .iter()
            .map(|e| (e.bus_id.0, e.stage_id))
            .collect();
        let keys_rev: Vec<(i32, i32)> = entries_rev
            .iter()
            .map(|e| (e.bus_id.0, e.stage_id))
            .collect();
        assert_eq!(keys_fwd, keys_rev);
    }
}
