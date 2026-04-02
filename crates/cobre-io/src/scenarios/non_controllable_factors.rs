//! Parsing for `scenarios/non_controllable_factors.json` — per-NCS-per-stage
//! block-level generation scaling factors.
//!
//! [`parse_non_controllable_factors`] reads `scenarios/non_controllable_factors.json`
//! and returns a sorted `Vec<NcsFactorEntry>`. When the file is absent, the optional
//! `load_non_controllable_factors` wrapper in `scenarios/mod.rs` returns `Ok(Vec::new())`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "...",
//!   "non_controllable_factors": [
//!     {
//!       "ncs_id": 0,
//!       "stage_id": 0,
//!       "block_factors": [
//!         { "block_id": 0, "factor": 0.6 },
//!         { "block_id": 1, "factor": 0.8 }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! ## Output ordering
//!
//! Entries are sorted by `(ncs_id, stage_id)` ascending. The `block_factors`
//! within each entry are sorted by `block_id` ascending.
//!
//! ## Validation
//!
//! - No two entries share the same `(ncs_id, stage_id)` pair.
//! - Every `factor` value must be strictly positive (> 0.0) and finite.

use cobre_core::EntityId;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::scenarios::BlockFactor;
use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `non_controllable_factors.json`.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawNcsFactorsFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of NCS factor entries.
    non_controllable_factors: Vec<RawNcsFactorEntry>,
}

/// Intermediate type for a single NCS factor entry (one NCS-stage pair).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct RawNcsFactorEntry {
    /// NCS entity identifier.
    ncs_id: i32,
    /// Stage identifier.
    stage_id: i32,
    /// Per-block scaling factors for this NCS-stage pair.
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

/// A single NCS factor entry from `scenarios/non_controllable_factors.json`.
///
/// One entry per `(ncs_id, stage_id)` pair, containing the per-block scaling
/// factors for NCS generation in that combination.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::{BlockFactor, NcsFactorEntry};
/// use cobre_core::EntityId;
///
/// let entry = NcsFactorEntry {
///     ncs_id: EntityId::from(0),
///     stage_id: 0,
///     block_factors: vec![
///         BlockFactor { block_id: 0, factor: 0.6 },
///         BlockFactor { block_id: 1, factor: 0.8 },
///     ],
/// };
/// assert_eq!(entry.ncs_id, EntityId::from(0));
/// assert_eq!(entry.block_factors.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NcsFactorEntry {
    /// NCS entity this entry applies to.
    pub ncs_id: EntityId,
    /// Stage this entry applies to.
    pub stage_id: i32,
    /// Per-block scaling factors sorted by `block_id` ascending.
    pub block_factors: Vec<BlockFactor>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse `scenarios/non_controllable_factors.json` and return a sorted entry list.
///
/// Reads the JSON file at `path`, deserializes it through intermediate types,
/// validates per-entry and per-block constraints, then returns all entries
/// sorted by `(ncs_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                             | Error variant              |
/// | ----------------------------------------------------- | -------------------------- |
/// | File not found / read failure                         | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field         | [`LoadError::ParseError`]  |
/// | `factor` not finite or `<= 0.0`                       | [`LoadError::SchemaError`] |
/// | Duplicate `(ncs_id, stage_id)` pair                   | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_non_controllable_factors;
/// use std::path::Path;
///
/// let entries = parse_non_controllable_factors(
///     Path::new("scenarios/non_controllable_factors.json"),
/// ).expect("valid NCS factors file");
/// println!("loaded {} NCS factor entries", entries.len());
/// ```
pub fn parse_non_controllable_factors(path: &Path) -> Result<Vec<NcsFactorEntry>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawNcsFactorsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    Ok(convert(raw))
}

// ── Validation ────────────────────────────────────────────────────────────────

fn validate_raw(raw: &RawNcsFactorsFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_entries(&raw.non_controllable_factors, path)?;
    for (i, entry) in raw.non_controllable_factors.iter().enumerate() {
        validate_block_factors(&entry.block_factors, i, path)?;
    }
    Ok(())
}

fn validate_no_duplicate_entries(
    entries: &[RawNcsFactorEntry],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<(i32, i32)> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        let key = (entry.ncs_id, entry.stage_id);
        if !seen.insert(key) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("non_controllable_factors[{i}]"),
                message: format!(
                    "duplicate (ncs_id={}, stage_id={}) in non_controllable_factors",
                    entry.ncs_id, entry.stage_id
                ),
            });
        }
    }
    Ok(())
}

fn validate_block_factors(
    block_factors: &[RawBlockFactor],
    entry_idx: usize,
    path: &Path,
) -> Result<(), LoadError> {
    for (j, bf) in block_factors.iter().enumerate() {
        if !bf.factor.is_finite() || bf.factor <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("non_controllable_factors[{entry_idx}].block_factors[{j}].factor"),
                message: format!("factor must be finite and > 0.0, got {}", bf.factor),
            });
        }
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

fn convert(raw: RawNcsFactorsFile) -> Vec<NcsFactorEntry> {
    let mut entries: Vec<NcsFactorEntry> = raw
        .non_controllable_factors
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
            NcsFactorEntry {
                ncs_id: EntityId::from(e.ncs_id),
                stage_id: e.stage_id,
                block_factors,
            }
        })
        .collect();

    entries.sort_by(|a, b| {
        a.ncs_id
            .0
            .cmp(&b.ncs_id.0)
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

    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    const VALID_JSON: &str = r#"{
  "non_controllable_factors": [
    {
      "ncs_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.6 },
        { "block_id": 1, "factor": 0.8 }
      ]
    },
    {
      "ncs_id": 1,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "factor": 0.9 }
      ]
    }
  ]
}"#;

    #[test]
    fn test_valid_2_entries() {
        let tmp = write_json(VALID_JSON);
        let entries = parse_non_controllable_factors(tmp.path()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].ncs_id, EntityId::from(0));
        assert_eq!(entries[0].block_factors.len(), 2);
        assert!((entries[0].block_factors[0].factor - 0.6).abs() < 1e-10);
        assert_eq!(entries[1].ncs_id, EntityId::from(1));
    }

    #[test]
    fn test_sorted_output() {
        let json = r#"{
  "non_controllable_factors": [
    { "ncs_id": 1, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 1.0 }] },
    { "ncs_id": 0, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 0.5 }] }
  ]
}"#;
        let tmp = write_json(json);
        let entries = parse_non_controllable_factors(tmp.path()).unwrap();
        assert_eq!(entries[0].ncs_id, EntityId::from(0));
        assert_eq!(entries[1].ncs_id, EntityId::from(1));
    }

    #[test]
    fn test_block_factors_sorted() {
        let json = r#"{
  "non_controllable_factors": [
    {
      "ncs_id": 0, "stage_id": 0,
      "block_factors": [
        { "block_id": 2, "factor": 1.0 },
        { "block_id": 0, "factor": 0.5 },
        { "block_id": 1, "factor": 0.7 }
      ]
    }
  ]
}"#;
        let tmp = write_json(json);
        let entries = parse_non_controllable_factors(tmp.path()).unwrap();
        assert_eq!(entries[0].block_factors[0].block_id, 0);
        assert_eq!(entries[0].block_factors[1].block_id, 1);
        assert_eq!(entries[0].block_factors[2].block_id, 2);
    }

    #[test]
    fn test_zero_factor_rejected() {
        let json = r#"{
  "non_controllable_factors": [
    { "ncs_id": 0, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 0.0 }] }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_non_controllable_factors(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("factor"), "field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_negative_factor_rejected() {
        let json = r#"{
  "non_controllable_factors": [
    { "ncs_id": 0, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": -0.5 }] }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_non_controllable_factors(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("factor"), "field: {field}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_duplicate_rejected() {
        let json = r#"{
  "non_controllable_factors": [
    { "ncs_id": 0, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 1.0 }] },
    { "ncs_id": 0, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 0.9 }] }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_non_controllable_factors(tmp.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("duplicate"), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_empty_array() {
        let json = r#"{ "non_controllable_factors": [] }"#;
        let tmp = write_json(json);
        let entries = parse_non_controllable_factors(tmp.path()).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_declaration_order_invariance() {
        let json_fwd = r#"{
  "non_controllable_factors": [
    { "ncs_id": 0, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 1.0 }] },
    { "ncs_id": 1, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 0.5 }] }
  ]
}"#;
        let json_rev = r#"{
  "non_controllable_factors": [
    { "ncs_id": 1, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 0.5 }] },
    { "ncs_id": 0, "stage_id": 0, "block_factors": [{ "block_id": 0, "factor": 1.0 }] }
  ]
}"#;
        let tmp_fwd = write_json(json_fwd);
        let tmp_rev = write_json(json_rev);
        let fwd = parse_non_controllable_factors(tmp_fwd.path()).unwrap();
        let rev = parse_non_controllable_factors(tmp_rev.path()).unwrap();
        assert_eq!(fwd, rev);
    }
}
