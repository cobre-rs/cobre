//! Parsing for `constraints/exchange_factors.json` — block-level transmission
//! capacity multipliers.
//!
//! [`parse_exchange_factors`] reads `constraints/exchange_factors.json` and
//! returns a sorted `Vec<ExchangeFactorEntry>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "exchange_factors": [
//!     {
//!       "line_id": 0,
//!       "stage_id": 0,
//!       "block_factors": [
//!         { "block_id": 0, "direct_factor": 0.9, "reverse_factor": 0.9 }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! ## Output ordering
//!
//! Entries are sorted by `(line_id, stage_id)` ascending.
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! - All `direct_factor` and `reverse_factor` values must be finite.
//! - All `direct_factor` and `reverse_factor` values must be > 0.0.
//!
//! Deferred validations (not performed here, Epic 06):
//!
//! - `line_id` existence in the line registry.
//! - `block_id` validity for the referenced stage.
//! - Duplicate `(line_id, stage_id)` pair detection.

use serde::Deserialize;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `exchange_factors.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawExchangeFactorsFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of exchange factor entries.
    exchange_factors: Vec<RawExchangeFactorEntry>,
}

/// Intermediate type for a single `(line_id, stage_id)` entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct RawExchangeFactorEntry {
    /// Transmission line identifier.
    line_id: i32,
    /// Stage index.
    stage_id: i32,
    /// Block-level capacity multipliers.
    block_factors: Vec<RawBlockExchangeFactor>,
}

/// Intermediate type for a single block's exchange factors.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
struct RawBlockExchangeFactor {
    /// Block index.
    block_id: i32,
    /// Direct (forward) flow capacity multiplier. Must be > 0.0 and finite.
    direct_factor: f64,
    /// Reverse flow capacity multiplier. Must be > 0.0 and finite.
    reverse_factor: f64,
}

// ── Output types ──────────────────────────────────────────────────────────────

/// Exchange factor for a single block on a transmission line.
///
/// Multiplies the line's base capacity in the given block. Both factors must
/// be positive and finite.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::BlockExchangeFactor;
///
/// let f = BlockExchangeFactor {
///     block_id: 0,
///     direct_factor: 0.9,
///     reverse_factor: 0.85,
/// };
/// assert!((f.direct_factor - 0.9).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BlockExchangeFactor {
    /// Block index.
    pub block_id: i32,
    /// Direct flow multiplier.
    pub direct_factor: f64,
    /// Reverse flow multiplier.
    pub reverse_factor: f64,
}

/// Exchange factor entry for one `(line_id, stage_id)` pair.
///
/// Carries block-level transmission capacity multipliers. The `block_factors`
/// vec holds one entry per block for which overrides are specified. An empty
/// `block_factors` is valid (means no overrides for this entry).
///
/// Entries are sorted by `(line_id, stage_id)` ascending.
///
/// # Examples
///
/// ```
/// use cobre_io::constraints::{BlockExchangeFactor, ExchangeFactorEntry};
///
/// let entry = ExchangeFactorEntry {
///     line_id: 0,
///     stage_id: 0,
///     block_factors: vec![
///         BlockExchangeFactor { block_id: 0, direct_factor: 0.9, reverse_factor: 0.9 },
///         BlockExchangeFactor { block_id: 1, direct_factor: 1.0, reverse_factor: 0.8 },
///     ],
/// };
/// assert_eq!(entry.block_factors.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExchangeFactorEntry {
    /// Line ID.
    pub line_id: i32,
    /// Stage ID.
    pub stage_id: i32,
    /// Block factors for this (line, stage) pair.
    pub block_factors: Vec<BlockExchangeFactor>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `constraints/exchange_factors.json` from `path`.
///
/// Reads the JSON file, deserialises it through intermediate serde types,
/// validates all exchange factor values, then converts to `Vec<ExchangeFactorEntry>`.
/// The result is sorted by `(line_id, stage_id)` ascending.
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// | --------------------------------------------- | -------------------------- |
/// | File not found / read failure                 | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field | [`LoadError::ParseError`]  |
/// | Exchange factor <= 0.0                        | [`LoadError::SchemaError`] |
/// | Exchange factor is NaN or infinite            | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::constraints::parse_exchange_factors;
/// use std::path::Path;
///
/// let entries = parse_exchange_factors(
///     Path::new("case/constraints/exchange_factors.json")
/// ).expect("valid exchange factors file");
/// println!("loaded {} exchange factor entries", entries.len());
/// ```
pub fn parse_exchange_factors(path: &Path) -> Result<Vec<ExchangeFactorEntry>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawExchangeFactorsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    Ok(convert(raw))
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized exchange factors data.
fn validate_raw(raw: &RawExchangeFactorsFile, path: &Path) -> Result<(), LoadError> {
    for (i, entry) in raw.exchange_factors.iter().enumerate() {
        for (j, bf) in entry.block_factors.iter().enumerate() {
            validate_factor(
                bf.direct_factor,
                &format!("exchange_factors[{i}].block_factors[{j}].direct_factor"),
                path,
            )?;
            validate_factor(
                bf.reverse_factor,
                &format!("exchange_factors[{i}].block_factors[{j}].reverse_factor"),
                path,
            )?;
        }
    }
    Ok(())
}

/// Validate a single exchange factor value: must be finite and > 0.0.
fn validate_factor(value: f64, field: &str, path: &Path) -> Result<(), LoadError> {
    if !value.is_finite() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field.to_string(),
            message: format!("exchange factor must be finite, got {value}"),
        });
    }
    if value <= 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field.to_string(),
            message: format!("exchange factor must be > 0.0, got {value}"),
        });
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert the validated raw data to `Vec<ExchangeFactorEntry>`, sorted by `(line_id, stage_id)`.
fn convert(raw: RawExchangeFactorsFile) -> Vec<ExchangeFactorEntry> {
    let mut result: Vec<ExchangeFactorEntry> = raw
        .exchange_factors
        .into_iter()
        .map(|entry| ExchangeFactorEntry {
            line_id: entry.line_id,
            stage_id: entry.stage_id,
            block_factors: entry
                .block_factors
                .into_iter()
                .map(|bf| BlockExchangeFactor {
                    block_id: bf.block_id,
                    direct_factor: bf.direct_factor,
                    reverse_factor: bf.reverse_factor,
                })
                .collect(),
        })
        .collect();

    // Sort by (line_id, stage_id) ascending.
    result.sort_by(|a, b| {
        a.line_id
            .cmp(&b.line_id)
            .then_with(|| a.stage_id.cmp(&b.stage_id))
    });

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(content.as_bytes()).expect("write");
        f
    }

    const VALID_JSON: &str = r#"{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.9, "reverse_factor": 0.9 },
        { "block_id": 1, "direct_factor": 1.0, "reverse_factor": 0.8 },
        { "block_id": 2, "direct_factor": 0.75, "reverse_factor": 1.0 }
      ]
    }
  ]
}"#;

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// AC-4 (ticket): Valid 1 entry with 3 block factors.
    #[test]
    fn test_parse_valid_single_entry_three_block_factors() {
        let f = write_json(VALID_JSON);
        let result = parse_exchange_factors(f.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].line_id, 0);
        assert_eq!(result[0].stage_id, 0);
        assert_eq!(result[0].block_factors.len(), 3);
        assert!((result[0].block_factors[0].direct_factor - 0.9).abs() < f64::EPSILON);
        assert!((result[0].block_factors[0].reverse_factor - 0.9).abs() < f64::EPSILON);
        assert!((result[0].block_factors[2].direct_factor - 0.75).abs() < f64::EPSILON);
    }

    /// Negative direct_factor → SchemaError.
    #[test]
    fn test_parse_negative_direct_factor_returns_schema_error() {
        let json = r#"{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "direct_factor": -0.5, "reverse_factor": 1.0 }
      ]
    }
  ]
}"#;
        let f = write_json(json);
        let err = parse_exchange_factors(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("direct_factor"),
                    "field should contain 'direct_factor', got: {field}"
                );
                assert!(
                    message.contains("> 0.0"),
                    "message should mention > 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Zero reverse_factor → SchemaError.
    #[test]
    fn test_parse_zero_reverse_factor_returns_schema_error() {
        let json = r#"{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": [
        { "block_id": 0, "direct_factor": 1.0, "reverse_factor": 0.0 }
      ]
    }
  ]
}"#;
        let f = write_json(json);
        let err = parse_exchange_factors(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reverse_factor"),
                    "field should contain 'reverse_factor', got: {field}"
                );
                assert!(
                    message.contains("> 0.0"),
                    "message should mention > 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// NaN factor → SchemaError.
    #[test]
    fn test_parse_nan_factor_returns_schema_error() {
        // NaN cannot be expressed in JSON without custom serializer; use infinity
        // which is also non-finite. Test the code path via direct validation.
        let err =
            validate_factor(f64::NAN, "test_field", std::path::Path::new("test.json")).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("test_field"));
                assert!(message.contains("finite"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Empty exchange_factors array → Ok(Vec::new()).
    #[test]
    fn test_parse_empty_array_returns_empty_vec() {
        let json = r#"{ "exchange_factors": [] }"#;
        let f = write_json(json);
        let result = parse_exchange_factors(f.path()).unwrap();
        assert!(result.is_empty());
    }

    /// Sorted output: entries come back in (line_id, stage_id) ascending order.
    #[test]
    fn test_parse_sorted_by_line_id_stage_id() {
        let json = r#"{
  "exchange_factors": [
    {
      "line_id": 1,
      "stage_id": 3,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.5, "reverse_factor": 0.5 }
      ]
    },
    {
      "line_id": 0,
      "stage_id": 5,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.8, "reverse_factor": 0.8 }
      ]
    },
    {
      "line_id": 0,
      "stage_id": 2,
      "block_factors": [
        { "block_id": 0, "direct_factor": 0.9, "reverse_factor": 0.9 }
      ]
    }
  ]
}"#;
        let f = write_json(json);
        let result = parse_exchange_factors(f.path()).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!((result[0].line_id, result[0].stage_id), (0, 2));
        assert_eq!((result[1].line_id, result[1].stage_id), (0, 5));
        assert_eq!((result[2].line_id, result[2].stage_id), (1, 3));
    }

    /// Entry with empty block_factors is valid.
    #[test]
    fn test_parse_entry_with_empty_block_factors() {
        let json = r#"{
  "exchange_factors": [
    {
      "line_id": 0,
      "stage_id": 0,
      "block_factors": []
    }
  ]
}"#;
        let f = write_json(json);
        let result = parse_exchange_factors(f.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].block_factors.is_empty());
    }
}
