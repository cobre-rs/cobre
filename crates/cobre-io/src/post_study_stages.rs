//! Parsing for `post_study_stages.json` — the post-horizon boundary calendar and
//! per-`(thermal, post-study stage)` cost/bounds table.
//!
//! [`parse_post_study_stages`] reads `post_study_stages.json` from the case
//! directory root and returns a [`PostStudyStages`] whose `stages` are ordered
//! ascending by `start_date` and whose `thermal_bounds` are in
//! `(thermal_id, post_study_stage_index)` order.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/post_study_stages.schema.json",
//!   "stages": [
//!     { "start_date": "2026-11-01", "duration_hours": 720.0 },
//!     { "start_date": "2026-12-01", "duration_hours": 744.0 }
//!   ],
//!   "thermal_bounds": [
//!     { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! The reader enforces the per-file invariants; the cross-file invariants
//! (date-contiguity, first `start_date` equals the study horizon end, a
//! `PostStudyThermalBound` for every post-study stage an anticipated lead
//! reaches, and no post-study delivery decided at a pre-study stage) are
//! enforced by the `cobre-io` semantic validator, which has the study
//! calendar this reader does not.
//!
//! 1. Every `start_date` parses as ISO 8601 (`YYYY-MM-DD`); no two stages share a
//!    `start_date`.
//! 2. Every `duration_hours` is finite and `> 0.0`.
//! 3. Every `cost_per_mwh`, `min_mw`, and `max_mw` is finite, and `min_mw <= max_mw`.
//! 4. Every `post_study_stage_index` is `< stages.len()`.
//! 5. No two thermal-bound rows share a `(thermal_id, post_study_stage_index)`.

use std::collections::HashSet;
use std::path::Path;

use cobre_core::{EntityId, PostStudyStage, PostStudyStages, PostStudyThermalBound};
use serde::Deserialize;

use crate::LoadError;
use crate::windowed_history::parse_iso_date;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Intermediate serde type for `post_study_stages.json`, deserialized then
/// validated before conversion to [`PostStudyStages`].
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawPostStudyStagesFile {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Ordered post-horizon calendar stages.
    stages: Vec<RawPostStudyStage>,

    /// Per-`(thermal, post-study stage)` cost and generation bounds.
    thermal_bounds: Vec<RawPostStudyThermalBound>,
}

/// One post-horizon calendar stage: a `[start_date, start_date + duration_hours)`
/// segment. `end_date` is not declared; it is derived downstream.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPostStudyStage {
    /// Stage start date (inclusive), as an ISO 8601 date (YYYY-MM-DD).
    start_date: String,
    /// Stage duration \[h\]. Must be finite and `> 0.0`.
    duration_hours: f64,
}

/// Cost and generation bounds for one `(thermal, post-study stage)` cell.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPostStudyThermalBound {
    /// Thermal plant identifier. Must reference an anticipated thermal.
    thermal_id: i32,
    /// Index into `stages`. Must be `< stages.len()`.
    post_study_stage_index: usize,
    /// Fuel cost (`$/MWh`) at this cell. Must be finite.
    cost_per_mwh: f64,
    /// Lower bound of the delivered MW rate at this cell. Must be finite.
    min_mw: f64,
    /// Upper bound of the delivered MW rate at this cell. Must be finite and
    /// `>= min_mw`.
    max_mw: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `post_study_stages.json` from `path`.
///
/// # Errors
///
/// | Condition                                                        | Error variant              |
/// | ---------------------------------------------------------------- | -------------------------- |
/// | File not found / read failure                                    | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field                    | [`LoadError::ParseError`]  |
/// | Invalid `start_date` / duplicate `start_date`                    | [`LoadError::SchemaError`] |
/// | Non-finite or non-positive `duration_hours`                      | [`LoadError::SchemaError`] |
/// | Non-finite cost/bound, `min_mw > max_mw`                         | [`LoadError::SchemaError`] |
/// | `post_study_stage_index` out of range                            | [`LoadError::SchemaError`] |
/// | Duplicate `(thermal_id, post_study_stage_index)`                 | [`LoadError::SchemaError`] |
pub fn parse_post_study_stages(path: &Path) -> Result<PostStudyStages, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawPostStudyStagesFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    let stages = convert_stages(&raw.stages, path)?;
    let thermal_bounds = convert_thermal_bounds(&raw.thermal_bounds, stages.len(), path)?;

    Ok(PostStudyStages {
        stages,
        thermal_bounds,
    })
}

// ── Conversion + validation ─────────────────────────────────────────────────────

/// Parse and validate `stages`, returning them sorted ascending by `start_date`.
fn convert_stages(
    entries: &[RawPostStudyStage],
    path: &Path,
) -> Result<Vec<PostStudyStage>, LoadError> {
    let mut stages: Vec<PostStudyStage> = Vec::with_capacity(entries.len());
    let mut seen: HashSet<chrono::NaiveDate> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        let start_date =
            parse_iso_date(&format!("stages[{i}].start_date"), &entry.start_date, path)?;
        if !entry.duration_hours.is_finite() || entry.duration_hours <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("stages[{i}].duration_hours"),
                message: format!(
                    "duration_hours must be a finite number > 0.0, got {}",
                    entry.duration_hours
                ),
            });
        }
        if !seen.insert(start_date) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("stages[{i}].start_date"),
                message: format!("duplicate start_date {start_date} in stages"),
            });
        }
        stages.push(PostStudyStage {
            start_date,
            duration_hours: entry.duration_hours,
        });
    }
    stages.sort_by_key(|s| s.start_date);
    Ok(stages)
}

/// Parse and validate `thermal_bounds`, returning them sorted by
/// `(thermal_id, post_study_stage_index)`.
fn convert_thermal_bounds(
    entries: &[RawPostStudyThermalBound],
    num_stages: usize,
    path: &Path,
) -> Result<Vec<PostStudyThermalBound>, LoadError> {
    let mut bounds: Vec<PostStudyThermalBound> = Vec::with_capacity(entries.len());
    let mut seen: HashSet<(i32, usize)> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        for (value, name) in [
            (entry.cost_per_mwh, "cost_per_mwh"),
            (entry.min_mw, "min_mw"),
            (entry.max_mw, "max_mw"),
        ] {
            if !value.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("thermal_bounds[{i}].{name}"),
                    message: format!("{name} must be a finite number, got {value}"),
                });
            }
        }
        if entry.min_mw > entry.max_mw {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermal_bounds[{i}].max_mw"),
                message: format!(
                    "max_mw ({}) must be >= min_mw ({}) for thermal_id {}, \
                     post_study_stage_index {}",
                    entry.max_mw, entry.min_mw, entry.thermal_id, entry.post_study_stage_index
                ),
            });
        }
        if entry.post_study_stage_index >= num_stages {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermal_bounds[{i}].post_study_stage_index"),
                message: format!(
                    "post_study_stage_index {} is out of range for {num_stages} post-study \
                     stage(s)",
                    entry.post_study_stage_index
                ),
            });
        }
        if !seen.insert((entry.thermal_id, entry.post_study_stage_index)) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermal_bounds[{i}]"),
                message: format!(
                    "duplicate (thermal_id {}, post_study_stage_index {}) in thermal_bounds",
                    entry.thermal_id, entry.post_study_stage_index
                ),
            });
        }
        bounds.push(PostStudyThermalBound {
            thermal_id: EntityId(entry.thermal_id),
            post_study_stage_index: entry.post_study_stage_index,
            cost_per_mwh: entry.cost_per_mwh,
            min_mw: entry.min_mw,
            max_mw: entry.max_mw,
        });
    }
    bounds.sort_by_key(|b| (b.thermal_id.0, b.post_study_stage_index));
    Ok(bounds)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::test_support::write_json;

    const VALID_JSON: &str = r#"{
      "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/post_study_stages.schema.json",
      "stages": [
        { "start_date": "2026-11-01", "duration_hours": 720.0 },
        { "start_date": "2026-12-01", "duration_hours": 744.0 }
      ],
      "thermal_bounds": [
        { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
        { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }"#;

    #[test]
    fn test_parse_valid_sorts_both_collections() {
        let f = write_json(VALID_JSON);
        let ps = parse_post_study_stages(f.path()).unwrap();
        assert_eq!(ps.stages.len(), 2);
        assert_eq!(
            ps.stages[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 11, 1).unwrap()
        );
        // thermal_bounds sorted by (thermal_id, post_study_stage_index).
        assert_eq!(ps.thermal_bounds[0].post_study_stage_index, 0);
        assert_eq!(ps.thermal_bounds[1].post_study_stage_index, 1);
    }

    #[test]
    fn test_declaration_order_invariance() {
        let reversed = r#"{
          "stages": [
            { "start_date": "2026-12-01", "duration_hours": 744.0 },
            { "start_date": "2026-11-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#;
        let f1 = write_json(VALID_JSON);
        let f2 = write_json(reversed);
        assert_eq!(
            parse_post_study_stages(f1.path()).unwrap(),
            parse_post_study_stages(f2.path()).unwrap()
        );
    }

    #[test]
    fn test_non_positive_duration_rejected() {
        let json = r#"{ "stages": [{ "start_date": "2026-11-01", "duration_hours": 0.0 }], "thermal_bounds": [] }"#;
        let err = parse_post_study_stages(write_json(json).path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("duration_hours")),
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_inverted_bounds_rejected() {
        let json = r#"{
          "stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
          "thermal_bounds": [{ "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 1.0, "min_mw": 10.0, "max_mw": 5.0 }]
        }"#;
        let err = parse_post_study_stages(write_json(json).path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => assert!(message.contains("max_mw")),
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_stage_index_out_of_range_rejected() {
        let json = r#"{
          "stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
          "thermal_bounds": [{ "thermal_id": 1, "post_study_stage_index": 3, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 }]
        }"#;
        let err = parse_post_study_stages(write_json(json).path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("post_study_stage_index"));
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_duplicate_cell_rejected() {
        let json = r#"{
          "stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
          "thermal_bounds": [
            { "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 },
            { "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 2.0, "min_mw": 0.0, "max_mw": 6.0 }
          ]
        }"#;
        let err = parse_post_study_stages(write_json(json).path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => assert!(message.contains("duplicate")),
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_unknown_field_rejected() {
        let json = r#"{ "stages": [], "thermal_bounds": [], "extra": 1 }"#;
        let err = parse_post_study_stages(write_json(json).path()).unwrap_err();
        assert!(matches!(err, LoadError::ParseError { .. }));
    }
}
