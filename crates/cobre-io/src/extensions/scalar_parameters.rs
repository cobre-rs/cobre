//! Parsing for the scalar parameters JSON case file.
//!
//! ## JSON schema (`constraints/generic_parameters.json`)
//!
//! The case file stores scalar parameters as a JSON object with a single
//! top-level key `"scalar_parameters"` whose value is an array of parameter
//! objects. Each parameter object has `"id"`, `"name"`, and `"kind"` fields,
//! plus kind-specific payload fields at the same level:
//!
//! ```json
//! {
//!   "scalar_parameters": [
//!     { "id": 1, "name": "discount_rate", "kind": "constant", "value": 0.05 },
//!     { "id": 2, "name": "demand", "kind": "per_stage",
//!       "values": [[0, 100.0], [1, 110.0], [2, 105.0]] },
//!     { "id": 3, "name": "wet_season_factor", "kind": "seasonal",
//!       "values": [[0, 1.2], [1, 0.8]] },
//!     { "id": 4, "name": "hydro_prod", "kind": "computed",
//!       "computed_spec": { "tag": "equivalent_productivity", "hydro_id": 7 } }
//!   ]
//! }
//! ```
//!
//! All variant discriminators (`kind`, `tag`) use `snake_case`.
//!
//! ### `kind` variants and payload fields
//!
//! | `kind`      | Extra fields                                                          |
//! | ----------- | --------------------------------------------------------------------- |
//! | `constant`  | `"value": <f64>` — one value for all stages                          |
//! | `per_stage` | `"values": [[stage_id, value], ...]` — contiguous from 0, sorted     |
//! | `seasonal`  | `"values": [[season_id, value], ...]`                                 |
//! | `computed`  | `"computed_spec": { "tag": "<variant>", "hydro_id": <int> }`         |
//! | `per_stage_block` | `"block_values": [[stage_id, block_id, value], ...]` — unique `(stage_id, block_id)`, sorted |
//!
//! ### `per_stage` contiguity rule
//!
//! The `stage_id` integers in a `"per_stage"` `"values"` array must form a
//! contiguous range starting at `0` (i.e. `[0, 1, 2, …, N-1]`).  Duplicate
//! `stage_id` keys and gaps are both rejected at parse time.  The in-memory
//! representation strips the stage indices and stores a dense `Vec<f64>`.
//!
//! ### `seasonal` uniqueness rule
//!
//! Duplicate `season_id` keys within a single `"seasonal"` `"values"` array
//! are rejected at parse time.  This is an authoring error; the parser does
//! **not** silently deduplicate.
//!
//! ### `computed_spec` tag values
//!
//! | `tag`                     | Meaning                                  |
//! | ------------------------- | ---------------------------------------- |
//! | `equivalent_productivity` | Equivalent productivity coefficient      |
//! | `accumulated_productivity`| Accumulated productivity coefficient     |
//! | `reference_volume`        | Reference reservoir volume               |
//! | `reference_turbine`       | Reference turbine flow                   |
//! | `min_storage`             | Minimum operational storage              |
//! | `max_storage`             | Maximum operational storage              |
//! | `specific_productivity`   | Specific productivity                    |
//! | `integrated_equivalent_productivity`  | Storage-range mean equivalent productivity  |
//! | `integrated_accumulated_productivity` | Storage-range mean accumulated productivity |
//! | `max_stored_energy`                   | Maximum stored energy over the physical per-stage storage range |
//!
//! ## Rejection rules
//!
//! The parser rejects with [`crate::LoadError::SchemaError`]:
//!
//! - Duplicate `id` across entries.
//! - Duplicate `name` across entries (case-sensitive).
//! - Empty `name` or `name` with leading/trailing whitespace.
//! - `"seasonal"` entries with duplicate `season_id` keys.
//! - `"per_stage"` entries whose `stage_id` keys are not a contiguous range
//!   from 0, or that contain duplicates.
//! - `"per_stage_block"` entries with a duplicate `(stage_id, block_id)` pair.
//! - `block_values` present on a non-`"per_stage_block"` entry, or absent on a
//!   `"per_stage_block"` entry.
//! - Non-finite `value` fields (NaN, ±∞).
//!
//! Top-level parse failures (malformed JSON) and unknown JSON fields, such as a
//! stale `"values_source"`, return [`crate::LoadError::ParseError`].  I/O
//! failures return [`crate::LoadError`].
//!
//! ## Output ordering
//!
//! The returned `Vec<ScalarParameter>` is sorted ascending by `id.0`,
//! regardless of the order entries appear in the file.

use std::collections::HashSet;
use std::path::Path;

use cobre_core::{ComputedParameter, EntityId, ParameterKind, ScalarParameter};
use serde::Deserialize;

use crate::LoadError;

// Intermediate types with inline payload fields: `ScalarParameter` cannot be
// deserialized directly because the JSON places payload fields flat alongside
// `id`/`name`, and `#[serde(flatten)]` does not cooperate with `tag = "kind"`.

// No deny_unknown_fields here: editor keys such as `$schema` must load.
/// Root object of `generic_parameters.json`.
///
/// Top-level keys other than `scalar_parameters`, such as `$schema`, are
/// ignored; an unknown key inside an entry is an error.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct ScalarParametersFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    /// Array of scalar parameter entries.
    scalar_parameters: Vec<ScalarParameterJsonEntry>,
}

/// One parameter entry.
///
/// An unknown key, such as the retired `values_source`, is a parse error that
/// names the key.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct ScalarParameterJsonEntry {
    /// Stable numeric identifier; unique within the file.
    id: i32,
    /// Human-readable name; unique within the file. Used as the `@name` reference
    /// from `generic_constraints.json`.
    name: String,
    /// Selects which payload field (`value`, `values`, `computed_spec` or
    /// `block_values`) the entry carries.
    kind: RawParameterKind,
    /// Scalar value. Required when `kind == "constant"`; must be absent for all
    /// other kinds.
    value: Option<f64>,
    /// `[[key, value], ...]` pairs. Required when `kind == "per_stage"` (keys are
    /// stage ids, must be a contiguous range starting at 0) or
    /// `kind == "seasonal"` (keys are season ids, must be unique). Must be absent
    /// for `"constant"` and `"computed"`.
    values: Option<Vec<(i32, f64)>>,
    /// Computed-parameter specification. Required when `kind == "computed"`;
    /// must be absent for all other kinds.
    computed_spec: Option<ComputedParameter>,
    /// `[[stage_id, block_id, value], ...]` triples. Required when
    /// `kind == "per_stage_block"` (each `(stage_id, block_id)` pair unique);
    /// must be absent for all other kinds.
    block_values: Option<Vec<(i32, i32, f64)>>,
}

/// `kind` discriminator of a `generic_parameters.json` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawParameterKind {
    /// One value for every stage, in `value`.
    Constant,
    /// One value per stage, in `values` keyed by stage id.
    PerStage,
    /// One value per season, in `values` keyed by season id.
    Seasonal,
    /// A quantity derived from hydro data, in `computed_spec`.
    Computed,
    /// One value per stage and block, in `block_values`.
    PerStageBlock,
}

impl RawParameterKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::PerStage => "per_stage",
            Self::Seasonal => "seasonal",
            Self::Computed => "computed",
            Self::PerStageBlock => "per_stage_block",
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse `constraints/generic_parameters.json` and return a fully-assembled,
/// sorted parameter vector.
///
/// # Errors
///
/// | Condition                                     | Error variant                  |
/// | --------------------------------------------- | ------------------------------ |
/// | File not found or permission denied           | [`LoadError`]              |
/// | Malformed JSON                                | [`LoadError::ParseError`]      |
/// | Unknown JSON field in any entry               | [`LoadError::ParseError`]      |
/// | Unknown `kind` value in any entry             | [`LoadError::ParseError`]      |
/// | Duplicate `id` across entries                 | [`LoadError::SchemaError`]     |
/// | Duplicate `name` across entries               | [`LoadError::SchemaError`]     |
/// | Empty or whitespace-trimmed `name`            | [`LoadError::SchemaError`]     |
/// | Duplicate `season_id` in `"seasonal"` values  | [`LoadError::SchemaError`]     |
/// | Non-contiguous `stage_id` in `"per_stage"`    | [`LoadError::SchemaError`]     |
/// | Duplicate `stage_id` in `"per_stage"` values  | [`LoadError::SchemaError`]     |
/// | Duplicate `(stage_id, block_id)` in `"per_stage_block"` | [`LoadError::SchemaError`] |
/// | `block_values` on wrong kind / absent on `"per_stage_block"` | [`LoadError::SchemaError`] |
/// | Non-finite numeric value                      | [`LoadError::SchemaError`]     |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::parse_scalar_parameters_json;
/// use std::path::Path;
///
/// let params = parse_scalar_parameters_json(
///     Path::new("constraints/generic_parameters.json")
/// ).expect("valid parameters file");
/// println!("loaded {} parameters", params.len());
/// ```
pub fn parse_scalar_parameters_json(path: &Path) -> Result<Vec<ScalarParameter>, LoadError> {
    let text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let file: ScalarParametersFile =
        serde_json::from_str(&text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    let entries = file.scalar_parameters;
    let mut seen_ids: HashSet<i32> = HashSet::with_capacity(entries.len());
    let mut seen_names: HashSet<String> = HashSet::with_capacity(entries.len());
    let mut result: Vec<ScalarParameter> = Vec::with_capacity(entries.len());

    for (i, entry) in entries.into_iter().enumerate() {
        if !seen_ids.insert(entry.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].id"),
                message: format!("duplicate id {}", entry.id),
            });
        }

        if entry.name.is_empty() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].name"),
                message: "name must not be empty".to_string(),
            });
        }
        if entry.name != entry.name.trim() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].name"),
                message: format!(
                    "name must not have leading or trailing whitespace, got {:?}",
                    entry.name
                ),
            });
        }

        if !seen_names.insert(entry.name.clone()) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].name"),
                message: format!("duplicate name {:?}", entry.name),
            });
        }

        let kind = convert_entry_to_kind(i, &entry, path)?;

        result.push(ScalarParameter {
            id: EntityId(entry.id),
            name: entry.name,
            kind,
        });
    }

    result.sort_by_key(|p| p.id.0);

    Ok(result)
}

// ── Private conversion helpers ────────────────────────────────────────────────

/// Convert a single `ScalarParameterJsonEntry` into a `ParameterKind`,
/// performing all per-kind validations.
fn convert_entry_to_kind(
    i: usize,
    entry: &ScalarParameterJsonEntry,
    path: &Path,
) -> Result<ParameterKind, LoadError> {
    if entry.kind != RawParameterKind::PerStageBlock && entry.block_values.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].block_values"),
            message: format!(
                "\"block_values\" is only valid for kind \"per_stage_block\", not \"{}\"",
                entry.kind.as_str()
            ),
        });
    }

    match entry.kind {
        RawParameterKind::Constant => convert_constant(i, entry, path),
        RawParameterKind::PerStage => convert_per_stage(i, entry, path),
        RawParameterKind::Seasonal => convert_seasonal(i, entry, path),
        RawParameterKind::Computed => convert_computed(i, entry, path),
        RawParameterKind::PerStageBlock => convert_per_stage_block(i, entry, path),
    }
}

/// Build a `ParameterKind::Constant` from an entry, validating field presence
/// and finiteness.
fn convert_constant(
    i: usize,
    entry: &ScalarParameterJsonEntry,
    path: &Path,
) -> Result<ParameterKind, LoadError> {
    let value = entry.value.ok_or_else(|| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: format!("scalar_parameters[{i}].value"),
        message: "\"constant\" kind requires a \"value\" field".to_string(),
    })?;
    if !value.is_finite() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].value"),
            message: format!("value must be finite, got {value}"),
        });
    }
    Ok(ParameterKind::Constant { value })
}

/// Build a `ParameterKind::PerStage`, rejecting non-finite values and a
/// `stage_id` set that is not a duplicate-free contiguous range from 0.
fn convert_per_stage(
    i: usize,
    entry: &ScalarParameterJsonEntry,
    path: &Path,
) -> Result<ParameterKind, LoadError> {
    let pairs = entry
        .values
        .as_deref()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].values"),
            message: "\"per_stage\" kind requires a \"values\" field".to_string(),
        })?;

    if pairs.is_empty() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].values"),
            message: "\"per_stage\" kind requires at least one entry".to_string(),
        });
    }

    let mut sorted: Vec<(i32, f64)> = pairs.to_vec();
    sorted.sort_by_key(|(k, _)| *k);

    for window in sorted.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].values"),
                message: format!("duplicate stage_id {} in per_stage values", window[0].0),
            });
        }
    }

    for (expected, &(actual, _)) in sorted.iter().enumerate() {
        let expected_i32 = i32::try_from(expected).unwrap_or(i32::MAX);
        if actual != expected_i32 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].values"),
                message: format!(
                    "per_stage values must have contiguous stage_ids starting at 0; \
                     expected stage_id {expected_i32} but got {actual}"
                ),
            });
        }
    }

    for &(stage_id, v) in &sorted {
        if !v.is_finite() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].values"),
                message: format!("value for stage_id {stage_id} must be finite, got {v}"),
            });
        }
    }

    Ok(ParameterKind::PerStage {
        values: sorted.into_iter().map(|(_, v)| v).collect(),
    })
}

/// Build a `ParameterKind::Seasonal`, rejecting non-finite values and duplicate
/// `season_id` keys.
fn convert_seasonal(
    i: usize,
    entry: &ScalarParameterJsonEntry,
    path: &Path,
) -> Result<ParameterKind, LoadError> {
    let pairs = entry
        .values
        .as_deref()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].values"),
            message: "\"seasonal\" kind requires a \"values\" field".to_string(),
        })?;

    if pairs.is_empty() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].values"),
            message: "\"seasonal\" kind requires at least one entry".to_string(),
        });
    }

    // Reject duplicate season_ids explicitly: new_seasonal silently dedups, but a
    // duplicate in a user-authored file is an error, not a value to drop.
    let mut seen_seasons: HashSet<i32> = HashSet::new();
    for &(season_id, v) in pairs {
        if !seen_seasons.insert(season_id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].values"),
                message: format!("duplicate season_id {season_id} in seasonal values"),
            });
        }
        if !v.is_finite() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].values"),
                message: format!("value for season_id {season_id} must be finite, got {v}"),
            });
        }
    }

    Ok(ParameterKind::new_seasonal(pairs.to_vec()))
}

/// Build a `ParameterKind::Computed` from an entry.
///
/// Validates that `"computed_spec"` is present.
fn convert_computed(
    i: usize,
    entry: &ScalarParameterJsonEntry,
    path: &Path,
) -> Result<ParameterKind, LoadError> {
    let computed_spec = entry.computed_spec.ok_or_else(|| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: format!("scalar_parameters[{i}].computed_spec"),
        message: "\"computed\" kind requires a \"computed_spec\" field".to_string(),
    })?;
    Ok(ParameterKind::Computed { computed_spec })
}

/// Build a `ParameterKind::PerStageBlock`, rejecting non-finite values and a
/// duplicate `(stage_id, block_id)` pair. Per-stage block-count coverage is
/// validated later, at resolution, not here — parse time has no block counts.
fn convert_per_stage_block(
    i: usize,
    entry: &ScalarParameterJsonEntry,
    path: &Path,
) -> Result<ParameterKind, LoadError> {
    let triples = entry
        .block_values
        .as_deref()
        .ok_or_else(|| LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].block_values"),
            message: "\"per_stage_block\" kind requires a \"block_values\" field".to_string(),
        })?;

    if triples.is_empty() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("scalar_parameters[{i}].block_values"),
            message: "\"per_stage_block\" kind requires at least one entry".to_string(),
        });
    }

    let mut sorted: Vec<(i32, i32, f64)> = triples.to_vec();
    sorted.sort_by_key(|&(stage_id, block_id, _)| (stage_id, block_id));

    for window in sorted.windows(2) {
        if window[0].0 == window[1].0 && window[0].1 == window[1].1 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].block_values"),
                message: format!(
                    "duplicate (stage_id, block_id) pair ({}, {}) in per_stage_block values",
                    window[0].0, window[0].1
                ),
            });
        }
    }

    for &(stage_id, block_id, v) in &sorted {
        if !v.is_finite() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("scalar_parameters[{i}].block_values"),
                message: format!(
                    "value for (stage_id {stage_id}, block_id {block_id}) must be finite, got {v}"
                ),
            });
        }
    }

    Ok(ParameterKind::PerStageBlock { values: sorted })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

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
    use crate::test_support::write_json;
    use cobre_core::EntityId;

    // ── Test 1: happy path — all four variants ─────────────────────────────────

    /// A file containing one entry of each kind round-trips correctly.
    #[test]
    fn scalar_parameters_json_happy_path_all_four_variants() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "const_p", "kind": "constant", "value": 3.6 },
                { "id": 2, "name": "stage_p", "kind": "per_stage",
                  "values": [[0, 1.0], [1, 2.0], [2, 3.0]] },
                { "id": 3, "name": "season_p", "kind": "seasonal",
                  "values": [[2, 0.8], [1, 1.2]] },
                { "id": 4, "name": "computed_p", "kind": "computed",
                  "computed_spec": { "tag": "equivalent_productivity", "hydro_id": 7 } }
            ]
        }"#;
        let tmp = write_json(json);
        let params = parse_scalar_parameters_json(tmp.path()).unwrap();

        assert_eq!(params.len(), 4);
        // Sorted by id ascending.
        assert_eq!(params[0].id, EntityId(1));
        assert_eq!(params[0].kind, ParameterKind::Constant { value: 3.6 });

        assert_eq!(params[1].id, EntityId(2));
        assert_eq!(
            params[1].kind,
            ParameterKind::PerStage {
                values: vec![1.0, 2.0, 3.0]
            }
        );

        assert_eq!(params[2].id, EntityId(3));
        // new_seasonal sorts ascending by season_id.
        assert_eq!(
            params[2].kind,
            ParameterKind::Seasonal {
                values: vec![(1, 1.2), (2, 0.8)]
            }
        );

        assert_eq!(params[3].id, EntityId(4));
        assert_eq!(
            params[3].kind,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(7)
                }
            }
        );
    }

    // ── Test 2: duplicate id rejected ─────────────────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_duplicate_id() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "a", "kind": "constant", "value": 3.6 },
                { "id": 1, "name": "b", "kind": "constant", "value": 4.0 }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("duplicate id"),
                    "message should contain 'duplicate id', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 3: duplicate name rejected ───────────────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_duplicate_name() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "rho_eq_h1", "kind": "constant", "value": 3.6 },
                { "id": 2, "name": "rho_eq_h1", "kind": "constant", "value": 4.0 }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("duplicate name"),
                    "message should contain 'duplicate name', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 4: empty name rejected ────────────────────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_empty_name() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "", "kind": "constant", "value": 3.6 }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("name"),
                    "field should contain 'name', got: {field}"
                );
                assert!(
                    message.contains("empty"),
                    "message should mention 'empty', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 5: whitespace name rejected ──────────────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_whitespace_name() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": " rho ", "kind": "constant", "value": 3.6 }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("name"),
                    "field should contain 'name', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 6: seasonal duplicate keys rejected ───────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_seasonal_duplicate_keys() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "s", "kind": "seasonal",
                  "values": [[1, 0.5], [1, 0.6]] }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("duplicate season_id"),
                    "message should contain 'duplicate season_id', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 7: per_stage non-contiguous keys rejected ─────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_per_stage_non_contiguous_keys() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "p", "kind": "per_stage",
                  "values": [[0, 1.0], [2, 2.0]] }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("contiguous") || message.contains("non-contiguous"),
                    "message should mention contiguity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 8: per_stage duplicate keys rejected ──────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_per_stage_duplicate_keys() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "p", "kind": "per_stage",
                  "values": [[0, 1.0], [0, 2.0], [1, 3.0]] }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("duplicate stage_id"),
                    "message should contain 'duplicate stage_id', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 9: non-finite value rejected ─────────────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_non_finite_value() {
        // JSON syntax cannot express NaN, so serde_json can never hand a NaN to
        // this path. To cover the runtime check anyway (in case a future JSON
        // extension passes NaN through), we call the conversion function directly.
        let result = convert_constant(
            0,
            &ScalarParameterJsonEntry {
                id: 1,
                name: "p".to_string(),
                kind: RawParameterKind::Constant,
                value: Some(f64::NAN),
                values: None,
                computed_spec: None,
                block_values: None,
            },
            std::path::Path::new("/test.json"),
        );
        assert!(result.is_err(), "NaN constant value should be rejected");
        let err = result.unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("finite"),
                    "message should mention 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 10: unknown field (`values_source`) rejected ──────────────────────

    #[test]
    fn scalar_parameters_json_rejects_unknown_field() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "a", "kind": "constant", "value": 3.6,
                  "values_source": "sidecar" }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        // deny_unknown_fields causes serde_json to return a parse error
        // whose message contains the unknown field name.
        match err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("values_source") || message.contains("unknown field"),
                    "message should mention 'values_source' or 'unknown field', got: {message}"
                );
            }
            other => panic!("expected ParseError for unknown field, got: {other:?}"),
        }
    }

    // ── Test 11: declaration-order invariance ──────────────────────────────────

    /// Entries declared in id order [3, 1, 2] must be returned sorted [1, 2, 3].
    #[test]
    fn scalar_parameters_json_normalizes_declaration_order() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 3, "name": "third", "kind": "constant", "value": 3.0 },
                { "id": 1, "name": "first",  "kind": "constant", "value": 1.0 },
                { "id": 2, "name": "second", "kind": "constant", "value": 2.0 }
            ]
        }"#;
        let tmp = write_json(json);
        let params = parse_scalar_parameters_json(tmp.path()).unwrap();

        assert_eq!(params.len(), 3);
        assert_eq!(params[0].id, EntityId(1));
        assert_eq!(params[1].id, EntityId(2));
        assert_eq!(params[2].id, EntityId(3));
    }

    // ── Test 12: per_stage empty values rejected ───────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_per_stage_empty_values() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "p", "kind": "per_stage", "values": [] }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("at least one entry"),
                    "message should mention 'at least one entry', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Test 13: seasonal empty values rejected ────────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_seasonal_empty_values() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "s", "kind": "seasonal", "values": [] }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("at least one entry"),
                    "message should mention 'at least one entry', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── per_stage_block: happy path ────────────────────────────────────────────

    #[test]
    fn scalar_parameters_json_parses_per_stage_block() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "block_p", "kind": "per_stage_block",
                  "block_values": [[0, 0, 1.0], [0, 1, 2.0]] }
            ]
        }"#;
        let tmp = write_json(json);
        let params = parse_scalar_parameters_json(tmp.path()).unwrap();

        assert_eq!(params.len(), 1);
        assert_eq!(params[0].id, EntityId(1));
        assert_eq!(
            params[0].kind,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0), (0, 1, 2.0)]
            }
        );
    }

    // ── per_stage_block: triples stored sorted ─────────────────────────────────

    #[test]
    fn scalar_parameters_json_per_stage_block_sorts_triples() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "block_p", "kind": "per_stage_block",
                  "block_values": [[1, 0, 3.0], [0, 1, 2.0], [0, 0, 1.0]] }
            ]
        }"#;
        let tmp = write_json(json);
        let params = parse_scalar_parameters_json(tmp.path()).unwrap();
        assert_eq!(
            params[0].kind,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)]
            }
        );
    }

    // ── per_stage_block: duplicate pair rejected ───────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_per_stage_block_duplicate_pair() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "b", "kind": "per_stage_block",
                  "block_values": [[0, 0, 1.0], [0, 0, 2.0]] }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── per_stage_block: block_values on a non-block kind rejected ──────────────

    #[test]
    fn scalar_parameters_json_rejects_block_values_on_wrong_kind() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "c", "kind": "constant", "value": 3.6,
                  "block_values": [[0, 0, 1.0]] }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("block_values"),
                    "field should contain 'block_values', got: {field}"
                );
                assert!(
                    message.contains("per_stage_block"),
                    "message should mention 'per_stage_block', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── per_stage_block: block_values absent rejected ──────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_per_stage_block_missing_block_values() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "b", "kind": "per_stage_block" }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("block_values"),
                    "field should contain 'block_values', got: {field}"
                );
                assert!(
                    message.contains("requires"),
                    "message should mention 'requires', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── kind: unlisted kind rejected ───────────────────────────────────────────

    #[test]
    fn scalar_parameters_json_rejects_unknown_kind_naming_every_kind() {
        let json = r#"{
            "scalar_parameters": [
                { "id": 1, "name": "p", "kind": "bogus", "value": 1.0 }
            ]
        }"#;
        let tmp = write_json(json);
        let err = parse_scalar_parameters_json(tmp.path()).unwrap_err();
        match err {
            LoadError::ParseError { message, .. } => {
                for spelling in [
                    "bogus",
                    "constant",
                    "per_stage",
                    "seasonal",
                    "computed",
                    "per_stage_block",
                ] {
                    assert!(
                        message.contains(&format!("`{spelling}`")),
                        "message should name `{spelling}`, got: {message}"
                    );
                }
            }
            other => panic!("expected ParseError for an unlisted kind, got: {other:?}"),
        }
    }

    // ── kind: as_str matches the JSON spelling ─────────────────────────────────

    #[test]
    fn raw_parameter_kind_as_str_is_its_json_spelling() {
        for kind in [
            RawParameterKind::Constant,
            RawParameterKind::PerStage,
            RawParameterKind::Seasonal,
            RawParameterKind::Computed,
            RawParameterKind::PerStageBlock,
        ] {
            let parsed: RawParameterKind =
                serde_json::from_str(&format!("\"{}\"", kind.as_str())).unwrap();
            assert_eq!(parsed, kind);
        }
    }
}
