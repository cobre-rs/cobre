//! Parsing for `scenarios/correlation.json` — spatial correlation profiles.
//!
//! [`parse_correlation`] reads `scenarios/correlation.json` and returns a
//! [`CorrelationModel`] holding all named profiles and the optional stage-to-profile
//! schedule.
//!
//! ## JSON structure (spec SS5)
//!
//! ```json
//! {
//!   "$schema": "...",
//!   "method": "spectral",
//!   "profiles": {
//!     "default": {
//!       "correlation_groups": [
//!         {
//!           "name": "southeast_cascade",
//!           "entities": [
//!             { "type": "inflow", "id": 0 },
//!             { "type": "inflow", "id": 1 }
//!           ],
//!           "matrix": [
//!             [1.0, 0.75],
//!             [0.75, 1.0]
//!           ]
//!         }
//!       ]
//!     }
//!   },
//!   "schedule": [
//!     { "stage_id": 0, "profile_name": "wet_season" }
//!   ]
//! }
//! ```
//!
//! ## Output
//!
//! Returns a [`CorrelationModel`] directly (the `cobre-core` type). The `profiles`
//! map uses [`BTreeMap`] for deterministic ordering (declaration-order invariance).
//!
//! ## Validation
//!
//! After deserializing, the following constraints are checked before conversion:
//!
//! - `profiles` must not be empty.
//! - For each group in each profile:
//!   - `matrix` must be square (`rows == cols`).
//!   - `matrix.len()` must equal `entities.len()`.
//!   - Diagonal entries must be exactly `1.0`.
//!   - Off-diagonal entries must be in `[-1.0, 1.0]`.
//!   - Matrix must be symmetric: `|m[i][j] - m[j][i]| <= 1e-10`.
//! - Each `schedule` entry's `profile_name` must exist in `profiles`.
//!
//! Deferred validations (not performed here):
//!
//! - Entity ID existence in entity registries.
//! - Positive semi-definiteness of correlation matrices.
//! - Requirement for a `"default"` profile when a schedule is present.
//! - Scenario count matching `stage.num_scenarios`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use cobre_core::EntityId;
use cobre_core::scenario::{
    CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
    CorrelationScheduleEntry,
};
use serde::{Deserialize, Deserializer, Serialize};

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Spatial correlation model loaded from `scenarios/correlation.json`.
///
/// Defines named correlation profiles and an optional stage-to-profile schedule
/// for stochastic scenario generation. Each profile contains one or more
/// correlation groups, each grouping a set of entity inflow series with a
/// symmetric correlation matrix decomposed via the method specified in the
/// `method` field (default: `"spectral"`).
///
/// Profiles are stored in deterministic order regardless of JSON declaration
/// order. Schedule entries reference profile names that must exist in the
/// `profiles` map.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawCorrelationFile {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Decomposition method for the correlation matrix. Accepts `spectral`.
    method: CorrelationMethod,

    /// Named correlation profiles. Must not be empty. Keys are profile names
    /// (e.g. `"default"`, `"wet_season"`).
    profiles: HashMap<String, RawProfile>,

    /// Optional stage-to-profile schedule. Absent or null = empty schedule
    /// (single profile applies everywhere). Each entry's `profile_name` must
    /// match a key in `profiles`.
    #[serde(default)]
    schedule: Option<Vec<RawScheduleEntry>>,
}

/// Correlation-matrix decomposition method (`scenarios/correlation.json`
/// `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum CorrelationMethod {
    /// Spectral (eigenvalue) decomposition.
    Spectral,
}

impl<'de> Deserialize<'de> for CorrelationMethod {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "spectral" => Ok(Self::Spectral),
            other => Err(serde::de::Error::unknown_variant(other, &["spectral"])),
        }
    }
}

/// A named correlation profile containing one or more correlation groups.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProfile {
    /// Groups of correlated entities, each with a correlation matrix.
    correlation_groups: Vec<RawCorrelationGroup>,
}

/// A named group of correlated entities with an associated correlation matrix.
///
/// All entities in a group must share the same `type` (same-type constraint).
/// Mixing entity types (e.g. inflow and load in the same group) produces a
/// runtime validation error.
///
/// The `matrix` must be square, symmetric, have 1.0 on every diagonal, and
/// have all off-diagonal entries in [-1.0, 1.0]. The number of rows (and
/// columns) must equal the number of `entities`.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCorrelationGroup {
    /// Human-readable group label.
    name: String,

    /// Ordered list of entity references. The matrix row/column order matches
    /// this list. All entities must have the same `type` value; mixing entity
    /// types in one group produces a runtime validation error.
    entities: Vec<RawEntity>,

    /// Symmetric correlation matrix in row-major order.
    /// - Must be square.
    /// - Diagonal entries must be exactly 1.0.
    /// - Off-diagonal entries must be in [-1.0, 1.0].
    /// - Must be symmetric: `|m[i][j] - m[j][i]| <= 1e-10`.
    matrix: Vec<Vec<f64>>,
}

/// Reference to a single entity within a correlation group.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEntity {
    /// Entity type tag. Valid values: `"inflow"` (hydro inflow series),
    /// `"load"` (bus load), `"ncs"` (non-controllable source availability).
    #[serde(rename = "type")]
    entity_type: String,

    /// Entity identifier matching the corresponding entity's `id` field.
    id: i32,
}

/// Maps a stage to the named correlation profile active for that stage.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScheduleEntry {
    /// Stage index (0-based) this entry applies to.
    stage_id: i32,

    /// Name of the correlation profile for this stage. Must match a key in
    /// `profiles`.
    profile_name: String,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse `scenarios/correlation.json` and return a [`CorrelationModel`].
///
/// When the file is absent, use the [`super::load_correlation`] wrapper which
/// accepts `Option<&Path>` and returns `Ok(CorrelationModel::default())`.
///
/// # Errors
///
/// | Condition                                                | Error variant              |
/// | -------------------------------------------------------- | -------------------------- |
/// | File not found / read failure                            | [`LoadError::IoError`]     |
/// | Malformed JSON / missing or unknown `method`             | [`LoadError::ParseError`]  |
/// | `profiles` is empty                                      | [`LoadError::SchemaError`] |
/// | Correlation matrix not square                            | [`LoadError::SchemaError`] |
/// | Matrix row count != entity count                         | [`LoadError::SchemaError`] |
/// | Diagonal entry != 1.0                                    | [`LoadError::SchemaError`] |
/// | Matrix element outside `[-1.0, 1.0]`                     | [`LoadError::SchemaError`] |
/// | Matrix not symmetric (`|m\[i\]\[j\] - m\[j\]\[i\]| > 1e-10`) | [`LoadError::SchemaError`] |
/// | Schedule `profile_name` not in `profiles`                | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_correlation;
/// use std::path::Path;
///
/// let model = parse_correlation(Path::new("scenarios/correlation.json"))
///     .expect("valid correlation file");
/// println!("method: {}", model.method);
/// println!("profiles: {}", model.profiles.len());
/// ```
pub fn parse_correlation(path: &Path) -> Result<CorrelationModel, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawCorrelationFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw(&raw, path)?;

    Ok(convert(raw))
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized data.
///
/// Called before conversion so that error messages can reference JSON field
/// paths rather than Rust field names.
fn validate_raw(raw: &RawCorrelationFile, path: &Path) -> Result<(), LoadError> {
    if raw.profiles.is_empty() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "profiles".to_string(),
            message: "at least one correlation profile must be defined".to_string(),
        });
    }

    // Sort keys for deterministic error ordering (matches BTreeMap output).
    let mut sorted_keys: Vec<&String> = raw.profiles.keys().collect();
    sorted_keys.sort();
    for profile_name in sorted_keys {
        let profile = &raw.profiles[profile_name];
        for (group_idx, group) in profile.correlation_groups.iter().enumerate() {
            validate_matrix(profile_name, group_idx, group, path)?;
        }
    }

    if let Some(schedule) = &raw.schedule {
        for (i, entry) in schedule.iter().enumerate() {
            if !raw.profiles.contains_key(&entry.profile_name) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("schedule[{i}].profile_name"),
                    message: format!(
                        "profile_name \"{}\" not found in profiles",
                        entry.profile_name
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Validate the correlation matrix for a single group within a profile.
fn validate_matrix(
    profile_name: &str,
    group_idx: usize,
    group: &RawCorrelationGroup,
    path: &Path,
) -> Result<(), LoadError> {
    let n_entities = group.entities.len();
    let n_rows = group.matrix.len();

    let field_prefix = format!("profiles.{profile_name}.correlation_groups[{group_idx}].matrix");
    if n_rows != n_entities {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix,
            message: format!("matrix row count ({n_rows}) must equal entity count ({n_entities})"),
        });
    }

    for (r, row) in group.matrix.iter().enumerate() {
        if row.len() != n_entities {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: field_prefix,
                message: format!(
                    "matrix is not square: row {r} has {} columns but expected {n_entities}",
                    row.len()
                ),
            });
        }
    }

    for r in 0..n_entities {
        for c in 0..n_entities {
            let val = group.matrix[r][c];

            if !(-1.0..=1.0).contains(&val) {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("{field_prefix}[{r}][{c}]"),
                    message: format!("correlation value {val} is outside [-1.0, 1.0]"),
                });
            }

            if r == c && (val - 1.0).abs() > f64::EPSILON {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("{field_prefix}[{r}][{r}]"),
                    message: format!("diagonal entry at [{r}][{r}] must be 1.0, got {val}"),
                });
            }

            if r < c {
                let mirror = group.matrix[c][r];
                if (val - mirror).abs() > 1e-10 {
                    return Err(LoadError::SchemaError {
                        path: path.to_path_buf(),
                        field: format!("{field_prefix}[{r}][{c}]"),
                        message: format!(
                            "matrix is not symmetric: m[{r}][{c}]={val} vs m[{c}][{r}]={mirror}"
                        ),
                    });
                }
            }
        }
    }

    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert validated raw data into [`CorrelationModel`].
///
/// Precondition: [`validate_raw`] has returned `Ok(())` for this data.
/// Profiles are inserted into a [`BTreeMap`] for deterministic ordering.
fn convert(raw: RawCorrelationFile) -> CorrelationModel {
    let profiles: BTreeMap<String, CorrelationProfile> = raw
        .profiles
        .into_iter()
        .map(|(name, raw_profile)| {
            let groups: Vec<CorrelationGroup> = raw_profile
                .correlation_groups
                .into_iter()
                .map(|raw_group| {
                    let entities: Vec<CorrelationEntity> = raw_group
                        .entities
                        .into_iter()
                        .map(|raw_entity| CorrelationEntity {
                            entity_type: raw_entity.entity_type,
                            id: EntityId(raw_entity.id),
                        })
                        .collect();
                    CorrelationGroup {
                        name: raw_group.name,
                        entities,
                        matrix: raw_group.matrix,
                    }
                })
                .collect();
            (name, CorrelationProfile { groups })
        })
        .collect();

    let schedule: Vec<CorrelationScheduleEntry> = raw
        .schedule
        .unwrap_or_default()
        .into_iter()
        .map(|entry| CorrelationScheduleEntry {
            stage_id: entry.stage_id,
            profile_name: entry.profile_name,
        })
        .collect();

    CorrelationModel {
        method: match raw.method {
            CorrelationMethod::Spectral => "spectral".to_string(),
        },
        profiles,
        schedule,
    }
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
    use crate::test_support::write_json;

    /// Canonical valid JSON for error-path tests (1 profile, 1 group, 2x2 matrix).
    const VALID_JSON: &str = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "se_cascade",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, 0.75],
            [0.75, 1.0]
          ]
        }
      ]
    }
  }
}"#;

    #[test]
    fn test_valid_3x3_identity_matrix() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "all_hydros",
          "entities": [
            { "type": "inflow", "id": 10 },
            { "type": "inflow", "id": 20 },
            { "type": "inflow", "id": 30 }
          ],
          "matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let model = parse_correlation(tmp.path()).unwrap();

        assert_eq!(model.method, "spectral");
        assert_eq!(model.profiles.len(), 1);
        assert!(model.profiles.contains_key("default"));
        assert!(model.schedule.is_empty());

        let profile = &model.profiles["default"];
        assert_eq!(profile.groups.len(), 1);

        let group = &profile.groups[0];
        assert_eq!(group.name, "all_hydros");
        assert_eq!(group.entities.len(), 3);
        assert!((group.matrix[1][1] - 1.0).abs() < f64::EPSILON);
        assert_eq!(group.entities[0].id, EntityId(10));
        assert_eq!(group.entities[0].entity_type, "inflow");
        assert_eq!(group.entities[1].id, EntityId(20));
        assert_eq!(group.entities[2].id, EntityId(30));
    }

    #[test]
    fn test_two_profiles_with_schedule() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "wet_season": {
      "correlation_groups": [
        {
          "name": "southeast",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, 0.9],
            [0.9, 1.0]
          ]
        }
      ]
    },
    "default": {
      "correlation_groups": [
        {
          "name": "all",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, 0.5],
            [0.5, 1.0]
          ]
        }
      ]
    }
  },
  "schedule": [
    { "stage_id": 0, "profile_name": "wet_season" },
    { "stage_id": 6, "profile_name": "default" }
  ]
}"#;
        let tmp = write_json(json);
        let model = parse_correlation(tmp.path()).unwrap();

        assert_eq!(model.profiles.len(), 2);
        assert_eq!(model.schedule.len(), 2);
        assert_eq!(model.method, "spectral");

        // BTreeMap ordering: "default" < "wet_season" alphabetically.
        let keys: Vec<&String> = model.profiles.keys().collect();
        assert_eq!(keys[0], "default");
        assert_eq!(keys[1], "wet_season");

        assert_eq!(model.schedule[0].stage_id, 0);
        assert_eq!(model.schedule[0].profile_name, "wet_season");
        assert_eq!(model.schedule[1].stage_id, 6);
        assert_eq!(model.schedule[1].profile_name, "default");
    }

    #[test]
    fn test_no_schedule_produces_empty_vec() {
        let tmp = write_json(VALID_JSON);
        let model = parse_correlation(tmp.path()).unwrap();

        assert!(model.schedule.is_empty());
    }

    #[test]
    fn test_non_symmetric_matrix_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "group_a",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, 0.8],
            [0.7, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("matrix"),
                    "field should contain 'matrix', got: {field}"
                );
                assert!(
                    message.contains("symmetric"),
                    "message should contain 'symmetric', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_diagonal_not_one_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "group_a",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [0.9, 0.5],
            [0.5, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("matrix"),
                    "field should contain 'matrix', got: {field}"
                );
                assert!(
                    message.contains("1.0"),
                    "message should mention 1.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_element_greater_than_one_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "group_a",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, 1.2],
            [1.2, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("matrix"),
                    "field should contain 'matrix', got: {field}"
                );
                assert!(
                    message.contains("[-1.0, 1.0]"),
                    "message should mention range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_element_less_than_minus_one_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "group_a",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, -1.5],
            [-1.5, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("matrix"),
                    "field should contain 'matrix', got: {field}"
                );
                assert!(
                    message.contains("[-1.0, 1.0]"),
                    "message should mention range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_non_square_matrix_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "group_a",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("matrix"),
                    "field should contain 'matrix', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_matrix_row_count_mismatch_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "group_a",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 },
            { "type": "inflow", "id": 2 }
          ],
          "matrix": [
            [1.0, 0.5],
            [0.5, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("matrix"),
                    "field should contain 'matrix', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_schedule_unknown_profile_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "group_a",
          "entities": [
            { "type": "inflow", "id": 0 }
          ],
          "matrix": [[1.0]]
        }
      ]
    }
  },
  "schedule": [
    { "stage_id": 0, "profile_name": "nonexistent" }
  ]
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("schedule[0].profile_name"),
                    "field should contain 'schedule[0].profile_name', got: {field}"
                );
                assert!(
                    message.contains("nonexistent"),
                    "message should mention the unknown profile name, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_empty_profiles_rejected() {
        let json = r#"{
  "method": "spectral",
  "profiles": {}
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("profiles"),
                    "field should contain 'profiles', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// An empty `method` is not one of the accepted variants, so it is rejected
    /// at parse (`ParseError`) rather than warned-and-accepted.
    #[test]
    fn test_empty_method_rejected() {
        let json = r#"{
  "method": "",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "g",
          "entities": [{ "type": "inflow", "id": 0 }],
          "matrix": [[1.0]]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for an empty method, got: {err:?}"
        );
    }

    /// An unrecognized `method` is rejected at parse (the tightening: no
    /// warn-and-continue), and the serde error names the accepted set.
    #[test]
    fn unknown_method_is_rejected_naming_accepted_set() {
        let json = r#"{
  "method": "bogus",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "g",
          "entities": [{ "type": "inflow", "id": 0 }],
          "matrix": [[1.0]]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("spectral"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    #[test]
    fn test_load_correlation_none_returns_default() {
        let result = super::super::load_correlation(None).unwrap();
        let expected = CorrelationModel::default();
        assert_eq!(result.method, expected.method);
        assert!(result.profiles.is_empty());
        assert!(result.schedule.is_empty());
    }

    #[test]
    fn test_single_entity_1x1_matrix_valid() {
        let json = r#"{
  "method": "spectral",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "solo",
          "entities": [{ "type": "inflow", "id": 5 }],
          "matrix": [[1.0]]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let model = parse_correlation(tmp.path()).unwrap();

        assert_eq!(model.profiles["default"].groups[0].entities.len(), 1);
        assert!((model.profiles["default"].groups[0].matrix[0][0] - 1.0).abs() < f64::EPSILON);
    }

    /// `method: cholesky` is now an unknown-variant load error naming the
    /// accepted set — the intended compat-break; it never drove a distinct
    /// decomposition.
    #[test]
    fn cholesky_method_is_rejected() {
        let json = r#"{
  "method": "cholesky",
  "profiles": {
    "default": {
      "correlation_groups": [
        {
          "name": "all_hydros",
          "entities": [
            { "type": "inflow", "id": 1 },
            { "type": "inflow", "id": 2 }
          ],
          "matrix": [
            [1.0, 0.5],
            [0.5, 1.0]
          ]
        }
      ]
    }
  }
}"#;
        let tmp = write_json(json);
        let err = parse_correlation(tmp.path()).unwrap_err();

        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("spectral"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }
}
