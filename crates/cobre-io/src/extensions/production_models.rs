//! Parsing for `system/hydro_production_models.json` — per-hydro production model configuration.
//!
//! [`parse_production_models`] reads `system/hydro_production_models.json` and returns a sorted
//! `Vec<ProductionModelConfig>` describing the HPF model selection for each configured hydro.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/production_models.schema.json",
//!   "production_models": [
//!     {
//!       "hydro_id": 0,
//!       "selection_mode": "stage_ranges",
//!       "stage_ranges": [
//!         {
//!           "start_stage_id": 0, "end_stage_id": 24,
//!           "model": "fpha",
//!           "fpha_config": { "source": "computed" }
//!         }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! ## Selection modes
//!
//! - **`stage_ranges`**: Each stage maps to a model via explicit `[start, end)` ranges.
//! - **`seasonal`**: Each stage maps to a model via its season index. Seasons not listed
//!   fall back to `default_model`.
//!
//! ## Output ordering
//!
//! Results are sorted by `hydro_id` ascending. Duplicate `hydro_id` values are rejected
//! as a `SchemaError`.
//!
//! ## Validation
//!
//! Per-entry constraints enforced by this parser:
//!
//! - No two entries share the same `hydro_id`.
//! - For `stage_ranges` mode: `start_stage_id <= end_stage_id` when `end_stage_id` is not null.
//! - In `fitting_window`: absolute bounds (`volume_min_hm3` / `volume_max_hm3`) and percentile
//!   bounds (`volume_min_percentile` / `volume_max_percentile`) are mutually exclusive.
//!
//! Deferred validations (not performed here):
//!
//! - `hydro_id` existence in the hydro registry — Layer 3, Epic 06.
//! - Cross-validation that `source: "precomputed"` hydros have FPHA hyperplanes — Layer 3/5.

use cobre_core::EntityId;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

/// Production model configuration for one hydro plant.
///
/// Loaded from `system/hydro_production_models.json`. Specifies how the hydro
/// production function (HPF) model is selected across stages or seasons.
///
/// # Examples
///
/// ```
/// use cobre_io::extensions::{ProductionModelConfig, SelectionMode};
/// use cobre_core::EntityId;
///
/// let config = ProductionModelConfig {
///     hydro_id: EntityId::from(0),
///     selection_mode: SelectionMode::StageRanges {
///         ranges: vec![],
///     },
/// };
/// assert_eq!(config.hydro_id, EntityId::from(0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProductionModelConfig {
    /// Hydro plant this configuration applies to.
    pub hydro_id: EntityId,
    /// How the model variant is selected for each stage.
    pub selection_mode: SelectionMode,
}

/// Model selection strategy for a hydro plant.
///
/// The two variants are mutually exclusive within a single hydro entry.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionMode {
    /// Models are selected by stage ID ranges.
    StageRanges {
        /// Ordered list of stage range descriptors.
        ranges: Vec<StageRange>,
    },
    /// Models are selected by season index, with a fallback default.
    Seasonal {
        /// Fallback model for seasons not listed in `seasons`.
        default_model: String,
        /// Season-specific overrides.
        seasons: Vec<SeasonConfig>,
    },
}

/// A stage range descriptor for the `stage_ranges` selection mode.
#[derive(Debug, Clone, PartialEq)]
pub struct StageRange {
    /// First stage (inclusive) to which this entry applies.
    pub start_stage_id: i32,
    /// Last stage (inclusive) to which this entry applies. `None` means "until end of horizon".
    pub end_stage_id: Option<i32>,
    /// Model name: `"constant_productivity"`, `"linearized_head"`, or `"fpha"`.
    pub model: String,
    /// FPHA configuration, required when `model == "fpha"`.
    pub fpha_config: Option<FphaColumnLayout>,
    /// Optional productivity override (MW per m3/s). When `Some`, this value
    /// replaces the entity's base `productivity_mw_per_m3s` for this stage range.
    /// Only valid for `"constant_productivity"` and `"linearized_head"` models.
    /// Must be positive when present.
    pub productivity_override: Option<f64>,
}

/// A season-specific model descriptor for the `seasonal` selection mode.
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonConfig {
    /// Season index (0-based, matching `stages.json` season map).
    pub season_id: i32,
    /// Model name: `"constant_productivity"`, `"linearized_head"`, or `"fpha"`.
    pub model: String,
    /// FPHA configuration, required when `model == "fpha"`.
    pub fpha_config: Option<FphaColumnLayout>,
    /// Optional productivity override (MW per m3/s). When `Some`, this value
    /// replaces the entity's base `productivity_mw_per_m3s` for this season.
    /// Only valid for `"constant_productivity"` and `"linearized_head"` models.
    /// Must be positive when present.
    pub productivity_override: Option<f64>,
}

/// Configuration for the FPHA production function model.
#[derive(Debug, Clone, PartialEq)]
pub struct FphaColumnLayout {
    /// `"computed"` (fit from topology) or `"precomputed"` (from `fpha_hyperplanes.parquet`).
    pub source: String,
    /// Number of volume discretization points used when computing hyperplanes.
    pub volume_discretization_points: Option<i32>,
    /// Number of turbine flow discretization points used when computing hyperplanes.
    pub turbine_discretization_points: Option<i32>,
    /// Number of spillage discretization points used when computing hyperplanes.
    pub spillage_discretization_points: Option<i32>,
    /// Maximum number of planes per hydro after heuristic selection.
    pub max_planes_per_hydro: Option<i32>,
    /// Optional fitting window restricting the volume range for hyperplane computation.
    pub fitting_window: Option<FittingWindow>,
}

/// Volume fitting window for computed FPHA hyperplanes.
///
/// Absolute bounds (`volume_min_hm3` / `volume_max_hm3`) and percentile bounds
/// (`volume_min_percentile` / `volume_max_percentile`) are mutually exclusive.
#[derive(Debug, Clone, PartialEq)]
pub struct FittingWindow {
    /// Explicit minimum volume for fitting (hm³). Mutually exclusive with `volume_min_percentile`.
    pub volume_min_hm3: Option<f64>,
    /// Explicit maximum volume for fitting (hm³). Mutually exclusive with `volume_max_percentile`.
    pub volume_max_hm3: Option<f64>,
    /// Minimum as percentile of the operating range. Mutually exclusive with `volume_min_hm3`.
    pub volume_min_percentile: Option<f64>,
    /// Maximum as percentile of the operating range. Mutually exclusive with `volume_max_hm3`.
    pub volume_max_percentile: Option<f64>,
}

/// Per-hydro production model configuration loaded from
/// `system/hydro_production_models.json`.
///
/// Specifies how the hydro production function (HPF) model variant is selected
/// for each stage or season. Two selection modes are supported:
///
/// - `stage_ranges`: maps each stage to a model via explicit `[start, end]`
///   intervals.
/// - `seasonal`: maps each stage to a model via its season index, with a
///   fallback default.
///
/// Each hydro may appear at most once. Results are sorted by `hydro_id`
/// ascending.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
pub(crate) struct RawProductionModelFile {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of per-hydro production model configurations. Each `hydro_id`
    /// must be unique.
    production_models: Vec<RawProductionModel>,
}

/// Production model configuration for one hydro plant.
///
/// The `selection_mode` field discriminates between two layouts:
/// `stage_ranges` carries a stage-range array, while `seasonal` carries a
/// `default_model` plus a `seasons` override list.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawProductionModel {
    /// Hydro plant identifier. Must be unique within the file.
    hydro_id: i32,

    /// Tagged-union payload for the model selection.
    #[serde(flatten)]
    selection: RawSelectionMode,
}

/// Model selection layout for a hydro plant, discriminated by the
/// `selection_mode` JSON field.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(tag = "selection_mode", rename_all = "snake_case")]
enum RawSelectionMode {
    /// Stage-range selection: each stage maps to a model via explicit
    /// `[start, end]` ranges.
    StageRanges {
        /// Ordered list of stage range descriptors.
        stage_ranges: Vec<RawStageRange>,
    },
    /// Seasonal selection: each stage maps to a model via its season index.
    Seasonal {
        /// Fallback model for seasons not listed in `seasons`. One of
        /// `"constant_productivity"`, `"linearized_head"`, or `"fpha"`.
        default_model: String,
        /// Season-specific model overrides.
        seasons: Vec<RawSeasonConfig>,
    },
}

/// Stage range descriptor for the `stage_ranges` selection mode.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawStageRange {
    /// First stage (inclusive) to which this entry applies. Must be <=
    /// `end_stage_id` when `end_stage_id` is set.
    start_stage_id: i32,
    /// Last stage (inclusive) to which this entry applies. `null` = until end
    /// of horizon.
    end_stage_id: Option<i32>,
    /// Model name: `"constant_productivity"`, `"linearized_head"`, or `"fpha"`.
    model: String,
    /// FPHA configuration. Required when `model` is `"fpha"`. Absent or null
    /// otherwise.
    fpha_config: Option<RawFphaColumnLayout>,
    /// Optional productivity override [MW/(m³/s)]. When present, replaces the
    /// entity's base `productivity_mw_per_m3s` for this stage range. Only
    /// valid for `"constant_productivity"` and `"linearized_head"` models.
    /// Must be positive when present.
    productivity_override: Option<f64>,
}

/// Season-specific model descriptor for the `seasonal` selection mode.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawSeasonConfig {
    /// Season index (0-based, matching the `stages.json` season map).
    season_id: i32,
    /// Model name: `"constant_productivity"`, `"linearized_head"`, or `"fpha"`.
    model: String,
    /// FPHA configuration. Required when `model` is `"fpha"`. Absent or null
    /// otherwise.
    fpha_config: Option<RawFphaColumnLayout>,
    /// Optional productivity override [MW/(m³/s)]. When present, replaces the
    /// entity's base `productivity_mw_per_m3s` for this season. Only valid
    /// for `"constant_productivity"` and `"linearized_head"` models. Must be
    /// positive when present.
    productivity_override: Option<f64>,
}

/// Configuration for the FPHA production function model.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
struct RawFphaColumnLayout {
    /// Hyperplane source: `"computed"` (fit from topology) or
    /// `"precomputed"` (from `fpha_hyperplanes.parquet`).
    source: String,
    /// Number of volume discretization points used when computing hyperplanes.
    /// Absent = algorithm default (5).
    volume_discretization_points: Option<i32>,
    /// Number of turbine flow discretization points used when computing
    /// hyperplanes. Absent = algorithm default (5).
    turbine_discretization_points: Option<i32>,
    /// Number of spillage discretization points used when computing
    /// hyperplanes. Absent = algorithm default (5).
    spillage_discretization_points: Option<i32>,
    /// Maximum number of planes per hydro after heuristic selection. Absent =
    /// algorithm default (10).
    max_planes_per_hydro: Option<i32>,
    /// Optional volume fitting window for hyperplane computation. Absent or
    /// null = full operating range.
    fitting_window: Option<RawFittingWindow>,
}

/// Volume fitting window restricting the range used for FPHA hyperplane
/// computation.
///
/// Absolute bounds (`volume_min_hm3` / `volume_max_hm3`) and percentile bounds
/// (`volume_min_percentile` / `volume_max_percentile`) are mutually exclusive:
/// set one pair or the other, not both for the same bound.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[allow(clippy::struct_field_names)]
#[derive(Deserialize)]
struct RawFittingWindow {
    /// Explicit minimum volume for fitting [hm³]. Mutually exclusive with
    /// `volume_min_percentile`.
    volume_min_hm3: Option<f64>,
    /// Explicit maximum volume for fitting [hm³]. Mutually exclusive with
    /// `volume_max_percentile`.
    volume_max_hm3: Option<f64>,
    /// Minimum as a percentile of the operating range. Mutually exclusive
    /// with `volume_min_hm3`.
    volume_min_percentile: Option<f64>,
    /// Maximum as a percentile of the operating range. Mutually exclusive
    /// with `volume_max_hm3`.
    volume_max_percentile: Option<f64>,
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parse `system/hydro_production_models.json` and return a sorted list of
/// per-hydro production model configurations.
///
/// Reads the JSON file, deserializes through intermediate serde types, validates
/// all invariants, then returns results sorted by `hydro_id` ascending.
///
/// # Errors
///
/// | Condition                                                   | Error variant              |
/// |------------------------------------------------------------ |--------------------------- |
/// | File not found or permission denied                         | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or unrecognised `selection_mode`        | [`LoadError::ParseError`] / [`LoadError::SchemaError`] |
/// | Duplicate `hydro_id`                                        | [`LoadError::SchemaError`] |
/// | `start_stage_id > end_stage_id` (when `end_stage_id` set)  | [`LoadError::SchemaError`] |
/// | Both absolute and percentile fitting bounds set             | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::extensions::parse_production_models;
/// use std::path::Path;
///
/// let models = parse_production_models(Path::new("system/hydro_production_models.json"))
///     .expect("valid production models file");
/// println!("loaded {} hydro model configs", models.len());
/// ```
pub fn parse_production_models(path: &Path) -> Result<Vec<ProductionModelConfig>, LoadError> {
    // Step 1: Read file.
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    // Step 2: Deserialize. Unrecognised `selection_mode` produces a serde error.
    let raw: RawProductionModelFile = serde_json::from_str(&raw_text).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("unknown variant") {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: "selection_mode".to_string(),
                message: msg,
            }
        } else {
            LoadError::parse(path, msg)
        }
    })?;

    // Step 3: Validate cross-entry constraints.
    validate_production_models(&raw.production_models, path)?;

    // Step 4: Convert and sort.
    let mut configs: Vec<ProductionModelConfig> = raw
        .production_models
        .into_iter()
        .map(convert_production_model)
        .collect();

    configs.sort_by_key(|c| c.hydro_id.0);

    Ok(configs)
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all cross-entry and per-entry constraints on raw production model data.
fn validate_production_models(models: &[RawProductionModel], path: &Path) -> Result<(), LoadError> {
    let mut seen_ids: HashSet<i32> = HashSet::new();

    for (entry_idx, model) in models.iter().enumerate() {
        // Duplicate hydro_id check.
        if !seen_ids.insert(model.hydro_id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("production_models[{entry_idx}].hydro_id"),
                message: format!(
                    "duplicate hydro_id {} — each hydro may appear at most once",
                    model.hydro_id
                ),
            });
        }

        // Mode-specific validation.
        match &model.selection {
            RawSelectionMode::StageRanges { stage_ranges } => {
                for (range_idx, range) in stage_ranges.iter().enumerate() {
                    validate_stage_range(range, entry_idx, range_idx, path)?;
                }
            }
            RawSelectionMode::Seasonal { seasons, .. } => {
                for (season_idx, season) in seasons.iter().enumerate() {
                    // Reject productivity_override on FPHA seasons.
                    if season.model == "fpha" && season.productivity_override.is_some() {
                        return Err(LoadError::SchemaError {
                            path: path.to_path_buf(),
                            field: format!(
                                "production_models[{entry_idx}].seasons[{season_idx}].productivity_override"
                            ),
                            message: "productivity_override is not valid for model \"fpha\""
                                .to_string(),
                        });
                    }

                    // Reject non-positive productivity_override.
                    if let Some(val) = season.productivity_override {
                        if val <= 0.0 {
                            return Err(LoadError::SchemaError {
                                path: path.to_path_buf(),
                                field: format!(
                                    "production_models[{entry_idx}].seasons[{season_idx}].productivity_override"
                                ),
                                message: format!(
                                    "productivity_override must be positive, got {val}"
                                ),
                            });
                        }
                    }

                    if let Some(cfg) = &season.fpha_config {
                        validate_fitting_window(
                            cfg,
                            &format!(
                                "production_models[{entry_idx}].seasons[{season_idx}].fpha_config.fitting_window"
                            ),
                            path,
                        )?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Validate one stage range descriptor.
fn validate_stage_range(
    range: &RawStageRange,
    entry_idx: usize,
    range_idx: usize,
    path: &Path,
) -> Result<(), LoadError> {
    // start_stage_id must not exceed end_stage_id.
    if let Some(end) = range.end_stage_id {
        if range.start_stage_id > end {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!(
                    "production_models[{entry_idx}].stage_ranges[{range_idx}].start_stage_id"
                ),
                message: format!(
                    "stage_ranges entry has start_stage_id ({}) > end_stage_id ({}); \
                     start_stage_id must be <= end_stage_id",
                    range.start_stage_id, end
                ),
            });
        }
    }

    // Reject productivity_override on FPHA stages.
    if range.model == "fpha" && range.productivity_override.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!(
                "production_models[{entry_idx}].stage_ranges[{range_idx}].productivity_override"
            ),
            message: "productivity_override is not valid for model \"fpha\"".to_string(),
        });
    }

    // Reject non-positive productivity_override.
    if let Some(val) = range.productivity_override {
        if val <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!(
                    "production_models[{entry_idx}].stage_ranges[{range_idx}].productivity_override"
                ),
                message: format!("productivity_override must be positive, got {val}"),
            });
        }
    }

    // Validate fitting_window if present.
    if let Some(cfg) = &range.fpha_config {
        validate_fitting_window(
            cfg,
            &format!(
                "production_models[{entry_idx}].stage_ranges[{range_idx}].fpha_config.fitting_window"
            ),
            path,
        )?;
    }

    Ok(())
}

/// Validate the mutually-exclusive fitting window bounds.
///
/// The spec states: use absolute bounds (`volume_min_hm3`, `volume_max_hm3`) OR
/// percentiles (`volume_min_percentile`, `volume_max_percentile`), not both.
fn validate_fitting_window(
    cfg: &RawFphaColumnLayout,
    field_prefix: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let Some(fw) = &cfg.fitting_window else {
        return Ok(());
    };

    if fw.volume_min_hm3.is_some() && fw.volume_min_percentile.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: "mutually exclusive bounds: volume_min_hm3 and volume_min_percentile \
                      cannot both be set; use absolute bounds OR percentiles, not both"
                .to_string(),
        });
    }

    if fw.volume_max_hm3.is_some() && fw.volume_max_percentile.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field_prefix.to_string(),
            message: "mutually exclusive bounds: volume_max_hm3 and volume_max_percentile \
                      cannot both be set; use absolute bounds OR percentiles, not both"
                .to_string(),
        });
    }

    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert a validated raw production model entry into the public type.
fn convert_production_model(raw: RawProductionModel) -> ProductionModelConfig {
    let selection_mode = match raw.selection {
        RawSelectionMode::StageRanges { stage_ranges } => SelectionMode::StageRanges {
            ranges: stage_ranges.into_iter().map(convert_stage_range).collect(),
        },
        RawSelectionMode::Seasonal {
            default_model,
            seasons,
        } => SelectionMode::Seasonal {
            default_model,
            seasons: seasons.into_iter().map(convert_season_config).collect(),
        },
    };

    ProductionModelConfig {
        hydro_id: EntityId::from(raw.hydro_id),
        selection_mode,
    }
}

fn convert_stage_range(raw: RawStageRange) -> StageRange {
    StageRange {
        start_stage_id: raw.start_stage_id,
        end_stage_id: raw.end_stage_id,
        model: raw.model,
        fpha_config: raw.fpha_config.map(convert_fpha_column_layout),
        productivity_override: raw.productivity_override,
    }
}

fn convert_season_config(raw: RawSeasonConfig) -> SeasonConfig {
    SeasonConfig {
        season_id: raw.season_id,
        model: raw.model,
        fpha_config: raw.fpha_config.map(convert_fpha_column_layout),
        productivity_override: raw.productivity_override,
    }
}

fn convert_fpha_column_layout(raw: RawFphaColumnLayout) -> FphaColumnLayout {
    FphaColumnLayout {
        source: raw.source,
        volume_discretization_points: raw.volume_discretization_points,
        turbine_discretization_points: raw.turbine_discretization_points,
        spillage_discretization_points: raw.spillage_discretization_points,
        max_planes_per_hydro: raw.max_planes_per_hydro,
        fitting_window: raw.fitting_window.map(|fw| convert_fitting_window(&fw)),
    }
}

fn convert_fitting_window(raw: &RawFittingWindow) -> FittingWindow {
    FittingWindow {
        volume_min_hm3: raw.volume_min_hm3,
        volume_max_hm3: raw.volume_max_hm3,
        volume_min_percentile: raw.volume_min_percentile,
        volume_max_percentile: raw.volume_max_percentile,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::match_wildcard_for_single_variants,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // ── AC: valid stage_ranges mode ───────────────────────────────────────────

    /// Given a valid file with one hydro using `stage_ranges` mode, returns Ok with
    /// one entry containing the correct SelectionMode variant.
    #[test]
    fn test_valid_stage_ranges_mode() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "fpha",
                "fpha_config": {
                  "source": "computed",
                  "volume_discretization_points": 7,
                  "turbine_discretization_points": 15,
                  "fitting_window": { "volume_min_hm3": null, "volume_max_hm3": null }
                }
              },
              { "start_stage_id": 25, "end_stage_id": null, "model": "constant_productivity" }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();

        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.hydro_id, EntityId::from(0));
        match &m.selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert_eq!(ranges.len(), 2);
                assert_eq!(ranges[0].start_stage_id, 0);
                assert_eq!(ranges[0].end_stage_id, Some(24));
                assert_eq!(ranges[0].model, "fpha");
                let fpha = ranges[0].fpha_config.as_ref().unwrap();
                assert_eq!(fpha.source, "computed");
                assert_eq!(fpha.volume_discretization_points, Some(7));
                assert_eq!(fpha.turbine_discretization_points, Some(15));
                // Fitting window present but both bounds null
                let fw = fpha.fitting_window.as_ref().unwrap();
                assert!(fw.volume_min_hm3.is_none());
                assert!(fw.volume_max_hm3.is_none());

                assert_eq!(ranges[1].start_stage_id, 25);
                assert!(ranges[1].end_stage_id.is_none());
                assert_eq!(ranges[1].model, "constant_productivity");
                assert!(ranges[1].fpha_config.is_none());
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    // ── AC: valid seasonal mode ───────────────────────────────────────────────

    /// Given a valid file with one hydro using `seasonal` mode, returns Ok with one
    /// entry containing the correct SelectionMode variant with default_model and seasons.
    #[test]
    fn test_valid_seasonal_mode() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 5,
            "selection_mode": "seasonal",
            "default_model": "linearized_head",
            "seasons": [
              {
                "season_id": 0,
                "model": "fpha",
                "fpha_config": { "source": "computed", "volume_discretization_points": 5 }
              },
              { "season_id": 1, "model": "fpha",
                "fpha_config": { "source": "computed", "turbine_discretization_points": 10 }
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();

        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.hydro_id, EntityId::from(5));
        match &m.selection_mode {
            SelectionMode::Seasonal {
                default_model,
                seasons,
            } => {
                assert_eq!(default_model, "linearized_head");
                assert_eq!(seasons.len(), 2);
                assert_eq!(seasons[0].season_id, 0);
                assert_eq!(seasons[0].model, "fpha");
                let fpha0 = seasons[0].fpha_config.as_ref().unwrap();
                assert_eq!(fpha0.source, "computed");
                assert_eq!(fpha0.volume_discretization_points, Some(5));
                assert!(fpha0.turbine_discretization_points.is_none());

                assert_eq!(seasons[1].season_id, 1);
                let fpha1 = seasons[1].fpha_config.as_ref().unwrap();
                assert_eq!(fpha1.turbine_discretization_points, Some(10));
                assert!(fpha1.volume_discretization_points.is_none());
            }
            other => panic!("expected Seasonal, got: {other:?}"),
        }
    }

    // ── AC: mixed — one stage_ranges, one seasonal ────────────────────────────

    /// Given a valid file with one hydro in stage_ranges mode and one in seasonal mode,
    /// returns Ok with 2 entries sorted by hydro_id.
    #[test]
    fn test_mixed_modes_sorted_by_hydro_id() {
        let json = r#"{
          "production_models": [
            {
              "hydro_id": 10,
              "selection_mode": "seasonal",
              "default_model": "constant_productivity",
              "seasons": []
            },
            {
              "hydro_id": 3,
              "selection_mode": "stage_ranges",
              "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }
              ]
            }
          ]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();

        assert_eq!(models.len(), 2);
        // Sorted by hydro_id ascending
        assert_eq!(models[0].hydro_id, EntityId::from(3));
        assert_eq!(models[1].hydro_id, EntityId::from(10));
        assert!(matches!(
            models[0].selection_mode,
            SelectionMode::StageRanges { .. }
        ));
        assert!(matches!(
            models[1].selection_mode,
            SelectionMode::Seasonal { .. }
        ));
    }

    // ── AC: duplicate hydro_id -> SchemaError ─────────────────────────────────

    /// Duplicate hydro_id in the file -> SchemaError mentioning the duplicate.
    #[test]
    fn test_duplicate_hydro_id() {
        let json = r#"{
          "production_models": [
            {
              "hydro_id": 5,
              "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "fpha", "fpha_config": { "source": "computed" } }]
            },
            {
              "hydro_id": 5,
              "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }]
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("hydro_id"),
                    "field should mention hydro_id, got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should mention duplicate, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: invalid stage range (start > end) -> SchemaError ─────────────────

    /// stage_ranges with start_stage_id > end_stage_id -> SchemaError with
    /// field containing "stage_ranges" and message containing "start_stage_id".
    #[test]
    fn test_invalid_stage_range_start_greater_than_end() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              { "start_stage_id": 25, "end_stage_id": 10, "model": "constant_productivity" }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("stage_ranges"),
                    "field should contain 'stage_ranges', got: {field}"
                );
                assert!(
                    message.contains("start_stage_id"),
                    "message should contain 'start_stage_id', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: equal start == end is valid ───────────────────────────────────────

    /// start_stage_id == end_stage_id is valid (single-stage range).
    #[test]
    fn test_stage_range_start_equals_end_is_valid() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              { "start_stage_id": 5, "end_stage_id": 5, "model": "constant_productivity" }
            ]
          }]
        }"#;
        let f = write_json(json);
        let result = parse_production_models(f.path());
        assert!(
            result.is_ok(),
            "equal start==end should be valid, got: {result:?}"
        );
    }

    // ── AC: mutually exclusive fitting window -> SchemaError ─────────────────

    /// Both volume_min_hm3 and volume_min_percentile set -> SchemaError with
    /// message containing "mutually exclusive".
    #[test]
    fn test_mutually_exclusive_fitting_window_min() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": {
                "source": "computed",
                "fitting_window": {
                  "volume_min_hm3": 1000.0,
                  "volume_max_hm3": null,
                  "volume_min_percentile": 0.1,
                  "volume_max_percentile": null
                }
              }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("mutually exclusive"),
                    "message should contain 'mutually exclusive', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Both volume_max_hm3 and volume_max_percentile set -> SchemaError with
    /// message containing "mutually exclusive".
    #[test]
    fn test_mutually_exclusive_fitting_window_max() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": {
                "source": "computed",
                "fitting_window": {
                  "volume_min_hm3": null,
                  "volume_max_hm3": 8000.0,
                  "volume_min_percentile": null,
                  "volume_max_percentile": 0.9
                }
              }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("mutually exclusive"),
                    "message should contain 'mutually exclusive', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Both absolute and percentile set in seasonal mode -> SchemaError.
    #[test]
    fn test_mutually_exclusive_fitting_window_seasonal() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 1,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [{
              "season_id": 0,
              "model": "fpha",
              "fpha_config": {
                "source": "computed",
                "fitting_window": {
                  "volume_min_hm3": 500.0,
                  "volume_min_percentile": 0.2
                }
              }
            }]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    // ── AC: None path wrapper returns empty vec ───────────────────────────────
    // (tested in extensions/mod.rs — see load_production_models)

    // ── AC: file not found -> IoError ────────────────────────────────────────

    /// Non-existent path -> IoError.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/path/hydro_production_models.json");
        let err = parse_production_models(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: unknown selection_mode -> SchemaError ─────────────────────────────

    /// Unknown selection_mode -> SchemaError (tagged union deserialization failure).
    #[test]
    fn test_unknown_selection_mode() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "unknown_mode"
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError for unknown selection_mode, got: {err:?}"
        );
    }

    // ── AC: empty production_models array -> Ok(vec![]) ──────────────────────

    /// An empty `production_models` array deserialises to `Ok(Vec::new())`.
    #[test]
    fn test_empty_array_returns_empty_vec() {
        let json = r#"{ "production_models": [] }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();
        assert!(models.is_empty());
    }

    // ── AC: declaration-order invariance ─────────────────────────────────────

    /// Reordering the entries in the JSON does not change the output ordering.
    #[test]
    fn test_declaration_order_invariance() {
        let json_asc = r#"{
          "production_models": [
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }] },
            { "hydro_id": 5, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }] },
            { "hydro_id": 99, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }] }
          ]
        }"#;
        let json_desc = r#"{
          "production_models": [
            { "hydro_id": 99, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }] },
            { "hydro_id": 5, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }] },
            { "hydro_id": 1, "selection_mode": "stage_ranges",
              "stage_ranges": [{ "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity" }] }
          ]
        }"#;
        let f_asc = write_json(json_asc);
        let f_desc = write_json(json_desc);
        let models_asc = parse_production_models(f_asc.path()).unwrap();
        let models_desc = parse_production_models(f_desc.path()).unwrap();

        let ids_asc: Vec<i32> = models_asc.iter().map(|m| m.hydro_id.0).collect();
        let ids_desc: Vec<i32> = models_desc.iter().map(|m| m.hydro_id.0).collect();
        assert_eq!(
            ids_asc, ids_desc,
            "output order must be hydro_id-sorted regardless of input"
        );
        assert_eq!(ids_asc, vec![1, 5, 99]);
    }

    // ── AC: fpha_config without fitting_window is valid ───────────────────────

    /// FPHA config with no fitting_window field at all is valid.
    #[test]
    fn test_fpha_config_without_fitting_window() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [{
              "start_stage_id": 0, "end_stage_id": null,
              "model": "fpha",
              "fpha_config": { "source": "precomputed" }
            }]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();
        assert_eq!(models.len(), 1);
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                let fpha = ranges[0].fpha_config.as_ref().unwrap();
                assert_eq!(fpha.source, "precomputed");
                assert!(fpha.fitting_window.is_none());
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    // ── productivity_override tests ───────────────────────────────────────────

    /// Parse stage range with `productivity_override` present.
    #[test]
    fn test_productivity_override_present() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_override": 0.85
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert_eq!(ranges[0].productivity_override, Some(0.85));
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    /// Backward compatibility: missing `productivity_override` defaults to None.
    #[test]
    fn test_productivity_override_absent_defaults_to_none() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity"
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();
        match &models[0].selection_mode {
            SelectionMode::StageRanges { ranges } => {
                assert!(ranges[0].productivity_override.is_none());
            }
            other => panic!("expected StageRanges, got: {other:?}"),
        }
    }

    /// Validation rejects negative `productivity_override`.
    #[test]
    fn test_productivity_override_negative_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_override": -1.0
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    /// Validation rejects zero `productivity_override`.
    #[test]
    fn test_productivity_override_zero_rejected() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "constant_productivity",
                "productivity_override": 0.0
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    /// Validation rejects `productivity_override` on FPHA stages.
    #[test]
    fn test_productivity_override_rejected_on_fpha() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
              {
                "start_stage_id": 0, "end_stage_id": 24,
                "model": "fpha",
                "fpha_config": { "source": "computed" },
                "productivity_override": 0.5
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let err = parse_production_models(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "expected SchemaError, got: {err:?}"
        );
    }

    /// Seasonal mode with `productivity_override` parses correctly.
    #[test]
    fn test_seasonal_productivity_override() {
        let json = r#"{
          "production_models": [{
            "hydro_id": 0,
            "selection_mode": "seasonal",
            "default_model": "constant_productivity",
            "seasons": [
              {
                "season_id": 0,
                "model": "constant_productivity",
                "productivity_override": 0.75
              }
            ]
          }]
        }"#;
        let f = write_json(json);
        let models = parse_production_models(f.path()).unwrap();
        match &models[0].selection_mode {
            SelectionMode::Seasonal { seasons, .. } => {
                assert_eq!(seasons[0].productivity_override, Some(0.75));
            }
            other => panic!("expected Seasonal, got: {other:?}"),
        }
    }
}
