//! Parsing for `stages.json` — temporal structure, policy graph, and scenario source.
//!
//! [`parse_stages`] reads `stages.json` from the case directory root and returns a
//! [`StagesData`] struct containing the sorted `Vec<Stage>`, the [`PolicyGraph`], and
//! the top-level [`ScenarioSource`].
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "...",
//!   "season_definitions": {
//!     "cycle_type": "monthly",
//!     "seasons": [{ "id": 0, "month_start": 1, "label": "January" }]
//!   },
//!   "policy_graph": {
//!     "type": "finite_horizon",
//!     "annual_discount_rate": 0.06,
//!     "transitions": [{ "source_id": 0, "target_id": 1, "probability": 1.0 }]
//!   },
//!   "scenario_source": {
//!     "sampling_scheme": "in_sample",
//!     "seed": 42
//!   },
//!   "pre_study_stages": [
//!     { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" }
//!   ],
//!   "stages": [
//!     {
//!       "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
//!       "season_id": 0,
//!       "blocks": [{ "id": 0, "name": "LEVE", "hours": 744 }],
//!       "block_mode": "parallel",
//!       "state_variables": { "storage": true, "inflow_lags": false },
//!       "risk_measure": "expectation",
//!       "num_scenarios": 50,
//!       "sampling_method": "saa"
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No duplicate stage `id` values.
//! 2. No duplicate pre-study stage `id` values.
//! 3. No stage `id` collision between study and pre-study sets.
//! 4. `num_scenarios > 0` for each study stage.
//! 5. `annual_discount_rate >= 0.0`.
//! 6. Each block has `hours > 0.0`.
//! 7. Block IDs within each stage are contiguous (0, 1, ..., n-1).
//! 8. `CVaR` `alpha` in (0, 1] and `lambda` in [0, 1].
//! 9. `start_date` and `end_date` parse as ISO 8601 dates with `start_date < end_date`.
//!
//! Cross-file validation (season-stage containment, transition probability sums,
//! block hours sum equals stage duration) is deferred to Epic 06 Layer 5.

use chrono::NaiveDate;
use cobre_core::{
    scenario::{ExternalSelectionMode, SamplingScheme, ScenarioSource},
    temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig,
        SeasonCycleType, SeasonDefinition, SeasonMap, Stage, StageRiskConfig, StageStateConfig,
        Transition,
    },
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `stages.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
struct RawStagesFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Optional season definitions. Absent or null means no seasonal structure.
    #[serde(default)]
    season_definitions: Option<RawSeasonDefinitions>,

    /// Policy graph: horizon type, discount rate, and transitions.
    policy_graph: RawPolicyGraph,

    /// Top-level scenario source. Absent or null means `ScenarioSource::default()`.
    #[serde(default)]
    scenario_source: Option<RawScenarioSource>,

    /// Pre-study stages (negative IDs). Absent or null means empty.
    #[serde(default)]
    pre_study_stages: Vec<RawPreStudyStage>,

    /// Study stages (non-negative IDs).
    stages: Vec<RawStage>,
}

/// Intermediate type for the `season_definitions` sub-object.
#[derive(Deserialize)]
struct RawSeasonDefinitions {
    /// Cycle type: `"monthly"`, `"weekly"`, or `"custom"`.
    cycle_type: String,
    /// List of season entries.
    seasons: Vec<RawSeasonEntry>,
}

/// Intermediate type for one season entry.
#[derive(Deserialize)]
struct RawSeasonEntry {
    /// Season index (0-based).
    id: usize,
    /// Human-readable label.
    label: String,
    /// Calendar month where the season starts (1-12).
    month_start: u32,
    /// Calendar day where the season starts (1-31). Only for `custom` cycle.
    #[serde(default)]
    day_start: Option<u32>,
    /// Calendar month where the season ends (1-12). Only for `custom` cycle.
    #[serde(default)]
    month_end: Option<u32>,
    /// Calendar day where the season ends (1-31). Only for `custom` cycle.
    #[serde(default)]
    day_end: Option<u32>,
}

/// Intermediate type for the `policy_graph` sub-object.
#[derive(Deserialize)]
struct RawPolicyGraph {
    /// Horizon type: `"finite_horizon"` or `"cyclic"`.
    #[serde(rename = "type")]
    graph_type: String,
    /// Global annual discount rate. Must be >= 0.0.
    annual_discount_rate: f64,
    /// Stage transitions.
    #[serde(default)]
    transitions: Vec<RawTransition>,
}

/// Intermediate type for one policy graph transition.
#[derive(Deserialize)]
struct RawTransition {
    /// Source stage ID.
    source_id: i32,
    /// Target stage ID.
    target_id: i32,
    /// Transition probability.
    probability: f64,
    /// Optional per-transition discount rate override.
    #[serde(default)]
    annual_discount_rate_override: Option<f64>,
}

/// Intermediate type for the `scenario_source` sub-object.
#[derive(Deserialize)]
struct RawScenarioSource {
    /// Noise source: `"in_sample"`, `"external"`, or `"historical"`.
    sampling_scheme: String,
    /// Optional random seed.
    #[serde(default)]
    seed: Option<i64>,
    /// Selection mode for external scenarios: `"random"` or `"sequential"`.
    #[serde(default)]
    selection_mode: Option<String>,
}

/// Intermediate type for a study stage entry.
#[derive(Deserialize)]
struct RawStage {
    /// Stage identifier (non-negative for study stages).
    id: i32,
    /// Start date as ISO 8601 string.
    start_date: String,
    /// End date as ISO 8601 string.
    end_date: String,
    /// Optional season index.
    #[serde(default)]
    season_id: Option<usize>,
    /// Load blocks within this stage.
    blocks: Vec<RawBlock>,
    /// Block mode: `"parallel"` (default) or `"chronological"`.
    #[serde(default = "default_block_mode_str")]
    block_mode: String,
    /// State variable flags.
    #[serde(default)]
    state_variables: Option<RawStateVariables>,
    /// Risk measure: `"expectation"` or `{"cvar": {"alpha": ..., "lambda": ...}}`.
    #[serde(default = "default_risk_measure")]
    risk_measure: RawRiskMeasure,
    /// Number of scenarios (branching factor). Must be > 0.
    num_scenarios: u32,
    /// Sampling method for noise generation. Default: `"saa"`.
    #[serde(default = "default_sampling_method_str")]
    sampling_method: String,
}

/// Intermediate type for a pre-study stage entry (negative IDs).
#[derive(Deserialize)]
struct RawPreStudyStage {
    /// Stage identifier (negative for pre-study stages).
    id: i32,
    /// Start date as ISO 8601 string.
    start_date: String,
    /// End date as ISO 8601 string.
    end_date: String,
    /// Optional season index.
    #[serde(default)]
    season_id: Option<usize>,
}

/// Intermediate type for one load block.
#[derive(Deserialize)]
struct RawBlock {
    /// Block index (0-based within the stage). Must be contiguous.
    id: usize,
    /// Human-readable block label.
    name: String,
    /// Block duration in hours. Must be > 0.0.
    hours: f64,
}

/// Intermediate type for the `state_variables` sub-object.
#[derive(Deserialize)]
struct RawStateVariables {
    /// Whether storage is a state variable. Default: true.
    #[serde(default = "default_true")]
    storage: bool,
    /// Whether inflow lags are state variables. Default: false.
    #[serde(default)]
    inflow_lags: bool,
}

/// Intermediate untagged union for the `risk_measure` field.
///
/// The JSON value can be:
/// - A string: `"expectation"`
/// - An object: `{"cvar": {"alpha": 0.95, "lambda": 0.5}}`
///
/// `#[serde(untagged)]` tries each variant in declaration order.
/// The `Expectation` string variant must come first so it is tried before
/// the `CVaR` object variant.
///
/// The inner string of `Expectation` is not read after deserialization;
/// presence of the variant alone signals `StageRiskConfig::Expectation`.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawRiskMeasure {
    /// String variant: any string (canonically `"expectation"`).
    ///
    /// The inner `String` is only used by serde during deserialization;
    /// the actual value is not inspected in `convert_risk_measure`.
    #[allow(dead_code)]
    Expectation(String),
    /// Object variant: `{"cvar": {"alpha": ..., "lambda": ...}}`.
    CVaR {
        /// Inner `CVaR` parameters object.
        cvar: RawCVarParams,
    },
}

/// `CVaR` parameters nested inside the `cvar` key.
#[derive(Deserialize)]
struct RawCVarParams {
    /// Confidence level alpha in (0, 1].
    alpha: f64,
    /// Risk aversion weight lambda in [0, 1].
    lambda: f64,
}

// ── Default functions ─────────────────────────────────────────────────────────

fn default_block_mode_str() -> String {
    "parallel".to_string()
}

fn default_sampling_method_str() -> String {
    "saa".to_string()
}

fn default_risk_measure() -> RawRiskMeasure {
    RawRiskMeasure::Expectation("expectation".to_string())
}

fn default_true() -> bool {
    true
}

// ── Output type ───────────────────────────────────────────────────────────────

/// Parsed output from `stages.json`.
///
/// Contains all stages (study + pre-study) sorted by `id` ascending, the
/// policy graph, and the top-level scenario source configuration.
///
/// # Examples
///
/// ```no_run
/// use cobre_io::stages::parse_stages;
/// use std::path::Path;
///
/// let data = parse_stages(Path::new("case/stages.json")).unwrap();
/// assert!(!data.stages.is_empty());
/// ```
#[derive(Debug)]
pub struct StagesData {
    /// All stages (study + pre-study), sorted by `id` ascending.
    /// Pre-study stages have negative IDs; study stages have non-negative IDs.
    pub stages: Vec<Stage>,
    /// Policy graph with transitions, horizon type, discount rate, and season map.
    pub policy_graph: PolicyGraph,
    /// Top-level scenario source configuration.
    pub scenario_source: ScenarioSource,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse `stages.json` from `path` and return the fully-validated temporal structure.
///
/// Reads the JSON file, deserializes through intermediate serde types, validates
/// all invariants, then converts to [`StagesData`]. The stages vector is sorted by
/// `id` ascending to satisfy declaration-order invariance.
///
/// # Errors
///
/// | Condition | Error variant |
/// |---|---|
/// | File not found / read failure | [`LoadError::IoError`] |
/// | Malformed JSON / missing required field | [`LoadError::ParseError`] |
/// | Duplicate stage `id` | [`LoadError::SchemaError`] |
/// | Duplicate pre-study stage `id` | [`LoadError::SchemaError`] |
/// | Stage `id` collision between study and pre-study | [`LoadError::SchemaError`] |
/// | `num_scenarios` <= 0 | [`LoadError::SchemaError`] |
/// | `annual_discount_rate` < 0.0 | [`LoadError::SchemaError`] |
/// | Block `hours` <= 0.0 | [`LoadError::SchemaError`] |
/// | Block IDs not contiguous (0..n-1) | [`LoadError::SchemaError`] |
/// | `CVaR` `alpha` not in (0.0, 1.0] | [`LoadError::SchemaError`] |
/// | `CVaR` `lambda` not in [0.0, 1.0] | [`LoadError::SchemaError`] |
/// | Date parse failure | [`LoadError::SchemaError`] |
/// | `start_date >= end_date` | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::stages::parse_stages;
/// use std::path::Path;
///
/// let data = parse_stages(Path::new("case/stages.json")).unwrap();
/// println!("Loaded {} stages", data.stages.len());
/// ```
pub fn parse_stages(path: &Path) -> Result<StagesData, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawStagesFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_stages(&raw, path)?;

    convert_stages(raw, path)
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized stages data.
fn validate_raw_stages(raw: &RawStagesFile, path: &Path) -> Result<(), LoadError> {
    validate_annual_discount_rate(raw.policy_graph.annual_discount_rate, path)?;
    validate_no_duplicate_stage_ids(&raw.stages, path)?;
    validate_no_duplicate_pre_study_stage_ids(&raw.pre_study_stages, path)?;
    validate_no_id_collision_between_sets(&raw.stages, &raw.pre_study_stages, path)?;
    for (i, stage) in raw.stages.iter().enumerate() {
        validate_num_scenarios(stage.num_scenarios, i, path)?;
        for (j, block) in stage.blocks.iter().enumerate() {
            validate_block_hours(block.hours, i, j, path)?;
        }
        validate_block_ids_contiguous(&stage.blocks, i, path)?;
        validate_risk_measure(&stage.risk_measure, i, path)?;
    }
    Ok(())
}

/// Check that `annual_discount_rate >= 0.0`.
fn validate_annual_discount_rate(rate: f64, path: &Path) -> Result<(), LoadError> {
    if rate < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "policy_graph.annual_discount_rate".to_string(),
            message: format!("annual_discount_rate must be >= 0.0, got {rate}"),
        });
    }
    Ok(())
}

/// Check that no two study stages share the same `id`.
fn validate_no_duplicate_stage_ids(stages: &[RawStage], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, stage) in stages.iter().enumerate() {
        if !seen.insert(stage.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("stages[{i}].id"),
                message: format!("duplicate id {} in stages array", stage.id),
            });
        }
    }
    Ok(())
}

/// Check that no two pre-study stages share the same `id`.
fn validate_no_duplicate_pre_study_stage_ids(
    stages: &[RawPreStudyStage],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, stage) in stages.iter().enumerate() {
        if !seen.insert(stage.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("pre_study_stages[{i}].id"),
                message: format!("duplicate id {} in pre_study_stages array", stage.id),
            });
        }
    }
    Ok(())
}

/// Check that no pre-study stage `id` collides with a study stage `id`.
fn validate_no_id_collision_between_sets(
    stages: &[RawStage],
    pre_study_stages: &[RawPreStudyStage],
    path: &Path,
) -> Result<(), LoadError> {
    let study_ids: HashSet<i32> = stages.iter().map(|s| s.id).collect();
    for (i, pss) in pre_study_stages.iter().enumerate() {
        if study_ids.contains(&pss.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("pre_study_stages[{i}].id"),
                message: format!(
                    "pre_study_stages id {} collides with a study stage id",
                    pss.id
                ),
            });
        }
    }
    Ok(())
}

/// Check that `num_scenarios > 0` for a study stage.
fn validate_num_scenarios(num: u32, stage_index: usize, path: &Path) -> Result<(), LoadError> {
    if num == 0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("stages[{stage_index}].num_scenarios"),
            message: "num_scenarios must be > 0".to_string(),
        });
    }
    Ok(())
}

/// Check that block `hours > 0.0`.
fn validate_block_hours(
    hours: f64,
    stage_index: usize,
    block_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if hours <= 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("stages[{stage_index}].blocks[{block_index}].hours"),
            message: format!("block hours must be > 0.0, got {hours}"),
        });
    }
    Ok(())
}

/// Check that block IDs within a stage are contiguous (0, 1, ..., n-1).
fn validate_block_ids_contiguous(
    blocks: &[RawBlock],
    stage_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    // Collect all IDs and check they form exactly the set {0, 1, ..., n-1}.
    let n = blocks.len();
    let mut ids: Vec<usize> = blocks.iter().map(|b| b.id).collect();
    ids.sort_unstable();
    let expected: Vec<usize> = (0..n).collect();
    if ids != expected {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("stages[{stage_index}].blocks"),
            message: format!(
                "block ids must be contiguous (0..{n}), got {:?}",
                blocks.iter().map(|b| b.id).collect::<Vec<_>>()
            ),
        });
    }
    Ok(())
}

/// Check `CVaR` alpha and lambda range constraints.
fn validate_risk_measure(
    risk: &RawRiskMeasure,
    stage_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if let RawRiskMeasure::CVaR { cvar } = risk {
        if cvar.alpha <= 0.0 || cvar.alpha > 1.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("stages[{stage_index}].risk_measure.cvar.alpha"),
                message: format!("cvar alpha must be in (0.0, 1.0], got {}", cvar.alpha),
            });
        }
        if cvar.lambda < 0.0 || cvar.lambda > 1.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("stages[{stage_index}].risk_measure.cvar.lambda"),
                message: format!("cvar lambda must be in [0.0, 1.0], got {}", cvar.lambda),
            });
        }
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert validated raw stages data into [`StagesData`].
///
/// Parses dates, constructs [`Stage`] and [`Block`] instances, builds the
/// [`PolicyGraph`] and [`ScenarioSource`], then sorts all stages by `id`
/// ascending and assigns `index` after sort.
fn convert_stages(raw: RawStagesFile, path: &Path) -> Result<StagesData, LoadError> {
    let scenario_source = convert_scenario_source(raw.scenario_source, path)?;
    let season_map = convert_season_definitions(raw.season_definitions, path)?;
    let mut policy_graph = convert_policy_graph(raw.policy_graph, path)?;
    policy_graph.season_map = season_map;

    // Convert study stages.
    let mut all_stages: Vec<Stage> =
        Vec::with_capacity(raw.stages.len() + raw.pre_study_stages.len());

    for (i, raw_stage) in raw.stages.into_iter().enumerate() {
        let start_date = parse_date(
            &raw_stage.start_date,
            &format!("stages[{i}].start_date"),
            path,
        )?;
        let end_date = parse_date(&raw_stage.end_date, &format!("stages[{i}].end_date"), path)?;
        if start_date >= end_date {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("stages[{i}].end_date"),
                message: format!("end_date ({end_date}) must be after start_date ({start_date})"),
            });
        }

        let blocks = convert_blocks(&raw_stage.blocks);
        let block_mode = convert_block_mode(
            &raw_stage.block_mode,
            &format!("stages[{i}].block_mode"),
            path,
        )?;
        let state_config = convert_state_config(raw_stage.state_variables);
        let risk_config = convert_risk_measure(raw_stage.risk_measure);
        let noise_method = convert_noise_method(
            &raw_stage.sampling_method,
            &format!("stages[{i}].sampling_method"),
            path,
        )?;
        let branching_factor = raw_stage.num_scenarios as usize;

        all_stages.push(Stage {
            // index will be assigned after sort
            index: 0,
            id: raw_stage.id,
            start_date,
            end_date,
            season_id: raw_stage.season_id,
            blocks,
            block_mode,
            state_config,
            risk_config,
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method,
            },
        });
    }

    // Convert pre-study stages with minimal defaults.
    for (i, raw_pss) in raw.pre_study_stages.into_iter().enumerate() {
        let start_date = parse_date(
            &raw_pss.start_date,
            &format!("pre_study_stages[{i}].start_date"),
            path,
        )?;
        let end_date = parse_date(
            &raw_pss.end_date,
            &format!("pre_study_stages[{i}].end_date"),
            path,
        )?;
        if start_date >= end_date {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("pre_study_stages[{i}].end_date"),
                message: format!("end_date ({end_date}) must be after start_date ({start_date})"),
            });
        }

        all_stages.push(Stage {
            index: 0,
            id: raw_pss.id,
            start_date,
            end_date,
            season_id: raw_pss.season_id,
            // Pre-study stages carry no blocks, blocks are irrelevant for PAR lag init.
            blocks: vec![],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        });
    }

    // Sort by id ascending (declaration-order invariance).
    all_stages.sort_by_key(|s| s.id);

    // Assign index after sort.
    for (idx, stage) in all_stages.iter_mut().enumerate() {
        stage.index = idx;
    }

    Ok(StagesData {
        stages: all_stages,
        policy_graph,
        scenario_source,
    })
}

/// Convert the raw `policy_graph` object into a [`PolicyGraph`].
fn convert_policy_graph(raw: RawPolicyGraph, path: &Path) -> Result<PolicyGraph, LoadError> {
    let graph_type = convert_policy_graph_type(&raw.graph_type, path)?;

    let transitions: Vec<Transition> = raw
        .transitions
        .into_iter()
        .map(|t| Transition {
            source_id: t.source_id,
            target_id: t.target_id,
            probability: t.probability,
            annual_discount_rate_override: t.annual_discount_rate_override,
        })
        .collect();

    Ok(PolicyGraph {
        graph_type,
        annual_discount_rate: raw.annual_discount_rate,
        transitions,
        // season_map is set by the caller after convert_season_definitions runs.
        season_map: None,
    })
}

/// Convert the raw `scenario_source` into a [`ScenarioSource`].
fn convert_scenario_source(
    raw: Option<RawScenarioSource>,
    path: &Path,
) -> Result<ScenarioSource, LoadError> {
    match raw {
        None => Ok(ScenarioSource::default()),
        Some(r) => {
            let sampling_scheme = convert_sampling_scheme(&r.sampling_scheme, path)?;
            let selection_mode = match r.selection_mode.as_deref() {
                None => None,
                Some("random") => Some(ExternalSelectionMode::Random),
                Some("sequential") => Some(ExternalSelectionMode::Sequential),
                Some(other) => {
                    return Err(LoadError::SchemaError {
                        path: path.to_path_buf(),
                        field: "scenario_source.selection_mode".to_string(),
                        message: format!(
                            "unknown selection_mode '{other}', expected 'random' or 'sequential'"
                        ),
                    });
                }
            };
            Ok(ScenarioSource {
                sampling_scheme,
                seed: r.seed,
                selection_mode,
            })
        }
    }
}

/// Convert raw blocks into sorted `Vec<Block>` (sorted by index ascending).
fn convert_blocks(raw_blocks: &[RawBlock]) -> Vec<Block> {
    let mut blocks: Vec<Block> = raw_blocks
        .iter()
        .map(|b| Block {
            index: b.id,
            name: b.name.clone(),
            duration_hours: b.hours,
        })
        .collect();
    blocks.sort_by_key(|b| b.index);
    blocks
}

/// Convert `state_variables` raw type to [`StageStateConfig`].
fn convert_state_config(raw: Option<RawStateVariables>) -> StageStateConfig {
    match raw {
        None => StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        Some(r) => StageStateConfig {
            storage: r.storage,
            inflow_lags: r.inflow_lags,
        },
    }
}

/// Convert `risk_measure` raw type to [`StageRiskConfig`].
fn convert_risk_measure(raw: RawRiskMeasure) -> StageRiskConfig {
    match raw {
        RawRiskMeasure::Expectation(_) => StageRiskConfig::Expectation,
        RawRiskMeasure::CVaR { cvar } => StageRiskConfig::CVaR {
            alpha: cvar.alpha,
            lambda: cvar.lambda,
        },
    }
}

// ── String-to-enum converters ─────────────────────────────────────────────────

/// Convert a `block_mode` string to [`BlockMode`].
fn convert_block_mode(s: &str, field: &str, path: &Path) -> Result<BlockMode, LoadError> {
    match s {
        "parallel" => Ok(BlockMode::Parallel),
        "chronological" => Ok(BlockMode::Chronological),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field.to_string(),
            message: format!(
                "unknown block_mode '{other}', expected 'parallel' or 'chronological'"
            ),
        }),
    }
}

/// Convert a `sampling_method` string to [`NoiseMethod`].
fn convert_noise_method(s: &str, field: &str, path: &Path) -> Result<NoiseMethod, LoadError> {
    match s {
        "saa" => Ok(NoiseMethod::Saa),
        "lhs" => Ok(NoiseMethod::Lhs),
        "qmc_sobol" => Ok(NoiseMethod::QmcSobol),
        "qmc_halton" => Ok(NoiseMethod::QmcHalton),
        "selective" => Ok(NoiseMethod::Selective),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: field.to_string(),
            message: format!(
                "unknown sampling_method '{other}', expected one of: saa, lhs, qmc_sobol, qmc_halton, selective"
            ),
        }),
    }
}

/// Convert a `sampling_scheme` string to [`SamplingScheme`].
fn convert_sampling_scheme(s: &str, path: &Path) -> Result<SamplingScheme, LoadError> {
    match s {
        "in_sample" => Ok(SamplingScheme::InSample),
        "external" => Ok(SamplingScheme::External),
        "historical" => Ok(SamplingScheme::Historical),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "scenario_source.sampling_scheme".to_string(),
            message: format!(
                "unknown sampling_scheme '{other}', expected one of: in_sample, external, historical"
            ),
        }),
    }
}

/// Convert a `policy_graph.type` string to [`PolicyGraphType`].
fn convert_policy_graph_type(s: &str, path: &Path) -> Result<PolicyGraphType, LoadError> {
    match s {
        "finite_horizon" => Ok(PolicyGraphType::FiniteHorizon),
        "cyclic" => Ok(PolicyGraphType::Cyclic),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "policy_graph.type".to_string(),
            message: format!(
                "unknown policy_graph type '{other}', expected 'finite_horizon' or 'cyclic'"
            ),
        }),
    }
}

/// Convert a `cycle_type` string to [`SeasonCycleType`].
fn convert_cycle_type(s: &str, path: &Path) -> Result<SeasonCycleType, LoadError> {
    match s {
        "monthly" => Ok(SeasonCycleType::Monthly),
        "weekly" => Ok(SeasonCycleType::Weekly),
        "custom" => Ok(SeasonCycleType::Custom),
        other => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "season_definitions.cycle_type".to_string(),
            message: format!(
                "unknown cycle_type '{other}', expected 'monthly', 'weekly', or 'custom'"
            ),
        }),
    }
}

/// Parse an ISO 8601 date string (`YYYY-MM-DD`) into a [`NaiveDate`].
fn parse_date(s: &str, field: &str, path: &Path) -> Result<NaiveDate, LoadError> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|_| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: field.to_string(),
        message: format!("invalid date '{s}', expected format YYYY-MM-DD"),
    })
}

/// Convert optional season definitions into an optional [`SeasonMap`].
fn convert_season_definitions(
    raw: Option<RawSeasonDefinitions>,
    path: &Path,
) -> Result<Option<SeasonMap>, LoadError> {
    match raw {
        None => Ok(None),
        Some(raw_sd) => {
            let cycle_type = convert_cycle_type(&raw_sd.cycle_type, path)?;
            let mut seasons: Vec<SeasonDefinition> = raw_sd
                .seasons
                .into_iter()
                .map(|s| SeasonDefinition {
                    id: s.id,
                    label: s.label,
                    month_start: s.month_start,
                    day_start: s.day_start,
                    month_end: s.month_end,
                    day_end: s.day_end,
                })
                .collect();
            seasons.sort_by_key(|s| s.id);
            Ok(Some(SeasonMap {
                cycle_type,
                seasons,
            }))
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants
)]
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

    /// Canonical minimal valid `stages.json` used as a baseline for error tests.
    const VALID_JSON: &str = r#"{
      "$schema": "https://cobre.dev/schemas/v2/stages.schema.json",
      "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": [
          { "source_id": 0, "target_id": 1, "probability": 1.0 },
          { "source_id": 1, "target_id": 2, "probability": 1.0 }
        ]
      },
      "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 },
      "stages": [
        {
          "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
          "season_id": 0,
          "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
          "num_scenarios": 50
        },
        {
          "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
          "season_id": 1,
          "blocks": [{ "id": 0, "name": "LEVE", "hours": 696.0 }],
          "num_scenarios": 50
        },
        {
          "id": 2, "start_date": "2024-03-01", "end_date": "2024-04-01",
          "season_id": 2,
          "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
          "num_scenarios": 50
        }
      ]
    }"#;

    // ── AC: valid 3-stage file with 6 pre-study stages (spec example SS1.9) ───

    /// Given a valid `stages.json` with 3 study stages and 6 pre-study stages,
    /// `parse_stages` returns `Ok` with `stages.len() == 9`, sorted by `id`,
    /// and `stages[0].id == -6`.
    #[test]
    fn test_parse_valid_3_study_6_pre_study() {
        let json = r#"{
          "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.06,
            "transitions": []
          },
          "pre_study_stages": [
            { "id": -6, "start_date": "2023-07-01", "end_date": "2023-08-01" },
            { "id": -5, "start_date": "2023-08-01", "end_date": "2023-09-01" },
            { "id": -4, "start_date": "2023-09-01", "end_date": "2023-10-01" },
            { "id": -3, "start_date": "2023-10-01", "end_date": "2023-11-01" },
            { "id": -2, "start_date": "2023-11-01", "end_date": "2023-12-01" },
            { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" }
          ],
          "stages": [
            {
              "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744.0 }],
              "num_scenarios": 50
            },
            {
              "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 696.0 }],
              "num_scenarios": 50
            },
            {
              "id": 2, "start_date": "2024-03-01", "end_date": "2024-04-01",
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744.0 }],
              "num_scenarios": 50
            }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert_eq!(
            data.stages.len(),
            9,
            "expected 9 stages (3 study + 6 pre-study)"
        );

        // Sorted by id: -6, -5, -4, -3, -2, -1, 0, 1, 2
        assert_eq!(data.stages[0].id, -6, "first stage should be id -6");
        assert_eq!(data.stages[1].id, -5);
        assert_eq!(data.stages[5].id, -1);
        assert_eq!(data.stages[6].id, 0);
        assert_eq!(data.stages[8].id, 2);

        // Index must match position after sort
        for (i, stage) in data.stages.iter().enumerate() {
            assert_eq!(stage.index, i, "stage index must match sort position");
        }

        // Pre-study stage defaults: empty blocks, Parallel, storage=true, inflow_lags=false
        let pss = &data.stages[0];
        assert!(
            pss.blocks.is_empty(),
            "pre-study stage should have empty blocks"
        );
        assert_eq!(pss.block_mode, BlockMode::Parallel);
        assert!(pss.state_config.storage);
        assert!(!pss.state_config.inflow_lags);
        assert_eq!(pss.risk_config, StageRiskConfig::Expectation);
        assert_eq!(pss.scenario_config.branching_factor, 1);
        assert_eq!(pss.scenario_config.noise_method, NoiseMethod::Saa);
    }

    // ── AC: CVaR risk measure ─────────────────────────────────────────────────

    /// Given a `stages.json` with `risk_measure: {"cvar": {"alpha": 0.95, "lambda": 0.5}}`,
    /// the corresponding stage has `risk_config == StageRiskConfig::CVaR { alpha: 0.95, lambda: 0.5 }`.
    #[test]
    fn test_parse_cvar_risk_measure() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
            "num_scenarios": 50,
            "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 0.5 } }
          }]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert_eq!(data.stages.len(), 1);
        match data.stages[0].risk_config {
            StageRiskConfig::CVaR { alpha, lambda } => {
                assert!(
                    (alpha - 0.95).abs() < f64::EPSILON,
                    "alpha: expected 0.95, got {alpha}"
                );
                assert!(
                    (lambda - 0.5).abs() < f64::EPSILON,
                    "lambda: expected 0.5, got {lambda}"
                );
            }
            other => panic!("expected CVaR, got {other:?}"),
        }
    }

    // ── AC: scenario_source with in_sample and seed ───────────────────────────

    /// Given `scenario_source: {"sampling_scheme": "in_sample", "seed": 42}`,
    /// `result.scenario_source.sampling_scheme == InSample` and `seed == Some(42)`.
    #[test]
    fn test_parse_scenario_source_in_sample_with_seed() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "scenario_source": { "sampling_scheme": "in_sample", "seed": 42 },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
            "num_scenarios": 50
          }]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert_eq!(
            data.scenario_source.sampling_scheme,
            SamplingScheme::InSample
        );
        assert_eq!(data.scenario_source.seed, Some(42));
        assert!(data.scenario_source.selection_mode.is_none());
    }

    // ── AC: scenario_source external with selection_mode ─────────────────────

    /// Given `scenario_source: {"sampling_scheme": "external", "selection_mode": "sequential"}`,
    /// `result.scenario_source` has `External` and `Sequential`.
    #[test]
    fn test_parse_scenario_source_external_sequential() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "scenario_source": { "sampling_scheme": "external", "selection_mode": "sequential" },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
            "num_scenarios": 50
          }]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert_eq!(
            data.scenario_source.sampling_scheme,
            SamplingScheme::External
        );
        assert_eq!(
            data.scenario_source.selection_mode,
            Some(ExternalSelectionMode::Sequential)
        );
    }

    // ── AC: season_definitions with 12 monthly seasons ────────────────────────

    /// Given `season_definitions` with 12 monthly seasons,
    /// `result.policy_graph.season_map` is `Some(SeasonMap { Monthly, ... })` with 12 seasons.
    #[test]
    fn test_parse_season_definitions_12_monthly() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "season_definitions": {
            "cycle_type": "monthly",
            "seasons": [
              { "id": 0, "month_start": 1, "label": "January" },
              { "id": 1, "month_start": 2, "label": "February" },
              { "id": 2, "month_start": 3, "label": "March" },
              { "id": 3, "month_start": 4, "label": "April" },
              { "id": 4, "month_start": 5, "label": "May" },
              { "id": 5, "month_start": 6, "label": "June" },
              { "id": 6, "month_start": 7, "label": "July" },
              { "id": 7, "month_start": 8, "label": "August" },
              { "id": 8, "month_start": 9, "label": "September" },
              { "id": 9, "month_start": 10, "label": "October" },
              { "id": 10, "month_start": 11, "label": "November" },
              { "id": 11, "month_start": 12, "label": "December" }
            ]
          },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
            "num_scenarios": 50
          }]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        let season_map = data
            .policy_graph
            .season_map
            .expect("expected season_map to be Some");
        assert_eq!(season_map.cycle_type, SeasonCycleType::Monthly);
        assert_eq!(season_map.seasons.len(), 12);
        assert_eq!(season_map.seasons[0].label, "January");
        assert_eq!(season_map.seasons[11].label, "December");
    }

    // ── AC: no season_definitions -> season_map is None ───────────────────────

    /// Given a `stages.json` without `season_definitions`, `policy_graph.season_map` is `None`.
    #[test]
    fn test_no_season_definitions_gives_none_season_map() {
        let f = write_json(VALID_JSON);
        let data = parse_stages(f.path()).unwrap();
        assert!(
            data.policy_graph.season_map.is_none(),
            "season_map should be None when season_definitions is absent"
        );
    }

    // ── AC: chronological block_mode ──────────────────────────────────────────

    /// Given a stage with `block_mode: "chronological"`, that stage has `BlockMode::Chronological`.
    #[test]
    fn test_parse_chronological_block_mode() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [
              { "id": 0, "name": "PEAK", "hours": 248.0 },
              { "id": 1, "name": "OFF", "hours": 496.0 }
            ],
            "block_mode": "chronological",
            "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert_eq!(data.stages[0].block_mode, BlockMode::Chronological);
        assert_eq!(data.stages[0].blocks.len(), 2);
    }

    // ── AC: per-transition discount rate override ─────────────────────────────

    /// Given a transition with `annual_discount_rate_override: 0.08`,
    /// the transition struct carries `annual_discount_rate_override == Some(0.08)`.
    #[test]
    fn test_parse_transition_discount_rate_override() {
        let json = r#"{
          "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.06,
            "transitions": [
              { "source_id": 0, "target_id": 1, "probability": 1.0, "annual_discount_rate_override": 0.08 }
            ]
          },
          "stages": [
            {
              "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
              "num_scenarios": 50
            },
            {
              "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "LEVE", "hours": 696.0 }],
              "num_scenarios": 50
            }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert_eq!(data.policy_graph.transitions.len(), 1);
        let override_rate = data.policy_graph.transitions[0].annual_discount_rate_override;
        let expected = Some(0.08);
        match override_rate {
            Some(r) => assert!(
                (r - 0.08).abs() < f64::EPSILON,
                "override rate expected 0.08, got {r}"
            ),
            None => panic!("expected Some(0.08), got None"),
        }
        let _ = expected;
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    /// Duplicate study stage IDs -> `SchemaError` with field containing `"stages["` and message `"duplicate"`.
    #[test]
    fn test_error_duplicate_stage_ids() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 5, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10 },
            { "id": 5, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 696.0 }], "num_scenarios": 10 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("stages["),
                    "field should contain 'stages[', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Duplicate pre-study stage IDs -> `SchemaError`.
    #[test]
    fn test_error_duplicate_pre_study_stage_ids() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "pre_study_stages": [
            { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" },
            { "id": -1, "start_date": "2023-11-01", "end_date": "2023-12-01" }
          ],
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("pre_study_stages["),
                    "field should contain 'pre_study_stages[', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Stage ID collision between study and pre-study -> `SchemaError`.
    #[test]
    fn test_error_id_collision_study_and_pre_study() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "pre_study_stages": [
            { "id": 0, "start_date": "2023-12-01", "end_date": "2024-01-01" }
          ],
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("pre_study_stages["),
                    "field should contain 'pre_study_stages[', got: {field}"
                );
                assert!(
                    message.contains("collides"),
                    "message should contain 'collides', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `num_scenarios = 0` -> `SchemaError` on `stages[i].num_scenarios`.
    #[test]
    fn test_error_num_scenarios_zero() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }],
            "num_scenarios": 0
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("num_scenarios"),
                    "field should contain 'num_scenarios', got: {field}"
                );
                assert!(
                    message.contains("> 0"),
                    "message should mention > 0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `annual_discount_rate` -> `SchemaError`.
    #[test]
    fn test_error_negative_annual_discount_rate() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": -0.01, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("annual_discount_rate"),
                    "field should contain 'annual_discount_rate', got: {field}"
                );
                assert!(
                    message.contains(">= 0"),
                    "message should mention >= 0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Block `hours = 0.0` -> `SchemaError` on `stages[i].blocks[j].hours`.
    #[test]
    fn test_error_block_hours_zero() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 0.0 }], "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("blocks[0].hours"),
                    "field should contain 'blocks[0].hours', got: {field}"
                );
                assert!(
                    message.contains("> 0"),
                    "message should mention > 0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// CVaR `alpha = 0.0` -> `SchemaError`.
    #[test]
    fn test_error_cvar_alpha_zero() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }],
            "num_scenarios": 10,
            "risk_measure": { "cvar": { "alpha": 0.0, "lambda": 0.5 } }
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("cvar.alpha"),
                    "field should contain 'cvar.alpha', got: {field}"
                );
                assert!(
                    message.contains("(0.0, 1.0]"),
                    "message should mention range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// CVaR `lambda = 1.5` -> `SchemaError`.
    #[test]
    fn test_error_cvar_lambda_out_of_range() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }],
            "num_scenarios": 10,
            "risk_measure": { "cvar": { "alpha": 0.95, "lambda": 1.5 } }
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("cvar.lambda"),
                    "field should contain 'cvar.lambda', got: {field}"
                );
                assert!(
                    message.contains("[0.0, 1.0]"),
                    "message should mention range, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `start_date >= end_date` -> `SchemaError`.
    #[test]
    fn test_error_start_date_not_before_end_date() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-02-01", "end_date": "2024-01-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("end_date"),
                    "field should contain 'end_date', got: {field}"
                );
                assert!(
                    message.contains("after start_date"),
                    "message should mention after start_date, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Non-contiguous block IDs -> `SchemaError`.
    #[test]
    fn test_error_non_contiguous_block_ids() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [
              { "id": 0, "name": "A", "hours": 248.0 },
              { "id": 2, "name": "B", "hours": 496.0 }
            ],
            "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("blocks"),
                    "field should contain 'blocks', got: {field}"
                );
                assert!(
                    message.contains("contiguous"),
                    "message should mention contiguous, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Invalid date string -> `SchemaError`.
    #[test]
    fn test_error_invalid_date_string() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "not-a-date", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("start_date"),
                    "field should contain 'start_date', got: {field}"
                );
                assert!(
                    message.contains("invalid date"),
                    "message should mention invalid date, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── Additional valid cases ────────────────────────────────────────────────

    /// No `scenario_source` in JSON -> `ScenarioSource::default()`.
    #[test]
    fn test_absent_scenario_source_uses_default() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10
          }]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();
        assert_eq!(
            data.scenario_source.sampling_scheme,
            SamplingScheme::InSample
        );
        assert!(data.scenario_source.seed.is_none());
        assert!(data.scenario_source.selection_mode.is_none());
    }

    /// Stages in reverse ID order in JSON are returned sorted ascending.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10 },
            { "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 696.0 }], "num_scenarios": 10 }
          ]
        }"#;
        let json_reversed = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 696.0 }], "num_scenarios": 10 },
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 10 }
          ]
        }"#;
        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let d1 = parse_stages(f1.path()).unwrap();
        let d2 = parse_stages(f2.path()).unwrap();
        assert_eq!(d1.stages[0].id, d2.stages[0].id);
        assert_eq!(d1.stages[1].id, d2.stages[1].id);
        assert_eq!(d1.stages[0].id, 0);
        assert_eq!(d1.stages[1].id, 1);
    }

    /// File not found -> `IoError`.
    #[test]
    fn test_file_not_found() {
        let err = parse_stages(Path::new("/nonexistent/stages.json")).unwrap_err();
        assert!(
            matches!(err, LoadError::IoError { .. }),
            "expected IoError, got: {err:?}"
        );
    }

    /// Invalid JSON -> `ParseError`.
    #[test]
    fn test_invalid_json_gives_parse_error() {
        let f = write_json(r#"{"stages": [not valid json}}"#);
        let err = parse_stages(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError, got: {err:?}"
        );
    }

    /// Cyclic policy graph type parses correctly.
    #[test]
    fn test_cyclic_policy_graph_type() {
        let json = r#"{
          "policy_graph": {
            "type": "cyclic",
            "annual_discount_rate": 0.1,
            "transitions": [
              { "source_id": 0, "target_id": 1, "probability": 1.0 },
              { "source_id": 1, "target_id": 0, "probability": 1.0 }
            ]
          },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-07-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 4344.0 }], "num_scenarios": 5 },
            { "id": 1, "start_date": "2024-07-01", "end_date": "2025-01-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 4416.0 }], "num_scenarios": 5 }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();
        assert_eq!(data.policy_graph.graph_type, PolicyGraphType::Cyclic);
    }
}
