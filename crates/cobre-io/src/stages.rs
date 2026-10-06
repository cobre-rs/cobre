//! Parsing for `stages.json` — temporal structure and policy graph.
//!
//! [`parse_stages`] reads `stages.json` from the case directory root and returns a
//! [`StagesData`] struct containing the sorted `Vec<Stage>` and the [`HorizonGraph`].
//!
//! Scenario source configuration has moved to `config.json`
//! (`training.scenario_source` / `simulation.scenario_source`). If a `stages.json`
//! still contains a `scenario_source` key, `parse_stages` returns a
//! [`LoadError::SchemaError`] directing the user to move the field.
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
//!       "num_openings": 50,
//!       "sampling_method": "saa"
//!     }
//!   ]
//! }
//! ```
//!
//! [`parse_stages`] validates the per-file invariants (see its `# Errors`).
//! Cross-file validation (season-stage containment, transition probability sums,
//! block hours sum equals stage duration) is deferred to the semantic layer.

use chrono::{Datelike, NaiveDate};
use cobre_core::HorizonGraph;
use cobre_core::temporal::{
    Block, BlockMode, Node, NoiseMethod, PolicyGraphType, SUB_PERIOD_TOLERANCE_DAYS,
    ScenarioSourceConfig, SeasonCycleType, SeasonCycles, SeasonDefinition, SeasonMap, Stage,
    StageRiskConfig, StageStateConfig, Transition,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Root object of `stages.json`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawStagesFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Optional season definitions. Absent or null means no seasonal structure.
    #[serde(default)]
    season_definitions: Option<RawSeasonDefinitions>,

    /// Policy graph: horizon type, discount rate, and transitions.
    policy_graph: RawPolicyGraph,

    /// Detection field: present when a `stages.json` still contains the old
    /// `scenario_source` key. `parse_stages` rejects it with a clear migration
    /// error directing the user to move the field to `config.json`.
    #[cfg_attr(feature = "schema", schemars(skip))]
    #[serde(default)]
    scenario_source: Option<serde_json::Value>,

    /// Pre-study stages (negative IDs). Absent or null means empty.
    #[serde(default)]
    pre_study_stages: Vec<RawPreStudyStage>,

    /// Study stages (non-negative IDs).
    stages: Vec<RawStage>,
}

/// The `season_definitions` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawSeasonDefinitions {
    /// Cycle type: `"monthly"`, `"weekly"`, or `"custom"`.
    cycle_type: String,
    /// List of season entries.
    seasons: Vec<RawSeasonEntry>,
}

/// One season entry.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawSeasonEntry {
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

/// The `policy_graph` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawPolicyGraph {
    /// Horizon type. Only `finite_horizon` is supported; `cyclic` is reserved
    /// and rejected during conversion (no engine consumer today).
    #[serde(rename = "type")]
    graph_type: RawPolicyGraphType,
    /// Global annual discount rate. Must be >= 0.0.
    annual_discount_rate: f64,
    /// Stage transitions.
    #[serde(default)]
    transitions: Vec<RawTransition>,
    /// Policy-graph nodes. Absent ⇒ the graph is a stage chain and `transitions`
    /// endpoints are stage ids.
    #[serde(default)]
    nodes: Vec<RawNode>,
}

/// One policy graph transition.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawTransition {
    /// Source endpoint: a node id when `policy_graph.nodes` is non-empty, a stage id otherwise.
    source_id: i32,
    /// Target endpoint: a node id when `policy_graph.nodes` is non-empty, a stage id otherwise.
    target_id: i32,
    /// Transition probability.
    probability: f64,
    /// Optional per-transition discount rate override.
    #[serde(default)]
    annual_discount_rate_override: Option<f64>,
}

/// One policy-graph node.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawNode {
    /// Unique node id, referenced by `transitions` endpoints when `nodes` is declared.
    id: i32,
    /// Declared study-stage id this node sits at; resolved against the study stages, never an array index.
    stage_id: i32,
    /// Per-stage external-library realization column this node carries. Required at a
    /// stage carrying a slot-occupying external class, omitted where none. `scenario_id: k`
    /// is exactly the degenerate one-element weighted opening set
    /// `[{ "scenario_id": k, "probability": 1.0 }]` (documented equivalence; the
    /// weighted-list spelling is not parsed here).
    #[serde(default)]
    scenario_id: Option<i32>,
    /// Optional human-readable label.
    #[serde(default)]
    label: Option<String>,
}

/// A study stage entry.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawStage {
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
    /// Block mode: `parallel` (default) or `chronological`.
    #[serde(default)]
    block_mode: RawBlockMode,
    /// State variable flags.
    #[serde(default)]
    state_variables: Option<RawStateVariables>,
    /// Risk measure: `"expectation"` or `{"cvar": {"alpha": ..., "lambda": ...}}`.
    #[serde(default = "default_risk_measure")]
    risk_measure: RawRiskMeasure,
    /// Number of within-node openings. Required (and `> 0`) on the chain dialect and
    /// at a generated-openings stage under `nodes[]`; rejected where a stage carries
    /// only external openings.
    #[serde(default)]
    num_openings: Option<u32>,
    /// Sampling method for noise generation. Default: `saa`.
    #[serde(default)]
    sampling_method: RawNoiseMethod,
    /// Per-stage annual discount rate override; absent uses `policy_graph.annual_discount_rate`.
    #[serde(default)]
    annual_discount_rate_override: Option<f64>,
}

// Both variants parse, so conversion rejects `cyclic` with the reserved message
// instead of serde's unknown-variant error.
/// Horizon type discriminator (`stages.json` `policy_graph.type`).
///
/// `cyclic` is reserved: the loader rejects it with a message that says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawPolicyGraphType {
    /// Acyclic stage chain with a definite end stage.
    FiniteHorizon,
    /// Infinite periodic horizon (reserved; not yet supported).
    Cyclic,
}

impl<'de> Deserialize<'de> for RawPolicyGraphType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "finite_horizon" => Ok(Self::FiniteHorizon),
            "cyclic" => Ok(Self::Cyclic),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["finite_horizon", "cyclic"],
            )),
        }
    }
}

/// Block formulation mode discriminator (`stages.json` `stages[].block_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawBlockMode {
    /// Independent sub-periods solved simultaneously.
    #[default]
    Parallel,
    /// Sequential blocks with intra-stage state transitions.
    Chronological,
}

impl<'de> Deserialize<'de> for RawBlockMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "parallel" => Ok(Self::Parallel),
            "chronological" => Ok(Self::Chronological),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["parallel", "chronological"],
            )),
        }
    }
}

/// Opening-tree noise-generation method discriminator (`stages.json`
/// `stages[].sampling_method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawNoiseMethod {
    /// Sample Average Approximation (pure Monte Carlo).
    #[default]
    Saa,
    /// Latin Hypercube Sampling.
    Lhs,
    /// Quasi-Monte Carlo with Sobol sequences.
    QmcSobol,
    /// Quasi-Monte Carlo with Halton sequences.
    QmcHalton,
    /// Selective/representative sampling.
    Selective,
    /// Historical residuals from the historical scenario library.
    HistoricalResiduals,
}

impl<'de> Deserialize<'de> for RawNoiseMethod {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "saa" => Ok(Self::Saa),
            "lhs" => Ok(Self::Lhs),
            "qmc_sobol" => Ok(Self::QmcSobol),
            "qmc_halton" => Ok(Self::QmcHalton),
            "selective" => Ok(Self::Selective),
            "historical_residuals" => Ok(Self::HistoricalResiduals),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &[
                    "saa",
                    "lhs",
                    "qmc_sobol",
                    "qmc_halton",
                    "selective",
                    "historical_residuals",
                ],
            )),
        }
    }
}

/// A pre-study stage entry (negative IDs).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawPreStudyStage {
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

/// One load block.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawBlock {
    /// Block index (0-based within the stage). Must be contiguous.
    id: usize,
    /// Human-readable block label.
    name: String,
    /// Block duration in hours. Must be > 0.0.
    hours: f64,
}

/// The `state_variables` sub-object.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawStateVariables {
    /// Whether storage is a state variable. Default: true.
    #[serde(default = "default_true")]
    storage: bool,
    /// Whether inflow lags are state variables. Default: false.
    #[serde(default)]
    inflow_lags: bool,
}

/// The `risk_measure` value, one of:
/// - the string `"expectation"`;
/// - an object such as `{"cvar": {"alpha": 0.95, "lambda": 0.5}}`.
#[derive(Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) enum RawRiskMeasure {
    // Untagged: variants are tried in declaration order; the string form stays first.
    /// String variant: any string (canonically `"expectation"`).
    // Rationale: serde's `#[serde(untagged)]` matches this variant by attempting to deserialize
    // the JSON value as a `String`; the inner field is structurally required for that match to
    // succeed. Removing it (e.g. to `Expectation`) changes the variant to a unit that serde
    // cannot match against a JSON string, breaking deserialization of all expectation configs.
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
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub(crate) struct RawCVarParams {
    /// Confidence level alpha in (0, 1].
    alpha: f64,
    /// Risk aversion weight lambda in [0, 1].
    lambda: f64,
}

// ── Default functions ─────────────────────────────────────────────────────────

fn default_risk_measure() -> RawRiskMeasure {
    RawRiskMeasure::Expectation("expectation".to_string())
}

fn default_true() -> bool {
    true
}

// ── Output type ───────────────────────────────────────────────────────────────

/// Parsed output from `stages.json`: all stages (study + pre-study) sorted by
/// `id` ascending, plus the policy graph. Scenario source configuration lives in
/// `config.json` (`parse_config`), not here.
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
    pub policy_graph: HorizonGraph,
    /// Study stage ids whose `num_openings` was declared; lets the semantic layer
    /// distinguish an absent count from a present one after conversion collapses it
    /// into `ScenarioSourceConfig::branching_factor`.
    pub openings_declared: HashSet<i32>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse and validate `stages.json` from `path` into [`StagesData`].
///
/// The stages vector is sorted by `id` ascending to satisfy declaration-order
/// invariance.
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
/// | `num_openings` present but <= 0, or absent on the chain dialect | [`LoadError::SchemaError`] |
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

    if raw.scenario_source.is_some() {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "scenario_source".to_string(),
            message: "the 'scenario_source' field has moved from stages.json to config.json \
                      (training.scenario_source / simulation.scenario_source). \
                      Remove it from stages.json."
                .to_string(),
        });
    }

    validate_raw_stages(&raw, path)?;

    convert_stages(raw, path)
}

// ── Validation ────────────────────────────────────────────────────────────────

fn validate_raw_stages(raw: &RawStagesFile, path: &Path) -> Result<(), LoadError> {
    validate_annual_discount_rate(raw.policy_graph.annual_discount_rate, path)?;
    validate_no_duplicate_ids(raw.stages.iter().map(|s| s.id), "stages", path)?;
    validate_no_duplicate_ids(
        raw.pre_study_stages.iter().map(|s| s.id),
        "pre_study_stages",
        path,
    )?;
    validate_no_id_collision_between_sets(&raw.stages, &raw.pre_study_stages, path)?;
    let nodes_declared = !raw.policy_graph.nodes.is_empty();
    for (i, stage) in raw.stages.iter().enumerate() {
        validate_num_openings(stage.num_openings, nodes_declared, i, path)?;
        for (j, block) in stage.blocks.iter().enumerate() {
            validate_block_hours(block.hours, i, j, path)?;
        }
        validate_block_ids_contiguous(&stage.blocks, i, path)?;
        validate_risk_measure(&stage.risk_measure, i, path)?;
    }
    Ok(())
}

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

fn validate_no_duplicate_ids(
    ids: impl Iterator<Item = i32>,
    array_name: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, id) in ids.enumerate() {
        if !seen.insert(id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{array_name}[{i}].id"),
                message: format!("duplicate id {id} in {array_name} array"),
            });
        }
    }
    Ok(())
}

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

/// A present `num_openings` must be `> 0`; on the chain dialect (no `nodes[]`) it is
/// also required. Under `nodes[]` an absent count is left to the semantic layer's
/// conditional-requirement rule, which reads whether the stage carries generated
/// openings.
fn validate_num_openings(
    num: Option<u32>,
    nodes_declared: bool,
    stage_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    match num {
        Some(0) => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("stages[{stage_index}].num_openings"),
            message: "num_openings must be > 0".to_string(),
        }),
        None if !nodes_declared => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("stages[{stage_index}].num_openings"),
            message: "num_openings is required".to_string(),
        }),
        _ => Ok(()),
    }
}

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

fn validate_block_ids_contiguous(
    blocks: &[RawBlock],
    stage_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
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

fn validate_risk_measure(
    risk: &RawRiskMeasure,
    stage_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if let RawRiskMeasure::Expectation(s) = risk
        && !s.eq_ignore_ascii_case("expectation")
    {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("stages[{stage_index}].risk_measure"),
            message: format!(
                "unrecognized risk measure string '{s}'; expected \"expectation\" or a CVaR object"
            ),
        });
    }
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

/// Convert validated raw stages data into [`StagesData`], sorting all stages by
/// `id` ascending. `index` is not assigned here — see [`Stage::index`].
fn convert_stages(raw: RawStagesFile, path: &Path) -> Result<StagesData, LoadError> {
    let season_map = convert_season_definitions(raw.season_definitions, path)?;

    let nodes_declared = !raw.policy_graph.nodes.is_empty();
    let openings_declared: HashSet<i32> = raw
        .stages
        .iter()
        .filter(|s| s.num_openings.is_some())
        .map(|s| s.id)
        .collect();
    // Chain dialect: the departing-edge override folds onto its source stage (exactly
    // one edge per stage); under nodes[] the per-edge spelling is rejected by the
    // semantic layer, never folded — the stage field is the only source.
    let stage_discount_rate_overrides: BTreeMap<i32, f64> = raw
        .stages
        .iter()
        .filter_map(|s| {
            let rate = s.annual_discount_rate_override.or_else(|| {
                if nodes_declared {
                    None
                } else {
                    raw.policy_graph
                        .transitions
                        .iter()
                        .find(|tr| tr.source_id == s.id)
                        .and_then(|tr| tr.annual_discount_rate_override)
                }
            });
            rate.map(|r| (s.id, r))
        })
        .collect();

    let mut policy_graph = convert_policy_graph(raw.policy_graph, path)?;
    policy_graph.season_map = season_map;
    policy_graph.stage_discount_rate_overrides = stage_discount_rate_overrides;

    let mut all_stages: Vec<Stage> =
        Vec::with_capacity(raw.stages.len() + raw.pre_study_stages.len());

    for (i, raw_stage) in raw.stages.into_iter().enumerate() {
        let (start_date, end_date) = parse_stage_date_range(
            "stages",
            i,
            &raw_stage.start_date,
            &raw_stage.end_date,
            path,
        )?;

        let blocks = convert_blocks(&raw_stage.blocks);
        let block_mode = convert_block_mode(raw_stage.block_mode);
        let state_config = convert_state_config(raw_stage.state_variables);
        let risk_config = convert_risk_measure(raw_stage.risk_measure);
        let noise_method = convert_noise_method(raw_stage.sampling_method);
        let branching_factor = raw_stage.num_openings.unwrap_or(1) as usize;
        let season_id = resolve_or_validate_season_id(
            raw_stage.season_id,
            raw_stage.id,
            start_date,
            end_date,
            policy_graph.season_map.as_ref(),
            &format!("stages[{i}].season_id"),
            path,
        )?;

        all_stages.push(Stage {
            index: 0,
            id: raw_stage.id,
            start_date,
            end_date,
            season_id,
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

    for (i, raw_pss) in raw.pre_study_stages.into_iter().enumerate() {
        let (start_date, end_date) = parse_stage_date_range(
            "pre_study_stages",
            i,
            &raw_pss.start_date,
            &raw_pss.end_date,
            path,
        )?;

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

    Ok(StagesData {
        stages: all_stages,
        policy_graph,
        openings_declared,
    })
}

fn convert_policy_graph(raw: RawPolicyGraph, path: &Path) -> Result<HorizonGraph, LoadError> {
    let graph_type = convert_policy_graph_type(raw.graph_type, path)?;

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

    let mut nodes: Vec<Node> = raw
        .nodes
        .into_iter()
        .map(|n| Node {
            id: n.id,
            stage_id: n.stage_id,
            scenario_id: n.scenario_id,
            label: n.label,
        })
        .collect();
    nodes.sort_by_key(|n| n.id);

    Ok(HorizonGraph {
        graph_type,
        annual_discount_rate: raw.annual_discount_rate,
        transitions,
        nodes,
        // season_map and stage_discount_rate_overrides are set by the caller, which
        // has the study stages and season definitions.
        stage_discount_rate_overrides: BTreeMap::new(),
        season_map: None,
    })
}

// ── Weight-vector normalization ─────────────────────────────────────────────────

/// A weight vector whose compensated sum cannot be scaled to 1.0.
enum WeightSumError {
    /// The sum is exactly zero (or `-0.0`).
    Zero,
    /// The sum is `NaN` or infinite.
    NonFinite,
}

/// Scale `weights` in place so they sum to 1.0 within 1 ULP, dividing each by the
/// Neumaier compensated sum of the slice in its given order; the caller supplies the
/// slice already in canonical order. The one weight vector normalized today is a
/// source's out-edge probabilities; within-node weighted opening vectors, when added,
/// normalize through this same routine unchanged — hence the generic slice rather than
/// an out-edge-specific signature.
///
/// # Errors
///
/// [`WeightSumError`] when the compensated sum is zero or non-finite — a vector that
/// cannot be scaled to 1.0.
fn normalize_weights(weights: &mut [f64]) -> Result<(), WeightSumError> {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for &w in weights.iter() {
        let t = sum + w;
        if sum.abs() >= w.abs() {
            compensation += (sum - t) + w;
        } else {
            compensation += (w - t) + sum;
        }
        sum = t;
    }
    let total = sum + compensation;
    if !total.is_finite() {
        return Err(WeightSumError::NonFinite);
    }
    if total == 0.0 {
        return Err(WeightSumError::Zero);
    }
    for w in weights.iter_mut() {
        *w /= total;
    }
    Ok(())
}

/// Normalize every source node's out-edge probability vector to sum to 1.0 within
/// 1 ULP, in canonical (ascending child node id) order with a compensated sum, so
/// nothing downstream re-normalizes. Sources are independent; each is scaled by its
/// own out-edge sum, stored back into [`Transition::probability`].
///
/// Runs once, after the semantic layer has rejected gross probability-sum errors, so
/// it only polishes an already-near-unit vector to exact unit and rejects the residual
/// zero/non-finite sum a tolerance gate cannot (rule 2: a `NaN` sum passes
/// `|s − 1| ≤ ε`).
///
/// # Errors
///
/// [`LoadError::SchemaError`] naming the source node whose out-edge sum is zero or
/// non-finite.
pub(crate) fn normalize_out_edge_probabilities(
    graph: &mut HorizonGraph,
    path: &Path,
) -> Result<(), LoadError> {
    let mut by_source: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (i, transition) in graph.transitions.iter().enumerate() {
        by_source.entry(transition.source_id).or_default().push(i);
    }
    for (source_id, mut edges) in by_source {
        edges.sort_by_key(|&i| graph.transitions[i].target_id);
        let mut weights: Vec<f64> = edges
            .iter()
            .map(|&i| graph.transitions[i].probability)
            .collect();
        normalize_weights(&mut weights).map_err(|err| out_edge_sum_error(source_id, &err, path))?;
        for (&i, &w) in edges.iter().zip(&weights) {
            graph.transitions[i].probability = w;
        }
    }
    Ok(())
}

fn out_edge_sum_error(source_id: i32, err: &WeightSumError, path: &Path) -> LoadError {
    let reason = match err {
        WeightSumError::Zero => "sums to zero",
        WeightSumError::NonFinite => "has a non-finite sum",
    };
    LoadError::SchemaError {
        path: path.to_path_buf(),
        field: "policy_graph.transitions".to_string(),
        message: format!(
            "outgoing transition probabilities from source {source_id} {reason}; a weight \
             vector must have a positive, finite sum to normalize to 1.0"
        ),
    }
}

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

fn convert_risk_measure(raw: RawRiskMeasure) -> StageRiskConfig {
    match raw {
        RawRiskMeasure::Expectation(_) => StageRiskConfig::Expectation,
        RawRiskMeasure::CVaR { cvar } => StageRiskConfig::CVaR {
            alpha: cvar.alpha,
            lambda: cvar.lambda,
        },
    }
}

// ── Raw-to-canonical converters ───────────────────────────────────────────────

fn convert_block_mode(mode: RawBlockMode) -> BlockMode {
    match mode {
        RawBlockMode::Parallel => BlockMode::Parallel,
        RawBlockMode::Chronological => BlockMode::Chronological,
    }
}

fn convert_noise_method(method: RawNoiseMethod) -> NoiseMethod {
    match method {
        RawNoiseMethod::Saa => NoiseMethod::Saa,
        RawNoiseMethod::Lhs => NoiseMethod::Lhs,
        RawNoiseMethod::QmcSobol => NoiseMethod::QmcSobol,
        RawNoiseMethod::QmcHalton => NoiseMethod::QmcHalton,
        RawNoiseMethod::Selective => NoiseMethod::Selective,
        RawNoiseMethod::HistoricalResiduals => NoiseMethod::HistoricalResiduals,
    }
}

fn convert_policy_graph_type(
    graph_type: RawPolicyGraphType,
    path: &Path,
) -> Result<PolicyGraphType, LoadError> {
    match graph_type {
        RawPolicyGraphType::FiniteHorizon => Ok(PolicyGraphType::FiniteHorizon),
        RawPolicyGraphType::Cyclic => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: "policy_graph.type".to_string(),
            message: "policy_graph type 'cyclic' is reserved and not yet supported; \
                      expected 'finite_horizon'"
                .to_string(),
        }),
    }
}

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

fn parse_date(s: &str, field: &str, path: &Path) -> Result<NaiveDate, LoadError> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|_| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: field.to_string(),
        message: format!("invalid date '{s}', expected format YYYY-MM-DD"),
    })
}

/// Parse and order-validate one stage's `start_date`/`end_date` pair, field-pathed
/// under `prefix[i]` (`"stages"` or `"pre_study_stages"`).
fn parse_stage_date_range(
    prefix: &str,
    i: usize,
    start_date: &str,
    end_date: &str,
    path: &Path,
) -> Result<(NaiveDate, NaiveDate), LoadError> {
    let start_date = parse_date(start_date, &format!("{prefix}[{i}].start_date"), path)?;
    let end_date = parse_date(end_date, &format!("{prefix}[{i}].end_date"), path)?;
    if start_date >= end_date {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{prefix}[{i}].end_date"),
            message: format!("end_date ({end_date}) must be after start_date ({start_date})"),
        });
    }
    Ok((start_date, end_date))
}

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
            let season_map = SeasonMap {
                cycle_type,
                seasons,
            };
            if let Some((a, b)) = SeasonCycles::new(&season_map).overlapping_pair() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: "season_definitions.seasons".into(),
                    message: format!(
                        "seasons {a} and {b} overlap within one resolution level; \
                         seasons whose spans differ by at most {SUB_PERIOD_TOLERANCE_DAYS} days \
                         form one level and must not share a calendar day"
                    ),
                });
            }
            Ok(Some(season_map))
        }
    }
}

// ── Season ID derivation ─────────────────────────────────────────────────────

/// Resolve or validate a study stage's `season_id` under the earliest-study-
/// period anchor: derive it from `start_date` via [`SeasonMap::season_for_date`]
/// when absent, or cross-check a declared id against the derivation when
/// present.
///
/// - No `season_map`: the field passes through unchanged (normally `None`).
/// - Multi-resolution `season_map` (`is_multi_resolution`): `season_for_date`
///   returns only the finest matching definition and cannot disambiguate a
///   coarse-level id, so an absent id is rejected and a present id is trusted.
/// - Single-resolution `season_map`: absent derives; a present id that
///   disagrees with the derivation is rejected UNLESS the stage is a
///   sub-period of its resolved season (see `period_width_days`) — a
///   Regime-A weekly stage sharing a coarser PAR draw (e.g.
///   `d28-decomp-weekly-monthly`) declares a grouping label, not a
///   calendar-date claim, and is trusted.
///
/// # Errors
/// Returns [`LoadError::SchemaError`] when a multi-resolution map has no
/// declared id, or when a full-period single-resolution stage declares an id
/// that disagrees with `season_for_date(start_date)`.
fn resolve_or_validate_season_id(
    raw_season_id: Option<usize>,
    stage_id: i32,
    start_date: NaiveDate,
    end_date: NaiveDate,
    season_map: Option<&SeasonMap>,
    field: &str,
    path: &Path,
) -> Result<Option<usize>, LoadError> {
    let Some(season_map) = season_map else {
        return Ok(raw_season_id);
    };

    if season_map.is_multi_resolution() {
        return match raw_season_id {
            Some(declared) => Ok(Some(declared)),
            None => Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: field.to_string(),
                message: format!(
                    "stage {stage_id}: declare an explicit season_id: a multi-resolution \
                     season_map cannot be auto-resolved"
                ),
            }),
        };
    }

    let derived = season_map.season_for_date(start_date);

    let Some(declared) = raw_season_id else {
        return Ok(derived);
    };

    let Some(derived_id) = derived else {
        return Ok(Some(declared));
    };

    if declared == derived_id {
        return Ok(Some(declared));
    }

    let duration_days = (end_date - start_date).num_days();
    let width_days = period_width_days(season_map, derived_id, start_date.year());
    if duration_days + SUB_PERIOD_TOLERANCE_DAYS <= width_days {
        return Ok(Some(declared));
    }

    Err(LoadError::SchemaError {
        path: path.to_path_buf(),
        field: field.to_string(),
        message: format!(
            "stage {stage_id} declares season_id {declared} but start_date {start_date} \
             resolves to season_id {derived_id}; declare {derived_id}, or omit season_id to \
             derive it automatically"
        ),
    })
}

/// Calendar width, in days, of the period identified by `season_id` under
/// `season_map`'s cycle: the specific month's length for `Monthly` (leap-aware,
/// using `year` — genuinely different from [`SeasonMap::resolution_level_of`]'s
/// year-independent classifier, so the `Monthly` arm stays local), always 7
/// for `Weekly`, and the matching definition's own calendar span for `Custom`
/// (delegated to `resolution_level_of`, since that arm's computation is
/// year-independent already). An unresolvable `season_id` falls back to the
/// widest plausible span (31d) so a dangling id never masquerades as a
/// sub-period stage — the cross-reference check (rule 27) rejects dangling
/// ids independently.
fn period_width_days(season_map: &SeasonMap, season_id: usize, year: i32) -> i64 {
    match season_map.cycle_type {
        SeasonCycleType::Weekly => 7,
        SeasonCycleType::Monthly => season_map
            .seasons
            .iter()
            .find(|s| s.id == season_id)
            .map_or(31, |s| days_in_month(year, s.month_start)),
        SeasonCycleType::Custom => season_map
            .resolution_level_of(season_id)
            .map_or(31, |days| i64::try_from(days).unwrap_or(366)),
    }
}

/// Number of days in `month` of `year` (1-based month, leap-aware).
fn days_in_month(year: i32, month: u32) -> i64 {
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    match (
        NaiveDate::from_ymd_opt(year, month, 1),
        NaiveDate::from_ymd_opt(next_year, next_month, 1),
    ) {
        (Some(start), Some(next)) => (next - start).num_days(),
        _ => 31,
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
    use crate::test_support::write_json;

    /// Canonical minimal valid `stages.json` used as a baseline for error tests.
    const VALID_JSON: &str = r#"{
      "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/stages.schema.json",
      "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": [
          { "source_id": 0, "target_id": 1, "probability": 1.0 },
          { "source_id": 1, "target_id": 2, "probability": 1.0 }
        ]
      },
      "stages": [
        {
          "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
          "season_id": 0,
          "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
          "num_openings": 50
        },
        {
          "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
          "season_id": 1,
          "blocks": [{ "id": 0, "name": "LEVE", "hours": 696.0 }],
          "num_openings": 50
        },
        {
          "id": 2, "start_date": "2024-03-01", "end_date": "2024-04-01",
          "season_id": 2,
          "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
          "num_openings": 50
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
              "num_openings": 50
            },
            {
              "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 696.0 }],
              "num_openings": 50
            },
            {
              "id": 2, "start_date": "2024-03-01", "end_date": "2024-04-01",
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744.0 }],
              "num_openings": 50
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
            "num_openings": 50,
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

    // ── AC: scenario_source in stages.json rejected ──────────────────────────

    /// Given a `stages.json` without `scenario_source`, `parse_stages` succeeds
    /// and returns a `StagesData` with no `scenario_source` field.
    #[test]
    fn test_stages_without_scenario_source_succeeds() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
            "num_openings": 50
          }]
        }"#;
        let f = write_json(json);
        parse_stages(f.path()).unwrap();
    }

    /// Given a `stages.json` with `"scenario_source": {"seed": 42}`,
    /// `parse_stages` returns `Err(LoadError::SchemaError)` with field
    /// `"scenario_source"` and the full migration message naming `config.json`.
    #[test]
    fn test_stages_with_scenario_source_rejected() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "scenario_source": { "seed": 42 },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "LEVE", "hours": 744.0 }],
            "num_openings": 50
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(
                    field, "scenario_source",
                    "field should be 'scenario_source', got: {field}"
                );
                assert_eq!(
                    message,
                    "the 'scenario_source' field has moved from stages.json to config.json \
                     (training.scenario_source / simulation.scenario_source). \
                     Remove it from stages.json."
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
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
            "num_openings": 50
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

    // ── AC: derive season_id from start_date (Weekly, single-resolution) ─────

    /// Given a `Weekly` single-resolution `season_map` and a stage with
    /// `season_id: None` starting in ISO week 3 (2024-01-15), when
    /// `parse_stages` runs, the derived `Stage.season_id` is `Some(2)`
    /// (week 3 -> id 2), matching `SeasonMap::season_for_date`.
    #[test]
    fn test_derive_weekly_season_id_from_start_date() {
        let seasons: String = (0..52)
            .map(|i| format!(r#"{{ "id": {i}, "month_start": 1, "label": "W{i:02}" }}"#))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{
              "policy_graph": {{ "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] }},
              "season_definitions": {{ "cycle_type": "weekly", "seasons": [{seasons}] }},
              "stages": [{{
                "id": 0, "start_date": "2024-01-15", "end_date": "2024-01-22",
                "blocks": [{{ "id": 0, "name": "SINGLE", "hours": 168.0 }}], "num_openings": 1
              }}]
            }}"#
        );
        let f = write_json(&json);
        let data = parse_stages(f.path()).unwrap();
        assert_eq!(
            data.stages[0].season_id,
            Some(2),
            "ISO week 3 (2024-01-15) must derive to season_id 2"
        );
    }

    // ── AC: declared Monthly season_id is an unchanged no-op ──────────────────

    /// Given a `Monthly` study that declares every `season_id` explicitly and
    /// consistently, when `parse_stages` runs, each parsed `Stage.season_id`
    /// equals its declared value byte-for-byte (no derivation, no rejection).
    #[test]
    fn test_declared_monthly_season_id_unchanged_no_derivation() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "season_definitions": {
            "cycle_type": "monthly",
            "seasons": [
              { "id": 0, "month_start": 1, "label": "January" },
              { "id": 1, "month_start": 2, "label": "February" },
              { "id": 2, "month_start": 3, "label": "March" }
            ]
          },
          "stages": [
            {
              "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "season_id": 0,
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744.0 }], "num_openings": 1
            },
            {
              "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "season_id": 1,
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 696.0 }], "num_openings": 1
            },
            {
              "id": 2, "start_date": "2024-03-01", "end_date": "2024-04-01",
              "season_id": 2,
              "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744.0 }], "num_openings": 1
            }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();
        assert_eq!(data.stages[0].season_id, Some(0));
        assert_eq!(data.stages[1].season_id, Some(1));
        assert_eq!(data.stages[2].season_id, Some(2));
    }

    // ── AC: a full-period mismatch is rejected ────────────────────────────────

    /// Given a single-resolution `Monthly` map and a FULL-MONTH-WIDTH stage
    /// (2024-03-01..2024-04-01, 31 days) declaring `season_id: Some(5)` (June)
    /// while its `start_date` resolves to season_id 2 (March), when
    /// `parse_stages` runs, it returns a `SchemaError` naming the stage, the
    /// declared id (5), and the derived id (2).
    #[test]
    fn test_declared_full_period_season_id_mismatch_rejected() {
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
              { "id": 5, "month_start": 6, "label": "June" }
            ]
          },
          "stages": [{
            "id": 7, "start_date": "2024-03-01", "end_date": "2024-04-01",
            "season_id": 5,
            "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744.0 }], "num_openings": 1
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("season_id"), "field: {field}");
                assert!(message.contains("stage 7"), "message: {message}");
                assert!(message.contains('5'), "message: {message}");
                assert!(message.contains('2'), "message: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: a sub-period (Regime A) mismatch is trusted, not rejected ────────

    /// Given a single-resolution `Monthly` map and a ~7-day sub-period stage
    /// (2026-05-02..2026-05-09) declaring `season_id: Some(3)` (April) even
    /// though its `start_date` resolves to season_id 4 (May) — the
    /// `d28-decomp-weekly-monthly` pattern, a Regime-A weekly stage sharing
    /// April's monthly PAR draw — when `parse_stages` runs, it succeeds and
    /// preserves the declared id unchanged.
    #[test]
    fn test_sub_period_season_id_mismatch_trusted() {
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
              { "id": 5, "month_start": 6, "label": "June" }
            ]
          },
          "stages": [{
            "id": 4, "start_date": "2026-05-02", "end_date": "2026-05-09",
            "season_id": 3,
            "blocks": [{ "id": 0, "name": "SINGLE", "hours": 168.0 }], "num_openings": 1
          }]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();
        assert_eq!(
            data.stages[0].season_id,
            Some(3),
            "a Regime-A sub-period stage keeps its declared operator-grouping season_id"
        );
    }

    // ── AC: multi-resolution Custom map (d30-shaped) ──────────────────────────

    /// `d30-multi-resolution-monthly-quarterly`-shaped `season_definitions`:
    /// 12 monthly + 4 quarterly `Custom` definitions whose calendar spans
    /// overlap by design.
    const D30_SHAPED_SEASON_DEFINITIONS: &str = r#"{
      "cycle_type": "custom",
      "seasons": [
        { "id": 0, "month_start": 1, "day_start": 1, "month_end": 1, "day_end": 31, "label": "January" },
        { "id": 1, "month_start": 2, "day_start": 1, "month_end": 2, "day_end": 28, "label": "February" },
        { "id": 2, "month_start": 3, "day_start": 1, "month_end": 3, "day_end": 31, "label": "March" },
        { "id": 3, "month_start": 4, "day_start": 1, "month_end": 4, "day_end": 30, "label": "April" },
        { "id": 4, "month_start": 5, "day_start": 1, "month_end": 5, "day_end": 31, "label": "May" },
        { "id": 5, "month_start": 6, "day_start": 1, "month_end": 6, "day_end": 30, "label": "June" },
        { "id": 6, "month_start": 7, "day_start": 1, "month_end": 7, "day_end": 31, "label": "July" },
        { "id": 7, "month_start": 8, "day_start": 1, "month_end": 8, "day_end": 31, "label": "August" },
        { "id": 8, "month_start": 9, "day_start": 1, "month_end": 9, "day_end": 30, "label": "September" },
        { "id": 9, "month_start": 10, "day_start": 1, "month_end": 10, "day_end": 31, "label": "October" },
        { "id": 10, "month_start": 11, "day_start": 1, "month_end": 11, "day_end": 30, "label": "November" },
        { "id": 11, "month_start": 12, "day_start": 1, "month_end": 12, "day_end": 31, "label": "December" },
        { "id": 12, "month_start": 7, "day_start": 1, "month_end": 9, "day_end": 30, "label": "Q3" },
        { "id": 13, "month_start": 10, "day_start": 1, "month_end": 12, "day_end": 31, "label": "Q4" },
        { "id": 14, "month_start": 1, "day_start": 1, "month_end": 3, "day_end": 31, "label": "Q1" },
        { "id": 15, "month_start": 4, "day_start": 1, "month_end": 6, "day_end": 30, "label": "Q2" }
      ]
    }"#;

    /// Given the d30-shaped multi-resolution Custom map and a quarterly stage
    /// (2024-07-01..2024-10-01, Q3) with an explicit `season_id: Some(12)`,
    /// when `parse_stages` runs, it succeeds with no error and the id is
    /// preserved unchanged (multi-resolution present-id is trusted, never
    /// cross-checked against `season_for_date`'s finest-match result).
    #[test]
    fn test_multi_resolution_explicit_id_accepted() {
        let json = format!(
            r#"{{
              "policy_graph": {{ "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] }},
              "season_definitions": {D30_SHAPED_SEASON_DEFINITIONS},
              "stages": [{{
                "id": 6, "start_date": "2024-07-01", "end_date": "2024-10-01",
                "season_id": 12,
                "blocks": [{{ "id": 0, "name": "SINGLE", "hours": 2208.0 }}], "num_openings": 1
              }}]
            }}"#
        );
        let f = write_json(&json);
        let data = parse_stages(f.path()).unwrap();
        assert_eq!(
            data.stages[0].season_id,
            Some(12),
            "a multi-resolution declared season_id must be trusted unchanged"
        );
    }

    /// Given the d30-shaped multi-resolution Custom map and a stage with
    /// `season_id: None`, when `parse_stages` runs, it returns a
    /// `SchemaError` instructing the author to declare an explicit id (never
    /// a silent finest-level guess).
    #[test]
    fn test_multi_resolution_absent_id_rejected() {
        let json = format!(
            r#"{{
              "policy_graph": {{ "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] }},
              "season_definitions": {D30_SHAPED_SEASON_DEFINITIONS},
              "stages": [{{
                "id": 6, "start_date": "2024-07-01", "end_date": "2024-10-01",
                "blocks": [{{ "id": 0, "name": "SINGLE", "hours": 2208.0 }}], "num_openings": 1
              }}]
            }}"#
        );
        let f = write_json(&json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("season_id"), "field: {field}");
                assert!(
                    message.contains("multi-resolution") && message.contains("explicit"),
                    "message should instruct declaring an explicit season_id, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Given the d30-shaped map plus a second January (`id` 16) that shares
    /// every day of monthly season 0, when `parse_stages` runs, it returns a
    /// `SchemaError` naming both seasons.
    #[test]
    fn test_overlapping_same_level_seasons_rejected() {
        let mut season_definitions: serde_json::Value =
            serde_json::from_str(D30_SHAPED_SEASON_DEFINITIONS).unwrap();
        season_definitions["seasons"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "id": 16, "month_start": 1, "day_start": 1,
                "month_end": 1, "day_end": 31, "label": "January bis"
            }));
        let json = format!(
            r#"{{
              "policy_graph": {{ "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] }},
              "season_definitions": {season_definitions},
              "stages": [{{
                "id": 6, "start_date": "2024-07-01", "end_date": "2024-10-01",
                "season_id": 12,
                "blocks": [{{ "id": 0, "name": "SINGLE", "hours": 2208.0 }}], "num_openings": 1
              }}]
            }}"#
        );
        let f = write_json(&json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "season_definitions.seasons");
                assert!(
                    message.contains("overlap within one resolution level")
                        && message.contains("seasons 0 and 16"),
                    "message should name both overlapping seasons, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
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
            "num_openings": 10
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
              "num_openings": 50
            },
            {
              "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "LEVE", "hours": 696.0 }],
              "num_openings": 50
            }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert_eq!(data.policy_graph.transitions.len(), 1);
        let override_rate = data.policy_graph.transitions[0].annual_discount_rate_override;
        match override_rate {
            Some(r) => assert!(
                (r - 0.08).abs() < f64::EPSILON,
                "override rate expected 0.08, got {r}"
            ),
            None => panic!("expected Some(0.08), got None"),
        }
    }

    /// Chain dialect (no `nodes[]`): a departing-edge `annual_discount_rate_override`
    /// folds onto its source stage in `stage_discount_rate_overrides`, so the edge
    /// spelling still applies, and a `stages[].annual_discount_rate_override`
    /// lands there directly.
    #[test]
    fn test_discount_override_folds_onto_stage_chain_dialect() {
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
              "num_openings": 5
            },
            {
              "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "LEVE", "hours": 696.0 }],
              "num_openings": 5,
              "annual_discount_rate_override": 0.09
            }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();
        let overrides = &data.policy_graph.stage_discount_rate_overrides;
        assert_eq!(
            overrides.get(&0).copied(),
            Some(0.08),
            "the departing-edge override must fold onto stage 0 (C1)"
        );
        assert_eq!(
            overrides.get(&1).copied(),
            Some(0.09),
            "the stage-field override must land on stage 1 (B2)"
        );
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    /// Duplicate study stage IDs -> `SchemaError` with field containing `"stages["` and message `"duplicate"`.
    #[test]
    fn test_error_duplicate_stage_ids() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 5, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10 },
            { "id": 5, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 696.0 }], "num_openings": 10 }
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
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10
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
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10
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

    /// `num_openings = 0` -> `SchemaError` on `stages[i].num_openings`.
    #[test]
    fn test_error_num_openings_zero() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }],
            "num_openings": 0
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("num_openings"),
                    "field should contain 'num_openings', got: {field}"
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
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10
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
            "blocks": [{ "id": 0, "name": "A", "hours": 0.0 }], "num_openings": 10
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
            "num_openings": 10,
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
            "num_openings": 10,
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
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10
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
            "num_openings": 10
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
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10
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

    /// Stages in reverse ID order in JSON are returned sorted ascending.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10 },
            { "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 696.0 }], "num_openings": 10 }
          ]
        }"#;
        let json_reversed = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 696.0 }], "num_openings": 10 },
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 10 }
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

    /// `policy_graph.type: "cyclic"` is rejected as reserved/unsupported (A4),
    /// naming the type and the accepted value.
    #[test]
    fn test_cyclic_policy_graph_type_rejected() {
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
              "blocks": [{ "id": 0, "name": "A", "hours": 4344.0 }], "num_openings": 5 },
            { "id": 1, "start_date": "2024-07-01", "end_date": "2025-01-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 4416.0 }], "num_openings": 5 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "policy_graph.type", "field: {field}");
                assert!(
                    message.contains("cyclic") && message.contains("reserved"),
                    "message should name 'cyclic' as reserved, got: {message}"
                );
                assert!(
                    message.contains("finite_horizon"),
                    "message should name the accepted value, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// An unknown `policy_graph.type` that is not the reserved `cyclic` is
    /// rejected at parse, and the serde error names the accepted set.
    #[test]
    fn unknown_policy_graph_type_is_rejected_naming_accepted_set() {
        let json = r#"{
          "policy_graph": { "type": "elliptic", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 5 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("finite_horizon") && message.contains("cyclic"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    /// An unknown `block_mode` is rejected at parse, and the serde error names
    /// the accepted set.
    #[test]
    fn unknown_block_mode_is_rejected_naming_accepted_set() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 5,
              "block_mode": "diagonal" }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("parallel") && message.contains("chronological"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    /// An unknown `sampling_method` is rejected at parse, and the serde error
    /// names the accepted set.
    #[test]
    fn unknown_sampling_method_is_rejected_naming_accepted_set() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 5,
              "sampling_method": "antithetic" }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("saa") && message.contains("historical_residuals"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    /// The removed per-stage `num_scenarios` spelling is a deserialize error
    /// naming the offending field (`deny_unknown_fields` rejects it at parse,
    /// surfaced as [`LoadError::ParseError`]); the canonical `num_openings`
    /// spelling loads and its count reaches `branching_factor`.
    #[test]
    fn test_num_scenarios_removed_field_rejected() {
        let rejected = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_scenarios": 5 }
          ]
        }"#;
        let f = write_json(rejected);
        match parse_stages(f.path()).unwrap_err() {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("num_scenarios") && message.contains("unknown field"),
                    "removed field must surface as an unknown-field error naming it, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }

        let accepted = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 5 }
          ]
        }"#;
        let f = write_json(accepted);
        let data = parse_stages(f.path()).unwrap();
        assert_eq!(
            data.stages[0].scenario_config.branching_factor, 5,
            "num_openings must populate branching_factor"
        );
        assert!(
            data.openings_declared.contains(&0),
            "a declared count marks the stage in openings_declared"
        );
    }

    /// An unrecognized `risk_measure` string is rejected naming the value and the
    /// accepted set (A3) — a mistyped `"cvar"` does not silently train risk-neutral.
    #[test]
    fn test_unknown_risk_measure_string_rejected() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "stages": [{
            "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }],
            "num_openings": 5,
            "risk_measure": "cvar"
          }]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("risk_measure"), "field: {field}");
                assert!(
                    message.contains("cvar"),
                    "message should name the offending value, got: {message}"
                );
                assert!(
                    message.contains("expectation") && message.contains("CVaR"),
                    "message should name the accepted set, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `convert_noise_method(RawNoiseMethod::HistoricalResiduals)` returns
    /// `NoiseMethod::HistoricalResiduals`.
    #[test]
    fn test_convert_noise_method_historical_residuals() {
        assert_eq!(
            convert_noise_method(RawNoiseMethod::HistoricalResiduals),
            NoiseMethod::HistoricalResiduals
        );
    }

    /// Given study stages whose ids are non-contiguous and non-0-based (a
    /// negative pre-study id then a gap: `-1, 0, 2, 5`), `parse_stages` returns
    /// `Ok` — no contiguity or 0-basedness rule fires. Pins the absence of any
    /// such rule so a later author cannot add one as a "tightening"; `stage_id`
    /// is a declared domain id, not a positional index.
    #[test]
    fn non_contiguous_stage_ids_validate_ok() {
        let json = r#"{
          "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0, "transitions": [] },
          "pre_study_stages": [
            { "id": -1, "start_date": "2023-12-01", "end_date": "2024-01-01" }
          ],
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }], "num_openings": 5 },
            { "id": 2, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "S", "hours": 696.0 }], "num_openings": 5 },
            { "id": 5, "start_date": "2024-03-01", "end_date": "2024-04-01",
              "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }], "num_openings": 5 }
          ]
        }"#;
        let f = write_json(json);
        let data =
            parse_stages(f.path()).expect("non-contiguous, non-0-based stage ids validate Ok");

        // Sorted by id: -1, 0, 2, 5 — the gaps between 0, 2, 5 are preserved,
        // never rejected or renumbered.
        let ids: Vec<i32> = data.stages.iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![-1, 0, 2, 5]);
    }

    // ── nodes[] parsing ───────────────────────────────────────────────────────

    /// A declared `nodes[]` parses into canonical `Node`s carrying
    /// `{id, stage_id, scenario_id, label}`.
    #[test]
    fn test_parse_nodes_carries_fields() {
        let json = r#"{
          "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.0,
            "nodes": [
              { "id": 0, "stage_id": 0, "scenario_id": 2, "label": "root" },
              { "id": 1, "stage_id": 1 }
            ],
            "transitions": [ { "source_id": 0, "target_id": 1, "probability": 1.0 } ]
          },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 5 },
            { "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 696.0 }], "num_openings": 5 }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();
        let nodes = &data.policy_graph.nodes;
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].id, 0);
        assert_eq!(nodes[0].stage_id, 0);
        assert_eq!(nodes[0].scenario_id, Some(2));
        assert_eq!(nodes[0].label.as_deref(), Some("root"));
        assert_eq!(nodes[1].id, 1);
        assert_eq!(nodes[1].scenario_id, None);
        assert_eq!(nodes[1].label, None);
    }

    /// Absent `nodes[]` leaves `policy_graph.nodes` empty and preserves the
    /// stage-endpoint transitions unchanged (chain-dialect parse).
    #[test]
    fn test_absent_nodes_gives_empty_vec_and_stage_endpoints() {
        let f = write_json(VALID_JSON);
        let data = parse_stages(f.path()).unwrap();
        assert!(data.policy_graph.nodes.is_empty());
        assert_eq!(data.policy_graph.transitions.len(), 2);
        assert_eq!(data.policy_graph.transitions[0].source_id, 0);
        assert_eq!(data.policy_graph.transitions[0].target_id, 1);
    }

    /// An unknown key inside a node is rejected (`deny_unknown_fields`).
    #[test]
    fn test_unknown_key_in_node_rejected() {
        let json = r#"{
          "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.0,
            "nodes": [ { "id": 0, "stage_id": 0, "noise_source": 3 } ],
            "transitions": []
          },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "blocks": [{ "id": 0, "name": "A", "hours": 744.0 }], "num_openings": 5 }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_stages(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "unknown node key should be a ParseError, got: {err:?}"
        );
    }

    /// `nodes[]` are stored sorted by `id` ascending, so declaration order does
    /// not change the parsed graph (declaration-order invariance).
    #[test]
    fn test_nodes_sorted_by_id_declaration_order_invariant() {
        let make = |order: &str| {
            let json = format!(
                r#"{{
                  "policy_graph": {{
                    "type": "finite_horizon",
                    "annual_discount_rate": 0.0,
                    "nodes": [{order}],
                    "transitions": []
                  }},
                  "stages": [
                    {{ "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
                       "blocks": [{{ "id": 0, "name": "A", "hours": 744.0 }}], "num_openings": 5 }}
                  ]
                }}"#
            );
            let f = write_json(&json);
            parse_stages(f.path())
                .unwrap()
                .policy_graph
                .nodes
                .iter()
                .map(|n| n.id)
                .collect::<Vec<_>>()
        };
        let forward = make(
            r#"{ "id": 0, "stage_id": 0 }, { "id": 1, "stage_id": 0 }, { "id": 2, "stage_id": 0 }"#,
        );
        let reversed = make(
            r#"{ "id": 2, "stage_id": 0 }, { "id": 1, "stage_id": 0 }, { "id": 0, "stage_id": 0 }"#,
        );
        assert_eq!(forward, vec![0, 1, 2]);
        assert_eq!(forward, reversed);
    }

    // ── Out-edge probability normalization ────────────────────────────────────

    fn tr(source_id: i32, target_id: i32, probability: f64) -> Transition {
        Transition {
            source_id,
            target_id,
            probability,
            annual_discount_rate_override: None,
        }
    }

    fn graph_with(transitions: Vec<Transition>) -> HorizonGraph {
        HorizonGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions,
            nodes: Vec::new(),
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        }
    }

    fn prob(graph: &HorizonGraph, source_id: i32, target_id: i32) -> f64 {
        graph
            .transitions
            .iter()
            .find(|t| t.source_id == source_id && t.target_id == target_id)
            .map(|t| t.probability)
            .expect("edge present")
    }

    fn ulp_diff(a: f64, b: f64) -> u64 {
        a.to_bits().abs_diff(b.to_bits())
    }

    /// Each normalized out-edge vector sums to 1.0 within 1 ULP, and the result is
    /// independent of the declaration order of the edges (canonical child-id order).
    #[test]
    fn test_normalize_out_edges_ulp_bound_and_canonical_order() {
        // Raw weights [1, 2, 3] from source 0; declared out of child-id order.
        let mut shuffled = graph_with(vec![tr(0, 3, 3.0), tr(0, 1, 1.0), tr(0, 2, 2.0)]);
        let mut sorted = graph_with(vec![tr(0, 1, 1.0), tr(0, 2, 2.0), tr(0, 3, 3.0)]);
        let path = Path::new("stages.json");
        normalize_out_edge_probabilities(&mut shuffled, path).unwrap();
        normalize_out_edge_probabilities(&mut sorted, path).unwrap();

        // Canonical order ⇒ declaration order cannot shift the bit pattern.
        for target in [1, 2, 3] {
            assert_eq!(
                prob(&shuffled, 0, target).to_bits(),
                prob(&sorted, 0, target).to_bits(),
                "edge 0->{target} must be declaration-order-invariant"
            );
        }

        let sum = prob(&sorted, 0, 1) + prob(&sorted, 0, 2) + prob(&sorted, 0, 3);
        assert!(
            ulp_diff(sum, 1.0) <= 1,
            "normalized vector must sum to 1.0 within 1 ULP, got {sum} ({} ULPs)",
            ulp_diff(sum, 1.0)
        );
    }

    /// Path probabilities assembled from the normalized vectors of a 3-stage binary
    /// tree deviate from 1.0 by at most T (= 3) ULPs. Raw weights force real
    /// normalization at every source (thirds, halves, quarters).
    #[test]
    fn test_normalize_out_edges_path_measure_binary_tree() {
        // T = number of stages in the fixture; the path-measure bound is T ULPs.
        const T: u64 = 3;
        let mut tree = graph_with(vec![
            tr(0, 1, 1.0),
            tr(0, 2, 2.0), // -> [1/3, 2/3]
            tr(1, 3, 1.0),
            tr(1, 4, 1.0), // -> [1/2, 1/2]
            tr(2, 5, 1.0),
            tr(2, 6, 3.0), // -> [1/4, 3/4]
        ]);
        normalize_out_edge_probabilities(&mut tree, Path::new("stages.json")).unwrap();

        let paths = [(0, 1, 3), (0, 1, 4), (0, 2, 5), (0, 2, 6)];
        let total: f64 = paths
            .iter()
            .map(|&(root, mid, leaf)| prob(&tree, root, mid) * prob(&tree, mid, leaf))
            .sum();
        assert!(
            ulp_diff(total, 1.0) <= T,
            "path measure must be within {T} ULPs of 1.0, got {total} ({} ULPs)",
            ulp_diff(total, 1.0)
        );
    }

    /// A chain (single unit out-edge per source) is bit-neutral under normalization —
    /// the C1 golden-parity precondition.
    #[test]
    fn test_normalize_out_edges_chain_is_bit_neutral() {
        let mut chain = graph_with(vec![tr(0, 1, 1.0), tr(1, 2, 1.0)]);
        normalize_out_edge_probabilities(&mut chain, Path::new("stages.json")).unwrap();
        assert_eq!(prob(&chain, 0, 1).to_bits(), 1.0_f64.to_bits());
        assert_eq!(prob(&chain, 1, 2).to_bits(), 1.0_f64.to_bits());
    }

    /// Re-running normalization is a bit-exact no-op, so a second normalization
    /// anywhere downstream cannot shift the path measure (the once-at-load contract).
    #[test]
    fn test_normalize_out_edges_is_idempotent() {
        let build = || {
            graph_with(vec![
                tr(0, 1, 1.0),
                tr(0, 2, 2.0),
                tr(1, 3, 1.0),
                tr(1, 4, 3.0),
            ])
        };
        let path = Path::new("stages.json");
        let mut once = build();
        normalize_out_edge_probabilities(&mut once, path).unwrap();
        let mut twice = once.clone();
        normalize_out_edge_probabilities(&mut twice, path).unwrap();
        for (a, b) in once.transitions.iter().zip(&twice.transitions) {
            assert_eq!(
                a.probability.to_bits(),
                b.probability.to_bits(),
                "second normalization must be a bit-exact no-op"
            );
        }
    }

    /// A source whose out-edge weights sum to zero is rejected naming the source.
    #[test]
    fn test_normalize_out_edges_rejects_zero_sum() {
        let mut graph = graph_with(vec![tr(0, 1, 1.0), tr(0, 2, -1.0)]);
        let err =
            normalize_out_edge_probabilities(&mut graph, Path::new("stages.json")).unwrap_err();
        let LoadError::SchemaError { message, .. } = err else {
            panic!("expected SchemaError, got {err:?}");
        };
        assert!(
            message.contains("source 0") && message.contains("sums to zero"),
            "message must name the source and the zero sum, got: {message}"
        );
    }

    /// A source with a non-finite out-edge (a `NaN` sum passes the tolerance gate but
    /// cannot normalize) is rejected naming the source.
    #[test]
    fn test_normalize_out_edges_rejects_non_finite_sum() {
        let mut graph = graph_with(vec![tr(0, 1, f64::NAN), tr(0, 2, 0.5)]);
        let err =
            normalize_out_edge_probabilities(&mut graph, Path::new("stages.json")).unwrap_err();
        let LoadError::SchemaError { message, .. } = err else {
            panic!("expected SchemaError, got {err:?}");
        };
        assert!(
            message.contains("source 0") && message.contains("non-finite"),
            "message must name the source and the non-finite sum, got: {message}"
        );
    }

    // ── pre-build lag-transition window (position, not `Stage.index`) ──────

    /// `precompute_stage_lag_transitions` must derive its "stages after this
    /// one" window from each stage's own position in the slice passed to it,
    /// not from `Stage.index` — [`Stage::index`](cobre_core::temporal::Stage)
    /// is 0 for every stage `parse_stages`/`convert_stages` produces, since
    /// [`SystemBuilder::build`](cobre_core::SystemBuilder::build) is the only
    /// writer. Three single-occurrence-per-season monthly stages are each the
    /// last (and only) occurrence of their season within the year, so
    /// `finalize_period` must be `true` for all three; an index-trusting
    /// implementation has every stage after the first wrongly match itself in
    /// its own "later occurrences" scan (since `.skip(0 + 1)` always starts
    /// from position 1) and report `false`.
    #[test]
    fn test_prebuild_lag_transition_uses_position_not_index() {
        let json = r#"{
          "season_definitions": {
            "cycle_type": "monthly",
            "seasons": [
              { "id": 0, "label": "Jan", "month_start": 1 },
              { "id": 1, "label": "Feb", "month_start": 2 },
              { "id": 2, "label": "Mar", "month_start": 3 }
            ]
          },
          "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.0,
            "transitions": [
              { "source_id": 0, "target_id": 1, "probability": 1.0 },
              { "source_id": 1, "target_id": 2, "probability": 1.0 }
            ]
          },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-02-01",
              "season_id": 0, "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
              "num_openings": 1 },
            { "id": 1, "start_date": "2024-02-01", "end_date": "2024-03-01",
              "season_id": 1, "blocks": [{ "id": 0, "name": "S", "hours": 696.0 }],
              "num_openings": 1 },
            { "id": 2, "start_date": "2024-03-01", "end_date": "2024-04-01",
              "season_id": 2, "blocks": [{ "id": 0, "name": "S", "hours": 744.0 }],
              "num_openings": 1 }
          ]
        }"#;
        let f = write_json(json);
        let data = parse_stages(f.path()).unwrap();

        assert!(
            data.stages.iter().all(|s| s.index == 0),
            "pre-build stages must all carry index == 0 -- the precondition the \
             bug traded on"
        );

        let season_map = data
            .policy_graph
            .season_map
            .as_ref()
            .expect("season_definitions parsed into a SeasonMap");
        let transitions =
            cobre_stochastic::par::precompute_stage_lag_transitions(&data.stages, season_map, 0);

        let finalize: Vec<bool> = transitions.iter().map(|t| t.finalize_period).collect();
        assert_eq!(
            finalize,
            vec![true, true, true],
            "each stage is the only occurrence of its season within its year, so \
             finalize_period must be true for all three regardless of position"
        );
    }
}
