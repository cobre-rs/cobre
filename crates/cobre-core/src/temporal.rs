//! Temporal domain types — stages, blocks, seasons, and the policy graph.
//!
//! This module defines the types that describe the time structure of a
//! multi-stage stochastic optimization problem: how the study horizon is
//! partitioned into stages, how stages are subdivided into load blocks,
//! how stages relate to seasonal patterns, and how the policy graph
//! encodes stage-to-stage transitions.
//!
//! These are clarity-first data types following the dual-nature design
//! principle: they use `Vec<T>`, `String`, and `Option` for readability
//! and correctness. LP-related fields (variable indices, constraint counts,
//! coefficient arrays) belong to the performance layer in downstream solver crates.
//!
//! Source: `stages.json`. See `internal-structures.md` SS12.

use chrono::{Datelike, NaiveDate};

// ---------------------------------------------------------------------------
// Supporting enums
// ---------------------------------------------------------------------------

/// Block formulation mode controlling how blocks within a stage relate
/// to each other in the LP.
///
/// See [Block Formulations](../math/block-formulations.md) for the
/// mathematical treatment of each mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlockMode {
    /// Blocks are independent sub-periods solved simultaneously.
    /// Water balance is aggregated across all blocks in the stage.
    /// This is the default and most common mode.
    Parallel,

    /// Blocks are sequential within the stage, with inter-block
    /// state transitions (intra-stage storage dynamics).
    /// Enables modeling of daily cycling patterns within monthly stages.
    Chronological,
}

/// Season cycle type controlling how season IDs map to calendar periods.
///
/// See [Input Scenarios §1.1](input-scenarios.md) for the JSON schema
/// and calendar mapping rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SeasonCycleType {
    /// Each season corresponds to one calendar month (12 seasons).
    Monthly,
    /// Each season corresponds to one ISO calendar week (52 seasons).
    Weekly,
    /// User-defined date ranges with explicit boundaries per season.
    Custom,
}

/// Opening tree noise generation algorithm for a stage.
///
/// Controls which algorithm is used to generate noise vectors for
/// the opening tree at this stage. This is orthogonal to
/// `SamplingScheme`, which selects the forward-pass noise *source*
/// (in-sample, external, historical). `NoiseMethod` governs *how*
/// the noise vectors are produced (SAA, LHS, QMC-Sobol, QMC-Halton,
/// Selective, or `HistoricalResiduals`).
///
/// See [Input Scenarios §1.8](input-scenarios.md) for the
/// full method catalog and use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NoiseMethod {
    /// Sample Average Approximation. Pure Monte Carlo random sampling.
    Saa,
    /// Latin Hypercube Sampling. Stratified sampling ensuring uniform coverage.
    Lhs,
    /// Quasi-Monte Carlo with Sobol sequences. Low-discrepancy.
    QmcSobol,
    /// Quasi-Monte Carlo with Halton sequences. Low-discrepancy.
    QmcHalton,
    /// Selective/Representative Sampling. Clustering on historical data.
    Selective,
    /// Historical residuals from the `HistoricalScenarioLibrary`.
    /// Copies pre-computed eta (residual) vectors from actual historical
    /// observations. Skips the parametric Cholesky correlation step since
    /// empirical cross-entity correlation is embedded in the residuals.
    /// Year pool configuration is sourced from the system-level
    /// `HistoricalYears` config (same as the Historical forward sampling
    /// scheme).
    HistoricalResiduals,
}

/// Horizon type tag for the policy graph.
///
/// Determines whether the study horizon is finite (acyclic linear chain or DAG)
/// or cyclic (infinite periodic horizon with at least one back-edge). The
/// solver-level `HorizonMode` enum in downstream solver crates is built from a
/// [`PolicyGraph`] that carries this tag — it precomputes transition maps,
/// cycle detection, and discount factors for efficient runtime dispatch.
///
/// Cross-reference: [Horizon Mode Trait SS3.1](../architecture/horizon-mode-trait.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PolicyGraphType {
    /// Acyclic stage chain: the study has a definite end stage.
    /// Terminal value is zero (no future-cost approximation beyond the horizon).
    FiniteHorizon,
    /// Infinite periodic horizon: at least one transition has
    /// `source_id >= target_id` (a back-edge). Requires a positive
    /// `annual_discount_rate` for convergence.
    Cyclic,
}

// ---------------------------------------------------------------------------
// Block (SS12.2)
// ---------------------------------------------------------------------------

/// A load block within a stage, representing a sub-period with uniform
/// demand and generation characteristics.
///
/// Blocks partition the stage duration into sub-periods (e.g., peak,
/// off-peak, shoulder). Block IDs are contiguous within each stage,
/// starting at 0. The block weight (fraction of stage duration) is
/// derived from `duration_hours` and is not stored — it is computed
/// on demand as `duration_hours / sum(all block hours in stage)`.
///
/// Source: `stages.json` `stages[].blocks[]`.
/// See [Input Scenarios §1.5](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Block {
    /// 0-based index within the parent stage.
    /// Matches the `id` field from `stages.json`, validated to be
    /// contiguous (0, 1, 2, ..., n-1) during loading.
    pub index: usize,

    /// Human-readable block label (e.g., "LEVE", "MEDIA", "PESADA").
    pub name: String,

    /// Duration of this block in hours. Must be positive.
    /// Validation: the sum of all block hours within a stage must
    /// equal the total stage duration in hours.
    /// See [Input Scenarios §1.10](input-scenarios.md), rule 3.
    pub duration_hours: f64,
}

// ---------------------------------------------------------------------------
// StageStateConfig (SS12.3)
// ---------------------------------------------------------------------------

/// State variable flags controlling which variables carry state
/// between stages for a given stage.
///
/// Source: `stages.json` `stages[].state_variables`.
/// See [Input Scenarios §1.6](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageStateConfig {
    /// Whether reservoir storage volumes are state variables.
    /// Default: true. Mandatory in most applications but kept as an
    /// explicit flag for transparency.
    pub storage: bool,

    /// Whether past inflow realizations (AR model lags) are state
    /// variables. Default: false. Required when PAR model order `p > 0`
    /// and inflow lag state tracking is enabled.
    pub inflow_lags: bool,
}

// ---------------------------------------------------------------------------
// StageRiskConfig (SS12.4)
// ---------------------------------------------------------------------------

/// Per-stage risk measure configuration, representing the parsed and
/// validated risk parameters for a single stage.
///
/// This is the clarity-first representation stored in the [`Stage`] struct.
/// The solver-level `RiskMeasure` enum in
/// [Risk Measure Trait](../architecture/risk-measure-trait.md) is the
/// dispatch type built FROM this configuration during the variant
/// selection pipeline.
///
/// Source: `stages.json` `stages[].risk_measure`.
/// See [Input Scenarios §1.7](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StageRiskConfig {
    /// Risk-neutral expected value. No additional parameters.
    Expectation,

    /// Convex combination of expectation and `CVaR`.
    /// See [Risk Measures](../math/risk-measures.md) for the
    /// mathematical formulation.
    CVaR {
        /// Confidence level `alpha` in (0, 1].
        /// `alpha = 0.95` means 5% worst-case scenarios are considered.
        alpha: f64,

        /// Risk aversion weight `lambda` in \[0, 1\].
        /// `lambda = 0` reduces to Expectation; `lambda = 1` is pure `CVaR`.
        lambda: f64,
    },
}

// ---------------------------------------------------------------------------
// ScenarioSourceConfig (SS12.5)
// ---------------------------------------------------------------------------

/// Scenario source configuration for one stage.
///
/// Groups the scenario-related settings that were formerly separate
/// `num_scenarios` and `sampling_method` fields. Sourced from
/// `stages.json` `scenario_source` and per-stage overrides.
///
/// See [Input Scenarios §1.4, §1.8](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScenarioSourceConfig {
    /// Number of noise realizations per stage for both the opening
    /// tree and forward pass. Formerly `num_scenarios`.
    /// Must be positive. Controls the per-stage branching factor.
    pub branching_factor: usize,

    /// Algorithm for generating noise vectors in the opening tree.
    /// Orthogonal to `SamplingScheme`, which selects the noise
    /// source (in-sample, external, historical).
    /// Can vary per stage, allowing adaptive strategies (e.g., LHS
    /// for near-term, SAA for distant stages).
    pub noise_method: NoiseMethod,
}

// ---------------------------------------------------------------------------
// Stage (SS12.6)
// ---------------------------------------------------------------------------

/// A single stage in the multi-stage stochastic optimization problem.
///
/// Stages partition the study horizon into decision periods. Each stage
/// has a temporal extent, block structure, scenario configuration, risk
/// parameters, and state variable flags. Stages are sorted by `id` in
/// canonical order after loading (see Design Principles §3).
///
/// Study stages have non-negative IDs; pre-study stages (used only for
/// PAR model lag initialization) have negative IDs. Pre-study stages
/// carry only `id`, `start_date`, `end_date`, and `season_id` — their
/// blocks, risk, and sampling fields are unused.
///
/// This struct does NOT contain LP-related fields (variable indices,
/// constraint counts, coefficient arrays). Those belong to the
/// downstream solver crate performance layer — see Solver Abstraction SS11.
///
/// Source: `stages.json` `stages[]` and `pre_study_stages[]`.
/// See [Input Scenarios §1.4](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stage {
    // -- Identity and temporal extent --
    /// 0-based index of this stage in the canonical-ordered stages
    /// vector. Used for array indexing into per-stage data structures
    /// (cuts, results, penalty arrays). Assigned during loading after
    /// sorting by `id`.
    pub index: usize,

    /// Unique stage identifier from `stages.json`.
    /// Non-negative for study stages, negative for pre-study stages.
    /// The `id` is the domain-level identifier; `index` is the
    /// internal array position.
    pub id: i32,

    /// Stage start date (inclusive). Parsed from ISO 8601 string.
    /// Uses `chrono::NaiveDate` — timezone-free calendar date, which
    /// is appropriate because stage boundaries are calendar concepts,
    /// not instants in time.
    pub start_date: NaiveDate,

    /// Stage end date (exclusive). Parsed from ISO 8601 string.
    /// The stage duration is `end_date - start_date`.
    pub end_date: NaiveDate,

    /// Season index linking to [`SeasonDefinition`]. Maps this stage to
    /// a position in the seasonal cycle (e.g., month 0-11 for monthly).
    /// Required for PAR model coefficient lookup and inflow history
    /// aggregation. `None` for stages without seasonal structure.
    pub season_id: Option<usize>,

    // -- Block structure --
    /// Ordered list of load blocks within this stage.
    /// Sorted by block index (0, 1, ..., n-1). The sum of all block
    /// `duration_hours` must equal the total stage duration in hours.
    pub blocks: Vec<Block>,

    /// Block formulation mode for this stage.
    /// Can vary per stage (e.g., chronological for near-term,
    /// parallel for distant stages).
    /// See [Block Formulations](../math/block-formulations.md).
    pub block_mode: BlockMode,

    // -- State, risk, and sampling --
    /// State variable flags controlling which variables carry state
    /// from this stage to the next.
    pub state_config: StageStateConfig,

    /// Risk measure configuration for this stage.
    /// Can vary per stage (e.g., `CVaR` for near-term, Expectation
    /// for distant stages).
    pub risk_config: StageRiskConfig,

    /// Scenario source configuration (branching factor and noise method).
    pub scenario_config: ScenarioSourceConfig,
}

// ---------------------------------------------------------------------------
// SeasonDefinition (SS12.7)
// ---------------------------------------------------------------------------

/// A single season entry mapping a season ID to a calendar period.
///
/// Season definitions are required when deriving AR parameters from
/// inflow history — the season determines how history values are
/// aggregated into seasonal means and standard deviations.
///
/// Source: `stages.json` `season_definitions.seasons[]`.
/// See [Input Scenarios §1.1](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeasonDefinition {
    /// Season index (0-based). For monthly cycles: 0 = January, ...,
    /// 11 = December. For weekly cycles: 0-51 (ISO week numbers).
    pub id: usize,

    /// Human-readable label (e.g., "January", "Q1", "Wet Season").
    pub label: String,

    /// Calendar month where this season starts (1-12).
    /// For monthly `cycle_type`, this uniquely identifies the month.
    pub month_start: u32,

    /// Calendar day where this season starts (1-31).
    /// Only used when `cycle_type` is `Custom`. Default: 1.
    pub day_start: Option<u32>,

    /// Calendar month where this season ends (1-12).
    /// Only used when `cycle_type` is `Custom`.
    pub month_end: Option<u32>,

    /// Calendar day where this season ends (1-31).
    /// Only used when `cycle_type` is `Custom`.
    pub day_end: Option<u32>,
}

// ---------------------------------------------------------------------------
// SeasonMap (SS12.8)
// ---------------------------------------------------------------------------

/// Complete season definitions including cycle type and all season entries.
///
/// The `SeasonMap` is the resolved representation of the `season_definitions`
/// section in `stages.json`. It provides the season-to-calendar mapping
/// consumed by the PAR model and inflow history aggregation.
///
/// Source: `stages.json` `season_definitions`.
/// See [Input Scenarios §1.1](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeasonMap {
    /// Cycle type controlling how season IDs map to calendar periods.
    pub cycle_type: SeasonCycleType,

    /// Season entries sorted by `id`. Length depends on `cycle_type`:
    /// 12 for `Monthly`, 52 for `Weekly`, user-defined for `Custom`.
    pub seasons: Vec<SeasonDefinition>,
}

impl SeasonMap {
    /// Resolve a calendar date to a season ID using the cycle definition.
    ///
    /// This mapping is purely calendar-based and does not depend on the study
    /// horizon — a date from 1931 maps to the same season as a date from 2026
    /// if they share the same calendar position. This is essential for PAR
    /// model estimation from historical inflow data that predates the study.
    ///
    /// Returns `None` only for `Custom` cycle types where the date does not
    /// fall within any defined season range.
    #[must_use]
    pub fn season_for_date(&self, date: NaiveDate) -> Option<usize> {
        match self.cycle_type {
            SeasonCycleType::Monthly => {
                let month = date.month();
                self.seasons
                    .iter()
                    .find(|s| s.month_start == month)
                    .map(|s| s.id)
            }
            SeasonCycleType::Weekly => {
                let iso_week = date.iso_week().week();
                let week_idx = (iso_week.saturating_sub(1)).min(51) as usize;
                self.seasons.iter().find(|s| s.id == week_idx).map(|s| s.id)
            }
            SeasonCycleType::Custom => {
                let (m, d) = (date.month(), date.day());
                self.seasons
                    .iter()
                    .find(|s| {
                        let ms = s.month_start;
                        let ds = s.day_start.unwrap_or(1);
                        let me = s.month_end.unwrap_or(ms);
                        let de = s.day_end.unwrap_or(31);
                        let start = (ms, ds);
                        let end = (me, de);
                        let cur = (m, d);
                        if start <= end {
                            cur >= start && cur <= end
                        } else {
                            cur >= start || cur <= end
                        }
                    })
                    .map(|s| s.id)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transition (SS12.9)
// ---------------------------------------------------------------------------

/// A single transition in the policy graph, representing a directed
/// edge from one stage to another with an associated probability and
/// optional discount rate override.
///
/// Transitions define the stage traversal order for both the forward
/// and backward passes. In finite horizon mode, transitions form a
/// linear chain. In cyclic mode, at least one transition creates a
/// back-edge (`source_id >= target_id`).
///
/// Source: `stages.json` `policy_graph.transitions[]`.
/// See [Input Scenarios §1.2](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transition {
    /// Source stage ID. Must exist in the stage set.
    pub source_id: i32,

    /// Target stage ID. Must exist in the stage set.
    pub target_id: i32,

    /// Transition probability. Outgoing probabilities from each source
    /// must sum to 1.0 (within tolerance).
    pub probability: f64,

    /// Per-transition annual discount rate override.
    /// When `None`, the global `annual_discount_rate` from the
    /// [`PolicyGraph`] is used. When `Some(r)`, this rate is converted to
    /// a per-transition factor using the source stage duration:
    /// `d = 1 / (1 + r)^dt`.
    /// See [Discount Rate §3](../math/discount-rate.md).
    pub annual_discount_rate_override: Option<f64>,
}

// ---------------------------------------------------------------------------
// PolicyGraph (SS12.10)
// ---------------------------------------------------------------------------

/// Parsed and validated policy graph defining stage transitions,
/// horizon type, and global discount rate.
///
/// This is the `cobre-core` clarity-first representation loaded from
/// `stages.json`. It stores the graph topology as specified by the
/// user. The solver-level `HorizonMode` enum (see Horizon Mode Trait
/// SS1) is built from this struct during initialization — it
/// precomputes transition maps, cycle detection, and discount factors
/// for efficient runtime dispatch.
///
/// Cross-reference: [Horizon Mode Trait](../architecture/horizon-mode-trait.md)
/// defines the `HorizonMode` enum that interprets this graph structure.
///
/// Source: `stages.json` `policy_graph`.
/// See [Input Scenarios §1.2](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PolicyGraph {
    /// Horizon type: finite (acyclic chain) or cyclic (infinite periodic).
    /// Determines which `HorizonMode` variant will be constructed at
    /// solver initialization.
    pub graph_type: PolicyGraphType,

    /// Global annual discount rate.
    /// Converted to per-transition factors using source stage durations:
    /// `d = 1 / (1 + annual_discount_rate)^dt`.
    /// A value of 0.0 means no discounting (`d = 1.0` for all transitions).
    /// For cyclic graphs, must be > 0 for convergence (validation rule 7).
    /// See [Discount Rate §3](../math/discount-rate.md).
    pub annual_discount_rate: f64,

    /// Stage transitions with probabilities and optional per-transition
    /// discount rate overrides. For finite horizon, these form a linear
    /// chain or DAG. For cyclic horizon, at least one transition has
    /// `source_id >= target_id` (the back-edge).
    pub transitions: Vec<Transition>,

    /// Season definitions loaded from `season_definitions` in
    /// `stages.json`. Required when PAR models or inflow history
    /// aggregation are used. `None` when no season definitions are
    /// provided and none are required.
    pub season_map: Option<SeasonMap>,
}

impl Default for PolicyGraph {
    /// Returns a finite-horizon policy graph with no transitions and no discounting.
    ///
    /// This is the minimal-viable-solver default: a finite study horizon with
    /// zero terminal value and no discount factor. `cobre-io` replaces this
    /// with the fully specified graph loaded from `stages.json`.
    fn default() -> Self {
        Self {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: Vec::new(),
            season_map: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_mode_copy() {
        let original = BlockMode::Parallel;
        let copied = original;
        assert_eq!(original, BlockMode::Parallel);
        assert_eq!(copied, BlockMode::Parallel);

        let chrono = BlockMode::Chronological;
        let copied_chrono = chrono;
        assert_eq!(chrono, BlockMode::Chronological);
        assert_eq!(copied_chrono, BlockMode::Chronological);
    }

    #[test]
    fn test_stage_duration() {
        let stage = Stage {
            index: 0,
            id: 1,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        };

        assert_eq!(
            stage.end_date - stage.start_date,
            chrono::TimeDelta::days(31)
        );
    }

    #[test]
    fn test_policy_graph_construction() {
        let transitions = vec![
            Transition {
                source_id: 1,
                target_id: 2,
                probability: 1.0,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 2,
                target_id: 3,
                probability: 1.0,
                annual_discount_rate_override: Some(0.08),
            },
            Transition {
                source_id: 3,
                target_id: 4,
                probability: 1.0,
                annual_discount_rate_override: None,
            },
        ];

        let graph = PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions,
            season_map: None,
        };

        assert_eq!(graph.graph_type, PolicyGraphType::FiniteHorizon);
        assert!((graph.annual_discount_rate - 0.06).abs() < f64::EPSILON);
        assert_eq!(graph.transitions.len(), 3);
        assert_eq!(
            graph.transitions[1].annual_discount_rate_override,
            Some(0.08)
        );
        assert!(graph.season_map.is_none());
    }

    #[test]
    fn test_season_map_construction() {
        let months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ];

        let seasons: Vec<SeasonDefinition> = months
            .iter()
            .enumerate()
            .map(|(i, &label)| SeasonDefinition {
                id: i,
                label: label.to_string(),
                month_start: u32::try_from(i + 1).unwrap(),
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();

        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };

        assert_eq!(season_map.cycle_type, SeasonCycleType::Monthly);
        assert_eq!(season_map.seasons.len(), 12);
        assert_eq!(season_map.seasons[0].label, "January");
        assert_eq!(season_map.seasons[11].label, "December");
        assert_eq!(season_map.seasons[0].month_start, 1);
        assert_eq!(season_map.seasons[11].month_start, 12);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_policy_graph_serde_roundtrip() {
        let graph = PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![
                Transition {
                    source_id: 1,
                    target_id: 2,
                    probability: 1.0,
                    annual_discount_rate_override: None,
                },
                Transition {
                    source_id: 2,
                    target_id: 3,
                    probability: 1.0,
                    annual_discount_rate_override: None,
                },
            ],
            season_map: None,
        };

        let json = serde_json::to_string(&graph).unwrap();

        // Acceptance criterion: JSON must contain both key-value pairs.
        assert!(
            json.contains("\"graph_type\":\"FiniteHorizon\""),
            "JSON did not contain expected graph_type: {json}"
        );
        assert!(
            json.contains("\"annual_discount_rate\":0.06"),
            "JSON did not contain expected annual_discount_rate: {json}"
        );

        // Round-trip must produce an equal value.
        let deserialized: PolicyGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(graph, deserialized);
    }
}
