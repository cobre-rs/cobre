//! Temporal domain types — stages, blocks, seasons, and stage transitions.
//!
//! This module defines the types that describe the time structure of a
//! multi-stage stochastic optimization problem: how the study horizon is
//! partitioned into stages, how stages are subdivided into load blocks,
//! how stages relate to seasonal patterns, and how a [`Transition`] represents
//! one stage-to-stage edge. [`super::horizon`] assembles transitions and nodes
//! into the horizon-wide topology.
//!
//! These are clarity-first data types following the dual-nature design
//! principle: they use `Vec<T>`, `String`, and `Option` for readability
//! and correctness. LP-related fields (variable indices, constraint counts,
//! coefficient arrays) belong to the performance layer in downstream solver crates.
//!
//! Source: `stages.json`. See `internal-structures.md` SS12.

use std::sync::LazyLock;

use chrono::{Datelike, NaiveDate};

pub mod overlap;
pub mod stage_key;
pub use overlap::{window_period_overlaps, window_period_reach_depth, window_reaches_any_period};
pub use stage_key::{CalendarMonth, StageId, StudyPos, month_of};

// ---------------------------------------------------------------------------
// Supporting enums
// ---------------------------------------------------------------------------

/// Block formulation mode controlling how blocks within a stage relate
/// to each other in the LP.
///
/// See [Block Formulations](../math/block-formulations.md) for the
/// mathematical treatment of each mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlockMode {
    /// Independent sub-periods solved simultaneously; water balance is
    /// aggregated across all blocks in the stage.
    #[default]
    Parallel,

    /// Sequential blocks with inter-block state transitions (intra-stage
    /// storage dynamics), e.g. daily cycling within a monthly stage.
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

/// Opening-tree noise generation algorithm for a stage.
///
/// Orthogonal to `SamplingScheme`: that selects the forward-pass noise *source*,
/// this governs *how* the opening-tree noise vectors are produced.
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
    /// Historical residuals from the `HistoricalScenarioLibrary`: copies
    /// pre-computed eta vectors and skips the parametric Cholesky step, since
    /// empirical cross-entity correlation is already embedded in the residuals.
    /// Year pool from the system-level `HistoricalYears`.
    HistoricalResiduals,
}

/// Horizon type tag for the policy graph: finite (acyclic chain/DAG) or cyclic
/// (infinite periodic, at least one back-edge).
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

/// A load block within a stage — a sub-period with uniform demand and
/// generation characteristics.
///
/// The block weight (fraction of stage duration) is not stored; it is computed
/// on demand as `duration_hours / sum(all block hours in stage)`.
///
/// Source: `stages.json` `stages[].blocks[]`.
/// See [Input Scenarios §1.5](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Block {
    /// 0-based index within the parent stage, validated contiguous during loading.
    pub index: usize,

    /// Human-readable block label (e.g., "LEVE", "MEDIA", "PESADA").
    pub name: String,

    /// Block duration in hours. Must be positive; the block hours within a stage
    /// must sum to the stage duration ([Input Scenarios §1.10](input-scenarios.md), rule 3).
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
    /// Whether reservoir storage volumes are state variables (default true).
    pub storage: bool,

    /// Whether past inflow realizations (AR lags) are state variables (default
    /// false); required when PAR order `p > 0`.
    pub inflow_lags: bool,
}

// ---------------------------------------------------------------------------
// StageLagTransition (SS12.3.1)
// ---------------------------------------------------------------------------

/// Pre-encoded contribution of a single stage to the lag accumulator.
///
/// Each stage in a multi-resolution study may span a different calendar
/// duration than the lag period it feeds into (for example, weekly stages
/// contributing to a monthly lag slot). `StageLagTransition` encodes the
/// fractional weights and finalization signal that the precomputation
/// algorithm derives from stage date boundaries, so the hot path can apply
/// them without recomputing calendar arithmetic at runtime.
///
/// A `Vec<StageLagTransition>` indexed by stage index is the canonical way
/// to carry this information alongside the stage vector.
///
/// Weights are validated to be consistent with the stage's temporal position by
/// the precomputation algorithm, not here. The `downstream_*` fields are inert
/// (`0.0` / `false`) unless `accumulate_downstream` is set, so uniform-resolution
/// studies carry zero hot-path overhead.
// Rationale: the bools encode orthogonal hot-path conditions, each
// independently tested by the lag-state accumulation kernel; an enum would
// need 2^N variants.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StageLagTransition {
    /// Fraction of this stage's realized value added to the current lag bucket.
    pub accumulate_weight: f64,

    /// Fraction carried into the next lag bucket when `finalize_period`; `0.0` when
    /// the stage falls entirely within one lag period.
    pub spillover_weight: f64,

    /// When `true`, the current lag bucket is committed to the lag state vector and
    /// reset (seeded with `spillover_weight`); otherwise accumulation continues.
    pub finalize_period: bool,

    /// Whether this stage also accumulates into the downstream (coarser) ring
    /// buffer — `true` only in the pre-transition window before a resolution change.
    pub accumulate_downstream: bool,

    /// `accumulate_weight` against the downstream (coarser) lag boundaries.
    pub downstream_accumulate_weight: f64,

    /// `spillover_weight` against the downstream lag boundary.
    pub downstream_spillover_weight: f64,

    /// When `true`, the downstream lag bucket is finalized and pushed to the
    /// downstream ring buffer.
    pub downstream_finalize: bool,

    /// When `true` (first coarse stage with a downstream PAR order), the
    /// lag-state accumulation kernel overwrites `state[lag_start..]` with the
    /// completed downstream lags before resuming primary accumulation.
    pub rebuild_from_downstream: bool,
}

// ---------------------------------------------------------------------------
// StageRiskConfig (SS12.4)
// ---------------------------------------------------------------------------

/// Per-stage risk measure configuration stored in the [`Stage`] struct.
///
/// The solver-level `RiskMeasure` dispatch enum is built from this.
/// See [Risk Measure Trait](../architecture/risk-measure-trait.md).
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
/// Groups the scenario-related settings. Sourced from
/// `stages.json` `scenario_source` and per-stage overrides.
///
/// See [Input Scenarios §1.4, §1.8](input-scenarios.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScenarioSourceConfig {
    /// Noise realizations per stage (opening tree and forward pass). Must be positive.
    pub branching_factor: usize,

    /// Opening-tree noise algorithm; may vary per stage.
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
    /// 0-based array position in the canonical-ordered stages vector. The sole
    /// writer for a `Stage` reachable via
    /// [`System::stages`](crate::System::stages) is
    /// [`SystemBuilder::build`](crate::SystemBuilder::build), which assigns it
    /// after sorting the stage vector by `id`; not meaningful on a `Stage`
    /// that has not been through the builder.
    pub index: usize,

    /// Unique domain-level stage identifier from `stages.json`; non-negative for
    /// study stages, negative for pre-study stages.
    pub id: i32,

    /// Stage start date (inclusive), a timezone-free calendar date.
    pub start_date: NaiveDate,

    /// Stage end date (exclusive); duration is `end_date - start_date`.
    pub end_date: NaiveDate,

    /// Season index into [`SeasonDefinition`]; `None` for stages without
    /// seasonal structure.
    pub season_id: Option<usize>,

    /// Load blocks, sorted by index; their `duration_hours` sum to the stage duration.
    pub blocks: Vec<Block>,

    /// Block formulation mode; may vary per stage.
    /// See [Block Formulations](../math/block-formulations.md).
    pub block_mode: BlockMode,

    /// Flags for which variables carry state to the next stage.
    pub state_config: StageStateConfig,

    /// Risk measure configuration; may vary per stage.
    pub risk_config: StageRiskConfig,

    /// Scenario source configuration (branching factor and noise method).
    pub scenario_config: ScenarioSourceConfig,
}

impl Stage {
    /// Total duration in hours: the sum of `blocks[].duration_hours`.
    #[must_use]
    pub fn total_hours(&self) -> f64 {
        self.blocks.iter().map(|b| b.duration_hours).sum::<f64>()
    }
}

// ---------------------------------------------------------------------------
// SeasonDefinition (SS12.7)
// ---------------------------------------------------------------------------

/// A single season entry mapping a season ID to a calendar period.
///
/// Required when deriving AR parameters from inflow history (the season governs
/// how history aggregates into seasonal means/stds). The `day_start`, `month_end`,
/// and `day_end` fields are read only for the `Custom` cycle type.
///
/// Source: `stages.json` `season_definitions.seasons[]`.
/// See [Input Scenarios §1.1](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeasonDefinition {
    /// 0-based season index: 0–11 (Monthly, Jan–Dec) or 0–51 (Weekly, ISO weeks).
    pub id: usize,

    /// Human-readable label (e.g., "January", "Q1", "Wet Season").
    pub label: String,

    /// Calendar start month (1-12); for Monthly cycles this identifies the month.
    pub month_start: u32,

    /// Calendar start day (1-31). Default: 1.
    pub day_start: Option<u32>,

    /// Calendar end month (1-12).
    pub month_end: Option<u32>,

    /// Calendar end day (1-31).
    pub day_end: Option<u32>,
}

impl SeasonDefinition {
    /// Whether this definition's calendar span covers `(month, day)`.
    ///
    /// Mirrors [`SeasonMap::season_for_date`]'s `Custom` match arm, generalized
    /// to test any one definition (not just the first match).
    fn covers(&self, month: u32, day: u32) -> bool {
        let start = (self.month_start, self.day_start.unwrap_or(1));
        let end = (
            self.month_end.unwrap_or(self.month_start),
            self.day_end.unwrap_or(31),
        );
        let cur = (month, day);
        if start <= end {
            cur >= start && cur <= end
        } else {
            cur >= start || cur <= end
        }
    }

    /// Canonical calendar width, in days, of this season's own span under
    /// `cycle_type`: a season/temporal-granularity classifier, not a
    /// study-year-specific duration (a leap `Monthly` February is still
    /// classified at its canonical 28-day width; year-specific leap-awareness
    /// is a stage-vs-season fact a caller derives separately when needed).
    ///
    /// `Monthly` reads a fixed non-leap month-length table (its February is 28);
    /// `Custom` counts covered days out of the leap `CANONICAL_CALENDAR_DAYS`
    /// calendar (Feb 29 included) — the two arms disagree on Feb 29 by design,
    /// the same sweep `is_multi_resolution` uses to detect overlapping definitions.
    #[must_use]
    pub fn span_days(&self, cycle_type: SeasonCycleType) -> usize {
        match cycle_type {
            SeasonCycleType::Weekly => 7,
            SeasonCycleType::Monthly => {
                const MONTH_DAYS: [usize; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
                let index = usize::try_from(self.month_start.saturating_sub(1)).unwrap_or(0);
                MONTH_DAYS.get(index).copied().unwrap_or(31)
            }
            SeasonCycleType::Custom => CANONICAL_CALENDAR_DAYS
                .iter()
                .filter(|&&(month, day)| self.covers(month, day))
                .count(),
        }
    }
}

/// Canonical 366-day `(month, day)` sequence for one leap year, swept when
/// comparing `Custom` season spans — the year itself is arbitrary since
/// `Custom` matching (`SeasonMap::season_for_date`, `SeasonDefinition::covers`)
/// operates on `(month, day)` alone, and omitting February 29 would silently
/// skip a checkable calendar position. Built once and borrowed by every
/// reader rather than rebuilt per call.
static CANONICAL_CALENDAR_DAYS: LazyLock<Vec<(u32, u32)>> = LazyLock::new(|| {
    const DAYS_IN_MONTH_LEAP: [u32; 12] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut days = Vec::with_capacity(366);
    for (month_index, &days_in_month) in DAYS_IN_MONTH_LEAP.iter().enumerate() {
        let month = u32::try_from(month_index).unwrap_or(0) + 1;
        for day in 1..=days_in_month {
            days.push((month, day));
        }
    }
    days
});

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
    /// Returns `None` only for `Custom` cycle types whose ranges leave the date
    /// uncovered; well-formed `Monthly`/`Weekly` maps always resolve (the `Weekly`
    /// arm folds ISO week 53 into week 52).
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
                // 52-bucket year (`id` 0..=51); `.min(51)` folds ISO week 53 into
                // week 52 — NOT `None` (callers skip None, dropping the observation
                // from PAR estimation) and NOT week 1 (wrong season for late December).
                let iso_week = date.iso_week().week();
                let week_idx = (iso_week.saturating_sub(1)).min(51) as usize;
                self.seasons.iter().find(|s| s.id == week_idx).map(|s| s.id)
            }
            SeasonCycleType::Custom => {
                let (m, d) = (date.month(), date.day());
                self.seasons.iter().find(|s| s.covers(m, d)).map(|s| s.id)
            }
        }
    }

    /// Detect whether this map layers more than one temporal-resolution level
    /// (e.g. a monthly + quarterly `Custom` map, the shape covered by the
    /// `test_d30_shaped_custom_map_is_multi_resolution` test below).
    /// `Monthly`/`Weekly` always tile their cycle by construction (12 disjoint
    /// months / 52 disjoint ISO weeks) and can never be multi-resolution; only
    /// `Custom` definitions legitimately overlap by design.
    ///
    /// Sweeps the 366-day canonical calendar (leap year, February 29 included,
    /// mirroring [`SeasonMap::season_for_date`]'s `Custom` arm) and flags any
    /// day covered by two or more definitions.
    #[must_use]
    pub fn is_multi_resolution(&self) -> bool {
        if self.cycle_type != SeasonCycleType::Custom {
            return false;
        }
        CANONICAL_CALENDAR_DAYS.iter().any(|&(month, day)| {
            self.seasons
                .iter()
                .filter(|def| def.covers(month, day))
                .count()
                >= 2
        })
    }

    /// Canonical calendar width, in days, of the entry identified by
    /// `season_id` — [`SeasonDefinition::span_days`] resolved through this
    /// map's own `cycle_type`. `None` when `season_id` matches no entry.
    #[must_use]
    pub fn resolution_level_of(&self, season_id: usize) -> Option<usize> {
        self.seasons
            .iter()
            .find(|s| s.id == season_id)
            .map(|s| s.span_days(self.cycle_type))
    }
}

// ---------------------------------------------------------------------------
// Transition (SS12.9)
// ---------------------------------------------------------------------------

/// A single transition in the horizon graph, representing a directed
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
    /// Source endpoint: a node id when the horizon graph declares `nodes`, a stage id otherwise.
    pub source_id: i32,

    /// Target endpoint: a node id when the horizon graph declares `nodes`, a stage id otherwise.
    pub target_id: i32,

    /// Transition probability. Outgoing probabilities from each source
    /// must sum to 1.0 (within tolerance).
    pub probability: f64,

    /// Per-transition annual discount rate override; `None` uses the
    /// [`HorizonGraph`](crate::HorizonGraph) global rate.
    /// See [Discount Rate §3](../math/discount-rate.md).
    pub annual_discount_rate_override: Option<f64>,
}

// ---------------------------------------------------------------------------
// Node (SS12.9.1)
// ---------------------------------------------------------------------------

/// A single horizon-graph node: one decision point at a study stage, carrying the
/// per-stage external-library realization it addresses.
///
/// A declared [`HorizonGraph::nodes`](crate::HorizonGraph::nodes) makes the graph
/// node-native — the engine solves the node graph directly and [`Transition`]
/// endpoints are node ids. An empty `nodes` leaves the graph a stage chain
/// (endpoints are stage ids).
///
/// Source: `stages.json` `policy_graph.nodes[]`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Node {
    /// Unique node id; [`Transition`] endpoints reference it under a declared graph.
    pub id: i32,

    /// Declared study-stage id this node sits at; a domain id, never an array index.
    pub stage_id: i32,

    /// Per-stage external-library realization column this node carries; `None` at a
    /// stage with no slot-occupying external class.
    pub scenario_id: Option<i32>,

    /// Optional human-readable label.
    pub label: Option<String>,
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
    fn test_stage_total_hours_is_the_left_to_right_block_sum() {
        let stage = Stage {
            index: 0,
            id: 1,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![
                Block {
                    index: 0,
                    name: "BLK0".to_string(),
                    duration_hours: 0.1,
                },
                Block {
                    index: 1,
                    name: "BLK1".to_string(),
                    duration_hours: 0.2,
                },
                Block {
                    index: 2,
                    name: "BLK2".to_string(),
                    duration_hours: 0.3,
                },
            ],
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
            stage.total_hours().to_bits(),
            (0.1_f64 + 0.2 + 0.3).to_bits()
        );
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

    #[test]
    fn test_weekly_season_iso_week_53_folds_into_week_52() {
        // `month_start` is unused for `Weekly` resolution but must be a valid month.
        let seasons: Vec<SeasonDefinition> = (0..52)
            .map(|i| SeasonDefinition {
                id: i,
                label: format!("W{:02}", i + 1),
                month_start: 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Weekly,
            seasons,
        };

        // 2020 is an ISO long year, so 2020-12-30 falls in ISO week 53.
        let week53_date = NaiveDate::from_ymd_opt(2020, 12, 30).unwrap();
        assert_eq!(
            week53_date.iso_week().week(),
            53,
            "precondition: ISO week 53"
        );
        assert_eq!(
            season_map.season_for_date(week53_date),
            Some(51),
            "ISO week 53 must fold into week 52 (id 51), not None or week 1 (id 0)"
        );

        // A normal week resolves to its own bucket: 2021-01-11 is ISO week 2 -> id 1.
        let week2_date = NaiveDate::from_ymd_opt(2021, 1, 11).unwrap();
        assert_eq!(week2_date.iso_week().week(), 2, "precondition: ISO week 2");
        assert_eq!(season_map.season_for_date(week2_date), Some(1));
    }

    /// `d30-multi-resolution-monthly-quarterly`-shaped `season_definitions`:
    /// 12 monthly + 4 quarterly `Custom` definitions whose calendar spans
    /// overlap by design.
    fn d30_shaped_season_map() -> SeasonMap {
        let mut seasons: Vec<SeasonDefinition> = (0..12u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("Month{}", i + 1),
                month_start: i + 1,
                day_start: Some(1),
                month_end: Some(i + 1),
                day_end: Some(if i == 1 { 28 } else { 31 }),
            })
            .collect();
        seasons.extend([
            SeasonDefinition {
                id: 12,
                label: "Q3".to_string(),
                month_start: 7,
                day_start: Some(1),
                month_end: Some(9),
                day_end: Some(30),
            },
            SeasonDefinition {
                id: 13,
                label: "Q4".to_string(),
                month_start: 10,
                day_start: Some(1),
                month_end: Some(12),
                day_end: Some(31),
            },
            SeasonDefinition {
                id: 14,
                label: "Q1".to_string(),
                month_start: 1,
                day_start: Some(1),
                month_end: Some(3),
                day_end: Some(31),
            },
            SeasonDefinition {
                id: 15,
                label: "Q2".to_string(),
                month_start: 4,
                day_start: Some(1),
                month_end: Some(6),
                day_end: Some(30),
            },
        ]);
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons,
        }
    }

    #[test]
    fn test_d30_shaped_custom_map_is_multi_resolution() {
        assert!(
            d30_shaped_season_map().is_multi_resolution(),
            "monthly + quarterly Custom definitions overlap by design"
        );
    }

    #[test]
    fn test_all_monthly_map_is_not_multi_resolution() {
        let seasons: Vec<SeasonDefinition> = (0..12u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("Month{}", i + 1),
                month_start: i + 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        };
        assert!(!season_map.is_multi_resolution());
    }

    #[test]
    fn test_weekly_map_is_not_multi_resolution() {
        let seasons: Vec<SeasonDefinition> = (0..52u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("W{:02}", i + 1),
                month_start: 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Weekly,
            seasons,
        };
        assert!(!season_map.is_multi_resolution());
    }

    #[test]
    fn test_resolution_level_of_d30_shaped_map() {
        let season_map = d30_shaped_season_map();
        assert_eq!(
            season_map.resolution_level_of(0),
            Some(31),
            "January spans 31 canonical days"
        );
        assert_eq!(
            season_map.resolution_level_of(1),
            Some(28),
            "the D30 fixture caps February at day_end 28, excluding the leap Feb 29"
        );
        assert_eq!(
            season_map.resolution_level_of(12),
            Some(92),
            "Q3 (Jul+Aug+Sep) spans 92 canonical days"
        );
        assert_eq!(
            season_map.resolution_level_of(14),
            Some(91),
            "Q1 (Jan+Feb+Mar) spans 91 canonical days, including the leap Feb 29"
        );
        assert_eq!(season_map.resolution_level_of(999), None);
    }

    #[test]
    fn test_resolution_level_of_weekly_is_always_seven() {
        let seasons: Vec<SeasonDefinition> = (0..52u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("W{:02}", i + 1),
                month_start: 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        let season_map = SeasonMap {
            cycle_type: SeasonCycleType::Weekly,
            seasons,
        };
        assert_eq!(season_map.resolution_level_of(0), Some(7));
        assert_eq!(season_map.resolution_level_of(51), Some(7));
    }

    #[test]
    fn test_season_for_date_custom_first_match_wins_on_overlap() {
        // Both "July" (id 6) and "Q3" (id 12) cover 2024-07-15; monthly entries
        // are listed first in the map, so `find` resolves the finer match.
        let season_map = d30_shaped_season_map();
        let july_15 = NaiveDate::from_ymd_opt(2024, 7, 15).unwrap();
        assert_eq!(season_map.season_for_date(july_15), Some(6));
    }
}
