//! Shared output-writing helpers consumed by both the CLI binary and the
//! Python bindings.
//!
//! Both front ends call this one implementation of the policy-checkpoint writer
//! and the stochastic-artifacts exporter, so their outputs cannot drift (Python
//! parity is a hard rule). Callers thread their own diagnostics via the
//! `on_warning` callback on [`export_stochastic_artifacts`].

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use cobre_io::config::CheckpointSchedule;
use cobre_io::output::policy::{
    CheckpointManifest, FORMAT_VERSION, GraphManifest, HydroSeasonOrders, ProducerBlock,
    SEASON_CYCLE_CODE_ABSENT, SEASON_CYCLE_CODE_CUSTOM, SEASON_CYCLE_CODE_MONTHLY,
    SEASON_CYCLE_CODE_WEEKLY, SeasonManifest, write_policy_checkpoint,
};
use cobre_io::output::{
    OutputError, write_correlation_json, write_fitting_report, write_inflow_annual_component,
    write_inflow_ar_coefficients, write_inflow_seasonal_stats, write_load_seasonal_stats,
    write_noise_openings,
};
use cobre_io::scenarios::LoadSeasonalStatsRow;
use cobre_io::scenarios::estimation::EstimationReport;
use cobre_io::scenarios::resolve_model_stage_seasons;
use cobre_io::{EntitySlot, SOFTWARE_NAME, SOFTWARE_VERSION};
use cobre_stochastic::StochasticContext;

use crate::TrainingResult;
use crate::cut::FutureCostFunction;
use crate::policy_export::{
    borrow_cut_records, build_active_indices, build_stage_basis_records, build_stage_cut_records,
    build_stage_cuts_payloads, build_stage_entity_manifest, build_stage_states_payloads,
    convert_basis_cache, scale_cut_records_for_export,
};
use crate::setup::{NodeGraph, NodePos, StudySetup};
use crate::stochastic_summary::{
    estimation_report_to_fitting_report, inflow_models_to_annual_component_rows,
    inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};
use crate::visited_states::VisitedStatesArchive;
use crate::workspace::CapturedBasis;

use cobre_core::{BlockMode, InflowModel, SeasonCycleType, System};

// ── Policy checkpoint ─────────────────────────────────────────────────────────

fn block_mode_label(mode: BlockMode) -> &'static str {
    match mode {
        BlockMode::Parallel => "parallel",
        BlockMode::Chronological => "chronological",
    }
}

fn training_block_provenance(modes: &[BlockMode]) -> (String, Vec<String>) {
    match modes.first() {
        None => (String::new(), Vec::new()),
        Some(&first) if modes.iter().all(|&m| m == first) => {
            (block_mode_label(first).to_string(), Vec::new())
        }
        Some(_) => (
            "mixed".to_string(),
            modes
                .iter()
                .map(|&m| block_mode_label(m).to_string())
                .collect(),
        ),
    }
}

/// One hydro's per-season autoregressive order, as the loading study itself
/// observes it. `orders[s] == None` means no inflow model of this hydro maps
/// to a stage on dense season ordinal `s` — the study has no PAR opinion
/// there. `orders[s] == Some(k)` is a fitted order, including `Some(0)`.
#[derive(Debug, Clone)]
pub struct StudyHydroSeasonOrders {
    /// Owning hydro's id.
    pub hydro_id: i32,
    /// Fitted AR order per dense season ordinal. Length equals
    /// [`StudySeasonManifest::n_seasons`].
    pub orders: Vec<Option<u32>>,
}

/// Study-side season-cycle and per-hydro PAR-order descriptor: the shape
/// [`SeasonManifest`] carries on the checkpoint wire, except each per-season
/// entry can be absent rather than zero-filled. A study whose horizon spans
/// only part of the declared cycle references a strict subset of its
/// seasons; this type lets the boundary-load season/PAR-identity gate tell
/// "the study never reached this season" from "the study fitted order zero
/// here", which a dense `u32` vector cannot.
#[derive(Debug, Clone)]
pub struct StudySeasonManifest {
    /// Season cycle discriminant; one of the `SEASON_CYCLE_CODE_*` constants.
    pub cycle_code: u8,
    /// Number of distinct seasons in the cycle; the length of every
    /// [`StudyHydroSeasonOrders::orders`] vector.
    pub n_seasons: u32,
    /// Per-hydro AR order opinions, in canonical ascending `hydro_id` order.
    pub hydro_orders: Vec<StudyHydroSeasonOrders>,
}

impl Default for StudySeasonManifest {
    /// The absent descriptor, mirroring [`SeasonManifest`]'s own `Default`.
    fn default() -> Self {
        Self {
            cycle_code: SEASON_CYCLE_CODE_ABSENT,
            n_seasons: 0,
            hydro_orders: Vec::new(),
        }
    }
}

impl StudySeasonManifest {
    /// Projects this descriptor onto the checkpoint wire type, mapping each
    /// `None` entry to `0`: the "no opinion" state lives only on this
    /// study-side type, never on the wire.
    #[must_use]
    pub fn to_season_manifest(&self) -> SeasonManifest {
        SeasonManifest {
            cycle_code: self.cycle_code,
            n_seasons: self.n_seasons,
            hydro_orders: self
                .hydro_orders
                .iter()
                .map(|h| HydroSeasonOrders {
                    hydro_id: h.hydro_id,
                    orders: h.orders.iter().map(|o| o.unwrap_or(0)).collect(),
                })
                .collect(),
        }
    }
}

/// Builds the study's season-cycle and per-hydro PAR-order descriptor.
///
/// This is the single owner both the checkpoint writer ([`write_checkpoint`],
/// via [`StudySeasonManifest::to_season_manifest`]) and the boundary-load
/// season/PAR-identity gate
/// ([`crate::policy::policy_load::BoundaryLoadRequest::with_study_seasons`])
/// build their descriptor from, so the two sides can never construct
/// incomparable descriptors.
///
/// Sources `n_seasons` from [`resolve_model_stage_seasons`]'s dense ordinals,
/// not raw `season_id`s — a sparse cycle (e.g. `Weekly` ids 21/26) would index
/// `orders` out of bounds otherwise. Resolving through the inflow models'
/// stage ids (not just `system.stages()`) keeps fitted models at synthesized
/// pre-study stage ids — partial-year studies whose AR lags reach past the
/// horizon — from silently dropping out of the descriptor.
#[must_use]
#[allow(clippy::cast_possible_truncation)] // season/AR-order counts are small
pub fn build_season_manifest(system: &System) -> StudySeasonManifest {
    let Some(season_map) = system.policy_graph().season_map.as_ref() else {
        return StudySeasonManifest::default();
    };

    let cycle_code = match season_map.cycle_type {
        SeasonCycleType::Monthly => SEASON_CYCLE_CODE_MONTHLY,
        SeasonCycleType::Weekly => SEASON_CYCLE_CODE_WEEKLY,
        SeasonCycleType::Custom => SEASON_CYCLE_CODE_CUSTOM,
    };

    let (stage_to_season, n_seasons) = resolve_model_stage_seasons(
        system.stages(),
        system.inflow_models().iter().map(|m| m.stage_id),
        season_map,
    );
    let hydro_orders = hydro_season_orders(system.inflow_models(), &stage_to_season, n_seasons);

    StudySeasonManifest {
        cycle_code,
        n_seasons: n_seasons as u32,
        hydro_orders,
    }
}

#[allow(clippy::cast_possible_truncation)] // season/AR-order counts are small
fn hydro_season_orders(
    inflow_models: &[InflowModel],
    stage_to_season: &HashMap<i32, usize>,
    n_seasons: usize,
) -> Vec<StudyHydroSeasonOrders> {
    // BTreeMap, not HashMap: ascending hydro_id order must not depend on
    // insertion order or hash-seed randomization.
    let mut orders_by_hydro = BTreeMap::new();
    for model in inflow_models {
        let Some(&season) = stage_to_season.get(&model.stage_id) else {
            continue;
        };
        let orders = orders_by_hydro
            .entry(model.hydro_id.0)
            .or_insert_with(|| vec![None; n_seasons]);
        orders[season] = Some(model.ar_order() as u32);
    }

    orders_by_hydro
        .into_iter()
        .map(|(hydro_id, orders)| {
            debug_assert_eq!(orders.len(), n_seasons, "orders must span every season");
            StudyHydroSeasonOrders { hydro_id, orders }
        })
        .collect()
}

/// Run-derived inputs to [`write_checkpoint`] (independent of the training
/// result). Stored in the checkpoint metadata for resume validation and
/// reproducibility; field types match `CheckpointManifest` widths to keep
/// the casts at this site.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointParams {
    /// Maximum iteration count from the active stopping rules.
    pub max_iterations: u64,
    /// Number of forward passes per iteration.
    pub forward_passes: u32,
    /// Random seed used for noise generation.
    pub seed: u64,
    /// When `false`, omit visited-state payloads even if the archive is
    /// populated. Controlled by the `exports.states` config flag.
    pub export_states: bool,
}

fn pool_entity_manifests(
    setup: &StudySetup,
    system: &System,
    n_pools: usize,
) -> Vec<Vec<EntitySlot>> {
    let global_layout = setup.stage_state();
    (0..n_pools)
        .map(|p| {
            // `p` is a pool ordinal; its owning stage resolves through
            // `pool_stage` — indexing `study_stage_ids` by `p` is OOB once
            // `n_pools > n_stages` on a branching graph (see `NodeGraph::pool_stage`).
            let stage_id = setup.inputs.study_stage_ids[setup.inputs.node_graph.pool_stage[p].0];
            build_stage_entity_manifest(
                system,
                global_layout,
                &setup.inputs.stage_data.study_dims.anticipated_plants,
                &setup.inputs.cut_state_layouts[p],
                stage_id,
            )
        })
        .collect()
}

/// The checkpoint inputs fixed for the whole run; a training-dependent value
/// belongs in [`CheckpointState`], or a repeated [`CheckpointLayout::write`]
/// would carry stale data.
#[derive(Debug, Clone)]
pub(crate) struct CheckpointLayout {
    stage_manifests: Vec<Vec<EntitySlot>>,
    study_stage_ids: Vec<i32>,
    study_stage_end_dates: Vec<NaiveDate>,
    cost_scale_factor: f64,
    n_stages: usize,
    graph_manifest: GraphManifest,
    season_manifest: SeasonManifest,
    training_block_mode: String,
    training_block_mode_per_stage: Vec<String>,
    params: CheckpointParams,
}

#[derive(Clone, Copy)]
pub(crate) struct CheckpointState<'a> {
    pub(crate) iterations: u64,
    pub(crate) final_lb: f64,
    pub(crate) final_ub: f64,
    pub(crate) basis_cache: &'a [Option<CapturedBasis>],
    pub(crate) visited_archive: Option<&'a VisitedStatesArchive>,
    pub(crate) lower_bound_history: &'a [f64],
}

impl<'a> CheckpointState<'a> {
    pub(crate) fn of(result: &'a TrainingResult) -> Self {
        Self {
            iterations: result.iterations,
            final_lb: result.final_lb,
            final_ub: result.final_ub,
            basis_cache: &result.basis_cache,
            visited_archive: result.visited_archive.as_ref(),
            lower_bound_history: &result.lower_bound_history,
        }
    }
}

impl CheckpointLayout {
    pub(crate) fn new(setup: &StudySetup, system: &System, params: CheckpointParams) -> Self {
        // `n_pools` sizes the pool-indexed manifests; `n_stages` is the metadata
        // field and comes from `setup.num_stages()` — NOT `fcf.pools.len()`, which
        // counts pools, equal to the stage count only on the chain degeneracy.
        let n_pools = setup.fcf.pools.len();
        let n_stages = setup.num_stages();

        let stage_manifests = pool_entity_manifests(setup, system, n_pools);

        let study_stage_end_dates: Vec<NaiveDate> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .map(|s| s.end_date)
            .collect();

        let study_modes: Vec<BlockMode> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .map(|s| s.block_mode)
            .collect();
        let (training_block_mode, training_block_mode_per_stage) =
            training_block_provenance(&study_modes);

        Self {
            stage_manifests,
            study_stage_ids: setup.inputs.study_stage_ids.clone(),
            study_stage_end_dates,
            cost_scale_factor: setup.inputs.stage_data.stage_templates.cost_scale_factor,
            n_stages,
            graph_manifest: setup.build_graph_manifest(),
            season_manifest: build_season_manifest(system).to_season_manifest(),
            training_block_mode,
            training_block_mode_per_stage,
            params,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn write(
        &self,
        policy_dir: &Path,
        fcf: &FutureCostFunction,
        node_graph: &NodeGraph,
        state: CheckpointState<'_>,
    ) -> Result<(), OutputError> {
        let cost_scale_factor = self.cost_scale_factor;
        let stage_records_internal = build_stage_cut_records(fcf);
        let stage_records_owned =
            scale_cut_records_for_export(&stage_records_internal, cost_scale_factor);
        let stage_records = borrow_cut_records(&stage_records_owned);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_cuts = build_stage_cuts_payloads(
            fcf,
            node_graph,
            &self.study_stage_ids,
            &self.study_stage_end_dates,
            cost_scale_factor,
            &stage_records,
            &stage_active_indices,
            &self.stage_manifests,
        );
        debug_assert!(
            stage_cuts
                .iter()
                .all(|p| p.cost_scale_factor.to_bits() == cost_scale_factor.to_bits()),
            "every pool must carry the study's single resolved cost_scale_factor"
        );

        let (basis_col_u8, basis_row_u8) = convert_basis_cache(state.basis_cache);
        let stage_bases = build_stage_basis_records(
            state.basis_cache,
            state.iterations,
            &basis_col_u8,
            &basis_row_u8,
        );

        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();

        let metadata = CheckpointManifest {
            format_version: FORMAT_VERSION,
            software: Some(SOFTWARE_NAME.to_string()),
            software_version: SOFTWARE_VERSION.to_string(),
            created_at: cobre_io::now_iso8601(),
            num_stages: self.n_stages as u32,
            graph_manifest: self.graph_manifest.clone(),
            producer: ProducerBlock {
                completed_iterations: state.iterations as u32,
                final_lower_bound: state.final_lb,
                best_upper_bound: Some(state.final_ub),
                max_iterations: self.params.max_iterations as u32,
                forward_passes: self.params.forward_passes,
                warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
                warm_start_counts,
                rng_seed: self.params.seed,
                total_visited_states: state.visited_archive.map_or(0, |a| {
                    (0..a.num_nodes()).map(|t| a.count(NodePos(t)) as u64).sum()
                }),
                training_block_mode: self.training_block_mode.clone(),
                training_block_mode_per_stage: self.training_block_mode_per_stage.clone(),
                cost_scale_factor: Some(cost_scale_factor),
                lower_bound_history: state.lower_bound_history.to_vec(),
            },
            season_manifest: self.season_manifest.clone(),
        };

        let stage_states = if self.params.export_states {
            build_stage_states_payloads(state.visited_archive, &self.stage_manifests, node_graph)
        } else {
            Vec::new()
        };

        write_policy_checkpoint(
            policy_dir,
            &stage_cuts,
            &stage_bases,
            &metadata,
            &stage_states,
        )
    }
}

/// Write the trained policy (cuts, bases, visited states, metadata) to
/// `policy_dir` as `FlatBuffers` files.
///
/// The single implementation shared by the CLI and the Python bindings, so the
/// on-disk format and write ordering cannot drift between them. `system` is
/// passed explicitly because [`StudySetup`] does not own it.
///
/// # Errors
///
/// Propagates [`OutputError`] from
/// [`cobre_io::output::policy::write_policy_checkpoint`] if any of the
/// `FlatBuffers` files cannot be written.
pub fn write_checkpoint(
    policy_dir: &Path,
    setup: &StudySetup,
    system: &System,
    training_result: &TrainingResult,
    params: &CheckpointParams,
) -> Result<(), OutputError> {
    CheckpointLayout::new(setup, system, *params).write(
        policy_dir,
        &setup.fcf,
        &setup.inputs.node_graph,
        CheckpointState::of(training_result),
    )
}

/// The checkpoint a run writes on the iterations its `policy.checkpointing`
/// schedule fires, built once by
/// [`StudySetup::enable_periodic_checkpoints`](crate::setup::StudySetup::enable_periodic_checkpoints)
/// with the final checkpoint's parameters and directory.
#[derive(Debug, Clone)]
pub struct PeriodicCheckpoint {
    schedule: CheckpointSchedule,
    policy_dir: PathBuf,
    layout: CheckpointLayout,
}

impl PeriodicCheckpoint {
    pub(crate) fn new(
        schedule: CheckpointSchedule,
        policy_dir: PathBuf,
        layout: CheckpointLayout,
    ) -> Self {
        Self {
            schedule,
            policy_dir,
            layout,
        }
    }

    pub(crate) fn fires_at(&self, iteration: u64) -> bool {
        self.schedule.fires_at(iteration)
    }

    pub(crate) fn policy_dir(&self) -> &Path {
        &self.policy_dir
    }

    pub(crate) fn write(
        &self,
        fcf: &FutureCostFunction,
        node_graph: &NodeGraph,
        state: CheckpointState<'_>,
    ) -> Result<(), OutputError> {
        self.layout.write(&self.policy_dir, fcf, node_graph, state)
    }
}

// ── Stochastic artifacts ──────────────────────────────────────────────────────

/// Write all applicable stochastic preprocessing artifacts to
/// `{output_dir}/stochastic/`.
///
/// Called when `exports.stochastic` is `true` in `config.json`. Each writer
/// invocation is independent: a failure produces a one-line message routed
/// through the caller-supplied `on_warning` callback and does not prevent
/// the remaining files (or training) from proceeding.
///
/// Every file is written unconditionally except `load_seasonal_stats.parquet`
/// (only when a load model has `std_mw > 0`) and `fitting_report.json` (only
/// when `estimation_report` is `Some`).
pub fn export_stochastic_artifacts(
    output_dir: &Path,
    stochastic: &StochasticContext,
    system: &System,
    estimation_report: Option<&EstimationReport>,
    on_warning: &mut dyn FnMut(&str),
) {
    let stochastic_dir = output_dir.join("stochastic");

    if let Err(e) = write_noise_openings(
        &stochastic_dir.join("noise_openings.parquet"),
        stochastic.opening_tree(),
    ) {
        on_warning(&format!("noise_openings: {e}"));
    }

    let stats_rows = inflow_models_to_stats_rows(system.inflow_models());
    if let Err(e) = write_inflow_seasonal_stats(
        &stochastic_dir.join("inflow_seasonal_stats.parquet"),
        &stats_rows,
    ) {
        on_warning(&format!("inflow_seasonal_stats: {e}"));
    }

    let ar_rows = inflow_models_to_ar_rows(system.inflow_models());
    if let Err(e) = write_inflow_ar_coefficients(
        &stochastic_dir.join("inflow_ar_coefficients.parquet"),
        &ar_rows,
    ) {
        on_warning(&format!("inflow_ar_coefficients: {e}"));
    }

    let annual_rows = inflow_models_to_annual_component_rows(system.inflow_models());
    if let Err(e) = write_inflow_annual_component(
        &stochastic_dir.join("inflow_annual_component.parquet"),
        &annual_rows,
    ) {
        on_warning(&format!("inflow_annual_component: {e}"));
    }

    if let Err(e) = write_correlation_json(
        &stochastic_dir.join("correlation.json"),
        system.correlation(),
    ) {
        on_warning(&format!("correlation: {e}"));
    }

    let has_stochastic_load = system.load_models().iter().any(|m| m.std_mw > 0.0);
    if has_stochastic_load {
        let load_rows: Vec<LoadSeasonalStatsRow> = system
            .load_models()
            .iter()
            .map(|m| LoadSeasonalStatsRow {
                bus_id: m.bus_id,
                stage_id: m.stage_id,
                mean_mw: m.mean_mw,
                std_mw: m.std_mw,
            })
            .collect();
        if let Err(e) = write_load_seasonal_stats(
            &stochastic_dir.join("load_seasonal_stats.parquet"),
            &load_rows,
        ) {
            on_warning(&format!("load_seasonal_stats: {e}"));
        }
    }

    if let Some(report) = estimation_report {
        let fitting = estimation_report_to_fitting_report(report);
        if let Err(e) = write_fitting_report(&stochastic_dir.join("fitting_report.json"), &fitting)
        {
            on_warning(&format!("fitting_report: {e}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::temporal::{
        Block, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{EntityId, HorizonGraph, SeasonDefinition, SeasonMap, Stage, SystemBuilder};
    use cobre_io::output::policy::SEASON_CYCLE_CODE_ABSENT;

    use super::{
        BlockMode, InflowModel, SEASON_CYCLE_CODE_WEEKLY, SeasonCycleType, System,
        build_season_manifest, hydro_season_orders, training_block_provenance,
    };

    #[test]
    fn uniform_parallel_study_summarizes_without_per_stage_list() {
        let (summary, per_stage) =
            training_block_provenance(&[BlockMode::Parallel, BlockMode::Parallel]);
        assert_eq!(summary, "parallel");
        assert!(per_stage.is_empty());
    }

    #[test]
    fn uniform_chronological_study_summarizes_without_per_stage_list() {
        let (summary, per_stage) =
            training_block_provenance(&[BlockMode::Chronological, BlockMode::Chronological]);
        assert_eq!(summary, "chronological");
        assert!(per_stage.is_empty());
    }

    #[test]
    fn mixed_study_reports_mixed_summary_and_full_per_stage_list() {
        let (summary, per_stage) = training_block_provenance(&[
            BlockMode::Parallel,
            BlockMode::Chronological,
            BlockMode::Parallel,
        ]);
        assert_eq!(summary, "mixed");
        assert_eq!(per_stage, vec!["parallel", "chronological", "parallel"]);
    }

    fn stage_with_season(id: i32, season_id: Option<usize>) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
            season_id,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn season_map(cycle_type: SeasonCycleType, n_seasons: usize) -> SeasonMap {
        SeasonMap {
            cycle_type,
            seasons: (0..n_seasons)
                .map(|id| SeasonDefinition {
                    id,
                    label: format!("S{id}"),
                    month_start: u32::try_from(id + 1).expect("season id fits a month"),
                    day_start: None,
                    month_end: None,
                    day_end: None,
                })
                .collect(),
        }
    }

    /// A 2-season `Weekly` cycle with sparse, non-contiguous ids `21`/`26` —
    /// mirrors the fixture that pinned `resolve_stage_seasons`'s densification.
    fn sparse_two_season_map() -> SeasonMap {
        SeasonMap {
            cycle_type: SeasonCycleType::Weekly,
            seasons: vec![
                SeasonDefinition {
                    id: 21,
                    label: "W22".to_string(),
                    month_start: 1,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                },
                SeasonDefinition {
                    id: 26,
                    label: "W27".to_string(),
                    month_start: 1,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                },
            ],
        }
    }

    fn inflow_model(hydro_id: i32, stage_id: i32, ar_coefficients: Vec<f64>) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 20.0,
            ar_coefficients,
            residual_std_ratio: 1.0,
            annual: None,
        }
    }

    fn system_with(
        stages: Vec<Stage>,
        season_map: Option<SeasonMap>,
        inflow_models: Vec<InflowModel>,
    ) -> System {
        SystemBuilder::new()
            .stages(stages)
            .policy_graph(HorizonGraph {
                season_map,
                ..HorizonGraph::default()
            })
            .inflow_models(inflow_models)
            .build()
            .expect("test system must be valid")
    }

    #[test]
    fn season_descriptor_reports_dense_ordinals_for_a_sparse_weekly_cycle() {
        let stages = vec![
            stage_with_season(0, Some(21)),
            stage_with_season(1, Some(26)),
        ];
        let system = system_with(
            stages,
            Some(sparse_two_season_map()),
            vec![
                inflow_model(1, 0, vec![0.4]),
                inflow_model(1, 1, vec![0.25, 0.1]),
            ],
        );

        let manifest = build_season_manifest(&system).to_season_manifest();

        assert_eq!(manifest.cycle_code, SEASON_CYCLE_CODE_WEEKLY);
        assert_eq!(manifest.n_seasons, 2);
        assert_eq!(manifest.hydro_orders.len(), 1);
        assert_eq!(manifest.hydro_orders[0].hydro_id, 1);
        assert_eq!(manifest.hydro_orders[0].orders, vec![1, 2]);
    }

    #[test]
    fn season_descriptor_absent_when_no_season_map_is_declared() {
        let system = SystemBuilder::new()
            .build()
            .expect("empty system must be valid");

        let manifest = build_season_manifest(&system);

        assert_eq!(manifest.cycle_code, SEASON_CYCLE_CODE_ABSENT);
        assert_eq!(manifest.n_seasons, 0);
        assert!(manifest.hydro_orders.is_empty());
    }

    #[test]
    fn season_descriptor_orders_vector_length_equals_n_seasons() {
        let stages = vec![
            stage_with_season(0, Some(0)),
            stage_with_season(1, Some(1)),
            stage_with_season(2, Some(2)),
            stage_with_season(3, Some(3)),
        ];
        let system = system_with(
            stages,
            Some(season_map(SeasonCycleType::Monthly, 4)),
            vec![inflow_model(1, 0, vec![0.3])],
        );

        let manifest = build_season_manifest(&system).to_season_manifest();

        assert_eq!(manifest.n_seasons, 4);
        assert_eq!(manifest.hydro_orders.len(), 1);
        assert_eq!(manifest.hydro_orders[0].orders, vec![1, 0, 0, 0]);
    }

    /// [`hydro_season_orders`] is fed the same multiset of models in two
    /// different orders (bypassing `SystemBuilder`, which requires
    /// `inflow_models` pre-sorted) — the `BTreeMap` grouping must not leak
    /// input order into the output.
    #[test]
    fn season_descriptor_is_invariant_to_hydro_declaration_order() {
        let stage_to_season: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();

        let hydro_2_stage_0 = inflow_model(2, 0, vec![0.4]);
        let hydro_2_stage_1 = inflow_model(2, 1, vec![0.1]);
        let hydro_5_stage_0 = inflow_model(5, 0, vec![0.2, 0.05]);
        let hydro_5_stage_1 = inflow_model(5, 1, vec![0.3]);

        let declared_5_first = vec![
            hydro_5_stage_0.clone(),
            hydro_5_stage_1.clone(),
            hydro_2_stage_0.clone(),
            hydro_2_stage_1.clone(),
        ];
        let declared_2_first = vec![
            hydro_2_stage_0,
            hydro_2_stage_1,
            hydro_5_stage_0,
            hydro_5_stage_1,
        ];

        let from_5_first = hydro_season_orders(&declared_5_first, &stage_to_season, 2);
        let from_2_first = hydro_season_orders(&declared_2_first, &stage_to_season, 2);

        assert_eq!(from_5_first.len(), 2);
        assert_eq!(from_5_first[0].hydro_id, 2);
        assert_eq!(from_5_first[0].orders, vec![Some(1), Some(1)]);
        assert_eq!(from_5_first[1].hydro_id, 5);
        assert_eq!(from_5_first[1].orders, vec![Some(2), Some(1)]);
        assert_eq!(
            format!("{from_5_first:?}"),
            format!("{from_2_first:?}"),
            "hydro declaration order must not affect the descriptor"
        );
    }

    #[test]
    fn season_descriptor_records_zero_for_a_missing_hydro_season_pair() {
        let stages = vec![
            stage_with_season(0, Some(0)),
            stage_with_season(1, Some(1)),
            stage_with_season(2, Some(2)),
        ];
        // Hydro 7 has no inflow model for season 1.
        let system = system_with(
            stages,
            Some(season_map(SeasonCycleType::Monthly, 3)),
            vec![
                inflow_model(7, 0, vec![0.4]),
                inflow_model(7, 2, vec![0.2, 0.1]),
            ],
        );

        let manifest = build_season_manifest(&system).to_season_manifest();

        assert_eq!(manifest.n_seasons, 3);
        assert_eq!(manifest.hydro_orders.len(), 1);
        assert_eq!(manifest.hydro_orders[0].hydro_id, 7);
        assert_eq!(manifest.hydro_orders[0].orders, vec![1, 0, 2]);
    }

    /// The study-side descriptor tells "no inflow model reaches this season"
    /// (`None`) from "the study fitted order zero here" (`Some(0)`), which the
    /// zero-filled checkpoint wire vector above cannot: hydro 7 has no model
    /// at season 1, so its study-side entry is `None`, not `Some(0)`.
    #[test]
    fn season_descriptor_marks_a_season_with_no_inflow_model_as_no_opinion() {
        let stages = vec![
            stage_with_season(0, Some(0)),
            stage_with_season(1, Some(1)),
            stage_with_season(2, Some(2)),
        ];
        let system = system_with(
            stages,
            Some(season_map(SeasonCycleType::Monthly, 3)),
            vec![
                inflow_model(7, 0, vec![0.4]),
                inflow_model(7, 2, vec![0.2, 0.1]),
            ],
        );

        let manifest = build_season_manifest(&system);

        assert_eq!(manifest.hydro_orders.len(), 1);
        assert_eq!(manifest.hydro_orders[0].hydro_id, 7);
        assert_eq!(
            manifest.hydro_orders[0].orders,
            vec![Some(1), None, Some(2)]
        );
    }

    /// A synthesized pre-study season a partial-year study's AR lags reach
    /// back into counts as referenced, the same as any of the study's own
    /// stages: its study-side entry is `Some`, never the `None` an unvisited
    /// season carries.
    #[test]
    fn season_descriptor_keeps_gap_season_orders_from_synthesized_prestudy_stages() {
        // Declared study stages: ids 0..3, seasons 8..11 (Sep-Dec).
        let month_stage = |id: i32, season: usize, start: (i32, u32), end: (i32, u32)| Stage {
            start_date: NaiveDate::from_ymd_opt(start.0, start.1, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(end.0, end.1, 1).expect("valid date"),
            ..stage_with_season(id, Some(season))
        };
        let stages = vec![
            month_stage(0, 8, (2024, 9), (2024, 10)),
            month_stage(1, 9, (2024, 10), (2024, 11)),
            month_stage(2, 10, (2024, 11), (2024, 12)),
            month_stage(3, 11, (2024, 12), (2025, 1)),
        ];
        let system = system_with(
            stages,
            Some(season_map(SeasonCycleType::Monthly, 12)),
            vec![
                // Synthesized pre-study ids: -1 -> season 7 (Aug), -2 -> season 6 (Jul).
                inflow_model(1, -2, vec![0.1, 0.05, 0.02]),
                inflow_model(1, -1, vec![0.2, 0.1]),
                inflow_model(1, 0, vec![0.3]),
                inflow_model(1, 1, vec![0.3]),
                inflow_model(1, 2, vec![0.3]),
                inflow_model(1, 3, vec![0.3]),
            ],
        );

        let manifest = build_season_manifest(&system);

        assert_eq!(manifest.n_seasons, 12);
        assert_eq!(manifest.hydro_orders.len(), 1);
        let orders = &manifest.hydro_orders[0].orders;
        assert_eq!(orders.len(), 12);
        assert_eq!(
            orders[7],
            Some(2),
            "gap season 7 (Aug) must keep the synthesized stage -1's order"
        );
        assert_eq!(
            orders[6],
            Some(3),
            "gap season 6 (Jul) must keep the synthesized stage -2's order"
        );
        for &season in &[0_usize, 1, 2, 3, 4, 5] {
            assert_eq!(
                orders[season], None,
                "untouched gap season {season} has no inflow model and carries no opinion"
            );
        }

        let wire_orders = manifest.to_season_manifest().hydro_orders[0].orders.clone();
        assert_eq!(
            wire_orders,
            orders.iter().map(|o| o.unwrap_or(0)).collect::<Vec<u32>>(),
            "the wire projection reads every None as 0, leaving the checkpoint byte-identical"
        );
    }
}
