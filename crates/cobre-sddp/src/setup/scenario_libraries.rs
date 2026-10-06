//! Scenario library builders for historical and external sampling schemes.
//!
//! Builders are not factored generically because external types have different
//! standardization semantics.

use std::collections::HashSet;

use cobre_core::{
    EntityId, Stage, System,
    scenario::{HistoricalYears, LoadModel, NcsModel, SamplingScheme},
    temporal::StageLagTransition,
};
use cobre_io::StageIdResolver;
use cobre_stochastic::{
    DerivedSeed, ExternalScenarioLibrary, HistoricalScenarioLibrary, PrecomputedNormal,
    PrecomputedPar, check_historical_structure, discover_historical_windows,
    pad_library_to_uniform, standardize_external_inflow, standardize_external_load,
    standardize_external_ncs, standardize_historical_windows, validate_external_library,
    validate_historical_library,
};

use crate::SddpError;
use crate::lp::builder::models_from_normal;

use super::{resolve_stage_lag_transitions, study_stages_slice};

/// Build and validate a [`HistoricalScenarioLibrary`] for inflow — the single
/// owner of window discovery, allocation, standardization and validation,
/// shared by the forward pass and the opening tree.
///
/// `par` is the PAR model the LP applies to this η; its width is
/// `par.max_order()`. `seed` ([`DerivedSeed`]) seeds the rolling η-inversion
/// chain, so every forward pass starting from the same derived seed exactly
/// reconstructs the raw historical observations. `min_windows` is the count
/// below which discovery and V2.6 warn; it never changes the pool.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on window discovery or validation failure.
pub(crate) fn build_historical_inflow_library(
    system: &System,
    par: &PrecomputedPar,
    seed: DerivedSeed<'_>,
    user_pool: Option<&HistoricalYears>,
    min_windows: u32,
) -> Result<HistoricalScenarioLibrary, SddpError> {
    let inflow_history = system.inflow_history();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let stages = study_stages_slice(system);
    let season_map = system.policy_graph().season_map.as_ref();
    let (downstream_par_order, stage_lag_transitions) =
        resolve_stage_lag_transitions(stages, par, season_map);

    let max_order = par.max_order();
    let window_years = discover_historical_windows(
        inflow_history,
        &hydro_ids,
        stages,
        user_pool,
        season_map,
        min_windows,
    )
    .map_err(SddpError::Stochastic)?;
    let mut library = HistoricalScenarioLibrary::new(
        window_years.len(),
        stages.len(),
        hydro_ids.len(),
        max_order,
        window_years.clone(),
    );
    let structure =
        check_historical_structure(&library, &hydro_ids, stages).map_err(SddpError::Stochastic)?;
    standardize_historical_windows(
        &structure,
        &mut library,
        inflow_history,
        par,
        &window_years,
        season_map,
        seed,
        &stage_lag_transitions,
        downstream_par_order,
    );
    validate_historical_library(&structure, &library, max_order, user_pool, min_windows)
        .map_err(SddpError::Stochastic)?;
    Ok(library)
}

fn per_stage_scenario_counts(
    stage_ids: impl Iterator<Item = i32>,
    stages: &[Stage],
    n_entities: usize,
) -> (Vec<usize>, Vec<usize>) {
    let resolver =
        StageIdResolver::from_study_stage_ids(&stages.iter().map(|s| s.id).collect::<Vec<_>>());
    let n_stages = stages.len();
    let mut rows_per_stage = vec![0usize; n_stages];
    for stage_id in stage_ids {
        if let Some(idx) = resolver.resolve(stage_id) {
            rows_per_stage[idx] += 1;
        }
    }
    let per_stage_scenarios = if n_entities > 0 {
        rows_per_stage.iter().map(|&r| r / n_entities).collect()
    } else {
        vec![0usize; n_stages]
    };
    (rows_per_stage, per_stage_scenarios)
}

/// Build and validate an [`ExternalScenarioLibrary`] for inflow.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_inflow_library(
    system: &System,
    par: &PrecomputedPar,
    seed: DerivedSeed<'_>,
    stage_lag_transitions: &[StageLagTransition],
    forward_passes: u32,
    downstream_par_order: usize,
) -> Result<ExternalScenarioLibrary, SddpError> {
    let external_rows = system.external_scenarios();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let stages = study_stages_slice(system);
    let n_stages = stages.len();
    let n_hydros = hydro_ids.len();
    let row_entity_ids: HashSet<EntityId> = external_rows.iter().map(|r| r.hydro_id).collect();
    let (rows_per_stage, per_stage_scenarios) =
        per_stage_scenario_counts(external_rows.iter().map(|r| r.stage_id), stages, n_hydros);
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library = ExternalScenarioLibrary::new(
        n_stages,
        n_scenarios_ext,
        n_hydros,
        "inflow",
        per_stage_scenarios,
    );
    standardize_external_inflow(
        &mut library,
        external_rows,
        &hydro_ids,
        stages,
        par,
        seed,
        stage_lag_transitions,
        downstream_par_order,
    );
    validate_external_library(
        &library,
        &hydro_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    pad_library_to_uniform(&mut library);
    Ok(library)
}

/// Build and validate an [`ExternalScenarioLibrary`] for load.
///
/// Canonical bus ID list from [`System::load_noise_member_bus_ids`] — the
/// single membership authority `noise_entity_order` and
/// [`resolve_lp_build_inputs`](super::lp_build_inputs::resolve_lp_build_inputs)
/// also route through, so a σ=0 or seasonal-stats-absent bus keeps the same
/// noise-vector slot everywhere.
/// `load_scheme` is the CALLING phase's own resolved scheme; a phase whose
/// scheme diverges from the training-derived noise-vector width is caught by
/// `assert_external_library_widths`, not here.
///
/// Standardizes against `normal_lp`/`normal_bus_ids` — the SAME
/// `PrecomputedNormal` and entity list `cobre_stochastic::context` builds for
/// reconstruction — rather than re-deriving moments independently from
/// `external_rows`, mirroring how [`build_external_inflow_library`]
/// standardizes against the reconstruction `PrecomputedPar` it receives.
/// `normal_bus_ids` is gated on the TRAINING scheme (the one scheme the
/// shared `PrecomputedNormal` is built under), which may be narrower than
/// this library's own `External`-gated `bus_ids` when the calling phase's
/// scheme diverges from training's; a bus outside `normal_bus_ids` then
/// standardizes to `(0.0, 0.0)`, matching what reconstruction itself would
/// give it — never a divergent, independently-derived value.
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_load_library(
    system: &System,
    load_scheme: SamplingScheme,
    forward_passes: u32,
    normal_lp: &PrecomputedNormal,
    normal_bus_ids: &[EntityId],
) -> Result<ExternalScenarioLibrary, SddpError> {
    let external_rows = system.external_load_scenarios();
    let stages = study_stages_slice(system);
    let n_stages = stages.len();
    let bus_ids = system.load_noise_member_bus_ids(load_scheme);
    let n_buses = bus_ids.len();
    let row_entity_ids: HashSet<EntityId> = external_rows.iter().map(|r| r.bus_id).collect();
    let (rows_per_stage, per_stage_scenarios) =
        per_stage_scenario_counts(external_rows.iter().map(|r| r.stage_id), stages, n_buses);
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library = ExternalScenarioLibrary::new(
        n_stages,
        n_scenarios_ext,
        n_buses,
        "load",
        per_stage_scenarios,
    );
    let stage_refs: Vec<&Stage> = stages.iter().collect();
    let derived_load_models = models_from_normal(
        normal_lp,
        normal_bus_ids,
        &stage_refs,
        |bus_id, stage_id, mean_mw, std_mw| LoadModel {
            bus_id,
            stage_id,
            mean_mw,
            std_mw,
        },
    );
    standardize_external_load(
        &mut library,
        external_rows,
        &bus_ids,
        &derived_load_models,
        stages,
    );
    validate_external_library(
        &library,
        &bus_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    pad_library_to_uniform(&mut library);
    Ok(library)
}

/// Build and validate an [`ExternalScenarioLibrary`] for NCS.
///
/// Canonical NCS ID list from [`System::ncs_noise_member_ids`] — this class
/// is only ever built under `External` (the caller gates the call site), so
/// an NCS with no `non_controllable_stats` row is still a noise member.
///
/// Standardizes against `ncs_normal`/`normal_ncs_ids` — the SAME
/// `PrecomputedNormal` and entity list `cobre_stochastic::context` builds for
/// reconstruction — rather than re-deriving moments independently, mirroring
/// [`build_external_load_library`]'s own fix (see its doc for the
/// TRAINING-scheme-gated `normal_ncs_ids` vs. this library's own
/// `External`-gated `ncs_ids` distinction).
///
/// # Errors
///
/// Returns `SddpError::Stochastic` on validation failure.
pub(crate) fn build_external_ncs_library(
    system: &System,
    forward_passes: u32,
    ncs_normal: &PrecomputedNormal,
    normal_ncs_ids: &[EntityId],
) -> Result<ExternalScenarioLibrary, SddpError> {
    let external_rows = system.external_ncs_scenarios();
    let stages = study_stages_slice(system);
    let n_stages = stages.len();
    let ncs_ids = system.ncs_noise_member_ids(SamplingScheme::External);
    let n_ncs = ncs_ids.len();
    let row_entity_ids: HashSet<EntityId> = external_rows.iter().map(|r| r.ncs_id).collect();
    let (rows_per_stage, per_stage_scenarios) =
        per_stage_scenario_counts(external_rows.iter().map(|r| r.stage_id), stages, n_ncs);
    let n_scenarios_ext = per_stage_scenarios.iter().copied().max().unwrap_or(0);
    let mut library =
        ExternalScenarioLibrary::new(n_stages, n_scenarios_ext, n_ncs, "ncs", per_stage_scenarios);
    let stage_refs: Vec<&Stage> = stages.iter().collect();
    let derived_ncs_models = models_from_normal(
        ncs_normal,
        normal_ncs_ids,
        &stage_refs,
        |ncs_id, stage_id, mean, std| NcsModel {
            ncs_id,
            stage_id,
            mean,
            std,
        },
    );
    standardize_external_ncs(
        &mut library,
        external_rows,
        &ncs_ids,
        &derived_ncs_models,
        stages,
    );
    validate_external_library(
        &library,
        &ncs_ids,
        &row_entity_ids,
        &rows_per_stage,
        n_stages,
        forward_passes,
    )
    .map_err(SddpError::Stochastic)?;
    pad_library_to_uniform(&mut library);
    Ok(library)
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, ExternalLoadRow, ExternalScenarioRow, InflowHistoryRow, InflowModel,
        NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig, System,
        SystemBuilder,
    };
    use cobre_stochastic::{PrecomputedNormal, StochasticError, derive_external_sample_moments};

    use super::{
        DerivedSeed, EntityId, LoadModel, PrecomputedPar, SamplingScheme, SddpError, Stage,
        StageLagTransition, build_external_inflow_library, build_external_load_library,
        build_historical_inflow_library,
    };

    fn single_stage(id: i32) -> Stage {
        Stage {
            index: 0,
            id,
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
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn finalizing_transition() -> StageLagTransition {
        StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        }
    }

    fn empty_derived_seed() -> DerivedSeed<'static> {
        DerivedSeed {
            lag_values: &[],
            l_state: 0,
            accum: &[],
            weight: &[],
        }
    }

    /// A `System` holding one hydro and the given external inflow rows — the
    /// production reader for [`build_external_inflow_library`]'s new
    /// `system`-sourced inputs.
    fn inflow_system(
        hydro_id: EntityId,
        stages: &[Stage],
        external_rows: Vec<ExternalScenarioRow>,
    ) -> System {
        let idx = usize::try_from(hydro_id.0).unwrap();
        SystemBuilder::new()
            .hydros(vec![crate::test_support::geometry_hydro(idx)])
            .stages(stages.to_vec())
            .external_scenarios(external_rows)
            .build()
            .expect("system must build")
    }

    /// A `sigma=0` hydro (deterministic PAR) whose external row does not match
    /// the deterministic value trips `solve_par_noise`'s `NEG_INFINITY`
    /// sentinel. `build_external_inflow_library` must surface it as a V3.7
    /// rejection through the real `standardize`-then-`validate` wiring, not
    /// silently accept it (the wiring previously ran `validate` against the
    /// still-zero-filled buffer, before `standardize_external_inflow` ever
    /// wrote eta, so V3.7 could never fire).
    #[test]
    fn external_inflow_sigma_zero_mismatch_rejected_by_v3_7() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0)];

        let models = vec![InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        let rows = vec![ExternalScenarioRow {
            stage_id: 0,
            scenario_id: 0,
            hydro_id,
            value_m3s: 999.0,
        }];
        let transitions = vec![finalizing_transition()];
        let system = inflow_system(hydro_id, &stages, rows);

        let result =
            build_external_inflow_library(&system, &par, empty_derived_seed(), &transitions, 1, 0);

        match result {
            Err(SddpError::Stochastic(StochasticError::InsufficientData { context })) => {
                assert!(
                    context.contains("V3.7"),
                    "expected a V3.7 rejection, got: {context}"
                );
            }
            other => panic!("expected a V3.7 rejection, got: {other:?}"),
        }
    }

    /// Negative control for the test above: when the external row matches the
    /// deterministic value exactly, `solve_par_noise` returns `0.0` (not the
    /// `NEG_INFINITY` sentinel) and the library builds successfully.
    #[test]
    fn external_inflow_sigma_zero_match_accepted() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0)];

        let models = vec![InflowModel {
            hydro_id,
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        let rows = vec![ExternalScenarioRow {
            stage_id: 0,
            scenario_id: 0,
            hydro_id,
            value_m3s: 100.0,
        }];
        let transitions = vec![finalizing_transition()];
        let system = inflow_system(hydro_id, &stages, rows);

        let result =
            build_external_inflow_library(&system, &par, empty_derived_seed(), &transitions, 1, 0);

        assert!(result.is_ok(), "expected Ok(()), got: {result:?}");
    }

    /// AC1: a σ=0 External LOAD bus reconstructs the external value, not a
    /// deliberately-disagreeing seasonal `mean_mw` (design doc §3/§4.1).
    #[test]
    fn external_load_sigma_zero_reconstructs_external_value_not_seasonal_mean() {
        let bus_id = EntityId(1);
        let stages = vec![single_stage(0)];
        let seasonal_load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: 999.0,
            std_mw: 0.0,
        }];
        let external_rows = vec![ExternalLoadRow {
            bus_id,
            stage_id: 0,
            scenario_id: 0,
            value_mw: 123.0,
        }];
        let system = SystemBuilder::new()
            .stages(stages.clone())
            .load_models(seasonal_load_models)
            .external_load_scenarios(external_rows.clone())
            .build()
            .expect("system must build");

        // Mirror context.rs's own training=External derivation: standardization
        // must consume this SAME `PrecomputedNormal`, not re-derive it.
        let moments = derive_external_sample_moments(
            &external_rows,
            &[bus_id],
            1,
            |row: &ExternalLoadRow| (row.bus_id, row.stage_id, row.scenario_id, row.value_mw),
        );
        let (mean, std) = moments[0];
        assert!(
            (mean - 123.0).abs() < 1e-10,
            "the standardization moment source must be the external sample, not the \
             seasonal mean_mw (999.0)"
        );
        assert!(
            std.abs() < 1e-10,
            "a single external sample -> sigma=0 exactly"
        );
        let derived_load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: mean,
            std_mw: std,
        }];
        let normal_lp = PrecomputedNormal::build(&derived_load_models, &[], &stages, &[bus_id], 1)
            .expect("normal_lp must build");

        let library = build_external_load_library(
            &system,
            SamplingScheme::External,
            1,
            &normal_lp,
            &[bus_id],
        )
        .expect("a single-column external load library must build");

        let eta = library.eta_slice(0, 0)[0];
        let realized = (mean + std * eta).max(0.0);
        assert!(
            (realized - 123.0).abs() < 1e-10,
            "reconstruction must equal the external value (123.0), not the seasonal \
             mean_mw (999.0); got {realized}"
        );
    }

    /// Regression: a divergent sim-only-External LOAD deck — training uses
    /// `InSample`, simulation uses `External` for the same bus — must
    /// standardize against the SAME `PrecomputedNormal`
    /// `cobre_stochastic::context` builds for the training-gated
    /// reconstruction, never re-derive independently from `external_rows`.
    /// Two distinct external scenarios give a nonzero external sample sigma
    /// (25.0), so a re-derivation would standardize a genuinely nonzero eta;
    /// pre-fix, `build_external_load_library` always called
    /// `derive_external_sample_moments` over the external sample regardless
    /// of which phase invoked it, so this bus's standardization sigma (25.0)
    /// disagreed with the training-gated reconstruction sigma (`0.0`, since a
    /// σ=0 bus is not an `InSample` noise member at all) — a silent
    /// divergence this test pins shut.
    #[test]
    fn external_load_divergent_sim_only_scheme_standardizes_against_training_gated_normal() {
        let bus_id = EntityId(1);
        let stages = vec![single_stage(0)];
        let seasonal_load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: 999.0,
            std_mw: 0.0,
        }];
        let external_rows = vec![
            ExternalLoadRow {
                bus_id,
                stage_id: 0,
                scenario_id: 0,
                value_mw: 100.0,
            },
            ExternalLoadRow {
                bus_id,
                stage_id: 0,
                scenario_id: 1,
                value_mw: 150.0,
            },
        ];
        let system = SystemBuilder::new()
            .stages(stages.clone())
            .load_models(seasonal_load_models)
            .external_load_scenarios(external_rows)
            .build()
            .expect("system must build");

        // Training's own scheme is InSample: default std_mw > 0.0 rule
        // excludes this sigma=0 bus from noise membership entirely, so
        // context.rs's non-External branch builds normal_lp over an empty
        // entity list for this bus.
        let normal_load_bus_ids = system.load_noise_member_bus_ids(SamplingScheme::InSample);
        assert!(
            normal_load_bus_ids.is_empty(),
            "a sigma=0 bus must not be an InSample noise member"
        );
        let max_blocks = stages.iter().map(|s| s.blocks.len()).max().unwrap_or(0);
        let normal_lp = PrecomputedNormal::build(
            system.load_models(),
            &[],
            &stages,
            &normal_load_bus_ids,
            max_blocks,
        )
        .expect("normal_lp must build");

        // Simulation's own scheme is External: external-additive
        // membership widens this library's own bus_ids to include the
        // sigma=0 bus, strictly wider than normal_load_bus_ids above --
        // exactly the divergent-scheme case the fix must not mis-standardize.
        let library = build_external_load_library(
            &system,
            SamplingScheme::External,
            1,
            &normal_lp,
            &normal_load_bus_ids,
        )
        .expect("a divergent sim-only-External load library must still build");

        for scenario in 0..2 {
            let eta = library.eta_slice(0, scenario)[0];
            assert!(
                eta.abs() < 1e-10,
                "a bus outside the training-gated normal_lp must standardize to eta=0 \
                 (the (0.0, 0.0) fallback) -- agreeing with what training's own \
                 reconstruction gives this bus -- not a nonzero eta derived from the \
                 external sample's own sigma (25.0), which would silently diverge \
                 from it; scenario {scenario} got {eta}"
            );
        }
    }

    /// AC2 + regression proof: a σ=0 External AR(0) inflow reconstructs the
    /// external value, including at a stage whose real scenario count is
    /// SMALLER than another stage's — a branching root's sole observation,
    /// the non-uniform shape the V3.7-vs-padding-order fix (design doc
    /// §3/§4.2) must not reject — and the padded phantom slot replicates the
    /// correct value, not a rejected/garbage sentinel.
    #[test]
    fn external_inflow_ar0_nonuniform_scenario_counts_reconstructs_external_values() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0), single_stage(1)];

        // Stage 0: a single real external scenario (sigma derives to 0
        // exactly). Stage 1: two distinct real scenarios (sigma > 0).
        let external_rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 60.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 10.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 1,
                hydro_id,
                value_m3s: 30.0,
            },
        ];

        // Mirror context.rs's AR(0) override: derive moments over the
        // external samples and rebuild PrecomputedPar from them.
        let moments = derive_external_sample_moments(
            &external_rows,
            &hydro_ids,
            2,
            |row: &ExternalScenarioRow| {
                (row.hydro_id, row.stage_id, row.scenario_id, row.value_m3s)
            },
        );
        let overridden_models: Vec<InflowModel> = (0..2_usize)
            .map(|s| {
                let (mean_m3s, std_m3s) = moments[s];
                InflowModel {
                    hydro_id,
                    stage_id: i32::try_from(s).unwrap(),
                    mean_m3s,
                    std_m3s,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                }
            })
            .collect();
        let par = PrecomputedPar::build(&overridden_models, &stages, &hydro_ids, None).unwrap();
        assert!(
            par.sigma(0, 0).abs() < 1e-10,
            "stage 0's single real scenario must derive sigma=0 exactly"
        );
        assert!(
            par.sigma(1, 0) > 0.0,
            "stage 1's two distinct real scenarios must derive sigma>0"
        );

        let transitions = vec![finalizing_transition(), finalizing_transition()];
        let system = inflow_system(hydro_id, &stages, external_rows);
        let library =
            build_external_inflow_library(&system, &par, empty_derived_seed(), &transitions, 2, 0)
                .expect(
                    "V3.7 must not reject stage 0 for having fewer real scenarios than stage 1",
                );

        let reconstruct = |stage: usize, scenario: usize| {
            let eta = library.eta_slice(stage, scenario)[0];
            par.deterministic_base(stage, 0) + par.sigma(stage, 0) * eta
        };

        assert!(
            (reconstruct(0, 0) - 60.0).abs() < 1e-10,
            "stage 0's real scenario must reconstruct the external value"
        );
        assert!(
            (reconstruct(1, 0) - 10.0).abs() < 1e-10,
            "stage 1's first real scenario must reconstruct the external value"
        );
        assert!(
            (reconstruct(1, 1) - 30.0).abs() < 1e-10,
            "stage 1's second real scenario must reconstruct the external value"
        );
        assert!(
            (reconstruct(0, 1) - 60.0).abs() < 1e-10,
            "the padded phantom slot at stage 0 must replicate the real root value \
             (60.0), not a rejected/garbage sentinel"
        );
    }

    /// Regression pin: an AR(p > 0) hydro in a non-uniform-per-stage External
    /// deck — stage 0 has one real scenario, a later stage has k — must NOT
    /// have a real later-stage slot's stored eta inverted against a
    /// fabricated phantom lag (`0.0`) instead of the lag the runtime forward
    /// pass actually feeds: every stage-1 branch descends from the SAME
    /// stage-0 root, so `accumulate_and_shift_lag_state`
    /// (`stochastic/noise.rs`) carries that root's own realized inflow to
    /// every child regardless of `scenario_id`. `standardize_external_inflow`
    /// (`sampling/external.rs`) replicates a stage's real raw values to
    /// uniform width BEFORE `run_eta_inversion`'s lag-chain advance for
    /// exactly this reason. `lag_realized` below is sourced independently of
    /// that fix: the branching root's own external value, recovered via this
    /// AR(1) hydro's own stage-0 stored eta (verified equal to the root's raw
    /// value) — never the lag the inversion used internally, which would mask
    /// a real bug were the fix ever weakened.
    #[test]
    fn external_inflow_ar1_nonuniform_scenario_counts_round_trip() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(0), single_stage(1)];

        // AR(1), untouched seasonal model: this ticket never overrides an
        // AR(p > 0) hydro's moments with derived external samples.
        let seasonal_model = |stage_id| InflowModel {
            hydro_id,
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![0.5],
            residual_std_ratio: 1.0,
            annual: None,
        };
        let seasonal_models = vec![seasonal_model(0), seasonal_model(1)];
        let par = PrecomputedPar::build(&seasonal_models, &stages, &hydro_ids, None).unwrap();
        let det_base = par.deterministic_base(0, 0);
        let psi = par.psi_slice(0, 0)[0];
        let sigma = par.sigma(0, 0);

        // Stage 0: a single real external scenario (the branching root).
        // Stage 1: two distinct real branches, both descending from that SAME
        // root.
        let external_rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 150.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 1,
                hydro_id,
                value_m3s: 250.0,
            },
        ];
        let derived_lag_values = [0.0_f64];
        let transitions = vec![finalizing_transition(), finalizing_transition()];
        let system = inflow_system(hydro_id, &stages, external_rows);

        let library = build_external_inflow_library(
            &system,
            &par,
            DerivedSeed {
                lag_values: &derived_lag_values,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &transitions,
            2,
            0,
        )
        .expect("V3.7 must not reject stage 0 for having fewer real scenarios than stage 1");

        let stage0_eta_real = library.eta_slice(0, 0)[0];
        let lag_realized = det_base + psi * derived_lag_values[0] + sigma * stage0_eta_real;
        assert!(
            (lag_realized - 200.0).abs() < 1e-10,
            "the branching root's own realized inflow must equal its external value; \
             got {lag_realized}"
        );

        let reconstruct_stage1 = |scenario: usize| {
            let eta = library.eta_slice(1, scenario)[0];
            det_base + psi * lag_realized + sigma * eta
        };

        let external_stage1 = [150.0_f64, 250.0_f64];
        for (scenario, &expected) in external_stage1.iter().enumerate() {
            let realized = reconstruct_stage1(scenario);
            assert!(
                (realized - expected).abs() < 1e-6,
                "stage 1 scenario {scenario}: reconstructed {realized}, expected {expected} \
                 (external value) using the REAL parent lag {lag_realized}"
            );
        }
    }

    /// Regression: a gapped/non-0-based External LOAD deck
    /// (declared stage ids `2`/`5`, never `0`/`1`) must resolve through the
    /// same canonical `stage_id -> index` mapping cobre-io's rule-47
    /// validator uses. Pre-fix, `rows_per_stage`'s `row.stage_id as usize`
    /// bound-check (`< n_stages`) silently dropped every row of a gapped
    /// deck outright, since the raw ids (2, 5) both exceed `n_stages` (2).
    #[test]
    fn external_load_library_resolves_gapped_stage_ids() {
        let bus_id = EntityId(1);
        let stages = vec![single_stage(2), single_stage(5)];
        let seasonal_load_models = vec![
            LoadModel {
                bus_id,
                stage_id: 2,
                mean_mw: 999.0,
                std_mw: 0.0,
            },
            LoadModel {
                bus_id,
                stage_id: 5,
                mean_mw: 999.0,
                std_mw: 0.0,
            },
        ];
        let external_rows = vec![
            ExternalLoadRow {
                bus_id,
                stage_id: 2,
                scenario_id: 0,
                value_mw: 123.0,
            },
            ExternalLoadRow {
                bus_id,
                stage_id: 5,
                scenario_id: 0,
                value_mw: 456.0,
            },
        ];

        let system = SystemBuilder::new()
            .stages(stages.clone())
            .load_models(seasonal_load_models)
            .external_load_scenarios(external_rows.clone())
            .build()
            .expect("system must build");

        // Resolve the declared ids the same way the fixed engine now must:
        // gapped id 2 -> canonical position 0, gapped id 5 -> position 1.
        let resolved_rows: Vec<(EntityId, i32, i32, f64)> = external_rows
            .iter()
            .map(|row| {
                let resolved = i32::from(row.stage_id == 5);
                (row.bus_id, resolved, row.scenario_id, row.value_mw)
            })
            .collect();
        let moments = derive_external_sample_moments(
            &resolved_rows,
            &[bus_id],
            2,
            |&(bus, stage_idx, scenario_id, value)| (bus, stage_idx, scenario_id, value),
        );
        // Mirror context.rs's own training=External derivation: standardization
        // must consume this SAME `PrecomputedNormal`, not re-derive it.
        let derived_load_models: Vec<LoadModel> = (0..2)
            .map(|resolved_idx| {
                let (mean, std) = moments[resolved_idx];
                LoadModel {
                    bus_id,
                    stage_id: stages[resolved_idx].id,
                    mean_mw: mean,
                    std_mw: std,
                }
            })
            .collect();
        let normal_lp = PrecomputedNormal::build(&derived_load_models, &[], &stages, &[bus_id], 1)
            .expect("normal_lp must build");

        let library = build_external_load_library(
            &system,
            SamplingScheme::External,
            1,
            &normal_lp,
            &[bus_id],
        )
        .expect("a gapped-stage-id external load deck must build, not drop every row");

        for (resolved_idx, expected) in [(0usize, 123.0_f64), (1usize, 456.0_f64)] {
            let (mean, std) = moments[resolved_idx];
            let eta = library.eta_slice(resolved_idx, 0)[0];
            let realized = (mean + std * eta).max(0.0);
            assert!(
                (realized - expected).abs() < 1e-10,
                "gapped declared stage id must resolve to canonical position {resolved_idx}; \
                 got {realized}, expected {expected}"
            );
        }
    }

    /// Regression, inflow counterpart: a gapped/non-0-based
    /// External INFLOW deck (declared stage ids `2`/`5`) must resolve
    /// through the same canonical mapping — both in the `rows_per_stage`
    /// count feeding V3.7 and in `standardize_external_inflow`'s own
    /// per-stage fill. Pre-fix, both silently dropped every row of this
    /// deck, since the raw ids exceed `n_stages` (2).
    #[test]
    fn external_inflow_library_resolves_gapped_stage_ids() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages = vec![single_stage(2), single_stage(5)];

        let external_rows = vec![
            ExternalScenarioRow {
                stage_id: 2,
                scenario_id: 0,
                hydro_id,
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 5,
                scenario_id: 0,
                hydro_id,
                value_m3s: 400.0,
            },
        ];

        // Mirror context.rs's AR(0) override: derive moments over the
        // RESOLVED (canonical) stage positions, the same mapping the engine
        // and the validator must both apply to the raw declared ids (2, 5).
        let resolved_rows: Vec<(EntityId, i32, i32, f64)> = external_rows
            .iter()
            .map(|row| {
                let resolved = i32::from(row.stage_id == 5);
                (row.hydro_id, resolved, row.scenario_id, row.value_m3s)
            })
            .collect();
        let moments = derive_external_sample_moments(
            &resolved_rows,
            &hydro_ids,
            2,
            |&(hydro, stage_idx, scenario_id, value)| (hydro, stage_idx, scenario_id, value),
        );
        let overridden_models: Vec<InflowModel> = stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| {
                let (mean_m3s, std_m3s) = moments[idx];
                InflowModel {
                    hydro_id,
                    stage_id: stage.id,
                    mean_m3s,
                    std_m3s,
                    ar_coefficients: vec![],
                    residual_std_ratio: 1.0,
                    annual: None,
                }
            })
            .collect();
        let par = PrecomputedPar::build(&overridden_models, &stages, &hydro_ids, None).unwrap();
        assert!(par.sigma(0, 0).abs() < 1e-10);
        assert!(par.sigma(1, 0).abs() < 1e-10);

        let transitions = vec![finalizing_transition(), finalizing_transition()];
        let system = inflow_system(hydro_id, &stages, external_rows);
        let library =
            build_external_inflow_library(&system, &par, empty_derived_seed(), &transitions, 1, 0)
                .expect("a gapped-stage-id external inflow deck must build, not drop every row");

        let reconstruct = |stage: usize| {
            let eta = library.eta_slice(stage, 0)[0];
            par.deterministic_base(stage, 0) + par.sigma(stage, 0) * eta
        };
        assert!(
            (reconstruct(0) - 200.0).abs() < 1e-10,
            "gapped declared id 2 must resolve to canonical position 0"
        );
        assert!(
            (reconstruct(1) - 400.0).abs() < 1e-10,
            "gapped declared id 5 must resolve to canonical position 1"
        );
    }

    #[test]
    fn historical_library_reports_a_seasonless_stage_before_standardizing() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let seasonless = Stage {
            index: 1,
            start_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
            season_id: None,
            ..single_stage(1)
        };
        let stages = vec![single_stage(0), seasonless];
        let models: Vec<InflowModel> = stages
            .iter()
            .map(|stage| InflowModel {
                hydro_id,
                stage_id: stage.id,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();
        let history: Vec<InflowHistoryRow> = (1990..=1992)
            .map(|year| InflowHistoryRow {
                hydro_id,
                start_date: NaiveDate::from_ymd_opt(year, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(year, 2, 1).unwrap(),
                value_m3s: 100.0,
            })
            .collect();
        let system = SystemBuilder::new()
            .hydros(vec![crate::test_support::geometry_hydro(1)])
            .stages(stages)
            .inflow_history(history)
            .build()
            .expect("system must build");

        let result = build_historical_inflow_library(&system, &par, empty_derived_seed(), None, 1);

        match result {
            Err(SddpError::Stochastic(StochasticError::InsufficientData { context })) => {
                assert!(
                    context.starts_with("V2.1: stage 1 (index 1)"),
                    "expected a V2.1 rejection naming stage 1, got: {context}"
                );
            }
            other => panic!("expected a V2.1 rejection, got: {other:?}"),
        }
    }
}
