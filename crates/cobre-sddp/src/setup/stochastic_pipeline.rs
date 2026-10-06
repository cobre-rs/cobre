//! Stochastic preprocessing pipeline: PAR estimation, opening tree loading, and stochastic context construction.

use std::collections::BTreeSet;
use std::path::Path;

use cobre_core::{
    EntityId, System,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_io::Config;
use cobre_io::LoadError;
use cobre_io::LoadFactorEntry;
use cobre_io::StageIdResolver;
use cobre_io::config::Openings;
use cobre_io::scenarios::assemble_opening_tree;
use cobre_io::scenarios::estimation::estimate_from_history;
use cobre_io::scenarios::load_noise_openings;
use cobre_io::scenarios::parse_load_factors;
use cobre_io::scenarios::validate_noise_openings;
use cobre_stochastic::BlockFactorPair;
use cobre_stochastic::ClassSchemes;
use cobre_stochastic::HistoricalScenarioLibrary;
use cobre_stochastic::build_inflow_par;
use cobre_stochastic::build_stochastic_context;
use cobre_stochastic::noise_entity_order;
use cobre_stochastic::normal::precompute::EntityFactorEntry;
use cobre_stochastic::par::lag_transition::precompute_noise_groups;
use cobre_stochastic::{OpeningTreeInputs, StochasticContext, context::OpeningTree};

use super::resolve_inflow_seeds;
use super::scenario_libraries;
use super::widen_lag_state_depth;
use crate::{EstimationPath, EstimationReport, SddpError};

/// Result of the stochastic preprocessing pipeline.
#[derive(Debug)]
pub struct PrepareStochasticResult {
    /// Updated system with estimated PAR models (if estimation ran).
    pub system: System,
    /// Built stochastic context, ready to pass to [`crate::setup::StudySetup::new`].
    pub stochastic: StochasticContext,
    /// Estimation report (`Some` if `inflow_history.parquet` was present and
    /// `inflow_seasonal_stats.parquet` was absent, triggering auto-estimation).
    pub estimation_report: Option<EstimationReport>,
    /// Which estimation path row was taken during preprocessing.
    pub estimation_path: EstimationPath,
}

fn class_schemes_for(training_source: &ScenarioSource) -> ClassSchemes {
    ClassSchemes {
        inflow: Some(training_source.inflow_scheme),
        load: Some(training_source.load_scheme),
        ncs: Some(training_source.ncs_scheme),
    }
}

pub(crate) fn forward_seed_for(source: &ScenarioSource) -> Option<u64> {
    source.seed.map(i64::unsigned_abs)
}

/// Load and validate a user-supplied opening tree when
/// `training.scenario_source.openings` declares `{source: file}`, reading the
/// convention-located `scenarios/noise_openings.parquet` — consumed by
/// declaration, not by file existence.
///
/// Returns `Ok(None)` when the declaration is absent or `generated` (the tree is
/// generated downstream). A present-but-undeclared file is therefore ignored.
///
/// # Errors
///
/// Returns [`SddpError::Io`] when `{source: file}` is declared but the
/// conventional file is absent, unreadable, or fails validation against each
/// stage's declared `num_openings` (its `branching_factor`).
fn load_user_opening_tree_inner(
    case_dir: &Path,
    system: &System,
    config: &Config,
    training_source: &ScenarioSource,
) -> Result<Option<OpeningTree>, SddpError> {
    match config.training_openings() {
        None | Some(Openings::Generated {}) => Ok(None),
        Some(Openings::File {}) => {
            let path = case_dir.join("scenarios").join("noise_openings.parquet");
            if !path.exists() {
                return Err(SddpError::Io(LoadError::io(
                    &path,
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "openings source is 'file' but the conventional \
                         scenarios/noise_openings.parquet is absent",
                    ),
                )));
            }

            let rows = load_noise_openings(Some(&path))?;

            let schemes = class_schemes_for(training_source);
            let expected_dim = noise_entity_order(system, &schemes).dim();

            let (study_stage_ids, expected_openings_per_stage): (Vec<i32>, Vec<usize>) = system
                .stages()
                .iter()
                .filter(|s| s.id >= 0)
                .map(|s| (s.id, s.scenario_config.branching_factor))
                .unzip();
            let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);

            validate_noise_openings(&rows, expected_dim, &expected_openings_per_stage, &resolver)?;

            Ok(Some(assemble_opening_tree(rows, expected_dim, &resolver)))
        }
    }
}

/// Noise-group ids for the study stages, delegating to [`precompute_noise_groups`].
///
/// Single owner: every site needing per-stage noise-group ids calls this rather
/// than inlining the filter — an inlined copy diverges whenever two consecutive
/// stages share a `(season_id, year)` group, changing which openings a rank solves
/// against.
#[must_use]
pub fn study_stage_noise_group_ids(system: &System) -> Vec<u32> {
    let study_stages: Vec<_> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect();
    precompute_noise_groups(&study_stages)
}

/// Build NCS entity factor entries from `System::resolved_ncs_factors()`.
///
/// Converts the dense 3D table into `(entity_id, stage_id, block_pairs)` tuples
/// for `PrecomputedNormal::build`.
#[must_use]
pub fn build_ncs_factor_entries(system: &System) -> Vec<(EntityId, i32, Vec<BlockFactorPair>)> {
    let stochastic_ncs: BTreeSet<EntityId> = system.ncs_models().iter().map(|m| m.ncs_id).collect();

    if stochastic_ncs.is_empty() {
        return Vec::new();
    }

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let ncs_ids: Vec<EntityId> = system
        .non_controllable_sources()
        .iter()
        .map(|n| n.id)
        .collect();

    let mut entries = Vec::new();
    for (ncs_idx, ncs_id) in ncs_ids.iter().enumerate() {
        if !stochastic_ncs.contains(ncs_id) {
            continue;
        }
        for (stage_idx, stage) in study_stages.iter().enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "block indices are far below i32::MAX"
            )]
            let block_pairs: Vec<BlockFactorPair> = stage
                .blocks
                .iter()
                .enumerate()
                .map(|(block_idx, _)| {
                    let factor = system
                        .resolved_ncs_factors()
                        .factor(ncs_idx, stage_idx, block_idx);
                    (block_idx as i32, factor)
                })
                .collect();
            entries.push((*ncs_id, stage.id, block_pairs));
        }
    }
    entries
}

/// Load `scenarios/load_factors.json` from the case directory, returning an
/// empty vec when the file is absent.
///
/// # Errors
///
/// Returns [`SddpError`] if the file exists but cannot be read or parsed.
pub fn load_load_factors_for_stochastic(
    case_dir: &Path,
) -> Result<Vec<LoadFactorEntry>, SddpError> {
    let path = case_dir.join("scenarios").join("load_factors.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    parse_load_factors(&path).map_err(SddpError::from)
}

/// Build the `HistoricalScenarioLibrary` for the opening tree when any stage
/// uses `NoiseMethod::HistoricalResiduals`. Returns `None` when no stage
/// needs historical draws.
fn build_opening_tree_library(
    system: &System,
    training_source: &ScenarioSource,
    declared_lag_depth: Option<u32>,
) -> Result<Option<HistoricalScenarioLibrary>, SddpError> {
    use cobre_core::temporal::NoiseMethod;
    let needs_historical_tree = system
        .stages()
        .iter()
        .any(|s| s.id >= 0 && s.scenario_config.noise_method == NoiseMethod::HistoricalResiduals);
    if !needs_historical_tree {
        return Ok(None);
    }
    let par = build_inflow_par(system, class_schemes_for(training_source).inflow)?;
    let lag_depth = widen_lag_state_depth(par.max_order(), declared_lag_depth);
    let seeds = resolve_inflow_seeds(system, lag_depth);
    Ok(Some(scenario_libraries::build_historical_inflow_library(
        system,
        &par,
        seeds.as_seed(lag_depth),
        training_source.historical_years.as_ref(),
        1,
    )?))
}

/// Per-class per-stage raw scenario count for opening-tree clamping, keyed by the
/// resolved study index. `None` when the class is not External. A row whose
/// `stage_id` does not resolve is impossible here — cobre-io's A2 rejects it at
/// load — and the resolve consumes the study stage-id resolver's map, never a
/// `stage_id as usize` index a gapped or 1-based domain would mis-key.
fn class_scenario_counts(
    scheme: SamplingScheme,
    stage_ids: impl Iterator<Item = i32>,
    n_entities: usize,
    n_stages: usize,
    resolver: &StageIdResolver,
) -> Option<Vec<usize>> {
    if scheme != SamplingScheme::External {
        return None;
    }
    let mut rows_per_stage = vec![0usize; n_stages];
    for stage_id in stage_ids {
        if let Some(idx) = resolver.resolve(stage_id) {
            rows_per_stage[idx] += 1;
        }
    }
    Some(if n_entities > 0 {
        rows_per_stage.iter().map(|&r| r / n_entities).collect()
    } else {
        vec![0usize; n_stages]
    })
}

/// Compute per-stage external scenario counts for opening tree clamping.
///
/// When any entity class uses External sampling, the external library is padded
/// to a uniform scenario count after loading. The opening tree generator must
/// clamp per-stage openings to the pre-padding raw count. P-B1 (cobre-io Layer
/// 5b) rejects any deck whose slot-occupying external classes disagree on the
/// per-stage raw count, so the present classes are guaranteed to agree here and
/// the first present class's vector is authoritative — never an element-wise
/// minimum, which would silently truncate a class's scenario set to a wrong
/// answer.
fn compute_external_scenario_counts(
    system: &System,
    training_source: &ScenarioSource,
) -> Option<Vec<usize>> {
    let study_stage_ids: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let n_stages = study_stage_ids.len();
    if n_stages == 0 {
        return None;
    }
    let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    let schemes = class_schemes_for(training_source);
    let noise_order = noise_entity_order(system, &schemes);

    let inflow_counts = class_scenario_counts(
        training_source.inflow_scheme,
        system.external_scenarios().iter().map(|r| r.stage_id),
        noise_order.hydro_ids.len(),
        n_stages,
        &resolver,
    );
    let load_counts = class_scenario_counts(
        training_source.load_scheme,
        system.external_load_scenarios().iter().map(|r| r.stage_id),
        noise_order.load_bus_ids.len(),
        n_stages,
        &resolver,
    );
    let ncs_counts = class_scenario_counts(
        training_source.ncs_scheme,
        system.external_ncs_scenarios().iter().map(|r| r.stage_id),
        noise_order.ncs_entity_ids.len(),
        n_stages,
        &resolver,
    );

    inflow_counts.or(load_counts).or(ncs_counts)
}

/// Run the stochastic preprocessing pipeline: PAR estimation, block factor
/// loading, opening-tree library construction (only when no opening tree is
/// supplied), and stochastic context build.
///
/// `inflow_lag_depth` is the boundary-inferred lag depth (from
/// `BoundaryStateRequirements::inflow_lag_depth`), or `None` for a study with no
/// loaded boundary; it sizes the opening-tree library's lag state to match the
/// state layout `StudySetup` reserves.
///
/// # Errors
///
/// Returns [`SddpError::Io`] on file read/parse/validation failure,
/// or [`SddpError::Stochastic`] on PAR/decomposition failure.
pub fn prepare_stochastic(
    system: System,
    case_dir: &Path,
    config: &Config,
    seed: u64,
    training_source: &ScenarioSource,
    inflow_lag_depth: Option<u32>,
) -> Result<PrepareStochasticResult, SddpError> {
    let (system, estimation_report, estimation_path) =
        estimate_from_history(system, case_dir, config)?;

    let user_opening_tree =
        load_user_opening_tree_inner(case_dir, &system, config, training_source)?;
    let external_scenario_counts = compute_external_scenario_counts(&system, training_source);

    let stochastic = build_stochastic_context_for_study(
        &system,
        case_dir,
        seed,
        training_source,
        inflow_lag_depth,
        user_opening_tree,
        external_scenario_counts,
    )?;

    Ok(PrepareStochasticResult {
        system,
        stochastic,
        estimation_report,
        estimation_path,
    })
}

/// Build the study's [`StochasticContext`] from a resolved (post-estimation)
/// `system` and the run's scenario inputs: the load/NCS factor entries (load
/// factors re-read from `case_dir`), the opening-tree historical library (only
/// when `user_tree` is `None`), and the forward seed. Both rank 0 (via
/// [`prepare_stochastic`], after PAR estimation and user-tree loading) and the CLI
/// non-root reconstruction call this one builder, so
/// the two paths cannot drift. `user_tree` and `external_scenario_counts` are the two inputs that
/// differ by rank — rank 0 loads / computes them; a non-root rank receives the tree
/// over the wire and passes `None` counts.
///
/// # Errors
///
/// Returns [`SddpError::Io`] on load-factor read/parse failure, or
/// [`SddpError::Stochastic`] on PAR / library-build / context-build failure.
pub fn build_stochastic_context_for_study(
    system: &System,
    case_dir: &Path,
    seed: u64,
    training_source: &ScenarioSource,
    inflow_lag_depth: Option<u32>,
    user_tree: Option<OpeningTree>,
    external_scenario_counts: Option<Vec<usize>>,
) -> Result<StochasticContext, SddpError> {
    let load_factor_entries = load_load_factors_for_stochastic(case_dir)?;
    let block_pairs: Vec<Vec<BlockFactorPair>> = load_factor_entries
        .iter()
        .map(|e| {
            e.block_factors
                .iter()
                .map(|bf| (bf.block_id, bf.factor))
                .collect()
        })
        .collect();
    let entity_factor_entries: Vec<EntityFactorEntry<'_>> = load_factor_entries
        .iter()
        .zip(block_pairs.iter())
        .map(|(e, pairs)| (e.bus_id, e.stage_id, pairs.as_slice()))
        .collect();

    let ncs_factor_entries = build_ncs_factor_entries(system);
    let ncs_entity_factor_entries: Vec<EntityFactorEntry<'_>> = ncs_factor_entries
        .iter()
        .map(|(ncs_id, stage_id, pairs)| (*ncs_id, *stage_id, pairs.as_slice()))
        .collect();

    let opening_tree_library = if user_tree.is_some() {
        None
    } else {
        build_opening_tree_library(system, training_source, inflow_lag_depth)?
    };

    Ok(build_stochastic_context(
        system,
        seed,
        forward_seed_for(training_source),
        &entity_factor_entries,
        &ncs_entity_factor_entries,
        OpeningTreeInputs {
            user_tree,
            historical_library: opening_tree_library.as_ref(),
            external_scenario_counts,
            noise_group_ids: Some(study_stage_noise_group_ids(system)),
        },
        class_schemes_for(training_source),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, HorizonGraph,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, InitialConditions, LineBlockBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
        ThermalStageBounds,
        entities::{
            bus::{Bus, DeficitSegment},
            hydro::{Hydro, HydroGenerationModel},
        },
        scenario::{
            ExternalLoadRow, ExternalScenarioRow, InflowHistoryRow, InflowModel, LoadModel,
        },
        temporal::{
            Block, BlockMode, NoiseMethod, PolicyGraphType, ScenarioSourceConfig, SeasonCycleType,
            SeasonDefinition, SeasonMap, Stage, StageLagTransition, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_stochastic::par::lag_transition::{
        derive_downstream_par_order, precompute_stage_lag_transitions,
    };
    use cobre_stochastic::{
        ComponentProvenance, PrecomputedPar, StochasticError, derive_inflow_seeds,
        par::lag_kernel::{DownstreamLagAccum, LagMajor, PrimaryLagAccum, advance_lag_chain},
        solve_par_noise,
    };

    /// Season definitions for the monthly->quarterly fixture below: three
    /// monthly seasons, then two quarterly seasons (`id >= 12`) — the same
    /// shape as the DLC fixture in `forward_sampler_integration.rs`.
    fn ring_season_map() -> SeasonMap {
        let def = |id: usize, month_start: u32, month_end: Option<u32>| SeasonDefinition {
            id,
            label: format!("S{id}"),
            month_start,
            day_start: None,
            month_end,
            day_end: None,
        };
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                def(0, 1, None),
                def(1, 2, None),
                def(2, 3, None),
                def(12, 4, Some(6)),
                def(13, 7, Some(9)),
            ],
        }
    }

    /// A standard 12-month `SeasonMap` (id `i` = calendar month `i+1`), the
    /// shape `derive_inflow_seeds`'s season-based backward walk needs.
    fn monthly_season_map() -> SeasonMap {
        let seasons: Vec<SeasonDefinition> = (0..12_u32)
            .map(|i| SeasonDefinition {
                id: i as usize,
                label: format!("Month{}", i + 1),
                month_start: i + 1,
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect();
        SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons,
        }
    }

    /// A `System` wired so `build_opening_tree_library` runs its production
    /// derivation end to end, rather than a hand-supplied `downstream_par_order`,
    /// for one hydro on `HistoricalResiduals`. [`ring_stage`] and [`ring_system`]
    /// build the calendar and the surrounding `System`; each of
    /// [`build_ring_fixture`] and [`build_nonaligned_ring_fixture`] supplies its
    /// own stages, PAR models, and history.
    struct RingFixture {
        system: System,
        stages: Vec<Stage>,
        hydro_id: EntityId,
        raw: Vec<f64>,
        past_inflow_seed: f64,
    }

    const RING_HYDRO_ID: EntityId = EntityId(2);
    const RING_BUS_ID: EntityId = EntityId(1);

    /// One `Stage` for the ring fixtures: single-block, `Parallel`, one hydro
    /// on `HistoricalResiduals` with both storage and inflow-lag state.
    fn ring_stage(
        index: usize,
        id: i32,
        start: NaiveDate,
        end: NaiveDate,
        season_id: usize,
        duration_hours: f64,
    ) -> Stage {
        Stage {
            index,
            id,
            start_date: start,
            end_date: end,
            season_id: Some(season_id),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::HistoricalResiduals,
            },
        }
    }

    /// The one-hydro, one-bus `System` shell shared by every ring fixture:
    /// `stages` supplies its calendar, `inflow_models`/`inflow_history` its PAR
    /// and window-discovery data, `season_map` its policy-graph season map.
    #[expect(
        clippy::too_many_lines,
        reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
    )]
    fn ring_system(
        stages: &[Stage],
        inflow_models: Vec<InflowModel>,
        inflow_history: Vec<InflowHistoryRow>,
        season_map: SeasonMap,
        external_scenarios: Vec<ExternalScenarioRow>,
    ) -> System {
        let n_stages = stages.len();

        let load_models: Vec<LoadModel> = stages
            .iter()
            .map(|s| LoadModel {
                bus_id: RING_BUS_ID,
                stage_id: s.id,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let bus = Bus {
            id: RING_BUS_ID,
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: RING_HYDRO_ID,
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(RING_BUS_ID);

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: Some(season_map),
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages.to_vec())
            .inflow_models(inflow_models)
            .load_models(load_models)
            .inflow_history(inflow_history)
            .external_scenarios(external_scenarios)
            .bounds(bounds)
            .penalties(penalties)
            .policy_graph(policy_graph)
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: vec![],
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .build()
            .expect("ring fixture: valid system")
    }

    /// A [`RingFixture`] with 5 study stages (three monthly, Jan-Mar 2026,
    /// then two quarterly, Q2-Q3 2026), AR(1) throughout.
    fn build_ring_fixture() -> RingFixture {
        let stages = vec![
            ring_stage(
                0,
                0,
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                0,
                31.0 * 24.0,
            ),
            ring_stage(
                1,
                1,
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                1,
                28.0 * 24.0,
            ),
            ring_stage(
                2,
                2,
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                2,
                31.0 * 24.0,
            ),
            ring_stage(
                3,
                3,
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
                12,
                91.0 * 24.0,
            ),
            ring_stage(
                4,
                4,
                NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 10, 1).unwrap(),
                13,
                92.0 * 24.0,
            ),
        ];

        let raw = vec![80.0, 150.0, 40.0, 190.0, 60.0];
        let past_inflow_seed = 70.0;

        let mut inflow_models = Vec::with_capacity(6);
        for stage_id in -1..5_i32 {
            inflow_models.push(InflowModel {
                hydro_id: RING_HYDRO_ID,
                stage_id,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: if stage_id >= 0 { vec![0.6] } else { vec![] },
                residual_std_ratio: 1.0,
                annual: None,
            });
        }

        let inflow_history: Vec<InflowHistoryRow> = stages
            .iter()
            .enumerate()
            .map(|(t, stage)| InflowHistoryRow {
                hydro_id: RING_HYDRO_ID,
                start_date: stage.start_date,
                end_date: stage.end_date,
                value_m3s: raw[t],
            })
            .collect();

        let system = ring_system(
            &stages,
            inflow_models,
            inflow_history,
            ring_season_map(),
            Vec::new(),
        );

        RingFixture {
            system,
            stages,
            hydro_id: RING_HYDRO_ID,
            raw,
            past_inflow_seed,
        }
    }

    /// Drive `advance_lag_chain::<LagMajor>` across `fx`'s stages with lag
    /// width `lag_width`, returning the incoming (pre-advance) lag vector at
    /// `at_stage` — the values the PAR model there reads. `downstream_par_order
    /// = 0` (`accumulator`/`completed_lags` empty) reproduces the literal-`0`
    /// regression `build_opening_tree_library` used to pass to
    /// `standardize_historical_windows`; a positive value reproduces the
    /// ring-aware, fixed behavior.
    fn ring_incoming_lags(
        fx: &RingFixture,
        transitions: &[StageLagTransition],
        downstream_par_order: usize,
        lag_width: usize,
        at_stage: usize,
    ) -> Vec<f64> {
        let layout = LagMajor {
            entity_count: 1,
            max_order: lag_width,
        };
        let mut lag_state = vec![fx.past_inflow_seed; lag_width];
        let mut incoming = vec![0.0; lag_width];
        let mut primary_acc = vec![0.0];
        let mut primary_w = vec![0.0];
        let mut ds_acc = if downstream_par_order > 0 {
            vec![0.0]
        } else {
            Vec::new()
        };
        let mut ds_completed = vec![0.0; downstream_par_order];
        let mut ds_w = 0.0_f64;
        let mut ds_n = 0usize;
        let mut captured = vec![0.0; lag_width];

        for (t, (stage_lag, &raw)) in transitions.iter().zip(fx.raw.iter()).enumerate() {
            incoming.copy_from_slice(&lag_state);
            if t == at_stage {
                captured.copy_from_slice(&incoming);
            }
            let mut primary = PrimaryLagAccum {
                accumulator: &mut primary_acc,
                weight_accum: &mut primary_w,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut ds_acc,
                weight_accum: &mut ds_w,
                completed_lags: &mut ds_completed,
                n_completed: &mut ds_n,
                par_order: downstream_par_order,
            };
            advance_lag_chain(
                layout,
                &mut lag_state,
                &incoming,
                &[raw],
                stage_lag,
                &mut primary,
                &mut downstream,
            );
        }
        captured
    }

    /// `build_opening_tree_library` (rank-0 opening-tree build) must thread
    /// `derive_downstream_par_order`'s result into `standardize_historical_windows`,
    /// not a literal `0`: the resulting eta at the quarterly transition must
    /// match the independent ring-aware forward-chain oracle and differ from
    /// the primary-only advance the literal-`0` regression produces.
    #[test]
    fn build_opening_tree_library_ring_aware_eta_at_quarterly_transition() {
        let fx = build_ring_fixture();
        let training_source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };

        let lib = build_opening_tree_library(&fx.system, &training_source, None)
            .expect("build_opening_tree_library must succeed")
            .expect("HistoricalResiduals noise method must build a library");
        assert_eq!(
            lib.n_windows(),
            1,
            "exactly one historical window (2026) is discoverable"
        );

        let par =
            PrecomputedPar::build(fx.system.inflow_models(), &fx.stages, &[fx.hydro_id], None)
                .expect("oracle PrecomputedPar must build");
        assert_eq!(par.max_order(), 1);

        let derived = derive_downstream_par_order(
            &fx.stages,
            &par,
            fx.system.policy_graph().season_map.as_ref(),
        );
        assert_eq!(
            derived,
            par.max_order(),
            "the fixture crosses season_id >= 12 at stage 3; derive_downstream_par_order \
             must gate to par.max_order(), not 0"
        );
        let season_map = ring_season_map();
        let transitions = precompute_stage_lag_transitions(&fx.stages, &season_map, derived);

        let ring_aware_incoming = ring_incoming_lags(&fx, &transitions, derived, 1, 4)[0];
        let naive_incoming = ring_incoming_lags(&fx, &transitions, 0, 1, 4)[0];
        assert!(
            (ring_aware_incoming - naive_incoming).abs() > 1.0,
            "ring-rebuilt lag feeding stage 4 must differ from the primary-only advance, \
             got ring_aware={ring_aware_incoming} vs naive={naive_incoming}"
        );

        let det_base = par.deterministic_base(4, 0);
        let psi = par.psi_slice(4, 0);
        let sigma = par.sigma(4, 0);
        let expected_ring_aware_eta =
            solve_par_noise(det_base, psi, &[ring_aware_incoming], sigma, fx.raw[4]);
        let expected_naive_eta =
            solve_par_noise(det_base, psi, &[naive_incoming], sigma, fx.raw[4]);

        let eta_stage4 = lib.eta_slice(0, 4)[0];
        assert_eq!(
            eta_stage4, expected_ring_aware_eta,
            "build_opening_tree_library's eta at the quarterly transition must match the \
             independent ring-aware forward-chain oracle"
        );
        assert!(
            (eta_stage4 - expected_naive_eta).abs() > 1e-6,
            "build_opening_tree_library's eta must differ from the primary-only advance \
             (the literal downstream_par_order=0 regression value), got eta={eta_stage4} \
             vs naive={expected_naive_eta}"
        );
    }

    fn build_pool_divergence_system() -> System {
        let stages = vec![
            ring_stage(
                0,
                0,
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                0,
                31.0 * 24.0,
            ),
            ring_stage(
                1,
                1,
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                1,
                28.0 * 24.0,
            ),
            ring_stage(
                2,
                2,
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                2,
                31.0 * 24.0,
            ),
        ];

        let mut inflow_models = Vec::with_capacity(4);
        for stage_id in -1..3_i32 {
            inflow_models.push(InflowModel {
                hydro_id: RING_HYDRO_ID,
                stage_id,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: if stage_id >= 0 { vec![0.5] } else { vec![] },
                residual_std_ratio: 1.0,
                annual: None,
            });
        }

        let month_row = |year: i32, month: u32, value: f64| InflowHistoryRow {
            hydro_id: RING_HYDRO_ID,
            start_date: NaiveDate::from_ymd_opt(year, month, 1).unwrap(),
            end_date: if month == 12 {
                NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap()
            },
            value_m3s: value,
        };

        let mut inflow_history = Vec::new();
        for m in 1..=3u32 {
            inflow_history.push(month_row(1990, m, 200.0 + f64::from(m)));
        }
        for m in 1..=3u32 {
            inflow_history.push(month_row(1991, m, 400.0 + f64::from(m)));
        }

        ring_system(
            &stages,
            inflow_models,
            inflow_history,
            monthly_season_map(),
            Vec::new(),
        )
    }

    /// The opening tree's window pool and eta must equal the forward pass's:
    /// both read the SAME applied PAR and the SAME derived seed, so the pool
    /// (and every shared window's eta) can never diverge.
    #[test]
    fn opening_tree_and_forward_libraries_share_windows_and_eta() {
        let system = build_pool_divergence_system();
        let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
        let study_stages: Vec<Stage> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let par = PrecomputedPar::build(system.inflow_models(), &study_stages, &hydro_ids, None)
            .expect("par must build");
        assert_eq!(par.max_order(), 1, "fixture precondition: p = 1");

        let training_source = ScenarioSource {
            inflow_scheme: SamplingScheme::Historical,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: Some(7),
            historical_years: None,
        };
        let config = config_from_json(&format!(
            r#"{{"training": {{{SAMPLED_TRAINING}, "scenario_source": {{"seed": 7, "inflow": {{"scheme": "historical"}}}}}}}}"#
        ));
        let tmp = tempfile::TempDir::new().expect("tempdir");

        let stochastic = build_stochastic_context_for_study(
            &system,
            tmp.path(),
            42,
            &training_source,
            Some(3),
            None,
            None,
        )
        .expect("stochastic context must build");

        let setup = crate::setup::StudySetup::new_with_boundary_requirements(
            &system,
            &config,
            stochastic,
            crate::hydro_models::PrepareHydroModelsResult::default_from_system(&system),
            crate::setup::BoundaryStateRequirements::present(3),
            Vec::new(),
        )
        .expect("StudySetup must build");

        assert_eq!(
            setup.inputs.stage_data.state.max_par_order, 3,
            "precondition: the state layout widens to the declared boundary depth"
        );
        let fwd = setup
            .inputs
            .scenario_libraries
            .training
            .historical
            .as_ref()
            .expect("training scheme is Historical: a forward library must build");
        let fwd_years: Vec<i32> = (0..fwd.n_windows()).map(|w| fwd.window_year(w)).collect();
        assert_eq!(
            fwd_years,
            vec![1990, 1991],
            "precondition: the forward pass admits both years holding their study seasons"
        );

        let tree = build_opening_tree_library(&system, &training_source, Some(3))
            .expect("build_opening_tree_library must succeed")
            .expect("HistoricalResiduals noise method must build a library");
        let tree_years: Vec<i32> = (0..tree.n_windows()).map(|w| tree.window_year(w)).collect();

        assert_eq!(
            tree_years, fwd_years,
            "the opening tree's window pool must equal the forward pass's"
        );
        assert_eq!(tree.max_order(), fwd.max_order());
        assert_eq!(tree.seed_digest(), fwd.seed_digest());
        for w in 0..fwd.n_windows() {
            for t in 0..study_stages.len() {
                assert_eq!(
                    tree.eta_slice(w, t),
                    fwd.eta_slice(w, t),
                    "window {w} stage {t}: eta must match between the tree and the forward pass"
                );
            }
        }
    }

    /// A two-stage (Jan-Feb 2026), one-AR(0)-hydro system with `External`
    /// inflow scenarios (mean 400, std 100) that diverge sharply from the
    /// fitted moments (mean 100, std 10) and a 1990 historical window the
    /// opening tree draws its eta from.
    fn build_external_ar0_system() -> System {
        let stages = vec![
            ring_stage(
                0,
                0,
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                0,
                31.0 * 24.0,
            ),
            ring_stage(
                1,
                1,
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                1,
                28.0 * 24.0,
            ),
        ];

        let mut inflow_models = Vec::with_capacity(3);
        for stage_id in -1..2_i32 {
            inflow_models.push(InflowModel {
                hydro_id: RING_HYDRO_ID,
                stage_id,
                mean_m3s: 100.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            });
        }

        let inflow_history = vec![
            InflowHistoryRow {
                hydro_id: RING_HYDRO_ID,
                start_date: NaiveDate::from_ymd_opt(1990, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(1990, 2, 1).unwrap(),
                value_m3s: 120.0,
            },
            InflowHistoryRow {
                hydro_id: RING_HYDRO_ID,
                start_date: NaiveDate::from_ymd_opt(1990, 2, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(1990, 3, 1).unwrap(),
                value_m3s: 130.0,
            },
        ];

        let external_scenarios = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id: RING_HYDRO_ID,
                value_m3s: 300.0,
            },
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 1,
                hydro_id: RING_HYDRO_ID,
                value_m3s: 500.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id: RING_HYDRO_ID,
                value_m3s: 300.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 1,
                hydro_id: RING_HYDRO_ID,
                value_m3s: 500.0,
            },
        ];

        ring_system(
            &stages,
            inflow_models,
            inflow_history,
            monthly_season_map(),
            external_scenarios,
        )
    }

    /// Under training `External`, the opening tree must invert eta with the
    /// PAR the LP applies (the External-overridden moments), not the fitted
    /// one — otherwise the reconstructed value diverges from the raw
    /// historical observation the tree drew its window from.
    #[test]
    fn opening_tree_inverts_eta_with_the_applied_par_under_external_training() {
        let system = build_external_ar0_system();
        let raw = [120.0, 130.0];

        let training_source = ScenarioSource {
            inflow_scheme: SamplingScheme::External,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: Some(7),
            historical_years: None,
        };
        let counts = compute_external_scenario_counts(&system, &training_source);
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let stochastic = build_stochastic_context_for_study(
            &system,
            tmp.path(),
            42,
            &training_source,
            None,
            None,
            counts,
        )
        .expect("stochastic context must build");
        let applied = stochastic.par();

        assert_eq!(applied.max_order(), 0, "fixture precondition: AR(0)");
        assert!(
            (applied.deterministic_base(0, 0) - 100.0).abs() > 1.0,
            "negative control: the External override must have moved the mean away from \
             the fitted value (100.0)"
        );

        let tree = build_opening_tree_library(&system, &training_source, None)
            .expect("build_opening_tree_library must succeed")
            .expect("HistoricalResiduals noise method must build a library");

        for (t, &raw_t) in raw.iter().enumerate() {
            let reconstructed =
                applied.deterministic_base(t, 0) + applied.sigma(t, 0) * tree.eta_slice(0, t)[0];
            assert!(
                (reconstructed - raw_t).abs() <= 1e-9 * raw_t.abs(),
                "stage {t}: reconstructing with the applied PAR must reproduce the raw \
                 historical observation ({raw_t}), got {reconstructed}"
            );
        }
    }

    /// Season definitions for the non-calendar-aligned ring fixture below:
    /// monthly Nov-Apr (seasons 10, 11, 0-3), then quarterly May-Jul and
    /// Aug-Oct (seasons 12, 13) — a quarterly regime that starts mid-calendar-
    /// quarter (May, not a calendar-quarter boundary).
    fn nonaligned_ring_season_map() -> SeasonMap {
        let month = |id: usize, month_start: u32| SeasonDefinition {
            id,
            label: format!("S{id}"),
            month_start,
            day_start: None,
            month_end: None,
            day_end: None,
        };
        let quarter = |id: usize, month_start: u32, month_end: u32| SeasonDefinition {
            id,
            label: format!("S{id}"),
            month_start,
            day_start: None,
            month_end: Some(month_end),
            day_end: None,
        };
        SeasonMap {
            cycle_type: SeasonCycleType::Custom,
            seasons: vec![
                month(10, 11),
                month(11, 12),
                month(0, 1),
                month(1, 2),
                month(2, 3),
                month(3, 4),
                quarter(12, 5, 7),
                quarter(13, 8, 10),
            ],
        }
    }

    /// A [`RingFixture`] whose quarterly regime starts mid-calendar-quarter
    /// (May, not a calendar-quarter boundary): 5 monthly stages (Dec 2025,
    /// season 11, through Apr 2026, season 3), then 2 quarterly stages
    /// (May-Jul 2026, season 12; Aug-Oct 2026, season 13), AR(1) on the
    /// monthly stages and AR(2) on the quarterly ones — only an AR order
    /// `p >= 2` lets the downstream ring's width (the PAR order, not a
    /// declared lag depth) change which completed quarter survives eviction
    /// at the transition.
    fn build_nonaligned_ring_fixture() -> RingFixture {
        let stages = vec![
            ring_stage(
                0,
                0,
                NaiveDate::from_ymd_opt(2025, 12, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                11,
                31.0 * 24.0,
            ),
            ring_stage(
                1,
                1,
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                0,
                31.0 * 24.0,
            ),
            ring_stage(
                2,
                2,
                NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                1,
                28.0 * 24.0,
            ),
            ring_stage(
                3,
                3,
                NaiveDate::from_ymd_opt(2026, 3, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                2,
                31.0 * 24.0,
            ),
            ring_stage(
                4,
                4,
                NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
                3,
                30.0 * 24.0,
            ),
            ring_stage(
                5,
                5,
                NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 8, 1).unwrap(),
                12,
                92.0 * 24.0,
            ),
            ring_stage(
                6,
                6,
                NaiveDate::from_ymd_opt(2026, 8, 1).unwrap(),
                NaiveDate::from_ymd_opt(2026, 11, 1).unwrap(),
                13,
                92.0 * 24.0,
            ),
        ];

        let raw = vec![60.0, 150.0, 40.0, 190.0, 80.0, 120.0, 90.0];
        let past_inflow_seed = 70.0;

        let inflow_models: Vec<InflowModel> = stages
            .iter()
            .enumerate()
            .map(|(idx, stage)| InflowModel {
                hydro_id: RING_HYDRO_ID,
                stage_id: stage.id,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: if idx < 5 { vec![0.6] } else { vec![0.5, 0.3] },
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let inflow_history: Vec<InflowHistoryRow> = stages
            .iter()
            .enumerate()
            .map(|(t, stage)| InflowHistoryRow {
                hydro_id: RING_HYDRO_ID,
                start_date: stage.start_date,
                end_date: stage.end_date,
                value_m3s: raw[t],
            })
            .collect();

        let system = ring_system(
            &stages,
            inflow_models,
            inflow_history,
            nonaligned_ring_season_map(),
            Vec::new(),
        );

        RingFixture {
            system,
            stages,
            hydro_id: RING_HYDRO_ID,
            raw,
            past_inflow_seed,
        }
    }

    /// `build_opening_tree_library` must size the downstream ring from the
    /// PAR model's own order, not the (possibly wider) declared lag depth: on
    /// this fixture the two diverge (`par.max_order() == 2` vs. the declared
    /// depth `3`), and only a `p >= 2` model with a mid-calendar-quarter
    /// transition can observe it.
    #[test]
    fn opening_tree_ring_order_is_par_order_not_declared_depth() {
        let fx = build_nonaligned_ring_fixture();
        let training_source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };

        let lib = build_opening_tree_library(&fx.system, &training_source, Some(3))
            .expect("build_opening_tree_library must succeed")
            .expect("HistoricalResiduals noise method must build a library");
        assert_eq!(
            lib.n_windows(),
            1,
            "exactly one historical window (2025) is discoverable"
        );

        let par = PrecomputedPar::build(
            fx.system.inflow_models(),
            &fx.stages,
            &[fx.hydro_id],
            fx.system.policy_graph().season_map.as_ref(),
        )
        .expect("oracle PrecomputedPar must build");
        assert_eq!(par.max_order(), 2);

        let derived = derive_downstream_par_order(
            &fx.stages,
            &par,
            fx.system.policy_graph().season_map.as_ref(),
        );
        assert_eq!(
            derived, 2,
            "the fixture crosses season_id >= 12 at stage 5; derive_downstream_par_order \
             must gate to par.max_order(), not the declared lag depth"
        );

        let season_map = nonaligned_ring_season_map();
        let transitions = precompute_stage_lag_transitions(&fx.stages, &season_map, derived);

        let q1_lag1 = ring_incoming_lags(&fx, &transitions, 3, 2, 6)[1];
        let dec_lag1 = ring_incoming_lags(&fx, &transitions, 2, 2, 6)[1];
        assert!(
            (q1_lag1 - dec_lag1).abs() > 1.0,
            "a width-3 ring keeps the Q1 average a width-2 ring evicts at the third \
             completed quarter, got q1_lag1={q1_lag1} vs dec_lag1={dec_lag1}"
        );

        let lags_p = ring_incoming_lags(&fx, &transitions, derived, 2, 6);
        let det_base = par.deterministic_base(6, 0);
        let psi = par.psi_slice(6, 0);
        let sigma = par.sigma(6, 0);
        let expected_eta = solve_par_noise(det_base, psi, &lags_p, sigma, fx.raw[6]);

        let eta_stage6 = lib.eta_slice(0, 6)[0];
        assert_eq!(
            eta_stage6, expected_eta,
            "build_opening_tree_library's eta at the quarterly transition must match the \
             PAR-order-sized ring oracle, not a ring sized by the declared lag depth"
        );
    }

    /// 12 monthly study stages (season 0=Jan .. 11=Dec, year 2026), one hydro,
    /// AR(0) — the declared depth alone drives widening, with no fitted lag
    /// dependence to entangle it, and a real monthly `SeasonMap` (needed for
    /// `derive_inflow_seeds`'s own season-based backward walk, which — unlike
    /// `discover_historical_windows`'s `month0()` fallback — requires a
    /// populated `SeasonMap.seasons` to resolve at all). History covers all 12
    /// study months plus `n_lag_months` pre-study lag months immediately
    /// preceding January 2026 (spanning back as many prior years as needed),
    /// each carrying a distinct `value_m3s = 1000.0 + k` (`k` = 1 for December
    /// 2025, the most recent, up to `n_lag_months` for the oldest) so a caller
    /// can identify exactly which calendar month landed in which lag slot.
    #[expect(
        clippy::too_many_lines,
        reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
    )]
    fn build_ar0_fixture_with_declared_lag_history(n_lag_months: u32) -> (System, ScenarioSource) {
        let hydro_id = EntityId(1);
        let bus_id = EntityId(2);

        let make_stage = |index: usize, month: u32| {
            let start = NaiveDate::from_ymd_opt(2026, month, 1).unwrap();
            let end = if month == 12 {
                NaiveDate::from_ymd_opt(2027, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(2026, month + 1, 1).unwrap()
            };
            Stage {
                index,
                id: index as i32,
                start_date: start,
                end_date: end,
                season_id: Some(index),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
                    duration_hours: 720.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: true,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::HistoricalResiduals,
                },
            }
        };
        let stages: Vec<Stage> = (1..=12_u32)
            .map(|m| make_stage((m - 1) as usize, m))
            .collect();

        let inflow_models: Vec<InflowModel> = (0..12_i32)
            .map(|stage_id| InflowModel {
                hydro_id,
                stage_id,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let mut inflow_history: Vec<InflowHistoryRow> = stages
            .iter()
            .map(|s| InflowHistoryRow {
                hydro_id,
                start_date: s.start_date,
                end_date: s.end_date,
                value_m3s: 100.0,
            })
            .collect();
        // Pre-study lag months immediately preceding January 2026 (month
        // index 0), walking back `n_lag_months` — possibly several prior
        // years — via Euclidean division so `k=12` lands on January 2025 and
        // `k=13` on December 2024, matching `nth_previous_occurrence`'s
        // monthly step-back semantics (`seeds.rs`).
        for k in 1..=n_lag_months {
            let month_index = -i32::try_from(k).unwrap();
            let year = 2026 + month_index.div_euclid(12);
            let month = u32::try_from(month_index.rem_euclid(12)).unwrap() + 1;
            let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap();
            let end = if month == 12 {
                NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(year, month + 1, 1).unwrap()
            };
            inflow_history.push(InflowHistoryRow {
                hydro_id,
                start_date: start,
                end_date: end,
                value_m3s: 1000.0 + f64::from(k),
            });
        }

        let load_models: Vec<LoadModel> = (0..12_i32)
            .map(|stage_id| LoadModel {
                bus_id,
                stage_id,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let bus = Bus {
            id: bus_id,
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: hydro_id,
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(bus_id);

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 12,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 12,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: Some(monthly_season_map()),
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .inflow_history(inflow_history)
            .bounds(bounds)
            .penalties(penalties)
            .policy_graph(policy_graph)
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: vec![],
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .build()
            .expect("AR(0) declared-lag-history fixture: valid system");

        let training_source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };
        (system, training_source)
    }

    /// Cross-source regression at the SAME declared depth (24, the worked
    /// acceptance example): `resolve_state_layout` (the dense-stride + mask
    /// source) still widens past the fixture's AR(0) order to exactly 24, but
    /// the historical library's width is the applied PAR's own
    /// order — the two sources address different facts and no longer agree.
    /// A forward twin built from the same applied PAR and the same depth-24
    /// seed (`resolve_inflow_seeds` at `state_layout.max_par_order`, the one
    /// seed owner at one depth) must be bit-identical to the tree: same
    /// window years, same `seed_digest()`, same eta.
    #[test]
    fn build_opening_tree_library_and_resolve_state_layout_agree_at_declared_depth() {
        let (system, training_source) = build_ar0_fixture_with_declared_lag_history(24);
        let declared_depth = Some(24);

        let study_stages: Vec<Stage> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
        let par_lp = PrecomputedPar::build(system.inflow_models(), &study_stages, &hydro_ids, None)
            .expect("PrecomputedPar must build");
        let calendar = crate::time_value::DeliveryCalendar::from_system(&system);
        let topology =
            crate::bucket_topology::build_transit_bucket_topology(&system, &calendar, false);
        let layout = crate::setup::resolve_state_layout(
            &system,
            &calendar,
            &par_lp,
            &topology,
            declared_depth,
        )
        .expect("resolve_state_layout must succeed with a declared depth");

        let lib = build_opening_tree_library(&system, &training_source, declared_depth)
            .expect("build_opening_tree_library must succeed with a declared depth")
            .expect("HistoricalResiduals noise method must build a library");

        assert_eq!(
            layout.state.max_par_order, 24,
            "resolve_state_layout must widen to the declared depth"
        );
        assert_eq!(
            lib.max_order(),
            par_lp.max_order(),
            "the historical library's width is the applied PAR's own order, never the \
             declared lag depth resolve_state_layout widens its dense stride to"
        );

        let fwd = resolve_inflow_seeds(&system, layout.state.max_par_order);
        let forward_twin = scenario_libraries::build_historical_inflow_library(
            &system,
            &par_lp,
            fwd.as_seed(24),
            None,
            1,
        )
        .expect("forward twin must build");

        assert_eq!(lib.n_windows(), forward_twin.n_windows());
        for w in 0..lib.n_windows() {
            assert_eq!(
                lib.window_year(w),
                forward_twin.window_year(w),
                "window {w}: the tree and the forward twin must draw the same year pool"
            );
        }
        assert_eq!(
            lib.seed_digest(),
            forward_twin.seed_digest(),
            "the tree and the forward twin share one seed owner at one depth"
        );
        for w in 0..lib.n_windows() {
            for t in 0..study_stages.len() {
                assert_eq!(
                    lib.eta_slice(w, t),
                    forward_twin.eta_slice(w, t),
                    "window {w} stage {t}: eta must be bit-identical between the tree and \
                     the forward twin"
                );
            }
        }
    }

    /// Given a declared lag depth (24) exceeding the fitted AR(0) order,
    /// `derive_inflow_seeds` must actually seed the DEEPEST declared lag slot
    /// from real history, not merely report a wider `max_order()` — the LP's
    /// stage-0 seed, `InitialConditions.inflow_seeds.as_seed(state.max_par_order)`,
    /// is what `run_eta_inversion`'s `max_order.min(seed.l_state)` copy loop
    /// reads. Complements
    /// `build_opening_tree_library_and_resolve_state_layout_agree_at_declared_depth`'s
    /// dimension-only check with the concrete seeded value.
    #[test]
    fn derive_inflow_seeds_populates_the_deepest_declared_lag_slot_from_real_history() {
        let (system, _training_source) = build_ar0_fixture_with_declared_lag_history(24);
        let declared_depth = widen_lag_state_depth(0, Some(24));
        assert_eq!(declared_depth, 24);

        let first_stage = system
            .stages()
            .iter()
            .find(|s| s.id >= 0)
            .expect("fixture has a study stage");
        let season_map = system
            .policy_graph()
            .season_map
            .as_ref()
            .expect("fixture carries a monthly season map");

        let seeds = derive_inflow_seeds(
            system.inflow_history(),
            &system.initial_conditions().recent_observations,
            system.hydros(),
            first_stage,
            season_map,
            declared_depth,
        );

        assert_eq!(
            seeds.lag_values.len(),
            system.hydros().len() * 24,
            "the seed buffer itself must be widened to the declared depth"
        );

        // Deepest slot: k=24 (index 23) is January 2024, this fixture's
        // uniquely-valued oldest pre-study month (1000.0 + 24 = 1024.0).
        let deepest = seeds.lag_values[23];
        assert!(
            (deepest - 1024.0).abs() < 1e-9,
            "the deepest declared lag slot (24 months back) must carry the real \
             January 2024 observation, not a truncated/default 0.0; got {deepest}"
        );

        // Negative control: an un-widened source (l_state left at the AR(0)
        // order, 0) cannot represent the deepest slot at all — the concrete
        // shape of "any source left un-widened".
        let unwidened_seeds = derive_inflow_seeds(
            system.inflow_history(),
            &system.initial_conditions().recent_observations,
            system.hydros(),
            first_stage,
            season_map,
            0,
        );
        assert!(
            unwidened_seeds.lag_values.is_empty(),
            "sanity: the un-widened AR(0) order provides no lag slots at all"
        );
    }

    // ── compute_external_scenario_counts: min + silent-drop deletion ──────────

    /// A `System` with one hydro and one std-bearing load bus over `stage_ids`,
    /// carrying `inflow_per_stage`/`load_per_stage` external realizations per stage
    /// (one row per entity per realization). Deliberately supports disagreeing
    /// per-class counts so a unit test can drive `compute_external_scenario_counts`
    /// directly — a disagreement P-B1 (cobre-io) would reject at load, unreachable
    /// through the public API but the exact input that separates take-first from an
    /// element-wise minimum.
    #[expect(
        clippy::too_many_lines,
        reason = "the fixture spells out one complete study inline so each assertion traces to a literal"
    )]
    fn external_count_system(
        stage_ids: &[i32],
        inflow_per_stage: usize,
        load_per_stage: usize,
    ) -> System {
        let hydro_id = EntityId(1);
        let bus_id = EntityId(2);
        let n_stages = stage_ids.len();

        let stages: Vec<Stage> = stage_ids
            .iter()
            .enumerate()
            .map(|(index, &id)| Stage {
                index,
                id,
                start_date: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                season_id: Some(0),
                blocks: vec![Block {
                    index: 0,
                    name: "S".to_string(),
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
            })
            .collect();

        let inflow_models: Vec<InflowModel> = stage_ids
            .iter()
            .map(|&stage_id| InflowModel {
                hydro_id,
                stage_id,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();
        let load_models: Vec<LoadModel> = stage_ids
            .iter()
            .map(|&stage_id| LoadModel {
                bus_id,
                stage_id,
                mean_mw: 100.0,
                std_mw: 10.0,
            })
            .collect();

        let mut external_scenarios = Vec::new();
        let mut external_load_scenarios = Vec::new();
        for &stage_id in stage_ids {
            for s in 0..inflow_per_stage {
                external_scenarios.push(ExternalScenarioRow {
                    stage_id,
                    scenario_id: i32::try_from(s).unwrap(),
                    hydro_id,
                    value_m3s: 1.0,
                });
            }
            for s in 0..load_per_stage {
                external_load_scenarios.push(ExternalLoadRow {
                    stage_id,
                    scenario_id: i32::try_from(s).unwrap(),
                    bus_id,
                    value_mw: 1.0,
                });
            }
        }

        let bus = Bus {
            id: bus_id,
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: hydro_id,
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(bus_id);

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    ..Default::default()
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let policy_graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .policy_graph(policy_graph)
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments: vec![],
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .external_scenarios(external_scenarios)
            .external_load_scenarios(external_load_scenarios)
            .build()
            .expect("external-count fixture: valid system")
    }

    fn external_source(inflow: SamplingScheme, load: SamplingScheme) -> ScenarioSource {
        ScenarioSource {
            inflow_scheme: inflow,
            load_scheme: load,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        }
    }

    /// The element-wise minimum is gone: with inflow (3) and load (2) external
    /// classes disagreeing on the per-stage raw count, the returned vector is the
    /// first present class's (inflow, 3) — never `min(3, 2) = 2`. This deck is
    /// P-B1-rejected at load; the direct call is the only way to observe the
    /// reconciliation was deleted rather than relocated.
    #[test]
    fn compute_external_scenario_counts_takes_first_never_min() {
        let system = external_count_system(&[0], 3, 2);
        let source = external_source(SamplingScheme::External, SamplingScheme::External);
        let counts = compute_external_scenario_counts(&system, &source);
        assert_eq!(
            counts,
            Some(vec![3]),
            "the first present class's raw count is authoritative, not the element-wise minimum"
        );
    }

    /// The `if stage_id as usize < n_stages` silent-drop is gone: a study numbered
    /// `[10, 11]` (never 0-based) counts every external row through the resolver.
    /// The pre-deletion code keyed on `stage_id as usize` and dropped all
    /// rows (`10 < 2` is false), returning `[0, 0]`; the resolver returns `[1, 1]`.
    #[test]
    fn compute_external_scenario_counts_resolves_non_zero_based_stage_ids() {
        let system = external_count_system(&[10, 11], 1, 0);
        let source = external_source(SamplingScheme::External, SamplingScheme::InSample);
        let counts = compute_external_scenario_counts(&system, &source);
        assert_eq!(
            counts,
            Some(vec![1, 1]),
            "rows must be counted through the resolver, not dropped by a stage_id-as-index guard"
        );
    }

    // ── load_user_opening_tree_inner: by declaration, not existence ───────────

    /// Parse a `Config` from an inline JSON string.
    fn config_from_json(json: &str) -> Config {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), json).expect("write config");
        cobre_io::parse_config(tmp.path()).expect("parse config")
    }

    const SAMPLED_TRAINING: &str = r#""selection": {"method": "sampled", "forward_passes": 10}, "stopping_rules": [{"type": "iteration_limit", "limit": 5}]"#;

    /// Write a 5-stage, 1-opening-per-stage, dim-1 tree at
    /// `case_dir/scenarios/noise_openings.parquet` — the shape the ring fixture's
    /// `branching_factor = 1` study stages declare.
    fn write_ring_noise_openings(case_dir: &Path) {
        let path = case_dir.join("scenarios").join("noise_openings.parquet");
        cobre_io::output::stochastic::write_noise_openings(&path, &ring_user_tree())
            .expect("write noise_openings");
    }

    fn ring_user_tree() -> OpeningTree {
        OpeningTree::from_parts(vec![0.1, 0.2, 0.3, 0.4, 0.5], vec![1; 5], 1)
    }

    /// A `noise_openings.parquet` physically present but with no `openings`
    /// declaration is ignored — consumption is by declaration, not existence.
    #[test]
    fn load_user_opening_tree_undeclared_file_is_ignored() {
        let fx = build_ring_fixture();
        let dir = tempfile::TempDir::new().expect("tempdir");
        write_ring_noise_openings(dir.path());

        let config = config_from_json(&format!(r#"{{"training": {{{SAMPLED_TRAINING}}}}}"#));
        let loaded = load_user_opening_tree_inner(
            dir.path(),
            &fx.system,
            &config,
            &ScenarioSource::default(),
        )
        .expect("an undeclared file must resolve to Ok(None)");
        assert!(
            loaded.is_none(),
            "a present-but-undeclared noise_openings.parquet must be ignored"
        );
    }

    /// A `{source: file}` declaration with a present, count-consistent file
    /// installs the user opening tree.
    #[test]
    fn load_user_opening_tree_declared_file_installs_tree() {
        let fx = build_ring_fixture();
        let dir = tempfile::TempDir::new().expect("tempdir");
        write_ring_noise_openings(dir.path());

        let config = config_from_json(&format!(
            r#"{{"training": {{{SAMPLED_TRAINING}, "scenario_source": {{"openings": {{"source": "file"}}}}}}}}"#
        ));
        let loaded = load_user_opening_tree_inner(
            dir.path(),
            &fx.system,
            &config,
            &ScenarioSource::default(),
        )
        .expect("a declared, present file must load")
        .expect("a declared file must install a tree");
        assert_eq!(loaded.n_stages(), 5);
        assert_eq!(loaded.dim(), 1);
        for s in 0..5 {
            assert_eq!(loaded.n_openings(s), 1, "stage {s} declares one opening");
        }
    }

    /// A `{source: file}` declaration with the file absent is a named error.
    #[test]
    fn load_user_opening_tree_declared_absent_file_errors() {
        let fx = build_ring_fixture();
        let dir = tempfile::TempDir::new().expect("tempdir");

        let config = config_from_json(&format!(
            r#"{{"training": {{{SAMPLED_TRAINING}, "scenario_source": {{"openings": {{"source": "file"}}}}}}}}"#
        ));
        let err = load_user_opening_tree_inner(
            dir.path(),
            &fx.system,
            &config,
            &ScenarioSource::default(),
        )
        .expect_err("a declared but absent file must error");
        match err {
            SddpError::Io(e) => assert!(
                e.to_string().contains("noise_openings.parquet"),
                "the error must name the missing file, got: {e}"
            ),
            other => panic!("expected SddpError::Io, got: {other:?}"),
        }
    }

    /// `build_opening_tree_library`'s rank-0 opening-tree build routes through
    /// `build_inflow_par`, which must surface `InvalidParParameters` itself —
    /// not some other error from downstream history discovery — when a PAR
    /// model carries AR order > 0 with zero standard deviation.
    #[test]
    fn build_opening_tree_library_rejects_invalid_par_parameters() {
        let stages = vec![ring_stage(
            0,
            0,
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
            0,
            31.0 * 24.0,
        )];
        let inflow_models = vec![
            InflowModel {
                hydro_id: RING_HYDRO_ID,
                stage_id: -1,
                mean_m3s: 100.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            InflowModel {
                hydro_id: RING_HYDRO_ID,
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 0.0,
                ar_coefficients: vec![0.6],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let system = ring_system(
            &stages,
            inflow_models,
            Vec::new(),
            monthly_season_map(),
            Vec::new(),
        );
        let training_source = ScenarioSource {
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: None,
            historical_years: None,
        };

        let err = build_opening_tree_library(&system, &training_source, None)
            .expect_err("a zero-std AR(1) model must be rejected");
        match err {
            SddpError::Stochastic(cobre_stochastic::StochasticError::InvalidParParameters {
                ..
            }) => {}
            other => panic!("expected InvalidParParameters, got: {other:?}"),
        }
    }

    /// The ring fixture with deterministic order-0 inflow (`std_m3s = 0`), so the
    /// stage-0 observation 80 against the mean 100 fails the historical library's
    /// V2.3 check.
    fn ring_with_failing_historical_library() -> System {
        let fx = build_ring_fixture();
        let models = fx
            .system
            .inflow_models()
            .iter()
            .cloned()
            .map(|m| InflowModel {
                std_m3s: 0.0,
                ar_coefficients: vec![],
                ..m
            })
            .collect();
        ring_system(
            &fx.stages,
            models,
            fx.system.inflow_history().to_vec(),
            ring_season_map(),
            Vec::new(),
        )
    }

    fn ring_without_inflow_history() -> System {
        let fx = build_ring_fixture();
        ring_system(
            &fx.stages,
            fx.system.inflow_models().to_vec(),
            Vec::new(),
            ring_season_map(),
            Vec::new(),
        )
    }

    fn assert_supplied_tree_reaches_context(system: &System) {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let ctx = build_stochastic_context_for_study(
            system,
            tmp.path(),
            42,
            &ScenarioSource::default(),
            None,
            Some(ring_user_tree()),
            None,
        )
        .expect("a supplied opening tree must not need the historical library");
        assert_eq!(
            ctx.provenance().opening_tree,
            ComponentProvenance::UserSupplied
        );
        assert_eq!(ctx.opening_tree().data(), [0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn supplied_opening_tree_ignores_a_historical_library_that_fails_validation() {
        assert_supplied_tree_reaches_context(&ring_with_failing_historical_library());
    }

    #[test]
    fn supplied_opening_tree_needs_no_inflow_history() {
        assert_supplied_tree_reaches_context(&ring_without_inflow_history());
    }

    #[test]
    fn generated_opening_tree_still_refuses_a_historical_library_that_fails_validation() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let result = build_stochastic_context_for_study(
            &ring_with_failing_historical_library(),
            tmp.path(),
            42,
            &ScenarioSource::default(),
            None,
            None,
            None,
        );
        match result {
            Err(SddpError::Stochastic(StochasticError::InsufficientData { context })) => {
                assert!(context.starts_with("V2.3"), "got: {context}");
            }
            Err(other) => panic!("expected a V2.3 refusal, got: {other:?}"),
            Ok(_) => panic!("a generated tree over a failing library must be refused"),
        }
    }
}
