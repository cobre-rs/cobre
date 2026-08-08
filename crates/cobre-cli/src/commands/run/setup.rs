//! Case-load, communicator setup, broadcast, and pre-training phases for `cobre run`.
//!
//! Rank 0 loads from disk; `System` and config are broadcast to all ranks, which
//! then build `StudySetup` from the shared data. Hydro model preprocessing is the
//! exception: it is not broadcast. Rank 0 reuses the artifacts
//! `load_case_with_artifacts` already parsed (`prepare_hydro_models_from_artifacts`);
//! non-root ranks independently re-read the case directory from disk
//! (`prepare_hydro_models`), relying on a shared filesystem instead.

use std::path::{Path, PathBuf};

use console::Term;

use cobre_comm::{
    AffinityReport, Communicator, TopologyProvider, WorkerAffinity, create_communicator,
};
use cobre_core::EntityId;
use cobre_core::ScalarParameter;
use cobre_core::ScenarioSource;
use cobre_core::System;
use cobre_core::temporal::SeasonCycleType::Monthly;
use cobre_core::temporal::SeasonMap;
use cobre_io::BroadcastScalarParameter;
use cobre_io::Config;
use cobre_io::PolicyMode;
use cobre_io::RankAffinity;
use cobre_io::SetupTimings;
use cobre_io::load_case_with_artifacts;
use cobre_io::parse_config;
use cobre_io::write_hydro_model_summary;
use cobre_io::write_provenance_report;
use cobre_io::write_scaling_report;
use cobre_sddp::EstimationPath;
use cobre_sddp::HydroFitTimings;
use cobre_sddp::build_provenance_report;
use cobre_sddp::hydro_models::prepare_hydro_models_from_artifacts;
use cobre_sddp::orchestration::export_stochastic_artifacts;
use cobre_sddp::reconcile_global_ok;
use cobre_sddp::{
    EstimationReport, PrepareHydroModelsResult, PrepareStochasticResult, StudySetup,
    build_hydro_model_summary, prepare_hydro_models, prepare_stochastic,
    setup::{
        ConstructionConfig, build_ncs_factor_entries, load_load_factors_for_stochastic,
        study_stage_noise_group_ids,
    },
};
use cobre_solver::active_solver_name;
use cobre_solver::active_solver_version;
use cobre_stochastic::ClassSchemes;
use cobre_stochastic::DerivedInflowSeeds;
use cobre_stochastic::HistoricalScenarioLibrary;
use cobre_stochastic::PrecomputedPar;
use cobre_stochastic::derive_inflow_seeds;
use cobre_stochastic::discover_historical_windows;
use cobre_stochastic::normal::precompute::BlockFactorPair;
use cobre_stochastic::normal::precompute::EntityFactorEntry;
use cobre_stochastic::par::lag_transition::derive_downstream_par_order;
use cobre_stochastic::par::lag_transition::precompute_stage_lag_transitions;
use cobre_stochastic::standardize_historical_windows;
use cobre_stochastic::{
    OpeningTreeInputs, build_stochastic_context, context::OpeningTree,
    provenance::ComponentProvenance,
};

use crate::error::CliError;

use crate::commands::broadcast::{
    BroadcastConfig, BroadcastOpeningTree, broadcast_value, stopping_rules_from_broadcast,
};

use super::{RunArgs, RunContext};
use crate::banner::print_banner;
use crate::progress::RenderMode;
use crate::progress::resolve_term_width;
use crate::summary::print_execution_topology;
use crate::summary::print_hydro_model_summary;
use crate::summary::print_provenance_summary;
use crate::summary::print_setup_summary;

pub(super) fn resolve_thread_count(cli_threads: Option<u32>) -> usize {
    match cli_threads {
        Some(n) => n as usize,
        None => 1,
    }
}

fn rank_affinity_from_report(rank: usize, report: &AffinityReport) -> RankAffinity {
    RankAffinity {
        rank: u32::try_from(rank).unwrap_or(u32::MAX),
        policy: report.policy.as_str().to_string(),
        online_processing_units: report
            .online_processing_units
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        visible_processing_units: report
            .visible_processing_units
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        physical_cores: report
            .physical_cores
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        numa_nodes: report
            .numa_nodes
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        visible_cpus: report
            .visible_cpus
            .iter()
            .map(|&cpu| u32::try_from(cpu).unwrap_or(u32::MAX))
            .collect(),
        memory_policy: report.memory_policy.clone(),
        memory_policy_nodes: report
            .memory_policy_nodes
            .iter()
            .map(|&node| u32::try_from(node).unwrap_or(u32::MAX))
            .collect(),
        allowed_memory_nodes: report
            .allowed_memory_nodes
            .iter()
            .map(|&node| u32::try_from(node).unwrap_or(u32::MAX))
            .collect(),
        memory_discovery_error: report.memory_discovery_error.clone(),
        worker_cpus: report
            .worker_cpus
            .iter()
            .map(|&cpu| u32::try_from(cpu).unwrap_or(u32::MAX))
            .collect(),
        discovery_error: report.discovery_error.clone(),
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RankAffinityWire {
    rank: u32,
    policy: String,
    online_processing_units: Option<u32>,
    visible_processing_units: Option<u32>,
    physical_cores: Option<u32>,
    numa_nodes: Option<u32>,
    visible_cpus: Vec<u32>,
    memory_policy: Option<String>,
    memory_policy_nodes: Vec<u32>,
    allowed_memory_nodes: Vec<u32>,
    memory_discovery_error: Option<String>,
    worker_cpus: Vec<u32>,
    discovery_error: Option<String>,
}

impl From<&RankAffinity> for RankAffinityWire {
    fn from(value: &RankAffinity) -> Self {
        Self {
            rank: value.rank,
            policy: value.policy.clone(),
            online_processing_units: value.online_processing_units,
            visible_processing_units: value.visible_processing_units,
            physical_cores: value.physical_cores,
            numa_nodes: value.numa_nodes,
            visible_cpus: value.visible_cpus.clone(),
            memory_policy: value.memory_policy.clone(),
            memory_policy_nodes: value.memory_policy_nodes.clone(),
            allowed_memory_nodes: value.allowed_memory_nodes.clone(),
            memory_discovery_error: value.memory_discovery_error.clone(),
            worker_cpus: value.worker_cpus.clone(),
            discovery_error: value.discovery_error.clone(),
        }
    }
}

impl From<RankAffinityWire> for RankAffinity {
    fn from(value: RankAffinityWire) -> Self {
        Self {
            rank: value.rank,
            policy: value.policy,
            online_processing_units: value.online_processing_units,
            visible_processing_units: value.visible_processing_units,
            physical_cores: value.physical_cores,
            numa_nodes: value.numa_nodes,
            visible_cpus: value.visible_cpus,
            memory_policy: value.memory_policy,
            memory_policy_nodes: value.memory_policy_nodes,
            allowed_memory_nodes: value.allowed_memory_nodes,
            memory_discovery_error: value.memory_discovery_error,
            worker_cpus: value.worker_cpus,
            discovery_error: value.discovery_error,
        }
    }
}

fn gather_rank_affinity(
    comm: &impl Communicator,
    local: &RankAffinity,
) -> Result<Vec<RankAffinity>, CliError> {
    let encoded = postcard::to_allocvec(&RankAffinityWire::from(local)).map_err(|error| {
        CliError::Internal {
            message: format!("failed to encode rank CPU affinity metadata: {error}"),
        }
    })?;
    let world_size = comm.size();
    let scalar_counts = vec![1usize; world_size];
    let scalar_displacements = (0..world_size).collect::<Vec<_>>();
    let mut encoded_lengths = vec![0u64; world_size];
    comm.allgatherv(
        &[u64::try_from(encoded.len()).unwrap_or(u64::MAX)],
        &mut encoded_lengths,
        &scalar_counts,
        &scalar_displacements,
    )
    .map_err(|error| CliError::Internal {
        message: format!("failed to gather rank CPU affinity lengths: {error}"),
    })?;

    let counts = encoded_lengths
        .iter()
        .map(|&length| {
            usize::try_from(length).map_err(|_| CliError::Internal {
                message: format!("rank CPU affinity payload length {length} exceeds usize"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut displacements = Vec::with_capacity(world_size);
    let mut total = 0usize;
    for &count in &counts {
        displacements.push(total);
        total = total.checked_add(count).ok_or_else(|| CliError::Internal {
            message: "rank CPU affinity payload size overflow".to_string(),
        })?;
    }
    let mut gathered = vec![0u8; total];
    comm.allgatherv(&encoded, &mut gathered, &counts, &displacements)
        .map_err(|error| CliError::Internal {
            message: format!("failed to gather rank CPU affinity metadata: {error}"),
        })?;

    let mut result: Vec<RankAffinity> = Vec::with_capacity(world_size);
    for (rank, (&offset, &count)) in displacements.iter().zip(&counts).enumerate() {
        let end = offset + count;
        let entry: RankAffinityWire =
            postcard::from_bytes(&gathered[offset..end]).map_err(|error| CliError::Internal {
                message: format!("failed to decode CPU affinity metadata for rank {rank}: {error}"),
            })?;
        result.push(entry.into());
    }
    result.sort_unstable_by_key(|entry| entry.rank);
    Ok(result)
}

/// Values loaded on rank 0 by [`load_case_and_config`]. The trailing
/// [`cobre_io::SetupTimings`] leaves `broadcast_seconds` zero;
/// [`broadcast_and_build_setup`] fills it after the broadcast region runs.
type LoadedCase = (
    PrepareStochasticResult,
    PrepareHydroModelsResult,
    BroadcastConfig,
    Config,
    Vec<ScalarParameter>,
    SetupTimings,
);

/// Load case and config on rank 0, capturing errors for MPI collective participation.
fn load_case_and_config(
    args: &RunArgs,
    quiet: bool,
    stderr: &Term,
) -> Result<LoadedCase, CliError> {
    if !args.case_dir.exists() {
        return Err(CliError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "case directory does not exist",
            ),
            context: args.case_dir.display().to_string(),
        });
    }
    if !quiet {
        let _ = stderr.write_line(&format!("Loading case: {}", args.case_dir.display()));
    }
    // Single load: downstream consumers reuse the artifacts returned here instead
    // of re-reading the same files from disk.
    let mut timings = SetupTimings::default();

    let load_start = std::time::Instant::now();
    let cobre_io::LoadedCase { system, artifacts } = load_case_with_artifacts(&args.case_dir)?;
    let config_path = args.case_dir.join("config.json");
    let config = parse_config(&config_path)?;
    timings.load_seconds = load_start.elapsed().as_secs_f64();

    let bcast = BroadcastConfig::from_config(&config)?;
    let seed = bcast.seed;

    let stochastic_start = std::time::Instant::now();
    let prepared = prepare_stochastic(
        system,
        &args.case_dir,
        &config,
        seed,
        &bcast.training_source,
    )
    .map_err(CliError::from)?;
    timings.stochastic_fit_seconds = stochastic_start.elapsed().as_secs_f64();

    let mut hydro_timings = HydroFitTimings::default();
    let hydro_models = prepare_hydro_models_from_artifacts(
        &prepared.system,
        &artifacts,
        config.exports.fpha_deviation_points,
        Some(&mut hydro_timings),
    )
    .map_err(CliError::from)?;
    timings.production_fit_seconds = hydro_timings.production_fit_seconds;
    timings.evaporation_fit_seconds = hydro_timings.evaporation_fit_seconds;

    Ok((
        prepared,
        hydro_models,
        bcast,
        config,
        artifacts.scalar_parameters,
        timings,
    ))
}

/// Output of [`broadcast_and_build_setup`]. `root_*` fields are `Some` only on
/// rank 0 (used for output writing); the flag fields are broadcast from rank 0.
pub(super) struct LoadBroadcastResult {
    pub(super) system: System,
    pub(super) setup: StudySetup,
    pub(super) root_config: Option<Config>,
    pub(super) root_estimation_report: Option<EstimationReport>,
    pub(super) root_estimation_path: Option<EstimationPath>,
    pub(super) training_enabled: bool,
    pub(super) policy_mode: PolicyMode,
    /// `None` on non-root ranks, which reconstruct setup independently and never
    /// write metadata.
    pub(super) setup_timings: Option<SetupTimings>,
}

/// Set up the communicator, terminal, rayon pool, and resolve the output directory.
pub(super) fn setup_communicator(
    args: &RunArgs,
) -> Result<RunContext<impl Communicator>, CliError> {
    let comm = create_communicator(args.comm_backend.into())?;
    let is_root = comm.rank() == 0;
    let quiet = args.quiet || !is_root;

    // mpiexec pipes rank 0's stderr without a PTY, so force colors on; console
    // would otherwise disable them on the non-TTY pipe.
    let mpi_active = comm.size() > 1;
    if mpi_active && is_root && !args.quiet {
        console::set_colors_enabled_stderr(true);
    }

    let stderr = Term::stderr();

    // Gather topology while the concrete backend type is still in scope.
    let topology = comm.topology().clone();

    let configured_threads = resolve_thread_count(args.threads);
    let affinity =
        WorkerAffinity::prepare(args.cpu_bind.into(), configured_threads).map_err(|error| {
            CliError::Internal {
                message: format!("CPU affinity setup failed: {error}"),
            }
        })?;
    let worker_affinity = affinity.clone();
    let actual_threads = match rayon::ThreadPoolBuilder::new()
        .num_threads(configured_threads)
        .start_handler(move |worker_index| worker_affinity.bind_worker(worker_index))
        .build_global()
    {
        Ok(()) => {
            // Start every worker before `verify`; start hooks cannot return errors.
            rayon::broadcast(|_| {});
            affinity.verify().map_err(|error| CliError::Internal {
                message: format!("CPU affinity setup failed: {error}"),
            })?;
            configured_threads
        }
        Err(err) if affinity.report().policy.binds_workers() => {
            return Err(CliError::Internal {
                message: format!(
                    "cannot apply requested CPU affinity because the global rayon pool already exists: {err}"
                ),
            });
        }
        Err(err) => {
            let actual = rayon::current_num_threads();
            tracing::warn!(
                configured = configured_threads,
                actual,
                %err,
                "rayon global thread pool init failed; using existing pool",
            );
            actual
        }
    };
    if actual_threads == 0 {
        return Err(CliError::Internal {
            message: "rayon reported zero active threads — unexpected state".to_string(),
        });
    }
    if affinity.report().is_oversubscribed(actual_threads) {
        tracing::warn!(
            workers = actual_threads,
            visible_cpus = affinity.report().visible_processing_units,
            "rayon workers exceed the visible CPU allocation",
        );
    }
    if affinity.report().is_cpuset_restricted() && topology.slurm.is_none() && comm.size() == 1 {
        tracing::warn!(
            visible_cpus = affinity.report().visible_processing_units,
            online_cpus = affinity.report().online_processing_units,
            "process inherited a restricted CPU set; workers remain inside it",
        );
    }

    let local_rank_affinity = rank_affinity_from_report(comm.rank(), affinity.report());
    let rank_affinity = gather_rank_affinity(&comm, &local_rank_affinity)?;

    let solver_version = active_solver_version();

    if !quiet {
        print_banner(&stderr);
        print_execution_topology(
            &stderr,
            &topology,
            actual_threads,
            active_solver_name(),
            Some(&solver_version),
            Some(&local_rank_affinity),
        );
    }

    let output_dir: PathBuf = args
        .output
        .clone()
        .unwrap_or_else(|| args.case_dir.join("output"));
    let term_width = resolve_term_width();
    let render_mode = RenderMode::auto();

    Ok(RunContext {
        comm,
        is_root,
        quiet,
        n_threads: actual_threads,
        rank_affinity,
        output_dir,
        term_width,
        stderr,
        render_mode,
        topology,
        solver_version,
    })
}

/// Load the case on rank 0, broadcast system/config/tree, and build `StudySetup` on all ranks.
// Rationale: one MPI coordination seam — splitting it would scatter the ordered
// broadcast-receive-build sequence across callsites.
#[allow(clippy::too_many_lines)]
pub(super) fn broadcast_and_build_setup(
    ctx: &RunContext<impl Communicator>,
    args: &RunArgs,
) -> Result<LoadBroadcastResult, CliError> {
    // Kept out of the broadcast tuple: timings are never broadcast — only rank 0
    // writes metadata.
    let mut root_setup_timings: Option<SetupTimings> = None;
    let (
        raw_system,
        raw_bcast_config,
        root_config,
        root_stochastic,
        root_estimation_report,
        root_estimation_path,
        raw_bcast_tree,
        root_hydro_models,
        raw_scalar_parameters,
        load_err,
    ) = if ctx.is_root {
        match load_case_and_config(args, ctx.quiet, &ctx.stderr) {
            Ok((prepared, hydro_models, bcast, config, scalar_parameters, timings)) => {
                root_setup_timings = Some(timings);
                let bcast_tree = if prepared.stochastic.provenance().opening_tree
                    == ComponentProvenance::UserSupplied
                {
                    let t = prepared.stochastic.opening_tree();
                    Some(BroadcastOpeningTree {
                        data: t.data().to_vec(),
                        openings_per_stage: t.openings_per_stage_slice().to_vec(),
                        dim: t.dim(),
                    })
                } else {
                    None
                };
                let PrepareStochasticResult {
                    system,
                    stochastic,
                    estimation_report,
                    estimation_path,
                } = prepared;
                let bcast_params: Vec<BroadcastScalarParameter> = scalar_parameters
                    .iter()
                    .map(BroadcastScalarParameter::from)
                    .collect();
                (
                    Some(system),
                    Some(bcast),
                    Some(config),
                    Some(stochastic),
                    estimation_report,
                    Some(estimation_path),
                    Some(bcast_tree),
                    Some(hydro_models),
                    Some(bcast_params),
                    None,
                )
            }
            Err(e) => (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(e),
            ),
        }
    } else {
        (None, None, None, None, None, None, None, None, None, None)
    };
    let broadcast_start = std::time::Instant::now();
    let system_result = broadcast_value(raw_system, &ctx.comm);
    let bcast_config_result = broadcast_value(raw_bcast_config, &ctx.comm);
    let scalar_parameters_result = broadcast_value(raw_scalar_parameters, &ctx.comm);

    let tree_result = broadcast_value(raw_bcast_tree, &ctx.comm);

    if let Some(e) = load_err {
        return Err(e);
    }
    let system = system_result?;
    let mut bcast_config = bcast_config_result?;

    let seed = bcast_config.seed;

    // Non-root reconstruction must reproduce rank 0's factor entries (load, NCS)
    // and forward seed exactly, for MPI reproducibility.
    let stochastic = if ctx.is_root {
        drop(tree_result);
        root_stochastic.ok_or_else(|| CliError::Internal {
            message: "stochastic context missing on rank 0 after successful load".to_string(),
        })?
    } else {
        let user_tree: Option<OpeningTree> =
            tree_result?.map(|bt| OpeningTree::from_parts(bt.data, bt.openings_per_stage, bt.dim));
        reconstruct_stochastic_context_non_root(
            &system,
            &bcast_config,
            user_tree,
            seed,
            &args.case_dir,
        )?
    };

    let hydro_models = if ctx.is_root {
        root_hydro_models.ok_or_else(|| CliError::Internal {
            message: "hydro models missing on rank 0 after successful load".to_string(),
        })?
    } else {
        // Deviation-points opt-in is `false` regardless of the run-level export
        // flag: non-root ranks never reach the write site (only rank 0 writes).
        prepare_hydro_models(&system, &args.case_dir, false).map_err(|e| CliError::Internal {
            message: format!("hydro model preprocessing error on non-root rank: {e}"),
        })?
    };

    let training_enabled = bcast_config.training_enabled;
    let policy_mode = bcast_config.policy_mode;
    let scalar_parameters: Vec<ScalarParameter> = scalar_parameters_result?
        .into_iter()
        .map(ScalarParameter::from)
        .collect();
    let setup = build_study_setup(
        &system,
        &mut bcast_config,
        stochastic,
        hydro_models,
        scalar_parameters,
    )?;

    if let Some(timings) = root_setup_timings.as_mut() {
        timings.broadcast_seconds = broadcast_start.elapsed().as_secs_f64();
    }

    Ok(LoadBroadcastResult {
        system,
        setup,
        root_config,
        root_estimation_report,
        root_estimation_path,
        training_enabled,
        policy_mode,
        setup_timings: root_setup_timings,
    })
}

/// Reconstruct the non-root `StochasticContext` from broadcast parameters.
fn reconstruct_stochastic_context_non_root(
    system: &System,
    bcast_config: &BroadcastConfig,
    user_tree: Option<OpeningTree>,
    seed: u64,
    case_dir: &Path,
) -> Result<cobre_stochastic::StochasticContext, CliError> {
    let training_src = &bcast_config.training_source;
    let forward_seed = training_src.seed.map(i64::unsigned_abs);

    let load_factor_entries =
        load_load_factors_for_stochastic(case_dir).map_err(|e| CliError::Internal {
            message: format!("load factor error on non-root rank: {e}"),
        })?;
    let load_block_pairs: Vec<Vec<BlockFactorPair>> = load_factor_entries
        .iter()
        .map(|e| {
            e.block_factors
                .iter()
                .map(|bf| (bf.block_id, bf.factor))
                .collect()
        })
        .collect();
    let load_entity_factors: Vec<EntityFactorEntry<'_>> = load_factor_entries
        .iter()
        .zip(load_block_pairs.iter())
        .map(|(e, pairs)| (e.bus_id, e.stage_id, pairs.as_slice()))
        .collect();

    let ncs_raw = build_ncs_factor_entries(system);
    let ncs_entity_factors: Vec<EntityFactorEntry<'_>> = ncs_raw
        .iter()
        .map(|(ncs_id, stage_id, pairs)| (*ncs_id, *stage_id, pairs.as_slice()))
        .collect();

    let opening_tree_library = rebuild_historical_library_non_root(system, training_src)?;

    build_stochastic_context(
        system,
        seed,
        forward_seed,
        &load_entity_factors,
        &ncs_entity_factors,
        OpeningTreeInputs {
            user_tree,
            historical_library: opening_tree_library.as_ref(),
            external_scenario_counts: None,
            noise_group_ids: Some(study_stage_noise_group_ids(system)),
        },
        ClassSchemes {
            inflow: Some(training_src.inflow_scheme),
            load: Some(training_src.load_scheme),
            ncs: Some(training_src.ncs_scheme),
        },
    )
    .map_err(|e| CliError::Internal {
        message: format!("stochastic context error: {e}"),
    })
}

/// Build the non-root `HistoricalScenarioLibrary` from broadcast parameters.
fn rebuild_historical_library_non_root(
    system: &System,
    training_src: &ScenarioSource,
) -> Result<Option<HistoricalScenarioLibrary>, CliError> {
    use cobre_core::temporal::NoiseMethod;

    // Mirrors `prepare_stochastic` on rank 0: build the library when any stage
    // uses HistoricalResiduals.
    let needs_historical_tree = system
        .stages()
        .iter()
        .any(|s| s.id >= 0 && s.scenario_config.noise_method == NoiseMethod::HistoricalResiduals);

    if needs_historical_tree {
        let study_stages: Vec<_> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
        let cycle_len = system
            .policy_graph()
            .season_map
            .as_ref()
            .map(|sm| sm.seasons.len());
        let par =
            PrecomputedPar::build(system.inflow_models(), &study_stages, &hydro_ids, cycle_len)
                .map_err(|e| CliError::Internal {
                    message: format!("PAR build error on non-root rank: {e}"),
                })?;
        let max_order = par.max_order();
        let user_pool = training_src.historical_years.as_ref();
        let window_years = discover_historical_windows(
            system.inflow_history(),
            &hydro_ids,
            &study_stages,
            max_order,
            user_pool,
            system.policy_graph().season_map.as_ref(),
            1,
        )
        .map_err(|e| CliError::Internal {
            message: format!("historical window discovery error on non-root rank: {e}"),
        })?;
        let mut lib = HistoricalScenarioLibrary::new(
            window_years.len(),
            study_stages.len(),
            hydro_ids.len(),
            max_order,
            window_years.clone(),
        );
        // The derived lag seed seeds the η-inversion chain from the same x₀ as
        // the forward pass — rank-invariant because it is a pure function of
        // the broadcast `system`, derived with the identical arguments rank 0
        // uses. Compute stage_lag_transitions via the production helper, not
        // the in-function uniform-monthly fallback, which silently misroutes
        // non-monthly study grids.
        let noop_season_map;
        let season_map_for_transitions: &SeasonMap =
            if let Some(sm) = system.policy_graph().season_map.as_ref() {
                sm
            } else {
                noop_season_map = SeasonMap {
                    cycle_type: Monthly,
                    seasons: Vec::new(),
                };
                &noop_season_map
            };
        let downstream_par_order = derive_downstream_par_order(
            &study_stages,
            max_order,
            system.policy_graph().season_map.as_ref(),
        );
        let stage_lag_transitions = precompute_stage_lag_transitions(
            &study_stages,
            season_map_for_transitions,
            downstream_par_order,
        );
        let derived_inflow_seeds = match study_stages.first() {
            None => DerivedInflowSeeds::zero(hydro_ids.len(), max_order),
            Some(first_stage) => derive_inflow_seeds(
                system.inflow_history(),
                &system.initial_conditions().recent_observations,
                system.hydros(),
                first_stage,
                season_map_for_transitions,
                max_order,
            ),
        };
        standardize_historical_windows(
            &mut lib,
            system.inflow_history(),
            &hydro_ids,
            &study_stages,
            &par,
            &window_years,
            system.policy_graph().season_map.as_ref(),
            &derived_inflow_seeds.lag_values,
            max_order,
            &derived_inflow_seeds.accum,
            &derived_inflow_seeds.weight,
            &stage_lag_transitions,
            downstream_par_order,
        );
        Ok(Some(lib))
    } else {
        Ok(None)
    }
}

/// Construct `StudySetup` on all ranks from broadcast parameters.
// Rationale: by-value because `StudySetup` construction moves these in; a
// reference would force an internal clone.
#[allow(clippy::needless_pass_by_value)]
fn build_study_setup(
    system: &System,
    bcast_config: &mut BroadcastConfig,
    stochastic: cobre_stochastic::StochasticContext,
    hydro_models: PrepareHydroModelsResult,
    scalar_parameters: Vec<ScalarParameter>,
) -> Result<StudySetup, CliError> {
    let stopping_rule_set = stopping_rules_from_broadcast(bcast_config);
    let cut_selection = bcast_config.cut_selection.take();
    let training_solver_backward = bcast_config.training_solver_backward.take();
    let training_solver_forward = bcast_config.training_solver_forward.take();
    let simulation_solver = bcast_config.simulation_solver.take();
    let config = ConstructionConfig {
        seed: bcast_config.seed,
        forward_passes: bcast_config.forward_passes,
        stopping_rule_set,
        n_scenarios: bcast_config.n_scenarios,
        io_channel_capacity: usize::try_from(bcast_config.io_channel_capacity).unwrap_or(64),
        policy_path: bcast_config.policy_path.clone(),
        inflow_method: bcast_config.inflow_method,
        cut_selection,
        cut_activity_tolerance: bcast_config.cut_activity_tolerance,
        budget: bcast_config.budget,
        export_states: bcast_config.export_states,
        scalar_parameters,
        training_solver_backward,
        training_solver_forward,
        simulation_solver,
        backward_scheduler: bcast_config.backward_scheduler.into(),
        cost_scale_factor: bcast_config.cost_scale_factor,
    };
    StudySetup::from_broadcast_params(
        system,
        stochastic,
        config,
        hydro_models,
        &bcast_config.training_source,
        &bcast_config.simulation_source,
    )
    .map_err(CliError::from)
}

pub(super) fn run_pre_training(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&Config>,
    root_estimation_report: Option<&EstimationReport>,
    root_estimation_path: Option<EstimationPath>,
    setup_timings: Option<&SetupTimings>,
) -> Result<(), CliError> {
    // Renders before the Hydro models block so the setup timings sit with the
    // other setup summaries.
    if ctx.is_root
        && !ctx.quiet
        && let Some(timings) = setup_timings
    {
        print_setup_summary(&ctx.stderr, timings);
    }

    let export_result: Result<(), CliError> = if ctx.is_root {
        run_root_exports(
            ctx,
            system,
            setup,
            root_config,
            root_estimation_report,
            root_estimation_path,
        )
    } else {
        Ok(())
    };

    // Reconcile the rank-0-only export writes BEFORE the barrier: a rank-0 I/O
    // failure (disk full, permissions) would otherwise strand every peer at the
    // barrier while rank 0 returned early.
    let mut reconcile_scratch = [0_i32];
    let global_ok = reconcile_global_ok(export_result.is_ok(), &ctx.comm, &mut reconcile_scratch)
        .map_err(|e| CliError::Internal {
        message: format!("pre-training export reconcile error: {e}"),
    })?;
    export_result?;
    if !global_ok {
        return Err(CliError::Internal {
            message: "rank 0 pre-training export failed; failing on every rank in lockstep"
                .to_string(),
        });
    }

    ctx.comm.barrier().map_err(|e| CliError::Internal {
        message: format!("post-export barrier error: {e}"),
    })?;

    Ok(())
}

/// Rank-0 pre-training export writes: hydro model summary, provenance report,
/// stochastic artifacts (non-fatal), and scaling report. Called only on rank 0;
/// the returned `Result` is reconciled across ranks before the post-export barrier.
fn run_root_exports(
    ctx: &RunContext<impl Communicator>,
    system: &System,
    setup: &StudySetup,
    root_config: Option<&Config>,
    root_estimation_report: Option<&EstimationReport>,
    root_estimation_path: Option<EstimationPath>,
) -> Result<(), CliError> {
    // Built regardless of `quiet`: it also feeds the persisted sidecar consumed
    // by `cobre summary`, not just the optional print.
    let hydro_summary = build_hydro_model_summary(&setup.hydro_models, system);
    if !ctx.quiet {
        print_hydro_model_summary(&ctx.stderr, &hydro_summary);
    }
    let hydro_models_path = ctx.output_dir.join("training/hydro_models.json");
    write_hydro_model_summary(&hydro_models_path, &hydro_summary).map_err(|e| {
        CliError::Internal {
            message: format!("failed to write hydro model summary: {e}"),
        }
    })?;

    if let Some(path) = root_estimation_path {
        let mut provenance = build_provenance_report(
            path,
            root_estimation_report,
            setup.stochastic.provenance(),
            system.hydros().len(),
            &setup.hydro_models.provenance,
        );
        // Fingerprint the derived lag seed (training-side library only) so
        // stale-library detection can compare against a fresh digest on later runs.
        provenance.inflow.historical_library_seed_digest = setup
            .scenario_libraries
            .training
            .historical
            .as_ref()
            .map(HistoricalScenarioLibrary::seed_digest);
        if !ctx.quiet {
            print_provenance_summary(&ctx.stderr, &provenance);
        }
        let provenance_path = ctx.output_dir.join("training/model_provenance.json");
        write_provenance_report(&provenance_path, &provenance).map_err(|e| CliError::Internal {
            message: format!("failed to write provenance report: {e}"),
        })?;
    }

    if root_config.is_some_and(|c| c.exports.stochastic) {
        if !ctx.quiet {
            let _ = ctx.stderr.write_line("Exporting stochastic artifacts...");
        }
        let stderr = &ctx.stderr;
        let quiet = ctx.quiet;
        let mut on_warning = |msg: &str| {
            if !quiet {
                let _ = stderr.write_line(&format!("warning: stochastic export failed ({msg})"));
            }
        };
        export_stochastic_artifacts(
            &ctx.output_dir,
            &setup.stochastic,
            system,
            root_estimation_report,
            &mut on_warning,
        );
    }

    let scaling_path = ctx.output_dir.join("training/scaling_report.json");
    write_scaling_report(&scaling_path, &setup.stage_data.scaling_report).map_err(|e| {
        CliError::Internal {
            message: format!("failed to write scaling report: {e}"),
        }
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use console::Term;

    use super::{
        gather_rank_affinity, load_case_and_config, rank_affinity_from_report,
        reconstruct_stochastic_context_non_root, study_stage_noise_group_ids,
    };
    use crate::commands::run::{CommBackendArg, CpuBindArg, RunArgs};
    use cobre_comm::{AffinityPolicy, AffinityReport, LocalBackend};

    #[test]
    fn local_rank_affinity_metadata_round_trips_through_collective() {
        let report = AffinityReport {
            policy: AffinityPolicy::Core,
            online_processing_units: Some(96),
            visible_processing_units: Some(48),
            physical_cores: Some(48),
            numa_nodes: Some(4),
            visible_cpus: vec![0, 2, 4, 6],
            memory_policy: Some("bind".to_string()),
            memory_policy_nodes: vec![0, 1],
            allowed_memory_nodes: vec![0, 1, 2, 3],
            memory_discovery_error: None,
            worker_cpus: vec![0, 2],
            discovery_error: None,
        };
        let local = rank_affinity_from_report(0, &report);
        let gathered = gather_rank_affinity(&LocalBackend, &local)
            .expect("local affinity gather must succeed");

        assert_eq!(gathered.len(), 1);
        assert_eq!(gathered[0].policy, "core");
        assert_eq!(gathered[0].visible_processing_units, Some(48));
        assert_eq!(gathered[0].worker_cpus, vec![0, 2]);
    }

    fn d29_case_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d29-weekly-par-noise-sharing")
    }

    /// D29's 4 weekly study stages all declare `season_id=0`, so
    /// `study_stage_noise_group_ids` groups them into a single shared noise
    /// group — the sharing case a monthly (all-unique-groups) fixture cannot
    /// exercise.
    #[test]
    fn non_root_opening_tree_matches_rank_0_under_shared_noise_groups() {
        let case_dir = d29_case_dir();
        let args = RunArgs {
            case_dir: case_dir.clone(),
            output: None,
            quiet: true,
            threads: None,
            cpu_bind: CpuBindArg::None,
            comm_backend: CommBackendArg::Local,
        };
        let (prepared, _hydro_models, bcast, _config, _scalars, _timings) =
            load_case_and_config(&args, true, &Term::stderr())
                .expect("D29 must load and prepare stochastic context on rank 0");

        let ids = study_stage_noise_group_ids(&prepared.system);
        assert!(
            ids.windows(2).any(|w| w[0] == w[1]),
            "D29 fixture must exercise shared noise groups; got {ids:?}",
        );

        let rank0_tree = prepared.stochastic.opening_tree();

        let non_root = reconstruct_stochastic_context_non_root(
            &prepared.system,
            &bcast,
            None,
            bcast.seed,
            &case_dir,
        )
        .expect("non-root reconstruction must succeed for D29");
        let non_root_tree = non_root.opening_tree();

        assert_eq!(
            non_root_tree.data(),
            rank0_tree.data(),
            "non-root opening tree data must match rank 0's under shared noise groups"
        );
        assert_eq!(
            non_root_tree.openings_per_stage_slice(),
            rank0_tree.openings_per_stage_slice(),
            "non-root opening tree shape must match rank 0's under shared noise groups"
        );
        assert_eq!(
            non_root_tree.dim(),
            rank0_tree.dim(),
            "non-root opening tree dim must match rank 0's under shared noise groups"
        );
    }
}
