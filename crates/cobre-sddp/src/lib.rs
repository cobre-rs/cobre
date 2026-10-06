//! SDDP solver for hydrothermal dispatch.

#![allow(
    rustdoc::private_intra_doc_links,
    reason = "public items link their pub(crate) collaborators for docs read with --document-private-items; an expect would be unfulfilled there"
)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
    )
)]

pub(crate) mod block_clock;
pub(crate) mod bucket_topology;
pub(crate) mod claim_scatter;
pub mod config;
pub mod convergence;
pub mod cut;
pub mod error;
pub(crate) mod fixed_delivery_echo;
pub(crate) mod gemm;
pub(crate) mod generic_constraint_echo;
pub mod horizon_mode;
pub(crate) mod hull;
pub mod lead_time;
pub mod lp;
pub mod policy;
pub mod production;
pub mod setup;
pub mod simulation;
pub mod solve;
pub mod solver_stats;
pub mod stochastic;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub(crate) mod time_value;
pub mod training;
pub mod validate_phases;
pub mod workspace;

pub use workspace::context;

pub use solve::solver_phase;
pub(crate) use solve::stage_solve;

pub use convergence::{risk_measure, stopping_rule};

pub use cut::{basis_reconstruct, cut_selection, cut_sync, dcs};

pub use lp::indexer;

pub use policy::{policy_export, resolved_parameters, scaling_report};

// `hydro_models::prepare_hydro_models_from_artifacts` is intentionally absent
// from the curated re-export — this shim is its sole resolution path.
pub(crate) use production::fpha_fitting;
pub use production::{energy_conversion, hydro_models};

pub use stochastic::inflow_method;
pub(crate) use stochastic::{noise, stochastic_summary};

pub(crate) use training::{
    backward, backward_pass_state, forward_pass_state, rank_reconcile, state_exchange, trajectory,
    visited_states,
};
pub use training::{forward, lower_bound};

// Test/tooling-only — integration tests need direct backward-pass access.
#[cfg(any(test, feature = "test-support"))]
pub use training::backward_pass_state::{BackwardPassInputs, BackwardPassState};

pub(crate) use training::session as training_session;

pub use cobre_io::scenarios::estimation::{
    EstimationPath, EstimationReport, estimate_from_history,
};
pub use config::TrainingConfig;
pub use convergence::convergence::ConvergenceMonitor;
pub use convergence::risk_measure::{BackwardOutcome, RiskMeasure};
pub use convergence::stopping_rule::{
    MonitorState, StopDecision, StopMask, StoppingMode, StoppingRule, StoppingRuleSet,
};
pub use cut::cut_selection::CutSelectionStrategy;
pub use cut::cut_sync::CutSyncBuffers;
pub use cut::row::build_cut_row_batch_into;
pub use cut::wire::{CutWireHeader, cut_wire_size, deserialize_cut, serialize_cut};
pub use cut::{CutPool, FutureCostFunction};
pub use error::{ErrorClass, SddpError};
pub use fixed_delivery_echo::build_fixed_delivery_rows;
pub use generic_constraint_echo::build_generic_constraint_echo_rows;
pub use lp::builder::StageTemplates;
pub use policy::policy_export::{ReservedInflowLagLayout, reserve_boundary_inflow_lag_slots};
pub use policy::policy_load::{
    BoundaryInjection, BoundaryLoadRequest, BoundaryReconciliation, FullFcf,
    LEGACY_COST_SCALE_FACTOR, PolicyLoadKind, PolicyLoadProof, PolicyStageManifest,
    StoredBasisLoad, StoredBasisMisfit, UnusedStoredBases, ValidatedBoundaryCuts,
    boundary_policy_required_lag_depth, build_basis_cache_from_checkpoint,
    checkpoint_terminal_cost_scale_factor, compare_manifest_slot_identity, inject_boundary_cuts,
    load_boundary_cuts, reconcile_boundary_policy, rescale_checkpoint_cuts_for_load,
    resolve_boundary_state_requirements, validate_policy_load,
};
pub use policy::provenance::{
    HydroProductionProvenance, InflowProvenance, ModelProvenanceReport, ProvenanceSource,
    build_provenance_report,
};
pub use policy::reconcile::{
    AnticipatedCoverage, BoundaryReconciliationReport, FamilyTally, SlotDetail,
};
pub use policy::resolved_parameters::{
    ResolvedParameters, ResolvedParametersError, build_resolved_parameters,
};
pub use production::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};
pub use production::hydro_models::{
    FphaFitDeviationEntry, FphaHydroDetail, HydroFitTimings, HydroModelSummary,
    NoTurbineCapacityHydro, PrepareHydroModelsResult, ProductionModelSource,
    build_deviation_summary, build_evaporation_model_rows, build_hydro_model_summary,
    prepare_hydro_models,
};
#[cfg(any(test, feature = "test-support"))]
pub use setup::lp_build_inputs::build_stage_templates_resolving_layout;
pub use setup::{
    BoundaryStateRequirements, DEFAULT_COST_SCALE_FACTOR, DEFAULT_SEED, PrepareStochasticResult,
    StudyParams, StudySetup, build_stochastic_context_for_study, prepare_stochastic,
    study_horizon_end, validate_generic_constraint_parameters,
};
pub use simulation::{
    ScenarioCategoryCosts, SimulationError, SimulationHydroResult, SimulationScenarioResult,
    SimulationStageResult, SimulationSummary, SimulationWeighting, aggregate_simulation, simulate,
};
#[cfg(feature = "highs")]
pub use solve::solver_phase::{BACKWARD_PROFILE, FORWARD_PROFILE, SIMULATION_PROFILE};
pub use solve::solver_phase::{Phase, SolverProfiles};
pub use solver_stats::{
    SOLVER_STATS_DELTA_SCALAR_FIELDS, SolverStatsDelta, SolverStatsLogEntry,
    aggregate_solver_stats_log, delta_to_stats_row, pack_delta_scalars, pack_scenario_stats,
    solver_stats_log_to_rows, unpack_delta_scalars, unpack_scenario_stats,
};
pub use stochastic::inflow_method::InflowNonNegativityMethod;
pub use stochastic::stochastic_summary::{
    ArOrderSummary, StochasticSource, StochasticSummary, build_stochastic_summary,
    estimation_report_to_fitting_report, inflow_models_to_annual_component_rows,
    inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};
pub use training::forward::SyncResult;
pub use training::rank_reconcile::reconcile_global_ok;
pub use training::state_exchange::ExchangeBuffers;
pub use training::training::{TrainingOutcome, TrainingResult, train};
pub use training::training_output::{
    PhaseTimingTotals, build_training_output, sum_phase_timing_ms,
};
pub use training::trajectory::TrajectoryRecord;
pub use workspace::workspace::{BASIS_BROADCAST_FORMAT_TAG, CapturedBasis};
