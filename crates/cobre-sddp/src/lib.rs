//! SDDP solver for hydrothermal dispatch.
//!
//! Implements the SDDP algorithm: forward/backward passes, Benders cuts, risk measures,
//! convergence monitoring, and policy simulation. Parallelized via rayon (intra-rank)
//! and ferrompi (inter-rank).

// Relax strict production lints for test builds (normal in test contexts).
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp,
    )
)]

pub mod backward;
pub(crate) mod backward_pass_state;
pub mod basis_reconstruct;
pub mod config;
pub mod context;
pub mod convergence;
pub mod conversion;
pub mod cut;
pub mod cut_selection;
pub mod cut_sync;
pub mod error;
pub mod estimation;
pub mod forward;
pub(crate) mod forward_pass_state;
pub(crate) mod fpha_fitting;
pub mod generic_constraints;
pub mod horizon_mode;
pub mod hydro_models;
pub mod indexer;
pub mod inflow_method;
pub(crate) mod lag_transition;
pub mod lower_bound;
pub mod lp_builder;
pub(crate) mod noise;
pub mod policy_export;
pub mod policy_load;
pub mod provenance;
pub mod risk_measure;
pub mod scaling_report;
pub mod setup;
pub mod simulation;
pub mod solver_stats;
pub mod stage_solve;
pub mod state_exchange;
pub mod stochastic_summary;
pub mod stopping_rule;
pub mod training;
pub mod training_output;
pub(crate) mod training_session;
pub mod trajectory;
pub mod visited_states;
pub mod workspace;

// ── config ────────────────────────────────────────────────────────────────────
pub use config::TrainingConfig;
// ── cut ───────────────────────────────────────────────────────────────────────
pub use cut::FutureCostFunction;
// ── cut_selection ─────────────────────────────────────────────────────────────
pub use cut_selection::CutSelectionStrategy;
// ── error ─────────────────────────────────────────────────────────────────────
pub use error::SddpError;
// ── estimation ────────────────────────────────────────────────────────────────
pub use estimation::{EstimationPath, EstimationReport};
// ── hydro_models ──────────────────────────────────────────────────────────────
pub use hydro_models::{
    FphaHydroDetail, HydroModelSummary, PrepareHydroModelsResult, ProductionModelSource,
    build_hydro_model_summary, prepare_hydro_models,
};
// ── inflow_method ─────────────────────────────────────────────────────────────
pub use inflow_method::InflowNonNegativityMethod;
// ── lp_builder ────────────────────────────────────────────────────────────────
pub use lp_builder::build_stage_templates;
// ── policy_load ───────────────────────────────────────────────────────────────
pub use policy_load::{
    build_basis_cache_from_checkpoint, inject_boundary_cuts, load_boundary_cuts,
    validate_policy_compatibility,
};
// ── provenance ────────────────────────────────────────────────────────────────
pub use provenance::{ModelProvenanceReport, ProvenanceSource, build_provenance_report};
// ── setup ─────────────────────────────────────────────────────────────────────
pub use setup::{
    DEFAULT_MAX_ITERATIONS, DEFAULT_SEED, PrepareStochasticResult, StudyParams, StudySetup,
    prepare_stochastic,
};
// ── simulation ────────────────────────────────────────────────────────────────
pub use simulation::{SimulationError, SimulationSummary, aggregate_simulation, simulate};
// ── solver_stats ──────────────────────────────────────────────────────────────
pub use solver_stats::{
    SOLVER_STATS_DELTA_SCALAR_FIELDS, SolverStatsDelta, pack_delta_scalars, pack_scenario_stats,
    unpack_delta_scalars, unpack_scenario_stats,
};
// ── stochastic_summary ────────────────────────────────────────────────────────
pub use stochastic_summary::{
    ArOrderSummary, StochasticSource, StochasticSummary, build_stochastic_summary,
    estimation_report_to_fitting_report, inflow_models_to_annual_component_rows,
    inflow_models_to_ar_rows, inflow_models_to_stats_rows,
};
// ── stopping_rule ─────────────────────────────────────────────────────────────
pub use stopping_rule::{StoppingMode, StoppingRule, StoppingRuleSet};
// ── training ──────────────────────────────────────────────────────────────────
pub use training::{TrainingResult, train};
// ── training_output ───────────────────────────────────────────────────────────
pub use training_output::build_training_output;
