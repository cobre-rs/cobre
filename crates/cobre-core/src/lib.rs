//! # cobre-core
//!
//! Shared data model for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate defines the fundamental types used across all Cobre tools:
//! buses, branches, generators (hydro, thermal, renewable), loads, and the
//! top-level [`system`] struct. A power system defined with
//! `cobre-core` types can be used for power flow analysis, optimization, dynamic
//! simulation, and any other analysis procedure in the ecosystem.
//!
//! ## Design principles
//!
//! - **Solver-agnostic**: no solver or algorithm dependencies.
//! - **Validate at construction**: invalid states are caught when building
//!   the system, not at solve time.
//! - **Shared types**: a `Hydro` is the same struct whether used in
//!   stochastic optimization or steady-state analysis.
//! - **Declaration-order invariance**: all entity collections are stored in
//!   canonical ID-sorted order so results are identical regardless of input ordering.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the current status.

// Internal (unpublished) workspace crate: public items intra-doc-link their
// pub(crate) collaborators as a maintainer aid (docs read with
// --document-private-items); the public-only doc gate flags these intentional links.
#![allow(rustdoc::private_intra_doc_links)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::panic,
        clippy::too_many_lines
    )
)]

pub mod commissioning;
pub mod constraints;
pub mod entities;
pub mod entity_id;
pub mod error;
pub mod model;
pub mod stats;
pub mod system;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod topology;

pub use model::{parameters, penalty, resolved, scenario, temporal};

pub use constraints::{generic_constraint, initial_conditions, training_event};

pub use constraints::generic_constraint::{
    AffineBound, ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
};
pub use constraints::initial_conditions::{
    AnticipatedCommitmentHistory, HydroPastDefluence, HydroStorage, InitialConditions,
    RecentObservation,
};
pub use constraints::training_event::{
    StageRowSelectionRecord, TrainingEvent, WORKER_TIMING_SLOT_BWD_SETUP,
    WORKER_TIMING_SLOT_BWD_WALL, WORKER_TIMING_SLOT_COUNT, WORKER_TIMING_SLOT_FWD_SETUP,
    WORKER_TIMING_SLOT_FWD_WALL, WORKER_TIMING_SLOT_SCORING, WorkerPhaseTimings, WorkerTimingPhase,
};
pub use entities::{
    AnticipatedConfig, Bus, ContractType, DeficitSegment, DiversionChannel, EfficiencyModel,
    EnergyContract, FillingConfig, HydraulicLossesModel, Hydro, HydroGenerationModel,
    HydroPenalties, HydroUnitGroup, Line, NonControllableSource, PumpingStation, TailraceModel,
    TailracePoint, Thermal,
};
pub use entity_id::EntityId;
pub use error::ValidationError;
pub use model::horizon::HorizonGraph;
pub use model::parameters::{CoefficientRef, ComputedParameter, ParameterKind, ScalarParameter};
pub use model::penalty::{
    GlobalPenaltyDefaults, HydroPenaltyOverrides, resolve_bus_deficit_segments,
    resolve_bus_excess_cost, resolve_hydro_penalties, resolve_line_exchange_cost,
    resolve_ncs_curtailment_cost,
};
pub use model::post_study::{PostStudyStage, PostStudyStages, PostStudyThermalBound};
pub use model::resolved::{
    BlockBoundsCountsSpec, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ContractBlockBounds, ContractBlockOverride, HydroBlockBounds, HydroBlockOverride,
    HydroStageBounds, HydroUnitGroupBoundsCountsSpec, HydroUnitGroupOverride, LineBlockBounds,
    LineBlockOverride, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
    PenaltiesDefaults, PumpingBlockBounds, PumpingBlockOverride, ResolvedBlockBounds,
    ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedHydroUnitGroupBounds,
    ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties,
    ThermalBlockBounds, ThermalBlockOverride, ThermalStageBounds,
};
pub use model::scenario::{
    CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
    CorrelationScheduleEntry, ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow,
    HistoricalYears, InflowHistoryRow, InflowModel, LoadModel, NcsModel, SamplingScheme,
    ScenarioSource,
};
pub use model::temporal::{
    Block, BlockMode, CalendarMonth, Node, NoiseMethod, PolicyGraphType, ScenarioSourceConfig,
    SeasonCycleType, SeasonDefinition, SeasonMap, Stage, StageId, StageRiskConfig,
    StageStateConfig, StudyPos, Transition, month_of, window_period_overlaps,
    window_period_reach_depth, window_reaches_any_period,
};
pub use stats::welford::WelfordAccumulator;
pub use system::{System, SystemBuilder};
pub use topology::CascadeTopology;
