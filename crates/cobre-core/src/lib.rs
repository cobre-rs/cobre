//! # cobre-core
//!
//! Shared data model for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate defines the fundamental types used across all Cobre tools:
//! buses, branches, generators (hydro, thermal, renewable), loads, network
//! topology, and the top-level [`system`] struct. A power system defined with
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
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.

// Relax strict production lints for test builds. These lints (unwrap_used,
// expect_used, etc.) guard library code but are normal in tests.
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

pub mod entities;
pub mod entity_id;
pub mod error;
pub mod penalty;
pub mod system;
pub mod topology;

pub use entities::{
    Bus, ContractType, DeficitSegment, DiversionChannel, EfficiencyModel, EnergyContract,
    FillingConfig, GnlConfig, HydraulicLossesModel, Hydro, HydroGenerationModel, HydroPenalties,
    Line, NonControllableSource, PumpingStation, TailraceModel, TailracePoint, Thermal,
    ThermalCostSegment,
};
pub use entity_id::EntityId;
pub use error::ValidationError;
pub use penalty::{
    resolve_bus_deficit_segments, resolve_bus_excess_cost, resolve_hydro_penalties,
    resolve_line_exchange_cost, resolve_ncs_curtailment_cost, GlobalPenaltyDefaults,
    HydroPenaltyOverrides,
};
pub use system::{System, SystemBuilder};
pub use topology::{BusGenerators, BusLineConnection, BusLoads, CascadeTopology, NetworkTopology};
