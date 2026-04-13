//! Entity type definitions for all power system elements.
//!
//! Each sub-module defines one entity type (Bus, Line, Hydro, Thermal, etc.).
//! All entity structs use [`crate::EntityId`] for identification and cross-references.

pub mod bus;
pub mod energy_contract;
pub mod hydro;
pub mod line;
pub mod non_controllable;
pub mod pumping_station;
pub mod thermal;

pub use bus::{Bus, DeficitSegment};
pub use energy_contract::{ContractType, EnergyContract};
pub use hydro::{
    DiversionChannel, EfficiencyModel, FillingConfig, HydraulicLossesModel, Hydro,
    HydroGenerationModel, HydroPenalties, TailraceModel, TailracePoint,
};
pub use line::Line;
pub use non_controllable::NonControllableSource;
pub use pumping_station::PumpingStation;
pub use thermal::{GnlConfig, Thermal};
