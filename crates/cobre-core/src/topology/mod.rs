//! Topology representations for cascade and transmission network structures.
//!
//! The topology sub-modules define resolved, validated representations of the
//! hydro cascade chain and the electrical transmission network. Both are built
//! during case loading and stored on the [`crate::system`] struct.

pub mod cascade;
pub mod network;

pub use cascade::CascadeTopology;
pub use network::{BusGenerators, BusLineConnection, BusLoads, NetworkTopology};
