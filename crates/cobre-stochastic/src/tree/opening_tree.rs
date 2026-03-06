//! Opening scenario tree data structure.
//!
//! Defines the in-memory representation of the opening tree: a rooted
//! directed acyclic graph where each node holds the initial hydro storage
//! levels and inflow realisations for one branch of the first-stage
//! scenario fan. The tree is constructed once per iteration and shared
//! (read-only) across all stages.
