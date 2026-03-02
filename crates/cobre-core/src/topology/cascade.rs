//! Resolved hydro cascade topology.
//!
//! `CascadeTopology` holds the validated, cycle-free directed graph of hydro plant
//! relationships. It is built during case loading after all `Hydro` entities have
//! been validated and their `downstream_id` cross-references verified.
