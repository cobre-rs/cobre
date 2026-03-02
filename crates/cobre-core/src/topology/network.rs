//! Resolved electrical transmission network topology.
//!
//! `NetworkTopology` holds the validated adjacency structure derived from the
//! `Line` entity collection. It is built during case loading after all `Bus` and
//! `Line` entities have been validated and their cross-references verified.
