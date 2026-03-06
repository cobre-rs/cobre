//! Scenario tree construction.
//!
//! Builds the opening scenario tree used during the first stage of
//! iterative optimization algorithms. The tree encodes a branching
//! structure of initial hydro storage states and inflow realisations
//! that are sampled at the start of each iteration.
//!
//! ## Submodules
//!
//! - [`opening_tree`] — data structure representing the opening scenario tree
//! - [`generate`] — constructs the tree from PAR model outputs and
//!   correlation factors

pub mod generate;
pub mod opening_tree;

pub use generate::generate_opening_tree;
pub use opening_tree::{OpeningTree, OpeningTreeView};
