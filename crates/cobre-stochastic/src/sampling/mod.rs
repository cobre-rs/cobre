//! Scenario sampling schemes.
//!
//! Provides sampling strategies that select which scenarios are simulated
//! during each iteration of a stochastic optimization algorithm. The
//! minimal viable implementation supports the `InSample` scheme, in which
//! the same set of scenario openings is used for both the forward and
//! backward phases of a given iteration.
//!
//! ## Submodules
//!
//! - [`insample`] — `InSample` scheme: draws scenarios uniformly from the
//!   opening tree and fixes them for the full iteration

pub mod insample;
