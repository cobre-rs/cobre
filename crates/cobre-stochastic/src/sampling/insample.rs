//! `InSample` scenario sampling scheme.
//!
//! The `InSample` scheme draws a fixed set of scenarios from the opening
//! tree at the start of each iteration and uses the same scenarios for
//! all stages. This scheme is the minimal viable sampling strategy for
//! iterative stochastic optimization algorithms and serves as the
//! baseline from which other sampling strategies can be derived.
