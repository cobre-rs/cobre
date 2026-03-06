//! PRNG wrapper for noise sampling.
//!
//! Initialises a `Pcg64` generator from a derived seed and samples
//! vectors of independent standard-normal (`N(0,1)`) variates. The
//! samples are consumed by the correlation module to produce spatially
//! correlated noise vectors for each scenario and stage.
