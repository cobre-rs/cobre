# cobre-stochastic

Stochastic process models for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

This crate provides the probabilistic building blocks used in scenario-based stochastic
optimization of power systems. It implements Periodic Autoregressive (PAR(p)) models
for inflow time series following the methodology used in the Brazilian power sector,
Cholesky-based spatial correlation for multi-variate scenario generation, and
deterministic communication-free noise generation via SipHash-1-3 seed derivation.
The `StochasticContext` bundles all precomputed parameters and the opening tree into
a single value ready for iterative optimization algorithms.

## When to Use

Depend on `cobre-stochastic` when you need to generate correlated stochastic scenarios
for a power system optimization algorithm. If you are implementing a new iterative
algorithm that draws inflow or load realisations at each iteration, `sample_forward`
and `StochasticContext` are the primary entry points. The crate is solver-agnostic
and carries no dependency on LP or MIP solvers.

## Key Types

- **`StochasticContext`** — Bundles precomputed PAR parameters, correlated factors, and the opening tree for use in iterative algorithms
- **`ForwardSampler`** — Composite struct holding one `ClassSampler` per entity class; entry point for forward-pass noise sampling
- **`ClassSampler`** — Per-entity-class noise source enum with variants InSample, OutOfSample, Historical, and External
- **`HistoricalScenarioLibrary`** — Pre-standardized historical inflow windows for historical replay sampling
- **`ExternalScenarioLibrary`** — Pre-standardized external scenarios for inflow, load, or NCS classes
- **`PrecomputedPar`** — Precomputed PAR(p) seasonal statistics and AR coefficients ready for fast evaluation
- **`OpeningTree`** — Scenario tree structure defining which openings are sampled at each stage
- **`CholeskyFactor`** — Lower-triangular Cholesky decomposition used to apply spatial correlation to noise draws
- **`build_forward_sampler`** — Factory function that constructs a `ForwardSampler` from a `ForwardSamplerConfig`

## Links

| Resource   | URL                                                       |
| ---------- | --------------------------------------------------------- |
| Book       | https://cobre-rs.github.io/cobre/crates/stochastic.html   |
| API Docs   | https://docs.rs/cobre-stochastic/latest/cobre_stochastic/ |
| Repository | https://github.com/cobre-rs/cobre                         |
| CHANGELOG  | https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md  |

## Status

Alpha — API is functional but not yet stable.

## License

Apache-2.0
