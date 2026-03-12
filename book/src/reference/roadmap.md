# Roadmap

The Cobre post-v0.1.0 roadmap documents deferred features, HPC optimizations,
and post-MVP crates planned for future releases. It covers implementation-level
work items only; for methodology-level roadmap pages (algorithm theory, spec
evolution), see the
[cobre-docs roadmap](https://cobre-rs.github.io/cobre-docs/roadmap/).

The roadmap is maintained in the repository at
[`docs/ROADMAP.md`](https://github.com/cobre-rs/cobre/blob/main/docs/ROADMAP.md).

## v0.1.1 Deliverables

The following features were delivered in v0.1.1:

- **PAR model simulation** -- Scenario generation using fitted PAR(p) models
  during the simulation pipeline, producing scenario-consistent inflow traces.
- **Inflow truncation** -- The `Truncation` non-negativity treatment method,
  which clamps negative PAR model draws to zero before applying noise.
- **Stochastic load noise** -- Correlated Gaussian noise added to load
  forecasts using the same Cholesky-based framework as inflow noise.
- **PAR estimation from history** -- Fitting PAR(p) model coefficients from
  historical inflow records provided in the case directory.
- **`cobre summary` subcommand** -- Post-run summary reporting subcommand that
  prints convergence statistics and output file locations.

## Sections

The roadmap is organized into four areas:

- **Inflow Truncation Methods** -- `TruncationWithPenalty` remains deferred
  from v0.1.0. (`Truncation` was delivered in v0.1.1.)
- **HPC Optimizations** -- Performance improvements beyond the rayon baseline,
  grouped into near-term (v0.1.x/v0.2.x) and longer-term (v0.3+) items.
- **Post-MVP Crates** -- Implementation plans for the three stubbed workspace
  crates: `cobre-mcp`, `cobre-python`, and `cobre-tui`.
- **Algorithm Extensions** -- Deferred solver variants: CVaR risk measure,
  multi-cut formulation, and infinite-horizon policy graphs.
