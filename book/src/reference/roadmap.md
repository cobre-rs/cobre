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
- **CVaR risk measure** -- Convex combination `(1 - λ)·E[Z] + λ·CVaR_α[Z]`
  configurable per stage via the `risk_measure` field in `stages.json`.
- **`cobre summary` subcommand** -- Post-run summary reporting subcommand that
  prints convergence statistics and output file locations.

## v0.1.2 Deliverables

The following features were delivered in v0.1.2:

- **Cut selection wiring** -- Level-1 cut selection strategy connected to the
  training loop; inactive cuts are retained in the pool for potential
  reactivation and reported in convergence output.
- **Generic PAR evaluation aliases** -- Public API aliases that decouple PAR
  evaluation from inflow-specific naming, enabling reuse across load and
  other stochastic processes.
- **Documentation accuracy** -- Roadmap sections, crate overview pages, and
  book reference pages updated to match the implemented state.
- **Design ADRs** -- Architecture Decision Records for opening tree
  user-supply, stochastic artifact export, complete tree work distribution,
  and per-stage warm-start cuts.

## v0.1.3 Deliverables

The following features were delivered in v0.1.3:

- **`StudySetup` extraction** -- Case-loading logic deduplicated into a shared
  `StudySetup` struct, eliminating divergence between the training and simulation
  pipelines.
- **Noise transformation deduplication** -- PAR noise draw logic consolidated
  into `noise.rs`, removing duplicated transformation paths across the forward
  pass and scenario generation.
- **CLI flag cleanup** -- Four flags removed (`--skip-simulation`, `--no-banner`,
  `--verbose`, `--export-stochastic`) in favour of structured configuration,
  reducing CLI surface and simplifying the execution lifecycle.
- **Stochastic summary module** -- Scenario statistics reporting extracted into
  `summary.rs` for cleaner separation from the training loop.
- **Simulation progress fix** -- `WelfordAccumulator` moved to `cobre-core` and
  per-worker accumulation bug corrected, restoring accurate progress reporting
  during the simulation pipeline.
- **User-supplied opening tree (ADR-008)** -- Callers can now supply a custom
  opening tree instead of the auto-generated one, enabling reproducible scenario
  studies with fixed initial conditions.
- **Stochastic artifact export (ADR-009)** -- The forward pass can export sampled
  scenario trajectories as Parquet artifacts for offline analysis and validation.
- **`ScratchBuffers` separation** -- Per-iteration scratch memory separated from
  training state, reducing allocation pressure on the hot path.
- **`StageContext` and `TrainingContext` structs** -- Structured context types
  introduced at pass boundaries to replace ad-hoc parameter threading.

## Sections

The roadmap is organized into four areas:

- **Inflow Truncation Methods** -- `TruncationWithPenalty` remains deferred
  from v0.1.0. (`Truncation` was delivered in v0.1.1.)
- **HPC Optimizations** -- Performance improvements beyond the rayon baseline,
  grouped into near-term (v0.1.x/v0.2.x) and longer-term (v0.3+) items.
- **Post-MVP Crates** -- Implementation plans for the three stubbed workspace
  crates: `cobre-mcp`, `cobre-python`, and `cobre-tui`.
- **Algorithm Extensions** -- Deferred solver variants: multi-cut formulation
  and infinite-horizon policy graphs. (CVaR risk measure was delivered in
  v0.1.1. Cut selection wiring was delivered in v0.1.2.)
