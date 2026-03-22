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

## v0.1.4 Deliverables

The following features were delivered in v0.1.4:

- **FPHA hydro production model** -- Four-piece hyperplane approximation for
  variable-head hydroelectric plants. Supports precomputed hyperplanes (supplied
  via `fpha_hyperplanes.parquet`) and computed-from-geometry mode (fitted from
  forebay/tailrace curves in `hydro_geometry.parquet`).
- **Evaporation linearization** -- Reservoir surface evaporation modeled as a
  linearized function of stored volume with per-season reference volumes.
- **Hydro model preprocessing pipeline** -- Unified resolution of production model
  configuration and evaporation parameters into solver-ready structures.
- **FPHA/evaporation result extraction** -- Simulation output now includes
  FPHA-related variables and evaporation volumes.
- **Deterministic regression test suite** -- 12 hand-computed test cases (D01--D12)
  covering thermal dispatch, single hydro, cascade, transmission, FPHA variants,
  evaporation, multi-deficit, and inflow non-negativity.
- **MSRV bumped to 1.86** -- Workspace `rust-version` updated from 1.85 to 1.86.

## v0.1.5 Deliverables

The following features were delivered in v0.1.5:

- **Python bindings (PyO3)** -- `cobre-python` package with case loading,
  validation, training, simulation, and Arrow zero-copy result inspection.
  Tested on Python 3.12/3.13/3.14.
- **Past inflows initialization** -- `past_inflows` field in
  `initial_conditions.json` for initializing PAR(p) lag state from historical
  inflow records.

## v0.1.6 Deliverables

The following features were delivered in v0.1.6:

- **Generic constraints LP wiring** -- User-defined generic constraints over 20
  LP variable types (thermal generation, hydro storage, spillage, line flow,
  etc.) with slack penalties and violation output.
- **Water withdrawal** -- Hydroelectric water withdrawal constraints modeled as
  LP bounds on turbined outflow.
- **JSON schemas** -- Formal JSON Schema definitions for all input files,
  enabling IDE validation and tooling integration.

## v0.1.7 Deliverables

The following features were delivered in v0.1.7:

- **Block factors** -- Per-bus load demand factors, per-line exchange capacity
  factors, and per-source NCS availability factors (per-stage, per-block).
- **NCS stochastic availability** -- Non-controllable source stochastic model
  via `non_controllable_stats.parquet` (mean + std availability factor per
  source per stage, clamped normal draw patched per scenario in
  forward/backward pass and lower bound evaluation).
- **Deterministic regression tests D13--D15** -- NCS availability, block
  factors, and combined NCS + block factor test cases.

## v0.1.8 Deliverables

The following features were delivered in v0.1.8:

- **LineExchange variable** -- `line_exchange` variant added to the generic
  constraint variable catalog, enabling constraints on net power flow across
  transmission lines.
- **Per-stage productivity override** -- `hydro_production_models.json` supports
  per-stage productivity replacement, overriding base `productivity_mw_per_m3s`
  for specific stages or seasons.

## v0.1.9 Deliverables

The following features were delivered in v0.1.9:

- **PAR estimation overhaul** -- Periodic Yule-Walker coefficient estimation,
  PACF-based order selection (replacing AIC), contribution-based validation,
  negative phi_1 rejection gate, and iterative PACF order reduction for
  improved numerical stability.
- **LP scaling** -- Row scaling with RHS prescaling and dual unscaling, plus
  internal objective cost scaling (`COST_SCALE_FACTOR`), improving solver
  conditioning on real-world systems.
- **Solver statistics** -- Three-channel instrumentation: LP scaling diagnostics
  report (JSON), per-phase solver statistics (Parquet), and enhanced CLI display
  with per-solve timing, basis reuse tracking, and simplex iteration counts.
- **Per-scenario simulation statistics** -- Individual scenario cost and LP solve
  metrics exported alongside aggregate simulation results.

## Sections

The roadmap is organized into four areas:

- **Inflow Truncation Methods** -- `TruncationWithPenalty` remains deferred
  from v0.1.0. (`Truncation` was delivered in v0.1.1.)
- **HPC Optimizations** -- Performance improvements beyond the rayon baseline,
  grouped into near-term (v0.1.x/v0.2.x) and longer-term (v0.3+) items.
- **Post-MVP Crates** -- `cobre-python` was delivered in v0.1.5. `cobre-mcp`
  and `cobre-tui` remain stubbed for future implementation.
- **Algorithm Extensions** -- Deferred solver variants: multi-cut formulation
  and infinite-horizon policy graphs. (CVaR risk measure was delivered in
  v0.1.1. Cut selection wiring was delivered in v0.1.2.)
