# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-header -->

## [Unreleased]

### Added

- **`StudySetup` struct** -- centralized study orchestration in cobre-sddp,
  encapsulating all precomputed study state (templates, indexer, FCF,
  stochastic context, risk measures, entity counts, block layout). Extracted
  from the CLI pipeline so that multiple entry points (CLI, Python, future TUI
  and MCP) share a single construction and orchestration path.
- **`StageContext` and `TrainingContext` structs** -- lightweight context
  bundles in a dedicated `context.rs` module that reduce hot-path function
  argument counts. `StageContext` groups stage templates and layout slices;
  `TrainingContext` groups the solver-level study parameters.
- **`ScratchBuffers` struct** -- separates per-worker noise and patch scratch
  space from the `SolverWorkspace`, improving memory layout and making the
  allocation boundary between solver state and algorithm scratch explicit.
- **`noise.rs` module** -- consolidates all noise-to-RHS-patch logic into a
  single module with shared `transform_inflow_noise` and `transform_load_noise`
  functions called from the forward pass, backward pass, and simulation
  pipeline, eliminating the three-way duplication identified in the v0.1.1
  post-release assessment.
- **`WelfordAccumulator` in cobre-core** -- streaming online statistics
  accumulator (mean, variance, standard deviation) for use in progress
  reporting and any algorithm that requires running statistics without storing
  all observations.
- **Stochastic summary** -- `summary.rs` in cobre-cli produces a structured
  post-setup report of the fitted PAR models with a three-tier AR order
  display: compact per-order form (≤10 hydros), range summary (11–30 hydros),
  and histogram (31+ hydros). Replaces the `[stochastic]` `eprintln!` pattern.
- **User-supplied opening tree** -- when `scenarios/noise_openings.parquet` is
  present in the case directory, Cobre loads, validates, and uses it as the
  backward-pass opening tree instead of generating one internally (ADR-008).
  The exported `output/stochastic/noise_openings.parquet` uses the same schema,
  so the round-trip from export to re-supply is a copy operation.
- **Stochastic artifact export** -- after the stochastic context is built,
  Cobre writes up to six artifact files to `output/stochastic/`: fitted
  seasonal statistics, AR coefficients, correlation matrix, fitting report,
  noise openings, and load seasonal statistics (ADR-009). Controlled by
  `exports.stochastic` in `config.json`.

### Changed

- **Hot-path argument counts reduced** -- `run_forward_pass`, `run_backward_pass`,
  and `simulate` each drop from 20–22 arguments to 7 by bundling dissolved
  context into `StageContext` and `TrainingContext`. The public `StudySetup`
  API exposes `train()` with 7 arguments and `simulate()` with 4.
- **Simulation progress uses a single global `WelfordAccumulator`** -- the
  progress thread now owns the sole accumulator and receives raw
  `scenario_cost: f64` events from workers. The `SimulationProgress` event is
  simplified to a single cost field, eliminating the per-worker pre-aggregation
  that caused the incorrect statistics bug.
- **CLI `execute()` uses `StudySetup`** -- training and simulation are
  orchestrated through `StudySetup::train()` and `StudySetup::simulate()`
  instead of threading individual parameters through the CLI.
- **Python `run_inner()` simplified** -- reduced from ~250 to ~125 lines by
  adopting `StudySetup::new()` for construction and `StudySetup::train()` /
  `StudySetup::simulate()` for orchestration.

### Removed

- **`--skip-simulation` CLI flag** -- simulation is now controlled exclusively
  via `config.json` (`simulation.enabled`).
- **`--no-banner` CLI flag** -- banner display is no longer user-configurable
  from the command line.
- **`--verbose` CLI flag** -- tracing subscriber setup has been removed from
  the CLI; structured logging configuration will be revisited in a future release.
- **`--export-stochastic` CLI flag** -- stochastic artifact export is controlled
  via `exports.stochastic` in `config.json`, not a command-line flag.
- **`format_stochastic_diagnostics` function** -- replaced by the structured
  `StochasticSummary` / `summary.rs` display system; the `[stochastic]`
  `eprintln!` pattern is removed from cobre-sddp.

### Fixed

- **Simulation progress bar statistics** -- the per-worker `WelfordAccumulator`
  accumulator produced incorrect mean and standard deviation because each
  worker tracked only its own subset of completed scenarios. The fix moves the
  single accumulator to the progress thread, which receives all scenario costs
  and computes globally correct statistics.
- **Clippy suppressions in cobre-sddp** -- remaining `#[allow(...)]`
  production suppressions reduced from 8+ to 2 by addressing each underlying
  lint finding.

## [0.1.2] - 2026-03-14

### Fixed

- Canonical upper bound summation for multi-rank determinism — the upper
  bound `allreduce` now uses a compensated (Kahan) summation that produces
  bit-for-bit identical results regardless of MPI rank count and scenario
  distribution across ranks.
- Removed all production clippy suppressions (`#[allow(...)]`) from cobre-sddp
  source files, addressing each underlying lint finding instead of silencing it.
- Addressed code review findings across cobre-sddp: simplified control flow,
  removed dead code, fixed off-by-one edge cases in stopping rules.

### Added

- ADR-008: User-supplied opening tree via Parquet file (design accepted,
  implementation planned for v0.1.3).
- ADR-009: Stochastic artifact export (design accepted, implementation
  planned for v0.1.3).
- ADR-010: Complete tree work distribution for forward/backward pass.
- ADR-011: Per-stage warm-start counts and terminal-stage boundary conditions.
- Generic PAR type aliases (`ParOrder`, `ParCoefficients`, `ParResidualStdRatio`)
  in cobre-stochastic for improved API clarity.

### Changed

- Updated software book: new 4-Region Example page, revised roadmap sections,
  fixed overview and SDDP crate pages, updated badges and DEC references.
- Updated cobre-stochastic docstrings to use generic terminology (no
  algorithm-specific language in infrastructure crate documentation).
- Python bindings: added Python 3.14 classifier and CI testing matrix
  (3.12, 3.13, 3.14).

## [0.1.1] - 2026-03-12

### Added

- PAR model simulation -- Scenario generation using fitted PAR(p) models
  during the simulation pipeline, producing scenario-consistent inflow traces.
- Inflow truncation -- The `Truncation` non-negativity treatment method,
  which clamps negative PAR model draws to zero before applying noise.
- Stochastic load noise -- Correlated Gaussian noise added to load
  forecasts using the same Cholesky-based framework as inflow noise.
- PAR estimation from history -- Fitting PAR(p) model coefficients from
  historical inflow records provided in the case directory.
- `cobre summary` subcommand -- Post-run summary reporting subcommand that
  prints convergence statistics and output file locations.

## [0.1.0] - 2026-03-09

### Added

- Phase 1 (cobre-core): Entity model (Bus, Line, Thermal, Hydro, Contract, PumpingStation, NonControllable), system registry, topology validation, three-tier penalty resolution
- Phase 2 (cobre-io): Case loader with five-layer validation, JSON/Parquet parsing for 33 input types, penalty/bound resolution
- Phase 3 (cobre-solver): LP solver abstraction with HiGHS backend, warm-start support, conformance tests
- Phase 4 (cobre-comm): Communicator trait with LocalBackend and FerrompiBackend, compile-time feature selection
- Phase 5 (cobre-stochastic): PAR(p) preprocessing, SipHash seed derivation, Cholesky correlation, opening trees, InSample sampling
- Phase 6 (cobre-sddp): SDDP training loop with forward/backward pass, Benders cuts, stopping rule set, convergence monitoring
- Phase 7 (cobre-sddp + cobre-io): Simulation pipeline with MPI aggregation, Hive-partitioned Parquet output, FlatBuffers policy checkpoint
- Phase 8 (cobre-cli): Command-line interface with `run`, `validate`, `report`, `version` subcommands, progress bars, exit codes

## [0.0.1] - 2026-02-23

### Added

- Initial workspace scaffold with 11 crates (`cobre`, `cobre-core`, `cobre-io`, `cobre-stochastic`, `cobre-solver`, `cobre-comm`, `cobre-sddp`, `cobre-cli`, `cobre-mcp`, `cobre-python`, `cobre-tui`)
- Reserved all crate names on crates.io
- CI pipeline: check, test, clippy, fmt, docs, security audit, license check, coverage
- Workspace lint configuration with clippy pedantic and `unsafe_code = "forbid"`
- cargo-dist configuration for multi-platform binary distribution

<!-- next-url -->

[Unreleased]: https://github.com/cobre-rs/cobre/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/cobre-rs/cobre/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cobre-rs/cobre/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cobre-rs/cobre/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/cobre-rs/cobre/releases/tag/v0.0.1
