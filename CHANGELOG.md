# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-header -->

## [Unreleased]

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
