# Cobre

**Open infrastructure for power system computation. Built in Rust.**

Cobre is an ecosystem of Rust crates for power system analysis and optimization. The first solver vertical implements Stochastic Dual Dynamic Programming (SDDP) for long-term hydrothermal dispatch -- a problem central to energy planning in systems with large hydroelectric capacity.

## Design goals

- **Production-grade HPC**: hybrid MPI + thread parallelism, designed for cluster execution via `mpiexec`
- **Reproducible results**: deterministic output regardless of rank count, thread count, or input ordering
- **Modular architecture**: 11 crates with clean boundaries, each independently testable
- **Open solver stack**: HiGHS LP solver, no proprietary dependencies for core functionality

## Current status

Phases 1 through 6 are complete. The ecosystem delivers a full SDDP training pipeline: entity model and topology validation (`cobre-core`), JSON/Parquet case loading with 5-layer validation (`cobre-io`), LP solver abstraction with HiGHS backend and warm-start basis management (`cobre-solver`), pluggable communication with MPI and local backends (`cobre-comm`), PAR(p) inflow models with deterministic correlated scenario generation (`cobre-stochastic`), and the SDDP training loop with forward/backward passes, Benders cut generation, cut synchronization, convergence monitoring, and composite stopping rules (`cobre-sddp`) -- verified by 1490 tests across the workspace. Implementation continues through the [8-phase build sequence](https://cobre-rs.github.io/cobre-docs/specs/overview/implementation-ordering.html), with Phase 7 (simulation + output) as the next candidate.

## Quick links

|                       |                                                                         |
| --------------------- | ----------------------------------------------------------------------- |
| GitHub                | [github.com/cobre-rs/cobre](https://github.com/cobre-rs/cobre)          |
| API docs (rustdoc)    | `cargo doc --workspace --no-deps --open`                                |
| Methodology reference | [cobre-rs.github.io/cobre-docs](https://cobre-rs.github.io/cobre-docs/) |
| License               | [Apache-2.0](https://github.com/cobre-rs/cobre/blob/main/LICENSE)       |
