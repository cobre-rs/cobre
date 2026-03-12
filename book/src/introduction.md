# Cobre

**Open infrastructure for power system computation. Built in Rust.**

Cobre is an ecosystem of Rust crates for power system analysis and optimization. The first solver vertical implements Stochastic Dual Dynamic Programming (SDDP) for long-term hydrothermal dispatch -- a problem central to energy planning in systems with large hydroelectric capacity.

## Design goals

- **Production-grade HPC**: hybrid MPI + thread parallelism, designed for cluster execution via `mpiexec`
- **Reproducible results**: deterministic output regardless of rank count, thread count, or input ordering
- **Modular architecture**: 11 crates with clean boundaries, each independently testable
- **Open solver stack**: HiGHS LP solver, no proprietary dependencies for core functionality

## Current status

All 8 implementation phases are complete. The ecosystem delivers a full SDDP training and simulation pipeline: entity model and topology validation (`cobre-core`), JSON/Parquet case loading with 5-layer validation (`cobre-io`), LP solver abstraction with HiGHS backend and warm-start basis management (`cobre-solver`), pluggable communication with MPI and local backends (`cobre-comm`), PAR(p) inflow models with deterministic correlated scenario generation and inflow non-negativity enforcement (`cobre-stochastic`), the SDDP training loop with forward/backward passes, Benders cut generation, cut synchronization, and composite stopping rules, plus the full simulation pipeline with Hive-partitioned Parquet output and FlatBuffers policy checkpointing (`cobre-sddp`). The CLI (`cobre-cli`) exposes six subcommands -- `run`, `validate`, `report`, `init`, `schema`, and `version` -- with rayon-based intra-rank thread parallelism via `--threads N`, progress bars, and a post-run summary. JSON Schema files for all input types are generated via `cobre schema export` and hosted at `https://cobre-rs.github.io/cobre/schemas/`, enabling `$schema` editor integration. Python bindings are available via `cobre-python` (PyO3). The workspace is verified by nearly 2,000 tests.

## Quick links

|                       |                                                                         |
| --------------------- | ----------------------------------------------------------------------- |
| GitHub                | [github.com/cobre-rs/cobre](https://github.com/cobre-rs/cobre)          |
| API docs (rustdoc)    | `cargo doc --workspace --no-deps --open`                                |
| Methodology reference | [cobre-rs.github.io/cobre-docs](https://cobre-rs.github.io/cobre-docs/) |
| License               | [Apache-2.0](https://github.com/cobre-rs/cobre/blob/main/LICENSE)       |
