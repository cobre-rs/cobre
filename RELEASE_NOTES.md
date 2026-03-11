# v0.1.0 — Initial Release

The first public release of the Cobre SDDP solver ecosystem.

## Highlights

Cobre v0.1.0 delivers a complete Stochastic Dual Dynamic Programming (SDDP) solver for hydrothermal dispatch, built in Rust across 8 workspace crates. All 8 implementation phases are complete with 1,982 passing tests.

### SDDP Training and Simulation

- Full SDDP training loop with forward/backward pass, Benders cut management, and convergence monitoring
- Post-training Monte Carlo simulation with configurable scenario counts
- Enhanced simulation progress bar showing real-time mean, standard deviation, and 95% confidence interval statistics
- FlatBuffers policy checkpoint for cut persistence
- Hive-partitioned Parquet output for costs, hydro dispatch, thermal dispatch, and bus balance

### CLI Subcommands

Six CLI subcommands for the full workflow:

- `cobre init` — scaffold a case directory from built-in templates
- `cobre run` — execute training and simulation with progress bars and post-run summary
- `cobre validate` — run the 5-layer validation pipeline with diagnostic output
- `cobre report` — query results from a completed run (JSON to stdout)
- `cobre schema` — export JSON Schema definitions for all input file types
- `cobre version` — print version, solver backend, and build information

### JSON Schema Generation

- `schemars`-derived JSON Schema for all input types (config, stages, penalties, buses, hydros, thermals, lines, energy contracts, pumping stations, non-controllable sources)
- `$schema` URLs in all input files for IDE validation and autocompletion
- Schema files hosted in the software book at `reference/schemas/`

### Performance

- Intra-rank thread parallelism via Rayon (`--threads N` or `COBRE_THREADS`)
- MPI distributed computing via ferrompi (`mpiexec -n N cobre run`)
- LP solver abstraction with HiGHS backend and warm-start basis management
- Zero allocation on hot paths in the training loop

### Data Model

- 7 system element types: buses, hydro plants, thermal units, transmission lines, energy contracts, pumping stations, non-controllable sources
- PAR(p) stochastic inflow models with Cholesky-correlated scenario generation
- Declaration-order invariance: bit-for-bit identical results regardless of input entity ordering
- 5-layer validation pipeline: schema, referential integrity, physical feasibility, stochastic consistency, solver feasibility

## Installation

```bash
cargo install cobre-cli
```

Requires Rust 1.85+ and HiGHS.

## What's Next

- v0.1.1: `cobre summary` subcommand
- v0.2.0: Python bindings (PyO3), import tool for legacy formats
