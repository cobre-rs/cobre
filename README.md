<p align="center">
  <img src="assets/cobre-logo-dark.svg" width="360" alt="Cobre — Power Systems in Rust"/>
</p>

<p align="center">
  <strong>Open infrastructure for power system computation</strong>
</p>

<p align="center">
  <a href="https://github.com/cobre-rs/cobre/actions/workflows/ci.yml"><img src="https://github.com/cobre-rs/cobre/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://codecov.io/gh/cobre-rs/cobre"><img src="https://codecov.io/gh/cobre-rs/cobre/branch/main/graph/badge.svg" alt="Coverage"/></a>
  <a href="https://crates.io/crates/cobre"><img src="https://img.shields.io/crates/v/cobre.svg" alt="crates.io"/></a>
  <a href="https://docs.rs/cobre"><img src="https://docs.rs/cobre/badge.svg" alt="docs.rs"/></a>
  <a href="https://github.com/cobre-rs/cobre/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache 2.0"/></a>
</p>

---

**Cobre** is an open-source ecosystem of Rust crates for power system analysis and optimization. It provides a shared data model, file format interoperability, stochastic scenario generation, and a distributed SDDP solver for hydrothermal dispatch — with interfaces for CLI, Python, and AI agents.

The name comes from the Portuguese word for **copper** — the metal that conducts electricity.

> **Documentation:** [cobre-rs.github.io/cobre-docs](https://cobre-rs.github.io/cobre-docs/)

## Why Cobre?

Power system computation today is split between closed-source commercial tools and fragmented academic projects. Cobre aims to provide:

- **A shared data model** — the same `HydroPlant`, `Bus`, or `ThermalUnit` type works whether you're running a 10-year stochastic dispatch or inspecting results from Python. Define your system once, analyze it from multiple angles.
- **Production performance** — Rust gives C/C++-level speed with memory safety. For software that dispatches national power grids, both matter.
- **Reproducibility** — declaration-order invariance guarantees bit-for-bit identical results regardless of input entity ordering.
- **Modularity** — pick the crates you need. Use `cobre-core` for data modeling without pulling in solver dependencies. Use `cobre-sddp` without caring about interfaces.
- **Interoperability** — JSON/Parquet input, MCP server for AI agents, Python bindings for Jupyter workflows.

## Crates

| Crate                                          | Status                                                                                     | Description                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| [`cobre-core`](crates/cobre-core/)             | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | Power system entity model — buses, hydros, thermals, lines, pumping stations, contracts         |
| [`cobre-io`](crates/cobre-io/)                 | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | Input loading (JSON/Parquet), output writing (Parquet/FlatBuffers), 5-layer validation pipeline |
| [`cobre-stochastic`](crates/cobre-stochastic/) | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | PAR(p) inflow models, correlated scenario generation, opening tree for SDDP backward pass       |
| [`cobre-solver`](crates/cobre-solver/)         | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | LP solver abstraction with HiGHS and CLP backends, warm-start basis management                  |
| [`cobre-comm`](crates/cobre-comm/)             | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | Pluggable communication abstraction — MPI, TCP, shared-memory, and local backends               |
| [`cobre-sddp`](crates/cobre-sddp/)             | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | Stochastic Dual Dynamic Programming — training loop, simulation, cut management, CVaR           |
| [`cobre-cli`](crates/cobre-cli/)               | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | Command-line interface: `run`, `validate`, `report`, `compare`, `serve`, `version`              |
| [`cobre-mcp`](crates/cobre-mcp/)               | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | MCP server exposing Cobre as tools and resources for AI coding assistants                       |
| [`cobre-python`](crates/cobre-python/)         | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | PyO3 Python bindings — `import cobre` from scripts and Jupyter notebooks                        |
| [`cobre-tui`](crates/cobre-tui/)               | ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) | Terminal UI for real-time training monitoring, convergence plots, and cut inspection            |

**Related:**

| Repository                                             | Description                                                    |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| [`ferrompi`](https://github.com/cobre-rs/ferrompi)     | MPI 4.x safe Rust bindings — optional backend for `cobre-comm` |
| [`cobre-docs`](https://github.com/cobre-rs/cobre-docs) | Full specification corpus and user guide (mdBook)              |

> Status badges use the [Cobre brand palette](docs/BRAND-GUIDELINES.md): ![experimental](https://img.shields.io/badge/status-experimental-DC4C4C?style=flat-square) ![alpha](https://img.shields.io/badge/status-alpha-F5A623?style=flat-square) ![beta](https://img.shields.io/badge/status-beta-4A90B8?style=flat-square) ![stable](https://img.shields.io/badge/status-stable-4A8B6F?style=flat-square)

## Architecture

The ecosystem is organized in five layers. `cobre-core` is the shared foundation — every other crate depends on it and nothing in the lower layers knows about SDDP.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│  ┌───────────┐  ┌───────────────┐  ┌───────────┐  ┌──────────┐   │
│  │ cobre-cli │  │  cobre-python │  │ cobre-mcp │  │cobre-tui │   │
│  └─────┬─────┘  └───────┬───────┘  └─────┬─────┘  └────┬─────┘   │
├────────┼────────────────┼───────────────┼──────────────┼─────────┤
│        │        Solver  │ Vertical      │              │         │
│  ┌─────┴────────────────┴───────────────┴──────────────┘         │
│  │                      cobre-sddp                               │
│  │     Training loop · Simulation · Cut management · CVaR        │
│  └────┬────────────────────────────────┬──────────────┘          │
├───────┼────────────────────────────────┼─────────────────────────┤
│       │        Infrastructure          │                         │
│  ┌────┴──────┐  ┌───────────┐  ┌──────┴───────┐  ┌──────────┐    │
│  │cobre-     │  │cobre-     │  │cobre-        │  │cobre-    │    │
│  │stochastic │  │solver     │  │comm          │  │io        │    │
│  │PAR(p) ·   │  │HiGHS · CLP│  │MPI · TCP ·   │  │JSON ·    │    │
│  │scenarios  │  │LP/warm-   │  │shm · local   │  │Parquet · │    │
│  │           │  │start      │  │              │  │FlatBufs  │    │
│  └────┬──────┘  └─────┬─────┘  └──────┬───────┘  └────┬─────┘    │
├───────┴───────────────┴───────────────┴───────────────┴──────────┤
│                       Foundation Layer                           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                       cobre-core                         │    │
│  │  Buses · Hydros · Thermals · Lines · Pumping · Contracts │    │
│  └──────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────┤
│            Optional: ferrompi (MPI 4.x Rust bindings)            │
└──────────────────────────────────────────────────────────────────┘
```

Key design decisions:

- **`cobre-core` has zero solver dependencies** — a pure data and validation crate
- **Static dispatch** — generics over solver and communicator traits eliminate vtable overhead
- **Declaration-order invariance** — entity collections are sorted by ID; results are reproducible regardless of input file ordering
- **`cobre-mcp` and `cobre-python` are single-process only** — MPI/GIL incompatibility means distributed execution always goes through the CLI

## Quick Start

> ⚠️ Cobre is under active development. The API is not stable yet.

```bash
# Clone the workspace
git clone https://github.com/cobre-rs/cobre.git
cd cobre

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace
```

Once implemented, studies are run from a case directory:

```bash
# Validate a case
cobre validate path/to/case/

# Run training + simulation
cobre run path/to/case/ --output path/to/results/

# Query results without re-running
cobre report path/to/results/ --format json
```

A case directory follows a JSON + Parquet layout:

```
case/
├── config.json          # Algorithm configuration (solver, iterations, risk measure…)
├── stages.json          # Stage definitions and policy graph
├── system/              # Entity registries (buses.json, hydros.json, thermals.json…)
├── scenarios/           # Stochastic data (PAR coefficients, correlation — Parquet)
└── constraints/         # Initial conditions and generic constraints (JSON)
```

See the [full documentation](https://cobre-rs.github.io/cobre-docs/) for the complete input/output specification.

## Context

Cobre was born from the need for an open, modern alternative to the legacy FORTRAN-based tools used for power system planning in Brazil (NEWAVE, DECOMP, DESSEM) and alongside the commercial PSR SDDP suite. While those tools are mature and production-proven, they present challenges in auditability, extensibility, and integration with modern computational infrastructure.

The project draws inspiration from:

- **NREL Sienna** (Julia) — ecosystem architecture with shared data model
- **PowSyBl** (Java) — modular design, institutional adoption path
- **SDDP.jl** (Julia) — algorithmic reference for SDDP implementation
- **SPARHTACUS** (C++) — auditable pre-processing approach

Cobre is not a replacement for these tools — it's a new entry in the ecosystem, offering the Rust community's strengths (safety, performance, modern tooling) to a domain that can benefit from them.

## Roadmap

### Phase 0 — Foundation (current)

Specification complete. Implementation underway.

- [x] Complete SDDP specification corpus ([cobre-docs](https://github.com/cobre-rs/cobre-docs))
- [x] Workspace scaffolding (all 11 crates)
- [ ] `cobre-core` — entity model, validation, resolution of cascaded defaults
- [ ] `cobre-io` — JSON/Parquet input, FlatBuffers policy output, 5-layer validation
- [ ] `cobre-stochastic` — PAR(p) models, correlated scenario generation
- [ ] `cobre-solver` — LP abstraction, HiGHS/CLP backends, warm-start
- [ ] `cobre-sddp` — training loop, simulation, cut management, CVaR
- [ ] `cobre-cli` — run/validate/report/compare/serve subcommands
- [ ] v0.1.0 — working SDDP solver validated against reference cases

### Phase 1 — Ecosystem Hardening

- [ ] `cobre-python` — PyO3 bindings with NumPy/Arrow zero-copy paths
- [ ] `cobre-tui` — ratatui convergence monitor, co-hosted and pipe modes
- [ ] `cobre-mcp` — MCP server for AI agent integration (stdio + HTTP/SSE)
- [ ] Benchmark suite with published results
- [ ] Comparison study: Cobre vs. NEWAVE on public test cases

### Phase 2 — Power Flow

- [ ] Newton-Raphson AC power flow
- [ ] DC power flow and OPF

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

The project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(sddp): implement multi-cut strategy
fix(core): correct reservoir volume bounds validation
docs(io): document Parquet schema for hydro inflows
```

## License

Cobre is licensed under the [Apache License, Version 2.0](LICENSE).

## Citation

If you use Cobre in academic work, please cite:

```bibtex
@software{cobre,
  author = {Alves, Rogerio J. M.},
  title = {Cobre: Open Infrastructure for Power System Computation},
  url = {https://github.com/cobre-rs/cobre},
  license = {Apache-2.0}
}
```
