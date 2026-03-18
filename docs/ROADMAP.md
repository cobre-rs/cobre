# Cobre -- Roadmap

This document tracks implementation-level work items planned for future releases.
For the full feature roadmap with motivation, prerequisites, and planned approach,
see the [methodology roadmap](https://cobre-rs.github.io/cobre-docs/roadmap/overview.html).

---

## Upcoming Features

### Solver Enhancements

- **Multi-cut formulation** (C.3) — One future-cost variable per forward scenario
  instead of the current single-cut (expected value) formulation. Improves lower
  bound convergence for problems with high-variance costs.

- **CVaR risk measure** — Risk-averse dispatch using Conditional Value-at-Risk.
  The `RiskMeasure` enum already has the variant stub; the LP modification and
  dual extraction need implementation.

### Equipment & Modeling

- **GNL thermal plants** (C.1) — Thermal units with gas supply contracts, take-or-pay
  clauses, and fuel substitution logic.

- **Battery energy storage** (C.2) — Charge/discharge variables with round-trip
  efficiency, degradation modeling, and state-of-charge tracking across stages.

- **FPHA enhancements** (C.6) — Per-unit efficiency curves, multiple generating units
  per plant, and forebay-tailrace coupling for cascaded reservoirs.

### Stochastic Enhancements

- **CEPEL PAR(p)-A variant** (C.8) — Modified PAR model with seasonal autoregressive
  structure used in the Brazilian power system.

- **Fine-grained temporal resolution** (C.10) — Sub-monthly blocks (weekly, hourly)
  with block-dependent load profiles and generation constraints.

### Post-MVP Crates

- **`cobre-mcp`** — A standalone MCP (Model Context Protocol) server that exposes
  Cobre case management and result queries to AI coding assistants. Runs as a
  single-process binary; does not use MPI.

- **`cobre-tui`** — A `ratatui`-based terminal UI library consumed by `cobre-cli`.
  Interactive dashboard with live convergence plots, scenario statistics, and solver
  timing. Depends only on `cobre-core`.

---

## HPC Optimizations

### Near-term

- **NUMA-aware thread/memory placement** — Pin rayon thread pools to NUMA domains;
  allocate per-thread workspaces from local memory. Expected 20--40% improvement on
  multi-socket servers.

- **Memory pool / arena allocation** — Replace per-iteration heap allocation with
  thread-local arenas to eliminate allocator contention at high thread counts.

- **Profile-guided optimization (PGO) builds** — Two-stage compiler build using
  representative training profiles. Expected 5--15% "free" throughput improvement.

### Longer-term

- **SIMD vectorization for cut evaluation** — Explicit SIMD for the backward-pass
  inner product loop. Natural target when cut pools reach thousands of entries.

- **Asynchronous cut exchange** — Overlap `allgatherv` communication with LP solves
  using non-blocking MPI collectives.

- **GPU acceleration** — GPU solver backends for large network topologies (> 500 buses).

---

**Methodology reference**: [cobre-rs.github.io/cobre-docs/roadmap/](https://cobre-rs.github.io/cobre-docs/roadmap/overview.html)
