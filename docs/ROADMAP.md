# Cobre -- Post-v0.1.0 Roadmap

This document tracks implementation-level work items planned for releases after
v0.1.0. It covers deferred features, HPC optimizations, and post-MVP crates.
For methodology-level roadmap pages (algorithm theory, spec evolution), see the
[cobre-docs roadmap](https://cobre-rs.github.io/cobre-docs/roadmap/).

---

## Inflow Truncation Methods

**Status**: Deferred from v0.1.0. Only the penalty method is available.

PAR(p) models can produce negative inflow realisations. The penalty method
(implemented in v0.1.0) handles this by adding a high-cost slack variable to
the water balance row. Two additional methods from the literature are planned:

- **Truncation** -- evaluate the AR model externally before LP patching; if
  the full inflow value `a_h < 0`, adjust the LP row bounds to force inflow to
  zero. Requires threading AR coefficients from `StochasticContext` through to
  the forward pass.
- **Truncation with penalty** -- combine modified bounds with a bounded slack
  variable, matching the SPTcpp reference implementation most closely.

The `InflowNonNegativityMethod` enum will gain `Truncation` and
`TruncationWithPenalty { cost }` variants when these are implemented. Existing
`"penalty"` and `"none"` configs will remain unchanged.

**Full design**: `docs/deferred-truncation-design.md`
**Paper reference**: Oliveira et al. (2022), _Energies_ 15(3):1115.

---

## HPC Optimizations

**Status**: Inter-rank (MPI) parallelism is implemented. Intra-rank thread
parallelism is the primary v0.1.x target (see `docs/PROJECT-STATUS.md` for
the blocking gap description).

Beyond the baseline thread parallelism, the following optimizations are
planned for later releases:

- **NUMA-aware allocation** -- pin thread pools and memory arenas to NUMA
  nodes to avoid cross-socket memory traffic on multi-socket servers.
- **Work-stealing tuning** -- profile rayon task granularity against real
  case sizes; consider hybrid chunking for uneven scenario distributions.
- **MPI + threads interaction** -- validate `MPI_THREAD_MULTIPLE` behaviour
  across OpenMPI and MPICH with the `FerrompiBackend`; document the minimum
  thread support level required.

**Spec references**: `cobre-docs/src/specs/hpc/parallelism-model.md`,
`cobre-docs/src/specs/hpc/memory-model.md`.

---

## Post-MVP Crates

Three crates are stubbed in the workspace but not yet implemented.

### `cobre-mcp`

A standalone MCP (Model Context Protocol) server that exposes Cobre case
management and result queries to AI coding assistants. Runs as a
single-process binary; does not use MPI. Depends only on `cobre-core` and
`cobre-io`.

### `cobre-python`

PyO3-based Python bindings compiled as a `cdylib`. Will expose case loading,
validation, training, and simulation to Python callers, enabling integration
with data science workflows and Jupyter notebooks. Single-process; no MPI.

### `cobre-tui`

A `ratatui`-based terminal UI library consumed by `cobre-cli`. Will provide
an interactive dashboard showing live convergence plots, scenario statistics,
and solver timing during a run. Depends only on `cobre-core`; the rendering
loop reads events published by `cobre-sddp`.

---

## Algorithm Extensions

The v0.1.0 solver uses the minimal variant set (Expectation risk measure,
Level-1 cut selection, InSample sampling, finite horizon). Future algorithm
extensions include:

- **CVaR risk measure** -- Conditional Value-at-Risk for risk-averse dispatch
  policies, as described in the cobre-docs methodology reference.
- **Multi-cut variant** -- one cut per scenario per stage rather than an
  averaged cut, for faster convergence on small scenario counts.
- **Infinite horizon** -- cyclic stage chains with a terminal value function,
  enabling long-run equilibrium studies.

**Methodology reference**: `cobre-docs/src/roadmap/` for the full algorithm
extension roadmap and theoretical grounding.
