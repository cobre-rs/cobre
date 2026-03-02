# Crate Overview

Cobre is organized as a Rust workspace with 11 crates. Each crate has a single responsibility and well-defined boundaries.

```
cobre/crates/
├── cobre-core/         Entity model (buses, hydros, thermals, lines)
├── cobre-io/           JSON/Parquet input, FlatBuffers/Parquet output
├── cobre-stochastic/   PAR(p) models, scenario generation
├── cobre-solver/       LP solver abstraction (HiGHS, CLP backends)
├── cobre-comm/         Communication abstraction (MPI, TCP, shm, local)
├── cobre-sddp/         SDDP training loop, simulation, cut management
├── cobre-cli/          Binary: run/validate/report/compare/serve
├── cobre-mcp/          Binary: MCP server for AI agent integration
├── cobre-python/       cdylib: PyO3 Python bindings
└── cobre-tui/          Library: ratatui terminal UI
```

For the full dependency graph and crate responsibilities, see the [methodology reference](https://cobre-rs.github.io/cobre-docs/specs/overview/implementation-ordering.html).

Individual crate pages below will be filled as each crate is implemented.
