# Cobre

**Open infrastructure for power system computation. Built in Rust.**

Cobre solves long-term hydrothermal dispatch -- the problem of scheduling water
and fuel across power grids with large hydroelectric capacity. It provides an
open-source, reproducible alternative built on modern infrastructure: Rust for
performance and safety, Parquet for data interchange, and Python for analysis
workflows.

## Choose Your Path

> **Coming from other energy optimization software?**
> If you already work with hydrothermal dispatch tools and want to convert
> existing case data, see the [cobre-bridge](./guide/cobre-bridge.md) conversion guide.

> **New to SDDP?**
> If you want to understand the algorithm before diving into code, read
> [What Cobre Solves](./tutorial/what-cobre-solves.md).

> **Python user?**
> If you want to run studies from Jupyter or a Python script, see the
> [Python Quickstart](./guide/python-quickstart.md).

## What Cobre Does

- **Solve long-term hydrothermal dispatch** via Stochastic Dual Dynamic
  Programming (SDDP), with training, simulation, and policy export.
- **Model complex power systems** -- hydro cascades with variable-head
  production, thermal units, transmission networks, non-controllable sources,
  and user-defined generic constraints.
- **Generate stochastic scenarios** using periodic autoregressive (PAR) inflow
  models with correlated multi-site noise.
- **Scale across clusters** with hybrid MPI + thread parallelism, producing
  bit-for-bit identical results regardless of rank or thread count.
- **Analyze results from Python** using Arrow zero-copy bindings, or directly
  from Parquet output files.

## Quick Links

|                       |                                                                         |
| --------------------- | ----------------------------------------------------------------------- |
| GitHub                | [github.com/cobre-rs/cobre](https://github.com/cobre-rs/cobre)          |
| Software Book         | You are here                                                            |
| API Docs              | [docs.rs/cobre](https://docs.rs/cobre)                                  |
| PyPI                  | [pypi.org/project/cobre-python](https://pypi.org/project/cobre-python/) |
| Methodology Reference | [cobre-rs.github.io/cobre-docs](https://cobre-rs.github.io/cobre-docs/) |
| License               | [Apache-2.0](https://github.com/cobre-rs/cobre/blob/main/LICENSE)       |
