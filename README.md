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
  <a href="https://pypi.org/project/cobre-python/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/cobre-python"></a>
  <a href="https://pypi.org/project/cobre-python/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/cobre-python"></a>
  <a href="https://docs.rs/cobre-sddp"><img src="https://docs.rs/cobre-sddp/badge.svg" alt="docs.rs"/></a>
  <a href="https://github.com/cobre-rs/cobre/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache 2.0"/></a>
  <a href="https://doi.org/10.5281/zenodo.23202161"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23202161.svg" alt="DOI"></a>
</p>

---

**Cobre** is a Rust ecosystem for power system optimization. It ships a distributed SDDP solver for hydrothermal dispatch with a CLI and Python bindings. The name comes from the Portuguese word for **copper**.

## Why Cobre?

- **Production performance** -- Rust gives C/C++-level speed with memory safety. For software that dispatches national power grids, both matter.
- **Reproducibility** -- Declaration-order invariance guarantees bit-for-bit identical results regardless of input entity ordering.
- **Modularity** -- Pick the crates you need. Use `cobre-core` for data modeling alone, or `cobre-sddp` for the full solver.
- **Interoperability** -- JSON/Parquet input, Python bindings for Jupyter workflows, MCP server for AI agents (reserved; not yet implemented).

## Install

```bash
# Rust CLI (requires Rust 1.88+ and HiGHS)
cargo install cobre-cli

# Python bindings (3.12 / 3.13 / 3.14)
pip install cobre-python
```

## Quick Links

| Resource      | Link                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| Documentation | [docs.cobre-rs.dev](https://docs.cobre-rs.dev/)                         |
| API Docs      | [docs.rs/cobre-sddp](https://docs.rs/cobre-sddp)                        |
| PyPI          | [pypi.org/project/cobre-python](https://pypi.org/project/cobre-python/) |

## Getting Started

- **Coming from other software?** -- See the [cobre-bridge guide](https://docs.cobre-rs.dev/guide/cobre-bridge.html)
- **New to SDDP?** -- Read [What Cobre Solves](https://docs.cobre-rs.dev/tutorial/what-cobre-solves.html)
- **Python user?** -- Try the [Python Quickstart](https://docs.cobre-rs.dev/guide/python-quickstart.html)

## Current Status

Cobre is alpha software with a fully functional SDDP solver. The pipeline covers case loading, stochastic scenario generation, training, simulation, policy checkpointing, and output writing. See the [CHANGELOG](CHANGELOG.md) for release history.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under [Apache-2.0](LICENSE).

```bibtex
@software{cobre,
  author = {Alves, Rogerio J. M.},
  title = {Cobre: Open Infrastructure for Power System Computation},
  url = {https://github.com/cobre-rs/cobre},
  license = {Apache-2.0}
}
```
