# Next Steps

You have completed the Cobre tutorial: you installed the tool, ran a complete
study with the 1dtoy template, inspected the case files, and interpreted the
output. This page points you to the resources that go deeper.

---

## Configuration

The `config.json` file controls every aspect of how training and simulation are
run: stopping rules, forward pass counts, simulation scenario counts, and more.
The configuration guide documents every field with examples.

- [Configuration](../guide/configuration.md) — complete `config.json` field reference

---

## System Modeling

The tutorial uses a minimal single-bus, one-hydro, two-thermal system. Real
studies model transmission networks, cascaded hydro plants, and many thermal
units. The system modeling guides explain every entity type and its parameters.

- [Hydro Plants](../guide/hydro-plants.md) — reservoir bounds, turbine models, cascade linkage
- [Thermal Units](../guide/thermal-units.md) — piecewise cost curves, generation limits
- [Network Topology](../guide/network-topology.md) — buses, transmission lines, and flow constraints

---

## CLI Reference

All subcommands (`run`, `validate`, `report`, `version`), their flags, exit
codes, and environment variables are documented in the CLI reference.

- [CLI Reference](../guide/cli-reference.md) — subcommand synopsis, options, and examples

---

## Methodology and Theory

The methodology reference describes the mathematical foundations of the solver:
the stochastic optimization formulation, the PAR(p) scenario model, the cut
management strategy, and the convergence theory. This is the right place to
start if you want to understand what the solver is doing, not just how to use it.

- [Methodology Reference](https://cobre-rs.github.io/cobre-docs/) — full theory documentation (external)

---

## API Documentation

The Rust API for all workspace crates is generated from inline doc comments.
To build and open it locally:

```bash
cargo doc --workspace --no-deps --open
```

This opens the documentation for all 10 workspace crates in your browser. The
`cobre-core` crate documents the entity model; `cobre-sddp` documents the
training loop and cut management types.

---

## Contributing

If you find a bug, want to add a feature, or want to improve the documentation,
the contributing guide explains how to set up the development environment, run
the test suite, and submit a pull request.

- [Contributing](../contributing.md) — development setup, branch conventions, and PR process

---

## Community and Support

- [GitHub Repository](https://github.com/cobre-rs/cobre) — source code, releases, and project board
- [Issue Tracker](https://github.com/cobre-rs/cobre/issues) — bug reports and feature requests
- [Discussions](https://github.com/cobre-rs/cobre/discussions) — questions, ideas, and community discussion
