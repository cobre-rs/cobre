# Getting Started

The User Guide is the reference path through Cobre. It documents all
configuration options, CLI flags, input schemas, and output formats in
detail — intended as a companion to working with real case directories
rather than as an initial walkthrough.

> **New to Cobre?** Start with the [Tutorial](../tutorial/installation.md)
> instead. The Tutorial walks you through installing Cobre, scaffolding a
> complete example study from a built-in template, and interpreting the
> results — all in a few minutes.

---

## Guide Contents

- [Installation](./installation.md) — pre-built binary, `cargo install`,
  and build-from-source instructions for all supported platforms
- [System Modeling](./system-modeling.md) — how to define buses, hydro
  plants, thermal units, transmission lines, and stochastic inflow models
  in the input JSON files
- [Running Studies](./running-studies.md) — the full validate → run →
  report workflow, output directory layout, and lifecycle exit codes
- [Configuration](./configuration.md) — every `config.json` field:
  stopping rules, forward pass count, seed, and simulation settings
- [CLI Reference](./cli-reference.md) — all subcommands, flags, and
  environment variables
