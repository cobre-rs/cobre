# cobre-cli

Command-line interface for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

Provides seven subcommands for running SDDP studies, scaffolding case
directories, validating input data, querying results, and inspecting build
information from the terminal.

## When to Use

Use `cobre-cli` when you want to run a complete SDDP study — training and
simulation — from the command line without writing Rust or Python code. For
programmatic embedding of the solver, depend on `cobre-sddp` directly.

## Key Subcommands

- **`init`** — scaffold a new case directory from an embedded template
- **`run`** — load a case directory, train an SDDP policy, and run simulation
- **`validate`** — validate a case directory and print a structured diagnostic report
- **`report`** — query results from a completed run and print them to stdout
- **`summary`** — display the post-run summary from a completed output directory
- **`schema`** — manage JSON Schema files for case directory input types
- **`version`** — print version, solver backend, and build information

## Links

| Resource             | URL                                                         |
| -------------------- | ----------------------------------------------------------- |
| Book — CLI reference | <https://cobre-rs.github.io/cobre/guide/cli-reference.html> |
| Repository           | <https://github.com/cobre-rs/cobre>                         |
| Changelog            | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md>  |

## Status

**Alpha** — API is functional but not yet stable. See the [main repository](https://github.com/cobre-rs/cobre) for the current release.

## License

Apache-2.0
