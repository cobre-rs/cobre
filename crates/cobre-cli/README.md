# cobre-cli

Command-line interface for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

Provides five subcommands for running SDDP studies, scaffolding case
directories, validating input data, and inspecting build
information from the terminal.

## When to Use

Use `cobre-cli` when you want to run a complete SDDP study — training and
simulation — from the command line without writing Rust or Python code. For
programmatic embedding of the solver, depend on `cobre-sddp` directly.

## Key Subcommands

- **`init`** — scaffold a new case directory from an embedded template
- **`run`** — load a case directory, train an SDDP policy, and run simulation
- **`validate`** — validate a case directory and print a structured diagnostic report
- **`schema`** — manage JSON Schema files for case directory input types
- **`version`** — print version, solver backend, and build information

## Exit Code Contract

All subcommands map failures to a typed exit code through the `CliError` type
(`src/error.rs`). The mapping is stable across releases:

| Exit Code | Variant      | Cause                                                                                                                                                                            |
| --------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`       | Success      | Command completed without errors                                                                                                                                                 |
| `1`       | `Validation` | Case directory failed the validation pipeline                                                                                                                                    |
| `2`       | `Io`         | Filesystem error during loading or output                                                                                                                                        |
| `3`       | `Solver`     | LP infeasible or numerical solver failure during training/simulation                                                                                                             |
| `4`       | `Internal`   | Communication failure or unexpected state                                                                                                                                        |
| `5`       | —            | `cobre run` only: a SIGTERM or SIGINT stopped training at an iteration boundary; the training outputs and policy checkpoint were written and a configured simulation was skipped |

Exit code `5` is not a failure and has no `CliError` variant. Under MPI every
rank exits 5, so the launcher reports 5. When writing the training outputs or
the skipped-simulation files fails, the run exits with that failure's code
instead. A signal received before training starts, or during a simulation,
terminates the process by that signal, and so does a second SIGINT in a
single-process run. Shells report 128 plus the signal number: 130 for SIGINT,
143 for SIGTERM. Under MPI a repeated SIGINT only repeats the stop request, and
aborting is left to the launcher.

This contract enables `cobre run` to be driven from shell scripts and batch
schedulers by inspecting the process exit code. `CliError` also carries the
`From` conversions that route every upstream error type (`cobre_io::LoadError`,
`cobre_io::OutputError`, `cobre_comm::BackendError`, `cobre_sddp::SddpError`,
`cobre_sddp::SimulationError`) onto one of these four variants.

## Output and Terminal Behavior

- **`cobre run`** writes a live progress bar to stderr and a run summary after
  completion (both suppressed in `--quiet` mode). Error messages are always
  written to stderr.
- **`cobre validate --json`** prints a single JSON object to stdout — stdout is
  reserved for machine-readable output.

## `cobre init`

Scaffolds a new case directory from a built-in template. This is the recommended
way to start a new study: the template provides a complete, valid case that passes
`cobre validate` out of the box and can be run immediately with `cobre run`.

### Arguments

| Argument      | Required              | Description                                   |
| ------------- | --------------------- | --------------------------------------------- |
| `<DIRECTORY>` | Yes (unless `--list`) | Path where the case directory will be created |

### Options

| Option              | Description                                                                  |
| ------------------- | ---------------------------------------------------------------------------- |
| `--template <NAME>` | Template name to scaffold. Required unless `--list` is given.                |
| `--list`            | List all available templates and exit. Mutually exclusive with `--template`. |
| `--force`           | Overwrite existing files in the target directory if it is non-empty.         |

### Available Templates

| Template | Description                                                         |
| -------- | ------------------------------------------------------------------- |
| `1dtoy`  | Single-bus hydrothermal system: 4 stages, 1 hydro plant, 2 thermals |

Templates are embedded at compile time (`src/templates.rs`, via
`include_bytes!`) from `examples/1dtoy/` at the workspace root, copied into
`OUT_DIR` by `build.rs` so the registry works under `cargo publish`.

### Usage Examples

```bash
# List all available templates
cobre init --list

# Scaffold the 1dtoy template into a new directory
cobre init --template 1dtoy my_study

# Overwrite an existing directory
cobre init --template 1dtoy my_study --force
```

After scaffolding, validate and run the case:

```bash
cobre validate my_study
cobre run my_study --output my_study/results
```

### Error Behavior

- Unknown template name: exits with code 1 (`CliError::Validation`) and lists
  available templates.
- Target directory is non-empty and `--force` is not set: exits with code 2
  (`CliError::Io`), with a hint to use `--force`.
- Write failure: exits with code 2 with the failing path in the error message.

## Links

| Resource             | URL                                                         |
| -------------------- | ----------------------------------------------------------- |
| CLI reference        | <https://docs.cobre-rs.dev/reference/cli-reference/>       |
| Repository           | <https://github.com/cobre-rs/cobre>                         |
| Changelog            | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md>  |

## Status

**Alpha** — API is functional but not yet stable. See the [main repository](https://github.com/cobre-rs/cobre) for the current release.

## License

Apache-2.0
