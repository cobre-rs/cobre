# cobre-cli

`cobre-cli` provides the `cobre` binary: the command-line interface for running
SDDP studies, validating input data, and inspecting results. It ties together
`cobre-io`, `cobre-stochastic`, `cobre-solver`, `cobre-comm`, and `cobre-sddp`
into a single executable with a consistent user interface.

## Subcommands

| Subcommand                   | Description                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `cobre run <CASE_DIR>`       | Load a case, train an SDDP policy, optionally simulate, and write all results |
| `cobre validate <CASE_DIR>`  | Run the 5-layer validation pipeline and print a structured diagnostic report  |
| `cobre report <RESULTS_DIR>` | Read result manifests and print a machine-readable JSON summary to stdout     |
| `cobre version`              | Print version, solver backend, communication backend, and build information   |
| `cobre init <DIRECTORY>`     | Scaffold a new case directory from an embedded template                       |

## Exit Code Contract

All subcommands map failures to a typed exit code through the `CliError` type.
The mapping is stable across releases:

| Exit Code | Category   | Cause                                     |
| --------- | ---------- | ----------------------------------------- |
| `0`       | Success    | Command completed without errors          |
| `1`       | Validation | Case directory failed validation          |
| `2`       | I/O        | Filesystem error during loading or output |
| `3`       | Solver     | LP infeasible or numerical solver failure |
| `4`       | Internal   | Communication failure or unexpected state |

This contract enables `cobre run` to be driven from shell scripts and batch
schedulers by inspecting the process exit code.

## Output and Terminal Behavior

`cobre run` writes a progress bar to stderr and a run summary after completion
(both suppressed in `--quiet` mode). Error messages are always written to stderr.

`cobre report` prints pretty-printed JSON to stdout, suitable for piping to `jq`.

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

- Unknown template name: exits with code 1 and lists available templates.
- Target directory is non-empty and `--force` is not set: exits with code 2.
- Write failure: exits with code 2 with the failing path in the error message.

## Related Documentation

- [Installation](../guide/installation.md) — how to install the `cobre` binary
- [Running Studies](../guide/running-studies.md) — end-to-end workflow guide
- [Configuration](../guide/configuration.md) — `config.json` reference
- [CLI Reference](../guide/cli-reference.md) — complete flag and subcommand reference
- [Error Codes](../reference/error-codes.md) — validation error catalog
