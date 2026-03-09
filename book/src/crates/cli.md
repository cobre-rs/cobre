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

## Related Documentation

- [Installation](../guide/installation.md) — how to install the `cobre` binary
- [Running Studies](../guide/running-studies.md) — end-to-end workflow guide
- [Configuration](../guide/configuration.md) — `config.json` reference
- [CLI Reference](../guide/cli-reference.md) — complete flag and subcommand reference
- [Error Codes](../reference/error-codes.md) — validation error catalog
