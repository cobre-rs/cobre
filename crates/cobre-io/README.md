# cobre-io

Case directory loading, validation, and result writing for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

This crate provides two top-level entry points for all I/O in the Cobre ecosystem.
`load_case` reads a case directory of JSON and Parquet files, executes a five-layer
validation pipeline (structural, schema, referential integrity, dimensional
consistency, and semantic), and produces a fully-validated `System` ready for the
solver. `write_results` accepts aggregate result types and writes all output
artifacts — Parquet tables, FlatBuffers policy checkpoints, and JSON manifests —
to a specified root directory.

## When to Use

Depend on `cobre-io` when you need to load a case directory from disk or write
solver outputs to a result directory. If you are building a new subcommand or
integration that reads case files and hands a `System` to an algorithm, this
crate is the boundary between the filesystem and `cobre-core` types. Do not
depend on it from pure algorithm crates — pass the `System` value instead.

## Key Types

- **`load_case`** — Reads and validates a case directory, returning a `System` or a `LoadError`
- **`write_results`** — Writes all output artifacts (Parquet, FlatBuffers, JSON) to a result directory
- **`Config`** — Deserialized run configuration loaded from `config.json` in the case directory
- **`LoadError`** — Typed error enum covering I/O, parse, schema, and constraint failures
- **`ValidationContext`** — Collects all validation diagnostics across all pipeline layers before failing

## Links

| Resource   | URL                                                      |
| ---------- | -------------------------------------------------------- |
| Book       | https://cobre-rs.github.io/cobre/crates/io.html          |
| API Docs   | https://docs.rs/cobre-io/latest/cobre_io/                |
| Repository | https://github.com/cobre-rs/cobre                        |
| CHANGELOG  | https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md |

## Status

Alpha — API is functional but not yet stable.

## License

Apache-2.0
