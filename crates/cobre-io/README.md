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

## Validation pipeline

`load_case` runs five layers in sequence; earlier layers gate later ones (a file
missing in Layer 1 is never parsed in Layer 2), and every layer collects all of
its diagnostics into a shared `ValidationContext` before the pipeline decides
whether to fail — a `ConstraintError` reports every problem found in one pass,
not just the first.

1. **Structural** — do the required files exist on disk? Missing required files
   fail; missing optional files are only noted in the file manifest.
2. **Schema** — parse every present file; check required fields, types, and
   value ranges. JSON parsing uses `#[serde(deny_unknown_fields)]`, so both
   missing fields and unrecognized keys surface as hard errors.
3. **Referential integrity** — every cross-entity ID reference (`bus_id`,
   `source_hydro_id`, bound/penalty override rows, etc.) must resolve to a
   known entity.
4. **Dimensional consistency** — optional per-entity files must cover every
   entity that needs them (e.g. inflow statistics must exist for every hydro;
   load seasonal statistics must cover every bus for every stage).
5. **Semantic** — domain business rules: acyclic hydro cascade, penalty
   ordering (lower tiers may not exceed upper), PAR model stationarity, stage
   count consistency, and estimation prerequisites (see below).

After all five layers pass, `load_case` resolves the three-tier penalty/bound
cascade, assembles the scenario models (running the estimation pipeline first
when `inflow_history.parquet` is present without `inflow_seasonal_stats.parquet`),
and calls `SystemBuilder::build()` to construct the immutable `System`.

## Error handling (`LoadError`)

| Variant              | Fields                                                                              | Pipeline phase                                                                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `IoError`            | `path`, `source: std::io::Error`                                                    | Layer 1/2 — file exists in the manifest but cannot be read from disk                                                                                                                    |
| `ParseError`         | `path`, `message`                                                                   | Layer 2 — file is readable but malformed (invalid JSON/Parquet)                                                                                                                         |
| `SchemaError`        | `path`, `field` (dot-separated, e.g. `"hydros[3].bus_id"`), `message`               | Layer 2 — required field missing or a value violates a schema constraint; also returned by `parse_config` when `training.selection` or `training.stopping_rules` is absent              |
| `ConstraintError`    | `description` (all collected messages, newline-joined, each `[ErrorKind]`-prefixed) | Layers 4/5, or a final `SystemBuilder::build()` rejection (duplicate IDs, cascade cycle)                                                                                                |
| `PolicyIncompatible` | `check`, `policy_value`, `system_value`                                             | After all layers pass, when `policy.mode` is `warm_start`/`resume` and the stored policy fails a compatibility check (hydro count, stage count, cut dimension, or entity identity hash) |

`LoadError::io(path, source)` is the constructor to use instead of a `From<std::io::Error>`
impl — the latter would lose the path context every diagnostic needs.

## `Config` struct (`config.json`)

`Config` (`src/config/mod.rs`) has seven sections, all but `training` defaulted:

| Section                  | Type                         | Default    | Purpose                                                |
| ------------------------ | ---------------------------- | ---------- | ------------------------------------------------------ |
| `modeling`               | `ModelingConfig`             | `{}`       | Inflow non-negativity treatment method and cost        |
| `training`               | `TrainingConfig`             | (required) | Iteration count, stopping rules, cut selection         |
| `upper_bound_evaluation` | `UpperBoundEvaluationConfig` | `{}`       | Inner-approximation upper-bound evaluation settings    |
| `policy`                 | `PolicyConfig`               | fresh mode | Policy directory path, warm-start / resume mode        |
| `simulation`             | `SimulationConfig`           | disabled   | Post-training simulation scenario count and output     |
| `exports`                | `ExportsConfig`              | all on     | Flags controlling which output files are written       |
| `estimation`             | `EstimationConfig`           | `{}`       | AR model fitting settings for history-based estimation |

`training.selection` and `training.stopping_rules` (must include at least one
`iteration_limit` rule) have no defaults; `parse_config` returns
`LoadError::SchemaError` if either is absent or `null`, or if
`training.stopping_rules` has no `iteration_limit` rule.

`training.stopping_rules` accepts four internally-tagged (`"type"`) rule
variants — `iteration_limit { limit }`, `time_limit { seconds }`,
`bound_stalling { iterations, tolerance }`, and
`gap { tolerance, relative_tolerance }` (parses but is rejected at load;
evaluation is not yet wired) — combined via `training.stopping_mode`: `"any"`
(default, OR) or `"all"` (AND).

`policy.mode` is one of `PolicyMode::Fresh` (default, start from scratch),
`WarmStart` (load existing cuts/states from `policy.path`), or `Resume`
(continue an interrupted run from the last checkpoint); the latter two trigger
the `PolicyIncompatible` compatibility checks above.

## Three-tier penalty/bound resolution

Penalty and bound values follow global → entity → stage precedence. Tiers 1
and 2 (`penalties.json` and per-entity JSON fields) resolve during Layer-2
parsing, so each entity struct already holds its tier-2 value. The dedicated
resolution step (after all validation layers pass) applies the sparse tier-3
`constraints/penalty_overrides_*.parquet` / `constraints/*_bounds.parquet`
overrides — a Parquet row only needs to exist for stages where the value
differs from tier 2 — and expands the result into dense
`[n_entities × n_stages]` `ResolvedPenalties` / `ResolvedBounds` arrays on
`System` for O(1), branch-free solver lookup.

## Estimation pipeline

When `scenarios/inflow_history.parquet` is present and
`scenarios/inflow_seasonal_stats.parquet` is **absent**, `load_case` derives
seasonal statistics and AR coefficients from the historical series instead of
reading pre-computed Parquet files. `config.estimation` controls the fit:

| Field                         | Type                        | Default  | Description                                                               |
| ----------------------------- | --------------------------- | -------- | ------------------------------------------------------------------------- |
| `max_order`                   | `u32`                       | `6`      | Maximum AR lag order considered during model selection                    |
| `order_selection`             | `"pacf"` \| `"pacf_annual"` | `"pacf"` | PACF significance testing, optionally with an annual component (PAR(p)-A) |
| `min_observations_per_season` | `u32`                       | `30`     | Minimum observations required per `(entity, season)` group                |

The estimation path additionally requires `season_definitions` in
`stages.json` (to group observations by season) and at least one history
observation per hydro plant; groups below `min_observations_per_season`
produce a `ModelQuality` warning rather than a hard failure. When explicit
stats files are provided instead, `inflow_history.parquet` (if present) is
still loaded and carried on the loaded system but does not influence model
assembly.

## Links

| Resource   | URL                                                      |
| ---------- | -------------------------------------------------------- |
| Docs site  | <https://docs.cobre-rs.dev/>                             |
| API Docs   | https://docs.rs/cobre-io/latest/cobre_io/                |
| Repository | https://github.com/cobre-rs/cobre                        |
| CHANGELOG  | https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md |

## Status

Alpha — API is functional but not yet stable.

## License

Apache-2.0
