# ADR-009: Stochastic Artifact Export

**Status:** Accepted
**Date:** 2026-03-13
**Spec reference:** N/A

## Context

The stochastic estimation pipeline in `cobre-stochastic` is a black box. When a user
supplies only `inflow_history.parquet`, Cobre runs `build_stochastic_context()`, which
executes five preprocessing steps: PAR order validation, PAR cache construction
(Levinson-Durbin with AIC-based order selection), Cholesky correlation decomposition,
opening tree generation, and normal cache construction. The resulting `StochasticContext`
owns all fitted artifacts in memory, but none of them are persisted to disk.

This opacity blocks three common workflows:

1. **Verification**: users cannot compare Cobre's fitted PAR parameters against an
   external reference implementation or a prior software version.
2. **Reuse**: users must re-run the full estimation pipeline on every invocation even
   when the input history has not changed. For large systems this estimation cost is
   non-trivial.
3. **Audit**: regulatory workflows often require a durable record of the fitted
   stochastic model used in a study. An in-memory artifact that evaporates after the
   run cannot satisfy this requirement.

A secondary motivation is the round-trip invariant introduced alongside user-supplied
opening trees (ADR-008): if exported files use the same schema as the corresponding
input files, the user can copy `output/stochastic/` to `scenarios/` and the next
invocation will load the artifacts directly, bypassing estimation entirely.

## Decision

Export stochastic artifacts to `output/stochastic/` when the `--export-stochastic`
flag is passed to the `cobre run` subcommand. Export is off by default in v0.1.x and
on by default starting in v0.2.

### Exported files

| File path                                          | Schema source                                            | Export condition         |
| -------------------------------------------------- | -------------------------------------------------------- | ------------------------ |
| `output/stochastic/inflow_seasonal_stats.parquet`  | Same as input `scenarios/inflow_seasonal_stats.parquet`  | Estimation was performed |
| `output/stochastic/inflow_ar_coefficients.parquet` | Same as input `scenarios/inflow_ar_coefficients.parquet` | Estimation was performed |
| `output/stochastic/correlation.json`               | Same as input `scenarios/correlation.json`               | Always                   |
| `output/stochastic/fitting_report.json`            | JSON diagnostic report (see below)                       | Estimation was performed |
| `output/stochastic/noise_openings.parquet`         | Same schema as defined in ADR-008                        | Always                   |
| `output/stochastic/load_seasonal_stats.parquet`    | Same as input `scenarios/load_seasonal_stats.parquet`    | Load buses exist         |

"Estimation was performed" means the user did not supply the corresponding scenario
file directly; Cobre derived it from `inflow_history.parquet`.

### Round-trip invariant

Every exported Parquet or JSON file uses the exact same column names, types, and
partition layout as the corresponding input file. Copying all files from
`output/stochastic/` to `scenarios/` and re-running Cobre causes the loader in
`cobre-io` to find the files already present and skip estimation. No schema translation
or file renaming is required.

### Fitting report format

`fitting_report.json` is a JSON object with a top-level key `"hydros"` mapping each
hydro plant ID to a diagnostics record:

```json
{
  "hydros": {
    "<hydro_id>": {
      "selected_order": 3,
      "aic_scores": [12.4, 11.1, 10.8, 11.3],
      "coefficients": [[0.42, -0.11, 0.07]]
    }
  }
}
```

`aic_scores[i]` is the AIC for order `i+1`. `coefficients` is a nested array with one
row per season, matching the layout of `inflow_ar_coefficients.parquet`. This file is
diagnostic only; it is not consumed as input on subsequent runs.

### Cholesky factor handling

The raw correlation matrix (same format as input `scenarios/correlation.json`) is
exported, not the Cholesky factor stored in `StochasticContext`. The Cholesky
decomposition is an internal implementation detail. On the next run, `cobre-io`
loads the raw matrix and `build_stochastic_context` re-decomposes it.

### Timing and synchronization

Export happens after `build_stochastic_context()` returns and before `train()` is
called. The export is synchronous; it blocks the start of training. This is acceptable
because stochastic context construction is already a one-time cost measured in seconds,
and export adds only I/O time to a step that already dominates the pre-training phase.

In MPI runs, rank 0 performs all file writes. All other ranks wait at a barrier after
rank 0 signals completion. This mirrors the pattern used for all other output writes in
`cobre-cli`.

### CLI surface

The flag is added to the `run` subcommand:

```
cobre run --export-stochastic <case_dir>
```

No short form is assigned. The flag is boolean (presence enables export). The default
changes from `false` (v0.1.x) to `true` (v0.2+).

## Consequences

### Benefits

- **Auditability**: fitted PAR parameters and the opening tree are persisted as
  durable files alongside the run output, satisfying regulatory record-keeping
  requirements.
- **Inspectability**: users can open `inflow_ar_coefficients.parquet` and
  `inflow_seasonal_stats.parquet` in any Parquet-aware tool to verify the fitted
  model without writing custom extraction code.
- **Reproducibility**: re-running Cobre with the exported files as inputs produces
  bit-for-bit identical stochastic artifacts, because the round-trip invariant
  eliminates the estimation step.
- **Performance on repeated runs**: users who copy exported artifacts back to
  `scenarios/` skip Levinson-Durbin fitting and Cholesky decomposition on subsequent
  runs of the same case.

### Costs and trade-offs

- **Output directory size**: a full stochastic export adds several Parquet files and
  two JSON files per run. For large systems with many hydro plants and long histories,
  `inflow_seasonal_stats.parquet` and `inflow_ar_coefficients.parquet` can be
  multi-megabyte.
- **Schema coupling**: the round-trip invariant binds the output schema to the input
  schema permanently. Any future change to an input file's schema is a breaking change
  for users who rely on the round-trip workflow, requiring a coordinated migration.
- **Cholesky re-decomposition cost**: exporting the raw correlation matrix instead of
  the Cholesky factor means a re-decomposition is required when the exported file is
  loaded on a subsequent run. For large correlation matrices this is measurable, but
  it is consistent with the cost incurred on the first run.
- **Barrier overhead in MPI runs**: the rank-0-writes + barrier pattern adds a
  synchronization point between stochastic context construction and training. At small
  node counts this is negligible; at large node counts the barrier may become
  noticeable if I/O is slow.
