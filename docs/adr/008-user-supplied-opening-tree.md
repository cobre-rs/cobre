# ADR-008: User-Supplied Opening Tree via Parquet File

**Status:** Accepted
**Date:** 2026-03-13

## Context

The opening tree in `cobre-stochastic` is always generated internally by
`generate_opening_tree()`. It uses SipHash-derived seeds and the spatial
correlation Cholesky factor to produce the backward-pass noise realizations
stored in `OpeningTree`. Users can supply some scenario input files (e.g.,
`inflow_seasonal_stats.parquet`, `inflow_ar_coefficients.parquet`) but have no
way to supply their own backward-pass noise realizations.

This limitation affects several advanced workflows:

- **Reproducibility across tools**: a practitioner who generated a specific set
  of opening scenarios in a reference tool and wants to compare `cobre-sddp`
  results against the same noise realizations cannot do so without modifying
  source code.
- **Sensitivity analysis**: researchers who want to study how a particular
  extreme scenario propagates through the policy need to inject that scenario
  deterministically.
- **Round-trip with stochastic artifact export**: ADR-009 establishes that
  `cobre-sddp` can export the opening tree it used during training to
  `scenarios/noise_openings.parquet`. Without a corresponding import path,
  that export cannot be replayed as input, breaking the round-trip invariant.

The `OpeningTree` struct holds a flat `Box<[f64]>` backing array with
stage-major ordering and `openings_per_stage` metadata. The downstream consumer
`sample_forward()` in `crates/cobre-stochastic/src/sampling/insample.rs`
indexes into `OpeningTree` using only its declared dimensions, regardless of
how the tree was populated. This means the bypass can be implemented entirely
in `build_stochastic_context()` without touching the sampling or backward-pass
code paths.

## Decision

An optional file `scenarios/noise_openings.parquet` is recognised by
`cobre-io`. When present, `build_stochastic_context()` in
`crates/cobre-stochastic/src/context.rs` loads it into `OpeningTree` via
`OpeningTree::from_parts()` instead of calling `generate_opening_tree()`.

**Parquet schema.** The file has exactly four columns:

| Column          | Arrow type | Semantics                                                               |
| --------------- | ---------- | ----------------------------------------------------------------------- |
| `stage_id`      | `i32`      | Zero-based stage index (0 to n_stages − 1)                              |
| `opening_index` | `u32`      | Zero-based opening index within the stage (0 to openings_per_stage − 1) |
| `entity_index`  | `u32`      | Zero-based entity index corresponding to the system dimension ordering  |
| `value`         | `f64`      | Noise realization for this (stage, opening, entity) triple              |

The entity ordering follows the system dimension convention: hydro entities
first (sorted by canonical ID), then load buses (sorted by canonical ID),
matching the ordering used by `generate_opening_tree()`. The total number of
rows must equal `n_stages × openings_per_stage × dim`, where
`dim = n_hydros + n_load_buses`.

**Bypass logic.** The decision tree in `build_stochastic_context()` is:

1. Attempt to load `scenarios/noise_openings.parquet` from the case directory.
2. If the file is absent, call `generate_opening_tree()` as before.
3. If the file is present, load it through the `cobre-io` validation layer and
   construct `OpeningTree::from_parts()` from the validated data.

**Validation rules.** The `cobre-io` validation layer checks:

- **Dimension mismatch**: the number of distinct `entity_index` values must
  equal the system `dim`. A mismatch is a hard error.
- **Stage count mismatch**: the number of distinct `stage_id` values must equal
  the configured number of study stages. A mismatch is a hard error.
- **Missing opening indices**: for each stage, every opening index from 0 to
  `openings_per_stage − 1` must be present for every entity. Gaps produce a
  hard error.

**Correlation bypass.** User-supplied noise is used as-is. The Cholesky
spatial correlation factor is not applied to user-supplied openings. The user
is responsible for incorporating any desired spatial correlation structure into
the values they supply. This matches the semantics of the export path in
ADR-009, which writes the post-Cholesky values that the solver actually used.

**Partial trees not supported.** The user must supply openings for all study
stages. Partial-stage override (supplying openings for a subset of stages while
generating the remainder internally) is deferred to a future version.

**Interaction with `base_seed`.** The `base_seed` configuration parameter
remains required even when a user-supplied opening tree is present. The opening
tree and forward-pass noise are separate concerns: `base_seed` governs the
forward-pass scenario sampling performed by `sample_forward()`, which uses
SipHash seeds derived independently of the opening tree. Supplying a custom
opening tree has no effect on forward-pass noise.

## Consequences

**Benefits:**

- Users can replay exact backward-pass noise realizations from external tools
  or reference runs, enabling cross-tool comparison and reproducibility.
- The round-trip invariant with stochastic artifact export (ADR-009) is
  complete: exported `noise_openings.parquet` uses the same schema and can be
  copied back as input without modification.
- `sample_forward()` requires no changes because it indexes `OpeningTree` by
  position, not by origin.
- The bypass is confined to a single decision point in
  `build_stochastic_context()`; no changes are required in the backward pass,
  cut management, or convergence monitoring.

**Costs and risks:**

- Validation complexity increases: `cobre-io` must check three new error
  conditions (dimension mismatch, stage count mismatch, missing opening
  indices) for this optional file.
- The user is fully responsible for the statistical properties of supplied
  openings. A poorly constructed opening tree (e.g., all zeros, non-physical
  correlations) will produce a valid but potentially poor policy without any
  warning from the solver.
- The entity ordering requirement (hydros first, then load buses, both sorted
  by canonical ID) is an implicit convention that users must follow; violating
  it causes silent value misassignment rather than a schema error, because the
  file only stores indices, not entity identifiers.
- Partial-stage override is not supported in v0.1.x. Users who want to replace
  a subset of stages must supply a complete tree, duplicating the internally
  generated values for the unmodified stages.
