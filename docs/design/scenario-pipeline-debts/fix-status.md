# Assessment Report Fix Status

**Date:** 2026-04-05
**Branch:** `develop`

## Summary

| Category                     | Count |
| ---------------------------- | ----- |
| Total findings               | 60    |
| Fixed                        | 54    |
| Skipped (with justification) | 6     |

---

## All Findings Status

### Critical (4/4 fixed)

| ID     | Status | Description                                                   |
| ------ | ------ | ------------------------------------------------------------- |
| C1     | FIXED  | stage_id range validation in IO parsers                       |
| C2     | FIXED  | Reorder validate-then-standardize in setup.rs                 |
| C3     | FIXED  | Divisibility check before library construction                |
| F1-101 | FIXED  | Correlation JSON example key `groups` -> `correlation_groups` |

### Major (8/8 fixed)

| ID  | Status | Description                                          |
| --- | ------ | ---------------------------------------------------- |
| M1  | FIXED  | NaN check in historical V2.3 validation              |
| M2  | FIXED  | Disjointness check in DecomposedCorrelation::build() |
| M3  | FIXED  | entity_type validation against {inflow, load, ncs}   |
| M4  | FIXED  | Same-type enforcement in IO semantic validation      |
| M5  | FIXED  | Exact divisibility check in V3.4                     |
| M6  | FIXED  | Historical scheme only valid for inflow class        |
| M7  | FIXED  | Single-season year-offset logic in window.rs         |
| M8  | FIXED  | Duplicate row check in external parsers              |

### Performance (3/4 fixed, 1 skipped)

| ID  | Status      | Description                                                     |
| --- | ----------- | --------------------------------------------------------------- |
| P1  | FIXED       | HashMap replaced with flat lookup in historical standardization |
| P2  | **SKIPPED** | Cache-friendly lag access in external standardization           |
| P3  | FIXED       | assert! -> debug_assert! in PrecomputedPar accessors            |
| P4  | FIXED       | Precomputed per-class correlation positions in resolve.rs       |

### Minor (9/11 fixed, 2 skipped)

| ID  | Status      | Description                                                |
| --- | ----------- | ---------------------------------------------------------- |
| m1  | FIXED       | Shared helper `build_observation_sequence` extracted       |
| m2  | FIXED       | debug_assert for HashMap key lookups in historical.rs      |
| m3  | FIXED       | Eq derive added to ScenarioSource                          |
| m4  | FIXED       | Debug derive added to CorrelationRef, ForwardSamplerConfig |
| m5  | FIXED       | Module doc rewritten in resolve.rs                         |
| m6  | FIXED       | Structural doc counts updated to 39                        |
| m7  | FIXED       | Dead test variables removed in setup.rs                    |
| m8  | FIXED       | Test for external NCS library construction added           |
| m9  | FIXED       | LHS/QMC equivalence tests added in generate.rs             |
| m10 | FIXED       | Redundant .max(1) removed in setup.rs                      |
| m11 | **SKIPPED** | Newtype wrappers for scheme/library invariant              |

### Documentation (all fixed)

| ID     | Status | Description                                                 |
| ------ | ------ | ----------------------------------------------------------- |
| F1-002 | FIXED  | ForwardSampler construction lifted (build_sampler_from_ctx) |
| F1-005 | FIXED  | debug_assert! -> assert! in eta_slice accessors             |
| F1-006 | FIXED  | Stack arrays in apply_group_scan for groups <= 64           |
| F1-007 | FIXED  | noise_methods shared via borrowed slice                     |
| F1-008 | FIXED  | n_hydros() accessor used in forward.rs and pipeline.rs      |
| F1-010 | FIXED  | ForwardNoise inner field made private                       |
| F1-101 | FIXED  | Correlation JSON example key fixed                          |
| F1-102 | FIXED  | ForwardSampler architecture section rewritten               |
| F1-103 | FIXED  | build_forward_sampler signature updated                     |
| F1-104 | FIXED  | HydroPenalties field count updated to 16                    |
| F1-105 | FIXED  | Stochastic README key types updated                         |
| F1-106 | FIXED  | Integration tests added for load/NCS classes                |
| F1-107 | FIXED  | IO integration tests for external load/NCS                  |
| F1-108 | FIXED  | OutOfSample dispatch docs rewritten                         |

### Findings F2-xxx (schema/validation)

| ID     | Status | Description                                           |
| ------ | ------ | ----------------------------------------------------- |
| F2-001 | FIXED  | Structural doc counts updated (same as m6)            |
| F2-002 | FIXED  | External scheme requires external files validation    |
| F2-003 | FIXED  | RiskMeasure string validated against "expectation"    |
| F2-004 | FIXED  | num_scenarios minimum: 0 -> 1 in schema               |
| F2-005 | FIXED  | Calendar field minimums: 0 -> 1 in schema             |
| F2-006 | FIXED  | scheme enum constraint added to schema                |
| F2-007 | FIXED  | Internal Rust language removed from schema            |
| F2-008 | FIXED  | RiskMeasure schema internals replaced                 |
| F2-009 | FIXED  | block_mode, sampling_method enum constraints added    |
| F2-101 | FIXED  | Per-class scheme fields added to StochasticSummary    |
| F2-103 | FIXED  | correlation_dim derived from actual entity count      |
| F2-104 | FIXED  | Unit test for build_stochastic_summary added          |
| F2-105 | FIXED  | n_stochastic_ncs field added to StochasticSummary     |
| F2-106 | FIXED  | Redundant opening_tree parameter removed from train() |

### Findings F3-xxx (CHANGELOG/config)

| ID     | Status             | Description                                                         |
| ------ | ------------------ | ------------------------------------------------------------------- |
| F3-001 | FIXED              | CHANGELOG per-class field description corrected                     |
| F3-002 | FIXED              | CHANGELOG "sub-objects" language fixed                              |
| F3-003 | FIXED              | deny_unknown_fields + additionalProperties: false on RawClassConfig |
| F3-006 | **FIXED (v0.4.3)** | month0() hardcoding in window.rs — resolved as Debt 1 (`316feab`)   |

### Findings F4-xxx (Python/misc)

| ID     | Status | Description                                   |
| ------ | ------ | --------------------------------------------- |
| F4-001 | FIXED  | Python run() docstring hydro_models key added |
| F4-003 | FIXED  | generate_opening_tree &mut changed to &       |

---

## Skipped Findings — Justification

### P2: Cache-friendly lag access in external standardization

**Location:** `cobre-stochastic/src/sampling/external.rs:315-351`
**Why skipped:** This is a cache-line optimization for the external inflow
standardization loop. The current code iterates in (stage, scenario, hydro,
lag) order which has a ~40KB stride for lag access. The fix requires
transposing the iteration order or restructuring the lag data layout. This is
a pure performance optimization that does not affect correctness, and the
external standardization runs once at startup (not on the hot path). The
benefit would only be measurable on large cases with many hydros and deep
PAR orders.
**Recommended approach:** Profile with `cargo flamegraph` on a realistic
large case to quantify the impact before investing in the refactor. If lag
access is measurable, transpose to (stage, hydro, lag, scenario) layout.

### m11: Newtype wrappers for scheme/library co-variance invariant

**Location:** `cobre-sddp/src/setup.rs:236-285`
**Why skipped:** This is a type-system design improvement, not a bug fix.
The invariant is: "when `inflow_scheme == External`, `external_inflow_library`
must be `Some`; when `inflow_scheme != External`, it must be `None`" (and
similarly for load/NCS). Currently this is enforced at runtime in setup.rs.
Encoding it in the type system (e.g., a `SchemeWithLibrary` enum that
carries the library in its `External` variant) would require refactoring
the `StudySetup` struct, all its accessors, and all callers.
**Recommended approach:** Design an RFC-style proposal. The enum approach
is cleanest:

```rust
enum ClassConfig {
    InSample,
    OutOfSample,
    Historical(HistoricalScenarioLibrary),
    External(ExternalScenarioLibrary),
}
```

This eliminates the `Option<Library>` fields entirely.

### F3-006: month0() hardcoding in window.rs — RESOLVED

**Location:** `cobre-stochastic/src/sampling/window.rs:140-148`
**Originally skipped** because `SeasonMap` was not passed to the Historical
pipeline and no non-monthly cycle was in use.
**Resolved** in v0.4.3 (`316feab`) as part of Temporal Resolution Debt 1.
Both `discover_historical_windows` and `standardize_historical_windows` now
accept a `SeasonMap` parameter and use a three-tier fallback chain instead
of `month0()`. See `docs/design/temporal-resolution-debts.md` Debt 1.

### P2 (duplicate entry — same as above)

See P2 above.
