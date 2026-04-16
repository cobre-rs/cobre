# Assessment Report: Scenario Sampling Pipeline (Final)

**Date:** 2026-04-05
**Branch:** `develop`
**Plan progress:** 46/46 tickets completed (100%)
**Scope:** all (architecture, performance, correctness, documentation, tests, examples)
**Rounds:** 4 (converged — finding rate: 23 > 15 > 6 > 3)
**Review depth:** 6 specialist adversarial agents, 8 defense passes

---

## Build & Test Health

| Metric                                               | Status                                     |
| ---------------------------------------------------- | ------------------------------------------ |
| `cargo check --workspace --all-features`             | PASS                                       |
| `cargo clippy --workspace --all-features`            | PASS (0 warnings)                          |
| `cargo test --workspace --all-features`              | PASS (3,600+ tests, 0 failures, 4 ignored) |
| Doc tests                                            | PASS (35 doc tests)                        |
| Suppression check (`check_suppressions.py --max 10`) | PASS (10 suppressions, at limit)           |
| Infrastructure genericity (core/io/stochastic)       | PASS (0 SDDP/Benders references)           |

---

## Summary

| Category                             | Count                  |
| ------------------------------------ | ---------------------- |
| **Total findings**                   | 60                     |
| Original assessment (pre-completion) | 26                     |
| New findings (Rounds 1-4)            | 47 raw, 34 after dedup |
| **Defended (closed)**                | 13                     |
| **Acknowledged (backlog)**           | 60                     |
| **Disputed (needs user review)**     | 0                      |

**Verdict:** The codebase is architecturally sound, compiles cleanly, and passes 3,400+ tests. The primary concerns are: (1) defense-in-depth validation gaps that allow silent misconfiguration, (2) stale documentation that contradicts the refactored API, and (3) several schema/CHANGELOG inaccuracies that would mislead users. None of the critical findings are data-corruption bugs in the happy path; all require specific malformed input to trigger.

---

## Backlog (prioritized by severity)

### Critical

| ID     | Location                                                                                                     | Problem                                                                                          | Suggested Resolution                                                                          | Effort  |
| ------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ------- |
| C1     | `cobre-io/src/scenarios/external.rs`, `cobre-stochastic/src/sampling/external.rs`, `cobre-sddp/src/setup.rs` | External scenario `stage_id` never validated; OOB in release builds via `as usize` cast          | Add `stage_id` range validation in IO Layer 3 or add runtime bounds checks in standardization | Small   |
| C2     | `cobre-sddp/src/setup.rs:753-873`                                                                            | Standardize-before-validate ordering allows panics on malformed input before validation runs     | Reorder to validate-then-standardize in setup.rs                                              | Small   |
| C3     | `cobre-sddp/src/setup.rs:747-748`                                                                            | Non-divisible `rows_per_stage / n_entities` silently truncates scenario count, causing OOB       | Add divisibility check: `if rows % n_entities != 0 { return Err(...) }`                       | Small   |
| F1-101 | `book/src/guide/stochastic-modeling.md:276-296`                                                              | User-facing correlation JSON example uses `"groups"` key; parser requires `"correlation_groups"` | Change example key from `"groups"` to `"correlation_groups"`                                  | Trivial |

### Major

| ID     | Location                                                                     | Problem                                                                                                                                                                                   | Suggested Resolution                                                       | Effort  |
| ------ | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------- |
| M1     | `cobre-stochastic/src/sampling/historical.rs:597-611`                        | Historical library validation checks NEG_INFINITY but not NaN; external validation checks both                                                                                            | Add `\|\| value.is_nan()` to V2.3 check                                    | Trivial |
| M2     | `cobre-stochastic/src/correlation/resolve.rs:333-338`                        | Correlation group disjointness not validated; overlapping groups produce wrong covariance                                                                                                 | Add disjointness check in `DecomposedCorrelation::build()`                 | Small   |
| M3     | `cobre-io/src/validation/referential.rs:416-427`                             | `entity_type` string not validated; typos silently produce uncorrelated noise                                                                                                             | Validate against `{"inflow", "load", "ncs"}` in IO Layer 3                 | Small   |
| M4     | `cobre-io/src/validation/semantic.rs`                                        | Same-type enforcement only in stochastic layer; late error in MPI context                                                                                                                 | Add same-type rule to IO Layer 5                                           | Medium  |
| M5     | `cobre-stochastic/src/sampling/external.rs:649-668`                          | V3.4 scenario count check uses integer division; intra-stage entity imbalance passes                                                                                                      | Check exact divisibility: `rows % n_entities == 0`                         | Small   |
| M6     | `cobre-io/src/stages.rs:801-803`                                             | `Historical` scheme accepted for load/NCS classes; error only at runtime deep in setup                                                                                                    | Add V1.x rule: Historical only valid for inflow                            | Small   |
| M7     | `cobre-stochastic/src/sampling/historical.rs:688-698`, `window.rs:293-303`   | Single-season year-offset logic: `0 < 0` never true, self-referential PAR calculation                                                                                                     | Use explicit year arithmetic instead of season-wrap detection              | Medium  |
| M8     | `cobre-io/src/scenarios/external.rs:94-161`                                  | Duplicate external scenario rows not validated; last-write-wins silently discards data                                                                                                    | Add uniqueness check after sorting in parsers                              | Small   |
| F1-005 | `cobre-stochastic/src/sampling/historical.rs:193-206`, `external.rs:151-163` | `eta_slice` uses `debug_assert!` only; in release, wrong (stage, scenario) silently returns valid but incorrect noise                                                                     | Replace `debug_assert!` with `assert!` in eta_slice accessors              | Small   |
| F1-102 | `book/src/crates/stochastic.md:245-305`                                      | ForwardSampler documented as 4-variant enum; actual is a struct with 3 ClassSampler fields. ForwardNoise documented as dual-lifetime enum; actual is single-lifetime newtype              | Rewrite forward sampler architecture section                               | Medium  |
| F1-103 | `book/src/crates/stochastic.md:270-288`                                      | `build_forward_sampler` signature documented as 3-arg; actual takes `ForwardSamplerConfig` struct                                                                                         | Update documented signature and description                                | Small   |
| F1-104 | `book/src/crates/core.md:223`                                                | HydroPenalties documented as "11 f64 fields"; actual struct has 16 fields                                                                                                                 | Update count to 16                                                         | Trivial |
| F2-002 | `cobre-io/src/validation/semantic.rs`                                        | No validation that External scheme has corresponding external scenario files present; error surfaces late                                                                                 | Add semantic rule: External scheme requires non-empty external file vector | Medium  |
| F2-003 | `cobre-io/src/stages.rs:309-330, 920-929`                                    | `RawRiskMeasure::Expectation(String)` accepts any string via serde(untagged); `"foobar"` silently becomes Expectation                                                                     | Validate inner string equals `"expectation"` in `validate_risk_measure`    | Small   |
| F3-001 | `CHANGELOG.md:53-57`                                                         | CHANGELOG claims per-class sub-objects carry `historical_years` and `file` fields; actual `RawClassConfig` has only `scheme`. `historical_years` is at parent level, `file` doesn't exist | Correct CHANGELOG description                                              | Small   |

### Performance

| ID  | Location                                                       | Problem                                                                                         | Suggested Resolution                                                      | Effort |
| --- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------ |
| P1  | `cobre-stochastic/src/sampling/historical.rs:360-437`          | Historical standardization uses HashMap in tight inner loop (~16.8M lookups for realistic case) | Replace with flat 3D array indexed by (hydro_idx, year_offset, season_id) | Medium |
| P2  | `cobre-stochastic/src/sampling/external.rs:315-351`            | External inflow standardization has L1 cache thrashing for lag access (40KB stride)             | Optimize lag access layout or transpose iteration order                   | Medium |
| P3  | `cobre-stochastic/src/par/precompute.rs:254-315`               | PrecomputedPar accessors use `assert!` instead of `debug_assert!` (600K-9.6M calls)             | Change to `debug_assert!` for consistency with library accessors          | Small  |
| P4  | `cobre-stochastic/src/correlation/resolve.rs:296-339, 375-377` | `apply_correlation_for_class` uses O(n\*m) scan instead of precomputed O(1) positions           | Precompute per-class positions and use fast path                          | Medium |

### Minor

| ID     | Location                                                              | Problem                                                                                            | Suggested Resolution                                                   | Effort  |
| ------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------- |
| m1     | `historical.rs` + `window.rs`                                         | `build_observation_sequence` and `build_required_sequence` duplicated (~40 lines)                  | Extract shared helper                                                  | Small   |
| m2     | `historical.rs:418-465`                                               | `unwrap_or(0.0)` fallbacks mask potential key mismatches                                           | Consider fail-fast or logging                                          | Small   |
| m3     | `cobre-core/src/scenario.rs:107`                                      | `ScenarioSource` derives `PartialEq` but not `Eq`                                                  | Add `Eq` derive                                                        | Trivial |
| m4     | `cobre-stochastic/src/sampling/mod.rs`                                | `CorrelationRef`, `SampleRequest`, `ForwardSamplerConfig` missing `Debug` derive                   | Add `Debug`                                                            | Trivial |
| m5     | `cobre-stochastic/src/correlation/resolve.rs:1-11`                    | Module doc claims range validation; that validation is in cobre-io                                 | Fix doc comment                                                        | Trivial |
| m6     | `cobre-io/src/validation/structural.rs:9,31,134`                      | File counts: 34, 36, 34 in doc comments; actual count is 39                                        | Update all doc comments to 39                                          | Trivial |
| m7     | `cobre-sddp/src/setup.rs:3870-4095`                                   | Dead test variables silenced with `let _ = ...`                                                    | Remove dead test code                                                  | Small   |
| m8     | `cobre-sddp/src/setup.rs`                                             | No test for external NCS library construction                                                      | Add test                                                               | Small   |
| m9     | `cobre-stochastic/src/tree/generate.rs`                               | Per-class vs full-vector equivalence tests only cover SAA noise                                    | Add LHS/QMC equivalence tests                                          | Small   |
| m10    | `cobre-sddp/src/setup.rs:747-748`                                     | Redundant `.max(1)` guard after `n_hydros > 0` check                                               | Remove dead guard                                                      | Trivial |
| m11    | `cobre-sddp/src/setup.rs:236-285`                                     | Scheme/library co-variance invariant not enforced by type system                                   | Consider newtype wrappers                                              | Medium  |
| F1-002 | `cobre-sddp/src/forward.rs:911-928`                                   | ForwardSampler rebuilt every training iteration; small heap allocation per class                   | Lift construction out of per-iteration loop                            | Small   |
| F1-006 | `cobre-stochastic/src/correlation/resolve.rs:369-391`                 | `apply_group_scan` allocates 3 Vec per call on hot path                                            | Use stack arrays or precomputed positions                              | Medium  |
| F1-007 | `cobre-stochastic/src/sampling/mod.rs:368, 443-447`                   | `noise_methods` Box<[NoiseMethod]> cloned into each OutOfSample class                              | Share via borrowed slice                                               | Small   |
| F1-008 | `cobre-sddp/src/forward.rs:919-923`, `simulation/pipeline.rs:842-846` | ClassDimensions computation duplicated; no `n_hydros()` accessor                                   | Add `n_hydros()` to StochasticContext                                  | Trivial |
| F1-010 | `cobre-stochastic/src/sampling/mod.rs:56-58`                          | `ForwardNoise<'b>(pub &'b [f64])` exposes inner field as pub                                       | Make field private, access via `as_slice()`                            | Small   |
| F1-105 | `cobre-stochastic/README.md:27`                                       | README lists `sample_forward` as key type; missing ForwardSampler, ClassSampler, libraries         | Update Key Types list                                                  | Small   |
| F1-106 | `cobre-sddp/tests/forward_sampler_integration.rs`                     | No Historical/External integration tests for load or NCS entity classes                            | Add tests with load/NCS external/historical data                       | Medium  |
| F1-107 | `cobre-io/tests/integration.rs:364-459`                               | IO integration tests only cover `external_inflow_scenarios.parquet`; no test for load/NCS          | Add IO integration tests for external load and NCS                     | Small   |
| F1-108 | `book/src/crates/stochastic.md:336-353`                               | OutOfSample dispatch path doc references old ForwardSampler::OutOfSample variant                   | Rewrite to describe per-class ClassSampler::OutOfSample::fill()        | Small   |
| F2-001 | `cobre-io/src/validation/structural.rs:9,31,36,134`                   | Four doc comments cite 34/36/39 inconsistently; actual count is 39                                 | Update all to 39 (overlaps m6)                                         | Trivial |
| F2-004 | `book/src/schemas/stages.schema.json:271-276`                         | `num_scenarios` has `"minimum": 0` but parser rejects 0                                            | Change to `"minimum": 1`                                               | Trivial |
| F2-005 | `book/src/schemas/stages.schema.json:206-241`                         | Calendar fields (`month_start`, `day_start`, etc.) have `"minimum": 0`; valid range starts at 1    | Change to `"minimum": 1`                                               | Trivial |
| F2-006 | `book/src/schemas/stages.schema.json:42-52`                           | `scheme` field has no `enum` constraint; parser only accepts 4 values                              | Add `"enum": ["in_sample", "out_of_sample", "external", "historical"]` | Trivial |
| F2-007 | `book/src/schemas/stages.schema.json:357-358`                         | Top-level description leaks internal Rust language ("Private", "Not re-exported")                  | Replace with user-facing description                                   | Trivial |
| F2-008 | `book/src/schemas/stages.schema.json:130-148`                         | RawRiskMeasure schema exposes serde internals                                                      | Replace with user-facing description                                   | Trivial |
| F2-009 | `book/src/schemas/stages.schema.json:250-285`                         | `block_mode` and `sampling_method` have no `enum` constraint                                       | Add enum values                                                        | Trivial |
| F2-101 | `cobre-sddp/src/stochastic_summary.rs:111-134`                        | StochasticSummary missing per-class scheme fields; provenance records them but summary discards    | Add scheme fields to summary, wire to CLI/Python                       | Small   |
| F2-103 | `cobre-sddp/src/stochastic_summary.rs:245-250`                        | `correlation_dim` hardcoded as `n_hydros x n_hydros`; per-class correlation spans all entities     | Derive from actual correlated entity count                             | Small   |
| F2-104 | `cobre-sddp/src/stochastic_summary.rs:365+`                           | No test for `correlation_dim` field or provenance/summary disconnect                               | Add unit test for `build_stochastic_summary`                           | Small   |
| F2-105 | `cobre-sddp/src/stochastic_summary.rs:111-134`                        | StochasticSummary has `n_load_buses` but no `n_stochastic_ncs`                                     | Add `n_stochastic_ncs` field, wire to CLI/Python                       | Small   |
| F2-106 | `cobre-sddp/src/training.rs:342-353`                                  | `train()` takes `opening_tree` as separate param; already accessible via `training_ctx.stochastic` | Remove redundant parameter                                             | Small   |
| F3-002 | `CHANGELOG.md:35-38`                                                  | "sub-objects" (plural) implies `historical_years` is per-class; it's at `scenario_source` level    | Rewrite to "scenario_source object" (singular)                         | Trivial |
| F3-003 | `book/src/schemas/stages.schema.json:42-52`                           | `RawClassConfig` schema lacks `additionalProperties: false`; extra fields silently accepted        | Add `additionalProperties: false` + `#[serde(deny_unknown_fields)]`    | Small   |
| F3-006 | `cobre-stochastic/src/sampling/window.rs:140-148`                     | `month0()` hardcoding assumes monthly seasons; weekly/custom cycles produce zero valid windows     | Derive season_id from `SeasonMap::season_for_date()`                   | Medium  |
| F4-001 | `cobre-python/src/run.rs:754-755`                                     | Python `run()` docstring omits `"hydro_models"` key from return dict                               | Add key to docstring                                                   | Trivial |
| F4-003 | `cobre-stochastic/src/tree/generate.rs:75`                            | `generate_opening_tree` takes `&mut DecomposedCorrelation`; only `&self` methods called            | Change to `&DecomposedCorrelation`                                     | Trivial |

---

## Defended Findings (for reference)

| ID     | Location                           | Claim                                              | Defense                                                                                                  |
| ------ | ---------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| F1-001 | `correlation/resolve.rs:320-339`   | apply_correlation_for_class uses O(n\*m) scan      | Duplicate of tracked finding P4                                                                          |
| F1-003 | `class_sampler.rs:196-203`         | debug_assert only for apply_initial_state          | Rust slice indexing performs bounds check in all builds; panic prevents corruption                       |
| F1-004 | `class_sampler.rs:252-261`         | InSample offset+len not validated at construction  | Rust slice bounds check fires in release; panics, does not corrupt                                       |
| F1-009 | `class_sampler.rs:157-165`         | u64-to-usize truncation on 32-bit targets          | All CI/release targets are 64-bit; HPC project has no 32-bit target                                      |
| F1-011 | `simulation/pipeline.rs:601-602`   | Stage ID vs array index mismatch in seeds          | Both training and simulation use `t as u32` consistently; opening tree indexed by stage_idx              |
| F1-012 | `forward.rs:957-959`               | Per-worker perm_scratch allocation                 | One allocation per worker per iteration (400-1600 bytes); negligible vs LP solve cost                    |
| F1-109 | `CHANGELOG`                        | No backward-compat detection test                  | Test `test_parse_old_flat_format_rejected` exists at `stages.rs:1403-1435`                               |
| F1-110 | `stochastic-modeling.md`           | Guide omits `load_factors.json`                    | `load_factors.json` is a deterministic input, not stochastic; editorial omission is appropriate          |
| F1-111 | `stochastic.md:40`                 | Claims "six failure domains" but 9 variants span 7 | Six domains is a design classification grouping 9 variants; documented in error.rs                       |
| F2-102 | `determinism.rs`, `integration.rs` | All tests pass empty stages                        | `forward_sampler_integration.rs` has dedicated non-InSample tests with populated stages via `StudySetup` |
| F3-004 | `stochastic_summary.rs`            | Summary missing per-class schemes                  | Duplicate of acknowledged F2-101                                                                         |
| F3-005 | `window.rs:283`                    | Lag season computation obscure, no comment         | Comments at lines 273-275 explain the intent; standard modular arithmetic pattern                        |
| F4-002 | `stochastic_summary.rs`            | Summary missing n_stochastic_ncs                   | Duplicate of acknowledged F2-105                                                                         |

---

## Recommended Fix Priority

### Must-fix before v0.4.0 release

| #   | Finding | Effort  | Description                                                              |
| --- | ------- | ------- | ------------------------------------------------------------------------ |
| 1   | C1      | Small   | Add `stage_id` range validation in IO Layer 3                            |
| 2   | C2      | Small   | Reorder to validate-then-standardize in setup.rs                         |
| 3   | C3      | Small   | Add divisibility check before library construction                       |
| 4   | F1-101  | Trivial | Fix correlation JSON example: `"groups"` -> `"correlation_groups"`       |
| 5   | M1      | Trivial | Add `is_nan()` to historical V2.3 check                                  |
| 6   | M2      | Small   | Add disjointness validation in `DecomposedCorrelation::build()`          |
| 7   | M3      | Small   | Validate `entity_type` against `{"inflow", "load", "ncs"}` in IO Layer 3 |
| 8   | M5      | Small   | Add exact divisibility check in V3.4                                     |
| 9   | M6      | Small   | Add V1.x rule: Historical only valid for inflow class                    |
| 10  | F1-005  | Small   | Change `debug_assert!` to `assert!` in eta_slice accessors               |
| 11  | F2-003  | Small   | Validate RiskMeasure string equals "expectation"                         |
| 12  | F3-001  | Small   | Correct CHANGELOG per-class field description                            |

### Should-fix before v0.4.0 release

| #   | Finding | Effort  | Description                                                      |
| --- | ------- | ------- | ---------------------------------------------------------------- |
| 13  | M4      | Medium  | Add same-type enforcement to IO Layer 5                          |
| 14  | M7      | Medium  | Fix single-season year-offset logic                              |
| 15  | M8      | Small   | Add uniqueness check in external parsers                         |
| 16  | F1-102  | Medium  | Rewrite ForwardSampler architecture section in stochastic.md     |
| 17  | F1-103  | Small   | Update build_forward_sampler documented signature                |
| 18  | F1-104  | Trivial | Update HydroPenalties field count to 16                          |
| 19  | F2-002  | Medium  | Add semantic validation: External scheme requires external files |
| 20  | P3      | Small   | Change `assert!` to `debug_assert!` in PrecomputedPar accessors  |
| 21  | F3-003  | Small   | Add `deny_unknown_fields` to RawClassConfig                      |

### Nice-to-have (post-release)

| #   | Finding   | Effort  | Description                                                                    |
| --- | --------- | ------- | ------------------------------------------------------------------------------ |
| 22  | P1        | Medium  | Replace HashMap with flat array in historical standardization                  |
| 23  | P2        | Medium  | Optimize lag access cache locality                                             |
| 24  | P4        | Medium  | Precompute per-class correlation positions                                     |
| 25  | F1-006    | Medium  | Eliminate Vec allocations in apply_group_scan                                  |
| 26  | F3-006    | Medium  | Replace month0() with SeasonMap dispatch in window discovery                   |
| 27  | m11       | Medium  | Encode scheme/library invariant in the type system                             |
| 28  | F1-106    | Medium  | Add Historical/External integration tests for load/NCS classes                 |
| 29  | F2-006    | Trivial | Add enum constraints to schema for scheme, block_mode, sampling_method         |
| 30  | F2-004    | Trivial | Fix num_scenarios minimum: 0 -> 1 in schema                                    |
| 31  | F2-005    | Trivial | Fix calendar field minimums: 0 -> 1 in schema                                  |
| 32  | F2-007    | Trivial | Remove internal Rust language from schema descriptions                         |
| 33  | F2-008    | Trivial | Replace RiskMeasure schema internals with user-facing text                     |
| 34  | F2-009    | Trivial | Add enum constraints for block_mode, sampling_method                           |
| 35  | F2-101    | Small   | Add per-class scheme fields to StochasticSummary                               |
| 36  | F2-105    | Small   | Add n_stochastic_ncs to StochasticSummary                                      |
| 37  | F2-106    | Small   | Remove redundant opening_tree parameter from train()                           |
| 38  | F4-003    | Trivial | Change generate_opening_tree &mut to &                                         |
| 39  | F1-002    | Small   | Lift ForwardSampler construction out of per-iteration loop                     |
| 40  | F1-007    | Small   | Share noise_methods via borrowed slice                                         |
| 41  | F1-008    | Trivial | Add n_hydros() accessor to StochasticContext                                   |
| 42  | Remaining | Various | m1-m10, F1-010, F1-105, F1-107, F1-108, F2-001, F2-103, F2-104, F3-002, F4-001 |

---

## Architecture Quality Scorecard

| Dimension                  | Score | Notes                                                                                                                                                                                    |
| -------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Type safety                | 7/10  | Stringly-typed `entity_type`, unvalidated `stage_id`, scheme/library invariant gap, any-string RiskMeasure                                                                               |
| Error handling             | 6/10  | `debug_assert` in release-facing code, standardize-before-validate, late error surfaces, no scheme/file cross-validation                                                                 |
| Performance (hot-path)     | 9/10  | Zero allocations in inner loops; one regression (P4 correlation scan + F1-006 Vec allocs)                                                                                                |
| Performance (construction) | 7/10  | HashMap in tight loops, `assert!` overhead, cache-unfriendly lag access                                                                                                                  |
| Modularity                 | 8/10  | Clean crate separation; setup.rs duplication; redundant train() parameter                                                                                                                |
| Test coverage              | 8/10  | 3,410+ tests; gaps in NCS library, load/NCS external integration, non-SAA equivalence                                                                                                    |
| Infrastructure genericity  | 10/10 | Zero SDDP references in core/io/stochastic                                                                                                                                               |
| Validation completeness    | 5/10  | 5-layer architecture is excellent but has 7+ missing rules (External file presence, RiskMeasure string, entity_type values, same-type in IO, disjointness, stage_id range, divisibility) |
| Documentation accuracy     | 4/10  | ForwardSampler docs fundamentally wrong, schema constraints incomplete, CHANGELOG inaccurate, 4 doc count mismatches, 2 stale README/doc sections                                        |
| Code hygiene               | 8/10  | Zero clippy warnings; dead test code; unnecessary &mut; pub inner field                                                                                                                  |

**Overall: 7.2/10** (down from 7.7 at 52% completion)

The score decreased despite plan completion because: (1) documentation was not updated to match the refactored architecture, (2) schema files auto-generated from Rust doc comments leaked internal implementation details, and (3) validation rules were not added for new configuration combinations.

---

## Test Coverage Gaps Identified

1. No test for external scenarios with non-contiguous `stage_id` values
2. No test for external scenarios with negative `stage_id`
3. No test for non-divisible `rows_per_stage / n_entities`
4. No test for overlapping correlation groups (same entity in two groups)
5. No test for misspelled `entity_type` in correlation groups
6. No test for `Historical` scheme on load/NCS classes
7. No test for single-season (annual) systems with PAR lags in historical sampling
8. No test for duplicate `(stage, scenario, entity)` rows in external scenarios
9. No test for external NCS library construction in setup.rs
10. No test for per-class vs full-vector bit-identity with LHS/QMC noise methods
11. No integration test for `external_load_scenarios.parquet` or `external_ncs_scenarios.parquet` through `load_case`
12. No Historical/External integration test for load or NCS entity classes through training
13. No test for `correlation_dim` computation in `build_stochastic_summary`
14. No test for `"risk_measure": "foobar"` being silently accepted as Expectation
15. No test verifying `RawClassConfig` rejects unknown fields (when `deny_unknown_fields` is added)

---

## Round-by-Round Detail

### Round 1 (Architecture/Performance/Correctness + Documentation/Tests)

**Attackers:** 2 parallel agents (arch + docs), 252s + 257s
**Defenders:** 2 parallel agents, 370s + 203s
**Findings:** 23 total (12 arch/perf/correctness + 11 docs/tests)
**Verdicts:** 10 defended, 13 acknowledged

Focused on the new ForwardSampler composite architecture (class_sampler.rs), sampling module refactoring, integration wiring (forward.rs, backward.rs, simulation), and user-facing documentation. Key discoveries: eta_slice silent wrong data (F1-005), wrong correlation JSON key in guide (F1-101), and fundamentally stale ForwardSampler documentation (F1-102, F1-103).

### Round 2 (IO/Config/Validation + Integration Wiring)

**Attackers:** 2 parallel agents (IO + integration), 312s + 355s
**Defenders:** 2 parallel agents, 196s + 236s
**Findings:** 15 total (9 IO/schema + 6 integration)
**Verdicts:** 1 defended, 14 acknowledged

Deep dive into all 5 IO validation layers, JSON schema accuracy, config parsing edge cases, and training/simulation integration. Key discoveries: RiskMeasure silent any-string acceptance (F2-003), missing External-scheme file presence validation (F2-002), and 8 schema constraint gaps (F2-004 through F2-009).

### Round 3 (Backward Pass/Python/MPI/Window/CHANGELOG)

**Attacker:** 1 agent, 589s
**Defender:** 1 agent, 134s
**Findings:** 6 total
**Verdicts:** 2 defended, 4 acknowledged

Covered backward pass consistency, Python binding parity, MPI broadcast paths, CHANGELOG accuracy, and example validity. Key discoveries: CHANGELOG falsely claims per-class configs carry `historical_years`/`file` (F3-001), and month0() hardcoding in window discovery breaks non-monthly cycles (F3-006).

### Round 4 (Final Deep Dive)

**Attacker:** 1 agent, 454s
**Defender:** 1 agent, 62s
**Findings:** 3 total (+ "NO NEW FINDINGS" convergence signal)
**Verdicts:** 1 defended, 2 acknowledged

Final sweep of remaining areas. Only minor findings: Python docstring omission (F4-001) and unnecessary &mut in tree generation (F4-003). Convergence reached.
