# Test Wall-Time Baseline — Phase 0b

Reference wall-time measurement of the default `cargo test --workspace --all-features --release` suite, captured after the HiGHS-integration merge and the test-inventory commits (Epic 01 tickets 001–003) and before any Epic 02 work. Every later epic in the architecture-unification plan compares against these numbers. The companion artifact is [`test-inventory.md`](test-inventory.md) (2,680 `#[test]` functions inventoried).

## 1. Machine spec

| Item            | Value                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------- |
| CPU model       | Intel(R) Xeon(R) Platinum 8259CL @ 2.50 GHz                                                        |
| Cores / threads | 8 cores / 16 threads (1 socket, SMT enabled)                                                       |
| RAM             | 62 GiB (AWS EC2 host; 48 GiB available at measurement)                                             |
| OS              | Ubuntu 20.04, Linux 5.15.0-1084-aws x86_64                                                         |
| Rust toolchain  | `rustc 1.94.1 (e408947bf 2026-03-25)` / `cargo 1.94.1 (29ea6fb6a 2026-03-24)`                      |
| `cargo-nextest` | `0.9.133 (65e806bd5 2026-04-14)`                                                                   |
| HiGHS vendor    | `1.13.1` (git `1d267d97c16928bb5f86fcb2cba2d20f94c8720c`)                                          |
| MPICH           | `/opt/mpich` (local install; `MPICH_HOME`, `PATH`, `LD_LIBRARY_PATH` exported per repo convention) |

Single-user developer workstation; no concurrent load during measurement.

## 2. Measurement protocol

- Command: `cargo test --workspace --all-features --release`.
- Wall time captured with `/usr/bin/time -f "%e"` per run.
- Run 1: cold (`cargo clean` immediately before).
- Runs 2 and 3: incremental (no clean; warm compile cache).
- Default `-j` (no thread-pool tuning). Cargo's default parallel target build and the built-in test runner (sequential per-test-binary launch; tests inside each binary run in parallel by default).
- Top-20 slowest tests were extracted via `cargo nextest run --workspace --all-features --release --no-fail-fast` using the per-test PASS timing lines. nextest uses parallel per-test scheduling so its aggregate wall time (~4 s) is not directly comparable to `cargo test`'s per-binary runner wall time; only the per-test durations are used.

Reported metrics:

- Cold-run wall time (measures build + run).
- Incremental-run median (runs 2 and 3; the metric Epic 09 compares against).
- Pass / fail / ignored counts reported by Cargo's test-result summary, aggregated across all 55 test binaries in the workspace.

## 3. Baseline results

| Run             | Wall-clock (s) |  Pass | Fail | Ignored | Notes                                                          |
| --------------- | -------------: | ----: | ---: | ------: | -------------------------------------------------------------- |
| 1 (cold)        |         175.16 | 3,676 |    0 |       6 | `cargo clean` immediately before; includes full release build. |
| 2 (incremental) |          10.87 | 3,676 |    0 |       6 | Warm compile cache; pure test-execution wall time.             |
| 3 (incremental) |          10.73 | 3,676 |    0 |       6 | Warm compile cache; pure test-execution wall time.             |

**Incremental median:** **10.80 s** (Epic 09 comparison point per source-doc lines 768–779).
**Cold run:** **175.16 s** (reference for build-included end-to-end time).
**Total across 3 runs:** **pass 11,028 · fail 0 · ignored 18**, summed from 55 test binaries per run.

Notes:

- The inventory script (`scripts/test_inventory.py`) counts 2,680 `#[test]` functions. Cargo reports 3,676 because fork tests and doc-tests are counted separately; the inventory covers the static `#[test]` annotations that Epic 09 will prune.
- The 6 ignored tests are the workspace's baseline set (no `--include-ignored` in measurement); they do not vary across runs.
- Pre-measurement test fixes (committed alongside this baseline): `#[cfg(debug_assertions)]` gates on 7 `#[should_panic]` tests whose expected messages come from `debug_assert!` (no release substring match); fixed `test_slot_lookup_growth_safe_in_release` to exercise the growth path via a stored reconcilable slot. Without these fixes, the suite reported 8 release-mode failures and no baseline was recordable.

## 4. Top-20 slowest tests (nextest per-test timings)

| Rank | Test name                                                                           | File                                              | Time (s) |
| ---: | ----------------------------------------------------------------------------------- | ------------------------------------------------- | -------: |
|    1 | `out_of_sample_convergence`                                                         | `cobre-sddp/tests/forward_sampler_integration.rs` |    2.124 |
|    2 | `mixed_scheme_convergence`                                                          | `cobre-sddp/tests/forward_sampler_integration.rs` |    1.199 |
|    3 | `external_inflow_convergence`                                                       | `cobre-sddp/tests/forward_sampler_integration.rs` |    0.974 |
|    4 | `historical_convergence`                                                            | `cobre-sddp/tests/forward_sampler_integration.rs` |    0.856 |
|    5 | `fpha_fitting::tests::select_planes_output_is_subset_of_input`                      | `cobre-sddp/src/fpha_fitting.rs`                  |    0.846 |
|    6 | `fpha_fitting::tests::select_planes_preserves_envelope_property`                    | `cobre-sddp/src/fpha_fitting.rs`                  |    0.842 |
|    7 | `fpha_fitting::tests::select_planes_approximation_error_not_catastrophically_worse` | `cobre-sddp/src/fpha_fitting.rs`                  |    0.841 |
|    8 | `fpha_fitting::tests::select_planes_reduces_to_target_count`                        | `cobre-sddp/src/fpha_fitting.rs`                  |    0.833 |
|    9 | `monthly_noise_sharing_regression`                                                  | `cobre-sddp/tests/forward_sampler_integration.rs` |    0.300 |
|   10 | `external_load_library_populated`                                                   | `cobre-sddp/tests/forward_sampler_integration.rs` |    0.269 |
|   11 | `d26_estimated_par2`                                                                | `cobre-sddp/tests/deterministic.rs`               |    0.265 |
|   12 | `external_ncs_library_populated`                                                    | `cobre-sddp/tests/forward_sampler_integration.rs` |    0.202 |
|   13 | `out_of_sample_declaration_order_invariance`                                        | `cobre-sddp/tests/forward_sampler_integration.rs` |    0.147 |
|   14 | `d30_pattern_d_monthly_quarterly_loads_and_trains`                                  | `cobre-sddp/tests/deterministic.rs`               |    0.107 |
|   15 | `pattern_d_structural_properties_and_training`                                      | `cobre-sddp/tests/pattern_d_integration.rs`       |    0.090 |
|   16 | `d19_multi_hydro_par_truncation`                                                    | `cobre-sddp/tests/deterministic.rs`               |    0.051 |
|   17 | `d29_pattern_c_weekly_par`                                                          | `cobre-sddp/tests/deterministic.rs`               |    0.049 |
|   18 | `d16_par1_lag_shift`                                                                | `cobre-sddp/tests/deterministic.rs`               |    0.046 |
|   19 | `decomp_boundary_cuts_compose_with_weekly_monthly`                                  | `cobre-sddp/tests/decomp_integration.rs`          |    0.044 |
|   20 | `d23_bidirectional_withdrawal`                                                      | `cobre-sddp/tests/deterministic.rs`               |    0.041 |

The `fpha_fitting::*` group in rows 5–8 is the slow-suite identified in `CLAUDE.md` and in the source doc; those four tests alone account for ~3.4 s of sequential test time (and run in parallel under nextest). Five of the top 20 are forward-sampler integration tests; nine are deterministic D-case cases. Epic 09 will consolidate D-case parameter variants and gate `fpha_*` behind a `slow-tests` feature.

## 5. Baseline commit SHA

This file and its lightweight tag `baseline/phase-0b-wall-time` are intended to be created together at commit time. Later epics reference the tag:

```
git show baseline/phase-0b-wall-time -- docs/assessments/test-wall-time-baseline.md
```

The tag will point at the commit landing this document on `feat/architecture-unification`. Epic 09's acceptance criterion (≥ 50% reduction of the incremental median) compares against the **10.80 s** figure above.
