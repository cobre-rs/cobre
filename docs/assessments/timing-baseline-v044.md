# Timing Baseline v0.4.4

**Date:** 2026-04-15
**Commit:** 37547b7
**System:** 12th Gen Intel(R) Core(TM) i7-12700KF, 31 GiB RAM, Linux 6.19.11-200.fc43.x86_64
**Rust:** rustc 1.94.1 (e408947bf 2026-03-25)

All timings use `cargo test --release -p cobre-sddp --test deterministic -- --test-threads 1`
(pre-built release binary; no compilation overhead included in wall-clock measurements).

## Full Deterministic Suite (36 tests, --test-threads 1)

| Run        | Wall-clock |
| ---------- | ---------- |
| 1          | 0.706s     |
| 2          | 0.657s     |
| 3          | 0.747s     |
| **Median** | **0.706s** |

## D28 Individual (d28_decomp_weekly_monthly_loads_and_trains)

| Run        | Wall-clock |
| ---------- | ---------- |
| 1          | 0.098s     |
| 2          | 0.093s     |
| 3          | 0.096s     |
| **Median** | **0.096s** |

## Notes

- The suite currently contains 36 test cases (including `incremental_*` and `model_persistence_*`
  tests), not 27 deterministic D-series cases. The ticket referenced 27 cases; the full binary
  includes additional regression tests that run alongside the D-series.
- `--test-threads 1` is used to eliminate parallelism variance between runs.
- `--release` profile uses `opt-level = 3`, `lto = "fat"`, `codegen-units = 1` as defined in
  `Cargo.toml` workspace release profile.
- No `--all-features` flag is used to avoid DHAT global allocator interference from the
  `dhat-heap` feature added in ticket-001.
