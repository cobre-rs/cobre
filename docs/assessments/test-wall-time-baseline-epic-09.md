# Test Wall-Time Baseline — Epic 09 Start

Reference wall-time measurement of the test suite at the start of Epic 09
(architecture-unification plan). This supersedes the Phase 0b baseline in
[`test-wall-time-baseline.md`](test-wall-time-baseline.md) as the comparison
point for Epic 09 ticket-007. The Phase 0b baseline was captured before Epics
02–07 and v0.5.0 feature work; this file reflects the current state after those
changes.

## 1. Machine spec

| Item            | Value                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| CPU model       | 12th Gen Intel(R) Core(TM) i7-12700KF                                                                        |
| Cores / threads | 12 cores / 20 threads (1 socket, SMT enabled)                                                                |
| RAM             | 32 GiB (11 GiB used at measurement start)                                                                    |
| OS              | Fedora Linux 43 (Workstation Edition), Linux 6.19.12-200.fc43.x86_64 x86_64                                  |
| Rust toolchain  | `rustc 1.94.1 (e408947bf 2026-03-25)` / `cargo 1.94.1 (29ea6fb6a 2026-03-24)`                                |
| `cargo-nextest` | `0.9.133 (65e806bd5 2026-04-14)`                                                                             |
| HiGHS vendor    | `1.13.1` (git `1d267d97c16928bb5f86fcb2cba2d20f94c8720c`, vendored under `crates/cobre-solver/vendor/HiGHS`) |
| MPICH           | `4.2.3` (system install; `MPICH_HOME`, `PATH`, `LD_LIBRARY_PATH` exported per repo convention)               |

Developer workstation; two users logged in. Load average ranged from ~1.6
(pre-measurement) to ~5.1 (during measurement due to background IDE activity).
Default `-j` (no thread-pool tuning).

## 2. Measurement protocol

Three command configurations, each run three times back-to-back:

- **Config A**: `cargo test --workspace --release`
- **Config B**: `cargo test --workspace --all-features --release`
- **Config C**: `cargo nextest run --workspace --all-features --release`

Wall time captured with `/usr/bin/time -f "%e"` per run.

Run ordering:

1. Warm build: `cargo test --workspace --all-features --release --no-run` (non-timed; primes the compile cache).
2. `cargo clean` immediately before Config A Run 1 (cold run).
3. Config A Runs 2 and 3: incremental (no clean; warm compile cache).
4. Config B Runs 1–3: incremental (no clean; Config A build cache reused for shared code; features differ so some recompilation occurs on Run 1).
5. Config C Runs 1–3: incremental (no clean; binaries already built by Config B).

Reported metrics:

- Per-config wall times for all three runs.
- Median of all three runs per config (ticket-007 comparison point).
- For Config A and B: pass / fail / ignored counts summed from all 52 test binaries, from a separate representative run after measurements.
- For Config C: nextest summary line (tests run / passed / skipped).

## 3. Baseline results

### Config A — `cargo test --workspace --release`

| Run             | Wall-clock (s) |  Pass | Fail | Ignored | Notes                                                          |
| --------------- | -------------: | ----: | ---: | ------: | -------------------------------------------------------------- |
| 1 (cold)        |         106.40 | 3,684 |    0 |      34 | `cargo clean` immediately before; includes full release build. |
| 2 (incremental) |           5.46 | 3,684 |    0 |      34 | Warm compile cache; pure test-execution wall time.             |
| 3 (incremental) |           5.53 | 3,684 |    0 |      34 | Warm compile cache; pure test-execution wall time.             |

**Median (all 3 runs):** **5.53 s**
**Incremental median (runs 2–3):** **5.50 s**
**Cold run:** **106.40 s**

### Config B — `cargo test --workspace --all-features --release`

| Run             | Wall-clock (s) |  Pass | Fail | Ignored | Notes                                                      |
| --------------- | -------------: | ----: | ---: | ------: | ---------------------------------------------------------- |
| 1 (incremental) |          66.91 | 3,716 |    0 |      34 | No `cargo clean`; partial recompile due to feature change. |
| 2 (incremental) |           5.47 | 3,716 |    0 |      34 | Warm compile cache; pure test-execution wall time.         |
| 3 (incremental) |           5.63 | 3,716 |    0 |      34 | Warm compile cache; pure test-execution wall time.         |

**Median (all 3 runs):** **5.63 s**
**Incremental median (runs 2–3):** **5.55 s**

Config B has 32 more passing tests than Config A (3,716 vs 3,684) because
`--all-features` enables additional test-gated code paths.

### Config C — `cargo nextest run --workspace --all-features --release`

| Run             | Wall-clock (s) | Tests run | Passed | Skipped | Notes                                     |
| --------------- | -------------: | --------: | -----: | ------: | ----------------------------------------- |
| 1 (incremental) |           2.06 |     3,429 |  3,429 |      31 | Binaries built by Config B; nextest only. |
| 2 (incremental) |           2.00 |     3,429 |  3,429 |      31 | Warm compile cache.                       |
| 3 (incremental) |           2.03 |     3,429 |  3,429 |      31 | Warm compile cache.                       |

**Median (all 3 runs):** **2.03 s**

nextest counts differ from `cargo test` counts: nextest does not run doc-tests
(which add ~287 tests to the `cargo test` total) and uses per-test parallelism
instead of per-binary parallelism, explaining the lower wall time.

## 4. Summary

| Config                                                      | Median wall-clock (s) | Incremental median (s) |
| ----------------------------------------------------------- | --------------------: | ---------------------: |
| A: `cargo test --workspace --release`                       |                  5.53 |                   5.50 |
| B: `cargo test --workspace --all-features --release`        |                  5.63 |                   5.55 |
| C: `cargo nextest run --workspace --all-features --release` |                  2.03 |                   2.03 |

The Epic 09 acceptance criterion (≥ 50% reduction of the incremental median)
compares against **Config B incremental median: 5.55 s**.

## 5. Starting LoC

`cobre-sddp/tests/` line count at the start of Epic 09:

```
wc -l crates/cobre-sddp/tests/*.rs
 16591 total  (19 files)
```

## 6. Commit SHA

```
c6379ecd9f38d5aea30cf05a7fc97922d97805d9
```

Branch: `feat/architecture-unification`
Date: 2026-04-22
