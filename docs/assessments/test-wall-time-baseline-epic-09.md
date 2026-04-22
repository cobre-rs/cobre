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

---

## 7. Post-epic verification

Date: 2026-04-22
Commit SHA: `f9f8c5aeb86a45da0d63d5dce28000fff1faf15b`
Branch: `feat/architecture-unification`
Machine load average at start: **1.10** (well under 3.0 threshold; safe to proceed)

### Commit

```
f9f8c5aeb86a45da0d63d5dce28000fff1faf15b docs: regenerate test-inventory.md for epic-09 close-out
```

### Current LoC

```
wc -l crates/cobre-sddp/tests/*.rs
 16780 total  (20 files)
```

### Config A — `cargo test --workspace --release`

| Run             | Wall-clock (s) |  Pass | Fail | Ignored | Notes                                                          |
| --------------- | -------------: | ----: | ---: | ------: | -------------------------------------------------------------- |
| 1 (cold)        |          98.67 | 3,612 |    0 |       3 | `cargo clean` immediately before; includes full release build. |
| 2 (incremental) |           4.62 | 3,612 |    0 |       3 | Warm compile cache; pure test-execution wall time.             |
| 3 (incremental) |           4.43 | 3,612 |    0 |       3 | Warm compile cache; pure test-execution wall time.             |

**Median (all 3 runs):** **4.62 s**
**Incremental median (runs 2–3):** **4.525 s**
**Cold run:** **98.67 s**

Note: pass/fail/ignored counts differ from ticket-001 baseline (3,684 pass / 34 ignored) because
some previously active tests are now gated behind `--all-features` (mpi feature flag). Counts
here reflect the non-feature-gated path only.

### Config B — `cargo test --workspace --all-features --release`

| Run             | Wall-clock (s) |  Pass | Fail | Ignored | Notes                                                      |
| --------------- | -------------: | ----: | ---: | ------: | ---------------------------------------------------------- |
| 1 (incremental) |          72.36 | 3,688 |    0 |       3 | No `cargo clean`; partial recompile due to feature change. |
| 2 (incremental) |           5.39 | 3,688 |    0 |       3 | Warm compile cache; pure test-execution wall time.         |
| 3 (incremental) |           5.51 | 3,688 |    0 |       3 | Warm compile cache; pure test-execution wall time.         |

**Median (all 3 runs):** **5.51 s**
**Incremental median (runs 2–3):** **5.45 s**

### Config C — `cargo nextest run --workspace --all-features --release`

| Run             | Wall-clock (s) | Tests run | Passed | Skipped | Notes                                     |
| --------------- | -------------: | --------: | -----: | ------: | ----------------------------------------- |
| 1 (incremental) |           2.14 |     3,419 |  3,419 |      31 | Binaries built by Config B; nextest only. |
| 2 (incremental) |           2.02 |     3,419 |  3,419 |      31 | Warm compile cache.                       |
| 3 (incremental) |           2.09 |     3,419 |  3,419 |      31 | Warm compile cache.                       |

**Median (all 3 runs):** **2.09 s**

### Delta table: pre-epic vs. post-epic

| Config | Pre-epic median (s) | Post-epic median (s) | Absolute delta (s) | Percentage delta |
| ------ | ------------------: | -------------------: | -----------------: | ---------------: |
| A      |                5.53 |                 4.62 |              -0.91 |           -16.5% |
| B      |                5.63 |                 5.51 |              -0.12 |            -2.1% |
| C      |                2.03 |                 2.09 |              +0.06 |            +3.0% |

Incremental medians (runs 2–3):

| Config | Pre-epic inc. median (s) | Post-epic inc. median (s) | Absolute delta (s) | Percentage delta |
| ------ | -----------------------: | ------------------------: | -----------------: | ---------------: |
| A      |                     5.50 |                     4.525 |             -0.975 |           -17.7% |
| B      |                     5.55 |                     5.450 |             -0.100 |            -1.8% |

### LoC delta

| Metric           |       Value |
| ---------------- | ----------: |
| Starting LoC     |      16,591 |
| Current LoC      |      16,780 |
| Absolute delta   |        +189 |
| Percentage delta |       +1.1% |
| File count       | 20 (was 19) |

### AC 2 / AC 3 gap — ESCALATION REQUIRED

**AC 2 — Wall-time reduction ≥ 40%** (measured on Config B incremental median):

- Measured: **−1.8%** (from 5.55 s to 5.45 s)
- Target: ≤ −40%
- **FAIL** — gap is 38.2 percentage points

**AC 3 — LoC reduction ≥ 15%**:

- Measured: **+1.1%** (LoC increased from 16,591 to 16,780)
- Target: ≤ −15%
- **FAIL** — actual direction is opposite (LoC grew)

#### Why the targets were not met

The Epic 09 tickets delivered the following changes to `crates/cobre-sddp/tests/`:

- **Ticket-002**: Gated 39 slow/regression tests behind `#[cfg_attr(not(feature="slow-tests"), ignore)]` and
  added a new `conformance.rs` file. The wall-time reduction from this was ~18% on Config B incremental
  median — but only Config A (without `--all-features`) benefits substantially because Config B re-enables
  those tests via the feature flag. This explains the Config A −17.7% improvement vs. Config B's near-zero
  change.
- **Ticket-003**: Consolidated two files using table-driven tests, reducing individual test functions by 10
  but adding ~49 LoC of scaffolding.
- **Ticket-004**: All tests audited and kept (0 deletions).
- **Tickets 005–006**: No test changes.

Net effect: LoC increased by 189 lines (+1.1%) and the wall-time benefit is only visible when running
without `--all-features` (Config A). The `--all-features` path (Config B, the AC 2 comparison point)
is essentially unchanged because the gated tests re-activate under that feature flag.

#### Escalation options

Per the ticket's Error Handling section, the options are:

**(a) Identify further consolidation** — additional table-driven rewrites or deduplication across the 20
test files. The test inventory (see `docs/assessments/epic-09-test-inventory.md`) identified
conservatively 25–35 more consolidation candidates. Estimated LoC reduction: 300–600 lines (~2–4%),
which would bring total LoC to ~16,180–16,480 but still not hit −15%.

**(b) Move more tests to slow-tests** — extend the `slow-tests` feature gating to Config B's path by
disabling affected tests by default regardless of feature flags (e.g., using `#[ignore]` unconditionally
for heavy regression tests, or restructuring so `slow-tests` is the positive opt-in for Config B too).
This would give a measurable Config B wall-time reduction but requires careful consideration of CI
policy.

**(c) Relax the ≥ 40% target for v0.5.x and document the rationale** — the original targets were set
before the extent of the test suite was fully understood. The Epic 09 work improved the developer
feedback loop (Config A is now 17.7% faster without features), tightened test naming and organization,
gated known-slow tests, and audited all 3,600+ tests for correctness. The infrastructure improvements
(naming, gating, table-driven structure) are real and durable even if the wall-time reduction is modest.
This option accepts the measured improvement as the final state for v0.5.x and defers further
optimization to a future epic.

**Recommendation**: Option (c) — relax the targets for v0.5.x. The gap between Config A and Config B
measurements reveals a structural issue with the AC definition: Config B's `--all-features` re-activates
the slow tests that Ticket-002 gated, making the comparison against Config B incremental median an
unfair measuring stick for that work. The actual developer workflow improvement (Config A, or nextest
without slow tests) is measurable and real. Revisiting the test architecture with a dedicated
`slow-tests` feature that applies consistently across all configs is the right work for a future epic,
not a patch against this one.
