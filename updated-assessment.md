# Cobre v0.1.3 — Progress Assessment

**Date:** 2026-03-14
**Scope:** Code quality audit of feat/v0.1.3-stochastic-io branch against the v0.1.1 post-release assessment findings, CLI interface redesign, pipeline extraction design, simulation progress fix
**Base document:** `cobre-v0.1.1-post-release-assessment.md` (2026-03-13)

---

## Table of Contents

1. [Assessment Scorecard — v0.1.1 Findings Resolution](#1-assessment-scorecard)
2. [Remaining Code Degradation — The Last Mile](#2-remaining-code-degradation)
3. [CLI Interface Redesign — Minimal `run` Command](#3-cli-interface-redesign)
4. [Stochastic Summary Design — Replacing `[stochastic]`](#4-stochastic-summary-design)
5. [Simulation Progress — The Per-Worker Accumulator Bug](#5-simulation-progress)
6. [Pipeline Extraction — `StudySetup` Design](#6-pipeline-extraction)
7. [Consolidated Task List](#7-consolidated-task-list)

---

## 1. Assessment Scorecard

### Critical Bug Fixes (B1-B4) — All Fixed

| # | Issue | Evidence |
|---|-------|---------|
| B1 | Simulation ignores inflow truncation | `simulation/pipeline.rs` calls `transform_inflow_noise` from shared `noise.rs` |
| B2 | CVaR unreachable | Both CLI and Python: `RiskMeasure::from(s.risk_config)` per stage |
| B3 | Cut selection not wired | `BroadcastCutSelection` in broadcast config, passed to `train()` |
| B4 | FPHA silent zero | `build_stage_templates` returns `Err` when any hydro uses `Fpha` |

### Code Quality Refactoring (Q1-Q5) — Nearly Complete

| # | Task | Status |
|---|------|--------|
| Q1 | `StageContext` struct | **Done** — 8 fields, used by forward/backward/simulate |
| Q2 | Noise transformation extraction | **Done** — `noise.rs`, 725 lines with tests, single source of truth |
| Q3 | `ScratchBuffers` separation | **Done** — nested inside `SolverWorkspace` |
| Q4 | `TrainingContext` bundle | **Done** — 5 fields |
| Q5 | Re-enable clippy lint | **Partial** — still suppressed on `train()` |

**Argument count improvement:**

| Function | v0.1.1 | v0.1.3 | Status |
|---|---|---|---|
| `train()` | 25 | 21 | Incomplete — see §2 |
| `run_forward_pass()` | 20 | **7** | Exceeded target |
| `run_backward_pass()` | 20 | **7** | Exceeded target |
| `simulate()` | 22 | **7** | Exceeded target |

### Documentation, Abstractions, Reproducibility, ADRs — Complete

DEC-XYZ references removed (0 in book). Status badges updated. README roadmap rewritten. 4ree example page created. PAR naming aliases with deprecations. Canonical UB summation via `allgatherv`. ADRs 008-011 written.

---

## 2. Remaining Code Degradation — The Last Mile

### 2.1 `train()` Is Still 21 Arguments

`train()` accepts `&TrainingContext` but also takes the 8 raw slices that compose a `StageContext` — then bundles them into a `StageContext` internally at line 311. Both CLI and Python destructure `StageTemplates` into 8 pieces at the call site only for `train()` to re-bundle them.

This is the last instance of the dissolved-context pattern. It will be resolved by the pipeline extraction in §6.

### 2.2 No Other Degradation Found

Systematic audit confirms: noise transformation is single-source (`noise.rs`), forward/backward/simulate are at 7 args each, `ScratchBuffers` is properly separated, CVaR and cut selection are wired, FPHA guard prevents silent failures. The codebase went from "actively degrading" to "one structural fix away from clean."

---

## 3. CLI Interface Redesign — Minimal `run` Command

### 3.1 Current Flags

```
cobre run <CASE_DIR>
    --output <DIR>           Output path override
    --skip-simulation        Train only
    --quiet                  Suppress all output
    --no-banner              Suppress banner only
    --verbose                Increase tracing level
    --export-stochastic      Export internal artifacts
    --threads <N>            Worker thread count
```

Seven flags, accumulated opportunistically. No design pass asked: *what is the minimal set a user needs?*

### 3.2 Flag-by-Flag Audit

**`--skip-simulation`** — `config.json` has `simulation.enabled: true/false`. A CLI flag that overrides config creates two places to control the same thing with an override precedence rule. Remove.

**`--no-banner`** — Subsumed by `--quiet`. The banner is 3 lines and takes zero time. Remove.

**`--verbose`** — Controls `tracing` log level, but no Cobre library crate emits tracing events on the hot path. The `[stochastic]` diagnostics don't respect it. A flag with no defined behavior is worse than no flag. Remove. If structured tracing is needed for HPC debugging later, design it as `COBRE_LOG=debug` following the `RUST_LOG` convention.

**`--export-stochastic`** — Exposes an implementation concept as a user-facing flag. Move to config-only (`exports.stochastic: true`). Remove the CLI flag.

### 3.3 The Minimal `run` Command

```
cobre run <CASE_DIR> [--output <DIR>] [--threads <N>] [--quiet]
```

Three flags. The principle: **`config.json` defines _what_ to compute; CLI flags define _how_ to run it.** A study is fully specified by its case directory. The same case directory produces the same results regardless of which CLI flags are used.

| Concern | Controlled by |
|---|---|
| Simulation on/off | `simulation.enabled` in config.json |
| Stochastic export on/off | `exports.stochastic` in config.json |
| Forward passes, iterations | `training.*` in config.json |
| Cut selection | `training.cut_selection` in config.json |
| Inflow method | `modeling.inflow_non_negativity` in config.json |

| Flag | Controls |
|---|---|
| `<CASE_DIR>` | Where the input is |
| `--output` | Where the output goes |
| `--threads` | How many cores to use |
| `--quiet` | Whether to show progress |

### 3.4 Migration

- `--skip-simulation` → set `simulation.enabled: false` in `config.json`
- `--export-stochastic` → set `exports.stochastic: true` in `config.json`
- `--no-banner` → use `--quiet`
- `--verbose` → set `COBRE_LOG=debug` (future; not implemented yet)

---

## 4. Stochastic Summary Design — Replacing `[stochastic]`

### 4.1 The Problem

The current output uses `eprintln!("[stochastic] ...")` — a homebrew logging prefix that violates every visual pattern the rest of the CLI establishes: no color, no `Term::stderr()`, no bold headers, no indentation hierarchy. It also doesn't scale: the AIC line lists every hydro's AR order by name.

### 4.2 The Existing Visual Language

Three established elements:

**Bold header line:** `Training complete in 5.0s (50 iterations, converged at iter 38)`

**Indented key-value pairs:**
```
  Lower bound:  1.00500e2 $/stage
  Upper bound:  1.05000e2 ± 2.50000e0 $/stage
```

**Blank line** between blocks. All through `Term::stderr()` with `console::style`.

### 4.3 The Replacement Design

A `StochasticSummary` struct in `summary.rs` with a `print_stochastic_summary` function:

```rust
pub struct StochasticSummary {
    pub inflow_source: StochasticSource,
    pub n_hydros: usize,
    pub n_seasons: usize,
    pub ar_summary: Option<ArOrderSummary>,
    pub correlation_source: StochasticSource,
    pub correlation_dim: Option<String>,
    pub opening_tree_source: StochasticSource,
    pub openings_per_stage: usize,
    pub n_stages: usize,
    pub n_load_buses: usize,
    pub seed: u64,
}

pub enum StochasticSource {
    Estimated,
    Loaded,
    None,
}

/// Compact summary of AR order selection across all hydros.
///
/// For small systems (≤10 hydros), the per-hydro order list is readable.
/// For production systems (160+ hydros), only the distribution matters.
pub struct ArOrderSummary {
    /// "AIC" or "fixed"
    pub method: String,
    /// Number of hydros per AR order: index = order, value = count.
    /// E.g., [0, 12, 45, 80, 28] means 12 at order 1, 45 at order 2, etc.
    pub order_counts: Vec<usize>,
    /// Minimum selected order across all hydros.
    pub min_order: usize,
    /// Maximum selected order across all hydros.
    pub max_order: usize,
    /// Total number of hydros with AR models.
    pub n_hydros: usize,
}
```

### 4.4 AR Order Display — Scaling Design

The current code builds `"SUDESTE=2, SUL=1, NORDESTE=3, NORTE=2"`. For 165 hydros, this would be a 1000+ character line that wraps multiple times and is unreadable.

The `ArOrderSummary` adapts its display based on hydro count:

**Small systems (≤10 hydros):** Per-hydro listing, compact form.
```
  Inflows:      PAR estimated, AIC (orders: 2/1/3/2)
```

The orders are listed in the system's hydro index order, separated by `/`. No names — at 4 hydros the user knows which is which. This is already more compact than the current `SUDESTE=2, SUL=1` format.

**Medium systems (11-30 hydros):** Order distribution.
```
  Inflows:      PAR estimated, AIC (orders 1-4, 30 hydros)
```

Min-max range plus count. The user knows the spread without a line that wraps three times.

**Large systems (31+ hydros):** Order distribution with histogram.
```
  Inflows:      PAR estimated, AIC (orders 1-6: 12×p1 45×p2 80×p3 28×p4 3×p5 2×p6)
```

The `N×pK` notation reads as "N hydros at order K." This is the same information density as NEWAVE's PMO report summary for AR orders. At 165 hydros, this line is ~80 characters — fits in one terminal width.

Implementation:

```rust
impl ArOrderSummary {
    pub fn display_string(&self) -> String {
        if self.n_hydros <= 10 {
            // Compact per-hydro listing: "AIC (orders: 2/1/3/2)"
            let orders: Vec<String> = self.order_counts.iter().enumerate()
                .flat_map(|(order, &count)| std::iter::repeat(order).take(count))
                .map(|o| o.to_string())
                .collect();
            format!("{} (orders: {})", self.method, orders.join("/"))
        } else if self.n_hydros <= 30 {
            // Range: "AIC (orders 1-4, 30 hydros)"
            format!(
                "{} (orders {}-{}, {} hydros)",
                self.method, self.min_order, self.max_order, self.n_hydros
            )
        } else {
            // Histogram: "AIC (orders 1-6: 12×p1 45×p2 80×p3 ...)"
            let parts: Vec<String> = self.order_counts.iter().enumerate()
                .filter(|(_, &count)| count > 0)
                .map(|(order, count)| format!("{count}×p{order}"))
                .collect();
            format!(
                "{} (orders {}-{}: {})",
                self.method, self.min_order, self.max_order,
                parts.join(" ")
            )
        }
    }
}
```

### 4.5 The Print Function

```rust
pub fn print_stochastic_summary(stderr: &Term, s: &StochasticSummary) {
    let _ = stderr.write_line(&format!(
        "{}",
        console::style("Stochastic setup").bold()
    ));

    // Inflows line
    let inflow_detail = match s.inflow_source {
        StochasticSource::Estimated => {
            let ar = s.ar_summary.as_ref()
                .map(|a| format!(", {}", a.display_string()))
                .unwrap_or_default();
            format!("PAR estimated ({} hydros, {} seasons{})", s.n_hydros, s.n_seasons, ar)
        }
        StochasticSource::Loaded => format!(
            "loaded ({} hydros, {} seasons)", s.n_hydros, s.n_seasons
        ),
        StochasticSource::None => "none (no hydro plants)".to_string(),
    };
    let _ = stderr.write_line(&format!("  Inflows:      {inflow_detail}"));

    // Correlation line
    let corr_detail = match s.correlation_source {
        StochasticSource::Estimated => {
            let dim = s.correlation_dim.as_deref().unwrap_or("?");
            format!("estimated from residuals ({dim})")
        }
        StochasticSource::Loaded => "loaded from correlation.json".to_string(),
        StochasticSource::None => "identity (default)".to_string(),
    };
    let _ = stderr.write_line(&format!("  Correlation:  {corr_detail}"));

    // Opening tree line
    let tree_detail = match s.opening_tree_source {
        StochasticSource::Loaded => "loaded from noise_openings.parquet".to_string(),
        _ => format!(
            "generated ({} openings/stage, {} stages)", s.openings_per_stage, s.n_stages
        ),
    };
    let _ = stderr.write_line(&format!("  Opening tree: {tree_detail}"));

    // Load noise line (only if present)
    if s.n_load_buses > 0 {
        let _ = stderr.write_line(&format!(
            "  Load noise:   {} stochastic buses", s.n_load_buses
        ));
    }

    // Seed line
    let _ = stderr.write_line(&format!("  Seed:         {}", s.seed));
}
```

### 4.6 Full Terminal Output After Changes

```
 ╺━━━━━━━━━━╻●
 ╺━━━━━━━━━━╻●⚡  COBRE v0.1.3
 ╺━━━━━━━━━━╻●   Power systems in Rust

Loading case: examples/4ree
Stochastic setup
  Inflows:      PAR estimated (4 hydros, 12 seasons, AIC (orders: 2/1/3/2))
  Correlation:  estimated from residuals (4×4)
  Opening tree: generated (10 openings/stage, 60 stages)
  Load noise:   4 stochastic buses
  Seed:         42

Training   ████████████████████████████████████████ 50/50 iter  LB 1.00e2  UB 1.05e2
Training complete in 5.0s (50 iterations, converged at iter 38)
  Lower bound:  1.00500e2 $/stage
  Upper bound:  1.05000e2 ± 2.50000e0 $/stage
  Gap:          4.8%
  Cuts:         480 active / 1200 generated
  LP solves:    36000

Simulation ████████████████████████████████████████ 200/200  mean 1.05e2 ± 2.50e0
Simulation complete in 10.0s (200 scenarios)
  Completed: 200  Failed: 0

Output written to examples/4ree/output/ (1.2s)
```

For a NEWAVE-scale case (165 hydros):

```
Stochastic setup
  Inflows:      PAR estimated (165 hydros, 12 seasons, AIC (orders 1-6: 12×p1 45×p2 80×p3 24×p4 3×p5 1×p6))
  Correlation:  estimated from residuals (165×165)
  Opening tree: generated (20 openings/stage, 120 stages)
  Seed:         12345
```

One line per concern, everything fits in 120 columns.

---

## 5. Simulation Progress — The Per-Worker Accumulator Bug

### 5.1 The Problem

The simulation progress bar was designed to show live cost statistics, and the infrastructure is fully in place:

- `TrainingEvent::SimulationProgress` carries `mean_cost`, `std_cost`, `ci_95_half_width`
- The progress bar handler in `progress.rs` formats them: `"mean: 1.05e2  std: 2.50e0  CI95: ±8.59e2"`
- `WelfordAccumulator` in `simulation/pipeline.rs` computes running statistics

**The bug is in the accumulation site.** Each worker thread has its own `WelfordAccumulator` (line 595):

```rust
let mut worker_acc = WelfordAccumulator::new();  // per-worker
// ...
worker_acc.update(total_cost);
let completed = scenarios_complete.fetch_add(1, Ordering::Relaxed) + 1;  // global atomic
emit_sim_progress(worker_sender.as_ref(), &worker_acc, completed, ...);
```

The `scenarios_complete` counter is a shared `AtomicU32` that counts completions across ALL workers. But `worker_acc` only knows about THIS worker's scenarios. So when worker 3 emits after completing its 5th scenario, the event says `scenarios_complete: 47` (global count from atomic) but `mean_cost` and `std_cost` are from only worker 3's 5 scenarios.

With 24 workers running in parallel, each emission shows a different worker's partial view. The statistics jump erratically on every update because the progress bar receives events from different workers in arbitrary interleaving order. The count advances monotonically (correct), but the mean/std/CI oscillate between workers' partial views (wrong).

### 5.2 The Fix — Move Accumulation to the Consumer

The correct architecture: workers emit per-scenario costs, the progress thread accumulates them in a single Welford accumulator.

**Step 1: Simplify the event.**

Replace the statistics fields in `SimulationProgress` with a single `scenario_cost`:

```rust
SimulationProgress {
    scenarios_complete: u32,
    scenarios_total: u32,
    elapsed_ms: u64,
    /// Cost of the most recently completed scenario.
    ///
    /// The progress thread accumulates these into a single Welford
    /// accumulator to produce global running statistics (mean, std,
    /// CI95). This eliminates the per-worker partial-view bug where
    /// each worker's accumulator only sees its own scenarios.
    scenario_cost: f64,
}
```

This removes `mean_cost`, `std_cost`, and `ci_95_half_width` from the event. These were incorrect anyway (per-worker partials, not global).

**Step 2: Remove per-worker accumulators from the simulation pipeline.**

In `simulate()`, replace:

```rust
let mut worker_acc = WelfordAccumulator::new();
// ...
worker_acc.update(total_cost);
emit_sim_progress(worker_sender.as_ref(), &worker_acc, completed, total, elapsed);
```

With:

```rust
// No worker_acc needed
emit_sim_progress(worker_sender.as_ref(), completed, total, elapsed, total_cost);
```

The `emit_sim_progress` function simplifies to just packaging the event fields.

**Step 3: Move `WelfordAccumulator` to the progress thread.**

In `progress.rs`, the simulation progress handler becomes:

```rust
// At the top of the progress thread, before the event loop:
let mut sim_acc = WelfordAccumulator::new();

// Inside the event loop:
TrainingEvent::SimulationProgress {
    scenarios_complete,
    scenarios_total,
    scenario_cost,
    ..
} => {
    sim_acc.update(scenario_cost);
    let bar = simulation_bar.get_or_insert_with(|| {
        create_simulation_bar(u64::from(scenarios_total))
    });
    bar.set_position(u64::from(scenarios_complete));
    let msg = if sim_acc.count() >= 2 {
        format!(
            "mean {} ± {}",
            fmt_sci(sim_acc.mean()),
            fmt_sci(sim_acc.ci_95_half_width()),
        )
    } else {
        format!("mean {}", fmt_sci(sim_acc.mean()))
    };
    bar.set_message(msg);
}
```

This is the canonical accumulation pattern — same principle as the forward pass reproducibility fix. One accumulator, one consumer, correct global statistics.

**Step 4: `WelfordAccumulator` needs to be shared.**

Currently `WelfordAccumulator` is `pub(crate)` in `simulation/pipeline.rs`. The progress thread is in `cobre-cli/src/progress.rs`. Two options:

**Option A:** Move `WelfordAccumulator` to `cobre-core` alongside `TrainingEvent`. It's a 30-line struct with no dependencies. The progress thread imports it from `cobre-core`.

**Option B:** The progress thread maintains its own inline accumulation (3 variables: count, mean, m2). No import needed.

Option A is cleaner — the type is reusable and the Welford algorithm should have one implementation.

### 5.3 Progress Bar Display Format

The simulation bar template becomes:

```
Simulation {bar:40} {pos}/{len} scenarios  {msg}  [{elapsed_precise} < {eta_precise}]
```

Where `{msg}` shows:

- Before 2 completions: `mean 1.05e2`
- After 2+ completions: `mean 1.05e2 ± 2.50e0`

The `±` value is the CI95 half-width, not the std. Users care about "how certain is this estimate" (CI), not "how spread is the distribution" (std). The std is available in the final printed summary after the bar finishes.

### 5.4 Effort

| Task | Effort |
|---|---|
| Move `WelfordAccumulator` to `cobre-core` | 30 min |
| Simplify `SimulationProgress` event (remove 3 fields, add `scenario_cost`) | 30 min |
| Remove per-worker accumulators from `simulate()` | 15 min |
| Update `progress.rs` to accumulate in the progress thread | 30 min |
| Update all tests that construct `SimulationProgress` events | 1h |
| **Total** | **~3 hours** |

---

## 6. Pipeline Extraction — `StudySetup` Design

### 6.1 The Duplication Problem

The orchestration logic — building stochastic context, stage templates, indexer, initial state, FCF, risk measures, then calling `train()` and `simulate()` — is duplicated between CLI (`run.rs`, ~440 lines) and Python (`run.rs`, ~250 lines). The two copies diverge on entry-point-specific concerns (MPI, progress bars, output writing) but the core sequence is identical. Every new feature had to be implemented in both copies — and wasn't in the Python copy (user opening trees are CLI-only).

### 6.2 The `StudySetup` Design

A struct in `cobre-sddp` that owns all computed study state and provides `train()` and `simulate()` as methods.

```rust
// cobre-sddp/src/setup.rs

/// All precomputed state needed to run an SDDP study.
///
/// Constructed once from a [`System`] and configuration parameters.
/// Provides [`train`](Self::train) and [`simulate`](Self::simulate)
/// methods that consume the minimum set of runtime arguments.
pub struct StudySetup {
    // ── Stochastic ────────────────────────────────────────────────
    stochastic: StochasticContext,
    /// Whether estimation was performed (vs loading pre-computed stats).
    estimation_report: Option<EstimationReport>,

    // ── LP structure ──────────────────────────────────────────────
    templates: Vec<StageTemplate>,
    base_rows: Vec<usize>,
    noise_scale: Vec<f64>,
    n_hydros: usize,
    n_load_buses: usize,
    load_balance_row_starts: Vec<usize>,
    load_bus_indices: Vec<usize>,
    block_counts_per_stage: Vec<usize>,
    max_blocks: usize,
    zeta_per_stage: Vec<Vec<f64>>,
    block_hours_per_stage: Vec<Vec<f64>>,

    // ── Algorithm ─────────────────────────────────────────────────
    indexer: StageIndexer,
    initial_state: Vec<f64>,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
    inflow_method: InflowNonNegativityMethod,
}
```

**Constructor:**

```rust
impl StudySetup {
    pub fn from_system(
        system: &System,
        seed: u64,
        inflow_method: InflowNonNegativityMethod,
        user_opening_tree: Option<OpeningTree>,
    ) -> Result<Self, SddpError> {
        // Single place where the entire orchestration sequence runs:
        // 1. build_stochastic_context
        // 2. build_stage_templates
        // 3. StageIndexer::with_equipment
        // 4. build_initial_state
        // 5. risk_measures from system stages
        // 6. HorizonMode from stage count
    }
}
```

**Internal context borrowing:**

```rust
impl StudySetup {
    fn stage_ctx(&self) -> StageContext<'_> {
        StageContext {
            templates: &self.templates,
            base_rows: &self.base_rows,
            noise_scale: &self.noise_scale,
            n_hydros: self.n_hydros,
            n_load_buses: self.n_load_buses,
            load_balance_row_starts: &self.load_balance_row_starts,
            load_bus_indices: &self.load_bus_indices,
            block_counts_per_stage: &self.block_counts_per_stage,
        }
    }

    fn training_ctx(&self) -> TrainingContext<'_> {
        TrainingContext {
            horizon: &self.horizon,
            indexer: &self.indexer,
            inflow_method: &self.inflow_method,
            stochastic: &self.stochastic,
            initial_state: &self.initial_state,
        }
    }
}
```

**The `train` method:**

```rust
impl StudySetup {
    pub fn train<S: SolverInterface + Send, C: Communicator>(
        &mut self,
        config: TrainingConfig,
        solver: &mut S,
        solver_factory: impl Fn() -> Result<S, SolverError>,
        n_threads: usize,
        comm: &C,
        stopping_rules: StoppingRuleSet,
        cut_selection: Option<&CutSelectionStrategy>,
        shutdown_flag: Option<&Arc<AtomicBool>>,
    ) -> Result<TrainingResult, SddpError> {
        let ctx = self.stage_ctx();
        let training_ctx = self.training_ctx();
        // FCF created internally, capacity from stopping_rules
        // Delegates to internal training::train with &StageContext
    }
}
```

9 arguments. Each represents a genuine caller decision (solver choice, parallelism, communication, stopping criteria). No raw slices.

### 6.3 The `train()` Signature Chain

The internal free function changes:

```
Before: train(..., 21 arguments including 8 raw slices)
After:  train(..., &StageContext, &TrainingContext, ...) → 13 arguments
```

The 8 raw slices become `&StageContext`. `max_blocks` is derived from `ctx.block_counts_per_stage.iter().max()` inside the function.

### 6.4 What the Entry Points Become

**CLI (`run.rs`):**

```rust
pub fn execute(args: RunArgs) -> Result<(), CliError> {
    let comm = create_communicator()?;
    let quiet = args.quiet || comm.rank() != 0;
    let stderr = Term::stderr();

    if !quiet { print_banner(&stderr); }

    // Phase 1: Load (rank 0) + broadcast
    let (system, config) = load_and_broadcast(&args, &comm)?;

    // Phase 2: Setup (all ranks, identical)
    let setup = StudySetup::from_system(&system, seed, inflow_method, user_tree)?;
    if !quiet { print_stochastic_summary(&stderr, &setup.stochastic_summary()); }

    // Phase 3: Train
    let result = setup.train(config, &mut solver, factory, n_threads, &comm, ...)?;
    if !quiet { print_training_summary(&stderr, ...); }

    // Phase 4: Simulate (optional, from config)
    if config.simulation.enabled {
        let sim = setup.simulate(sim_config, &comm)?;
        if !quiet { print_simulation_summary(&stderr, ...); }
    }

    // Phase 5: Write (rank 0 only)
    if comm.rank() == 0 { write_all_outputs(...)?; }
    if !quiet { print_output_path(&stderr, ...); }
    Ok(())
}
```

~60 lines instead of 530. The orchestration sequence is gone — it lives in `StudySetup::from_system`.

**Python (`run.rs`):**

```rust
fn run_inner(case_dir: &Path, ...) -> Result<RunResult, String> {
    let system = load_case(case_dir)?;
    let config = parse_config(case_dir)?;
    let setup = StudySetup::from_system(&system, seed, inflow_method, None)?;
    let result = setup.train(config, &mut solver, factory, n_threads, &LocalBackend, ...)?;
    if config.simulation.enabled {
        let sim = setup.simulate(sim_config, &LocalBackend)?;
        // write results
    }
    Ok(result)
}
```

~40 lines instead of 250. Automatically gets every feature `StudySetup` supports — user opening trees, stochastic export, cut selection — without per-entry-point wiring.

### 6.5 What `StudySetup` Provides Beyond Deduplication

**`stochastic_summary()`** — Has all the data to construct a `StochasticSummary` without threading 7 arguments to a formatting function.

**`export_stochastic(output_dir)`** — Stochastic export becomes a method, callable from any entry point.

**Single validation point** — Study-level invariants (empty templates, zero blocks, FPHA guard) are checked in `from_system`, not in every entry point.

### 6.6 Effort

| Task | Effort |
|---|---|
| Create `StudySetup` struct with `from_system` constructor | 1d |
| Move orchestration logic from CLI into constructor | 1d |
| Add `train()` method; update internal `training::train` to take `&StageContext` | 1d |
| Add `simulate()` method | 0.5d |
| Add `stochastic_summary()` and `export_stochastic()` methods | 0.5d |
| Rewrite CLI `execute()` to use `StudySetup` | 0.5d |
| Rewrite Python `run_inner()` to use `StudySetup` | 0.5d |
| **Total** | **5 days** |

---

## 7. Consolidated Task List

### Phase 1 — CLI Cleanup (1 day)

| # | Task | Effort |
|---|------|--------|
| C1 | Remove `--skip-simulation` flag; document migration to `simulation.enabled` | 30 min |
| C2 | Remove `--no-banner` flag | 15 min |
| C3 | Remove `--verbose` flag and `tracing` subscriber setup | 30 min |
| C4 | Remove `--export-stochastic` flag; keep config-only `exports.stochastic` | 30 min |
| C5 | Remove `format_stochastic_diagnostics` and `print_stochastic_diagnostics` | 15 min |
| C6 | Simplify `BroadcastConfig` (remove `should_simulate`, `export_stochastic` fields) | 30 min |

### Phase 2 — Stochastic Summary + Simulation Progress Fix (1.5 days)

| # | Task | Effort |
|---|------|--------|
| S1 | Create `StochasticSummary`, `StochasticSource`, `ArOrderSummary` in `summary.rs` | 1h |
| S2 | Implement `ArOrderSummary::display_string` with 3-tier scaling | 1h |
| S3 | Implement `print_stochastic_summary` using `Term` + `console::style` | 1h |
| S4 | Build `StochasticSummary` in `execute()` (temporary — moves to `StudySetup` in Phase 3) | 1h |
| S5 | Tests: formatting for each source variant and AR order scale | 1h |
| W1 | Move `WelfordAccumulator` to `cobre-core` | 30 min |
| W2 | Simplify `SimulationProgress` event: remove 3 stats fields, add `scenario_cost` | 30 min |
| W3 | Remove per-worker accumulators from `simulate()` | 15 min |
| W4 | Accumulate in progress thread; display `mean X ± Y` on simulation bar | 30 min |
| W5 | Update all tests that construct `SimulationProgress` events | 1h |

### Phase 3 — Pipeline Extraction (5 days)

| # | Task | Effort |
|---|------|--------|
| P1 | Create `cobre-sddp/src/setup.rs` with `StudySetup` struct | 1d |
| P2 | Implement `from_system()` constructor | 1d |
| P3 | Implement `train()` method; update internal `training::train` to take `&StageContext` | 1d |
| P4 | Implement `simulate()` method | 0.5d |
| P5 | Add `stochastic_summary()` and `export_stochastic()` methods | 0.5d |
| P6 | Rewrite CLI `execute()` to use `StudySetup` | 0.5d |
| P7 | Rewrite Python `run_inner()` to use `StudySetup` | 0.5d |

### Ordering

```
Phase 1 (day 1):     CLI cleanup — remove flags, remove [stochastic] prints
Phase 2 (days 2-3):  Stochastic summary + simulation progress fix
Phase 3 (days 4-8):  Pipeline extraction — StudySetup

After Phase 3:
  train() drops from 21 to 13 args
  CLI execute() drops from 530 to ~60 lines
  Python run_inner() drops from 250 to ~40 lines
  Simulation progress bar shows correct global statistics
  AR order display scales to 165+ hydros
  Codebase is structurally ready for FPHA
```

---

**End of Assessment.**