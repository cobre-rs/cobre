# Phase 8 CLI Design Recommendations

**Date:** 2026-03-08
**Scope:** `cobre-cli` — the binary entry point for the Cobre ecosystem
**Audience context:** Power systems engineers, HPC researchers, energy planning analysts. Not web developers. These users run long batch jobs, operate on HPC clusters, and need tools that feel authoritative and professional.

---

## 1. Design Principles

Three principles from the brand guidelines apply directly to the CLI:

1. **Technical, not trendy.** No emoji in output. No animated spinners where a progress bar suffices. Clean, information-dense output that respects the user's terminal.
2. **Data-dense when needed.** Training runs produce numbers — lower bounds, upper bounds, gap percentages, iteration counts. Show them inline, don't hide them behind `--verbose`.
3. **Dark-first, pipe-aware.** Colors are copper-warm by default on dark terminals. All color and progress output is suppressed when stdout is piped or when `NO_COLOR` is set, per the [no-color.org](https://no-color.org) convention.

---

## 2. Terminal Banner

### Concept

The SVG logo (`assets/cobre-logo-dark.svg`) depicts three horizontal busbars connected by a vertical bar on the left, with node dots on the right and a spark accent. This translates to Unicode box-drawing characters:

```
 ╺━━━━━━━━━╸●
 ╺━━━━━━━━━╸●⚡  COBRE v0.1.0
 ╺━━━━━━━━━╸●   Power systems in Rust
```

Three bars + three bus dots + spark. Version and tagline to the right. Three lines total.

### Color mapping (from brand palette)

| Element | Color | ANSI 256 approximation |
|---------|-------|----------------------|
| Busbars (`━`, `╺`, `╸`) | Copper `#B87333` | 172 |
| Bus dots (`●`) | Copper Light `#D4956A` | 179 |
| Spark (`⚡`) | Spark Amber `#F5A623` | 214 |
| "COBRE" text | Bright `#E8E6E3` | 253 (bold) |
| Tagline | Muted `#8B9298` | 245 |

Use 256-color codes for broad terminal compatibility. Detect truecolor support via `COLORTERM=truecolor` and upgrade to exact hex values when available.

### Display rules

| Context | Show banner? |
|---------|-------------|
| `cobre run` | Yes (unless `--quiet` or `--no-banner`) |
| `cobre simulate` (if separated) | Yes |
| `cobre validate` | No — output is the validation report |
| `cobre report` | No — output is query results |
| `cobre version` | No — version info only |
| stdout is piped | No |
| `NO_COLOR` env var set | Yes, but without color |

### Implementation notes

Hand-craft the banner — don't use ASCII art generators. The box-drawing characters (`━`, `╺`, `╸`, `●`) render consistently across modern terminal emulators (iTerm2, Windows Terminal, GNOME Terminal, Alacritty, kitty). The `console` crate's `Term::stdout().is_term()` detects pipe vs. interactive.

---

## 3. Dependency Stack

Four production dependencies cover everything:

| Crate | Role | Notes |
|-------|------|-------|
| `clap` (derive feature) | Argument parsing, help generation, subcommands | The Rust ecosystem standard. Use derive style exclusively. |
| `indicatif` | Progress bars for training iterations and simulation scenarios | Multi-progress bar support for showing forward/backward phases. Already depends on `console`. |
| `console` | Terminal detection, styled text, color support | Handles `NO_COLOR`, pipe detection, terminal width. Used by `indicatif` internally. |
| `tracing-subscriber` (fmt layer) | Structured logging to stderr | Wires up the `tracing` infrastructure already used by library crates. `--verbose` increases filter level. |

### What NOT to add

| Crate | Why not |
|-------|---------|
| `dialoguer` | No interactive prompts. HPC tools run in batch mode — prompting blocks unattended jobs. |
| `comfy-table` | `cobre report` should emit JSON by default. Pretty tables are a `--format table` option, implementable with simple `format!` or deferred to post-v0.1.0. |
| `clap_complete` | Shell completion generation adds nothing to v0.1.0 UX. Easy to add later. |
| `colored` / `owo-colors` | Redundant with `console` (which `indicatif` already pulls in). |
| `anyhow` / `eyre` | Use `thiserror` for typed errors (consistent with all other crates). Map to exit codes explicitly. |

---

## 4. Subcommand Structure

```
cobre validate <CASE_DIR>
cobre run <CASE_DIR> [OPTIONS]
cobre report <RESULTS_DIR> [OPTIONS]
cobre version
```

### `cobre validate <CASE_DIR>`

Runs the 5-layer validation pipeline (`cobre_io::load_case`). Prints a structured diagnostic report to stdout. No banner, no progress bar — the output IS the deliverable.

Exit codes: 0 = valid, 1 = validation errors found, 2 = I/O error (missing files, permission denied).

### `cobre run <CASE_DIR> [OPTIONS]`

The main entry point. Executes the full lifecycle: load → validate → train → simulate → write.

| Flag | Default | Effect |
|------|---------|--------|
| `--output <DIR>` | `<CASE_DIR>/output/` | Output directory for results |
| `--max-iterations <N>` | From config.json | Override max iterations (convenience for quick tests) |
| `--skip-simulation` | false | Train only, skip simulation phase |
| `--quiet` | false | Suppress banner and progress bars. Errors still go to stderr. |
| `--no-banner` | false | Suppress banner only (keep progress bars) |
| `--verbose` | false | Increase log level (debug-level tracing output to stderr) |

### `cobre report <RESULTS_DIR> [OPTIONS]`

Queries results without re-running. Reads Parquet files and manifests. Default output format is JSON to stdout (machine-readable). Optional `--format table` for human-readable terminal display (post-v0.1.0 scope — for v0.1.0, JSON only is fine).

### `cobre version`

Prints version, solver backend, and build info:

```
cobre 0.1.0
solver: HiGHS 1.x.y
build:  release (lto=thin)
```

Useful for bug reports and HPC environment debugging.

---

## 5. Progress Reporting

### Training progress

One line, updating in place via `indicatif::ProgressBar`:

```
Training ━━━━━━━━━━━━━━━━━━━━╺━━━━  12/50 iter  LB: 45,230.4  UB: 47,100.2 ± 312  gap: 4.1%
```

The bar advances by iteration count. `set_message` updates the statistics (LB, UB, gap) after each iteration's convergence check. When convergence is detected before max iterations, the bar jumps to 100% with a "converged" suffix.

### Simulation progress

```
Simulation ━━━━━━━━━━━━━━━━━━╺━━━  1,247/2,000 scenarios  [00:42 < 01:15]
```

Standard ETA-based progress bar. Advances by completed scenario count.

### Event channel integration

The `TrainingEvent` channel (`mpsc::Sender<TrainingEvent>`) already emitted by the training loop is the data source. The CLI spawns a thread (or uses the main thread) to consume events and update progress bars. No modifications to library crates are needed — the CLI is just another event consumer, alongside `build_training_output`.

Key events to consume:

| Event variant | Progress bar update |
|---------------|-------------------|
| `IterationComplete` | Advance bar by 1, update LB/UB/gap message |
| `ForwardPassComplete` | Optional: update phase indicator |
| `BackwardPassComplete` | Optional: update phase indicator |
| `TrainingFinished` | Finish bar, print summary |

---

## 6. Post-Run Summary

After training and simulation complete, print a compact summary block. This is what users screenshot for reports and presentations.

```
Training complete in 3m 42s (50 iterations, converged at iter 38)
  Lower bound:  45,230.41 $/stage
  Upper bound:  45,410.22 ± 89.3 $/stage  (95% CI)
  Gap:          0.40%
  Cuts:         2,847 active / 10,000 generated
  LP solves:    6,120,000  (avg 4.2 iter/solve with warm-start)

Simulation complete in 1m 15s (2,000 scenarios)
  Mean cost:    45,312.7 $/stage  ± 156.2

Output written to output/
  training/convergence.parquet    (50 records)
  training/_manifest.json
  simulation/costs/               (2,000 partitions)
  policy/cuts/                    (60 stages)
```

Color: section headers in bold, numerical values in default (no color — they may be copied). File paths in muted. The summary goes to stderr (so it doesn't interfere with piped stdout from `cobre report`).

---

## 7. Error Formatting

### Structured errors with actionable hints

Don't dump Rust backtraces. Use `console::style` for colored labels:

```
error: LP infeasible at stage 12, iteration 3, scenario 47
  → check constraint bounds for stage 12 (hydros may have conflicting min/max storage)
  → run `cobre validate path/to/case/` for a full diagnostic report
```

The `→` hint lines are what distinguish a tool from a program. They tell the user what to do next.

### Validation error format

For `cobre validate`, group errors by layer and severity:

```
Validation: 3 errors, 2 warnings in path/to/case/

  Layer 3 — Referential integrity:
    error: hydros.json: downstream_id 999 not found in hydro registry (hydro "Itaipu", id=42)
    error: thermals.json: bus_id 7 not found in bus registry (thermal "Angra I", id=15)

  Layer 5 — Semantic:
    warning: cascade contains hydro "Tucuruí" with zero max_storage — will be treated as run-of-river
```

Entity names in the error messages (not just IDs) make the output immediately useful to the power systems engineer who knows their system by plant name.

---

## 8. Exit Codes

| Code | Meaning | Mapped from |
|------|---------|-------------|
| 0 | Success (training converged or hit max iterations; simulation completed) | `Ok(TrainingResult)` + `Ok(SimulationResult)` |
| 1 | Validation error (case directory failed validation) | `LoadError::ConstraintError` |
| 2 | I/O error (file missing, permission denied, disk full, write failure) | `LoadError::IoError`, `OutputError::Io` |
| 3 | Solver error (LP infeasible/unbounded during training or simulation) | `SddpError::Infeasible`, `SddpError::Solver`, `SimulationError::LpInfeasible` |
| 4 | Internal error (panics, unexpected state, channel failures) | `SddpError::Internal`, catch-all |

Document these in `cobre run --help` and in the software book.

---

## 9. Logging Strategy

### Two output channels

| Channel | Content | Control |
|---------|---------|---------|
| stderr | Banner, progress bars, summary, errors, tracing logs | `--quiet` suppresses banner + progress; `--verbose` increases log level |
| stdout | Reserved for machine-readable output (`cobre report`, `cobre validate --format json`) | Never mixed with progress output |

This separation means `cobre report results/ | jq .lower_bound` works correctly — progress bars on stderr never corrupt the JSON on stdout.

### Tracing integration

Default filter: `warn` for library crates, `info` for `cobre-cli`. With `--verbose`: `info` for library crates, `debug` for `cobre-cli`. This uses `tracing_subscriber::EnvFilter` so users can also override with `RUST_LOG=cobre_sddp=debug` for fine-grained control.

---

## 10. What to Defer to Post-v0.1.0

| Feature | Why defer |
|---------|----------|
| `cobre init` (scaffold case directory) | Useful but not core. The case format spec is documented; users can create directories manually. |
| `cobre compare` (diff two result sets) | Requires a comparison algorithm that doesn't exist yet. |
| Shell completions (`clap_complete`) | Nice-to-have, zero impact on first-run experience. |
| `--format table` for `cobre report` | JSON is machine-readable and sufficient. Pretty tables can come later. |
| `cobre serve` (MCP server mode) | Depends on `cobre-mcp` which is post-v0.1.0. |
| Configuration file for CLI defaults | Premature — let usage patterns emerge first. |
| Internationalization / PT-BR output | English-only for v0.1.0 per brand guidelines. |

---

## 11. File Structure Recommendation

```
crates/cobre-cli/
├── Cargo.toml
├── src/
│   ├── main.rs          # Entry point: parse args, dispatch subcommand, set exit code
│   ├── banner.rs         # Banner rendering with terminal detection
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── run.rs        # cobre run: load → train → simulate → write
│   │   ├── validate.rs   # cobre validate: load_case → print report
│   │   ├── report.rs     # cobre report: read results → print JSON
│   │   └── version.rs    # cobre version: print build info
│   ├── progress.rs       # TrainingEvent consumer → indicatif progress bars
│   ├── summary.rs        # Post-run summary formatting
│   ├── error.rs          # Error formatting with hints, exit code mapping
│   └── logging.rs        # tracing-subscriber setup
└── README.md
```

Each subcommand is a self-contained module that returns a `Result<(), CliError>`. `main.rs` maps `CliError` variants to exit codes and prints formatted errors. No library logic lives in the CLI crate — it's pure orchestration and presentation.
