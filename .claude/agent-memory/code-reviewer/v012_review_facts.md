---
name: v0.1.2-immediate-fixes review facts
description: Key findings and architecture notes from the v0.1.2 branch code review
type: project
---

Key facts confirmed during review of feat/v0.1.2-immediate-fixes (3 commits):

**B1 fix (inflow truncation in simulation)**: Confirmed correct. `transform_inflow_noise` in `noise.rs`
is now called from the simulation pipeline (was pre-declaration before). The truncation logic correctly
uses `has_negative` to gate the `eta_floor_buf` access, avoiding stale buffer reads.

**B2 fix (per-stage risk measures)**: `RiskMeasure::from(StageRiskConfig)` `From` impl added in
`risk_measure.rs`. Both CLI (`run.rs`) and Python (`run.rs`) now derive `risk_measures` from
`system.stages().filter(id >= 0).map(s.risk_config)` instead of `vec![Expectation; n_stages]`.

**B3 fix (cut selection config)**: `parse_cut_selection_config` added in `cut_selection.rs`.
Bug: guard is `!enabled && config.method.is_none()` — when `enabled = false` but `method = Some(...)`,
the guard is NOT taken and cut selection is activated despite `enabled = false`. Missing test case.
`cut_activity_tolerance` from config is silently discarded (field exists in config, not used in parse).

**B4 fix (FPHA validation)**: `build_stage_templates` now returns `Result<StageTemplates, SddpError>`
instead of `StageTemplates`. FPHA plants are rejected early with the plant name in the error message.

**Epic-02 (StageContext + noise extraction)**:

- `StageContext` and `TrainingContext` structs in `context.rs` reduce parameter count in hot-path fns.
- `transform_inflow_noise` and `transform_load_noise` extracted to `noise.rs` (previously duplicated
  in forward pass and NOT in simulation at all).
- `ScratchBuffers` moved to its own named struct inside `SolverWorkspace` (field: `ws.scratch`).

**Epic-03 (Workspace split + clippy cleanup)**:

- `ScratchBuffers` grouped under `ws.scratch` in `workspace.rs`.
- `clippy::too_many_arguments` and `clippy::too_many_lines` suppressions removed from `run_forward_pass`,
  `run_backward_pass`. These functions now delegate to `run_forward_stage` and `process_stage_backward` helpers.
- `#[allow(clippy::too_many_arguments)]` still present on `train()` — train() still takes ~16 params
  including flat `noise_scale`, `n_hydros`, etc. that should eventually move into a `StageContext` arg.
  This suppression was pre-existing and not addressed in this plan.

**CRITICAL BUG — cobre-python compile error**:

- `crates/cobre-python/src/run.rs` is EXCLUDED from the workspace (`exclude` in root `Cargo.toml`).
- Both `simulate` and `train` call sites in `cobre-python/src/run.rs` use the PRE-REFACTOR API:
  - `simulate` now takes 7 parameters (ctx, fcf, training_ctx, config, output, comm) but the
    Python binding passes 14 parameters (old flat list).
  - `train` now takes `training_ctx: &TrainingContext` in place of `&indexer, &initial_state` but
    the Python binding still passes `&indexer, &initial_state` separately.
- `cargo build --workspace` passes because cobre-python is excluded; the crate won't compile as-is.

**`parse_cut_selection_config` enabled=false + method=Some bug**:

- Logic: `if !enabled && config.method.is_none() { return Ok(None); }`
- When `enabled = Some(false)` and `method = Some("level1")`: guard is `!false && false` = `false`,
  falls through, parses and returns `Ok(Some(Level1{..}))` — cut selection is activated.
- Correct logic should be: `if !enabled { return Ok(None); }` (check enabled first, unconditionally).
- No test covers `enabled = false, method = Some(...)`.

**Lower bound allocation**:

- `vec![uniform_prob; n_openings]` inlined from local variable: `&vec![...]` is idiomatic in Rust
  but slightly surprising. Not a performance regression (was a `let` before, same allocation).
- `assert!` on `n_openings > 0` is a non-debug assert on every LB evaluation call (pre-existing,
  not introduced in this PR — noted in memory already).

**Why**: Record for future reviews of this branch and follow-up tickets.
**How to apply**: When reviewing cobre-python changes, always verify compile separately (excluded from workspace).
