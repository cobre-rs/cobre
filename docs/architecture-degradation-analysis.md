# Architectural Degradation Analysis

**Date:** 2026-03-27
**Scope:** Clippy suppressions, function signature budgets, function length, detection tooling

---

## 1. Detection Gap — Real Count is 16, Not 12

`check_suppressions.py` checks each line for both `"allow"` and
`"clippy::too_many_arguments"` on the **same line**. Multi-line `#[allow(...)]`
blocks split across lines are invisible.

**4 hidden production suppressions:**

| File               | Function                | Params | Notes                                |
| ------------------ | ----------------------- | ------ | ------------------------------------ |
| `training.rs:222`  | `train`                 | **15** | Most critical function in the solver |
| `setup.rs:327`     | `from_broadcast_params` | 12     | Setup orchestrator                   |
| `bounds.rs:139`    | `resolve_bounds`        | 12     | Production resolver                  |
| `penalties.rs:139` | `resolve_penalties`     | 10     | Production resolver                  |

These are the **largest** offenders by parameter count, and the script doesn't
see them.

---

## 2. Hot-Path Budget Table — Stale and Violated

The `architecture-rules.md` budget table says `train` is at 14 params. It's
actually at **15** (`max_blocks` was added without updating the table):

| Function                   | Budget | Actual | Suppression         | Status          |
| -------------------------- | ------ | ------ | ------------------- | --------------- |
| `train`                    | 14     | **15** | hidden (multi-line) | **OVER BUDGET** |
| `run_forward_pass`         | 8      | 8      | yes                 | at budget       |
| `run_backward_pass`        | 8      | 8      | yes (multi-line)    | at budget       |
| `simulate`                 | 8      | 8      | no                  | clean           |
| `evaluate_lower_bound`     | 9      | 9      | yes (multi-line)    | at budget       |
| `build_row_lower_unscaled` | —      | 8      | yes                 | not tracked     |

The architecture rules state: _"Never suppress the lint on hot-path functions."_
Yet 4 of 5 tracked functions carry suppressions. These are grandfathered from
when the rules were written, but `train` has actively grown past its budget.

---

## 3. Full `too_many_arguments` Inventory — 16 Production Suppressions

### Refactorable (7) — clear struct extraction opportunities

| Function                          | Params | Proposed Structs                                |
| --------------------------------- | ------ | ----------------------------------------------- |
| `resolve_bounds` (production)     | 12     | `EntitySlices` + `BoundsOverrides`              |
| `resolve_penalties` (production)  | 10     | `EntitySlices` + `PenaltyOverrides`             |
| `ResolvedBounds::new`             | 11     | `BoundsCountsSpec` + `BoundsDefaults`           |
| `with_equipment_and_evaporation`  | 11     | `EquipmentCounts` + `FphaConfig` + `EvapConfig` |
| `resolve_variable_ref`            | 10     | `EntityPositionMaps`                            |
| `fill_generic_constraint_entries` | 10     | `LpMatrixBuffers`                               |
| `with_equipment`                  | 9      | `EquipmentCounts` + `FphaConfig`                |

### Hot-path — needs design discussion (5)

| Function                   | Params | Already uses context structs?                               |
| -------------------------- | ------ | ----------------------------------------------------------- |
| `train`                    | 15     | Yes, but loose params leaked in                             |
| `run_forward_pass`         | 8      | Yes (`StageContext`, `TrainingContext`, `ForwardPassBatch`) |
| `run_backward_pass`        | 8      | Yes (`StageContext`, `TrainingContext`, `BackwardPassSpec`) |
| `evaluate_lower_bound`     | 9      | Yes (`LbEvalSpec`)                                          |
| `build_row_lower_unscaled` | 8      | No — pure computation helper                                |

### Acceptable (4)

| Function                   | Params | Reason                               |
| -------------------------- | ------ | ------------------------------------ |
| `from_broadcast_params`    | 12     | Setup entry point, called once       |
| `ResolvedPenalties::new`   | 9      | Constructor pattern                  |
| `transform_ncs_noise`      | 8      | Borderline, mild extraction possible |
| `iterative_pacf_reduction` | 8      | At threshold, statistically complex  |

---

## 4. `too_many_lines` — 39 Production Suppressions

### Worst offenders

| Function                               | Lines   | File                        |
| -------------------------------------- | ------- | --------------------------- |
| `validate_referential_integrity`       | **742** | `validation/referential.rs` |
| `execute` (CLI run)                    | **503** | `commands/run.rs`           |
| `solve` (HiGHS retry)                  | 392     | `highs.rs`                  |
| `estimate_correlation_with_season_map` | 296     | `par/fitting.rs`            |
| `fill_stage_columns`                   | 261     | `lp_builder.rs`             |
| `StageLayout::new`                     | 241     | `lp_builder.rs`             |
| `resolve_variable_ref`                 | 231     | `generic_constraints.rs`    |

### 4 unnecessary suppressions — thin wrappers that no longer fire

- `validate_semantic_hydro_thermal` (10 lines)
- `validate_semantic_stages_penalties_scenarios` (13 lines)
- `estimate_ar_coefficients` (16 lines)
- `estimate_correlation` (16 lines)

---

## 5. Overall Suppression Landscape — 289 Production

| Category                 | Count  | %   | Assessment                             |
| ------------------------ | ------ | --- | -------------------------------------- |
| Cast-related (`cast_*`)  | 187    | 65% | Legitimate — numeric solver code       |
| `too_many_lines`         | 39     | 13% | Mixed — some splittable, some inherent |
| `needless_pass_by_value` | 17     | 6%  | Mostly PyO3 FFI constraints            |
| `too_many_arguments`     | **16** | 6%  | Actionable — struct extraction         |
| `struct_field_names`     | 9      | 3%  | Domain naming, low priority            |
| Other                    | 21     | 7%  | Various                                |

Concentration: `cobre-sddp` has 120 (42%), `cobre-io` has 54 (19%).

---

## Tackling Strategy

### Tier 1: Fix the Detection (immediate, prevents further silent regression)

1. **Fix `check_suppressions.py`** to handle multi-line `#[allow]` blocks. The
   current line-by-line check misses the 4 worst offenders. This is the
   highest-priority item — you can't manage what you can't measure.
2. **Add `--check too_many_lines`** to the script and establish a baseline.
3. **Update the budget table** in `architecture-rules.md` — `train` is at 15,
   not 14. Add `build_row_lower_unscaled` to tracking.

### Tier 2: Quick Wins (low risk, remove unnecessary noise)

1. **Remove 4 stale `too_many_lines` suppressions** on thin wrappers — they no
   longer fire.
2. **Split `validate_referential_integrity`** (742 lines into ~9 helpers, one
   per entity type). This is pure validation code with no shared mutable
   state — mechanical extraction.

### Tier 3: Struct Extraction (medium effort, kills 7 suppressions)

Introduce small grouping structs for the non-hot-path functions:

- `EntitySlices` / `PenaltyOverrides` / `BoundsOverrides` — fixes
  `resolve_bounds` + `resolve_penalties` (production)
- `EquipmentCounts` + `FphaConfig` + `EvapConfig` — fixes both `StageIndexer`
  constructors
- `EntityPositionMaps` — fixes `resolve_variable_ref`
- `LpMatrixBuffers` — fixes `fill_generic_constraint_entries`
- `BoundsCountsSpec` + `BoundsDefaults` — fixes `ResolvedBounds::new`

These are all non-hot-path, so the risk is low and the refactors are
straightforward.

### Tier 4: Strategic — `train` Signature (design decision needed)

`train` at 15 params is the biggest architectural debt. Three params leaked past
the context struct pattern:

- `cut_activity_tolerance: f64` — should be in `TrainingConfig`
- `n_fwd_threads: usize` — should be in `TrainingConfig`
- `max_blocks: usize` — should be in `TrainingConfig` or `StageContext`

Absorbing these into `TrainingConfig` would drop `train` from 15 to **12**. The
question is whether `from_broadcast_params` (which constructs `TrainingConfig`)
should also be refactored simultaneously.

### Tier 5: Long-Tail Function Length

For the 503-line `execute` and 392-line `solve`, these are orchestration
functions that are long because they coordinate many steps. Splitting them
requires phase-extraction patterns (e.g., `fn training_phase(...)`,
`fn simulation_phase(...)`). Worth doing eventually, but lower priority than the
argument-count work.
