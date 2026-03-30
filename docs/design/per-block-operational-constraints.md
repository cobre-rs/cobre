# Refactor: Per-Block Operational Violation Constraints

## Problem

The four operational violation constraint families (min outflow, max outflow,
min turbined, min generation) are currently formulated as **stage-level
volume/energy aggregates**. Each family has one constraint row and one slack
column per hydro per stage.

This produces non-physical simulation results: the optimizer concentrates all
flow into high-demand blocks and zeros out others, because the constraint only
requires that the _total volume across the stage_ meets the minimum, not the
_instantaneous flow in each block_.

### Observed symptoms

| Hydro                                                 | Constraint            | Block behavior               |
| ----------------------------------------------------- | --------------------- | ---------------------------- |
| 106 SOBRADINHO (min_outflow=800 m3/s)                 | Stage-level satisfied | outflow=0 in blocks 0,2      |
| 33 TRES IRMAOS (min_outflow=165 m3/s)                 | Stage-level satisfied | outflow=0 in multiple blocks |
| 35 P. PRIMAVERA (min_outflow=1881 m3/s, run-of-river) | Stage-level satisfied | outflow=0 in some blocks     |

All `outflow_slack_below_m3s` values are zero because the stage-level constraint
IS satisfied. The per-block violations are invisible to the LP.

### Root cause

In `matrix.rs`, the constraint formulation aggregates across blocks:

```text
Current (stage-level, hm3 units):
  sum_k[ tau_k * (q_k + s_k + d_k) ] + sigma >= min_outflow * zeta
  where tau_k = block_hours[k] * 0.0036, zeta = total_hours * 0.0036
```

One sigma per hydro (stage-level), objective = `penalty * total_stage_hours`.

## Design

Convert all four constraint families to **per-block** formulation: one constraint
row and one slack column per hydro per block per family.

### New constraint formulation

| Family              | Old (stage-level)                                         | New (per-block)                      |
| ------------------- | --------------------------------------------------------- | ------------------------------------ |
| Min outflow (>=)    | `sum_k[tau_k * outflow_k] + sigma >= min_outflow * zeta`  | `outflow_k + sigma_k >= min_outflow` |
| Max outflow (<=)    | `sum_k[tau_k * outflow_k] - sigma <= max_outflow * zeta`  | `outflow_k - sigma_k <= max_outflow` |
| Min turbine (>=)    | `sum_k[tau_k * q_k] + sigma >= min_turbined * zeta`       | `q_k + sigma_k >= min_turbined`      |
| Min generation (>=) | `sum_k[hours_k * gen_k] + sigma >= min_gen * total_hours` | `gen_k + sigma_k >= min_gen`         |

Units are now in m3/s (flow constraints) and MW (generation constraint) --
the natural per-block rate units. No tau/zeta conversion needed.

### Slack objective

Old: `penalty * total_stage_hours` (one sigma per hydro).
New: `penalty * block_hours_k` (one sigma per hydro per block).

The cost contribution for a 1 m3/s violation in block k lasting `hours_k` hours
is `penalty * hours_k`, matching the existing cost convention for per-block
decision variables (spillage, thermals, etc.).

### Column activation logic (unchanged)

Slack column upper bounds remain conditionally set:

- `min_outflow > 0` => upper = +inf, else pinned to 0
- `max_outflow.is_some()` => upper = +inf, else pinned to 0
- `min_turbined > 0` => upper = +inf, else pinned to 0
- `min_generation > 0` => upper = +inf, else pinned to 0

This is applied identically for every block of the same hydro.

## LP size impact

For N hydros and B blocks per stage:

| Region          | Old | New |
| --------------- | --- | --- |
| Slack columns   | 4N  | 4NB |
| Constraint rows | 4N  | 4NB |

Typical case (150 hydros, 3 blocks): 600 -> 1800 columns and rows. These are
extremely sparse rows (1 flow variable + 1 slack each), so impact on HiGHS
solve time is negligible. Presolve will eliminate any row where the slack is
pinned to zero and the flow variable naturally satisfies the bound.

## Files to modify

### 1. `lp_builder/layout.rs` -- StageLayout offsets

Change 4 column regions and 4 row regions from `n_hydros` to `n_hydros * n_blks`:

```rust
// Old
let col_outflow_below_start = withdrawal_slack_end;
let col_outflow_above_start = col_outflow_below_start + ctx.n_hydros;
let col_turbine_below_start = col_outflow_above_start + ctx.n_hydros;
let col_generation_below_start = col_turbine_below_start + ctx.n_hydros;
let operational_slack_end = col_generation_below_start + ctx.n_hydros;

// New
let n_op_slack = ctx.n_hydros * n_blks;
let col_outflow_below_start = withdrawal_slack_end;
let col_outflow_above_start = col_outflow_below_start + n_op_slack;
let col_turbine_below_start = col_outflow_above_start + n_op_slack;
let col_generation_below_start = col_turbine_below_start + n_op_slack;
let operational_slack_end = col_generation_below_start + n_op_slack;
```

Same change for the 4 row regions:

```rust
// Old
let row_min_outflow_start = evap_rows_end;
let row_max_outflow_start = row_min_outflow_start + ctx.n_hydros;
let row_min_turbine_start = row_max_outflow_start + ctx.n_hydros;
let row_min_generation_start = row_min_turbine_start + ctx.n_hydros;
let operational_rows_end = row_min_generation_start + ctx.n_hydros;

// New
let n_op_rows = ctx.n_hydros * n_blks;
let row_min_outflow_start = evap_rows_end;
let row_max_outflow_start = row_min_outflow_start + n_op_rows;
let row_min_turbine_start = row_max_outflow_start + n_op_rows;
let row_min_generation_start = row_min_turbine_start + n_op_rows;
let operational_rows_end = row_min_generation_start + n_op_rows;
```

Update doc comments on the 8 fields to say "per hydro per block" and document
layout as `start + h_idx * n_blks + blk`.

### 2. `indexer.rs` -- StageIndexer ranges

Same change: 4 column ranges and 4 row ranges from `hydro_count` to
`hydro_count * n_blks`.

```rust
// Old
let ob = ws_end..ws_end + hydro_count;
let oa = ob.end..ob.end + hydro_count;
let tb = oa.end..oa.end + hydro_count;
let gb = tb.end..tb.end + hydro_count;

let r_min_out = evap_rows_end..evap_rows_end + hydro_count;
let r_max_out = r_min_out.end..r_min_out.end + hydro_count;
let r_min_turb = r_max_out.end..r_max_out.end + hydro_count;
let r_min_gen = r_min_turb.end..r_min_turb.end + hydro_count;

// New
let n_op = hydro_count * n_blks;
let ob = ws_end..ws_end + n_op;
let oa = ob.end..ob.end + n_op;
let tb = oa.end..oa.end + n_op;
let gb = tb.end..tb.end + n_op;

let r_min_out = evap_rows_end..evap_rows_end + n_op;
let r_max_out = r_min_out.end..r_min_out.end + n_op;
let r_min_turb = r_max_out.end..r_max_out.end + n_op;
let r_min_gen = r_min_turb.end..r_min_turb.end + n_op;
```

The `StageIndexer` constructor already receives `n_blks` (from its
`with_equipment_and_evaporation` constructor). Thread it through.

### 3. `lp_builder/matrix.rs` -- Column bounds

`fill_stage_columns`: change the 4 slack column loops from iterating over
`0..n_h` to iterating over `0..n_h` x `0..n_blks`:

```rust
// Old: outflow-below-minimum slacks
for h_idx in 0..layout.n_h {
    let col = layout.col_outflow_below_start + h_idx;
    // ...
    objective[col] = hp.outflow_violation_below_cost * total_stage_hours;
}

// New: outflow-below-minimum slacks (per block)
for h_idx in 0..layout.n_h {
    let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
    let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
    for blk in 0..layout.n_blks {
        let col = layout.col_outflow_below_start + h_idx * layout.n_blks + blk;
        if hb.min_outflow_m3s > 0.0 {
            col_upper[col] = f64::INFINITY;
        } else {
            col_upper[col] = 0.0;
        }
        let block_hours = stage.blocks[blk].duration_hours;
        objective[col] = hp.outflow_violation_below_cost * block_hours;
    }
}
```

Repeat for outflow-above, turbine-below, generation-below.

### 4. `lp_builder/matrix.rs` -- Row bounds

`fill_operational_violation_rows`: change from `n_h` rows to `n_h * n_blks` rows.
RHS is now in rate units (m3/s or MW), not volume/energy:

```rust
// Old: min outflow rows
for h_idx in 0..layout.n_h {
    let row = layout.row_min_outflow_start + h_idx;
    let rhs = hb.min_outflow_m3s * layout.zeta;
    row_lower[row] = rhs;
    row_upper[row] = f64::INFINITY;
}

// New: min outflow rows (per block)
for h_idx in 0..layout.n_h {
    let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
    for blk in 0..layout.n_blks {
        let row = layout.row_min_outflow_start + h_idx * layout.n_blks + blk;
        row_lower[row] = hb.min_outflow_m3s;
        row_upper[row] = f64::INFINITY;
    }
}
```

For min generation, the RHS is `min_generation_mw` (MW, not MWh).

### 5. `lp_builder/matrix.rs` -- Matrix entries

`fill_operational_violation_entries`: each constraint row now references a single
block's variables with coefficient 1.0 (not tau):

```rust
// Old: min outflow (stage-level, tau coefficients)
let row = layout.row_min_outflow_start + h_idx;
for blk in 0..n_blks {
    let tau = stage.blocks[blk].duration_hours * M3S_TO_HM3;
    col_entries[col_q].push((row, tau));
    col_entries[col_s].push((row, tau));
    col_entries[col_d].push((row, tau));
}
col_entries[col_slack].push((row, 1.0));

// New: min outflow (per-block, unit coefficients)
for blk in 0..n_blks {
    let row = layout.row_min_outflow_start + h_idx * n_blks + blk;
    let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
    let col_s = layout.col_spillage_start + h_idx * n_blks + blk;
    let col_d = layout.col_diversion_start + h_idx * n_blks + blk;
    col_entries[col_q].push((row, 1.0));
    col_entries[col_s].push((row, 1.0));
    col_entries[col_d].push((row, 1.0));
    let col_slack = layout.col_outflow_below_start + h_idx * n_blks + blk;
    col_entries[col_slack].push((row, 1.0));
}
```

For min generation with constant productivity:

```rust
// Old
col_entries[col_q].push((row, rho * block_hours));

// New
col_entries[col_q].push((row, rho));  // gen_k = rho * q_k, constraint is rho*q_k + sigma_k >= min_gen
```

For min generation with FPHA:

```rust
// Old
col_entries[col_g].push((row, block_hours));

// New
col_entries[col_g].push((row, 1.0));  // g_k + sigma_k >= min_gen
```

### 6. `simulation/extraction.rs` -- Slack extraction

The extraction code currently reads one stage-level slack per hydro and divides
by zeta to convert from hm3 to m3/s. With per-block slacks, the value is
already in m3/s:

```rust
// Old
let outflow_slack_below = view.primal[indexer.outflow_below_slack.start + h] / zeta;

// New
let outflow_slack_below = view.primal[indexer.outflow_below_slack.start + h * n_blks + blk];
```

For generation slack, the old code divides by `total_hours` to convert MWh -> MW.
The new code reads MW directly:

```rust
// Old
let generation_slack = view.primal[indexer.generation_below_slack.start + h] / total_hours;

// New
let generation_slack = view.primal[indexer.generation_below_slack.start + h * n_blks + blk];
```

This simplification applies to both the stage-level and per-block extraction
paths in `extract_hydro_stage_level` and `extract_hydro_per_block`.

### 7. `conformance.rs` -- Column family inventory

The conformance test `constraint_inventory_ranges_are_nonempty` validates that
all `StageIndexer` ranges are non-empty. The ranges change size but remain
non-empty, so the test should pass without changes.

The test `column_family_slack_ranges_no_overlap` checks that slack column
families don't overlap. The ranges change offsets but the test is structural,
so it should pass with the new sizes.

### 8. Deterministic tests -- D20, D21

The training policy and simulation results will change because:

- Per-block constraints are tighter than stage-level (they imply the stage-level
  constraint but not vice versa)
- Plants that previously concentrated flow into one block will now be forced to
  spread flow across blocks to meet per-block minimums
- This will increase operational costs (more constrained dispatch)

Expected cost constants (`D20_EXPECTED_COST`, `D21_EXPECTED_COST`) must be
recomputed. The D20 case (which has min_outflow on a hydro) will show the
largest change. The D21 case may also change if its test hydros have active
operational bounds.

## What does NOT change

- **Cut generation**: operational violation slacks are not state variables.
  Benders cuts use duals from state-fixing rows (storage, AR lags), which are
  unaffected. The LP objective changes, which affects cut intercepts, but the
  cut structure is the same.
- **Water balance**: the storage-flow coupling is unchanged.
- **Output schema**: the Parquet writer already emits per-block rows with
  `outflow_slack_below_m3s`, `turbined_slack_m3s`, `generation_slack_mw` fields.
  No schema change needed.
- **Python bindings**: output schema is unchanged, so `cobre-python` needs no
  modification.
- **Forward/backward pass logic**: these operate on the LP template API without
  direct knowledge of constraint internals.
- **Policy checkpoint format**: cuts are stored per stage, not per constraint.
  No format change needed.

## Verification

1. `cargo test --workspace --all-features` passes.
2. D20 and D21 deterministic tests updated with new expected costs.
3. Re-run the `convertido` case and confirm:
   - `outflow_slack_below_m3s > 0` for blocks where `outflow < min_outflow`.
   - `evaporation_violation_cost` decomposition is non-zero where expected.
   - Per-block outflow respects `min_outflow` (within violation penalty tolerance).
4. Conformance tests pass with updated column/row counts.
