---
paths:
  - "crates/cobre-sddp/**/*.rs"
---

# SDDP Numerical & Algorithm Conventions

Hard-won correctness contracts of the SDDP solver. Each one is a _contract_, not
a style preference: a plausible-looking deviation produces wrong bounds, rejected
warm-starts, or silently understated cuts that still compile and pass most tests.
Verify against the cited code before changing any of them.

## Benders cut sign & subgradient extraction

The FCF stores the **raw subgradient** `∂Q/∂x` as a cut's `coefficients` (it is
_not_ negated at storage). That subgradient is the incoming-state column's
reduced cost **divided** by `col_scale`:
`∂Q/∂x_orig = rc_scaled / col_scale[col]` — divided, not multiplied, because the
pin sets `v_scaled = v_orig / col_scale`. Cut-row construction then negates the
gradient so the LP row reads `−∇·x + θ ≥ intercept`, yielding the Benders cut
`θ ≥ Q(x̂) + π'(x − x̂)`.
Read: `training/backward/duals_extraction.rs` (`extract_duals_from_view`), `cut/fcf.rs`, and
`cut::row::push_scaled_coefficient`, where `batch.values.push(-coeff * d)`
applies the negation.

### The cut intercept dots the trial state through the projection, never positionally

The intercept `Q(x̂) − Σ_j β_j·x̂[dim(j)]` gathers the full trial-state vector
`x̂` (length `StateSpace::n_state`) through the pool's `CutStateProjection`,
pairing projected coefficient slot `j` with global state dimension
`global_state_index(j)`. `CutStateProjection::dot_trial_state` is the single
owner; every backward intercept site routes through it
(`backward/replicated.rs`, `backward/outcome_aggregation.rs`'s
`write_opening_outcome`, which the by-scenario and by-node schedulers share).

A positional `coefficients.iter().zip(x̂)` is the wrong-but-compiling
alternative. It agrees ONLY for the all-enabled identity projection; the moment a
pool drops a region — a `storage:false`/`inflow_lags:false` pool, e.g. a study
whose stages disable inflow-lag cut-state — slot `j` past the gap projects a
global dimension `> j`, so the zip multiplies that coefficient by the WRONG
dimension's value (an anticipated coefficient against a lag entry). The intercept
gains a per-cut roughly constant bias, so the cut sits too high: an invalid,
still-converging bound that, under a state-coupling terminal boundary FCF, drives
`final_lb > final_ub` — a persistent negative gap halting a `gap` rule at a
spurious crossover. Reduces to the positional dot bit-for-bit for an all-enabled
pool (`global_state_index(j) == j`), so no existing full-projection study moves.
Read: `lp/indexer/cut_state_projection.rs` (`dot_trial_state`,
`global_state_index`), `training/backward/replicated.rs`,
`training/backward/outcome_aggregation.rs` (`write_opening_outcome`). Pinned by
`dot_trial_state_gathers_reduced_projection_not_positional` (the gather picks the
anticipated dimension past the dropped lag block) and
`dot_trial_state_all_enabled_matches_positional_zip` (byte-neutral for the
identity projection), both in `lp/indexer/cut_state_projection.rs`.

## State pinning uses column bounds, not equality rows

Incoming state is pinned with `set_col_bounds` on the incoming-state LP column;
there is no state-fixing row range in the LP. Always resolve the LP column —
for both pinning and dual extraction — via
`StateSpace::state_to_lp_incoming_column`; never assume a fixing-row index.
Read: `lp/indexer/state_space.rs`.

## Inflow noise enters only through the z rows

The inflow-noise patch touches hydro `h`'s z-inflow row and, under `truncation`
and `truncation_with_penalty` (the truncation methods), only the column bounds
of hydro `h`'s own inflow-slack column (`apply_inflow_slack_col_bounds`,
divided by `col_scale`), never a water-balance row. Every water-balance row
instead reads the deterministic column `z_h` through
`push_local_inflow_coupling`: at `−ζ` on a parallel stage, and at `−τ_k` (block
`k`'s duration hours times `M3S_TO_HM3`) on each chronological block row — on
the hydro's own water-balance row(s), or its `PreFilling` short-circuit
target's. The same router gives the hydro's inflow-slack column, when the
method has one, exactly `z_h`'s water-row entries: the target's rows for a
`PreFilling` hydro, no row at a sink, never a frozen identity row. The water
rows themselves carry no lag, base, or patch of their own; re-encoding the
inflow there routes a chronological hydro's noise onto another hydro's block
row.

Read: `lp/builder/entries.rs` (`push_local_inflow_coupling`),
`stochastic/noise.rs` (`transform_inflow_noise`). Pinned by
`chronological_inflow_noise_moves_only_its_own_hydro`
(`tests/chronological_inflow_noise.rs`, the z-row part: method `none`) and
`every_noise_dimension_patches_only_its_own_entity`
(`tests/patch_ownership_sweep.rs`), whose inflow ownership set is the z row
under `none`/`penalty` and the z row plus that column's bounds under the
truncation methods, and `inflow_slack_follows_z_routing_for_prefilling`
(`lp/builder/entries.rs`) for the slack route.

## Truncation keeps the PAR state raw; the correction is a bounded column

η is never clamped. The z-inflow RHS is `b + σ·η` (0 without a PAR model), so z
carries the raw PAR value `r = b + Σψ·â + σ·η` at this solve's pinned incoming
lags `â`, and `r` may be negative. Outgoing lag 0 is built from the raw z
primal — directly, or through the multi-resolution lag accumulator — with no
clamp, so it may be negative too (`assemble_outgoing_state` never clamps inflow
lags).

Under the truncation methods (`truncation` and `truncation_with_penalty`) a
negative `r` is covered by a separate column, the correction, never by changing
z. The correction is the hydro's own inflow-slack column. Whenever the family
exists it carries exactly z's water-row entries (`push_local_inflow_coupling`);
only under the truncation methods does it also carry `+1` in the generic
`hydro_inflow` term (`push_local_inflow_rate`). Under the truncation methods its
template bounds are `[0, 0]` (`fill_inflow_slack_columns`) and every solve
patches them to the truncation amount `l* = max(−r(â, η), 0)`, written as `+0.0`
when `r ≥ 0`: `[l*, l*]` at cost 0 under `truncation`, `[0, l*]` at `c_inf·T`
(the hydro's `inflow_nonnegativity_cost` times the stage hours) under
`truncation_with_penalty`. `r` comes from `evaluate_par_batch` at
`current_state[inflow_lags]` (this solve's pinned lags) and this solve's noise.
`transform_inflow_noise` writes the bounds, and `apply_inflow_slack_col_bounds`
applies them divided by `col_scale`, identically at every solve site through
`StageSolvePrep`.

The wrong-but-compiling alternatives:

- clamping η (raising it to the value that makes `r = 0` before the solve)
  censors the PAR state: z, the outgoing lags and every later `r` would be
  computed from the clamped value;
- computing `l*` from a precomputed scenario path instead of the pinned lags
  makes `l*` ignore the trial state this solve pins;
- an unbounded slack under `truncation_with_penalty` can add more than the
  truncation amount, buying water the PAR value never lacked;
- an unscaled bound write mis-sizes `l*` by the column's `col_scale`;
- a non-zero cost on the `truncation` column adds `c_inf·T·l*` to every
  objective and cut.

Known limitation (Larroyd et al. (2022), the paper's §2.5): `l*` depends on the
trial state, so a cut taken with the correction frozen at `l*(x̂_k, ω)` can
exceed the true successor cost at any `x` where `l*(x, ω) ≠ l*(x̂_k, ω)` — where
truncation is deeper, and, when the extra water must be spilled at a cost or
against a spillage bound, also where it is shallower. The lower bound is then
not guaranteed to stay below the upper bound, and a `gap` rule (admitted only
under enumerated forward selection, `reject_gap_under_sampled_selection`) can
report false convergence because it reads `(UB − LB).max(0.0)`. The limitation
is not remedied. Each stored cut stays a valid minorant of the successor value
with the slack bounds frozen at its own trial state, which is what the cut
oracles assert (`check_pool` in `tests/common/cut_oracles.rs`), so validity is
checked on every truncation deck with no exclusion.

Read: `stochastic/noise.rs` (`transform_inflow_noise`,
`apply_inflow_slack_col_bounds`), `training/stage_solve_prep.rs`
(`StageSolvePrep::run`), `lp/builder/columns.rs` (`fill_inflow_slack_columns`),
`lp/builder/entries.rs` (`push_local_inflow_coupling`),
`lp/builder/generic_constraints.rs` (`push_local_inflow_rate`),
`stochastic/inflow_method.rs` (`truncates`, `slack_is_priced`),
`solve/stage_solve.rs` (`assemble_outgoing_state`).

Pinned by:

- `test_transform_inflow_noise_truncation_writes_the_raw_rhs_and_fixes_the_correction`
  (`stochastic/noise.rs`): the z RHS is never clamped and the bounds are
  `[l*, l*]` — forbids clamping η;
- `truncation_writes_the_raw_rhs_and_fixes_the_slack_at_the_correction`
  (`training/forward/tests.rs`): the forward solve site writes the raw RHS and
  fixes the slack at `l*`;
- `test_transform_inflow_noise_truncation_with_penalty_caps_the_correction`
  (`stochastic/noise.rs`): the bounds are `[0, l*]` — forbids an unbounded
  slack;
- `truncation_slack_bound_is_divided_by_col_scale` (`stochastic/noise.rs`):
  forbids an unscaled bound write;
- `inflow_slack_index_buffer_rebuilds_on_start_change_at_equal_length`
  (`stochastic/noise.rs`): the patch addresses the current stage's slack
  columns;
- `inflow_slack_follows_z_routing_for_prefilling` (`lp/builder/entries.rs`): the
  slack column carries z's water-row entries;
- `inflow_slack_objective_and_template_bounds_follow_the_method`
  (`tests/inflow_nonnegativity.rs`): cost 0 and template `[0, 0]` under
  `truncation` — forbids a non-zero cost;
- `hydro_inflow_counts_the_truncation_correction_only_under_truncation_methods`
  (`tests/hydro_inflow_travel_time.rs`): `+1` in `hydro_inflow` only under the
  truncation methods;
- `every_noise_dimension_patches_only_its_own_entity`
  (`tests/patch_ownership_sweep.rs`): noise changes only the hydro's own slack
  bounds;
- `inflow_slack_bounds_for_probe_evaluates_the_given_state`
  (`tests/cut_oracles.rs`): `l*` follows the pinned lags of the given state —
  forbids a precomputed scenario path;
- `frozen_inflow_slack_bounds_reach_the_successor_lp` (`tests/cut_oracles.rs`):
  patched slack bounds reach the solved LP;
- `cut_oracles_hold_on_every_committed_deck_with_lag_or_anticipated_state`
  (`tests/cut_oracles.rs`): each stored cut is a minorant of its own
  frozen-correction successor value;
- `simulation_truncation_keeps_the_raw_rhs_and_fixes_the_slack_bounds`
  (`simulation/pipeline/tests.rs`): the simulation solve site writes the raw RHS
  and fixes the slack bounds at `l*`;
- `truncation_keeps_the_raw_par_value_as_the_next_lag`
  (`tests/inflow_nonnegativity.rs`): the next stage's incoming lag 0 is the raw,
  negative PAR value of the stage before — forbids clamping η;
- `truncation_slack_is_fixed_at_the_negated_raw_inflow`
  (`tests/inflow_nonnegativity.rs`): in a solved LP the `truncation` slack equals
  `max(−z, 0)` on every record;
- `truncation_handles_zero_sigma` (`tests/inflow_nonnegativity.rs`): with `σ = 0`
  the slack still equals `−z` for a negative deterministic PAR value;
- `truncation_with_penalty_slack_is_capped_by_the_truncation_amount`
  (`tests/inflow_nonnegativity.rs`): the solved slack never exceeds `l*`, and
  stays below it where stored water is cheaper than the penalty — forbids an
  unbounded slack;
- `truncation_with_penalty_slack_is_zero_for_non_negative_inflow`
  (`tests/inflow_nonnegativity.rs`): the slack is zero where `z ≥ 0`, while the
  uncapped `penalty` twin buys water there — forbids an unbounded slack;
- `truncation_slack_bounds_are_identical_across_solve_sites`
  (`tests/patch_ownership_sweep.rs`): the lower-bound and forward root LPs carry
  the same slack bounds, and a later stage's bounds follow the lags pinned for
  that solve, not the initial state — forbids a precomputed scenario path;
- `truncation_out_of_lp_par_value_matches_the_z_row` (`tests/hydro_sim.rs`): `r`
  from `evaluate_par_batch` equals the solved z, including under the
  multi-resolution lag accumulator;
- `truncation_lag_sequence_is_order_invariant`
  (`tests/template_permutation.rs`): the patched bounds, the slack, the inflow
  and the lags are bit-identical under a permuted hydro declaration order and
  across runs;
- `simulated_water_balance_closes_from_outputs` and
  `simulated_water_balance_closes_with_a_nonzero_release` (`tests/hydro_sim.rs`):
  the written hydros columns close the water balance with the raw inflow plus
  the slack as water added;
- `cut_oracles_resolve_frozen_corrections_on_the_negative_draw_par1_study` and
  `cut_oracles_hold_with_sloped_cuts_on_the_negative_draw_water_value_study`
  (`tests/cut_oracles.rs`): the frozen-correction comparison runs its re-solve
  branch, which needs a negative draw, and holds on sloped cuts.

## FPHA uses average storage

The FPHA generation constraint is
`g ≤ γ₀ + (γᵥ/2)·(V_in + V_out) + γ_q·q (+ γ_s·s)`. The `−γᵥ/2` coefficient
appears on **both** the incoming and outgoing storage columns — not on `V_out`
alone. (Discovered during deterministic case D06.)
Read: `lp/builder/entries.rs` (`fill_fpha_entries` — pushes `−γᵥ/2` onto both the
incoming- and outgoing-storage columns), `lp/builder/rows.rs` (`fill_fpha_rows`),
and `lp/builder/template.rs`.

## Parallel evaporation is one stage-level slot

On a parallel stage each evaporating hydro has **one** evaporation slot, on the
stage endpoints `(S⁰, Sᴷ)`, coupled into the single water-balance row with
`+ζ`; its violation slacks (`f_evap_plus`/`f_evap_minus`) are priced at the
violation cost times the **total stage hours**
(`stage.blocks.iter().map(|b| b.duration_hours).sum::<f64>()`). A chronological
stage keeps one slot per block, each on that block's own `(Sᵏ⁻¹, Sᵏ)`, priced at
that block's own hours. `evaporation_slot_count(block_mode, n_blks)` is the
single owner of the slot count (`1` parallel, `n_blks` chronological); every
column/row family and the generic-constraint resolver derive their stride from
it — no consumer keeps a `* n_blks` evaporation stride on a parallel stage.

Allocating one evaporation slot per BLOCK on a parallel stage is the
wrong-but-compiling alternative: every extra slot beyond the first is a
decoupled variable (no water-row or objective term ties it to anything), and
the one coupled slot's violation slack is still priced at only ONE block's
hours while its flow moves the WHOLE stage's water — understating the true
violation cost by a factor of `K`.

Read: `lp/builder/layout.rs` (`evaporation_slot_count`), `lp/builder/columns.rs`
(`fill_evaporation_columns`).
Pinned by `parallel_multi_block_evap_slack_price_matches_water_it_moves` and
`parallel_multiblock_evaporation_study_has_one_priced_stage_slot`.

## A generic-constraint slack is priced by its declared basis, never by its terms

Each soft generic-constraint row prices its slack columns with
`SlackConfig::unit_cost(row_hours)`, where `row_hours` is the stage's total
hours for a stage-level row and the block's hours for a block row.
`PenaltyBasis::PerHour` gives `penalty × row_hours`, for a rate row (MW, m³/s).
`PenaltyBasis::PerUnit` gives `penalty` with no hours, for an amount row (MWh,
hm³). This is the same no-hours convention as the filling-target and
storage-floor slacks (`fill_filling_target_columns`,
`fill_filled_min_storage_floor_columns` in `lp/builder/columns.rs`). The LP
objective (`fill_generic_constraint_entries`) and the simulation's reported
`slack_cost` (`extract_generic_violations`) both call `unit_cost` with the same
`row_hours`, so the reported cost always matches the LP's price.

`PerUnit` is charged once per violated LP row. A block-independent row whose
bound or coefficient uses a per-(stage, block) parameter is split into one row
per block (`constraint_collapses_to_stage_level`), and each violated block row
is charged `penalty`. This is the intended meaning: each block row is its own
constraint with its own bound. Scaling a split row by its block's share of the
stage would require reading the row's terms, which this contract forbids.

The user declares the basis; `cobre-io` rejects an enabled slack without one.
Inferring it from the row's terms is the wrong-but-compiling alternative. An
amount can be written with flow terms, for example a volume row built from
hours-weighted turbined and spilled flow. Adding or removing one term would
then silently change what the penalty means and rescale the slack price by the
stage hours, which still compiles and still converges.

Read: `crates/cobre-core/src/constraints/generic_constraint.rs`
(`SlackConfig::unit_cost`, `PenaltyBasis`), `lp/builder/entries.rs`
(`fill_generic_constraint_entries`), `simulation/extraction.rs`
(`extract_generic_violations`), `crates/cobre-io/src/constraints/generic.rs`
(`resolve_slack`). Pinned by `per_unit_storage_slack_price_ignores_stage_hours`
(`tests/template_integration/generic_constraints.rs`, objective coefficient on
720 h and 2160 h stages),
`per_unit_storage_shortfall_costs_the_same_on_a_week_and_a_month` and
`per_hour_storage_shortfall_cost_scales_with_stage_hours` (`tests/hydro_sim.rs`,
reported cost on 168 h and 744 h stages), and
`generic_constraint_collapsed_slack_priced_by_total_stage_hours` (a stage-level
`per_hour` row prices at the total stage hours).

## Hydro-cell aggregation assumes one production map per cell

`HydroCellIndex` partitions a plant's `unit_groups` into `bus_id`-equivalence
cells; a same-bus group pair's bounds sum exactly into one cell's LP columns
only because every group sharing a cell also shares the plant's production
map and objective coefficients. `HydroGenerationModel` is a field on `Hydro`,
never on `HydroUnitGroup`, and the resolved coefficients
(`ResolvedProductionModel`, `FphaPlane`'s `gamma_v`/`gamma_q`/`gamma_s`) are
keyed `[hydro][stage]` (`ProductionModelSet::model`) — there is no group or
cell axis to key on, and `HydroUnitGroup` itself carries no productivity,
efficiency, or cost field. That is what makes a same-bus group pair a
segment (one shared production ray) rather than a 2-D zonotope, so summing
member bounds is exact for the turbined-flow and generation-MW box
constraints considered independently — see the fold-order sub-contract below
for the one place that independence breaks down.

A per-group productivity `ρ_g` would break this: pricing the cell at
`ρ_cell = max_g ρ_g` lets the LP draw the efficient unit's MW from the
inefficient unit's water, understating cost — an invalid lower bound that
still converges and still looks plausible. The fix, if a per-group
production field is ever introduced, is to widen the cell partition key to
`(bus_id, production-coefficient signature)`; partitioning by `bus_id` alone
would then silently misprice any mixed-productivity cell.

Read: `crates/cobre-sddp/src/production/hydro_models/types.rs`
(`ProductionModelSet::model`), `crates/cobre-core/src/entities/hydro.rs`
(`HydroGenerationModel` on `Hydro`, absent from `HydroUnitGroup`),
`crates/cobre-sddp/src/lp/indexer/hydro_cell.rs` (`HydroCellIndex::build`).
Pinned by `production_model_set_model_returns_correct_variant` (the
`(hydro, stage)` lookup has no group or cell dimension to key on) and
`test_multi_bus_plant_splits_into_bus_ordered_cells` (partitioning depends on
`bus_id` alone, blind to differing group bounds).

### ConstantProductivity's bound fold must run per group, then sum the cell

A `ConstantProductivity` plant has no separate generation column: its MW cap
folds into the turbine bound as
`min(max_turbined_m3s, max_generation_mw / ρ)` (`fill_turbine_columns`,
`lp/builder/columns.rs`). That fold is exact for a single-group cell; once it
resolves per cell instead of per plant, the fold ORDER becomes load-bearing
the moment a cell holds more than one group. The correct cell bound is
**fold-then-sum** — each group's own `min(q̄_g, p̄_g / ρ)` computed first, then
summed over the cell's groups — because each group is independently limited
by whichever of its own two caps binds first. **Sum-then-fold**
(`min(Σ_g q̄_g, (Σ_g p̄_g) / ρ)`) is the wrong-but-compiling alternative: `min`
does not distribute over a sum of independent terms
(`min(Σa, Σb) ≥ Σ min(a, b)`, strict whenever the binding side — flow-limited
or MW-limited — differs across the cell's groups), so it silently overstates
the cell's true capacity, producing an invalid, too-loose bound that still
converges: with `ρ = 1` and groups `(q̄=100, p̄=50)` and `(q̄=10, p̄=100)`,
fold-then-sum gives `50 + 10 = 60` while sum-then-fold gives
`min(110, 150) = 110` — with one shared `ρ` and no per-group productivity at
all.

A bound multiplier applied to BOTH `q̄_g` and `p̄_g` identically (a shared
availability derate, a per-unit nominal-capability scalar) commutes with the
per-group fold (`min(k·q̄_g, k·p̄_g/ρ) = k·min(q̄_g, p̄_g/ρ)` for `k > 0`), so
fold-then-sum stays exact under any number of such multipliers. A multiplier
on only one side of the pair (an MW-only forced-outage derate against a
fixed mechanical flow limit) does not commute the same way: it can flip
which side binds for a group at a given stage — harmless under
fold-then-sum, which re-folds independently per group regardless, but it
makes sum-then-fold's overstatement vary stage-to-stage instead of vanishing.

Live: `cell_max_turbined` (`fill_turbine_columns`, `lp/builder/columns.rs`)
resolves this bound per CELL, folding each of the cell's own member groups
before summing them, exactly as this sub-contract requires — each group's
`q̄_g`/`p̄_g` is its RESOLVED per-block value (the override when the study
supplies one, the declaration otherwise, via `GroupBoundLookup`), never the
bare declared value.

Read: `crates/cobre-sddp/src/lp/builder/hydro_state.rs` (`cell_max_turbined`).
Pinned by `test_same_bus_groups_sum_into_one_cell_box`, mutation-verified
against sum-then-fold on a two-group fixture whose groups bind on opposite
sides.

### Both terms of the cell bound's closing `min` are load-bearing

`cell_max_turbined`/`cell_max_generation` close with `sum.min(hb...)`: `sum`
folds/sums the cell's OWN member groups; `hb...` is the plant's resolved
bound. Neither term is a redundant guard over the other — each dominates a
disjoint regime, and dropping either one compiles and passes today's
single-group fixtures.

Drop the plant term (`min` degenerates to `sum`) and a lowering
`hydro_bounds` override on a single-cell plant is silently discarded: a
mid-horizon capacity cut, declared exactly the way the no-raising rule's own
rejection message prescribes (declare the plant at final capacity, tighten the
earlier stages with override rows), stops reaching the LP the moment the plant
declares more than one group. A single-cell plant has no cap row (the section
below), so its column bound is the only carrier of the plant maximum.

Drop the group term (`min` degenerates to `hb...`) and a cell can turbine or
generate past its own groups' capacity: the plant cap row below bounds only the
cells' total, never one cell. Three independent, fully-valid-input mechanisms
reach the group term, the third strictly inside a single cell:

- **Cell subsetting.** A two-bus plant's cell sums only its own bus's groups,
  necessarily less than the plant total whenever the other bus's groups are
  nonzero.
- **Rule 41 slack.** The declaring-plant sum check is `Σ_g g.max_* ≤
declared`, not `=` — groups summing to less than the declared value satisfy
  it.
- **Fold-then-sum vs. raw-sum, on a SINGLE cell.** Rule 41 checks the RAW
  group sum; `cell_max_turbined` checks the FOLD-then-sum. These can diverge
  even at rule-41 EQUALITY with no override at all, because "one cell" is not
  "one group": with ρ = 1, a plant declaring `(110, 150)` and two SAME-BUS
  groups `(q̄ 100, p̄ 50)` and `(q̄ 10, p̄ 100)` satisfies rule 41 exactly on
  both columns (100+10=110, 50+100=150), yet the folded group side is
  `min(100,50) + min(10,100) = 60` against the plant's own
  `min(110, 150) = 110`. The group term binds by 50 m³/s with no override, no
  cell split, and no rule-41 slack — the same non-distributivity that
  motivates fold-then-sum over sum-then-fold, surfacing on the OTHER side of
  the `min`.

The plant term collapses to a no-op only for a plant with **no declared
groups** (the implicit single group mirrors the plant's declared value
exactly, and the fold is monotone in both its inputs) — never merely "one
cell", which a same-bus multi-group plant also has while still hitting the
third mechanism above. This is inert on TODAY'S fixtures, not provably inert:
rule 41 and the no-raising rule both admit `value ≤ declared +
ENVELOPE_TOLERANCE`, so even a no-declared-groups plant's resolved value may
sit up to that tolerance above declared — the plant term could tighten by
that same margin. No shipped fixture exercises this; do not round it up to
"provably inert."

Read: `crates/cobre-sddp/src/lp/builder/hydro_state.rs` (`cell_max_turbined`,
`cell_max_generation`), `crates/cobre-io/src/validation/semantic/block_bounds.rs`
(`check_bound_raises_declared_capacity`, the no-raising rule),
`crates/cobre-io/src/validation/semantic/hydro.rs` (rule 41). Pinned by
`test_same_bus_groups_sum_into_one_cell_box`'s third plant (a same-bus pair at
rule-41 equality, no override), which pins the group term binding, and by
`test_cell_columns_take_their_own_group_box`'s block-2 override, which pins
the plant term binding.

### A plant maximum below its cells' caps binds their total in one row

The per-cell `sum.min(hb...)` clamp above cannot hold a multi-cell plant's
total: each cell is clamped to the plant maximum `P` alone, so `|C|` cells can
reach `|C|·P` together. `plant_cap_rhs` (`lp/builder/hydro_state.rs`) owns the
decision: a plant with two or more cells gets one hard `≤ P` row per `MaxCap`
kind and block, without a slack, where
`P + envelope_tolerance(P) < Σ_c cap_c` — the tolerance rule 41 admits a declared
maximum with, so a declared maximum never changes the LP and every study without
a lowering override builds byte-identical rows. `cap_c` is the largest value the
cell's term on that row can take: `cell_max_turbined` for `Turbined`;
`ρ · cell_max_turbined` (`ConstantProductivity`) or `cell_max_generation`
(FPHA) for `Generation`. The kinds are decided independently. The row's LHS is
`Σ_c` of the cell's production terms, the same terms its floor row carries
(`push_cell_power_terms`, the one owner of both): the turbine column, or the FPHA
generation column, or `ρ·q_c`, or nothing for a dormant plant. A single-cell
plant gets no row; its column bound already clamps to `P`. A plant suspended at
the stage (`PreFilling`/`Filling`, `is_suspended`) gets no row, since its
turbines are pinned at 0.

A plant joins `plant_cap_hydro_indices` for a stage when either kind binds at
any block. Both families are sized `n_caps * n_blks`
(`plant_max_turbine`/`plant_max_generation` in `StageGeometry`), and a row that
does not bind at its block is free (`+∞`).

The wrong-but-compiling alternatives:

- relying on the per-cell clamp alone: a two-bus plant whose `P` is lowered
  below each cell's own cap turbines `2P`. Every column bound holds, so nothing
  fails, and the cost is understated (D51);
- a cap row with a slack column: a priced slack lets the plant buy its way past
  a capacity that is physical, unlike the floors, which are soft;
- binding wherever `P < Σ_c cap_c` with no tolerance: a declared maximum at
  rule 41's tolerance then adds a row and moves every such study's LP;
- summing the raw group maxima for `cap_c` instead of the cell's own column
  bound (the fold and the clamp): the decision then disagrees with what the
  columns can carry, adding a row where the cells already fit under `P` or
  skipping one where they do not.

Read: `crates/cobre-sddp/src/lp/builder/hydro_state.rs` (`MaxCap`,
`plant_cap_rhs`, `is_suspended`), `lp/builder/layout.rs`
(`identify_plant_cap_hydros`, `StageLayout::plant_max_row`), `lp/builder/rows.rs`
(`fill_plant_cap_rows`), `lp/builder/entries.rs` (`fill_plant_cap_entries`,
`push_cell_power_terms`). Pinned by
`test_plant_cap_holds_the_cells_total_at_the_plant_maximum` (a solve: the total
stays at `P` and the shortfall moves to the floor slack; mutation-verified against
a free cap row), `test_plant_cap_below_the_cells_gets_a_cap_row_free_where_they_fit`
(the per-kind, per-block decision),
`test_plant_cap_the_cells_fit_under_leaves_the_stage_lp_unchanged` (the shared
tolerance), `test_single_cell_plant_clamps_its_column_instead_of_a_cap_row`,
`test_suspended_plant_gets_no_cap_row`, the
`*_cap_rows_sum_their_cells_production_terms` pair (constant, FPHA) and D51
(`d51_split_plant_two_bus_converges_to_known_optimum`: the simulated stage-0
plant totals stay within the lowered envelope and reach it).

### The per-cell floor is a plain sum; the plant floor adds only where it is tighter

`cell_min_turbined`/`cell_min_generation` (`lp/builder/hydro_state.rs`) are the
MIN-side mirror of `cell_max_turbined`/`cell_max_generation` above, and
deliberately do NOT mirror their shape: a cell's min-turbine/min-generation
floor `F_c` is `Σ_{g∈cell} resolved_min_*(g)` — the cell's OWN member groups'
resolved minima, summed, with no fold and no closing `.min(plant)` term. This is
the correct shape because the floor bounds a SUM of variables: the cell's
member groups all feed one shared aggregate column (one turbine column, one
FPHA-generation column, or the same column read at ρ for
`ConstantProductivity`), so each group's own mandatory minimum adds to the
others' — it is not one quantity two caps compete to bound tighter, which is
what licenses a `min` fold on the MAX side.

The plant's resolved minimum `P` (`hydro_bounds_at_block`: the `hydro_bounds`
override, else the declared value) is a second floor on the plant's total, and
the TIGHTER of the two wins. `PlantFloor` owns the decision: `P` binds only
where `P > Σ_c F_c + envelope_tolerance(P)` — the same tolerance rule 44 admits
a declared minimum with (`P_declared ≤ Σ F_declared + tolerance`), so a
declared minimum never changes the LP and every study without a raising
override builds byte-identical rows and columns. Where `P` binds:

- **A single-cell plant folds `P` into its one cell row** (`cell_rhs`); no
  row is added. Every plant without multi-bus groups takes this path.
- **A multi-cell plant gets one residual row per `MinFloor` and block**,
  `Σ_c LHS_c + σ_plant ≥ P`, where `LHS_c` repeats cell `c`'s own row terms —
  the turbine column (or FPHA generation column, or `ρ·q_c`; nothing for a
  dormant plant) **and the cell's own slack `σ_c`** (`push_cell_floor_lhs`, the
  single owner of a cell's floor LHS for both row kinds). Carrying `σ_c` is
  what makes the charge `max(P − Σq, Σ_c σ_c)`: `σ_plant` tops up only what
  the cell slacks leave. The wrong-but-compiling alternative — a residual row
  on the bare columns, `Σ_c q_c + σ_plant ≥ P` — charges every unit of a cell
  shortfall twice whenever both floors bind.

A plant joins `plant_floor_hydro_indices` for a stage when its residual binds
at any block; at a block where the cells cover `P` its row is free (`−∞`) and
its slack pinned `[0, 0]`. Both slacks are priced at the plant's FULL penalty
per hour (`MinFloor::below_cost`). An override that LOWERS `P` below `Σ F_c`
cannot loosen the floor — the groups' minima still bind — so setup emits one
warning per plant and `MinFloor` (`warn_on_inert_plant_floor_overrides`)
pointing at `hydro_unit_group_bounds` instead. The MAX side mirrors this
decision with a hard row and no slack (the section above).

Four wrong-but-compiling alternatives for the cell floor, each with a close
MAX-side cousin that makes it easy to copy over by habit:

- **`.min(plant)` clamp** — closing the sum with `.min(P)`, copying
  `cell_max_turbined`'s closing term verbatim. It silently loosens the floor
  whenever `P` is lower than a cell's group-sum, clamping every cell to that
  plant value. A lower `P` never loosens a cell floor; a higher `P` tightens
  the plant total through the fold or the residual row above, never by
  raising one cell's own floor.
- **`max`-fold over a cell's own groups** — `max_{g∈cell} min_*(g)` instead of
  `Σ_{g∈cell} min_*(g)`. Wrong for the reason above: each group's own mandatory
  floor adds to the cell's aggregate minimum, so `max` understates a
  multi-group cell's true floor by every group's contribution except the
  largest.
- **ρ-folding the generation floor** — computing `cell_min_generation` by
  folding each group's turbined-derived floor through `ConstantProductivity`'s
  ρ instead of summing `group.min_generation_mw` directly. The min-generation
  row's LHS already carries `ρ * q_c` for a `ConstantProductivity` cell;
  folding ρ into the RHS too double-prices the productivity.
- **`1/|cells|` price or RHS apportionment** — dividing the penalty
  (`turbined_violation_below_cost`/`generation_violation_below_cost`, both
  plant-level `HydroPenalties`) or the RHS by the plant's cell count before
  applying it per cell. The penalty is priced at FULL magnitude on every one
  of the plant's cells (mirroring how an arc's `k_d` release-weight replicates
  onto every cell's turbine column, never apportioned — see the water
  travel-time section below).

The cell rows `min_turbine`/`min_generation` are sized `n_cells * n_blks`
(`OperViolationSizes::cells`) and the residual rows
`plant_min_turbine`/`plant_min_generation` `n_plant_floor * n_blks`
(`plant_floors`); the min/max-outflow families stay hydro-keyed
(`hydros`), since outflow has no per-cell column to attribute to.

Output stays plant-keyed: `simulation/hydros/**.parquet`'s
`turbined_slack_m3s`/`generation_slack_mw` and the `turbined`/`generation`
violation costs read a plant's own cells' slacks (`sum_cell_slack`) plus its
residual slack when it owns one (`hydro_operational_slacks`,
`compute_hydro_violation_costs`) — the plant's whole shortfall, the same
pattern `turbined_m3s`/`generation_mw` use for the max-side columns. A per-cell
floor VIOLATION is DETERMINED (each cell owns its own row and slack), unlike
the undetermined same-bus generation split the output-axis decision (§7.10 of
the blocks-and-units design) rules out.

Read: `crates/cobre-sddp/src/lp/builder/hydro_state.rs` (`cell_min_turbined`,
`cell_min_generation`, `MinFloor`, `PlantFloor`,
`warn_on_inert_plant_floor_overrides`), `crates/cobre-sddp/src/lp/builder/columns.rs`
(`fill_cell_floor_slacks`, `fill_plant_floor_slacks`),
`crates/cobre-sddp/src/lp/builder/rows.rs` (`fill_operational_violation_rows`),
`crates/cobre-sddp/src/lp/builder/entries.rs` (`push_cell_floor_lhs`),
`crates/cobre-sddp/src/lp/builder/layout.rs` (`identify_plant_floor_hydros`,
`OperViolationRanges`), `crates/cobre-sddp/src/simulation/extraction.rs`
(`sum_cell_slack`, `hydro_operational_slacks`),
`crates/cobre-io/src/validation/semantic/hydro.rs` (rule 44) and
`crates/cobre-io/src/validation/semantic/mod.rs` (`envelope_tolerance`).
Pinned by `test_min_floor_rhs_is_the_cells_own_group_sum` (mutation-verified
against the `.min(plant)` clamp, the `max`-fold, the ρ-fold, and `1/|cells|`
apportionment), `test_floor_shortfall_is_charged_once_as_the_larger_of_plant_and_cells`
(fails at 16 against 12 when the residual row drops `σ_c`), the
`*_residual_rows_sum_their_cell_rows` trio (constant, FPHA, dormant),
`test_plant_floor_the_cells_cover_leaves_the_stage_lp_unchanged` (the shared
tolerance), the d53 binding fixture, and the group-declaration-order
determinism regression.

### Both outflow rows bind the non-diverted river-remnant flow

Both per-hydro outflow rows couple turbine + spillage only, symmetrically:
`q + s + σ_below ≥ min_outflow` and `q + s − σ_above ≤ max_outflow`. The
diversion column `d` is DELIBERATELY EXCLUDED from BOTH. A diversion routes water
to a _different_ downstream target — the water balance books it `+τ_h` on the
source's own row and `−τ_h` on the diversion target's row
(`fill_state_and_water_entries`), and the water travel-time arc deposits only
`q + s` (`stage_release_rate_m3s` excludes diversion for exactly this reason) — so
`d` is a separate flow path, capped by its own `max_diversion_m3s` column bound,
not part of the natural river reach the defluência bounds govern.

Coupling `d` into either row is the wrong-but-compiling alternative: on the
minimum it lets diverted water satisfy the floor, so a diverting plant reports
zero below-slack while its own channel carries less than `min_outflow` (the Belo
Monte / Volta Grande under-release); on the maximum it double-governs the
diversion's own cap, capping total release rather than the natural channel.

The rows are byte-neutral on any non-diverting deck: the diversion column is
dense but pinned `[0, 0]` (`fill_diversion_columns`) and presolve-eliminated, so
omitting its zero-valued coefficient leaves the solved LP identical. Both flow
families stay hydro-keyed (`n_op_hydro`), never per-cell.

Read: `crates/cobre-sddp/src/lp/builder/entries.rs`
(`fill_operational_violation_entries` — both outflow blocks omit `d`),
`crates/cobre-sddp/src/lp/builder/rows.rs` (`fill_operational_violation_rows`),
`crates/cobre-core/src/entities/hydro.rs` (`Hydro::min_outflow_m3s` /
`max_outflow_m3s` docs). Pinned by `both_outflow_rows_exclude_diversion`
(`crates/cobre-sddp/src/lp/builder/template/tests.rs`, the structural coefficient
check, mutation-verified against re-adding `d` to either row),
`min_outflow_binds_the_non_diverted_flow_on_a_diverter`, and
`max_outflow_binds_the_non_diverted_flow_on_a_diverter` (`tests/hydro_sim.rs`,
run-of-river diverters where the diverted flow would otherwise satisfy the
minimum or evade the maximum).

## Cut pool is append-only; basis matches by slot identity

**One pool per pool id.** A pool is addressed by its 0-based **pool id**,
resolved from the node graph's `node → pool` map (`NodeGraph`,
`NodeRuntime.pool_id`); on the degenerate one-node-per-stage graph (`nodes[]`
absent) `pool_id == stage`, so every read reduces byte-for-byte to the
pre-node-native stage read. Sibling fan nodes at one level may share a pool.

**Append-only within a pool.** A cut is never removed from its pool. Deactivation
clears the slot's activity flag (`CutPool::set_active`) and keeps the slot and
its data; every cut keeps a stable slot index for the lifetime of the run,
placed by `slot_index`'s deterministic function of `slot_base`,
`iteration`, `iteration_base`, `visit_stride`, and `forward_pass_index`. The
per-iteration template refreeze encodes **only active cuts** (one row per
`active_cuts()` entry), so a refrozen stage template holds no inactive cut. The
lower-bound LP only appends rows (`append_new_cuts_to_lp`) and never removes
one, so a root cut stays in it once added.

**Growth is between-iteration and append-only.** A pool's capacity may grow
between iterations (`CutPool::grow`, when a node's realized visit rate would
exceed its construction-time `visit_stride` floor) — never mid-iteration. Growth
is `Vec::resize`, which only appends new slots, so every populated slot keeps its
index across the realloc. Relocating or re-packing a populated slot on growth is
the wrong-but-compiling alternative — it silently invalidates every stored
basis's slot-identity match. Each cut record also carries its generating
`node_id` (`CutMetadata.node`, set by `add_cut(node_id, …)`); this is
**provenance only** (carried onto the MPI cut wire) and never affects which slot
the cut lands in — the append-only, slot-identity contract is independent of
`node_id`.

Among generated slots, slot order refines generation order, also after growth,
while every `forward_pass_index` is below the pool's `visit_stride`:
`(iteration_generated, forward_pass_index)` increases with the slot, and a slot
written in an iteration lies above every slot written in an earlier one, so the
slot tie-break of the budget's eviction key is a generation-order tie-break.
Swapping the roles of the iteration and the pass in `slot_index` is the
wrong-but-compiling alternative: a later iteration's cut then lands below an
earlier one's. Pinned by `generated_slots_follow_generation_order_across_growth`
(`cut/pool.rs`).

**A loaded cut keeps its real generating iteration; the loaded prefix is a
stored role.** A cut loaded from a checkpoint (`warm_start` or `resume`) is stored
with the `iteration` its record carries, in `CutMetadata.iteration_generated`.
Whether a slot held a cut before the run started is the slot's role:
`CutPool::is_prefix_slot` reads `SlotRole::origin`, which the loading
constructors write and `add_cut` writes as `Generated`:
`CutPool::new_with_warm_start` writes `Prefix` for every record's slot, while
`CutPool::from_deserialized` and `CutPool::new_resumed` write `Prefix` for the
records stored below the recorded prefix length (`warm_start_count`) and
`Generated` for the others. Every load reads each record's `scored` flag as
written and never infers it from the role. `LoadedSlotMap`
(`policy/full_fcf_load.rs`) is the only translation from a stored slot to the slot
its record occupies: the identity for a resume, the record's rank in ascending
slot order for a warm start and a simulation-only load, which pack. The role is
never a value of `iteration_generated`. Under `warm_start` the run's iteration
counter restarts at 1, so a loaded cut's generating iteration can equal the
current one; every reader that separates a loaded cut from a cut of this run
tests the predicate before the iteration.

A sentinel in `iteration_generated` (a value no iteration reaches) is the
wrong-but-compiling alternative: it keeps loaded cuts out of the per-iteration
sets, but the field reaches the checkpoint writer, which writes it as the cut's
`iteration`, so every loaded cut is exported with a corrupted iteration, or the
write is refused when the sentinel does not fit the record's 32-bit field.
Comparing `iteration_generated` to the current iteration without the
predicate is the second: a loaded cut whose generating iteration equals the
current one is taken for a cut of this iteration, appended to the LP a second
time as a delta row and packed for the cut exchange as a new cut. Answering by
position (`slot < slot_base`) is the third: a pool built by `CutPool::new`
with a non-zero `slot_base` reserves leading slots that no record fills,
and the position test calls each of them a loaded cut. Writing every loaded
record unscored is the fourth: a cut its source scored then keeps the flag it was
loaded with, so a warm-started `level1` run never deactivates or reactivates it,
and a simulation-only `all` sees only the stored flags.
Read: `cut/pool.rs` (`SlotRole`, `CutPool::new_with_warm_start`,
`CutPool::from_deserialized`, `CutPool::new_resumed`, `is_prefix_slot`,
`active_delta_cuts`, `enforce_budget`), `cut/cut_sync.rs` (`pack_pool_into`),
`cut/dcs.rs` (`score_violated_candidates`, `build_initial_resident_set`),
`policy/policy_export.rs` (`build_stage_cut_records`), `policy/full_fcf_load.rs`
(`LoadedSlotMap`). Pinned by
`exported_loaded_cut_keeps_its_generating_iteration` (`policy/policy_export.rs`:
a loaded cut is exported at its record's iteration),
`a_loaded_cut_sharing_the_current_iteration_number_is_not_a_delta_cut`
(`cut/pool.rs`) and `a_reserved_leading_slot_is_vacant_not_loaded` (`cut/pool.rs`:
a reserved leading slot is vacant and no slot of the pool is a loaded cut), with
one pin per other reader:
`a_loaded_cut_is_never_packed_as_a_new_cut` (`cut/cut_sync.rs`),
`enforce_budget_never_evicts_a_loaded_cut`
(`cut/pool.rs`), `candidate_recency_window_keeps_warm_start_cuts` and
`a_loaded_slot_is_not_seeded_as_a_current_iteration_cut` (`cut/dcs.rs`); and,
for the role and the flag each load writes,
`the_loading_constructors_read_each_records_scored_flag` (`cut/pool.rs`),
`from_deserialized_gives_each_pool_its_recorded_prefix` (`cut/fcf.rs`),
`a_slot_map_places_each_record_by_kind_and_leaves_a_gap_empty`
(`policy/full_fcf_load.rs`) and, through the real writer and reader in
`tests/cut_basis.rs`, `a_warm_started_scored_cut_is_selected_like_a_generated_one`
and `a_warm_started_unscored_cut_keeps_its_flag`.

**The activity clock is the current run's.** `CutMetadata::last_active_iter` and
`CutMetadata::active_count` are on the current run's iteration clock. Two readers
use them: the eviction key of the active-cut budget (`CutPool::enforce_budget`),
and the seed window of a lazy solve under `dynamic`, which compares
`last_active_iter` with the run's iteration counter and seeds an active cut when
`current_iteration.saturating_sub(last_active_iter) <= seed_window`
(`build_initial_resident_set`). `add_cut` starts a cut of this run at
`last_active_iter = iteration_generated` and `active_count = 0`.

A warm start and a boundary injection restart both counters at 0 for the loaded
prefix (`CutPool::restart_activity_clock`), because their records carry the
iterations of another run or study: `check_full_fcf_load` restarts every pool
for `FullFcfLoadKind::WarmStart`, and `inject_boundary_cuts` restarts the
terminal pool it installs. A resume restores the counters the checkpoint carries
(below), the terminal pool's boundary rows included (the boundary section at the
end of the resume sections); a simulation-only load keeps the record's
iteration and reads no clock, since the simulation seeds at iteration 0. The
restart touches the two counters of the prefix slots only:
`iteration_generated` stays the record's iteration, as the paragraph above
requires. A restarted active loaded cut is in the seed of iterations
`1..=seed_window` and then of the `seed_window` iterations that follow each
iteration in which it bound, and it is a candidate of every lazy solve
throughout. `build_initial_resident_set` still tests `is_prefix_slot` before it
compares `iteration_generated` with the current iteration: a restarted loaded
cut whose generating iteration equals the current one lies outside the window
and would otherwise be seeded as an untested cut of this iteration.

Seeding a warm start from `record.iteration` is the wrong-but-compiling
alternative: the run's counter restarts at 1, so the window's subtraction
saturates to 0 until the new run passes the source run's iteration count, every
loaded active cut is seeded in every lazy solve, and the window selects
nothing. Restarting the clock under a resume is the opposite error: a reloaded
cut the run generated would fall out of the seed window it was in before the
checkpoint.

**A checkpoint carries the counters while a reader is configured.**
`CheckpointLayout::new` keeps `CutManagementConfig::tracks_binding_activity` of
the setup's config. While it holds, `build_stage_cut_records` called with
`BindingCounters::Carried` writes every piece's `active_count` and
`last_active_iter` (`last_active_iteration` on disk) from its slot's metadata,
and the manifest's `activity_counters` is set;
otherwise no piece carries them and the flag is unset. The write issues no
collective: every rank applies the same `CutPool::record_binding` after the
reduction in `sync_stage_metadata`, inside the backward pass, so rank 0 holds the
run's counters at the periodic and at the final write. `CutPool::new_resumed`
writes a record's counters into its slot, so a resumed run with a budget or
`dynamic` continues the eviction order and the seed window of the uninterrupted
run, the prefix of a run that was itself warm-started included: its counters are
on that run's clock. A record without counters resumes at `active_count = 0` and, in
a slot below the recorded prefix length, at `last_active_iter = 0`, where the
restart of a warm start or a boundary injection left it; in any other slot at
`last_active_iter = iteration`, where `add_cut` starts a cut. Only a run with no
reader configured writes such a record, and nothing reads either counter in
that run. A resume whose run configures a reader refuses a checkpoint written
without counters (the resume rejects below), since the recorded run did not count
the iterations in which its cuts bound. A warm start, a boundary injection outside
a resume and a simulation-only load read no counter.

Writing the counters always and reading them under `warm_start` is the
wrong-but-compiling alternative: the records carry the source run's clock, which
the new run's counter does not continue (the seeding error above). No test pins
this one: the restart of a warm start zeroes the prefix counters after the load,
so a load that read them leaves the same pool. Not restoring
them on a resume is the opposite one: a reloaded cut that had bound resumes at the
iteration that generated it, so the budget evicts it ahead of cuts the
uninterrupted run evicts and `dynamic` drops it from the seed window it was in.
Resuming a prefix record without counters at its generating iteration is the
last: no result moves, since no reader exists, but the resumed pool no longer
equals the uninterrupted warm-started run's, whose prefix holds 0.

Read: `cut/pool.rs` (`CutPool::restart_activity_clock`, `CutPool::new_resumed`,
`add_cut`, `enforce_budget`), `cut/dcs.rs` (`build_initial_resident_set`),
`policy/full_fcf_load.rs` (`check_full_fcf_load`), `policy/policy_load.rs`
(`inject_boundary_cuts`), `policy/orchestration.rs` (`CheckpointLayout::new`,
`CheckpointLayout::write`), `policy/policy_export.rs`
(`build_stage_cut_records`). Pinned by
`a_layout_with_a_budget_writes_both_counters_on_every_piece` and
`a_layout_without_a_reader_writes_no_counter` (`policy/policy_export.rs`),
`a_resumed_pool_restores_the_binding_counters_of_its_checkpoint`
(`tests/full_fcf_load.rs`: after a resume of a budget run's checkpoint every slot
holds that run's counters, some of them non-zero),
`a_warm_start_restarts_the_binding_counters_and_its_resume_restores_them`
(`tests/full_fcf_load.rs`: a warm start from that checkpoint holds 0 in every
prefix slot, and a resume of the warm-started run restores its counters, some
prefix cut's non-zero),
`resumed_run_equals_the_uninterrupted_run_with_one_forward_pass`
(`tests/periodic_checkpoint.rs`: without a reader, the counters of every slot
equal the uninterrupted run's), `loaded_activity_clock_follows_the_load_kind`
(`tests/full_fcf_load.rs`: after a load of a checkpoint without counters, every
loaded slot has `active_count = 0`; under a resume the slots are not prefix slots
and `last_active_iter` equals the record's iteration, under a warm start they are
prefix slots and it is 0),
`injected_boundary_cuts_start_on_this_runs_activity_clock`
(`policy/policy_load.rs`), `restart_activity_clock_touches_only_the_loaded_prefix`
(`cut/pool.rs`), `a_resumed_record_without_counters_resumes_on_the_clock_its_run_left_it`
(`cut/pool.rs`: a record without counters resumes at 0 below the prefix length
and at its iteration above it, one with counters at its counters),
`a_resumed_warm_started_run_equals_its_uninterrupted_run`
(`tests/periodic_checkpoint.rs`: the counters of every slot of a warm-started run
without a reader, prefix included, equal the uninterrupted run's),
`a_warm_started_cut_is_seeded_in_the_first_window_and_after_it_binds`
(`cut/dcs.rs`: the window and its saturation under the stored clock) and
`a_loaded_slot_is_not_seeded_as_a_current_iteration_cut` (`cut/dcs.rs`: the
guard).

**A never-written slot is not a cut.** A slot holds a cut only once `add_cut` or a
loading constructor has written it; until then its role origin is `Vacant` and its
intercept and coefficients are zero. A vacant slot lies below
`CutPool::populated()` when a pool's slot block for an iteration is not filled:
on a sampled branching graph the pool's `visit_stride` is a statistical floor
(`NodeGraph::pool_cut_stride`), so a node visited fewer times than its stride
leaves the tail of its block unwritten and the next iteration's block starts above
it; and a `CutPool::new` pool with a non-zero `slot_base` leaves its
leading slots unwritten. `populated()` is one past the highest written slot: it
bounds slot-indexed reads and does not count cuts, which `generated()` does.
Readers skip a vacant slot. The checkpoint writer (`build_stage_cut_records`)
writes no record for it, and `build_stage_cuts_payloads` writes `populated_count`
as the number of records. The ownership record treats it as not live (`is_live`),
so its zero owns no state, and `CutPool::ownership_activity_updates` lists scored
slots only, so it is never activated. A debug build refuses to activate a vacant
slot (`CutPool::set_active`).

Treating `populated()` as a cut count is the wrong-but-compiling alternative: the
vacant zero then competes in the maximum and owns states, and the checkpoint
exports it as an inactive zero record, which a load reads back as a loaded cut.
Read: `cut/pool.rs` (`SlotRole`, `CutPool::role`, `CutPool::populated`),
`cut/ownership.rs` (`is_live`), `policy/policy_export.rs`
(`build_stage_cut_records`, `build_stage_cuts_payloads`). Pinned by
`a_slot_role_is_fixed_when_the_slot_is_written` and
`a_reserved_leading_slot_is_vacant_not_loaded` (`cut/pool.rs`: a role is written
with its slot and by nothing else), `a_never_written_slot_is_not_exported`
(`policy/policy_export.rs`: records at slots 0, 2 and 3 and a `populated_count`
of 3) and `sampled_fan_checkpoint_holds_no_never_written_slot`
(`tests/branching_value_oracle.rs`: on a trained sampled fan, one record per
written slot), the `vacant_slot` fixture of
`the_record_reproduces_every_reference_fixture` (`cut/ownership.rs`: the vacant
zero owns no state) and `incremental_record_equals_rebuild_and_reference`
(`cut/ownership.rs`: its draws leave vacant holes below the highest written
slot), and `level1_follows_the_reference_rule_on_every_fixture`
(`cut/cut_selection.rs`: the vacant slot of a fixture is in neither update list),
with `activating_a_vacant_slot_panics_in_debug` (`cut/pool.rs`) for the debug
assertion.

**A resume places each cut at its slot.** `check_full_fcf_load` builds the FCF of
a `resume` through `FutureCostFunction::new_resumed`, which builds every pool
through `CutPool::new_resumed`; a `warm_start` keeps `new_with_warm_start`. A
record goes to its own `slot_index`, not to its position in the list. The slot
base is the pool's recorded `warm_start_count`, and the capacity is the larger of
the recorded `capacity` and `slot_base + max_iterations * visit_stride`, so a pool
that grew keeps its slots. A slot below `warm_start_count` is `Prefix`, every
other record is `Generated`, and each keeps the `scored` flag it was written with.
A slot with no record stays `Vacant`, so `populated()` is one past the highest
record and `generated()` counts records. The constructor stores no trial state:
the session does, once it has enabled the store (the trial-state section below).
`TrainingSession::new` sets the iteration base to 1 on every run, so the slot of
iteration `k` and pass `p` is
`slot_base + (k - 1) * visit_stride + p` on a resumed run as in the uninterrupted
one; `start_iteration` only drives the loop range. A record that shares a slot,
lies at or above the capacity, or leaves a slot of the prefix empty is refused
with `SddpError::Validation` naming the pool and slot.

The wrong-but-compiling alternatives:

- packing the records into slots `0..n`, as `new_with_warm_start` does: a cut
  after a vacant slot moves down, so the loop's cuts no longer land where the
  uninterrupted run's did and every stored basis's slot match breaks;
- treating every reloaded cut as `Prefix`: the cuts the recorded run generated
  become unselectable and unevictable, so selection and the budget treat the
  resumed run differently;
- an iteration base of `start_iteration + 1`: the first resumed iteration maps to
  slot `slot_base`, which a reloaded cut holds, and the window boundary shifts by
  the iterations before the checkpoint.

Read: `cut/pool.rs` (`CutPool::new_resumed`), `cut/fcf.rs`
(`FutureCostFunction::new_resumed`), `policy/full_fcf_load.rs`
(`check_full_fcf_load`), `training/session/mod.rs` (`TrainingSession::new`).
Pinned by `a_resumed_pool_keeps_each_record_at_its_slot_and_a_hole_vacant`,
`a_resumed_pool_takes_the_loop_cuts_where_the_recorded_run_put_them` and the
`a_resumed_pool_refuses_*` tests (`cut/pool.rs`),
`new_resumed_names_the_pool_whose_records_it_refuses` (`cut/fcf.rs`),
`a_resume_restores_the_prefix_length_of_a_warm_started_run`
(`tests/full_fcf_load.rs`: the prefix length and the prefix slots of a resumed
warm-started run), `a_fan_checkpoint_resumes_every_record_at_its_slot`
(`tests/deterministic.rs`: a trained sampled fan written by the real writer and
read back, every slot's role, flag and value bits, vacant slots included, and the
node id of each pool) and `resumed_run_equals_the_uninterrupted_run_with_one_forward_pass`
(`tests/periodic_checkpoint.rs`: the final checkpoint of the resumed run equals
the uninterrupted run's).

**Basis matches by slot, never by count or column.** Warm-start basis
reconstruction matches stored cut rows to current LP rows by **`CutPool` slot
identity**, never by row count and never by absolute column index. On the
frozen hot path `reconstruct_basis` is the single entry point for every pool
whose cut set can still grow — the entire interior — and must never be
bypassed there. The terminal-static short-circuit below is the SOLE licensed
bypass, and only because a terminal pool's cut set is provably invariant.
Read: `cut/pool.rs` (`CutPool::grow`, `add_cut`, `CutMetadata.node`,
`slot_index`), `cut/fcf.rs` (the `node → pool` map / pool-id addressing),
`cut/basis_reconstruct.rs`. Pinned by
`test_anticipated_5stage_k2_warm_start_zero_basis_rejections`
(`tests/anticipated_scenarios.rs` — the anticipated ring shifts every downstream
column, yet the run records zero basis rejections because reconstruction matches
by slot identity, not column index) and the slot-identity reconstruction
regressions in `tests/cut_basis.rs`.

### The terminal stage bypasses reconstruction with a 1:1 basis apply

The terminal stage solves against a baked static template whose active-cut set
is fixed once primed — a leaf never gains or loses a cut: no backward pass adds
to a leaf pool, selection scores interior nodes only, and the budget visits
interior pools only — so the template's shape never changes across iterations.
There the slot-identity reconstruction
above is REPLACED by a plain 1:1 basis apply (`run_stage_solve_terminal_static`,
selected only at the terminal forward solve by `solve_forward_node`'s
`is_terminal` gate): a node-matching stored basis maps onto the current LP by
position and is copied verbatim into `scratch_basis` with no reconstruction.
This is the sole licensed bypass of `reconstruct_basis` on the frozen hot path,
and it is safe ONLY because the terminal cut set is provably invariant. The
node-tag filter and a shape guard still gate the apply: a stored basis whose
`node_id` mismatches the node being solved (`filtered_stored_basis`), or whose
column/row status length does not match the current template
(`terminal_basis_shape_matches`), drops to cold rather than applying a
wrong-shaped basis.

Applying this short-circuit on an interior node — or on any node whose cut set
is NOT provably invariant — is the wrong-but-compiling alternative: an interior
pool's shape changes as deeper backward levels append cuts, so a 1:1 apply
matches a stored basis against a differently-shaped LP and silently warm-starts
from the wrong factorization. The interior hot path is untouched —
`reconstruct_basis` remains its sole entry, reached through `run_stage_solve`.
Read: `solve/stage_solve.rs` (`run_stage_solve_terminal_static`,
`filtered_stored_basis`, `terminal_basis_shape_matches`, and `run_stage_solve`
for the interior path), `training/forward/enumerated.rs` (`solve_forward_node`'s
`is_terminal` gate). Pinned by
`run_stage_solve_terminal_static_applies_basis_1to1_without_reconstruct_basis`
(the verbatim copy, no reconstruction) and its interior counterpart
`run_stage_solve_interior_warm_start_invokes_reconstruct_basis` (an interior
warm start still invokes `reconstruct_basis`), plus
`run_stage_solve_terminal_static_cross_node_stored_basis_is_treated_as_cold` and
`run_stage_solve_terminal_static_shape_mismatch_is_treated_as_cold` (the
node-tag and shape guards each drop to cold), all in `solve/stage_solve.rs`, and
by `budget_caps_only_interior_pools_and_templates_match_their_pools`
(`training/session/mod.rs`: after the budget runs, every pool's frozen template
holds exactly its pool's active cuts, the terminal pool's included).

### A stored basis records its own cut-row count

Each checkpoint `StageBasis` record holds one `BasisEntry` per scenario, and each
entry carries `piece_row_slots`, the pool slot of every trailing cut row of the
captured basis itself (`CapturedBasis::cut_row_slots`, one per row of
`row_status.len() - base_row_count`). Writing the node's pool count
(`populated()`) instead is the wrong-but-compiling alternative: it overstates the
basis's cut rows, because a forward-pass capture precedes that iteration's
backward pass, which appends cuts to the pool afterwards. `FORMAT_VERSION` 4
marks this layout. The loader still takes the template row count from the
current LP (`node_dims`), never from the record. Read: `policy/policy_export.rs`
(`basis_entry`, the one builder of an entry, `build_basis_store_entries`,
`build_stage_basis_records`). Pinned by
`build_stage_basis_records_writes_each_basis_own_trailing_cut_row_count`
(`policy/policy_export.rs`: two scenarios per node, each with its own slots) and
`exported_basis_records_count_the_cut_rows_each_basis_was_captured_with`
(`tests/cut_basis.rs`, through the checkpoint's own `build_basis_store_entries`).

The load decodes every entry of every node record (`build_basis_cache_for_nodes`)
and admits each by its stored slots. An entry's cut rows keep the slots it
recorded, each translated through its node's pool's `LoadedSlotMap`
(`policy/full_fcf_load.rs`), the only translation from a stored slot: the stored
slot under a resume, the record's rank under a warm start and a simulation-only
load, which pack. Its template row count is the current LP's (`node_dims`). An
entry is used only when it fits that node's LP exactly: `admit_stored_basis`
checks, in order, `found_cols == template_cols`,
`found_rows == template_rows + piece_row_slots.len()`, and a basic count (column
and row statuses decoding to `Basic`) equal to `found_rows`; then every recorded
slot must map to a loaded cut (`StoredBasisMisfit::UnmappedSlot` otherwise). An
entry that fails is left out and the load proceeds: `UnusedStoredBases` owns the
one aggregated warning, `stored bases not used: …`, which counts entries. The
basic count is checked because `reconstruct_basis` assumes it and a deficit
aborts the run with `BasisShapeMismatch` (`cut/basis_reconstruct.rs`, "Basic-count
invariant"); `reconstruct_basis` is never called with a dropped entry, so the
append-only pool and slot identity are untouched.

Each admitted entry seeds its own scenario. `StoredBasisLoad::store` keeps, per
node, the admitted entries with their global scenario; a training load hands them
to `train` in `LoadedPolicy::scenario_bases`, and `TrainingSession::seed_basis_store`
fills, on every rank and before `prime_frozen_templates`, local row
`m < my_actual_fwd` of each node with `LoadedPolicy::seed_basis` of global scenario
`fwd_offset + m`: that scenario's own entry; when the node holds none for it, a
`resume` seeds nothing (`MissingScenarioBasis::Cold`) and a `warm_start` seeds
the node's lowest-scenario entry (a source that ran fewer forward passes). The
padding rows
`my_actual_fwd..max_local_fwd` stay empty. Simulation keeps one basis per node,
the lowest-scenario admitted entry (`StoredBasisLoad::cache`).

The wrong-but-compiling alternatives:

- refusing a `warm_start`, simulation-only or boundary load over a basis, which
  only warm-starts a solve. A `resume` is the one load that refuses over the store:
  one whose `basis_store_complete` is unset (a failure exit's) is refused (the
  resume rejects below);
- seeding a resumed scenario from another scenario's entry where the node holds
  none of its own: on a branching sampled graph the uninterrupted run holds no
  basis for a node the scenario has not visited and solves it cold, and a seeded
  solve can settle on another vertex;
- admitting on shape alone, which lets a basic-count deficit reach
  `reconstruct_basis`;
- adding a cold retry after `reconstruct_basis` fails, which hides an entry this
  rule should have dropped;
- copying one basis to every scenario: a resumed run with several forward passes
  restarts scenarios 1 and up from scenario 0's basis, and at a degenerate optimum
  their solves settle on other vertices than the uninterrupted run's;
- inferring a basis's cut-row slots from the pool's active set (its first slots
  when the active slots are an unbroken prefix, no cut rows otherwise): a pool
  under selection is not a prefix, so every stored basis loses its cut rows;
- keying a row by its local index instead of its global scenario: every rank
  but rank 0 restores rank 0's scenarios.

Read: `policy/policy_load.rs` (`build_basis_cache_for_nodes`,
`admit_stored_basis`, `StoredBasisLoad`), `policy/full_fcf_load.rs`
(`check_full_fcf_load`, `LoadedSlotMap`), `training/training.rs`
(`LoadedPolicy::seed_basis`), `training/session/mod.rs` (`seed_basis_store`).
Pinned by, in `policy/policy_load.rs`:
`stored_basis_whose_basic_count_differs_from_its_rows_is_dropped`,
`unused_stored_bases_report_is_independent_of_record_order`,
`a_stored_basis_is_admitted_at_its_translated_slots` (cut rows at slots that are
not a prefix of the pool, each at its translated slot under every load kind),
`a_stored_basis_with_an_unmapped_slot_is_left_out_and_counted` and
`every_entry_of_a_record_is_kept_by_scenario_and_the_cache_takes_the_lowest_admitted`;
`seeding_gives_each_local_row_its_global_scenario_and_leaves_padding_empty`
(`training/session/mod.rs`: rank 1 of 2 seeds its row from scenario 2 and leaves
its padding row empty, a node holding scenario 0 only seeds every row from it;
fails under the local-row key and under seeding the padding rows);
`resumed_run_equals_the_uninterrupted_run_with_several_forward_passes`
(`tests/periodic_checkpoint.rs`: the resumed setup seeds every scenario and node
with the interrupted run's checkpoint entry, and the resumed run equals the
uninterrupted one bit for bit; under copying scenario 0's basis to every row the
cut intercepts of pool 1 differ, and under decoding only the lowest-scenario entry
the seeded basis of scenario 1 does);
`a_resumed_fan_run_ends_with_the_cut_pools_of_the_uninterrupted_run`
(`policy/full_fcf_load.rs`: on a sampled fan where some nodes hold the bases of
some scenarios only, the resumed pools equal the uninterrupted run's slot by slot;
the comparison passes under copying scenario 0's basis, so it pins the pools and
not the seeding, which the next test checks);
`only_a_resume_leaves_a_scenario_without_a_stored_basis_cold`
(`policy/full_fcf_load.rs`: on the same fan a scenario with no entry at a node
gets none under a resume and the node's lowest-scenario entry under a warm start);
`a_resume_refuses_a_checkpoint_whose_bases_are_not_whole` (`tests/full_fcf_load.rs`:
the resume is refused and a warm start of the same checkpoint loads);
`warm_start_skips_a_stored_basis_with_too_few_basic_entries_instead_of_aborting`
(`tests/cut_basis.rs`),
`simulation_only_loads_a_policy_with_a_wider_stored_basis_and_warns`
(`crates/cobre-cli/tests/cli_run.rs`) and
`test_load_policy_with_a_wider_stored_basis_loads_and_warns_once`
(`crates/cobre-python/tests/test_policy_load_validation.py`).

### Simulation loads the cuts of one chosen set

Simulation builds each stage LP from the cuts of a `CutSet`, the field
`StudySetup::simulation_cut_set`. The key `simulation.cut_set` sets it, `all` or
`active`, and `all` is the default (`SimulationConfig::cut_set` in `cobre-io`
owns the description). `StudyParams::from_config` maps the key; the CLI carries
it in `BroadcastConfig::simulation_cut_set`, and every rank builds its setup from
that value (`build_study_setup`, `StudySetup::from_broadcast_params`), so every
rank simulates the same set. `Active` is the cuts whose flag is set. `All` is
decided by slot role (`CutPool::cuts_in`, `CutPool::count_in`): every `Generated`
slot whatever its flag and whatever `SlotRole::scored` says, since a cut written
before the pool stored trial states can be evicted by the cap too; every scored
`Prefix` slot whatever its flag; every unscored `Prefix` slot whose flag is set;
never a `Vacant` slot. `Active` is a subset of `All`, so equal counts mean equal
sets. The choice changes no flag: it selects which slots a simulation LP holds
and nothing else, and training, the lower bound, the checkpoint and the selection
record never read it.

The template, the basis reconstruction and the load read the same set.
`refreeze_templates_if_needed` builds the template from `cuts_in(set)`, and the
non-lazy branch of `solve_simulation_stage` reconstructs the stored basis over
the same iterator (`run_stage_solve_over`), so a stored training basis applies by
slot over the rows the template holds and a row training never had starts basic.
The caller's templates are used only when they hold the chosen set: always under
`Active`, and under `All` when `count_in(All) == active_count()` in every pool.
The lazy branch under `dynamic` reads the active cuts and asserts that the two
counts are equal. A run that trains under `dynamic` deactivates nothing, so a
pool breaks the assert only through a cut it loaded, from a source that
deactivated or evicted one. `check_full_fcf_load` refuses that pairing for every
load kind, and `reconcile_boundary_policy` for `policy.boundary`
(`check_dynamic_simulation_serves`, one function for both), when the run's
strategy is `Dynamic`, the run simulates (`simulation_config.n_scenarios > 0`) with
`simulation_cut_set = All`, and some loaded pool, or the terminal pool the
boundary injection installs, has `count_in(All) != active_count()`. The loaded
terminal pool is not checked in a warm start or a resume of a study that declares
`policy.boundary`, which replaces it after the load; a simulation-only load
injects no boundary and checks it. A load is
refused as `FullFcfLoadError::Refused(SddpError::Validation(..))`, class
`IncompatiblePolicy`; the boundary as `SddpError::Validation` of the boundary
phase (`PrepPhase::Boundary`), which `cobre validate`, a training `cobre run` and
the Python bindings reach through `reconcile_boundary_policy`. Both name both settings,
the first such pool with its two counts (`the policy.boundary pool` for the
boundary) and the way out: `simulation.cut_set = active`, or the `level1` method.
The assert then never meets a pool a load or the boundary admitted. Under `Active`
the two counts are equal by construction, and training under `dynamic` over loaded
inactive cuts with no simulation of `All` is admitted: those cuts are valid, a
subset of the pool.

The boundary check runs on the pool `terminal_boundary_pool` builds, which
`inject_boundary_cuts` installs, so what is checked is what is installed. Its
rows are `Prefix` rows that keep the `scored` and active flags of their source
records: a `level1` source of a longer horizon, whose interior pool is priced at
the study's boundary date, supplies a scored inactive row, which `All` includes;
a source that stored no trial states, such as a capped one, supplies an unscored
inactive row, which `All` leaves out with its flag, and that pairing loads.

The wrong-but-compiling alternatives:

- reconstructing over `active_cuts()` against a template built from `All`: the
  basis has the rows of the active cuts and the LP those of `All`, which differ in
  length wherever a pool holds a cut outside its active cuts, and both backends
  reject that with `BasisRowCountMismatch`. A simulation under `All` then fails
  outright instead of warm-starting by slot;
- reusing training's templates under `All` when the sets differ: the LP holds the
  rows of the active cuts and the reconstruction counts the rows of `All`, so the
  basis and the LP disagree in length. The stored basis is truncated to the LP's
  rows, and the backends then differ: HiGHS rejects it with `BasisInconsistent`
  and CLP accepts it;
- including an unscored `Prefix` slot whose flag is off: on the terminal pool it
  is a boundary record that training's LP left out, so the post-horizon is priced
  differently from training;
- honouring the cap under `All`: the cap clears the flag of cuts a run generated,
  and `All` holds them whatever the flag. A set that leaves them out is `Active`
  under another name;
- `StudySetup::simulate` passing a fixed set instead of the field: the key then
  changes nothing;
- relaxing the lazy-branch assert in place of refusing the load: a debug build no
  longer stops, and a release build simulates the active cuts of the pool under
  the name `all`, silently, for the default value of the key;
- checking the loaded pools only: `inject_boundary_cuts` replaces the terminal pool
  after the load, and that pool is the one the assert meets at the last stage, so
  a `policy.boundary` over a `level1` source breaks the assert for the default
  value of the key;
- refusing a boundary that holds any inactive record: an unscored inactive record
  is in neither set, so the refusal is spurious;
- checking the loaded terminal pool of a warm start or a resume under
  `policy.boundary`: a checkpoint whose terminal pool holds boundary rows outside
  its active set is refused for a boundary that replaces them, though the boundary
  itself loads. Skipping it in a simulation-only load is the opposite error: no
  boundary replaces the pool there, and the lazy-branch assert meets it.

A simulation-only load restores each record's origin and scored flag
(`CutPool::from_deserialized`: `Prefix` below the recorded `warm_start_count`,
`Generated` above it, `scored` as written), so `all` over a loaded pool holds what
training's `all` held, the cuts selection deactivated and the cap evicted
included. Only the terminal pool's boundary records are `Prefix` there, and
`has_prefix_cuts` still reads them. Writing every loaded record as an unscored
`Prefix` is the wrong-but-compiling alternative: `All` then sees only the stored
flags, and a simulation-only `all` simulates fewer cuts than the run that wrote
the policy (`a_loaded_pool_keeps_the_cut_sets_of_its_run`).

Read: `cut/pool.rs` (`CutSet`, `CutPool::cuts_in`, `CutPool::count_in`),
`cut/row.rs` (`build_cut_set_row_batch_into`), `solve/stage_solve.rs`
(`run_stage_solve_over`), `simulation/state.rs` (`refreeze_templates_if_needed`),
`simulation/pipeline.rs` (`solve_simulation_stage`), `setup/orchestration.rs`
(`StudySetup::simulate`), `policy/full_fcf_load.rs`
(`check_dynamic_simulation_serves`), `policy/policy_load.rs`
(`reconcile_boundary_policy`, `terminal_boundary_pool`). Pinned by `cuts_in_all_follows_the_slot_roles`,
`cuts_in_active_equals_active_cuts` (`cut/pool.rs`: the rule by role),
`a_loaded_pool_keeps_the_cut_sets_of_its_run` (`cut/pool.rs`: an inactive scored
prefix cut and an inactive generated cut are in `All` and not in `Active`, an
inactive unscored prefix cut is in neither),
`stage_solve_applies_the_basis_over_the_given_cut_rows`
(`solve/stage_solve.rs`), `simulation_builds_its_templates_from_the_chosen_cut_set`
(`simulation/pipeline/tests.rs`: the templates simulation owns, and when the
caller's are used) and, in `tests/cut_basis.rs`,
`simulating_all_cuts_matches_the_flagged_twin` (an `All` simulation with training's
templates and bases equals an `Active` simulation of a copy of the pools with every
cut of `All` flagged, with no basis rejection),
`selection_off_simulates_identically_under_either_cut_set`,
`a_dynamic_run_has_one_cut_set`,
`a_simulation_only_load_holds_the_cuts_training_all_held` (a load through the real
writer and reader has the source's `All` and active counts in every pool) and
`simulation_follows_the_cut_set_of_the_setup`
(on a run whose sets differ, `StudySetup::simulate` under `All` and under `Active`
gives different results). The load refusal is pinned by
`a_dynamic_simulation_only_run_refuses_to_simulate_all_over_deactivated_cuts`
(`tests/full_fcf_load.rs`: a `level1` checkpoint with a deactivated cut is refused
for a `dynamic` simulation of `All` and loads under `Active`),
`a_dynamic_warm_start_refuses_to_simulate_all_over_deactivated_cuts` and
`a_dynamic_resume_refuses_to_simulate_all_over_deactivated_cuts` (the same for the
other two load kinds) and
`a_load_is_refused_for_a_dynamic_all_simulation_over_deactivated_cuts_only` (a
warm start and a resume of a run that does not simulate, a `level1` run and a
`dynamic` source load), and by
`validate_rejects_a_dynamic_simulation_over_deactivated_cuts`
(`crates/cobre-cli/tests/cli_validate.rs`, after a control that validates) with the
row `dynamic_simulation_over_deactivated_cuts` of
`crates/cobre-cli/tests/refusal_parity.rs` and
`crates/cobre-python/tests/test_refusal_parity.py`. The boundary refusal is pinned
by `a_dynamic_simulation_of_all_refuses_boundary_cuts_outside_the_active_set`
(`tests/full_fcf_load.rs`: a `level1` source's pool as the boundary of a shorter
study is refused under `All` and loads under `Active`),
`a_dynamic_simulation_of_all_loads_boundary_cuts_a_cap_evicted` (the same with a
capped source, whose inactive rows are unscored),
`a_boundary_replaces_the_loaded_terminal_pool_the_dynamic_check_reads` (a
checkpoint whose terminal pool holds a scored inactive boundary row warm-starts and
resumes for a study whose own boundary is whole, and a simulation-only load of it is
refused), and by
`validate_rejects_a_dynamic_simulation_over_deactivated_boundary_cuts`
(`crates/cobre-cli/tests/cli_validate.rs`, after a control that validates) with the
row `dynamic_simulation_over_deactivated_boundary_cuts` of the two parity tables.
The key is pinned by
`simulation_cut_set_defaults_to_all` and `with_overrides_names_the_simulation_cut_set_key` (`cobre-io`: the default, the
values, and an unknown value rejected naming the key),
`from_config_maps_the_simulation_cut_set` (`setup/params.rs`),
`broadcast_config_carries_the_simulation_cut_set`
(`crates/cobre-cli/src/commands/broadcast.rs`: the postcard round trip),
`non_root_rebuild_carries_the_simulation_cut_set`
(`crates/cobre-cli/src/commands/run/setup.rs`: the setup a rank builds from the
decoded config holds each value) and
`validate_rejects_an_unknown_simulation_cut_set`
(`crates/cobre-cli/tests/cli_validate.rs`: `active` validates, then an unknown
value is rejected naming the key).

## A stored basis warm-starts only at its own node (node-tag)

A `CapturedBasis` carries the declared `node_id` it was captured at
(`CapturedBasis::new(…, NodeId)`, `NodeGraph::node_ids[node]`). Every apply site
warm-starts from a stored basis **only when its `node_id` matches the node being
solved** and treats a mismatch as **cold** (`stored_basis.filter(|c| c.node_id
== node_id)`). A resampled path may revisit a stage at a different node than the
one whose basis is cached there, and warm-starting across that boundary reuses a
basis built against a different LP.

This node-tag check is the **sole** line of defence, not defence in depth:
**CLP accepts a right-dimension but internally inconsistent warm basis
silently** (every column and row status `Basic`, over-determining the rank —
`Clp_dual` repairs it), whereas HiGHS's `isBasisConsistent` check rejects the
same input loudly with `BasisInconsistent`; a basis with the wrong row count is
rejected symmetrically by both backends with `BasisRowCountMismatch`, so shape
alone is not where the asymmetry lives. Pinned by
`test_solver_clp_solve_accepts_inconsistent_basis_status_combination_silently`,
`test_solver_highs_solve_rejects_inconsistent_basis_status_combination`, and the
symmetric pair `test_solver_{clp,highs}_solve_rejects_undersized_row_basis` in
`crates/cobre-solver/tests/conformance.rs`. A cross-node warm-start is
therefore a silent wrong-vertex / wrong-dual on the CLP backend with no solver
backstop — so the check must live in cobre's own apply path, never be delegated
to the solver. Dropping the `node_id` filter, or comparing pool id instead of the
declared node id (sibling fan nodes share a pool — see the append-only section
above), is the wrong-but-compiling alternative: it compiles, warm-starts from the
wrong LP, and at a degenerate optimum settles on a different-but-equally-valid
vertex, silently breaking the run-to-run reproducibility and declaration-order
invariance the determinism contract requires.
Read: `workspace/workspace.rs` (`CapturedBasis::node_id`, `CapturedBasis::new`),
`solve/stage_solve.rs` (`run_stage_solve`'s `node_id` filter, `StageInputs::
node_id`), `cut/dcs.rs` (the same cross-node-reuse rejection on the DCS path).
Pinned directly by `run_stage_solve_cross_node_stored_basis_is_treated_as_cold`
(`solve/stage_solve.rs` — a deficit-shaped basis tagged at a mismatching node
drops to cold instead of erroring) and its DCS companion in `cut/dcs.rs`; the
reproducibility the check protects is pinned by the `opening_order_determinism`
gate in `tests/mpi_wire.rs` (bitwise `final_lb` across thread and rank shapes).
The CLP/HiGHS basis-validation asymmetry itself is unpinned by any test.

### Simulation pool-fill re-tags a shared-pool sibling basis, never pool-matches

Enumerated simulation warms each terminal leaf's solve from a stored basis, but
training captures one only for the single leaf its scenario-0 forward walked; the
other same-pool leaves would otherwise cold-solve the boundary-cut-heavy terminal
LP. `pool_fill_basis_cache` (`setup/node_graph.rs`, called once from
`StudySetup::simulate`, gated on `simulation_enumerated == Enumerated`) fills each
empty leaf slot with a same-pool sibling's `CapturedBasis` **re-tagged with the
target leaf's own `node_id`**. This is the licensed way to warm sibling fan leaves
WITHOUT weakening the node-tag filter above: the filter still matches `node_id` to
node exactly, and the re-tag is sound ONLY because same-`pool_id` nodes share one
frozen template, so a sibling's basis has identical column/cut-row shape and is
structurally valid at the target leaf. Reuse routes through the tolerant
slot-identity `reconstruct_basis` path, which re-validates shape. A slot the
stored-basis admit rule leaves empty is filled the same way, so a dropped
enumerated leaf may warm from a fitting sibling.

Relaxing the filter to a pool-id match instead of re-tagging — the exact
wrong-but-compiling alternative the paragraph above forbids — would let a basis
from a genuinely different-shaped LP through, which CLP accepts silently. The
precondition is same-`pool_id` ⇒ same-template-shape; if a future change bakes
node-specific data into a shared template or widens pool sharing across differing
shapes, this bypass is no longer safe. Read: `setup/node_graph.rs`
(`pool_fill_basis_cache`), `setup/orchestration.rs` (`StudySetup::simulate` call
site). Pinned by `enumerated_census_pool_fill_warms_previously_cold_leaves`
(`tests/simulation_integration.rs`): zero basis-consistency failures and
warm-vs-cold per-scenario cost bit-identity.

## NCS stochastic availability is a dimensionless factor

Non-controllable-source availability `α_r(ω) ∈ [0, 1]` is dimensionless. The
realized cap is `A_r = max_gen · clamp(mean + std·η, 0, 1)`. The
`non_controllable_stats.parquet` stores `(mean, std)` **as factors**, not as MW.
Read: `stochastic/noise.rs` (`transform_ncs_noise`).

## Lower-bound evaluation must patch NCS

`evaluate_lower_bound` patches NCS column bounds per opening via
`StageSolvePrep::run`'s internal `transform_ncs_noise` call, exactly as the
forward and backward passes do. Skipping the patch understates the bound (a
real bug caught during D15). The patch inputs ride on `StageContext`
(`ncs_max_gen`, `ncs_allow_curtailment`), the same struct every other solve
site reads.
Read: `training/lower_bound.rs`, `training/stage_solve_prep.rs`.

The lower bound patches stochastic load-balance rows the same way, through the
same `StageSolvePrep` call's load patch over the root opening's own load
segment. Skipping that patch leaves stage-0 load uncertainty out of the bound
and still compiles: the LP solves, converges, and reports a bound that never
reflects the root's load-noise draw. Pinned by
`lower_bound_root_lp_matches_the_forward_root_lp` in
`tests/patch_ownership_sweep.rs`.

## Per-level exchange in the backward pass

`exchange()` is called inside the reverse-topological sweep, once per
cut-sharing level (one node — == one stage — per level absent `nodes[]`), not
in a separate pre-pass before the loop. The level driver owns the one state
exchange and the one batched cut exchange per level; a per-node collective
would scale the collective count with node count.
Read: `training/backward_pass_state.rs` (`run_one_backward_level`).

### A multi-node level exchanges the node of each trial

A level with more than one node issues one `allgatherv` between the state
exchange and the node solves (`BackwardPassState::compute_node_visit_offsets`).
Each rank sends one `u64` per local trial slot, `max_local_fwd` entries: entry
`m` is the level index (the position in the level list, not the node's declared
id or its `NodePos`) of the node local trial `m` was routed to, and `NO_TRIAL`
for a slot with no real trial (`m >= local_work`). The gathered buffer holds
`num_ranks * max_local_fwd` entries, rank-major, with the same count and
displacement stride for every rank: the layout of the state exchange
(`ExchangeBuffers::state_at`), so entry `r * max_local_fwd + m` names the node
of the gathered trial state of rank `r`, local trial `m`. A level with one node
issues no such collective and touches none of its buffers: every trial visits
the sole node, and the offset is `fwd_offset`.

A cut's node-relative index (its `forward_pass_index` on a multi-node level) is
the number of trials routed to the same node that precede its trial in
rank-major, then local, order: the visits to the node from lower ranks
(`node_visit_offsets`) plus the cut's position among this rank's own trials for
the node. Each rank counts the per-`(rank, node)` visits (`rank_node_visits`)
from the gathered positions in one pass; the offsets and `check_visit_bound`
read that table, so the counts are not exchanged and have one source. A
gathered entry that is neither `NO_TRIAL` nor below the level's node count, or
a number of routed trials other than the level's real trial count
(`ExchangeBuffers::real_total_scenarios`), fails with `SddpError::Validation`.

Every rank enters the collective under the same condition and with the same
length. The condition is `level.len() > 1`, and the level list is
`NodeGraph::backward_cut_levels` of the graph every rank builds from the same
broadcast system (`StudySetup::from_broadcast_params`). The length is
`max_local_fwd`, which `RankDistribution::new` computes from the forward-pass
count and the rank count; a rank with no real trial sends `max_local_fwd`
padding entries. No config value, strategy, pool content or local trial count
decides whether or how much a rank sends. The buffers are sized in
`BackwardPassState::new` and the receive buffer starts filled with `NO_TRIAL`,
so a segment an in-process communicator leaves unwritten reads as a rank with
no trial.

The wrong-but-compiling alternatives:

- exchanging per-node counts: the offsets are right and no rank can tell which
  gathered state a peer's cut was computed at. Counts sent through the position
  buffer are read as positions: the visit counts and the indices come out
  wrong, or the level is rejected for its total;
- padding with a valid level index: the padded slots of a rank with fewer
  trials are counted as visits to that node, and the level is rejected for one
  routed trial too many per padded slot;
- starting the receive buffer at a valid level index: a segment left unwritten
  is counted as trials of that node;
- deciding the exchange, or its length, from a value a rank computes locally
  (its trial count, sending only its real trials): one rank skips a collective
  its peers enter, or the uniform counts and displacements no longer describe
  what was sent, and a multi-rank run blocks or gathers misaligned segments. No
  single-process test sees it; the rank symmetry holds by the derivation above.

Read: `training/backward_pass_state.rs` (`compute_node_visit_offsets`,
`check_visit_bound`, `build_trial_routing`, `NO_TRIAL`,
`BackwardPassState::new`), `training/state_exchange.rs` (`ExchangeBuffers`),
`training/session/rank_distribution.rs` (`RankDistribution::new`),
`setup/node_graph.rs` (`NodeGraph::backward_cut_levels`),
`training/backward/by_scenario.rs` (`process_by_scenario_backward`),
`training/backward/by_node.rs` (`by_node_finish`). Pinned by, in
`training/backward_pass_state.rs`:
`node_positions_give_every_rank_the_same_visit_counts` (two ranks' views of one
level hold the same gathered positions and visit counts; the higher rank's
offsets are the lower rank's counts),
`node_relative_indices_enumerate_each_node_without_collision` (the index a cut
carries is its trial's rank among the gathered positions of its node, and the
two ranks' indices cover each node's range once),
`a_single_node_level_issues_no_collective`,
`a_malformed_node_position_exchange_is_rejected` and
`an_unwritten_peer_segment_reads_as_a_rank_with_no_trial`. Every test is one
rank's view through a communicator that writes that rank's segment only; no
in-process test runs two cooperating ranks.

## Each scored cut's trial state is stored by slot

A pool that stores trial states (`CutPool::enable_trial_states`, or
`CutPool::enable_trial_states_within` for a window) holds, for each slot whose
cut was written scored, the state the cut was computed at, until a window drops
it (the window section below; without a window every scored slot keeps its
state): one row at the pool's projected dimension (`CutPool::state_dimension`),
addressed by the slot (`CutPool::trial_state`) and written by
`CutPool::store_trial_state` through `CutStateProjection::gather_trial_state`,
the single owner of "global state to projected state". A pool's rows are those
of its own cuts: it holds only the states of the trials routed to its node.
`SlotRole::scored` records that the pool stored states when the slot was
written (`add_cut` writes it); enabling the store afterwards does not change the
flag of a cut already written, and a loaded cut keeps the flag its record
carries. A loaded scored slot receives its stored row through
`CutPool::load_trial_state` once the session has enabled the store (below); one
the checkpoint holds no row for, because its source's window dropped the state,
has none. A pool that does not store states allocates nothing. Enabling
reserves `capacity * state_dimension * 8` bytes, of which
`stored_trial_states() * state_dimension * 8` hold a state, and per slot one
`u32` for the slot-to-row lookup and one `f64` and two `u32` for the ownership
record. With a window the rows reserved are those of the window section.

**Who enables it.** `TrainingSession::new` enables the store on every rank, on
the pool of every node of `interior_cut_nodes` (stage 1 or later, with a
successor), through one total match over the strategy variants: `Level1` stores,
with `enable_trial_states_within` when the strategy carries a window and
`enable_trial_states` otherwise; `Dynamic` and no strategy store, with
`enable_trial_states`, only when `exports.states` is on. `exports.states` forces
the store on whatever the strategy, and also on the root's pool (stage 0, with a
successor), which stores without a window because selection never scores or
expires it. The leaf pool receives no cut and never stores. The enabling in
`TrainingSession::new` is the only place that decides which pools store, and
nothing else allocates per-node states: the exported states are what the pools
store, each pool's by slot at the pool's dimension, and a run without
`exports.states` that is not `level1` allocates no row. A forced store only sets
`scored` on the cuts a run generates, which `dynamic` and a run without selection
never read, so forcing it changes no result. The strategy comes from the config
the CLI broadcasts and the node set from the graph every rank builds from it, so
the same pools store, with the same window, on every rank for the whole run.

**Loaded rows.** A `resume` and a `warm_start` load carry the rows of the
checkpoint's states files (a simulation-only load keeps none).
`check_full_fcf_load` places each row at the slot its cut occupies in the loaded
pool through `LoadedSlotMap::dest`, the stored slot under a resume and the
record's rank under a warm start, which packs. It refuses, as
`FullFcfLoadError::FcfConstruction` naming the pool, a states file of a pool the
study does not have, of another row dimension than the study's, or with a row at
a slot that holds no cut. Right after the enabling,
`TrainingSession::new` stores each row with `CutPool::load_trial_state`
in every pool that stores trial states and drops the rows of every other pool,
so the enabling stays the one place that decides which pools store. The loader
stores nothing: a store it created would be unbounded where the strategy asks for
a window. No ownership rebuild follows: the record is empty and its scanned mark
is 0, so the pool's first `update_ownership` scans every loaded slot against every
loaded state, which is the rebuild. Every rank runs the load on the whole
checkpoint, so every rank stores the same rows.

**Two fill sites.** Sampled: `run_one_backward_level` calls
`store_level_trial_states` once per level, after
`CutSyncBuffers::sync_level_records`, on every rank. One loop walks the level's
real trials rank-major, then local (`ExchangeBuffers::actual_counts`), takes
each trial's level index from the gathered node positions
(`BackwardPassState::trial_node_recv`) and counts the trials of each node; the
count before a trial is the node-relative index of its cut, which is the cut's
`forward_pass_index`, and the state is `ExchangeBuffers::state_at(rank, local)`.
A single-node level has no positions to read (the buffer holds padding or an
earlier level's content there): every trial visits the sole node, and the count
is the global real-trial index its cut carries. The loop is the same for this
rank's cuts and for the cuts of peers, which arrive without a state; the
gathered buffers are the only place the state a cut was computed at exists on a
rank that did not compute it. Enumerated: `run_enumerated_backward` calls
`store_trial_state` right after `add_cut`, for pass index 0, with the node's
persisted outgoing state the cut was computed at.

Every condition that decides whether the fill runs or what it reads is the same
on every rank: the pool's `stores_trial_states` (above), `level.len() > 1`,
`actual_counts` (the forward-pass count and the rank count), and the gathered
states and positions. The fill issues no collective. Selection runs on each
rank's own pools with no collective, so a rank whose pools held other states
than its peers' would select other cuts.

The wrong-but-compiling alternatives:

- the positional prefix `x_hat[..n_slots]` in place of the gather: it equals the
  gather only for a pool that projects the leading dimensions, and a pool that
  drops a leading region stores the wrong dimensions. Pinned by
  `gather_trial_state_follows_the_projection_not_position` and
  `a_rank_stores_the_state_of_a_peers_cut_at_a_sibling_node` (one pool projects
  the lag dimensions), both of which fail under it.
  `a_level1_run_stores_states_at_each_pools_projected_dimension` does not: its
  reduced pool drops a trailing region, so its prefix is its gather, and the
  test pins the projected dimension and the values, not the gather index;
- on a multi-node level, the gathered row at the cut's node-relative index: it is
  the row of the cut's trial only on a single-node level, and on a multi-node
  level names a trial routed to another node. Pinned by
  `a_rank_stores_the_state_of_a_peers_cut_at_a_sibling_node`, which fails under
  it while the single-node test passes;
- storing a state only for the cuts this rank generated: a peer's cut has no
  state. The mutation shown is a fill that walks rank 0's trials only, which
  `a_rank_stores_the_state_of_a_peers_cut_at_a_sibling_node` and
  `a_single_node_level_stores_each_trial_state_at_its_cut` fail on the peer's
  cuts (`None` for `Some`). Storing inside `by_scenario_finish` or
  `by_node_finish` is the same alternative and is not itself pinned by a test.
  That the ranks then select different cuts is not demonstrable in-process;
- reading the gathered positions on a single-node level: the buffer is padding or
  an earlier level's content, so the index is out of bounds or names another
  node. A training run fails at the fill, in
  `a_level1_run_stores_the_trial_state_of_every_generated_cut` among others;
- walking the padded trial slots of a rank with fewer trials than
  `max_local_fwd`: they have no cut, and a debug build of `store_trial_state`
  refuses the slot. Pinned by
  `a_single_node_level_stores_each_trial_state_at_its_cut`;
- enabling the store for a strategy other than `Level1` when `exports.states` is
  off, or setting `scored` on the cuts already written when the store is enabled.
  Pinned by `a_run_without_level1_stores_no_trial_state`, which fails on its
  `dynamic` leg under the first: the mutation shown admits `Dynamic` in the
  enabling match of `TrainingSession::new` whatever `exports.states` says; and by
  `a_slot_role_is_fixed_when_the_slot_is_written` and
  `a_pool_that_stores_trial_states_marks_its_cuts_scored` under the second;
- forcing the store for `exports.states` in the interior pools only: the root's
  cuts keep no state and the checkpoint holds no states file for the root pool.
  The mutation shown removes the root's arm of the enabling; under it
  `a_level1_run_with_exported_states_keeps_every_trial_state`,
  `a_dynamic_run_with_exported_states_stores_every_trial_state` and
  `exported_states_make_every_pool_that_receives_cuts_store_without_level1` fail.
  Leaving `dynamic` and the run without a strategy out of the forcing fails the
  last two;
- forcing the root's pool without `exports.states`, which `level1` selection never
  scores: `level1_without_exported_states_stores_in_the_interior_pools_only` fails;
- giving the root's pool the strategy's window: no step expires the root's states,
  so a bounded store would claim a window it does not honour. Not pinned by a test;
- deciding the enabling or the fill from a value a rank computes locally (its
  trial count, the cuts it generated): the ranks' stores differ. Not
  demonstrable in-process;
- not storing the loaded rows: a resumed `level1` pool holds no state for the
  scored cuts it loaded, so its first selection run scores them at the new run's
  states only and deactivates cuts that owned a state. The mutation shown skips
  the storing in `TrainingSession::new`; under it
  `a_resumed_level1_pool_holds_the_states_and_record_of_the_interrupted_run`
  fails on a stored state, and
  `a_warm_started_cut_whose_state_aged_out_competes_at_the_new_states` on a
  loaded state;
- placing a row at its stored slot under a warm start: the load packs the cuts,
  so after a vacant slot of the source every row sits on another cut. Pinned by
  `a_stored_state_follows_its_cut_to_the_slot_the_load_puts_it_at`; no training
  run shows it, since a chain source leaves no vacant slot;
- creating the store in the loader, or storing before the enabling: the
  store is unbounded where the strategy asks for a window, and the enabling is no
  longer the one owner of who stores. The mutation shown has the loader call
  `enable_trial_states` for each pool it holds rows for, before the enabling, which
  then skips a pool that already stores. Pinned by
  `level1_with_a_window_resume_equals_the_uninterrupted_run`
  (`tests/periodic_checkpoint.rs`), which fails under it: the states of the
  iterations after the resume never leave the window.

Read: `cut/trial_states.rs`, `cut/pool.rs` (`CutPool::enable_trial_states`,
`CutPool::enable_trial_states_within`, `CutPool::store_trial_state`,
`CutPool::load_trial_state`, `CutPool::trial_state`, `add_cut`),
`policy/full_fcf_load.rs` (`place_trial_states`, `LoadedSlotMap::dest`),
`lp/indexer/cut_state_projection.rs` (`gather_trial_state`),
`training/backward_pass_state.rs` (`store_level_trial_states`,
`run_one_backward_level`, `run_enumerated_backward`),
`training/state_exchange.rs` (`ExchangeBuffers::actual_counts`),
`training/session/mod.rs` (`TrainingSession::new`). Pinned by
`a_trial_state_row_is_addressed_by_its_slot` (`cut/trial_states.rs`),
`a_pool_that_stores_trial_states_marks_its_cuts_scored` and
`a_slot_role_is_fixed_when_the_slot_is_written` (`cut/pool.rs`),
`a_loaded_state_is_stored_at_a_scored_prefix_slot_past_the_reserve` (`cut/pool.rs`:
a row at a scored loaded slot of a bounded store past its reserve, and the first
`update_ownership` equal to the rebuild),
`loading_a_state_at_an_unscored_slot_panics_in_debug` (`cut/pool.rs`),
`a_stored_state_follows_its_cut_to_the_slot_the_load_puts_it_at` and
`a_stored_state_the_study_cannot_hold_is_refused_naming_its_pool`
(`policy/full_fcf_load.rs`), and, in `tests/cut_basis.rs`,
`a_resumed_level1_pool_holds_the_states_and_record_of_the_interrupted_run` (a
resumed pool's roles, stored states and rebuilt record equal the interrupted
run's, with and without a window) and
`a_warm_started_cut_whose_state_aged_out_competes_at_the_new_states` (a warm
start restores the source's rows at their slots, and cuts whose state the
source's window dropped own states and are deactivated at the new run's),
`gather_trial_state_follows_the_projection_not_position`
(`lp/indexer/cut_state_projection.rs`),
`a_rank_stores_the_state_of_a_peers_cut_at_a_sibling_node` and
`a_single_node_level_stores_each_trial_state_at_its_cut`
(`training/backward_pass_state.rs`: the gathered states of two ranks, two
sibling nodes, one pool projecting a non-leading dimension, and a padded slot),
and, in `tests/cut_basis.rs`,
`a_level1_run_stores_the_trial_state_of_every_generated_cut`,
`a_level1_run_stores_states_at_each_pools_projected_dimension`,
`an_enumerated_level1_run_stores_one_trial_state_per_cut` and
`a_sampled_fan_stores_only_each_nodes_own_trial_states` (the states sit at the
generated slots, one per cut, at the pool's dimension; their values are pinned
by the unit tests above), `a_run_without_level1_stores_no_trial_state` (no pool
stores without `level1` and without `exports.states`),
`level1_without_exported_states_stores_in_the_interior_pools_only`,
`a_level1_run_with_exported_states_keeps_every_trial_state`,
`a_dynamic_run_with_exported_states_stores_every_trial_state` and
`exported_states_make_every_pool_that_receives_cuts_store_without_level1` (with
`exports.states` every pool that receives cuts, the root's included, stores the
state of each); and, in `tests/mpi_wire.rs` on a
sampled fan whose two sibling nodes receive trials with different states,
`level1_thread_shapes_agree_on_a_branching_fan` and
`level1_declaration_order_is_invariant_on_a_branching_fan` (each sibling pool
stores one state per generated cut, the two pools store different states, and
the stored states do not depend on the thread count or the declaration order)
and, under an MPI launcher only, as the ownership section below states,
`level1_world_size_is_invariant_on_a_branching_fan` (nor on the rank count).

## A cut's value at a state is canonical

A cut's value at a stored state is the `gemm_block` element plus the cut's
intercept, added once, and the ownership record (`cut/ownership.rs`) computes it
no other way. Every call passes at most `ROW_BLOCK` slots and `M_BLOCK` states.
An incremental update evaluates a few new cuts against many old states and a
rebuild evaluates every cut against every state, so the two agree bit for bit
only because an element of `gemm_block` does not depend on which other rows and
columns share its call. That independence is a property of one build of the
`matrixmultiply` kernel, single-threaded and unfused (`gemm_block`'s
determinism section), not of the arithmetic.

The independence is pinned for up to `PINNED_MAX_ROWS` rows,
`PINNED_MAX_COLUMNS` columns and an inner dimension up to the largest of
`INNER_DIMS` (`gemm.rs`), and nowhere else. The kernel's column block (`D_NC`)
is never reached, because a call passes at most `M_BLOCK` states. An inner
dimension above the largest of `INNER_DIMS` is summed in further blocks of
`D_KC` and is drawn by no property; the inexact property draws 3, 257, 300 and
600.

The wrong-but-compiling alternatives:

- a hand-written dot product in one of the two paths: for an inner dimension
  above `D_KC` it rounds differently from the blocked kernel, so a state taken
  over by a later cut carries another best value than the rebuilt record's. The
  mutation shown computes the second step's value as a plain left-to-right dot
  product, which `incremental_record_equals_rebuild_on_inexact_values` fails on
  a case of dimension 300. The integer-valued property cannot show it, since
  every summation order is exact there;
- more rows or columns than the pinned ones in one call, such as all slots in
  one block: the shape is outside what the block-shape property draws. The
  mutation shown raises `ROW_BLOCK` above `PINNED_MAX_ROWS`: only
  `ownership_blocks_stay_inside_the_pinned_gemm_envelope` fails and no
  comparison of values does, so that test and the single value function are the
  guard. More states than the column block in one call is not demonstrated.

Read: `cut/ownership.rs` (`scan_block`, `ROW_BLOCK`, `M_BLOCK`), `gemm.rs`
(`gemm_block`, `PINNED_MAX_ROWS`, `PINNED_MAX_COLUMNS`).
Pinned by `gemm_block_element_is_independent_of_the_block_shape` (`gemm.rs`: an
element equals the same element alone, in column chunks and in row chunks),
`resolved_kernel_is_the_unfused_baseline` (`gemm.rs`: the kernel the build
resolves rounds each product), `ownership_blocks_stay_inside_the_pinned_gemm_envelope`,
`incremental_record_equals_rebuild_and_reference` (structure, on values every
summation order computes exactly) and
`incremental_record_equals_rebuild_on_inexact_values` (`cut/ownership.rs`).

## The ownership record is a cache of the ownership function

Under the Level 1 rule of de Matos, Philpott and Finardi (2015), a stored trial
state has one owner: the lowest live slot whose value at the state equals the
maximum over the live set, compared exactly, with no tolerance. A slot is live
when it is scored, or when it is an unscored loaded (`Prefix`) slot whose
activity flag is set; a `Vacant` slot and an unscored `Generated` slot are not
live. `is_live` owns the predicate. The activity flag of a scored slot is never
read: it is an output of selection, derived from this record by
`CutPool::ownership_activity_updates`, and a deactivated scored cut still owns
the states it wins. The flag of an unscored loaded slot is an input: a scan
reads it once, so a flip after the scan leaves the record stale until
`rebuild_ownership`. Nothing flips one during training: the budget skips loaded
cuts and the derivation lists scored slots only.
`selection_reference::owners` is the brute-force statement of the rule that the
record is compared with.

`CutPool` keeps, with the trial-state store, a record with one `f64` and two
`u32` per slot: the best value and the owning slot of the state stored at the
slot, and the number of states the slot's cut owns, plus the number of slots
already scanned. It is allocated by `enable_trial_states` or
`enable_trial_states_within`, extended by `grow`, and a pool without a store has
no record, so `update_ownership` and `rebuild_ownership` are no-ops there. It is
never written to a checkpoint and never exchanged between ranks: each rank
derives it from its own pool.

`update_ownership` takes the slots written since the last call in two steps,
each state block being at most `M_BLOCK` states: each new stored state against
every live slot below `populated()`, giving its owner and best value, then each
new live slot against every state stored before the call, taking a state over
only when `takes_over` says so. It then adjusts the counts and moves the
scanned mark to `populated()`. The state blocks run in parallel with each
state's result written to its own position, and the results are committed
sequentially in slot order, so the record does not depend on the thread count.
`rebuild_ownership` resets the record and runs the same update from slot 0; it
costs about the pool's cut count squared times its dimension in multiply-adds
on every rank. `state_owner` and `state_best_value` are `None` for a slot with
no stored state or one the record has not scanned; bound slot-indexed reads by
`populated()`, since like `trial_state` they panic at a slot at or above the
capacity.

`takes_over` is the only comparison of two cut values in the module, used by
every scan in ascending slot order: a later cut takes a state only when
strictly higher, so ties stay with the lowest slot. Reversing the tie direction
is a change to `takes_over` and to the reference rule, nothing else. The
update relies on every slot written since the last call lying at or above the
scanned mark, which holds while every `forward_pass_index` is below the pool's
`visit_stride` (the append-only section above), and on a state being stored
before the update that passes its slot. A slot written or a state stored below
the mark is not seen until `rebuild_ownership`. A debug build refuses the second
case: `Ownership::update` asserts that no state stored below the mark lacks an
owner, which `rebuild_ownership` never trips because it resets the mark first.
A load needs no `rebuild_ownership`: its rows are stored while the mark is 0, so
the first `update_ownership` scans them all.
Dropping a stored state, which the window section below describes, clears its
entry and its owner's count, so `state_owner` and `state_best_value` read `None`
for it as for a slot that never stored one.

The wrong-but-compiling alternatives:

- `>=` in the takeover: ties then go to the highest slot among those scanned,
  so the outcome follows arrival order and differs from the reference. Pinned
  by `the_record_reproduces_every_reference_fixture` (on `ties`),
  `a_new_cut_takes_a_state_only_when_strictly_higher` and
  `incremental_record_equals_rebuild_and_reference`, which fail under it;
- reading the activity flag of a scored cut in the live predicate: a
  deactivated cut can never win a new state. Pinned by
  `an_inactive_scored_cut_wins_a_new_state`, which fails under it, as does the
  property through the update's assertion that a stored state's own slot is
  live;
- skipping the second step: states stored before a call never meet the cuts
  written after it, and the record differs from a rebuild. Pinned by
  `a_new_cut_takes_a_state_only_when_strictly_higher`,
  `an_inactive_scored_cut_wins_a_new_state`, both properties and, through its
  rebuild comparison, `ownership_record_is_identical_across_thread_counts`,
  which fail under it;
- counting a vacant slot as live: its zero intercept and coefficients compete
  in the maximum and the vacant slot owns states. Pinned by the `vacant_slot`
  fixture of `the_record_reproduces_every_reference_fixture` and by
  `incremental_record_equals_rebuild_and_reference`, whose draws leave vacant
  holes below the highest written slot, which fail under it;
- a thread-count dependence in the commit (committing in arrival order): not
  demonstrated, since no mutation of the shipped code depends on the thread
  count; `ownership_record_is_identical_across_thread_counts` compares 1, 4 and
  8 threads on a pool wider than `ROW_BLOCK` and with more states than
  `M_BLOCK`.

Read: `cut/ownership.rs` (`Ownership::update`, `is_live`, `takes_over`,
`scan`), `cut/pool.rs` (`CutPool::enable_trial_states`, `grow`,
`update_ownership`, `rebuild_ownership`, `state_owner`, `state_best_value`,
`owned_states`), `cut/selection_reference.rs` (`owners`). Pinned by
`the_record_reproduces_every_reference_fixture`,
`an_inactive_scored_cut_wins_a_new_state`,
`a_new_cut_takes_a_state_only_when_strictly_higher`,
`incremental_record_equals_rebuild_and_reference`,
`incremental_record_equals_rebuild_on_inexact_values`,
`windowed_record_equals_rebuild_and_reference` (the same comparison with states
dropped by a window),
`a_state_stored_below_the_scanned_mark_panics_in_debug` and
`ownership_record_is_identical_across_thread_counts` (`cut/ownership.rs`), and
`no_live_cut_leaves_a_state_unowned` (`cut/selection_reference.rs`: the rule
leaves a state unowned when no slot is live, which a stored state's scored slot
rules out in the record).

In a training run the record is pinned by the `level1_` gates of
`selection_on_determinism` (`tests/mpi_wire.rs`), which read every slot's best
value, owner and owned count through the public readers:
`level1_thread_shapes_agree_on_the_uniform_fixture` and
`level1_thread_shapes_agree_on_the_non_uniform_projection_fixture` (the record
does not depend on the thread count),
`level1_two_rank_stub_matches_single_rank_on_the_uniform_fixture` and
`level1_two_rank_stub_matches_single_rank_on_the_non_uniform_projection_fixture`
(nor on the two-rank buffer layout, with one trial on rank 0), and
`level1_declaration_order_is_invariant_on_the_uniform_fixture`,
`level1_declaration_order_is_invariant_on_the_non_uniform_projection_fixture`
and `level1_declaration_order_is_invariant_on_a_two_hydro_fixture` (nor on the
declaration order; the last on a state of two dimensions, where it also pins
that of bit-identical cuts none above the lowest slot is active). Every run of
those gates also asserts the rule from cut values the test sums itself: each
stored state's owner attains the maximum of the live cuts within a rounding
bound, and no live cut at a lower slot is bit-identical to it. No in-process
test runs two cooperating ranks, so none shows two ranks that route trials to
different sibling nodes deriving the same record, and no two-rank stub leg
runs on a level with several nodes, where the stub reads a stale per-pool cut
count. `level1_world_size_is_invariant_on_a_branching_fan` shows it only when
its test binary, built with the `mpi` feature of `cobre-comm`, is launched
under an MPI launcher with two or more ranks; under `cargo test` it is a
single-rank pass and pins nothing here.

## `level1` keeps one owner per stored state

Under `level1` a selection run sets the activity flag of every scored cut of an
interior pool from the ownership record (the section above): active when the
cut owns at least one stored state, inactive when it owns none.
`CutPool::ownership_activity_updates` is the derivation. It reads each slot's
role, its flag and `owned_states`, and compares no cut values: an active scored
slot that owns no state is a deactivation, an inactive scored slot that owns one
is a reactivation. No tolerance, no iteration counter and no strategy parameter
enter it, so a cut of the current iteration is judged like any other.

By slot role:

- generated and scored (every cut a `level1` run generates in an interior
  pool): live whatever its flag, active after a run exactly when it owns a
  state;
- loaded and scored (a record written scored): the same: it competes at each
  state the pool stores, the row loaded for it included, and is active after a
  selection run exactly when it owns one; one whose row its source's window
  dropped has no state of its own and competes at the stored ones;
- loaded and stateless: live while its flag is set, and in neither list, so it
  keeps the flag it was loaded with. While active it can own a state and keep a
  tied generated cut above it inactive;
- generated and stateless (written before the pool stored states): not live and
  in neither list;
- vacant: not live, in neither list, never activated.

`run_cut_management` runs it in every iteration `should_run` holds, for the pool
of each node of `interior_cut_nodes`: `expire_trial_states` (a no-op without a
window), `update_ownership`, then
`apply_updates(ownership_activity_updates())`. Selection drops no stored state,
only the window of the lookback section does, so a `level1` run without a window
exports every state of the run. The call site meets what the update relies on:

- `TrainingSession::new` enabled the store on these pools before the first cut,
  so every generated cut there is scored;
- the backward pass has ended, so every cut of the iteration, a peer's
  included, is written and its trial state stored;
- every slot written since the last run lies above the slots written before it
  (the append-only section);
- nothing has flipped a loaded cut's flag since the last run;
- the derivation walks slots below `populated()` and lists scored slots only,
  so a vacant slot is never activated;
- no collective is involved: every rank holds the same cuts and states and
  runs selection in the same iterations, from the broadcast config.

**Envelope invariant.** Right after a selection run in which the budget evicts
nothing, each stored state's owner is active, so the active cuts of the pool
attain the maximum of its live cuts at every stored state; with a window the
stored states are those inside it (the section below). Both conditions are
needed. A state stored after the run can be owned by an inactive cut until the
next run. And the budget, which runs after selection in the same iteration,
clears the flag of scored cuts that own states.

**Budget.** The record does not read the flag the budget clears, so an evicted
cut still owns its states. The next selection run finds it inactive and owning a
state and reactivates it, and the budget is applied again in that iteration; the
template refreeze that follows encodes the flags the budget left. With no budget
a deactivated cut returns only when it comes to own a state.

The wrong-but-compiling alternatives, each shown by a mutation:

- a tolerance band in the derivation (keeping a scored cut whose value is
  within a tolerance of some state's best): duplicates and near-ties stay
  active and the active set is no longer the set of owners. Under it
  `level1_gives_a_tied_state_to_the_lowest_slot`,
  `level1_deactivates_a_generated_cut_tied_with_a_lower_loaded_cut`,
  `level1_keeps_one_of_two_identical_cuts`,
  `level1_follows_the_reference_rule_on_every_fixture`,
  `level1_matches_the_reference_rule`,
  `level1_deactivates_cuts_in_a_training_run`,
  `level1_keeps_the_owner_of_every_stored_state_after_a_training_run` and both
  `with_level1` oracle legs fail;
- exempting the cuts of the latest iteration: a new cut that owns no state
  stays active. Under it
  `level1_deactivates_a_current_iteration_cut_that_owns_no_state` fails, with
  `level1_gives_a_tied_state_to_the_lowest_slot`,
  `level1_keeps_one_of_two_identical_cuts`, the reference tests, the two
  training tests above and both oracle legs;
- reading the activity flag of a scored cut when deciding who competes: a
  deactivated cut can never win a state again. Under it
  `level1_reactivates_the_owner_of_a_new_state`,
  `level1_drops_the_owner_of_a_state_that_left_the_window`,
  `level1_follows_the_reference_rule_on_every_fixture` (which starts every
  scored cut inactive in one of its two passes), the `level1` properties and
  `level1_keeps_the_owner_of_every_stored_state_after_a_training_run` fail;
- giving a tie to the highest slot (`>=` in `takes_over`): of two identical
  cuts the newer survives. Under it
  `level1_gives_a_tied_state_to_the_lowest_slot`,
  `level1_keeps_one_of_two_identical_cuts` and the reference tests fail, and so
  does the lowest-slot assertion of one `with_level1` oracle leg: the leg in
  which an active cut has an identical twin, which is not the same leg on both
  solver backends;
- a derivation that never emits a reactivation: an owner the budget evicted
  stays inactive. Under it `level1_reactivates_an_owner_the_budget_evicted`,
  `level1_reactivates_the_owner_of_a_new_state`,
  `level1_drops_the_owner_of_a_state_that_left_the_window` and the reference
  tests fail.

Read: `cut/pool.rs` (`CutPool::ownership_activity_updates`, `update_ownership`,
`apply_updates`, `enforce_budget`), `cut/ownership.rs` (`is_live`,
`takes_over`), `cut/cut_selection.rs` (the module documentation),
`training/session/mod.rs` (`run_cut_management`, `TrainingSession::new`),
`cut/selection_reference.rs` (`active_after`). Pinned by, in
`cut/cut_selection.rs`: `level1_selects_the_two_steep_cuts_of_paper_figure_four`,
`level1_loses_the_middle_cut_of_paper_figure_four_at_an_unvisited_state`,
`level1_reactivates_the_owner_of_a_new_state`,
`level1_drops_the_owner_of_a_state_that_left_the_window`,
`level1_gives_a_tied_state_to_the_lowest_slot`,
`level1_deactivates_a_generated_cut_tied_with_a_lower_loaded_cut`,
`level1_deactivates_a_current_iteration_cut_that_owns_no_state`,
`level1_keeps_one_of_two_identical_cuts`,
`level1_follows_the_reference_rule_on_every_fixture`,
`level1_matches_the_reference_rule` and its sibling properties; in `cut/pool.rs`:
`ownership_activity_updates_leave_an_unscored_generated_cut_alone`; in
`tests/cut_basis.rs`: `level1_deactivates_cuts_in_a_training_run` (after a run,
a scored cut is active exactly when it owns a stored state),
`level1_keeps_the_owner_of_every_stored_state_after_a_training_run` (the
envelope invariant, and the run's record against a rebuild),
`level1_reactivates_an_owner_the_budget_evicted` (the budget paragraph, as
existence over a run's events, which carry counts and no slot),
`a_level1_run_with_exported_states_keeps_every_trial_state` (no selection run
drops a stored state),
`a_warm_started_scored_cut_is_selected_like_a_generated_one` (a loaded scored cut
changes its flag in a warm-started run) and
`a_warm_started_unscored_cut_keeps_its_flag` (a loaded unscored cut never does);
in `tests/cut_oracles.rs`:
`cut_oracles_hold_on_deactivated_cuts_under_level1` (the run leaves inactive cuts
and the validity oracle checks them); in `tests/extensive_form_oracle.rs`:
`branching_tree_with_level1_stays_below_and_closes_to_the_extensive_form` and
`chain_with_level1_stays_below_and_closes_to_the_extensive_form` (among
bit-identical cuts only the lowest slot is active, every stored state's owner is
active, and both bounds close).

## The lookback window removes trial states, never cuts

A `level1` strategy with `lookback_iterations: Some(w)` keeps the trial states of
the last `w` iterations of the run; with `None` every state of the run is kept
and nothing below applies. The key
`training.cut_selection.selection.lookback_iterations` sets the window: its
config type (`SelectionMethod::Level1` in `cobre-io`) is `Option<NonZeroU32>`, so
a zero fails at load with a message naming the key, and
`parse_cut_selection_config` maps it onto the strategy. The parse rejects a value
below `check_frequency`, because the states of the iterations between two
selection runs would then be dropped before a run scores them; a value equal to
`check_frequency` is admitted. The window counts iterations of the current run
from its first, under every `policy.mode` and with `policy.boundary`; a resumed
run counts the iterations before its checkpoint, as the uninterrupted run does.

**The boundary.** The window is a slot boundary on this run's clock.
`CutPool::window_start_slot`, beside `slot_index`, returns the first slot whose
state is inside the window at a selection run of iteration `k`. With
`r = k - iteration_base + 1`, the iterations this run has completed, it is slot 0
while `r < w`, and `slot_base + (r - w) * visit_stride` from `r = w` on: the first
slot of iteration `k - w + 1`. A generated cut's state is therefore inside
exactly when its generating iteration exceeds `k - w`. A state loaded at a
`Prefix` slot (`CutPool::load_trial_state`) is at iteration 0 of the run's clock,
whatever iteration its record carries: it is inside while `r < w` and leaves at
the run's iteration `w`. One loaded at a `Generated` slot, under a resume, is at
the iteration its slot gives it, as in the uninterrupted run.

**The step.** `CutPool::expire_trial_states(k)` removes the states that left. For
each slot from the previous boundary up to the new one that holds a state, it
decrements the count of the state's owner when the record has one, clears the
entry and returns the row to the store; the boundary only moves up, and a pool
without a window does nothing. It removes states and nothing else: the slot's
role, coefficients, intercept and flag are untouched and `is_live` is unchanged,
so a scored cut whose state left stays live and competes at every state that
remains. The derivation of the section above then deactivates it when it owns
none and keeps it active when it owns one. With a window the scored set of a pool
is the states inside it, and the envelope invariant holds at those.
`rebuild_ownership` needs no change: it resets the record and scans the stored
states, which are those inside the window. On a pool given a window below
`check_frequency` directly, which no configuration reaches, the states of the
iterations between two runs are dropped before a run scores them, and the record
still equals a rebuild.

**Where it runs.** `run_cut_management` calls `expire_trial_states` for each
interior pool before `update_ownership`; swapping the two scores states that are
about to leave and ends in the same record, so the order is cost only.
`windowed_record_equals_rebuild_and_reference` runs the swapped order on a clone
at every run and compares the record, the stored states and the derived flags.
`TrainingSession::new` enables the store with
`enable_trial_states_within(w, check_frequency)` when the strategy carries
`Some(w)`.

**Memory.** `enable_trial_states_within` reserves
`(w + check_frequency) * visit_stride` rows, never more than one per slot: the
`w` iterations the previous run kept plus the `check_frequency` iterations added
since, which is what a pool holds right before a run when selection runs on
schedule and a pool receives at most `visit_stride` cuts per iteration. The store
is a `u32` lookup per slot, cleared when a state is dropped, over rows handed out
from a free list; `grow` extends the lookup, and a store without a window also
appends a row per new slot. Past the reserve the store appends a row and never
overwrites one, and that fill allocates; the reserve counts no loaded state, so a
loaded state held beside a full reserve extends the store too. A clone of the
pool, such as the one the Python `Policy` object holds, carries the reserved
rows, so the window bounds it as well.

**Same on every rank.** The window is a field of the strategy, which rank 0
parses from the config and broadcasts in `BroadcastConfig::cut_selection`, and
every rank builds its setup from that strategy, so each rank enables the same
store with the same window. The boundary is a function of the window, the
iteration and the slot layout; no collective reads the window and no value
derived from it crosses ranks.

The wrong-but-compiling alternatives:

- clearing the scored flag of a cut when its state leaves: the cut is no longer
  live and stops competing at the states that remain, so a state it would own
  goes to another cut. Under it
  `a_loaded_scored_cut_competes_only_at_this_runs_states_once_its_state_left_the_window`,
  `level1_drops_the_owner_of_a_state_that_left_the_window`,
  `windowed_record_equals_rebuild_and_reference` and
  `a_level1_window_keeps_only_the_states_of_the_last_iterations` fail.
  Deactivating the cut instead is shown only at the derivation's lists, since the
  next run recomputes the flag: the two fixtures and the property fail, and
  neither training test does;
- dropping a state without taking it from its owner's count: the owner keeps
  owning a state that is gone, stays active, and the record differs from a
  rebuild. Under it
  `level1_drops_the_owner_of_a_state_that_left_the_window`,
  `a_loaded_scored_cut_competes_only_at_this_runs_states_once_its_state_left_the_window`,
  `windowed_record_equals_rebuild_and_reference` and
  `a_level1_window_keeps_only_the_states_of_the_last_iterations` fail;
- reading the iteration a loaded record carries against this run's counter: the
  record holds the iteration of the run that wrote it, so its state would leave
  when this run's counter reaches that iteration plus `w`, not at the run's
  iteration `w`. The mutation shown keeps a loaded state for good, as a record
  of iteration 40 does in the fixture. Under it
  `a_loaded_scored_cut_competes_only_at_this_runs_states_once_its_state_left_the_window`,
  `a_loaded_state_leaves_a_window_at_the_runs_iteration_w` and
  `windowed_record_equals_rebuild_and_reference` fail;
- a boundary that keeps one iteration more or one fewer than the last `w`: under
  both, `the_window_start_slot_agrees_with_the_slot_formula`,
  `a_windowed_pool_never_holds_more_rows_than_its_reserve`,
  `level1_drops_the_owner_of_a_state_that_left_the_window`,
  `a_loaded_scored_cut_competes_only_at_this_runs_states_once_its_state_left_the_window`,
  `windowed_record_equals_rebuild_and_reference` and
  `a_level1_window_keeps_only_the_states_of_the_last_iterations` fail, and one
  fewer also fails `a_window_longer_than_the_run_changes_nothing`;
- returning a row to the free list and keeping the lookup entry: a dropped slot
  still reads its state. Under it `a_bounded_store_frees_the_row_of_a_dropped_state`,
  `a_windowed_pool_never_holds_more_rows_than_its_reserve`, both fixtures,
  `windowed_record_equals_rebuild_and_reference` and
  `a_level1_window_keeps_only_the_states_of_the_last_iterations` fail;
- doing nothing when the window start lies above `populated()`, a tail of
  never-written slots: the states older than the window stay stored and owned.
  Under it `a_window_start_above_the_populated_slots_releases_every_older_state`
  and `windowed_record_equals_rebuild_and_reference` fail. The `populated()`
  bound that `expire_trial_states` puts on the start is not itself pinned: the
  same tests pass without it, since a slot above `populated()` holds no state;
- overwriting a row past the reserve instead of appending one: two slots share a
  row. Under it `a_bounded_store_extends_when_more_states_are_held_than_reserved`
  and `windowed_record_equals_rebuild_and_reference` fail;
- deciding the window, or whether a rank enables the bounded store, from a value
  a rank computes locally: the ranks then hold other states and select other
  cuts. Not demonstrable in-process.

Read: `cut/pool.rs` (`CutPool::window_start_slot`, `expire_trial_states`,
`enable_trial_states_within`, `slot_index`), `cut/ownership.rs`
(`Ownership::expire`), `cut/trial_states.rs` (`TrialStates::claim`, `release`,
`grow`), `cut/cut_selection.rs` (`CutSelectionStrategy::Level1`,
`parse_cut_selection_config`), `crates/cobre-io/src/config/training.rs`
(`SelectionMethod::Level1`, `positive_count`),
`crates/cobre-cli/src/commands/broadcast.rs` (`BroadcastConfig`),
`training/session/mod.rs` (`run_cut_management`, `TrainingSession::new`). Pinned
by, in `cut/trial_states.rs`: `a_bounded_store_frees_the_row_of_a_dropped_state`
and `a_bounded_store_extends_when_more_states_are_held_than_reserved`; in
`cut/pool.rs`: `the_window_start_slot_agrees_with_the_slot_formula` (the start
against the slot formula, loaded slots included) and
`a_windowed_pool_never_holds_more_rows_than_its_reserve` and
`a_window_start_above_the_populated_slots_releases_every_older_state` (a start
above `populated()` releases every older state, and the record equals a
rebuild); in
`cut/cut_selection.rs`: `level1_drops_the_owner_of_a_state_that_left_the_window`
and
`a_loaded_scored_cut_competes_only_at_this_runs_states_once_its_state_left_the_window`
(a loaded cut deactivated when it owns none, then winning a later state); in
`cut/ownership.rs`: `windowed_record_equals_rebuild_and_reference` (the record
against a rebuild and the states of the window against the reference rule, under
drawn windows, strides and run frequencies, with the swapped order of expiring
and updating on a clone); in `tests/cut_basis.rs`:
`a_level1_window_keeps_only_the_states_of_the_last_iterations` and
`a_window_longer_than_the_run_changes_nothing`; in `tests/mpi_wire.rs`:
`level1_window_thread_shapes_agree_on_the_uniform_fixture` and
`level1_window_thread_shapes_agree_on_the_non_uniform_projection_fixture` (with
a window, the states kept, the record and the activity flags do not depend on
the thread count, in runs where states left and their cuts stayed scored; the
first also requires a reactivation). The two load rejects are pinned by
`parse_level1_lookback_below_check_frequency_is_rejected` and
`parse_level1_lookback_equal_to_check_frequency_is_admitted`
(`cut/cut_selection.rs`), `a_zero_lookback_window_names_its_key`
(`crates/cobre-io/src/config/training.rs`),
`parse_config_rejects_a_zero_lookback_window` and
`with_overrides_rejects_a_zero_lookback_window`
(`crates/cobre-io/src/config/mod.rs`), and
`validate_rejects_a_zero_lookback_window` and
`validate_rejects_a_lookback_window_below_the_check_frequency`
(`crates/cobre-cli/tests/cli_validate.rs`, each after a window equal to
`check_frequency` validates); the broadcast of the window by
`broadcast_config_carries_the_level1_window`
(`crates/cobre-cli/src/commands/broadcast.rs`: a `Level1` strategy with a window
survives the postcard round trip).

## Binding activity is recorded only for a configured reader

A cut's binding activity (`CutMetadata::active_count`,
`CutMetadata::last_active_iter`) has two readers: the eviction key of the
active-cut budget (`CutPool::enforce_budget`) and the seed window of a lazy
solve under the `dynamic` method (`build_initial_resident_set`).
`CutManagementConfig::tracks_binding_activity` owns the predicate: true when a
budget is set or the strategy is `CutSelectionStrategy::Dynamic`. It is a method
over the two config values, not a stored flag, and matches every strategy
variant by name, so a new variant must be classified to compile.

`TrainingSession::new` passes the predicate's value once to
`BackwardPassState::set_binding_tracking`, on every rank, and
`CheckpointLayout::new` reads it to decide whether a checkpoint carries the
counters (the activity clock paragraph above). Both inputs come from
the config every rank holds identically: the CLI broadcasts one `BroadcastConfig`
from rank 0, every rank builds its `StudySetup` from it
(`StudySetup::from_broadcast_params`), no `StudySetup` setter changes the
strategy variant or the budget, and the training entry
points of `StudySetup` copy both into the `TrainingConfig` the session reads.
The switch is therefore the same on every rank for the whole run. A `BackwardPassState` built without the session
keeps the default, which records.

With the switch off a sampled backward pass does none of the recording work:

- `extract_duals_from_view` copies no cut-row dual;
- `accumulate_opening_outcome` runs no slot loop;
- neither scheduler grows or zero-fills a worker's `slot_increments` and
  `metadata_sync_contribution`, or calls `fold_slot_increments_into_sync`;
- `run_one_backward_level` does not call `sync_stage_metadata`, so no per-worker
  sum, no `allreduce` and no `CutPool::record_binding` runs.

The workers read the switch from the threshold on the successor view
(`SuccessorOutcomes`, `SuccessorChild`), an option that is `None` when binding is
not tracked; the lazy solve reads the same option before
`accumulate_dcs_binding_counts`. Every cut then keeps `active_count = 0` and
`last_active_iter` at the value it started with (the activity clock paragraph
above). The state duals are
filled by the same routine in both states, so the cuts and the bounds are
bit-identical with the switch on and off. The worker buffers stay allocated.
The enumerated backward records no binding activity in either state.

The wrong-but-compiling alternatives:

- deciding at the `sync_stage_metadata` call from a value a rank computes
  locally (this rank's trial-point count, an empty contribution buffer, a
  pool's local content): one rank then skips a collective its peers enter and
  a multi-rank run blocks, which no single-process test sees. The early return
  of `sync_stage_metadata` on an empty pool is not such a decision: it rests on
  the populated count being the same on every rank, as the method's doc states;
- skipping only the reduction while the workers keep scanning cut-row duals
  and folding increments: no result changes and the work per solve stays.

Read: `config.rs` (`CutManagementConfig::tracks_binding_activity`),
`training/session/mod.rs` (`TrainingSession::new`),
`training/backward_pass_state.rs` (`BackwardPassState::set_binding_tracking`,
`run_one_backward_level`, `compute_one_backward_node`,
`process_stage_backward`, `sync_stage_metadata`), `training/backward/mod.rs`
(`SuccessorOutcomes`, `SuccessorChild`),
`training/backward/duals_extraction.rs` (`extract_duals_from_view`),
`training/backward/outcome_aggregation.rs` (`accumulate_opening_outcome`,
`accumulate_dcs_binding_counts`, `fold_slot_increments_into_sync`),
`training/backward/by_scenario.rs` (`process_by_scenario_backward`),
`training/backward/by_node.rs` (`process_stage_backward_by_node`),
`cut/pool.rs` (`CutPool::record_binding`, `CutPool::enforce_budget`),
`cut/dcs.rs` (`build_initial_resident_set`), `setup/mod.rs`
(`StudySetup::from_broadcast_params`), `setup/orchestration.rs`
(`StudySetup::train`, `StudySetup::train_with_solver_profiles`),
`crates/cobre-cli/src/commands/broadcast.rs` (`BroadcastConfig`,
`broadcast_value`). Pinned by
`tracks_binding_activity_only_for_a_budget_or_the_dynamic_method` (`config.rs`:
the truth table over a budget and every strategy),
`binding_is_recorded_only_when_tracking_is_on`
(`training/backward/tests.rs`: with the switch off, under both schedulers, the
pass issues no `u64` sum reduction, moves no slot's metadata, copies no
cut-row dual and touches no worker buffer, and its cuts equal those of the
pass with the switch on) and `binding_activity_is_recorded_only_for_a_reader`
(`tests/cut_basis.rs`: a `level1` run without a budget records nothing, a
budget-only run and a `dynamic` run record, and a budget that evicts nothing
gives the cuts and bounds of a run without one). No in-process test can show
one rank skipping alone; the rank symmetry holds by the derivation above.

## Backward opening order is warm-start-only

A trial point's backward openings are SOLVED in the installed `solve_order`
permutation (`OpeningTree::set_solve_order`, keyed by
`noise_key::build_noise_key_table` — the intrinsic shortest-chain order, a
nearest-neighbor + 2-opt minimum-distance path over the openings'
inflow-noise vectors; a stage below 3 openings keeps its σ-weighted key, the
live fallback that also owns the noise-dimension validation) but each
opening's outcome is WRITTEN and AGGREGATED by **canonical ω**. The
aggregation therefore carries no solve-order dependence: results are
declaration-order-invariant and run-to-run reproducible across thread and
rank shapes (the pinned gates). No config field selects the order.
CHANGING the order (a code change to `noise_key`) changes the warm-start
chain each opening's solve starts from, and at a degenerate optimum a
differently-warmed solve may settle on a different-but-equally-valid vertex
with different duals — the hot≠cold divergence the Cobre determinism contract
permits — so an order change re-checks the golden parity baselines instead of
assuming byte-identical outputs. Aggregating the outcome slice indexed by
solve position — or handing solve-order-permuted probabilities to
`RiskMeasure::aggregate_cut_into` — is the wrong-but-compiling alternative: it
makes the cut depend on solve order, silently
breaking declaration-order invariance and run-to-run reproducibility.
Read: `stochastic/noise_key.rs` (`build_noise_key_table`, `apply_chain_order`),
`training/backward/by_scenario.rs` (`process_by_scenario_backward` — solves by
`solve_order`, aggregates by canonical ω), `training/backward/outcome_aggregation.rs`
(`write_opening_outcome`). Pinned by the `opening_order_determinism` gate in
`tests/mpi_wire.rs` (threads=k / threads=1 / a same-shape repeat / a 2-rank
stub, bitwise `final_lb`) and the MPI SLURM Integration job's rank-invariance
comparison on `examples/4ree`.

## By-node scheduler is warm-start-only

The live scheduler spellings are `by_scenario` (the default) and `by_node`, both
under `training.parallelism.backward_scheduler`. The retired `trial_point` /
`opening_block` spellings are unknown-variant deserialize errors — a clean break
with no `serde(alias)` fallback — pinned by
`retired_scheduler_spellings_are_deserialize_error`
(`crates/cobre-io/src/config/training.rs`).

The opt-in by-node scheduler
(`training.parallelism.backward_scheduler = { method = by_node }`)
reassigns the backward pass's work unit from a whole trial point to an
opening-block: workers claim `(trial point, block)` units in any order from a
shared atomic counter, warm-chaining each block's openings from a fresh
frozen-LP load. Units are SOLVED in claim order — dependent on worker count and
scheduling timing — but each opening's outcome is WRITTEN into a per-`(m, ω)`
arena and AGGREGATED per trial point over CANONICAL ω, in ASCENDING m. The
generated cut set is therefore independent of claim order and worker count:
reordering claims changes only which worker warms which block, never which cut
is produced. Aggregating the arena in claim/solve-position order, or keying it
on the claim index instead of `(m, ω)`, is the wrong-but-compiling
alternative — CVaR's tail weighting is order-sensitive, so it silently breaks
CVaR reproducibility and declaration-order invariance the same way a
solve-order-keyed aggregation would break the by-scenario path above. An
active `dynamic` iteration always falls back to the by-scenario path: the
by-node scheduler's frozen-LP load is incompatible with the lazy solve's
cut-free core.
Read: `training/backward/by_node.rs`
(`process_stage_backward_by_node`'s claim loop,
`by_node_finish`'s per-`(m, ω)` arena and ascending-m aggregation),
`training/backward_pass_state.rs` (`compute_one_backward_node`'s
scheduler dispatch via `resolve_backward_scheduler`). Pinned by
`by_node_scheduler_determinism_expectation` and
`by_node_scheduler_determinism_cvar` in `tests/mpi_wire.rs` (threads=4
/ a same-shape threads=4 repeat / threads=2 / threads=1 / a `Rank0Of2`
2-rank stub, bitwise `final_lb`, on both an expectation and a `CVaR`
configuration), `by_node_degenerates_on_single_opening`
(by-node-vs-by-scenario equality on a single-opening case whose
resolved block count is `1`), and
`by_node_handles_non_uniform_cut_projection`
(by-node-vs-by-scenario equality on a case whose per-stage cut-state
projection dimension varies across stages).

**Hardest-first claim order is result-neutral.** Under `ByNode`,
claims are further ordered hardest-`(stage, block)`-first
(longest-processing-time, LPT) by the PREVIOUS iteration's per-`(stage,
block)` mean `simplex_iterations` pivot — never per-`(m, block)`, since
resampled trial points make per-m hardness noise where the opening-block
component is iteration-stable. The hardest-first order touches only the
claim decode: the per-`(m, ω)` write and the ascending-m aggregation above
are unchanged, so hardest-first-on and the canonical identity order produce
a bit-identical cut set and `final_lb`. Keying the order on per-`(m, block)`
pivots, reordering the arena or the aggregation instead of only the claim
decode, and a tie-break that leaves equal-mean blocks unordered (not a total
order) are each wrong-but-compiling: the first two reintroduce a
claim-order dependence the invariant above forbids; the third makes the
claim order itself nondeterministic across otherwise-identical runs.
`block_pivots_prev` is the previous iteration's fully-merged row —
`BackwardPassState::run` swaps it in from `block_pivots` once per call, never
per stage; reading `block_pivots` instead during the sweep is stale
(reset-then-partially-filled).
Read: `training/backward/by_node.rs`
(`process_stage_backward_by_node`'s `block_order`-indexed decode,
`hardest_first_block_order`, `identity_block_order`),
`training/backward_pass_state.rs` (`compute_one_backward_node`'s block-order
computation, the `run` swap). Pinned by
`hardest_first_claim_order_is_result_neutral` in `tests/mpi_wire.rs`
(hardest-first on vs off, bitwise `final_lb`).

## CVaR weights are the probability floor plus the pure-CVaR allocation

`ρ^{λ,α}[Z] = (1−λ)·E[Z] + λ·CVaR_α[Z]` is realized by one weight kernel in
`convergence/risk_measure.rs`, the only body behind both
`compute_cvar_weights_into` and `compute_cvar_weights_from_costs_into`:
`μ_ω = (1−λ)·p_ω + λ·ν_ω`, where `ν` is the pure `CVaR_α` greedy allocation (cap
`p_ω/α`, total mass 1, openings visited by cost descending then canonical index
ascending via `sort_unstable_by` on the pre-allocated scratch). Every opening keeps at
least `(1−λ)·p_ω` and at most `(1−λ)·p_ω + λ·p_ω/α`. The same kernel feeds
`RiskMeasure::aggregate_cut_into` (every backward scheduler), `evaluate_risk_into`
(the nested upper bound) and the root lower bound, so LB and UB sit under one measure.

A single greedy whose per-opening cap is `(1−λ)·p_ω + λ·p_ω/α`, with no floor, is
the wrong-but-compiling alternative: it is a consistent pure `CVaR` at
`α' = α/(λ + α(1−λ))`, so LB and UB still bracket each other and no bracketing test
catches it. It agrees with the correct kernel at `λ = 1`, at `λ = 0` and on every
two-opening fan; only three or more openings with `0 < λ < 1`, where the floor binds,
discriminate. Switching one consumer without the others opens a spurious LB/UB gap.

Read: `convergence/risk_measure.rs` (the kernel and its two entry points),
`training/lower_bound.rs` (`lb_aggregate_and_broadcast`),
`training/forward/stats_aggregation.rs` (`nested_ub_recursion`). Pinned by
`cvar_weights_match_analytic_table` (equiprobable `10/20/30/40` at `α = λ = 0.5`
gives `30.0`; the cap-only form gives `31.25`),
`cvar_weights_match_rockafellar_uryasev_oracle`,
`cvar_cost_ties_break_by_canonical_index`,
`cvar_weights_reduce_bitwise_at_lambda_endpoints`,
`aggregate_cut_into_applies_the_probability_floor`,
`nested_ub_recursion_applies_the_probability_floor` and
`lower_bound_aggregation_applies_the_probability_floor`.

## Joint risk is applied once over the flattened successor×opening vector

A branching node's backward cut applies the stage `RiskMeasure` **once** over the
single flattened joint outcome vector spanning every successor and every one of
their openings, weighted by the product `P(n→child)·q_{child,ω}` and ordered
canonically — ascending child node id, then within-child opening.
`RiskMeasure::aggregate_cut_into` runs exactly once per trial point over that joint
arena; `assemble_outcome_weights` fills the product weights in the canonical order
the aggregation depends on (`CVaR`'s tail weighting is index-order-sensitive).

Applying the measure per child and then probability-averaging the children — a
NESTED measure — is the wrong-but-compiling alternative. It is indistinguishable
from the joint form in both degenerate regimes (a single successor, or a single
opening per successor), and with one opening per node — the pure-branching case the
measure exists for — the within-node measure is vacuous, so the nested form
collapses to plain expectation with NO tail weighting at all. On a genuine fan the
two differ: joint `CVaR₀.₅` over outcomes `[10, 20, 30, 40]` at weight `0.25`
concentrates on the worst two → `35`, while `max`-per-child-then-average gives
`(20 + 40)/2 = 30`.

Read: `convergence/risk_measure.rs` (`RiskMeasure::aggregate_cut_into`),
`setup/node_graph.rs` (`assemble_outcome_weights` — the canonical product-weight
fill), `training/backward/by_scenario.rs` (`process_by_scenario_backward`),
`training/backward/by_node.rs` (`by_node_finish`), `training/backward/replicated.rs`
(the replicated path applies the same single aggregation over the same flattened
arena). Pinned by `joint_cvar_differs_from_nested_per_child_then_average` in
`convergence/risk_measure.rs` (the analytical `35`-vs-`30` mutation control).

## The branching backward integrates every successor exhaustively

The backward at a node solves **every** successor and **every** one of their
openings, regardless of which successor the forward pass drew. Sampling selects
WHICH TRIAL STATES receive a cut; it never truncates a cut's INTEGRATION AXIS.
`assemble_outcome_weights` iterates the node's whole successor list independent of
the forward draw, and each child loads its OWN LP — frozen template, delta cut
batch, pool, basis key, External column — and solves its own openings. A chain is
the one-element case: one successor, one LP load per trial point, exactly as the
chain backward solves all of a trial point's openings rather than only the sampled
one.

"Solve only the sampled child" (the child-0 collapse) is the forbidden
optimization: pricing every leaf against a single child's LP overstates future
cost, so `final_lb` overshoots the true first-stage value — `final_lb > final_ub`,
an invalid lower bound that still compiles and still converges.

Read: `setup/node_graph.rs` (`assemble_outcome_weights`, `successor_outcome_count`
— the full-successor flatten), `training/backward/by_scenario.rs`
(`process_by_scenario_backward` — each child loads its own LP),
`training/backward/by_node.rs` (`by_node_finish` — the opening-block scheduler over
the same reified outcome set). Pinned by
`water_binding_external_fan_final_lb_matches_extensive_form` (a distinct-column
external fan whose reservoir binds, where the child-0 collapse would overshoot the
extensive-form optimum) and its by-node companion
`water_binding_external_fan_by_node_matches_extensive_form`, both in
`tests/branching_value_oracle.rs`.

## No EWMA upper bound

`ConvergenceMonitor::upper_bound()` returns the raw per-iteration upper bound —
there is no exponentially-weighted smoothing. Gap closure is immediate for
deterministic cases.
Read: `convergence/convergence.rs`.

## The enumerated CVaR upper bound is NESTED, not end-of-horizon

Under an effective `CVaR` the enumerated forward's upper bound is computed by a
NESTED backward risk recursion over the enumerated scenario tree
(`nested_ub_recursion` behind the `ForwardBound::NestedRisk` arm of `sync_forward`
in `training/forward/stats_aggregation.rs`, which the session's forward-sync
selects for a uniform effective `CVaR`):
`Ṽ(n) = cum_d[stage(n)]·c(n) + ρ_children(Ṽ(child))`,
where `ρ` is the same `RiskMeasure::evaluate_risk` weighting the per-node cut /
lower-bound aggregation applies, over each node's children weighted by their
conditional probabilities. This mirrors the nested measure SDDP optimizes
(`ρ = ρ₁(c₁ + ρ₂(c₂ + … ρ_T(c_T)))`), the time-consistent CVaR (per-stage α
compounding to ≈ α^T; CEPEL NT-66 §3.2.1) that DECOMP uses.

Applying `evaluate_risk` ONCE to whole-path root-to-leaf totals — the
end-of-horizon form `(1−λ)E[Z] + λ·CVaR_α[Z]` over path totals — is the
wrong-but-compiling alternative. For a nested measure `ρ_nested ≥
ρ_end-of-horizon`, so the end-of-horizon bound is NOT a valid upper bound on the
nested objective: it can (and on `decomp-mar-26-rv2-reduced` does) fall BELOW the
nested lower bound, giving a persistent negative gap that halts a `gap` rule at a
spurious LB/UB crossover before the policy has converged. Same `evaluate_risk`
weighting, wrong recursion. The nested `Ṽ(root)` is `≥ V* ≥ LB` at every
iteration, so the gap stays non-negative and closes only at true convergence.

`Expectation` (and `CVaR { lambda: 0 }`, which
[`effective`](RiskMeasure::effective) collapses to it) leaves the bound at the
risk-neutral compensated `Σ wᵢ·cᵢ`: the session selects `ForwardBound::Exact` in
`sync_forward` there, since nesting is linear under expectation.
`ForwardBound::NestedRisk` is selected only for a uniform effective `CVaR`
(`uniform_effective_measure`); a stage-varying measure (reachable only without a
`gap` rule) falls back to `ForwardBound::Exact`.

**The `gap` stopping rule admits an effective `CVaR` only under enumerated
forwards with a uniform measure.** The exact nested bound exists only when the
forward is enumerated (a sampled forward's UB is a statistical estimate) and the
measure is uniform across stages (one measure aggregates the tree).
`reject_gap_under_effective_risk_aversion` (`setup/mod.rs`) enforces both: sampled
forwards reject any effective risk aversion (and `reject_gap_under_sampled_selection`
rejects the expectation case too); enumerated forwards defer to
`reject_gap_under_nonuniform_risk`, which admits a uniform measure and rejects a
stage-varying one.

Read: `training/forward/stats_aggregation.rs` (`nested_ub_recursion`, the
`ForwardBound::{Exact, NestedRisk}` arms of `sync_forward`), `training/session/mod.rs`
(the forward-sync bound selection), `convergence/risk_measure.rs` (`evaluate_risk`,
`effective`, `uniform_effective_measure`), `setup/mod.rs`
(`reject_gap_under_effective_risk_aversion`, `reject_gap_under_nonuniform_risk`).
Pinned by `nested_ub_recursion_is_nested_not_end_of_horizon`
(`training/forward/tests.rs` — the nested bound exceeds the end-of-horizon bound
on a tree whose worst branch compounds), the `admission_gate_*` gate tests
(`setup/tests.rs`), and the end-to-end `enumerated_cvar_gap` module
(`tests/deterministic.rs`).

## Enumerated traversal excludes dynamic cut selection

`admission_gate` (`setup/mod.rs`) rejects `Traversal::Enumerated` paired with
`CutSelectionStrategy::Dynamic` as a hard `SddpError::Validation`, through
`reject_dynamic_cut_selection_under_enumerated`. The enumerated backward never
solves lazily: `run_backward_node_replicated` loads the frozen template of each
successor it solves and reads state duals only, and nothing in
`run_enumerated_backward` reaches `sync_stage_metadata`, the only caller of
`CutPool::record_binding`, so no binding activity is recorded. The enumerated
forward does have a lazy branch (`solve_forward_node`): it seeds its resident set
through `build_initial_resident_set` from a window over each cut's
`last_active_iter`, and under enumerated traversal that clock never advances past
the cut's generating iteration. Admitting the pairing is the wrong-but-compiling
alternative — it would run the lazy branch on a seed window whose activity clock
nothing feeds, a path no test exercises; the reject is what makes it unreachable.
No shipped deck pairs them, so the reject is byte-neutral. Any non-`Dynamic`
strategy (or none) under enumerated forwards, and `Dynamic` under sampled
forwards, are admitted — the reject discriminates on the `Dynamic` variant, not on
any strategy being present.

Read: `setup/mod.rs` (`admission_gate`,
`reject_dynamic_cut_selection_under_enumerated`), `training/forward/enumerated.rs`
(`solve_forward_node`), `cut/dcs.rs` (`build_initial_resident_set`),
`training/backward_pass_state.rs` (`run_enumerated_backward`,
`sync_stage_metadata`), `training/backward/replicated.rs`
(`run_backward_node_replicated`). Pinned by
`admission_gate_rejects_dynamic_cut_selection_under_enumerated` (`setup/mod.rs` —
the positive enumerated + `Dynamic` reject, plus the enumerated-only and
dynamic-only negatives that do not reject).

## A lazy solve returns only when no candidate is violated

`lazy_solve_preloaded` returns only when `score_violated_candidates` reports no
violated candidate at the point it returns, so its result is the optimum over the
candidate set up to the tolerance below. A candidate is violated when
`alpha - theta > eps * max(|theta|, |alpha|)`, strict, with `alpha` the cut's value
and `theta` the future-cost value at the point, both in the unscaled units the
function computes, and `eps` the resolved `violation_tolerance`. Candidates are
reported by `alpha - theta` descending, ties to the lower slot. With `z_R` the
objective returned, `z_C` the optimum over the candidate set, `theta*` the
future-cost value at the returned point and `d` the objective coefficient of the
future-cost column, `z_R <= z_C <= z_R + d * eps / (1 - eps) * |theta*|`, for
`eps < 1`: the load enforces it for a present value; an absent value resolves to
a profile's `primal_feasibility_tolerance`, which no check bounds above, and at 1
or more no candidate is ever violated. At `theta* = 0` every candidate with a positive value is
appended, which costs rows and rounds and never exactness.

An absent `violation_tolerance` is resolved by `DcsConfig::resolve`, the only
place that gives it a value, to the `primal_feasibility_tolerance` of the solver
profile of the phase that derives its lazy parameters: the sampled and enumerated
forward, the backward and simulation each use their own. The HiGHS retry ladder
can solve a retried LP looser than its profile, so the test is then stricter than
the solve's accuracy and can append a candidate that noise made look violated;
that costs a row and cannot loop, because an appended candidate is resident.
Simulation seeds every active cut, so its tolerance is never exercised.

A present `violation_tolerance` is range-checked at load by two owners.
`parse_cut_selection_config` rejects a value of 1 or more: the bound divides by
`1 - eps`, and at 1 no candidate with a non-negative future cost is ever
violated. `StudySetup::from_broadcast_params` rejects a value below the
`primal_feasibility_tolerance` of the backward, forward or simulation profile,
the first failing one in that order, whether or not that phase runs; it owns the
floor because the resolved profiles exist only there (`admission_gate` receives
none, and the config type cannot see the solver default). A value equal to a
profile's tolerance is admitted. The wrong-but-compiling alternatives, each shown
by a mutation: comparing only the backward profile admits a value that a forward
or simulation override makes too small; `<=` in place of `<` rejects the equal
value; dropping the upper bound admits a value of 1 or more.

The rounds that add a bounded number of cuts each end at a cap; past it the
per-round limit is lifted and the score, append, solve cycle continues, adding
every violated candidate per cycle, until none is left. Each solve that reaches
the cap with a candidate still violated is counted. The cycle ends because scoring
reports only non-resident candidates beyond the tolerance and every one reported
is appended, so each cycle shrinks the finite non-resident set; a candidate inside
the tolerance is neither reported nor appended.

Appending the cuts violated at the last point and returning after one solve is
the wrong-but-compiling alternative: the new optimum moves, and a cut that was
satisfied before can be violated there. On the cuts `(intercept, slope)`
`(15, -3)`, `(6, -0.5)`, `(20, -4)` seeded with the first, the optimum is 5 at
`x = 5`, where only the second cut is violated; one solve with it gives 7.8 at
`x = 3.6`, where the third is violated by 1.4, against 8 at `x = 4` with all
three. The returned value is a valid but loose relaxation, so nothing fails.

The wrong-but-compiling alternatives for the test itself, each shown by a
mutation: an absolute test `alpha - theta > eps` selects other candidates when the
cost scale changes (`scoring_is_invariant_to_the_cost_scale` and
`a_lazy_solve_stays_within_the_relative_bound` fail); `min` in place of `max` in
the threshold appends a candidate the tolerance admits
(`violation_tolerance_is_strict` and `a_lazy_solve_stays_within_the_relative_bound`
fail); `>=` in place of `>` reports a candidate on the boundary, including one
worth exactly `theta*` when `theta* = 0` (`violation_tolerance_is_strict` and
`lazy_solve_is_exact_after_the_round_cap` fail).

The claim is exact over the candidate set, which is the whole pool only when the
candidate recency window is unset; with a finite window it holds over the cuts
inside the window and the result is deliberately not the all-cuts optimum.
Read: `cut/dcs.rs` (`lazy_solve_preloaded`, `score_violated_candidates`,
`DcsConfig::resolve`, `DcsSolveScratch::round_cap_hits`,
`DcsParams::max_inner_iterations`), `cut/cut_selection.rs`
(`parse_cut_selection_config`), `setup/mod.rs`
(`reject_violation_tolerance_below_a_phase_solver_tolerance`). Pinned by
`lazy_solve_is_exact_after_the_round_cap`
(`cut/dcs.rs`: 8 at `x = 4`, not 7.8, and one cap hit), with
`lazy_objective_equals_the_all_cuts_objective` (random cuts, caps 0 to 3, the
two-sided bound), `a_lazy_solve_stays_within_the_relative_bound` (an unappended
candidate, 7.8 against 8 within the bound), `lazy_solve_terminates_after_the_round_cap`,
`violation_tolerance_is_strict` (the relative boundary),
`scoring_is_invariant_to_the_cost_scale`,
`an_absent_violation_tolerance_resolves_to_the_phase_solver_tolerance`,
`each_phase_resolves_a_dynamic_violation_tolerance_from_its_own_profile`
(`tests/cut_basis.rs`, `highs` only: a profile tolerance of 1 leaves a phase's
lazy solves on their seed and a tight one re-solves, so a forward or backward
site that read the other phase's profile swaps the two LP-solve counts; the
enumerated forward site is unreachable under `dynamic`, and the simulation site
is not exercised),
`parse_dynamic_rejects_a_violation_tolerance_of_one` (`cut/cut_selection.rs`),
`a_violation_tolerance_below_a_phase_solver_tolerance_is_rejected` (`setup/mod.rs`:
the per-phase order, the equal value, the absent value and a `level1` strategy
admitted) and `validate_rejects_a_violation_tolerance_below_the_solver_tolerance`
(`crates/cobre-cli/tests/cli_validate.rs`: two controls that validate, then the
reject).

## The active-cut budget caps only cuts this run generated in interior pools

`run_cut_management` enforces `max_active_per_stage` every iteration when it is
set, after selection and before the template refreeze, on the pools of
`interior_cut_nodes` only. Every non-leaf node owns its own pool and all leaves
share one, so those pools are distinct from each other, from the root's pool and
from the shared leaf pool. In such a pool `CutPool::enforce_budget` counts and
evicts the active non-prefix cuts only: a loaded cut (`CutPool::is_prefix_slot`)
neither counts toward the cap nor is evicted, and a cut of the current iteration
counts and is never evicted. A pool's active count can therefore exceed the
budget by its loaded cuts. Under `resume` the prefix is the recorded run's own
(`warm_start_count`), so the cap counts every cut that run generated, as in the
uninterrupted run.

The root pool is exempt because no solve records binding for it (`record_binding`
runs for successor pools only), so its generated cuts would rank by generation
age alone, and because the lower-bound LP only appends cuts: a root cut evicted
after an earlier lower-bound solve stays in that LP while the refrozen stage-0
template and the bitmap no longer hold it. The terminal pool is exempt because
its template is baked once and the per-iteration refreeze skips it, so an
eviction would leave the bitmap disagreeing with the LP the terminal node solves,
and a simulation that builds its own templates would solve another LP.

The evicted set is the lexicographic minimum of the candidates under
`(last_active_iter, active_count, slot)`: the smallest keys go first, up to the
excess over the cap. `add_cut` writes the generating iteration into
`last_active_iter`, so a cut that never bound ranks by generation age, and the
cuts of one iteration that share a key are told apart by slot. The key is total,
so the evicted set is a function of the pool, the budget and the current
iteration.

The budget is rejected at load, as `SddpError::Validation` naming
`training.cut_selection.max_active_per_stage`, when it is set together with the
`dynamic` method (`reject_budget_under_dynamic_cut_selection`) or with enumerated
training (`reject_budget_under_enumerated`). A lazy solve takes its candidates
from the active cuts (`score_violated_candidates` iterates
`CutPool::active_cuts`) and `dynamic` runs no selection pass
(`CutSelectionStrategy::should_run` is false for it), so no step of the training
loop reactivates a cut and one the budget deactivates is never a candidate again.
The enumerated backward records no binding activity (`sync_stage_metadata`, the
only caller of `record_binding`, runs in the sampled level driver only), so every
cut this run generates keeps `active_count = 0` and its generating iteration as
`last_active_iter`, and the eviction key reduces to generation order.

The wrong-but-compiling alternatives:

- enforcing on every pool: the loop then evicts root cuts, which no solve records
  binding for and which rank by generation age alone;
- letting a loaded cut be a candidate in a loop that also visits the terminal
  pool: it deactivates boundary cuts the baked terminal LP keeps, so a basis has
  fewer rows than that LP and the next iteration fails with
  `BasisRowCountMismatch`;
- counting prefix cuts toward the cap without evicting them: once the prefix
  alone reaches the cap, every generated cut older than the current iteration is
  evicted at each iteration;
- the two-part key `(last_active_iter, active_count)`: equal keys are then ordered
  by an unstable sort, which the language leaves unspecified, so the evicted set
  can differ between toolchains while every build agrees with itself and no
  same-build gate sees it;
- admitting the budget with `dynamic`: the lazy solve returns the optimum over the
  active cuts, and every cut the budget deactivates leaves the candidate set for
  good. Every remaining cut is valid, so that optimum is a valid but looser
  relaxation of the optimum over the whole pool, and nothing fails;
- admitting the budget under enumerated training: the cap evicts the oldest
  generated cuts first whatever their use, and nothing fails.

Read: `training/session/mod.rs` (`run_cut_management`, `interior_cut_nodes`,
`freeze_active_cuts_into_templates`), `cut/pool.rs` (`CutPool::enforce_budget`,
`is_prefix_slot`), `setup/node_graph.rs` (the `node → pool` assignment),
`training/backward_pass_state.rs` (`sync_stage_metadata`, the only caller of
`record_binding`), `training/lower_bound.rs` (`lb_init_rank0`), `cut/row.rs`
(`append_new_cuts_to_lp`), `simulation/state.rs` (`refreeze_templates_if_needed`),
`setup/mod.rs` (`admission_gate`), `cut/dcs.rs` (`score_violated_candidates`),
`cut/cut_selection.rs` (`CutSelectionStrategy::should_run`). Pinned by
`enforce_budget_never_evicts_a_loaded_cut` (a loaded cut stays, including one
whose generating iteration equals the current one) and
`enforce_budget_does_not_count_loaded_cuts` (loaded cuts do not count toward the
cap), both in `cut/pool.rs`;
`enforce_budget_breaks_ties_by_lowest_slot` (ties go to the lowest slot, and the
key parts rank in order, on hand-derived keys),
`evicted_set_is_the_lexicographic_minimum` (random pools with frequent ties
against a reference sorted from tuples; it is the pin against the two-part key,
since a hand fixture of a few cuts sorts stably by accident) and
`enforce_budget_tiebreaks_by_active_count` (the `active_count` part), all in
`cut/pool.rs`;
`budget_caps_only_interior_pools_and_templates_match_their_pools`
(`training/session/mod.rs`: the root and terminal pools keep their cuts and
every frozen template matches its pool); and
`budget_below_the_boundary_cut_count_leaves_the_boundary_run_unchanged`
(`tests/cut_basis.rs`: a budget below the boundary cut count trains without a
basis rejection and leaves the run unchanged);
`admission_gate_rejects_a_budget_with_dynamic_cut_selection` and
`admission_gate_rejects_a_budget_under_enumerated` (`setup/mod.rs`: each pairing
is rejected naming the key and the setting, and the neighbouring configs are
admitted).

## Terminal boundary FCF is booked in the reported total cost

The forward trajectory cost and the simulation per-scenario cost both reconstruct
a path total as `Σ_t cum_d(t)·stage_cost(t)`, where the interior `stage_cost(t) =
(view.objective − d_t·θ_t)·cost_scale` subtracts the discounted epigraph `θ_t` —
the future cost-to-go a later stage realizes as its own immediate cost. At the
TERMINAL stage under a boundary policy (`CutPool::has_prefix_cuts` on the
terminal stage's pool, i.e. `prefix_len() > 0`) `θ_t` prices the
POST-HORIZON value-to-go, which no later stage realizes, so it is KEPT in the
reported cost (`stage_cost = view.objective·cost_scale`) — matching the lower
bound, which already carries it through `θ_0`'s cuts (`evaluate_lower_bound`
pushes the full stage-0 `view.objective`). The present values coincide exactly
because `θ_t`'s objective coefficient IS `d_t`, so
`cum_d(t)·d_t·θ_t = cum_d(t+1)·θ_t` is the same term the LB books.

Subtracting `θ_t` at the terminal boundary stage — the interior form — is the
wrong-but-compiling alternative: it drops the post-horizon FCF from the UB /
simulation cost alone, leaving `LB ≫ UB` (a NEGATIVE gap the stopping rule's
`.max(0.0)` clamp then reads as "converged"). The branch is gated on
`terminal && pool.has_prefix_cuts()`; a non-boundary study pins terminal `θ`
to `[0, 0]` (forward) or leaves the terminal pool empty with `θ`'s `0.0` lower
bound driving it to `0` (simulation), so the fix is byte-neutral there.

Read: `cut/pool.rs` (`CutPool::has_prefix_cuts`, the flag's only formula),
`training/forward/enumerated.rs` (`solve_forward_node`),
`training/forward/stage_solve.rs` (`run_forward_stage`), `simulation/pipeline.rs`
(`extract_sim_stage_result`, flag read from the stage's pool in
`solve_simulation_stage`). Do NOT change `evaluate_lower_bound`
(`training/lower_bound.rs`) — it is correct — and do NOT fold `θ` into the
per-stage `compute_cost_result` breakdown (`simulation/extraction.rs`), which
reports `immediate_cost`/`future_cost` separately by design. Pinned by
`terminal_boundary_fcf_training_gap_is_consistent`,
`terminal_boundary_fcf_simulation_cost_includes_post_horizon`,
`terminal_boundary_records_all_inactive_leave_the_plain_chain_bounds`, and
`terminal_boundary_records_all_inactive_leave_the_plain_fan_bounds`
(`tests/branching_value_oracle.rs`), and
`terminal_boundary_flag_formulas_agree_on_chain_and_terminal_fan`
(`setup/tests.rs`).

## Fused terminal slice projects with the parent pool, not the leaf pool

Terminal-leaf fusion reuses an External terminal leaf's forward-solved
`(objective, duals)` as the penultimate-stage Benders cut instead of re-solving
the leaf in the backward. The forward MUST project that slice with the leaf's
CUT-GENERATING PARENT pool (`cut_state_layouts[parent.pool_id]`, where `parent =
build_parent_map()[leaf]`) — the identical projection the backward's
`SuccessorSpec::cut_state` uses for the currently-solving parent node — NOT the
leaf's own (terminal, no-successor) pool. The two pools' `n_slots()` differ
whenever the terminal pool and the parent's successor-sized pool project
different state dimensions (`build_cut_state_layouts` seeds every pool at full
state and only shrinks non-leaf pools to their successor's `state_config`).
Projecting with the leaf's own pool is the wrong-but-compiling alternative: it
emits a wrong-length/wrong-projection dual slice the consumer still accepts,
corrupting the fused cut (surfaces as an `allgather_outcomes` length-invariant
violation, or a silently mis-projected cut driving a NEGATIVE gap).

Fusion reuses the forward-captured slice ONLY for a leaf
`is_external_terminal_leaf` admits (External, single-opening, terminal) — the
one case where the forward and the backward read the byte-identical LP — and is
DISABLED under DCS (`params.dcs.is_none()`): DCS solves a lazily cut-reduced
forward LP that need not match the full frozen template the backward loads, so a
fused DCS slice could under-price the cut. A Generated terminal leaf (forward
samples one opening, backward integrates all of them) is NOT eligible and keeps
the exhaustive backward integration the branching contract requires (see "The
branching backward integrates every successor exhaustively" above); reusing its
single forward opening would understate the cut and drive `final_lb` above
`final_ub`. A parentless leaf (a malformed graph) captures nothing and falls
back to a direct backward solve.

Read: `training/forward/enumerated.rs` (`solve_forward_node`, `fusion_cut_state`),
`training/backward_pass_state.rs` (`SuccessorSpec::cut_state`),
`setup/node_graph.rs` (`NodeGraph::is_external_terminal_leaf`, the single
eligibility source the forward capture and the backward consume both read, so the
two cannot drift). The projection is pinned by
`enumerated_forward_fused_slice_projects_with_parent_pool_not_leaf_pool` and
`external_distinct_fan_heterogeneous_cut_state_matches_extensive_form`; the
eligibility gate by
`is_external_terminal_leaf_true_for_terminal_external_single_opening`,
`is_external_terminal_leaf_false_for_generated_terminal_leaf`, and
`is_external_terminal_leaf_false_for_interior_external_node`
(`setup/node_graph.rs`); and the External-fuses / Generated-integrates-exhaustively
split end-to-end by the `terminal_fusion` oracle's
`water_binding_external_fan_fused_cut_matches_independently_derived_cut` (the fused
cut is a valid supporting hyperplane),
`external_distinct_fan_backward_performs_zero_terminal_leaf_solves` (fusion removes
every External terminal-leaf backward solve), and
`terminal_generated_fan_integrates_exhaustively_and_backward_solves_every_leaf` (a
Generated fan is never fused and solves every leaf every iteration), all in
`tests/branching_value_oracle.rs`.

## Spillage is frozen `[0, 0]` during PreFilling

A `PreFilling` hydro's spillage column is pinned `[0, 0]` — no dam exists yet to
spill from, and its incremental inflow has already left via the short-circuit, so a
free spillage column injects phantom water onto the first active downstream hydro's
water-balance row (a conservation violation). The freeze is gated on
`Phase::PreFilling` ALONE. Two wrong-but-compiling alternatives: extending the
freeze to `Filling` removes the legitimate over-dam relief valve an impounding
reservoir needs (D40); gating on `filling.is_none()` leaves the phantom-spill hole
open for a filling hydro in its own `PreFilling` sub-phase (D38, D39). Turbine and
diversion differ — they are frozen in BOTH `PreFilling` and `Filling` (no installed
machinery), whereas spillage is legitimately free in `Filling`.
Read: `lp/builder/columns.rs` (`fill_spillage_columns`). Cases: D38, D39, D42
(phantom PreFilling spill removed); D40 (legitimate Filling-phase spill retained).

## Checkpoint cut values are stored in the run's scaled units

A checkpoint stores every cut's intercept and coefficients as the pool holds them
in memory, in the writing study's scaled units, and each pool's
`StageCuts.cost_scale_factor` names that scale (`build_stage_cut_records`). A load
converts each pool by its own factor (`rescale_cut_records_for_load`): when the
pool's factor is bit-equal (`to_bits`) to the loading study's, the records are
used unchanged, so a same-factor round trip, a resume included, is bit for bit;
otherwise every value becomes `(value * source) / loading`, multiplying first.
`value * (source / loading)` rounds the ratio first and moves the bits of a
pool written at `1.0`.

The wrong-but-compiling alternatives:

- multiplying by the factor on export and dividing on load: the two roundings do
  not cancel at a factor that is not a power of two, the default included, so a
  resumed run drifts from the run it continues;
- ignoring the source factor on load (dividing by the loading factor alone): a
  checkpoint written at another factor loads at the wrong scale.

Read: `policy/policy_export.rs` (`build_stage_cut_records`),
`policy/policy_load.rs` (`rescale_cut_records_for_load`, `pool_cost_scale_factor`,
`convert_checkpoint_cuts_for_load`), `policy/full_fcf_load.rs`
(`check_full_fcf_load`). Pinned by `export_then_load_same_factor_round_trips_bit_for_bit`,
`a_marked_checkpoint_converts_by_its_own_factor`,
`a_cross_factor_load_multiplies_by_the_source_before_dividing` and
`a_full_load_converts_each_pool_by_its_own_factor` (`policy/policy_load.rs`),
`boundary_load_reads_cost_scale_from_bin`
(`tests/boundary_self_describing_clean_break.rs`), and the one-pass resume leg
`resumed_run_equals_the_uninterrupted_run_with_one_forward_pass`
(`tests/periodic_checkpoint.rs`), which runs at the default factor.

## The per-scenario basis store is gathered at every checkpoint write

A checkpoint's `basis/` files hold one `BasisEntry` per (global scenario, node)
whose basis the run's `BasisStore` holds. A rank's store is keyed by its local
row `m`, and the global scenario of that row is `fwd_offset + m`
(`RankDistribution`); rows `my_actual_fwd..max_local_fwd` are padding and never
sent. `gather_basis_store` (`training/training.rs`) assembles the store on rank
0: per canonical node, each rank packs its rows `0..my_actual_fwd` with
`CapturedBasis::to_broadcast_payload`, the sole owner of the bytes (`0_i32` for a
row with no basis), and the node issues one `allgatherv` of the two payload
lengths of every rank, then one `allgatherv` each of the `i32` and the `f64`
payloads at the displacements those lengths give. Rank 0 decodes each rank's
segment with `CapturedBasis::try_from_broadcast_payload`, numbers its rows from
the rank's first scenario (the ranks' scenarios are contiguous blocks in rank
order, so no scenario index travels), drops `state_at_capture`, and keeps the
rows that hold a basis, in a `CheckpointBasisStore` marked `complete`. The
writer (`build_basis_store_entries`) writes one entry per kept row, and the
manifest's `basis_store_complete` is the store's `complete`.
`TrainingResult::basis_cache` and `broadcast_basis_cache` keep their own job:
scenario 0 per node, for simulation.

Every rank enters the gather under a condition all ranks share: the schedule in
`write_periodic_checkpoint` (`PeriodicCheckpoint::fires_at`, built on every rank
from the broadcast config), and the clean `finalize` once `reconcile_error_flag`
has agreed, before `broadcast_basis_cache`. A length that does not fit an `i32`
count travels as `-1`, so every rank fails at the same node; a segment rank 0
cannot decode ends the gather only after the last node, so no rank leaves the
collectives early. On the periodic path either failure is reconciled like a
failed write; on the clean path `finalize` reconciles the gather result with
`reconcile_result` before `broadcast_basis_cache`, whose length broadcast
(`broadcast_len`) carries an oversized length as `-1` so every rank fails after
it. The exits that skip the collectives, `finalize_with_error` and
`finalize_without_broadcast`, gather nothing: `TrainingResult::checkpoint_bases`
holds rank 0's own rows keyed by global scenario
(`CheckpointBasisStore::rank_local`), marked not `complete`, which is also what a
one-rank failure exit writes although it holds every scenario.

The wrong-but-compiling alternatives:

- keying an entry by the local row `m`: on one rank it is the global scenario,
  so nothing fails there; on several ranks every rank's row 0 is written as
  scenario 0, so the checkpoint depends on the writing run's rank count;
- writing rank 0's rows only, skipping the gather: on one rank it is the whole
  store; on several ranks the other ranks' scenarios are missing from a
  checkpoint that says it is whole;
- gathering on an error exit: a peer that failed never enters the collective, so
  the ranks that reach it block;
- returning rank 0's gather error from `finalize` before the broadcast: the other
  ranks enter `broadcast_basis_cache` and block. Pinned by
  `a_gather_failure_on_rank_zero_does_not_leave_a_peer_in_the_broadcast`
  (`training/session/mod.rs`: two ranks' collective sequences, rank 0 meeting a
  decode failure) and, for the length, by
  `an_oversized_payload_fails_every_rank_after_the_same_broadcast`
  (`training/training/tests.rs`).

Read: `training/training.rs` (`gather_basis_store`, `CheckpointBasisStore`,
`CheckpointBasisStore::rank_local`), `training/session/mod.rs`
(`write_periodic_checkpoint`, `finalize`, `finalize_without_broadcast`),
`workspace/workspace.rs` (`CapturedBasis::to_broadcast_payload`,
`CapturedBasis::try_from_broadcast_payload`), `policy/orchestration.rs`
(`CheckpointLayout::write`), `policy/policy_export.rs`
(`build_basis_store_entries`). Pinned by, in `training/training/tests.rs`:
`a_single_rank_gather_keeps_every_stored_row_with_its_scenario` (every stored row
with its scenario, a node with no basis has no entry),
`a_rank_zero_of_two_gather_reads_an_empty_peer_segment_as_no_entries` (a
communicator that writes rank 0's segment only) and
`a_two_rank_gather_numbers_each_peer_row_from_its_offset` (two cooperating ranks
in one process, the peer's padding row stored and not sent: fails under the
local-row key and under rank 0's rows only),
`a_gather_that_cannot_decode_a_peer_segment_still_issues_every_collective` (rank
0 returns the decode error after consuming every collective the peer issued) and
`a_payload_length_that_does_not_fit_an_i32_count_fails_naming_its_rank` (a `-1`
length is refused naming its rank and node), with the `complete` flag of a clean
end and of a failure exit in `ac_train_result_fields_populated` and
`ac_train_partial_result_on_mid_iteration_failure`; and by
`periodic_checkpoint_with_several_forward_passes_holds_every_scenario_basis_and_equals_the_final_one`
(`tests/periodic_checkpoint.rs`: three forward passes, the periodic and the final
checkpoint at the same iteration hold scenarios 0, 1 and 2 at every node with a
basis, both whole, and are equal). The clean end's gather runs under an MPI
launcher in
`a_failed_periodic_write_on_the_world_leaves_the_last_whole_checkpoint_and_a_partial_beside_it`
(`tests/periodic_checkpoint.rs`, launched on 2 and 4 ranks by
`scripts/ci/run-mpi-launcher-gates.sh`): rank 0's whole checkpoint holds every
forward pass of every rank at each node record, which fails under the local-row
key. Under `cargo test` it is a single-rank pass and pins nothing here. No test
pins the error-exit alternative, which no single-process run can show.

## A failed run never replaces a whole checkpoint

A run that ends in an error (`TrainingOutcome::error`) writes its policy to the
sibling `partial_policy_dir(policy_dir)`, `<policy dir name>` followed by
`.partial`, and never to the policy directory. The policy directory only ever
holds whole checkpoints: the periodic writes, the clean end and a graceful stop
write it, and each of them holds every rank's bases
(`CheckpointBasisStore::complete`). A failure exit holds rank 0's own rows,
marked not complete, and the failed iteration may be half applied: the FCF can
hold the cuts of an iteration that `completed_iterations` does not count.
`write_checkpoint` is the single owner of the rule and of the name, for the CLI
and the Python bindings alike. It decides the destination from
`TrainingResult::checkpoint_bases.complete` alone, never from the directory's
content, and returns the directory it wrote; both front ends name that directory
in the failure message. A second failed run replaces the sibling, as the writer
replaces any target. No load reads the sibling by itself: it is read only when
`policy.path` names it.

The wrong-but-compiling alternatives:

- writing the failure exit's policy into the policy directory: it swaps a partial
  policy over the last periodic checkpoint, so a multi-rank run that fails
  in-band after a periodic write loses the one checkpoint it could resume from;
- skipping the write when the policy directory already holds a checkpoint: the
  destination then depends on the directory's content, a failed run with no
  earlier checkpoint writes to the policy directory again, and the partial
  policy of the other case is lost;
- a collective on the error path, such as gathering the other ranks' bases to
  make the failure checkpoint whole: a rank that failed never enters it, and the
  ranks that reach it block.

Read: `policy/orchestration.rs` (`write_checkpoint`, `partial_policy_dir`),
`training/session/mod.rs` (`finalize_without_broadcast`),
`crates/cobre-cli/src/commands/run/outputs.rs` (`write_training_outputs`),
`crates/cobre-python/src/run.rs` (`write_training_outputs`). Pinned by, in
`policy/orchestration.rs`,
`a_result_without_every_ranks_bases_is_written_beside_the_policy_directory` (the
whole checkpoint is byte for byte unchanged),
`a_failed_runs_partial_policy_is_written_without_a_whole_checkpoint_present` and
`a_result_with_every_ranks_bases_is_written_to_the_policy_directory`; by
`a_failed_periodic_write_leaves_the_last_whole_checkpoint_and_a_partial_beside_it`
(`tests/periodic_checkpoint.rs`: an in-band periodic write failure of a run that
resumed a whole checkpoint) and, on 2 and 4 ranks under the MPI launcher gates,
`a_failed_periodic_write_on_the_world_leaves_the_last_whole_checkpoint_and_a_partial_beside_it`
(rank 0's write fails and every peer fails at the same iteration through the
error reconciliation); and by
`a_failed_run_keeps_the_last_whole_checkpoint`
(`crates/cobre-cli/tests/cli_run.rs`) and
`test_a_failed_run_keeps_the_last_whole_checkpoint`
(`crates/cobre-python/tests/test_periodic_checkpoint.py`): a resumed run that
fails in-band leaves the policy directory byte for byte and writes the sibling,
and the next resume of the policy directory continues from the whole checkpoint.

## A resume refuses a checkpoint that cannot continue the run identically

A resume continues the run that wrote the checkpoint, so it reads state the file
must carry. `check_full_fcf_load` refuses a `FullFcfLoadKind::Resume`, right after
`validate_policy_load`, as `FullFcfLoadError::Refused(SddpError::Validation(..))`
(class `IncompatiblePolicy`, phase `ResumeIncompatible`), on the first of these, in
this order (`check_resume_serves`), reading the manifest's `ProducerBlock`:

1. `forward_passes` differs from the run's `loop_params.forward_passes`: the slot
   layout and the stored scenarios follow the count;
2. the run's strategy is `Level1`, the study has an interior pool
   (`interior_cut_nodes`), and `state_history` is `None`;
3. the same run keeps every state (`lookback_iterations: None`) and the file holds
   a `Window`, or the run's window is longer than the file's;
4. `CutManagementConfig::tracks_binding_activity` holds and `activity_counters`
   is unset;
5. `basis_store_complete` is unset.

Each message names the setting, what the checkpoint holds and the way out. A
stored window longer than the run's is admitted: the extra states leave at the
first selection pass, as they would have. A study with no interior pool stores no
trial state under `level1`, so its `None` is not refused. `warm_start`,
simulation-only and boundary loads read none of these fields. The run's own
`check_frequency`, budget value and cost scale apply after a resume, and none of
them is state the file must carry.

The wrong-but-compiling alternatives:

- deciding the history reject from missing state rows: a windowed source drops the
  states of its older cuts, so a scored slot without a row is admitted by design.
  `state_history` describes the interior pools only; the root's states, written
  when `exports.states` is on, are unwindowed;
- refusing a missing counter per generated record only: a checkpoint written
  without a reader carries no counter on any record, and its warm-started prefix
  cuts would restart at the source run's iterations;
- refusing a store that is not whole for a `warm_start`: it seeds each scenario's
  basis or none, never needs all of them, and only warm-starts a solve;
- claiming that bases are missing: a one-rank failure exit holds every scenario
  and is still marked not whole, so the message says the run failed.

Read: `policy/full_fcf_load.rs` (`check_resume_serves`, `check_full_fcf_load`),
`cobre_io::ProducerBlock`, `config.rs`
(`CutManagementConfig::tracks_binding_activity`), `training/session/mod.rs`
(`interior_cut_nodes`). Pinned by, in `tests/full_fcf_load.rs`, each with a warm
start of the same checkpoint that loads:
`a_resume_refuses_another_forward_pass_count`,
`a_level1_resume_refuses_a_checkpoint_without_the_trial_state_history`,
`a_level1_resume_refuses_a_shorter_stored_window` (a run keeping every state and
one with a longer window are refused, the same and a shorter window resume),
`a_resume_with_a_reader_refuses_a_checkpoint_without_the_binding_counters` (a
budget and `dynamic`) and `a_resume_refuses_a_checkpoint_whose_bases_are_not_whole`;
`a_whole_checkpoint_resumes_under_the_settings_that_wrote_it` (a `level1` run with
a window and a budget resumes its own checkpoint),
`a_level1_resume_of_a_study_without_an_interior_pool_needs_no_history`; and, in
`crates/cobre-cli/tests/cli_validate.rs`,
`validate_rejects_a_resume_without_the_trial_state_history` and
`validate_rejects_a_resume_from_a_partial_policy` (each after a control that
validates), with the rows `resume_without_the_trial_state_history` and
`resume_from_a_partial_policy` of `crates/cobre-cli/tests/refusal_parity.rs` and
`crates/cobre-python/tests/test_refusal_parity.py` (the same line from
`cobre validate`, `cobre run` and the Python bindings).

## A resume continues the uninterrupted run

A run resumed from the checkpoint of iteration `k` and trained to iteration `n`
holds what the run uninterrupted to `n` holds, both stopped by an iteration limit:
a stopping rule that reads the lower bound can stop the two runs an iteration
apart.

Bit for bit:

- in every pool, the cut values, the activity flags, the generating iteration and
  forward pass of every slot, the binding counters, each slot's role, the stored
  trial states and the ownership record (the owner, the best value and the owned
  count of each state);
- the final upper bound, the forward and backward simplex counts of the iterations
  after `k`, and the pool row counts those iterations report
  (`TrainingEvent::PolicyPoolSummary`, bar its wall-clock field);
- the bases the resume seeds, per node and scenario, against the ones the
  checkpoint holds;
- the final checkpoint: manifest, cuts, bases, trial states and prefix length, bar
  `created_at` and its lower-bound fields.

Within a relative `1e-6`: the lower-bound series, and the checkpoint's copy of it.
The uninterrupted run rebuilds the lower-bound LP warm and the resumed run builds
it cold, so two solves of one LP start from different solver states. The simplex
counts of the lower-bound solve are not compared.

The resumed run's own settings apply after the resume: its iteration limit,
`check_frequency`, budget value and cost scale. The state the resume rejects name
(the forward-pass count, the trial-state history and window, the binding counters
and a whole basis store) must come with the checkpoint; the legs resume under the
settings that wrote it, which is what makes the comparison with the uninterrupted
run one of equals.

The wrong-but-compiling alternatives:

- comparing the cuts and the bounds only: a counter, a role, a trial state or a
  basis that differs changes no cut value until a later pass reads it;
- comparing the lower bounds bit for bit: the cold rebuild differs in the last
  digits.

The legs catch each of these mutations of the resume:

- every record loaded as `Prefix`: every leg fails, on the slot roles or, once a
  budget evicts, on the cuts;
- no trial state loaded: the legs that run `level1` fail on the cuts the first
  selection pass deactivates, and the `dynamic` and selection-off legs, whose pools
  store none, are unchanged;
- an iteration base of `start_iteration + 1`: a `debug_assert` stops a debug build.
  With debug assertions off the legs still fail: the resumed run aborts with
  `BasisRowCountMismatch` in most, and in the `dynamic` and the warm-start legs,
  whose runs complete, the comparison of the final checkpoints fails by itself.

Read: `training/session/mod.rs` (`TrainingSession::new`), `cut/pool.rs`
(`CutPool::new_resumed`), `policy/full_fcf_load.rs` (`check_full_fcf_load`).
Pinned by, in `tests/periodic_checkpoint.rs`, the legs
`level1_resume_equals_the_uninterrupted_run`,
`level1_with_a_window_resume_equals_the_uninterrupted_run`,
`level1_with_a_budget_resume_equals_the_uninterrupted_run`,
`dynamic_resume_equals_the_uninterrupted_run`,
`a_resume_on_more_threads_equals_the_uninterrupted_run` (checkpointed on one thread,
resumed on two), `a_resumed_warm_started_run_equals_its_uninterrupted_run` and the
selection-off legs `resumed_run_equals_the_uninterrupted_run_with_one_forward_pass`
and `resumed_run_equals_the_uninterrupted_run_with_several_forward_passes`. Each
`level1` leg asserts that a selection pass ran at the last iteration, so the
ownership record is current, and that a slot generated by the checkpoint changed
its flag afterwards; the window legs, that the window had dropped a state before
the checkpoint (the warm-start leg, a loaded one); the budget leg, that an owner
generated by the checkpoint is inactive at `n`, which only the final budget can
have made it; the `dynamic` leg, that a slot generated by the checkpoint bound
again.

## A resume that declares a boundary keeps the terminal binding counters

A run that declared `policy.boundary` trains against the boundary rows in its
terminal pool, and its checkpoint carries those rows with the binding counters
the run recorded for them. A resume continues that run, so the terminal pool it
trains keeps the counters. The front ends load the checkpoint, reconcile the
declared boundary and call `inject_boundary_cuts`, which replaces the terminal pool
with the reconciled rows. When `loop_params.start_iteration` is above 0, which only
a resume sets (`CheckedFullFcfLoad::apply_to_training`), it first compares the
reconciled rows with the loaded terminal pool: the state dimension, the slot count
and, slot by slot in order, the role, the active flag, the intercept bits and the
coefficient bits. A difference is refused as `SddpError::Validation` naming the
pool, the first difference and the way out: resume with the `policy.boundary` and
the cost scale (`modeling.cost_scale_factor`) the interrupted run used, or continue
with `policy.mode = warm_start`. Rows that match take the `active_count` and
`last_active_iter` of the loaded row at the same slot, through
`CutPool::record_binding` on the restarted pool. A warm start and a
simulation-only load do not compare and keep the restart (the activity clock
paragraph above).

The compare is one function, `refuse_changed_boundary`, which
`inject_boundary_cuts` runs on the pool it installs and `check_resume_boundary`
runs on the pool it builds for `validate_study`, so `cobre validate`,
`cobre run` and the Python bindings refuse the same resume with the same line.
`validate_study` applies a resume load to its setup before it reconciles the
boundary, in the order of a run; without that the setup it reconciles on has
`start_iteration = 0` and the compare never fires there.

A resume at another `modeling.cost_scale_factor` is not refused earlier
(`check_resume_serves` reads no cost scale), and the compare fires on it only when
the boundary file's factor differs from the interrupted run's and the interrupted
run's differs from the resume's: the loaded checkpoint rows were converted twice
(the boundary's factor to the run's, then the run's to the resume's) and the
reconciled boundary rows once (the boundary's factor to the resume's), and two
roundings are not one. The message therefore names the factor along with the
boundary.

Under a budget the counters of the terminal rows reach the checkpoint the resumed
run writes, and under `dynamic` the seed window reads them.

The wrong-but-compiling alternatives:

- leaving the restart in place on a resume: the terminal rows restart at the
  resume, so the checkpoint the resumed run writes carries lower counters than the
  uninterrupted run's, and under `dynamic` the boundary rows leave the seed window
  at the first resumed iteration, which moves the training numerics in the last
  bits;
- restoring without comparing: a boundary that changed between the interrupted run
  and the resume gives its rows the counters of other rows;
- comparing within a tolerance: the interrupted run trained against exactly these
  bits, and a row that differs in the last bit is another row;
- skipping the injection on a resume, so the loaded pool stays: the declared
  boundary is then never read, and a boundary that changed is ignored without a
  word;
- comparing on a warm start: its loaded terminal pool is another run's, and a
  warm start with a new boundary is legitimate;
- reconciling on an unloaded setup in `validate_study`: `start_iteration` is 0
  there, the compare does not fire, and `cobre validate` accepts a resume that
  `cobre run` refuses.

Read: `policy/policy_load.rs` (`inject_boundary_cuts`, `check_resume_boundary`,
`refuse_changed_boundary`, `terminal_boundary_pool`), `validate_phases.rs` (`validate_study`,
`check_configured_full_fcf_load`), `cut/pool.rs` (`CutPool::record_binding`,
`CutPool::restart_activity_clock`).
Pinned by `level1_with_a_budget_and_a_boundary_resume_equals_the_uninterrupted_run`
(`tests/periodic_checkpoint.rs`: the resumed run's terminal counters equal the
uninterrupted run's; with the restore removed the budget leg fails on a terminal
`active_count`), with `dynamic_with_a_boundary_resume_equals_the_uninterrupted_run`
(the seed window) and `level1_with_a_boundary_resume_equals_the_uninterrupted_run`
(a run with no reader, which the restore leaves unchanged);
`a_resume_refuses_a_boundary_that_no_longer_matches_its_checkpoint`
(`tests/full_fcf_load.rs`: the boundary the run used is admitted and keeps the
checkpoint's counters, another boundary is refused, and a warm start of the same
checkpoint takes it and restarts them) and, in `policy/policy_load.rs`,
`a_resume_keeps_the_binding_counters_of_the_checkpoints_terminal_rows`,
`a_resume_refuses_a_boundary_that_differs_from_the_checkpoints_terminal_rows` (one
case for each kind of difference),
`a_resume_at_another_cost_scale_factor_is_refused_naming_the_factor` and
`only_a_resume_compares_the_boundary_with_the_loaded_terminal_pool`; and by the row
`resume_over_a_changed_boundary` of `crates/cobre-cli/tests/refusal_parity.rs` and
`crates/cobre-python/tests/test_refusal_parity.py` (the same line from
`cobre validate`, `cobre run` and the Python bindings), with
`a_resume_over_the_boundary_it_was_interrupted_over_validates` (the control that
validates).

## Policy-load compatibility validation is mandatory

Every policy load — full-FCF warm-start/resume/simulation-only and terminal
boundary-cut injection — routes through `validate_policy_load`, the single
entry point; there is no opt-out or bypass path. Its check matrix keys off
`PolicyLoadKind`: `state_dimension` equality is hard-rejected only for `FullFcf`
(`CHECK_STATE_DIMENSION`); a `BoundaryInjection` load skips it in
`validate_policy_load` and defers to the per-slot reconciliation in
`load_boundary_cuts` as the authority — the C17 source-drop surfacing
(`BoundaryReconciliationReport::superset_summary`, `FamilyTally::dropped_source`/
`dropped_source_slots`, serialized by `cobre validate --json`, and rejected
outright under `policy.boundary.strict`) is what makes relaxing it safe, letting a
NEWAVE-shaped source (no transit buckets, monthly anticipated slots) feed a
DECOMP-shaped current study at a differing state dimension. That deferral holds
only while the entity manifest is verifiable; an absent (empty) manifest cannot
reconcile per-slot, so `load_boundary_cuts` falls back to a `state_dimension`
equality guard there, rejecting an unreconcilable differing-dimension load rather
than panicking in the fixed-length cut-pool copy (`CutPool::new_with_warm_start`'s
`copy_from_slice`, which panics on a source-vs-current length mismatch).
`num_stages` equality is hard-rejected only for
`FullFcf` — a `BoundaryInjection` load skips it deliberately, since a monthly
source study may legitimately feed a weekly+monthly current study. Per-slot
`slot_identity` (`entity_type`, `entity_id`, `subindex`) is an EXACT positional
match (`compare_manifest_slot_identity`) only under `FullFcf`; a
`BoundaryInjection` load does NOT exact-match here — its slot identity is
RECONCILED instead, by `reconcile::build_rebind`/`rebind_cut` inside
`load_boundary_cuts`, confined to that one load path. Storage and inflow-lag are
the state's must-correspond core: a target slot of either family with no source
counterpart REJECTS, naming the offending hydro or lag depth — checked up front,
all at once over the whole manifest, by the storage/inflow-lag topology gate
below, with `build_rebind`'s own per-slot rejects retained as that gate's
postcondition (see "Boundary loads gate on the storage/inflow-lag topology
before reconciling"). The entity
(`entity_type`/`entity_id`) is NEVER relaxed for any family — only the matching
MECHANISM changes (identity hashmap vs. exact position), and only a date family's
calendar `subindex` is ever relaxed (that relaxation is the dated fan-out
reconciliation, not this core). `col_scale`/LP prescaling is explicitly NOT a
compatibility dimension: a state variable's identity and physical unit are
independent of how the LP happens to scale its column, so comparing `col_scale`
would falsely reject a policy whose entities genuinely match but whose scaling
strategy or magnitude differs from the current study's — the forbidden
alternative this contract rules out.

A `BoundaryInjection` reads its study-global facts from the resolved pool's own
`cuts/<pool>.bin`, never `metadata.json`. It selects the pool by DATE EQUALITY,
never by index: the study's boundary date (`study_horizon_end`, the last study
stage's exclusive `end_date`) is encoded once via `cobre_io::encode_slot_date`
and compared as an integer against each pool's own `priced_state_date`, walking
`checkpoint.stage_cuts` in its pool-id-sorted order — an index into
`graph_stage_id` or the raw pool id could silently pick a different, wrong
future-cost function whose reconciliation tally is identical to the correct
one. It then reads the selected pool's `cost_scale_factor` through
`pool_cost_scale_factor` to feed `rescale_cut_records_for_load` — the additive `StageCuts`
`cost_scale_factor`/`node_id`/`priced_state_date` fields make one
`cuts/<pool>.bin` self-describing. Four named rejects guard the boundary load:
every pool in the checkpoint carrying the `priced_state_date` sentinel (a
pre-dated checkpoint) REJECTS, advising re-export
(`load_boundary_cuts_undated_pools_reject_with_reexport_hint`); a boundary date
matching zero pools, or more than one pool, REJECTS — naming the boundary date
and every pool's own priced date, or every matching pool id, respectively; a
resolved pool whose `cost_scale_factor` reads `None` (a pre-self-describing
`.bin`) REJECTS (`boundary_predates_self_describing_cuts`, advising
re-export), never silently defaulting to `LEGACY_COST_SCALE_FACTOR`; and a
resolved pool whose `node_id` is the `-1` sentinel (a shared, multi-owner
pool) REJECTS — a boundary source must be a single-node terminal pool. The
`FullFcf` path mirrors the `cost_scale_factor` read and its rejects through the
per-pool check `pool_cost_scale_factor`, applied to every pool by
`convert_checkpoint_cuts_for_load`: each pool converts by its own factor, and a
factor that is absent, not finite or not positive is refused naming the pool. Resurrecting a read of `metadata.graph_manifest` or
`metadata.producer.cost_scale_factor` on either load path is the
wrong-but-compiling alternative this contract rules out: `metadata.json` is
retired — the study-global graph, `num_stages`, and provenance now live on the
`manifest.bin` `CheckpointManifest` root, consumed only by the `FullFcf`
graph-identity check — so a metadata read would fail to compile or silently
reintroduce a stale-scale bug.

The shared-pool `node_id` reject above is pinned by
`multi_node_shared_pool_is_rejected` in
`tests/shared_boundary_terminal_fan_probe.rs`.

Read: `policy/policy_load.rs` (`validate_policy_load`, `slot_identity`,
`pool_cost_scale_factor`, `convert_checkpoint_cuts_for_load`,
`boundary_predates_self_describing_cuts`,
`PolicyLoadKind::CHECK_STATE_DIMENSION`,
`PolicyLoadKind::CHECK_SLOT_IDENTITY_EXACT`), `policy/reconcile.rs`
(`build_rebind`, `rebind_cut`). Pinned by the `validate_policy_load_full_fcf_*`
(FullFcf exact match, unchanged, including
`validate_policy_load_full_fcf_still_rejects_differing_state_dimension`),
`validate_policy_load_boundary_injection_allows_differing_state_dimension` and
`validate_policy_load_boundary_injection_does_not_check_slot_identity`
(BoundaryInjection defers to reconcile), and
`load_boundary_cuts_empty_manifest_differing_state_dimension_rejects` (the
unverifiable-manifest `state_dimension` fallback guard) tests, plus
`policy::reconcile`'s unit tests, `tests/boundary_reconcile_defaults.rs`, the
end-to-end NEWAVE→DECOMP acceptance regression
`tests/boundary_dim_mismatch_reconcile.rs`
(`newave_source_reconciles_into_decomp_current`,
`newave_boundary_injected_decomp_run_converges`), and the self-describing
clean-break rejects in `tests/boundary_self_describing_clean_break.rs`
(`boundary_load_reads_cost_scale_from_bin`,
`boundary_load_rejects_pre_self_describing_checkpoint`). The date-equality
selector itself is pinned by `policy_load.rs`'s own
`load_boundary_cuts_selects_the_pool_priced_at_the_boundary_date`,
`load_boundary_cuts_unmatched_boundary_date_rejects_naming_available_dates`,
`load_boundary_cuts_multi_pool_date_tie_rejects_naming_both_pools`, and
`load_boundary_cuts_undated_pools_reject_with_reexport_hint`.

A checkpoint predating `FORMAT_VERSION` is pinned by
`boundary_load_rejects_pre_format_version_checkpoint`, also in
`tests/boundary_self_describing_clean_break.rs`.

### A policy written by other software, or another version, is refused first

The first check of `validate_policy_load`, for every `PolicyLoadKind` and
ahead of its check matrix, refuses a source whose recorded writer (the
checkpoint's `manifest.bin` `CheckpointManifest` root: `software` and
`software_version`, read through `CheckpointManifest::written_by`) is not
exactly `SoftwareIdentity::THIS_BUILD`, the identity this build stamps into
every checkpoint it writes: the trainer's `write_checkpoint` and
`cobre.write_policy_checkpoint`, which ignores a caller-supplied identity. A
checkpoint that recorded no `software` is refused like one from another
program. The refusal is `SddpError::PolicySoftwareMismatch`, naming both
writers. Only same-software, same-version loads are supported; a looser rule
(a version range, a `major.minor` match, a per-study predicate, or comparing
the version alone) is the wrong-but-compiling alternative, since cuts and
stored bases from another version or another program describe that writer's
LP, and a product built from this code can carry the same version string.
The identity is checkpoint provenance, not a study-global fact, so reading it
from the root on a `BoundaryInjection` load does not widen the
season-descriptor carve-out below. Every step after `validate_policy_load`
(the check matrix, the boundary rebind, FCF construction and stored-basis
decoding) sees only checkpoints this build wrote.

Read: `policy/policy_load.rs` (`validate_policy_load`) and
`cobre_io::SoftwareIdentity`. Pinned by `policy_version_refused_for_every_kind`,
`policy_from_other_software_refused_at_the_same_version`,
`policy_without_recorded_software_refused`,
`policy_version_checked_before_the_layout` and
`policy_version_refused_at_boundary_load` in `policy/policy_load.rs`, and by
`warm_start_refuses_a_policy_written_by_another_version` and
`boundary_policy_written_by_another_version_is_refused_at_run` in
`crates/cobre-cli/tests/cli_validate.rs`.

### Boundary loads gate on season-cycle and PAR-order identity before reconciling

The inflow-lag family's join maps a lag depth `d` to a calendar season (`d`
seasons before the boundary date) and to the PAR order that season's
coefficient was fitted under. `check_season_compatibility`, called from
`load_boundary_cuts` after the `effective_inflow_lag_depth` guard and before
`validate_policy_load` (so this gate's specific message wins over the generic
`state_dimension` reject), rejects a load whose source and study disagree on
either before any lag coefficient moves — a weekly-versus-monthly or
order-6-versus-order-4 boundary must reject, never blend. The gate is SKIPPED
entirely when the study side is absent — `BoundaryLoadRequest::study_seasons`
is `None`, or its `cycle_code` reads `SEASON_CYCLE_CODE_ABSENT` — since a study
declaring no season map has no PAR context to compare and, having no inflow
models, no lag slots to protect.

With a present study descriptor, the gate rejects on the FIRST of five ordered
checks: the source descriptor is itself absent (a pre-`id:19` checkpoint;
rejects advising re-export, in the same register as
`boundary_predates_self_describing_cuts` but with its own season-descriptor
wording); `cycle_code` differs (named by word via `season_cycle_label`); `n_seasons`
differs (both counts named); a hydro the study's `hydro_orders` models has no
entry in the source's (the hydro named — "the boundary was fitted on a
different set of inflow processes", a more diagnostic reject than the
storage/inflow-lag topology gate a genuinely different deck would otherwise
trip first); or a hydro present on both sides has a differing `orders` vector
AT A SEASON THE STUDY REFERENCES (the hydro, the first such differing season
ordinal, and both orders there named; a length mismatch between the two
`orders` vectors rejects first, naming both counts, so a truncated source
descriptor can never pass by comparing fewer seasons —
`boundary_load_rejects_truncated_source_par_orders_for_a_hydro`).

A study season no inflow model reaches carries no opinion and is skipped by
the per-season comparison above — it is not read as a fitted order-zero. The
study side of the descriptor is `orchestration::StudySeasonManifest`, whose
`hydro_orders[_].orders` is `Vec<Option<u32>>`: `None` at dense season ordinal
`s` means no inflow model of that hydro maps to a stage on season `s` (a
season the study's own stages never reach, including a stage a horizon
reduction truncates away from a longer source study), `Some(k)` is a fitted
order, including `Some(0)`. The source side stays the dense, zero-filled
`cobre_io::SeasonManifest` the checkpoint wire carries — only the loading
study's own opinion can be absent; a source that priced a season the study
never visited is not, on that account alone, treated as absent.
`StudySeasonManifest::to_season_manifest` projects every `None` to `0` when
`write_checkpoint` builds the outgoing wire manifest, so a study's own
checkpoint is byte-identical to what it was before this relaxation existed —
the "no opinion" state lives only on the study-side type, never on the wire.
This is why a pure horizon reduction against a longer source now loads: the
reduction's stages reference a strict subset of the source's seasons, and the
seasons only the source visited are skipped rather than compared against the
reduction's zero allocation.

Both hydro checks walk the STUDY's `hydro_orders` positionally — both sides are
canonical ascending `hydro_id` by construction
(`orchestration::hydro_season_orders`'s `BTreeMap` grouping) — and look each
hydro up in the source's by binary search, never a `HashMap`; a hydro present
only in the source is a superset drop and is never examined. Comparing
`SeasonManifest` wholesale via a derived `PartialEq` is the wrong-but-compiling
alternative this staged form forbids: a single `!=` cannot produce the five
distinct messages the reject tier requires, and it would report a hydro-SET
difference (a missing hydro) as if it were a PAR-ORDER difference.

The wire-side `SeasonManifest`'s ascending-`hydro_id` order and each hydro's
`orders.len() == n_seasons` are no longer assumed from the writer alone: `cobre-io`
rejects a decoded manifest violating either shape before this gate ever runs,
pinned by `checkpoint_manifest_rejects_unsorted_season_hydro_orders` and
`checkpoint_manifest_rejects_season_orders_length_mismatch` in
`crates/cobre-io/src/output/policy/codec.rs`.

`orchestration::build_season_manifest` (renamed from the file-private
`season_manifest`, now `pub`) is the single owner both the checkpoint writer
(`write_checkpoint`) and this gate build their descriptor from via
`BoundaryLoadRequest::with_study_seasons`, so the two sides can never be
constructed by diverging code. This makes the gate's read of
`checkpoint.metadata.season_manifest` a DELIBERATE, narrow carve-out of the
study-global-facts contract above (the one that reads `cost_scale_factor` and
the graph from the resolved pool's own `cuts/<pool>.bin`, never
`metadata.json`): the season descriptor is the one study-global fact this path
reads from the metadata root, because it is genuinely study-global (one season
cycle per study, not one per pool), has no per-pool counterpart to go stale
against, and duplicating it onto every pool would itself be the drift hazard
the metadata-avoidance contract exists to prevent.

Read: `policy/policy_load.rs` (`check_season_compatibility`,
`season_cycle_label`, `BoundaryLoadRequest::study_seasons`/`with_study_seasons`,
`load_boundary_cuts`), `policy/orchestration.rs` (`build_season_manifest`,
`hydro_season_orders`, `StudySeasonManifest`, `StudyHydroSeasonOrders`,
`StudySeasonManifest::to_season_manifest`). Pinned by
`boundary_load_rejects_differing_season_cycle`,
`boundary_load_rejects_differing_season_count`,
`boundary_load_rejects_source_missing_a_modeled_hydro_par_order`,
`boundary_load_rejects_differing_par_order_naming_hydro_and_season`,
`boundary_load_rejects_absent_source_season_descriptor_with_reexport_hint`,
`boundary_load_without_a_study_season_descriptor_skips_the_gate`,
`boundary_load_accepts_a_source_par_order_at_a_season_the_study_never_references`,
and
`boundary_load_rejects_differing_par_order_at_a_referenced_season_despite_unreferenced_gaps`
(all in `policy/policy_load.rs`); the producer-to-consumer end-to-end round trip
`boundary_load_accepts_the_studys_own_checkpoint_under_its_own_season_gate` in
`tests/deterministic.rs`'s `boundary_season_gate_round_trip` module; and the
real-season-map horizon reduction
`horizon_reduction_under_a_real_season_map_reconciles_with_zero_dropped_couplings`
in `tests/boundary_horizon_reduction.rs`.

### Boundary loads gate on the storage/inflow-lag topology before reconciling

The state's must-correspond core (storage and inflow-lag) is checked UP FRONT,
all at once, rather than discovered mid-rebind. `check_topology_subset`, called
from `load_boundary_cuts` immediately after the (hoisted)
`manifest_identity_verifiable` check and before `validate_policy_load` — sharing
that same emptiness gate, so it is SKIPPED on the absent-manifest path exactly
like the intercept fold and the rebind below — walks the CURRENT manifest's own
storage and inflow-lag slots against a `HashSet` built once over the SOURCE
manifest's, and rejects naming EVERY missing entity in one message per family
(storage first, then inflow-lag — a missing reservoir is the louder
"different deck" signal, and a deck missing a plant is usually missing its lag
block too), rather than the first one `reconcile::build_rebind` would otherwise
report. A source slot with no current counterpart is a superset drop, not
examined here — that is `BoundaryReconciliationReport::superset_summary`'s and
`load_boundary_cuts`'s own `policy.boundary.strict` gate's concern; the boundary
path emits no warning for a dropped source slot.

`reconcile::resolve_storage`/`resolve_inflow_lag`'s own `RebindOp::Reject` arms
are UNCHANGED: through `load_boundary_cuts` they are now unreachable once the
gate has passed, but they remain `build_rebind`'s own postcondition for a direct
in-crate caller. Downgrading either arm to a `debug_assert!` instead is the
wrong-but-compiling alternative this contract forbids: it would let a future gate
regression silently zero a water value in a release build with no backstop at
all.

Read: `policy/policy_load.rs` (`check_topology_subset`, `load_boundary_cuts`),
`policy/reconcile.rs` (`resolve_storage`, `resolve_inflow_lag`). Pinned by
`boundary_load_rejects_every_unpriced_hydro_in_one_message`,
`boundary_load_rejects_missing_inflow_lag_depth_naming_hydro_and_depth`,
`boundary_load_superset_source_passes_the_topology_gate`,
`boundary_load_absent_manifest_skips_the_topology_gate`,
`load_boundary_cuts_entity_id_mismatch_rejects`, and
`load_boundary_cuts_storage_slot_absent_from_differently_typed_source_rejects`
(all in `policy/policy_load.rs`), plus `reconcile.rs`'s own
`build_rebind_storage_miss_rejects_naming_hydro` and
`build_rebind_lag_miss_rejects_naming_lag_depth`, which exercise
`build_rebind`'s postcondition directly.

### Inflow-lag reconciliation validates the reference date, not just identity

The topology gate above proves the ENTITY matches; it says nothing about
WHEN either side's lag points. `resolve_inflow_lag`'s identity hit is
followed by a `reference_date` check: the two `YYYYMMDD` `i32` stamps
(`lag_reference_anchor`'s stamped past-stage date, or
`ENTITY_SLOT_DATE_SENTINEL`) are compared RAW, never as decoded
`NaiveDate`s. Both dated and equal copies exactly as before; both dated and
DIFFERENT rejects, naming the hydro, the lag depth, and both dates (each
rendered through `cobre_io::decode_slot_date`, degrading to the raw integer
on an undecodable stamp) — a "different past" diagnosis, worded distinctly
from the identity-miss lag-depth-incompatibility reject above so the two
failures, which have different remedies, are never conflated.

A sentinel on EITHER side falls back to the identity copy, never a reject —
the `reserve_boundary_inflow_lag_slots` bridge carve-out. That DECOMP-bridge
bootstrap builds slots from a manifest and coefficients, never a calendar, so
it can only ever emit the sentinel; rejecting on it would break the pinned
NEWAVE-to-DECOMP acceptance regression below with no replacement path. A
sentinel is the absence of a date, not a wrong one — `lag_reference_anchor`
emits it when a lag reaches past the earliest declared stage, which two
studies with different pre-study window lengths legitimately do at differing
depths.

Two wrong-but-compiling alternatives: comparing DECODED `NaiveDate`s instead
of the raw `i32` stamps — an unparseable non-sentinel stamp (never expected
past `read_policy_checkpoint`'s date validation, but not ruled out by the
type system) would decode-fail into `None` on both sides and compare
spuriously equal instead of on its own raw value, so the comparison and the
message-rendering decode must stay two separate steps; and keying the join
on `(hydro, reference_date)` instead of the unchanged `(hydro, lag-index)` —
`build_stage_entity_manifest` is the sole owner of the 1-based `subindex`
convention on both sides, so there is no lag-index convention drift to guard
against, and re-keying would make every undated slot unjoinable, breaking
the same carve-out this section protects.

Read: `policy/reconcile.rs` (`resolve_inflow_lag`, `render_reference_date`),
`policy/policy_export.rs` (`lag_reference_anchor`,
`reserve_boundary_inflow_lag_slots`, `build_stage_entity_manifest`). Pinned
by `inflow_lag_differing_reference_dates_reject_naming_both_dates`,
`inflow_lag_matching_reference_dates_copy`, and
`inflow_lag_sentinel_on_either_side_copies_by_identity` (`policy/reconcile.rs`
unit tests), and `boundary_injection_differing_lag_reference_date_rejects`
and `boundary_injection_undated_source_lag_still_copies` (both in
`tests/boundary_reconcile_defaults.rs`), alongside the unchanged
no-regression case `boundary_injection_storage_lag_identity_match_succeeds`
and the bridge regressions `newave_source_reconciles_into_decomp_current`
and `newave_boundary_injected_decomp_run_converges`
(`tests/boundary_dim_mismatch_reconcile.rs`), whose
`reserve_boundary_inflow_lag_slots`-built source stays sentinel-dated and
must keep loading.

### Dated forward-family fan-out reconciliation is hour-weighted by the SOURCE interval

A forward-family slot's identity date is its `[interval_start, interval_end)`
span, never a single anchor — `HydroInflowLag` is the one family keyed on a
single `reference_date` instead (a past stage, not a delivery target).

The calendar-`subindex` relaxation the parent section reserves for a date family
IS this fan-out, and it now covers BOTH forward-dated families —
`AnticipatedThermalState` and `HydroTransitBucket` — inside `load_boundary_cuts`,
dispatched through the single `resolve_by_interval_overlap` resolver; there is no
per-family miss-rule parameter, because both families miss to `Zero` for their own
expected reason (below). A source study prices its anticipated commitments on a
monthly delivery calendar and its transit-bucket arrivals on their own per-arc
arrival calendar; the current study may deliver either on a differently-shaped
calendar (weekly, monthly, or a differing arc topology). `resolve_by_interval_overlap`
reconciles each target slot `w` against the source intervals `M` of the SAME family
it overlaps **by real calendar date**, not by subindex: full coverage yields
`RebindOp::Blend` with per-interval weight `overlap_hours(w, M) / H_M` — divided by
the **source** interval's hours `H_M`, the conservation identity that makes a
slot's fanned coefficients sum back to the source's (a covered target slot's coeff
ratio equals `H_w / H_M`). A target slot straddling into unpriced time yields
`RebindOp::Renormalize`: the same weights additionally scaled by
`H_w / Σ_covered overlap`, so the covered intervals' price density replicates
across the uncovered span instead of deflating the boundary FCF with an implicit
`0.0` term. No covered interval yields `Zero` — for `AnticipatedThermalState` this
is usually an in-study delivery (below); for `HydroTransitBucket` this is the
expected outcome when the source is NEWAVE-shaped and carries no transit arcs at
all, the same "miss is expected, not incompatible" status the identity join used
to grant it, now reached through the calendar join instead.

Only a live anticipated target slot whose interval reaches past
`boundary_date` fans out. A live target slot whose `interval_end` is at or
before `boundary_date` is an IN-STUDY ring slot — its resolved physical
delivery target still lands within the current horizon (chiefly a `K = 0`
sub-stage-lead delivery, self-delivered at its own stage) — and resolves to
`Zero`: the terminal boundary FCF prices only post-study obligations, so a
within-horizon delivery, already discharged inside the study, contributes
nothing. This is sound BECAUSE a ring slot's `interval_start`/`interval_end`
are stamped from the SAME modular delivery target, resolved at the SAME
OUTGOING anchor (`current_stage_idx + 1`, the state leaving the pool's own
stage — see the manifest's own contract above) — `build_stage_entity_manifest`
recovers `(slot, plant)` from a ring column via `slot_lane_at` (the exact
inverse of `out_col`/`in_col`) and intervals the slot at its modular delivery
stage in one pass. Under a shared stage calendar, a study-stage delivery
always ends at or before the study's own boundary date and a post-study-stage
delivery always ends after it, so `interval_end <= boundary_date` partitions
the ring exactly into in-study (`Zero`) and post-study-targeted (fan-out)
slots — never a failed post-study resolution.

The terminal pool's OWN maturing residue — the slot whose subindex matches
the terminal stage's own delivery class, the same slot the always-fish arm
reads via `in_col` that stage — is NOT this in-study case. Resolved at the
outgoing anchor (one stage past the terminal stage), its next same-residue
ring-axis target is always a full `k_max` stages past the terminal stage:
the slot now dates and intervals onto its REAL post-study delivery and fans
out (`interval_end` past `boundary_date`, `Blend`/`Renormalize`) whenever a
declared post-study calendar reaches that far, or reads the sentinel — still
`Zero`, but because no calendar exists to date it against, never because the
delivery is discharged in-study — when none does. Anchoring this slot on the
entering `current_stage_idx` instead of the outgoing anchor — the RETIRED
behavior — is now the wrong-but-compiling alternative: it dates the terminal
maturing residue onto the terminal stage's own in-study month, silently
zeroing a real post-study delivery a declared post-study calendar should
have priced. Pinned by
`terminal_maturing_residue_dates_onto_its_post_study_delivery` and
`terminal_maturing_residue_stays_sentinel_without_a_post_study_calendar`
(`policy/policy_export.rs`). Two further wrong-but-compiling alternatives,
retired earlier and still forbidden: `RebindOp::Reject` here aborts a
legitimate boundary load the moment any anticipated thermal targets an
in-study delivery (the `K = 0` case); resolving an in-study slot an
in-horizon interval and fanning it out would wrongly `Blend` a within-horizon
delivery against the source's months.

`Blend` and `Renormalize` are semantically distinct and MUST NOT be collapsed:
`rebind_cut` applies both through the identical weighted-sum, so unifying them
reads like harmless dedup — but only `build_rebind` carries the `Renormalize`
anti-deflation scale, and dropping it silently understates the boundary FCF on any
partial-coverage calendar. Two further wrong-but-compiling alternatives: dividing
by the **target** slot's `H_w` instead of the source `H_M` (breaks the
`H_w / H_M` conservation ratio), and joining on `subindex` instead of the real
interval (anticipated delivery is NON-monotone in subindex — the modular
delivery-target residue of the ring contract above — so a subindex join misaligns
months to weeks). Source intervals are read from each live source slot's OWN
`interval_start`/`interval_end` (`build_source_interval_index`, decoding through
`cobre_io::decode_slot_date`) — the source-side counterpart of the target-side
read the parent section describes, never a `YYYYMM01` anchor reconstruction; an
exact or superset match reconciles byte-for-byte (`Copy`). The `H_w / covered`
division is guarded: `resolve_by_interval_overlap` returns `Zero` on
`terms.is_empty()` before it can divide by a zero covered span.

`HydroTransitBucket` target and source slots reconcile through this EXACT SAME
`resolve_by_interval_overlap` call and the exact same `÷H_m` math above — there is
no separate transit resolver. `build_source_interval_index` selects live
(non-sentinel) source slots of EITHER family into one shared index, keyed
`(entity_type, entity_id)`; the family byte already in that key is what keeps an
anticipated and a transit source entry sharing one `entity_id` disjoint, so
widening the index to a second family needed no new discriminator (see the
must-never-collapse-to-bare-`entity_id` contract on `build_boundary_fold`'s own key
above — the identical family-byte discipline). A transit target's `subindex` (its
maturity lag) plays no part in the join, for a reason distinct from anticipated's
non-monotone-residue one: two buckets of one downstream plant at different lags
have disjoint arrival intervals (each stamped from its own arrival stage on the
extended delivery calendar) and are told apart by that alone; re-introducing
`subindex` into the join would reject a legitimate source whose lag indexing
differs from the current study's. Under an identical source/current arrival
interval the overlap is full and the single term's weight is
`overlap / H_m == 1.0`, reproducing today's coefficient bit-for-bit — but the
reconciliation report tallies that slot as `fan_out`, never `copy`: the
coefficients are unchanged, the JOIN MECHANISM is not, and collapsing a
full-coverage single term back into `RebindOp::Copy` would hide that the family is
now date-validated (and would have to apply to the anticipated family's own
copy-equivalent, fully-inside-one-source-interval case too, silently changing its
tally as well). A dated transit miss that is NOT the NEWAVE-shaped no-arcs case —
both sides declare transit arcs but their intervals do not overlap — still resolves
`Zero` at the same report tier as any other miss; promoting it to a louder tier is
diagnostics work this contract does not cover.

The constant intercept fold (`build_boundary_fold`) reuses this SAME
`overlap/H_m` source-interval weighting on a different object: a class-4 fixed
post-horizon window's declared MW is a CONSTANT, not a state dimension, so
`Σ_m (overlap_hours(w, m) / H_m) · v_w` sums directly into a `(source_pos,
factor)` fold vector instead of a `Blend`/`Renormalize` op — a delivery is
either a fixed window folded into the intercept or an in-study/post-study ring
decision resolved through `Blend`/`Renormalize`, never both, since the two
paths draw from disjoint inputs (`fixed_windows` vs. `target`'s ring slots) and
write to disjoint outputs (the intercept scalar vs. a state-dimension
coefficient). A fixed window overlapping no source interval contributes zero —
mirroring `RebindOp::Zero`, never an error — and the fold's own filter drops
every exact `0.0` factor rather than emit a zero-weighted term. `load_boundary_cuts`
adds the fold onto each cut's RAW intercept (`record.intercept += Σ
coeff[source_pos] · factor`) BEFORE `rescale_cut_records_for_load`'s cost-scale
and legacy-ratio transforms, in the same source-coefficient frame `rebind_cut`
reads — so the folded future-cost term rides both rescale transforms with the
rest of the intercept. Folding after rescale, or reading a rescaled coefficient
into the fold sum, is the wrong-but-compiling alternative: the transforms are
not idempotent across that boundary, so a post-rescale fold prices the fixed
commitment at the wrong scale.
Read: `policy/reconcile.rs` (`resolve_by_interval_overlap`, `build_rebind`,
`rebind_cut`, `build_source_interval_index`, `overlap_hours`, `build_boundary_fold`,
the `Blend`/`Renormalize` variants), `policy/policy_export.rs`
(`build_stage_entity_manifest`, intervalling each ring slot at its modular
delivery stage via `slot_lane_at`), `policy/policy_load.rs` (`load_boundary_cuts` builds one
`source_index` shared by `build_rebind`/`build_reconciliation_report`, threads
`boundary_date` into `build_rebind`, and applies the fold before rescale).
Pinned by the `hm_distribute_conservation` fixtures in
`tests/anticipated_core.rs` (coeff ratio equals `H_w / H_M`, invariant to the
delivery stage's hours), the `Blend`/`Renormalize` `rebind_cut` unit tests (both
apply identical mechanics, distinction is only the weight),
`tests/boundary_reconcile_defaults.rs` (the fan-out matrix and the superset
bit-identity `to_bits` pin),
`anticipated_target_ending_at_the_boundary_date_zeroes` (a live target
interval ending at or before the boundary date resolves to `Zero`, not a
reject, even when a source month would overlap it), and the constant
intercept fold's own regressions
`boundary_fold_marked_frame_moves_intercept_by_hand_computed_delta`,
`boundary_fold_multi_month_window_sums_contributions`,
`boundary_fold_no_overlap_window_leaves_intercept_bit_identical`, and
`boundary_fold_empty_windows_intercept_bit_identical`, all in
`policy/policy_load.rs`. The transit-bucket join shares every one of the
`resolve_by_interval_overlap`/`build_source_interval_index` pins above and is
additionally pinned by `policy/reconcile.rs`'s own
`transit_bucket_identical_arrival_interval_blends_at_unit_weight`,
`transit_bucket_target_inside_one_source_interval_blends_at_fractional_source_hours`,
`transit_bucket_non_overlapping_arrival_interval_zeroes`,
`transit_bucket_join_ignores_subindex`,
`transit_bucket_sentinel_interval_is_a_structural_pad`, and
`unrecognized_entity_type_still_rejects_by_identity`, plus the boundary-load
round trip `load_boundary_cuts_matching_transit_bucket_arrival_interval_round_trips`
(`policy/policy_load.rs`) and
`boundary_injection_transit_bucket_blends_on_identical_arrival_interval`
(`tests/boundary_reconcile_defaults.rs`).

A sentinel-interval forward-family slot is a structural ring pad on BOTH sides
of reconciliation, not just the target: `dropped_source_positions` excludes it
from the source-side `dropped_source` tally exactly as `classify_op` excludes
it from the target-side `default_zero` tally above, through the single shared
`is_structural_pad` predicate — the two tallies can never disagree on what
counts as a pad. Counting an unreferenced pad as a genuine drop is the
wrong-but-compiling alternative: a target-shaped or fully-covered source's own
ring pads are then never referenced by any op, so its `dropped_source` reads
non-zero on a load that is otherwise bit-identical, and would wrongly fire a
superset-source warning on a faithful reload of a study's own terminal
manifest. Pinned by
`dropped_source_positions_excludes_sentinel_forward_family_pads` and
`dropped_source_positions_still_reports_an_unreferenced_storage_slot`
(`policy/reconcile.rs`), and by
`boundary_injection_report_target_shaped_superset_is_copy_only` and
`boundary_injection_report_fan_out_matrix_coverage`
(`tests/boundary_reconcile_defaults.rs`), each asserting an empty
`dropped_source_slots` where the only unreferenced source positions are pads.

## Initial-state seeding resolves IDs through a position map, never `binary_search`

`System::hydros()`/`thermals()` sort canonically by `(operational_start_date,
id)`, which is id-ascending only when every entity shares one operational
start date. A staggered-commissioning system (filling reservoirs, future-entry
plants — the entire point of `operational_start_date`) breaks that
coincidence, so `binary_search_by_key` over the canonical slice — which
requires id-ascending order — silently returns `Err` (or the wrong index) for
an out-of-id-order entity, dropping its seed to the default `0.0`. Every
id-keyed initial-condition lookup (`storage`, `filling_storage`, thermal
`past_anticipated_commitments`) resolves through an `id -> position` map built
once from the canonical slice, never a `binary_search_by_key` call. The map is
built from the canonical order, but every write still iterates the IC record
list (not the map) — a map iteration order is unspecified and would violate
declaration-order invariance if used to drive writes.

The derived inflow lag seed (`derive_inflow_seeds`) satisfies the same
invariant a different way: it carries no id->position map at all — it
iterates `hydros` directly, so the loop index IS the canonical position, then
filters each hydro's own historical windows by id. `build_initial_state`'s lag
block trusts this pre-ordering and does a plain positional read, with no id
lookup of its own.
Read: `setup/mod.rs` (`id_to_position`, `build_initial_state`),
`crates/cobre-stochastic/src/seeds.rs` (`derive_inflow_seeds`). Pinned by
`test_initial_state_seeds_correctly_under_staggered_commissioning_dates`,
`build_initial_state_anticipated_seed_correct_under_staggered_commissioning_dates`,
and `test_seed_correct_under_staggered_commissioning_dates`, each using a
staggered-date fixture where the canonical order is id-descending.

## The initial storage takes the hydro's physical range, never the stage-0 box

The stage-0 `StateBox` storage bounds are that stage's OUTGOING column bounds,
so they carry its end-of-stage storage targets (`hydro_bounds`
`min_storage_hm3`/`max_storage_hm3`). The incoming initial storage has no
producing stage: `canonicalize_initial_state` projects it onto the hydro's own
`[min_storage_hm3, max_storage_hm3]`, the floor lifted to `0` where
`storage_floor_relaxed` holds at stage 0. Projecting it onto the stage-0 box is
the wrong-but-running alternative: a stage-0 floor above the declared storage
raises it and a ceiling below lowers it, creating or removing water before the
first solve with no diagnostic. Travel-time buckets keep the stage-0 box.
Read: `setup/mod.rs` (`canonicalize_initial_state`), `lp/builder/hydro_state.rs`
(`storage_floor_relaxed`, the one owner of which hydros have a hard floor).
Pinned by `initial_storage_below_stage0_storage_floor_is_not_raised` and
`initial_storage_above_stage0_storage_ceiling_is_not_lowered`
(`tests/state_drift_absorption.rs`).

## Water travel time

A declared upstream→downstream arc introduces in-transit "bucket" state: one
Markov-1 volume slot per `(downstream plant, lag)` absorbs water in flight. With
the feature compiled in but no arc declared (`n_buckets == 0`), every path below
collapses to the pre-bucket layout byte-for-byte; the moment any arc is
declared, each of the following is a contract.

### Shared lagged-delivery ring skeleton

The water in-transit bucket ring and the anticipated-thermal ring are one
lagged-delivery ring construct, owned by `DeliveryRing`. They share a borrowed
outgoing block (identity-resolved, contributing to `n_state`), a separate
borrowed incoming block (pinned via `state_to_lp_incoming_column`), and the
paired row-cap/column-freeze masking (`DeliveryRing::freeze_masked_columns`).
The interior transition differs: water shifts one Markov-1 slot per stage
(`DeliveryRing::emit_shift_rows`, slot → slot+1), while anticipated holds each
commitment in its own slot (`DeliveryRing::emit_carry_rows`, same slot). The
rings also differ in how each deposits into its newest slot and in what a
masked terminal slot means. Every difference lives entirely at each ring's own
call site, never a second skeleton implementation; the side-by-side table in
`docs/design/anticipated-thermals-and-water-travel-time.md` §4 lists them:

- **Deposit.** Water's block-mode-coupled per-lag deposit share is emitted at
  its own call site (`fill_arc_release_block_entries`), never through
  `DeliveryRing::emit_deposit`. Anticipated's deposit IS `emit_deposit`: it
  pins the ring's newest slot to a single decision column, `+1` on
  `out_col(slot, lane)` and `−1` on `decision_col`.
- **Masked terminal slot.** Water's masked slot discards a genuine share the
  ring would otherwise deposit — an admitted target-stage imprecision (see
  Terminal credit deferred below). Anticipated's masked slot never held a
  value in the first place, because no anticipated commitment is ever created
  past the horizon (see End-of-horizon masking below). Both render the SAME
  masking output (frozen `[0, 0]`, scale-independent) — only the per-ring
  subsection below states what the masked slot MEANS.

The masking contract is always two-sided and ships together: a masked
position (`row_pos[i] == None`) gets NO definition row (the row-cap side) AND
a frozen `[0, 0]` outgoing column (`freeze_masked_columns`, the column-freeze
side) in the SAME pass — wiring only one side leaves either a dangling row
referencing a frozen column or a free column with no defining constraint, both
wrong-but-compiling. Water instantiates one ring per downstream plant
(`DeliveryRing::transit_buckets`, `n_lanes = 1`, over that plant's ragged
contiguous sub-range); anticipated instantiates ONE dense ring spanning every
plant (`DeliveryRing::anticipated`, `n_lanes = n_anticipated`,
slot-major/plant-minor) — both addressing schemes resolve through the same
`out_col`/`in_col` formula (`block.start + slot * n_lanes + lane`).
Read: `lp/builder/delivery_ring.rs` (`DeliveryRing::emit_shift_rows`,
`freeze_masked_columns`, `emit_deposit`, `out_col`/`in_col`, `slot_target`,
`DeliveryRing::anticipated`, `DeliveryRing::transit_buckets`).

### In-transit bucket dynamics & sign

`fill_transit_bucket_definition_entries` routes the bucket-definition ring
shift through `DeliveryRing::emit_shift_rows` (the shared skeleton above,
`b_d^out = b_{d+1}^in + k_d·D_i`); `fill_arc_release_block_entries` deposits
the arc's `k_d`-weighted release from the SAME release column that also
carries `k_0` onto the balance row — never a separate once-per-stage family
(the deposit share itself is emitted at the call site, never through
`DeliveryRing::emit_deposit`, which only the anticipated ring calls). Incoming
buckets are pinned via column bounds, resolved through
`StateSpace::state_to_lp_incoming_column`'s explicit `transit_buckets_in` arm,
never falling through to the commitment-hold `commit_in` arm. Subgradient extraction
divides the incoming bucket column's reduced cost by `col_scale`
(`extract_duals_from_view`, the same rc/col_scale contract as storage); the
cut row renders the **outgoing** bucket column through
`StateSpace::lp_column_for_state`'s identity arm and multiplies `col_scale`
back on via `push_scaled_coefficient` — divided on extract, multiplied on
render, identical to storage. Swapping which column is pinned/read, or
dividing on render instead of extract, prices the in-transit water in the
wrong direction — a wrong bound that still compiles. A fold implementation
(crossing mass absorbed same-stage, no bucket at all) can reach the same total
cost as the correct one, so total cost alone cannot discriminate — only the
dual's sign/magnitude and the per-stage delivery split do.
Read: `lp/builder/entries.rs` (`fill_transit_bucket_definition_entries`,
`fill_arc_release_block_entries`), `lp/builder/delivery_ring.rs`
(`DeliveryRing::transit_buckets`), `lp/indexer/state_space.rs`
(`StateSpace::state_to_lp_incoming_column`, `StateSpace::lp_column_for_state`),
`training/backward/duals_extraction.rs` (`extract_duals_from_view`), `cut/row.rs`
(`push_scaled_coefficient`, `push_cut_row`). Pinned by the bucket-arm
column-resolution tests (outgoing resolves by identity, incoming resolves to the
pinned column via an explicit arm, never the anticipated catch-all) and the
per-stage-visit bucket-pinning regressions in the backward pass and lower-bound
evaluation; a sub-stage-delay bucket-dual regression is the fold-discriminating
pin for the sign/magnitude itself.

### k-factor conservation

`resolve_spread` sums the stage-clock weights to `Σ_d k_d = 1` per arc per
anchor stage (`debug_assert`-enforced), and `fill_arc_release_block_entries`
asserts the same sum immediately before it deposits. A closed-form ceiling
depth (e.g. `⌈t_v/h_t⌉`) is a plausible-looking replacement for the resolver's
overlap-based depth and silently drops trailing mass on a non-uniform calendar
— conservation violated, not a compile error.
Read: `lead_time/mod.rs` (`resolve_spread`), `lp/builder/entries.rs`
(`fill_arc_release_block_entries`). Pinned by the resolver's monthly-then-weekly
counterexample regression (asserting the correct, deeper depth against the
closed-form ceiling's shallower, wrong one) and the stage-level conservation
regression exercising the `Σ_d k_d = 1` debug_assert directly across
non-uniform calendars; a mixed-calendar end-to-end regression extends the pin
to delivered-plus-horizon-drop equalling released, per arc, to floating-point
tolerance.

A plant's turbined flow is `Σ_c q_c` over its `HydroCellIndex` cells — a
disjoint CSR partition, not a duplicate representation — so an arc's `k_d`
prices the plant's TOTAL release and is REPLICATED onto every cell's turbine
column at the same magnitude, never apportioned (divided) across them:
apportioning by `1/|C|` discards `(1 − 1/|C|)` of the released mass, an
under-delivery no less wrong than the ceiling-depth bug above. Conservation
holds PER CELL, not merely in the aggregate — every cell of a plant feeds the
same arc at the same travel time, so `stage_weights` is cell-invariant by
construction and the `Σ_d k_d = 1` debug_assert stays exactly where it is
(once per arc per stage), never moved inside a per-cell loop. This holds only
while travel time is an ARC (plant) attribute; if a cell ever acquires its own
`t_v`, each cell needs its own weight vector (each still summing to 1) and the
assertion moves inside the per-cell loop — still never apportioned even then.
Read: `lp/builder/entries.rs` (`fill_arc_release_block_entries`,
`fill_arc_release_chrono_block_entries`). Pinned by
`test_cascade_release_sums_the_upstream_plants_cells` (same-magnitude,
not-divided per cell) and `test_plant_total_release_is_invariant_to_cell_partition`
(a solved-LP objective/dual comparison between a one-cell and an evenly-split
two-cell plant releasing the same total).

### Canonical bucket ordering

Bucket columns sort by the downstream plant's canonical
`(operational_start_date, id)` index — the same order `System::hydros` already
carries — then by lag; never by raw declared id, never by cascade-traversal
order. `build_transit_bucket_topology` derives `column_order` from that canonical
iteration alone. Emitting buckets in traversal order instead makes the state
layout input-declaration-order-dependent, breaking the
declaration-order-invariance hard rule.
Read: `bucket_topology.rs` (`build_transit_bucket_topology`,
`TransitBucketTopology::column_order`). Pinned by the bucket column-order
declaration-invariance regression: two systems differing only in the
declaration order of their hydros produce identical `column_order` and
`n_buckets`.

### Stage-0 seed: windowed IC anchor

`build_initial_transit_bucket_state` seeds every declared arc's stage-0
incoming buckets directly from its `past_defluences` windows — never a
positional walk over a fixed pre-study calendar. For upstream hydro `i`'s
window `[start_date, end_date)`, `e_off = start_0 − end_date` and
`width = end_date − start_date` feed the shared `StageCalendar`'s
`hour_window_shares(t_v, cumulative_before, period_duration)` — a pure
hour-clock overlap over the study-stage durations, exactly as it already
takes `(cumulative_before, period_duration)`: the windowed derivation lives
entirely in how the caller computes those two offsets from calendar dates,
never inside the resolver itself. A hydro may carry multiple, non-contiguous
windows; the seed must `filter` over every
window with a matching `hydro_id` and deposit each one independently
(`volume = width · M3S_TO_HM3 · value_m3s`, `seed[start+d] += k[d] · volume`)
— a `.find()` would silently keep only the first window and drop the rest,
understating the seed with no error. There is no fallback for incomplete
coverage: `cobre-io`'s `validate_travel_time` row-5 gate guarantees every
declared arc's windows cover `[start_0 − t_v, start_0)` before setup ever
runs this seed.
Read: `setup/mod.rs` (`build_initial_transit_bucket_state`,
`splice_transit_bucket_seed`), `cobre-stochastic`'s
`season_cast::StageCalendar::hour_window_shares`. Pinned by the single-window
unroll regression (the `k`-weighted deposit matches the closed-form
half-share), the gapped-two-window additive regression (two non-contiguous
windows for one arc contribute independently), and the seed's own
declaration-order-invariance regression (distinct from, and in addition to,
the topology-level ordering pin above).

### Delivery-family right-boundary pricing

Both delivery-family carriers — the anticipated-thermal hold ring
(`## Anticipated thermal commitments`) and the water travel-time bucket ring —
keep terminal in-flight state LIVE and price it through the SAME already-generic
cut-state projection (`β·state`), never a per-family pricing arm.
`StateRegion::cut_enabled` returns `true` for both `Buckets` and `CommitmentHold`
at every pool, the terminal pool included, and `CutStateProjection::new` walks
every cut-enabled region's `state_dim_range` with no entity-type and no per-stage
arm — so a kept-live terminal slot of either family is already a priced FCF
dimension a loaded boundary cut's `β` lands on directly. The projection carries
every such slot today; only whether the slot holds live value or a masked
`[0, 0]` structural zero depends on the per-family fill. A per-family
terminal-pricing arm is the forbidden alternative: the projection already owns
the coefficient, so a second path double-counts or misaligns it.

The two carriers reach that live terminal state through a LOAD-BEARING asymmetry
that must not be flattened into one shared keep-live helper:

- **Thermal is the single anticipated ring's own reachable slots, NOT a
  `config.policy.boundary`-gated appendage.** A post-study-targeted slot is one of
  the ring's own slots, held open `(-inf, inf)` by `fill_anticipated_slot_columns`
  whenever reachable — the `freeze_masked_columns` reachability geometry over the
  whole `anticipated_slot_row_pos`, never a boundary-conditional appended block. Its
  EXISTENCE is gated on the study declaring a `post_study_stages.json` calendar (the
  only thing that extends the delivery axis past the horizon:
  `StateSpace::n_delivery` is the attached resolution's decider length, the
  study stages plus the post-study continuation, so
  `build_anticipated_slot_row_pos` gates reachability on `m < n_delivery`, not
  `m < n_stages`) plus the plant's own lead reaching the slot — never on a loaded
  boundary. With no post-study calendar the axis is study-only and every
  `m >= n_stages` slot is masked `[0, 0]`, none created (End-of-horizon masking
  above). This is the re-derived form of the asymmetry, NOT its collapse: water
  gates terminal live-STATE on `config.policy.boundary` presence; anticipated gates
  post-study slot EXISTENCE on a declared `post_study_stages.json` plus lead reach.
  Applying water's `config.policy.boundary` gate to the anticipated ring — masking a
  reachable post-study slot `[0, 0]` unless a boundary is loaded — is the
  wrong-but-compiling alternative: a legitimate `min_mw == max_mw` replay deck with
  no boundary then loses the state slot its declared post-study commitment is
  carried in, and the generic `β·state` projection already owns that slot's
  coefficient whether or not a boundary fills the FCF.
- **Water is the ring's own capped slots and is NOT inert.** Its terminal state
  is the bucket ring's own horizon-capped deep-lag slots, which
  `horizon_cap_active` masks `[0, 0]` at the terminal (Terminal credit deferred
  below). Un-masking them re-enables their definition rows, deposit share, and
  outgoing columns — a live LP change, not an inert appendage — so it MUST be
  gated on `config.policy.boundary` presence: a zero-terminal-value study (no
  boundary) keeps the masked layout byte-for-byte, and only a boundary-loaded
  study opens the terminal slots.

Un-masking the water terminal slots UNCONDITIONALLY — dropping the
`config.policy.boundary` gate — is the wrong-but-compiling alternative. It
re-enables the terminal deposit and outgoing columns for every study, so a
zero-terminal-value water-travel-time study, whose terminal value is still zero,
now routes the end-of-horizon release into a bucket slot it used to drop; the LP
the solver sees changes, silently perturbing every existing water-travel-time
golden even though the optimal cost is unchanged. Byte-neutrality for the
no-boundary case is the property the gate protects.

The water terminal state is also EMITTED for rolling seeding (the `transit_seed`
output, reconstructed from realized turbined+spilled releases in the
`past_defluences` schema so a follow-on run reuses `build_initial_transit_bucket_state`
verbatim). That rolling round-trip is faithful ONLY for `t_v <= horizon`: the seed
reader derives its `StageCalendar` from the receiving study's own un-padded stage
list, so it cannot represent an in-transit lag deeper than that study's stage
count. For `t_v > horizon` the deep pre-study mass beyond the horizon is not
carried across the seam — the SAME Terminal credit deferred imprecision (below),
surfacing at the rolling boundary rather than being introduced by the emit format
(a direct bucket-state emit would hit the identical reader truncation). This is a
ratified scope boundary, not a bug; lifting it requires the seed reader to
represent lags past the horizon. Pinned by the `#[ignore]`d
`round_trip_continuity_needs_the_leftover_seed_stitch_when_travel_time_exceeds_horizon`
reproduction, alongside the passing `t_v <= horizon` round-trip, in
`tests/hydro_sim.rs`.
Read: `lp/indexer/state_space.rs` (`StateRegion::cut_enabled`),
`lp/indexer/cut_state_projection.rs` (`CutStateProjection::new`),
`bucket_topology.rs` (`horizon_cap_active`), `lp/builder/columns.rs`
(`fill_anticipated_slot_columns`, `fill_transit_bucket_columns`),
`crates/cobre-io/src/config/policy.rs` (`PolicyConfig::boundary`). Pinned by
`every_bucket_dim_projects_including_deep_terminal_lags` (every bucket dim, the
deep-lag terminal slots included, appears exactly once in the cut-state
projection with no entity-type or per-stage gate) and
`commitment_hold_post_study_target_joins_the_projection` (the single anticipated
ring's post-study-targeted slots join the same projection), both in
`lp/indexer/cut_state_projection.rs`.

### Terminal credit deferred

`horizon_cap_active` caps each stage's active lag at `n_stages − 1 − t`, the
deepest lag whose target stage still lands inside the horizon;
`build_transit_bucket_row_pos` gates the per-stage LP fill on that cap, so a lag beyond
it gets no bucket-definition row at that stage — dropped by construction, not
retained and silently zeroed elsewhere. `fill_arc_release_block_entries` /
`fill_arc_release_chrono_block_entries` drop the matching deposit share rather
than write it to a stale row index, and `fill_transit_bucket_columns` freezes the
masked slot's outgoing column `[0, 0]` (the commissioning-dormant-column
convention) so no row is needed to define it. The complementary guarantee is
why dropping the row is safe: the finite horizon's zero terminal value
(`HorizonMode::Finite`, the only implemented mode) makes a masked slot's cut
coefficient structurally zero, so no solution loses value by never routing
water into it — the residual mass has no receiving stage either way. This
safe-drop is scoped to a zero-terminal-value study — one with no
`config.policy.boundary`: with a boundary loaded the terminal value is not zero,
the masked slot's coefficient is no longer structurally zero, and dropping the
slot would lose real value — the case the Delivery-family right-boundary pricing
convention above handles, keeping the terminal bucket state live and pricing it
through the shared cut-state projection, gated on `config.policy.boundary`
presence so this drop stays byte-for-byte for a zero-terminal-value study. This
under-values end-of-horizon upstream release; it is a documented target-stage
imprecision, not a bug to patch by capping `TransitBucketTopology::column_order`
too — it sizes from the global max over every anchor and must retain what the
earliest stages need. Both
drop sites (`fill_arc_release_block_entries`, `fill_arc_release_chrono_block_entries`)
now assert the confinement directly: a debug-only check at the `None` row-lookup
arm requires the dropped lag `d` at stage `t` to satisfy `t + d >= n_stages`,
so the drop is provably confined to a target past the horizon and unreachable
once `boundary_present` un-caps the mask; the `HashMap` lookup the check needs
lives inside the `debug_assert!` argument, so it does not exist in release.
Read: `bucket_topology.rs` (`horizon_cap_active`), `lp/builder/layout.rs`
(`build_transit_bucket_row_pos`), `lp/builder/columns.rs` (`fill_transit_bucket_columns`).
Pinned by the horizon-depth-cap regression (the last stage's active-lag cap
reaches zero, so no slot targets past the horizon), `build_transit_bucket_row_pos`'s
own consumption regression (that same cap sequence emitting correspondingly
fewer rows), a sub-stage-delay case's last-stage release, whose dropped
share surfaces as an uneven per-stage delivery split rather than a credited
one, and `transit_bucket_mask_covers_every_arc_deposit_depth_under_boundary`
(`bucket_topology.rs`), which asserts `per_stage_mask` dominates every
arc's deposit depth on both the parallel and chronological tables under
`boundary_present` and is strictly exceeded at some stage without it.

### Sub-contracts: mode-independent sizing, aggregation consistency, fixed delivery density

The bucket state stays a pure function of stage lengths, never of
`n_blks`/`block_mode`, only because each of the following holds:

- **Depth from stage lengths alone.** Bucket depth and `n_buckets` derive from
  the per-stage calendar, the declared post-study calendar, and the pre-study
  anchor alone (`DeliveryCalendar`, `build_transit_bucket_topology`) — never
  from `n_blks` or `block_mode`. Deriving any part of the depth inside a
  block-aware code path re-couples the state dimension to how a stage happens
  to be resolved.
- **Shared arrival density.** A chronological stage's per-block deposit shares
  `block_deposits`/`within_stage_routing` and the stage-level `stage_weights`
  come from the same shared arrival density (`resolve_spread`'s
  `stage_weights`/`block_deposits`/`within_stage_routing`,
  `resolve_block_factors`'s `BlockFactors`), so `Σ_b w_b·χ_{b,d} = k_d` holds
  by construction. Building `block_deposits`/`within_stage_routing` from one
  density and `stage_weights` from another lets the chronological and
  parallel cuts diverge and silently breaks conservation.
- **Fixed delivery density.** A maturing bucket delivers into its arrival
  stage's blocks through a fixed, `block_mode`-independent `arrival_density`
  looked up from the setup-precomputed per-`(arc, arrival stage)` table
  (`resolve_bucket_arrival_density` reading
  `TemplateBuildCtx::arc_arrival_density`, built by `build_arc_arrival_density`
  as a blend over every contributing source stage's lag, resolved in the
  ARRIVAL stage's own frame), never by tracking which origin block a unit came
  from. Tracking origin-to-arrival-block correlation would grow the bucket
  into a per-block vector whose length scales with the receiving stage's
  `n_blks` — re-violating the depth-from-stage-lengths property above.

Read: `lead_time/mod.rs` (`resolve_spread`'s
`block_deposits`/`within_stage_routing`/`arrival_density` fields,
`resolve_block_factors`'s `BlockFactors`, `resolve_arrival_density_at`),
`bucket_topology.rs` (`build_arc_arrival_density`), `lp/builder/entries.rs`
(`fill_chronological_water_entries`),
`lp/builder/delivery_ring.rs` (`resolve_bucket_arrival_density`). Pinned
by the shared-density-consistency regression exercising the aggregation
debug_assert directly, the chronological block-table regression matching the
worked kappa/chi numbers, and the `K = 1` chronological-vs-parallel
byte-identity regression; a state-dimension-equality regression across
parallel and chronological builds is the direct pin for mode-independent
sizing. The arrival-frame lookup regression (the resolved density equals the
precomputed `arc_arrival_density` table entry verbatim) is the direct pin for
the fixed-delivery-density clause itself; the parallel-fill regression (the
maturing bucket keeps a single `-1.0` regardless of the table's contents)
pins that `fill_parallel_water_entries` never reads it.

### Maturing transit into a `PreFilling` plant follows the short-circuit target

A plant `h` that is `PreFilling` at the stage its lag-1 bucket `b_1^in(h)`
matures (in a validated study, a stage at or after its `exit_stage_id`) puts
that bucket's REAL incoming column onto the rows of
`resolve_shortcircuit_target(h)`: `-1.0` on the target's row in `Parallel`,
`-arrival_density[k]` (`resolve_bucket_arrival_density` for `h`) on the
target's block rows in `Chronological`. `h`'s frozen identity row gets nothing.
With no non-`PreFilling` downstream (sink) nothing is routed and the water
leaves at the system outlet. Wrong-but-compiling alternatives: leaving the
column on no row loses the water and makes `b_1(h)`'s cut coefficient
structurally zero; routing onto the immediate `downstream(h)` corrupts that
plant's frozen row when it is itself `PreFilling`; a synthesized coefficient
instead of the real column breaks the `rc / col_scale` subgradient. The
`hydro_inflow` generic-constraint term mirrors the same route, counting the
routed bucket at `arrival_density[blk] / τ(blk)`.
Read: `lp/builder/entries.rs` (`push_maturing_bucket_coupling`,
`fill_prefilling_shortcircuit`), `lp/builder/hydro_state.rs`
(`resolve_shortcircuit_target`), `lp/builder/generic_constraints.rs`
(`resolve_hydro_inflow`, `push_maturing_bucket_rate`). Pinned by
`prefilling_plants_maturing_bucket_lands_on_the_short_circuit_target_row`,
`prefilling_plants_maturing_bucket_spreads_by_arrival_density_on_the_target_block_rows`
and `prefilling_plants_maturing_bucket_at_a_sink_lands_on_no_row` (entries.rs),
`exited_plant_transit_reaches_the_next_operating_plant_at_the_hand_derived_cost`
(hand-derived lower bound 2380 $ and a -10 $/hm³ bucket cut coefficient) and
`exited_plant_transit_conserves_every_released_hm3` (`filling_commissioning.rs`),
and `hydro_inflow_rows_count_an_exited_plants_maturing_transit_water_on_a_parallel_stage`
(`hydro_inflow_travel_time.rs`).

## Anticipated thermal commitments

### Pre-study anticipated commitments: calendar-derived coverage

`AnticipatedCommitmentHistory` (`cobre-core`) is a windowed record —
`{thermal_id, start_date, end_date, value_mw}`, one commitment window per
entry, mirroring `HydroPastDefluence`'s shape — never a per-stage array
indexed by delivery order. A plant's commitment windows must TILE EXACTLY its
pre-study-decided delivery set at coverage `1.0` — the calendar-derived
leading in-study stages and, when the plant's lead reaches past the horizon,
the post-horizon stages it ALSO decides before the study (classes 2 and 4 of
the delivery taxonomy) — with two named failure directions: a GAP (a
pre-study-decided stage left uncovered, in-study or post-horizon) and
OVER-COVERAGE (a window reaching a stage the study itself decides, class 3,
or one beyond the plant's decision reach, class 5); overlap between two
windows of the same plant is rejected earlier, at parse time, by the shared
windowed-record validator that also serves `past_defluences` and
`recent_observations`.

A single window may not STRADDLE the study horizon — `start_date < horizon_end
< end_date`, spanning the in-study prefix (class 2) and the post-horizon fixed
set (class 4) in one record — because the two are semantically distinct
(in-study deliveries mature in an LP stage; post-horizon ones never enter the
ring and are priced by the boundary fold), and the downstream class-4 selectors
key on `start_date >= horizon_end`, so a straddling record would be silently
dropped from the fold and the outputs. `check_no_straddling_commitment_window`
rejects it with a `BusinessRuleViolation` instructing the author to split the
coverage into two windows at `horizon_end`; the boundary cases (`end_date ==
horizon_end`, purely in-study; `start_date == horizon_end`, purely post-horizon)
are legal. This makes "no window straddles the horizon" an enforced precondition
the class-4 date selectors rely on, not an assumed one. Pinned by
`test_straddling_commitment_window_rejected` and
`test_horizon_split_commitment_window_pair_loads_cleanly`
(`crates/cobre-io/tests/post_study_stages.rs`).

The in-study half is calendar-derived, computed independently of the solver
crate's point-commitment resolver (`cobre-io` is upstream and cannot depend on
it): `LeadStages(l)` clamps to `min(l, n_stages)`; `LeadTime(delta)` counts the
leading study stages whose stage-end cumulative hours are `<= delta`
(tie-inclusive). `check_anticipated_thermals` resolves this count, then hands
the plant's windows to the shared `StageCalendar` resolver — the same
calendar walk `past_defluences` coverage uses — via `covers_exactly` (gap
detection over the leading count) and a per-stage `coverage` sum
(over-coverage detection beyond it); either failure hard-rejects as a
`BusinessRuleViolation`, no fallback. A count-only gate
(`records.len() == leading_stage_count`) is a plausible-looking alternative
that accepts the right NUMBER of windows while missing a leading stage and
duplicating another — silently mis-covering the plant — since only per-stage
tiling, not a count, proves every leading stage is covered exactly once.

The post-horizon half mirrors this pair exactly, against
`classify_deliveries`'s four-way partition of every post-study index
(`fixed_post_study` = class 4, `carried` = class 3, `beyond_reach` = class 5,
`commissioning_inactive`): `check_fixed_post_study_tiling` is V2, the gap
check (every `fixed_post_study` index tiled at coverage `1.0`, an explicit
`0 MW` window included); `check_post_study_window_excludes_unreachable_stages`
is V3, the over-coverage check (no window covers a `carried` or
`beyond_reach` index, reported as two distinct errors since the remedies are
opposite — lengthen the lead for a `carried` miss, shorten it for a
`beyond_reach` one). Together V2 and V3 make a plant's covered post-study
stages EXACTLY its `fixed_post_study` class. V4 (`check_committed_value_bounds`)
extends the SAME committed-value envelope check to post-horizon windows
exactly as to in-study ones — no separate post-horizon envelope rule.

A class-4 delivery — pre-study-decided, post-study-delivered — is
representable as a DECLARED CONSTANT that NEVER ENTERS THE RING once its
window tiles exactly: priced by the boundary intercept fold, reported at its
real delivery date, with no decision column, no ring slot, and no carry row.
The retired reject that used to fire on any window reaching past the horizon
is gone, but the scope boundary it protected is NOT: no pre-study decision is
EVER carried through the ring into the post-study — the fixed commitment
bypasses the ring precisely to keep that boundary. Carrying a class-4
delivery through the ring instead — the boundary that retired reject used to
enforce — is the wrong-but-compiling alternative this contract still
forbids.

The DECLARATION requirement above is commissioning-FILTERED — a
`commissioning_inactive` post-study stage needs NO declared window at all (V2
is vacuously satisfied there) — while the RING'S OWN excision is
decider-derived, owned by `fixed_post_horizon_width` (The ring axis subsection
above), a different axis entirely: excision keys on `decider == None` alone;
declaration keys on `fixed_post_study` (`decider == None` AND
commissioning-active). Using the commissioning-filtered declaration set as
the ring's excision source instead is the wrong-but-compiling alternative
this split forbids: it would skip
excising a commissioning-inactive class-4 stage (routed to
`commissioning_inactive`, never `fixed_post_study`), leaving it a ring member
the ring never gives a carrier to — exactly the corruption the
commissioning-blind, decider-derived excision exists to avoid. Commissioning
gates only WHETHER a window must be declared and WHETHER a non-zero value
there is a modelling error — never ring membership.

A non-zero fixed value covering a `commissioning_inactive` post-study stage
is rejected (`check_fixed_commitment_within_window`, V5); an explicit `0 MW`
window there stays legal. V5 shares its predicate with the in-study seed rule
(`check_seed_within_window`) but not its justification: in-study, a non-zero
value in a closed commissioning window is an LP FISHING-EQUALITY
INFEASIBILITY (the matured generation column is pinned `[0, 0]`, so
`0 == seed` is unsatisfiable); post-horizon there is no LP column to reject
it — the value would instead be SILENTLY FOLDED into the terminal-boundary
valuation and reported as a delivery from a plant not in service, a
mispriced output with no LP backstop at all.

Read: `crates/cobre-core/src/constraints/initial_conditions.rs`
(`AnticipatedCommitmentHistory`), `crates/cobre-io/src/validation/semantic/thermal.rs`
(`check_anticipated_thermals`, `lead_delivery_stage_count`,
`check_commitment_coverage`, `check_post_study_stages`, `classify_deliveries`,
`DeliveryClasses`, `check_fixed_post_study_tiling`,
`check_post_study_window_excludes_unreachable_stages`,
`check_committed_value_bounds`, `check_fixed_commitment_within_window`,
`check_seed_within_window`), `crates/cobre-stochastic/src/season_cast/mod.rs`
(`StageCalendar::covers_exactly`, `StageCalendar::coverage`).
Pinned by `test_anticipated_lead_time_coverage_pmo_calendar` and
`test_anticipated_lead_time_coverage_pmo_calendar_under_coverage_rejected`
(in-study coverage, `thermal.rs`); and, all in
`crates/cobre-io/tests/post_study_stages.rs`:
`test_fixed_post_horizon_windows_tiling_class_four_stages_loads` and
`test_untiled_fixed_post_horizon_stage_rejected` (V2),
`test_window_on_a_study_decided_post_study_stage_rejected`,
`test_window_beyond_the_decision_reach_rejected`, and
`test_window_confined_to_the_fixed_set_accepted` (V3),
`test_untiled_fixed_window_and_out_of_envelope_value_both_reported` (V4),
`test_nonzero_fixed_commitment_outside_commissioning_window_rejected`,
`test_zero_fixed_commitment_outside_commissioning_window_accepted`, and
`test_fixed_commitment_inside_commissioning_window_accepted` (V5),
`test_retired_no_carrier_advice_is_absent_from_every_diagnostic` and
`test_fixed_commitment_window_message_makes_no_infeasibility_claim` (the
converted class-4 record carries no reject/infeasibility language),
`test_v2_v3_v4_and_v5_violations_are_all_reported` (multi-violation), and
`test_reference_shaped_fixed_post_horizon_deck_loads` (a reference-shaped
deck loads clean).

### The ring axis: the delivery axis with the fixed post-horizon window excised

Every `m mod k_max` statement below is a RING-AXIS statement: indices are
ring-axis indices; the ring axis is the delivery axis with the fixed
post-horizon window excised; identity whenever no fixed window exists.
`PointResolution::ring_index(m)` maps a physical delivery target `m` to its
ring-axis index: `Some(m)` for `m < n_decision` (the study's decision-stage
count, `n_stages` elsewhere in this file — in-study, identity), `None` inside
the excised fixed post-horizon window `[n_decision, n_decision + g)` (never a
ring member), `Some(m − g)` above it (the class-3/5 tail). `physical_target`
is its left inverse (`r < n_decision ↦ r`, `r ↦ r + g` otherwise). `g` is the
leading `None`-run width at the post-study end of `decider`, derived on the
resolution alone via the private `fixed_post_horizon_width` helper
(`take_while(is_none).count()`) — it carries NO SEPARATE STATE, so
`ring_index` and `physical_target` cannot disagree about `g`. `g == 0` (no
declared fixed post-horizon window) collapses both maps to the identity — the
byte-neutrality anchor every existing deck relies on.

A parallel per-plant structure that caches the excision instead of deriving it
from `PointResolution` on demand is the forbidden alternative: it re-introduces
the foreign-index-space alignment bug the excision exists to remove, the
moment a second copy of `g` drifts from the resolution's own. `PointResolution`
is the single owner of both maps — no new type, no new state, never a second
copy.
Read: `lead_time/mod.rs` (`PointResolution::ring_index`, `physical_target`,
`fixed_post_horizon_width`). Pinned by
`ring_index_is_the_identity_without_a_post_study_none_run`,
`ring_index_excises_the_fixed_post_horizon_window`,
`physical_target_is_the_left_inverse_of_ring_index`, and
`ring_index_degrades_to_the_identity_on_a_short_decider`, all in
`lead_time/tests.rs`.

### Ring depth sizing: `k_max = max(occupancy_max, n_none_in_study)`

The ring depth is `k_max = max(occupancy_max, n_none_in_study)`, resolved in
ring-axis (excised) space and owned by `PointResolution::ring_depth`: the
global `k_max = max_i ring_depth_i` (`AnticipatedResolution::resolve`) and the
per-plant lead `k_i` (`StateSpace::anticipated_lead_stages`, the
`LeadTime` arm) both read it. The `LeadStages(l)` arm instead returns `l`
VERBATIM — the byte-identity anchor `n_none_in_study <= l` by construction
makes safe (`debug_assert!(ring_depth() <= l)`). Sizing from `occupancy_max`
alone is the wrong-but-plausible under-sizing alternative this closes — the
filed ring-depth under-sizing defect: `occupancy_max` SUBTRACTS the seed
maturing at stage 0, so it under-counts the one moment
every simultaneous pre-study seed is in flight (stage 0, before the first
fishing) — the last seeded stage then silently delivers an earlier stage's MW,
a silent-wrong-value bug that still compiles and still converges.
Read: `lead_time/mod.rs` (`PointResolution::ring_depth`,
`AnticipatedResolution::resolve`, `AnticipatedResolution::ring_size`),
`setup/mod.rs` (`resolve_anticipated_commitments_core`'s `LeadStages`/`LeadTime`
split).
Pinned by `ring_depth_covers_every_simultaneous_pre_study_seed`,
`ring_depth_equals_the_occupancy_max_when_no_seed_overflows`,
`ring_depth_ignores_post_study_none_deciders`, and
`resolve_sizes_k_max_from_the_deepest_plant_ring_depth`, all in
`lead_time/tests.rs`.

### In-LP anticipated ring: definition-row sign, hold carry & asymmetric masking

The in-study anticipated ring is `DeliveryRing`'s other instantiation (the shared
skeleton above), borrowing the LEADING `n_anticipated * k_max` sub-range of the
merged commitment-hold region: an outgoing block (`StateSpace::commit_out`,
identity-resolved by `state_to_lp_column`, contributing to `n_state`) and a
separate incoming block (`StateSpace::commit_in`, pinned via
`state_to_lp_incoming_column`) — never one dual-purpose range shifted out-of-LP.
There is no Rust-side shift step: the ring transition is resolved entirely by the
definition rows below, and `current_state`/`state_at_capture` read the outgoing
block by the same plain copy already used for storage and travel-time buckets.
Slots are keyed by DELIVERY-TARGET RESIDUE, not by distance to maturity: delivery
target `m`'s slot is `ring_index(m) mod k_max` — `m mod k_max` where the index
is already a ring-axis index (The ring axis subsection above) —
slot-major/plant-minor (`StateSpace::commitment_hold_in_study_offset(plant, m) =
(ring_index(m) mod k_max) * n_anticipated + plant`).

The interior transition is the same-slot HOLD identity, not a Markov-1 shift. An
in-flight, not-yet-due slot's outgoing column is pinned to its OWN incoming column,
`slot^out − slot^in = 0`, via `DeliveryRing::emit_carry_rows` (`+1` on
`out_col(slot, lane)`, `−1` on `in_col(slot, lane)` — the SAME slot), routed by
`fill_anticipated_slot_definition_entries`. This REPLACES the retired Markov-1
shift `slot_k^out − slot_{k+1}^in = 0` (`emit_shift_rows`, whose `−1` lands on the
NEXT slot): a commitment does not migrate slots stage-to-stage — it is held at its
delivery-target residue until it matures. The water travel-time ring keeps
`emit_shift_rows`, because its physics genuinely shift; only the anticipated family
carries. `build_anticipated_slot_row_pos` covers the ring window `{t+1 .. t+k_max}`
— the strictly-future, not-yet-due delivery targets, a contiguous run over which
the modular key `r mod k_max` (ring-axis `r`; The ring axis subsection above) is
injective (`modular_slot_key_is_injective_on_the_carried_in_flight_set`), so the
same-slot hold never collides two in-flight commitments onto one slot; the commitment
maturing THIS stage is always fished (see the always-fish contract below), never
carried here.

The deposit / latch pins a plant's fresh decision into the slot of its OWN delivery
target, `slot^out = decision_col` (`slot = ring_index(delivery_stage) mod k_max`
— The ring axis subsection above), via the
shared skeleton's deposit primitive (`DeliveryRing::emit_deposit`, `+1` on
`out_col(slot, lane)`, `−1` on `decision_col`), routed by
`fill_anticipated_state_out_def_entries`. Both row families render `[0, 0]` bounds
(`fill_anticipated_slot_definition_rows` / `fill_anticipated_state_out_def_rows`):
the `+1`/`−1` structural coefficients on each side do the carry/deposit, never the
bounds.

Masking is reachability masking over the ENTIRE ring — there is no separate
appended block with a rule of its own. Every position keeps the two-sided
reachability masking the shared skeleton always ships together: a masked position
(`build_anticipated_slot_row_pos`'s per-slot `None`) gets NO definition row (the
row-cap side) AND a frozen `[0, 0]` outgoing column
(`DeliveryRing::freeze_masked_columns`, the column-freeze side, over the open
signed `(-inf, inf)` reachable bound a committed MW value needs) in the SAME pass;
wiring only one side leaves either a dangling row on a frozen column or a free
column with no defining constraint, both wrong-but-compiling.
`fill_anticipated_slot_columns` applies `freeze_masked_columns` over the WHOLE
`anticipated_slot_row_pos`: a reachable post-study-targeted slot is open
`(-inf, inf)` because it is NOT masked (its `row_pos` is `Some`), never because it
is exempt from freezing, and the boundary FCF then prices its carried state
directly through the generic `β·state` projection (Delivery-family right-boundary
pricing above). The surviving masking asymmetry is anticipated-vs-water, stated at
the shared skeleton's "Masked terminal slot" contrast (an anticipated masked slot
never held a value), not a per-slot freeze exemption here. Treating any slot as
freeze-exempt — the retired appended-block rule — is the wrong-but-compiling
alternative: a masked (unreachable) slot left open `(-inf, inf)` with no definition
row is a free, undefined state column the projection still prices, an
out-of-nowhere commitment value.

The policy manifest resolves a ring column back to `(slot, plant)` via
`DeliveryRing::slot_lane_at` — the exact inverse of `out_col`/`in_col`, never a
hand-rolled `offset / n_anticipated`/`offset % n_anticipated` pair — and dates it
at its MODULAR delivery stage, reached through the RING-AXIS residue evaluated
at the OUTGOING anchor `t_out = current_stage_idx + 1` (the state leaving the
pool's own stage, never the entering `current_stage_idx`): the next ring-axis
target `r >= t_out` in the slot's residue class (`delta = (slot_idx + k_max −
t_out mod k_max) mod k_max`, `r = t_out + delta`), mapped to the physical
delivery stage `m = physical_target(r)` (The ring axis subsection above)
before dating. Three wrong-but-compiling alternatives: anchoring on the
entering `current_stage_idx` instead of `t_out` — the RETIRED form this
section itself described before the re-anchoring — dates the terminal pool's
own maturing-residue slot onto the terminal stage's own in-study month
instead of a full `k_max` stages past it, silently zeroing a real post-study
delivery; `t_out + slot_idx` (the older retired shift-ring form, wrong
whenever `t_out mod k_max != 0`); and dating the raw ring-axis `r` directly
instead of `physical_target(r)` — it lands on the excised fixed post-horizon
window's stub stage whenever a plant declares one. A ring slot is live at
the pool's stage `t` if and only if `for_each_live_commitment_slot` visits
it: its target lands in the window `{t+1 ..= t+k_max}`, inside the delivery
calendar, and is ready (`PointResolution::is_ready_at`). The retired
per-plant lead bound `slot_idx < k_i` is the wrong-but-compiling
alternative — under residue keying a short-lead plant cycles through every
residue, so a slot beyond its own lead can still be a live carry or
deposit — the multi-plant heterogeneous-lead case, where plants sharing one
`k_max`-wide ring have different reachable widths.
`build_stage_entity_manifest` applies this same rule before populating
`EntitySlot::interval_start`/`interval_end`, and `StateSpace::set_nonzero_mask`
applies it to the cut mask's `CommitmentHold` region as the union of this rule
over every decision stage, computed once at construction from the attached
resolution.

The sign / `col_scale` invariants are unchanged from storage and the water buckets:
the incoming column's reduced cost is DIVIDED by `col_scale` on extract
(`extract_duals_from_view`) and the outgoing column's cut coefficient is MULTIPLIED
back on render (`push_scaled_coefficient`); `col_scale` is forced to `1.0` across
the whole region (the reconcile contract below).

Read: `lp/indexer/state_space.rs` (`StateSpace::commit_out`,
`StateSpace::commit_in`, `commitment_hold_in_study_offset`, `state_to_lp_column`,
`state_to_lp_incoming_column`), `lp/builder/delivery_ring.rs`
(`DeliveryRing::emit_carry_rows`, `emit_deposit`, `freeze_masked_columns`,
`slot_lane_at`, `DeliveryRing::anticipated`), `lp/builder/entries.rs`
(`fill_anticipated_slot_definition_entries`,
`fill_anticipated_state_out_def_entries`), `lp/builder/rows.rs`
(`fill_anticipated_slot_definition_rows`, `fill_anticipated_state_out_def_rows`),
`lp/builder/layout.rs` (`build_anticipated_slot_row_pos`), `lp/builder/columns.rs`
(`fill_anticipated_slot_columns`), `policy/policy_export.rs`
(`build_stage_entity_manifest`). Pinned by the `state_to_lp_column`
`commit_out`-identity regressions
(`state_to_lp_column_commit_out_is_identity_no_lag`,
`state_to_lp_column_commit_out_identity_multi_plant_heterogeneous_k`), the
carry-vs-shift and masking primitives
(`emit_carry_rows_targets_the_same_slot_where_emit_shift_rows_targets_the_next`,
`emit_carry_rows_masked_position_emits_no_row`,
`freeze_masked_columns_masks_identically_across_reachable_bound`), the open-coded
carry-formula regression
(`fill_anticipated_slot_definition_entries_matches_open_coded_carry_formula_across_heterogeneous_plants`),
the backward-cut coefficient-propagation regressions
(`two_stage_k1_anticipated_cut_coefficient_matches_analytical`,
`three_stage_k2_anticipated_cut_coefficient_propagates_correctly`,
`four_stage_k3_anticipated_cut_coefficient_propagates_correctly`), and the
manifest delivery-anchor regressions
(`anticipated_slot_delivery_anchor_matches_delivery_stage_year_month`,
`anticipated_slot_delivery_anchor_past_horizon_is_sentinel`,
`anticipated_short_lead_slot_dates_the_residue_it_latches`), the
outgoing-anchor re-anchoring regressions
(`terminal_maturing_residue_dates_onto_its_post_study_delivery`,
`terminal_maturing_residue_stays_sentinel_without_a_post_study_calendar`,
`anticipated_slot_date_matches_the_resolved_physical_delivery_stage`,
`anticipated_slots_the_lp_does_not_latch_stay_sentinel`), and the
mixed-lead reachability regressions
(`mixed_lead_nonzero_mask_covers_every_slot_the_lp_latches`,
`mixed_lead_manifest_dates_exactly_the_slots_the_lp_latches`). The ring-axis
excision itself is additionally pinned by the collision/identity regressions
`excision_keeps_each_study_stage_fishing_its_own_seed`,
`zero_gap_with_post_study_resolves_an_identity_ring_and_occupancy_depth`, and
`zero_gap_carry_slot_addressing_matches_the_open_coded_identity_formula`, all
in `tests/anticipated_core.rs`, and the excised-space `physical_target`/`ring_index`
contract, pinned by `ring_index_excises_the_fixed_post_horizon_window` and
`physical_target_is_the_left_inverse_of_ring_index` (`lead_time/tests.rs`).

### In-study maturity always fishes; carry-to-terminal is the post-study-targeted ring slot's alone

The in-study maturity arm ALWAYS fishes. For every in-study delivery maturing this
stage (`build_anticipated_fishing_row_pos`'s `Some`, driven by
`PointResolution::is_anticipated_at`; `None` only at a `K = 0` self-delivery),
`fill_anticipated_fishing_entries` emits the must-generate coupling
UNCONDITIONALLY — active OR commissioning-inactive alike: `+h_b` (block hours) on
each of the plant's per-block thermal generation columns and `−H` (the stage's
total hours) on the maturing slot's INCOMING column `in_col(ring_index(stage_idx)
mod k_max)` — identity here (The ring axis subsection above): a delivery matures
only at its own in-study stage index, always `< n_decision`.
It reads `commit_in` and NEVER writes `commit_out`. A commissioning-inactive
delivery was never latched — its decision column stays dormant `[0, 0]`
(`fill_anticipated_columns`) — so its `in_col` carries `0` and this equality pins
that stage's thermal generation to `0`, the correct, harmless outcome for a
delivery the plant's window cannot receive.

The wrong-but-compiling alternative is a two-way
`fish-iff-commissioning-active-else-carry` branch. Because fishing reads only
`in_col` while a carry WRITES `out_col`, the carry arm's `out_col` write collides
with the SAME stage's fresh delivery latch on that slot whenever a future-entry
plant's pre-entry ramp shares the maturing slot's modular residue (the case a
plant's own lead defines `k_max`, so no other plant reaches deeper): two definition
rows on one `out_col` pin a freshly-costed decision to a stale carried value, a
release-silent LP corruption surfacing as a false `Infeasible` or a silent
zero-commit (the guarding `debug_assert` is compiled out of release).
Carry-to-terminal is owned SOLELY by the ring's interior-carry rows
(`DeliveryRing::emit_carry_rows`, routed by `fill_anticipated_slot_definition_entries`),
never the maturity arm. The always-fish `+h_b`/`−H` coefficient shape is exactly
the pre-migration one; only its slot addressing is modular
(`ring_index(stage_idx) mod k_max`, identity for an in-study index — The ring
axis subsection above — via `commitment_hold_in_study_offset`).

Read: `lp/builder/entries.rs` (`fill_anticipated_fishing_entries`,
`fill_anticipated_slot_definition_entries`), `lp/builder/layout.rs`
(`build_anticipated_fishing_row_pos`), `lp/builder/columns.rs`
(`fill_anticipated_columns`). Pinned by `fishing_rows_always_active_stage_zero`
(every plant gets a fishing row regardless of activity, coupling on the maturing
slot's `commit_in` column at `−H`), `fishing_rows_fill_all_plants`,
`anticipated_commissioning_window_gates_simulation_output` (a
commissioning-inactive delivery), and
`simulation_commitment_hold_carries_anticipated_state_k2` (the maturing seed is
fished, not carried, across the pre-horizon stages).

### End-of-horizon masking is exact, never a dropped commitment

Unlike the water ring's Terminal credit deferred subsection, no anticipated
commitment — in-study or post-study-targeted — is ever discarded at the
delivery-axis boundary; none is created past it in the first place.
`is_anticipated_decision_active_for_delivery` gates a decision column's
existence on the strict clause `stage_idx + K_i < n_delivery`, against the
EXTENDED delivery calendar (`n_delivery = StateSpace::n_delivery`, the
study stages plus the `post_study_stages.json` continuation), not merely
`n_stages`; `PointResolution::decider` has the matching domain
`m in [0, n_delivery)`, so no code path ever computes a commitment targeting a
delivery past the extended axis and then truncates it. A post-study-targeted
delivery (`m` in `[n_stages, n_delivery)`) is CREATED and rides the ring, priced
through the boundary FCF — masking is exact at `n_delivery`, not `n_stages`.
`build_anticipated_slot_row_pos`'s per-slot `None` (no carry row) and
`fill_anticipated_slot_columns`'s frozen `[0, 0]` outgoing column, at a slot
whose ring-axis target `r = stage_idx + depth + 1` (`depth in 0..k_max`) maps,
through `physical_target` (The ring axis subsection above), to a physical
delivery target `m >= n_delivery`, are therefore always vacuous: the masked
slot is provably zero for every valid configuration, never a real value the
model declines to
route anywhere. A commissioning-inactive in-study delivery is likewise pinned to
`0` — not by masking but by the always-fish arm reading a dormant slot's
`in_col` of `0` (the always-fish contract above) — so it too loses nothing of
value. This differs in kind from water's masking: a masked bucket discards a
genuine non-zero `k_d`-weighted release share deposited every stage regardless
of the arc's travel time — an admitted target-stage imprecision — while the
anticipated gate prevents the decision from ever existing. Crediting a masked
slot as if it held a dropped commitment would introduce value the model never
computed, for a delivery target past the extended delivery axis `n_delivery`.
Read: `lp/indexer/anticipated_gate.rs`
(`is_anticipated_decision_active_for_delivery`), `lead_time/mod.rs`
(`PointResolution::decider`), `lp/indexer/state_space.rs`
(`StateSpace::n_delivery`), `lp/builder/layout.rs`
(`build_anticipated_slot_row_pos`), `lp/builder/columns.rs`
(`fill_anticipated_slot_columns`). Pinned by
`is_anticipated_decision_active_for_delivery_strict_extended_bound` (the strict
`< n_delivery` gate on the extended axis; `<=` would admit a delivery at
`n_delivery`) and `a1c_lead_stages_is_pure_index_shift`'s
empty-`decision_sets`-past-the-delivery-bound assertion.

### In-LP anticipated ring: single-decider deposit & `K = 0` exclusion

Each anticipated plant gets AT MOST ONE decision column per stage
(`col_anticipated_decision_start + local_idx`), driven by
`PointResolution::genuine_decisions_at(stage_idx).next()` (a `K = 0`
self-delivery already excluded — see below). That decision deposits into its
OWN ring slot, `slot = ring_index(delivery_stage) mod k_max` (The ring axis
subsection above) — computed DIRECTLY from the decision's own delivery stage
(`fill_anticipated_state_out_def_entries`), never from a `depth`-derived
boundary.

**`depth[t]` is not the ring's per-stage occupancy boundary.** `depth[t]`
(`PointResolution::depth`) counts only IN-STUDY decided items still in flight
— `build_decision_sets_and_depth`'s sweep adds a delta only for `Some(t)`
deciders, structurally excluding pre-study (`None`, IC-seeded) occupancy. A
plant can have BOTH an IC-seeded item and a fresh in-study decision occupying
the ring at the same stage (e.g. a constant-lead plant's stage 0), so
`depth[t] − genuine_count(t)` under-counts and mis-targets the slot — the
wrong-but-plausible shortcut `PointResolution::is_ready_at`'s doc comment
warns against. The correct interior/deposit/padding split is checked PER
DELIVERY TARGET directly (`build_anticipated_slot_row_pos`): for each
ring-axis target `r = stage_idx + depth + 1` (`depth in 0..k_max`), ring slot
`r mod k_max` — `ring_index(m) mod k_max` under the physical delivery target
`m = physical_target(r)` (The ring axis subsection above) — is a deposit iff
`decider[m] == Some(stage_idx)`, an interior carry iff `is_ready_at(m,
stage_idx)` and not a deposit, else padding or past-horizon.
`decider` is nondecreasing in `m`, so readiness is monotonic and the ready
delivery targets form a contiguous prefix — the property that makes the
per-target check well-founded without needing an aggregate boundary.

**`K = 0` (sub-stage lead, `c(m) = m`) is excluded from the ring entirely —
exclude-with-advisory, never a hard error, never an underflow.** A
delivery whose physical lead is shorter than its own stage's duration is
decided inside its own delivery stage; `PointResolution::self_delivered_stages`
identifies these, and `genuine_decisions_at`/`is_anticipated_at` filter them
out of the decision and fishing gates respectively — the plant's ordinary
thermal generation column is priced and bounded normally (no fishing
coupling, no anticipated row at all) at that stage. A setup-time
`tracing::warn!` (`setup::warn_on_sub_stage_lead`, the same channel
`StudyParams::from_config`'s budget advisory uses) names the plant, the
stage, and the `lead_stages == 0` alternative — never emitted per-scenario or
per-trajectory.

The single-decider deposit is TODAY's fill; making a coarse decision stage
anchor several delivery stages (`|genuine C(t)| > 1`, fan-out) is the deferred
multi-decider capability the fan-out contract below reserves.

Read: `lead_time/mod.rs` (`PointResolution::genuine_decisions_at`,
`self_delivered_stages`, `is_anticipated_at`, `is_ready_at`, `depth`),
`lp/indexer/anticipated_gate.rs`
(`is_anticipated_decision_active_for_delivery`,
`anticipated_resolution_for`), `lp/builder/layout.rs`
(`build_anticipated_slot_row_pos`, `build_anticipated_decision_row_pos`,
`build_anticipated_fishing_row_pos`), `lp/builder/columns.rs`
(`fill_anticipated_columns`), `lp/builder/entries.rs`
(`fill_anticipated_state_out_def_entries`, `fill_anticipated_fishing_entries`),
`setup/mod.rs` (`warn_on_sub_stage_lead`). Pinned by
`k0_sub_stage_lead_emits_no_anticipated_rows_or_fishing_coupling` (no
anticipated slot/row/fishing coupling at any stage, one advisory per
self-delivered stage) and `five_stage_k2_anticipated_state_ring_buffer_evolution`
(the modular deposit/carry occupancy across stages).

### Fan-out is representable, but the LP fill retains its setup-time reject

The hold family MAKES fan-out representable: the ring holds N independent fixed
slots keyed by delivery-target residue, and the modular key `m mod k_max` is a
bijection on the in-flight set regardless of fan-out — several deliveries
anchored to one coarse decision stage occupy distinct residues, so there is no
slot collision and no extra state sizing. What is NOT yet built is the LP FILL:
every anticipated plant still gets at most one decision column per stage
(`PointResolution::genuine_decisions_at(stage_idx).next()`, the single-decider
contract above), so a `LeadTime` plant whose resolution would fan out
(`|genuine C(t)| > 1` at any decision stage) has no way to deposit its several
decisions. `resolve_state_layout` therefore RETAINS the reject — it fails any
`AnticipatedResolution::max_fanout > 1` configuration with
`SddpError::Validation`, naming the fanning plant (`first_fanned_plant_id`)
before a study's stage templates exist. This is the SOLE fan-out guard: a
reserved-capability gate pending the deferred multi-decider fill, NOT a
belt-and-braces check backed by column/entry/row-position handling that no
longer exists.
Read: `setup/mod.rs` (`resolve_state_layout`, `first_fanned_plant_id`). Pinned
by `lead_time_fanout_rejected_at_setup` (asserts `SddpError::Validation`, not a
panic, after confirming the fixture genuinely fans out).

### Delivery-anchoring preservation

Every anticipated plant's decision column is bounded, costed, and
commissioning-gated at ITS OWN delivery stage `m` (its
`genuine_decisions_at(t)` target, when one exists), never the decision stage
`t`. `fill_anticipated_columns` reads `thermal_block_base(thermal_idx,
delivery_stage)` for the column's `[min, max]` bounds (the overlay-ignoring
base is safe here only because a load-time rule rejects a `block_id` bound row
on an anticipated thermal — see `cobre-io`'s
`check_block_id_on_anticipated_thermal`),
`thermal_bounds(thermal_idx, delivery_stage).cost_per_mwh` for its cost,
`TimeValue::delivery_total_hours(delivery_stage)` for its hours,
`TimeValue::relative_delivery_discount(stage_idx, delivery_stage)` (the delivery
stage's cumulative discount over the decision stage's, `D(m)/D(t)`, one
division) for its discount, so the objective `cost * hours * D(m)/D(t)` is in
stage-`t` units like every other stage-`t` cost,
and `is_anticipated_decision_active_for_delivery` (the plant's window at
`delivery_stage`) for its dormancy — each read at the plant's own genuine
delivery stage (the discount relative to the decision stage), never at
`stage_idx` alone. Pricing with the absolute `D(m)` discounts a decision taken
after stage 0 twice (once in its own coefficient and once through the
discounted future cost) and still compiles, since the two agree at stage 0
(`D(0) = 1`). The delivered commitment is a hard equality with
no slack (the fishing coupling pins the plant's delivery-stage generation to
the committed value), so relatively-complete recourse requires the committed
value always lie within the delivery stage's own generation bounds. A
DECISION-anchored read (`thermal_block_base(thermal_idx, stage_idx)`) is the
forbidden alternative: it
reintroduces the capacity-drop infeasibility — a commitment placed under the
decision stage's larger capacity that no scenario can deliver under the delivery
stage's smaller one, stranded with no feasibility cut to absorb it — and still
compiles, since constant-across-lead bounds make the two reads indistinguishable.

Residual audit complete: no mechanism other than `thermal_block_base` can
strand a delivered commitment. The only generic-constraint handle on an anticipated plant,
`VariableRef::AnticipatedDecision` (`resolve_anticipated_decision`), binds the
fresh decision column at its own decision stage (the recourse variable, already
delivery-anchored here), never an in-flight matured commitment (no `VariableRef`
targets the ring state slots) nor the delivery-stage generation; constraining it
cannot strand a delivered value. The one path that touches the delivery-stage
generation, `VariableRef::ThermalGeneration` on an anticipated plant, is already
surfaced by `warn_thermal_generation_on_anticipated_thermal` and is the general
"a hard generic constraint may be infeasible" class, not an anticipated-specific
hole.

The StateBox commitment-slot box is the SECOND reader of this same
delivery-anchored base: `fill_commitment_hold_box` (`lp/builder/state_box.rs`)
resolves each reachable hold slot's box from the SAME
`thermal_block_base(delivery_stage)` the decision column reads, and depends on the
SAME `check_block_id_on_anticipated_thermal` load-time rule for the
overlay-ignoring base read's safety. They are one delivery-anchored dependency
with two readers — and the box the read-back seam clamps onto IS this box — so a
change to the delivery-stage bound source, or to the block-id rule that makes the
overlay-ignoring base read safe, must update BOTH readers; updating only one
prices a commitment against a different bound than its own box permits.

Read: `lp/builder/columns.rs` (`fill_anticipated_columns`),
`time_value.rs` (`TimeValue::relative_delivery_discount`),
`lp/builder/template.rs` (`finalize_stage_objective`, which divides the stage-`t` price by the cost scale and gives θ the one-step factor that carries it to the root),
`lp/builder/state_box.rs` (`fill_commitment_hold_box`, the box reader of the same
delivery-anchored base),
`lp/indexer/anticipated_gate.rs` (`is_anticipated_decision_active_for_delivery`),
`lp/builder/generic_constraints.rs` (`resolve_anticipated_decision`),
`cobre-io` `validation/semantic/thermal.rs`
(`warn_thermal_generation_on_anticipated_thermal`), `cobre-io`
`validation/semantic/block_bounds.rs`
(`check_block_id_on_anticipated_thermal`, the rule the base read's safety
depends on). Pinned by
`test_anticipated_decision_after_stage_zero_is_priced_relative_to_its_own_stage`
(a decision after stage 0 priced relative to its own stage),
`test_anticipated_decision_delivery_anchored_bounds` (stage-varying delivery
bounds/cost, mutation-verified against the decision-anchored read), the
end-to-end
`a1b_lead_time_equals_lead_stages_uniform_calendar` (the same
decision-anchored mutation turns the forward solve infeasible; pinned by
training and simulating both `LeadTime` and `LeadStages` configurations of
the same calendar to bit-identical solutions), and
`a1c_lead_stages_is_pure_index_shift` (pins the delivery-anchored decider
`c(m) = m - lead` those bounds are read against).

### Delivered commitments reconcile against solver drift; exactness is unreachable

Delivery-anchoring keeps the committed value inside the delivery stage's
generation bounds **in exact arithmetic only**. The value that actually reaches
the delivery stage is the solver's computed value for a **basic** ring-slot
column: `slot_out` is defined by an equality row (`slot_out − decision = 0`, or
the interior carry), so the simplex produces it through the basis factorization,
and it is accurate only to the backend's `primal_feasibility_tolerance` (`1e-9`
on HiGHS and CLP) — never to 1 ULP. A commitment at its cap therefore arrives a
hair outside it, and the fishing equality's no-slack pin turns that hair into
`SddpError::Infeasible`: a false infeasibility over a physically meaningless
quantity that would abort training outright if the state were pinned raw.

That sub-tolerance drift is absorbed at the outgoing-state read-back seam, not
judged on the solve path: `assemble_outgoing_state` projects every outgoing state
onto its admissible box before the value is pinned, solved against, or dotted
into a cut, so the commitment reaches the delivery stage already inside its
bound; the clamp absorbs the drift silently — no runtime verdict, no telemetry.
The `commit_out ∪ commit_in` carry is additionally made bit-exact by
`apply_commitment_hold_col_scale_unscale` (`col_scale = 1.0`), removing the
ring-carry drift at its source; the basis-factorization drift at the deposit row
is what the seam absorbs, since exactness there is the solver's to give and it
does not give it.

A *genuine* over-commitment — past the delivery bound by more than solver noise —
is NOT a runtime verdict and NOT the seam's to catch: it is rejected before the
study runs, at `cobre-io` load time, by `check_committed_value_bounds`
(`validation/semantic/thermal.rs`), which checks each committed value against the
delivery-stage resolved generation box. The seam absorbs solver noise; the
load-time validator rejects the modelling error. Letting the seam decide which
overshoot is "genuine" is the wrong-but-compiling alternative this split forbids:
the clamp is applied unconditionally to every solve site — forward, backward,
lower bound, and simulation each canonicalize through `assemble_outgoing_state`,
a per-site opt-out being what once let solve sites silently diverge — and it never
distinguishes noise from error, because the validator has already rejected any
genuine over-commitment at load.

Read: `solve/stage_solve.rs` (`assemble_outgoing_state`, the read-back seam),
`lp/builder/scaling.rs` (`apply_commitment_hold_col_scale_unscale`), and `cobre-io`
`validation/semantic/thermal.rs` (`check_committed_value_bounds`, the load-time
over-commitment reject). Pinned for the seam by
`anticipated_commitment_drifted_over_cap_is_absorbed` (a seed a hair past the cap
trains to completion rather than aborting, `tests/anticipated_scenarios.rs`), and
for the load-time validator by `delivery_box_sub_tolerance_drift_accepted`
(sub-tolerance accepted) and `delivery_box_override_tightens_max_rejects_over_commitment`
plus `test_committed_value_above_max_bounds_error` (genuine over-commitment
rejected). `anticipated_commitment_at_cap_survives_ring_carry` does NOT pin the
seam: a seed exactly at the cap carries zero drift and never exercises the
absorption.

### Post-study delivery without a boundary carries zero value, never a reject

The anticipated analog of water's `t_v > horizon` seed boundary (Delivery-family
right-boundary pricing above): a plant whose decider anchors an in-study decision
to a delivery target `m >= n_stages` — a post-study-targeted delivery — in a study
that declares NO `config.policy.boundary` carries ZERO terminal value. The ring
slot still exists and joins the `β·state` projection, but the terminal boundary FCF
it would price against is empty, so `β·state` contributes nothing. A class-4
fixed post-horizon commitment is a different object this contract does not
cover: never an in-study decision, excised from the ring entirely (The ring
axis subsection above), and priced — when a boundary is declared — by the
boundary intercept fold on the raw cut intercept, never by `β·state` (the fold
clause in the fan-out reconciliation subsection below). This is a
ratified scope boundary, NOT a reject: a `min_mw == max_mw` replay deck is a
legitimate use of a fixed post-study profile with no boundary, and rejecting it
would abort a valid study. Setup emits exactly ONE advisory naming every affected
plant (`setup::warn_on_boundary_absent_post_study_delivery`, once at setup on the
same `tracing::warn!` channel as the `K = 0` advisory, but a DISTINCT condition —
`warn_on_sub_stage_lead` fires on a sub-stage lead resolving `c(m) = m`, this one on
a post-study target with no boundary; do not conflate them). Hard-rejecting instead
of warning is the wrong-but-compiling alternative: it turns a legitimate
no-boundary replay deck into a spurious setup failure. Silence is the opposite
failure — it hides a modelling error where the user expected the commitment valued
against a real future.
Read: `setup/mod.rs` (`warn_on_boundary_absent_post_study_delivery`,
`warn_on_sub_stage_lead`). Pinned by
`warn_on_boundary_absent_post_study_delivery_fires_once_when_boundary_absent` and
`warn_on_boundary_absent_post_study_delivery_silent_when_boundary_present`
(`setup/tests.rs`).
