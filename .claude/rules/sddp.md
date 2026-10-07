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

The inflow-noise patch touches only hydro `h`'s z-inflow row. Every
water-balance row instead reads the deterministic column `z_h` through
`push_z_inflow_coupling`: at `−ζ` on a parallel stage, and at `−τ_k` (block
`k`'s duration hours times `M3S_TO_HM3`) on each chronological block row — on
the hydro's own water-balance row(s), or its `PreFilling` short-circuit
target's. The water rows themselves carry no lag, base, or patch of their
own; re-encoding the inflow there routes a chronological hydro's noise onto
another hydro's block row.

Read: `lp/builder/entries.rs` (`push_z_inflow_coupling`), `stochastic/noise.rs`
(`transform_inflow_noise`). Pinned by
`chronological_inflow_noise_moves_only_its_own_hydro`
(`tests/chronological_inflow_noise.rs`) and
`every_noise_dimension_patches_only_its_own_entity`
(`tests/patch_ownership_sweep.rs`), whose inflow ownership set is the z row
alone.

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

Drop the plant term (`min` degenerates to `sum`) and every lowering
`hydro_bounds` override in a study is silently discarded: a mid-horizon
capacity cut, declared exactly the way the no-raising rule's own rejection
message prescribes (declare the plant at final capacity, tighten the earlier
stages with override rows), stops reaching the LP the moment the plant
declares more than one group.

Drop the group term (`min` degenerates to `hb...`) and a multi-cell plant can
turbine or generate past its own declared capacity, because
`cell_max_turbined`/`cell_max_generation` are the ONLY consumers of
`hb.max_turbined_m3s`/`hb.max_generation_mw` in the hydro LP path — no
plant-level aggregate-max row exists to catch the overshoot. Three
independent, fully-valid-input mechanisms reach the group term, the third
strictly inside a single cell:

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

### The per-cell floor is a plain sum, never a fold or a plant clamp

`cell_min_turbined`/`cell_min_generation` (`lp/builder/columns.rs`) are the
MIN-side mirror of `cell_max_turbined`/`cell_max_generation` above, and
deliberately do NOT mirror their shape: a cell's min-turbine/min-generation
floor is `Σ_{g∈cell} resolved_min_*(g)` — the cell's OWN member groups' resolved
minima, summed, with no fold and no closing `.min(plant)` term at all. This is
the correct shape because the floor bounds a SUM of variables: the cell's
member groups all feed one shared aggregate column (one turbine column, one
FPHA-generation column, or the same column read at ρ for
`ConstantProductivity`), so each group's own mandatory minimum adds to the
others' — it is not one quantity two caps compete to bound tighter, which is
what licenses a `min` fold on the MAX side.

Four wrong-but-compiling alternatives, each of which has a close MAX-side
cousin that makes it easy to copy over by habit:

- **`.min(plant)` clamp** — closing the sum with `.min(hydro.min_turbined_m3s /
min_generation_mw)`, copying `cell_max_turbined`'s closing term verbatim.
  Wrong: the plant's declared minimum has no role in the per-cell floor at
  all (validation rule 44 checks it can be REACHED by the groups' sum, but the
  LP itself never reads it here). A `.min(plant)` clamp silently loosens the
  floor whenever the plant's own declared minimum is lower than a cell's
  group-sum, and clamps to that plant value uniformly on every cell —
  understating the true floor without invalidating it that the group data
  independently supports.
- **`max`-fold over a cell's own groups** — `max_{g∈cell} min_*(g)` instead of
  `Σ_{g∈cell} min_*(g)`. Wrong for the reason above: each group's own mandatory
  floor adds to the cell's aggregate minimum, so `max` understates a
  multi-group cell's true floor by every group's contribution except the
  largest.
- **ρ-folding the generation floor** — computing `cell_min_generation` by
  folding each group's turbined-derived floor through `ConstantProductivity`'s
  ρ (mirroring the MAX-side fold that combines two independent caps into one
  tighter one) instead of summing `group.min_generation_mw` directly. The
  min-generation row's LHS already carries `ρ * q_c` for a
  `ConstantProductivity` cell (`fill_operational_violation_entries`); folding
  ρ into the RHS too double-prices the productivity and produces a floor with
  no physical meaning.
- **`1/|cells|` price or RHS apportionment** — dividing the penalty
  (`turbined_violation_below_cost`/`generation_violation_below_cost`, both
  plant-level `HydroPenalties`) or the RHS by the plant's cell count before
  applying it per cell. The penalty is priced at FULL magnitude on every one
  of the plant's cells (mirroring how an arc's `k_d` release-weight replicates
  onto every cell's turbine column, never apportioned — see the water
  travel-time section below); apportioning it discounts the true cost of a
  multi-cell plant's violation by `|cells|`, and apportioning the RHS instead
  of summing it produces a floor with no basis in either the cell's own
  groups or the plant's declared value.

Two per-cell soft rows exist per the same design that governs the MAX-side
columns — never folded into one: `min_turbine_rows` couples the cell's own
turbine column (`+1`) to the cell's own `turbine_below_slack` (`+1`);
`min_generation_rows` couples the cell's own generation column — the cell's
FPHA-generation column (`+1`) or the cell's turbine column at `ρ` for
`ConstantProductivity` — to the cell's own `generation_below_slack` (`+1`).
Both families are sized `n_cells * n_blks` (`OperViolationRanges::new`'s
`n_op_cell` parameter), never `n_h * n_blks` — the two flow families
(min/max-outflow) stay hydro-keyed at `n_op_hydro`, since outflow has no
per-cell column to attribute to.

Output stays plant-keyed: `simulation/hydros/**.parquet`'s
`turbined_slack_m3s`/`generation_slack_mw` columns are unchanged in shape —
extraction sums a plant's own cells' slack columns into the existing
plant-level field (`sum_cell_slack` in `simulation/extraction.rs`), the same
pattern `turbined_m3s`/`generation_mw` already use for the max-side columns.
This differs from the same-bus generation-split problem the output-axis
decision (§7.10 of the blocks-and-units design) rules out: that problem is
genuinely UNDETERMINED (several same-bus groups sharing one column have no
basis to split by), while a per-cell floor VIOLATION is DETERMINED — each
cell owns its own row and its own slack column, so there is exactly one
correct per-cell slack value to sum, never a manufactured one.

Read: `crates/cobre-sddp/src/lp/builder/hydro_state.rs` (`cell_min_turbined`,
`cell_min_generation`), `crates/cobre-sddp/src/lp/builder/columns.rs`
(`fill_cell_block_family`), `crates/cobre-sddp/src/lp/builder/rows.rs`
(`fill_operational_violation_rows`), `crates/cobre-sddp/src/lp/builder/entries.rs`
(`fill_operational_violation_entries`), `crates/cobre-sddp/src/lp/builder/layout.rs`
(`OperViolationRanges`), `crates/cobre-sddp/src/simulation/extraction.rs`
(`sum_cell_slack`, `hydro_operational_slacks`), `crates/cobre-io/src/validation/semantic/hydro.rs`
(rule 44). Pinned by the per-cell analytical row/coefficient test in
`crates/cobre-sddp/src/lp/builder/entries.rs` (mutation-verified against the
`.min(plant)` clamp, the `max`-fold, the ρ-fold, and `1/|cells|`
apportionment), the d53 binding fixture, and the group-declaration-order
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

**Append-only within a pool.** Cuts are never removed from the LP. Deactivation
toggles a cut row's RHS bounds to the `±f64::INFINITY` sentinel (trivially
satisfied); every cut keeps a stable slot index for the lifetime of the run,
placed by `slot_index`'s deterministic function of `warm_start_count`,
`iteration`, `iteration_base`, `visit_stride`, and `forward_pass_index`. The
per-iteration template refreeze encodes **only active cuts** (one row per
`active_cuts()` entry), not inactive cuts at sentinel bounds.

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
is fixed once primed — a leaf never gains or loses a cut — so the template's
shape never changes across iterations. There the slot-identity reconstruction
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
node-tag and shape guards each drop to cold), all in `solve/stage_solve.rs`.

### A stored basis records its own cut-row count

Each checkpoint `StageBasis` record carries `num_cut_rows =
row_status.len() - base_row_count`, the trailing cut rows of the captured basis
itself. Writing the node's pool count (`populated()`) instead is the
wrong-but-compiling alternative: it overstates the basis's cut rows, because a
forward-pass capture precedes that iteration's backward pass, which appends cuts
to the pool afterwards. `FORMAT_VERSION` 3 marks this meaning. The loader still
takes the template row count from the current LP (`node_dims`), never from the
record. Read: `policy/policy_export.rs` (`build_stage_basis_records`). Pinned by
`build_stage_basis_records_writes_each_basis_own_trailing_cut_row_count`
(`policy/policy_export.rs`) and
`exported_basis_records_count_the_cut_rows_each_basis_was_captured_with`
(`tests/cut_basis.rs`).

The load uses a stored basis for its node only when it fits that node's LP
exactly. `admit_stored_basis` checks, in order, `found_cols == template_cols`,
`found_rows == template_rows + num_cut_rows`, and a basic count (column and row
statuses decoding to `Basic`) equal to `found_rows`. A record that fails is not
used for its node and the load proceeds: its slot stays `None`, and
`UnusedStoredBases` owns the one aggregated warning, `stored bases not used: …`.
The basic count is checked because `reconstruct_basis` assumes it and a deficit
aborts the run with `BasisShapeMismatch` (`cut/basis_reconstruct.rs`, "Basic-count
invariant"); `reconstruct_basis` is never called for a dropped node, so the
append-only pool and slot identity are untouched. The wrong-but-compiling
alternatives: refusing the load over a basis, which only warm-starts a solve;
admitting on shape alone, which lets a basic-count deficit reach
`reconstruct_basis`; and adding a cold retry after `reconstruct_basis` fails,
which hides a record this rule should have dropped. Read: `policy/policy_load.rs`
(`build_basis_cache_for_nodes`, `admit_stored_basis`, `StoredBasisLoad`),
`policy/full_fcf_load.rs` (`check_full_fcf_load`). Pinned by
`stored_basis_whose_basic_count_differs_from_its_rows_is_dropped` and
`unused_stored_bases_report_is_independent_of_record_order`
(`policy/policy_load.rs`),
`warm_start_skips_a_stored_basis_with_too_few_basic_entries_instead_of_aborting`
(`tests/cut_basis.rs`),
`simulation_only_loads_a_policy_with_a_wider_stored_basis_and_warns`
(`crates/cobre-cli/tests/cli_run.rs`) and
`test_load_policy_with_a_wider_stored_basis_loads_and_warns_once`
(`crates/cobre-python/tests/test_policy_load_validation.py`).

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
Read: `stochastic/noise.rs` (`transform_ncs_noise`, `compute_effective_eta`).

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
active Dynamic Cut Selection iteration always falls back to the by-scenario
path: the by-node scheduler's frozen-LP load is incompatible with
DCS's cut-free lazy core.
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
`reject_dynamic_cut_selection_under_enumerated`. The enumerated engine seeds each
cut pool at its node-native stride (`enumerated_pool_cut_stride`), whereas dynamic
cut selection's lazy core drives `build_initial_resident_set` (`cut/dcs.rs`) and
`CutPool::enforce_budget`'s eviction-key reader (`cut/pool.rs`) under the
sampled-selection eviction-key discipline; the pairing would drive that reader
down an untested eviction path. Admitting the pairing is the wrong-but-compiling
alternative — it compiles and runs, exercising an unvalidated eviction path.
No shipped deck pairs them, so the reject is byte-neutral. Any non-`Dynamic`
strategy (or none) under enumerated forwards, and `Dynamic` under sampled
forwards, are admitted — the reject discriminates on the `Dynamic` variant, not on
any strategy being present, and does not change `enforce_budget` or
`build_initial_resident_set`, which the gate protects.

Read: `setup/mod.rs` (`admission_gate`,
`reject_dynamic_cut_selection_under_enumerated`), `cut/dcs.rs`
(`build_initial_resident_set`), `cut/pool.rs` (`CutPool::enforce_budget`),
`setup/node_graph.rs` (`enumerated_pool_cut_stride`). Pinned by
`admission_gate_rejects_dynamic_cut_selection_under_enumerated` (`setup/mod.rs` —
the positive enumerated + `Dynamic` reject, plus the enumerated-only and
dynamic-only negatives that do not reject).

## Terminal boundary FCF is booked in the reported total cost

The forward trajectory cost and the simulation per-scenario cost both reconstruct
a path total as `Σ_t cum_d(t)·stage_cost(t)`, where the interior `stage_cost(t) =
(view.objective − d_t·θ_t)·cost_scale` subtracts the discounted epigraph `θ_t` —
the future cost-to-go a later stage realizes as its own immediate cost. At the
TERMINAL stage under a boundary policy (`CutPool::has_warm_start_cuts` on the
terminal stage's pool, i.e. `warm_start_count > 0`) `θ_t` prices the
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
`terminal && pool.has_warm_start_cuts()`; a non-boundary study pins terminal `θ`
to `[0, 0]` (forward) or leaves the terminal pool empty with `θ`'s `0.0` lower
bound driving it to `0` (simulation), so the fix is byte-neutral there.

Read: `cut/pool.rs` (`CutPool::has_warm_start_cuts`, the flag's only formula),
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
one. It then reads the selected pool's `cost_scale_factor` to feed
`rescale_cut_records_for_load` — the additive `StageCuts`
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
`FullFcf` path mirrors the `cost_scale_factor` read + `None`
clean-break reject through `checkpoint_terminal_cost_scale_factor` (the terminal
pool's own value). Resurrecting a read of `metadata.graph_manifest` or
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
`checkpoint_terminal_cost_scale_factor`, `boundary_predates_self_describing_cuts`,
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
