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

## FPHA uses average storage

The FPHA generation constraint is
`g ≤ γ₀ + (γᵥ/2)·(V_in + V_out) + γ_q·q (+ γ_s·s)`. The `−γᵥ/2` coefficient
appears on **both** the incoming and outgoing storage columns — not on `V_out`
alone. (Discovered during deterministic case D06.)
Read: `lp/builder/entries.rs` (`fill_fpha_entries` — pushes `−γᵥ/2` onto both the
incoming- and outgoing-storage columns), `lp/builder/rows.rs` (`fill_fpha_rows`),
and `lp/builder/template.rs`.

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

Read: `crates/cobre-sddp/src/lp/builder/columns.rs` (`cell_max_turbined`).
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

Read: `crates/cobre-sddp/src/lp/builder/columns.rs` (`cell_max_turbined`,
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

Read: `crates/cobre-sddp/src/lp/builder/columns.rs` (`cell_min_turbined`,
`cell_min_generation`, `fill_cell_block_family`), `crates/cobre-sddp/src/lp/builder/rows.rs`
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

## A stored basis warm-starts only at its own node (node-tag)

A `CapturedBasis` carries the declared `node_id` it was captured at
(`CapturedBasis::new(…, NodeId)`, `NodeGraph::node_ids[node]`). Every apply site
warm-starts from a stored basis **only when its `node_id` matches the node being
solved** and treats a mismatch as **cold** (`stored_basis.filter(|c| c.node_id
== node_id)`). A resampled path may revisit a stage at a different node than the
one whose basis is cached there, and warm-starting across that boundary reuses a
basis built against a different LP.

This node-tag check is the **sole** line of defence, not defence in depth:
**CLP accepts a shape-mismatched (or otherwise wrong) warm basis silently**,
whereas HiGHS validates and rejects it loudly
(`reference_solver_basis_validation_asymmetry`). A cross-node warm-start is
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
slot-identity `reconstruct_basis` path, which re-validates shape.

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
`non_controllable_models.parquet` stores `(mean, std)` **as factors**, not as MW.
Read: `stochastic/noise.rs` (`transform_ncs_noise`, `compute_effective_eta`).

## Lower-bound evaluation must patch NCS

`evaluate_lower_bound` patches NCS column bounds per opening via
`StageSolvePrep::run`'s internal `transform_ncs_noise` call, exactly as the
forward and backward passes do. Skipping the patch understates the bound (a
real bug caught during D15). The patch inputs ride on `StageContext`
(`ncs_max_gen`, `ncs_allow_curtailment`), the same struct every other solve
site reads.
Read: `training/lower_bound.rs`, `training/stage_solve_prep.rs`.

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

## Terminal boundary FCF is booked in the reported total cost

The forward trajectory cost and the simulation per-scenario cost both reconstruct
a path total as `Σ_t cum_d(t)·stage_cost(t)`, where the interior `stage_cost(t) =
(view.objective − d_t·θ_t)·cost_scale` subtracts the discounted epigraph `θ_t` —
the future cost-to-go a later stage realizes as its own immediate cost. At the
TERMINAL stage under a boundary policy (`terminal_has_boundary_cuts`, i.e.
`fcf.pools[terminal].warm_start_count > 0`) `θ_t` prices the POST-HORIZON
value-to-go, which no later stage realizes, so it is KEPT in the reported cost
(`stage_cost = view.objective·cost_scale`) — matching the lower bound, which
already carries it through `θ_0`'s cuts (`evaluate_lower_bound` pushes the full
stage-0 `view.objective`). The present values coincide exactly because `θ_t`'s
objective coefficient IS `d_t`, so `cum_d(t)·d_t·θ_t = cum_d(t+1)·θ_t` is the same
term the LB books.

Subtracting `θ_t` at the terminal boundary stage — the interior form — is the
wrong-but-compiling alternative: it drops the post-horizon FCF from the UB /
simulation cost alone, leaving `LB ≫ UB` (a NEGATIVE gap the stopping rule's
`.max(0.0)` clamp then reads as "converged"). The branch is gated on
`terminal && terminal_has_boundary_cuts`; a non-boundary study pins terminal `θ`
to `[0, 0]` (forward) or leaves the terminal pool empty with `θ`'s `0.0` lower
bound driving it to `0` (simulation), so the fix is byte-neutral there.

Read: `training/forward/enumerated.rs` (`solve_forward_node`),
`training/forward/stage_solve.rs` (`run_forward_stage`), `simulation/pipeline.rs`
(`extract_sim_stage_result`, flag computed in `solve_simulation_stage`). Do NOT
change `evaluate_lower_bound` (`training/lower_bound.rs`) — it is correct — and do
NOT fold `θ` into the per-stage `compute_cost_result` breakdown
(`simulation/extraction.rs`), which reports `immediate_cost`/`future_cost`
separately by design. Pinned by `terminal_boundary_fcf_training_gap_is_consistent`
and `terminal_boundary_fcf_simulation_cost_includes_post_horizon`
(`tests/branching_value_oracle.rs`).

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
`load_boundary_cuts` as the authority — the C17 source-drop warning
(`warn_dropped_source_couplings`) is what makes relaxing it safe, letting a
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
counterpart REJECTS, naming the offending hydro or lag depth. The entity
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
`cuts/<pool>.bin`, never `metadata.json`. It resolves `source_stage -> pool` by
matching a pool's own `graph_stage_id`, and reads that pool's `cost_scale_factor`
to feed `rescale_cut_records_for_load` — the additive `StageCuts`
`cost_scale_factor`/`node_id`/`graph_stage_id` fields make one `cuts/<pool>.bin`
self-describing. Three named rejects guard the boundary load: a resolved pool
whose `cost_scale_factor` reads `None` (a pre-self-describing `.bin`) REJECTS
(`boundary_predates_self_describing_cuts`, advising re-export), never silently
defaulting to `LEGACY_COST_SCALE_FACTOR`; a resolved pool whose `node_id` is the
`-1` sentinel (a shared, multi-owner pool) REJECTS — a boundary source must be a
single-node terminal pool; and a `source_stage` matching zero or more than one
pool REJECTS. The `FullFcf` path mirrors the `cost_scale_factor` read + `None`
clean-break reject through `checkpoint_terminal_cost_scale_factor` (the terminal
pool's own value). Resurrecting a read of `metadata.graph_manifest` or
`metadata.producer.cost_scale_factor` on either load path is the
wrong-but-compiling alternative this contract rules out: `metadata.json` is
retired — the study-global graph, `num_stages`, and provenance now live on the
`manifest.bin` `CheckpointManifest` root, consumed only by the `FullFcf`
graph-identity check — so a metadata read would fail to compile or silently
reintroduce a stale-scale bug.

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
`boundary_load_rejects_pre_self_describing_checkpoint`,
`auto_resolver_rejects_sentinel_graph_stage_id`).

### Dated anticipated fan-out reconciliation is hour-weighted by the SOURCE month

The calendar-`subindex` relaxation the parent section reserves for a date family
IS this fan-out, and it is confined to `AnticipatedThermalState` slots inside
`load_boundary_cuts`. A source study prices its anticipated commitments on a
monthly delivery calendar; the current study delivers them on a post-study weekly
(and monthly) calendar. `resolve_anticipated` reconciles each target slot `w`
against the source months `M` it overlaps **by real calendar date**, not by
subindex: full coverage yields `RebindOp::Blend` with per-month weight
`overlap_hours(w, M) / H_M` — divided by the **source** month's hours `H_M`, the
conservation identity that makes a slot's fanned coefficients sum back to the
source's (a covered target slot's coeff ratio equals `H_w / H_M`). A target slot
straddling into unpriced time yields `RebindOp::Renormalize`: the same weights
additionally scaled by `H_w / Σ_covered overlap`, so the covered months' price
density replicates across the uncovered span instead of deflating the boundary FCF
with an implicit `0.0` term. No covered month yields `Zero`.

Only a post-study-targeted ring slot fans out. A dated target slot with NO
resolved delivery interval is an IN-STUDY ring slot — a commitment delivered
WITHIN the current horizon (a matured commitment fished at the terminal stage, or
a `K = 0` sub-stage-lead delivery self-delivered there) — and resolves to `Zero`:
the terminal boundary FCF prices only post-study obligations, so a within-horizon
delivery, already discharged inside the study, contributes nothing. This is sound
BECAUSE a ring slot's `delivery_date` and its target interval are BOTH derived
from the SAME modular delivery target at the same stage — the manifest recovers
`(slot, plant)` from a ring column via `slot_lane_at` (the exact inverse of
`out_col`/`in_col`) and dates the slot at its modular delivery stage
(`build_stage_entity_manifest`), while the companion
`build_terminal_anticipated_delivery_intervals` derives the interval from that
same target — so a slot dated onto a post-study stage always carries
`Some(interval)` and a slot dated onto an in-study stage always carries `None`:
`dated ⟺ Some(interval)` holds by construction, and a `None` interval on a dated
slot marks the in-study ring uniquely, never a failed post-study resolution. Two
wrong-but-compiling alternatives: `RebindOp::Reject` here (the
retired behavior) aborts a legitimate boundary load the moment any anticipated
thermal delivers in-horizon (the K=0-at-terminal case — the manifest DELIBERATELY
dates a matures-this-stage slot, pinned by
`anticipated_slot_delivery_anchor_matches_delivery_stage_year_month`); resolving
the in-study slot an in-horizon interval and fanning it out would wrongly `Blend`
a within-horizon delivery against the source's months.

`Blend` and `Renormalize` are semantically distinct and MUST NOT be collapsed:
`rebind_cut` applies both through the identical weighted-sum, so unifying them
reads like harmless dedup — but only `build_rebind` carries the `Renormalize`
anti-deflation scale, and dropping it silently understates the boundary FCF on any
partial-coverage calendar. Two further wrong-but-compiling alternatives: dividing
by the **target** slot's `H_w` instead of the source `H_M` (breaks the
`H_w / H_M` conservation ratio), and joining on `subindex` instead of the real
interval (anticipated delivery is NON-monotone in subindex — the modular
delivery-target residue of the ring contract above — so a subindex join misaligns
months to weeks). Source month intervals are reconstructed from the `YYYYMM01`
day-01 delivery anchor (`decode_month_anchor`, the exact inverse of
`year_month_day_anchor`); an exact or superset match reconciles byte-for-byte
(`Copy`). The `H_w / covered` division is guarded: `resolve_anticipated` returns
`Zero` on `terms.is_empty()` before it can divide by a zero covered span.

The constant intercept fold (`build_boundary_fold`) reuses this SAME
`overlap/H_M` source-month weighting on a different object: a class-4 fixed
post-horizon window's declared MW is a CONSTANT, not a state dimension, so
`Σ_M (overlap_hours(w, M) / H_M) · v_w` sums directly into a `(source_pos,
factor)` fold vector instead of a `Blend`/`Renormalize` op. A fixed window
overlapping no source month contributes zero — mirroring `RebindOp::Zero`,
never an error — and the fold's own filter drops every exact `0.0` factor
rather than emit a zero-weighted term. `load_boundary_cuts` adds the fold onto
each cut's RAW intercept (`record.intercept += Σ coeff[source_pos] · factor`)
BEFORE `rescale_cut_records_for_load`'s cost-scale and legacy-ratio
transforms, in the same source-coefficient frame `rebind_cut` reads — so the
folded future-cost term rides both rescale transforms with the rest of the
intercept. Folding after rescale, or reading a rescaled coefficient into the
fold sum, is the wrong-but-compiling alternative: the transforms are not
idempotent across that boundary, so a post-rescale fold prices the fixed
commitment at the wrong scale.
Read: `policy/reconcile.rs` (`resolve_anticipated`, `build_rebind`, `rebind_cut`,
`decode_month_anchor`, `overlap_hours`, `build_boundary_fold`, the
`Blend`/`Renormalize` variants), `setup/mod.rs` (`year_month_day_anchor`),
`setup/accessors.rs` (`build_terminal_anticipated_delivery_intervals`, the
target-interval companion), `policy/policy_export.rs`
(`build_stage_entity_manifest`, dating each ring slot at its modular delivery
stage via `slot_lane_at`), `policy/policy_load.rs` (`load_boundary_cuts`
threads the target delivery intervals and applies the fold before rescale).
Pinned by the `hm_distribute_conservation` fixtures in
`tests/anticipated_core.rs` (coeff ratio equals `H_w / H_M`, invariant to the
delivery stage's hours), the `Blend`/`Renormalize` `rebind_cut` unit tests (both
apply identical mechanics, distinction is only the weight),
`tests/boundary_reconcile_defaults.rs` (the fan-out matrix and the superset
bit-identity `to_bits` pin),
`build_rebind_dated_in_study_ring_slot_with_no_interval_yields_zero` (a dated
target slot with no interval resolves to `Zero`, not a reject, even when a source
month would overlap it), and the constant intercept fold's own regressions
`boundary_fold_marked_frame_moves_intercept_by_hand_computed_delta`,
`boundary_fold_legacy_frame_moves_intercept_by_hand_computed_delta`,
`boundary_fold_no_overlap_window_leaves_intercept_bit_identical`, and
`boundary_fold_empty_windows_intercept_bit_identical`, all in
`policy/policy_load.rs`.

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
lagged-delivery ring construct, owned by `DeliveryRing`: a borrowed outgoing
block (identity-resolved, contributing to `n_state`) and a separate borrowed
incoming block (pinned via `state_to_lp_incoming_column`), advanced one
Markov-1 slot per stage by the same interior shift row
(`DeliveryRing::emit_shift_rows`) and the same paired row-cap/column-freeze
masking (`DeliveryRing::freeze_masked_columns`). The two rings differ only in
how each deposits into its newest slot and in what a masked terminal slot
means — both differences live entirely at each ring's own call site, never a
second skeleton implementation:

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
(`transit_bucket_ring`, `n_lanes = 1`, over that plant's ragged contiguous
sub-range); anticipated instantiates ONE dense ring spanning every plant
(`anticipated_ring`, `n_lanes = n_anticipated`, slot-major/plant-minor) — both
addressing schemes resolve through the same `out_col`/`in_col` formula
(`block.start + slot * n_lanes + lane`).
Read: `lp/builder/delivery_ring.rs` (`DeliveryRing::emit_shift_rows`,
`freeze_masked_columns`, `emit_deposit`, `out_col`/`in_col`, `slot_target`),
`lp/builder/entries.rs` (`transit_bucket_ring`, `anticipated_ring`).

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
`fill_arc_release_block_entries`, `transit_bucket_ring`), `lp/indexer/state_space.rs`
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
Read: `setup/bucket_topology.rs` (`build_transit_bucket_topology`,
`TransitBucketTopology::column_order`). Pinned by the bucket column-order
declaration-invariance regression: two systems differing only in the
declaration order of their hydros produce identical `column_order`,
`per_plant_depth`, and `n_buckets`.

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
  `delivery_stage_count = n_delivery.max(n_stages)`, so
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
`setup/bucket_topology.rs` (`horizon_cap_active`), `lp/builder/columns.rs`
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
imprecision, not a bug to patch by capping
`TransitBucketTopology::per_plant_depth`/`column_order` too — those size from the
global max over every anchor and must retain what the earliest stages need.
Read: `setup/bucket_topology.rs` (`horizon_cap_active`), `lp/builder/layout.rs`
(`build_transit_bucket_row_pos`), `lp/builder/columns.rs` (`fill_transit_bucket_columns`).
Pinned by the horizon-depth-cap regression (the last stage's active-lag cap
reaches zero, so no slot targets past the horizon), `build_transit_bucket_row_pos`'s
own consumption regression (that same cap sequence emitting correspondingly
fewer rows), and a sub-stage-delay case's last-stage release, whose dropped
share surfaces as an uneven per-stage delivery split rather than a credited
one.

### Sub-contracts: mode-independent sizing, aggregation consistency, fixed delivery density

The bucket state stays a pure function of stage lengths, never of
`n_blks`/`block_mode`, only because each of the following holds:

- **Depth from stage lengths alone.** Bucket depth and `n_buckets` derive from
  the per-stage calendar and the pre-study anchor alone
  (`study_stage_durations`, `build_transit_bucket_topology`) — never from `n_blks` or
  `block_mode`. Deriving any part of the depth inside a block-aware code path
  re-couples the state dimension to how a stage happens to be resolved.
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
  (`resolve_chrono_arrival_density` reading
  `TemplateBuildCtx::arc_arrival_density`, built by `build_arc_arrival_density`
  as a blend over every contributing source stage's lag, resolved in the
  ARRIVAL stage's own frame), never by tracking which origin block a unit came
  from. Tracking origin-to-arrival-block correlation would grow the bucket
  into a per-block vector whose length scales with the receiving stage's
  `n_blks` — re-violating the depth-from-stage-lengths property above.

Read: `lead_time/mod.rs` (`resolve_spread`'s
`block_deposits`/`within_stage_routing`/`arrival_density` fields,
`resolve_block_factors`'s `BlockFactors`, `resolve_arrival_density_at`),
`setup/bucket_topology.rs` (`build_arc_arrival_density`), `lp/builder/entries.rs`
(`fill_chronological_water_entries`, `resolve_chrono_arrival_density`). Pinned
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
per-plant reachability bound `k_i` (`StateSpace::anticipated_lead_stages`, the
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
`AnticipatedResolution::resolve`), `setup/mod.rs`
(`resolve_anticipated_commitments_core`'s `LeadStages`/`LeadTime` split).
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
at its MODULAR delivery stage, reached through the RING-AXIS residue: the next
ring-axis target `r >= t` in the slot's residue class (`delta = (slot_idx +
k_max − t mod k_max) mod k_max`, `r = t + delta`), mapped to the physical
delivery stage `m = physical_target(r)` (The ring axis subsection above)
before dating. Two wrong-but-compiling alternatives: `t + slot_idx` (the
retired shift-ring form, wrong whenever `t mod k_max != 0`), and dating the
raw ring-axis `r` directly instead of `physical_target(r)` — it lands on the
excised fixed post-horizon window's stub stage whenever a plant declares one.
Reachability uses the plant's OWN `StateSpace::anticipated_lead_stages[plant]`
bound (`slot_idx < k_i`), not a depth- or decider-only check
(`AnticipatedResolution::decision_sets`/`depth` count only within-study-decided
commitments and silently exclude a still-draining pre-study seed): a slot beyond
that bound is structural padding dated at the sentinel even when its delivery
target `m` still lands inside the horizon — the multi-plant heterogeneous-lead
case, where plants sharing one `k_max`-wide ring have different reachable widths.
`build_stage_entity_manifest` applies this before populating
`EntitySlot::delivery_date`.

The sign / `col_scale` invariants are unchanged from storage and the water buckets:
the incoming column's reduced cost is DIVIDED by `col_scale` on extract
(`extract_duals_from_view`) and the outgoing column's cut coefficient is MULTIPLIED
back on render (`push_scaled_coefficient`); `col_scale` is forced to `1.0` across
the whole region (the reconcile contract below).

Read: `lp/indexer/state_space.rs` (`StateSpace::commit_out`,
`StateSpace::commit_in`, `commitment_hold_in_study_offset`, `state_to_lp_column`,
`state_to_lp_incoming_column`), `lp/builder/delivery_ring.rs`
(`DeliveryRing::emit_carry_rows`, `emit_deposit`, `freeze_masked_columns`,
`slot_lane_at`), `lp/builder/entries.rs`
(`fill_anticipated_slot_definition_entries`,
`fill_anticipated_state_out_def_entries`, `anticipated_ring`), `lp/builder/rows.rs`
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
`anticipated_slot_padding_beyond_own_lead_is_sentinel`). The ring-axis
excision itself is additionally pinned by the collision/identity regressions
`excision_keeps_each_study_stage_fishing_its_own_seed`,
`zero_gap_with_post_study_resolves_an_identity_ring_and_occupancy_depth`, and
`zero_gap_carry_slot_addressing_matches_the_open_coded_identity_formula`, all
in `tests/anticipated_core.rs`, and the excised-space manifest-dating
regression `date_ring_slots_in_excised_space_maps_through_physical_target`
(`policy/policy_export.rs`).

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
EXTENDED delivery calendar (`n_delivery = StateSpace::delivery_stage_count`, the
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
(`StateSpace::delivery_stage_count`), `lp/builder/layout.rs`
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
`delivery_total_hours[delivery_stage]` and
`delivery_cumulative_discount_factors[delivery_stage]` for its present-value objective,
and `is_anticipated_decision_active_for_delivery` (the plant's window at
`delivery_stage`) for its dormancy — each at the plant's own genuine delivery
stage, never at `stage_idx`. The delivered commitment is a hard equality with
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

Read: `lp/builder/columns.rs` (`fill_anticipated_columns`),
`lp/indexer/anticipated_gate.rs` (`is_anticipated_decision_active_for_delivery`),
`lp/generic_constraints.rs` (`resolve_anticipated_decision`),
`cobre-io` `validation/semantic/thermal.rs`
(`warn_thermal_generation_on_anticipated_thermal`), `cobre-io`
`validation/semantic/block_bounds.rs`
(`check_block_id_on_anticipated_thermal`, the rule the base read's safety
depends on). Pinned by
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
quantity that aborts training outright.

`StageSolvePrep::run` therefore reconciles every pinned commitment against the
delivery generation column's **enforced** bound (`col_upper * col_scale`, the
round-tripped value the solver applies — not the template's raw `max_gen`),
relaxing the column just far enough to admit drift within `drift_margin`. Drift
beyond that margin is `SddpError::AnticipatedCommitmentOutOfBounds`, never
absorbed: the margin is the discrimination line between solver noise and a
modelling error, and a guard that relaxes for ANY overshoot silently admits a
plant generating past its cap.

Two forbidden alternatives, both of which have shipped:

- **Deleting the reconciliation on the premise that unscaling makes it
  redundant.** `apply_commitment_hold_col_scale_unscale` (`col_scale = 1.0` on
  `commit_out ∪ commit_in`) removes the ring _carry_ drift and
  is retained — the carry is bit-exact and the decision column's own value is
  bit-exact at its bound. It cannot remove the drift the basis factorization
  introduces at the deposit row, because exactness there is the solver's to give
  and it does not give it. No amount of unscaling closes this.
- **Making the reconciliation an opt-in hook.** It is not a variation point and
  takes no parameter: `run` derives its own gate, so all four solve sites (forward,
  backward, lower bound, simulation) get it and none can opt out. An
  `Option<..Ctx>` hook threaded per call site is what let all four silently lose it
  in one commit.

Read: `lp/builder/commitment_reconcile.rs` (`reconcile_commitment`,
`fill_bound_relaxations`, `drift_margin`), `training/stage_solve_prep.rs`
(`StageSolvePrep::reconcile_commitments`), `lp/builder/scaling.rs`
(`apply_commitment_hold_col_scale_unscale`). Pinned by
`anticipated_commitment_drifted_over_cap_is_absorbed` (a seed a hair past the cap
trains; it returns `Infeasible` the moment the reconciliation is disabled) and
`anticipated_commitment_over_cap_seed_is_refused` (a genuine over-commitment is
named, not absorbed). `anticipated_commitment_at_cap_survives_ring_carry` does
NOT pin this contract and must never be mistaken for it: a seed exactly at the cap
carries zero drift, never reaches the reconciliation, and stays green with the
guard deleted — an at-cap-only suite is what let this regression ship.

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


## Expectation/CVaR mixture keeps the expectation floor

Both risk-weight entry points in `convergence/risk_measure.rs` initialize
`mu[i] = (1-lambda)*p[i]` and allocate only `lambda` additional mass, capped
per scenario by `lambda*p[i]/alpha`. Starting from zero with a full unit of
mass and only the combined upper bound changes the risk measure and can
invalidate cuts for the intended objective. The scalar and cut paths share
this contract, pinned by `cvar_mixture_preserves_expectation_floor_in_value_and_cut`
and `cvar_mixture_matches_primal_tail_formula_and_envelope`.
