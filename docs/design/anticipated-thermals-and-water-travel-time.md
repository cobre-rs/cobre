# Anticipated thermal generation & water travel time

A reference explanation of how cobre models two _time-delayed delivery_ phenomena
— thermal generation that must be **committed a lead time in advance**, and water
that **takes travel time to flow** from an upstream reservoir to a downstream one —
from input parsing, through state-layout sizing, into the LP.

The two look unrelated in the input, but internally they are **the same
construct**: a _lagged-delivery ring_. This document leads with that shared
substrate, then treats each subsystem, then contrasts the two. Symbols are cited
by name and file so the pointers survive line-number churn; the authoritative
correctness contracts live in `.claude/rules/sddp.md` (§ "Water travel time" and §
"Anticipated thermal commitments"), which this document summarizes but does not
replace.

> New to these features? Start with the primer,
> [`../guide/anticipated-thermals-and-water-travel-time.md`](../guide/anticipated-thermals-and-water-travel-time.md)
> (contextualization, worked examples, diagrams), then return here for the contracts.

---

## 0. The unifying idea

Both subsystems track a quantity **produced at one stage** and **realized at a
later stage**, with the intervening amount held as _state_ that advances one step
per stage:

|                 | Water travel time                                | Anticipated thermal                                    |
| --------------- | ------------------------------------------------ | ------------------------------------------------------ |
| Produced        | water **released/spilled** at the upstream plant | a generation MW value **decided** (committed)          |
| Delayed by      | `travel_time_hours` on the arc                   | a lead (`lead_stages` or `lead_time_hours`)            |
| Realized as     | inflow **arriving** at the downstream plant      | thermal **generation delivered** at the delivery stage |
| In-flight state | in-transit "bucket" volumes                      | held commitment MW                                     |

The in-flight amount lives in a **ring of state slots**. Each stage, the ring
advances one slot; the slot that matures this stage is consumed; a fresh slot is
deposited. That ring is one shared code primitive — `DeliveryRing`
(`crates/cobre-sddp/src/lp/builder/delivery_ring.rs`) — and each subsystem
occupies its own region of the SDDP state vector (`Buckets` and
`CommitmentHold`). They differ in the call-site-local ways §4 tabulates.

---

## 1. The shared substrate

### 1.1 State-vector layout

Every stage subproblem carries a stage-invariant state vector whose layout is
owned by `StateSpace` (`crates/cobre-sddp/src/lp/indexer/state_space.rs`). Both
delivery rings sit inside it:

```text
[0, N)                     storage             — outgoing storage volumes (N = hydro count)
[N, N*(1+L))               inflow_lags         — AR lag variables (L lags per hydro)
[N*(1+L), N*(1+L)+B)       transit_buckets_out — WATER ring, outgoing (identity-resolved)
[N*(1+L)+B, …+S)           commit_out          — ANTICIPATED ring, outgoing
[…+S, …+N)                 z_inflow            — realized inflow (auxiliary, not state)
[…+N, …+2N)                storage_in          — incoming storage (pinned)
[…+2N, …+2N+B)             transit_buckets_in  — WATER ring, incoming (pinned)
[…+B, …+B+S)               commit_in           — ANTICIPATED ring, incoming (pinned)
…+S                        theta               — future-cost variable
```

with `B = n_buckets` (water) and `S = A·k_max` (anticipated: `A =
n_anticipated`, `k_max` = ring depth). The
region walk order — `Storage → Lag → Buckets → CommitmentHold` — is owned once by
`REGION_ORDER`; every consumer dispatches over it exhaustively, so a new region
fails to compile until handled.

Two structural facts that recur below:

- **Each ring is an _outgoing_ block and a separate _incoming_ block**, never one
  dual-purpose range. The outgoing block contributes to `n_state` and is resolved
  to its LP column **by identity** (`state_to_lp_column`); the incoming block is
  **pinned** to the previous stage's value via `set_col_bounds` on the column
  `state_to_lp_incoming_column` resolves (there is no state-fixing _row_ — pinning
  is column bounds).
- **`n_state` counts each dimension once** (out and in describe the same
  dimensions). With no arc / no anticipated plant, the corresponding range
  collapses to `0..0` and the whole layout reproduces the pre-feature bytes
  exactly.

### 1.2 The `DeliveryRing` primitive

`DeliveryRing::new(out_block, in_block, n_lanes, depth)` borrows its out/in column
blocks from `StateSpace` (never a private copy) and owns the ring arithmetic. A
ring is a dense **slot-major, lane-minor** grid of `n_lanes × depth` columns:

```
out_col(slot, lane) = out_block.start + slot * n_lanes + lane
in_col (slot, lane) = in_block.start  + slot * n_lanes + lane
```

`slot_lane_at` is the exact inverse (used by the policy manifest). The primitive
emits four kinds of CSC entries; each subsystem uses a subset:

| Method                  | Emits                                                            | Used by         |
| ----------------------- | ---------------------------------------------------------------- | --------------- |
| `emit_shift_rows`       | `out[slot] +1`, `in[slot+1] −1` (advance to the **next** slot)   | **water**       |
| `emit_carry_rows`       | `out[slot] +1`, `in[slot] −1` (**same-slot** hold)               | **anticipated** |
| `emit_deposit`          | `out[slot] +1`, `decision_col −1` (latch a decision into a slot) | anticipated     |
| `freeze_masked_columns` | masked slot → `[0,0]`; reachable → the ring's open bound         | both            |

All ring definition rows carry `[0,0]` equality bounds — the `+1`/`−1` structural
coefficients do the work, never the bounds.

### 1.3 State → LP column, and duals

- **Outgoing** state (storage, buckets, commitment-hold) maps to its LP column by
  **identity** in `state_to_lp_column`; only the inflow-lag region remaps (lag 0 →
  `z_inflow`, lag ℓ → previous lag). This is why a cut row's coefficient on a ring
  slot lands on the right outgoing column.
- **Incoming** state resolves through `state_to_lp_incoming_column`, dispatched by
  region to the matching pinned block start (`incoming_block_start`). The water
  arm (`transit_buckets_in`) and the anticipated arm (`commit_in`) are **explicit**
  — a bucket index must not fall through to the commitment-hold catch-all.
- **Dual sign convention (identical to storage):** the incoming column's reduced
  cost is **divided** by `col_scale` on extraction (`extract_duals_from_view`); the
  outgoing column's cut coefficient is **multiplied** back on render
  (`push_scaled_coefficient`). Divided on extract, multiplied on render.

### 1.4 Masking is two-sided and ships together

A masked ring position (`row_pos[i] == None`) gets **no definition row** _and_ a
**frozen `[0,0]` outgoing column**, in the same pass. Wiring only one side leaves
either a dangling row referencing a frozen column, or a free column with no
defining constraint — both compile and both are wrong. The row half lives in each
ring's layout table; the column half is `freeze_masked_columns`.

### 1.5 Cut-state projection keeps ring state priced at every pool

`StateRegion::cut_enabled` returns `true` for both `Buckets` and `CommitmentHold`
at **every** pool, terminal included, ignoring the per-stage `StageStateConfig`.
So every ring slot is a priced future-cost dimension that a loaded boundary cut's
coefficient `β` lands on directly, via the one generic `β·state` projection
(`CutStateProjection::new`) — never a per-family pricing arm.

---

## 2. Water travel time

### 2.1 Input

Travel time is an **arc attribute declared on the upstream hydro, in hours** —
`travel_time_hours: Option<f64>` on `Hydro`
(`crates/cobre-core/src/entities/hydro.rs`). There is **no bucket count and no
discretization in the input**: the user supplies one scalar per plant, and cobre
derives everything. An arc exists iff `travel_time_hours == Some(t) && t > 0.0 &&
downstream_id.is_some()` — `0.0` means _undeclared_, not "instant-with-a-bucket"
(`TransitBucketTopology::arcs`, resolved once in
`crates/cobre-sddp/src/bucket_topology.rs`).

The companion input is the **pre-study release history** that seeds the buckets:
`HydroPastDefluence { hydro_id, start_date, end_date, value_m3s }`
(`crates/cobre-core/src/constraints/initial_conditions.rs`) — windowed records
(end-exclusive), multiple non-overlapping windows per hydro allowed.

Validation (`crates/cobre-io/src/validation/semantic/travel_time.rs`) is a
config-time matrix; the load-bearing gates: negative/non-finite `t` is a hard
error; `t == 0` and negligible/horizon-inert travel times warn; **`past_defluences`
must cover `(0, t]` hours before the study start, contiguously** (a hard
`BusinessRuleViolation` — this is why the seed builder has no fallback); a
heterogeneous-`t` confluence into one downstream under chronological blocks is
`NotImplemented`; releasing while the downstream is not yet operating is rejected.

### 2.2 Sizing — how many bucket slots

The number of slots is a **measured overlap**, not a chosen count. The building
block is `window_period_overlaps` (`crates/cobre-core/src/model/temporal/overlap.rs`):
it intersects a time window against consecutive stage periods and returns _one
entry per period from period 0 through the deepest overlapped period_ — a
contiguous run, keeping interior zeros, truncating only trailing zeros.

`resolve_spread` (`crates/cobre-sddp/src/lead_time/mod.rs`) turns a travel time
into per-stage **k-weights** by overlapping the _arrival window_ `[t_v, t_v + h_t)`
(a uniform release over the anchor stage, delayed by the travel time) against the
stage calendar counted from the anchor stage:

```
k_d(t) = | [t_v, t_v + h_t) ∩ [S_d, S_{d+1}) | / h_t
```

where `S_d` are cumulative stage boundaries from stage `t`, and `h_t` is the anchor
stage's duration. Then:

- `Σ_d k_d = 1` **exactly** (debug-asserted). Conservation of released water.
- `k_0` is the **same-stage** share — delivered on the water-balance row directly,
  **no bucket**.
- `stage_reach = max{ d : k_d > 0 }` — the deepest _index_, never a count. This is
  the in-study depth at that anchor stage.

The per-plant slot count folds two anchors (`build_transit_bucket_topology`,
`bucket_topology.rs`):

- **In-study depth**: `in_study_depth = resolve_spread(t_v, stage, …).stage_reach`
  — discards the index-0 same-stage share (no bucket needed for it).
- **Pre-study residual depth**: `ic_only_depth = |window_period_overlaps(0, t_v,
study_durations)|` — the raw overlap _count_ (the in-transit water arriving over
  `[0, t_v)` has no same-stage share to discard). This is the one place the depth
  arithmetic diverges, and it is deliberate.

The formula:

```
L_j = max over arcs t_v into plant j of
        max( max over stages t of stage_reach(t_v, t, extended),
             |window_period_overlaps(0, t_v, study_durations)| )

n_buckets = Σ_j L_j
column_order = [(j, 1), (j, 2), …, (j, L_j)] for each plant j in canonical order
```

`extended` is the study calendar followed by any declared post-study calendar
(`DeliveryCalendar::total_hours`), padded with copies of its trailing duration
only past what that base calendar already covers (`extend_for_resolution`,
`bucket_topology.rs`). With no declared post-study calendar the base is the
study-only vector and the formula above is unchanged.

Key properties (each pinned by a named test in `bucket_topology.rs`):

- **Confluence aggregates** — all arcs into one downstream collapse to a _single_
  block of depth `max_i L_i`, never one block per arc.
- **The IC anchor can dominate** — a fine-then-coarse calendar can need more depth
  for the draining pre-study mass than any in-study anchor.
- **Sizing is uncapped by the horizon** — `L_j` retains what the _earliest_ stages
  need even though the per-stage mask decays to 0 at the terminal.
- **Depth is a pure function of stage lengths** — never `n_blks` / `block_mode`.
- **Canonical ordering** — buckets sort by the downstream plant's
  `(operational_start_date, id)` canonical index, then lag; never declared-id or
  cascade-traversal order (declaration-order invariance).

There is **no closed-form `⌈t_v / h_t⌉`** depth: on a non-uniform calendar it
silently drops trailing mass. The overlap measure is the contract.

### 2.3 The per-stage reachability mask and `boundary_present`

Sizing gives the _global_ depth; a per-stage mask (`per_stage_mask`) says which
lags are live at each stage: the union of this stage's own-release depth with the
**decaying IC residual** `ic_depth − stage`. With no boundary policy the mask is
**horizon-capped** at `n_stages − 1 − stage` (deep terminal slots masked `[0,0]` —
the "terminal credit deferred" imprecision, confined to this no-boundary path);
with a boundary policy present
(`config.policy.boundary.is_some()`, threaded as `boundary_present`) the mask is
the **raw uncapped** depth at every stage so those terminal slots stay live and
reach the boundary-priced cut projection. **Sizing is identical either way** —
only the mask changes — and the gate is required, not cosmetic: un-capping
unconditionally would perturb every existing no-boundary golden even at unchanged
optimal cost.

### 2.4 LP entry

The topology tables are resolved once and borrowed by the template build
(`lp/builder/template.rs`); they are never stored on the templates and never
re-derived. Per stage, `build_transit_bucket_row_pos` turns the mask into compact
row positions (`None` = masked). Then:

- **One `DeliveryRing` per downstream plant**, `n_lanes = 1`, over that plant's
  contiguous sub-range (`DeliveryRing::transit_buckets`,
  `crates/cobre-sddp/src/lp/builder/delivery_ring.rs`). Ring slot `k` ↔ lag
  `k+1`.
- **The shift** (`fill_transit_bucket_definition_entries` → `emit_shift_rows`):
  `b_d^out = b_{d+1}^in + (deposits)` — each stage the mass advances one slot
  toward maturity. Emitted mode-independently, outside the block-mode match.
- **Release routing** (`fill_arc_release_block_entries`): the arc's release column
  carries `k_0` onto the downstream water-balance row (same-stage share) and
  `k_1…k_depth` into the bucket definition rows (`−k_d · τ`). The **same** release
  column feeds both — never a separate once-per-stage family. A masked deep lag's
  share is **dropped**, never misdirected onto another lag's row.
- **Replication, not apportionment** (`push_plant_release`): a plant's release is
  `Σ_c q_c + s` over a disjoint cell partition, so the `k_d` coefficient is
  **replicated** at the same magnitude onto every cell's turbine column and the
  spillage column — never divided by cell count. Conservation holds _per cell_, so
  the `Σ_d k_d = 1` assertion stays once per arc per stage.
- **Maturity** (on the downstream water-balance row): a single entry `−1.0` on
  `in_col(0, 0)` — the confluence sum over every upstream arc already lives inside
  the state variable, and the bucket state is already a volume (hm³), so no `τ`
  scaling. Under chronological blocks the maturing mass instead spreads across the
  arrival stage's block rows by a fixed, `block_mode`-independent arrival density
  `ρ` looked up from a setup-precomputed table (`resolve_bucket_arrival_density`).

### 2.5 Seed and rolling output

`build_initial_transit_bucket_state` seeds stage-0 incoming buckets directly from
`past_defluences`: for each matching window it computes the k-weighted share of the
window's volume over the arc's travel time (`hour_window_shares`, the same overlap
engine) and deposits **additively** across windows (`filter`, never `find`). The
splice writes into the outgoing block, which the incoming-column resolver remaps to
the pinned incoming column. Simulation emits a `transit_seed` parquet
(`build_transit_seed`) shaped like `past_defluences` so a follow-on run re-seeds
verbatim — faithful only for `t_v ≤ horizon` (a ratified scope boundary: the reader
cannot represent a lag deeper than the receiving study's stage count).

---

## 3. Anticipated thermal generation

### 3.1 Input

A thermal plant declares **only a lead** via `AnticipatedConfig`
(`crates/cobre-core/src/entities/thermal.rs`), one of two mutually-exclusive modes:

- `LeadStages(u32 ≥ 1)` — a stage-count lead; the calendar is never consulted.
- `LeadTime(f64 > 0)` — a physical lead time in hours, delivery-anchored (the same
  clock as a water arc's `travel_time_hours`).

The wire form (`crates/cobre-io/src/system/thermals.rs`) is `#[serde(untagged,
deny_unknown_fields)]` over `{lead_stages}` / `{lead_time_hours}` — supplying both
keys or neither matches no variant and is a parse error; that _is_ the exclusion
mechanism.

There is **no delivery-window field on the config**. The delivered quantities live
on two separate surfaces:

- `initial_conditions.past_anticipated_commitments[]`
  (`AnticipatedCommitmentHistory { thermal_id, start_date, end_date, value_mw }`) —
  pre-study **decided** MW windows that deliver into the study's leading stages
  (the stage-0 ring seed; sunk cost, never in the objective).
- `post_study_stages.json` (`crates/cobre-core/src/model/post_study.rs`) — the
  **sole** post-study declaration surface: an ordered post-study calendar
  segment plus a per-`(thermal, post-study stage)` `cost_per_mwh`/`min_mw`/`max_mw`
  an in-study decision delivering past the horizon is priced and bounded against.
  It extends the delivery axis to `n_delivery = n_stages + n_post` — the runtime
  axis is `DeliveryCalendar::total_hours`, read by `resolve_anticipated_commitments_core`
  (`build_extended_delivery_axis` is the separate cobre-io _validation_ axis used
  for the reach check below); with it absent the axis is study-only and no lead can
  reach a post-study stage. `min_mw == max_mw` pins a fixed post-study profile (a
  legitimate replay deck).

Validation (`crates/cobre-io/src/validation/semantic/thermal.rs`): the
lead-versus-horizon bound rejects a `LeadTime` whose lead exceeds the summed
study-stage durations (strict `>`) _only_ when the plant's extended lead reaches
**no** declared post-study stage at all — a lead that reaches one legitimately
delivers past the horizon, either decided in-study or decided pre-study as a
**fixed post-horizon commitment** (below), and is exempt either way. Reach is
resolved by `classify_deliveries`'s four-way partition over the concatenated
study + post-study calendar that `build_extended_delivery_axis` assembles and
`extended_deciders` walks.

A pre-study-decided, post-study-delivered commitment — a **fixed post-horizon
commitment** — is representable: the
`initial_conditions.past_anticipated_commitments` surface (above) extends
unchanged to carry it — its windows are allowed to sit past the study horizon;
no new field, no schema reshape. It is a declared constant that never
enters the ring, priced by the boundary intercept fold and reported at its real
delivery date (§3.5). The commitment windows must **tile exactly** the plant's
pre-study-decided delivery set — the leading in-study stages and, past the
horizon, the stages the plant also decides before the study — at coverage
`1.0`: a gap (a pre-study-decided stage left uncovered) and an over-coverage (a
window reaching a stage the study itself decides, or one beyond the plant's
decision reach) are both rejected. The committed value must lie inside the
delivery stage's generation bounds, in-study and post-horizon alike; and —
load-bearing for the LP — `check_block_id_on_anticipated_thermal` rejects any
per-block bound row on an anticipated thermal, which is what licenses the LP's
overlay-ignoring bound read (§3.4).

### 3.2 Resolution & sizing — how many ring slots

The lead resolves to a **decider** `c(m)` = the decision stage for each delivery
stage `m` on the **extended delivery axis** `[0, n_delivery)` (`n_delivery =
n_stages + n_post`, study-only when no `post_study_stages.json` is declared), via
`PointResolution` (`crates/cobre-sddp/src/lead_time/mod.rs`):

- `LeadTime` (`resolve_decider_physical`): `c(m)` is the stage containing
  `stage_end(m) − Δ` — **end-anchored** (a sub-stage lead `Δ < h_m` then gives
  `c(m) = m`, a case a start-anchored form could never reach), ties to the earlier
  stage; `None` means a pre-study decider.
- `LeadStages` (`resolve_decider_stage_count`): `c(m) = m − ℓ`, calendar never read.

From `c(m)` a difference-array prefix sum builds two per-decision-stage series:
`depth[t] = |{ m > t : c(m) = Some(c'), c' ≤ t }|` (in-study deciders only —
pre-study deciders contribute nothing) and the **full in-flight occupancy**
`occupancy[t] = |{ m > t : m has a carrier as of t }|`, which additionally counts
every pre-study (`None`-decider) commitment still in flight. The **ring depth**
per plant is

```
ring_depth = max(occupancy_max, n_none_in_study)
k_max = max over plants i of ring_depth_i
```

where `occupancy_max = max_t occupancy(t)` and `n_none_in_study` is the
plant's leading pre-study (`None`-decider) run inside the decision axis alone.
Sizing from `occupancy_max` alone under-counts the one moment every
simultaneous pre-study seed is in flight — stage 0, before the first fishing —
whenever that run outlives the occupancy term: the last seeded stage then
silently delivers an earlier stage's MW (the ring-depth under-sizing defect
this form closes). Sizing from `depth` instead under-counts even earlier and
collides a carry row with a deposit row the moment a pre-study seed outlives
the in-study in-flight count; `occupancy` and `depth` coincide only on a
uniform-calendar lead (`occupancy[t] = depth[t] + max(0, n_none − 1 − t)`,
`n_none` = the leading pre-study run), so a pure-`LeadStages` study still gets
`k_max = max_i ℓ_i` byte-for-byte — `LeadStages(l)` returns `l` **verbatim**
(the pre-delivery-anchor sizing anchor). `k_max` is a pure function of the
per-plant leads and the delivery calendar — independent of `n_blks`,
`block_mode`, or the number of decisions. It is computed in
`AnticipatedResolution::resolve`, and the final widen is
`AnticipatedResolution::ring_size`, called by `resolve_state_layout`
(`crates/cobre-sddp/src/setup/mod.rs`).

The state region is `S = A·k_max` (see §3.3). The **modular slot key** is keyed
on the **ring axis**, not the raw delivery axis: the delivery axis with each
plant's own fixed post-horizon commitment (if any) excised — identity whenever
the plant declares none. `PointResolution::ring_index`/`physical_target` own
the forward map and its inverse; no new type, no new state.

```
slot(m) = ring_index(m) mod k_max        (slot-major, plant-minor: commitment_hold_in_study_offset)
```

This is injective over the **ring window** in ring-axis terms — the
strictly-future, not-yet-due delivery targets a stage carries, a contiguous run
of `k_max` consecutive ring-axis integers, hence distinct residues mod
`k_max`. Keying on the raw `m mod k_max` instead is the wrong-but-compiling
alternative once a plant declares a fixed post-horizon commitment: the excised
window breaks the raw axis's contiguity, so two in-flight commitments can
collide onto one slot. (The trap: `depth[t]` **excludes** pre-study occupancy,
so it is _not_ the ring's per-stage occupancy boundary — the
interior/deposit/padding split is decided per delivery target, not from
`depth`.)

Two sizing-adjacent gates:

- **Fan-out reject.** If any decision stage anchors more than one delivery
  (`max_fanout > 1` — a coarse decision stage upstream of several short delivery
  stages under `LeadTime`), `resolve_state_layout` returns `SddpError::Validation`
  naming the plant. The _state_ is already fan-out-ready (distinct residues), but
  the LP fill and output extractor assume one decision column per plant per stage,
  so this is a reserved-capability gate and the **sole** fan-out guard.
- **`K = 0` exclusion** (sub-stage lead, `c(m) = m`): excluded from the ring
  entirely and dispatched as ordinary unconstrained thermal generation at that
  stage, with one setup-time advisory per self-delivered stage (never a hard error,
  never a per-scenario log).

### 3.3 State layout

The anticipated ring occupies the whole merged **commitment-hold** region
(`commit_out` / `commit_in`), spanning exactly `A·k_max` slots —
slot-major/plant-minor, keyed by delivery-target residue. There is no separate
appended block: a slot whose delivery target lands past the horizon (`m >=
n_stages`, reachable only when `post_study_stages.json` extends the delivery axis)
is one of the ring's own slots, not a trailing lane. A plant's commitment for
delivery `m` occupies slot `ring_index(m) mod k_max`, so every plant cycles
through all `k_max` slots over the horizon; only a slot the LP never latches
is structurally zero.

`n_anticipated` is the count of thermals with `anticipated_config.is_some()` in
canonical `System::thermals()` order. The stage-0 seed writes
`past_anticipated_commitments` into the outgoing block at `slot·n_ant + local_idx`,
`.take(K_i)` (seeds exactly the plant's own pre-study deliveries), with
ids resolved through a position map (never `binary_search`, which breaks under
staggered commissioning).

### 3.4 LP entry

The commitment transition is realized entirely by three `[0,0]`-equality row
families over the one dense ring `DeliveryRing::anticipated` builds
(`n_lanes = n_anticipated`, `depth = k_max`); `commit_in` is pinned to the
previous stage's `commit_out` by column
bounds, so there is no Rust-side shift step. Every `mod k_max` slot key below is
a **ring-axis** residue (§3.2) — `ring_index(m) mod k_max`, the identity whenever
the plant declares no fixed post-horizon commitment:

- **Latch / deposit** (`fill_anticipated_state_out_def_entries` → `emit_deposit`):
  a plant's fresh decision at stage `t` pins the slot of its **own delivery
  target** — `out_col(delivery_stage mod k_max) − decision_col = 0`. The slot comes
  directly from the decision's delivery stage, never a `depth`-derived boundary.
- **Interior carry / hold** (`fill_anticipated_slot_definition_entries` →
  `emit_carry_rows`): an in-flight, not-yet-due slot is pinned to its **own**
  incoming column — `out[slot] − in[slot] = 0`, the **same slot**. This replaces
  the retired Markov-1 shift (`out[slot] − in[slot+1] = 0`): a commitment does not
  migrate slots, it is _held_ at its delivery-target residue until it matures. (The
  water ring keeps `emit_shift_rows` because its physics genuinely shift — this is
  the single most important difference between the two rings.)
- **Maturity / "fishing"** (`fill_anticipated_fishing_entries`): the delivery
  maturing this stage couples the plant's per-block generation to the matured
  commitment — `Σ_b h_b · gen[b] − H · in_col(stage mod k_max) = 0` (MW → MWh). It
  reads `commit_in` and **never writes `commit_out`** — which is exactly why it
  cannot collide with the same stage's fresh latch. It fires **unconditionally**
  for every maturing delivery: a commissioning-inactive delivery was never latched,
  so its `in_col` is `0` and the equality harmlessly pins that stage's generation
  to `0`. (The forbidden "fish-if-active-else-carry" alternative writes `out_col`
  and collides with a fresh latch on the same residue — a release-silent LP
  corruption.)

The **decision column** is bounded, costed, and commissioning-gated at the plant's
**own delivery stage** (`thermal_block_base(thermal_idx, delivery_stage)`, cost and
discount at `delivery_stage`), never at the decision stage `t` — a decision-anchored
read reintroduces a capacity-drop infeasibility. The anticipated plant's per-block
generation columns carry **no objective coefficient**; fuel is booked on the
decision column so nothing double-counts.

Masking is the ring's **two-sided reachability masking over the whole slot range**
— there is no separate appended block with a rule of its own. Every position keeps
the two-sided discipline the shared skeleton ships together: a masked (unreachable)
position gets **no** definition row _and_ a frozen `[0,0]` outgoing column in the
same pass. `fill_anticipated_slot_columns` (`crates/cobre-sddp/src/lp/builder/columns.rs`)
applies `freeze_masked_columns` over the whole `anticipated_slot_row_pos`: a
reachable post-study-targeted slot is open `(-inf, inf)` because it is **not**
masked (its `row_pos` is `Some`), never because it is exempt from freezing, and the
boundary FCF then prices its carried state directly through the generic `β·state`
projection. (Reachable slots use `(-inf, inf)`, not water's `[0, inf)`, because a
committed MW value carries either sign.)

**Drift reconciliation** (`crates/cobre-sddp/src/solve/stage_solve.rs`,
`assemble_outgoing_state`): a latched `commit_out` is a _basic_ variable produced by
the simplex factorization, so it is accurate only to the backend's
`primal_feasibility_tolerance` (`1e-9`), never 1 ULP; a commitment at its cap arrives
a hair outside it and the no-slack fishing equality would turn that hair into a false
`Infeasible`. The outgoing-state read-back seam projects every outgoing state onto its
admissible box before the value is pinned, solved against, or dotted into a cut, so the
sub-tolerance drift is absorbed silently — no runtime verdict, no telemetry. The clamp
runs at all four solve sites uniformly (forward, backward, lower bound, simulation). A
_genuine_ over-commitment — a declared value outside the delivery stage's resolved
generation box — is rejected before the study runs, at cobre-io load time
(`check_committed_value_bounds`), never on the solve path.

### 3.5 `col_scale` and boundary pricing

`col_scale` is forced to `1.0` across the whole `commit_out ∪ commit_in` region.
The same divide-on-extract / multiply-on-render dual convention as storage and the
water buckets applies. The ring's reachable slots — post-study-targeted ones
included — join the same generic `β·state` cut projection as everything else, no
per-family terminal-pricing arm.

A post-study-targeted slot in a study that declares **no** `config.policy.boundary`
carries **zero** terminal value: the slot still exists and still joins the
`β·state` projection, but the terminal boundary FCF it would price against is
empty, so the term contributes nothing. This is a ratified scope boundary, **never
a reject** — a `min_mw == max_mw` replay deck is a legitimate use of a fixed
post-study profile with no boundary, and rejecting it would abort a valid study.
Setup emits exactly one advisory naming every affected plant
(`setup::warn_on_boundary_absent_post_study_delivery`, on the same `tracing::warn!`
channel as the `K = 0` advisory but a distinct condition — do not conflate it with
`warn_on_sub_stage_lead`, which fires on a sub-stage lead resolving `c(m) = m`).
Silence is the opposite failure: it hides a modelling error where the user expected
the commitment valued against a real future.

---

## 4. Side by side

Both rings share the `DeliveryRing` skeleton, the out-by-identity / in-pinned
column resolution, the two-sided masking discipline, and the dual sign
convention; each occupies its own state region (`Buckets`, `CommitmentHold`).
They differ in the call-site-local ways the table lists:

| Aspect                   | Water travel time                                                                 | Anticipated thermal                                                             |
| ------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Ring instances           | one per downstream plant (`n_lanes = 1`)                                          | one dense ring (`n_lanes = n_anticipated`)                                      |
| Transition               | **shift** (`emit_shift_rows`, `slot → slot+1`)                                    | **hold** (`emit_carry_rows`, same slot)                                         |
| Slot key                 | lag (distance in flight), one plant per ring                                      | delivery-target residue on the **ring axis** (`ring_index(m) mod k_max`)        |
| Deposit                  | `k_d`-weighted release share at the call site                                     | single decision latch (`emit_deposit`)                                          |
| Depth sizing             | overlap measure (`stage_reach`, IC overlap)                                       | `max(occupancy_max, n_none_in_study)` per plant                                 |
| Reachable column bound   | `[0, inf)` (a volume)                                                             | `(-inf, inf)` (a signed MW value)                                               |
| Masked terminal slot     | drops a **genuine** deposited share — only on the horizon-capped no-boundary path | provably **zero** (no commitment targets a masked slot)                         |
| Terminal live-state gate | needs `config.policy.boundary` (not inert)                                        | post-study slot **existence** on declared `post_study_stages.json` + lead reach |

These last two rows carry the deepest asymmetry, and it **survives** the
ring-native re-derivation rather than collapsing into one rule. On the masked slot:
the water ring _discards a real end-of-horizon release share_ (a documented
target-stage imprecision, safe only because a finite horizon's terminal value is
zero), whereas an anticipated masked slot is _provably_ vacuous — no commitment
ever targets it, because the decision-existence gate —
`is_anticipated_decision_active_for_delivery`, the production entry point (the
constant-lead `is_anticipated_decision_active` wrapper is test-only) — keys on the
extended delivery axis `[0, n_delivery)` and a masked slot is unreachable. On the terminal live-state
gate: un-masking the water ring's horizon-capped deep-lag slots re-enables their
definition rows, deposit share, and outgoing columns — a live LP change — so it
**must** be gated on `config.policy.boundary` presence, keeping a no-boundary study
byte-for-byte. The anticipated ring gates the **existence** of a post-study slot
instead — on a declared `post_study_stages.json` extending the delivery axis past
the horizon, plus the plant's own lead reaching that target. A post-study-targeted
slot is one of the ring's **own** reachable slots, held open `(-inf, inf)` whenever
reachable and priced through the same generic `β·state` projection, never a
boundary-conditional block; with no post-study calendar the axis is study-only and
every `m >= n_stages` slot is masked. Water gates terminal live-**state** on a
loaded boundary; anticipated gates post-study slot **existence** on the declared
calendar and lead reach — the same asymmetry, re-derived (`.claude/rules/sddp.md` §
"Delivery-family right-boundary pricing").

---

## 5. The correctness contracts

The invariants that make the above _correct_ rather than merely plausible live in
`.claude/rules/sddp.md`, each pinned to a named regression test:

- **Shared:** the `DeliveryRing` skeleton (one implementation, two rings);
  two-sided masking ships together; out-by-identity / in-pinned column resolution;
  divide-on-extract / multiply-on-render duals; every ring slot priced at every
  pool through the one `β·state` projection.
- **Water:** `Σ_d k_d = 1` (no closed-form ceiling depth); `k_d` replicated across
  cells, never apportioned; canonical bucket ordering; windowed additive IC seed
  with no fallback; terminal-credit-deferred (masked share dropped, never
  misdirected); `boundary_present`-gated terminal live-state; mode-independent
  sizing; shared arrival density (`Σ_b w_b·χ_{b,d} = k_d`); fixed delivery density.
- **Anticipated:** delivery-anchored end-anchored decider; slots keyed by residue
  with a same-slot hold carry; `depth[t]` is not the occupancy boundary; in-study
  maturity always fishes (carry-to-terminal is the post-study-targeted ring slot's
  alone); end-of-horizon masking is exact (never a dropped commitment); fan-out
  reject is the sole guard; `K = 0` is exclude-with-advisory; delivery-anchoring
  preservation (bounds/cost/discount read at the delivery stage); mandatory
  non-parametric drift reconciliation; hour-weighted post-study boundary
  reconciliation.

When changing anything in either subsystem, read the matching `sddp.md` section
first: a plausible-looking deviation in either ring produces wrong bounds, rejected
warm-starts, or silently understated cuts that still compile and pass most tests.
