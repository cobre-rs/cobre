# Fixed post-horizon anticipated commitments — design proposal

**Status: Implemented** (retained pending fold into the live spec). This is the
design that answered the design brief for representing DECOMP anticipated
dispatch faithfully on a fixed study-stage set — correct delivery-date
discounting and a carrier-free post-horizon case. The behaviour below has
shipped; the authoritative homes are the live spec
[`anticipated-thermals-and-water-travel-time.md`](anticipated-thermals-and-water-travel-time.md)
and the numerical/algorithm contracts in `.claude/rules/sddp.md`'s
"Anticipated thermal commitments" section. Three choices in it were
owner-ratified and are not open for re-litigation: the input surface (§4), the
sunk-cost booking (§7), and the gap-excised ring indexing (§5).

The brief's two out-of-scope options remain out of scope here: the study horizon
is never extended, and no pre-study decision is ever carried through the ring
into the post-study. The fixed commitment introduced below deliberately never
enters the ring — this design keeps that boundary rather than lifting it.

---

## 1. Summary

One new modelling concept closes every gap in the brief: the **fixed
post-horizon commitment** — a delivery decided before the study that matures
after it (DECOMP's "já-comandada"). It is a declared constant, not a decision:
it never enters any LP, occupies no ring slot, and adds no state dimension. It
is consumed in exactly three places — input validation, a constant fold into
the boundary FCF's cut intercepts, and the outputs.

Everything else the brief demands falls out of machinery that already exists:

- **Real delivery dates.** With the post-study calendar declared at the real
  delivery windows, the existing deciders (`resolve_decider_physical` /
  `resolve_decider_stage_count`) map each study stage to exactly the delivery
  DECOMP assigns it. No new resolver semantics.
- **Correct discounting.** `delivery_cumulative_discount_factors` continues the
  study discount recurrence over the post-study durations, so a delivery placed
  at its real date is discounted at its real date. The mirror-shift's discount
  error is a calendar-content problem, not a machinery problem.
- **Correct valuation of the fixed generation.** The boundary reconciliation
  already joins anticipated slots to source months by real calendar date
  (`resolve_anticipated`'s overlap weights); the fixed values reuse those exact
  weights as a constant intercept fold (§6).

## 2. The delivery taxonomy

On the extended delivery axis `[0, n_delivery)` every delivery of an
anticipated plant falls into exactly one of five classes, determined by the
decider's side of the horizon on each axis end:

| #   | delivered  | decided    | representation                                                   |
| --- | ---------- | ---------- | ---------------------------------------------------------------- |
| 1   | in-study   | in-study   | ordinary anticipated decision + ring deposit (existing)          |
| 2   | in-study   | pre-study  | `past_anticipated_commitments` ring seed (existing)              |
| 3   | post-study | in-study   | ring deposit carried to the terminal, boundary-priced (existing) |
| 4   | post-study | pre-study  | **fixed post-horizon commitment (new — today's E2 reject)**      |
| 5   | post-study | post-study | beyond the study's decision reach; unrepresented (existing)      |

Because the decider is nondecreasing in the delivery index, class 4 is always a
single contiguous run starting exactly at the horizon:
`[n_stages, n_stages + g_i)` for a per-plant width `g_i >= 0` (the plant's
`post_study_reach` `no_carrier` set). `g_i > 0` occurs only in the
full-anticipation regime where every in-study delivery is also pre-study-decided
(class 2), which is exactly the DECOMP shape. This contiguity is an invariant
the implementation asserts, not an assumption.

A class-4 commitment touches three surfaces and nothing else:

1. an input window (§4),
2. a constant fold into each boundary cut's intercept (§6),
3. output rows at its real delivery date (§8).

It has no decision column, no ring slot, no carry row, no fishing coupling, no
state dimension, and no objective term. The study's LPs are byte-identical to
the same study with the fixed values replaced by any other values — only the
boundary intercepts, the reported outputs, and validation see them.

## 3. Faithful calendar resolution (verified, no new machinery)

For the reference deck (`decomp-mar-26-rv2-reduced`: four study stages
2026-03-14 → 2026-05-01, two GNL plants, two-whole-month anticipation), the
faithful post-study calendar is

| index | window                 | class                        |
| ----- | ---------------------- | ---------------------------- |
| 4     | 05-01 → 05-02 (24 h)   | 4 — fixed (0 MW stub)        |
| 5     | 05-02 → 05-09 (Sem 8)  | 4 — fixed (já-comandada)     |
| 6     | 05-09 → 05-16 (Sem 9)  | 4 — fixed (já-comandada)     |
| 7     | 05-16 → 05-23 (Sem 10) | 3 — decided by study stage 0 |
| 8     | 05-23 → 05-30 (Sem 11) | 3 — decided by study stage 1 |
| 9     | 05-30 → 06-06 (Sem 12) | 3 — decided by study stage 2 |
| 10    | 06-06 → June end       | 3 — decided by study stage 3 |

With `lead_stages = 7` (= 4 study stages + 3 fixed stages), or equivalently any
`lead_time_hours` in `[1512, 1680)`, the existing end-anchored decider produces
exactly this mapping, and the four in-study deliveries (indices 0–3, all
pre-study-decided) take the existing class-2 seeds. Both lead modes were
verified arithmetically against `extended_deciders` and
`resolve_decider_physical`; on the faithful axis the stage-count form is exact
by construction (`lead_stages = n_stages + g`), so the bridge needs no
hour-interval tuning.

Because the post-study half of `delivery_cumulative_discount_factors`
continues the study recurrence over these real durations, Sem 10's commitment
is discounted to 2026-05-16 and the June remainder's to its real June window —
the brief's goal 3, with zero new code.

## 4. Input surface and validation

**Owner-ratified: extend `past_anticipated_commitments`.** The field already
means "commitments decided before this study"; both pre-study-decided classes
(2 and 4) belong to it. Windows are allowed to extend past the study horizon
onto the post-study calendar; the validator splits consumption by delivery
side. No new field, no schema shape change (the shared windowed-record form
`{thermal_id, start_date, end_date, value_mw}` is unchanged); the retired
`future_anticipated_deliveries` surface is not resurrected.

The validation matrix for an anticipated plant's windows becomes:

- **V1 (unchanged).** The leading `k_i` in-study delivery stages are tiled at
  coverage 1.0 (`check_anticipated_thermals` via `covers_exactly`).
- **V2 (new — replaces the E2 reject).** The plant's class-4 stages (its
  `post_study_reach` `no_carrier` set) are tiled at coverage 1.0 over the
  post-study calendar, an explicit 0 MW window included (the stub stage is a
  declared zero, mirroring V1's "a committed 0 MW is explicit" convention). The
  rejection message names the uncovered post-study stages and says to declare
  the fixed commitments — not to shorten the lead.
- **V3 (new).** No window may cover a class-3 (in-study-decided) or class-5
  (beyond-reach) post-study stage: a fixed value for a stage the study decides
  is a contradiction, and one beyond the decision reach is unrepresentable.
  This generalizes the current over-coverage rule, whose boundary moves from
  "the leading `k_i` study stages" to "the pre-study-decided window" (classes
  2 and 4).
- **V4 (extended).** `check_committed_value_bounds` applies to the post-horizon
  windows exactly as to the in-study ones: finite, within the plant's declared
  `[min_generation_mw, max_generation_mw]` envelope tolerance.
- **V5 (new).** A non-zero fixed value must lie inside the plant's
  commissioning window at its delivery stage, mirroring the in-study seed rule
  (`check_seed_within_window`): a plant whose window closes cannot deliver
  after it, so a non-zero value there is a modelling error, never a foldable
  constant. Explicit 0 MW tiling windows stay legal on inactive stages.
- **Rule 1 (unchanged).** Class-3 stages still require their
  `PostStudyThermalBound` cells; class-4 stages require none (no decision
  exists to bound).

## 5. Ring geometry: gap-excised ring indexing

**Owner-ratified: excise the fixed window from the ring's index space.**

### The problem the excision dissolves

Excluding class-4 deliveries from the ring breaks the contiguity of the
in-flight set on the raw delivery axis. In the reference deck, occupancy-sized
`k_max = 4` puts the study-stage-3 seed (delivery 3) and the Sem 10 deposit
(delivery 7) on the same residue (`3 ≡ 7 mod 4`) at stage 0 — two definition
rows on one outgoing column, the same corruption class as the ring
under-sizing defect this design closes together with the excision (the sizing
contract now lives in `.claude/rules/sddp.md`, "Ring depth sizing"). The
modular key `m mod k_max` is injective only on contiguous runs, and the raw
axis is no longer contiguous for the ring's members.

### The design

The root cause is that the ring keys its slots on a foreign index space — the
full delivery axis — which now contains entries the ring does not own. The fix
is to key the ring on its own axis: **the plant's ring index is its delivery
index with the fixed post-horizon window excised**,

```text
r_i(m) = m            for m <  n_stages          (in-study: identity)
r_i(m) = m − g_i      for m >= n_stages + g_i    (class 3 and 5)
r_i(m) = undefined    for the class-4 window     (never a ring member)
```

with `slot = r_i(m) mod k_max` replacing `m mod k_max` at every ring
addressing site (deposit targeting, the slot row-position sweep, fishing's
in-study arm — where it is the identity — and the manifest's modular dating).
Because class 4 is a single contiguous run at a fixed position (§2), the
excision is one per-plant integer subtraction, not a table — and it is owned
as two pure methods on the plant's own point resolution (the forward map and
its inverse), derived from data the resolution already carries. No new type,
no new state: a new parallel structure would re-introduce the
foreign-index-space alignment problem the excision exists to remove.

The ring depth becomes `k_max = max(occupancy_max, n_none_in_study)` in excised
space — the second term is the incoming-state window at stage 0 (every seed
present simultaneously before the first fishing), which closes the ring
under-sizing defect this design also fixes; the formula is now the sizing
contract in `.claude/rules/sddp.md`, "Ring depth sizing".

In excised space the in-flight set is contiguous by construction — the property
every pinned injectivity contract rests on becomes true again rather than being
patched around. The reference-deck walk confirms the classic recycling pattern
returns: seeds occupy slots 0–3; each stage fishes its seed and deposits its
post-study decision into the just-freed residue (`r(7) = 4 ≡ 0`,
`r(8) = 5 ≡ 1`, …); the terminal state carries exactly the four decided
deliveries in four slots, eight anticipated state dimensions for two plants,
zero padding.

### Why this and not the alternatives

Two alternatives were considered and rejected:

- **Width-based modular sizing** (`k_max` = the maximal residency span on the
  raw axis, 7 in the reference deck) keeps identity indexing but permanently
  wastes `g_i` masked slots per plant (14 state dimensions instead of 8 here),
  inflates the boundary checkpoint the bridge must author with synthetic
  never-live slots that need sentinel dating, and redefines `k_max` as "span
  including holes" instead of the honest "maximal simultaneous occupancy".
- **A setup-time slot table** (interval coloring) reaches the same minimal
  depth but replaces the modular key everywhere — deposit, carry, fishing,
  `slot_lane_at`'s closed-form inverse, manifest dating — rewriting every
  pinned ring contract and threading a new data structure through all of them,
  for zero additional benefit over the excision.

The excision gets the minimum of both costs: minimal state (the true occupancy,
smaller cut-state vectors, smaller checkpoints), zero hot-path change (one
setup-time constant; every ring structure is precomputed at layout/template
build exactly as today), and contract continuity — the modular key, the
masking machinery, `slot_lane_at`, and the carry/deposit/fishing fills survive
verbatim, with the contracts amended by one clause ("indices are ring-axis
indices; the ring axis is the delivery axis with the fixed post-horizon window
excised; identity whenever no fixed window exists"). When `g_i = 0` — every
existing deck, every shipped golden — the excision is the identity map and the
change is provably byte-neutral.

### Manifest dating amendment

`build_stage_entity_manifest` (and its interval companion
`build_terminal_anticipated_delivery_intervals`) date a slot by the next
in-flight index in its residue class; that search now runs in excised space and
maps back to the physical delivery (`m = r` in-study, `m = r + g_i`
post-study) before resolving the date. In the reference deck the terminal
manifest therefore dates slots 0–3 at Sem 10, Sem 11, Sem 12, and the June
remainder — the real dates the boundary join needs.

## 6. Boundary pricing: the constant intercept fold

The boundary FCF is a function of the committed-generation state at the
horizon. The true state there has two parts: the in-study-decided post-study
deliveries (class 3, live ring slots the cuts' coefficients land on through the
existing date-join rebind) and the fixed class-4 values — constants. A cut
`θ ≥ intercept + Σ β·x` evaluated with a subset of dimensions held constant is
the same cut with those terms folded into the intercept:

```text
intercept' = intercept + Σ_w Σ_M coeff[M] · (overlap_hours(w, M) / H_M) · v_w
```

over each fixed window `w` (value `v_w` MW) and each source anticipated slot
`M` (a source month) overlapping it — precisely the weights
`resolve_anticipated` already computes for live slots (`RebindOp::Blend`'s
`÷H_M` distribute), applied to a constant instead of a state dimension. This
reproduces exactly how the DECOMP←NEWAVE coupling values já-comandadas: the
source FCF's committed-generation coefficients meet the fixed MW at the real
overlap of their calendars.

Mechanics, inside `load_boundary_cuts` alongside `build_rebind`/`rebind_cut`:

- Precompute once per load a fold vector `(source_pos, factor)` where
  `factor = Σ_w (overlap_hours(w, M) / H_M) · v_w`, iterating plants in
  canonical order and each plant's windows in date order — a deterministic,
  declaration-order-invariant sum. Per cut, `intercept += Σ coeff[source_pos]
· factor` — O(fold terms) per cut, once at load, zero hot-path cost.
- **Frame contract:** the fold reads the cut's coefficients in the same frame
  `rebind_cut` reads them, and lands on the intercept before the intercept's
  own load transforms (both applied by `rescale_cut_records_for_load`: the
  cost-scale division and the legacy-ratio rescale) — the folded term is
  future cost exactly like the rest of the intercept and must ride every
  intercept transform with it.
- A fixed window overlapping no source month contributes nothing (the source
  does not price those dates) — mirroring `RebindOp::Zero`, never an error.
- With no `config.policy.boundary` declared there is no fold target: the fixed
  values are inert, never a reject. The single-shot advisory
  (`warn_on_boundary_absent_post_study_delivery`) widens to also name a plant
  declaring a **non-zero** fixed value with no boundary — the user presumably
  expected it priced. All-zero tilings (the mandatory stub) stay silent: a
  zero value is provably inert, so warning on it would be noise.

No new state, no new rows, no interaction with `Renormalize` (class-3 slot
intervals and class-4 windows partition the post-study calendar, so a live
slot never straddles a fixed window).

## 7. Cost booking: sunk

**Owner-ratified: no objective term.** The revision that decided a
já-comandada charged its fuel at that decision — the same convention this
study applies to its own in-study decisions (cost read at the delivery stage
and discounted relative to the decision stage, `D(m)/D(t)`, charged on the
decision column). Re-charging a fixed value
here would double-count across revisions of a rolling study chain. Its effect
on this study is exactly its displacement value, which the boundary fold (§6)
prices. Outputs report the MW (§8); no cost output ever books it.

Constants in an objective cannot change a policy either way — this choice is
about honest reported totals, not optimization.

## 8. Outputs

Fixed post-horizon commitments appear in the anticipated output surfaces at
their real delivery dates, distinguishable from decided deliveries (a source
marker: decided vs fixed), with the values echoed from the input — the
analogue of the `*` rows DECOMP's `relgnl` re-prints. In-study surfaces
(`anticipated_committed_mw`, the carried ring-slot readings) are unchanged.

Fixed commitments are scenario- and stage-independent constants, so they never
enter the per-scenario anticipated-lanes partition (which is built from solver
primals and would repeat the constant per scenario while forcing its non-null
columns nullable). They get a **run-level table** instead — one row per plant ×
fixed window (dates, MW), written once per run by both writers — the CLI and
the Python bindings — under the workspace's output parity rule. The table holds
only fixed rows, so no source-marker column is needed: the file is the marker,
and decided post-study deliveries keep their existing per-scenario report.

## 9. Verification plan

Regressions to pin at implementation, each named for the contract it guards:

- **Faithful resolution end-to-end:** a reference-deck-shaped case asserting
  each study stage's delivery lands at its real post-study date with the real
  duration and the real cumulative discount factor.
- **Excision collision regression:** the stage-3-seed vs first-deposit residue
  collision (`3 ≡ 7 mod 4` on the raw axis) solves correctly under excised
  indexing — distinct seeds, each stage fishing its own value.
- **Under-sizing closure:** the ring-depth under-sizing reproduction
  (`initial_state_seeds_every_leading_commitment_under_the_widened_ring_depth`,
  `n_post < n_stages`, distinct seeds) turns green under
  `k_max = max(occupancy_max, n_none_in_study)`.
- **Fold analytics:** a boundary checkpoint with hand-authored anticipated
  coefficients loads against declared fixed windows and the cut intercepts
  move by the hand-computed `Σ coeff · (overlap/H_M) · v` — including a
  no-overlap window contributing zero, and composition with
  `rescale_cut_records_for_load`'s intercept transforms.
- **Byte-neutrality:** with no fixed window declared (`g_i = 0`), existing
  goldens and parity baselines are bit-identical (the excision is the
  identity; the fold vector is empty).
- **Validation matrix:** V2 rejects an untiled fixed window naming the
  post-study stages; V3 rejects a window covering a class-3 stage; V4 rejects
  an out-of-envelope fixed value; V5 rejects a non-zero fixed value outside
  the plant's commissioning window; the E2 reject message is retired.
- **No-boundary inertness:** fixed values with no boundary leave every LP and
  every output cost untouched and fire the existing advisory once.

## 10. Out of scope

- Extending the study horizon (owner-rejected in the brief; unchanged).
- Carrying a pre-study decision through the ring into the post-study — the
  fixed commitment bypasses the ring precisely to keep this boundary.
- Class-5 deliveries (post-study-decided) remain unrepresented.
- Multi-decider fan-out: the single-decider deposit contract and its setup
  reject are untouched by the excision (the classification is per delivery
  target, not per decision count).

## 11. Bridge follow-up (cobre-bridge, separate repository)

- Build the faithful post-study calendar: the horizon-end stub, the real
  operative weeks, and the month remainders — replacing the mirror-shift
  (`_build_post_study_calendar`).
- Emit `lead_stages = n_stages + g` (exact on the faithful axis) instead of
  the mirror-shift's hour lead; retire `_cobre_safe_lead_hours`.
- Map `dadgnl` `GL` registers to `past_anticipated_commitments` windows on
  both sides of the horizon (in-study months as class-2 seeds, post-study
  weeks as class-4 fixed windows, the stub as an explicit zero).
- Author the boundary checkpoint against the excised ring's state dimension
  and manifest.

## 12. Contract amendments on implementation

This shipped by amending in place (never duplicating): the ring injectivity
and slot-addressing contracts and the manifest-dating contract (the ring-axis
clause, §5); the ring-sizing formula; the validation-layer statements of the
retired E2 reject (now V2/V3); and the live spec
`anticipated-thermals-and-water-travel-time.md` plus the first-timers guide.
The ring under-sizing defect this design closes (§5) was tracked in a design
doc that has since been deleted per the maintenance convention, its residue
being the sizing clause in §5 and the `.claude/rules/sddp.md` sizing contract
it amended.
The design brief this document answered has likewise been deleted once the
behavior landed, per the same convention.
