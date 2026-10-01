# Post-horizon input schema unification

> **Status:** Proposal (not yet implemented). A target design for the input
> surface that declares post-horizon anticipated-thermal commitments. Supersedes
> the `post_study_stages.json` shape shipped in 0.15.0. Timing (whether it lands
> before or after 0.15.0) is an open owner decision — see §9.

**Scope:** how a study declares (a) the post-horizon calendar and (b) the
cost/bounds of an anticipated thermal's post-horizon delivery. Narrows both onto
the surfaces that already own the analogous in-study and pre-study data, and
keys everything by stage `id` instead of an array index.

---

## 1. Problem statement

The post-horizon feature (0.15.0) introduced `post_study_stages.json`, a file
that bundles two unrelated concerns and breaks with every convention the rest of
the input model follows:

```json
{
  "stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
  "thermal_bounds": [
    {
      "thermal_id": 86,
      "post_study_stage_index": 0,
      "cost_per_mwh": 210.0,
      "min_mw": 0.0,
      "max_mw": 350.0
    }
  ]
}
```

Five concrete asymmetries against `stages.json` and the rest of the input model:

1. **Positional index, not identity.** Every other stage reference in cobre is a
   declared `id` resolved through `StageIdResolver` (transitions, nodes,
   `entry_stage_id`/`exit_stage_id`, initial-condition seeding). A post-study
   stage has no `id`; `thermal_bounds` joins to it by `post_study_stage_index`, a
   raw index into the start-date-sorted array. Insert an earlier post-study stage
   and every bound silently re-points. This is the parser-index-vs-canonical-order
   hazard the codebase is otherwise careful about, and it is gratuitous: the
   engine already lifts these stages into id-bearing `Stage` objects on a
   continued calendar (`post_study_calendar_stages` →
   `extended_delivery_stages`), discarding the index immediately.

2. **The post-horizon calendar is the dual of the pre-study calendar, modeled
   nothing like it.** `stages.json` already carries `pre_study_stages[]` =
   `{ id (negative), start_date, end_date, season_id? }` — thin calendar stages,
   no blocks/openings/risk. A post-study stage needs exactly that shape (it is
   never solved). Instead it lives in a separate file with a different date
   convention.

3. **`{start_date, duration_hours}` vs `{start_date, end_date}`.** Two spellings
   of a calendar segment. `duration_hours` is what discounting needs, but
   `end_date` supplies it (`end − start`) exactly as the study calendar already
   works.

4. **A single plant's post-horizon story is scattered across three files with
   three key types.** `thermals.json` (identity, lead, in-study cost/bounds — by
   **id**); `initial_conditions.json` `past_anticipated_commitments` (the fixed
   já-comandada values — by **date window**); `post_study_stages.json`
   `thermal_bounds` (the cost/bounds envelope for the same cells — by **array
   index**).

5. **Mandatory fully-redundant per-cell declaration.** The engine derives _which_
   `(thermal, post-stage)` cells exist from the plant's lead and the calendar
   (`classify_deliveries` / `reaches_post_study`; validation V2 checks the
   declared set tiles the derived set exactly). Yet the user must hand-write a
   full `{cost, min, max}` row for every derived cell, with no fallback to the
   plant's own installed `cost_per_mwh` / `generation`, even for a flat profile.

## 2. Principle

> Post-horizon data is not a new kind of thing. The post-horizon **calendar** is
> the forward dual of the pre-study calendar and belongs beside it in
> `stages.json`, id-keyed. An anticipated thermal's post-horizon **delivery
> envelope** is an attribute of that plant and belongs on the plant in
> `thermals.json`, keyed by stage `id`, defaulting to the plant's own installed
> cost and bounds. Nothing is keyed by array index.

## 3. Target schema

### 3.1 Calendar → `stages.json` `post_study_stages[]`

A sibling of `pre_study_stages[]`, ids continuing **positive** past the largest
study stage id:

```json
{
  "pre_study_stages": [
    { "id": -1, "start_date": "2025-12-01", "end_date": "2026-01-01" }
  ],
  "stages": [/* … study stages, ids 0..N-1 … */],
  "post_study_stages": [
    {
      "id": 120,
      "start_date": "2026-11-01",
      "end_date": "2026-12-01",
      "season_id": 10
    }
  ]
}
```

- Shape is exactly `RawPreStudyStage`: `{ id, start_date, end_date, season_id? }`.
  No blocks, no `num_openings`, no `risk_measure`, no `block_mode` (never solved).
- Ids are any distinct positive integers greater than every study stage id,
  ordered ascending by `start_date`; resolved like any stage id.
- Contiguity (`post_study_stages[0].start_date == study horizon end`; each stage's
  `end_date == next.start_date`) is validated exactly as the study calendar is.
- `duration_hours` is dropped; the delivery discount factor derives hours from
  `end − start`, as the study stages already do.
- The standalone `post_study_stages.json` file is retired.

### 3.2 Thermal delivery → `thermals.json` `anticipated_config.post_horizon_delivery[]`

The plant owns its post-horizon envelope, keyed by stage `id`, with per-field
defaults from the plant:

```json
"anticipated_config": {
  "lead_stages": 7,
  "post_horizon_delivery": [
    { "stage_id": 120, "max_mw": 350.0 },       // cost, min default from the plant
    { "stage_id": 121, "cost_per_mwh": 220.0 }  // override only what differs
  ]
}
```

- `cost_per_mwh` defaults to the plant's `cost_per_mwh`; `min_mw`/`max_mw` default
  to the plant's `generation.{min_mw,max_mw}`. A plant whose post-horizon profile
  is its installed capacity at its own cost declares **`post_horizon_delivery: []`**
  (or omits it) — the engine derives the reached stages from the lead and applies
  the plant's own numbers. Entries appear only where the future differs.
- `stage_id` references a `post_study_stages[]` id (§3.1).
- This requires reshaping `anticipated_config`'s cobre-io parse from the current
  two-variant untagged union (`{lead_stages}` xor `{lead_time_hours}`) into a
  struct carrying the (still mutually-exclusive) lead spelling plus the optional
  `post_horizon_delivery[]`. `cobre_core::AnticipatedConfig` keeps its plain,
  postcard-broadcast-safe derive; only the parse layer changes, and the
  post-horizon envelope resolves into the same `PostStudyThermalBound`-shaped
  data the engine consumes today.

### 3.3 Relationship to `past_anticipated_commitments` (unchanged)

The já-comandada fixed _values_ stay in `initial_conditions.json`
`past_anticipated_commitments` as date windows, mirroring `past_defluences` /
`recent_observations`. A past decision is naturally a dated historical record;
the forward cost/bounds envelope is naturally stage-id-keyed. They are kept
distinct on purpose — merging them is a non-goal (§10) — but the two now share
one consistent identity story: the window's dates resolve against the same
`post_study_stages[]` calendar the envelope's `stage_id` references.

## 4. Semantics & derivation

- **Derived cell existence.** Which `(thermal, post-stage)` cells exist is
  derived from the plant's lead and the extended calendar, exactly as today. The
  input no longer restates that set; it supplies only per-cell _overrides_.
- **Defaulting.** For each reached post-study stage, the effective cell is
  `(cost_per_mwh ?? plant.cost_per_mwh, min_mw ?? plant.generation.min_mw, max_mw
?? plant.generation.max_mw)`, then a `post_horizon_delivery` entry for that
  `stage_id` overrides any field it sets.
- **Internal representation is unchanged.** After resolution the engine still has
  a `PostStudyThermalBound`-equivalent keyed by the (now id-derived) post-study
  stage position; `post_study_calendar_stages` / `extended_delivery_stages` /
  `StateSpace::n_delivery` are unaffected in behavior. The redesign is an
  input-surface change, not a state-space change.

## 5. Validation changes (`cobre-io`)

`validation/semantic/thermal.rs` (the V2–V5 matrix) is re-expressed against stage
`id`s instead of `post_study_stage_index`, and split across its two natural homes:

- **Calendar validation** joins the pre-/study-/post-study contiguity and
  ordering checks already applied to `stages.json` — first post-study
  `start_date` equals the horizon end; ids distinct and greater than every study
  id; calendar tiles contiguously.
- **Delivery validation** (V2 tiling, V3 unreachable-stage exclusion, V4 committed
  value envelope, V5 commissioning-window fixed value) keys on `stage_id`. V2's
  "a bound for every reached cell" weakens to "every `post_horizon_delivery`
  entry references a reached, in-service post-study stage" — existence is derived,
  not required, because the plant's own defaults now cover every reached cell.
- The `check_no_straddling_commitment_window` precondition and the reader/semantic
  split (per-file invariants in the reader; cross-file calendar invariants in the
  semantic validator) are preserved, re-pointed at the new surfaces.

## 6. Affected surfaces

| Crate              | Surface                                                                                                                                                                                                      | Change                                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cobre-core`       | `model/post_study.rs`, `system/{mod,builder}.rs`                                                                                                                                                             | `PostStudyThermalBound` keyed by resolved stage position (not raw index); calendar carried as id-bearing stages                                                         |
| `cobre-io`         | `post_study_stages.rs` (retire), `config/{stages,thermals}.rs`, `validation/semantic/thermal.rs` (the largest consumer surface), `validation/{schema,structural,dimensional}.rs`, `schema.rs`, `pipeline.rs` | Parse `post_study_stages[]` in `stages.json`; parse `post_horizon_delivery[]` in `anticipated_config`; re-key validation                                                |
| `cobre-sddp`       | `setup/mod.rs` (id-resolve the delivery calendar + bounds lookup), `policy/policy_export.rs`, `lp/builder/columns.rs`                                                                                        | Resolve `(thermal, stage_id)` → delivery position via the calendar, not a raw index                                                                                     |
| `cobre-stochastic` | `season_cast/mod.rs` (`post_study_calendar_stages`)                                                                                                                                                          | Consume id-bearing post-study stages                                                                                                                                    |
| `cobre-python`     | bindings + schema export                                                                                                                                                                                     | Regenerate `schemas/` (`stages.schema.json`, `thermals.schema.json`; retire `post_study_stages.schema.json`); output parity is via shared cobre-io parse and unaffected |
| examples / docs    | `examples/deterministic/d55-*`, `anticipated-thermals-and-water-travel-time.md`, `CHANGELOG.md`                                                                                                              | Migrate the d55 deck; update the guide; BREAKING changelog entry                                                                                                        |

## 7. Migration

The transform from the 0.15.0 shape is mechanical:

1. Move each `post_study_stages.json` `stages[]` entry into `stages.json`
   `post_study_stages[]`, assigning ascending positive ids past the last study id
   and converting `duration_hours` to an `end_date` (`start + duration`).
2. For each `thermal_bounds` row, add a `post_horizon_delivery` entry to that
   thermal's `anticipated_config`, replacing `post_study_stage_index` with the new
   `stage_id`; drop any field equal to the plant's own default.
3. Delete `post_study_stages.json`.

The post-horizon feature is **new in 0.15.0**, so if the redesign lands before
0.15.0 has real adopters the migration burden is ~nil (see §9). A converter
(`cobre-bridge`) emitting the old shape updates once.

## 8. Byte-neutrality & determinism

- No golden `parity_hash_*` or `mpi_wire` baseline exercises a post-horizon
  anticipated deck; the only exercisers are the `d55` deck and the anticipated
  unit/analytical tests. The change is **result-neutral by coverage**: after the
  d55 deck is migrated, its solved outputs must be bit-identical to the 0.15.0
  outputs (the internal state space is unchanged — §4). A moved hash is an
  escalation signal, never a re-baseline.
- Moment/geometry derivation is untouched; declaration-order invariance holds
  because the calendar sorts by `(start_date)` and bounds resolve through the
  id-keyed calendar in canonical order.
- `schemas/` regenerates (schema-bearing types change); CI's `schemas` job gates
  the drift.

## 9. Sequencing (owner decision)

The redesign is a breaking input-schema change regardless of when it lands. Two
paths:

- **Ship 0.15.0 as-is, redesign in 0.16.0.** The large, ready 0.15.0 work
  (external scenarios, self-describing checkpoints, the anticipated ring) ships
  now; the redesign is the first breaking change of 0.16.0, before real adoption.
  The awkward schema appears in exactly one release.
- **Hold 0.15.0, redesign first.** The awkward schema never ships and no migration
  ever exists, at the cost of delaying all the other ready 0.15.0 work behind this
  one sub-feature's schema.

This document does not decide the fork; it makes either path executable.

## 10. Non-goals

- **No merge of `past_anticipated_commitments` into the plant.** The fixed-value
  history stays a dated record in `initial_conditions.json` (§3.3).
- **No change to the state space, delivery axis, ring geometry, or boundary
  pricing.** Input surface only.
- **No new discounting/season semantics for post-study stages** beyond what the
  pre-study/study calendar already defines.

## 11. Open questions

- Should post-study `season_id` be required, optional, or omitted? (Pre-study
  makes it optional; post-study never consults seasons in the LP — likely omit.)
- Does any post-horizon delivery legitimately need a cost/bound the plant's own
  declaration cannot express (e.g. a plant not yet in service in-study)? If so,
  confirm the plant's `generation` still provides a sane default or require an
  explicit entry for such plants.
- Confirm `cobre-bridge`'s emitter is the only external producer of the old shape.
