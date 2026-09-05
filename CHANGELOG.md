# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-header -->

## [Unreleased]

### Fixed

- Preserve the expectation floor in expectation/CVaR mixture weights for
  backward cuts and risk-adjusted cost evaluation. Mixed-risk runs previously
  could overweight expensive scenarios; retrain affected policies and cuts.
  Pure CVaR and expectation retain their mathematical definitions. See the
  `cobre-sddp` risk aggregation contract and its reproducible example.

## [0.15.0] - 2026-08-24

### Added

- **A post-horizon anticipated thermal commitment is now declared in
  `post_study_stages.json` and carried on the anticipated-commitment ring.**
  `post_study_stages.json` gains a `thermal_bounds[]` table — one
  `{thermal_id, post_study_stage_index, cost_per_mwh, min_mw, max_mw}` row per
  post-study delivery cell — and this file is the sole surface for declaring a
  commitment whose delivery lands past the study horizon. A delivery decided
  inside the study but maturing after it now rides the same ring as an in-study
  delivery, over a delivery axis extended across the post-study calendar,
  instead of a separate carrier; its generation is bounded, costed, and priced
  against the extended horizon.

- **A deck may declare commitments decided before the study that deliver after
  it.** `past_anticipated_commitments` windows may now extend past the study
  horizon — the DECOMP "já-comandada" case — with no new field and no schema
  reshape. Such a window is validated like any other: tiled at coverage 1.0
  over every post-horizon delivery stage the plant decided before the study,
  with an explicit `0 MW` window accepted as legitimate coverage. It is priced
  against the terminal boundary by a constant fold into the future-cost cut
  intercepts — a sunk cost, never an objective term — and reported at its real
  delivery date in a run-level fixed-delivery table written by both the CLI
  and the Python bindings.

- **A policy checkpoint is now self-describing.** Every study-global fact a load
  needs — the study graph, the stage count, and the producer provenance — lives
  in a new `policy/manifest.bin` (a FlatBuffers root written last as the commit
  signal and read first behind a format-version gate), and each `cuts/<pool>.bin`
  now carries its own cost-scale factor and graph identity. A checkpoint no longer
  depends on a hand-editable `policy/metadata.json`, which is removed; the Python
  bindings read and write the new layout with an unchanged `load_policy` dict
  shape.

- **A terminal boundary future cost function may be injected from a source study
  whose state shape differs from the loading study's.** A boundary-injection load
  with a legitimately different state dimension — a source with no transit buckets
  and monthly anticipated slots feeding a differently-shaped study — now
  reconciles per state slot by entity identity and delivery date instead of
  rejecting on the state-dimension check. A source cut coefficient that couples a
  state slot the loading study does not model is reported with a named warning per
  family and the load still succeeds, where the previous surface discarded it
  silently.

### Changed

- **BREAKING — every policy checkpoint written by an earlier release must be
  re-exported; none loads as-is.** The checkpoint format is now self-describing
  (see Added): the study-global metadata moved from `policy/metadata.json` to
  `policy/manifest.bin`, and each `cuts/<pool>.bin` gained the self-describing
  cost-scale and identity fields every load path reads. A checkpoint predating
  this format — warm-start, resume, simulation-only, or terminal boundary
  injection alike — is rejected by name with no fallback and no in-place upgrade;
  re-export it with this release, or retrain. This supersedes the two
  configuration-scoped rejections below: it applies to every earlier checkpoint
  regardless of the study's configuration.

- **BREAKING — a policy checkpoint written by an earlier release for a study
  that carries post-study anticipated deliveries is rejected when loaded for
  warm-start or resume.** Moving a post-study delivery onto the ring changes the
  anticipated-commitment ring's state dimension and its per-slot manifest
  identity, so such a checkpoint no longer matches the current layout and fails
  the state-dimension check with a named error. There is no in-place upgrade —
  retrain to produce a current-version checkpoint. This state-dimension change is
  a second, configuration-specific reason such a study cannot be upgraded in
  place, on top of the universal format re-export above.

- **A post-study delivery is now commissioning-gated at its own delivery
  stage.** The delivery-anchored commissioning gate extends over the post-study
  calendar through a continued stage-id sequence, so a plant whose service
  window ends inside the study is inactive post-horizon and can no longer be
  committed to a post-horizon delivery. The previous post-horizon surface
  applied no commissioning gate at that stage, so a plant exiting inside the
  study could still be committed post-horizon; a decommissioned plant committing
  post-horizon generation is now the modeling error the gate catches.

- **The anticipated-commitment ring is sized from full in-flight occupancy,
  including pre-study seeds.** Ring depth now counts every carried delivery in
  flight at a stage — a pre-study-seeded commitment included — instead of a
  quantity that structurally excluded pre-study occupancy. This is
  byte-identical on every well-behaved calendar and on every shipped case, but a
  study whose ring was previously under-sized — a physical-lead plant with more
  pre-study commitments in flight than the old depth counted — gets a wider
  state vector, a different checkpoint, and different, correct results: the old
  depth silently under-sized the ring and pinned a freshly-costed decision onto
  a carried seed.

- **Two carrier-absent configurations are now rejected at load instead of
  silently mis-modeled.** An anticipated thermal whose lead reaches a post-study
  stage for which `post_study_stages.json` declares no `thermal_bounds[]` cell
  is rejected, naming the plant and the post-study stage — where the previous
  surface dropped the delivery with only a log warning. A commitment decided at
  a pre-study stage whose delivery lands past the horizon has no ring slot to
  carry it and is rejected, naming the plant.

- **BREAKING — a policy checkpoint written by an earlier release for a study
  whose ring depth the sizing fix above widens no longer loads.** The
  corrected ring depth changes that study's committed-state dimension, so such
  a checkpoint fails the state-dimension check on load and must be
  regenerated — it encoded the wrong committed value under the old sizing. A
  study whose pre-study run never exceeded its in-flight occupancy keeps its
  state dimension, though its checkpoints, like all others, still require the
  format re-export above.

- **Under the External sampling scheme, a class's realized values now come
  from its external scenario file, including the deterministic (σ = 0)
  case.** Load, NCS, and inflow validation and standardization derive their
  mean and standard deviation from the external scenario samples themselves
  rather than from the seasonal-statistics files, so a seasonal-stats file is
  optional under External. A constant (σ = 0) external column is accepted for
  load, NCS, and an AR(0) inflow model (no declared lag coefficient or annual
  component); for an AR(p > 0) inflow model a deterministic column is still
  rejected, and the message now states the real reason — the column would
  have to equal that model's own deterministic output — instead of a message
  about requiring a positive standard deviation.

### Removed

- **BREAKING — `initial_conditions.json`'s `future_anticipated_deliveries[]` is
  removed.** A post-horizon delivery is declared in `post_study_stages.json`
  (see above), now the sole post-horizon declaration surface. An
  `initial_conditions.json` still carrying `future_anticipated_deliveries[]` is
  rejected by the file's deny-unknown-fields contract; move each entry to a
  `post_study_stages.json` `thermal_bounds[]` row.

### Fixed

- **An anticipated thermal whose lead reaches the full study horizon no longer
  panics the LP build.** Such a plant produced an empty commitment ring while
  still emitting a commitment-maturity row, and the maturity fill divided by the
  zero ring depth and aborted the run — validation passed, so the crash surfaced
  only at run time. Sizing the ring from full occupancy keeps a carried
  delivery's depth positive, and the maturity fill is now gated on a positive
  ring depth, so the divide is unreachable by construction.

- **The anticipated ring depth now sizes as the larger of the in-flight
  occupancy and the pre-study seed run.** A configuration whose pre-study run
  outlived the in-flight occupancy previously sized the ring one or more slots
  too shallow, silently aliasing a later delivery stage's committed value onto
  an earlier one. A configuration where the occupancy term already dominated
  is unaffected.

### Migration

Moving a study from 0.14 to 0.15:

1. **Re-export or retrain every policy checkpoint.** The checkpoint format is now
   self-describing (`policy/manifest.bin` replaces `policy/metadata.json`, and
   `cuts/<pool>.bin` carries its own cost-scale and identity fields), so a
   checkpoint written by an earlier release is rejected by name on every load
   path — warm-start, resume, simulation-only, and terminal boundary injection
   alike — with no in-place upgrade. A study that carries post-study anticipated
   deliveries, or whose corrected ring depth widens (see above), additionally
   needs a genuine retrain, since its state dimension changed.
2. **Re-declare post-horizon deliveries.** Move every `initial_conditions.json`
   `future_anticipated_deliveries[]` entry to a `post_study_stages.json`
   `thermal_bounds[]` row keyed by `(thermal_id, post_study_stage_index)`,
   supplying `cost_per_mwh`, `min_mw`, and `max_mw`. A cell missing for a plant
   whose lead reaches that post-study stage now fails the load, naming the plant
   and stage.

## [0.14.3] - 2026-08-19

### Added

- **`cobre.write_policy_checkpoint` accepts an `inflow_lag_depth` argument so a
  boundary policy authored outside cobre reserves the canonical inflow-lag state
  slots.** When set to `N > 0`, cobre widens every stage's manifest with `N`
  `HydroInflowLag` slots per storage hydro — in its own canonical layout — and
  places each cut's per-hydro `inflow_lag_coefficients` at their `(hydro, depth)`
  positions, so the checkpoint self-describes depth `N` and the load path
  reserves the matching forward lag state. This lets a case with no autoregressive
  inflow model to infer the depth from (explicit-scenario inflows) carry an
  inflow-lag-coupled terminal boundary cost-to-go that would otherwise be
  silently dropped. Absent or `0`, the written checkpoint is byte-identical.

## [0.14.2] - 2026-08-19

### Changed

- **Internal: the SDDP training loop reuses pre-allocated buffers on two hot
  paths that previously allocated on every iteration.** The forward pass's
  cross-rank per-stage stats aggregation now reuses buffers held on the per-run
  iteration scratch, and the interior cut-generating node set is resolved once
  per run instead of being re-derived on every cut-selection cycle. Byte-neutral:
  solver output, cuts, bounds, and run-to-run and declaration-order determinism
  are unchanged.

## [0.14.1] - 2026-08-17

### Added

- **A `gap` stopping rule is now admitted under a CVaR risk measure when the
  forward selection is `enumerated` and the measure is uniform across stages.**
  An enumerated forward visits every path, so its upper bound is computed
  exactly — aggregated with the study's own CVaR weighting through a nested,
  time-consistent backward recursion over the scenario tree, the same measure
  the cuts and lower bound already use. That risk-adjusted bound brackets the
  risk-averse lower bound (a risk-neutral bound never would), so the optimality
  gap stays non-negative and closes only at true convergence. Sampled forwards
  and stage-varying measures remain rejected; `Expectation` runs are
  byte-identical.

### Changed

- **Both the minimum- and maximum-outflow rows now bind the non-diverted
  river-remnant flow — turbine plus spill — and exclude the diversion channel.**
  A diversion routes water to a different downstream target and is governed
  solely by its own `max_diversion_m3s` bound, so a plant that diverts can no
  longer meet its minimum outflow with diverted water, and its diversion is no
  longer double-capped by the maximum-outflow row; the two bounds are now
  symmetric. Byte-neutral on any non-diverting deck — the diversion column is
  pinned to `[0, 0]` and presolve-eliminated.

### Removed

- **The `state_space` config section (`state_space.inflow_lag_depth`) is
  removed — the inflow-lag state depth is now always inferred from a loaded
  boundary policy's cuts.** The field was an optional override that only ever
  widened the depth _beyond_ what the boundary and PAR model already require;
  the inference covers every case, so the knob is redundant. A `config.json`
  that still declares `state_space` now fails to load with an unknown-field
  error (naming the section) rather than being silently ignored — delete the
  section; the depth is resolved automatically.

### Fixed

- **A loaded terminal boundary future cost function no longer produces an
  invalid lower bound (a persistent negative optimality gap) for a study whose
  stages disable inflow-lag cut-state.** The backward cut-intercept dot paired a
  reduced projection's coefficients against the full-length trial state
  positionally, so a commitment coefficient multiplied a lag value and biased
  the intercept — leaving cuts too high and driving the lower bound above the
  upper bound. The trial state is now gathered through the projection. Studies
  whose stages keep the full state projection are byte-identical.

## [0.14.0] - 2026-08-13

### Added

- **`cobre.write_policy_checkpoint(...)`** writes a policy checkpoint to disk
  from plain Python dicts/sequences, mirroring the shape
  `cobre.results.load_policy` returns so a loaded checkpoint round-trips (load,
  edit, write). It exposes the existing Rust writer, keeping the on-disk
  FlatBuffers layout single-sourced, and lets an external producer author a
  checkpoint from raw cut data. It validates each cut's coefficient count
  against the stage's `state_dimension` and each stage's state-data length
  before writing.

- **A study's temporal structure can declare a policy graph directly, rather
  than only an implicit stage chain.** `stages.json`'s `policy_graph.nodes[]`
  lists one entry per decision point — `id`, `stage_id` (the study stage the
  node sits at), an optional `scenario_id` (the column this node selects from
  a slot-occupying external-scenario class), and an optional `label` — and
  `policy_graph.transitions[]`'s `source_id`/`target_id` become node ids once
  `nodes[]` is non-empty (they stay stage ids otherwise, unchanged). A study
  that declares no `nodes[]` is a stage chain exactly as before, byte-for-byte.
  `nodes[]` lives in `stages.json` (JSON, not Parquet) because it is
  structure, not bulk: the declared graph is `O(K)` nodes for a fan of width
  `K` or `O(T + K)` for a recombining hybrid over a `T`-stage horizon, and a
  fully-enumerated `K^T` scenario tree is never written out as an explicit
  `nodes[]` array — enumeration computes the root-to-leaf path count with
  checked arithmetic, and a `u64` overflow is a hard setup error. A future
  input whose size scales with entities, stages, blocks, and scenarios still
  belongs in Parquet; `nodes[]` is the JSON side's one bounded exception.
  `transitions[].probability` is the only place a transition probability is
  declared — there is no `scenario_probabilities.parquet`, and none is
  planned.

- **Declaring a user-supplied opening tree together with a node graph, or
  under enumerated forward selection, is rejected.** Setting
  `training.scenario_source.openings` to `{"source": "file"}` (which reads
  `scenarios/noise_openings.parquet`) conflicts with a declared
  `policy_graph.nodes[]`, since `nodes[]` already declares the opening set,
  and conflicts with an enumerated `training.selection.method`, which
  consumes external columns rather than a generated or file-sourced opening
  tree. Both combinations fail to load, naming the conflicting declaration;
  declare one or the other.

- **Three new external-scenario Parquet files carry pre-computed realizations
  for a node graph's `scenario_id` columns:**
  `scenarios/external_inflow_scenarios.parquet`,
  `scenarios/external_load_scenarios.parquet`, and
  `scenarios/external_ncs_scenarios.parquet`, one row per
  `(stage_id, scenario_id, entity_id)`. The loader rejects, rather than
  silently repairs, several malformed shapes:
  - a `scenario_id` set that is not exactly `{0..raw_c(t)-1}` for a given
    entity and stage — a 1-based set such as `{1, 2}` would otherwise load,
    aliasing stage `t+1`'s realization 0 onto the current stage and
    zero-filling realization 0;
  - a row whose `stage_id` names no declared study stage — otherwise
    silently dropped rather than reported;
  - two slot-occupying external classes disagreeing on their per-stage raw
    column count — otherwise reconciled by taking the element-wise minimum,
    silently truncating the wider class's scenario set to the narrower
    one's.

- **The `gap` stopping rule and the exact upper bound it depends on.**
  `training.stopping_rules[]` accepts a `"gap"` entry with `tolerance`
  and/or `relative_tolerance`, stopping once the clamped canonical-R$ gap
  between the exact upper bound and the lower bound satisfies either
  tolerance arm. It is admitted only under enumerated forward selection with
  an expectation risk measure at every stage; declaring it under sampled
  forward selection fails with a named error stating that a gap rule admits
  only enumerated forward selection, and declaring it alongside an
  effectively risk-averse measure at any stage fails with a named error
  identifying the offending stage and stating that a gap rule admits only an
  expectation risk measure at every stage. `training/convergence.parquet`
  gains `upper_bound_kind`: `"exact"` when the enumerated weighted bound is
  in force, `"statistical"` otherwise.

- **A `by_node` backward-pass scheduler, alongside the default
  `by_scenario`.** Setting `training.parallelism.backward_scheduler.method`
  to `"by_node"`, with an optional `block_size`, reassigns backward-pass work
  from a whole trial point to a `(trial point, opening-block)` tile;
  `block_size` sizes the tile and defaults to half the stage's opening
  count, rounded up.

- **`node_id` on every simulation entity row, and a new
  `simulation/paths.parquet`.** Every `simulation/` entity Parquet file's row
  now carries `(scenario_id, stage_id, node_id)` instead of `stage_id` alone;
  `node_id` is the visited node's declared id, and on a stage chain (no
  declared `nodes[]`) it equals `stage_id` — the degenerate case, so a
  consumer reads the column the same way regardless of whether the study
  declared a graph. The new run-level, unpartitioned `paths.parquet` records
  the node path each simulated scenario visited, one row per
  `(scenario_id, stage_id)`.

- **Named expressions in generic constraints.** Declare a linear combination
  once and reference it by `@name` anywhere an expression or a bound is written;
  references are inlined at load with cycle detection and coefficient
  distribution, so the solver still sees a flat term list. A net line flow can be
  addressed by its `(source_bus, target_bus)` pair.

- **An inline relational right-hand side for generic constraints.** A
  constraint's `expression` may be written as `lhs <op> rhs` (`<=`, `>=`, `==`).
  At load the two sides are normalized onto the constraint's interval: every
  variable term folds onto the left, and constants and parameters fold onto the
  bound. A parenthesised group scaled by a literal coefficient distributes
  (`0.5 * (a - b)`). The resolved LP is byte-identical to the hand-flattened
  one-sided form.

- **A per-`(stage, block)` axis for scalar parameters, usable in bound
  positions.** A parameter may vary by block as well as stage, and a parameter
  may supply a constraint's bound (resolved per block) — so a value declared once
  is referenced by `@name` in many bounds instead of being pre-baked into every
  row.

- **Resolved generic constraints are written to
  `generic_constraints/resolved_echo.parquet`,** one row per
  `(constraint, stage, block, term)`, by both the CLI and the Python bindings —
  the fully desugared flat form the solver receives, for debugging the authoring
  layer and comparing against an external producer.

- **A minimum-flow floor for hydro diversion and a min/max band for spillage,**
  authored as per-`(stage, block)` bound columns alongside the existing entity
  bounds.

- **A terminal boundary future cost function loaded from an external policy
  checkpoint (a DECOMP-style right boundary).** `config.json`'s `policy.boundary`
  points at a previously trained checkpoint (`path`, with an optional
  `source_stage`); its cuts are injected as fixed boundary conditions at the
  study's terminal stage, so the horizon is priced against an externally supplied
  continuation value instead of a zero terminal value. When `source_stage` is
  omitted it is auto-resolved by calendar overlap against the terminal window.
  A source anticipated-thermal or water travel-time delivery state is reconciled
  onto the current study's calendar by dated, hour-weighted fan-out (a monthly
  source delivering onto a weekly/monthly study), and a per-family reconciliation
  summary is reported at load. The source is addressed by a single leaf pool; a
  multi-node source is rejected.

### Changed

- **BREAKING — the policy checkpoint format gains a required version marker;
  a checkpoint written before this release fails to load.** `metadata.json`
  now carries `format_version`, checked before any cut, basis, or state
  payload is parsed; each `.bin` payload now carries a FlatBuffers
  `file_identifier` (`"CBVF"`). A checkpoint written by an earlier release
  has neither, and loading it now fails with a named error citing
  `format_version` and reporting the expected value against the value found
  in the checkpoint — there is no conversion path. Retrain to produce a new
  checkpoint. The per-slot entity manifest's delivery marker is also
  retyped: `EntitySlot`'s `delivery_anchor` (a month-level integer, its
  calendar encoding left to the caller) is replaced by `delivery_date`, a
  `YYYYMMDD` calendar date owned by the format itself.

- **BREAKING — `initial_conditions.json`'s `past_anticipated_commitments`
  are windowed, delivery-anchored records, not a per-lead-stage value
  array.** Each entry is now a `{thermal_id, start_date, end_date, value_mw}`
  record — the externally-decided MW rate held constant over the window it
  declares — replacing the previous `{thermal_id, values_mw: [...]}` shape,
  whose values were implicitly ordered by delivery lead rather than dated. A
  deck still carrying `values_mw` is rejected by the file's
  deny-unknown-fields contract; re-emit the commitment history as dated
  windows.

- **BREAKING — every simulation output file spells each axis with exactly
  one canonical name, and `stage_id` never carries a `-1` sentinel.** Bare
  `stage` is rejected in favor of `stage_id`, bare `opening` in favor of
  `opening_index`, and `upper_bound_mean` in favor of `upper_bound` (a
  non-nullable float on `training/convergence.parquet`, unchanged in
  meaning). A not-applicable `stage_id` — a lower-bound-phase row on
  `training/solver/iterations.parquet`, for instance — is now `NULL` rather
  than the previous `-1` sentinel; a consumer that filtered on
  `stage_id != -1` must switch to a null check.

- **The per-transition discount-rate override moves to `stages[]` under a
  declared node graph.** `transitions[].annual_discount_rate_override` is
  rejected once `policy_graph.nodes[]` is non-empty — declare
  `stages[].annual_discount_rate_override` instead, a per-stage quantity
  rather than a per-edge one. The chain dialect (no declared `nodes[]`) is
  unaffected and keeps the transition-level override exactly as before.

- **BREAKING — an unrecognized `risk_measure` string is rejected instead of
  silently training risk-neutral.** `stages[].risk_measure` accepts only the
  string `"expectation"` or a `{"cvar": {...}}` object; any other string (a
  typo such as `"cvar"` without the object form) now fails to load, naming
  the offending value and the accepted set. Previously any string parsed
  successfully and was treated as `"expectation"` regardless of its content.

- **BREAKING — a `policy_graph.type` of `"cyclic"` is rejected as
  reserved.** No engine consumer implements the cyclic (infinite-horizon,
  periodic) mode; a deck declaring it now fails to load naming the
  reservation, instead of parsing successfully and then silently training
  as a finite-horizon chain with a zero terminal value — the declared
  cyclic type was never read by anything downstream.

- **BREAKING — the scalar-parameters input moved from
  `system/scalar_parameters.json` to `constraints/generic_parameters.json`,**
  beside the generic constraints that reference its `@name` parameters. The
  file's contents and schema are unchanged; only its directory and filename
  changed. A case still carrying `system/scalar_parameters.json` now fails to
  load with an error naming `constraints/generic_parameters.json` — a clean
  break with no silent fallback.

- **BREAKING — a generic constraint's shape is derived from its bounds; the
  authored `sense` field is removed.** `constraints/generic_constraints.json` no
  longer carries `"sense"` (a constraint is `{id, name, description, expression,
slack}`). The shape comes from which endpoints
  `constraints/generic_constraint_bounds.parquet` supplies per `(stage, block)`,
  now two nullable columns `bound_lower` / `bound_upper` in place of the single
  required `bound`: lower-only is `>=`, upper-only is `<=`, both-equal is `==`,
  and both-present-and-differing is a two-sided range. A file still carrying
  `"sense"`, or a bounds table still carrying a single `bound` column, fails to
  load. This lets a constraint's active sides vary by period — one side unbounded
  in some periods and present in others — without a sentinel value.

### Removed

- **BREAKING — three legacy Parquet input-column spellings are rejected;
  each concept now has exactly one accepted name.**
  `scenarios/external_ncs_scenarios.parquet` requires `availability_factor`
  (the legacy `value` is rejected); `constraints/penalty_overrides_ncs.parquet`
  requires `ncs_id` (the legacy `source_id` is rejected);
  `constraints/pumping_bounds.parquet` requires `pumping_station_id` (the
  legacy `station_id` is rejected). A file still carrying the old name fails
  to load with a named error reporting the required column by name; rename
  the column, keeping the data in the same cell.

- **BREAKING — the `trial_point` / `opening_block` backward-scheduler
  spellings are gone, with no alias.**
  `training.parallelism.backward_scheduler.method` is now `by_scenario`
  (renamed from `trial_point`, still the default) or `by_node` (renamed
  from `opening_block`); a config still using either old spelling fails to
  load as an unknown enum variant.

- **BREAKING — the per-stage `num_scenarios` field is gone; `num_openings`
  replaces it.** `stages[].num_openings` names the same within-node opening
  count `num_scenarios` used to; a `stages.json` still carrying
  `num_scenarios` is rejected as an unknown field.

- **BREAKING — the root `training.forward_passes` and flat
  `simulation.num_scenarios` count aliases are gone; `selection` is the sole
  spelling.** `training.selection.forward_passes` (under a sampled method)
  and `simulation.selection.num_scenarios` (under a sampled method) are the
  only accepted homes for the count; a config declaring it at the old flat
  location is rejected as an unknown field.

### Fixed

- **Reservoir evaporation is scaled by the stage's calendar month, not its stage
  duration.** The mm·km²/month evaporation rate is converted to m³/s by the
  calendar month's hours, so a stage deposits only its own share of the month's
  evaporation rather than a full month's worth on every stage.

- **A commissioning-dormant FPHA hydro no longer aborts the LP build.** A plant
  whose FPHA production model is not yet commissioned at a stage is gated out of
  the FPHA index and priced as dormant, instead of reaching an unreachable path.

- **The Windows (`x86_64-pc-windows-msvc`) wheel builds.** The qhull convex-hull
  shim's diagnostic capture used the POSIX-only `open_memstream`, which left an
  unresolved symbol at link time on MSVC; it now falls back to a temporary file
  on Windows.

### Migration

Moving a study from 0.13 to 0.14, in order:

1. **Retrain.** A checkpoint from an earlier release fails to load (a
   `format_version` mismatch); there is no in-place upgrade for a policy
   artifact.
2. **Rename the removed config spellings.** `stages[].num_scenarios` becomes
   `num_openings`; the root `training.forward_passes` alias becomes
   `training.selection.forward_passes` under a sampled selection; the flat
   `simulation.num_scenarios` alias becomes
   `simulation.selection.num_scenarios` under the same shape;
   `training.parallelism.backward_scheduler.method` values `trial_point` /
   `opening_block` become `by_scenario` / `by_node`.
3. **Rename the three legacy Parquet columns**, keeping the data unchanged:
   `scenarios/external_ncs_scenarios.parquet`'s `value` becomes
   `availability_factor`; `constraints/penalty_overrides_ncs.parquet`'s
   `source_id` becomes `ncs_id`; `constraints/pumping_bounds.parquet`'s
   `station_id` becomes `pumping_station_id`.
4. **Re-emit `past_anticipated_commitments`** as dated windows (`start_date`,
   `end_date`, `value_mw`) instead of a `values_mw` array.
5. **Adjust any deck relying on a shape that now rejects**: a 1-based or
   gapped external `scenario_id` set, an out-of-range `stage_id` row,
   disagreeing per-stage external counts across classes, a misspelled
   `risk_measure` string, a `policy_graph.type` of `"cyclic"`, or
   `training.scenario_source.openings` set to `{"source": "file"}` declared
   alongside `policy_graph.nodes[]` or under enumerated forward selection —
   each of these previously loaded and produced a wrong or ignored result
   rather than an error.
6. **Move the scalar-parameters file** into the constraints directory,
   contents unchanged:
   `mkdir -p <case>/constraints && git mv <case>/system/scalar_parameters.json <case>/constraints/generic_parameters.json`.
7. **Convert generic constraints to the bounds-derived shape.** Drop `"sense"`
   from every `constraints/generic_constraints.json`. In
   `constraints/generic_constraint_bounds.parquet`, replace the single `bound`
   column with nullable `bound_lower` / `bound_upper`: put the old value in
   `bound_upper` for a former `<=`, in `bound_lower` for a former `>=`, and in
   both for a former `==`.

## [0.13.0] - 2026-07-30

### Added

- **A hydro plant can declare more than one turbine group, each on its own
  bus.** Every entry in `unit_groups` carries its own `id`, `name`, `bus_id`,
  and generation/turbined envelope, so one plant's units can span several
  electrical buses. `constraints/hydro_unit_group_bounds.parquet` overlays
  stage-varying, optionally per-block overrides on a group's four declared
  bounds (`min_turbined_m3s`, `max_turbined_m3s`, `min_generation_mw`,
  `max_generation_mw`); a group with no override reads its declared value.

- **`thermal_bounds.parquet`, `hydro_bounds.parquet`, `line_bounds.parquet`,
  `pumping_bounds.parquet`, and `contract_bounds.parquet` gain a per-block
  axis.** An optional `block_id` column selects one block within a stage for
  the row's override, including the per-block `contract_bounds.price_per_mwh`
  a study's simulation cost path now honors. A study with no per-block rows
  produces byte-identical resolved bounds and output, so this is not a
  breaking change. Thermal `cost_per_mwh` is deliberately not block-eligible
  — a per-block value is rejected at validation — unlike contract
  `price_per_mwh`, which is: commitment is a stage-level decision, so a
  per-block dispatch cost has nothing to attach to.

- **A generic constraint can address one bus of a hydro plant split across
  several unit groups.** The `hydro_turbined` and `hydro_generation`
  variable references accept a named `bus=` selector, e.g.
  `hydro_turbined(5, bus=2)` — an optional positional block argument, when
  present, precedes it — resolving to the LP column for that `(hydro, bus)`
  cell.

- **Simulation output gains a `simulation/hydro_bus_generation/` partition,
  reporting turbined flow and generation per `(hydro, bus)` cell.**
  `simulation/hydros/` is unchanged and keeps reporting each plant's total: a
  plant split across buses has no per-group quantity to report, since every
  split of a shared cell's flow across its same-bus groups is an equally
  optimal solution with no dual to distinguish them. The new partition
  reports at the bus-cell granularity the LP itself solves.

### Changed

- **A negative realized inflow is accepted again, in both
  `scenarios/inflow_history.parquet` and `initial_conditions.json`'s
  `recent_observations`.** Both parsers reject only non-finite values;
  `value_m3s < 0.0` loads. The quantity is _incremental_ inflow — a plant's
  natural flow minus its upstream plants' — so a negative window is real
  hydrology (a reach that loses water over the window), and the LP already
  prices it through the inflow non-negativity slack. Semantic validation
  reports one warning per file naming the negative count and the most
  negative value with its hydro, so a genuinely sign-flipped series still
  stands out. Any file that loaded before still loads.

- **BREAKING — `scenarios/inflow_history.parquet` is now windowed, and the
  legacy point-dated layout is rejected outright.** Every row carries
  `hydro_id`, `start_date` (inclusive), `end_date` (exclusive), and
  `value_m3s` — the mean inflow observed over that window — replacing the
  previous single-`date`-per-row layout; windowed `inflow_history` is the
  sole record layout the loader accepts. A file still carrying the legacy
  `date` column is rejected at load, with no inference or conversion path
  back to the windowed form — every case must re-emit its history as dated
  windows before it will load.

- **BREAKING — `unit_groups` is now mandatory on every hydro.** A
  `hydros.json` entry with an absent, `null`, or empty `unit_groups` array is
  rejected at load; every hydro must declare at least one group with its own
  `id`, `name`, `bus_id`, and generation/turbined bounds. The exported
  schema marks the key required, matching the loader.

- **BREAKING — the top-level `hydro.bus_id` field is removed.** A hydro's
  bus association now lives exclusively on `unit_groups[].bus_id`; the plant
  itself no longer carries one. A deck still carrying the top-level field is
  rejected by `hydros.json`'s deny-unknown-fields contract.

### Removed

- **BREAKING — `past_inflows` is gone from `initial_conditions.json`.** The
  positional `past_inflows` array — backed by the `HydroPastInflows` type,
  which paired a raw per-lag value list with a parallel `season_ids` array —
  is removed entirely, along with its dedicated coverage and season-id
  validation rules. An `initial_conditions.json` still carrying `past_inflows`
  is rejected at load: the file's existing deny-unknown-field contract turns
  the field into a named parse error rather than a silent partial load.
  Seeding the PAR lag chain and the mid-period accumulator from history no
  longer reads a positional array at all — it derives from the windowed
  `inflow_history` record above, shadowed day-wise by `recent_observations`
  wherever the two overlap.

- **BREAKING — `constraints/exchange_factors.json` is removed.** Per-block
  transmission-line capacity is now declared as absolute MW directly on
  `constraints/line_bounds.parquet`, via `direct_mw`/`reverse_mw` rows
  carrying a `block_id`: `direct_mw = base_capacity × direct_factor` (and
  the reverse equivalent) reproduces the multiplicative form's effect per
  `(line, stage, block)`. A study still carrying the file is rejected at
  load, naming the replacement. Absolute capacity also expresses
  `direct_mw = 0.0` — a line fully closed in one block — which the previous
  factor, constrained strictly positive, could never represent.
  `scenarios/load_factors.json` and `scenarios/non_controllable_factors.json`
  are unaffected: a factor scaling an authored stage quantity is redundant
  once its bounds carry a block axis, as line capacity's now do, while a
  factor scaling a generated quantity — load and non-controllable-source
  availability are sampled and occupy their own noise-vector dimensions —
  remains the only way to express that quantity's block split.

### Fixed

- **Resolved bound, penalty, and factor overrides could be applied to the
  wrong entity when a system's commissioning dates ordered its buses, lines,
  hydros, thermals, pumping stations, energy contracts, or non-controllable
  sources differently from their declared IDs.** An override declared for a
  given entity now always resolves onto that same entity. Every case shipped
  or referenced by this project already orders entities identically under
  both the ID and the commissioning-date order, so no existing output changes.

- **A user-supplied `scenarios/noise_openings.parquet` no longer panics a
  study that has non-controllable sources.** The reader sized the opening
  tree at `n_hydros + n_load_buses`, omitting the NCS block, while the
  samplers slice that same tree at the NCS class offset — so any case with
  NCS entities indexed past the end of a row and aborted with a slice-range
  panic instead of a diagnostic, and padding the file to its true width was
  rejected by the same short dimension. The noise-vector layout now has a
  single owner (`cobre_stochastic::noise_entity_order`) that the reader and
  the context builder share, so a full-width openings file loads and a
  genuinely mis-sized one is reported as a dimension mismatch. A study with
  no NCS entities is unaffected.

- **The PAR lag-slot and mid-period accumulator seed derivation is
  corrected, and any case that supplied `recent_observations` now trains
  against different — and correct — seed values.** The previous derivation
  mixed an inflow value scaled by raw observation hours against a stage
  weight scaled by the _fraction_ of the period observed — an hours-scale
  numerator against a fraction-scale weight — inflating the seeded share of
  the most-recent lag by a factor on the order of the observation period's
  own hour count, and averaged one scalar weight across every hydro instead
  of weighting each hydro by its own observed coverage. The derivation now
  casts the layered `inflow_history` record and `recent_observations`
  conditioning through one coverage-normalized, day-weighted operator per
  hydro, and an under-covered lag slot or a partially-covered mid-period
  window is now flagged or rejected at load instead of silently seeding a
  wrong value. A case with no `recent_observations` reproduces its pre-fix
  training trace bit-exactly; a case that supplied it gets corrected seed
  values and can therefore retrain to a different trajectory. No released
  version of the conversion tooling ever emitted `recent_observations`, so
  this is a bug fix, not a compatibility break — no input migration is
  owed.

- **A `thermal_bounds.parquet` row carrying a `block_id` is no longer
  silently dropped.** The per-block override on `min_generation_mw` /
  `max_generation_mw` was parsed and discarded, so a deck declaring
  per-block thermal capacity loaded cleanly while every stage was folded to
  one hours-weighted value, misallocating must-run energy and over-allowing
  capacity. The block-scoped override now reaches the LP.

- **`training/dictionaries/bounds.parquet` no longer reports an
  unconditionally-null `block_id`.** The column existed, but every row wrote
  `null` regardless of whether the resolved bound was block-scoped,
  under-reporting the resolved model for any study using the block axis. It
  now emits a row per resolved per-block override; a study with none
  produces a byte-identical file.

## [0.12.0] - 2026-07-21

### Added

- **The objective cost-scale factor is now configurable** via
  `modeling.cost_scale_factor` (`config.json`), replacing a hard-coded
  divisor applied to every non-theta objective coefficient at template
  build time. Absent uses the previous default, byte-identical to prior
  behavior; the field is validated finite and `> 0`, with an advisory
  warning outside `[1.0, 1e12]`. The resolved factor is visible in the
  training scaling report and in exported policy metadata. Effective dual
  tolerance in currency units is `dual_feasibility_tolerance × this factor`
  — raising the factor without adjusting the tolerance loosens optimality
  in currency terms even though the configured tolerance value is
  unchanged.

- **Exported policies now store cut coefficients and intercepts in
  canonical currency units, independent of the writing study's cost-scale
  factor.** Export multiplies every value by the writing study's factor;
  every load — warm-start, resume, simulation-only, and boundary-cut
  injection — divides by the loading study's own factor, so a policy
  trained under one factor loads correctly into a study configured with a
  different one. `policy/metadata.json` gains a `cost_scale_factor`
  provenance field; a checkpoint written before this field existed is
  interpreted as scaled at the previous hard-coded default, so every
  existing policy directory remains loadable.

### Changed

- **Loading a policy written by this release's export path applies one
  additional floating-point division that was not needed before**, since
  cut values now round-trip through canonical currency units rather than
  being carried in the writer's internal scaled representation unchanged.
  The shift is below solver tolerance and does not change training or
  simulation results, but it moves cut coefficients, intercepts, and
  therefore any bit-exact hash computed over a policy-load path by up to a
  few ULP — a resumed-run trajectory or a golden case that hashes a
  loaded policy shifts once. A checkpoint written before this release and
  loaded at the still-default cost-scale factor is unaffected (bit-exact,
  no re-baseline). This is one-directional: a checkpoint written by this
  release stores canonical currency-unit values and must not be read by an
  earlier release, which would silently reinterpret them under the old
  scaled convention; a checkpoint written by an earlier release remains
  fully loadable by this one.

- **Per-phase LP solver settings are now configurable.**
  `training.solver.backward`, `training.solver.forward`, and
  `simulation.solver` each accept an optional solver-profile block of
  per-field overrides (`dual_edge_weight`, `scale`, `price`,
  `primal_feasibility_tolerance`) — resolved once at setup and broadcast
  identically to every MPI rank, a later field always overriding the
  phase's built-in profile. A tuned bundle for the backward pass
  (dual `SteepestEdge`, Curtis–Reid scaling, `Row` pricing, a loosened
  primal feasibility tolerance) is configured explicitly:

  ```json
  "backward": {
    "dual_edge_weight": "steepest_edge",
    "scale": "solver_scaling",
    "price": "row",
    "primal_feasibility_tolerance": 1e-7
  }
  ```

  Every field is optional, and a study with no solver-profile config
  resolves byte-identically to the prior, unconfigurable per-phase
  defaults. On the CLP backend, any solver-profile config — any override
  field — is rejected at setup with a named error identifying the phase and
  the unsupported setting, instead of silently applying a HiGHS-tuned
  option to CLP's own option surface; CLP solver-profile support is
  deferred until it is separately measured.

- **The per-phase solver-profile block gains further override fields**:
  `dual_feasibility_tolerance`, `presolve`, `simplex_update_limit`,
  `cost_perturbation`, `refactor_error_tolerance`, `factor_pivot_threshold`,
  `use_warm_start`, and `steepest_edge_devex_fallback_threshold`, layered on top of
  the base profile the same way as the existing override fields.
  `presolve` only affects a solve that starts genuinely cold — a
  warm-started solve skips presolve regardless of the setting.
  `use_warm_start` is a diagnostic override: setting it `false` forces
  every solve in the phase cold and is not an intended production
  configuration. Every new field is optional; leaving it unset resolves to
  the value already in effect before this override existed, so a study
  with no override for a new field resolves byte-identically to before. On
  the CLP backend, setting any new field is rejected at setup the same way
  the existing override fields are.

- **An opt-in opening-block backward scheduler distributes backward-pass
  work at finer granularity than a whole trial point.** Setting
  `training.parallelism.backward_scheduler` to
  `{ "method": "opening_block" }` (the default remains
  `{ "method": "trial_point" }`) reassigns each backward work unit from a
  whole trial point to a `(trial point, opening block)` pair, claimed off a
  shared counter and warm-chained from a fresh frozen-LP load. The optional
  `block_size` field of the `opening_block` method controls the block size
  (default: half of each stage's opening count, rounded up; an explicit
  value is clamped to the stage's own opening count); supplying
  `block_size` under `trial_point` is a load-time error rather than a
  silently ignored key. Claims within a stage are ordered
  hardest-first by each `(stage, block)`'s mean simplex-iteration cost
  measured on the previous iteration, load-balancing workers without
  changing which cuts are generated or how they are aggregated — the
  produced cut set and the training lower bound are identical to claiming
  blocks in their canonical order. An active Dynamic Cut Selection
  iteration always falls back to the `trial_point` scheduler: the
  opening-block path's frozen-LP load is incompatible with Dynamic Cut
  Selection's cut-free lazy core.

- **The training `Time split` report now reflects coordinator-measured
  phase time instead of a per-worker average.** The `Forward`/`Backward`
  lines report the coordinator's own measured phase wall (previously a
  per-worker mean), each decomposed into `solve` (mean worker busy time)
  and `wait` (load imbalance across workers); a new `Serial` line replaces
  the previous opaque `Other` bucket, breaking the non-parallel portion of
  training down into lower-bound evaluation, row (cut) selection,
  cross-rank cut synchronization, MPI allreduce, and a residual `other`
  bucket that absorbs scheduling overhead.
  The three lines' durations sum to the training wall and match the
  per-iteration progress line. A summary reconstructed from a completed
  output directory (`cobre summary`) omits the `Time split` block entirely,
  since per-iteration phase timing is not persisted to `metadata.json`.

- **The backward pass now solves each stage's openings along a shortest
  warm-start chain.** Each stage's openings are ordered along a
  nearest-neighbor-plus-2-opt minimum-distance path over their inflow-noise
  vectors before solving, replacing the previous fixed
  sigma-weighted-key order (a stage with fewer than three openings keeps
  the sigma-weighted order — there is no path to improve). The order is
  intrinsic; no config field selects it. The chosen order only changes
  which warm-start chain the backward pass walks: each opening's cut is
  still written and aggregated by its own canonical opening identity,
  independent of solve order, so declaration-order invariance and
  run-to-run reproducibility are unaffected. The warm-start chain
  itself changes, not only training time: at a degenerate optimum a
  multi-opening stage can settle on a different — but equally optimal —
  vertex than the prior order, so training and simulation outputs on such a
  stage can shift.

## [0.11.1] - 2026-07-17

### Fixed

- **The anticipated-commitment drift margin no longer refuses genuine solver
  noise.** The margin introduced in 0.11.0 was anchored to the backends'
  raw `primal_feasibility_tolerance`, but that tolerance bounds the scaled
  residuals — the carried commitment is a basic-variable value whose error is
  amplified by the basis conditioning, and production-scale studies produce
  drift severalfold past it (observed: `3.8e-6` MW over a `1593` MW cap,
  refused as a training error). The relative headroom is now two orders above
  the raw tolerance with a matching absolute floor — still sub-watt at any
  realistic capacity, so a genuine over-commitment (kilowatts and up) is
  refused exactly as before.

## [0.11.0] - 2026-07-16

### Changed

- **`residual_std_ratio` is now derived, not trusted, whenever
  `inflow_ar_coefficients.parquet` supplies AR coefficients.** The residual
  std ratio (`σₘ/sₘ` per season) is computed at load from the standardized
  AR coefficients via the periodic-ACF closure under the unit-marginal-
  variance contract, instead of being read from a stored or previously
  fitted value. For a season whose per-season AR order is uniform across the
  cycle, the derived value matches a Yule-Walker fit to numerical precision
  (unchanged solved outputs); for heterogeneous per-season orders, it
  differs from a Yule-Walker fit by roughly `1e-4`. A study that supplies AR
  coefficients directly (bypassing estimation) now has any stored
  `residual_std_ratio` in the coefficients file ignored outright and
  replaced by the closure-derived value — if that stored value was not
  itself consistent with the supplied coefficients, solved outputs can
  shift by more than the `1e-4` mixed-order band. Declaration-order
  invariance and run-to-run reproducibility are unaffected.
- `historical`-scheme inflow sampling's residual standardization (`σ = s·r`,
  `sampling/historical.rs` reading the model's `sigma()`) is unchanged in
  code but now reads the closure-derived `r` above it, so a `historical`
  study with heterogeneous per-season AR order sees its sampled residuals
  standardized by a correspondingly shifted `σ` (~`1e-4`).
- The spectral-clipping diagnostic emitted while estimating correlation
  matrices is logged at debug level instead of warn: a finite-sample
  correlation estimate is routinely indefinite and clipping to the nearest
  positive-semidefinite factor is the expected remedy, so healthy studies no
  longer warn on every run. The largest negative eigenvalue magnitude stays
  on the debug record.

### Fixed

- **A study whose anticipated thermal commitment reaches its delivery
  generation cap now trains instead of aborting as infeasible.** The
  commitment a delivery stage receives is the solver's computed value for a
  ring state column, so it is accurate only to the backend's primal
  feasibility tolerance rather than exact. It can therefore arrive a hair
  outside the generation bound the must-generate coupling pins it to, and the
  stage LP was reported infeasible over a physically meaningless quantity,
  ending the run. The delivery generation bound is now reconciled against that
  drift on every solve. A commitment genuinely beyond its cap is unaffected:
  it is refused as a named error identifying the thermal, stage, and
  overshoot, never absorbed and never reported as a bare infeasibility.

- **A non-stationary fitted inflow model now fails the load with a named
  error instead of silently corrupting scenarios.** When history estimation
  produces AR coefficients whose implied residual variance is negative (a
  degenerate near-unit-root fit), the closure-derived `residual_std_ratio`
  is non-finite; loading previously wrote that `NaN` into the model — user-
  supplied coefficient files were already rejected by the stationarity
  validation rule, but internally-estimated coefficients bypassed it. The
  derivation now hard-errors, naming the hydro and season.
- **Multi-rank training with an auto-generated opening tree now derives the
  same tree on every rank.** The tree is regenerated locally per rank, but
  only rank 0 passed the noise-group ids that make consecutive stages
  sharing a `(season, year)` group reuse one noise draw — its peers drew
  independently, solved against different openings, and exchanged cuts for
  models that disagreed. Uniform monthly studies were unaffected (every
  group is unique there); a weekly study placing several stages in one
  season triggered it. Both sides now derive the ids from the broadcast
  system through one owner.
- **Multi-rank runs no longer overstate the lower bound through empty
  entity lookups on non-root ranks.** `System`'s lookup indices are derived
  data skipped by serialization, and the broadcast path left them empty on
  every rank but 0 — downstream, a missing downstream-hydro lookup fell
  back to a different tailrace, so non-root ranks fitted different FPHA
  planes, built structurally different LPs, and rejected rank 0's
  warm-start basis; lower bounds came out roughly 3% above the true value.
  The indices are now rebuilt as part of deserialization itself.
- **A stored policy basis with too few basic entries is rejected with a
  named error instead of being handed to the solver.** A deficit proves the
  basis was captured against a differently-shaped LP; previously HiGHS
  aborted on it while CLP silently accepted it — the same defect was a hard
  abort on one backend and a silent wrong answer on the other. Both now
  reject it identically, reporting the basic-count arithmetic.
- **A rank failing under MPI now prints its own error before aborting.**
  The abort path bypassed error rendering, so a failed run reported only
  the launcher's bare abort code — and, since ranks abort in lockstep,
  often from a peer echoing the failure rather than the rank that caused
  it. The failing rank's rendered error now reaches stderr first.
- **PAR history estimation no longer panics when the fitting lag depth
  exceeds the season cycle.** Evaluating a candidate lag `k ≥ n_seasons + 2`
  (reachable at the default `max_order = 6` with a short season cycle, e.g.
  two seasons) underflowed a season index in the conditional-FACP
  covariance and aborted the fit in debug builds. The lag season now wraps
  around the cycle; in-range lags are computed identically.

## [0.10.0] - 2026-07-10

### Added

- **A stage's blocks can now be solved chronologically, chaining storage across
  the blocks within a stage instead of treating them as parallel load levels.**
  `stages.json` gains a per-stage `block_mode`, either `"parallel"` (the default,
  and the prior behavior) or `"chronological"`; any other value is rejected with a
  schema error naming the offending stage and value. A chronological stage chains
  its per-block storage boundaries `S⁰ … Sᴷ` through one water-balance row per
  block, computes each block's FPHA production and evaporation on that block's own
  average storage, and freezes the per-block storage identity for a PreFilling
  hydro. A `parallel` stage, and any single-block (`K = 1`) stage, produce an LP
  that is bit-for-bit identical to the prior release; the future-cost epigraph and
  the state-vector dimension are independent of a stage's block count. Simulation's
  `simulation/hydros/` output reports each block's own storage boundaries and
  evaporation on a chronological stage — the same columns as before, resolved per
  block — emitted identically by the CLI and the Python bindings. Because a cut's
  coefficients do not depend on block count, a policy trained in one block mode can
  be loaded and simulated in the other; `policy/metadata.json` records the training
  block mode in `training_block_mode` (and, when it varies across stages,
  `training_block_mode_per_stage`).

- **Water travel time now delays a hydro release before it reaches its
  downstream plant.** The `travel_time_hours` field on the upstream hydro's
  cascade arc to `downstream_id` declares, in hours, how long the arc's release
  stays in transit before it reaches the downstream plant's water balance; v1
  covers the main cascade arc only — diversion and pumping-conduit arcs do not
  carry travel time. The in-transit volume is carried as augmented Benders
  state — one state slot per downstream plant per maturity — until it matures
  onto the receiving plant's balance. A hydro whose `travel_time_hours` is
  absent or `0.0` keeps the existing instantaneous-transfer behavior: no arc is
  declared and no state is added. `past_defluences` on `InitialConditions`
  supplies the pre-study releases already in transit at study start for a
  declared arc; config validation requires history at least as deep as the
  arc's travel time and either finds it directly, derives a proxy from
  `past_inflows` with a logged caveat, or rejects the study when neither is
  available. A declared arc that releases while its downstream plant has not
  yet reached operating status (still `PreFilling`/`Filling`, or before its
  `entry_stage_id`) is also rejected at validation. Simulation writes a new
  `simulation/in_transit/` output (`stage_id`, `hydro_id`, `lag`,
  `in_transit_volume_hm3`, `delayed_arrival_hm3` per downstream plant per
  maturity lag), emitted by both the CLI and the Python bindings and absent for
  a study with no declared travel-time arc. In-transit volume that would mature
  past the study's last stage is dropped rather than credited to terminal
  storage — a documented target-stage imprecision at the end of the horizon.

- **A maturing in-transit water bucket now splits its delivery across a
  chronological arrival stage's own blocks, blended over every source stage
  whose released water is still arriving there.** When the arrival stage's
  block partition differs from the stage(s) the water was released from — a
  coarsening or refining resolution boundary — the delivered volume is
  resolved directly against the arrival stage's own blocks rather than a
  density carried over from a sending stage, weighted by each contributing
  source stage's own share of the maturing volume. A parallel source stage
  maturing into a chronological arrival stage is covered by the same blend:
  it no longer falls back to splitting the delivery in proportion to block
  duration regardless of the travel time. The delivered split remains a
  single fixed density per maturing bucket — it does not vary by which block
  released the water that is arriving — a documented, accepted modeling
  bound.

- **Generic constraints can reference per-block hydro storage and evaporation.**
  `hydro_storage_initial(h)` and `hydro_storage_final(h)` reference a stage's
  initial (`S⁰`) and final (`Sᴷ`) storage; supplying a block index —
  `hydro_storage_initial(h, k)` / `hydro_storage_final(h, k)` — references the
  storage boundary at the start / end of block `k` (available on chronological
  stages; a parallel stage exposes only its two endpoints, so an interior block
  reference there is rejected). `hydro_evaporation(h, k)` selects block `k`'s
  evaporation, mirroring the block-index convention used by the flow variables;
  bare `hydro_evaporation(h)` is the stage evaporation in parallel mode (every
  block shares the stage's storage endpoints) but is rejected on a chronological
  stage with more than one block, where the blocks differ and a block must be
  named. A block reference a stage cannot expose is rejected at load with a
  message naming the constraint, the block, and the stage's block count.

- **A stage can now carry storage-only cuts under a PAR(p) inflow model, so a
  PAR(p) study can be coupled to a storage-only downstream boundary.** The
  per-stage `state_variables` block in `stages.json` — `{ "storage": <bool>,
"inflow_lags": <bool> }`, defaulting to storage-only — now governs the
  dimension of the cuts a stage emits, not only the state it reports. A stage
  configured with `inflow_lags: false` produces cuts spanning the storage
  dimension alone even when the study fits a PAR(p) inflow model, rather than
  zero-padding the inflow-lag coefficients; in a multi-rank run each stage's cut
  synchronization derives its wire layout from that stage's own cut dimension. A
  study that fits a PAR(p > 0) model but disables inflow lags on every stage is
  surfaced with a model-quality warning. Studies that leave every state dimension
  enabled produce bit-for-bit identical cuts and results.

- **Every `system/*.json` entity now carries a required `operational_start_date`
  field (an ISO-8601 `YYYY-MM-DD` calendar date).** Buses, hydros, thermals,
  lines, non-controllable sources, pumping stations, and energy contracts must
  each declare the date the entity enters service. The field has no default: a
  registry file that omits it is rejected, and a value that is not a valid
  ISO-8601 date produces a schema error naming the file, the field, and the
  offending string.

- **A study now fails validation instead of silently mis-modeling previously
  unguarded inflow-PAR and anticipated-thermal configurations.** An
  anticipated thermal's `lead_time_hours` lead exceeding the study's full
  horizon is rejected — the plant could never deliver within the horizon, and
  previously it fell through validation and dispatched as an ordinary thermal
  with no anticipation. A non-Monthly season cycle supplying an inflow annual
  component is rejected, because that long-memory extension is
  Monthly-exclusive by design.

### Changed

- **Operational entities now sort canonically by `(operational_start_date, id)`
  instead of `(operational_start_date, name)`.** The entity id is the stable
  canonical key; entity names are user-chosen and vary between authors of the same
  system, so a rename no longer changes the LP layout, cut ordering, or output
  column order (whereas an id renumbering now does). A study whose same-date
  entities were ordered differently by name than by id sees a reordered — but
  numerically equivalent — LP; the optimum and reported costs are unchanged.

- **Cobre is configured only through config/data files and `cobre` CLI
  arguments; environment variables are no longer an input channel.** The
  `COBRE_THREADS`, `COBRE_COLOR`, `COBRE_COMM_BACKEND`, `COBRE_W1_DIAG`,
  `FORCE_COLOR`, `NO_COLOR`, `COLUMNS`, and `HOSTNAME` reads are removed. Thread
  count is set with `--threads`; color with `--color <auto|always|never>`
  (default `auto`, colored only when stderr is a terminal); and the communication
  backend with `--comm-backend <auto|local|mpi>` (default `auto`). `auto` selects
  the MPI backend when the process is launched under an MPI launcher
  (`mpiexec`/`mpirun`/`srun`) and the local backend otherwise — detecting a
  launcher is a runtime fact, not a configuration channel, so this is retained;
  `--comm-backend mpi` forces the MPI backend and fails with a clear message on a
  binary built without MPI support. Terminal width for progress rendering and the
  hostname recorded in run provenance are now obtained by querying the terminal
  and the OS directly.

- **Anticipated (pre-committed) thermal dispatch, previously configured only as a
  stage-count lead, now also accepts a physical-duration lead and anchors every
  commitment at its delivery stage.** `system/thermals.json`'s `anticipated_config`
  gains `lead_time_hours` (a duration in hours, resolved against the stage
  calendar) as an alternative to the existing `lead_stages` (a stage count that
  never consults the calendar); the two are mutually exclusive. Every committed
  decision is now bounded, costed, and commissioning-gated at its own delivery
  stage rather than the stage where it is decided, so a plant whose generation cap
  differs across stages cannot be committed to a value its delivery stage cannot
  honor. The in-flight commitment between decision and delivery is carried as
  augmented Benders state — a ring of per-plant slots advanced by a plain copy each
  stage, with no out-of-LP shift step. A `lead_time_hours` configuration whose
  coarser decision stage would anchor more than one delivery stage — a single
  decision committing several deliveries at once — is rejected at setup:
  per-delivery-stage simulation output for that configuration is not yet supported.

- **A non-filling hydro's commissioning window is now honored, and spillage is
  frozen for any hydro that has not yet entered service.** The
  `entry_stage_id`/`exit_stage_id` window on a non-filling hydro — previously
  parsed but applied nowhere, and flagged only by a model-quality warning — now
  takes effect: outside its window the plant is modeled as PreFilling, with its
  turbine, spillage, and diversion pinned to zero, its storage decoupled by the
  frozen storage identity, and its inflow passed through to the first active
  downstream plant (or the network sink), so the river flows past a site that does
  not yet exist without trapping water or injecting phantom storage. The obsolete
  parsed-but-inert warning is removed. Independently, spillage is now frozen to
  zero for every hydro while it is PreFilling — a dam that does not yet exist
  cannot spill; previously the spillage column was left free during PreFilling,
  which let the optimizer route phantom water onto a downstream balance. Spillage
  remains free during Filling, where a real impounding reservoir can shed inflow.
  Studies with pre-commissioning or PreFilling hydros produce different numerical
  results as a consequence.

- **Each policy cut and state file now embeds the per-slot identity of its
  state-vector dimensions.** Every `policy/cuts/stage_NNN.bin` and
  `policy/states/stage_NNN.bin` carries an entity manifest with one entry per
  coefficient position, recording the owning entity's type, id, and secondary
  index (the inflow-lag order or anticipated-commitment ring slot) plus whether
  that entity was operationally active at the stage. This identity is now read
  directly from the policy file rather than cross-referenced against a separate
  sidecar.

- **Loading a trained policy is now always validated against the current study.**
  Warm-start, resume, simulation-only runs, and the Python bindings route every
  policy load through a single validation entry point that cannot be disabled.
  Beyond the state-vector dimension it checks the per-slot entity manifest each
  policy file now embeds, rejecting a policy whose stored dimensions match the
  current study by count but attach to different entities — a mismatch the integer
  dimension check alone accepted.

### Removed

- **The `training/dictionaries/state_dictionary.json` sidecar is no longer
  written.** Its per-slot state-variable mapping is now embedded in each policy
  cut and state file (see above). `training/dictionaries/` still contains
  `codes.json`, `entities.csv`, `variables.csv`, and `bounds.parquet`.

- **The `policy.validate_compatibility` configuration field is removed.**
  Policy-load validation is now unconditional (see above), so the opt-out no
  longer exists. A config file that still sets `validate_compatibility` is
  rejected — the policy config section rejects unknown fields — and the Python
  bindings no longer accept a `validate_compatibility` argument.

### Fixed

- **A study on a non-monthly (weekly or custom) season cycle that fits an
  inflow PAR model now advances its inflow-lag state at the end of every
  period, instead of silently freezing it at the initial condition for the
  whole run.** Only a monthly season cycle previously accumulated, finalized,
  and spilled the per-period lag average forward; a weekly or custom cycle
  computed zero-weight contributions every stage, so the lag state driving the
  PAR forecast never left its initial value. The day-weighted accumulate /
  spillover / finalize arithmetic that already covered monthly periods now
  applies uniformly to ISO-week and user-defined custom periods, including a
  season map that layers more than one period resolution together (for
  example, monthly and quarterly definitions in the same map) — each stage
  advances within its own resolution's period, never a different one.

- **A study whose stage ids do not start at zero now receives the correct FPHA
  productivity coefficients.** The per-stage productivity-override table was keyed
  by a stage's position in the solve loop rather than its domain stage id, so a
  study numbered from anything other than zero silently paired each stage with
  another stage's energy coefficient (the reference-turbine and specific-productivity
  conversion paths were affected too). Coefficients are now keyed by the domain
  stage id; studies numbered from zero are unaffected.

- **Studies on weekly, custom, or mixed-resolution season calendars are no longer
  mis-modeled in evaporation, inflow-lag advancement, and correlation
  estimation.** A weekly cycle with evaporation configured no longer fails to
  build: a block's evaporation now takes its month from the stage's start date
  rather than reinterpreting the season index, which also corrects custom cycles
  that indexed the wrong calendar month. On a multi-resolution season map (for
  example a monthly definition layered with a quarterly one), the opening-tree
  residual standardization and the external/historical replay samplers now advance
  the downstream inflow-lag ring the same way the forward pass does, instead of
  assuming a single primary lag order; a monthly-only study is unchanged. A
  partial-year study that estimates its own inflow correlation no longer drops the
  residual seasons that fall outside the study window.

- **A failure on one MPI rank no longer hangs the healthy ranks at a mismatched
  collective.** Across the backward pass, the lower-bound evaluation, the forward
  loop, run finalization, and the CLI's post-simulation and post-export steps, a
  rank-local failure is now reconciled across all ranks before the next collective,
  so a subset-of-ranks error — a failed worker solve, a stage-zero evaluation
  failure, or a rank-0 export write to a full or read-only disk — surfaces as a
  coordinated shutdown rather than a hang. Single-rank runs are unaffected.

- **Reloading a policy checkpoint no longer degrades a CLP warm-start.** Basis
  statuses now share one canonical representation that the HiGHS and CLP backends
  translate to and from at the solver boundary, and a basis stored in a policy
  checkpoint round-trips all of its statuses losslessly — the earlier export folded
  CLP's superbasic and fixed statuses into a lossy encoding, so a warm-start
  reconstructed from a reloaded checkpoint began from a weaker basis. A binary from
  this release reading an older checkpoint, and an older binary reading a checkpoint
  from this release, both stay safe: an unrecognized status falls back to a cold
  start rather than being misread. Deterministic results are unchanged.

## [0.9.1] - 2026-06-26

### Fixed

- **Future-cost discounting is now applied to the correct LP column when a study
  combines anticipated (GNL-style) thermal pre-commitment with a non-zero
  `stages.json` `annual_discount_rate`.** The per-stage discount factor multiplies
  the future-cost epigraph (θ) column of the LP objective. Its column index was
  derived by hand from the state and hydro counts, which omits the anticipated-state
  block; in a deck with anticipated thermals this landed on a zero-cost column,
  leaving the future-cost term undiscounted. The discount now reads the authoritative
  per-stage θ column index. Decks without anticipated thermals, or with
  `annual_discount_rate: 0`, produce bit-for-bit identical output.

## [0.9.0] - 2026-06-25

### Added

- **Energy contracts now participate in dispatch.** A `system/energy_contracts.json`
  contract (a bilateral purchase or sale obligation) contributes one LP column per
  block per direction (`type: "import"` or `"export"`) on its `bus_id`, bounded by
  `[limits.min_mw, limits.max_mw]`. An import column injects `+1.0` MW into the bus
  power-balance row; an export column withdraws `−1.0` MW. A positive `price_per_mwh`
  is a cost (import); a negative value is revenue (export). Contracts honor a
  commissioning window (`entry_stage_id`/`exit_stage_id`): outside `[entry, exit)` the
  column is pinned to `[0, 0]` and a zero-power row is emitted. Stage-varying bounds
  and prices are supplied via `constraints/contract_bounds.parquet`; a non-zero
  `min_mw` row acts as a take-or-pay floor enforced as a hard LP column lower bound.
  Simulation writes a new `simulation/contracts/` output (`stage_id`, `block_id`,
  `contract_id`, `power_mw`, `energy_mwh`, `price_per_mwh`, `total_cost`,
  `operative_state_code` per contract per block) plus a `contract_cost` cost column —
  both emitted by the CLI and the Python bindings.

- **Pumping stations now participate in dispatch.** A `system/pumping_stations.json`
  station (a pumped-storage / reversible plant) contributes a per-block pumped-flow
  decision bounded by `flow.min_m3s`/`flow.max_m3s` that transfers water from its
  `source_hydro_id` reservoir to its `destination_hydro_id` reservoir and draws
  `consumption_mw_per_m3s × flow` of electrical power on its `bus_id`. Stations honor
  a commissioning window (`entry_stage_id`/`exit_stage_id`): a station is active only
  at stages in `[entry, exit)` and contributes no LP columns outside it. Simulation
  writes a new `simulation/pumping_stations/` output (pumped flow, volume, power,
  energy, cost, and operative state per station per block) plus a `pumping_cost`
  cost column — both emitted by the CLI and the Python bindings.

### Changed

- **Commissioning windows are now honored by thermal units, transmission lines,
  and anticipated (GNL) thermals.** The `entry_stage_id`/`exit_stage_id` fields on
  `system/thermals.json` and `system/lines.json` — previously parsed but inert —
  now take effect. Outside `[entry, exit)` a thermal has both generation bounds
  zeroed (including any must-run floor, so a windowed-out must-run unit stays
  feasible) and a line has both flow caps zeroed. An anticipated thermal gates its
  commitment and generation on the delivery stage's window. NCS and pumping
  commissioning is unified onto the same zero-influence treatment, so a dormant
  NCS, pumping, thermal, or line entity now emits a uniform zero-valued output row
  rather than being omitted from the output. Only non-filling hydro commissioning
  windows remain parsed-but-inert (still surfaced by a model-quality warning).

- **`constraints/pumping_bounds.parquet` override rows are validated for domain
  sanity.** A pumped-flow override row is now rejected when a bound is negative or
  when `min_m3s > max_m3s`, matching the checks already enforced on the
  `system/pumping_stations.json` entity path.

- **BREAKING — `filling_min_rate_m3s` replaces `filling_inflow_m3s` and flips
  semantics.** The `filling` block in `system/hydros.json` and the override
  column in `constraints/hydro_bounds.parquet` are renamed from
  `filling_inflow_m3s` to `filling_min_rate_m3s`. The field is no longer a
  retention cap on impounded inflow (a ceiling limiting how much of the natural
  inflow is kept): it is now a **per-stage minimum accumulation rate** (a floor
  the reservoir must clear). Each Filling stage derives a minimum target storage
  anchored on `min_storage_hm3` and accumulating `filling_min_rate_m3s` over
  the stage duration; the reservoir must reach at least that running level by the
  end of the stage. Natural inflow (PAR / AR coupling) is no longer cap-limited
  during Filling — inflow above the old cap flows freely. The impound / retention
  cap mechanism is removed entirely. Per-stage `V_target` filling floors replace
  the single terminal filling target that was enforced only at the entry stage.
  Any existing case that sets `filling_inflow_m3s` must rename the field and
  review the value: the two semantics are inverses, not aliases.

## [0.8.2] - 2026-06-17

### Added

- **`system/tailrace_curves.parquet` (optional).** Exact piecewise-quartic
  tailrace curves with backwater families keyed by the downstream
  reservoir's reference level. When a plant has rows here, the computed-FPHA fit
  evaluates its tailrace from these quartics — selecting the segment by
  downstream flow and interpolating between families at the downstream plant's
  stage reference level — instead of the entity-level tailrace model. Plants
  without a row keep the existing polynomial / piecewise-linear tailrace, so the
  input is inert for cases that omit it.

- **`reference_volume` in `system/hydro_production_models.json`.** A per-model
  reference operating volume (sibling of `fpha_config`), declared as either
  `volume_hm3` (absolute) or `percentile` (fraction of useful volume) —
  mutually exclusive, set exactly one. This is now the single source of truth
  for the reference volume the computed-FPHA fit and the equivalent-productivity
  derivation consume.

- **`fpha_plane_reduction` in the production-model config (optional).**
  Similar-hyperplane simplification that merges near-parallel / near-coincident
  FPHA planes into their mean hyperplane to shrink the LP. Two mutually-exclusive
  methods: `{ "method": "angle", "tolerance_deg": <0–90> }` and
  `{ "method": "distance", "tolerance_pct": <f64>, "n_samples": <u32> }`. Off by
  default; the origin plane (zero generation at zero turbining) is never merged.
  The distance method's sampling uses a deterministically seeded PRNG, so results
  are bit-identical across input ordering and rank count.

- **Computed-FPHA fit-quality warning.** When a fitted plane set deviates from
  the exact production function by more than 5 % (relative mean absolute
  deviation over the spill = 0 grid), the run logs a warning naming the plant and
  stage — typically a strongly non-concave production surface no single `α`
  correction can track. This is the operator-facing replacement for the retired
  low-`kappa` warning. Warnings are emitted in canonical plant/stage order.

- **`output/hydro_models/evaporation_models.parquet`.** The resolved per-hydro
  evaporation coefficients are now written as an output sidecar, emitted whenever
  any hydro declares an evaporation model. Written by both the CLI and the Python
  bindings.

- **Setup-phase timings in `training/metadata.json`.** A new `setup` section
  records the wall-clock cost of each study-setup phase (input load, stochastic
  fit, production-model fit, evaporation fit, and MPI broadcast). The timings are
  observational only — they are non-deterministic and never participate in any
  parity hash.

### Changed

- **Computed FPHA is now fit by a 3-D convex hull.** The computed path evaluates
  the production cloud on a `(volume, turbined)` grid at spillage = 0, takes its
  3-D convex hull, applies a least-squares `α` correction, and fits a per-plane
  lateral-flow secant — fully replacing the previous tangent-plane sampling,
  greedy plane selection, and `kappa` shrink. Fits are now resolved **per stage**
  (one per season / stage range), where they were previously stage-independent.
  Run-of-river (single fitting volume) plants, which previously failed to fit,
  are now supported. Computed-FPHA deterministic baselines were re-blessed to the
  new coefficients.

- **New vendored dependency: `qhull`.** The reentrant `libqhull_r` is statically
  linked into `cobre-sddp` (git submodule pinned to tag `2020.2`); see
  `THIRD_PARTY_NOTICES.md`. The hull input and output are canonically sorted, so
  hyperplanes stay bit-identical regardless of input ordering and MPI rank count.

- **PAR(p) per-hydro AR fit parallelized.** The initial autoregressive fit of
  the periodic inflow model now fits each hydro in parallel (rayon). Results are
  bit-identical regardless of thread count — per-hydro blocks reassemble in
  canonical entity order.

- **BREAKING — `training.cut_selection` restructured.** Method-specific
  parameters now live inside a tagged `selection` object instead of a flat bag
  of fields. The top level keeps only the always-on knobs
  (`row_activity_tolerance`, `max_active_per_stage`) plus `selection`; the chosen
  `selection.method` (`"level1"`, `"lml1"`, `"domination"`, or `"dynamic"`)
  carries only its own parameters. Supplying a parameter that belongs to a
  different method, or misspelling `method`, is now a config-load error rather
  than a silently ignored value. Omitting `selection` disables row selection.

  Field renames (all now scoped to their method's `selection` block unless
  noted):
  - `cut_activity_tolerance` → `row_activity_tolerance` (top-level, always-on)
  - `active_window` → `seed_window` (dynamic)
  - `candidate_window` → `candidate_recency` (dynamic)
  - `nadic` → `max_added_per_round` (dynamic)
  - `domination_epsilon` → `domination_tolerance` (domination)

  Under the dynamic method, `violation_tolerance` no longer falls back to a
  level-1 tie tolerance; it defaults to `1e-10` directly.

### Removed

- **`training.cut_selection` legacy fields.** `enabled` and `method` are gone —
  the presence of `selection` enables row selection and `selection.method` is the
  discriminator. The dead fields `threshold`, `memory_window`, and
  `basis_activity_window` are removed entirely. Existing configs that set any
  removed or renamed key must be migrated to the `selection` block; an unmigrated
  flat config now fails to load with a clear deserialize error.

- **Computed-FPHA `kappa` shrink and low-kappa warnings.** The computed path no
  longer derives a `kappa` correction factor or emits low-kappa warnings (the
  CLI display is gone): the least-squares `α` correction replaces the shrink
  factor, and the fit-quality deviation warning (above) replaces the low-kappa
  warning. The `kappa` column in the precomputed `fpha_hyperplanes.parquet` input
  is retained for back-compatibility and defaults to `1.0`.

- **`reference_volume_hm3` input column.** No longer read from
  `system/hydro_energy_productivity.parquet` (a stale column is ignored with a
  warning); declare the reference volume via the production-model
  `reference_volume` field instead.

- **BREAKING — `config.json` `energy` section.** The top-level `energy` block
  (and its `reference_volume_fraction`) is removed; the reference volume is now
  declared per production model via `reference_volume` in
  `system/hydro_production_models.json`. Because the config rejects unknown
  fields, an existing config that still carries an `energy` block now fails to
  load — remove the block and migrate the reference volume to the production-model
  field.

## [0.8.1] - 2026-06-13

### Added

- Dynamic cut-selection method, selected with
  `training.cut_selection.method = "dynamic"`. Rather than carrying the entire cut
  pool into every LP, a dynamic run loads only a small resident subset of cuts per
  solve — keeping per-solve LP size bounded as the pool grows while the full pool
  is retained — and applies uniformly across the backward pass, forward pass, and
  simulation. The resident set is tuned by `active_window` (the seed window `k2`,
  below), `candidate_window`, and `nadic`, and is mutually exclusive with the
  periodic-pruning methods (`level1` / `lml1` / `domination`).

- `training.cut_selection.active_window` — a first-class config field for the
  dynamic cut-selection active-set seed window (`k2`). Applies only when
  `method = "dynamic"`. Default `5`; `0` is valid and meaningful (seeds only the
  current iteration's cuts, matching NEWAVE `selcor.dat`). Previously this value
  had to be supplied through `check_frequency`, which overloaded the
  periodic-pruning cadence used by the `level1` / `lml1` / `domination` methods.

- Selectable LP backend at build time. A binary is now bound to exactly
  one LP solver backend, chosen via Cargo features:
  - `highs` (enabled by default) — [HiGHS](https://highs.dev) LP solver, MIT-licensed.
  - `clp` (opt-in) — [CLP/CoinUtils](https://github.com/coin-or/Clp) LP solver, EPL-2.0-licensed.
    Build with `--no-default-features --features clp`.

  The two features are **mutually exclusive**: enabling both is a compile
  error. Enabling `clp` selects CLP; `highs` applies only when `clp` is
  not enabled. The active backend is reflected in `cobre version` and in
  the `solver` / `solver_version` fields of the run output metadata.

  The CLP backend ships the following capabilities: dual and primal
  simplex algorithms; native incremental row and bound mutation
  (appending rows or patching bounds preserves the solver's factorization
  across mutations, enabling warm-start continuity across solve sequences);
  per-phase tuning covering the dual-simplex pricing strategy,
  factorization cadence, feasibility and optimality tolerances, and
  iteration limits; and hot-start (snapshot and restore of the simplex
  rim), delivered through the C++ class interface.

  Each backend is internally deterministic: run-to-run bit-for-bit
  reproducible and declaration-order invariant (a permutation of the
  input entities produces the correspondingly permuted result). Switching
  backends may legitimately change numerical results — the two solvers can
  reach different optimal vertices on degenerate problems — so each
  backend maintains its own deterministic parity baselines.

  Existing builds are unaffected: the default backend remains HiGHS, and
  the CLP backend is strictly opt-in.

- `anticipated_thermal_cost` — a new per-stage field in the run cost output that
  attributes the forward-committed (anticipated) thermal commitment cost, so the
  sum of the named cost categories reconciles to `immediate_cost`. It is zero for
  cases with no anticipated units and is written identically by the CLI and Python
  paths.

- Dynamic cut selection now reports the per-solve resident-set size — the cuts
  actually loaded into each LP. Surfaced as a run-level mean and max in the console
  summary and `training/metadata.json`, and as a per-iteration `mean_rows_in_lp`
  column in `training/convergence.parquet`. The pool `active / generated` line is
  retained as the pool/memory-footprint figure. The metric is work-distribution
  invariant (bit-identical across thread counts).

### Changed

- `method = "dynamic"` no longer reads `check_frequency` for its `k2` window;
  use the new `active_window` field instead. A dynamic config that relied on
  `check_frequency` to set `k2` now falls back to the default `k2 = 5` unless it
  sets `active_window`. `check_frequency` remains the periodic-pruning cadence
  for `level1` / `lml1` / `domination`, where `0` is still rejected; under
  `dynamic` an explicit `check_frequency` is ignored rather than rejected.
- The deprecated `training.cut_selection.threshold` and `memory_window` fields
  are now silently ignored for **every** method, including `dynamic`. They were
  previously consumed as undocumented fallbacks for `nadic` and the
  candidate-recency window (`k1`) under `dynamic`; that honoring is removed. The
  fields remain accepted in config files (so existing configs still parse) but
  no longer affect behavior. Configure `nadic` and `candidate_window` directly.
- Distributed release artifacts (CLI archives, Python wheels, MPI tarball) now
  bundle the complete third-party license notices for the Rust dependency graph
  (`THIRD_PARTY_LICENSES.md`), in addition to the vendored C++ solver attributions
  already recorded in `NOTICE` and `THIRD_PARTY_NOTICES.md`.

### Fixed

- Per-entity hydro penalty overrides: the directional water-withdrawal and
  evaporation violation costs (`*_violation_pos_cost` / `*_violation_neg_cost`)
  now fall back to the entity's resolved symmetric cost when left unset, instead
  of the global directional default. An entity that overrode only the symmetric
  `water_withdrawal_violation_cost` / `evaporation_violation_cost` previously had
  its directional costs silently revert to the global value.
- PAR(p) estimation no longer panics for studies whose horizon is narrower than
  the season cycle (for example a monthly model running only September–December).
  Seasons that are not lag-reachable are skipped, and for PAR(p > 0) the
  recent-past months before the study start are synthesized from history so their
  seasonal statistics feed the pre-study lags. Studies that span the full cycle
  (or carry no out-of-window history) are unaffected and remain bit-identical.
- Water-withdrawal violation modeling: the under-delivery slack is now bounded so
  realized withdrawal cannot cross zero past its target. Previously a run-of-river
  plant could "un-withdraw" well beyond its target and inject phantom water; the
  bound is sign-aware for negative/return targets. This affects only degenerate
  cases — realized withdrawal now pins at the target.

## [0.8.0] - 2026-06-01

### Deprecated

- `training.cut_selection.basis_activity_window` is now ignored at
  config load and emits a `tracing::warn!` naming the field, its
  deprecation status, and the ignored semantics. The previous
  validation of the 1..=31 range is gone; any value (including
  formerly out-of-range values) loads successfully. The field will
  be removed from the schema in the next release; remove the entry
  from `config.json` to silence the warning. The rationale is that
  basis reconstruction now matches stored cut rows by slot identity
  alone, which makes the activity-window mask unobservable.

### Changed

- Stage-LP state pinning uses column bounds (`set_col_bounds`) on the
  incoming-state columns instead of equality rows. The `storage_fixing`,
  `lag_fixing`, and `anticipated_state_fixing` row ranges in
  `StageIndexer` are permanent empty sentinels (`0..0`); callers must
  use `StageIndexer::state_to_lp_incoming_column` to resolve the column
  index for both pinning and dual extraction.
- Cut subgradient extraction reads `view.reduced_costs[col]` (unscaled
  by multiplying by `col_scale[col]`) instead of `view.dual[row]`. The
  per-LP backward solve avoids `N + N*L + A*K` redundant equality rows
  per stage.
- Cut deactivation toggles a cut row's RHS bounds to the
  `f64::INFINITY` sentinel (trivially satisfied) instead of removing
  the row from the LP. The cut pool is append-only: every cut ever
  generated remains stored at a stable slot index for the lifetime of
  the run. Stage-LP cut rows are stable across iterations, including
  after cut-selection deactivation.
- Iteration template rebake now includes only active cuts. The
  per-iteration baked stage template carries one row per active cut in
  `active_cuts()` iteration order — inactive cuts are not encoded at
  sentinel `[-INF, +INF]` bounds. Recovers ~29% wall-time regression
  observed on production-scale convertido cases under the earlier
  sentinel-bake design.
- `training/metadata.json` `row_pool` carries `cuts_active` (active
  cuts at end-of-run) and `peak_active`. The `cuts_in_lp` field
  introduced earlier in this development cycle is removed; the value
  was tied to the sentinel-bake model and is no longer meaningful.
  Existing manifests that carry `cuts_in_lp` are silently accepted —
  the field is dropped on read.
- `training/convergence.parquet` row-selection schema carries
  10 columns. The `cuts_in_lp` column from the sentinel-bake model is
  removed; `cuts_active` stays.
- `ModelProvenanceReport` is restructured into nested `inflow` and
  `hydro_production` sub-sections, aggregating FPHA-plane and
  evaporation-reference source counts from `HydroModelProvenance` via
  order-invariant tallies. The report and `HydroModelSummary` are now
  `Serialize + Deserialize` so they can be persisted and read back by
  `cobre summary`.
- Cobre's offline geometric-mean column/row prescaler is restored and
  HiGHS's internal simplex scaler is disabled (`simplex_scale_strategy:
0`) across all phase profiles. The prescaler conditions the LP matrix
  before the first solve; HiGHS therefore operates on an already-scaled
  system and no double-scaling occurs. The retry escalation ladder
  inherits the `Off` strategy throughout.
- HiGHS primal and dual feasibility tolerances are tightened to `1e-9`
  for all three phase profiles (forward, backward, simulation). Prior
  releases used the HiGHS default of `1e-7`.
- The `cobre-comm` shared-memory subsystem (`SharedMemoryProvider`,
  `SharedRegion`, `LocalCommunicator`, `HeapRegion`) is now gated
  behind an off-by-default `shared-memory` Cargo feature. Downstream
  crates that previously compiled against these types unconditionally
  must add `features = ["shared-memory"]` to their dependency
  declaration, or remove the usage.
- `SharedMemoryProvider::split_local` now returns `LocalCommKind` (a
  concrete enum: `Local(LocalBackend)` or, under `mpi + shared-memory`,
  `Ferrompi(FerrompiLocalComm)`) instead of `Box<dyn LocalCommunicator>`.
  This is a breaking change for crates that stored or forwarded the
  return value as a trait object.
- `training/metadata.json` and `simulation/metadata.json` now persist
  the full execution topology (per-host rank layout via `DistributionInfo.hosts`),
  training and simulation solve-stat summaries, and the expected-cost
  statistics (mean, std, CVaR), so `cobre summary` can reproduce the
  complete live run end-block from a finished output directory.
- Warm-start and resume training reuse the prior policy's stored LP bases:
  the first training iteration warm-starts its solves from the checkpoint
  basis instead of cold-starting. The same checkpoint basis reconstruction
  (which recovers per-cut slot identity from the stored basis) also speeds
  up simulation-only / `cobre simulate` runs.

### Added

- **Python bindings** (`cobre-python` crate, `import cobre`). A PyO3
  extension module exposing the full solver lifecycle to Python 3.12+,
  built with maturin's mixed layout. The public surface includes:
  - `cobre.Study` — load a case directory once; call `.train()` and
    `.simulate(policy)` without re-reading disk between calls; accepts
    `config_overrides` (dotted-key flat dict), `output_dir`, and
    `threads`.
  - `cobre.Policy` — in-memory trained-policy handle returned by
    `Study.train()`; exposes `.iterations`, `.final_lower_bound`,
    `.final_upper_bound`, `.evaluate(stage, state)`, and
    `.cut_matrix(stage)` for stochastic introspection. Load from disk
    via `Study.load_policy()`.
  - `cobre.run.run` — single-call equivalent of `cobre run` (load →
    train → simulate), GIL released for all Rust computation, same
    output artifacts as the CLI.
  - `cobre.io.load_case`, `cobre.io.validate` — case loading and
    validation with the same structured report as the CLI.
  - `cobre.results.*` — `load_results`, `load_convergence`,
    `load_convergence_arrow`, `load_simulation`, `load_simulation_arrow`,
    `load_policy`, `load_stochastic`, `report`; `Stochastic` read-only
    introspection class.
  - `cobre.model.*` — `System`, `Bus`, `Line`, `Thermal`, `Hydro`,
    `EnergyContract`, `PumpingStation`, `NonControllableSource`
    read-only model view classes.
  - `cobre.errors.*` — structured exception hierarchy rooted at
    `CobreError(Exception)`: `ValidationError(ValueError)`,
    `PolicyIncompatibleError(ValueError)`, `CaseIoError(OSError)`,
    `OutputError(OSError)`, `SolverError(RuntimeError)`,
    `SimulationError(RuntimeError)`. Every leaf also subclasses the
    matching Python builtin so existing `except OSError` handlers
    continue to work.
  - `cobre.schema.export` — JSON schema export.
  - `Study.train(on_iteration=...)` accepts an optional Python callable
    invoked at each training-iteration boundary (dict with `kind`,
    `iteration`, `lower_bound`, `upper_bound`, `gap`, `wall_time_ms`);
    a truthy return requests a cooperative stop; a raising callback or
    `KeyboardInterrupt` is propagated after training artifacts are
    written. The GIL is reacquired only at iteration boundaries — never
    in the LP hot loop.
  - A per-call scoped rayon thread pool is created for each `run`,
    `train`, and `simulate` invocation so sequential calls with
    different `threads` values each honour their own count.
  - Output parity with the CLI: every file written by `cobre run` is
    also written by the Python bindings on the same code path, and
    vice-versa.
- Per-phase LP solve profile mechanism. The LP solver is wrapped by
  `ProfiledSolver<S>`, which carries a solver-specific profile type
  (`S::Profile`) and applies it to the inner solver at phase boundaries.
  The profile type is exposed via the `SolverInterface::Profile`
  associated type, so different solvers (e.g. `HiGHS`, CLP) can each
  declare their full native tuning surface without a lossy abstraction.
  For the `HiGHS` backend the concrete type is `HighsProfile`, with
  fields for feasibility tolerances, iteration caps, dual edge-weight
  strategy, scale strategy, and price strategy. The profile is
  re-applied automatically before every solve to survive solver-internal
  option resets.
- Backward-phase LP solver tuning. `BACKWARD_PROFILE` overrides one
  `HiGHS` option relative to the forward/simulation default:
  `simplex_price_strategy` switches from Row (`1`) to Row Hyper-Sparse
  (`2`), exploiting the sparsity of cut-subgradient rows that dominate
  the backward LP row count. The dual edge-weight strategy remains Devex
  on all three phases; empirical sweeps showed Dantzig and Steepest-Edge
  alternatives net worse on wall time and tail latency respectively.
- Cut-selection kernel rebuilt around an m-block GEMM
  (`matrixmultiply::dgemm`). The kernel evaluates all populated cuts
  (active and inactive) at every visited forward-pass trial point, and
  applies the configured survival rule (Level1, LML1, Dominated). All
  three variants share one implementation that treats reactivation and
  deactivation symmetrically. Intra-stage work is distributed across
  the rayon thread pool via 8-trial-point blocks; the OR-merge across
  tasks is commutative so determinism is preserved regardless of thread
  count. The hot path allocates nothing beyond the bounded fold-leaf
  scratch.
- `tie_tolerance` field on `CutSelectionStrategy` variants controls
  the per-state max-survival tolerance used by the value-evaluation
  kernel.
- Trial-point sliding window on `VisitedStatesArchive`: the archive
  retains the most recent `window_size` forward-pass trial points and
  evicts older entries, bounding memory growth on long runs while
  keeping recent states visible to cut selection.
- Cut reactivation wired into the training loop: cuts deactivated in
  a prior iteration that the kernel selects for survival are reactivated
  before the next backward pass.
- `cobre report` now surfaces final bounds (`final_lower_bound`,
  `final_upper_bound`) and expected cost (`mean_cost`, `std_cost`,
  `cvar`) as top-level convenience keys in the JSON output, alongside
  the full nested metadata. Legacy output directories degrade gracefully
  to default values.
- `cobre summary` reproduces the complete live run end-block (execution
  topology, hydro models, model provenance, training, simulation) from
  a finished output directory without re-running the solver. Reads the
  persisted metadata sidecars and degrades gracefully when a sidecar is
  absent.
- Legacy lower bound recovery: `cobre summary` and `cobre report` can
  reconstruct the final lower bound from `training/convergence.parquet`
  when `training/metadata.json` predates the bounds field, preserving
  read-back for output directories from prior releases.
- `cobre-io` exposes generic parquet readers and JSON sidecar
  read/write helpers (`read_provenance_report`, `write_hydro_model_summary`,
  etc.) so the CLI and Python bindings can share one load path for
  structured output artifacts.

### Removed

- `basis_reconstructions` column from `training/solver/iterations.parquet`.
  The column was always zero for every row and phase (the field was
  never packed into the MPI solver-stats wire format) and carried no
  diagnostic value. Consumers reading this column by name must remove
  the reference; the column is absent from the schema going forward.

### Fixed

- Forward-pass solver statistics are now aggregated across all MPI
  ranks before being written to `training/solver/iterations.parquet`.
  Previously only rank 0's contribution reached the parquet writer,
  so `lp_solves`, `simplex_iterations`, and timing fields for forward
  rows were understated by a factor of `world_size` on multi-rank runs.
- `cobre report` and `cobre summary` now report the true generated cut
  count (cuts actually added by `add_cut`) rather than the slot
  high-water mark, which over-counted by `forward_passes × stages` due
  to the 1-based iteration indexing leaving the warm-start reservation
  block permanently empty.
- Algorithmic strategy options (`simplex_dual_edge_weight_strategy`,
  `simplex_scale_strategy`, `simplex_price_strategy`) are now fully
  re-applied after a retry-level restore alongside the feasibility
  tolerances. Previously only the tolerance fields were re-applied,
  silently reverting the backward-phase price strategy to the HiGHS
  default on the post-retry attempt.
- Simulation-phase output-write errors (simulation results, solver
  stats) are now routed to `CaseIoError(OSError)` and
  `SimulationError(RuntimeError)` respectively in the Python bindings,
  matching the exception types raised by the equivalent training-phase
  failures. Previously these errors fell through to `SolverError`.
- `cobre init` / `1dtoy` quickstart template corrected: the bus deficit
  cost is raised so load shedding dominates operational constraint
  penalties, and `initial_conditions.json` gains the `$schema` URL
  present on all other template files. The scaffolded case now passes
  `cobre validate` with zero errors and zero warnings.
- The redundant Hydro-production sub-section is removed from the
  `cobre summary` and live run end-block output. It duplicated the
  Hydro-models section and always printed a misleading
  `FPHA planes: 0 computed, 0 precomputed` line for studies with no
  FPHA hydros. The source breakdown remains in `model_provenance.json`
  and is accessible via `cobre report` and the Python bindings.
- Warm-start and resume training now apply the loaded policy's cuts on the
  first iteration. Previously the first post-resume iteration solved a
  cut-less, myopic policy — producing a spuriously high upper-bound estimate
  (the lower bound was already correct) and a wasted iteration — before
  self-correcting on the next iteration.
- Loading a policy for simulation-only mode (training disabled) no longer
  trips a basis-reconstruction assertion in debug builds. Release builds were
  unaffected (the assertion compiles out), but debug builds — e.g. the Python
  bindings under `maturin develop` — previously panicked when reconstructing
  the LP basis from a checkpoint.

### Note

The cut-sync wire format was briefly bumped to version 2 during
development to carry an `ActivityUpdateRecord` alongside cut records;
this never shipped and has been removed. The wire format remains at
version 1 (cut records only); cross-version MPI runs are not a
supported deployment mode.

## [0.7.0] - 2026-05-24

### Added

- Add per-phase solver profile mechanism for training and simulation passes.

- Anticipated thermal dispatch is now fully implemented. Plants declared
  with `anticipated_config = { lead_stages: K }` in `system/thermals.json`
  participate in the LP via a decision variable at stage `t` that becomes
  generation at stage `t + K` (fishing constraint), with the ring-buffer
  state propagated across stages by the forward pass and the backward
  pass extracting cut subgradients w.r.t. the matured slot.
- `initial_conditions.json` accepts a new `past_anticipated_commitments`
  array seeding pre-horizon commitments. Each entry maps a `thermal_id` to
  a `values_mw` array whose length must equal that plant's `lead_stages`.
  The semantic validator rejects mismatched lengths and references to
  non-anticipated thermals.
- `simulation/thermals.parquet` populates three columns for anticipated
  plants: `is_anticipated` (bool); `anticipated_committed_mw` (Float64,
  null outside delivery stages) — the scalar committed MW that matures at
  this stage, read from slot 0 of the `anticipated_state` ring buffer; and
  `anticipated_decision_mw` (Float64, null outside the decision horizon) —
  the commitment placed at this stage for delivery `K` stages later.
- `training/dictionaries/state_dictionary.json` includes
  `anticipated_state` entries in slot-major plant-minor order
  (`K_max * n_anticipated` entries with `entity_type: "thermal"`,
  `slot_index`, and `entity_id`).
- Generic constraints now support `anticipated_decision(N)` as an
  expression term, where `N` is the anticipated thermal's `id`. The term
  references the stage-level commitment variable — the MW quantity placed
  at the current stage for delivery `K` stages later. Referencing a
  non-anticipated thermal is a hard semantic error. Using
  `thermal_generation(N)` on an anticipated thermal emits a
  `SemanticAmbiguity` warning to flag likely intent mismatch.

### Changed (breaking)

- The `gnl_config` field in `system/thermals.json` is renamed to
  `anticipated_config`. Its sub-field `lag_stages` is renamed to
  `lead_stages`. Files using the old keys will fail to parse.
- Three columns in `simulation/thermals.parquet` are renamed for
  consistency: `is_gnl` → `is_anticipated`, `gnl_committed_mw` →
  `anticipated_committed_mw`, `gnl_decision_mw` → `anticipated_decision_mw`.
  Code reading these columns by name must be updated.

### Fixed

- Anticipated-thermal cut machinery: the post-shift `anticipated_state`
  ring-buffer value is now decoupled from the state-fixing column via a
  new `anticipated_state_out` LP column block, eliminating a
  decision-write coefficient that corrupted `state_to_lp_column`'s
  Less-branch routing and previously drove `d_t = 0` for in-horizon
  stages `t >= 1` when `K >= 2`. The fishing predicate is now active at
  every stage in `[0, T-1]`, so pre-horizon seeded `values_mw[k]` is
  delivered end-to-end (committed_at(0) == values_mw[0], etc.). Verified
  by K=1, K=2, and K=3 pre-horizon delivery integration tests plus a
  NEWAVE-parity bridge fixture (ST.CRUZ NOVA, K=1,
  values_mw=[204.5647]).
- Anticipated semantic validator no longer rejects in-bounds non-zero
  `past_anticipated_commitments.values_mw` entries. The previous
  `SemanticAmbiguity` warning ("non-zero seeds will load but produce the
  same dispatch as all-zero seeds") is removed; the bounds-check rule
  (`v_k ∈ [min_mw, max_mw]`) remains. The `SemanticAmbiguity` warning
  for `thermal_generation(N)` on an anticipated thermal in generic
  constraints is preserved.

## [0.6.2] - 2026-05-20

### Added

- New `allow_curtailment: bool` field on `NonControllableSource` (default
  `true`, preserving existing case behaviour). When set to `false` on an
  entity, the LP pins its generation column to the realized availability
  on every scenario: lower and upper column bounds both equal
  `max_generation_mw * alpha * block_factor`, where
  `alpha = clamp(mean + std*eta, 0, 1)` is the per-(stage, scenario)
  availability ratio drawn from `non_controllable_stats.parquet` and
  `block_factor` is the per-(stage, block) shape factor from
  `non_controllable_factors.json`. Use this on NEWAVE-derived
  `geracao_usinas_nao_simuladas` aggregates (PCH, PCT, EOL, UFV, MMGD)
  that the source model pre-nets from `MERC`; with the default Cobre
  schema the LP was free to curtail these because curtailment is one of
  the cheapest LP slacks, leading on the bundled deterministic 1983 case
  to ≈ 18 % of total NCS supply being curtailed, a ≈ +15 % hydro-dispatch
  swing, and ≈ −23 % spillage versus NEWAVE. Setting
  `allow_curtailment = false` on the must-run aggregates restores
  dispatch parity with NEWAVE while preserving per-source observability
  in the simulation outputs. JSON schema accepts the field as optional;
  absent → `true`.
- `ModelProvenanceReport.historical_library_past_inflows_digest: Option<u64>`
  records a SipHash-1-3 fingerprint of `initial_conditions.past_inflows`
  when the historical scenario scheme is active. Downstream consumers can
  compare the stored digest against the current `past_inflows` to detect
  a stale historical library that needs re-standardisation. Serialised
  under `#[serde(skip_serializing_if)]` so non-historical runs keep their
  JSON provenance unchanged. The cobre-python production path populates
  the field from `setup.scenario_libraries.training.historical`.

### Changed

- **API (breaking, hydro penalty rename)**: `HydroPenalties.fpha_turbined_cost`
  is renamed to `HydroPenalties.turbined_cost`, cascading through
  `RawHydroPenalties`, the `penalty_overrides.parquet` column, the output
  schemas, and the `simulation_writer` record. The turbined-flow
  regularisation cost is now applied universally to every hydro's turbine
  column in the LP objective (previously gated behind `is_fpha` in
  `fill_turbine_columns`, so constant-productivity plants paid nothing and
  diverged from NEWAVE). The dead `turbined_cost` field on
  `ResolvedProductionModel::Fpha` is dropped (the cost now lives in the
  penalty cascade). A latent extraction bug is fixed in the process:
  `SimulationCostResult.turbined_cost` was summing `primal*obj` over
  `indexer.generation` (FPHA generation columns, objective = 0), so the
  field was silently zero in real runs; it now sums over `indexer.turbine`
  where the cost lives. JSON schemas regenerated via `cobre schema export`
  with no hand edits. Two LB regression pins shift by ≈ 0.10 % under the
  new universal cost (`basis_reconstruct_churn::PINNED_FINAL_LB`,
  `deterministic::D03_EXPECTED_COST`).
- **API (breaking, PAR primitives)**: dropped the explicit `order: usize`
  parameter from `evaluate_par`, `evaluate_par_inflow`, `solve_par_noise`, and
  removed the equivalent `par.order(h)` inner-loop bound from
  `evaluate_par_batch` / `solve_par_noise_batch` / `solve_par_noises` /
  `evaluate_par_inflows`. All four primitives now iterate the full
  `psi.iter()` slice (via `psi.len()` in the batch variants), and require
  `lags.len() >= psi.len()`. `psi.len()` is the authoritative number of lag
  terms in the model — equal to the AR order for classical PAR(p) and to 12
  when PAR(p)-A annual is active (the materialised annual coefficient is
  spread across the extra positions in `PrecomputedPar::psi_slice`). Callers
  that previously truncated to AR order silently dropped the annual
  contribution; see the matching `Fixed` entry below.

### Fixed

- **PAR(p)-A annual coefficients were omitted from the cut sparse mask,
  producing over-estimating Benders cuts and LB > UB at convergence.**
  `StudySetup::new` was passing `par.order(h)` (the classical AR order) as
  the per-hydro lag-state-slot count to `StageIndexer::set_nonzero_mask`.
  With PAR(p)-A active, `PrecomputedPar::psi_slice` widens to
  `max_order = 12` and the standardised annual coefficient `ψ̂/12` fills
  positions `p..12`. The LP loads all 12 psi entries into the AR-dynamics
  row, but the cut sparse mask omitted state coefficients on lag slots
  `p..12` — so cut rows built via `build_cut_row_batch_into` were missing
  the trailing state dependencies. At the visited state this shifted the
  cut hyperplane above the true LP value, producing systematically
  over-estimating cuts. On the bundled NEWAVE 1983 case (Camargos
  AR(4)+A on every hydro), the gap converged to LB > UB by ≈ 7 % with 41
  of 50 iterations recording negative gaps; disabling annual via
  `order_selection = pacf` collapsed the gap to 0.05 %. After the fix the
  same case with `pacf_annual` converges to 0.003 % final gap with zero
  negative-gap iterations across all 50. `PrecomputedPar` now records a
  per-hydro `has_annual: Box<[bool]>` at build time and exposes
  `effective_lag_count(h)` (returns `max_order` when `has_annual[h]`,
  else `orders[h]`); `StageIndexer::set_nonzero_mask`'s `ar_orders`
  parameter is renamed to `lag_counts` to reflect its true meaning. New
  regression test `nonzero_mask_par_a_includes_full_psi_stride` pins
  the contract. This is the direct analog of the η-inversion bug fixed
  above — same `order`-vs-`psi.len()` confusion, cut-construction path
  instead of standardisation path.
- **PAR(p)-A annual was silently dropped at standardisation time when the
  scheme was `historical` or `external`.** `standardize_historical_windows`
  (and the analogous external path) built their per-hydro `lag_buf` only up
  to `par.order(h)` lag slots and passed `order_h` as the iteration bound to
  `solve_par_noise`. With PAR-A active, `PrecomputedPar::psi_slice` returns
  12 entries (AR coefficients in slots `0..order`, plus the spread annual
  coefficient `ψ̂/12` in slots `order..12`), all of which the LP loads via
  `lp_builder/matrix.rs:834-841`. The standardised `η` therefore omitted the
  annual contribution from lags `order..12`, while the LP applied it in
  full — so forward replays produced biased inflows even when the stage-0
  lag state matched the window's pre-study lags exactly. Worst-case observed
  on the bundled NEWAVE 1983-anchored case (Camargos, AR(4) +
  annual_coefficient ≈ −0.225): a ≈ 11 % shortfall vs the raw historical
  observation. After the fix, the same case reconstructs every hydro at
  stage 0 to within 10⁻¹² of the historical observation. The fix combines
  the API change above with two callsite updates that now fill the full
  `psi.len()` lag slots in `standardize_historical_windows` and
  `standardize_external_inflow`. New regression test
  `crates/cobre-stochastic/tests/par_a_historical_replay.rs` pins the
  invariant directly.
- `ClassSampler::Historical::apply_initial_state` no longer overwrites the
  inflow-lag portion of the stage-0 state vector with the window-preceding
  raw historical inflows. Previously, the forward pass replayed the lag
  state of the historical year being sampled (e.g. when scenario `m`
  replayed 1983, the forward LP started from the 1982-Q4 lag values), while
  the lower-bound and backward-pass evaluators kept the user-supplied
  `initial_conditions.past_inflows` lags. The two paths therefore evaluated
  V₀ at different `x_0` on every historical-replay case, producing a
  structural, typically negative SDDP gap (≈ −19 % on the bundled
  NEWAVE-derived 1983 deterministic case) that did not close with
  iteration count. The historical window now contributes only its
  standardized noise residuals via `fill`; the initial inflow lags come
  uniformly from `initial_conditions.past_inflows` for every scenario,
  matching NEWAVE's `TENDENCIA HIDROLOGICA` convention. Cases using
  `InSample`, `OutOfSample`, or `External` schemes are unaffected
  (bit-identical output — those variants were already no-ops). Cases using
  the `Historical` scheme will see different forward upper bounds and
  meaningful gap closure to cut tightness.

- **Historical η inversion re-rooted at `past_inflows` (LB/UB x₀ sharing).**
  `standardize_historical_windows` previously inverted the PAR noise residual
  η for each window stage using `window_pre_study_lags` as the lag state seed
  — i.e. the raw historical inflows from the year before each window year.
  After commit dc96030 (`apply_initial_state` no-op), the SDDP forward pass
  starts every scenario from `initial_conditions.past_inflows` (the
  user-supplied x₀), not from the window-preceding lags. The two paths
  therefore built their lag chains from different x₀, producing a
  systematic per-stage offset `z_h = target + Σψ·(past_inflows −
window_lag)` that propagated through all stages and prevented exact
  historical replay even at stage 0 after d0e4a42. The fix re-roots the
  inversion on a rolling lag chain seeded from `past_inflows`, following the
  same accumulate/finalize pattern as `standardize_external_inflow`. The
  rolling chain is advanced each month via `StageLagTransition`; uniform-
  monthly transitions are used by default, with noop transitions (produced
  by an empty `SeasonMap`) falling back to uniform-monthly in the inner loop.
  A `past_inflows_digest: u64` field (SipHash-1-3 of all `past_inflows`
  values) is stored on `HistoricalScenarioLibrary` to enable stale-library
  detection when `past_inflows` change between calls. A `debug_assert!`
  guard fires when `stage_lag_transitions` contain non-trivial (non-monthly)
  entries, marking `TODO(historical-replay-non-monthly)` for future work on
  sub-monthly and multi-monthly study configurations.
  `build_historical_inflow_library` and `build_opening_tree_library` both
  forward `past_inflows` and `stage_lag_transitions` to
  `standardize_historical_windows`. After the fix, every historical-scheme
  forward pass that starts from `past_inflows` reconstructs the raw
  historical observation to within floating-point precision at every stage,
  restoring LB/UB consistency at x₀. New tests T2–T6 in
  `crates/cobre-stochastic/tests/par_a_historical_replay.rs` pin the
  invariant for differing `past_inflows`, AR(0), truncated lag vectors,
  multi-window cases, and the non-trivial-transition guard.

## [0.6.1] - 2026-05-18

### Fixed

- The linearised evaporation flow variable `Q_ev` is now bounded
  symmetrically `[-q_max, +q_max]` instead of `[0, q_max]`, where
  `q_max = |k_evap0 + k_evap_v · v_max| · margin`. Previously, when
  the per-stage net evaporation coefficient `c_ev[month]` was negative
  (net rainfall over the lake surface exceeds open-water evaporation,
  common in 5–7 months/year on tropical and subtropical basins), the
  upper bound clamped to zero and the equality row forced the
  over-evaporation slack to absorb the imbalance at every wet-month
  stage of every scenario. Negative `Q_ev` values now flow into the
  storage continuity equation as net rainfall input, and the violation
  slacks fire only on genuine modelling infeasibility. Cases with
  all-positive monthly coefficients are unaffected (bit-identical
  output).

- `evaporation_m3s` in `simulation/hydros.parquet` is now a signed
  value: positive entries continue to represent net evaporative loss,
  while negative entries represent net rainfall input absorbed by the
  reservoir. Downstream consumers that filtered or asserted
  `evaporation_m3s >= 0` will need to be updated.

### Changed

- The `HydroEvaporation` variable in
  `cobre_core::generic_constraint::VariableRef` is documented as a
  signed net-flow quantity rather than an outflow-only quantity.
  Generic constraints referencing it that previously assumed
  non-negativity may need their bounds revisited.

- Docstring corrections in `LinearizedEvaporation`,
  `EvaporationIndices`, and the `penalty_overrides` schema reference:
  the linearised coefficients `k_evap0`, `k_evap_v`, and the violation
  slacks `f_evap_plus`/`f_evap_minus` are stage-averaged flows in m³/s,
  not volumes in hm³ as the documentation previously claimed. No
  runtime behaviour change.

### Added

- New deterministic regression case `d17-evaporation-mixed-sign`
  exercising a single-hydro four-stage horizon spanning Oct → Jan with
  monthly evaporation coefficients that switch sign across stages.
  Guarded by `parity_hash_d17` (byte stability) and
  `d17_signed_evaporation::d17_evaporation_is_signed_per_month`
  (explicit sign assertions per stage).

## [0.6.0] - 2026-05-18

### Added

- Reserved three new crate names in the workspace as placeholders for
  upcoming verticals: `cobre-flow` (power flow algorithms), `cobre-uc`
  (MILP-based unit commitment for hydrothermal dispatch), and
  `cobre-emt` (electromagnetic transient analysis). Each ships as an
  empty library following the existing `cobre-tui` reservation pattern
  and is not yet implemented.

- Five new columns in `simulation/hydros.parquet`, populated for every
  `(stage, block, hydro)` row:
  - `equivalent_productivity_mw_per_m3s` (MW/(m³/s)) — equivalent
    productivity `ρ_eq`. For `ConstantProductivity` and
    `LinearizedHead` hydros it is the stored input scalar; for FPHA
    hydros it is derived from VHA geometry, the specific productivity
    `ρ_esp`, and the reference operating point `(V_ref, Q_ref)`.
  - `accumulated_productivity_mw_per_m3s` (MW/(m³/s)) — accumulated
    productivity `ρ_acum`: the sum of `ρ_eq` over each hydro and every
    plant downstream of it along the cascade.
  - `incremental_inflow_energy_mw` (MW) — incremental natural inflow
    expressed as energy, computed as
    `ρ_acum · incremental_inflow_m3s`.
  - `stored_energy_initial_mwh` (MWh) — stored reservoir energy at the
    start of the block,
    `(storage_initial_hm3 − min_storage_hm3) · ρ_acum · 10⁶ / 3600`.
  - `stored_energy_final_mwh` (MWh) — stored reservoir energy at the
    end of the block, using the same formula with the final storage.

### Removed

- `cobre_stochastic::par::fit_par_annual_with_reduction` and its
  `ReducedOrderFit` return type. The function had been exposed alongside
  the wired PAR(p)-A reduction loop in `cobre-sddp` but was never called
  from production code, and its `φ + ψ̂/12` "effective coefficient"
  contribution check disagreed with the φ-only check used by the actual
  estimation pipeline. The wired path
  (`apply_annual_prepass_reductions` → `reduce_entity_orders_annual`)
  remains the single source of truth.

- Column `productivity_mw_per_m3s` from `simulation/hydros.parquet`. The
  column was `null` for FPHA hydros and duplicated input for non-FPHA
  hydros. It is replaced by the five always-populated energy columns
  listed above.

- Inline `productivity_mw_per_m3s` field on
  `system/hydros.json` `generation` blocks. The per-stage
  productivity coefficient is now authored on every
  `stage_ranges[]` or `seasons[]` entry in
  `system/hydro_production_models.json` under the same key.
  Cases that previously omitted
  `system/hydro_production_models.json` and relied on the
  entity-level scalar must now ship the file with one entry per
  hydro.

- Input files `system/scalar_parameter_definitions.parquet` and
  `system/scalar_parameter_values.parquet`. Scalar parameters
  for generic constraints (`@name` references) are now authored
  in a single `system/scalar_parameters.json` carrying one
  object per parameter with a `kind` discriminator and
  kind-specific payload (`value`, `values`, or
  `computed_spec`).

- Several `config.json` fields whose values were no longer wired to any
  downstream consumer: `modeling.inflow_non_negativity.penalty_cost` (the
  per-hydro slack cost is now sourced exclusively from
  `penalties.json::hydro.inflow_nonnegativity_cost`),
  `training.cut_formulation`, `training.forward_pass`,
  `simulation.policy_type`, `simulation.output_mode`, `simulation.output_path`,
  and the legacy export flags
  (`training`, `cuts`, `vertices`, `simulation`, `forward_detail`,
  `backward_detail`, `compression`) under `exports`. Any case still carrying
  these keys is rejected at parse with an `unknown field` error.

- The `"fixed"` value of `estimation.order_selection`. Only `"pacf"` and
  `"pacf_annual"` remain valid; the alias was previously accepted and silently
  mapped to `"pacf"`.

- The `version` field on `penalties.json`. The field was parsed but never
  consulted; schema-version gating is enforced via the `$schema` URL instead.

### Changed

- `system/hydro_production_models.json` —
  `productivity_mw_per_m3s` is now optional for
  `constant_productivity` and `linearized_head` models. Omit the
  field (or set it to `null`) when the value is supplied per stage
  by `system/hydro_energy_productivity.parquet`.

- Productivity validation relaxed from strictly positive (`> 0.0`)
  to non-negative (`>= 0.0`) for `productivity_mw_per_m3s` in
  `system/hydro_production_models.json`,
  `equivalent_productivity_mw_per_m3s` in
  `system/hydro_energy_productivity.parquet`, and
  `specific_productivity_mw_per_m3s_per_m` in the same parquet.
  A value of `0.0` is accepted as a planned-outage marker for the
  affected `(hydro, stage)`; the LP treats these coefficients as
  multipliers, so zero produces zero generation without any
  division-by-zero hazard. Negative values are still rejected.

- `config.json::modeling.inflow_non_negativity.method` is now a typed
  enum with four valid values (`"none"`, `"truncation"`, `"penalty"`,
  `"truncation_with_penalty"`). Unrecognised strings were previously
  silently coerced to `"none"`; they are now rejected at parse.

- `config.json::training.cut_selection.threshold` is consulted only
  when `method = "level1"` (it is the activity-count cutoff for cut
  deactivation). The `"lml1"` and `"domination"` methods no longer
  fall back to `threshold`; they now require `memory_window` and
  `domination_epsilon` respectively, and reject configurations that
  omit them.

### Breaking Changes

**Output schema** — `simulation/hydros.parquet` grows from 31 to 35
columns. The column `productivity_mw_per_m3s` is removed; five new
non-nullable `Float64` columns are added after `generation_mwh`:
`equivalent_productivity_mw_per_m3s`,
`accumulated_productivity_mw_per_m3s`, `incremental_inflow_energy_mw`,
`stored_energy_initial_mwh`, `stored_energy_final_mwh`. The
accompanying `variables.csv` dictionary file is updated to reflect the
new columns and drops the entry for the removed one.

**FPHA validation** — Studies that declare a hydro with
`generation_model: "fpha"` must now supply either VHA geometry plus
`specific_productivity_mw_per_m3s_per_m`, or (when the optional
`system/hydro_energy_productivity.parquet` override is in place) a
per-`(hydro, stage)` `equivalent_productivity` entry. Cases that
previously loaded with neither source now fail fast at setup time with
an error that names the offending plant and lists the three accepted
remediations.

**Hydro productivity authoring** — Studies must remove
`productivity_mw_per_m3s` from every `hydros.json` `generation`
block and supply the same value via a
`system/hydro_production_models.json` entry. For non-FPHA models
(`constant_productivity`, `linearized_head`) the field is
positive when present on every `stage_ranges[]` / `seasons[]`
entry, and may be omitted (or `null`) to defer to the parquet
override (see the next entry). The previous opt-in
`productivity_override` key on those entries is renamed to
`productivity_mw_per_m3s`; the override semantics (replace the
entity-level base value) no longer apply because there is no
longer an entity-level base value.

**Productivity resolution across files** — The
`equivalent_productivity_mw_per_m3s` column in
`system/hydro_energy_productivity.parquet` now applies to **all**
hydro generation models, not only FPHA. A row supplying this value
for a non-FPHA hydro is honoured as the equivalent productivity
ρ_eq for that `(hydro, stage)` instead of being silently ignored.
Studies that supplied a value for the same `(hydro, stage)` pair
in both `system/hydro_production_models.json`
(`productivity_mw_per_m3s`) and `system/hydro_energy_productivity.parquet`
(`equivalent_productivity_mw_per_m3s`) are now rejected at load time
with a schema error naming both files. Studies that supplied a value
in neither file for a non-FPHA `(hydro, stage)` pair are likewise
rejected with a clear coverage-gap error, rather than failing deeper
in the SDDP setup layer.

**Scalar parameters file format** — The pair of input parquet
files `system/scalar_parameter_definitions.parquet` and
`system/scalar_parameter_values.parquet` is removed. Studies
that use scalar parameters in generic constraints must now ship
`system/scalar_parameters.json` instead. The new file uses one
array entry per parameter with a `kind` discriminator
(`constant`, `per_stage`, `seasonal`, or `computed`) and a
kind-specific payload. Authoring order is preserved at parse
time; LP coefficients are unaffected by authoring order
(parameters are looked up by `id` at LP-build time).

**Strict JSON parsing** — every input JSON file (`config.json`,
`penalties.json`, `stages.json`, `initial_conditions.json`, and every
file under `system/`, `constraints/`, and `scenarios/`) now rejects
unknown top-level and per-entry fields with a hard parse error.
Misspellings (`"max_outflow"` vs `"max_outflow_m3s"`) and stale
configuration keys from earlier releases that previously survived
silently must be fixed before a case will load. The error message
names the unrecognised field.

**Migration**

- Replace reads of `productivity_mw_per_m3s` with
  `equivalent_productivity_mw_per_m3s`. For `ConstantProductivity` and
  `LinearizedHead` hydros the numeric value is identical to the
  previous column; for FPHA hydros it is the derived `ρ_eq`
  (previously `null`).
- Notebooks and dashboards that filtered out the old `null` values for
  FPHA hydros can now treat all hydros uniformly.
- Downstream consumers that need accumulated cascade productivity,
  energy-units inflows, or reservoir energy state should read from the
  four new columns instead of computing them externally.
- Remove `productivity_mw_per_m3s` from every `generation` block
  in `system/hydros.json`; transfer the same numeric value into
  the matching entry in `system/hydro_production_models.json`
  under the same key.
- Rename any `productivity_override` keys in
  `system/hydro_production_models.json` to
  `productivity_mw_per_m3s`. The values do not change; the
  positive-value validation rule still applies, but the
  override-versus-default distinction is gone.
- Replace `system/scalar_parameter_definitions.parquet` and
  `system/scalar_parameter_values.parquet` with a single
  `system/scalar_parameters.json` (see the schema at
  `book/src/schemas/scalar_parameters.schema.json` and the
  authoring guide at `book/src/guide/scalar-parameters.md`).

## [0.5.1] - 2026-04-28

### Added

- Annual component extension to the periodic AR inflow model. Selecting
  `"order_selection": "pacf_annual"` in the estimation config activates the
  PAR(p)-A path: the fitting pipeline emits a new `AnnualComponent` triple
  (`coefficient`, `mean_m3s`, `std_m3s`) per (hydro, season) on top of the
  classical AR coefficients, allowing the model to capture multi-year
  hydrological persistence. The triple is exposed on `InflowModel.annual` and
  is also written to a new output file
  `output/stochastic/inflow_annual_component.parquet` (5 columns:
  `hydro_id`, `stage_id`, `annual_coefficient`, `annual_mean_m3s`,
  `annual_std_m3s`) by both the CLI and the Python bindings.

- `HistoryClass` taxonomy on per-(hydro, stage) historical buckets. Each
  observation series is classified inside the shared
  `estimate_seasonal_stats` routine, so the override applies on **both**
  the classical PAR(p) and PAR(p)-A paths: constant series (`Constant`),
  saturating caps such as turbine flow ceilings or low-flow constants
  (`Saturated`), and series dominated by more than 10% strictly negative
  observations (`ManyNegative`) are now detected automatically.
  `Constant` and `Saturated` buckets force the seasonal `(mean, std)` to
  `(value, 0)`, which yields a degenerate fit (order 0, no AR/annual
  coefficients) on either path — explicitly via the structural-zero
  short-circuit on PAR(p)-A, and implicitly via zeroed periodic
  autocorrelation on classical PAR(p). `ManyNegative` is purely
  diagnostic and does not override the fit.

- Two extensions to the PACF order-selection rule for the PAR(p)-A path:
  a structural-zero short-circuit forces the model to order 0 when the
  lag-1 conditional FACP is exactly zero (degenerate Schur complement),
  and a minimum-order-1 default keeps an AR(1) base whenever the lag-1
  FACP is well defined but no lag exceeds the 95% significance threshold.

- Maceira-Damazio iterative order reduction across the full periodic
  cycle for the PAR(p)-A path. After the initial PACF + Yule-Walker fit,
  the recursively-composed contributions of each lag through the
  periodic monthly chain are computed; if any contribution is negative,
  the offending season's AR ceiling is reduced and the fit is re-run at
  the new ceiling. The reduction iterates across all seasons until every
  season's contribution recursion yields non-negative entries. This
  prevents negative chain-composed contributions from propagating as
  unstable Benders cuts in downstream SDDP recursions.

### Changed

- Seasonal stats (`σ^Z_m`) now use the population (`1/N`) divisor on
  **both** the classical PAR(p) and PAR(p)-A paths, matching the
  Maceira-Damazio standard-deviation convention. Previously
  `estimate_seasonal_stats` used the Bessel-corrected (`1/(N-1)`)
  divisor; the function is shared between paths, so any consumer of
  `inflow_seasonal_stats.parquet` will see std values about 0.5–1.1%
  smaller than before (proportional to `sqrt((N-1)/N)`). The change
  propagates through `periodic_autocorrelation` (used by both paths) and
  affects the classical PACF order selection too.

- PAR(p)-A Z⊗A cross-covariance (`cross_correlation_z_a` /
  `cross_correlation_a_z_neg1`) now uses the max-bucket-size population
  divisor (`max(|A|, |Z|)`) instead of the strict-pair count. This
  change is PAR(p)-A specific — classical PAR(p) does not use Z⊗A
  cross-correlations.

- HiGHS default options retuned for warm-started master LPs dominated by
  many slack rows: Devex dual-edge weight pricing, dual-simplex cost
  perturbation disabled, initial-condition check disabled, row-wise PRICE
  strategy, and a loosened rebuild-refactor solution-error tolerance
  (`1e-6`). These changes alter the simplex trajectory and may yield a
  different optimal basis representation at the same objective value;
  the deterministic-suite parity hashes for D03, D06, and D07 were
  refreshed accordingly. Numerical answers are unchanged within solver
  tolerances.

- JSON Schemas under `book/src/schemas/` regenerated from the current
  Rust structs via `cobre schema export`. The previously committed
  schemas had drifted from the source of truth (most visibly,
  `CutSelectionConfig` was renamed to `RowSelectionConfig` and
  `ExportsConfig` was trimmed to its two active flags). No
  config-file shape change for users on the supported variants.

## [0.5.0] - 2026-04-25

Major refactor. Consumers must update `config.json`, any code calling
solver traits, and any tooling reading `solver/iterations.parquet`
before upgrading.

### Breaking Changes

**Public Rust API**:

- `TrainingResult` is `#[non_exhaustive]`; use `TrainingResult::new(...)`.
- `SolverInterface::record_padding_stats` → `record_reconstruction_stats(&mut self)`.
- `SolverInterface::clear_solver_state` removed.
- `WarmStartBasisMode` and `CanonicalStateStrategy` enums removed along
  with the `canonical_state_strategy` config fields.
- Four classification counters (`basis_preserved`, `basis_new_tight`,
  `basis_new_slack`, `basis_demotions`) collapsed into `basis_reconstructions`.
- `basis_rejections` + `basis_non_alien_rejections` → `basis_consistency_failures`.
- `add_rows_count`, `add_rows_time_ms`, `clear_solver_count`, and
  `clear_solver_failures` counters removed.
- `StoppingRuleResult.rule_name` is now `&'static str`;
  `StoppingRuleResult.detail` is now `Cow<'static, str>`. Call
  `.to_string()` on `rule_name` or use `detail.as_ref()` at consumers
  that previously expected `String`.
- `TrainingEvent::WorkerTiming.timings` field type changed from
  `[f64; 16]` (where 12 of 16 slots were always zero on per-worker
  events) to a new `WorkerPhaseTimings` struct with four named fields:
  `forward_wall_ms`, `backward_wall_ms`, `fwd_setup_ms`, `bwd_setup_ms`.
  The output Parquet schema for `training/timing/iterations.parquet`
  is unchanged. Consumers that read the variant payload directly must
  access the named fields rather than slot indices. The rank-aggregated
  `WorkerTimingRecord` writer record retains its 16-column layout and
  the `WORKER_TIMING_SLOT_*` constants remain public as the bridge
  between named fields and writer slots.

**Output schema** — `solver/iterations.parquet` drops from 23 to 19
columns. Added: `opening`, `rank`, `worker_id` (nullable i32),
`basis_reconstructions` (u64). Removed: the eight counters listed
above. Backward rows gain an `opening` dimension; forward rows are now
one per `(iteration, stage)` rather than one per iteration.

`cut_selection/iterations.parquet` drops `active_after_angular`
(10 → 9 columns).

**Policy FlatBuffer** — `CUT_FIELD_DOMINATION_COUNT` slot removed. Old
policies deserialise via graceful-absence; `"domination_count"`
disappears from `cobre.results.load_policy` per-cut dicts.

### Added

- **Backward-pass basis cache** — rank 0 captures a fresh basis per
  stage during the backward pass and broadcasts it end-of-iteration;
  next-iteration backward solves warm-start from the cache.
- **Basis reconstruction** — `CapturedBasis` wrapper with slot and
  state metadata; `reconstruct_basis` applies a stored basis across
  cut-set churn on forward, backward, and simulation paths. Controlled
  by `training.cut_selection.basis_activity_window` (1-31, default 5).
- **Weekly+monthly studies** — sub-monthly lag accumulation,
  `recent_observations` input for mid-season starts, terminal boundary
  cuts (`policy.boundary.{path, source_stage}`) for Cobre-to-Cobre FCF
  coupling, and non-uniform per-stage scenario counts.
- **Multi-resolution studies** — same-season noise group sharing,
  observation aggregation from finer to coarser resolution, and
  monthly→quarterly PAR transition.
- **Per-opening solver statistics** — backward rows carry `rank` and
  `worker_id` metadata for per-worker parity testing.
- `TrainingEvent::SimulationStarted` variant mirrors `TrainingStarted`
  for the simulation phase. Carries `n_scenarios`, `n_stages`, `ranks`,
  and `threads_per_rank`. Emitted once per rank before the parallel
  scenario loop so consumers can render a banner before any scenario
  completes.
- Non-TTY progress lines (pipes, `mpirun` aggregators, log files) now
  include an `[elapsed HH:MM:SS < eta HH:MM:SS]` trailing cell matching
  the interactive bar.
- Wire-format version bytes added to both the basis broadcast payload
  and cut records. The basis version field is at position 1 of the
  `i32` metadata buffer; the cut version byte is at byte 0 of each
  serialized cut record. Mismatches between sender and receiver produce
  typed validation errors rather than silent data corruption or panics.
- Typed errors for MPI correctness conditions:
  - Non-uniform worker counts across ranks are detected during the
    backward-pass handshake and reported as a validation error naming
    both the minimum and maximum observed counts.
  - Basis broadcast `i32` length overflow (more than `i32::MAX`
    elements) is detected before the MPI call and reported as
    `CommError::InvalidBufferSize`.
  - Wire-format version mismatches in the basis broadcast and in cut
    deserialization are reported as validation errors naming both
    expected and received versions.
  - Violated cut-count invariants in `sync_cuts` (negotiated count
    differs from actual serialized count) are reported as a validation
    error.
  - Stats counter values exceeding `2^53` (the f64 integer precision
    limit), which would corrupt MPI allreduce sums, are detected before
    the allreduce call and reported as an internal error.

### Changed

- **Lower-bound LP is strictly append-only.** Cuts are never removed
  from the LB LP, guaranteeing monotonic non-decreasing lower bound.
  Cut selection and budget enforcement continue to deactivate cuts in
  the shared pool for the forward and backward passes.
- **Cut management pipeline reduced to two stages**: strategy-based
  selection followed by budget enforcement.
- Internal refactor of `cobre-io` semantic-validation: the
  6 319-line `validation/semantic.rs` file is now split into
  7 cohesive domain submodules (`hydro`, `thermal`, `stages`,
  `scenarios`, `season`, `correlation`, `sobol`) plus a
  placeholder `shared` module for future cross-domain helpers.
  The two public entry functions
  (`validate_semantic_hydro_thermal`,
  `validate_semantic_stages_penalties_scenarios`) keep their
  paths under `cobre_io::validation::semantic::*`. No semantic-
  validation rule was added, removed, or modified.
- `TrainingEvent::SimulationProgress.scenarios_complete` now carries a
  global estimate under multi-rank execution (`local × ranks`, clamped
  to `scenarios_total`), matching the `scenarios_total` field's global
  scope. Single-rank runs are unchanged. Previously, rank 0 reported
  its local count against the global total, producing a misleading
  `50/100` final line on a 2-rank run.
- `cobre-comm` now requires `ferrompi >= 0.4.0`. The MPI bitwise-OR
  allreduce used to synchronize the active-window bitmap is dispatched
  directly to `MPI_Allreduce` with `MPI_BOR` instead of being emulated
  via an `allgatherv` + bytewise fold workaround.
- FPHA hyperplane file is now written to
  `<output_dir>/hydro_models/fpha_hyperplanes.parquet` in both the CLI
  (`--output` flag) and Python bindings (`output_dir` parameter).
  Previously the file was always written to
  `<case_dir>/output/hydro_models/`, ignoring the caller-specified
  output directory. Under multi-rank execution the file is written by
  rank 0 only; previously all ranks raced to write the same path.
- All sort comparators on `f64` in the hot path now use `f64::total_cmp`
  for deterministic NaN handling. `partial_cmp(...).unwrap_or(Equal)`
  patterns on the production training and simulation paths have been
  replaced.
- Rayon global thread pool initialization failure now emits a structured
  warning (configured threads, actual threads, error reason) and
  continues using the already-initialized pool. Previously a silent
  fallback was applied without any diagnostic output.

### Deprecated

- `RowSelectionConfig::threshold` is deprecated. The field remains
  parseable; supplying it now emits a `WARN`-level `tracing` event
  directing the user to `memory_window` (for `"lml1"`) or
  `domination_epsilon` (for `"domination"`).

### Removed

- **Angular diversity pruning** — config section, module, output
  column, and training event variant.
- **`SparseCut` module** — handled by the indexer.
- **`CutMetadata::domination_count`** — unused.
- **LB LP periodic rebuild** and its `CutRowMap` helpers.
- `FerrompiScratch` and the `FerrompiBackend::scratch` interior-
  mutability field. The native bitwise-OR allreduce no longer needs
  per-call scratch for counts/displs vectors.
  `unsafe impl Send + Sync for FerrompiBackend` comments were updated
  accordingly.
- `SimulationSummary::stage_stats: Option<Vec<StageSummaryStats>>`
  field and the `StageSummaryStats` struct were removed; the field was
  always `None` in production output and had no consumers.
- `ExportsConfig` reduced to two flags: `states` and `stochastic`. The
  seven previously declared fields (`training`, `cuts`, `vertices`,
  `simulation`, `forward_detail`, `backward_detail`, `compression`)
  had no runtime consumers and have been removed from the public
  config surface. Existing `config.json` files that set these keys
  will continue to load — the keys are silently ignored.
- Cargo features `tcp = []` and `shm = []` removed from `cobre-comm`
  (no runtime backends were declared under either feature).
  `highs = []` removed from `cobre-solver`; the HiGHS FFI is compiled
  unconditionally. Downstream `Cargo.toml` files that specified
  `cobre-comm = { features = ["tcp"] }` or
  `cobre-solver = { features = ["highs"] }` must drop those entries.
- `cobre-sddp` crate root re-export surface reduced from ~85 to ~50
  symbols. Implementation-detail types (`BackwardOutcome`,
  `CutSyncBuffers`, `StageIndexer`, `FphaColumnLayout`, `EvapConfig`,
  `LbEvalScratch*`, `PatchBuffer`, `WorkspacePool`, `BasisStore*`,
  `CapturedBasis`, `RiskMeasure*`, `EvaporationModel*`, `FphaPlane`,
  `LinearizedEvaporation`, `MonitorState`, `TrainingOutcome`,
  `TrainingContext`, `StageContext`, and similar) are no longer
  accessible at `cobre_sddp::Type`. They remain reachable via their
  full module path (e.g., `cobre_sddp::cut::pool::CutPool`,
  `cobre_sddp::workspace::SolverWorkspace`).

### Verified

- D01-D30 + convertido SHA256 map matches v0.4.5 reference except for
  the documented schema renames. All 90 stable parquet entries are
  byte-identical versus the previous baseline.
- D01-D15 parity hash holds across all internal refactors landed in
  this release (cut coefficients, primal/dual trajectories,
  convergence trajectory bit-for-bit identical).

## [0.4.4] - 2026-04-14

### Added

- **Per-stage thermal cost override** — `thermal_bounds.parquet` now supports
  an optional `cost_per_mwh` column alongside `block_id` to override thermal
  costs per `(plant, stage)`. D27 deterministic regression test verifying
  per-stage dispatch ordering.
- **Per-stage warm-start cut counts** — `FutureCostFunction::new` accepts
  per-stage `warm_start_counts: &[u32]` instead of a uniform scalar.
  `WARM_START_ITERATION` sentinel distinguishes warm-start from training cuts.
  Terminal-stage theta conditionally activated for boundary cuts.
- **Angular-accelerated dominance pruning** (Stage 2 of the cut management
  pipeline) — two-phase algorithm: cluster cuts by cosine similarity of their
  coefficient vectors, then perform pointwise dominance verification within
  each cluster. Preserves Assumption (H2) from Guigues 2017 and finite
  convergence. Config: `angular_pruning.enabled`, `cosine_threshold`,
  `check_frequency`.
- **Active cut budget enforcement** (Stage 3 of the cut management pipeline)
  — hard cap on LP size via `max_active_per_stage`. Evicts stalest cuts
  when the budget is exceeded, protecting current-iteration cuts. Runs every
  iteration (not gated by `check_frequency`).
- **Basis-aware warm-start padding** — `pad_basis_for_cuts` evaluates each
  active cut at the warm-start state and assigns informed basis status
  (`NONBASIC_LOWER` for tight, `BASIC` for slack). Gated by `basis_padding`
  config flag (default: `false`).
- **Performance Accelerators documentation** — new mdBook chapter documenting
  LP setup optimizations, solver safeguards, LP scaling, cut management
  pipeline, basis warm-start, parallel execution, and memory efficiency.
- **Comprehensive documentation update** — updated output-format reference
  (timing schema, solver stats, cut selection, metadata), configuration guide,
  crate developer docs (sddp, solver, overview), and JSON schemas.

### Changed

- **Cut selection Parquet schema** (breaking) — `cut_selection/iterations.parquet`
  expanded from 7 to 10 columns: added `active_after_angular`,
  `budget_evicted`, `active_after_budget`.
- **JSON schemas** regenerated for config (angular pruning, cut budget, basis
  padding, memory_window, domination_epsilon) and thermals (schemars update).
- Root-level `.schema.json` files moved to `book/src/schemas/` (canonical
  location); `.gitignore` updated to prevent re-accumulation.

## [0.4.3] - 2026-04-13

### Added

- **HistoricalResiduals noise method** — new `NoiseMethod::HistoricalResiduals`
  variant that copies pre-computed eta vectors from `HistoricalScenarioLibrary`
  into the backward-pass opening tree, preserving empirical cross-entity
  correlation from actual inflow observations. Uses hash-based deterministic
  window selection and skips Cholesky decorrelation.
- **Season ID consistency validation** (rules 27–30) — four new semantic
  validation sub-rules: season range coverage, observation coverage per season,
  resolution consistency across seasons, and contiguity of defined seasons.
- **Observation alignment validation** (rule 31) — validates observation-to-season
  alignment with a three-tier season mapping fallback chain. Replaces hardcoded
  `month0()` in historical window discovery and standardization with
  `SeasonMap`-aware logic.
- **SLURM MPI test infrastructure** — replaced `mpi_smoke.sh` with
  `mpi_slurm.sh` for Dockerized SLURM cluster testing.

### Changed

- **Timing instrumentation columns** — renamed for accuracy:
  `forward_solve_ms` → `forward_wall_ms`, `backward_solve_ms` →
  `backward_wall_ms`, `mpi_broadcast_ms` → `cut_sync_ms`,
  `rayon_overhead_ms` → `bwd_rayon_overhead_ms`. Added `lower_bound_ms`
  and `fwd_rayon_overhead_ms`. Removed stub columns `forward_sample_ms`,
  `backward_cut_ms`, `io_write_ms`.

### Fixed

- **LP determinism across MPI configurations** — reload LP model per
  scenario/trial-point to prevent HiGHS internal state carry-over (basis, RNG
  position) from making results depend on rank/thread count. Verified
  bit-identical lower bounds across 1r/1t, 1r/2t, 2r/1t, and 1r/4t.
- **Forward pass sampler count** — pass global `total_forward_passes` to the
  sampler instead of the per-rank local count, fixing LHS/QMC determinism
  across MPI configurations.
- **Timing overhead double-counting** — fix `overhead_ms` always being zero
  due to `cut_sync_ms` included in both `backward_ms` and the attributed sum.
- **Historical window year normalization** — `build_observation_sequence` now
  normalizes year offsets so the first study stage gets `year_offset=0`,
  fixing incorrect window matching for studies not starting in January.
- **MPI handling** — fix MPI communication edge cases.

## [0.4.2] - 2026-04-10

### Added

- **Execution topology reporting** — leverage ferrompi v0.3.0 to gather MPI
  library version, rank-to-host mapping, thread level, and SLURM job metadata
  at initialization. New `TopologyProvider` trait in cobre-comm with
  `ExecutionTopology` types. Displayed after the banner during `cobre run` and
  persisted in metadata JSON output.
- **Solver version reporting** — expose HiGHS version via `Highs_versionMajor`,
  `Highs_versionMinor`, `Highs_versionPatch` C API wrappers. Version displayed
  in the `Execution` section, `cobre version`, and metadata JSON.
- **Backward pass work-stealing** — replace static partitioning with atomic
  counter work-stealing for better load balance across MPI ranks.
- **Stage-major simulation loop** — refactor simulation pipeline from
  scenario-major to stage-major ordering, eliminating redundant LP setup calls.
- **Lazy FCF growth** — `CutPool` grows its coefficient storage on demand
  rather than pre-allocating to max capacity.
- **Parallel lower bound evaluation** — evaluate lower bound across openings
  in parallel using the rayon thread pool.
- **`SolverInterface::solver_name_version()`** — new trait method for solver
  identity reporting.

### Changed

- **Metadata JSON schema** (breaking) — `mpi` object replaced by `distribution`
  with richer fields: `backend`, `world_size`, `ranks_participated`,
  `num_nodes`, `threads_per_rank`, `mpi_library`, `mpi_standard`,
  `thread_level`, `slurm_job_id`, and `solver_version`.
- **`cobre version` output** — now shows the HiGHS version alongside the
  solver name.

### Fixed

- Fixed 34 assessment findings across the workspace (error handling, edge cases,
  documentation accuracy).
- Fixed rustdoc private-intra-doc-link warning in `visited_states.rs`.
- MPICH multi-line `MPI_Get_library_version` output is now sanitized to a
  single-line identifier for display.

## [0.4.1] - 2026-04-06

### Fixed

- **Documentation accuracy** — fixed 20 documentation issues: stale JSON schemas,
  wrong default values, dead field references, outdated examples, and missing
  content across 12 mdbook pages.
- `case-format.md` — `training.seed` renamed to `training.tree_seed`; `exports.states`
  default corrected from `true` to `false`; added `estimation` section.
- `configuration.md` — null `tree_seed` correctly documented as default seed 42
  (not OS entropy); removed dead `simulation.sampling_scheme` reference.
- `stochastic-modeling.md` — corrected seed behavior and `scenario_source.seed`
  optionality rules.
- `1dtoy.md` / `4ree.md` — updated config.json examples with `scenario_source`
  block; replaced non-existent `relative_gap` stopping rule with `bound_stalling`.
- `thermal-units.md` — marked GNL dispatch anticipation as not yet implemented.
- `network-topology.md` — corrected deficit penalty resolution from three-tier to
  two-tier.
- `python-quickstart.md` — documented all 11 result dict keys with `None` guard.
- `error-codes.md` — replaced stale FPHA example with `linearized_head`.

### Changed

- Updated VHS terminal recordings and asciinema casts to reflect v0.4.0 CLI output.
- Embedded `validation.gif`, `validation-error.gif`, and `multithreading.gif` in
  the Running Studies guide.

## [0.4.0] - 2026-04-06

### Added

- **Per-class scenario sampling** -- Each entity class (inflow, load, NCS)
  can independently use `InSample`, `OutOfSample`, `Historical`, or `External`
  sampling schemes via per-class sub-objects in `training.scenario_source`
  (or `simulation.scenario_source`) in `config.json`.
- **Historical inflow sampling** -- Replays standardized noise drawn from
  historical observation windows discovered in `inflow_history.parquet`.
  The window pool is controlled by the `historical_years` field in
  `training.scenario_source` in `config.json`, which accepts a list of
  years or a `{from, to}` range.
- **External scenario sources** -- Reads pre-generated scenario realizations
  from per-class Parquet files: `external_inflow_scenarios.parquet`,
  `external_load_scenarios.parquet`, and `external_ncs_scenarios.parquet`.
  Replaces the old single `external_scenarios.parquet`.
- **`HistoricalScenarioLibrary` and `ExternalScenarioLibrary`** -- New types
  in `cobre-stochastic` for pre-computed scenario storage shared across
  forward-pass iterations.
- **`ClassSampler` enum** -- Per-class noise dispatch type in
  `cobre-stochastic` routing each entity class to its configured sampling
  scheme during the forward pass.
- **Composite `ForwardSampler` architecture** -- Holds per-class
  `ClassSampler` instances and applies inter-class correlation after
  per-class noise generation, replacing the previous monolithic sampler.
- **`historical_years` config** -- New field in `training.scenario_source`
  (or `simulation.scenario_source`) in `config.json` for specifying which
  historical years are eligible as inflow replay windows. Accepts a list
  (`[2010, 2015, 2020]`) or a range (`{from: 2010, to: 2023}`).
- **Same-type enforcement for correlation groups** -- All entities in a
  correlation group must share the same `entity_type`. Mixed-type groups
  are rejected at parse time with a descriptive error.
- **`LoadModel` and `NcsModel` types** -- New types in `cobre-core` for
  per-class external standardization, mirroring the existing `InflowModel`.
- **Window discovery algorithm** -- Historical sampling uses an automatic
  window discovery pass over `inflow_history.parquet` to build the eligible
  replay pool before the first forward iteration.
- **Noise method dispatch** -- `cobre-stochastic` supports pluggable noise
  generation methods (`InSample`, `LatinHypercube`, `QmcSobol`, `QmcHalton`)
  via the `noise_method` field in `config.json`. Latin Hypercube Sampling,
  Sobol, and Halton quasi-Monte Carlo sequences are new options for
  low-discrepancy scenario generation.
- **Per-season correlation estimation** -- Correlation matrices are now
  estimated independently for each season when sufficient paired observations
  exist, with fallback to the pooled matrix for seasons below the
  `MIN_CORRELATION_PAIRS` threshold.
- **Stochastic provenance summary** -- New `stochastic_provenance.json` output
  file records PAR model fitting diagnostics, correlation estimation metadata,
  and sampler configuration for reproducibility auditing.

### Changed

- **Replace Cholesky-based spatial correlation with spectral decomposition** --
  Correlation matrices are now factored via eigendecomposition and a symmetric
  matrix square root `D = V * diag(sqrt(lambda)) * V^T` (cyclic Jacobi
  algorithm with negative-eigenvalue clipping). This eliminates
  positive-definiteness requirements on estimated correlation matrices and
  handles rank-deficient matrices naturally. The `method` field in
  `correlation.json` now defaults to `"spectral"`; `"cholesky"` is accepted
  for backward compatibility.
- **Degenerate hydro filtering removed** -- The `classify_degenerate_hydros`
  function and associated constants (`MAX_NEGATIVE_FRACTION`,
  `MIN_RESIDUAL_STD`) have been removed. With spectral decomposition,
  degenerate hydros are included in correlation estimation; their near-zero
  correlations produce near-zero eigenvalues naturally.
- **Output metadata overhaul** -- Training and simulation output directories
  now write structured `metadata.json` files with timing, iteration counts,
  and completion status. The retry histogram is normalized into a separate
  Parquet file.
- **Solver retry budget table** -- The 12-level retry escalation ladder is
  decoupled from a magic constant and uses a configurable budget table.
- **`training.seed` renamed to `training.tree_seed`** -- The `config.json`
  field controlling the scenario-tree random seed is now `tree_seed`. No
  backward-compatible alias is provided; old configs must be updated.
- **`scenario_source` moved from `stages.json` to `config.json`** -- The
  `scenario_source` configuration now lives under `training.scenario_source`
  and `simulation.scenario_source` in `config.json`, enabling independent
  per-phase sampling scheme selection. Training and simulation can use
  different sampling schemes (e.g., InSample for training, OutOfSample for
  simulation). When `simulation.scenario_source` is absent, it falls back to
  `training.scenario_source`. The `stages.json` parser rejects the old
  location with a migration error.
- **`scenario_source` per-class format** -- The `scenario_source` object
  uses per-class sub-objects with `inflow`, `load`, and `ncs` keys, each
  carrying its own `scheme` field. The `historical_years` field is at the
  top-level `scenario_source` object, not per-class.
- **`InflowHistoryRow` and `ExternalScenarioRow` relocated** -- These row
  types moved from `cobre-io` to `cobre-core::scenario` so that
  `cobre-stochastic` can reference them without depending on `cobre-io`.
- **Per-class external files** -- `external_scenarios.parquet` is replaced
  by three separate files (`external_inflow_scenarios.parquet`,
  `external_load_scenarios.parquet`, `external_ncs_scenarios.parquet`),
  one per entity class.
- **Correlation `entity_type` expanded** -- Correlation groups previously
  accepted only `"inflow"` as the entity type. The field now accepts
  `"inflow"`, `"load"`, and `"ncs"`.

### Removed

- **`ExternalSelectionMode` type** -- Sequential and random selection modes
  for external scenarios have been removed. External scenarios are now
  selected by scenario index, consistent with the other sampling schemes.
- **`selection_mode` field** -- Removed from `scenario_source`; no
  replacement.
- **`seed` alias in `config.json`** -- The deprecated `seed` alias for
  `tree_seed` has been removed. Use `training.tree_seed` directly.
- **Flat `sampling_scheme` field** -- The old top-level `sampling_scheme`
  string in `scenario_source` is gone. Configs using the flat format
  receive a descriptive parse-time error directing them to the per-class
  format.
- **`simulation.sampling_scheme` in `config.json`** -- The dead
  `sampling_scheme` field in the simulation config section has been removed.
  Simulation sampling is now controlled by `simulation.scenario_source`.
- **`classify_degenerate_hydros` function** -- Removed from `fitting.rs`
  along with `MAX_NEGATIVE_FRACTION` and `MIN_RESIDUAL_STD` constants.
  Spectral decomposition handles degenerate series naturally.

### Fixed

- **MPI reproducibility** -- Fixed four bugs affecting multi-rank training
  reproducibility: NCS/load factors and `forward_seed` are now broadcast to
  non-root MPI ranks; training stats and simulation costs are aggregated
  across ranks.
- **Cut lag coefficient remapping** -- Fixed incorrect LP column mapping for
  cut lag coefficients in backward pass.
- **Periodic Yule-Walker solver** -- Fixed forward periodic YW equation
  assembly that could produce incorrect PAR coefficients for multi-season
  models.

## [0.3.2] - 2026-03-30

### Added

- **Dominated cut selection** -- New `"domination"` method for
  `training.cut_selection`. Deactivates cuts that are dominated at every
  visited forward-pass trial point. Most aggressive strategy; configurable
  via `threshold` (consecutive domination checks before deactivation) and
  `check_frequency`. Includes current-iteration protection and stage-0
  exemption.
- **Visited states archive** -- Forward-pass trial points are now always
  collected in memory during training, regardless of the cut selection
  method. Useful for post-hoc analysis and required by dominated cut
  selection at pruning time.
- **`exports.states` config flag** -- Controls whether visited states are
  persisted to the policy checkpoint (`policy/states/stage_NNN.bin`).
  Defaults to `false` because the archive scales as
  `iterations × forward_passes × stages × state_dimension × 8 bytes`.
  Set to `true` to opt in.
- **`total_visited_states` in policy metadata** -- The `metadata.json`
  checkpoint file now includes a `total_visited_states` field.
- **`compute_effective_eta` helper** -- Extracted reusable helper for
  inflow noise path computation, reducing duplication across forward,
  backward, and lower-bound evaluation passes.

### Changed

- **Policy checkpoint exports all cuts** -- Both active and inactive cuts
  are now serialized to the policy checkpoint. The `is_active` flag on
  each cut record and the `active_cut_indices` vector preserve which cuts
  are currently in the LP. Previously, only active cuts were exported.

### Fixed

- **Inflow truncation in lower bound evaluation** -- The lower bound
  evaluation pass (`evaluate_lower_bound`) now applies the same inflow
  truncation logic as the forward and backward passes. Previously,
  negative truncated inflows were not applied when evaluating stage-0
  openings, producing optimistic lower bounds when
  `inflow_non_negativity.method` was `"truncation"` or
  `"truncation_with_penalty"`.

## [0.3.1] - 2026-03-30

### Added

- **Discount rate support** -- Annual discount rate from the policy graph is
  now wired into the SDDP solver. Per-stage one-step discount factors scale
  the theta (future cost) objective coefficient, and cumulative discount
  factors weight stagewise costs in both the training upper bound and
  simulation cost accumulation.
- **Deterministic regression test D25** -- Verifies discounted lower bound
  and simulation discount factors against undiscounted baseline (D02).

### Fixed

- **Discounted upper bound** -- The statistical upper bound now applies
  cumulative discount factors to stagewise immediate costs, making it
  comparable to the discounted lower bound. Previously, undiscounted stage
  costs were summed, producing a UB incommensurate with the LB when a
  non-zero discount rate was active.
- **Immediate cost extraction under discount** -- Stage cost extraction
  (`objective - theta`) now accounts for the discount factor on the theta
  coefficient (`objective - d_t * theta`), correctly isolating the
  undiscounted immediate cost at each stage.

## [0.3.0] - 2026-03-30

### Added

- **Policy warm-start and resume-from-checkpoint** -- Training can now load a
  prior policy checkpoint and either warm-start (inject cuts into a fresh FCF)
  or resume (continue from the saved iteration). Configured via
  `policy.mode`: `"fresh"`, `"warm_start"`, or `"resume"`.
- **Simulation-only mode** -- Run simulation against a saved policy without
  re-training. Enabled when `training.enabled = false` with a valid policy.
- **Truncation-with-penalty inflow method** -- Combined truncation and penalty
  enforcement for inflow non-negativity, matching SPTcpp's
  `truncamento_penalizacao` mode. Configured via
  `modeling.inflow_non_negativity.method = "truncation_with_penalty"`.
- **Per-plant inflow penalty via cascade** -- Inflow non-negativity penalty
  cost can now be overridden per hydro plant through the penalty cascade
  (`penalties.json` hydro overrides → `inflow_nonnegativity_cost`).
- **Bidirectional withdrawal and evaporation slacks** -- Withdrawal and
  evaporation violation slacks are now split into directional (pos/neg)
  components with independent costs, enabling asymmetric penalisation.
- **Per-block operational violations** -- Min/max outflow, turbined flow, and
  generation constraints now have per-block slack columns with independent
  penalty costs.
- **Cost decomposition** -- Simulation output now includes 6 granular violation
  cost columns (`outflow_violation_below_cost`, `outflow_violation_above_cost`,
  `turbined_violation_cost`, `generation_violation_cost`,
  `evaporation_violation_cost`, `withdrawal_violation_cost`) alongside the
  aggregate `hydro_violation_cost`.
- **Per-stage productivity override** -- `hydro_production_models.json` can now
  override the generation model for specific hydros at specific stages,
  validated via the D24 regression test.
- **Deterministic regression tests D19--D24** covering multi-hydro PAR(p),
  operational violations, min-outflow regression, per-block violations,
  bidirectional withdrawal, and productivity override.

### Fixed

- **LP bus balance productivity** -- Constant-productivity hydros now use the
  resolved per-stage production model (accounting for `hydro_production_models`
  overrides) instead of the static entity model. Fixes incorrect load-balance
  coefficients when per-stage overrides are active.
- **Withdrawal cost extraction** -- `compute_cost_result` now sums both
  `withdrawal_slack_neg` and `withdrawal_slack_pos`, fixing understated
  withdrawal violation costs in simulation output.
- **Pre-study stage handling in estimation** -- PAR(p) estimation pipeline
  correctly handles pre-study stages with season fallback for lag statistics.

### Changed

- **`policy.mode` is now a validated enum** -- Invalid values (typos like
  `"warmstart"`) are rejected at config parse time with a clear error message
  listing valid options, instead of silently defaulting to fresh training.

## [0.2.2] - 2026-03-27

### Fixed

- **Solver safeguards against stuck LP solves** -- Added iteration limits
  (simplex: `max(100K, 50 × num_cols)`, IPM: 10K) and wall-clock budgets
  (15s/30s per-level, 120s overall) to the retry escalation sequence.
  `ITERATION_LIMIT` and `TIME_LIMIT` from the initial solve are now retryable.
  Production runs with large, numerically difficult LPs could previously hang
  indefinitely. HiGHS `time_limit` option is not used because HiGHS tracks
  time cumulatively from instance creation, not per-`run()` call.

### Changed

- **Architecture degradation cleanup** -- Extracted grouping structs for 7
  functions, reducing `#[allow(clippy::too_many_arguments)]` suppressions from
  17 to 9. Absorbed 3 parameters into `TrainingConfig`, reducing `train()`
  from 15 to 12 parameters. Split 3 oversized functions (`execute`, `solve`,
  `estimate_correlation_with_season_map`) into focused sub-functions.
- **ferrompi dependency** bumped to 0.2.1 (removes RPATH from MPI binaries
  for HPC cluster compatibility).

## [0.2.1] - 2026-03-26

### Fixed

- **Cut selection phantom deactivation** -- `select_for_stage` now receives
  the pool's `active` slice and skips already-inactive slots. Previously,
  unpopulated slots below the high-water mark matched the deactivation filter,
  inflating `cuts_deactivated` counts in the convergence output. The actual cut
  pool was unaffected (the `deactivate` guard prevented double-decrements), but
  the convergence record's `cuts_active` field systematically underestimated the
  true count.

### Changed

- **Book documentation audit** -- Version references updated to v0.2.0. Four
  undocumented `config.json` sections added to the guide (`modeling`,
  `cut_selection`, `estimation`, `checkpointing`). `simulation` stopping rule
  documented. CVaR `risk_measure` field documented in stages.json coverage.
  Incorrect "only supported model" claim corrected. Expanded case directory
  layout documented for production cases. Test and subcommand counts updated.
- **cobre-bridge documentation** -- Replaced the "NEWAVE Migration" page with
  a comprehensive reference for the `cobre-bridge` conversion package,
  documenting CLI, Python API, entity mapping, output structure, and bounds
  comparison. Removed explicit references to external software names from the
  introduction and guide pages.

## [0.2.0] - 2026-03-26

### Added

- **Cut selection observability** -- New `CutSelectionRecord` and
  `StageSelectionRecord` data model in `cobre-core`. Per-stage cut selection
  statistics (cuts populated, active before/after, deactivated) are written to
  `training/cut_selection/iterations.parquet` by both CLI and Python bindings.
  Dictionary and schema definitions added to `cobre-io`.
- **Configurable simplex strategy** -- `simplex_strategy: Option<u32>` in
  `TrainingSolverConfig` allows benchmarking HiGHS strategies (0=auto, 1=dual,
  4=primal) without code changes. Threaded through `BroadcastConfig`.
- **Backward pass instrumentation** -- New timing columns in solver statistics:
  `cut_sync_ms`, `state_exchange_ms`, `cut_batch_build_ms`, `rayon_overhead_ms`,
  and `solve_with_basis` overhead tracking. All propagated to Parquet output.
- **Cut selection integration tests** -- D17 (Level1) and D18 (Lml1) regression
  tests validating convergence with bounded pool growth, re-deactivation safety,
  and `memory_window` boundary behavior.
- **Documentation overhaul** -- Slimmed README (218→74 lines), rewrote book
  introduction with audience paths, added NEWAVE migration guide, "What Cobre
  Solves" page, and Python quickstart. Brand CSS with copper headings and
  flow-blue links.
- **Quality tooling** -- Pre-commit hook (`scripts/pre-commit`), Python parity
  checker (`scripts/check_python_parity.py`), CLAUDE.md version currency checker
  (`scripts/check_claudemd_version.py`). Release checklist added to
  `CONTRIBUTING.md`.

### Changed

- **Backward pass performance** -- Sparse cut injection precomputes nonzero
  state index masks from per-hydro AR orders, skipping structurally zero
  coefficients (~29.5% NNZ reduction). Openings 1+ use `solver.solve()` (HiGHS
  internal hot-start) instead of `solve_with_basis`, eliminating ~95% of basis
  installation overhead. `HashMap<usize, u64>` binding slot increments replaced
  with `Vec<u64>` indexed by pool slot. Backward coefficient buffers
  pre-allocated and overwritten in-place via `copy_from_slice`.

### Fixed

- **Multi-rank cut sync** -- Per-stage cut sync (`allgatherv`) moved from
  post-sweep loop into the backward per-stage loop, fixing a correctness
  violation (DEC-009) for multi-rank MPI runs.
- **Cut selection** -- Fixed cut selection event propagation and Parquet output
  wiring.
- **Python parity** -- Added 3 missing output writes to `cobre-python`
  (scaling report, training solver stats, simulation solver stats).
- **Clippy compliance** -- Removed 3 dead code items (unused import, dead field,
  dead method), reduced `too_many_arguments` suppressions from 15 to 12,
  fixed `cast_possible_wrap` and doc backtick warnings.

## [0.1.11] - 2026-03-23

### Added

- **LP setup timing instrumentation** -- `SolverStatistics` now tracks cumulative
  wall-clock time for `load_model`, `add_rows`, and `set_row_bounds`/`set_col_bounds`
  separately from solve time. Three new columns (`load_model_time_ms`,
  `add_rows_time_ms`, `set_bounds_time_ms`) in `training/solver/iterations.parquet`
  enable diagnosing LP rebuild overhead vs simplex time.
- **LP setup optimisation** -- Model persistence across scenarios at the same
  stage (`S1`), active-cut-count caching, incremental cut append, sparse cut
  representation, and bound-zeroing deactivation. Reduces LP rebuild overhead
  by avoiding redundant `load_model` + `add_rows` calls.
- **Simulation basis warm-start** -- Simulation LPs are warm-started with the
  per-stage basis from the training checkpoint. The basis is read-only and shared
  across all threads, preserving determinism while reducing simplex iterations.
- **Physical evaporation upper bounds** -- Evaporation flow (`Q_ev`) is now
  bounded above by a physical estimate derived from linearisation coefficients
  and maximum storage, with a 2x safety margin. Over-evaporation slack
  (`f_evap_minus`) is penalised at 100x the under-evaporation cost to prevent
  the solver from inflating evaporation as a dump valve.
- **Cost breakdown extraction** -- `inflow_penalty_cost`, `hydro_violation_cost`
  (evaporation + withdrawal violations), and diversion cost are now extracted
  from LP primal values into the simulation cost breakdown. Previously these
  LP objective contributions were included in `immediate_cost` but not reported
  in any named component field.

### Fixed

- **Stale z_inflow column offset formulas** -- Corrected 8 test column offset
  formulas and 10+ comments in `lp_builder.rs` and `noise.rs` that still
  referenced the old z_inflow-at-end-of-columns layout after the N\*(1+L)
  refactoring. Tests passed coincidentally because adjacent columns had
  identical bounds and objective values.
- **Simulation cost extraction with LP prescaling** -- Per-variable cost
  extraction (spillage, thermal, exchange, NCS curtailment, and the aggregate
  cost breakdown) now divides by `col_scale[j]` to undo column prescaling.
  Without this, per-entity costs were inflated by the column scale factor when
  LP prescaling was active.
- **NCS curtailment cost semantics** -- Changed curtailment cost to use
  `curtailment_mw` (available minus generation) instead of `generation_mw`.
  The field now reports the actual penalty for not generating, matching the
  `curtailment_cost` field name semantics.
- **HiGHS internal scaling disabled** -- Set `simplex_scale_strategy = 0` (off)
  in default solver options. Cobre's own prescaler handles conditioning; the
  HiGHS internal scaler interfered with basis reuse and dual extraction.

## [0.1.10] - 2026-03-23

### Added

- **Z-inflow LP variable** -- New `z_inflow` LP variable tracks realized inflow
  per hydro at fixed column offset `N*(1+L)`, enabling accurate inflow reporting
  in simulation output independent of lag dynamics. Includes stage-invariant
  row placement and deterministic test D16 for PAR(1) lag shift verification.
- **`water_balance` field on `StageIndexer`** -- Explicit row range for water
  balance constraints, replacing the fragile `n_state + h` offset that broke
  after the z-inflow row insertion.

### Changed

- **LP column/row layout refactor** -- Z-inflow columns and rows relocated to
  fixed offset `N*(1+L)`, shifting `storage_in` to `N*(2+L)` and `theta` to
  `N*(3+L)`. All 56 affected test assertions updated for the new layout.

### Fixed

- **Water value extraction** -- Fixed simulation output reading z-inflow row
  duals instead of water balance duals after the layout refactor. Water values
  were reporting null/wrong for all hydros with the new column layout.
- **PAR seasonal model expansion** -- Fixed auto-estimation emitting one model
  per season instead of one per stage. Stages beyond the first in each season
  were missing inflow AR coefficients.
- **Lag state transition** -- Wired lag state shift in forward/backward pass
  so PAR(p) lag variables propagate correctly across stages.
- **Inflow reporting** -- Fixed inflow values in simulation output to reflect
  realized inflow rather than stale lag-derived values.

### Performance

- **`load_model` hoisting** -- Moved `load_model` call out of the backward pass
  inner opening loop, avoiding redundant LP reloads per stage.

## [0.1.9] - 2026-03-22

### Added

- **PAR estimation overhaul** -- Replaced AIC-based order selection with
  periodic Yule-Walker coefficient estimation and PACF-based order selection.
  Added contribution-based validation, negative phi_1 rejection gate, and
  iterative PACF order reduction for improved numerical stability on
  real-world inflow series.
- **LP scaling** -- Row scaling with RHS prescaling and dual unscaling, plus
  internal objective cost scaling (`COST_SCALE_FACTOR = 1000`), improving
  solver conditioning on large systems with heterogeneous constraint
  magnitudes.
- **Solver statistics** -- Three-channel instrumentation architecture:
  LP scaling diagnostics report (`solver_stats/scaling_report.json`),
  per-phase solver statistics Parquet output (`solver_stats/solver_stats.parquet`),
  and enhanced CLI display with per-solve timing, basis reuse tracking, and
  simplex iteration counts for both training and simulation.
- **Per-scenario simulation statistics** -- Individual scenario cost and LP
  solve metrics in simulation CLI summary output alongside aggregate results.

### Changed

- **Simulation pipeline performance** -- Eliminated two per-LP-solve `Vec<f64>`
  clones in `solve_simulation_stage` by using `std::mem::take` to temporarily
  move unscaled buffers out of `ScratchBuffers`, resolving a borrow conflict
  without allocation.

### Fixed

- **Clippy compliance** -- Resolved all clippy warnings across the workspace
  for CI compliance with `-D warnings`.

## [0.1.8] - 2026-03-21

### Added

- **`LineExchange` generic constraint term** -- New `VariableRef::LineExchange`
  variant (the 20th) enables generic constraints to reference net line flow
  (direct - reverse) as a single expression term via `line_exchange(id)`. The
  resolver returns two LP column entries: `(fwd_col, +1.0)` and
  `(rev_col, -1.0)`. Includes referential validation for line ID existence.
- **Per-stage productivity override** -- `productivity_override: Option<f64>`
  field on `StageRange` and `SeasonConfig` in `hydro_production_models.json`.
  When present, replaces the entity's base `productivity_mw_per_m3s` for the
  covered stages. Validated to be positive and rejected on FPHA stages. Enables
  exact reproduction of NEWAVE cases with temporal head/elevation overrides.

## [0.1.7] - 2026-03-21

### Added

- **Block factors** -- Per-block scaling multipliers for load demand
  (`scenarios/load_factors.json`), transmission line capacity
  (`constraints/exchange_factors.json`), and non-controllable source
  availability (`scenarios/non_controllable_factors.json`). Factors default
  to 1.0 when absent. Includes validation rules 36--41 and deterministic
  test D14.
- **NCS stochastic availability** -- Non-controllable sources (wind, solar,
  run-of-river) now support stochastic availability modeling via
  `scenarios/non_controllable_stats.parquet`. Each source has a per-stage
  mean and standard deviation availability factor (0--1), drawn from a
  normal distribution and clamped to [0, 1]. Availability is multiplied by
  `max_generation_mw` and per-block factors. The SDDP policy learns to
  hedge against NCS variability. Includes NCS noise dimension in the opening
  tree, per-scenario LP column bound patching in forward, backward, and
  lower bound evaluation passes, and deterministic test D15.
- **NCS JSON schema** -- `non_controllable_factors.schema.json` added to the
  schema reference. The schema generator now produces 17 schemas (up from 16).
- **Deterministic tests D14--D15** -- Two new regression test cases: D14
  (block factor load scaling) and D15 (non-controllable source with
  stochastic pipeline, mean factor 0.5, std 0).

### Changed

- **NCS entity promoted to Full** -- Non-controllable sources are no longer
  stub entities. They now contribute LP generation variables, stochastic
  availability, simulation output, and full validation rules.

## [0.1.6] - 2026-03-19

### Added

- **Generic constraints** -- User-defined linear constraints over LP variables,
  specified via `constraints/generic_constraints.json` with stage-varying bounds
  from `constraints/generic_constraint_bounds.parquet`. Supports all 19 variable (now 20 with `line_exchange`)
  types (thermal generation, hydro storage, hydro outflow, line flows, etc.),
  optional slack variables with per-constraint penalties, and three constraint
  senses (`<=`, `>=`, `==`). Includes dual and slack extraction during training
  and simulation, violation cost accounting, and Hive-partitioned Parquet output
  for generic constraint violations.
- **Water withdrawal** -- Hydro plants can now model water withdrawal schedules
  (e.g., irrigation, municipal supply) with configurable bounds and violation
  penalties. Withdrawal constraints are integrated into the LP water balance,
  with slack variables and violation cost tracking in simulation output.
- **Generic constraint validation rules** -- Three new referential validation
  rules (33--35) in cobre-io: entity ID existence in constraint expressions,
  block ID validity for referenced stages, and duplicate bounds key detection.
- **Deterministic test D13** -- New regression test case exercising generic
  constraints with a thermal plant capped by a user-defined constraint,
  verifying the hand-computed expected cost.
- **JSON schemas for generic constraints, exchange factors, and load factors** --
  Added `generic_constraints.schema.json`, `exchange_factors.schema.json`, and
  `load_factors.schema.json` to the schema reference. The schema generator now
  produces 13 schemas (up from 10).

### Changed

- **Schema reference expanded** -- The JSON Schemas reference page now lists all
  16 available schemas, including `production_models`, `initial_conditions`,
  `correlation`, and the three newly generated schemas.

## [0.1.5] - 2026-03-18

### Added

- **Multi-segment deficit pricing** -- The LP builder now supports N deficit
  columns per bus per block (one per segment), with capacity constraints.
  NEWAVE-converted cases with tiered deficit costs produce correct results.
  Deterministic test D09 is un-ignored and passes.
- **Arrow zero-copy result loading** -- `load_convergence_arrow()` and
  `load_simulation_arrow()` in cobre-python return Arrow RecordBatches via
  the Arrow C Data Interface, enabling zero-copy `polars.from_arrow()` in
  Python without intermediate serialization.
- **Jupyter quickstart notebook** -- `examples/notebooks/quickstart.ipynb`
  demonstrates the end-to-end Python workflow: run a study, load results
  with Arrow zero-copy, and visualize convergence with matplotlib.
- **Past inflows for PAR lag initialization** -- New `past_inflows` field in
  `initial_conditions.json` allows users to specify historical inflow values
  for PAR(p) lag initialization at stage 0. Values are provided in recency
  order (most recent first) per hydro, replacing zero-initialization with
  actual historical data for accurate first-stage noise realization.
- **Past inflows validation rules** -- Three new semantic validation rules
  (22--24) in cobre-io check that `past_inflows` provides sufficient coverage
  when `inflow_lags: true` and PAR order > 0: entries must be non-empty
  (rule 22), per-hydro value count must meet the PAR order (rule 23), and
  all hydro IDs in past_inflows must exist in the registry (rule 24).

### Changed

- **Infrastructure crate doc cleanup** -- Replaced "Benders" terminology with
  generic "cutting plane" language in cobre-io and cobre-solver doc comments.
- **Backward pass sort verified redundant** -- The sort in the backward pass
  was confirmed redundant and replaced with a `debug_assert!` for safety.

### Fixed

- **D09 deterministic test** -- Previously `#[ignore]` due to missing
  multi-segment deficit support; now passes with correct expected cost.

## [0.1.4] - 2026-03-17

### Added

- **FPHA hydro production model** -- Four-piece hyperplane approximation for
  variable-head hydroelectric plants. Supports two modes: precomputed hyperplanes
  (supplied via `fpha_hyperplanes.parquet`) and computed-from-geometry (fitted from
  forebay volume-elevation and tailrace flow-elevation curves in `hydro_geometry.parquet`).
  Includes the full FPHA fitting pipeline (`fpha_fitting.rs`) with least-squares
  hyperplane generation and production function evaluation.
- **Evaporation linearization** -- Reservoir surface evaporation modeled as a
  linearized function of stored volume. Per-season evaporation reference volumes
  allow seasonal variation in the linearization point. Evaporation variables and
  constraints are integrated into the LP water balance.
- **Hydro model preprocessing pipeline** -- `hydro_models.rs` provides a unified
  preprocessing pipeline that resolves production model configuration (constant
  productivity, precomputed FPHA, or computed FPHA) and evaporation parameters
  into solver-ready structures consumed by the LP builder.
- **Hydro model output writer** -- `output/hydro_models.rs` in cobre-io serializes
  the resolved hydro model parameters (FPHA hyperplanes, evaporation coefficients)
  to Parquet for auditability and debugging.
- **FPHA/evaporation result extraction** -- Simulation extraction pipeline now
  reports FPHA-related variables (production per segment, active hyperplane) and
  evaporation volumes alongside existing hydro results.
- **Deterministic regression test suite** -- 12 hand-computed test cases (D01--D12)
  covering thermal dispatch, single hydro, cascade, transmission, FPHA (constant
  head, variable head, computed), evaporation, multi-deficit, and inflow
  non-negativity. Each case has an analytically derived expected cost that serves
  as a regression anchor.
- **New example case `4ree-fpha-evap`** -- 4-region system demonstrating FPHA with
  evaporation, including hydro geometry data and production model configuration.
- **JSON schemas** -- Added `correlation.schema.json`, `initial_conditions.schema.json`,
  and `production_models.schema.json` to the software book.

### Changed

- **LP builder expanded** -- FPHA constraints (hyperplane cuts, segment bounds) and
  evaporation constraints (linearized volume-area, water balance contribution) are
  now generated by the LP builder for hydros configured with non-constant production
  models or evaporation parameters.
- **`StudySetup` indexer** -- Extended to populate FPHA and evaporation index maps,
  enabling O(1) lookup of hydro model parameters during LP construction.
- **Backward pass** -- Updated to extract FPHA/evaporation duals for cut coefficient
  computation.
- **MSRV bumped to 1.86** -- Workspace `rust-version` updated from 1.85 to 1.86.

## [0.1.3] - 2026-03-15

### Added

- **`StudySetup` struct** -- centralized study orchestration in cobre-sddp,
  encapsulating all precomputed study state (templates, indexer, FCF,
  stochastic context, risk measures, entity counts, block layout). Extracted
  from the CLI pipeline so that multiple entry points (CLI, Python, future TUI
  and MCP) share a single construction and orchestration path.
- **`StageContext` and `TrainingContext` structs** -- lightweight context
  bundles in a dedicated `context.rs` module that reduce hot-path function
  argument counts. `StageContext` groups stage templates and layout slices;
  `TrainingContext` groups the solver-level study parameters.
- **`ScratchBuffers` struct** -- separates per-worker noise and patch scratch
  space from the `SolverWorkspace`, improving memory layout and making the
  allocation boundary between solver state and algorithm scratch explicit.
- **`noise.rs` module** -- consolidates all noise-to-RHS-patch logic into a
  single module with shared `transform_inflow_noise` and `transform_load_noise`
  functions called from the forward pass, backward pass, and simulation
  pipeline, eliminating the three-way duplication identified in the v0.1.1
  post-release assessment.
- **`WelfordAccumulator` in cobre-core** -- streaming online statistics
  accumulator (mean, variance, standard deviation) for use in progress
  reporting and any algorithm that requires running statistics without storing
  all observations.
- **Stochastic summary** -- `summary.rs` in cobre-cli produces a structured
  post-setup report of the fitted PAR models with a three-tier AR order
  display: compact per-order form (≤10 hydros), range summary (11–30 hydros),
  and histogram (31+ hydros). Replaces the `[stochastic]` `eprintln!` pattern.
- **User-supplied opening tree** -- when `scenarios/noise_openings.parquet` is
  present in the case directory, Cobre loads, validates, and uses it as the
  backward-pass opening tree instead of generating one internally.
  The exported `output/stochastic/noise_openings.parquet` uses the same schema,
  so the round-trip from export to re-supply is a copy operation.
- **Stochastic artifact export** -- after the stochastic context is built,
  Cobre writes up to six artifact files to `output/stochastic/`: fitted
  seasonal statistics, AR coefficients, correlation matrix, fitting report,
  noise openings, and load seasonal statistics. Controlled by
  `exports.stochastic` in `config.json`.

### Changed

- **Hot-path argument counts reduced** -- `run_forward_pass`, `run_backward_pass`,
  and `simulate` each drop from 20–22 arguments to 7 by bundling dissolved
  context into `StageContext` and `TrainingContext`. The public `StudySetup`
  API exposes `train()` with 7 arguments and `simulate()` with 4.
- **Simulation progress uses a single global `WelfordAccumulator`** -- the
  progress thread now owns the sole accumulator and receives raw
  `scenario_cost: f64` events from workers. The `SimulationProgress` event is
  simplified to a single cost field, eliminating the per-worker pre-aggregation
  that caused the incorrect statistics bug.
- **CLI `execute()` uses `StudySetup`** -- training and simulation are
  orchestrated through `StudySetup::train()` and `StudySetup::simulate()`
  instead of threading individual parameters through the CLI.
- **Python `run_inner()` simplified** -- reduced from ~250 to ~125 lines by
  adopting `StudySetup::new()` for construction and `StudySetup::train()` /
  `StudySetup::simulate()` for orchestration.

### Removed

- **`--skip-simulation` CLI flag** -- simulation is now controlled exclusively
  via `config.json` (`simulation.enabled`).
- **`--no-banner` CLI flag** -- banner display is no longer user-configurable
  from the command line.
- **`--verbose` CLI flag** -- tracing subscriber setup has been removed from
  the CLI; structured logging configuration will be revisited in a future release.
- **`--export-stochastic` CLI flag** -- stochastic artifact export is controlled
  via `exports.stochastic` in `config.json`, not a command-line flag.
- **`format_stochastic_diagnostics` function** -- replaced by the structured
  `StochasticSummary` / `summary.rs` display system; the `[stochastic]`
  `eprintln!` pattern is removed from cobre-sddp.

### Fixed

- **Simulation progress bar statistics** -- the per-worker `WelfordAccumulator`
  accumulator produced incorrect mean and standard deviation because each
  worker tracked only its own subset of completed scenarios. The fix moves the
  single accumulator to the progress thread, which receives all scenario costs
  and computes globally correct statistics.
- **Clippy suppressions in cobre-sddp** -- remaining `#[allow(...)]`
  production suppressions reduced from 8+ to 2 by addressing each underlying
  lint finding.

## [0.1.2] - 2026-03-14

### Fixed

- Canonical upper bound summation for multi-rank determinism — the upper
  bound `allreduce` now uses a compensated (Kahan) summation that produces
  bit-for-bit identical results regardless of MPI rank count and scenario
  distribution across ranks.
- Removed all production clippy suppressions (`#[allow(...)]`) from cobre-sddp
  source files, addressing each underlying lint finding instead of silencing it.
- Addressed code review findings across cobre-sddp: simplified control flow,
  removed dead code, fixed off-by-one edge cases in stopping rules.

### Added

- Generic PAR type aliases (`ParOrder`, `ParCoefficients`, `ParResidualStdRatio`)
  in cobre-stochastic for improved API clarity.

### Changed

- Updated software book: new 4-Region Example page, revised roadmap sections,
  fixed overview and SDDP crate pages, updated badges and DEC references.
- Updated cobre-stochastic docstrings to use generic terminology (no
  algorithm-specific language in infrastructure crate documentation).
- Python bindings: added Python 3.14 classifier and CI testing matrix
  (3.12, 3.13, 3.14).

## [0.1.1] - 2026-03-12

### Added

- PAR model simulation -- Scenario generation using fitted PAR(p) models
  during the simulation pipeline, producing scenario-consistent inflow traces.
- Inflow truncation -- The `Truncation` non-negativity treatment method,
  which clamps negative PAR model draws to zero before applying noise.
- Stochastic load noise -- Correlated Gaussian noise added to load
  forecasts using the same Cholesky-based framework as inflow noise.
- PAR estimation from history -- Fitting PAR(p) model coefficients from
  historical inflow records provided in the case directory.
- `cobre summary` subcommand -- Post-run summary reporting subcommand that
  prints convergence statistics and output file locations.

## [0.1.0] - 2026-03-09

### Added

- Entity model and core (cobre-core): Bus, Line, Thermal, Hydro, Contract, PumpingStation, NonControllable types; system registry; topology validation; three-tier penalty resolution
- Case loader (cobre-io): five-layer validation; JSON/Parquet parsing for 33 input types; penalty/bound resolution
- LP solver abstraction (cobre-solver): HiGHS backend; warm-start support; conformance tests
- Communication layer (cobre-comm): Communicator trait with LocalBackend and FerrompiBackend; compile-time feature selection
- Stochastic preprocessing (cobre-stochastic): PAR(p) preprocessing; SipHash seed derivation; Cholesky correlation; opening trees; InSample sampling
- SDDP training loop (cobre-sddp): forward/backward pass; Benders cuts; stopping rule set; convergence monitoring
- Simulation pipeline (cobre-sddp + cobre-io): MPI aggregation; Hive-partitioned Parquet output; FlatBuffers policy checkpoint
- Command-line interface (cobre-cli): `run`, `validate`, `report`, `version` subcommands; progress bars; exit codes

## [0.0.1] - 2026-02-23

### Added

- Initial workspace scaffold with 11 crates (`cobre`, `cobre-core`, `cobre-io`, `cobre-stochastic`, `cobre-solver`, `cobre-comm`, `cobre-sddp`, `cobre-cli`, `cobre-mcp`, `cobre-python`, `cobre-tui`)
- Reserved all crate names on crates.io
- CI pipeline: check, test, clippy, fmt, docs, security audit, license check, coverage
- Workspace lint configuration with clippy pedantic and `unsafe_code = "forbid"`
- cargo-dist configuration for multi-platform binary distribution

<!-- next-url -->

[Unreleased]: https://github.com/cobre-rs/cobre/compare/v0.14.2...HEAD
[0.14.2]: https://github.com/cobre-rs/cobre/compare/v0.14.1...v0.14.2
[0.14.1]: https://github.com/cobre-rs/cobre/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/cobre-rs/cobre/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/cobre-rs/cobre/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/cobre-rs/cobre/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/cobre-rs/cobre/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/cobre-rs/cobre/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/cobre-rs/cobre/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/cobre-rs/cobre/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/cobre-rs/cobre/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/cobre-rs/cobre/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/cobre-rs/cobre/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/cobre-rs/cobre/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/cobre-rs/cobre/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/cobre-rs/cobre/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/cobre-rs/cobre/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/cobre-rs/cobre/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/cobre-rs/cobre/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/cobre-rs/cobre/compare/v0.4.4...v0.5.0
[0.4.4]: https://github.com/cobre-rs/cobre/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/cobre-rs/cobre/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/cobre-rs/cobre/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/cobre-rs/cobre/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/cobre-rs/cobre/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/cobre-rs/cobre/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/cobre-rs/cobre/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/cobre-rs/cobre/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/cobre-rs/cobre/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cobre-rs/cobre/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cobre-rs/cobre/compare/v0.1.11...v0.2.0
[0.1.11]: https://github.com/cobre-rs/cobre/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/cobre-rs/cobre/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/cobre-rs/cobre/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/cobre-rs/cobre/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/cobre-rs/cobre/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/cobre-rs/cobre/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/cobre-rs/cobre/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/cobre-rs/cobre/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/cobre-rs/cobre/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/cobre-rs/cobre/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cobre-rs/cobre/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cobre-rs/cobre/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/cobre-rs/cobre/releases/tag/v0.0.1
