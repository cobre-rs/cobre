# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-header -->

## [Unreleased]

### Breaking Changes

### Added

### Changed

### Removed

### Verified

## [0.5.0] - 2026-04-20

Major refactor. Consumers must update `config.json`, any code calling
solver traits, and any tooling reading `solver/iterations.parquet`
before upgrading.

### Breaking Changes

**Config schema** — remove from `config.json`:

- `training.solver.warm_start_basis_mode` (rejected at parse time)
- `training.solver.canonical_state` (silently ignored with stderr warning)
- `training.cut_selection.angular_pruning` (rejected at parse time)
- `training.cut_selection.basis_padding` (soft-deprecated with stderr warning)

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
- **Weekly+monthly DECOMP studies** — sub-monthly lag accumulation,
  `recent_observations` input for mid-season starts, terminal boundary
  cuts (`policy.boundary.{path, source_stage}`) for Cobre-to-Cobre FCF
  coupling, and non-uniform per-stage scenario counts.
- **Multi-resolution studies** — same-season noise group sharing,
  observation aggregation from finer to coarser resolution, and
  monthly→quarterly PAR transition.
- **Per-opening solver statistics** — backward rows carry `rank` and
  `worker_id` metadata for per-worker parity testing.

### Changed

- **Lower-bound LP is strictly append-only.** Cuts are never removed
  from the LB LP, guaranteeing monotonic non-decreasing lower bound.
  Cut selection and budget enforcement continue to deactivate cuts in
  the shared pool for the forward and backward passes.
- **Cut management pipeline reduced to two stages**: strategy-based
  selection followed by budget enforcement.

### Removed

- **Angular diversity pruning** — config section, module, output column,
  and training event variant.
- **`SparseCut` module** — handled by the indexer.
- **`CutMetadata::domination_count`** — unused.
- **LB LP periodic rebuild** and its `CutRowMap` helpers.

### Verified

- D01-D30 + convertido SHA256 map matches v0.4.5 reference except for
  the documented schema renames. All 90 stable parquet entries are
  byte-identical versus the previous baseline.

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

- Phase 1 (cobre-core): Entity model (Bus, Line, Thermal, Hydro, Contract, PumpingStation, NonControllable), system registry, topology validation, three-tier penalty resolution
- Phase 2 (cobre-io): Case loader with five-layer validation, JSON/Parquet parsing for 33 input types, penalty/bound resolution
- Phase 3 (cobre-solver): LP solver abstraction with HiGHS backend, warm-start support, conformance tests
- Phase 4 (cobre-comm): Communicator trait with LocalBackend and FerrompiBackend, compile-time feature selection
- Phase 5 (cobre-stochastic): PAR(p) preprocessing, SipHash seed derivation, Cholesky correlation, opening trees, InSample sampling
- Phase 6 (cobre-sddp): SDDP training loop with forward/backward pass, Benders cuts, stopping rule set, convergence monitoring
- Phase 7 (cobre-sddp + cobre-io): Simulation pipeline with MPI aggregation, Hive-partitioned Parquet output, FlatBuffers policy checkpoint
- Phase 8 (cobre-cli): Command-line interface with `run`, `validate`, `report`, `version` subcommands, progress bars, exit codes

## [0.0.1] - 2026-02-23

### Added

- Initial workspace scaffold with 11 crates (`cobre`, `cobre-core`, `cobre-io`, `cobre-stochastic`, `cobre-solver`, `cobre-comm`, `cobre-sddp`, `cobre-cli`, `cobre-mcp`, `cobre-python`, `cobre-tui`)
- Reserved all crate names on crates.io
- CI pipeline: check, test, clippy, fmt, docs, security audit, license check, coverage
- Workspace lint configuration with clippy pedantic and `unsafe_code = "forbid"`
- cargo-dist configuration for multi-platform binary distribution

<!-- next-url -->

[Unreleased]: https://github.com/cobre-rs/cobre/compare/v0.2.2...HEAD
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
