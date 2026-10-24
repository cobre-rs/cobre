# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-header -->

## [Unreleased]

### Added

- `training.cut_selection.selection.lookback_iterations` for `level1`: the
  trial states of a run are kept only for the rows added in the last
  `lookback_iterations` iterations of that run, which bounds the memory they
  take. Absent, the run keeps every trial state. A zero, or a value below
  `check_frequency`, is refused.
- `simulation.cut_set`, `all` or `active`, chooses which rows of each stage
  the simulation loads. `all` loads every row training added, including rows
  the selection method deactivated and rows `max_active_per_stage` evicted;
  `active` loads the rows that were active when training ended.
- `training/convergence.parquet` has a `lazy_round_cap_hits` column: the
  lazy-selection LP solves of the iteration that reached the round cap with a
  candidate row still violated.
- A resumed or warm-started `level1` run restores the stored trial states, and
  a resumed run with `max_active_per_stage` or the `dynamic` method restores
  the binding counters of each row.
- Checkpoints hold one solver basis per forward pass and node from every rank,
  and record whether the set is whole. A resumed run with several forward
  passes starts each pass from its own basis, and a warm start or resume
  restores the bases of a policy trained with a selection method, with their
  row slots. The unused-bases warning counts per scenario.
- `policy.mode = resume` is refused, with a message naming what is missing and
  the way out, when the checkpoint lacks the `level1` trial-state history the
  run needs, lacks the binding counters the run reads, holds a lookback window
  shorter than the run's, holds an incomplete basis store, or was written with
  another `training.selection.forward_passes`.
- `policy.mode = resume` is refused by `cobre validate` and `cobre run` when
  the rows of `policy.boundary` differ from the terminal pool of the
  checkpoint. A resume whose `modeling.cost_scale_factor` differs from the run
  that wrote the checkpoint is refused only when the boundary rows then
  differ. The resume applies the scale of its own configuration.
- Computed scalar parameters in MWh: `max_stored_energy_mwh`,
  `integrated_accumulated_productivity_mwh_per_hm3` and
  `integrated_equivalent_productivity_mwh_per_hm3`. Each is the raw tag of the
  same name times 10⁶/3600. With them a stored-energy constraint states its
  amounts in MWh and no stage duration enters either side; price its slack with
  `"penalty_basis": "per_unit"` so the penalty is a price per MWh on every stage.
- A generic-constraint expression accepts arithmetic over numbers and
  `@parameters` on both sides of the relation: `+`, `-`, `*`, `/`, parentheses
  and unary minus, for example
  `(@weight_a + @weight_b) * hydro_useful_volume_final(0) >= @pct * @emax`. It
  must stay linear in LP variables: a product of two variables, or a division
  by one, is rejected at load. A constant that divides by zero or is not finite
  is rejected at load, and a value that resolves from parameters to a
  non-finite number stops setup with an error naming the stage and block.
- A scalar parameter in `constraints/generic_parameters.json` may be
  `kind: "expression"`, a formula over other parameters, for example
  `{ "id": 5, "name": "scaled_demand", "kind": "expression", "expression": "@discount_rate * @demand" }`.
  A constraint reads it like any other parameter. A reference cycle or an LP
  variable in the formula is rejected at load. The value is stored per block
  when a parameter it reads varies by block, and the stored-energy pairing
  warning (rule 51) checks the tags it reads.
- `system/hydro_energy_productivity.parquet` has a new
  `integrated_equivalent_productivity_mw_per_m3s` column. A value sets that
  plant's own useful-range mean productivity for the stage outright; the
  cascade sum, `max_stored_energy*`, the `integrated_*` tags and output
  columns and `stored_energy_*` are computed from it. Use it when a source
  model's stored-energy convention differs from integrating the plant's
  geometry over its physical range. Finite and `>= 0.0`; NULL means no
  override.
- A hydro's `diversion` channel in `system/hydros.json` may leave out
  `max_flow_m3s` or set it to `null` to mean the channel has no capacity limit.
  A per-stage or per-block `max_diversion_m3s` in
  `constraints/hydro_bounds.parquet` still caps it.

### Changed

- **BREAKING:** `truncation` and `truncation_with_penalty` follow
  Larroyd et al. (2022): a negative inflow is no longer raised to zero.
  `incremental_inflow_m3s` reports the inflow model's value, which can be
  negative, the inflow lags are built from it, and
  `inflow_nonnegativity_slack_m3s` reports the water added to it. `truncation`
  adds exactly the amount that lifts the inflow to zero, at no cost;
  `truncation_with_penalty` adds at most that amount, priced at
  `inflow_nonnegativity_cost`. Results change for studies using either
  method when an inflow draw is negative. Retrain policies trained with either
  method. Known limitation (the paper's §2.5): the lower bound can exceed the
  upper bound, and a `gap` stopping rule, admitted only under enumerated
  forward selection, can then report convergence that has not occurred.
- Under `penalty`, the `inflow_nonnegativity_slack_m3s` of a PreFilling plant
  adds water at the downstream plant that receives its inflow, if there is
  one. It added none before, so results can change for `penalty` studies with
  PreFilling plants.
- **BREAKING:** `level1` keeps exactly the cuts that own a stored trial state.
  At each trial state the run stored, the cut with the highest value owns it,
  and the lowest position wins an exact tie. The run stores the trial states
  of the whole run unless `lookback_iterations` sets a window. Results change
  for every study that uses `level1`. Retrain policies trained with `level1`.
- **BREAKING:** `simulation.cut_set` defaults to `all`. A run with `level1` or
  `max_active_per_stage` now simulates with every row training added,
  including deactivated and evicted ones. Set `simulation.cut_set` to
  `active` for the earlier behaviour. Runs without a selection method or cap
  and runs under `dynamic` are unchanged. A simulation over a loaded policy
  also includes the rows the writing run deactivated or evicted, as in the
  `warm_start` entry below.
- **BREAKING:** `violation_tolerance` of the `dynamic` method is a relative
  factor, absent by default. Absent, each phase uses the primal feasibility
  tolerance of its solver profile. A value of 1 or more, or below the primal
  feasibility tolerance of any phase's solver profile, is refused at load.
  Runs can return different cuts and bounds, and a value written for the
  earlier absolute tolerance can now fail at load: delete the key or raise it.
- **BREAKING:** `row_activity_tolerance` defaults to 1e-6 instead of 0, and an
  explicit `null` is refused. A run that sets `max_active_per_stage` or uses
  `dynamic` and omits the key can evict and seed other rows. Set the key to 0
  to keep the earlier behaviour.
- **BREAKING:** a zero in `check_frequency`, `lookback_iterations`,
  `start_iteration`, `candidate_recency` or `max_added_per_round` is a parse
  error that names the key path. Set the key to 1 or more, or delete it to use
  its default.
- **BREAKING:** a `selection` object given as an array is refused. Write it as
  an object with a `method` key.
- **BREAKING:** `max_active_per_stage` is refused at load with `dynamic` and
  with `training.selection.method` set to `enumerated`. Remove the cap, or use
  `level1`; for enumerated, set `training.selection.method` to `sampled`.
- `max_active_per_stage` applies to the rows the current run added to each
  stage strictly between the first and the last, and not to loaded rows. Ties
  for eviction go to the row binding in fewer solves, then to the lower
  position in the pool.
- Loaded cuts keep the `iteration` they were generated in on export. The cuts
  loaded by `policy.boundary` and by a warm start restart their activity clock.
- **BREAKING:** `training/cut_selection/iterations.parquet` has one row per
  `(iteration, pool_id)` on every iteration of a run with a selection method or
  a cap. `cuts_in_pool` replaces `cuts_populated`, and `stage_id` is joined by
  `pool_id` and a nullable `node_id`. `cuts_deactivated`, `cuts_reactivated`
  and `selection_time_ms` are null where no selection pass ran on the pool;
  `budget_evicted` and `active_after_budget` are null where the cap was not
  applied; `cuts_in_lp` is null under `dynamic`. Update readers of the file.
- **BREAKING:** policy checkpoints are written with `format_version` 4, and
  version-3 checkpoints are refused. Cut values are stored in the scaled units
  of the run that wrote them, so a resume with the same
  `modeling.cost_scale_factor` is exact. Each file in the policy's `states/`
  directory is keyed by pool and holds each stored trial state with its slot,
  and each file in `basis/` holds one entry per scenario. Re-run the program
  that produced the policy.
- **BREAKING:** a resumed run places every cut at its original position and
  restores the prefix the original run started with. `level1` and
  `max_active_per_stage` treat the reloaded cuts as the uninterrupted run
  does; a resumed run with a cap used to count only the cuts added after the
  resume.
- **BREAKING:** under `warm_start`, a loaded cut its source scored is selected
  by `level1` like a generated cut, and can be deactivated and reactivated.
  `simulation.cut_set = all` over a loaded policy includes the cuts the
  writing run deactivated or evicted.
- **BREAKING:** `exports.states` writes each pool's stored trial states, by
  position and at the pool's state dimension, for every selection setting, the
  root pool included. A `dynamic` run or a run without selection now writes
  them. The per-stage visited-states export at full dimension is removed.
- A resumed run starts a scenario that has no basis of its own cold, as the
  original run did, instead of from another scenario's basis.
- **BREAKING:** a run that fails writes its partial policy to
  `<policy.path>.partial` and leaves the last whole checkpoint in
  `policy.path`. The failure message names the directory.
- A `dynamic` run that simulates with `simulation.cut_set = all` is refused
  when the pool it loads holds inactive rows that `all` includes, and only
  when the run simulates. For a policy loaded to resume or only to simulate,
  these are the rows a selection method deactivated or `max_active_per_stage`
  evicted in the run that wrote it. For a `warm_start` policy and for
  `policy.boundary` rows, they are the scored inactive rows; inactive rows
  that were never scored, such as those a capped source leaves, are not
  included. The terminal pool of a checkpoint is not checked when
  `policy.boundary` replaces it. Set `simulation.cut_set` to `active` or use
  `level1`.
- `cobre validate` reconciles `policy.boundary` only when the run trains.
- **BREAKING:** a refused policy loaded only to simulate is reported by
  `cobre validate --json` and `cobre.io.validate` as
  `SimulationPolicyIncompatible`, no longer as `WarmStartIncompatible`, which
  now covers only `policy.mode = warm_start`. The warnings of an accepted
  simulation-only load, which the Python validation also returns, change kind
  the same way. Filter on both kinds to keep the previous behaviour.
- x86_64 Linux builds of the MPI release archive and builds from source,
  `cargo install` included, no longer enable AVX2 and FMA. Every x86_64 build
  of a version now uses one floating-point kernel, so they select the same
  cuts. Results of those builds can differ from 0.18.0's.
- **BREAKING:** `constraints/hydro_bounds.parquet` rejects a positive
  `max_diversion_m3s` on a hydro with no `diversion` channel, as it already
  rejected a positive `min_diversion_m3s` (rule 46). Before, the diverted water
  left the system without reaching any plant.
- **BREAKING:** a negative `diversion.max_flow_m3s` in `system/hydros.json` is
  rejected at load.
- **BREAKING:** `equivalent_productivity_mw_per_m3s` in
  `system/hydro_energy_productivity.parquet` sets only the reference-point
  productivity. A plant with VHA geometry and a specific productivity now
  integrates its useful-range mean productivity even when this override is
  present, so its `integrated_*` and `max_stored_energy*` tags, the
  `integrated_*_productivity_mw_per_m3s` output columns and
  `stored_energy_*` change wherever the override differed from the integral.
  The same productivity supplied from `hydro_production_models.json` already
  behaved this way. To keep a study's previous values, write the same number
  to the new `integrated_equivalent_productivity_mw_per_m3s` column.
- **BREAKING:** `system/hydro_energy_productivity.parquet` must carry the
  `integrated_equivalent_productivity_mw_per_m3s` column; a file without it
  is rejected at load. Add it as an all-NULL column to keep current behavior
  for the reference-point productivity.
- A plant whose geometry gives a non-positive useful-range mean head is
  rejected at setup even when it has an `equivalent_productivity_mw_per_m3s`
  override; the error names the plant and the integrated column that
  supplies its value instead.
- **BREAKING:** the `linearized_head` production model is removed. Both
  `hydros.json` and `hydro_production_models.json` accepted it, but it always
  ran as `constant_productivity`, in training and in simulation. A file that
  names it is rejected at load; write `constant_productivity` instead, which
  gives the same results. Such a hydro no longer needs two rows in
  `system/hydro_geometry.parquet`.
- **BREAKING:** `model` and `default_model` in
  `system/hydro_production_models.json` accept only `constant_productivity`
  and `fpha`. Any other name used to run silently as constant productivity;
  it is now rejected at load, naming the `model` field.
- The stored-energy pairing warning (rule 51) also flags a constraint that
  mixes raw and MWh stored-energy tags, and a `max_stored_energy*` tag beside a
  same-unit productivity tag of another scope or evaluator for the same hydro.
- **BREAKING:** an enabled generic-constraint slack must declare
  `slack.penalty_basis`. `per_hour` charges the penalty for every hour the
  row covers, as before: the stage's total hours for a stage-level row and
  the block's hours for a block row. Use it for a row in MW or m³/s.
  `per_unit` charges the penalty once per unit of violation, whatever the
  stage length. Use it for a row in MWh or hm³. To keep a study's current
  costs, add `"penalty_basis": "per_hour"` to every enabled slack.
- **BREAKING:** a disabled generic-constraint slack that sets `penalty` or
  `penalty_basis` is rejected. These fields used to be ignored.
- `generic_constraints/resolved_echo.parquet` has a new
  `slack_penalty_basis` column, null for a hard constraint.
- HiGHS forward and simulation solves are cold by default: `use_warm_start`
  now defaults to `false` for those phases, so their LPs ignore the stored
  basis and run presolve. Backward solves still warm-start. Set
  `"use_warm_start": true` in `training.solver.forward` or `simulation.solver`
  to restore warm-starting for that phase. A cold solve can stop at a
  different but equally optimal vertex, so simulated duals and the final
  lower bound may differ slightly from 0.18.0 on the same inputs.
- HiGHS retry escalation keeps the phase's `use_warm_start` at every retry
  level; levels 5-11 previously reset it to the solver default.
- `cobre run` writes its banner, progress lines and summaries to stdout;
  warnings and errors stay on stderr, and progress bars on an interactive
  terminal still draw on stderr. Scripts and job logs that read the run report
  from stderr must read stdout instead. `--color` now applies to both streams.
- The training gap in the progress lines and the training summary is printed
  with 5 decimals instead of 1, so small gaps no longer round to `0.0%`.
- A generic constraint written without a relational operator now drops a
  variable whose coefficient is exactly zero, such as `0 * x` or `x - x`, as a
  relation already did. When every remaining variable is stage-level and
  no coefficient or bound resolves to a per-block value, the row now collapses
  to one stage-level row, so under `"penalty_basis": "per_unit"` a violation of
  it costs the penalty once instead of once per block.
- A generic-constraint expression nested deeper than 256 levels, counting
  parentheses and unary operators alike, is rejected with a diagnostic.
  Expanding named expressions into a constraint is likewise refused once the
  expansion exceeds an operation budget set far above the size of an authored
  constraint.
- **BREAKING:** in `constraints/generic_parameters.json`, a payload field that
  does not belong to the entry's kind (for example `values` on a `constant`)
  stops the load. Only `block_values` was rejected before; the other fields
  were ignored.

### Removed

- **BREAKING:** the `lml1` and `domination` values of
  `training.cut_selection.selection.method`. A config naming either fails at
  load and points to `level1`.
- **BREAKING:** `tie_tolerance` of `level1`. A config that sets it fails at
  load. `level1` compares row values exactly: delete the key.

### Fixed

- The initial storage was moved onto the first stage's `min_storage_hm3` and
  `max_storage_hm3` from `constraints/hydro_bounds.parquet`, which bound the
  storage at the end of that stage. A first-stage minimum above the initial
  storage added water before the first solve, and a maximum below it removed
  water. The initial storage is now kept within the hydro's own storage range.
- A `min_turbined_m3s` or `min_generation_mw` value in
  `constraints/hydro_bounds.parquet` had no effect: the LP read only the unit
  groups' minima. A plant minimum above the sum of its groups' minima now
  binds on the plant's total, and a shortfall is charged once, at the plant's
  penalty, in `turbined_slack_m3s` and `generation_slack_mw`. A value below
  that sum cannot loosen the groups' minima, and setup now warns about it.
- A `max_turbined_m3s` or `max_generation_mw` value in
  `constraints/hydro_bounds.parquet` capped each bus's share of a plant whose
  unit groups sit on several buses, but not their total, so the plant could
  turbine or generate up to that value on every bus at once. A plant maximum
  below the sum of its buses' capacities now caps the plant's total.
- `truncation` corrects a negative inflow when the hydro's PAR residual
  standard deviation is zero. Such a study could stop on an infeasible LP
  before.
- `cuts_removed` in `training/convergence.parquet` was always 0.
  `cuts_active` in the same file ignored evictions and reactivations, and
  `row_pool.peak_active` in `training/metadata.json` missed a peak reached
  after a reactivation.
- A `max_active_per_stage` below the number of cuts loaded by `policy.boundary`
  aborted training on a basis row-count mismatch. The cap now counts and
  evicts only the rows the run generated.
- A lazy solve under `dynamic` that reached the round cap returned a relaxed
  result. It now ends only when no candidate row is violated.
- `cobre validate` printed two warnings twice.
- A checkpoint write refuses a payload a load would reject, and a file over the
  2 GiB limit of the policy format, naming the file. The writers refuse an
  entity manifest whose dates are inconsistent and repeated ids.
- A basis-store decode failure at the end of a multi-rank run no longer
  leaves the other ranks blocked.
- Under MPI, training multiplied its LP-solve count by the number of ranks, and
  the first-try/retried/failed split and the forward solve seconds with it. The
  training summary and `solve_stats` in `training/metadata.json` now report the
  same counts for any rank count, and the split adds up to the total.
- `cobre run` under MPI no longer panics when the forward passes do not
  divide evenly across ranks. A rank holding fewer passes than another could
  index its stored warm-start bases out of bounds, and the job then hung. Such
  a run now writes the same outputs as a single-process run.
- A rank that panics during `cobre run` under MPI aborts the whole job
  with exit code 101. The other ranks used to wait in a collective until the
  job was killed.
- A generic constraint whose terms land on the same LP column (a repeated
  variable, or an aliased pair such as `hydro_storage` and `hydro_storage_final`)
  no longer aborts the run when HiGHS loads the LP. The terms are summed into one
  coefficient per column, and a column whose coefficients sum to exactly zero
  gets no entry.

## [0.18.0] - 2026-10-07

### Added

- `cobre validate --output <DIR>`, resolved like `cobre run --output`.
- Periodic policy checkpoints: with `policy.checkpointing.enabled: true`,
  training writes a checkpoint every `interval_iterations` iterations.
- `cobre run` stops at the next iteration boundary on SIGTERM or SIGINT. It
  writes the training outputs and policy checkpoint, skips the simulation, and
  exits 5.
- `training/hydro_models.json` lists the plants with no turbine capacity under
  `no_turbine_capacity`.
- The MPI release archive's `README.txt` explains how to stop before a SLURM
  time limit with `--signal`.

### Changed

- **BREAKING:** policy checkpoints are written with `format_version` 3, and
  policies written by 0.17 or earlier are refused. Re-run the program that
  produced the policy, or convert a boundary policy again.
- **BREAKING:** a policy loads only in the software and version that wrote it.
  `metadata.json` replaces `cobre_version` with `software` and
  `software_version`.
- **BREAKING:** `training.stopping_rules` must contain an `iteration_limit`
  rule. The implicit 100-iteration limit is gone.
- **BREAKING:** validation rejects a pumping station that is active while its
  source or destination hydro is not operating, two seasons of one resolution
  level that share a calendar day, and a generic-constraint variable that
  repeats its block argument.
- With `stopping_mode: "all"`, `iteration_limit` caps the run instead of being
  one of the rules that must all hold.
- `convergence.achieved` is true when a `gap` or `bound_stalling` rule ends
  training.
- `status` in `metadata.json` is `complete` or `partial`. `partial` marks a
  training stopped by a signal and the simulation it skipped.
- Stored bases that no longer fit the LP are skipped with a warning instead of
  refusing the policy.
- `cobre validate` runs the study-setup and policy-load checks `cobre run`
  runs, exits with the same codes, and prints every refusal on stdout. New
  `--json` phase kinds: `StudySetupError`, `WarmStartIncompatible` and
  `ResumeIncompatible`.
- `policy.path` is refused (exit 1) when it is empty, names the output
  directory or a directory above it, names or contains a directory a run
  clears, or lies inside one the run removes whole. For a run that trains, it
  is also refused when it is a regular file or holds files cobre did not write.
  These checks run before training.
- Every policy-checkpoint refusal gives the same remedy: re-run the program
  that produced the policy, or convert a boundary policy again.
- Exit codes follow the error: a missing or unparseable policy exits 1, an
  unreadable one 2, refused stochastic data 1 and an infeasible LP 3. All of
  these used to exit 4.
- `policy.checkpointing.enabled: true` requires `interval_iterations` of at
  least 1.
- `scenario_source.seed` is optional when the only non-`in_sample` class is
  `historical`.

### Fixed

- `cvar` with `0 < lambda < 1` now computes
  `(1 − lambda)·E[Z] + lambda·CVaR_alpha[Z]`. It computed a pure CVaR at another
  confidence level, so results for these studies change.
- Custom and weekly season maps follow the calendar, not season id numbers or
  calendar years. This applies to inflow lag seeds, PAR lag seasons and
  statistics, PAR fitting, monthly-to-quarterly lag aggregation, inflow-history
  grouping and historical windows. Results can change for custom maps whose ids
  are not consecutive or not in calendar order, that layer several
  resolutions, or that have a season spanning 1 January, and for weekly maps.
  This change leaves monthly maps unchanged.
- PAR estimation pairs each observation with the lag of the right year. Fits
  change when the inflow history does not start at the first season of the
  year, or when the maximum AR order exceeds the number of seasons.
- PAR lag statistics no longer vary between runs or MPI ranks.
- Historical inflow sampling admits every year whose study-season observations
  are complete. Results change for historical sampling or `historical_residuals`
  openings with a PAR order above 0.
- Water in transit toward a plant that retires before it arrives now reaches
  the next operating plant downstream.
- Directional evaporation and withdrawal costs reach the LP. A hydro's
  `penalties` block no longer discards those of `penalties.json`, and a
  symmetric cost on a `constraints/penalty_overrides_hydro.parquet` row now
  sets that stage's directional costs.
- A run killed while writing its policy checkpoint leaves a loadable
  checkpoint. The write uses `<policy.path>.staging` and
  `<policy.path>.previous`, so keep policy backups under other names.
- `_SUCCESS` markers are written after every other file of their phase. Before
  writing, a run removes the previous run's markers, simulation files and
  optional training files for each phase it runs.
- A resumed run continues the `bound_stalling` window and stops at the absolute
  `iteration_limit`.
- `simulation.scenario_source.seed` now seeds the simulation's out-of-sample
  draws.
- A study with a file-supplied opening tree is no longer refused over a
  historical library it does not use.
- Computed FPHA exports load unedited as precomputed input. A plant with no
  turbine capacity needs no hyperplane rows, and hyperplanes with
  `gamma_q = 0` are accepted.
- A historical-inflow study with a stage that has no season is refused (V2.1)
  instead of crashing.
- Clearer errors:
  - V2.3 names the window year, stage id and hydro id;
  - block-selector errors print the whole term;
  - generic-constraint reference errors name
    `constraints/generic_constraints.json`.
- Penalty-ordering warnings compare only costs in the same unit.
- Corrected units and descriptions in `training/dictionaries/variables.csv`.
- Input schemas:
  - descriptions are cleaned up for case authors, with consistent units;
  - `generic_parameters` lists every `kind`;
  - `config.schema.json` marks `training.selection` and
    `training.stopping_rules` required;
  - `stages.schema.json` drops `scenario_source`.

## [0.17.0] - 2026-10-01

### Changed

- **BREAKING:** a policy loads only in the Cobre version that wrote it.
  Warm-start, resume, simulation-only and boundary-cut loads refuse a
  checkpoint from another version, naming both. Retrain the policy, or
  re-export the boundary policy, with the running version. The CLI reports a
  validation error.
- **BREAKING:** a policy checkpoint whose stored basis does not match the
  study's LP dimensions is refused at load, naming the node and the expected
  and found columns and rows. The CLI reports a validation error.
- **BREAKING:** a generic constraint naming an evaporation block other than 0
  on a parallel stage with two or more blocks is rejected at validation.
  Reference block 0 or no block.
- **BREAKING:** a study whose precomputed inflow model does not match its
  stages and hydros is refused at setup, naming both shapes. It used to run as
  a zero-inflow model.
- **BREAKING:** under historical inflow sampling, a study whose backward pass
  cannot build a valid historical opening tree is refused at setup. This covers
  a stage without a season, no complete historical window and a non-finite
  standardized inflow.

### Fixed

- A hydro with no turbine capacity no longer aborts the run when its
  production model is computed FPHA. It is modeled with zero productivity,
  counted as a constant-productivity plant in the run summary, and named in a
  warning.
- A generic constraint's `hydro_inflow` term accounts for water travel time,
  including transit water from earlier stages, and counts water passing
  through a not-yet-built or retired upstream plant. Studies without these are
  unchanged.
- On a chronological stage with two or more blocks, pumping-station flow and
  inflow noise now reach their own hydro, and `water_value_per_hm3` reports
  each block's own water-balance dual. Parallel-stage results are unchanged.
- On a parallel stage with two or more blocks, an evaporating hydro evaporates
  as one stage-level quantity, its violation priced for the whole stage.
- Cuts and the policy file keep every in-flight commitment when anticipated
  thermals have different leads. The lower bound could be wrong before.
- An anticipated thermal decision after the first stage is no longer
  discounted twice under a positive discount rate.
- The training lower bound includes stage-0 load uncertainty.
- Historical inflow sampling finds lag years from the season calendar, and the
  backward pass's opening tree draws from the same years and uses the same
  inflow model as the forward pass. Studies without historical inflow
  sampling are unaffected.

## [0.16.0] - 2026-09-22

### Added

- Simulation output columns
  `integrated_equivalent_productivity_mw_per_m3s` and
  `integrated_accumulated_productivity_mw_per_m3s` (unit `MW/(m3/s)`), written
  after the existing productivity pair.
- Columns `stored_energy_initial_mw` and `stored_energy_final_mw`: the
  `stored_energy_{initial,final}_mwh` value divided by the stage's total block
  hours.
- Computed scalar-parameter tags `integrated_equivalent_productivity`,
  `integrated_accumulated_productivity` and `max_stored_energy`.
- Generic-constraint variables `hydro_useful_volume_initial(id)` and
  `hydro_useful_volume_final(id)`: the `hydro_storage_{initial,final}` column
  with the bound shifted by the plant's physical dead volume.
- A non-blocking validation warning for a security-curve constraint that pairs
  `max_stored_energy(h)` with `accumulated_productivity(h)`. The matching tag
  is `integrated_accumulated_productivity`.
- Stored state values are projected onto their bounds at read-back, so solver
  drift no longer aborts a run. A committed value outside the delivery stage's
  generation bounds is rejected at case load.
- `policy.boundary.strict` (default `false`): when `true`, a source pool that
  prices an entity or commitment the study does not model is rejected, naming
  every dropping family and its count. Boundary loading no longer emits
  warning lines. `cobre validate --json` gains `boundary_date`,
  `dropped_source_slots` and `straddling_slots`.

### Changed

- `stored_energy_initial_mwh` and `stored_energy_final_mwh` now use the
  `integrated_accumulated_productivity` grid and the plant's physical dead
  volume. Values change for studies with a non-collapsed storage range and
  resolvable VHA geometry; others are unchanged.
- The `specific_productivity_mw_per_m3s_per_m` column of
  `hydro_energy_productivity.parquet` now reaches the head-derived
  productivity of both evaluators. An all-`NULL` column is unchanged.
- A stage with no block, or a block whose duration is not finite and positive,
  is rejected at validation.
- `cobre validate` runs the generic-constraint parameter check for every deck,
  reporting an unresolved scalar parameter as
  `GenericConstraintValidationError`.
- A negative `volume_discretization_points`, `turbine_discretization_points`,
  `spillage_discretization_points` or `max_planes_per_hydro` in
  `hydro_production_models.json` is rejected at load, naming the field and
  hydro.
- Enumerated forward traversal combined with dynamic cut selection is rejected
  at setup.
- **BREAKING:** `policy.boundary.source_stage` is removed and rejected by
  `config.json`; delete the key. The source pool is chosen by the date equal to
  the study's last stage `end_date`. Checkpoints written before
  `format_version` 2 are rejected on load; re-export the source policy.
  `report.anticipated_coverage.source_month_count` in `cobre validate --json`
  is renamed `source_interval_count`, and the object drops `source_span` and
  `target_span`.
- **BREAKING:** a bound-override row whose `stage_id` is not a declared study
  stage is rejected at validation, for every bound family. Remove the rows or
  correct their `stage_id`.
- Rows of `constraints/generic_constraint_bounds.parquet` follow the other
  bound families: an out-of-range `block_id` is a `BusinessRuleViolation`
  instead of an `InvalidValue`, and rows setting disjoint columns for the same
  constraint, stage and block are no longer duplicates.
- A missing required column in `hydro_geometry.parquet`,
  `hydro_energy_productivity.parquet` or `tailrace_curves.parquet` reads
  `missing required column "<name>"`.
- **BREAKING:** the `kind` of a `cobre validate --json` load-phase error is now
  one of `IoError`, `ParseError`, `SchemaError`, `ConstraintError` or
  `PolicyIncompatible`, replacing `CaseValidationError`. Filter on
  `ConstraintError` for a duplicate bus id.
- `cobre validate --json` emits an error object for a `config.json` parse
  failure, an invalid `training.scenario_source` and a boundary reconciliation
  reject (`kind` `BoundaryReconciliationError`). Stdout was empty before.

### Removed

- **BREAKING:** the `cobre report` and `cobre summary` subcommands. Read
  `training/metadata.json` and `simulation/metadata.json` directly.

### Fixed

- `cobre validate` rejects a boundary-configured study whose scalar-parameter
  table has a genuine gap, and a generic constraint referencing a scalar
  parameter that never resolves, instead of resolving it to `0.0`.
- A boundary policy load whose study covers only part of the source's season
  cycle compares season and PAR order only at the seasons the study
  references. A checkpoint season descriptor with hydros out of ascending id
  order, or an order list not spanning the season count, is rejected.
- A multi-rank `cobre run` applies the terminal boundary policy on every rank.
  Multi-rank results with a boundary policy change and are bit-identical
  across rank counts.
- Out-of-sample `inflow`, `load` and `ncs` classes no longer share a noise
  stream. Results change for decks with two or more out-of-sample classes.
- `penalty_overrides_ncs.parquet` curtailment overrides now apply to the LP
  objective.
- An invalid `simulation.scenario_source` is rejected at case load and by
  `cobre validate`, as `cobre run` already did.
- Thermal bound-override rows naming the last declared stage are no longer
  rejected when stage ids do not start at 0.
- Policy checkpoint files and dictionary CSVs are written atomically, and a
  rewrite first removes the previous checkpoint's manifest and payload files.
  A crash leaves the complete previous checkpoint or none, and stale cut,
  basis and state files are no longer read back.
- `simulation/metadata.json` written by `cobre run` carries `solver_version`.

## [0.15.0] - 2026-08-24

### Added

- `post_study_stages.json` gains `thermal_bounds[]`, one
  `{thermal_id, post_study_stage_index, cost_per_mwh, min_mw, max_mw}` row per
  post-study delivery cell. It is the only place to declare a thermal
  commitment that delivers past the study horizon.
- `past_anticipated_commitments` windows may extend past the study horizon.
  They are validated like any other window: coverage 1.0 over every
  post-horizon delivery stage the plant decided before the study, and an
  explicit `0 MW` window counts as coverage. The run reports them at their real
  delivery date in a run-level fixed-delivery table.
- A policy checkpoint is self-describing: `policy/manifest.bin` replaces
  `policy/metadata.json`.
- A terminal boundary policy may come from a source study whose state shape
  differs from the loading study's. A source cut coefficient on a state slot
  the loading study does not model gives a named warning per family, and the
  load succeeds.

### Changed

- **BREAKING:** every policy checkpoint written by an earlier release is
  rejected by name on every load path (warm-start, resume, simulation-only,
  boundary injection). Re-export it with this release, or retrain. A study
  with post-study anticipated deliveries, or whose ring depth widens (see
  below), must retrain because its state dimension changed.
- A plant whose service window ends inside the study can no longer be
  committed to a post-horizon delivery.
- The anticipated-commitment ring is sized from the full in-flight occupancy,
  pre-study seeds included. A study whose ring was too small gets a wider
  state vector, a different checkpoint and different, correct results. Other
  studies are unchanged.
- Load rejects an anticipated thermal whose lead reaches a post-study stage
  with no `thermal_bounds[]` cell, naming the plant and stage. It also rejects
  a commitment decided at a pre-study stage whose delivery lands past the
  horizon, naming the plant. The first used to log only a warning.
- Under the External sampling scheme, a class's realized values come from its
  external scenario file. Load, NCS and inflow standardization use the
  external samples, so a seasonal-statistics file is optional. A constant
  (σ = 0) column is accepted for load, NCS and an AR(0) inflow model. For an
  AR(p > 0) inflow model it is still rejected, with a message that now states
  the real reason.

### Removed

- **BREAKING:** `initial_conditions.json`'s `future_anticipated_deliveries[]`
  is removed and the file rejects it. Move each entry to a
  `post_study_stages.json` `thermal_bounds[]` row.

### Fixed

- An anticipated thermal whose lead reaches the full study horizon no longer
  aborts the run at LP build.
- The anticipated ring depth is the larger of the in-flight occupancy and the
  pre-study seed run. A too-shallow ring aliased a later delivery stage's
  committed value onto an earlier one.

## [0.14.3] - 2026-08-19

- No changes visible to CLI users.

## [0.14.2] - 2026-08-19

- No changes visible to CLI users.

## [0.14.1] - 2026-08-17

### Added

- A `gap` stopping rule is accepted under a CVaR risk measure when the forward
  selection is `enumerated` and the measure is uniform across stages. Sampled
  forwards and stage-varying measures are still rejected.

### Changed

- The minimum- and maximum-outflow rows both bind the non-diverted flow
  (turbine plus spill) and exclude the diversion channel. A plant that diverts
  can no longer meet its minimum outflow with diverted water, and its
  diversion is no longer capped by the maximum-outflow row. Decks without
  diversion are unchanged.

### Removed

- **BREAKING:** the `state_space` config section
  (`state_space.inflow_lag_depth`) is removed; the depth is inferred from a
  loaded boundary policy. A `config.json` that declares it fails to load.
  Delete the section.

### Fixed

- A loaded terminal boundary policy no longer gives an invalid lower bound
  (a persistent negative gap) for a study whose stages disable inflow-lag
  cut-state.

## [0.14.0] - 2026-08-13

### Added

- `stages.json` accepts `policy_graph.nodes[]`, one entry per decision point:
  `id`, `stage_id`, optional `scenario_id` and optional `label`. Once
  `nodes[]` is non-empty, `transitions[].source_id` and `target_id` are node
  ids, and `transitions[].probability` is the only place a transition
  probability is declared. A study without `nodes[]` behaves as before.
- Three external-scenario files carry realizations for a node graph's
  `scenario_id` columns: `scenarios/external_inflow_scenarios.parquet`,
  `scenarios/external_load_scenarios.parquet` and
  `scenarios/external_ncs_scenarios.parquet`, one row per
  `(stage_id, scenario_id, entity_id)`. Load rejects a `scenario_id` set that
  is not exactly `{0..raw_c(t)-1}` per entity and stage, a row whose
  `stage_id` names no declared stage, and two external classes that disagree
  on their per-stage raw column count.
- Load rejects `training.scenario_source.openings` set to
  `{"source": "file"}` together with `policy_graph.nodes[]` or an enumerated
  `training.selection.method`, naming the conflict.
- The `gap` stopping rule: `training.stopping_rules[]` accepts a `"gap"` entry
  with `tolerance` and/or `relative_tolerance`. It is admitted only under
  enumerated forward selection with an expectation risk measure at every
  stage, and other uses fail with a named error. `training/convergence.parquet`
  gains `upper_bound_kind`: `"exact"` or `"statistical"`.
- `training.parallelism.backward_scheduler.method` accepts `"by_node"` with an
  optional `block_size`, next to the default `by_scenario`.
- Every `simulation/` entity Parquet file carries `(scenario_id, stage_id,
node_id)`. On a stage chain `node_id` equals `stage_id`. A new run-level
  `simulation/paths.parquet` records the node path of each simulated scenario,
  one row per `(scenario_id, stage_id)`.
- Generic constraints support named expressions, referenced as `@name` in any
  expression or bound, and net line flow addressed by `(source_bus,
target_bus)`.
- A generic constraint's `expression` may be written as `lhs <op> rhs` with
  `<=`, `>=` or `==`. A parenthesised group scaled by a literal coefficient
  distributes.
- Scalar parameters can vary per `(stage, block)` and can supply a
  constraint's bound.
- `generic_constraints/resolved_echo.parquet` holds the resolved generic
  constraints, one row per `(constraint, stage, block, term)`.
- A minimum-flow floor for hydro diversion and a min/max band for spillage,
  as per-`(stage, block)` bound columns.
- `config.json`'s `policy.boundary` (`path`, optional `source_stage`) loads
  a terminal future cost function from a previously trained checkpoint.
  `source_stage` defaults to a calendar-overlap match with the terminal
  window. Source delivery states are reconciled onto the study calendar and a
  per-family summary is reported at load. A multi-node source is rejected.

### Changed

- **BREAKING:** `metadata.json` carries `format_version`, and a checkpoint
  written before this release fails to load with a named error. Retrain.
  `EntitySlot`'s `delivery_anchor` is replaced by `delivery_date`
  (`YYYYMMDD`).
- **BREAKING:** `past_anticipated_commitments` in `initial_conditions.json`
  are `{thermal_id, start_date, end_date, value_mw}` records. A deck with
  `values_mw` is rejected. Re-emit the history as dated windows.
- **BREAKING:** simulation outputs use one name per axis. `stage` becomes
  `stage_id`, `opening` becomes `opening_index`, and `upper_bound_mean`
  becomes `upper_bound` on `training/convergence.parquet`. A not-applicable
  `stage_id` is `NULL`, not `-1`.
- **BREAKING:** `stages[].risk_measure` accepts only `"expectation"` or a
  `{"cvar": {...}}` object. Any other string is rejected instead of training
  risk-neutral.
- **BREAKING:** `policy_graph.type` of `"cyclic"` is rejected as reserved.
- **BREAKING:** `system/scalar_parameters.json` moves to
  `constraints/generic_parameters.json`, with unchanged contents. The old path
  fails to load.
- **BREAKING:** `constraints/generic_constraints.json` no longer carries
  `"sense"`. `constraints/generic_constraint_bounds.parquet` replaces `bound`
  with nullable `bound_lower` and `bound_upper`: lower-only is `>=`,
  upper-only is `<=`, equal is `==`, and both differing is a range. Move a
  former `<=` value to `bound_upper`, a `>=` value to `bound_lower`, and an
  `==` value to both.
- With a declared node graph, `transitions[].annual_discount_rate_override` is
  rejected. Use `stages[].annual_discount_rate_override`.

### Removed

- **BREAKING:** three legacy Parquet column names are rejected. Rename
  `value` to `availability_factor` in
  `scenarios/external_ncs_scenarios.parquet`, `source_id` to `ncs_id` in
  `constraints/penalty_overrides_ncs.parquet`, and `station_id` to
  `pumping_station_id` in `constraints/pumping_bounds.parquet`.
- **BREAKING:** `backward_scheduler.method` values `trial_point` and
  `opening_block` are gone. Use `by_scenario` and `by_node`.
- **BREAKING:** `stages[].num_scenarios` is replaced by `stages[].num_openings`.
- **BREAKING:** the root `training.forward_passes` and flat
  `simulation.num_scenarios` are rejected. Use `training.selection.forward_passes`
  and `simulation.selection.num_scenarios`.

### Fixed

- Reservoir evaporation is scaled by the stage's calendar month, not its stage
  duration, so a stage deposits only its share of the month's evaporation.
- A commissioning-dormant FPHA hydro no longer aborts the LP build.

## [0.13.0] - 2026-07-30

### Added

- A hydro can declare several turbine groups in `unit_groups`, each with its
  own `id`, `name`, `bus_id` and generation/turbined bounds, so one plant can
  span several buses. `constraints/hydro_unit_group_bounds.parquet` overlays
  stage-varying, optionally per-block overrides on `min_turbined_m3s`,
  `max_turbined_m3s`, `min_generation_mw` and `max_generation_mw`.
- `thermal_bounds.parquet`, `hydro_bounds.parquet`, `line_bounds.parquet`,
  `pumping_bounds.parquet` and `contract_bounds.parquet` accept an optional
  `block_id` column for per-block overrides, including per-block
  `contract_bounds.price_per_mwh`, which the simulation cost now honors. A
  per-block thermal `cost_per_mwh` is rejected at validation. Studies without
  per-block rows are unchanged.
- Generic-constraint variables `hydro_turbined` and `hydro_generation` accept a
  `bus=` selector, e.g. `hydro_turbined(5, bus=2)`. An optional block argument
  precedes it.
- `simulation/hydro_bus_generation/` reports turbined flow and generation per
  `(hydro, bus)`. `simulation/hydros/` keeps reporting each plant's total.

### Changed

- A negative realized inflow loads again in `scenarios/inflow_history.parquet`
  and in `recent_observations` of `initial_conditions.json`. Validation warns
  once per file with the negative count and the most negative value.
- **BREAKING:** `scenarios/inflow_history.parquet` uses windowed rows
  (`hydro_id`, `start_date` inclusive, `end_date` exclusive, `value_m3s`). A
  file with the legacy `date` column is rejected; re-emit the history as dated
  windows.
- **BREAKING:** `unit_groups` is required on every hydro. An absent, `null` or
  empty array is rejected, and the exported schema marks the key required.
- **BREAKING:** the top-level `hydro.bus_id` is removed. Use
  `unit_groups[].bus_id`; `hydros.json` rejects the old field.

### Removed

- **BREAKING:** `past_inflows` is removed from `initial_conditions.json` and
  is rejected at load. PAR lag and mid-period seeds derive from the windowed
  `inflow_history`, overridden day-wise by `recent_observations`.
- **BREAKING:** `constraints/exchange_factors.json` is removed and rejected at
  load, naming the replacement. Declare line capacity in absolute MW with
  `direct_mw`/`reverse_mw` rows carrying a `block_id` in
  `constraints/line_bounds.parquet`. `direct_mw = 0.0` now closes a line in one
  block.

### Fixed

- Bound, penalty and factor overrides no longer land on the wrong entity when
  the commissioning-date order of entities differs from the declared-id order.
- A user-supplied `scenarios/noise_openings.parquet` no longer panics a study
  with non-controllable sources. A mis-sized file is reported as a dimension
  mismatch.
- The PAR lag-slot and mid-period seed derivation is corrected. Cases that
  supply `recent_observations` train against different, corrected seeds, and an
  under-covered lag slot or partially covered mid-period window is flagged or
  rejected at load. Cases without `recent_observations` are unchanged.
- A `thermal_bounds.parquet` row with a `block_id` is no longer dropped; its
  per-block `min_generation_mw`/`max_generation_mw` override reaches the LP.
- `training/dictionaries/bounds.parquet` writes `block_id` for per-block
  overrides instead of always `null`.

## [0.12.0] - 2026-07-21

### Added

- `modeling.cost_scale_factor` in `config.json` sets the objective cost-scale
  factor. Absent keeps the previous default. It must be finite and `> 0`, with
  a warning outside `[1.0, 1e12]`. The factor appears in the training scaling
  report and in the policy metadata. Raising it without adjusting
  `dual_feasibility_tolerance` loosens the effective tolerance in currency
  terms.
- Exported policies store cut coefficients and intercepts in canonical
  currency units, so a policy loads into a study with a different cost-scale
  factor. `policy/metadata.json` gains `cost_scale_factor`. Older checkpoints
  still load.
- `training.solver.backward`, `training.solver.forward` and `simulation.solver`
  accept an optional solver-profile block with `dual_edge_weight`, `scale`,
  `price`, `primal_feasibility_tolerance`, `dual_feasibility_tolerance`,
  `presolve`, `simplex_update_limit`, `cost_perturbation`,
  `refactor_error_tolerance`, `factor_pivot_threshold`, `use_warm_start` and
  `steepest_edge_devex_fallback_threshold`. Unset fields keep the previous
  behavior. `use_warm_start: false` is a diagnostic that forces cold solves. The
  CLP backend rejects any solver-profile field at setup, naming the phase and
  setting.
- `training.parallelism.backward_scheduler` accepts
  `{ "method": "opening_block" }` to split backward work by
  `(trial point, opening block)`; the default stays
  `{ "method": "trial_point" }`. The optional `block_size` defaults to half the
  stage's opening count, rounded up. `block_size` under `trial_point` is a
  load-time error. Cuts and the lower bound are unchanged, and a Dynamic Cut
  Selection iteration falls back to `trial_point`.

### Changed

- A checkpoint written by this release must not be read by an earlier
  release. Checkpoints from earlier releases still load.
- The training `Time split` report shows the coordinator's measured phase time:
  `Forward` and `Backward` lines split into `solve` and `wait`, and a `Serial`
  line replaces `Other`. `cobre summary` omits the `Time split` block.
- The backward pass orders each stage's openings along a shortest warm-start
  chain over their inflow-noise vectors, with no config field. At a degenerate
  optimum, training and simulation outputs of a multi-opening stage can shift.
  Results stay reproducible and order-invariant.

## [0.11.1] - 2026-07-17

### Fixed

- The anticipated-commitment drift margin no longer refuses genuine solver
  noise, which had aborted production-scale training. A real over-commitment is
  still refused.

## [0.11.0] - 2026-07-16

### Changed

- `residual_std_ratio` is derived from the AR coefficients whenever
  `inflow_ar_coefficients.parquet` supplies them, and a stored value in that
  file is ignored. Solved outputs differ by about `1e-4` for per-season AR
  orders that vary across the cycle, and by more if the stored value was
  inconsistent with the coefficients. `historical` inflow sampling shifts by
  about `1e-4` in the same case.
- The spectral-clipping diagnostic from correlation estimation is logged at
  debug level instead of warn.

### Fixed

- A study whose anticipated thermal commitment reaches its delivery generation
  cap trains instead of aborting as infeasible. A commitment beyond its cap is
  refused with an error naming the thermal, stage and overshoot.
- A non-stationary fitted inflow model fails the load with an error naming the
  hydro and season, instead of writing `NaN` into the model.
- Multi-rank training with an auto-generated opening tree derives the same tree
  on every rank. Weekly studies with several stages in one season were
  affected.
- Multi-rank runs no longer overstate the lower bound, by about 3%, on
  non-root ranks.
- A stored policy basis with too few basic entries is rejected with a named
  error, reporting the basic-count arithmetic, on both HiGHS and CLP.
- A rank failing under MPI prints its own error to stderr before aborting.
- PAR history estimation no longer panics when the fitting lag depth exceeds
  the season cycle, e.g. two seasons at the default `max_order = 6`.

## [0.10.0] - 2026-07-10

### Added

- `stages.json` gains a per-stage `block_mode`, `"parallel"` (default) or
  `"chronological"`, which chains storage across the stage's blocks. Any other
  value is a schema error naming the stage and value. `parallel` and
  single-block stages are unchanged. `simulation/hydros/` reports each block's
  storage boundaries and evaporation on a chronological stage.
  `policy/metadata.json` records `training_block_mode` and, when it varies,
  `training_block_mode_per_stage`. A policy trained in one block mode loads and
  simulates in the other.
- `travel_time_hours` on a hydro's cascade arc to `downstream_id` delays the
  release by that many hours before it reaches the downstream plant. Absent or
  `0.0` keeps instantaneous transfer. Diversion and pumping-conduit arcs take
  no travel time. `past_defluences` in the initial conditions supplies releases
  already in transit. Validation requires history at least as deep as the
  travel time, derives a proxy from `past_inflows` with a logged caveat, or
  rejects the study. An arc that releases while its downstream plant is not yet
  operating is rejected. Water maturing past the last stage is dropped.
- `simulation/in_transit/` (`stage_id`, `hydro_id`, `lag`,
  `in_transit_volume_hm3`, `delayed_arrival_hm3`) is written for studies with a
  travel-time arc.
- Maturing in-transit water is split across the blocks of a chronological
  arrival stage, blended over the source stages, including a parallel source
  stage. The split is one fixed density per maturing bucket.
- Generic constraints accept `hydro_storage_initial(h)`,
  `hydro_storage_final(h)`, and the per-block forms `hydro_storage_initial(h, k)`
  and `hydro_storage_final(h, k)`. They also accept `hydro_evaporation(h, k)`.
  Bare `hydro_evaporation(h)` is rejected on a chronological stage with more
  than one block. An interior block reference on a parallel stage is rejected.
  A block a stage cannot expose is rejected at load, naming the constraint,
  block and stage block count.
- `state_variables` in `stages.json` (`{ "storage": <bool>, "inflow_lags":
<bool> }`) sets the dimension of the cuts a stage emits. `inflow_lags: false`
  gives storage-only cuts under a PAR(p) model. A study that fits PAR(p > 0)
  but disables inflow lags on every stage gets a model-quality warning.
- Validation rejects an anticipated thermal whose `lead_time_hours` exceeds the
  study horizon, and a non-Monthly season cycle that supplies an inflow annual
  component.

### Changed

- **BREAKING:** every `system/*.json` entity (buses, hydros, thermals, lines,
  non-controllable sources, pumping stations, energy contracts) requires
  `operational_start_date` (`YYYY-MM-DD`). A missing or invalid value is a
  schema error naming the file, field and string.
- **BREAKING:** environment variables are no longer read. `COBRE_THREADS`,
  `COBRE_COLOR`, `COBRE_COMM_BACKEND`, `COBRE_W1_DIAG`, `FORCE_COLOR`,
  `NO_COLOR`, `COLUMNS` and `HOSTNAME` are gone. Use `--threads`,
  `--color <auto|always|never>` (default `auto`) and
  `--comm-backend <auto|local|mpi>` (default `auto`, which picks MPI under
  `mpiexec`/`mpirun`/`srun`). `--comm-backend mpi` fails on a binary built
  without MPI.
- Entities sort by `(operational_start_date, id)` instead of
  `(operational_start_date, name)`. A rename no longer changes the LP layout,
  cut order or output column order. Same-date entities ordered differently by
  name than by id give a reordered but equivalent LP.
- `anticipated_config` in `system/thermals.json` accepts `lead_time_hours`, a
  duration in hours, as an alternative to `lead_stages`; the two are mutually
  exclusive. Each commitment is bounded, costed and commissioning-gated at its
  delivery stage. A `lead_time_hours` configuration whose decision stage would
  anchor more than one delivery stage is rejected at setup.
- A non-filling hydro's `entry_stage_id`/`exit_stage_id` window now takes
  effect: outside it the plant is modeled as PreFilling, with turbine, spillage
  and diversion at zero, and its inflow passes downstream. The parsed-but-inert
  warning is removed. Spillage is zero for every PreFilling hydro and stays free
  during Filling. Studies with pre-commissioning or PreFilling hydros give
  different results.
- Policy cut and state files embed a per-slot entity manifest, and every policy
  load (warm-start, resume, simulation-only) is validated against it. A policy
  whose dimensions match by count but belong to different entities is rejected.

### Removed

- **BREAKING:** `training/dictionaries/state_dictionary.json` is no longer
  written. `training/dictionaries/` still holds `codes.json`, `entities.csv`,
  `variables.csv` and `bounds.parquet`.
- **BREAKING:** `policy.validate_compatibility` is removed, since policy-load
  validation is unconditional. A config that sets it is rejected.

### Fixed

- Weekly or custom season cycles that fit an inflow PAR model now advance the
  inflow-lag state every period instead of freezing it at the initial
  condition. This includes season maps that layer several resolutions.
- A study whose stage ids do not start at zero now gets the correct FPHA
  productivity coefficients.
- Weekly, custom and mixed-resolution calendars: evaporation with a weekly
  cycle builds, a block's evaporation month comes from the stage start date,
  multi-resolution samplers advance the inflow-lag ring like the forward pass,
  and a partial-year study keeps residual seasons outside its window in the
  correlation estimate.
- A failure on one MPI rank no longer hangs the healthy ranks; the run shuts
  down in coordination.
- Reloading a policy checkpoint no longer degrades a CLP warm-start. An
  unrecognized basis status falls back to a cold start, in both directions
  between this and older releases.

## [0.9.1] - 2026-06-26

### Fixed

- A non-zero `stages.json` `annual_discount_rate` now discounts the future-cost
  term when a study has anticipated (GNL-style) thermals. It was left
  undiscounted. Studies without anticipated thermals, or with
  `annual_discount_rate: 0`, are unchanged.

## [0.9.0] - 2026-06-25

### Added

- Energy contracts in `system/energy_contracts.json` take part in dispatch. Each
  contract gives one import or export (`type`) column per block on its `bus_id`,
  bounded by `[limits.min_mw, limits.max_mw]`. A positive `price_per_mwh` is a
  cost (import), a negative one is revenue (export). A non-zero `min_mw` is a
  take-or-pay floor. `constraints/contract_bounds.parquet` supplies
  stage-varying bounds and prices.
- Pumping stations in `system/pumping_stations.json` take part in dispatch. Each
  station gives a per-block pumped flow bounded by `flow.min_m3s`/`flow.max_m3s`
  from `source_hydro_id` to `destination_hydro_id`, and draws
  `consumption_mw_per_m3s × flow` of power on its `bus_id`.
- `entry_stage_id`/`exit_stage_id` set the active window `[entry, exit)` of
  contracts and pumping stations.
- Simulation writes `simulation/contracts/` and `simulation/pumping_stations/`,
  and adds the `contract_cost` and `pumping_cost` cost columns.

### Changed

- **BREAKING:** `filling_inflow_m3s` is renamed `filling_min_rate_m3s`, in the
  `filling` block of `system/hydros.json` and in
  `constraints/hydro_bounds.parquet`. It is now a per-stage minimum accumulation
  rate, not a cap on retained inflow, so the meaning is inverted. Rename the
  field and review its value.
- `entry_stage_id`/`exit_stage_id` now take effect on thermals
  (`system/thermals.json`), lines (`system/lines.json`) and anticipated
  thermals. Outside the window a thermal's generation bounds (including a
  must-run floor) and a line's flow caps are zero. A dormant thermal, line, NCS
  or pumping entity now writes a zero-valued output row instead of being
  omitted.
- `constraints/pumping_bounds.parquet` rows are rejected when a bound is
  negative or `min_m3s > max_m3s`.

## [0.8.2] - 2026-06-17

### Added

- Optional `system/tailrace_curves.parquet`: piecewise-quartic tailrace curves
  with backwater families, used by the computed-FPHA fit for plants that have
  rows.
- `reference_volume` in `system/hydro_production_models.json`: `volume_hm3` or
  `percentile`, exactly one. It is the reference volume for the computed-FPHA
  fit and the equivalent productivity.
- Optional `fpha_plane_reduction` in the production-model config merges
  near-parallel FPHA planes: `{ "method": "angle", "tolerance_deg": <0-90> }` or
  `{ "method": "distance", "tolerance_pct": <f64>, "n_samples": <u32> }`. Off by
  default.
- A warning names the plant and stage when a computed-FPHA fit deviates from the
  exact production function by more than 5 %.
- `output/hydro_models/evaporation_models.parquet` holds the resolved per-hydro
  evaporation coefficients, written when any hydro declares an evaporation
  model.
- `training/metadata.json` has a `setup` section with the wall-clock time of each
  study-setup phase.

### Changed

- Computed FPHA is fit with a 3-D convex hull and a least-squares `α`
  correction, resolved per stage. Run-of-river plants now fit. Computed-FPHA
  results change.
- **BREAKING:** `training.cut_selection` method parameters move into a
  `selection` object. `selection.method` is `"level1"`, `"lml1"`, `"domination"`
  or `"dynamic"`, and a parameter of another method or a misspelled `method`
  fails config load. Omitting `selection` disables row selection. Renames:
  `cut_activity_tolerance` to `row_activity_tolerance` (top level),
  `active_window` to `seed_window`, `candidate_window` to `candidate_recency`,
  `nadic` to `max_added_per_round`, `domination_epsilon` to
  `domination_tolerance`. The dynamic `violation_tolerance` defaults to `1e-10`.

### Removed

- **BREAKING:** `training.cut_selection` fields `enabled`, `method`,
  `threshold`, `memory_window` and `basis_activity_window`. Move the config to
  the `selection` block.
- **BREAKING:** the `config.json` `energy` section and its
  `reference_volume_fraction`. Remove the block and set `reference_volume` in
  `system/hydro_production_models.json`.
- The `kappa` shrink and low-`kappa` warnings of computed FPHA. The `kappa`
  column of `fpha_hyperplanes.parquet` is still accepted and defaults to `1.0`.
- The `reference_volume_hm3` column of `system/hydro_energy_productivity.parquet`
  is ignored with a warning.

## [0.8.1] - 2026-06-13

### Added

- `training.cut_selection.method = "dynamic"` loads only a small resident subset
  of cuts into each LP and keeps the full pool. It is tuned by `active_window`,
  `candidate_window` and `nadic`, and excludes `level1`, `lml1` and `domination`.
- `training.cut_selection.active_window`: the seed window `k2` of the dynamic
  method. Default `5`; `0` seeds only the current iteration's cuts.
- Source builds pick the LP backend with Cargo features: `highs` (default) or
  `clp` (`--no-default-features --features clp`). They are mutually exclusive.
  `cobre version` and the `solver` / `solver_version` metadata fields show the
  active backend. Results can differ between backends.
- `anticipated_thermal_cost` in the run cost output, so the named cost
  categories sum to `immediate_cost`.
- Dynamic cut selection reports the mean and max cuts per LP in the console
  summary and `training/metadata.json`, and `mean_rows_in_lp` per iteration in
  `training/convergence.parquet`.

### Changed

- `method = "dynamic"` no longer reads `check_frequency`; use `active_window`.
  An explicit `check_frequency` is ignored under `dynamic`.
- `training.cut_selection.threshold` and `memory_window` are ignored for every
  method. They still parse.
- Release archives bundle `THIRD_PARTY_LICENSES.md`.

### Fixed

- A hydro that overrides only the symmetric `water_withdrawal_violation_cost` or
  `evaporation_violation_cost` keeps it for the directional `*_violation_pos_cost`
  and `*_violation_neg_cost`, instead of reverting to the global default.
- PAR(p) estimation no longer panics when the horizon is narrower than the
  season cycle, such as a monthly model running September to December.
- The water-withdrawal under-delivery slack is bounded, so realized withdrawal
  cannot cross zero past its target. Only degenerate cases change.

## [0.8.0] - 2026-06-01

### Added

- `cobre report` adds top-level `final_lower_bound`, `final_upper_bound`,
  `mean_cost`, `std_cost` and `cvar` keys to its JSON output.
- `cobre summary` reproduces the full run end-block from a finished output
  directory without re-running the solver.
- `cobre summary` and `cobre report` recover the final lower bound from
  `training/convergence.parquet` when `training/metadata.json` lacks it.

### Changed

- HiGHS primal and dual feasibility tolerances are `1e-9` (was `1e-7`), and the
  offline geometric-mean prescaler replaces the HiGHS internal simplex scaler.
- Warm-start and resume training start from the checkpoint's LP bases. Simulation
  runs and `cobre simulate` also speed up.
- `training/metadata.json` and `simulation/metadata.json` store the execution
  topology, solve-stat summaries and expected-cost statistics (mean, std, CVaR).
- In `training/metadata.json`, `row_pool` has `cuts_active` and `peak_active`;
  `cuts_in_lp` is gone and old manifests that carry it still load.
- `training/convergence.parquet` has 10 row-selection columns; `cuts_in_lp` is
  removed.

### Deprecated

- `training.cut_selection.basis_activity_window` is ignored with a warning, and
  any value loads. Remove it from `config.json`.

### Removed

- The `basis_reconstructions` column of `training/solver/iterations.parquet`. It
  was always zero.

### Fixed

- Forward-pass solver statistics in `training/solver/iterations.parquet` now sum
  all MPI ranks. They were understated by a factor of `world_size`.
- `cobre report` and `cobre summary` report the true generated cut count, not an
  over-count of `forward_passes × stages`.
- The `cobre init` / `1dtoy` template passes `cobre validate` with no errors or
  warnings.
- `cobre summary` and the run end-block drop the duplicate Hydro-production
  section and its misleading `FPHA planes: 0 computed, 0 precomputed` line.
- Warm-start and resume training apply the loaded policy's cuts on the first
  iteration. It used to give an inflated upper bound.

## [0.7.0] - 2026-05-24

### Added

- Anticipated thermal dispatch: a plant with
  `anticipated_config = { lead_stages: K }` in `system/thermals.json` commits
  generation at stage `t` for stage `t + K`.
- `initial_conditions.json` accepts `past_anticipated_commitments`: entries of
  `thermal_id` and a `values_mw` array whose length equals that plant's
  `lead_stages`. Mismatched lengths and non-anticipated thermals are rejected.
- `simulation/thermals.parquet` fills `is_anticipated`,
  `anticipated_committed_mw` and `anticipated_decision_mw` for anticipated
  plants.
- `training/dictionaries/state_dictionary.json` lists `anticipated_state`
  entries.
- Generic constraints accept `anticipated_decision(N)`, where `N` is an
  anticipated thermal's `id`. A non-anticipated thermal is an error, and
  `thermal_generation(N)` on an anticipated thermal warns with
  `SemanticAmbiguity`.

### Changed

- **BREAKING:** `gnl_config` is renamed `anticipated_config` and `lag_stages`
  `lead_stages` in `system/thermals.json`. Old keys fail to parse.
- **BREAKING:** `simulation/thermals.parquet` columns `is_gnl`,
  `gnl_committed_mw` and `gnl_decision_mw` are renamed `is_anticipated`,
  `anticipated_committed_mw` and `anticipated_decision_mw`.

### Fixed

- Anticipated thermals deliver pre-horizon `values_mw` end to end. Commitments
  for in-horizon stages `t >= 1` with `K >= 2` were forced to zero.
- Non-zero in-bounds `past_anticipated_commitments.values_mw` entries no longer
  raise the `SemanticAmbiguity` warning. The bounds check
  `v_k ∈ [min_mw, max_mw]` remains.

## [0.6.2] - 2026-05-20

### Added

- `allow_curtailment: bool` on `NonControllableSource` (default `true`). When
  `false`, the LP pins the source's generation to the realized availability in
  every scenario. Use it for must-run aggregates already netted from the load.

### Changed

- **BREAKING:** `fpha_turbined_cost` is renamed `turbined_cost`, including the
  `penalty_overrides.parquet` column and the output schemas.
- The turbined-flow regularisation cost applies to every hydro's turbine
  column, not only FPHA plants, so lower bounds shift by about 0.10 % on
  constant-productivity cases. The simulated `turbined_cost` no longer reports
  zero.

### Fixed

- PAR(p)-A annual coefficients are now included in the cuts. They were
  omitted, which over-estimated cuts and gave LB > UB at convergence (about
  7 % on the bundled 1983 case, 0.003 % after the fix).
- PAR(p)-A annual coefficients are no longer dropped when standardising
  `historical` or `external` inflows, which biased forward replays.
- Historical scenarios start every forward pass from
  `initial_conditions.past_inflows` and replay the historical inflows exactly.
  Forward upper bounds change for the `Historical` scheme, and the structural
  negative gap closes. Other schemes are unchanged.

## [0.6.1] - 2026-05-18

### Changed

- `HydroEvaporation` is a signed net-flow quantity. Generic constraints that
  assumed it is non-negative may need their bounds revisited.

### Fixed

- The linearised evaporation flow is bounded `[-q_max, +q_max]` instead of
  `[0, q_max]`. Net-rainfall months no longer force over-evaporation slack.
  Cases with all-positive monthly coefficients are unchanged.
- `evaporation_m3s` in `simulation/hydros.parquet` is signed: negative values
  are net rainfall input. Consumers that assumed `evaporation_m3s >= 0` need
  updating.

## [0.6.0] - 2026-05-18

### Added

- Five columns in `simulation/hydros.parquet`:
  `equivalent_productivity_mw_per_m3s`, `accumulated_productivity_mw_per_m3s`,
  `incremental_inflow_energy_mw`, `stored_energy_initial_mwh` and
  `stored_energy_final_mwh`. They replace `productivity_mw_per_m3s` and are
  non-nullable `Float64` after `generation_mwh`.

### Changed

- **BREAKING:** `simulation/hydros.parquet` grows from 31 to 35 columns.
  Read `equivalent_productivity_mw_per_m3s` instead of
  `productivity_mw_per_m3s`. The `variables.csv` dictionary is updated.
- **BREAKING:** every input JSON file rejects unknown top-level and per-entry
  fields with a parse error that names the field. Fix misspellings and stale
  keys before loading a case.
- **BREAKING:** `productivity_mw_per_m3s` moves from the `generation` block of
  `system/hydros.json` to every `stage_ranges[]` or `seasons[]` entry of
  `system/hydro_production_models.json`, which every hydro now needs. Rename
  `productivity_override` to `productivity_mw_per_m3s`.
- **BREAKING:** a hydro with `generation_model: "fpha"` needs VHA geometry plus
  `specific_productivity_mw_per_m3s_per_m`, or a per-`(hydro, stage)`
  `equivalent_productivity` in `system/hydro_energy_productivity.parquet`.
  Otherwise setup fails with an error that names the plant.
- **BREAKING:** `equivalent_productivity_mw_per_m3s` in
  `system/hydro_energy_productivity.parquet` applies to all generation models.
  A `(hydro, stage)` pair supplied in both it and
  `system/hydro_production_models.json` is rejected naming both files. A
  non-FPHA pair supplied in neither is rejected as a coverage gap.
- `productivity_mw_per_m3s` in `system/hydro_production_models.json` is optional
  for `constant_productivity` and `linearized_head` models. Omit it or set it
  to `null` when `system/hydro_energy_productivity.parquet` supplies it.
- Productivity values are validated as `>= 0.0` instead of `> 0.0` in
  `productivity_mw_per_m3s`, `equivalent_productivity_mw_per_m3s` and
  `specific_productivity_mw_per_m3s_per_m`. `0.0` marks a planned outage.
- `modeling.inflow_non_negativity.method` accepts only `"none"`,
  `"truncation"`, `"penalty"` and `"truncation_with_penalty"`. Other strings
  are rejected instead of being read as `"none"`.
- `training.cut_selection.threshold` applies only to `method = "level1"`.
  `"lml1"` requires `memory_window` and `"domination"` requires
  `domination_epsilon`.

### Removed

- **BREAKING:** `system/scalar_parameter_definitions.parquet` and
  `system/scalar_parameter_values.parquet`. Author scalar parameters in one
  `system/scalar_parameters.json`, one object per parameter with a `kind` of
  `constant`, `per_stage`, `seasonal` or `computed`.
- **BREAKING:** these `config.json` keys are rejected with `unknown field`:
  `modeling.inflow_non_negativity.penalty_cost` (use
  `penalties.json::hydro.inflow_nonnegativity_cost`),
  `training.cut_formulation`, `training.forward_pass`,
  `simulation.policy_type`, `simulation.output_mode`, `simulation.output_path`
  and the `exports` flags `training`, `cuts`, `vertices`, `simulation`,
  `forward_detail`, `backward_detail` and `compression`.
- **BREAKING:** `estimation.order_selection` no longer accepts `"fixed"`. Use
  `"pacf"` or `"pacf_annual"`.
- The `version` field of `penalties.json`.

## [0.5.1] - 2026-04-28

### Added

- `"order_selection": "pacf_annual"` activates the PAR(p)-A model with an
  annual component. It is written to `output/stochastic/inflow_annual_component.parquet`
  with columns `hydro_id`, `stage_id`, `annual_coefficient`, `annual_mean_m3s`
  and `annual_std_m3s`.
- Constant and saturated inflow histories get an order-0 fit with zero
  standard deviation. Histories with more than 10 % negative observations are
  classified as `ManyNegative` for diagnosis only.
- PAR(p)-A order selection forces order 0 when the lag-1 conditional FACP is
  exactly zero, keeps at least AR(1) otherwise, and iteratively reduces orders
  that give negative chain-composed contributions.

### Changed

- Seasonal standard deviations use the population `1/N` divisor, so
  `inflow_seasonal_stats.parquet` std values are about 0.5-1.1 % smaller and
  classical PACF order selection can change.
- The PAR(p)-A cross-covariance uses the `max(|A|, |Z|)` divisor.
- HiGHS defaults are retuned for warm-started master LPs. Objective values
  are unchanged within solver tolerances, but the optimal basis may differ.

## [0.5.0] - 2026-04-25

### Added

- Backward-pass basis cache and basis reconstruction across cut-set churn on
  forward, backward and simulation paths. Tune it with
  `training.cut_selection.basis_activity_window` (1-31, default 5).
- Weekly+monthly studies: sub-monthly lag accumulation, `recent_observations`
  input for mid-season starts, terminal boundary cuts through
  `policy.boundary.{path, source_stage}`, and non-uniform per-stage scenario
  counts.
- Multi-resolution studies: shared same-season noise groups, observation
  aggregation from finer to coarser resolution, and a monthly-to-quarterly PAR
  transition.
- Non-TTY progress lines include a trailing `[elapsed HH:MM:SS < eta HH:MM:SS]`.
- Clearer errors for MPI conditions: non-uniform worker counts across ranks,
  basis or cut wire-format version mismatches, and counters above `2^53`.

### Changed

- **BREAKING:** `solver/iterations.parquet` drops from 23 to 19 columns. It
  gains `opening`, `rank`, `worker_id` and `basis_reconstructions` and loses
  eight counter columns. Backward rows gain an `opening` dimension and forward
  rows are one per `(iteration, stage)`.
- **BREAKING:** `cut_selection/iterations.parquet` drops
  `active_after_angular` (10 to 9 columns).
- The lower-bound LP is append-only, so the lower bound is monotonically
  non-decreasing.
- `<output_dir>/hydro_models/fpha_hyperplanes.parquet` follows `--output`
  instead of always going to `<case_dir>/output/hydro_models/`. Rank 0 alone
  writes it.
- Simulation progress on multi-rank runs shows a global scenario count instead
  of a misleading `50/100` on 2 ranks.
- A failed Rayon thread-pool initialisation emits a warning with the
  configured and actual thread counts.

### Deprecated

- `threshold` in the cut-selection config still parses, but warns and points
  to `memory_window` (`"lml1"`) or `domination_epsilon` (`"domination"`).

### Removed

- **BREAKING:** angular diversity pruning: its config section and the
  `active_after_angular` output column.
- The `exports` keys `training`, `cuts`, `vertices`, `simulation`,
  `forward_detail`, `backward_detail` and `compression`. Only `states` and
  `stochastic` remain, and existing keys are silently ignored.
- The cut `domination_count` field. Old policies still load.

## [0.4.4] - 2026-04-14

### Added

- `thermal_bounds.parquet` accepts an optional `cost_per_mwh` column alongside
  `block_id` to override thermal costs per `(plant, stage)`.
- Angular dominance pruning of cuts, set with `angular_pruning.enabled`,
  `cosine_threshold` and `check_frequency`.
- Active cut budget: `max_active_per_stage` caps the cuts in each stage LP and
  evicts the stalest first. It runs every iteration.
- Basis-aware warm-start padding, enabled with the `basis_padding` config flag
  (default `false`).

### Changed

- **BREAKING:** `cut_selection/iterations.parquet` grows from 7 to 10 columns:
  `active_after_angular`, `budget_evicted` and `active_after_budget`.

## [0.4.3] - 2026-04-13

### Added

- `HistoricalResiduals` noise method: the backward-pass opening tree copies
  residuals from historical inflow observations, keeping their cross-entity
  correlation.
- Validation rules 27-30 check season ids: range coverage, observation coverage
  per season, resolution consistency across seasons and contiguity.
- Validation rule 31 checks that observations align with seasons. Historical
  window discovery and standardization follow the season map instead of
  calendar months.

### Changed

- Timing columns renamed: `forward_solve_ms` to `forward_wall_ms`,
  `backward_solve_ms` to `backward_wall_ms`, `mpi_broadcast_ms` to
  `cut_sync_ms`, `rayon_overhead_ms` to `bwd_rayon_overhead_ms`. Added
  `lower_bound_ms` and `fwd_rayon_overhead_ms`. Removed `forward_sample_ms`,
  `backward_cut_ms` and `io_write_ms`.

### Fixed

- Results no longer depend on MPI rank or thread count: lower bounds are
  bit-identical across 1r/1t, 1r/2t, 2r/1t and 1r/4t, and LHS/QMC sampling is
  deterministic across MPI configurations.
- `overhead_ms` was always zero in the timing output.
- Historical windows match correctly for studies that do not start in January.

## [0.4.2] - 2026-04-10

### Added

- Execution topology (MPI library version, rank-to-host mapping, thread level,
  SLURM job metadata) is shown after the banner during `cobre run` and written
  to the metadata JSON.
- The HiGHS version appears in the `Execution` section, in `cobre version` and
  in the metadata JSON.

### Changed

- **BREAKING:** the metadata JSON `mpi` object is replaced by `distribution`
  with `backend`, `world_size`, `ranks_participated`, `num_nodes`,
  `threads_per_rank`, `mpi_library`, `mpi_standard`, `thread_level`,
  `slurm_job_id` and `solver_version`.
- Better backward-pass load balance across MPI ranks, no redundant LP setup in
  simulation, a parallel lower-bound evaluation, and cut storage that grows on
  demand.

### Fixed

- Multi-line MPICH library version output is shown as a single line.

## [0.4.1] - 2026-04-06

- No changes visible to CLI users.

## [0.4.0] - 2026-04-06

### Added

- Per-class scenario sampling: inflow, load and NCS each use `InSample`,
  `OutOfSample`, `Historical` or `External` through the `inflow`, `load` and
  `ncs` sub-objects, each with a `scheme` field, of `training.scenario_source`
  and `simulation.scenario_source` in `config.json`.
- Historical inflow sampling replays standardized noise from windows found in
  `inflow_history.parquet`. The top-level `historical_years` of
  `scenario_source` takes a list (`[2010, 2015, 2020]`) or a range
  (`{from: 2010, to: 2023}`).
- External scenarios are read from `external_inflow_scenarios.parquet`,
  `external_load_scenarios.parquet` and `external_ncs_scenarios.parquet`.
- `noise_method` in `config.json` accepts `InSample`, `LatinHypercube`,
  `QmcSobol` and `QmcHalton`.
- Correlation matrices are estimated per season when enough paired
  observations exist, with the pooled matrix as fallback.
- Correlation `entity_type` accepts `"inflow"`, `"load"` and `"ncs"`. A group
  with mixed entity types is rejected at parse time.
- `stochastic_provenance.json` records PAR fitting diagnostics, correlation
  estimation metadata and sampler configuration.

### Changed

- **BREAKING:** `training.seed` is renamed `training.tree_seed`, with no alias.
  Update old configs.
- **BREAKING:** `scenario_source` moves from `stages.json` to
  `training.scenario_source` and `simulation.scenario_source` in `config.json`;
  `stages.json` with the old location fails with a migration error.
  `simulation.scenario_source` falls back to `training.scenario_source` when
  absent.
- **BREAKING:** `external_scenarios.parquet` is replaced by the three per-class
  files above.
- Correlation matrices are factored by spectral decomposition, so they no
  longer need to be positive definite and degenerate hydros stay in the
  estimation. The `method` in `correlation.json` defaults to `"spectral"`;
  `"cholesky"` is still accepted.
- Training and simulation output directories write `metadata.json` with
  timing, iteration counts and completion status. The retry histogram moves to
  a separate Parquet file.

### Removed

- **BREAKING:** the flat `sampling_scheme` string in `scenario_source` fails
  with a parse error pointing to the per-class format.
- **BREAKING:** `selection_mode` is removed from `scenario_source`, with no
  replacement. External scenarios are selected by scenario index.
- **BREAKING:** the `seed` alias of `tree_seed` is removed. Use
  `training.tree_seed`.
- **BREAKING:** `simulation.sampling_scheme` is removed. Use
  `simulation.scenario_source`.

### Fixed

- Multi-rank training is reproducible: NCS and load factors and `forward_seed`
  reach non-root ranks, and training stats and simulation costs are aggregated
  across ranks.
- Wrong LP column mapping for cut lag coefficients in the backward pass.
- Forward periodic Yule-Walker assembly could give wrong PAR coefficients for
  multi-season models.

## [0.3.2] - 2026-03-30

### Added

- `"domination"` method for `training.cut_selection`: deactivates cuts
  dominated at every visited trial point. Configure it with `threshold` and
  `check_frequency`.
- `exports.states` (default `false`) writes the visited states to
  `policy/states/stage_NNN.bin`.
- `total_visited_states` in the policy `metadata.json`.

### Changed

- The policy checkpoint holds all cuts, active and inactive, with an
  `is_active` flag per cut and an `active_cut_indices` vector. It used to hold
  only active cuts.

### Fixed

- The lower bound applied no inflow truncation to stage-0 openings, giving
  optimistic bounds with `inflow_non_negativity.method` set to `"truncation"`
  or `"truncation_with_penalty"`.

## [0.3.1] - 2026-03-30

### Added

- The annual discount rate of the policy graph now applies: it scales the
  future cost and weights the stagewise costs in the training upper bound and
  the simulation cost.

### Fixed

- With a non-zero discount rate, the upper bound summed undiscounted stage
  costs and did not match the discounted lower bound, and the immediate cost
  extracted at each stage was wrong.

## [0.3.0] - 2026-03-30

### Added

- Policy warm-start and resume from a checkpoint, set with `policy.mode`:
  `"fresh"`, `"warm_start"` or `"resume"`.
- Simulation-only mode against a saved policy: `training.enabled = false` with
  a valid policy.
- `modeling.inflow_non_negativity.method = "truncation_with_penalty"`.
- The inflow non-negativity penalty can be overridden per hydro through the
  hydro overrides in `penalties.json` (`inflow_nonnegativity_cost`).
- Withdrawal and evaporation violation slacks split into positive and negative
  parts with independent costs.
- Min/max outflow, turbined flow and generation constraints have per-block
  slacks with independent penalty costs.
- Simulation output adds the violation cost columns `outflow_violation_below_cost`,
  `outflow_violation_above_cost`, `turbined_violation_cost`,
  `generation_violation_cost`, `evaporation_violation_cost` and
  `withdrawal_violation_cost`, next to `hydro_violation_cost`.
- `hydro_production_models.json` can override the generation model of a hydro
  at specific stages.

### Changed

- An invalid `policy.mode` (for example `"warmstart"`) is rejected with an
  error listing the valid values. It used to fall back to fresh training.

### Fixed

- Constant-productivity hydros use the per-stage production model, so
  `hydro_production_models` overrides reach the load-balance coefficients.
- Withdrawal violation costs in the simulation output were understated.
- PAR(p) estimation handles pre-study stages.

## [0.2.2] - 2026-03-27

### Changed

- MPI binaries no longer carry an RPATH, for HPC cluster compatibility.

### Fixed

- Stuck LP solves no longer hang. The retry sequence has iteration limits
  (simplex `max(100K, 50 × num_cols)`, IPM 10K) and wall-clock budgets (15s/30s
  per level, 120s overall), and `ITERATION_LIMIT` and `TIME_LIMIT` from the
  initial solve are retried.

## [0.2.1] - 2026-03-26

### Fixed

- Cut selection counted unpopulated cut slots as deactivated, which inflated
  `cuts_deactivated` and understated `cuts_active` in the convergence output.

## [0.2.0] - 2026-03-26

### Added

- Per-stage cut selection statistics (cuts populated, active before and after,
  deactivated) are written to `training/cut_selection/iterations.parquet`.
- `simplex_strategy` selects the HiGHS strategy: 0=auto, 1=dual, 4=primal.
- Solver statistics add the timing columns `cut_sync_ms`, `state_exchange_ms`,
  `cut_batch_build_ms` and `rayon_overhead_ms`.

### Changed

- Faster backward pass: about 29.5% fewer nonzeros in cuts and about 95% less
  basis installation overhead.

### Fixed

- Multi-rank MPI runs synchronized cuts in the wrong place in the backward
  pass.
- Cut selection events did not reach the Parquet output.

## [0.1.11] - 2026-03-23

### Added

- `load_model_time_ms`, `add_rows_time_ms` and `set_bounds_time_ms` columns in
  `training/solver/iterations.parquet` time the LP setup separately from the
  solve.
- Less LP rebuild overhead: the model persists across scenarios of a stage and
  cuts are appended incrementally.
- Simulation LPs warm-start from the per-stage basis of the training
  checkpoint.
- Evaporation flow is bounded above by a physical estimate with a 2x safety
  margin. The over-evaporation slack costs 100x the under-evaporation cost.
- The simulation cost breakdown reports `inflow_penalty_cost`,
  `hydro_violation_cost` (evaporation and withdrawal violations) and diversion
  cost.

### Fixed

- Per-entity simulation costs were inflated by the column scale factor when LP
  prescaling was active.
- NCS curtailment cost uses `curtailment_mw` (available minus generation)
  instead of `generation_mw`.
- HiGHS internal scaling is off by default (`simplex_scale_strategy = 0`),
  because it interfered with basis reuse and dual extraction.

## [0.1.10] - 2026-03-23

### Fixed

- Simulation water values read the wrong duals and were null or wrong for all
  hydros.
- PAR auto-estimation produced one model per season instead of one per stage,
  so later stages of each season had no inflow AR coefficients.
- PAR(p) lag variables now propagate across stages.
- Inflow in the simulation output is the realized inflow, not a stale
  lag-derived value.

## [0.1.9] - 2026-03-22

### Added

- PAR estimation uses periodic Yule-Walker coefficients and PACF-based order
  selection instead of AIC, with a negative `phi_1` rejection and iterative
  PACF order reduction.
- LP row scaling with RHS prescaling and dual unscaling, and objective cost
  scaling (`COST_SCALE_FACTOR = 1000`), improve conditioning on large systems.
- `solver_stats/scaling_report.json` and `solver_stats/solver_stats.parquet`.
  The CLI shows per-solve timing, basis reuse and simplex iteration counts for
  training and simulation.
- The simulation summary shows per-scenario cost and LP solve metrics.

## [0.1.8] - 2026-03-21

### Added

- `line_exchange(id)` generic-constraint term: net line flow (direct minus
  reverse). The line id must exist.
- `productivity_override` on stage ranges and seasons of
  `hydro_production_models.json` replaces `productivity_mw_per_m3s` for the
  covered stages. It must be positive and is rejected on FPHA stages.

## [0.1.7] - 2026-03-21

### Added

- Block factors that scale load demand (`scenarios/load_factors.json`), line
  capacity (`constraints/exchange_factors.json`) and non-controllable source
  availability (`scenarios/non_controllable_factors.json`). They default to 1.0
  and have validation rules 36-41.
- Stochastic availability of non-controllable sources (wind, solar,
  run-of-river) from `scenarios/non_controllable_stats.parquet`: a per-stage
  mean and standard deviation availability factor (0-1), drawn from a normal
  distribution clamped to [0, 1] and multiplied by `max_generation_mw` and the
  block factors. The policy hedges against it.

### Changed

- Non-controllable sources are no longer stubs: they have LP generation
  variables, stochastic availability, simulation output and validation rules.

## [0.1.6] - 2026-03-19

### Added

- Generic constraints in `constraints/generic_constraints.json`, with
  stage-varying bounds from `constraints/generic_constraint_bounds.parquet`.
  They support 20 variable types, optional slack variables with
  per-constraint penalties, and the senses `<=`, `>=` and `==`. Duals, slacks
  and violation costs are extracted in training and simulation, and violations
  are written as Hive-partitioned Parquet.
- Water withdrawal for hydros, with bounds and violation penalties. Slack and
  violation cost appear in the simulation output.
- Validation rules 33-35: entity ids in constraint expressions must exist,
  block ids must be valid for the referenced stages, and duplicate bounds keys
  are rejected.

## [0.1.5] - 2026-03-18

### Added

- Multi-segment deficit pricing: N deficit columns per bus per block, one per
  segment, with capacity constraints. Cases with tiered deficit costs give
  correct results.
- `past_inflows` in `initial_conditions.json` initializes PAR(p) lags at stage
  0, most recent first per hydro, instead of zeros.
- Validation rules 22-24, when `inflow_lags: true` and the PAR order is above
  0: `past_inflows` entries must be non-empty, each hydro needs at least the
  PAR order in values, and every hydro id must exist in the registry.

## [0.1.4] - 2026-03-17

### Added

- FPHA hydro production model, a four-piece hyperplane approximation for
  variable-head plants. Hyperplanes come from `fpha_hyperplanes.parquet` or are
  computed from the forebay volume-elevation and tailrace flow-elevation curves
  in `hydro_geometry.parquet`.
- Evaporation linearization as a function of stored volume, with per-season
  reference volumes.
- The resolved hydro model parameters (FPHA hyperplanes, evaporation
  coefficients) are written to Parquet.
- Simulation results report FPHA production per segment, the active
  hyperplane and evaporation volumes.

## [0.1.3] - 2026-03-15

### Added

- Post-setup stochastic summary of the fitted PAR models, replacing the
  `[stochastic]` lines. AR orders show per order for up to 10 hydros, as a
  range for 11-30 and as a histogram for 31 or more.
- `scenarios/noise_openings.parquet`, when present, is loaded, validated and
  used as the backward-pass opening tree. The exported
  `output/stochastic/noise_openings.parquet` has the same schema.
- Up to six stochastic artifact files are written to `output/stochastic/`:
  fitted seasonal statistics, AR coefficients, correlation matrix, fitting
  report, noise openings and load seasonal statistics. `exports.stochastic` in
  `config.json` controls it.

### Removed

- **BREAKING:** the CLI flags `--skip-simulation` (use `simulation.enabled` in
  `config.json`), `--export-stochastic` (use `exports.stochastic`),
  `--no-banner` and `--verbose`.

### Fixed

- The simulation progress bar showed wrong mean and standard deviation.

## [0.1.2] - 2026-03-14

### Fixed

- The upper bound uses compensated (Kahan) summation, so it is bit-for-bit
  identical regardless of MPI rank count and scenario distribution.
- Off-by-one edge cases in stopping rules.

## [0.1.1] - 2026-03-12

### Added

- Scenario generation from fitted PAR(p) models in simulation.
- `Truncation` inflow non-negativity method, which clamps negative PAR draws
  to zero.
- Correlated Gaussian load noise, using the same framework as inflow noise.
- PAR(p) coefficient estimation from the historical inflow records in the case
  directory.
- `cobre summary` subcommand: prints convergence statistics and output file
  locations.

## [0.1.0] - 2026-03-09

### Added

- Entity types Bus, Line, Thermal, Hydro, Contract, PumpingStation and
  NonControllable, with topology validation and three-tier penalty resolution.
- Case loader with five-layer validation and JSON/Parquet parsing for 33 input
  types.
- SDDP training with Benders cuts, stopping rules and convergence monitoring,
  on the HiGHS solver with warm starts. PAR(p) preprocessing, correlated
  noise, opening trees and `InSample` sampling.
- MPI support through ferrompi.
- Simulation with MPI aggregation, Hive-partitioned Parquet output and a
  FlatBuffers policy checkpoint.
- CLI subcommands `run`, `validate`, `report` and `version`, with progress
  bars and exit codes.

## [0.0.1] - 2026-02-23

### Added

- Multi-platform binary distribution through cargo-dist.

<!-- next-url -->

[Unreleased]: https://github.com/cobre-rs/cobre/compare/v0.18.0...HEAD
[0.18.0]: https://github.com/cobre-rs/cobre/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/cobre-rs/cobre/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/cobre-rs/cobre/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/cobre-rs/cobre/compare/v0.14.3...v0.15.0
[0.14.3]: https://github.com/cobre-rs/cobre/compare/v0.14.2...v0.14.3
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
