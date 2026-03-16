# Code Reviewer Memory

## Project: cobre (Rust SDDP solver ecosystem)

### Key architecture facts

- `System` struct has both `pub` fields AND `pub fn` accessor methods — this dual API is intentional
  (fields for `PartialEq`/`Debug`, methods for slice access). But public fields break encapsulation.
- `SystemBuilder::build()` sorts entities by `id.0`, then checks duplicates via `windows(2)`,
  then validates cross-refs. Error collection is non-short-circuiting per design (DEC-005 style).
- `HasId` trait is private; `check_duplicates` and `build_index` are private fns.
- `CascadeTopology::build()` uses Kahn's algorithm with a `BinaryHeap<Reverse<i32>>` as the min-heap
  ready queue. Push/pop are O(log n). Changed from a sorted `Vec<i32>` (which had O(n) `remove(0)`)
  in commit d5a8c3b. Determinism is preserved: min-heap pops smallest ID first within a topo level.
- `NetworkTopology` uses two `static OnceLock<T>` globals as empty-collection sentinels to
  return `&T` (not `Option<&T>`) from accessor methods.

### Recurring patterns to watch in this codebase

- `#[allow(unused_variables)]` at function level to suppress pre-declared Phase N+1 params is too broad.
  The correct idiom is underscore-prefix: `_param_name`. Function-level allow masks future accidental
  unused locals introduced inside the body. Found in `backward.rs:103` (Phase 7 pre-declarations).
- `vec![...]` in test code triggers `clippy::useless_vec` when the vec doesn't need heap allocation.
  CI runs clippy with `-D warnings`, so this silently breaks CI while `cargo test` still passes.
  Always check with `cargo clippy --all-targets --all-features -- -D warnings`. Found in `types.rs:414`.
- `Option<usize>` comparisons in tests can silently pass when `None < Some(x)` in Rust's `Ord`.
  Always `.expect()` before comparing positions. Found in `system.rs:1058-1060`.
- `windows(2)` duplicate detection reports N-1 errors for N identical IDs (a triplicate gets 2 errors).
  This is technically correct but may surprise users. Confirm intentional before flagging.
- `pub` fields on `System` allow mutation of entity collections bypassing index invariants. This is
  intentional for Phase 1 (builder is the only constructor; `cobre-io` will own construction) but
  worth noting when reviewing Phase 2.

### Penalty resolution architecture

- `penalty.rs` exports standalone `resolve_*` free functions that are pub but NOT called
  inside `SystemBuilder::build()`. They are Phase 2 utilities: `cobre-io` will call them
  when constructing entities from JSON. `SystemBuilder` is a Phase 1 test-only API that
  accepts already-resolved entities directly (e.g. `Hydro::penalties` field is always set).
- `ValidationError` has two variants that are NEVER constructed in Phase 1:
  `DisconnectedBus` and `InvalidPenalty`. Both have TODO/Phase-2 deferred comments.
  These are pre-declared for Phase 2. Don't flag as dead code.

### Phase 2 cobre-io key facts

- `Stage.index` (0-based positional) ≠ `Stage.id` (user-facing domain ID, can be non-zero/non-contiguous).
  `Stage.index` is documented as the key for penalty/bounds array indexing. Override rows carry `stage_id`.
- `resolve_penalties` and `resolve_bounds` treat `stage_id` from override rows as positional `stage_idx`
  via `usize::try_from(row.stage_id)`. This is correct ONLY if stage IDs happen to equal their 0-based index.
  This is the most important correctness risk identified in Phase 2.
- `n_stages` in pipeline.rs counts only study stages (id >= 0); pre-study stages excluded.
- `dead_code` allowances on `config`, `penalties`, `load_factors`, `exchange_factors` in `ParsedData` are
  intentional: these fields are parsed for Layer 2 validation but not forwarded to `System` yet (Phase 3+).
- `ValidationContext::into_result()` silently discards warnings — this is explicitly documented behavior.
- `ValidationContext.errors()` returns `Vec<&ValidationEntry>` (heap-allocated filter), not a slice.
  This is slightly inefficient but clean. Called at most once per file parse in validate_schema.
- `#[allow(clippy::struct_excessive_bools)]` on `FileManifest` is necessary and correct.
- `unreachable!` in pipeline.rs line 52 is actually sound — it maps the `Ok(())` branch of
  `into_result()` when validate_schema returned `None` (no errors added), which logically cannot happen.
  But in practice it CAN be reached if no new errors were added by validate_schema (pre-existing errors
  from Layer 1 caused validate_schema to return None). This is a logic flaw — see FINDING.

### Phase 3 cobre-solver key facts (post solver-hot-path-refactor)

- `unsafe_code = "allow"` is set at crate level in `Cargo.toml` (overriding workspace `forbid`)
  because FFI to HiGHS requires unsafe blocks. All other workspace lints are manually replicated.
- `HighsSolver` holds a raw `*mut c_void` handle; `Send` is manually impl'd (not `Sync`).
- `ffi.rs` uses `#![allow(dead_code)]` intentionally — it's a sys-crate-style bindings file.
- `BasisStatus` enum and `Basis` (old enum-based) are REMOVED. `RawBasis` was renamed to `Basis`
  (raw i32 codes). `SolverInterface` now has: `solve()→SolutionView`, `get_basis(&mut Basis)`,
  `solve_with_basis(&Basis)→SolutionView`. The old `solve_view`, `get_raw_basis`,
  `solve_with_raw_basis_view`, `solve_with_raw_basis` names are gone.
- `SolverError` variants simplified: `Infeasible` and `Unbounded` are now unit variants.
  `NumericalDifficulty`, `TimeLimitExceeded`, `IterationLimit` lost their `partial_solution` field.
  All callers treat these as training termination (propagate as `SddpError::Solver`).
- `try_extract_partial_solution` and `make_infeasible_error`/`make_unbounded_error` helper methods
  in `HighsSolver` are removed. Ray/direction certificate extraction is still done for
  `UNBOUNDED_OR_INFEASIBLE` classification, but the ray values are discarded (only classification matters).
- `basis_col_i32` and `basis_row_i32` are pre-allocated internal buffers in `HighsSolver` for
  zero-copy basis injection in `solve_with_basis`. Resized in `load_model`/`add_rows`.
- `get_basis` takes `&mut Basis` (out-param). Resizes `out.col_status` and `out.row_status` to match
  current LP dimensions before calling `cobre_highs_get_basis`.
- `solve_with_basis` has a hard assert (not debug_assert) on `basis.col_status.len() == self.num_cols`.
  Row mismatch is handled silently: extra rows filled with `HIGHS_BASIS_STATUS_BASIC` (1).
- `cobre_highs_get_solution` return is silently discarded in `extract_solution_view`. Called after
  OPTIMAL is confirmed.
- `restore_default_settings` is private and called unconditionally after retry loop.
- The retry loop (5 levels: clear solver, enable presolve, primal simplex, relax tolerances, IPM)
  only activates on `SOLVE_ERROR` or `UNKNOWN` model status. No test exercises it directly.
- `test-support` feature re-exports raw FFI option-setters for integration tests that need
  time/iteration limits (e.g., `test_solver_highs_solve_time_limit`).
- SAFETY comment in `reset()` says "model is still loaded" (true for HiGHS) but code sets
  `has_model = false`. This is correct behavior but the comment is potentially misleading.

### Phase 4 cobre-comm key facts

- `FerrompiBackend::allreduce` does NOT validate `send.len() == recv.len()` or `send.len() > 0` before calling MPI.
  ferrompi internally validates `send.len() != recv.len()` and returns `Error::InvalidBuffer`, which maps to
  `CommError::InvalidBufferSize { expected: 0, actual: 0 }` (sentinel). The empty-buffer case is NOT validated.
- `FerrompiBackend::allgatherv` does NOT validate `counts.len() == self.size` or `send.len() == counts[self.rank()]`.
  ferrompi's `allgatherv` passes directly to MPI FFI without validating count array lengths. If `counts.len() < size`,
  MPI reads past the end of the counts array (UB).
- `BackendKind` is pub-exported but `create_communicator` doesn't accept it as a parameter — intentional forward-
  declaration for ticket-011. Same pattern as Phase 1/2 pre-declarations.
- `CommBackend` enum (feature-gated) has no `#[derive(Debug)]` — `missing_debug_implementations` is not enforced.
- `FerrompiBackend::new()` calls `world.split_shared()` → stored in `self.shared`. `split_local()` calls
  `world.split_shared()` AGAIN for each invocation — a new collective per call. `is_leader()` uses the cached
  `self.shared` from `new()`. This is consistent (same deterministic split result) but slightly wasteful.
- Unit test `ENV_LOCK` in `factory.rs` and integration test `ENV_LOCK` in `factory_tests.rs` are separate mutexes
  in separate binaries — they do not serialize across binary boundaries. Safe in default `cargo test` (sequential
  binary execution) but not with concurrent harnesses like nextest.
- `LocalBackend` uses `saturating_add` for recv-length validation but plain `+` for the actual slice index on line 95.
  Theoretical inconsistency; not reachable on any real system (no allocation could overflow usize).

### Phase 5 cobre-stochastic key facts

- `StochasticContext` bundles `PrecomputedParLp`, `DecomposedCorrelation`, `OpeningTree`, `base_seed`, `dim`.
  All fields immutable after construction; consumed read-only by the optimization loop.
- `build_stochastic_context` calls `validate_par_parameters` first (fatal check + warnings), then
  `PrecomputedParLp::build`. The `ParValidationReport` returned from the first call is silently discarded
  (`let _report = ...`). The report's warning list is never exposed. This is a design gap — callers cannot
  see low-residual-variance warnings. If warning propagation is needed in Phase 6, refactor to return the report.
- `PrecomputedParLp::build` uses stage-major layout: `array[stage * n_hydros + hydro]`. The 3-D `psi` array
  uses `psi[stage * n_hydros * max_order + hydro * max_order + lag]`. All padded with 0.0.
- `CholeskyFactor` uses packed lower-triangular storage: element (i, j) at index `i*(i+1)/2 + j`.
  Symmetry tolerance is 1e-10. PD check: diagonal element <= 0 during Cholesky → error.
- `DecomposedCorrelation` uses `BTreeMap<String, Vec<GroupFactor>>` (deterministic iteration order).
  `resolve_positions` must be called once before `apply_correlation` to pre-compute entity indices.
  `apply_group_precomputed` uses stack-allocated buffers (128 bytes × 2) for groups ≤ 64 entities.
- `derive_forward_seed` (20 bytes: base+iter+scenario+stage) and `derive_opening_seed` (16 bytes:
  base+opening+stage) use `SipHasher13::new()` with zero key. Domain separation relies on different
  message lengths — SipHash-1-3 incorporates length into state. Golden-value test pins output.
- `sample_forward` uses `rng.random::<u64>() % n` (modulo bias). For small `n` (branching factor),
  bias is negligible in practice but is a correctness defect. Fix: `rng.random_range(0..n)`.
- `StochasticError::SeedDerivationError` and `InsufficientData` are pre-declared but never constructed
  in Phase 5 production code. They lack deferred-phase comments (unlike Phase 1/2 pre-declarations).
- `OpeningTree::from_parts` is `pub(crate)` — only callable by `generate.rs` (tree generation module).
  Public API is `generate_opening_tree` (free function) and `OpeningTree` accessors.
- `infrastructure_genericity_no_sddp_references` test in `reproducibility.rs` spawns `grep` subprocess.
  Linux-only; works in CI but would fail on Windows. Acceptable for this codebase (Linux CI only).
- Broken intra-doc links in Phase 5: `[j]` in `cholesky.rs:22` and `[apply_correlation]` in
  `resolve.rs:178` produce rustdoc warnings. Fix: escape brackets in cholesky.rs, use
  `Self::apply_correlation` in resolve.rs.

### Phase 6 cobre-sddp key facts (post solver-hot-path-refactor)

- `train()` in `training.rs` allocates `basis_cache: Vec<Option<Basis>>` (length = num_stages) and
  passes `&mut basis_cache` to both `run_forward_pass` and `run_backward_pass` each iteration.
  The cache is NOT reset between forward and backward passes — both share it within an iteration.
- `run_forward_pass` and `run_backward_pass` warm-start by calling `solve_with_basis(cache[t])` if
  `Some`, else `solve()`. After each successful solve, `get_basis(&mut cache[t])` is called to
  update the cache. On solver error, `cache[t] = None` (invalidation).
- `run_backward_pass` has `config: &TrainingConfig` and `comm: &C` as unused Phase 7 pre-declarations.
  These are suppressed via `#[allow(unused_variables)]` at function level (line 103 of backward.rs).
  The correct idiom for unused pre-declared params is underscore-prefix (`_config`, `_comm`).
- `train()` in `training.rs` destructures `TrainingConfig` into `loop_config` with `event_sender: None`
  using struct update syntax (`..config`) after extracting `config_forward_passes`, `max_iterations`,
  `event_sender`. The old explicit field-by-field reconstruction is replaced.
- Backward pass inner opening loop allocates `Vec<f64>` per opening for `coefficients` despite the
  doc claiming "no allocations inside the inner opening loop." Also allocates `Vec<usize>` for
  `binding_slots` per opening when cuts are present. Both are confirmed hot-path allocations.
- `ConvergenceMonitor::update` clones `lower_bound_history` (O(iterations)) on every call.
  `MonitorState` owns the history; `BoundStalling` is the only consumer.
- `evaluate_lower_bound` uses non-debug `assert!` for `n_openings > 0` — fires in release builds
  on every iteration despite the opening tree being immutable. Should be `debug_assert!`.
- `TrainingEvent::TrainingStarted` is emitted with `case_name: String::new()` and `timestamp:
String::new()` — placeholder values because `TrainingConfig` has no mechanism to supply them.
- `FutureCostFunction::pools` is `pub Vec<CutPool>` — direct access is used extensively in
  `training.rs` and `backward.rs` for metadata updates and deactivation.
- Cut slot assignment formula: `slot = warm_start_count + iteration * forward_passes + forward_pass_index`.
  Deterministic, enables reproducibility and checkpointing.
- `CutSyncBuffers::sync_cuts` assumes all ranks generate the same number of cuts (uniform `counts`).
  Acknowledged in doc comment. `allgatherv` is called with the actual send slice length.
- `collect_local_cuts_for_stage` in `training.rs` causes a double-collect: returns `Vec<(u32, u32,
u32, f64, Vec<f64>)>` that is immediately re-mapped to `Vec<(u32, u32, u32, f64, &[f64])>`.
  Documented as acceptable on the backward/sync path (not inner LP loop).
- `StageIndexer` has `pub` fields used throughout; it is a plain data struct, no invariants to protect.
- `HorizonMode::Finite` is the only implemented variant; `Cyclic` is deferred.
- All 20 integration+conformance tests pass; 34 unit tests pass.

### Phase 8 cobre-cli key facts

- `cobre-cli` is the binary crate; it has `main.rs` as entry point. Subcommands in `src/commands/`.
- `CliError` has 4 variants → exit codes 1-4. `format_error` writes to `Term::stderr()` with ANSI color.
  `From` impls for `LoadError`, `OutputError`, `SddpError`, `SimulationError`.
- Banner/progress/summary are all fire-and-forget: `write_line` errors are silently discarded via `let _ = ...`.
- `run_progress_thread` spawns a background thread that consumes `TrainingEvent` via mpsc and drives
  indicatif progress bars. Returns `ProgressHandle`; `join()` returns all collected events.
  `ProgressHandle::join` uses `#[allow(clippy::expect_used)]` intentionally — thread panic = programming error.
- `format_summary_string` is gated `#[cfg(test)]` but `print_summary` duplicates the format strings for
  production use. This split was designed to enable pure-string unit tests but creates a maintenance burden.
- `validate.rs` prints entity counts twice: once as a summary line, then 4 individual indented lines.
  The individual lines are the grep-friendly output; the summary line is supplemental.
- `#![allow(dead_code)]` on `error.rs` is stale as of Phase 8 — all variants are used.
- Arrow/Parquet v55→v58 API change: `set_max_row_group_size` renamed to `set_max_row_group_row_count(Some(...))`.
  Applied in 3 writer files: `dictionary.rs`, `simulation_writer.rs`, `training_writer.rs`.
- `build_stage_templates` and `StageTemplates` newly re-exported from `cobre-sddp::lp_builder` in Phase 8.
- `DecomposedCorrelation::empty()` added for thermal-only systems (dim == 0 → skip Cholesky).
- Theta pinning fix in `forward.rs`: for terminal stages, `set_col_bounds(&[theta], &[0.0], &[0.0])` before
  solve to prevent unbounded LP when cut pool is empty at iteration 1.
- io_thread in `run.rs` simulate path: on `simulate()` error, the JoinHandle is dropped (thread detached).
  The io_thread self-terminates when `result_rx` gets disconnected. No hard resource leak, but unjoined.
- `config.simulation.io_channel_capacity` is `u32` in cobre-io; cast via `as usize` in run.rs (not via
  `usize::try_from`) — safe on all 64-bit platforms, minor stylistic inconsistency.

### Phase v0.1.1 stochastic-foundation key facts

- `evaluate_par_inflows` and `solve_par_noises` are new public exports from `cobre-stochastic::par::evaluate`.
  Infrastructure crate — confirmed zero SDDP references.
- Truncation path in `forward.rs` (training): lag state is transposed from hydro-major
  (`current_state[inflow_lags.start + h * max_par_order + l]`) to lag-major
  (`lag_matrix_buf[l * n_hydros + h]`) before calling `evaluate_par_inflows`. Correct transposition.
- Simulation pipeline (`simulation/pipeline.rs`) keeps `_inflow_method` as unused pre-declaration.
  Truncation is NOT applied in simulation — only in training forward pass. Pre-existing limitation.
- `SolverWorkspace::new` signature changed to include `max_par_order` parameter (public API break).
- `WelfordAccumulator` in the main-thread post-parallel loop (`acc` at line 443) accumulates costs
  but its values are never used. Dead code after refactor. The `#[allow(dead_code)]` on the struct's
  `impl` block documents intent for future `SimulationFinished` event enrichment.
- `simulate()` signature changed from `event_sender: Option<&Sender<T>>` to `Option<Sender<T>>` (owned).
  Workers clone the sender; the original is used for `SimulationFinished` after the parallel region.
- Python `run.rs`: `sim_event_tx` is cloned from `event_tx` before training. Both training and simulation
  events are sent to the same `event_rx` channel. After training, events are drained; after simulation,
  remaining simulation events in `event_rx` are dropped silently at function exit. No deadlock.
- `cobre summary` subcommand added: reads `training/_manifest.json` (required),
  `training/convergence.parquet` (optional, fallback to zeroed summary), and
  `simulation/_manifest.json` (optional). Prints the same summary as `cobre run`.
- `ConvergenceSummary` reads all batches and sums `lp_solves` + `time_total_ms`;
  takes bound/gap fields from the last row of the last batch.
- `read_optional_simulation_manifest` returns `Ok(None)` on `ErrorKind::NotFound`, propagates other errors.
- `SimulationProgress` stats (`mean_cost`, `std_cost`, `ci_95_half_width`) are per-worker
  (from `worker_acc`), not global. This is intentional — a global atomic accumulator across threads
  would require a Mutex. The progress bar shows per-worker running statistics, not the true global mean.
- `validate_semantic_stages_penalties_scenarios` doc comment says "All 18 rules" but the function
  now enforces 21 rules (17–21 added in v0.1.1). Comment was not updated.
- `estimate_correlation` in `fitting.rs` line 1075: `pairs.sort_unstable_by(partial_cmp)` sorts by
  residual value, not by date. Comment says "Sort by date" but is wrong. Since Pearson correlation
  is sum-based (order-invariant), this doesn't affect correctness but is misleading.
- `find_season_for_date` is duplicated: identical private fn in both `fitting.rs` and `estimation.rs`.
- `hydro_ids.contains` in `estimate_ar_coefficients` and `estimate_ar_with_aic` is O(n) per observation.
  Not on the training hot path, but scales poorly for very large observation sets.
- `active_load_patches` in `PatchBuffer` is never reset to 0. After `fill_load_patches` is called
  for a stage with `n_blocks`, the value persists until the next `fill_load_patches` call. This is
  safe because `forward_patch_count()` is always called AFTER `fill_load_patches` in all code paths.

### v0.1.1 review findings (confidence >= 75, confirmed)

- **CRITICAL BUG**: `estimate_ar_coefficients` (fitting.rs ~line 722) does NOT clamp autocorrelations
  to `[-1.0, 1.0]` before passing to `levinson_durbin`. The AIC path helper `compute_autocorrelations`
  (line 396) does clamp. When `|rho| > 1` (possible from floating-point rounding in cross-covariance),
  `sigma2_next = sigma2 * (1 - kappa^2)` goes negative. The near-singularity guard fires on
  `sigma2 <= f64::EPSILON` (positive threshold only), so negative sigma2 passes through and garbage
  coefficients propagate into the PAR model. Fix: add `.clamp(-1.0, 1.0)` on line 722.
- **ERROR HANDLING GAP**: `estimate_from_history` (estimation.rs lines 89-90) calls `validate_structure`
  into a `ValidationContext` that is never checked. Structural errors are silently discarded. If the
  partial manifest has `scenarios_inflow_history_parquet == true` despite errors, estimation proceeds
  on invalid data. Fix: check `ctx.into_result()` before using the manifest.
- **STANDARDS**: `PrecomputedNormalLp::mean`, `std`, and `block_factor` accessors use `assert!` (not
  `debug_assert!`) for bounds checking. Called in forward/backward pass hot paths. Pattern is inconsistent
  with all other hot-path precondition checks in this codebase. Fix: change to `debug_assert!`.
- **AIC BIAS**: `select_order_aic` in `estimate_ar_with_aic` (estimation.rs line 319) is called with
  `n_obs = pair_obs.len()` (all season observations) but autocorrelations are computed from a smaller
  effective sample (observations with lag predecessors). AIC penalty is underweighted, biasing toward
  higher-order models. Fix: use `pair_obs.len().saturating_sub(actual_order)` as approximation.

### v0.1.3 stochastic-io plan key facts

See [v013_review_facts.md](v013_review_facts.md) for details on noise_openings parser/validator/assembler,
BroadcastOpeningTree MPI correctness, EstimationReport -> FittingReport conversion, ScenarioData.noise_openings
design, and EntityId Ord derive. Key confirmed finding: "backward-pass" in `cobre-io/src/scenarios/noise_openings.rs`
line 1 violates the infrastructure genericity rule.

### FPHA + evaporation plan key facts

See [v014_fpha_evaporation_review.md](v014_fpha_evaporation_review.md) for full details.
Critical bug: `setup.indexer()` is built with empty FPHA/evap indices, so simulation
extraction uses wrong (zero) `generation_mw` for FPHA hydros. MPI concern: non-root ranks
re-run `prepare_hydro_models` independently (no broadcast); shared-filesystem assumption.

### FPHA hyperplane fitting plan key facts (feat/fpha-fitting)

See [project_fpha_fitting_phase.md](project_fpha_fitting_phase.md) for full details.
Key design decisions: `validate_computed_prerequisites` intentionally requires all 3 optional models
(tailrace, hydraulic_losses, efficiency) even though the math supports None for each.
`select_planes` has an undocumented early-break that can return > max_planes_per_hydro planes when no
valid removal exists — doc says "at most N" but implementation can exceed N. `eprintln!` for kappa < 0.95
is intentional (no event system in preprocessing). Grid formula duplicated in 4 functions.

### Review workflow notes

- Run `cargo clippy --package cobre-core` and `cargo test --package cobre-core` to baseline.
- This project uses workspace lints: `unsafe_code=forbid`, `unwrap_used=deny`, `missing_docs=warn`,
  `clippy::pedantic`. Clippy clean = confirmed by running.
- All entities are sorted by `id.0` (inner `i32`) not by `EntityId` itself (no `Ord` on `EntityId`).
- `EntityId.0` is intentionally `pub` — used throughout tests and sorting code (`sort_by_key(|e| e.id.0)`).
- `CascadeTopology::build()` is public; callers are responsible for providing sorted input.
  `SystemBuilder` always sorts before calling it. Unit tests calling it directly can pass
  unsorted input but the output remains deterministic because Kahn's ready queue is always sorted.
