---
name: v0.1.3 stochastic-io review facts
description: Key architecture and correctness facts learned during v0.1.3-stochastic-io code review
type: project
---

### v0.1.3 stochastic-io plan key facts

**Git range**: d582984..HEAD (branch feat/v0.1.3-stochastic-io)
**Commits**: 6 commits (epics 02-07), 54 files changed

#### noise_openings parser/validator/assembler (cobre-io)

- `validate_noise_openings` openings check is partially tautological when called from `load_user_opening_tree`:
  the `expected_openings_per_stage` slice is derived from the same rows being validated. Still meaningful
  for gap detection (ensures openings form a contiguous range 0..n). The count check itself is tautological.
- `validate_noise_openings` doc says "Panics if expected_count > u32::MAX" but code uses `unwrap_or(0)`.
  Actual behavior on overflow: `expected_set` becomes empty, any non-empty `opening_set` triggers a
  `SchemaError` ("missing opening indices"). Safe direction (false negative turned false positive) but doc is wrong.
- `assemble_opening_tree` counts openings by boundary-crossing on sorted rows. Relies on prior
  `validate_noise_openings` call to guarantee opening contiguity.

#### BroadcastOpeningTree MPI correctness

- `broadcast_value` is called with `T = Option<BroadcastOpeningTree>` and `value: Option<Option<BroadcastOpeningTree>>`.
  `postcard` encodes `Option::None` as a single tag byte (not zero bytes), so `len != 0` holds for the "no user tree"
  case. The "len==0 = broadcast failure" guard does NOT misfire. Broadcast is correct.
- `raw_opening_tree` wrapping on root: `Some(None)` = file absent, `Some(Some(bt))` = file present.
  Non-root ranks start with `None`; after `broadcast_value` they hold the deserialized `T` from root.

#### EstimationReport -> FittingReport conversion

- `estimation_report_to_fitting_report` maps `EntityId` keys (numeric `BTreeMap` order, i.e. i32 ascending)
  to `String` keys stored in `BTreeMap<String>` (lexicographic). "10" < "2" lexicographically — output JSON
  has different key order than the source entries for entity IDs > 9. `FittingReport` is write-only diagnostic;
  no round-trip impact, but the doc saying "ascending sort order" is technically lexicographic, not numeric.
- `build_estimation_report` uses season 0 as the representative season for `aic_scores`. If season 0 was
  skipped during estimation (zero std or insufficient observations), `aic_scores` is silently empty `Vec<f64>`.
  This is documented in the function comment. Cosmetic gap in diagnostic output.

#### Infrastructure genericity violation (CONFIRMED finding)

- `crates/cobre-io/src/scenarios/noise_openings.rs` line 1 module doc: "user-supplied **backward-pass** noise
  realisations". "backward-pass" is an SDDP algorithm term. CLAUDE.md requires infrastructure crates
  (cobre-io, cobre-stochastic) to use only generic language. Fix: replace with "iterative optimization pass"
  or "backward-phase" is also algorithm-specific — use "optimization pass" or "solver-defined evaluation pass".

#### ScenarioData.noise_openings field

- Loaded by `load_scenarios` (via `load_noise_openings`) into `ScenarioData.noise_openings`.
  NOT consumed by `load_case` in `pipeline.rs` — intentional design. The actual tree assembly happens
  in `load_user_opening_tree` (cobre-cli), which bypasses `ScenarioData` and calls the parser directly.
  `ScenarioData.noise_openings` exists as a carrier for potential future validation use.

#### EntityId now has Ord + PartialOrd derives

- `crates/cobre-core/src/entity_id.rs`: `EntityId` now derives `PartialOrd, Ord` (by inner `i32`).
  Enables `BTreeMap<EntityId, _>` in `EstimationReport.entries`. Semantically correct: ordering by entity ID value.
  Previously: all code sorted via `entity.id.0` comparisons. With this change, `EntityId` itself is `Ord`.

#### Stochastic diagnostics (cobre-cli)

- `format_stochastic_diagnostics` infers `n_seasons` from `inflow_models` filtered to `hydros[0].id`.
  If the system has 0 hydros but `n_hydros == 0` guard fires first, so the filter is never reached on empty.
  If the system has hydros but `inflow_models` is empty (no estimation ran), the filter returns 0 seasons.
  This is correct but worth noting for the "estimation branch" path.
- Export warnings (`eprintln!`) bypass the `quiet` flag by design — errors always surface.
  This matches the pre-existing pattern for other non-fatal warnings in run.rs.

**Why:** Captured during code review of v0.1.3-stochastic-io to avoid re-investigating the same questions.
**How to apply:** When reviewing follow-on Phase v0.1.3 work or any PR touching noise_openings, ScenarioData,
or EstimationReport conversion, use these facts as baseline. The genericity violation (finding F1) was confirmed
but is NOT in the v013_review_facts.md list because it belongs to the review report, not persistent facts.

---

### v0.1.3 plan v013-assessment review facts (Epic-01 through Epic-03 + bugfixes)

**Git range**: 6a4b8d1..HEAD (branch feat/v0.1.3-stochastic-io)
**Commits**: 8 commits (3 epics + 2 fixes + cleanup)

#### BasisStore indexing fix (backward.rs)

- `BasisStore::new(max_local_fwd, num_stages)` is sized with LOCAL scenario count (0..max_local_fwd).
  `BasisStore::get(scenario, stage)` uses `scenario` directly as the row index.
  The old code passed `scenario = spec.fwd_offset + m` (global index) — OOB panic on MPI rank > 0.
  Fix: `succ.basis_store.get(m, s)` using local index `m`. Confirmed correct.
- `BasisStoreSliceMut::get` takes an ABSOLUTE `scenario` index and subtracts `self.scenario_offset`.
  `SuccessorSpec.basis_store` is `&BasisStore` (not `BasisStoreSliceMut`), so no offset subtraction applies.

#### SimulationProgress event redesign

- `SimulationProgress` event now carries `scenario_cost: f64` (single scenario cost) instead of
  `mean_cost`, `std_cost`, `ci_95_half_width`. Statistics accumulation moved to progress thread
  via `WelfordAccumulator`.
- `scenarios_complete` in the event is the RANK-LOCAL count (from `AtomicU32::new(0)` per simulate() call).
  Progress thread only exists on rank 0 (`quiet = !is_root` for non-root ranks).
  The `scenarios_complete >= 2` condition in `progress.rs` equals `acc.count >= 2` in practice
  (no cross-rank event routing). NOT a bug.

#### StudySetup ownership model

- `StudySetup::new` builds all precomputed state (templates, indexer, FCF, initial state, horizon, risk measures,
  entity counts). `StudySetup::from_broadcast_params` does the expensive work; `new` extracts scalars from
  `cobre_io::Config` then delegates.
- `event_sender: Option<Sender<TrainingEvent>>` is moved into `TrainingConfig` inside `StudySetup::train()`.
  After `train()` returns, `event_tx` (the original) is consumed — correct lifecycle.
- In quiet mode (CLI), `sim_event_rx` is dropped before `setup.simulate(Some(sim_event_tx))`.
  `emit_sim_progress` uses `let _ = s.send(...)` — send on disconnected channel is silently discarded. Safe.

#### Unconditional barrier after conditional export

- Old code: `comm.barrier()` was inside `if bcast_config.export_stochastic { ... }`.
  New code: barrier is always called unconditionally after the conditional export block.
  This adds a no-cost barrier on `LocalBackend` and an extra collective on MPI when no export happens.
  Behaviour is correct (barrier is safe to call anytime). Minor unnecessary synchronisation on non-export runs.

#### WelfordAccumulator moved to cobre-core

- `WelfordAccumulator` is now `pub` in `cobre_core::welford`, re-exported from `cobre_core::WelfordAccumulator`.
- No `#[derive(Debug)]` on `WelfordAccumulator`. Workspace has `clippy::pedantic = warn` which includes
  `missing_debug_implementations`. All other public structs in cobre-core have `Debug`. Will fire as clippy warning.
- `Default` impl added correctly; `#[must_use]` added to `new()` and all query methods. Good.

#### format_stochastic_summary_string maintenance pattern

- `print_stochastic_summary` (production) and `format_stochastic_summary_string` (`#[cfg(test)]`) have
  duplicate format strings. Same split pattern as `print_summary`/`format_summary_string` from Phase 8.
  Both are in `summary.rs`. Two maintenance touch points for stochastic summary text. Known pattern.
  `StochasticSource`, `ArOrderSummary`, `StochasticSummary` have no `#[derive(Debug)]` — same pedantic lint gap.

#### Python run.rs refactoring

- `should_simulate` in `run_inner` is computed from `skip_simulation && config` before config moves into `StudySetup`.
  `setup.n_scenarios()` > 0 also reflects the same config flag. The two predicates agree — no regression.
  `skip_simulation=true` overrides `setup.n_scenarios() > 0` via the outer `if should_simulate { ... }` block.
- `DEFAULT_SEED`, `DEFAULT_FORWARD_PASSES`, `DEFAULT_MAX_ITERATIONS` constants removed from `cobre-python`
  (now internal to `cobre-sddp::setup`). Only `DEFAULT_SEED` remains in Python run.rs for seed extraction
  before `StudySetup::new` moves config.

**Why:** Captured during code review of v0.1.3 plan assessment phase to avoid re-investigation.
**How to apply:** Use when reviewing follow-on v0.1.3 work or PRs touching StudySetup, BasisStore, SimulationProgress, or WelfordAccumulator.

---

### v0.1.3 pre-release plan review (epic-01 through epic-04)

**Git range**: main..HEAD (4 commits, branch feat/v0.1.3-pre-release)

#### StudyParams extraction

- `StudyParams::from_config()` extracts scalars from `cobre_io::Config`. No `max_iterations` field — it is derived by `StudySetup::from_broadcast_params` via `max_iterations_from_rules()`. `StudyParams` does NOT have a `max_iterations` field; only `stopping_rule_set` is stored.
- All 3 constants (`DEFAULT_FORWARD_PASSES`, `DEFAULT_MAX_ITERATIONS`, `DEFAULT_SEED`) are now `pub const` in `cobre-sddp::setup`. `cobre-python/src/run.rs` still imports `DEFAULT_SEED` from `cobre_sddp` for extracting seed before config is moved into setup.

#### StochasticProvenance correctness

- Provenance is computed BEFORE `user_opening_tree` is consumed (the comment says so explicitly).
- `correlation_prov` uses `system.correlation().profiles.is_empty()` AND `dim > 0`. If dim > 0 but profiles empty → `NotApplicable`. If dim > 0 and profiles non-empty → `Generated`. `UserSupplied` is never set by `build_stochastic_context` — the CorrelationModel always comes from the System.
- `inflow_prov` uses `hydro_ids.is_empty()` (hydro_ids are computed inside `build_stochastic_context` from the system). This is consistent with the stochastic context's internal dim computation.

#### broadcast_value sentinel analysis

- Zero-length sentinel conflict: types used at all three call sites (System, BroadcastConfig, Option<BroadcastOpeningTree>) all serialize to >0 bytes. `postcard` encodes `Option::None` as a single tag byte. The sentinel is safe for all current call sites but is a generic footgun.

#### correlation_dim display is correct

- `correlation_dim = Some(format!("{n_hydros}x{n_hydros}"))` is correct because the spatial correlation model covers only inflow entities (entity_type = "inflow"), not load buses. Load noise is always independent. The comment on line 251 of stochastic_summary.rs is accurate.

#### BroadcastStoppingRule silent data loss for SimulationBased/GracefulShutdown

- `BroadcastConfig::from_config()` folds `SimulationBased` and `GracefulShutdown` rules into `IterationLimit { DEFAULT_MAX_ITERATIONS }` silently. These variants are not yet in production configs (deferred), so risk is low. If they were user-configured, the behaviour change would be silent.

#### gather_simulation_results reviewed

- `gather_simulation_results` correctly uses two allgatherv calls (lengths then data). `recv_counts` is `Result`-mapped for u64→usize conversion. `recv_displs` is computed via `scan`. `offset` tracking is correct (no off-by-one: `offset += count` after using `all_bytes[offset..offset+count]`). No panic paths.

#### stage_ctx / training_ctx duplication pattern in StudySetup::train()

- `train()` cannot call `stage_ctx()` and `training_ctx()` because both take `&self` while `train()` also needs `&mut self.fcf`. The fix is correct: inline field borrows for `train()`, method calls for `simulate()`. The comment in setup.rs explains this.

#### n_seasons display heuristic

- Uses `system.hydros()[0].id` as representative hydro. Safe: guarded by `n_hydros > 0`. If all hydros have the same seasonal coverage (guaranteed by cobre-io validation), this is correct. If coverage differs, it would show the first hydro's count. Not a bug for current validation rules.
