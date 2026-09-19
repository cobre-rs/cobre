## INGEST ANCHOR PROBE — sddp (2026-09, baseline)

Register-shaped stub: one block per candidate, every anchor of every candidate rendered so
`check-anchors.py "INGEST ANCHOR PROBE — sddp (2026-09, baseline)" --register <this file> --baseline 077dbe2c` resolves it with the
register's own parser and `git show <sha>:<path>` resolver. Every anchor of this station is a
declaration anchor rendered `path::symbol` — the E05-4 gate demoted every line-only, struct-field,
out-of-station and out-of-manifest anchor into `citedContext` before ingest, so the
coincidental-until-symbolised flag has nothing to flag and no `path:line` form appears below.
Ingest ref `<subStation>-<lens>-<nn>` (nn = zero-based index in the envelope).

### 5a-architecture-00 · The run-parameter half of StudySetup uses two incompatible conventions (named projections vs raw config types), so train_inner re-inflates TrainingConfig field-by-field out of the god-struct

**Anchors:** `crates/cobre-sddp/src/setup/mod.rs::StudySetup` `crates/cobre-sddp/src/setup/orchestration.rs::train_inner` `crates/cobre-sddp/src/config.rs::TrainingConfig` `crates/cobre-sddp/src/setup/accessors.rs::set_budget`

### 5a-architecture-01 · The MPI wire config is not a superset-projection of StudyParams: one field is computed on params then dropped and re-read from Config, and two run-mode fields exist only on the wire

**Anchors:** `crates/cobre-sddp/src/setup/params.rs::StudyParams` `crates/cobre-sddp/src/setup/params.rs::from_config`

### 5a-architecture-02 · scalar_parameters is a caller-patched placeholder rather than a constructor input, so both validate paths build stage templates against an empty parameter table and silently resolve every parameterized coefficient to 0.0

**Anchors:** `crates/cobre-sddp/src/setup/params.rs::StudyParams` `crates/cobre-sddp/src/setup/mod.rs::new_with_boundary_requirements` `crates/cobre-sddp/src/policy/resolved_parameters.rs::ResolvedParameters`

### 5a-architecture-03 · CutManagementConfig::warm_start_cuts is a public field with a capacity contract in its doc, hard-coded to 0 at both production construction sites and read by no production consumer

**Anchors:** `crates/cobre-sddp/src/config.rs::CutManagementConfig` `crates/cobre-sddp/src/setup/orchestration.rs::train_inner`

### 5b-architecture-00 · fill_anticipated_columns re-opens the latched commitment slot at the RAW delivery-axis residue that its two sibling residue owners' own rustdoc calls the forbidden, wrong-but-compiling alternative

**Anchors:** `crates/cobre-sddp/src/lp/builder/columns.rs::fill_anticipated_columns` `crates/cobre-sddp/src/lp/builder/entries.rs::fill_anticipated_state_out_def_entries` `crates/cobre-sddp/src/lp/builder/layout.rs::build_anticipated_slot_row_pos`

### 5b-architecture-01 · Pin-round-trip exactness for the four state families is handled by three unrelated per-family mechanisms in three modules, and the commitment family's two mechanisms never cross-reference each other

**Anchors:** `crates/cobre-sddp/src/lp/builder/scaling.rs::apply_commitment_hold_col_scale_unscale` `crates/cobre-sddp/src/lp/builder/scaling.rs::apply_bucket_col_scale` `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs::fill_bound_relaxations` `crates/cobre-sddp/src/lp/builder/patch.rs::fill_col_state_patches`

### 5b-architecture-02 · Commitment hold is the only state family with no typed InCol/OutCol resolver, so four production sites recompose its column by hand as a bare usize - across two different index axes

**Anchors:** `crates/cobre-sddp/src/lp/indexer/state_space.rs::commitment_hold_in_study_offset` `crates/cobre-sddp/src/lp/indexer/state_space.rs::bucket_incoming_col` `crates/cobre-sddp/src/lp/builder/delivery_ring.rs::out_col`

### 5b-architecture-03 · transit_buckets_in is the only incoming state family whose template bounds are never written, inheriting them from buffer initialization while storage_in and commit_in write theirs explicitly

**Anchors:** `crates/cobre-sddp/src/lp/builder/columns.rs::fill_transit_bucket_columns` `crates/cobre-sddp/src/lp/builder/columns.rs::fill_anticipated_state_columns` `crates/cobre-sddp/src/lp/builder/columns.rs::fill_stage_columns`

### 5b-architecture-04 · A production gate predicate takes an unused _state: &StateSpace parameter justified by uniformity with a sibling that is #[cfg(test)]-gated

**Anchors:** `crates/cobre-sddp/src/lp/indexer/anticipated_gate.rs::is_anticipated_decision_active_for_delivery` `crates/cobre-sddp/src/lp/indexer/anticipated_gate.rs::is_anticipated_decision_active`

### 5c-architecture-00 · The cut-binding metadata channel is a sampled-traversal-only side duty: the enumerated backward commits cuts but records no binding activity, so DCS resident-set seeding silently degrades to a generation-age filter on a combination no admission gate rejects

**Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward` `crates/cobre-sddp/src/training/backward/replicated.rs::run_backward_node_replicated` `crates/cobre-sddp/src/cut/dcs.rs::build_initial_resident_set`

### 5c-architecture-01 · CD-015 sharpened: by_scenario's risk aggregation HAS moved out into process_by_scenario_backward, leaving exactly the cross-worker staged-cut merge and the add_cut commit loop inline in compute_one_backward_node against by_node_finish's extracted scatter-aggregate-commit, and the register's proposed commit_by_scenario_cuts does not exist at the pin

**Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` `crates/cobre-sddp/src/training/backward/by_node.rs::by_node_finish` `crates/cobre-sddp/src/training/backward/by_scenario.rs::process_by_scenario_backward`

### 5c-architecture-02 · CD-016 sharpened: the per-worker backward_accum bookkeeping is duplicated at more than the pre-allocation the register names -- the slot_increments into metadata_sync_contribution fold is a second copied block, in two different modules on two different reset schedules, and the replicated driver participates in neither

**Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::process_stage_backward` `crates/cobre-sddp/src/training/backward/by_node.rs::process_stage_backward_by_node` `crates/cobre-sddp/src/training/backward/by_scenario.rs::process_by_scenario_backward` `crates/cobre-sddp/src/workspace/workspace.rs::BackwardAccumulators`

### 5c-architecture-03 · The replicated backward driver never enters the outcome_aggregation module and open-codes a second owner of the Benders intercept derivation, so the crate's cut-intercept formula lives in two places with no shared owner

**Anchors:** `crates/cobre-sddp/src/training/backward/replicated.rs::solve_replicated_outcome_slice` `crates/cobre-sddp/src/training/backward/replicated.rs::run_backward_node_replicated`

### 5c-architecture-04 · CD-028 sharpened: the shared claim-and-scatter owner already exists and its module doc declares a boundary narrower than the duplication, so what remains twice-declared between the enumerated forward and the enumerated simulation is the path marking and the worker-result fold that sit outside any stated boundary -- the fix is extending an owner with two consumers, not minting a new one

**Anchors:** `crates/cobre-sddp/src/training/forward/enumerated.rs::run_enumerated_forward` `crates/cobre-sddp/src/simulation/enumerated.rs::run_sweep` `crates/cobre-sddp/src/simulation/enumerated.rs::mark_own_paths` `crates/cobre-sddp/src/claim_scatter.rs::canonical_scatter` `crates/cobre-sddp/src/simulation/enumerated.rs::enumerated_sim_stage_worker` `crates/cobre-sddp/src/simulation/enumerated.rs::re_expand` `crates/cobre-sddp/src/simulation/enumerated.rs::EnumeratedSimScratch` `crates/cobre-sddp/src/simulation/enumerated.rs::SimNodeVisit` `crates/cobre-sddp/src/simulation/enumerated.rs::EnumeratedSimParams` `crates/cobre-sddp/src/training/forward/enumerated.rs::enumerated_stage_worker` `crates/cobre-sddp/src/training/forward/enumerated.rs::EnumeratedForwardScratch` `crates/cobre-sddp/src/training/forward/enumerated.rs::NodeVisit` `crates/cobre-sddp/src/training/forward/enumerated.rs::EnumeratedParams`

### 5d-architecture-00 · A crate-root flat alias namespace shadows the directory-module tree, so the same file reaches the same cluster by two different paths and an import states nothing about which cluster it crosses into

**Anchors:** `crates/cobre-sddp/src/lib.rs::lp` `crates/cobre-sddp/src/lib.rs::production` `crates/cobre-sddp/src/lib.rs::workspace` `crates/cobre-sddp/src/lib.rs::solve` `crates/cobre-sddp/src/simulation/pipeline.rs::solve_simulation_stage` `crates/cobre-sddp/src/simulation/enumerated.rs::run_enumerated_simulation` `crates/cobre-sddp/src/simulation/extraction.rs::extract_stage_result` `crates/cobre-sddp/src/production/hydro_models/production.rs::resolve_production_models_from_artifacts` `crates/cobre-sddp/src/production/hydro_models/mod.rs::prepare_hydro_models`

### 5d-architecture-01 · The simulation-result to cobre-io write-payload conversion is homed in the production-modeling cluster, whose module doc had to be amended to describe it

**Anchors:** `crates/cobre-sddp/src/production/conversion.rs::IntoWriteRecord` `crates/cobre-sddp/src/production/conversion.rs::with_node` `crates/cobre-sddp/src/production/mod.rs::conversion` `crates/cobre-sddp/src/simulation/types.rs::SimulationStageResult`

### 5a-performance-00 · The opening-order chain builder recomputes a whole-tour cost sum for every candidate 2-opt reversal, and fills a symmetric distance matrix twice

**Anchors:** `crates/cobre-sddp/src/stochastic/noise_key.rs::two_opt_improve` `crates/cobre-sddp/src/stochastic/noise_key.rs::tour_cost` `crates/cobre-sddp/src/stochastic/noise_key.rs::shortest_chain_path` `crates/cobre-sddp/src/stochastic/noise_key.rs::l2_distance_matrix` `crates/cobre-sddp/src/stochastic/noise_key.rs::nearest_neighbor_tour` `crates/cobre-sddp/src/stochastic/noise_key.rs::apply_chain_order`

### 5a-performance-01 · The reverse-topological cut level decomposition is a study invariant but is rescanned and reallocated on every backward pass

**Anchors:** `crates/cobre-sddp/src/setup/node_graph.rs::backward_cut_levels` `crates/cobre-sddp/src/setup/node_graph.rs::NodeGraph` `crates/cobre-sddp/src/setup/node_graph.rs::build_node_graph`

### 5a-performance-02 · Stage frontier resolution scans the entire node array per stage, so an enumerated sweep pays a whole-graph scan for every stage it visits

**Anchors:** `crates/cobre-sddp/src/setup/node_graph.rs::stage_frontier` `crates/cobre-sddp/src/setup/node_graph.rs::frontier_node` `crates/cobre-sddp/src/setup/node_graph.rs::any_stage_node` `crates/cobre-sddp/src/setup/node_graph.rs::build_declared_node_graph`

### 5b-performance-00 · The incoming state-to-LP-column resolver is uncached on the per-solve pin path while its outgoing twin and the cut-extraction path both cache it

**Anchors:** `crates/cobre-sddp/src/lp/builder/patch.rs::fill_col_state_patches` `crates/cobre-sddp/src/lp/indexer/state_space.rs::state_to_lp_incoming_column` `crates/cobre-sddp/src/lp/indexer/state_space.rs::classify`

### 5b-performance-01 · The backward per-opening prep recomputes and re-submits an opening-invariant column-bound state pin and an opening-invariant commitment relaxation once per opening

**Anchors:** `crates/cobre-sddp/src/lp/builder/patch.rs::fill_col_state_patches` `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs::fill_bound_relaxations`

### 5b-performance-02 · Every generic-constraint term allocates and immediately drops a one- or two-element heap vector during stage-template construction

**Anchors:** `crates/cobre-sddp/src/lp/generic_constraints.rs::resolve_variable_ref` `crates/cobre-sddp/src/lp/builder/entries.rs::fill_generic_constraint_entries`

### 5c-performance-00 · The cut-selection value sweep heap-allocates six times per call and twice per rayon fold task, contradicting run_cut_management's stated no-allocation contract

**Anchors:** `crates/cobre-sddp/src/cut/cut_selection.rs::select_for_stage` `crates/cobre-sddp/src/training/session/mod.rs::run_cut_management` `crates/cobre-sddp/src/cut/cut_selection.rs::M_BLOCK`

### 5c-performance-01 · apply_column_rule walks the row-major gemm output panel column-strided twice for every trial column

**Anchors:** `crates/cobre-sddp/src/cut/cut_selection.rs::apply_column_rule` `crates/cobre-sddp/src/gemm.rs::gemm_block`

### 5c-performance-02 · The cut-management projection gather pushes one scalar at a time through the slot-index table with no bulk-copy path for the identity projection

**Anchors:** `crates/cobre-sddp/src/training/session/mod.rs::run_cut_management`

### 5c-performance-03 · The backward level archives the whole level's gathered state block into every sibling node's bucket, so archive footprint and the downstream value sweep both scale with nodes per level

**Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::run_one_backward_level` `crates/cobre-sddp/src/training/visited_states.rs::archive_gathered_states` `crates/cobre-sddp/src/training/visited_states.rs::append`

### 5c-performance-04 · build_slot_lookup clears the whole pool-length slot table on every warm-started solve although only the reconcilable slots are ever written

**Anchors:** `crates/cobre-sddp/src/cut/basis_reconstruct.rs::build_slot_lookup` `crates/cobre-sddp/src/solve/stage_solve.rs::run_stage_solve`

### 5c-performance-05 · enforce_basic_count_invariant recounts BASIC statuses with two full filter passes over the status vectors the reconstruction just wrote

**Anchors:** `crates/cobre-sddp/src/cut/basis_reconstruct.rs::enforce_basic_count_invariant` `crates/cobre-sddp/src/cut/basis_reconstruct.rs::reconstruct_basis` `crates/cobre-sddp/src/cut/basis_reconstruct.rs::reconstruct_basis_uniform_basic` `crates/cobre-sddp/src/solve/stage_solve.rs::run_stage_solve` `crates/cobre-sddp/src/cut/dcs.rs::lazy_solve_preloaded`

### 5c-performance-06 · process_stage_backward allocates a fresh owned staged-cut vector per worker per node at its return boundary, defeating the reused workspace buffer it drains

**Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::process_stage_backward` `crates/cobre-sddp/src/training/backward/by_node.rs::process_stage_backward_by_node`

### 5d-performance-00 · Per-(scenario, stage) element-wise unscale of a per-stage constant row_lower template in the simulation hot path

**Anchors:** `crates/cobre-sddp/src/simulation/pipeline.rs::build_row_lower_unscaled` `crates/cobre-sddp/src/simulation/pipeline.rs::extract_sim_stage_result`

### 5d-performance-01 · Four per-(scenario, stage) extraction vectors collect from flat_map with no reservation while six siblings in the same file pre-reserve the same product

**Anchors:** `crates/cobre-sddp/src/simulation/extraction.rs::extract_hydros` `crates/cobre-sddp/src/simulation/extraction.rs::extract_exchanges` `crates/cobre-sddp/src/simulation/extraction.rs::extract_buses` `crates/cobre-sddp/src/simulation/extraction.rs::extract_stub_collections`

### 5d-performance-02 · re_expand deep-clones each captured node result once per visiting leaf path because its scratch is borrowed immutably

**Anchors:** `crates/cobre-sddp/src/simulation/enumerated.rs::re_expand` `crates/cobre-sddp/src/simulation/types.rs::SimulationStageResult`

### 5d-performance-03 · FPHA fitting rebuilds the shared (V, Q) grid and re-walks the same nodes at four pipeline stages instead of building it once per plant

**Anchors:** `crates/cobre-sddp/src/production/fpha_fitting/grid.rs::build_grid` `crates/cobre-sddp/src/production/fpha_fitting/hull_fit.rs::build_cloud` `crates/cobre-sddp/src/production/fpha_fitting/alpha.rs::compute_alpha_fpha` `crates/cobre-sddp/src/production/fpha_fitting/deviation.rs::compute_fit_deviation`

### 5d-performance-04 · fit_gamma_s_for_planes rebuilds the grid and recomputes the whole min-over-planes envelope once per plane

**Anchors:** `crates/cobre-sddp/src/production/fpha_fitting/secant.rs::fit_gamma_s_for_planes` `crates/cobre-sddp/src/production/fpha_fitting/secant.rs::representative_operating_point`

### 5d-performance-05 · long_term_mean_inflow rescans the entire inflow-history table once per hydro inside the parallel production fit

**Anchors:** `crates/cobre-sddp/src/production/hydro_models/production.rs::long_term_mean_inflow` `crates/cobre-sddp/src/production/hydro_models/production.rs::fit_one_hydro`

### 5d-performance-06 · Hull facet dedup uses a linear Vec::contains membership scan whose cost grows with the number of planes already kept

**Anchors:** `crates/cobre-sddp/src/production/fpha_fitting/hull_fit.rs::fit_hull_planes`

### 5d-performance-07 · prepare_hydro_models_from_artifacts groups and per-hydro sorts the same geometry table three times in one call

**Anchors:** `crates/cobre-sddp/src/production/hydro_models/mod.rs::prepare_hydro_models_from_artifacts` `crates/cobre-sddp/src/production/hydro_models/production.rs::build_geometry_map` `crates/cobre-sddp/src/production/hydro_models/evaporation.rs::resolve_evaporation_models_from_artifacts`

### 5a-over-engineering-00 · Four production #[allow(clippy::...)] sites in the 5a manifest carry no written rationale, so the census's sanctioned-by-construction claim is false at those four

**Anchors:** `crates/cobre-sddp/src/policy/orchestration.rs::write_checkpoint` `crates/cobre-sddp/src/setup/node_graph.rs::build_chain_node_graph` `crates/cobre-sddp/src/setup/node_graph.rs::NodeOpenings` `crates/cobre-sddp/src/stochastic/noise_key.rs::chain_position_key`

### 5a-over-engineering-01 · ncs_stochastic_dormant_for_test ships in the default public API while its four sibling test hooks in the same file are test-support-gated

**Anchors:** `crates/cobre-sddp/src/setup/accessors.rs::ncs_stochastic_dormant_for_test`

### 5a-over-engineering-02 · train_inner hand-mirrors the 19-field TrainingContext and the StageContext literal that accessors.rs already constructs, and the training_ctx constructor it duplicates is cfg-gated with no production consumer

**Anchors:** `crates/cobre-sddp/src/setup/accessors.rs::training_ctx` `crates/cobre-sddp/src/setup/accessors.rs::stage_ctx` `crates/cobre-sddp/src/setup/orchestration.rs::train_inner`

### 5a-over-engineering-03 · setup/scenario_library_set.rs is a 46-line public module of two impl-less structs whose name collides with the sibling setup/scenario_libraries.rs, which does not contain the type ScenarioLibraries

**Anchors:** `crates/cobre-sddp/src/setup/scenario_library_set.rs::ScenarioLibraries` `crates/cobre-sddp/src/setup/scenario_library_set.rs::PhaseLibraries` `crates/cobre-sddp/src/setup/scenario_libraries.rs::build_historical_inflow_library`

### 5b-over-engineering-00 · The generic Col and Row LP-index newtypes have zero consumers: the typed-address vocabulary their module doc describes is carried entirely by StateDim, CutSlot, BlockIdx, InCol and OutCol

**Anchors:** `crates/cobre-sddp/src/lp/indexer/index.rs::Col` `crates/cobre-sddp/src/lp/indexer/index.rs::Row`

### 5b-over-engineering-01 · FphaRowRange is a zero-consumer public type whose module doc claims a production carrier that does not exist and whose row formula contradicts the live FPHA walker

**Anchors:** `crates/cobre-sddp/src/lp/indexer/layout.rs::FphaRowRange` `crates/cobre-sddp/src/lp/builder/fpha_cursor.rs::for_each_fpha_plane`

### 5c-over-engineering-00 · Five public read-surface items on the stats and convergence types have no caller outside #[cfg(test)] code, and the buffer-size helper's documented job is done inline at the two sites it names

**Anchors:** `crates/cobre-sddp/src/solver_stats.rs::worker_opening_stats_buffer_size` `crates/cobre-sddp/src/solver_stats.rs::StageWorkerStatsBuffer` `crates/cobre-sddp/src/convergence/convergence.rs::ci_95_half_width`

### 5c-over-engineering-01 · RankDistribution::actual_per_rank forwards to cobre_comm::per_rank_counts without deciding anything, for one caller, while re-taking a total the struct already stores

**Anchors:** `crates/cobre-sddp/src/training/session/rank_distribution.rs::actual_per_rank` `crates/cobre-sddp/src/training/session/rank_distribution.rs::RankDistribution`

### 5d-over-engineering-00 · 42 of 69 production-region `#[allow(...)]` openers in the 5d manifest carry no rationale, refuting the premise the census clearing rests on

**Anchors:** `crates/cobre-sddp/src/production/fpha_fitting/geometry.rs::resolve_fitting_bounds` `crates/cobre-sddp/src/production/fpha_fitting/grid.rs::build_grid` `crates/cobre-sddp/src/production/fpha_fitting/secant.rs::fit_gamma_s` `crates/cobre-sddp/src/simulation/aggregation.rs::aggregate_simulation` `crates/cobre-sddp/src/simulation/extraction.rs::extract_hydro_per_block` `crates/cobre-sddp/src/simulation/state.rs::run_worker_scenarios`

### 5d-over-engineering-01 · Five `pub(crate)` items in production/ and simulation/ have zero consumers outside their defining file and nothing forces the wider visibility

**Anchors:** `crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs::QuarticSegment` `crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs::TailraceSegments` `crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs::TailraceFamily` `crates/cobre-sddp/src/production/hydro_models/production.rs::DEFAULT_REFERENCE_VOLUME_FRACTION` `crates/cobre-sddp/src/simulation/state.rs::SimWorkerParams`

### 5d-over-engineering-02 · `SimulationInputs::new` is a ten-parameter pass-through constructor whose only production caller already carries the suppression it duplicates

**Anchors:** `crates/cobre-sddp/src/simulation/state.rs::SimulationInputs` `crates/cobre-sddp/src/simulation/pipeline.rs::simulate`

### 5a-test-bloat-00 · Inline-vs-sibling test homing in setup/, policy/ and stochastic/ is not just unapplied but inverted: the only file with an extracted sibling has the fewest inline tests and carries three test homes at once

**Anchors:** `crates/cobre-sddp/src/setup/mod.rs::post_study_resolution_tests` `crates/cobre-sddp/src/setup/mod.rs::transit_seed_round_trip_tests` `crates/cobre-sddp/src/setup/tests.rs::minimal_system` `crates/cobre-sddp/src/policy/policy_load.rs::tests` `crates/cobre-sddp/src/policy/policy_export.rs::tests` `crates/cobre-sddp/src/policy/reconcile.rs::tests` `crates/cobre-sddp/src/setup/node_graph.rs::tests` `crates/cobre-sddp/src/stochastic/noise.rs::tests`

### 5a-test-bloat-01 · The transit-seed fixture family is declared three times byte-identically inside one directory: hydro(), zero_penalties() and date() each exist in setup/bucket_topology.rs, setup/mod.rs and setup/tests.rs

**Anchors:** `crates/cobre-sddp/src/setup/bucket_topology.rs::hydro` `crates/cobre-sddp/src/setup/mod.rs::hydro` `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_hydro` `crates/cobre-sddp/src/setup/bucket_topology.rs::zero_penalties` `crates/cobre-sddp/src/setup/mod.rs::zero_penalties` `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_zero_penalties` `crates/cobre-sddp/src/setup/mod.rs::date` `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_date`

### 5a-test-bloat-02 · HydroPenalties is hand-enumerated field by field 46 times in setup/tests.rs and 276 times across the crate with zero struct-update spreads, defeating the one-place field-addition invariant tests/common/builders.rs documents

**Anchors:** `crates/cobre-sddp/src/setup/tests.rs::minimal_system` `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_zero_penalties` `crates/cobre-sddp/src/setup/mod.rs::zero_penalties` `crates/cobre-sddp/src/setup/bucket_topology.rs::zero_penalties` `crates/cobre-sddp/tests/common/builders.rs::neutral_hydro_penalties` `crates/cobre-sddp/tests/common/builders.rs::make_hydro` `crates/cobre-sddp/tests/common/builders.rs::HydroSpec`

### 5a-test-bloat-03 · The four right_boundary_* binaries each re-declare the same ResolvedPenalties, ResolvedBounds and study_start prelude, and right_boundary_validation.rs carries a ~125-line copy of right_boundary_pricing.rs's ring-injection prelude

**Anchors:** `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::penalties` `crates/cobre-sddp/tests/right_boundary_output.rs::penalties` `crates/cobre-sddp/tests/right_boundary_pricing.rs::penalties` `crates/cobre-sddp/tests/right_boundary_validation.rs::penalties` `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::bounds` `crates/cobre-sddp/tests/right_boundary_output.rs::bounds` `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::study_start` `crates/cobre-sddp/tests/right_boundary_pricing.rs::freeze_terminal_template` `crates/cobre-sddp/tests/right_boundary_validation.rs::freeze_terminal_template` `crates/cobre-sddp/tests/right_boundary_pricing.rs::inject_ring_boundary` `crates/cobre-sddp/tests/right_boundary_validation.rs::inject_ring_boundary`

### 5a-test-bloat-04 · anticipated_core.rs declares build_config 14 times, build_system 13 times and default_hydro_penalties 6 times inside a single 8,828-line binary that already links mod common

**Anchors:** `crates/cobre-sddp/tests/anticipated_core.rs::default_hydro_penalties` `crates/cobre-sddp/tests/anticipated_core.rs::build_config` `crates/cobre-sddp/tests/anticipated_core.rs::build_system` `crates/cobre-sddp/tests/anticipated_core.rs::default_hydro_bounds` `crates/cobre-sddp/tests/anticipated_scenarios.rs::default_hydro_penalties` `crates/cobre-sddp/tests/anticipated_scenarios.rs::build_config` `crates/cobre-sddp/tests/anticipated_scenarios.rs::build_system` `crates/cobre-sddp/tests/common/builders.rs::make_hydro`

### 5a-test-bloat-05 · state_layout_for is cloned byte-identically into seven integration binaries under a doc comment whose stated rationale is false at the pin, while the shared test_support symbol it duplicates is already imported by those same files

**Anchors:** `crates/cobre-sddp/tests/integration.rs::state_layout_for` `crates/cobre-sddp/tests/load_integration.rs::state_layout_for` `crates/cobre-sddp/tests/conformance.rs::state_layout_for` `crates/cobre-sddp/tests/integration.rs::study_dims` `crates/cobre-sddp/tests/load_integration.rs::study_dims`

### 5a-test-bloat-06 · A test-only reference oracle for the lag-shift kernel is homed in the hot-path production module stochastic/noise.rs rather than beside the tests that are its only callers

**Anchors:** `crates/cobre-sddp/src/stochastic/noise.rs::shift_lag_state` `crates/cobre-sddp/src/stochastic/noise.rs::accumulate_and_shift_lag_state` `crates/cobre-sddp/src/stochastic/noise.rs::tests`

### 5b-test-bloat-00 · Both unit-test homing conventions are declared back to back inside one file: lp/builder/layout.rs carries `mod tests;` and an inline `collapse_stage_level_tests` block on adjacent lines

**Anchors:** `crates/cobre-sddp/src/lp/builder/layout.rs::collapse_stage_level_tests` `crates/cobre-sddp/src/lp/builder/layout.rs::resolve_affine_of_two_term_remainder_sums_constant_and_terms` `crates/cobre-sddp/src/lp/builder/commitment_reconcile/tests.rs::commitment_inside_bounds_needs_no_patch`

### 5b-test-bloat-01 · The extracted-versus-inline choice is inverted against the yardstick threshold and reaches beyond lp/builder into lp/indexer: the directory's smallest test body is extracted while its largest four stay inline

**Anchors:** `crates/cobre-sddp/src/lp/indexer/state_space.rs::lp_column_map_matches_resolver_with_lags_and_anticipated` `crates/cobre-sddp/src/lp/builder/patch.rs::state_col_patch_count_returns_n_times_one_plus_l` `crates/cobre-sddp/src/lp/builder/columns.rs::filling_phase_gating_tests` `crates/cobre-sddp/src/lp/builder/entries.rs::pumping_water_tests` `crates/cobre-sddp/src/lp/builder/commitment_reconcile/tests.rs::commitment_exactly_at_cap_needs_no_patch`

### 5b-test-bloat-02 · entries.rs's `pumping_water_tests` has become the crate's catch-all test dump: 6576 lines and 77 test fns under a name that describes one entry family

**Anchors:** `crates/cobre-sddp/src/lp/builder/entries.rs::pumping_water_tests` `crates/cobre-sddp/src/lp/builder/entries.rs::csc_byte_identical_under_permuted_declaration_order` `crates/cobre-sddp/src/lp/builder/entries.rs::zero_cost_tests`

### 5b-test-bloat-03 · tests/par_a_lag12_lp_coefficient.rs's `build_classical_fixture` is a 243-line clone of `build_par_a_fixture` whose entire semantic delta is one field set to None

**Anchors:** `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs::build_classical_fixture` `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs::build_par_a_fixture` `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs::classical_par_has_no_lag_11_column`

### 5b-test-bloat-04 · tests/template_integration.rs grows nineteen whole-system fixture builders in copy-paste-then-parameterize families, two of which are roughly 85 percent textually identical to their sibling

**Anchors:** `crates/cobre-sddp/tests/template_integration.rs::build_hydro_one_ant_system` `crates/cobre-sddp/tests/template_integration.rs::one_hydro_one_ant_system` `crates/cobre-sddp/tests/template_integration.rs::two_anticipated_thermal_system` `crates/cobre-sddp/tests/template_integration.rs::one_anticipated_thermal_system` `crates/cobre-sddp/tests/template_integration.rs::one_bus_system_n_blks_with_generic`

### 5b-test-bloat-05 · template_integration/generic_constraints.rs silently shadows the parent's `one_hydro_system` builder with a same-named local of a different signature, under `use super::*`

**Anchors:** `crates/cobre-sddp/tests/template_integration/generic_constraints.rs::one_hydro_system` `crates/cobre-sddp/tests/template_integration.rs::one_hydro_system` `crates/cobre-sddp/tests/template_integration/generic_constraints.rs::generic_constraint_two_hydros_sum_csc_entries`

### 5b-test-bloat-06 · Three separate Stage fixture surfaces with divergent defaults serve one crate, and the builder-module one is `#[cfg(test)]`-only so no integration binary can reach it

**Anchors:** `crates/cobre-sddp/src/lp/builder/test_support.rs::two_block_stage` `crates/cobre-sddp/src/lp/builder/test_support.rs::three_block_stage` `crates/cobre-sddp/src/lp/builder/mod.rs::test_support` `crates/cobre-sddp/tests/common/builders.rs::make_stage` `crates/cobre-sddp/tests/common/builders.rs::StageSpec`

### 5b-test-bloat-07 · The four-line `case_dir` helper is declared twice inside tests/lp_builder.rs alone, and tests/common/ hosts a third copy it does not export

**Anchors:** `crates/cobre-sddp/tests/lp_builder.rs::case_dir` `crates/cobre-sddp/tests/common/parity_hash.rs::case_dir` `crates/cobre-sddp/tests/common/mod.rs::build_setup_for_case`

### 5c-test-bloat-00 · The sync_cuts count-mismatch rejection is asserted twice, in two separate integration binaries, each carrying its own byte-identical 2-rank Communicator stub

**Anchors:** `crates/cobre-sddp/tests/test_mpi_allgatherv_nonuniform_workers.rs::sync_cuts_rejects_mismatched_local_cut_count` `crates/cobre-sddp/tests/test_mpi_allgatherv_nonuniform_workers.rs::StubComm2Rank` `crates/cobre-sddp/tests/test_mpi_sync_cuts_invariant.rs::sync_cuts_invariant_rejected_when_cut_count_mismatches` `crates/cobre-sddp/tests/test_mpi_sync_cuts_invariant.rs::StubComm2Rank`

### 5c-test-bloat-01 · cut_basis.rs re-declares five fixture helpers two and three times inside a single binary that already wires mod common

**Anchors:** `crates/cobre-sddp/tests/cut_basis.rs::write_test_checkpoint` `crates/cobre-sddp/tests/cut_basis.rs::d01_case_dir` `crates/cobre-sddp/tests/cut_basis.rs::d03_case_dir` `crates/cobre-sddp/tests/cut_basis.rs::ascending_stage_end_dates` `crates/cobre-sddp/tests/cut_basis.rs::build_setup` `crates/cobre-sddp/tests/common/mod.rs::build_setup_for_case`

### 5c-test-bloat-02 · No Communicator test double lives in the shared test_support surface, so src-side unit tests re-declare StubComm four times and Rank0Of2 twice while tests/common already exports both

**Anchors:** `crates/cobre-sddp/tests/common/mod.rs::StubComm` `crates/cobre-sddp/tests/common/mod.rs::Rank0Of2` `crates/cobre-sddp/src/training/backward/tests.rs::StubComm` `crates/cobre-sddp/src/training/backward_pass_state.rs::StubComm` `crates/cobre-sddp/src/training/session/mod.rs::StubComm` `crates/cobre-sddp/src/training/training/tests.rs::StubComm` `crates/cobre-sddp/src/training/backward_pass_state.rs::Rank0Of2` `crates/cobre-sddp/src/training/session/mod.rs::Rank0Of2` `crates/cobre-sddp/src/cut/cut_sync.rs::ThreeRankComm` `crates/cobre-sddp/src/cut/cut_sync.rs::TwoRankStubComm`

### 5c-test-bloat-03 · Eight same-named MockSolver doubles re-implement the SolverInterface required methods across the manifest, with no shared base in test_support

**Anchors:** `crates/cobre-sddp/src/training/backward/tests.rs::MockSolver` `crates/cobre-sddp/src/training/backward_pass_state.rs::MockSolver` `crates/cobre-sddp/src/training/forward/tests.rs::MockSolver` `crates/cobre-sddp/src/training/forward_pass_state.rs::MockSolver` `crates/cobre-sddp/src/training/lower_bound.rs::MockSolver` `crates/cobre-sddp/src/training/session/mod.rs::MockSolver` `crates/cobre-sddp/src/training/training/tests.rs::MockSolver` `crates/cobre-sddp/src/workspace/workspace.rs::MockSolver`

### 5c-test-bloat-04 · The inline-versus-extracted test homing split is accidental in training, cut, workspace and solve: three measured inversions and a third homing convention

**Anchors:** `crates/cobre-sddp/src/training/backward/tests.rs::MockSolver` `crates/cobre-sddp/src/training/forward/tests.rs::MockSolver` `crates/cobre-sddp/src/training/training/tests.rs::MockSolver` `crates/cobre-sddp/src/training/stage_solve_prep/tests.rs::RecordingSolver` `crates/cobre-sddp/src/training/backward_pass_state.rs::MockSolver` `crates/cobre-sddp/src/training/lower_bound.rs::MockSolver` `crates/cobre-sddp/src/training/session/mod.rs::MockSolver` `crates/cobre-sddp/src/cut/cut_selection.rs::make_pool` `crates/cobre-sddp/src/cut/pool.rs::new_creates_pool_with_correct_capacity_and_all_inactive` `crates/cobre-sddp/src/solve/solver_phase.rs::highs_tests` `crates/cobre-sddp/src/solve/solver_phase.rs::clp_tests` `crates/cobre-sddp/src/solve/solver_phase.rs::validate_phase_solver_config_tests`

### 5c-test-bloat-05 · HydroPenalties has no Default, so fixtures spell all sixteen fields; the crate answers with per-file private constructors including one inside test_support itself

**Anchors:** `crates/cobre-sddp/src/training/lower_bound.rs::zero_hydro_penalties` `crates/cobre-sddp/src/training/lower_bound.rs::default_hydro_penalties` `crates/cobre-sddp/tests/common/builders.rs::neutral_hydro_penalties` `crates/cobre-sddp/tests/forward_sampler_no_alloc.rs::make_hydro_spec`

### 5d-test-bloat-00 · Five integration binaries hand-copy test_support helpers behind a doc claim that the crate surface is unreachable, while every one of them already calls into that surface

**Anchors:** `crates/cobre-sddp/tests/simulation_integration.rs::state_layout_for` `crates/cobre-sddp/tests/simulation_pipeline_integration.rs::state_layout_for` `crates/cobre-sddp/tests/inflow_nonnegativity.rs::state_layout_for` `crates/cobre-sddp/tests/load_integration.rs::state_layout_for` `crates/cobre-sddp/tests/integration.rs::state_layout_for` `crates/cobre-sddp/tests/simulation_integration.rs::all_enabled_cut_state_layouts` `crates/cobre-sddp/tests/integration.rs::all_enabled_cut_state_layouts` `crates/cobre-sddp/tests/load_integration.rs::all_enabled_cut_state_layouts` `crates/cobre-sddp/tests/inflow_nonnegativity.rs::all_enabled_cut_state_layouts` `crates/cobre-sddp/tests/simulation_integration.rs::study_dims_for` `crates/cobre-sddp/tests/inflow_nonnegativity.rs::study_dims_for` `crates/cobre-sddp/src/test_support.rs::state_layout` `crates/cobre-sddp/src/test_support.rs::all_enabled_cut_state_layouts` `crates/cobre-sddp/src/test_support.rs::study_dims_for`

### 5d-test-bloat-01 · hold_k1_byte_stability_probe.rs asserts over its own string literals and one required fixture name no longer exists, so the guard passes green while verifying nothing

**Anchors:** `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::lead_ge2_rebaseline_contains_the_named_k2_k3_fixtures` `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::LEAD_GE2_REBASELINE` `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::K1_SURVIVORS` `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::k1_survivors_and_lead_ge2_rebaseline_are_disjoint` `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::k1_byte_stability_verdict` `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::AUDITED_FIXTURE_FILES` `crates/cobre-sddp/tests/anticipated_scenarios.rs::simulation_ring_buffer_shifts_anticipated_state_k1`

### 5d-test-bloat-02 · The four right_boundary binaries carry md5-identical 40-to-41-line fixture prologues and a shared-helper trio, with the fixture preamble occupying 54 to 72 percent of each file

**Anchors:** `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::penalties` `crates/cobre-sddp/tests/right_boundary_output.rs::penalties` `crates/cobre-sddp/tests/right_boundary_pricing.rs::penalties` `crates/cobre-sddp/tests/right_boundary_validation.rs::penalties` `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::bounds` `crates/cobre-sddp/tests/right_boundary_output.rs::bounds` `crates/cobre-sddp/tests/right_boundary_pricing.rs::bounds` `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::stages` `crates/cobre-sddp/tests/right_boundary_output.rs::stages` `crates/cobre-sddp/tests/right_boundary_pricing.rs::leaf_positions` `crates/cobre-sddp/tests/shared_boundary_terminal_fan_probe.rs::leaf_positions` `crates/cobre-sddp/tests/right_boundary_pricing.rs::fixture_priced_date` `crates/cobre-sddp/tests/right_boundary_validation.rs::fixture_priced_date` `crates/cobre-sddp/tests/shared_boundary_terminal_fan_probe.rs::fixture_priced_date` `crates/cobre-sddp/src/test_support.rs::fixture_priced_date`

### 5d-test-bloat-03 · anticipated_core.rs re-declares md5-identical hydro-default fixtures six times inside one binary and build_config fourteen times, while already importing the shared O(1) field-add builder

**Anchors:** `crates/cobre-sddp/tests/anticipated_core.rs::default_hydro_penalties` `crates/cobre-sddp/tests/anticipated_core.rs::default_hydro_bounds` `crates/cobre-sddp/tests/anticipated_core.rs::default_hydro_block_bounds` `crates/cobre-sddp/tests/anticipated_core.rs::build_config` `crates/cobre-sddp/tests/anticipated_core.rs::build_system` `crates/cobre-sddp/tests/anticipated_scenarios.rs::default_hydro_penalties` `crates/cobre-sddp/tests/anticipated_scenarios.rs::build_config` `crates/cobre-sddp/tests/hydro_sim.rs::build_system` `crates/cobre-sddp/tests/common/builders.rs::HydroSpec`

### 5d-test-bloat-04 · The shared harness carries two hardcoded single-shape communicators while one binary carries the parameterized one that subsumes both, so the rank shape a test needs decides which of three types it reaches for

**Anchors:** `crates/cobre-sddp/tests/common/mod.rs::StubComm` `crates/cobre-sddp/tests/common/mod.rs::Rank0Of2` `crates/cobre-sddp/tests/simulation_pipeline_integration.rs::StubComm` `crates/cobre-sddp/tests/integration.rs::ShutdownComm` `crates/cobre-sddp/tests/conformance.rs::LocalComm`

### 5d-test-bloat-05 · The sibling-versus-inline test homing split across simulation and production is not threshold-driven: one file declares both homes at once, four inline modules sit above the proposed threshold, and one sibling was extracted at eleven test fns

**Anchors:** `crates/cobre-sddp/src/simulation/extraction.rs::transit_seed_tests` `crates/cobre-sddp/src/simulation/extraction.rs::tests` `crates/cobre-sddp/src/production/energy_conversion/builder.rs::tests` `crates/cobre-sddp/src/production/hydro_models/evaporation.rs::tests` `crates/cobre-sddp/src/production/hydro_models/summary.rs::tests` `crates/cobre-sddp/src/simulation/types.rs::tests` `crates/cobre-sddp/src/simulation/pipeline.rs::tests` `crates/cobre-sddp/src/lead_time/mod.rs::tests` `crates/cobre-sddp/src/production/fpha_fitting/mod.rs::tests` `crates/cobre-sddp/src/production/hydro_models/production.rs::tests`
