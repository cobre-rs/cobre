# Test Inventory — Architecture Unification

Canonical test-suite inventory. Taxonomy in `docs/assessments/test-inventory-taxonomy.md`.

## 1. Summary

**Total tests:** 3438

### Per-crate breakdown

| Crate | Count |
| ----- | ----: |
| `cobre-cli` | 187 |
| `cobre-comm` | 129 |
| `cobre-core` | 210 |
| `cobre-io` | 952 |
| `cobre-python` | 1 |
| `cobre-sddp` | 1,334 |
| `cobre-solver` | 102 |
| `cobre-stochastic` | 523 |

### Per-category breakdown

| Category | Count |
| -------- | ----: |
| `conformance` | 31 |
| `coverage-matrix` | 0 |
| `e2e` | 4 |
| `integration` | 278 |
| `parameter-sweep` | 0 |
| `regression` | 32 |
| `unit` | 3,093 |

### Per-guard breakdown

Guards relevant to Epic 03/04/05 deletions are highlighted. Tests carrying multiple guards contribute one count per guard label, so the sum may exceed the total test count.

| Guard | Count | Epic |
| ----- | ----: | ---- |
| `add-rows-trait` | 34 | 05 |
| `alien-only` | 0 | 03 |
| `baked` | 0 | — |
| `broadcast-canonical-field` | 0 | 04 |
| `broadcast-warm-start-field` | 0 | 03 |
| `canonical-clearsolver` | 0 | — |
| `canonical-config-flag` | 4 | 04 |
| `canonical-disabled` | 0 | 04 |
| `clear-solver-state-trait` | 0 | 04 |
| `convertido-determinism` | 3 | — |
| `d-case-determinism` | 35 | — |
| `fpha-slow` | 164 | 09 |
| `generic` | 3,198 | — |
| `non-alien-first` | 0 | — |
| `non-baked` | 0 | 05 |
| `solve-with-basis-trait` | 0 | 04 |
| `stored-cut-row-offset` | 2 | 05 |
| `training-result-struct-literal` | 0 | — |
| `unified-path` | 0 | — |
| `warm-start-config-flag` | 7 | 03 |

## 2. Inventory Table

All 3438 tests sorted by (crate, file, line).

| Crate | File | Line | Function | Body LoC | Test module | Category | Guards | Notes |
| ----- | ---- | ---: | -------- | -------: | ----------- | -------- | ------ | ----- |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1552 | `backward_result_fields_accessible` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1577 | `backward_result_clone_and_debug` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1597 | `dual_extraction_formula_coefficients_are_negated_duals` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1610 | `intercept_formula_matches_spec` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1627 | `single_stage_system_produces_no_cuts` | 89 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1718 | `two_stage_system_two_trial_points_generates_two_cuts_at_stage_0` | 99 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1819 | `cut_inserted_with_correct_stage_iteration_and_forward_pass_index` | 96 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 1917 | `no_cuts_generated_at_last_stage` | 94 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2013 | `elapsed_ms_is_non_negative` | 87 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2102 | `infeasible_solver_returns_sddp_infeasible_error` | 91 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2195 | `expectation_aggregation_mean_of_per_opening_intercepts` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2228 | `cut_coefficients_and_intercept_match_dual_extraction_formula` | 112 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2342 | `cut_gradient_sign_physically_correct` | 115 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2459 | `cut_is_tight_at_trial_point` | 115 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2576 | `single_rank_backward_pass_with_local_backend_produces_correct_fcf` | 95 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2673 | `forward_pass_index_matches_global_scenario_index` | 115 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2800 | `warm_start_uses_prepopulated_forward_basis` | 96 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 2906 | `multi_opening_subsequent_openings_use_internal_hotstart` | 97 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 3015 | `backward_solver_error_propagates` | 100 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 3122 | `test_backward_pass_parallel_cut_determinism` | 248 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 3553 | `backward_pass_load_patches_applied` | 167 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 3727 | `backward_pass_no_load_buses_unchanged` | 153 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 3889 | `backward_pass_cut_coefficients_unaffected` | 158 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4067 | `per_stage_cut_sync_invariant_after_bug1_fix` | 112 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4189 | `metadata_sync_updates_active_count_and_last_active_iter` | 127 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4471 | `work_stealing_produces_identical_results_across_worker_counts` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4603 | `decompose_four_workers_different_solve_times` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4630 | `decompose_setup_time_is_aggregate_non_solve_work` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4658 | `decompose_identical_workers_zero_imbalance` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4676 | `decompose_single_worker` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/backward.rs` | 4695 | `decompose_scheduling_clamped_when_worker_exceeds_wall` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 460 | `test_empty_stored_all_new_cuts` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 512 | `test_all_preserved_same_slots_same_order` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 551 | `test_drops_only` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 594 | `test_reorder` | 61 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 659 | `test_adds_only` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 707 | `test_mixed_drop_and_add` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 749 | `test_empty_iterator_preserves_template_rows` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 783 | `test_slot_lookup_growth_safe_in_release` | 35 | `tests` | `unit` | `stored-cut-row-offset` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 832 | `test_stored_cut_row_offset_skips_baked_rows` | 58 | `tests` | `unit` | `stored-cut-row-offset` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 902 | `test_forward_reconstruct_preserves_slots_after_churn` | 59 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 967 | `test_forward_reconstruct_three_new_slack_cuts` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 1020 | `test_capture_metadata_invariants` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 1106 | `reconstructed_basis_preserves_basic_count_invariant_backward_all_preserved` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 1154 | `reconstructed_basis_preserves_basic_count_invariant_backward_with_new_cuts` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 1247 | `reconstructed_basis_preserves_basic_count_invariant_forward_all_preserved` | 61 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 1322 | `reconstructed_basis_preserves_basic_count_invariant_forward_drops_with_lower` | 75 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 1413 | `reconstructed_basis_preserves_basic_count_invariant_forward_new_cuts_after_drops` | 70 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/config.rs` | 302 | `field_access_forward_passes_and_max_iterations` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/config.rs` | 317 | `checkpoint_interval_none_and_some` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/config.rs` | 345 | `warm_start_cuts_field_accessible` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/config.rs` | 364 | `event_sender_none` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/config.rs` | 374 | `event_sender_some_can_send_training_event` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/config.rs` | 417 | `debug_output_non_empty` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 262 | `new_initializes_all_fields_to_default` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 274 | `update_increments_iteration_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 284 | `update_stores_lb_and_ub_correctly` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 301 | `gap_formula_uses_max_guard` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 317 | `gap_formula_normal_case` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 332 | `lower_bound_history_grows` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 342 | `set_shutdown_triggers_graceful_rule` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 363 | `set_simulation_costs_populates_monitor_state` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 393 | `iteration_limit_triggers_at_limit` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 408 | `bound_stalling_triggers_when_stable` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 445 | `ac_iteration_limit_triggers_at_third_call` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 467 | `ac_gap_formula_with_ub_110_lb_100` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 491 | `ac_set_shutdown_triggers_graceful_shutdown_rule` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/convergence.rs` | 510 | `ac_lb_and_iteration_count_track_correctly` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/conversion.rs` | 414 | `convert_scenario_result_to_write_payload_round_trip` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/conversion.rs` | 447 | `convert_stage_result_preserves_all_entity_types` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 374 | `new_creates_correct_number_of_pools` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 380 | `new_each_pool_has_correct_capacity_no_warmstart` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 392 | `new_each_pool_has_correct_capacity_with_warmstart` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 402 | `new_all_pools_start_with_zero_active_cuts` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 408 | `new_zero_stages_is_valid` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 415 | `new_non_uniform_warm_start_counts_per_stage_capacity` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 430 | `new_uniform_zero_counts_matches_old_scalar_zero_behavior` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 442 | `new_mismatched_length_panics_in_debug` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 449 | `add_cut_and_active_cuts_round_trip_at_specific_stage` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 462 | `active_cuts_at_other_stage_returns_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 471 | `add_cut_multiple_stages_are_independent` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 484 | `evaluate_at_state_delegates_to_correct_pool` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 500 | `total_active_cuts_sums_across_stages` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 512 | `total_active_cuts_reflects_deactivation` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 524 | `deactivate_delegates_to_correct_pool` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 539 | `ac_new_5_stages_pools_len_is_5` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 545 | `ac_active_cuts_at_stage_with_cut_yields_it` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 555 | `ac_active_cuts_at_different_stage_yields_none` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 564 | `ac_total_active_cuts_is_sum_across_stages` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 575 | `fcf_derives_debug_and_clone` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 622 | `from_deserialized_empty_input_returns_err` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 630 | `from_deserialized_inconsistent_dimensions_returns_err` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 642 | `from_deserialized_preserves_active_flags` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 660 | `from_deserialized_evaluate_at_state_matches_original` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 690 | `from_deserialized_empty_stage_is_valid` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 704 | `from_deserialized_single_cut_stage` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 721 | `warm_start_capacity_includes_training_slots` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 742 | `warm_start_training_cuts_at_correct_offset` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 762 | `warm_start_empty_stage_has_training_capacity` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/fcf.rs` | 778 | `warm_start_preserves_inactive_flags` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 802 | `new_creates_pool_with_correct_capacity_and_all_inactive` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 817 | `new_zero_capacity_is_valid` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 824 | `add_cut_at_slot_zero_stores_intercept_coefficients_and_active_flag` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 839 | `add_cut_deterministic_slot_formula_no_warmstart` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 862 | `add_cut_warm_start_count_offsets_slot` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 874 | `add_cut_metadata_initialized_correctly` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 886 | `populated_count_tracks_high_water_mark` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 900 | `active_cuts_returns_only_active_cuts` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 918 | `active_cuts_empty_pool_returns_empty_iterator` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 925 | `active_count_is_correct_after_add_and_deactivate` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 937 | `deactivate_sets_flags_correctly` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 952 | `deactivate_multiple_indices` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 967 | `deactivate_empty_slice_is_noop` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 975 | `evaluate_at_state_returns_max_cut_value` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 990 | `evaluate_at_state_selects_correct_max` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1003 | `evaluate_at_state_empty_pool_returns_neg_infinity` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1009 | `evaluate_at_state_all_deactivated_returns_neg_infinity` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1017 | `evaluate_at_state_ignores_deactivated_cuts` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1029 | `ac_add_cut_stores_at_slot_zero_and_active_count_is_one` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1041 | `ac_deactivate_reduces_active_count_correctly` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1056 | `ac_evaluate_at_state_returns_correct_max` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1067 | `ac_warm_start_count_offsets_slot` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1079 | `ac_empty_pool_evaluate_returns_neg_infinity` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1086 | `cut_pool_derives_debug_and_clone` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1101 | `sparsity_report_empty_pool` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1111 | `sparsity_report_all_nonzero` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1124 | `sparsity_report_all_zero` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1137 | `sparsity_report_mixed` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1150 | `sparsity_report_excludes_inactive_cuts` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1164 | `sparsity_report_per_dimension_zeros_correct` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1180 | `warm_start_cuts_have_sentinel_iteration` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1218 | `terminal_has_boundary_cuts_when_warm_start_count_positive` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1237 | `no_boundary_cuts_when_warm_start_count_zero` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1245 | `enforce_budget_noop_when_under_budget` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1257 | `enforce_budget_evicts_oldest_last_active_iter` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1279 | `enforce_budget_tiebreaks_by_active_count` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1297 | `enforce_budget_protects_current_iteration` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1315 | `enforce_budget_all_current_iteration_no_eviction` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/pool.rs` | 1329 | `enforce_budget_result_fields` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 156 | `new_creates_empty_map` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 164 | `new_zero_capacity_is_valid` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 170 | `insert_assigns_sequential_rows_from_base_offset` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 187 | `insert_records_slot_to_row_mapping` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 195 | `lp_row_for_slot_returns_none_for_unmapped_slot` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 201 | `lp_row_for_slot_returns_none_for_out_of_range` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 207 | `multiple_inserts_preserve_mappings` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 226 | `insert_same_slot_twice_panics_in_debug` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 235 | `cut_row_map_derives_debug_and_clone` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/row_map.rs` | 250 | `base_row_offset_zero_works` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 262 | `cut_wire_size_zero_state_returns_24` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 267 | `cut_wire_size_one_state_returns_32` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 272 | `cut_wire_size_three_hydro_ar2_returns_96` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 278 | `cut_wire_size_production_scale_returns_16664` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 284 | `round_trip_all_fields_match_exactly` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 300 | `round_trip_verifies_bit_for_bit_coefficient_integrity` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 318 | `byte_offsets_match_wire_format_spec` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 359 | `round_trip_production_scale_n_state_2080` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 378 | `edge_case_n_state_zero_header_only_24_bytes` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 393 | `edge_case_n_state_one_produces_32_byte_record` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 408 | `padding_bytes_at_offset_12_to_15_are_zero` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 415 | `multi_cut_five_cuts_round_trip_all_match` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 453 | `multi_cut_ten_cuts_round_trip_order_preserved` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 480 | `deserialize_cuts_from_empty_buffer_returns_empty_vec` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut/wire.rs` | 486 | `cut_wire_header_derives_debug_clone_copy_partialeq` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 500 | `should_run_false_at_zero` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 509 | `should_run_false_between_multiples` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 519 | `should_run_true_at_multiples` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 530 | `should_run_lml1_respects_check_frequency` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 542 | `should_run_dominated_respects_check_frequency` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 552 | `level1_deactivates_zero_activity_cuts` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 567 | `level1_retains_positive_activity_cuts` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 581 | `level1_threshold_1_deactivates_cuts_with_count_at_most_1` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 595 | `level1_empty_metadata_returns_empty_set` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 606 | `lml1_deactivates_cuts_outside_memory_window` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 617 | `lml1_retains_cuts_within_memory_window` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 630 | `lml1_retains_cuts_exactly_at_boundary` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 644 | `lml1_mixed_cuts_deactivates_correct_indices` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 660 | `dominated_select_always_returns_empty_set` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 671 | `ac_level1_threshold_0_deactivates_zero_activity_cut` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 690 | `ac_lml1_deactivates_cut_outside_memory_window` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 709 | `select_for_stage_sets_stage_index` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 720 | `select_sets_stage_index_to_zero` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 731 | `deactivation_set_derives_debug_and_clone` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 743 | `cut_metadata_derives_debug_and_clone` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 755 | `test_parse_disabled_default` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 766 | `test_parse_level1` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 795 | `test_parse_lml1` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 823 | `test_parse_domination` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 852 | `test_parse_unknown_method` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 873 | `test_parse_enabled_without_method` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 889 | `test_parse_enabled_false_with_method_returns_none` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 908 | `test_parse_zero_check_frequency` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 934 | `select_skips_already_inactive_slots` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 973 | `lml1_memory_window_boundary_behavior` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1003 | `level1_spares_cuts_from_current_iteration` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1038 | `lml1_spares_cuts_from_current_iteration` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1100 | `dominated_select_deactivate_dominated` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1124 | `dominated_select_partial_domination_retained` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1152 | `dominated_select_none_dominated_when_all_achieve_max` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1181 | `dominated_select_empty_states` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1201 | `dominated_select_single_active_cut` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1222 | `dominated_select_current_iteration_excluded` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_selection.rs` | 1250 | `aggressiveness_ordering_level1_leq_lml1_leq_dominated` | 81 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 520 | `new_send_buf_capacity_is_max_cuts_times_record_size` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 529 | `new_recv_buf_capacity_is_max_cuts_times_num_ranks_times_record_size` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 539 | `new_counts_length_equals_num_ranks` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 546 | `new_displs_length_equals_num_ranks` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 553 | `new_counts_and_displs_initialized_to_max_uniform_values` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 565 | `new_n_state_zero_record_size_is_24` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 573 | `send_buf_serialization_round_trip_two_cuts` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 620 | `counts_and_displs_computation_for_various_cut_counts` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 645 | `sync_cuts_single_rank_returns_zero_remote_cuts` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 662 | `sync_cuts_single_rank_does_not_insert_local_cuts_into_fcf` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 684 | `sync_cuts_serialization_round_trip_via_allgatherv_identity` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 707 | `sync_cuts_zero_local_cuts_returns_zero` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 720 | `sync_cuts_error_maps_to_sddp_communication_error` | 67 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 789 | `sync_cuts_three_ranks_returns_four_remote_cuts` | 123 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/cut_sync.rs` | 914 | `sync_cuts_preserves_cut_fields_after_deserialization` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 124 | `sddp_error_is_send_sync_static` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 129 | `display_solver_variant_contains_solver_and_underlying_message` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 138 | `display_communication_variant_contains_message` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 150 | `display_stochastic_variant_contains_stochastic_and_underlying_message` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 161 | `display_io_variant_contains_io_and_underlying_message` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 175 | `display_validation_variant_contains_message` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 186 | `display_infeasible_variant_contains_stage_iteration_scenario` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 199 | `from_solver_error` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 209 | `from_stochastic_error` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 218 | `from_load_error` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 229 | `from_comm_error_wraps_directly` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 241 | `from_fpha_fitting_error_wraps_as_validation` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 255 | `sddp_error_satisfies_std_error_trait` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/error.rs` | 279 | `all_variants_debug_non_empty` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 1849 | `test_with_scenario_models_replaces_fields` | 72 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 1924 | `test_with_scenario_models_clears_when_empty` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 1945 | `test_estimate_explicit_stats_returns_unchanged` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 1990 | `test_estimate_no_history_returns_unchanged` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2028 | `test_estimation_path_resolve_all_8_combinations` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2082 | `test_estimation_path_as_str_round_trip` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2114 | `test_user_stats_to_rows_maps_all_models` | 48 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2165 | `test_user_stats_to_rows_empty_system` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2382 | `test_partial_estimation_preserves_user_stats` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2439 | `test_partial_estimation_returns_report` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2572 | `test_lag_scale_warning_fires_when_closer_to_estimated` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2600 | `test_lag_scale_warning_not_fires_when_closer_to_user` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2616 | `test_lag_scale_warning_empty_past_inflows` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2632 | `test_lag_scale_warning_skips_hydro_without_history` | 81 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2760 | `test_estimation_report_structure` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2798 | `test_estimation_report_empty_for_pacf` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2834 | `test_contribution_order_zero_fallback` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2848 | `test_contribution_order_zero_input_passes` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2857 | `test_contribution_stable_model_passes` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2876 | `test_apply_contribution_validation_reduces_explosive` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 2933 | `test_pimental_like_multi_season_reduction` | 71 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3010 | `test_all_negative_fallback_to_white_noise` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3052 | `phi1_rejection_sets_order_to_zero` | 70 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3124 | `phi1_rejection_before_contribution_analysis` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3163 | `phi1_zero_is_not_rejected` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3196 | `phi1_rejection_interacts_with_magnitude_bound` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3266 | `iterative_reduction_terminates_at_zero` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3324 | `iterative_reduction_only_affects_failing_seasons` | 60 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3386 | `iterative_pacf_reduction_with_synthetic_observations` | 58 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3446 | `fixed_path_uses_truncation_not_reselection` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3500 | `combined_strategies_produce_correct_reduction_reasons` | 108 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3647 | `seasonal_stats_to_rows_includes_prestudy_stages` | 70 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3719 | `ar_estimates_to_rows_includes_prestudy_stages` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3776 | `full_estimation_produces_prestudy_inflow_models` | 78 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 3980 | `iterative_pacf_reduction_stable_par2_not_spuriously_reduced` | 161 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 4148 | `roundtrip_estimation_two_season_par2_recovers_coefficients` | 113 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 4274 | `test_ar_rows_to_estimates_groups_by_season` | 135 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 4608 | `test_user_ar_estimation_preserves_ar_coefficients` | 54 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 4668 | `test_user_ar_estimation_estimates_stats_from_history` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 4708 | `test_user_ar_estimation_returns_user_provided_report` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 4919 | `test_partial_estimation_direction_a_missing_stats` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 4969 | `test_partial_estimation_direction_b_white_noise_fallback` | 62 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 5037 | `test_partial_estimation_exact_coverage_no_fallback` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 5063 | `test_full_estimation_report_has_empty_fallbacks` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 5174 | `test_std_ratio_divergence_fires_when_ratios_diverge` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 5204 | `test_std_ratio_divergence_not_fires_when_similar` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 5220 | `test_std_ratio_divergence_skips_near_zero_std` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/estimation.rs` | 5241 | `test_std_ratio_divergence_wraps_last_to_first` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 1922 | `forward_result_field_access` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 1937 | `forward_result_clone_and_debug` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 1959 | `forward_overhead_decomposition_four_workers` | 99 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2061 | `forward_overhead_decomposition_single_worker_zero_imbalance` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2099 | `forward_overhead_scheduling_clamped_to_zero_on_clock_skew` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2130 | `build_cut_row_batch_empty_cuts_returns_empty_batch` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2144 | `build_cut_row_batch_one_cut_correct_structure` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2159 | `build_cut_row_batch_two_cuts_correct_row_starts` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2186 | `build_cut_row_batch_zero_coefficient_state_variable` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2273 | `ac_two_scenarios_three_stages_fixed_solution` | 112 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2393 | `ac_infeasible_at_stage_1_scenario_0_returns_infeasible_error` | 111 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2508 | `ac_global_scenario_index_rank1_scenario0` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2524 | `cost_statistics_accumulated_correctly` | 109 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2638 | `sync_result_field_access` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2652 | `sync_result_clone_and_debug` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2676 | `ub_statistics_four_scenarios_correct_mean_and_std` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2717 | `ac_ticket_acceptance_criterion_ub_mean` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2745 | `canonical_summation_identical_regardless_of_partition` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2787 | `bessel_correction_single_scenario_zero_std_and_ci` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2811 | `negative_variance_guard_produces_zero_std_not_nan` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2851 | `sync_forward_local_backend_global_equals_local` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2873 | `sync_forward_sync_time_ms_is_valid_u64` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 2891 | `sync_forward_comm_error_wraps_as_sddp_communication` | 67 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 3069 | `warm_start_first_iteration_cold_second_iteration_warm` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 3110 | `basis_invalidated_on_solver_error` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 3156 | `test_forward_pass_parallel_cost_agreement` | 130 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 3301 | `test_forward_pass_work_distribution` | 114 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 3709 | `truncation_clamps_negative_inflow_noise` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 3745 | `truncation_no_clamp_when_inflow_positive` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 3781 | `none_method_unchanged_with_truncation_code_present` | 106 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4038 | `test_forward_pass_parallel_infeasibility` | 109 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4160 | `forward_pass_load_noise_positive_realization` | 123 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4291 | `forward_pass_load_noise_clamped_to_zero` | 119 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4418 | `forward_pass_no_load_buses_unchanged` | 85 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4581 | `append_new_cuts_returns_zero_when_no_new_cuts` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4605 | `append_new_cuts_appends_all_on_empty_row_map` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4635 | `append_new_cuts_skips_already_mapped_cuts` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4668 | `append_new_cuts_matches_build_cut_row_batch_into` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4705 | `append_new_cuts_with_scaling_matches_build` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4749 | `test_build_delta_empty_pool` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4766 | `test_build_delta_single_iteration_filter` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4793 | `test_build_delta_skips_deactivated_cuts` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4816 | `test_build_delta_excludes_warm_start_cuts` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4851 | `test_build_delta_matches_full_batch_when_pool_has_only_current_iter` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4877 | `test_build_delta_sparse_path` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4916 | `test_build_delta_reuses_out_buffer` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4939 | `test_build_delta_clears_row_starts` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 4973 | `forward_pass_baked_ready_skips_cut_batches_rebuild` | 88 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/forward.rs` | 5068 | `forward_pass_baked_not_ready_rebuilds_cut_batches` | 93 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1677 | `valid_five_point_curve_construction_succeeds` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1688 | `interpolation_at_midpoint_segment_0_to_2000` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1697 | `interpolation_at_breakpoints_returns_exact_values` | 13 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1714 | `height_clamped_below_v_min` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1724 | `height_clamped_above_v_max` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1734 | `derivative_first_segment_correct` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1743 | `derivative_last_segment_and_at_v_max` | 20 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1767 | `derivative_at_interior_breakpoint_uses_right_segment` | 12 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1783 | `derivative_clamped_below_v_min_returns_first_segment_slope` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1794 | `insufficient_points_zero_rows` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1807 | `insufficient_points_one_row` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1820 | `non_monotonic_volume_duplicate` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1841 | `non_monotonic_volume_decreasing` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1855 | `non_monotonic_height_decreasing` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1876 | `equal_consecutive_heights_accepted` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1885 | `display_insufficient_points_contains_name_and_count` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1893 | `display_non_monotonic_volume_contains_name_and_index` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1905 | `display_non_monotonic_height_contains_name_and_index` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1917 | `fpha_fitting_error_implements_std_error` | 4 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1925 | `tailrace_polynomial_constant_one_coefficient` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1934 | `tailrace_polynomial_linear_two_coefficients` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1943 | `tailrace_polynomial_quadratic_acceptance_criterion` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1952 | `tailrace_polynomial_quartic_five_coefficients` | 8 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1964 | `tailrace_polynomial_derivative_constant_is_zero` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1972 | `tailrace_polynomial_derivative_linear` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1980 | `tailrace_polynomial_derivative_quadratic_acceptance_criterion` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1989 | `tailrace_polynomial_derivative_quartic` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2020 | `tailrace_piecewise_midpoint_first_segment_acceptance_criterion` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2027 | `tailrace_piecewise_at_breakpoints_exact` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2035 | `tailrace_piecewise_clamp_below_range` | 4 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2041 | `tailrace_piecewise_clamp_above_range` | 4 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2049 | `tailrace_piecewise_derivative_first_segment_acceptance_criterion` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2056 | `tailrace_piecewise_derivative_second_segment` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2063 | `tailrace_piecewise_derivative_at_q_max_returns_last_segment_slope` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2070 | `tailrace_piecewise_derivative_clamp_above_returns_last_segment_slope` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2077 | `tailrace_piecewise_derivative_clamp_below_returns_first_segment_slope` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2086 | `losses_factor_acceptance_criterion` | 4 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2092 | `losses_factor_scales_with_gross_head` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2099 | `losses_factor_turbined_has_no_effect` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2109 | `losses_constant_acceptance_criterion` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2116 | `losses_constant_independent_of_all_inputs` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2127 | `losses_factor_extraction_returns_factor` | 4 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2133 | `losses_factor_extraction_constant_returns_zero` | 4 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2141 | `two_point_minimum_curve_works` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2155 | `interpolation_second_segment_correct` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2248 | `net_head_no_tailrace_no_losses_equals_h_fore` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2257 | `net_head_polynomial_tailrace_constant_losses_acceptance_criterion` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2278 | `net_head_piecewise_tailrace_factor_losses` | 22 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2302 | `net_head_clamped_to_zero_when_losses_exceed_forebay` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2325 | `evaluate_acceptance_criterion` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2348 | `partial_derivatives_no_tailrace_ds_is_zero` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2360 | `partial_derivatives_no_tailrace_dv_is_positive` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2385 | `partial_derivatives_polynomial_tailrace_constant_losses` | 38 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2425 | `partial_derivatives_polynomial_tailrace_constant_losses_dv_positive` | 15 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2442 | `partial_derivatives_polynomial_tailrace_ds_nonpositive` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2463 | `partial_derivatives_factor_losses_dv_accounts_for_k_factor` | 26 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2504 | `partial_derivatives_piecewise_tailrace_factor_losses` | 42 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2548 | `partial_derivatives_piecewise_tailrace_ds_negative` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2575 | `finite_difference_cross_check_dv` | 27 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2605 | `finite_difference_cross_check_dq` | 27 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2635 | `finite_difference_cross_check_ds` | 27 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2665 | `finite_difference_cross_check_all_derivatives_factor_losses` | 42 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2786 | `no_fitting_window_uses_forebay_defaults` | 14 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2806 | `absolute_bounds_both_set` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2829 | `absolute_bounds_only_min` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2852 | `absolute_bounds_only_max` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2876 | `percentile_bounds_both_set` | 20 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2901 | `mixed_absolute_min_percentile_max` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2925 | `conflicting_min_bound_returns_error` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2949 | `conflicting_max_bound_returns_error` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2973 | `inverted_absolute_bounds_returns_empty_window_error` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2995 | `equal_absolute_bounds_returns_empty_window_error` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3019 | `absolute_min_below_forebay_gets_clamped` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3041 | `absolute_max_above_forebay_gets_clamped` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3065 | `discretization_all_none_defaults_to_five` | 11 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3079 | `discretization_explicit_values_passed_through` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3102 | `volume_discretization_one_returns_error` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3124 | `volume_discretization_zero_returns_error` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3145 | `turbine_discretization_one_returns_error` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3166 | `spillage_discretization_one_returns_error` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3189 | `max_planes_per_hydro_none_defaults_to_ten` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3201 | `max_planes_per_hydro_explicit_value` | 12 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3216 | `max_planes_per_hydro_zero_returns_error` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3270 | `tangent_plane_at_known_operating_point_coefficients` | 40 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3313 | `tangent_plane_identity_at_operating_point` | 16 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3332 | `tangent_plane_identity_at_second_operating_point` | 16 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3351 | `tangent_plane_identity_with_spillage` | 16 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3370 | `compute_tangent_plane_zero_flow_returns_none` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3378 | `compute_tangent_plane_negative_flow_returns_none` | 5 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3389 | `compute_tangent_plane_zero_production_returns_none` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3410 | `raw_hyperplane_evaluate_linear_combination` | 15 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3428 | `gamma_v_positive_for_positive_net_head` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3441 | `gamma_s_nonpositive_with_tailrace` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3454 | `raw_hyperplane_implements_debug_clone_copy` | 16 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3506 | `sample_tangent_planes_count_between_100_and_125_for_5x5x5` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3519 | `sample_tangent_planes_count_at_most_n_v_times_n_q_times_n_s` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3540 | `sample_tangent_planes_flow_grid_avoids_zero_q` | 12 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3557 | `sample_tangent_planes_spillage_grid_starts_at_zero` | 12 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3574 | `eliminate_redundant_strictly_reduces_count_for_non_trivial_geometry` | 14 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3592 | `eliminate_redundant_envelope_upper_bounds_production_function` | 44 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3641 | `eliminate_redundant_constant_head_produces_one_plane` | 28 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3672 | `eliminate_redundant_empty_input_returns_empty` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3682 | `eliminate_redundant_spillage_planes_can_survive` | 15 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3704 | `eliminate_redundant_output_is_subset_of_input` | 24 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3773 | `select_planes_reduces_to_target_count` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3794 | `select_planes_approximation_error_not_catastrophically_worse` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3815 | `select_planes_passthrough_when_input_is_small` | 38 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3857 | `select_planes_preserves_envelope_property` | 42 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3902 | `select_planes_empty_input_returns_empty` | 6 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3911 | `select_planes_single_plane_returns_unchanged` | 23 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3942 | `compute_max_approximation_error_is_zero_for_linear_production_function` | 27 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3972 | `compute_max_approximation_error_empty_planes_returns_zero` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3984 | `compute_max_approximation_error_is_non_negative` | 8 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3995 | `select_planes_output_is_subset_of_input` | 13 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4016 | `compute_kappa_in_valid_range_for_realistic_geometry` | 11 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4038 | `compute_kappa_in_range_for_realistic_geometry` | 31 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4076 | `compute_kappa_is_one_for_linear_production_function` | 28 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4110 | `compute_kappa_less_than_one_for_nonlinear_production_function` | 19 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4132 | `compute_kappa_empty_planes_returns_one` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4146 | `validate_fitted_planes_valid_input_returns_ok` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4167 | `validate_fitted_planes_zero_kappa_returns_invalid_kappa` | 13 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4183 | `validate_fitted_planes_kappa_above_one_returns_invalid_kappa` | 13 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4199 | `validate_fitted_planes_negative_kappa_returns_invalid_kappa` | 13 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4215 | `validate_fitted_planes_empty_planes_returns_no_hyperplanes` | 7 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4225 | `validate_fitted_planes_negative_gamma_v_returns_invalid_coefficient` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4245 | `validate_fitted_planes_negative_gamma_q_returns_invalid_coefficient` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4265 | `validate_fitted_planes_positive_gamma_s_returns_invalid_coefficient` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4285 | `validate_fitted_planes_near_zero_gamma_v_within_tolerance_passes` | 13 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4301 | `validate_fitted_planes_near_zero_gamma_s_within_tolerance_passes` | 13 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4317 | `validate_fitted_planes_kappa_exactly_one_passes` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4382 | `fit_fpha_planes_sobradinho_style_end_to_end` | 42 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4430 | `fit_fpha_planes_intercepts_are_kappa_scaled` | 24 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4459 | `fit_fpha_planes_linear_function_produces_one_plane_with_kappa_one` | 77 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4539 | `fit_fpha_planes_propagates_forebay_error_on_insufficient_rows` | 23 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4566 | `display_invalid_kappa_contains_name_and_value` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4577 | `display_no_hyperplanes_produced_contains_name` | 10 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4589 | `display_invalid_coefficient_contains_name_and_index_and_detail` | 11 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4606 | `fit_fpha_planes_result_kappa_in_range_and_intercept_consistent` | 39 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 500 | `thermal_generation_block_id_none_at_block_1` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 531 | `thermal_generation_block_id_some_at_block_2` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 561 | `thermal_generation_second_thermal` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 593 | `hydro_storage_stage_level_ignores_block` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 643 | `hydro_outflow_expands_to_turbine_and_spillage` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 674 | `hydro_outflow_block_id_some_uses_explicit_block` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 709 | `hydro_generation_constant_productivity_maps_to_turbine` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 741 | `hydro_generation_fpha_maps_to_generation_column` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 773 | `hydro_generation_fpha_second_hydro_block_2` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 814 | `hydro_evaporation_maps_to_q_ev_col` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 873 | `hydro_evaporation_no_evap_model_returns_empty` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 935 | `pumping_flow_returns_empty` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 962 | `pumping_power_returns_empty` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 989 | `contract_import_returns_empty` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1016 | `non_controllable_generation_returns_empty` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1045 | `missing_entity_id_returns_empty` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1080 | `bus_deficit_returns_one_entry_per_segment` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1116 | `bus_deficit_second_bus_block_1` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1150 | `bus_excess_maps_to_excess_column` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1182 | `line_direct_maps_to_fwd_column` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1212 | `line_reverse_maps_to_rev_column` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1245 | `line_exchange_maps_to_fwd_and_rev_columns` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1275 | `line_exchange_with_explicit_block` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1302 | `line_exchange_unknown_id_returns_empty` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1331 | `hydro_turbined_maps_to_turbine_column` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/generic_constraints.rs` | 1359 | `hydro_spillage_maps_to_spillage_column` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 203 | `successors_mid_stage_returns_next` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 210 | `successors_first_stage_returns_second` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 216 | `successors_terminal_stage_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 222 | `successors_beyond_terminal_returns_empty` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 229 | `successors_all_non_terminal_stages` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 238 | `successors_consistent_with_is_terminal` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 253 | `is_terminal_last_stage_is_true` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 260 | `is_terminal_preceding_stage_is_false` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 266 | `is_terminal_first_stage_is_false` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 272 | `is_terminal_single_stage_is_terminal` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 281 | `validate_accepts_two_or_more_stages` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 289 | `validate_rejects_one_stage` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 300 | `validate_rejects_zero_stages` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 310 | `validate_error_message_contains_stage_count` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 323 | `num_stages_returns_field_value` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 329 | `num_stages_single` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 337 | `debug_output_contains_variant_name` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/horizon_mode.rs` | 345 | `clone_produces_equal_num_stages` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 1872 | `all_constant_no_config_returns_default_constant_provenance` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 1914 | `linearized_head_entity_resolves_to_constant_productivity` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 1945 | `fpha_entity_without_config_entry_returns_validation_error` | 9 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 1957 | `computed_source_returns_computed_from_geometry` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2000 | `computed_source_missing_tailrace_returns_validation_error` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2024 | `computed_source_missing_geometry_returns_validation_error` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2044 | `find_fpha_config_for_stage_returns_config_in_range` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2062 | `find_fpha_config_for_stage_returns_none_outside_range` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2095 | `gamma_0_is_scaled_by_kappa` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2143 | `validation_rejects_gamma_v_negative` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2160 | `validation_accepts_gamma_v_zero` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2173 | `validation_rejects_gamma_s_positive` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2190 | `validation_rejects_gamma_q_nonpositive` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2207 | `validation_rejects_kappa_zero` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2224 | `validation_rejects_kappa_above_one` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2241 | `stage_specific_hyperplanes_override_all_stage` | 61 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2305 | `all_stage_hyperplanes_used_when_no_stage_specific_rows` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2351 | `zero_hyperplanes_for_stage_returns_validation_error` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2378 | `mixed_system_model_returns_correct_variant_for_all_pairs` | 59 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2440 | `find_model_for_stage_returns_correct_model_name_in_range` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2449 | `find_model_for_stage_returns_none_when_before_range_start` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2472 | `find_model_for_stage_open_ended_range_covers_all_stages` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2500 | `resolve_stage_model_uses_productivity_override` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2539 | `resolve_stage_model_uses_entity_productivity_when_no_override` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2578 | `find_model_for_stage_returns_override_in_tuple` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2602 | `find_model_for_stage_seasonal_with_override` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2633 | `precomputed_config_returns_precomputed_source` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2643 | `fpha_plane_is_copy` | 11 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2656 | `linearized_evaporation_is_copy` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2666 | `all_types_implement_debug` | 80 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2750 | `production_model_set_model_returns_correct_variant` | 73 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2826 | `production_model_set_out_of_bounds_hydro_panics_in_debug` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2851 | `evaporation_model_set_has_evaporation_true_when_any_linearized` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2884 | `evaporation_model_set_has_evaporation_false_when_all_none` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2898 | `evaporation_model_set_model_returns_correct_variant` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2935 | `evaporation_model_set_empty_has_no_evaporation` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2960 | `interpolate_area_exact_first_point` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2972 | `interpolate_area_exact_last_point` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2984 | `interpolate_area_exact_middle_point` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2999 | `interpolate_area_midpoint_between_two_points` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3018 | `interpolate_area_clamps_below_first_point` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3030 | `interpolate_area_clamps_above_last_point` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3047 | `area_derivative_correct_finite_difference` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3066 | `area_derivative_single_point_returns_zero` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3078 | `area_derivative_at_or_below_first_point_uses_first_interval` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3091 | `area_derivative_at_or_above_last_point_uses_last_interval` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3172 | `resolve_evaporation_all_none_when_no_hydro_has_coefficients` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3220 | `resolve_evaporation_known_geometry_produces_correct_coefficients` | 69 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3294 | `resolve_evaporation_negative_coefficient_produces_valid_results` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3338 | `resolve_evaporation_missing_geometry_returns_validation_error` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3367 | `resolve_evaporation_mixed_system_returns_correct_model_mix` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3428 | `resolve_evaporation_degenerate_geometry_nan_detected` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3473 | `resolve_evaporation_per_season_ref_vols_produces_per_stage_coefficients` | 109 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3586 | `resolve_evaporation_none_ref_vols_produces_default_midpoint_provenance` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3647 | `resolve_evaporation_mixed_ref_vol_provenance` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 3689 | `build_hydro_model_summary_ref_source_counts` | 81 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4001 | `build_hydro_model_summary_all_constant` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4039 | `build_hydro_model_summary_mixed_counts_and_plane_total` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4080 | `build_hydro_model_summary_acceptance_criterion` | 119 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4203 | `build_hydro_model_summary_evaporation_counts_from_provenance` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4286 | `computed_source_end_to_end_produces_valid_fpha_planes` | 71 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4364 | `mixed_precomputed_and_computed_sources_resolve_correctly` | 93 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4461 | `computed_source_all_stages_produce_identical_planes` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4533 | `computed_source_in_summary_counts_correctly` | 65 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4602 | `computed_source_missing_efficiency_returns_validation_error` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 4628 | `computed_source_missing_losses_returns_validation_error` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1240 | `storage_range_3_2` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1245 | `inflow_lags_range_3_2` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1251 | `z_inflow_range_3_2` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1257 | `storage_in_range_3_2` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1263 | `theta_index_3_2` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1269 | `n_state_3_2` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1275 | `storage_fixing_range_3_2` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1280 | `lag_fixing_range_3_2` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1286 | `row_column_symmetry_3_2` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1299 | `n_state_production_scale` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1305 | `theta_production_scale` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1311 | `row_column_symmetry_production_scale` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1320 | `single_hydro_no_lags` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1348 | `degenerate_zero_hydros` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1393 | `from_stage_template_matches_new_3_2` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1410 | `from_stage_template_matches_new_160_12` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1422 | `from_stage_template_matches_new_edge_cases` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1438 | `clone_and_debug` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1450 | `new_equipment_ranges_are_empty` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1479 | `with_equipment_doctest_n1_l0_t2_l1_b2_k1` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1520 | `with_equipment_n2_l1_t3_l2_b4_k2` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1540 | `with_equipment_all_counts_zero_matches_new` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1563 | `with_equipment_ranges_are_contiguous` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1580 | `with_equipment_column_index_formulas` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1610 | `with_equipment_inflow_penalty_appends_slack` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1651 | `fpha_no_hydros_generation_is_empty` | 11 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1685 | `fpha_one_hydro_one_block_three_planes` | 18 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1716 | `fpha_two_hydros_two_blocks_different_planes` | 30 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1752 | `fpha_generation_contiguous_with_prior_region` | 17 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1772 | `fpha_rows_contiguous_with_load_balance` | 21 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1798 | `evap_no_hydros_indices_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1827 | `evap_one_hydro_column_row_positions` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1872 | `evap_two_hydros_with_fpha_contiguous` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1904 | `new_evap_ranges_are_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1929 | `withdrawal_slack_with_equipment_and_evaporation_n3_evap1` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1951 | `withdrawal_slack_zero_hydros_is_empty` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1965 | `withdrawal_slack_from_new_is_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1974 | `withdrawal_slack_length_equals_hydro_count` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2012 | `withdrawal_slack_immediately_after_evap_columns` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2035 | `evap_indices_debug_clone_copy` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2052 | `fpha_row_range_debug_clone_copy` | 11 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2066 | `new_fpha_ranges_are_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2076 | `extended_adjacency_invariant_with_fpha` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2102 | `test_diversion_range_n3_l0_k2` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2112 | `test_diversion_zero_hydros` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2122 | `z_inflow_range_new_constructor` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2131 | `z_inflow_range_zero_hydros` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2140 | `z_inflow_row_fields` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2150 | `z_inflow_range_with_equipment` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2162 | `z_inflow_single_hydro_no_lags` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2172 | `nonzero_mask_default_is_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2181 | `nonzero_mask_mixed_ar_orders` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2212 | `nonzero_mask_zero_par_order` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2221 | `nonzero_mask_all_full_order` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 138 | `none_has_no_slack_columns` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 143 | `truncation_has_no_slack_columns` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 148 | `penalty_has_slack_columns` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 155 | `penalty_cost_for_penalty_variant` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 163 | `penalty_cost_none_for_none_variant` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 168 | `truncation_penalty_cost_is_none` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 175 | `test_inflow_method_conversion_none` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 187 | `test_inflow_method_conversion_penalty` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 199 | `test_inflow_method_conversion_truncation` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 211 | `test_truncation_ignores_penalty_cost` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 223 | `test_inflow_method_conversion_unknown_falls_back_to_none` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 235 | `test_penalty_config_propagation` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 250 | `truncation_with_penalty_has_slack_columns` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 257 | `truncation_with_penalty_cost` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/inflow_method.rs` | 265 | `test_inflow_method_conversion_truncation_with_penalty` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 606 | `test_uniform_monthly_identity` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 655 | `test_pmo_apr_2026_rv0_trace` | 99 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 768 | `test_boundary_straddling_week` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 801 | `test_no_season_id_produces_noop` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 820 | `test_single_stage_per_month_finalizes` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 844 | `test_multiple_weekly_stages_only_last_finalizes` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 963 | `test_seed_empty_observations_returns_zero` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 980 | `test_seed_one_observation_one_hydro` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1009 | `test_seed_two_observations_same_hydro_additive` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1037 | `test_seed_two_observations_different_hydros_independent` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1077 | `test_seed_weight_independent_of_hydro_count` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1108 | `test_seed_unknown_hydro_id_silently_skipped` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1127 | `test_seed_no_season_id_returns_zero` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1140 | `test_noise_groups_monthly_unique` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1163 | `test_noise_groups_weekly_shared` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1194 | `test_noise_groups_mixed_weekly_monthly` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1219 | `test_noise_groups_none_season_id` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lag_transition.rs` | 1238 | `test_noise_groups_cross_year` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 679 | `one_opening_expectation_lb_equals_single_objective` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 726 | `three_openings_expectation_lb_equals_mean` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 775 | `two_openings_pure_cvar_alpha_half_lb_equals_worst` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 831 | `two_openings_cvar_alpha_one_equals_expectation` | 48 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 882 | `infeasible_solve_maps_to_sddp_infeasible` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 928 | `broadcast_failure_maps_to_communication_error` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 980 | `integration_two_openings_local_backend_expectation` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 1033 | `integration_monotonicity_more_cuts_yields_higher_or_equal_lb` | 60 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 1102 | `test_lb_none_method_unchanged` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 1154 | `test_lb_truncation_no_crash` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lower_bound.rs` | 1200 | `test_lb_truncation_with_penalty_no_crash` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 559 | `new_3_2_sizes_to_15` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 568 | `new_160_12_sizes_to_2400` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 577 | `new_zero_lags_sizes_to_3n` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 584 | `new_zero_hydros_sizes_to_zero` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 590 | `forward_patch_count_without_z_inflow_fill` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 598 | `state_patch_count_is_n_times_one_plus_l` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 605 | `state_patch_count_zero_lags` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 612 | `ar_dynamics_row_offset_adds_base_plus_hydro` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 619 | `ar_dynamics_row_offset_zero_base` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 624 | `fill_forward_patches_category1_indices` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 637 | `fill_forward_patches_category2_indices` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 656 | `fill_forward_patches_category3_indices` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 670 | `fill_forward_patches_category1_values` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 686 | `fill_forward_patches_category2_values` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 710 | `fill_forward_patches_category3_values` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 726 | `fill_forward_patches_all_equality_constraints` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 745 | `fill_state_patches_count_is_n_times_one_plus_l` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 756 | `fill_state_patches_category1_correct` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 773 | `fill_state_patches_category2_correct` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 788 | `fill_state_patches_equality_constraints_in_active_range` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 806 | `forward_patches_zero_lags_only_storage_and_noise` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 833 | `state_patches_zero_lags_only_storage` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 851 | `production_scale_forward_patch_count` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 860 | `production_scale_fill_forward_patches_smoke` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 887 | `clone_and_debug` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 902 | `new_with_load_allocates_correct_capacity` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 918 | `fill_load_patches_correct_indices` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 933 | `fill_load_patches_correct_values` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 951 | `fill_load_patches_equality_constraints` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 977 | `forward_patch_count_includes_load` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 992 | `state_patch_count_excludes_load` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/patch.rs` | 1007 | `zero_load_buses_no_category4` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/scaling.rs` | 273 | `row_scale_identity_for_uniform_matrix` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/scaling.rs` | 305 | `row_scale_geometric_mean` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/scaling.rs` | 330 | `apply_row_scale_scales_values_and_bounds` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/scaling.rs` | 386 | `row_scale_empty_row_gets_one` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 952 | `empty_stages_returns_empty` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 969 | `one_stage_one_template` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 986 | `num_cols_formula_no_hydro_no_thermal_no_line` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1007 | `num_cols_formula_one_hydro_lag_zero` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1031 | `num_cols_formula_one_hydro_lag_two` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1055 | `num_rows_formula_no_hydro` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1074 | `num_rows_formula_one_hydro_lag_zero` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1096 | `num_rows_formula_one_hydro_lag_two` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1118 | `n_state_matches_indexer` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1136 | `n_transfer_is_n_times_lag_order` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1153 | `n_dual_relevant_equals_n_state_for_constant_productivity` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1173 | `base_row_is_n_dual_relevant_plus_n_hydros` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1196 | `csc_col_starts_monotone_nondecreasing` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1217 | `csc_row_indices_in_range` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1241 | `csc_nz_count_matches_col_starts` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1269 | `theta_column_has_unit_objective` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1291 | `spillage_objective_nonzero_for_nonzero_penalty` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1546 | `fpha_turbined_cost_applied_to_fpha_turbine_column` | 27 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1575 | `constant_hydro_turbine_column_has_zero_objective` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1600 | `fpha_turbined_cost_multi_block_uses_per_block_hours` | 34 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1636 | `fpha_turbined_cost_mixed_system_only_fpha_hydros_carry_cost` | 98 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1736 | `load_balance_rhs_matches_load_model_mean_mw` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1762 | `multiple_stages_produce_same_count_templates_and_base_rows` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1779 | `stage_templates_clone_and_debug` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1804 | `test_fpha_model_accepted` | 190 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1999 | `test_constant_productivity_accepted` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2027 | `test_penalty_columns_added` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2056 | `test_penalty_columns_added_3_hydros` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2091 | `test_penalty_objective_coefficient` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2120 | `test_no_penalty_columns_when_none` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2150 | `test_penalty_slack_in_water_balance` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2189 | `test_penalty_slack_bounds` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2222 | `test_penalty_water_balance_coefficient_value` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2264 | `test_penalty_multi_stage_consistent` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2309 | `test_penalty_slack_absorbs_negative_inflow` | 73 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2602 | `stage_templates_load_balance_row_starts_correct` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2639 | `stage_templates_n_load_buses_matches_stochastic_buses` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2670 | `stage_templates_no_load_buses_gives_zero` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3139 | `fpha_ac1_dimensions_one_fpha_hydro_five_planes` | 39 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3200 | `fpha_ac2_generation_column_entries` | 44 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3250 | `fpha_ac3_v_in_column_entries` | 32 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3288 | `fpha_ac4_v_out_column_entries` | 31 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3335 | `fpha_ac5_mixed_system_load_balance_uses_generation_col` | 79 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3482 | `fpha_solve_one_hydro_optimal` | 45 | `tests` | `unit` | `add-rows-trait,fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3537 | `fpha_solve_hyperplane_constraints_hold` | 71 | `tests` | `unit` | `add-rows-trait,fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3647 | `fpha_solve_storage_fixing_dual_differs_from_constant` | 95 | `tests` | `unit` | `add-rows-trait,fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3756 | `fpha_solve_mixed_system_optimal` | 62 | `tests` | `unit` | `add-rows-trait,fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3871 | `evap_zero_hydros_layout_unchanged` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3914 | `evap_two_hydros_increases_cols_and_rows` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3965 | `evap_row_bounds_equality_at_k_evap0` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3999 | `evap_col_bounds_and_objective` | 54 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4110 | `evap_csc_entries_one_hydro_correct_coefficients` | 121 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4234 | `evap_csc_entries_coefficient_scaling` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4279 | `evap_csc_entries_zero_hydros_no_op` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4311 | `evap_csc_entries_two_hydros_independent_rows` | 71 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4386 | `evap_csc_entries_zero_k_evap_v_produces_zero_volume_coefficients` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4434 | `evap_water_balance_one_hydro_coefficient_is_zeta` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4482 | `evap_water_balance_only_second_hydro_has_evap` | 210 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4699 | `evap_water_balance_zero_hydros_no_op` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4947 | `evap_violation_cost_applied_to_slack_columns` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 4993 | `evap_q_ev_objective_is_zero` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5026 | `evap_lp_solvable_and_q_ev_nonnegative` | 47 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5082 | `evap_violation_slacks_near_zero_feasible_constraint` | 52 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5143 | `evap_storage_fixing_dual_differs_from_no_evaporation` | 68 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5217 | `evap_bound_prevents_dump_valve` | 84 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5408 | `test_multi_segment_deficit_column_count` | 173 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5586 | `test_multi_segment_deficit_bounds_and_objective` | 61 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5651 | `test_single_segment_backward_compat` | 48 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5714 | `test_multi_segment_deficit_load_balance_coefficients` | 75 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6030 | `withdrawal_rhs_subtracted_from_water_balance` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6071 | `withdrawal_zero_leaves_rhs_unchanged_from_base` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6124 | `withdrawal_slack_matrix_entry_coefficient_is_minus_zeta` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6167 | `withdrawal_slack_objective_equals_cost_times_hours` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6194 | `withdrawal_slack_objective_zero_when_cost_is_zero` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6220 | `withdrawal_slack_bounds_are_zero_to_infinity` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6252 | `two_hydro_withdrawal_slack_entries_per_hydro` | 264 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6524 | `three_hydro_num_cols_includes_three_withdrawal_slacks` | 245 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6925 | `generic_constraints_zero_does_not_change_layout` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6949 | `generic_constraint_no_slack_block_id_none_3_blocks` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 6983 | `generic_constraint_le_slack_enabled_2_blocks` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7017 | `generic_constraint_equal_sense_two_slacks_per_row` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7051 | `generic_constraint_specific_block_id_generates_one_row` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7084 | `generic_constraint_inactive_does_not_contribute_rows` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7114 | `stage_indexer_generic_fields_empty_from_new` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7273 | `generic_constraint_thermal_le_row_bounds_and_csc_entry` | 71 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7351 | `generic_constraint_thermal_le_slack_column_and_csc_entry` | 81 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7438 | `generic_constraint_thermal_ge_row_bounds` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7494 | `generic_constraint_thermal_equal_two_slacks` | 91 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 7592 | `generic_constraint_two_hydros_sum_csc_entries` | 278 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8254 | `operational_violation_row_col_counts` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8301 | `min_outflow_active_col_bounds` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8332 | `max_outflow_active_col_bounds` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8362 | `operational_violation_inactive_pinned` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8408 | `operational_violation_objective_costs` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8452 | `min_outflow_row_bounds` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8493 | `max_outflow_row_bounds` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8533 | `min_turbine_row_bounds` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8573 | `min_generation_row_bounds` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8613 | `min_outflow_matrix_coefficients` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8666 | `max_outflow_matrix_slack_is_negative` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8701 | `min_turbine_matrix_only_turbine_cols` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8754 | `min_generation_constant_productivity_coefficients` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8800 | `turbine_column_lower_bound_is_zero` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8834 | `operational_violation_rows_outside_dual_relevant` | 54 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 8895 | `diagnostic_template_operational_violation_correctness` | 107 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 870 | `test_transform_inflow_noise_none_method` | 63 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 941 | `test_transform_inflow_noise_truncation_clamps` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1012 | `test_transform_inflow_noise_truncation_passthrough` | 66 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1086 | `test_transform_load_noise_basic` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1113 | `test_transform_load_noise_clamped_non_negative` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1135 | `shift_lag_state_par0_is_noop` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1149 | `shift_lag_state_par1_single_hydro` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1162 | `shift_lag_state_par3_single_hydro` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1178 | `shift_lag_state_par1_two_hydros` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1193 | `shift_lag_state_preserves_storage` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1209 | `test_compute_effective_eta_none_passes_through` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1226 | `test_compute_effective_eta_penalty_passes_through` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1243 | `test_compute_effective_eta_truncation_clamps_negative` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1263 | `test_compute_effective_eta_truncation_passes_positive` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1281 | `test_compute_effective_eta_truncation_with_penalty_clamps` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1325 | `test_accumulate_monthly_identity` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1386 | `test_accumulate_four_weeks_then_finalize` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1446 | `test_accumulate_spillover_seeds_next_period` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1502 | `test_accumulate_noop_for_par0` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1550 | `test_accumulate_preserves_storage` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1668 | `test_downstream_par1_accumulation_and_rebuild` | 96 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1772 | `test_downstream_par2_two_quarters` | 118 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1898 | `test_no_downstream_for_uniform_monthly` | 78 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 1982 | `test_rebuild_resets_downstream_state` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 2044 | `test_downstream_spillover_seeds_next_quarter` | 61 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/noise.rs` | 2112 | `test_downstream_multi_hydro` | 97 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 288 | `load_boundary_cuts_valid_stage` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 314 | `load_boundary_cuts_missing_stage_returns_error` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 336 | `load_boundary_cuts_state_dimension_mismatch_returns_error` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 353 | `load_boundary_cuts_nonexistent_path_returns_error` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 383 | `compatible_metadata_passes` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 389 | `state_dimension_mismatch_fails` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 400 | `num_stages_mismatch_fails` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 411 | `both_dimensions_mismatched_returns_err` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 454 | `resolve_warm_start_counts_new_format_returns_per_stage_counts` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 461 | `resolve_warm_start_counts_old_format_broadcasts_scalar` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 469 | `resolve_warm_start_counts_old_format_zero_scalar_broadcasts_zeros` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 476 | `resolve_warm_start_counts_wrong_length_returns_validation_error` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 491 | `resolve_warm_start_counts_single_stage_new_format` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/policy_load.rs` | 498 | `resolve_warm_start_counts_zero_stages_old_format_returns_empty` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 278 | `deterministic_path_both_na` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 305 | `user_stats_white_noise_path` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 327 | `user_provided_no_history_path` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 346 | `full_estimation_path` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 368 | `user_ar_history_stats_path` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 388 | `partial_estimation_path` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 409 | `user_provided_all_path` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 434 | `user_supplied_tree_maps_to_user_file` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 450 | `full_estimation_json_round_trip` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 475 | `deterministic_json_na_variant` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 494 | `provenance_source_display` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 503 | `white_noise_fallbacks_propagated_as_raw_ids` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/provenance.rs` | 519 | `no_estimation_report_yields_empty_fallbacks` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 397 | `expectation_aggregate_cut_equal_probs_mean_intercept` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 412 | `expectation_aggregate_cut_nonuniform_probs` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 428 | `expectation_aggregate_cut_coefficients_weighted` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 441 | `expectation_evaluate_risk_equal_probs` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 449 | `expectation_evaluate_risk_nonuniform_probs` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 461 | `cvar_evaluate_risk_pure_cvar_alpha_half` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 473 | `cvar_evaluate_risk_alpha_one_equals_expectation` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 489 | `cvar_evaluate_risk_lambda_zero_equals_expectation` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 505 | `cvar_evaluate_risk_convex_combination` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 517 | `cvar_aggregate_cut_pure_cvar_selects_worst` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 534 | `cvar_aggregate_cut_with_coefficients` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 552 | `cvar_aggregate_cut_alpha_one_equals_expectation` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 574 | `cvar_aggregate_cut_lambda_zero_equals_expectation` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 596 | `cvar_aggregate_cut_weights_sum_to_one` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 633 | `risk_measure_debug_and_clone` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 645 | `backward_outcome_debug_and_clone` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 658 | `test_from_stage_risk_config_expectation` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 665 | `test_from_stage_risk_config_cvar` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 681 | `aggregate_weighted_into_matches_aggregate_weighted` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 718 | `aggregate_cut_into_matches_aggregate_cut_expectation` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/risk_measure.rs` | 743 | `aggregate_cut_into_matches_aggregate_cut_cvar` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 231 | `test_compute_abs_range_basic` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 239 | `test_compute_abs_range_all_zero` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 247 | `test_compute_abs_range_empty` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 254 | `test_median_odd` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 259 | `test_median_even` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 264 | `test_median_single` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 269 | `test_median_empty` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 274 | `test_summarize_scale_factors` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 284 | `test_summarize_scale_factors_empty` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 291 | `test_coefficient_range_from_template` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/scaling_report.rs` | 322 | `test_build_scaling_report_summary` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1144 | `new_minimal_valid_system_returns_ok` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1176 | `new_zero_stages_returns_validation_error` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1211 | `accessor_methods_return_expected_values` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1269 | `fcf_mut_allows_cut_insertion` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1303 | `inflow_method_reflects_config` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1340 | `cut_selection_none_when_disabled` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1373 | `stage_ctx_fields_match_study_setup` | 54 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1429 | `training_ctx_fields_match_study_setup` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1478 | `train_completes_within_iteration_limit` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1528 | `train_generates_cuts_in_fcf` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1572 | `simulation_config_reflects_setup_fields` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1616 | `create_workspace_pool_returns_correct_size` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1655 | `build_training_output_non_empty` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1711 | `simulate_after_train_returns_nonempty_costs` | 72 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1787 | `study_params_from_config_defaults` | 73 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 1864 | `study_params_from_config_explicit` | 78 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2000 | `prepare_stochastic_no_history_no_tree_returns_none_report_and_generated_provenance` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2039 | `prepare_stochastic_with_stats_file_present_skips_estimation` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2084 | `prepare_stochastic_no_opening_tree_gives_non_user_supplied_provenance` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2119 | `test_prepare_stochastic_historical_residuals_noise_method` | 250 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2378 | `default_from_system_gives_constant_and_no_evaporation` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2426 | `hydro_models_accessor_returns_stored_result` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2720 | `build_initial_state_populates_lags_from_past_inflows` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2763 | `build_initial_state_empty_past_inflows_leaves_zero_lags` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2784 | `build_initial_state_unknown_hydro_in_past_inflows_stays_zero` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2816 | `study_setup_initial_state_has_nonzero_lags_from_past_inflows` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2875 | `build_initial_state_no_lags_state_is_storage_only` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 2896 | `historical_library_none_for_insample` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 3191 | `historical_library_built_when_scheme_is_historical` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 3236 | `external_inflow_library_built_when_scheme_is_external` | 259 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 3507 | `external_load_library_built_when_scheme_is_external` | 254 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 3773 | `external_ncs_library_built_when_scheme_is_external` | 282 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 4067 | `historical_library_fails_when_no_valid_windows` | 243 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 4323 | `test_simulate_uses_simulation_scheme` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/setup/mod.rs` | 4381 | `test_sim_historical_library_built_when_sim_scheme_is_historical` | 48 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 439 | `local_min_max_basic` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 448 | `local_min_max_multiple` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 460 | `local_min_max_empty_returns_infinities` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 469 | `mean_std_five_costs` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 483 | `mean_std_single_scenario_yields_zero_std` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 491 | `mean_std_empty_yields_zeros` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 500 | `cvar_five_scenarios_alpha_095` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 510 | `cvar_single_scenario_equals_cost` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 517 | `cvar_empty_returns_zero` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 523 | `cvar_100_scenarios_alpha_095` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 535 | `pack_category_costs_layout` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 554 | `pack_category_costs_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 562 | `aggregate_basic_three_scenarios_mean_min_max` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 582 | `aggregate_cvar_five_scenarios` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 596 | `aggregate_single_scenario_std_zero_cvar_equals_cost` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 611 | `aggregate_category_stats_frequency` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 646 | `aggregate_category_stats_mean_max` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 694 | `aggregate_category_names_in_order` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 710 | `aggregate_operational_stats_are_zero_placeholders` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 722 | `aggregate_stage_stats_is_none` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 732 | `aggregate_cvar_100_scenarios` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/aggregation.rs` | 746 | `aggregate_std_five_costs_bessel_corrected` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/config.rs` | 52 | `simulation_config_construction` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/config.rs` | 62 | `simulation_config_arbitrary_values` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/config.rs` | 72 | `simulation_config_debug_non_empty` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 93 | `simulation_error_is_send_sync_static` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 98 | `simulation_error_lp_infeasible_display` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 111 | `simulation_error_solver_error_display` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 124 | `simulation_error_io_error_display` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 133 | `simulation_error_policy_incompatible_display` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 142 | `simulation_error_channel_closed_display` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 153 | `simulation_error_satisfies_std_error_trait` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 179 | `from_simulation_error_to_sddp_error` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/error.rs` | 217 | `sddp_error_simulation_variant_display` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1361 | `assign_scenarios_uneven_rank0` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1368 | `assign_scenarios_uneven_rank2` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1374 | `assign_scenarios_single_rank` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1380 | `assign_scenarios_uneven_rank1` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1387 | `assign_scenarios_exact_division` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1395 | `assign_scenarios_zero_scenarios` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1403 | `assign_scenarios_more_ranks_than_scenarios` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1413 | `assign_scenarios_sum_equals_n_scenarios` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1470 | `extract_costs_has_one_entry_matching_stage_id` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1511 | `extract_cost_splits_objective_correctly` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1554 | `extract_hydro_storage_values_from_primal` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1599 | `extract_inflow_lag_values_from_primal` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1644 | `extract_no_lags_when_max_par_order_zero` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1693 | `extract_stage_id_propagates_to_all_results` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1736 | `extract_equipment_zero_when_indexer_has_no_equipment_ranges` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1805 | `extract_equipment_reads_primal_when_with_equipment` | 168 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 1975 | `extract_optional_entity_types_are_empty_when_absent` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2085 | `accumulate_single_stage_all_categories` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2104 | `accumulate_two_consecutive_stages_sums_correctly` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2131 | `accumulate_all_zeros_leaves_accum_unchanged` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2152 | `accumulate_violation_all_five_components` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2167 | `accumulate_regularization_all_four_components` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2193 | `test_slack_extraction_with_penalty_active` | 103 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2300 | `test_slack_extraction_without_penalty_is_zero` | 74 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2378 | `test_slack_extraction_fallback_path_with_penalty` | 96 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2512 | `fpha_generation_read_from_lp_column` | 82 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2598 | `fpha_productivity_is_none` | 56 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2690 | `evaporation_read_from_lp_column` | 72 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2767 | `evaporation_violation_is_sum_of_slacks` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2842 | `fpha_turbined_cost_in_compute_cost_result` | 63 | `tests` | `unit` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2935 | `cost_breakdown_sums_to_immediate_identity_scale` | 79 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 3022 | `cost_unscaled_by_col_scale` | 66 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 3092 | `hydro_violation_cost_decomposition` | 152 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 1643 | `simulate_single_rank_4_scenarios_produces_4_results` | 96 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 1752 | `simulate_infeasible_returns_lp_infeasible_error` | 94 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 1852 | `simulate_infeasible_at_scenario2_stage3` | 94 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 1949 | `simulate_channel_closed_returns_error` | 88 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2044 | `simulate_total_cost_equals_sum_of_stage_costs` | 97 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2147 | `simulate_cost_buffer_scenario_ids_match_assigned_range` | 95 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2245 | `simulate_channel_receives_results_in_scenario_order` | 85 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2336 | `test_simulation_parallel_cost_determinism` | 208 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2551 | `simulate_emits_progress_events` | 117 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2672 | `simulate_no_events_when_sender_is_none` | 89 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2771 | `simulate_progress_events_received_before_return` | 103 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2882 | `simulate_progress_scenario_cost_equals_total_cost` | 112 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 2998 | `simulate_emits_simulation_finished_as_last_event` | 122 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 3126 | `simulate_progress_scenario_cost_is_finite` | 103 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 3387 | `simulation_load_patches_applied` | 199 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 3593 | `simulation_no_load_buses_unchanged` | 96 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 3698 | `simulation_inflow_extraction_unaffected` | 163 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 4091 | `simulation_truncation_clamps_negative_inflow_noise` | 105 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 4204 | `simulation_none_method_produces_raw_negative_noise` | 103 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 4314 | `simulate_baked_path_issues_zero_add_rows` | 98 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 4433 | `simulate_fallback_path_issues_expected_add_rows` | 99 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 4537 | `simulate_baked_length_mismatch_returns_error` | 94 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 4650 | `simulate_with_captured_basis_preserves_row_statuses` | 168 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/pipeline.rs` | 4825 | `simulate_with_empty_stage_bases_cold_starts` | 103 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 546 | `cost_result_construction_all_fields` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 599 | `hydro_result_optional_fields` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 641 | `thermal_result_gnl_fields_nullable` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 674 | `exchange_result_construction` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 695 | `bus_result_construction` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 716 | `pumping_result_construction` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 737 | `contract_result_construction` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 758 | `non_controllable_result_construction` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 781 | `inflow_lag_result_construction` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 796 | `generic_violation_result_construction` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 813 | `stage_result_empty_optional_vecs` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 836 | `scenario_result_is_send` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 843 | `scenario_result_with_multiple_stages` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 878 | `category_costs_construction` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 895 | `category_cost_stats_construction` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 910 | `stage_summary_stats_construction` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 925 | `simulation_summary_construction` | 48 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/types.rs` | 975 | `simulation_summary_optional_stage_stats` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 395 | `test_from_snapshots_all_deltas` | 69 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 466 | `test_from_snapshots_zero_delta` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 510 | `test_aggregate_empty_returns_default` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 517 | `test_aggregate_sums_all_fields` | 71 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 590 | `test_aggregate_solver_statistics_sums_all_fields` | 74 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 693 | `test_pack_unpack_delta_scalars_round_trip` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 724 | `test_pack_unpack_delta_scalars_identity_for_lp_solves_600` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 733 | `test_pack_unpack_scenario_stats_round_trip_three_entries` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 763 | `test_pack_scenario_stats_empty_round_trip` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/solver_stats.rs` | 771 | `test_solver_stats_delta_includes_reconstruction_fields` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stage_solve.rs` | 431 | `run_stage_solve_cold_start_returns_outcome` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stage_solve.rs` | 469 | `run_stage_solve_warm_start_excess_basic_demotes` | 63 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stage_solve.rs` | 538 | `run_stage_solve_propagates_infeasible` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stage_solve.rs` | 586 | `basis_inconsistent_propagates_as_sddp_solver_error` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 404 | `new_allocates_correct_send_buf_length` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 411 | `new_allocates_correct_recv_buf_length` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 418 | `new_allocates_correct_counts_length_and_values` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 427 | `new_allocates_correct_displs_length_and_values` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 436 | `new_single_rank_counts_is_one_element` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 444 | `total_scenarios_returns_local_count_times_num_ranks` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 450 | `total_scenarios_single_rank` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 456 | `state_at_indexing_arithmetic` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 493 | `exchange_single_rank_three_scenarios_two_state` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 513 | `exchange_selects_correct_stage_in_multi_stage_records` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 545 | `state_at_matches_record_state_after_exchange` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 565 | `exchange_error_maps_to_sddp_communication_error` | 65 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 634 | `real_total_scenarios_uneven_distribution` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 643 | `real_total_scenarios_even_distribution` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 651 | `real_total_scenarios_single_rank` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 659 | `pack_real_states_into_excludes_padding` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 697 | `pack_real_states_into_even_distribution_matches_gathered_states` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/state_exchange.rs` | 717 | `pack_real_states_into_reuses_buffer_capacity` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 565 | `estimation_report_to_fitting_report_two_hydros` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 610 | `inflow_models_to_stats_rows_field_values` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 643 | `inflow_models_to_ar_rows_lag_numbering_and_count` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 682 | `build_stochastic_summary_loaded_source_when_no_estimation_report` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 713 | `build_stochastic_summary_estimated_source_with_estimation_report` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 766 | `build_stochastic_summary_no_hydros_yields_none_source` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 803 | `build_stochastic_summary_stages_and_load_buses` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 834 | `opening_tree_source_user_supplied` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 869 | `opening_tree_source_generated` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 895 | `correlation_source_estimated_when_estimation_ran` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 949 | `correlation_source_loaded_when_no_estimation` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 985 | `correlation_source_none_when_empty` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 1023 | `build_stochastic_summary_new_fields_and_correlation_dim` | 58 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 1085 | `build_stochastic_summary_correlation_dim_uses_full_dim` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 1127 | `fitting_report_includes_reason_strings` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stochastic_summary.rs` | 1197 | `estimation_report_tracks_all_reductions` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 483 | `iteration_limit_triggered_at_limit` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 492 | `iteration_limit_triggered_above_limit` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 500 | `iteration_limit_not_triggered_below_limit` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 508 | `time_limit_triggered_at_threshold` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 517 | `time_limit_triggered_above_threshold` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 525 | `time_limit_not_triggered_below_threshold` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 533 | `bound_stalling_not_triggered_with_insufficient_history` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 545 | `bound_stalling_triggered_when_lb_stable` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 557 | `bound_stalling_not_triggered_when_lb_improving` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 569 | `bound_stalling_near_zero_lb_uses_max_guard` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 581 | `graceful_shutdown_triggered_when_requested` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 591 | `graceful_shutdown_not_triggered_when_not_requested` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 599 | `rule_set_any_mode_stops_on_first_triggered_rule` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 616 | `rule_set_any_mode_does_not_stop_when_no_rules_trigger` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 630 | `rule_set_all_mode_stops_only_when_all_rules_trigger` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 646 | `rule_set_all_mode_does_not_stop_when_only_one_triggers` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 660 | `rule_set_graceful_shutdown_bypasses_all_mode` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 675 | `rule_set_graceful_shutdown_bypasses_any_mode` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 687 | `rule_set_returns_all_results_regardless_of_mode` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 702 | `ac_iteration_limit_triggered_at_10` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 711 | `ac_bound_stalling_with_6_history_entries` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/stopping_rule.rs` | 723 | `ac_rule_set_any_mode_stops_at_iteration_100` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 1654 | `ac_train_completes_with_iteration_limit` | 87 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 1749 | `ac_train_returns_partial_on_infeasible` | 98 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 1860 | `ac_train_emits_correct_event_sequence` | 154 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2020 | `ac_train_result_fields_populated` | 87 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2113 | `ac_train_with_no_event_sender` | 84 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2203 | `ac_total_time_ms_is_non_negative` | 90 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2299 | `cut_selection_none_skips_step` | 96 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2402 | `cut_selection_level1_runs_at_frequency` | 111 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2520 | `cut_selection_stage0_exempt_preserves_cuts` | 128 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2656 | `existing_train_tests_pass_with_none` | 87 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2753 | `ac_train_partial_result_on_mid_iteration_failure` | 114 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2871 | `start_iteration_resumes_from_offset` | 89 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 2964 | `start_iteration_at_or_beyond_max_runs_zero_iterations` | 92 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3066 | `ac_broadcast_basis_cache_uses_scenario_0_not_last` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3129 | `ac_broadcast_basis_cache_none_slots_preserved` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3153 | `broadcast_basis_cache_single_rank_preserves_metadata` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3395 | `broadcast_basis_cache_multi_rank_round_trips_full_metadata` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3457 | `broadcast_basis_cache_empty_cut_slots_round_trips_ok` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3511 | `broadcast_basis_cache_truncated_cut_slots_returns_validation` | 58 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3579 | `broadcast_basis_cache_truncated_state_returns_validation` | 60 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training.rs` | 3647 | `template_bake_event_emitted` | 132 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 448 | `records_count_matches_iteration_summaries` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 463 | `converged_true_for_bound_stalling` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 474 | `converged_true_for_simulation_based` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 485 | `converged_false_for_iteration_limit` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 496 | `cut_stats_from_fcf` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 522 | `gap_percent_none_when_lb_nonpositive` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 536 | `converged_false_for_all_other_reasons` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 555 | `empty_events_produces_zero_records` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 569 | `gap_percent_computed_correctly` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 579 | `iteration_gap_percent_none_when_lb_zero_or_negative` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 590 | `upper_bound_std_from_forward_sync_complete` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 609 | `forward_passes_from_forward_pass_complete` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 629 | `cut_fields_from_backward_and_sync_events` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 663 | `peak_active_tracks_maximum_cuts_active` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 699 | `iterations_completed_from_result` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 709 | `termination_reason_copied_from_result` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 719 | `per_phase_timing_captured_from_sync_and_selection_events` | 67 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 788 | `overhead_ms_is_total_minus_attributed_phases` | 62 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 852 | `overhead_ms_saturates_at_zero_when_attributed_exceeds_total` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 884 | `cut_selection_records_extracted_from_events` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/training_output.rs` | 935 | `no_cut_selection_events_produces_empty_records` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/trajectory.rs` | 58 | `construct_and_access_all_fields` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/trajectory.rs` | 73 | `stage_cost_value_is_accessible` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/trajectory.rs` | 86 | `clone_produces_identical_independent_copy` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/trajectory.rs` | 110 | `debug_format_is_non_empty` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/trajectory.rs` | 122 | `flat_vec_indexing_pattern_works_correctly` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 176 | `stage_states_new_preallocates` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 186 | `stage_states_append_single_batch` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 195 | `stage_states_append_multiple_batches` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 211 | `stage_states_empty_states` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 218 | `archive_new_creates_correct_stages` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 228 | `archive_gathered_states_delegates` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 239 | `archive_accumulates_across_iterations` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/visited_states.rs` | 249 | `archive_states_for_stage_returns_flat_slice` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 658 | `test_workspace_send_bound` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 678 | `test_workspace_pool_size` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 684 | `test_workspace_buffer_dimensions` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 696 | `test_workspace_pool_zero_threads` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 702 | `test_workspace_pool_single_thread` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 709 | `test_workspace_pool_each_solver_independent` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 718 | `test_scratch_buffers_zero_downstream_par_order_empty_buffers` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 747 | `test_scratch_buffers_nonzero_downstream_par_order_allocates_correctly` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 780 | `test_workspace_pool_propagates_downstream_par_order` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 816 | `basis_store_new_all_none` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 831 | `basis_store_get_mut_set_and_retrieve` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 841 | `basis_store_zero_scenarios` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 848 | `basis_store_zero_stages` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 855 | `basis_store_split_workers_mut_disjoint_writes` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 884 | `basis_store_split_single_worker` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 894 | `basis_store_split_more_workers_than_scenarios` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 907 | `basis_store_slice_offset_correct` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 930 | `test_captured_basis_new_capacities` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 957 | `test_basis_store_holds_captured_basis` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/src/workspace.rs` | 991 | `test_recon_slot_lookup_presized` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/basis_reconstruct_churn.rs` | 182 | `basis_reconstruct_churn` | 127 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/basis_reconstruct_churn.rs` | 339 | `test_basis_reconstruct_no_churn_full_preservation` | 85 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/basis_reconstruct_churn.rs` | 458 | `test_basis_reconstruct_full_churn_no_rows_preserved` | 210 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/basis_reconstruct_churn.rs` | 697 | `simulate_warm_start_basis_preserved_gt_zero` | 115 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/boundary_cuts.rs` | 136 | `boundary_cuts_improve_terminal_stage_objective` | 73 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 274 | `risk_measure_expectation_aggregate_cut_sums_to_weighted_mean` | 28 | `risk_measure_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 306 | `risk_measure_cvar_alpha_one_equals_expectation` | 41 | `risk_measure_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 351 | `risk_measure_cvar_alpha_half_concentrates_on_worst` | 24 | `risk_measure_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 379 | `risk_measure_cvar_weights_sum_to_one` | 37 | `risk_measure_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 441 | `stopping_rule_bound_stalling_uses_max_guard` | 19 | `stopping_rule_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 464 | `stopping_rule_set_all_mode_requires_simultaneous` | 46 | `stopping_rule_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 516 | `stopping_rule_graceful_shutdown_bypasses_all_mode` | 18 | `stopping_rule_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 549 | `cut_wire_record_serialize_deserialize_round_trip` | 48 | `cut_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 601 | `cut_pool_add_then_active_cuts_returns_correct_data` | 52 | `cut_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 676 | `convergence_monitor_gap_formula_matches_spec` | 21 | `convergence_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 704 | `convergence_monitor_lb_history_grows_monotonically_when_lb_increases` | 44 | `convergence_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 751 | `convergence_monitor_iteration_limit_triggers_at_exact_count` | 30 | `convergence_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 801 | `evaluate_lower_bound_monotonicity_with_additional_cuts` | 83 | `lb_conformance` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 896 | `indexer_constraint_inventory` | 110 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/conformance.rs` | 1020 | `constraint_extraction_regression_guard` | 71 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/decomp_integration.rs` | 166 | `decomp_structural_properties_and_training` | 151 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/decomp_integration.rs` | 336 | `decomp_boundary_cuts_compose_with_weekly_monthly` | 75 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/determinism.rs` | 707 | `test_training_determinism_across_thread_counts` | 35 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/determinism.rs` | 902 | `test_canonical_ub_determinism_across_rank_counts` | 73 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/determinism.rs` | 987 | `test_simulation_determinism_across_thread_counts` | 34 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 253 | `d01_thermal_dispatch` | 15 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 288 | `d02_single_hydro` | 15 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 311 | `d03_two_hydro_cascade` | 15 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 334 | `d04_transmission` | 15 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 355 | `d05_fpha_constant_head` | 15 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 446 | `d06_fpha_variable_head` | 19 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 493 | `d07_fpha_computed` | 19 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 579 | `d08_evaporation` | 21 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 639 | `d09_multi_deficit` | 15 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 747 | `d10_inflow_nonnegativity` | 21 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 841 | `d11_water_withdrawal` | 26 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 877 | `d11_warm_start_verification` | 18 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 915 | `d12_checkpoint_round_trip` | 199 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1138 | `d13_generic_constraint` | 50 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1206 | `d14_block_factors` | 64 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1293 | `d15_non_controllable_source` | 93 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1417 | `d16_par1_lag_shift` | 27 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1456 | `model_persistence_regression_d01` | 48 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1526 | `incremental_lb_reduces_load_model_count` | 46 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1581 | `incremental_lb_add_rows_exceeds_load_model` | 24 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1614 | `incremental_bit_for_bit_d01_trace` | 15 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1655 | `d19_multi_hydro_par_truncation` | 16 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1712 | `d20_operational_violations` | 44 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1764 | `d21_min_outflow_regression` | 192 | `` | `regression` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 1967 | `d22_per_block_min_outflow` | 121 | `` | `conformance` | `d-case-determinism,convertido-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2106 | `d20_convertido2_truncation_feasibility` | 22 | `` | `conformance` | `d-case-determinism,convertido-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2161 | `d23_bidirectional_withdrawal` | 142 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2435 | `d24_productivity_override` | 86 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2544 | `d25_discount_rate` | 22 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2573 | `d25_simulation_discount_factors` | 28 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2613 | `d26_estimated_par2` | 16 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2632 | `d26_estimated_par2_order_selection` | 25 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2697 | `d27_per_stage_thermal_cost` | 29 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2739 | `d28_decomp_weekly_monthly_loads_and_trains` | 10 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2767 | `d29_pattern_c_weekly_par` | 87 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2874 | `d30_pattern_d_monthly_quarterly_loads_and_trains` | 47 | `` | `conformance` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/deterministic.rs` | 2938 | `baked_vs_fallback_simulation_costs_are_identical` | 104 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/estimation_integration.rs` | 319 | `test_estimate_from_history_fixed_order` | 50 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/estimation_integration.rs` | 379 | `test_estimate_from_history_aic_order` | 49 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/estimation_integration.rs` | 704 | `test_estimation_round_trip_par1` | 82 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/estimation_integration.rs` | 890 | `test_partial_estimation_end_to_end` | 64 | `` | `e2e` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/estimation_integration.rs` | 962 | `test_estimation_round_trip_two_hydros` | 60 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 549 | `insample_equivalence_d01` | 27 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 585 | `out_of_sample_convergence` | 52 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 651 | `out_of_sample_declaration_order_invariance` | 34 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 1054 | `historical_convergence` | 18 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 1076 | `external_inflow_convergence` | 28 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 1108 | `mixed_scheme_convergence` | 30 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 1332 | `external_load_library_populated` | 26 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 1366 | `external_ncs_library_populated` | 26 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/forward_sampler_integration.rs` | 1510 | `monthly_noise_sharing_regression` | 45 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_computed.rs` | 310 | `fpha_computed_case_converges` | 82 | `` | `e2e` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_evaporation.rs` | 92 | `fpha_evaporation_case_converges` | 123 | `` | `e2e` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_evaporation.rs` | 233 | `test_4ree_fpha_evap_seasonal_ref_provenance` | 92 | `` | `e2e` | `fpha-slow` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/inflow_nonnegativity.rs` | 685 | `test_penalty_method_prevents_infeasibility` | 8 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/inflow_nonnegativity.rs` | 703 | `test_penalty_slack_value_matches_negative_inflow` | 23 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/inflow_nonnegativity.rs` | 736 | `test_simulation_slack_output_populated` | 29 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/inflow_nonnegativity.rs` | 774 | `truncation_with_penalty_training_completes` | 30 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/inflow_nonnegativity.rs` | 814 | `per_plant_inflow_penalty_differentiates_objective_coefficients` | 136 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 624 | `train_converges_with_mock_solver` | 79 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 707 | `train_deterministic_with_same_seed` | 18 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 729 | `train_lb_monotonically_nondecreasing` | 94 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 829 | `train_emits_correct_event_sequence` | 98 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 931 | `train_stops_at_iteration_limit` | 74 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 1009 | `train_stops_on_graceful_shutdown` | 84 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 1097 | `train_propagates_infeasible_error` | 83 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 1193 | `d17_level1_cut_selection_convergence` | 149 | `` | `integration` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 1352 | `d17_level1_cut_selection_reconstruction` | 92 | `` | `integration` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 1458 | `d18_lml1_cut_selection_convergence` | 134 | `` | `integration` | `d-case-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 1602 | `test_forward_basis_reconstruct_bit_identical_d01` | 90 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 1776 | `forward_pass_uses_baked_template_on_iter_2` | 263 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 2047 | `backward_pass_uses_delta_batch_on_iter_2` | 309 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/integration.rs` | 2364 | `baked_backward_pass_smoke_test` | 89 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/load_integration.rs` | 361 | `test_stochastic_load_context_construction` | 17 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/load_integration.rs` | 382 | `test_stochastic_load_training_completes` | 126 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/load_integration.rs` | 515 | `test_deterministic_load_training_matches_baseline` | 99 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/load_integration.rs` | 618 | `test_stochastic_load_seed_determinism` | 141 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/pattern_d_integration.rs` | 131 | `pattern_d_structural_properties_and_training` | 203 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/simulation_integration.rs` | 541 | `train_simulate_write_cycle` | 363 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/simulation_integration.rs` | 1205 | `simulation_min_outflow_slack_extracted_from_primal` | 258 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/simulation_only.rs` | 80 | `simulation_only_fcf_round_trip` | 128 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/sparse_dense.rs` | 44 | `sparse_full_mask_equals_dense` | 91 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/sparse_dense.rs` | 140 | `sparse_partial_mask_produces_correct_subset` | 66 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/sparse_dense.rs` | 209 | `sparse_dense_with_scaling` | 43 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/unified_run_path.rs` | 314 | `forward_uses_stage_solve` | 155 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/unified_run_path.rs` | 482 | `backward_uses_stage_solve` | 216 | `` | `integration` | `convertido-determinism` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/unified_run_path.rs` | 714 | `simulation_zero_rejections_on_cut_churn` | 212 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/unified_run_path.rs` | 1135 | `cross_phase_identical_inputs_identical_reconstruction` | 74 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/unified_run_path.rs` | 1223 | `phase_only_affects_outcome_variant_not_solve_path` | 97 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/unified_run_path.rs` | 1330 | `cold_start_zero_recon_stats` | 33 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 294 | `v045_d01_thermal_dispatch` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 300 | `v045_d02_single_hydro` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 306 | `v045_d03_two_hydro_cascade` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 312 | `v045_d04_transmission` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 318 | `v045_d05_fpha_constant_head` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 324 | `v045_d06_fpha_variable_head` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 330 | `v045_d07_fpha_computed` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 336 | `v045_d08_evaporation` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 342 | `v045_d09_multi_deficit` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 348 | `v045_d10_inflow_nonnegativity` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 354 | `v045_d11_water_withdrawal` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 360 | `v045_d13_generic_constraint` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 366 | `v045_d14_block_factors` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 372 | `v045_d15_non_controllable_source` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 378 | `v045_d16_par1_lag_shift` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 384 | `v045_d19_multi_hydro_par` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 390 | `v045_d20_operational_violations` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 396 | `v045_d21_min_outflow_regression` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 402 | `v045_d22_per_block_min_outflow` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 408 | `v045_d23_bidirectional_withdrawal` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 414 | `v045_d24_productivity_override` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 420 | `v045_d25_discount_rate` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 426 | `v045_d26_estimated_par2` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 432 | `v045_d27_per_stage_thermal_cost` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 438 | `v045_d28_decomp_weekly_monthly` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 444 | `v045_d29_pattern_c_weekly_par` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/v045_regression.rs` | 450 | `v045_d30_pattern_d_monthly_quarterly` | 3 | `` | `regression` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/warm_start.rs` | 133 | `resume_training_from_checkpoint` | 69 | `` | `integration` | `generic` |  |
| `cobre-sddp` | `crates/cobre-sddp/tests/warm_start.rs` | 205 | `warm_start_training_preserves_cuts_and_trains_further` | 70 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 300 | `test_bake_empty_rows_copies_base` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 332 | `test_bake_single_row_appends_correct_column_entries` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 374 | `test_bake_preserves_row_scale_and_defaults_cut_rows_to_one` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 401 | `test_bake_preserves_empty_row_scale_when_no_rows` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 417 | `test_bake_reuses_out_buffer_capacity` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 491 | `test_bake_determinism` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 520 | `test_bake_multi_column_distribution` | 72 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 651 | `test_bake_load_model_row_count` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 683 | `test_bake_empty_base_row_scale_with_cut_rows_appends_ones` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/baking.rs` | 722 | `test_bake_panics_on_nnz_overflow` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/ffi.rs` | 299 | `test_ffi_smoke_create_solve_destroy` | 61 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1363 | `test_highs_solver_create_and_name` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1370 | `test_highs_solver_send_bound` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1376 | `test_highs_solver_statistics_initial` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1388 | `test_highs_load_model_updates_dimensions` | 29 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1419 | `test_highs_add_rows_updates_dimensions` | 23 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1444 | `test_highs_set_row_bounds_no_panic` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1454 | `test_highs_set_col_bounds_no_panic` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1464 | `test_highs_set_bounds_empty_no_panic` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1477 | `test_highs_solve_basic_lp` | 31 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1513 | `test_highs_solve_with_cuts` | 32 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1549 | `test_highs_solve_after_rhs_patch` | 20 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1572 | `test_highs_solve_statistics_increment` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1592 | `test_highs_solve_preserves_stats` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1615 | `test_highs_solve_iterations_positive` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1630 | `test_highs_solve_time_positive` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1646 | `test_highs_solve_statistics_single` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1666 | `test_get_basis_valid_status_codes` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1694 | `test_get_basis_resizes_output` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1731 | `test_solve_warm_start_reproduces_cold_objective` | 37 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1773 | `test_solve_warm_start_extends_missing_rows_as_basic` | 32 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1814 | `test_solve_warm_start_non_alien_success` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1856 | `test_solve_warm_start_rejects_inconsistent_basis` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1922 | `test_solve_increments_clear_solver_count_per_call` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 2026 | `test_research_probe_limit_status_on_ss11_lp` | 33 | `research_tests_ticket_023` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 2126 | `test_research_time_limit_zero_triggers_time_limit_status` | 32 | `research_tests_ticket_023` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 2168 | `test_research_iteration_limit_zero_triggers_iteration_limit_status` | 35 | `research_tests_ticket_023` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 2210 | `test_research_partial_solution_availability` | 40 | `research_tests_ticket_023` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 2254 | `test_research_restore_defaults_allows_subsequent_optimal_solve` | 90 | `research_tests_ticket_023` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 2350 | `test_research_iteration_limit_one_triggers_iteration_limit_status` | 51 | `research_tests_ticket_023` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 209 | `test_trait_compiles_as_generic_bound` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 214 | `test_solver_interface_send_bound` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 219 | `test_noop_solver_name` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 226 | `test_noop_solver_statistics_initial` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 237 | `test_noop_solver_get_basis_noop` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 250 | `test_noop_solver_solve_with_optional_basis_returns_internal_error` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 260 | `test_unsupported_display_format` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 269 | `test_noop_solver_all_methods` | 42 | `tests` | `unit` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 573 | `test_basis_new_dimensions_and_zero_fill` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 582 | `test_basis_new_empty` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 589 | `test_basis_debug_and_clone` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 601 | `test_solver_error_display_infeasible` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 607 | `test_solver_error_display_all_variants` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 639 | `test_solver_error_is_std_error` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 648 | `test_solver_statistics_default_all_zero` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 669 | `default_stats_has_zero_clear_solver_counters` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 699 | `test_stage_template_construction` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 726 | `test_solver_error_display_all_branches` | 60 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 788 | `test_solver_error_is_std_error_all_variants` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 821 | `test_solution_view_to_owned` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 843 | `test_solution_view_is_copy` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/src/types.rs` | 860 | `test_row_batch_construction` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 61 | `test_solver_highs_load_model_and_solve` | 32 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 96 | `test_solver_highs_load_model_replaces_previous` | 27 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 127 | `test_fixture_stage_template_data` | 21 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 150 | `test_fixture_row_batch_data` | 10 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 164 | `test_solver_highs_add_rows_tightens` | 28 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 195 | `test_solver_highs_add_rows_single_cut` | 32 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 231 | `test_solver_highs_set_row_bounds_state_change` | 35 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 270 | `test_solver_highs_set_col_bounds_basic` | 18 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 291 | `test_solver_highs_set_col_bounds_tightens` | 24 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 318 | `test_solver_highs_set_col_bounds_repatch` | 34 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 356 | `test_solver_highs_solve_dual_values` | 28 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 387 | `test_solver_highs_solve_dual_values_with_cuts` | 27 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 417 | `test_solver_highs_solve_reduced_costs` | 22 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 442 | `test_solver_highs_solve_iterations_reported` | 21 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 468 | `test_solver_highs_dual_normalization_cut_relevant_row` | 16 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 487 | `test_solver_highs_dual_normalization_sensitivity_check` | 29 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 519 | `test_solver_highs_dual_normalization_with_binding_cut` | 18 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 545 | `test_solver_highs_statistics_are_cumulative` | 27 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 577 | `test_solver_highs_statistics_initial` | 33 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 613 | `test_solver_highs_statistics_increment` | 41 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 659 | `test_solver_highs_name_returns_identifier` | 7 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 680 | `test_solver_highs_lifecycle_repeated_patch_solve` | 35 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 729 | `test_solver_highs_solve_infeasible` | 48 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 792 | `test_solver_highs_solve_unbounded` | 48 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 913 | `test_solver_highs_solve_time_limit` | 24 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 946 | `test_solver_highs_solve_iteration_limit` | 29 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 984 | `test_solver_highs_restore_defaults_after_limit` | 30 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1060 | `test_solver_highs_infeasible_with_rows` | 35 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1103 | `test_solver_highs_infeasible_with_presolve` | 41 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1154 | `test_solver_highs_unbounded_with_primal_ray` | 35 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1207 | `test_solver_highs_unbounded_or_infeasible` | 48 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1263 | `solve_equals_solve_owned` | 33 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1303 | `solve_borrows_internal_buffers` | 22 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1332 | `solve_after_add_rows` | 26 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1361 | `solve_statistics_updated` | 16 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1383 | `basis_dimensions_after_solve` | 25 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1412 | `basis_cut_extension` | 22 | `` | `integration` | `add-rows-trait` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1438 | `basis_warm_start_iterations` | 29 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1472 | `test_basis_roundtrip` | 29 | `` | `integration` | `generic` |  |
| `cobre-solver` | `crates/cobre-solver/tests/ffi_set_basis_non_alien_smoke.rs` | 70 | `non_alien_basis_loop_low_rejection_rate` | 44 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/broadcast.rs` | 188 | `test_round_trip_minimal_system` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/broadcast.rs` | 202 | `test_round_trip_populated_system` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/broadcast.rs` | 234 | `test_deserialize_corrupted_bytes` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/broadcast.rs` | 243 | `test_deserialize_empty_bytes` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/broadcast.rs` | 252 | `test_serialized_size_reasonable` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1089 | `test_parse_minimal_config` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1120 | `test_missing_forward_passes` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1138 | `test_missing_stopping_rules` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1154 | `test_nonexistent_file` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1167 | `test_parse_full_config` | 97 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1267 | `test_invalid_json_syntax` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1281 | `test_stopping_rule_variants` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1329 | `test_unknown_stopping_rule_type` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1343 | `test_config_has_no_version_field` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1356 | `test_schema_field_accepted` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1378 | `test_invalid_policy_mode_rejected` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1393 | `test_legacy_version_field_silently_ignored` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1412 | `test_truncation_method_accepted` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1439 | `test_estimation_config_defaults` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1454 | `test_estimation_config_order_selection_fixed_deprecated` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1472 | `test_estimation_config_order_selection_pacf` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1491 | `test_estimation_config_unknown_order_selection` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1515 | `test_exports_stochastic_explicit_true` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1534 | `test_exports_stochastic_defaults_to_false` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1570 | `test_training_scenario_source_default` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1584 | `test_training_scenario_source_explicit` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1603 | `test_simulation_scenario_source_fallback` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1618 | `test_simulation_scenario_source_independent` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1635 | `test_scenario_source_historical_inflow_valid` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1646 | `test_scenario_source_historical_load_rejected` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1666 | `test_scenario_source_historical_ncs_rejected` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1685 | `test_scenario_source_seed_required_for_oos` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1703 | `test_scenario_source_historical_years_range` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1720 | `test_scenario_source_historical_years_without_historical_scheme` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1742 | `test_dead_sampling_scheme_field_removed` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1759 | `max_active_per_stage_serde_roundtrip` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1779 | `max_active_per_stage_absent_defaults_none` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1799 | `test_boundary_policy_present` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1823 | `test_boundary_policy_absent` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1842 | `test_boundary_policy_explicit_null` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1861 | `test_policy_config_default_boundary_is_none` | 6 | `tests` | `unit` | `warm-start-config-flag` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1871 | `test_boundary_policy_round_trip` | 17 | `tests` | `unit` | `warm-start-config-flag` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1921 | `warm_start_basis_mode_returns_migration_error` | 11 | `tests` | `unit` | `warm-start-config-flag,canonical-config-flag` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1934 | `warm_start_basis_mode_non_alien_also_rejected` | 8 | `tests` | `unit` | `warm-start-config-flag,canonical-config-flag` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1944 | `config_without_warm_start_basis_mode_parses_cleanly` | 5 | `tests` | `unit` | `warm-start-config-flag,canonical-config-flag` |  |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1954 | `canonical_state_key_is_obsolete_but_parses_cleanly` | 22 | `tests` | `unit` | `canonical-config-flag` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 989 | `test_thermal_valid_3_rows_sorted` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1015 | `test_thermal_missing_stage_id` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1044 | `test_thermal_nan_max_generation` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1066 | `test_thermal_empty_parquet` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1075 | `test_thermal_declaration_order_invariance` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1106 | `test_load_thermal_bounds_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1113 | `test_thermal_cost_and_block_id_columns_read` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1151 | `test_thermal_missing_cost_and_block_id_columns_are_none` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1171 | `test_thermal_non_null_block_id_is_parsed` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1202 | `test_thermal_nan_cost_per_mwh` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1243 | `test_thermal_negative_cost_per_mwh` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1304 | `test_hydro_all_11_columns_mixed_null` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1346 | `test_hydro_subset_of_optional_columns` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1386 | `test_hydro_missing_stage_id` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1415 | `test_hydro_nan_in_optional_column` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1450 | `test_hydro_empty_parquet` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1470 | `test_hydro_declaration_order_invariance` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1512 | `test_load_hydro_bounds_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1548 | `test_line_valid_rows_sorted` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1570 | `test_line_missing_stage_id` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1596 | `test_line_nan_direct_mw` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1612 | `test_line_empty_parquet` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1620 | `test_line_declaration_order_invariance` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1649 | `test_load_line_bounds_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1685 | `test_pumping_valid_rows_sorted` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1707 | `test_pumping_missing_stage_id` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1733 | `test_pumping_nan_max_m3s` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1749 | `test_pumping_empty_parquet` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1757 | `test_pumping_declaration_order_invariance` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1788 | `test_load_pumping_bounds_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1827 | `test_contract_valid_rows_sorted` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1852 | `test_contract_missing_stage_id` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1878 | `test_contract_nan_price` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1900 | `test_contract_empty_parquet` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1908 | `test_contract_declaration_order_invariance` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/bounds.rs` | 1941 | `test_load_contract_bounds_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/exchange_factors.rs` | 298 | `test_parse_valid_single_entry_three_block_factors` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/exchange_factors.rs` | 312 | `test_parse_negative_direct_factor_returns_schema_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/exchange_factors.rs` | 343 | `test_parse_zero_reverse_factor_returns_schema_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/exchange_factors.rs` | 374 | `test_parse_nan_factor_returns_schema_error` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/exchange_factors.rs` | 390 | `test_parse_empty_array_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/exchange_factors.rs` | 399 | `test_parse_sorted_by_line_id_stage_id` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/exchange_factors.rs` | 436 | `test_parse_entry_with_empty_block_factors` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 823 | `test_expr_simple_single_term` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 838 | `test_expr_addition_two_terms` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 861 | `test_expr_coefficient_and_subtraction` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 884 | `test_expr_subtraction_negates_coefficient` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 893 | `test_expr_block_specific_variable` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 907 | `test_expr_line_exchange_with_block` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 921 | `test_expr_stage_only_hydro_storage` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 934 | `test_expr_stage_only_with_block_is_error` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 944 | `test_expr_unknown_variable_name` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 954 | `test_expr_missing_closing_paren` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 964 | `test_expr_empty_is_error` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 974 | `test_expr_all_19_variable_types_recognised` | 153 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1132 | `test_parse_valid_two_constraints` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1165 | `test_parse_coefficient_and_subtraction_expression` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1193 | `test_parse_invalid_expression_returns_schema_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1224 | `test_parse_duplicate_ids_returns_schema_error` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1262 | `test_parse_slack_enabled_without_penalty_returns_schema_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1293 | `test_parse_invalid_sense_returns_schema_error` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1322 | `test_parse_empty_constraints_array` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1331 | `test_parse_sorted_by_id` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1367 | `test_parse_line_exchange_json_constraint` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1395 | `test_parse_slack_zero_penalty_returns_schema_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic.rs` | 1426 | `test_parse_description_optional` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic_bounds.rs` | 264 | `test_parse_with_nullable_block_id` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic_bounds.rs` | 296 | `test_parse_without_block_id_column` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic_bounds.rs` | 308 | `test_parse_non_finite_bound_returns_schema_error` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic_bounds.rs` | 329 | `test_parse_infinite_bound_returns_schema_error` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic_bounds.rs` | 346 | `test_parse_missing_bound_column_returns_schema_error` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic_bounds.rs` | 379 | `test_parse_empty_parquet` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/generic_bounds.rs` | 388 | `test_parse_sort_order_invariance` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/ncs_bounds.rs` | 188 | `test_parse_valid_2_rows` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/ncs_bounds.rs` | 200 | `test_parse_sorted_output` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/ncs_bounds.rs` | 209 | `test_parse_negative_value_rejected` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/ncs_bounds.rs` | 222 | `test_parse_nan_rejected` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/ncs_bounds.rs` | 234 | `test_parse_missing_column_rejected` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/ncs_bounds.rs` | 262 | `test_parse_empty_file` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/ncs_bounds.rs` | 269 | `test_parse_zero_available_gen_accepted` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 846 | `test_bus_valid_3_rows_sorted` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 870 | `test_bus_missing_bus_id` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 899 | `test_bus_negative_excess_cost` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 921 | `test_bus_nan_excess_cost` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 943 | `test_bus_empty_parquet` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 952 | `test_load_bus_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 985 | `test_line_valid_2_rows_sorted` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1002 | `test_line_missing_line_id` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1031 | `test_line_negative_exchange_cost` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1053 | `test_line_nan_exchange_cost` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1075 | `test_line_empty_parquet` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1084 | `test_load_line_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1112 | `test_hydro_two_columns_non_null_rest_none` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1161 | `test_hydro_all_11_columns_mixed` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1203 | `test_hydro_only_2_columns_in_schema` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1244 | `test_hydro_missing_hydro_id` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1273 | `test_hydro_negative_spillage_cost` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1308 | `test_hydro_nan_penalty` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1343 | `test_hydro_empty_parquet` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1363 | `test_load_hydro_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1370 | `test_hydro_declaration_order_invariance` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1448 | `test_ncs_valid_3_rows_sorted` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1468 | `test_ncs_missing_source_id` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1497 | `test_ncs_negative_curtailment_cost` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1519 | `test_ncs_nan_curtailment_cost` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1541 | `test_ncs_empty_parquet` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/constraints/penalty_overrides.rs` | 1550 | `test_load_ncs_none` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/error.rs` | 151 | `test_load_error_io_display` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/error.rs` | 166 | `test_load_error_parse_display` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/error.rs` | 180 | `test_load_error_schema_display` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/error.rs` | 202 | `test_load_error_cross_reference_display` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/error.rs` | 229 | `test_load_error_is_std_error` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/error.rs` | 241 | `test_load_error_io_helper` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 313 | `test_valid_itaipu_5_planes_all_columns` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 367 | `test_optional_columns_absent_kappa_defaults_to_1` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 397 | `test_kappa_column_present_but_null_defaults_to_1` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 435 | `test_missing_gamma_0_column` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 471 | `test_missing_hydro_id_column` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 505 | `test_wrong_type_gamma_0_as_int32` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 541 | `test_sorted_output` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 572 | `test_null_stage_id_sorts_before_non_null` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 609 | `test_file_not_found` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 625 | `test_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 636 | `test_optional_validity_ranges_preserved` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 670 | `test_declaration_order_invariance` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/fpha_hyperplanes.rs` | 712 | `test_field_values_preserved` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 301 | `test_valid_single_hydro_five_rows` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 324 | `test_multiple_hydros_sorted` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 357 | `test_missing_area_km2_column` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 393 | `test_missing_hydro_id_column` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 424 | `test_wrong_type_hydro_id_as_float64` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 456 | `test_negative_volume_hm3` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 479 | `test_negative_height_m` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 494 | `test_negative_area_km2` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 511 | `test_file_not_found` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 527 | `test_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 538 | `test_nan_volume_rejected` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 553 | `test_infinite_area_rejected` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 571 | `test_field_values_preserved` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 588 | `test_multiple_record_batches` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 605 | `test_zero_volume_is_valid` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/hydro_geometry.rs` | 617 | `test_schema_error_field_format_includes_row_index` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/mod.rs` | 131 | `test_load_production_models_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/mod.rs` | 138 | `test_load_fpha_hyperplanes_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/mod.rs` | 145 | `test_load_hydro_geometry_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 580 | `test_valid_stage_ranges_mode` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 635 | `test_valid_seasonal_mode` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 687 | `test_mixed_modes_sorted_by_hydro_id` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 726 | `test_duplicate_hydro_id` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 763 | `test_invalid_stage_range_start_greater_than_end` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 794 | `test_stage_range_start_equals_end_is_valid` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 817 | `test_mutually_exclusive_fitting_window_min` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 853 | `test_mutually_exclusive_fitting_window_max` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 888 | `test_mutually_exclusive_fitting_window_seasonal` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 922 | `test_file_not_found` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 937 | `test_unknown_selection_mode` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 956 | `test_empty_array_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 967 | `test_declaration_order_invariance` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 1006 | `test_fpha_config_without_fitting_window` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 1035 | `test_productivity_override_present` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 1061 | `test_productivity_override_absent_defaults_to_none` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 1086 | `test_productivity_override_negative_rejected` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 1110 | `test_productivity_override_zero_rejected` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 1134 | `test_productivity_override_rejected_on_fpha` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/extensions/production_models.rs` | 1159 | `test_seasonal_productivity_override` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 522 | `test_parse_valid_initial_conditions` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 559 | `test_parse_valid_past_inflows` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 587 | `test_parse_empty_arrays` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 602 | `test_negative_storage_value` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 630 | `test_negative_filling_storage_value` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 660 | `test_duplicate_hydro_id_in_storage` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 689 | `test_duplicate_hydro_id_in_filling_storage` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 720 | `test_hydro_id_in_both_lists` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 751 | `test_duplicate_hydro_id_in_past_inflows` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 782 | `test_file_not_found` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 797 | `test_zero_storage_value_is_valid` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 816 | `test_filling_storage_below_dead_volume_is_valid` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 833 | `test_declaration_order_invariance` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 875 | `test_invalid_json_syntax` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 886 | `test_missing_required_field` | 9 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 898 | `test_zero_past_inflow_value_is_valid` | 15 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 916 | `test_empty_values_m3s_is_valid` | 15 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 938 | `test_recent_observations_absent_defaults_to_empty` | 8 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 950 | `test_recent_observations_empty_array` | 10 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 967 | `test_recent_observations_valid_two_entries` | 24 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 998 | `test_recent_observations_invalid_start_date_format` | 20 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1023 | `test_recent_observations_invalid_end_date_format` | 20 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1050 | `test_recent_observations_end_date_equals_start_date` | 20 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1075 | `test_recent_observations_end_date_before_start_date` | 20 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1102 | `test_recent_observations_negative_value` | 20 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1129 | `test_recent_observations_overlapping_ranges` | 21 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1157 | `test_recent_observations_adjacent_ranges_are_valid` | 16 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1179 | `test_recent_observations_sorted_by_hydro_id_then_start_date` | 26 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1210 | `test_recent_observations_declaration_order_invariance` | 30 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1247 | `test_parse_past_inflows_with_valid_season_ids` | 15 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1267 | `test_parse_past_inflows_season_ids_length_mismatch` | 30 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/initial_conditions.rs` | 1301 | `test_parse_past_inflows_without_season_ids_backward_compat` | 16 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/convergence_reader.rs` | 351 | `read_convergence_summary_from_real_parquet` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/convergence_reader.rs` | 386 | `read_convergence_summary_empty_file` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/convergence_reader.rs` | 419 | `read_convergence_summary_missing_file` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/convergence_reader.rs` | 432 | `read_convergence_summary_single_row` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1325 | `codes_json_roundtrip` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1366 | `entities_csv_correct_rows` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1406 | `entities_csv_entity_type_order` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1431 | `entities_csv_system_id_is_zero` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1448 | `variables_csv_total_columns` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1463 | `variables_csv_has_required_columns_in_header` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1504 | `variables_csv_nullable_reflects_schema` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1530 | `bounds_parquet_roundtrip` | 86 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1620 | `state_dictionary_hydro_storage_entries` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/dictionary.rs` | 1660 | `state_dictionary_version_field` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 113 | `display_io_error_contains_path_and_source` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 128 | `display_serialization_error_contains_entity_and_message` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 142 | `display_schema_error_contains_file_and_column` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 164 | `display_manifest_error_contains_type_and_message` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 181 | `output_error_is_send_sync_static` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 186 | `output_error_satisfies_std_error_trait` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 207 | `io_helper_constructs_correct_variant` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 220 | `serialization_helper_constructs_correct_variant` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/error.rs` | 232 | `all_variants_debug_non_empty` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/hydro_models.rs` | 226 | `round_trip_5_rows_hydro_66` | 109 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/hydro_models.rs` | 341 | `empty_slice_produces_valid_parquet_with_zero_rows` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/hydro_models.rs` | 357 | `schema_has_exactly_11_fields_with_correct_names_and_types` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/hydro_models.rs` | 407 | `nullable_columns_round_trip_as_none` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/hydro_models.rs` | 450 | `multi_hydro_rows_sorted_by_parse` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/hydro_models.rs` | 485 | `parent_directory_created_automatically` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 413 | `training_metadata_roundtrip` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 445 | `simulation_metadata_roundtrip` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 464 | `write_training_metadata_creates_file` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 478 | `write_simulation_metadata_creates_file` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 492 | `write_training_metadata_fields_survive_write_read_cycle` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 506 | `write_simulation_metadata_fields_survive_write_read_cycle` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 521 | `write_training_metadata_missing_parent_returns_io_error` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 535 | `write_simulation_metadata_missing_parent_returns_io_error` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 549 | `read_training_metadata_missing_file` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 562 | `read_training_metadata_malformed_json` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 580 | `write_metadata_atomic_no_tmp_remains` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 598 | `training_metadata_cobre_version_matches_cargo_pkg_version` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/manifest.rs` | 617 | `now_iso8601_returns_valid_format` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 438 | `training_output_construction_and_field_access` | 63 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 503 | `iteration_record_construction_and_field_access` | 54 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 559 | `simulation_output_construction_and_field_access` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 579 | `cut_statistics_construction` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 592 | `test_merge_empty_slice` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 602 | `test_merge_single_output` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 619 | `test_merge_two_outputs` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/mod.rs` | 647 | `test_merge_partitions_sorted` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/parquet_config.rs` | 72 | `parquet_writer_config_default_values` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/parquet_config.rs` | 90 | `parquet_writer_config_zstd_level_is_three` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/parquet_config.rs` | 102 | `parquet_writer_config_clone_is_independent` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/parquet_config.rs` | 114 | `parquet_writer_config_debug_non_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1302 | `serialize_stage_cuts_single_cut_round_trip` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1328 | `serialize_stage_cuts_empty_cuts_valid_buffer` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1344 | `serialize_stage_cuts_multiple_cuts_deterministic` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1362 | `serialize_stage_cuts_non_empty_for_varying_state_dimensions` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1385 | `serialize_stage_basis_round_trip` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1406 | `serialize_stage_basis_empty_status_vectors` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1428 | `serialize_stage_basis_deterministic` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1451 | `policy_checkpoint_metadata_serializes_to_json` | 60 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1513 | `policy_checkpoint_metadata_none_upper_bound_serializes_to_null` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1593 | `write_policy_checkpoint_creates_directory_structure` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1641 | `write_policy_checkpoint_metadata_json_valid` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1688 | `write_policy_checkpoint_cut_files_non_empty` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1725 | `write_policy_checkpoint_basis_files_non_empty` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1749 | `write_policy_checkpoint_empty_bases_no_basis_files` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1801 | `write_policy_checkpoint_error_on_readonly_dir` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1836 | `write_policy_checkpoint_stage_numbering_zero_padded` | 65 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1905 | `deserialize_stage_cuts_single_cut_all_fields` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 1948 | `deserialize_stage_cuts_three_cuts_all_match` | 59 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2009 | `deserialize_stage_cuts_empty_cut_pool` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2024 | `deserialize_stage_cuts_zero_length_coefficients` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2045 | `deserialize_stage_cuts_large_coefficient_vector` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2066 | `deserialize_stage_cuts_truncated_buffer_returns_error` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2077 | `deserialize_stage_cuts_stage_id_nonzero` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2086 | `deserialize_stage_basis_all_fields` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2114 | `deserialize_stage_basis_empty_status_vectors` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2138 | `deserialize_stage_basis_large_status_vectors` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2158 | `deserialize_stage_basis_truncated_buffer_returns_error` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2178 | `policy_checkpoint_metadata_deserializes_from_json` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2212 | `policy_checkpoint_metadata_deserializes_none_upper_bound` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2242 | `read_policy_checkpoint_full_round_trip` | 69 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2313 | `read_policy_checkpoint_no_bases_empty_stage_bases` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2334 | `read_policy_checkpoint_missing_metadata_returns_error` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2349 | `read_policy_checkpoint_stages_sorted_by_id` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2386 | `read_policy_checkpoint_metadata_json_field_by_field` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2427 | `policy_checkpoint_metadata_warm_start_counts_round_trips` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2456 | `policy_checkpoint_metadata_warm_start_counts_absent_defaults_to_empty` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/policy.rs` | 2487 | `read_policy_checkpoint_warm_start_counts_in_metadata` | 76 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/provenance.rs` | 70 | `write_and_read_back_json` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/provenance.rs` | 84 | `round_trip_all_fields` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/provenance.rs` | 112 | `tmp_file_is_cleaned_up` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 339 | `write_results_creates_training_directories` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 365 | `write_results_creates_simulation_directory` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 386 | `write_results_returns_ok_on_success` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 412 | `write_results_creates_success_marker` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 433 | `write_results_creates_metadata` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 461 | `write_results_creates_convergence_parquet` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 482 | `write_results_convergence_parquet_row_count` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 512 | `write_results_empty_training_convergence_parquet_correct_schema` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 553 | `write_results_simulation_success_marker_conditional` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 601 | `write_results_simulation_metadata_scenarios_total` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 635 | `write_results_creates_dictionaries` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 658 | `write_results_codes_json_contains_operative_state` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 688 | `write_training_results_produces_complete_output` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 713 | `write_simulation_results_produces_metadata_and_success` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 726 | `split_functions_match_write_results_output` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 775 | `extract_max_iterations_from_config` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/results_writer.rs` | 781 | `training_metadata_has_max_iterations` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/scaling_report.rs` | 54 | `write_and_read_back_json` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/scaling_report.rs` | 71 | `tmp_file_is_cleaned_up` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 379 | `parquet_writer_config_default_values` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 389 | `costs_schema_field_count_and_names` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 431 | `costs_schema_types_and_nullability` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 479 | `hydros_schema_field_count_and_names` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 528 | `hydros_schema_nullable_fields` | 48 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 578 | `thermals_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 588 | `thermals_schema_gnl_fields_nullable` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 598 | `exchanges_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 608 | `buses_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 618 | `pumping_stations_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 628 | `contracts_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 638 | `non_controllables_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 648 | `inflow_lags_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 658 | `inflow_lags_schema_all_non_nullable` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 670 | `generic_violations_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 680 | `convergence_schema_field_count_and_types` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 699 | `convergence_schema_nullable_fields` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 726 | `iteration_timing_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 736 | `iteration_timing_schema_all_non_nullable` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 748 | `rank_timing_schema_field_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 758 | `rank_timing_schema_all_non_nullable` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 770 | `cut_selection_schema_field_count_and_types` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/schemas.rs` | 801 | `all_schema_functions_return_valid_schemas` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1727 | `build_costs_batch_from_two_stages` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1745 | `build_hydros_batch_derived_columns` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1815 | `build_exchanges_batch_net_flow_and_losses` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1863 | `build_costs_batch_block_id_nullable` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1883 | `simulation_parquet_writer_is_send` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1889 | `write_scenario_creates_hive_partitions` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1922 | `write_scenario_skips_empty_entity_types` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1958 | `finalize_returns_correct_counts` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 1982 | `finalize_partitions_written_contains_all_paths` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 2019 | `write_scenario_parquet_roundtrip_costs_row_count` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 2055 | `write_scenario_parquet_roundtrip_hydros_derived_mwh` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/simulation_writer.rs` | 2125 | `write_scenario_atomic_no_tmp_file_remaining` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/solver_stats_writer.rs` | 356 | `write_and_read_back` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/solver_stats_writer.rs` | 413 | `write_empty_rows` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/solver_stats_writer.rs` | 431 | `retry_histogram_sparse_encoding` | 80 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 785 | `write_then_read_round_trips` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 821 | `write_creates_parent_directory` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 844 | `write_correct_schema` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 881 | `write_row_count_matches_tree` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 915 | `write_atomic_no_tmp_file_remains` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 935 | `write_correct_row_tuples` | 72 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1013 | `write_empty_tree_zero_rows` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1072 | `write_then_read_inflow_stats_round_trips` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1104 | `write_inflow_stats_creates_parent_directory` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1129 | `write_inflow_stats_empty_rows_valid_schema` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1169 | `write_inflow_stats_no_tmp_file_remains` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1240 | `write_then_read_ar_coefficients_round_trips` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1274 | `write_ar_coefficients_creates_parent_directory` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1299 | `write_ar_coefficients_empty_rows_valid_schema` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1340 | `write_ar_coefficients_no_tmp_file_remains` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1454 | `write_then_read_correlation_json_round_trips_simple` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1487 | `write_then_read_correlation_json_round_trips_with_schedule` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1525 | `write_correlation_json_field_names_match_input_format` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1557 | `write_correlation_json_creates_parent_directory` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1579 | `write_correlation_json_no_tmp_file_remains` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1632 | `write_then_read_load_stats_round_trips` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1664 | `write_load_stats_empty_rows_valid_schema` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1704 | `write_load_stats_creates_parent_directory` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1754 | `write_fitting_report_two_hydros` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1799 | `write_fitting_report_empty_hydros` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1825 | `write_fitting_report_creates_parent_directory` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1847 | `write_fitting_report_no_tmp_file_remains` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/stochastic.rs` | 1871 | `write_correlation_json_multi_profile_round_trip` | 92 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 439 | `convergence_batch_from_empty_records` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 446 | `convergence_batch_field_count_and_types` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 461 | `convergence_batch_nullable_columns` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 484 | `iteration_timing_batch_field_count` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 503 | `iteration_timing_columns_six_decomposed_overhead` | 109 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 618 | `write_convergence_parquet_roundtrip` | 59 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 679 | `write_convergence_parquet_atomic_rename` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 703 | `writer_fails_if_training_dir_missing` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 713 | `writer_fails_if_timing_dir_missing` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 728 | `writer_writes_empty_training_output` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 766 | `writer_writes_five_records` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 801 | `writer_gap_percent_null_at_correct_row` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 844 | `write_cut_selection_empty_is_noop` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 856 | `write_cut_selection_roundtrip` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/output/training_writer.rs` | 900 | `write_cut_selection_with_budget_columns_roundtrip` | 79 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 442 | `test_parse_valid_penalties` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 485 | `test_parse_penalties_matches_manual_construction` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 531 | `test_parse_penalties_negative_value` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 571 | `test_parse_penalties_negative_deficit_segment_cost` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 614 | `test_parse_penalties_capped_last_segment` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 658 | `test_parse_penalties_uncapped_last_segment` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 672 | `test_parse_penalties_monotonic_deficit` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 719 | `test_parse_penalties_missing_field` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 741 | `test_parse_penalties_file_not_found` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 756 | `test_parse_penalties_negative_hydro_value` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 794 | `test_parse_penalties_fpha_turbined_cost_zero_valid` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 827 | `test_parse_penalties_fpha_turbined_cost_negative_invalid` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/penalties.rs` | 868 | `test_parse_penalties_three_deficit_segments` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/report.rs` | 186 | `test_generate_report_errors_and_warnings` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/report.rs` | 197 | `test_generate_report_empty_context` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/report.rs` | 208 | `test_report_to_json_valid` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/report.rs` | 220 | `test_report_entry_fields` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/report.rs` | 240 | `test_generate_report_does_not_consume_context` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 728 | `test_base_values_no_overrides` | 75 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 809 | `test_single_hydro_override` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 847 | `test_hydro_with_diversion_and_filling` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 878 | `test_hydro_without_diversion` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 895 | `test_thermal_override` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 929 | `test_line_override` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 967 | `test_pumping_override` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1002 | `test_contract_override_with_price` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1041 | `test_unknown_entity_id_skipped` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1075 | `test_empty_entities_no_panic` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1108 | `test_ac1_hydro_storage_override` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1142 | `test_ac2_thermal_max_generation_override` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1175 | `test_ac3_hydro_diversion_no_filling` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1194 | `test_ac4_empty_overrides_base_values` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1235 | `test_ac5_contract_price_override` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1268 | `test_water_withdrawal_negative_accepted` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1298 | `test_thermal_cost_override_null_block_id` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1347 | `test_thermal_cost_fallback_to_base` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1363 | `test_thermal_cost_block_id_row_ignored` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/bounds.rs` | 1399 | `test_thermal_zero_base_cost_no_override` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/exchange_factors.rs` | 157 | `test_empty_entries_returns_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/exchange_factors.rs` | 165 | `test_basic_resolution` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/exchange_factors.rs` | 176 | `test_unknown_line_id_skipped` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/exchange_factors.rs` | 186 | `test_unknown_stage_id_skipped` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 139 | `test_empty_constraints_empty_bounds` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 147 | `test_constraints_no_bounds` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 157 | `test_two_constraints_sparse_bounds` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 179 | `test_block_specific_bounds` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 198 | `test_unknown_constraint_id_skipped` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 215 | `test_ac_is_active_true` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 224 | `test_ac_is_active_false` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/generic_bounds.rs` | 233 | `test_ac_bounds_for_stage` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/load_factors.rs` | 144 | `test_empty_entries_returns_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/load_factors.rs` | 152 | `test_basic_resolution` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/load_factors.rs` | 166 | `test_unknown_bus_id_skipped` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/load_factors.rs` | 176 | `test_unknown_stage_id_skipped` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/load_factors.rs` | 186 | `test_pre_study_stages_excluded` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_bounds.rs` | 98 | `test_empty_overrides_uses_defaults` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_bounds.rs` | 109 | `test_overrides_applied` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_bounds.rs` | 125 | `test_unknown_ncs_id_skipped` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_bounds.rs` | 138 | `test_unknown_stage_id_skipped` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_bounds.rs` | 151 | `test_empty_ncs_returns_empty` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_bounds.rs` | 158 | `test_zero_stages_returns_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_factors.rs` | 150 | `test_empty_entries_returns_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_factors.rs` | 158 | `test_basic_resolution` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_factors.rs` | 173 | `test_unknown_ncs_id_skipped` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_factors.rs` | 183 | `test_unknown_stage_id_skipped` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/ncs_factors.rs` | 193 | `test_pre_study_stages_excluded` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 667 | `test_tier2_only_no_overrides` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 688 | `test_single_field_override` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 724 | `test_full_11_field_hydro_override` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 773 | `test_partial_override_mix_some_none` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 808 | `test_bus_override` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 833 | `test_line_override` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 861 | `test_ncs_override` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 900 | `test_unknown_entity_id_skipped` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 924 | `test_multiple_entities_multiple_stages` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 977 | `test_empty_entities_no_panic` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 996 | `test_empty_overrides_entity_level_values` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 1012 | `test_ac_hydro_spillage_cost_override` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 1038 | `test_ac_bus_excess_cost_override` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/resolution/penalties.rs` | 1058 | `test_ac_single_field_filling_target_violation_cost` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 240 | `test_valid_6_rows_sorted_by_hydro_stage_lag` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 267 | `test_lag_zero_is_schema_error` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 288 | `test_missing_coefficient_column` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 322 | `test_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 330 | `test_coefficient_values_preserved` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 345 | `test_valid_residual_std_ratio_preserved` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 367 | `test_residual_std_ratio_zero_is_schema_error` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 388 | `test_residual_std_ratio_above_one_is_schema_error` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 409 | `test_residual_std_ratio_nan_is_schema_error` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/ar_coefficients.rs` | 426 | `test_missing_residual_std_ratio_column` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/assembly.rs` | 193 | `test_assemble_inflow_models_matching_join` | 72 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/assembly.rs` | 267 | `test_assemble_inflow_models_no_coefficients` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/assembly.rs` | 282 | `test_assemble_inflow_models_orphaned_coefficients` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/assembly.rs` | 315 | `test_assemble_inflow_models_both_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/assembly.rs` | 329 | `test_assemble_inflow_models_empty_stats_non_empty_ar_returns_empty` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/assembly.rs` | 352 | `test_assemble_load_models_four_rows` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/assembly.rs` | 405 | `test_assemble_load_models_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 430 | `test_valid_3x3_identity_matrix` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 480 | `test_two_profiles_with_schedule` | 58 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 543 | `test_no_schedule_produces_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 554 | `test_non_symmetric_matrix_rejected` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 597 | `test_diagonal_not_one_rejected` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 640 | `test_element_greater_than_one_rejected` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 683 | `test_element_less_than_minus_one_rejected` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 726 | `test_non_square_matrix_rejected` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 766 | `test_matrix_row_count_mismatch_rejected` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 806 | `test_schedule_unknown_profile_rejected` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 848 | `test_empty_profiles_rejected` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 871 | `test_empty_method_rejected` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 904 | `test_load_correlation_none_returns_default` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 916 | `test_single_entity_1x1_matrix_valid` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/correlation.rs` | 944 | `parse_accepts_cholesky_method` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 539 | `test_parse_external_inflow_valid` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 571 | `test_parse_external_inflow_nan_rejected` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 586 | `test_parse_external_inflow_missing_hydro_id_column` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 617 | `test_parse_external_inflow_negative_scenario_id_rejected` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 632 | `test_parse_external_inflow_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 640 | `test_load_external_inflow_scenarios_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 646 | `test_parse_external_load_valid` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 678 | `test_parse_external_load_missing_bus_id_column` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 709 | `test_parse_external_load_nonfinite_value_rejected` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 724 | `test_load_external_load_scenarios_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 730 | `test_parse_external_ncs_valid` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 762 | `test_parse_external_ncs_negative_scenario_id_rejected` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 777 | `test_parse_external_ncs_infinity_value_rejected` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/external.rs` | 792 | `test_load_external_ncs_scenarios_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_history.rs` | 193 | `test_valid_24_rows_sorted_by_hydro_date` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_history.rs` | 230 | `test_infinite_value_m3s` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_history.rs` | 248 | `test_missing_date_column` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_history.rs` | 280 | `test_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_history.rs` | 288 | `test_date_values_preserved` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_history.rs` | 302 | `test_nan_value_m3s` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_stats.rs` | 223 | `test_valid_4_rows_sorted_by_hydro_stage` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_stats.rs` | 247 | `test_missing_mean_m3s_column` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_stats.rs` | 281 | `test_negative_std_m3s` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_stats.rs` | 298 | `test_nan_mean_m3s` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_stats.rs` | 315 | `test_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/inflow_stats.rs` | 323 | `test_declaration_order_invariance` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 330 | `test_valid_2_entries_sorted_and_block_factors_correct` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 354 | `test_entries_sorted_by_bus_stage` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 390 | `test_block_factors_sorted_by_block_id` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 418 | `test_zero_factor_rejected` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 446 | `test_negative_factor_rejected` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 475 | `test_duplicate_bus_stage_rejected` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 512 | `test_empty_load_factors_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 523 | `test_missing_block_factors_field_is_parse_error` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 544 | `test_very_small_positive_factor_accepted` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_factors.rs` | 565 | `test_declaration_order_invariance` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_stats.rs` | 228 | `test_valid_4_rows_sorted_by_bus_stage` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_stats.rs` | 256 | `test_zero_std_mw_is_accepted` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_stats.rs` | 269 | `test_negative_std_mw` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_stats.rs` | 289 | `test_nan_mean_mw` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_stats.rs` | 309 | `test_missing_mean_mw_column` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_stats.rs` | 346 | `test_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/load_stats.rs` | 357 | `test_declaration_order_invariance` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 422 | `test_load_inflow_seasonal_stats_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 429 | `test_load_inflow_ar_coefficients_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 436 | `test_load_inflow_history_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 443 | `test_load_load_seasonal_stats_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 450 | `test_load_load_factors_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 457 | `test_load_correlation_none_returns_default` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 465 | `test_load_external_inflow_scenarios_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 472 | `test_load_external_load_scenarios_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 479 | `test_load_external_ncs_scenarios_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 487 | `test_load_scenarios_all_flags_false_returns_empty` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 534 | `test_load_noise_openings_none_returns_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/mod.rs` | 542 | `test_load_scenarios_noise_openings_absent` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/noise_openings.rs` | 385 | `parse_valid_file_returns_sorted_rows` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/noise_openings.rs` | 425 | `parse_missing_column_returns_schema_error` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/noise_openings.rs` | 462 | `validate_correct_dimensions_returns_ok` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/noise_openings.rs` | 472 | `validate_dimension_mismatch_returns_error` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/noise_openings.rs` | 493 | `validate_stage_count_mismatch_returns_error` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/noise_openings.rs` | 514 | `validate_missing_openings_returns_error` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/noise_openings.rs` | 546 | `assemble_produces_correct_opening_tree` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 275 | `test_valid_2_entries` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 286 | `test_sorted_output` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 300 | `test_block_factors_sorted` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 321 | `test_zero_factor_rejected` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 338 | `test_negative_factor_rejected` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 355 | `test_duplicate_rejected` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 373 | `test_empty_array` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_factors.rs` | 381 | `test_declaration_order_invariance` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 193 | `test_valid_4_rows_sorted_by_ncs_stage` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 220 | `test_zero_std_is_accepted` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 232 | `test_negative_std` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 251 | `test_nan_mean` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 270 | `test_mean_out_of_range` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 293 | `test_missing_mean_column` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 329 | `test_empty_parquet_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/scenarios/non_controllable_stats.rs` | 339 | `test_declaration_order_invariance` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/schema.rs` | 135 | `test_generate_schemas_returns_expected_count` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/schema.rs` | 146 | `test_all_schema_filenames_and_values_non_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/schema.rs` | 156 | `test_all_schemas_are_objects` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/schema.rs` | 169 | `test_all_schemas_have_structure_keys` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/schema.rs` | 190 | `test_config_schema_contains_expected_fields` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/schema.rs` | 215 | `test_buses_schema_contains_buses_array` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/schema.rs` | 238 | `test_all_expected_schema_filenames_present` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 928 | `test_parse_valid_3_study_6_pre_study` | 67 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1001 | `test_parse_cvar_risk_measure` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1035 | `test_stages_without_scenario_source_succeeds` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1053 | `test_stages_with_scenario_source_rejected` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1086 | `test_parse_season_definitions_12_monthly` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1129 | `test_no_season_definitions_gives_none_season_map` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1142 | `test_parse_chronological_block_mode` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1167 | `test_parse_transition_discount_rate_override` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1209 | `test_error_duplicate_stage_ids` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1238 | `test_error_duplicate_pre_study_stage_ids` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1269 | `test_error_id_collision_study_and_pre_study` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1299 | `test_error_num_scenarios_zero` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1327 | `test_error_negative_annual_discount_rate` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1354 | `test_error_block_hours_zero` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1381 | `test_error_cvar_alpha_zero` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1410 | `test_error_cvar_lambda_out_of_range` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1439 | `test_error_start_date_not_before_end_date` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1466 | `test_error_non_contiguous_block_ids` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1497 | `test_error_invalid_date_string` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1526 | `test_declaration_order_invariance` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1557 | `test_file_not_found` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1567 | `test_invalid_json_gives_parse_error` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1578 | `test_cyclic_policy_graph_type` | 21 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1604 | `test_build_season_stage_map_basic` | 38 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1645 | `test_build_season_stage_map_empty` | 4 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1652 | `test_build_season_stage_map_none_season_ids` | 25 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/stages.rs` | 1681 | `test_convert_noise_method_historical_residuals` | 9 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 330 | `test_parse_valid_buses` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 381 | `test_duplicate_bus_id` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 411 | `test_declaration_order_invariance` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 443 | `test_entity_deficit_segment_negative_cost` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 475 | `test_entity_deficit_segment_last_not_uncapped` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 508 | `test_entity_deficit_segment_non_monotonic` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 543 | `test_file_not_found` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 559 | `test_invalid_json` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 573 | `test_empty_buses_array` | 7 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/buses.rs` | 583 | `test_excess_cost_always_from_global` | 13 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 285 | `test_parse_valid_contracts` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 341 | `test_unknown_contract_type` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 365 | `test_duplicate_contract_id` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 403 | `test_negative_limits_min_mw` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 433 | `test_max_mw_less_than_min_mw` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 465 | `test_negative_price_per_mwh_is_valid_for_export` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 491 | `test_declaration_order_invariance` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 543 | `test_file_not_found` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 559 | `test_invalid_json` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 572 | `test_empty_contracts_array` | 6 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 581 | `test_min_equals_max_mw_is_valid` | 18 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/energy_contracts.rs` | 602 | `test_zero_price_is_valid` | 18 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 986 | `test_parse_valid_full_and_minimal` | 84 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1076 | `test_parse_fpha_generation_model` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1106 | `test_parse_linearized_head_generation_model` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1142 | `test_parse_tailrace_piecewise` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1183 | `test_parse_hydraulic_losses_constant` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1219 | `test_entity_level_penalty_partial_override` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1262 | `test_entity_level_penalty_all_global_defaults` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1320 | `test_filling_inflow_defaults_to_zero` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1357 | `test_duplicate_hydro_id` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1394 | `test_invalid_reservoir_bounds_min_gt_max` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1429 | `test_invalid_reservoir_negative_min` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1455 | `test_invalid_reservoir_negative_max` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1483 | `test_invalid_generation_bounds_max_lt_min` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1518 | `test_invalid_turbined_bounds_max_lt_min` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1544 | `test_invalid_outflow_negative_min` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1572 | `test_invalid_evaporation_wrong_length` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1610 | `test_unknown_generation_model` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1638 | `test_declaration_order_invariance` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1685 | `test_file_not_found` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1701 | `test_invalid_json` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1715 | `test_empty_hydros_array` | 7 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1725 | `test_reservoir_min_equals_max_is_valid` | 23 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1771 | `test_evaporation_reference_volumes_happy_path` | 34 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1810 | `test_evaporation_reference_volumes_absent_is_none` | 27 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1841 | `test_evaporation_reference_volumes_wrong_length` | 38 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1883 | `test_evaporation_reference_volumes_nan_value` | 51 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1939 | `test_evaporation_reference_volumes_exceeds_max_storage` | 38 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 1982 | `test_evaporation_reference_volumes_below_min_storage` | 38 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 2026 | `test_no_evaporation_block_both_fields_none` | 9 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/hydros.rs` | 2038 | `test_schema_field_is_ignored` | 24 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 304 | `test_parse_valid_lines` | 60 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 371 | `test_duplicate_line_id` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 406 | `test_negative_direct_mw` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 435 | `test_negative_reverse_mw` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 464 | `test_negative_losses_percent` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 497 | `test_declaration_order_invariance` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 541 | `test_file_not_found` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 557 | `test_invalid_json` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 571 | `test_empty_lines_array` | 7 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 581 | `test_losses_percent_defaults_to_zero` | 18 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/lines.rs` | 602 | `test_zero_capacity_is_valid` | 17 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/mod.rs` | 148 | `test_load_ncs_none_returns_empty` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/mod.rs` | 159 | `test_load_pumping_stations_none_returns_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/mod.rs` | 169 | `test_load_energy_contracts_none_returns_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/non_controllable.rs` | 264 | `test_parse_valid_ncs` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/non_controllable.rs` | 319 | `test_duplicate_ncs_id` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/non_controllable.rs` | 348 | `test_negative_max_generation_mw` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/non_controllable.rs` | 377 | `test_declaration_order_invariance` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/non_controllable.rs` | 410 | `test_file_not_found` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/non_controllable.rs` | 427 | `test_invalid_json` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/non_controllable.rs` | 441 | `test_empty_ncs_array` | 7 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 262 | `test_parse_valid_pumping_stations` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 319 | `test_duplicate_pumping_station_id` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 357 | `test_negative_consumption` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 387 | `test_negative_flow_min` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 417 | `test_max_flow_less_than_min_flow` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 450 | `test_declaration_order_invariance` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 502 | `test_file_not_found` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 518 | `test_invalid_json` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 531 | `test_empty_pumping_stations_array` | 6 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/pumping_stations.rs` | 540 | `test_min_equals_max_flow_is_valid` | 18 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 305 | `test_parse_valid_thermals` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 347 | `test_duplicate_thermal_id` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 384 | `test_parse_cost_per_mwh_75` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 410 | `test_legacy_array_cost_format_fails` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 431 | `test_negative_cost_per_mwh` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 458 | `test_negative_min_generation_mw` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 487 | `test_negative_max_generation_mw` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 516 | `test_max_mw_less_than_min_mw` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 548 | `test_declaration_order_invariance` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 596 | `test_gnl_thermal_not_rejected` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 608 | `test_file_not_found` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 623 | `test_invalid_json` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 636 | `test_empty_thermals_array` | 6 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 645 | `test_optional_stage_ids_default_to_none` | 15 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 663 | `test_min_equals_max_is_valid` | 17 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/system/thermals.rs` | 683 | `test_zero_cost_is_valid` | 17 | `` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 606 | `test_valid_coverage_no_errors` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 651 | `test_missing_inflow_stats_one_hydro_one_stage` | 62 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 720 | `test_correlation_matrix_row_count_mismatch` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 771 | `test_correlation_matrix_non_square_row` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 816 | `test_empty_optional_data_no_false_positives` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 842 | `test_fpha_hydro_missing_hyperplane_rows` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 891 | `test_hydro_lifecycle_entry_stage_id_skips_earlier_stages` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 922 | `test_hydro_lifecycle_exit_stage_id_skips_later_stages` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 952 | `test_pre_study_stages_not_checked` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 988 | `test_load_stats_missing_for_one_stage` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 1027 | `test_correlation_schedule_missing_profile` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 1059 | `test_linearized_head_hydro_missing_geometry` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/dimensional.rs` | 1091 | `test_all_rules_checked_independently` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/mod.rs` | 287 | `test_context_empty` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/mod.rs` | 305 | `test_context_errors_collected` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/mod.rs` | 337 | `test_context_warnings_not_errors` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/mod.rs` | 367 | `test_context_into_result_with_errors` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/mod.rs` | 386 | `test_context_into_result_warnings_only_is_ok` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/mod.rs` | 401 | `test_context_into_result_multiple_errors_joined` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/mod.rs` | 429 | `test_error_kind_default_severity` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1214 | `test_all_valid_references_no_errors` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1231 | `test_line_invalid_source_bus` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1265 | `test_hydro_invalid_downstream_id` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1301 | `test_empty_optional_collections_no_errors` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1319 | `test_multiple_invalid_references_all_collected` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1367 | `test_hydro_valid_bus_ref` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1379 | `test_hydro_invalid_bus_ref` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1400 | `test_hydro_downstream_id_none_no_error` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1417 | `test_hydro_diversion_none_no_error` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1434 | `test_hydro_diversion_invalid_downstream` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1459 | `test_pumping_valid_refs` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1472 | `test_pumping_invalid_source_hydro` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1494 | `test_inflow_seasonal_stats_invalid_hydro_ref` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1520 | `test_load_seasonal_stats_invalid_bus_ref` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1547 | `test_correlation_entity_inflow_invalid_hydro` | 51 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1602 | `test_correlation_entity_inflow_valid_hydro` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1637 | `test_thermal_bounds_invalid_thermal_ref` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1665 | `test_hydro_bounds_invalid_hydro_ref` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1700 | `test_line_bounds_invalid_line_ref` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1726 | `test_generic_constraint_bounds_invalid_constraint_ref` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1752 | `test_bus_penalty_override_invalid_bus_ref` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1777 | `test_ncs_penalty_override_invalid_ncs_ref` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1802 | `test_ncs_penalty_override_valid_ncs_ref` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1820 | `test_load_factors_invalid_bus_ref` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1856 | `test_load_factors_invalid_stage_ref` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1892 | `test_load_factors_valid_refs_no_error` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1915 | `test_ncs_bounds_valid_refs_no_error` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1935 | `test_ncs_bounds_invalid_ncs_ref` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1959 | `test_ncs_bounds_negative_available_generation` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 1983 | `test_ncs_factors_valid_refs_no_error` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 2006 | `test_ncs_factors_invalid_ncs_ref` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 2038 | `test_ncs_factors_invalid_stage_ref` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/referential.rs` | 2071 | `test_ncs_factors_negative_factor` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/schema.rs` | 888 | `test_valid_case_returns_some_and_no_errors` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/schema.rs` | 930 | `test_invalid_json_returns_none_and_parse_error` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/schema.rs` | 974 | `test_two_invalid_files_both_errors_collected` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/schema.rs` | 1025 | `test_absent_optional_file_yields_none_no_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/schema.rs` | 1056 | `test_map_load_error_io_error` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/schema.rs` | 1075 | `test_map_load_error_parse_error` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/schema.rs` | 1091 | `test_map_load_error_schema_error` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2156 | `test_cascade_acyclic_valid` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2174 | `test_cascade_cycle_detected` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2197 | `test_cascade_empty_hydros` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2209 | `test_hydro_storage_min_greater_than_max` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2243 | `test_hydro_storage_equal_bounds_valid` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2268 | `test_hydro_turbine_min_greater_than_max` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2296 | `test_hydro_outflow_no_max_no_error` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2315 | `test_hydro_outflow_min_greater_than_max` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2336 | `test_hydro_lifecycle_entry_gte_exit` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2360 | `test_hydro_lifecycle_only_entry_no_error` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2383 | `test_hydro_lifecycle_valid` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2404 | `test_geometry_empty_no_error` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2420 | `test_geometry_valid_monotonic` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2445 | `test_geometry_non_monotonic_volume` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2487 | `test_geometry_non_monotonic_height` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2521 | `test_fpha_one_plane_valid` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2542 | `test_fpha_two_planes_valid` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2563 | `test_fpha_minimum_planes_valid` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2590 | `test_fpha_negative_gamma_v` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2616 | `test_fpha_positive_gamma_s` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2642 | `test_fpha_gamma_s_zero_valid` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2669 | `test_fpha_gamma_v_zero_valid` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2692 | `test_fpha_empty_no_error` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2710 | `test_thermal_generation_min_greater_than_max` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2738 | `test_thermal_generation_equal_bounds_valid` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2758 | `test_all_rules_checked_no_short_circuit` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2788 | `test_ac1_valid_data_no_errors` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2815 | `test_ac2_hydro_storage_bounds_error` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2843 | `test_ac3_cycle_detected` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2863 | `test_ac4_geometry_non_monotonic_volume_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 2894 | `test_ac5_empty_geometry_and_fpha_no_false_positives` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3073 | `test_5b_all_valid_no_errors` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3106 | `test_5b_transition_invalid_source_id` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3134 | `test_5b_transition_invalid_target_id` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3166 | `test_5b_transition_probability_sum_wrong` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3204 | `test_5b_transition_probability_sum_valid` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3245 | `test_5b_cyclic_zero_discount_rate` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3270 | `test_5b_cyclic_positive_discount_rate_valid` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3299 | `test_5b_block_zero_duration` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3327 | `test_5b_block_positive_duration_valid` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3359 | `test_5b_cvar_alpha_zero_invalid` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3386 | `test_5b_cvar_lambda_out_of_range` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3416 | `test_5b_penalty_ordering_filling_less_than_storage_violation` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3459 | `test_5b_fpha_penalty_violated` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3497 | `test_5b_fpha_penalty_zero_valid` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3524 | `test_5b_fpha_penalty_equal_spillage_valid` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3552 | `test_5b_fpha_penalty_valid` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3582 | `test_5b_inflow_std_zero_warning` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3620 | `test_5b_residual_std_ratio_consistent_no_error` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3665 | `test_5b_residual_std_ratio_inconsistent_error` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3716 | `test_5b_correlation_asymmetric` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3756 | `test_5b_correlation_diagonal_not_one` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3788 | `test_5b_correlation_off_diagonal_out_of_range` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3818 | `test_5b_correlation_valid_symmetric` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3842 | `test_5b_no_correlation_no_inflow_no_false_positives` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3884 | `test_5b_load_factors_invalid_block_id` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3928 | `test_5b_load_factors_deterministic_bus_warning` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 3974 | `test_5b_load_factors_empty_no_errors` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4161 | `test_estimation_requires_season_definitions` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4187 | `test_estimation_warns_low_observations` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4222 | `test_estimation_error_missing_hydro` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4249 | `test_no_estimation_when_stats_and_coefficients_present` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4292 | `test_estimation_active_when_stats_present_but_coefficients_absent` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4445 | `test_rule22_lags_enabled_no_past_inflows_errors` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4486 | `test_rule23_sufficient_past_inflows_no_error` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4519 | `test_rule23_insufficient_past_inflows_errors` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4553 | `test_rules_skip_when_lags_disabled` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4575 | `test_rules_skip_when_par_order_zero` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4595 | `test_rule24_unknown_hydro_in_past_inflows_errors` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4670 | `test_past_inflows_season_ids_invalid_season` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4705 | `test_past_inflows_season_ids_valid` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4739 | `test_past_inflows_season_ids_no_season_map_skipped` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4764 | `test_sobol_non_power_of_2_emits_warning` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4806 | `test_sobol_power_of_2_no_warning` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4838 | `test_saa_non_power_of_2_no_warning` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4871 | `test_sobol_mixed_stages_only_warns_non_power` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4960 | `test_training_external_inflow_without_file_is_error` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 4997 | `test_simulation_external_load_without_file_is_error` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5032 | `test_training_external_inflow_with_file_is_ok` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5121 | `test_season_id_range_coverage_valid_monthly` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5154 | `test_season_id_range_coverage_undefined_season` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5196 | `test_season_id_range_coverage_no_season_map` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5225 | `test_season_id_range_coverage_multiple_violations` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5328 | `test_resolution_consistency_monthly_valid` | 103 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5436 | `test_resolution_consistency_mixed_monthly_quarterly` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5498 | `test_resolution_consistency_disjoint_resolutions` | 130 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5633 | `test_resolution_consistency_weekly_vs_monthly` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5692 | `test_observation_coverage_all_seasons_have_obs` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5719 | `test_observation_coverage_season_missing_obs_non_external` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5769 | `test_observation_coverage_season_missing_obs_external` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5812 | `test_contiguity_no_gaps` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5844 | `test_contiguity_gap_detected` | 80 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5931 | `test_observation_alignment_valid_monthly` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 5968 | `test_observation_alignment_duplicate_obs` | 88 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 6069 | `test_observation_alignment_coarser_than_season` | 82 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 6155 | `test_observation_alignment_no_season_map` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 6174 | `test_observation_alignment_estimation_inactive` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 6208 | `test_observation_alignment_partial_boundary_years_no_error` | 57 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/semantic.rs` | 6271 | `test_observation_alignment_missing_interior_season_produces_error` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 424 | `test_structural_all_required_present` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 464 | `test_structural_missing_required_hydros` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 496 | `test_structural_optional_absent_no_error` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 518 | `test_structural_optional_present_in_manifest` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 537 | `test_structural_multiple_missing_required` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 566 | `test_structural_manifest_fields_count` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 584 | `test_manifest_noise_openings_absent` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/src/validation/structural.rs` | 603 | `test_manifest_noise_openings_present` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 15 | `test_minimal_config_all_defaults` | 66 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 83 | `test_config_explicit_seed_preserved` | 18 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 103 | `test_config_absent_seed_is_none` | 16 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 121 | `test_config_all_sections_explicit_no_defaults_applied` | 69 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 195 | `test_config_absent_modeling_uses_defaults` | 20 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 217 | `test_config_absent_simulation_uses_defaults` | 20 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 239 | `test_config_absent_exports_uses_defaults` | 20 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 282 | `test_stages_absent_scenario_source_succeeds` | 7 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/defaults_cascade.rs` | 293 | `test_stages_variable_branching_factor_preserved` | 51 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/genericity_gate.rs` | 6 | `infrastructure_genericity_no_sddp_references` | 19 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 26 | `test_minimal_valid_case` | 33 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 65 | `test_multi_entity_case` | 45 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 116 | `test_missing_required_file` | 20 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 142 | `test_malformed_json` | 27 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 176 | `test_referential_integrity_violation` | 20 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 208 | `test_inflow_history_wired_into_system` | 148 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 363 | `test_external_scenarios_wired_into_system` | 97 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 466 | `test_inflow_history_absent_returns_empty` | 13 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 485 | `test_external_scenarios_absent_returns_empty` | 13 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 506 | `test_external_load_scenarios_wired_into_system` | 64 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 578 | `test_external_ncs_scenarios_wired_into_system` | 81 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/integration.rs` | 666 | `test_postcard_round_trip` | 47 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/invariance.rs` | 160 | `test_bus_ordering_invariance` | 51 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/invariance.rs` | 221 | `test_stage_ordering_invariance` | 43 | `` | `integration` | `generic` |  |
| `cobre-io` | `crates/cobre-io/tests/invariance.rs` | 274 | `test_full_case_ordering_invariance` | 44 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/banner.rs` | 70 | `test_render_banner_colored_contains_ansi_escapes` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/banner.rs` | 77 | `test_render_banner_plain_no_ansi_escapes` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/banner.rs` | 83 | `test_render_banner_contains_version` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/banner.rs` | 93 | `test_render_banner_contains_unicode_busbars` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/banner.rs` | 100 | `test_print_banner_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 325 | `broadcast_value_local_round_trips_simple` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 339 | `broadcast_value_local_round_trips_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 351 | `broadcast_value_local_round_trips_config_like` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 373 | `broadcast_value_returns_err_when_root_passes_none` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 392 | `broadcast_value_round_trips_u64` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 408 | `broadcast_opening_tree_round_trips_via_postcard` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 432 | `broadcast_config_propagates_training_enabled` | 21 | `tests` | `unit` | `warm-start-config-flag` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 456 | `broadcast_config_roundtrips_via_postcard_after_warm_start_basis_mode_deletion` | 43 | `tests` | `unit` | `warm-start-config-flag` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 506 | `broadcast_optional_opening_tree_local_round_trips` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 193 | `test_init_list_prints_template_names` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 200 | `test_init_list_execute_returns_ok` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 211 | `test_init_unknown_template_returns_validation_error` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 227 | `test_init_creates_directory_and_files` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 260 | `test_init_config_json_contains_schema_url` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 280 | `test_init_system_json_files_contain_schema_urls` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 314 | `test_init_existing_non_empty_dir_without_force_returns_io_error` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/init.rs` | 332 | `test_init_existing_non_empty_dir_with_force_succeeds` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 212 | `report_output_serializes_to_valid_json` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 232 | `report_output_contains_iterations_field` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 253 | `report_output_with_simulation_not_null` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 276 | `report_output_status_from_training_metadata` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 295 | `read_optional_returns_none_for_nonexistent_file` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 303 | `read_optional_returns_value_for_existing_file` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 316 | `read_optional_returns_internal_error_for_malformed_json` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/report.rs` | 329 | `read_training_metadata_returns_io_error_for_missing_file` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/run.rs` | 1767 | `test_resolve_thread_count_cli_value` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/run.rs` | 1772 | `test_resolve_thread_count_default` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/summary.rs` | 265 | `summary_args_parses_output_dir` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/summary.rs` | 273 | `construct_training_summary_from_metadata` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/summary.rs` | 294 | `convergence_fallback_uses_metadata_gap_percent` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/summary.rs` | 304 | `convergence_fallback_gap_none_when_metadata_has_no_gap` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/summary.rs` | 314 | `build_training_summary_gap_defaults_to_zero_when_none` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/summary.rs` | 327 | `build_training_summary_converged_at_none_when_metadata_has_none` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/validate.rs` | 170 | `format_report_contains_error_label` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/validate.rs` | 180 | `format_report_contains_warning_label` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/validate.rs` | 190 | `format_report_contains_file_path` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/validate.rs` | 200 | `format_report_contains_error_message` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/validate.rs` | 210 | `format_report_summary_header_present` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/validate.rs` | 220 | `format_entry_with_entity` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/commands/validate.rs` | 234 | `format_entry_without_entity` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 254 | `validation_exit_code_is_1` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 262 | `io_exit_code_is_2` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 271 | `solver_exit_code_is_3` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 279 | `internal_exit_code_is_4` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 287 | `display_validation_non_empty` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 297 | `display_io_non_empty` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 309 | `display_solver_non_empty` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 319 | `display_internal_non_empty` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 329 | `from_load_error_io_maps_to_cli_io` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 345 | `from_load_error_constraint_maps_to_validation` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 358 | `from_load_error_schema_maps_to_validation` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 375 | `from_load_error_cross_reference_maps_to_validation` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 393 | `from_sddp_error_infeasible_maps_to_solver` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 408 | `from_sddp_error_solver_maps_to_solver` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 419 | `from_sddp_error_io_maps_to_cli_io_or_validation` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 437 | `from_sddp_error_validation_maps_to_validation` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 448 | `from_sddp_error_communication_maps_to_internal` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 464 | `from_sddp_error_stochastic_maps_to_internal` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 478 | `from_sddp_error_simulation_maps_to_internal` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 489 | `from_simulation_error_lp_infeasible_maps_to_solver` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 504 | `from_simulation_error_solver_error_maps_to_solver` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 519 | `from_simulation_error_io_maps_to_internal` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 532 | `from_simulation_error_policy_incompatible_maps_to_internal` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 545 | `from_simulation_error_channel_closed_maps_to_internal` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/error.rs` | 556 | `from_backend_error_maps_to_internal` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/main.rs` | 153 | `test_resolve_color_always_enables_color` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/main.rs` | 162 | `test_resolve_color_never_disables_color` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 360 | `test_progress_handle_training_events_returned` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 375 | `test_progress_handle_simulation_events_returned` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 414 | `test_progress_handle_returns_all_events` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 449 | `test_empty_channel_returns_empty_vec` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 463 | `test_training_only_no_simulation_events` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 491 | `test_simulation_progress_message_with_statistics` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 523 | `test_simulation_progress_message_single_scenario` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 554 | `test_non_ui_events_are_collected` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 595 | `test_simulation_progress_five_events_no_panic` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 634 | `test_simulation_progress_accumulator_costs_collected_correctly` | 69 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/progress.rs` | 705 | `test_simulation_progress_accumulator_three_events_no_panic` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 872 | `test_format_duration_seconds` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 877 | `test_format_duration_minutes` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 882 | `test_format_duration_hours` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 887 | `test_format_duration_exactly_zero` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 892 | `test_format_duration_exactly_60s` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 897 | `test_format_duration_exactly_1h` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 902 | `test_format_summary_training_only` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 917 | `test_format_summary_with_simulation` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 950 | `test_format_summary_contains_bounds` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 968 | `test_format_summary_converged_detail` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 988 | `test_format_summary_non_converged_shows_reason` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1008 | `test_format_summary_time_3m42s` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1026 | `test_format_summary_scientific_notation` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1044 | `test_format_summary_output_dir` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1059 | `test_format_summary_cut_stats` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1078 | `test_print_summary_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1084 | `test_print_summary_with_simulation_does_not_panic` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1170 | `format_hydro_model_summary_with_fpha_contains_key_terms` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1190 | `format_hydro_model_summary_without_fpha_contains_constant_not_fpha` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1206 | `format_hydro_model_summary_contains_header` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1218 | `format_hydro_model_summary_mixed_production_line` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1234 | `format_hydro_model_summary_all_fpha_shows_filename` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1254 | `format_hydro_model_summary_singular_evaporation` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1275 | `format_hydro_model_summary_plural_evaporation` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1287 | `print_hydro_model_summary_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1293 | `print_hydro_model_summary_all_constant_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1299 | `print_hydro_model_summary_all_fpha_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1308 | `test_evaporation_line_all_midpoint` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1333 | `test_evaporation_line_all_user_supplied` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1366 | `test_evaporation_line_mixed` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1395 | `test_evaporation_line_no_evaporation` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1444 | `print_provenance_summary_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1450 | `print_provenance_summary_deterministic_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1456 | `format_provenance_summary_contains_all_section_keys` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1486 | `format_provenance_summary_full_estimation_includes_ar_detail` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1500 | `format_provenance_summary_deterministic_no_ar_detail` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1519 | `format_provenance_summary_user_file_source` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1547 | `test_format_rank_list_empty` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1552 | `test_format_rank_list_single` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1557 | `test_format_rank_list_contiguous` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1562 | `test_format_rank_list_non_contiguous` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/summary.rs` | 1567 | `test_format_rank_list_mixed` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/templates.rs` | 150 | `test_available_templates_contains_1dtoy` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/templates.rs` | 156 | `test_find_template_1dtoy_returns_some` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/templates.rs` | 163 | `test_find_template_unknown_returns_none` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/templates.rs` | 168 | `test_1dtoy_template_has_files` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/templates.rs` | 174 | `test_1dtoy_files_have_descriptions` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/templates.rs` | 182 | `test_1dtoy_files_have_relative_paths` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/src/templates.rs` | 191 | `test_1dtoy_config_json_content_matches_source` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_color.rs` | 129 | `color_always_flag_forces_ansi_in_banner` | 21 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_color.rs` | 159 | `color_never_flag_suppresses_ansi_in_banner` | 19 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_color.rs` | 182 | `color_always_global_flag_before_subcommand_is_accepted` | 18 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_color.rs` | 203 | `cobre_color_env_always_forces_ansi` | 19 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_color.rs` | 225 | `force_color_env_forces_ansi` | 19 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_color.rs` | 251 | `cobre_color_env_invalid_value_is_silently_ignored` | 18 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 127 | `training_only_exits_0` | 9 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 138 | `training_only_stdout_is_valid_json` | 28 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 168 | `training_only_simulation_key_is_null` | 18 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 190 | `full_results_exits_0` | 9 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 201 | `full_results_both_training_and_simulation_present` | 27 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 232 | `nonexistent_directory_exits_2` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 241 | `nonexistent_directory_stderr_contains_error` | 8 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 253 | `status_field_is_valid_string` | 26 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 283 | `output_directory_field_contains_absolute_path` | 21 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_report.rs` | 306 | `missing_training_manifest_exits_2` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 118 | `valid_case_exits_0` | 16 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 136 | `valid_case_creates_training_metadata` | 18 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 156 | `valid_case_creates_convergence_parquet` | 18 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 176 | `disabled_simulation_does_not_produce_manifest` | 18 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 196 | `custom_output_dir_receives_training_artifacts` | 20 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 218 | `missing_required_file_exits_1` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 231 | `missing_required_file_stderr_contains_validation_error` | 12 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 245 | `nonexistent_path_exits_2` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 254 | `nonexistent_path_stderr_contains_io_error` | 8 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_run.rs` | 264 | `test_run_quiet_suppresses_banner_and_summary` | 17 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_schema.rs` | 20 | `test_schema_export_writes_files` | 64 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_schema.rs` | 88 | `test_schema_export_default_dir` | 21 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_schema.rs` | 113 | `test_schema_export_creates_dir` | 30 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_schema.rs` | 148 | `test_schema_export_unwritable_dir_exits_nonzero` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 17 | `help_exits_0_and_lists_subcommands` | 10 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 29 | `run_help_exits_0_and_lists_flags` | 10 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 46 | `run_threads_zero_exits_with_clap_error` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 61 | `run_threads_positive_is_accepted_by_clap` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 70 | `version_exits_0_and_contains_version_string` | 9 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 81 | `version_exits_0_and_stdout_contains_cobre_prefix` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 90 | `version_stdout_contains_solver_highs` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 99 | `run_nonexistent_path_exits_2_with_io_error` | 8 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 109 | `validate_nonexistent_path_exits_2` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 118 | `report_nonexistent_path_exits_2` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_smoke.rs` | 127 | `unknown_subcommand_exits_nonzero` | 3 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 105 | `valid_case_exits_0` | 8 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 115 | `valid_case_stdout_contains_buses_count` | 9 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 128 | `missing_buses_json_exits_1` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 141 | `missing_buses_json_stdout_contains_error` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 154 | `missing_buses_json_stdout_mentions_file` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 169 | `nonexistent_path_exits_2` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 178 | `nonexistent_path_stderr_mentions_path` | 8 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/cli_validate.rs` | 190 | `valid_case_piped_stdout_has_no_ansi_escapes` | 15 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/init.rs` | 21 | `test_init_list_shows_1dtoy` | 7 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/init.rs` | 30 | `test_init_1dtoy_creates_valid_case` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/init.rs` | 43 | `test_init_unknown_template_fails` | 10 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/init.rs` | 55 | `test_init_no_args_fails` | 3 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/init.rs` | 60 | `test_init_existing_non_empty_dir_fails` | 11 | `` | `integration` | `generic` |  |
| `cobre-cli` | `crates/cobre-cli/tests/init.rs` | 73 | `test_init_force_overwrites` | 15 | `` | `integration` | `generic` |  |
| `cobre-python` | `crates/cobre-python/src/run.rs` | 958 | `prepare_stochastic_succeeds_for_d01_case_via_python_path` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 450 | `test_available_backends_contains_local` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 460 | `test_available_backends_no_feature_exact` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 473 | `test_mpi_launch_detected_false_by_default` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 503 | `test_mpi_launch_detected_pmi_rank` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 518 | `test_mpi_launch_detected_ompi` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 534 | `test_create_communicator_no_feature_auto` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 547 | `test_create_communicator_no_feature_invalid` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 567 | `test_create_communicator_no_feature_unavailable` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 593 | `test_backend_kind_derives` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 623 | `test_comm_backend_send_sync` | 4 | `comm_backend` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 630 | `test_comm_backend_local_rank_size` | 5 | `comm_backend` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 638 | `test_comm_backend_local_barrier` | 4 | `comm_backend` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 645 | `test_comm_backend_local_allreduce` | 7 | `comm_backend` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 655 | `test_comm_backend_local_allgatherv` | 7 | `comm_backend` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 665 | `test_comm_backend_local_broadcast` | 6 | `comm_backend` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/factory.rs` | 676 | `test_comm_backend_local_shared_memory` | 20 | `comm_backend` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 593 | `test_ferrompi_backend_send_sync` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 599 | `sanitize_mpich_multiline` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 608 | `sanitize_openmpi_clean` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 616 | `sanitize_intel_mpi` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 622 | `sanitize_empty` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 628 | `test_ferrompi_local_comm_send_sync` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 635 | `test_ferrompi_local_comm_is_object_safe` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 646 | `test_map_reduce_op_exhaustive` | 14 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 662 | `test_to_i32_vec_valid` | 4 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 668 | `test_to_i32_vec_overflow` | 15 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 685 | `test_to_i32_vec_empty` | 4 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 691 | `test_map_ferrompi_error_invalid_buffer` | 13 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 706 | `test_map_ferrompi_error_already_initialized` | 7 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 715 | `test_map_ferrompi_error_internal` | 17 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 734 | `test_map_ferrompi_error_not_supported` | 14 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 750 | `test_map_ferrompi_error_mpi_comm_class` | 12 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 764 | `test_map_ferrompi_error_mpi_root_class` | 12 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 778 | `test_map_ferrompi_error_mpi_buffer_class` | 18 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 798 | `test_map_ferrompi_error_mpi_count_class` | 18 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/ferrompi.rs` | 818 | `test_map_ferrompi_error_mpi_other_class` | 19 | `mpi_helpers` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 304 | `test_local_backend_is_zst` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 309 | `test_local_allgatherv_identity` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 318 | `test_local_allgatherv_with_offset` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 327 | `test_local_allgatherv_invalid_counts_len` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 347 | `test_local_allgatherv_invalid_displs_len` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 367 | `test_local_allgatherv_send_count_mismatch` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 386 | `test_local_allgatherv_recv_too_small` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 405 | `test_local_allreduce_identity_sum` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 414 | `test_local_allreduce_identity_min` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 423 | `test_local_allreduce_identity_max` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 432 | `test_local_allreduce_buffer_mismatch` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 451 | `test_local_allreduce_empty` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 470 | `test_local_broadcast_root0_noop` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 479 | `test_local_broadcast_invalid_root` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 490 | `test_local_barrier_noop` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 496 | `test_local_rank` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 502 | `test_local_size` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 508 | `test_local_send_sync` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 514 | `test_local_communicator_rank` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 520 | `test_local_communicator_size` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 526 | `test_local_communicator_barrier_noop` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 532 | `test_local_communicator_as_dyn` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 541 | `test_heap_region_create` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 548 | `test_heap_region_write_read` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 558 | `test_heap_region_fence_noop` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 565 | `test_heap_region_zero_count` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 572 | `test_local_create_shared_region` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 580 | `test_local_split_local` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 588 | `test_local_is_leader` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 594 | `test_heap_region_send_sync` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 600 | `test_heap_region_new_crate_visible` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/local.rs` | 608 | `test_heap_region_lifecycle` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 117 | `test_num_hosts_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 123 | `test_num_hosts_single` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 129 | `test_num_hosts_multiple` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 135 | `test_is_homogeneous_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 141 | `test_is_homogeneous_single_host` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 147 | `test_is_homogeneous_equal_rank_counts` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 153 | `test_is_homogeneous_unequal_rank_counts` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 159 | `test_leader_hostname_empty` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 165 | `test_leader_hostname_returns_first` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 171 | `test_debug_clone_derive` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/topology.rs` | 180 | `test_slurm_job_info_fields` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/traits.rs` | 747 | `test_commdata_blanket_impl` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/traits.rs` | 773 | `test_communicator_generic_and_send_sync_compile` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/traits.rs` | 833 | `test_local_communicator_object_safe_and_generic_compile` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/traits.rs` | 859 | `test_local_communicator_requires_send_sync` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/traits.rs` | 896 | `test_shared_region_trait_bounds` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/traits.rs` | 929 | `test_shared_memory_provider_gat` | 64 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/traits.rs` | 995 | `test_shared_memory_provider_requires_send_sync` | 59 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/types.rs` | 174 | `test_reduce_op_debug_format` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/types.rs` | 181 | `test_reduce_op_copy_eq` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/types.rs` | 193 | `test_comm_error_display` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/types.rs` | 234 | `test_comm_error_debug` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/types.rs` | 248 | `test_backend_error_display` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/types.rs` | 283 | `test_comm_error_std_error` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/src/types.rs` | 290 | `test_backend_error_std_error` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 25 | `test_factory_no_feature_auto` | 7 | `no_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 35 | `test_factory_no_feature_explicit_local` | 9 | `no_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 47 | `test_factory_no_feature_explicit_auto` | 7 | `no_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 57 | `test_factory_no_feature_mpi_unavailable` | 20 | `no_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 80 | `test_factory_no_feature_tcp_unavailable` | 8 | `no_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 91 | `test_factory_no_feature_shm_unavailable` | 8 | `no_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 102 | `test_factory_no_feature_invalid_name` | 17 | `no_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 130 | `test_factory_any_feature_local` | 9 | `any_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 142 | `test_factory_any_feature_tcp_unavailable` | 10 | `any_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 155 | `test_factory_any_feature_shm_unavailable` | 10 | `any_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 168 | `test_factory_any_feature_invalid_name` | 19 | `any_feature_factory` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 197 | `test_available_backends_contains_local` | 4 | `available_backends_tests` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 204 | `test_available_backends_mpi_feature` | 4 | `available_backends_tests` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 217 | `test_ferrompi_backend_send_sync` | 4 | `compile_time_checks` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 225 | `test_ferrompi_backend_communicator` | 4 | `compile_time_checks` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 233 | `test_ferrompi_backend_shared_memory_provider` | 4 | `compile_time_checks` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 248 | `test_comm_error_std_error_send_sync` | 4 | `error_type_checks` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/factory_tests.rs` | 255 | `test_backend_error_std_error_send_sync` | 4 | `error_type_checks` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 36 | `test_local_allgatherv_identity_size1` | 10 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 49 | `test_local_allgatherv_with_displacement` | 10 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 64 | `test_local_allreduce_identity_sum` | 10 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 77 | `test_local_allreduce_identity_min` | 10 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 90 | `test_local_allreduce_identity_max` | 10 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 103 | `test_local_allreduce_single_element` | 10 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 118 | `test_local_broadcast_root0_noop` | 9 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 132 | `test_local_barrier_repeated` | 7 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 144 | `test_local_rank_size` | 6 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 156 | `test_local_collective_sequence` | 23 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 184 | `test_local_shared_region_lifecycle` | 20 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 207 | `test_local_shared_region_is_leader` | 4 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 214 | `test_local_split_local_rank_size` | 8 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 227 | `test_local_allreduce_buffer_mismatch` | 21 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 251 | `test_local_allreduce_empty_buffer` | 21 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 275 | `test_local_allgatherv_recv_too_small` | 21 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 299 | `test_local_allgatherv_counts_mismatch` | 21 | `` | `integration` | `generic` |  |
| `cobre-comm` | `crates/cobre-comm/tests/local_conformance.rs` | 323 | `test_local_broadcast_invalid_root` | 11 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 708 | `stochastic_context_is_send_sync` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 715 | `build_succeeds_with_valid_system` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 762 | `par_lp_has_expected_dimensions` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 808 | `opening_tree_has_expected_dimensions` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 854 | `tree_view_returns_valid_view` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 897 | `build_fails_on_invalid_par` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 934 | `build_succeeds_on_non_pd_correlation` | 70 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1009 | `build_succeeds_with_hydros_and_empty_correlation` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1051 | `pre_study_stages_excluded_from_opening_tree` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1108 | `context_with_load_buses_has_expanded_dim` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1164 | `context_without_load_has_original_dim` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1214 | `context_load_bus_deterministic_excluded` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1266 | `opening_tree_noise_length_matches_expanded_dim` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1318 | `normal_lp_accessible_from_context` | 65 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1389 | `build_with_none_matches_original` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1447 | `build_with_user_supplied_tree_uses_provided_tree` | 90 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1540 | `test_entity_order_accessor` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1595 | `test_entity_order_with_user_tree` | 64 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1662 | `test_forward_seed_from_config` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/context.rs` | 1711 | `test_forward_seed_none_when_absent` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 598 | `build_single_default_profile` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 606 | `build_single_non_default_profile_used_as_default` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 614 | `build_fails_with_no_profiles` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 628 | `build_fails_with_multiple_profiles_and_no_default` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 655 | `build_with_schedule_mapping` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 691 | `apply_correlation_with_identity_factor_leaves_noise_unchanged` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 704 | `apply_correlation_with_known_factor` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 730 | `apply_correlation_leaves_unmatched_entities_unchanged` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 748 | `apply_correlation_with_reordered_entities` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 794 | `apply_correlation_uses_correct_profile_for_stage` | 58 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 876 | `test_build_rejects_mixed_entity_types` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 891 | `test_build_accepts_same_type_entities` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 901 | `test_build_accepts_single_entity_group` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 911 | `test_build_mixed_type_error_includes_group_name` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 937 | `test_apply_correlation_for_class_inflow_only` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 968 | `test_apply_correlation_for_class_skips_other_types` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 991 | `test_apply_correlation_for_class_no_matching_groups` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/resolve.rs` | 1005 | `test_group_factor_stores_entity_type` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 338 | `spectral_of_1x1_identity` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 345 | `spectral_of_2x2_identity` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 354 | `spectral_of_2x2_correlated_matrix` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 362 | `spectral_of_3x3_known_matrix` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 374 | `spectral_of_4x4_identity` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 384 | `spectral_handles_non_pd_matrix` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 393 | `spectral_fails_on_non_square_matrix` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 403 | `spectral_fails_on_non_symmetric_matrix` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 414 | `transform_with_identity_factor_equals_input` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 424 | `transform_with_known_2x2_factor` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 459 | `spectral_symmetric_tolerance_boundary` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 471 | `spectral_of_rank_deficient_matrix` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 489 | `spectral_of_non_pd_clips_negative_eigenvalue` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 517 | `spectral_of_20x20_identity` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/correlation/spectral.rs` | 538 | `spectral_matches_cholesky_for_pd_matrix` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 127 | `test_invalid_par_parameters_implements_std_error` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 141 | `test_spectral_decomposition_failed_implements_std_error` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 153 | `test_invalid_correlation_implements_std_error` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 165 | `test_insufficient_data_implements_std_error` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 176 | `test_seed_derivation_error_implements_std_error` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 186 | `test_all_variants_debug` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 232 | `test_unsupported_noise_method_display` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 245 | `test_dimension_exceeds_capacity_display` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 258 | `test_unsupported_sampling_scheme_display` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/error.rs` | 269 | `test_missing_scenario_source_display` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/lib.rs` | 55 | `stochastic_error_is_send_sync_static` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/lib.rs` | 61 | `all_public_modules_accessible` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 164 | `quantile_0_5_is_zero` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 171 | `quantile_symmetry` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 186 | `quantile_known_values` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 198 | `quantile_monotonicity` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 215 | `quantile_panics_at_0` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 221 | `quantile_panics_at_1` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 227 | `quantile_extreme_tail_clamp` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/quantile.rs` | 236 | `quantile_extreme_tail_symmetry` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/rng.rs` | 51 | `rng_from_seed_is_deterministic` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/rng.rs` | 64 | `rng_from_seed_differs_for_different_seeds` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/rng.rs` | 73 | `rng_from_seed_zero_is_valid` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/rng.rs` | 81 | `rng_from_seed_max_u64_is_valid` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 95 | `forward_seed_is_deterministic` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 103 | `forward_seed_varies_with_stage` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 111 | `forward_seed_varies_with_scenario` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 119 | `forward_seed_varies_with_iteration` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 127 | `forward_seed_varies_with_base_seed` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 139 | `opening_seed_is_deterministic` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 144 | `opening_seed_varies_with_stage` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 149 | `opening_seed_varies_with_opening_index` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 154 | `opening_seed_varies_with_base_seed` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 167 | `forward_and_opening_seeds_differ_for_same_partial_inputs` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 188 | `forward_seed_golden_value` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 199 | `stage_seed_is_deterministic` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 204 | `stage_seed_varies_with_stage` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 209 | `stage_seed_varies_with_base_seed` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 222 | `stage_seed_differs_from_opening_seed` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 231 | `stage_seed_differs_from_forward_seed` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 247 | `stage_seed_golden_value` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 259 | `test_derive_forward_seed_grouped_deterministic` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/noise/seed.rs` | 270 | `test_derive_forward_seed_grouped_differs_from_forward` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 405 | `build_empty_returns_zero_mean_std_one_factors` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 438 | `build_with_models_populates_mean_std` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 475 | `build_with_factors_populates_block_factors` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 511 | `build_with_missing_entity_stage_defaults_to_zero` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 558 | `build_with_missing_factor_defaults_to_one` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 586 | `accessor_consistency_across_stages` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 646 | `acceptance_criterion_mean_stage1_entity0` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 676 | `declaration_order_invariance` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 706 | `mean_out_of_bounds_panics` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 716 | `mean_entity_out_of_bounds_panics` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 726 | `std_out_of_bounds_panics` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 736 | `block_factor_out_of_bounds_panics` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/normal/precompute.rs` | 748 | `default_is_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/aggregate.rs` | 356 | `test_quarterly_aggregation_single_entity` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/aggregate.rs` | 397 | `test_identity_case_monthly_obs_monthly_seasons` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/aggregate.rs` | 431 | `test_multi_entity_two_entities_four_quarters` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/aggregate.rs` | 473 | `test_multi_year_two_years_four_quarters` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/aggregate.rs` | 504 | `test_season_map_fallback_for_out_of_range_date` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/aggregate.rs` | 531 | `test_unresolvable_date_returns_error` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/aggregate.rs` | 572 | `test_leap_year_february_weight` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 188 | `order_zero_returns_empty_vec` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 196 | `order_one_single_season_uniform_std` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 207 | `order_one_two_seasons_different_std` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 219 | `order_two_single_season_recursive_composition` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 233 | `order_two_two_seasons_recursive_composition` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 250 | `negative_contribution_detection` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 265 | `pimental_like_explosive_scenario` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 302 | `zero_std_for_lagged_season_produces_zero_contribution` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 318 | `check_empty_contributions_returns_false` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 323 | `check_all_positive_returns_false` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 328 | `check_one_negative_returns_true` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 333 | `check_zero_is_not_negative` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 342 | `find_max_order_empty_returns_zero` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 347 | `find_max_order_all_positive_returns_full_length` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 352 | `find_max_order_negative_at_index_two` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 357 | `find_max_order_negative_at_index_zero` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 366 | `order_three_single_season_full_recursion` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 389 | `phi1_negative_returns_true` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 394 | `phi1_positive_returns_false` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 399 | `phi1_zero_returns_false` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 404 | `phi1_empty_returns_false` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/contribution.rs` | 409 | `phi1_near_zero_negative_returns_true` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 592 | `ar0_produces_mean_plus_noise` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 602 | `ar1_acceptance_criterion` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 612 | `ar2_known_values` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 624 | `zero_sigma_returns_deterministic_value` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 634 | `negative_noise_can_produce_negative_inflow` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 646 | `order_respected_from_psi_slice_longer_than_order` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 686 | `batch_matches_single_hydro_at_stage_1` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 740 | `batch_matches_single_hydro_at_stage_0_ar2` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 784 | `declaration_order_invariance_batch` | 62 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 848 | `solve_noise_ar1_for_zero_target` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 859 | `solve_noise_roundtrip_zero_target` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 875 | `solve_noise_roundtrip_ar2` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 890 | `solve_noise_roundtrip_nonzero_target` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 907 | `solve_noise_ar0_case` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 920 | `solve_noise_zero_sigma_returns_neg_infinity` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 929 | `test_solve_par_noise_sigma_zero_matching_target` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 940 | `test_solve_par_noise_sigma_zero_non_matching_target` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 952 | `test_solve_par_noise_sigma_zero_near_matching_target` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 963 | `test_solve_par_noise_batch_sigma_zero_matching_target` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1012 | `solve_noise_positive_deterministic_gives_negative_eta_for_zero` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1024 | `solve_noise_negative_deterministic_gives_positive_eta_for_zero` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1040 | `batch_solve_matches_single_hydro_at_stage_0` | 49 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1091 | `batch_solve_roundtrip_makes_all_inflows_hit_target` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1119 | `batch_solve_roundtrip_nonzero_targets` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1148 | `test_evaluate_par_matches_inflow` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1170 | `test_evaluate_par_batch_matches_inflows` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/evaluate.rs` | 1196 | `test_solve_par_noise_batch_matches_noises` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 1850 | `estimate_seasonal_stats_two_hydros_twelve_seasons` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 1891 | `estimate_seasonal_stats_known_values` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 1937 | `estimate_seasonal_stats_bessel_correction` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 1970 | `estimate_seasonal_stats_insufficient_data_one_obs` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 1991 | `estimate_seasonal_stats_unmapped_date` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2013 | `estimate_seasonal_stats_ignores_unknown_hydros` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2051 | `estimate_seasonal_stats_empty_history` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2064 | `estimate_seasonal_stats_thirty_years_single_season` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2141 | `estimate_correlation_identical_series` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2185 | `estimate_correlation_single_hydro` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2219 | `estimate_correlation_empty_hydros` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2239 | `estimate_correlation_canonical_order` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2273 | `estimate_correlation_symmetric` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2308 | `estimate_correlation_unit_diagonal` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2349 | `estimate_correlation_independent_series` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2410 | `estimate_correlation_multi_season_produces_per_season_profiles` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2453 | `estimate_correlation_multi_season_schedule_maps_stages_to_seasons` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2492 | `estimate_correlation_multi_season_per_season_values_differ` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2537 | `select_order_aic_known_values` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2564 | `select_order_aic_white_noise_preferred` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2572 | `select_order_aic_ar1_selected` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2581 | `select_order_aic_empty_sigma2` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2588 | `select_order_aic_non_positive_sigma2_excluded` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2597 | `select_order_aic_tie_prefers_lower_order` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2617 | `select_order_aic_monotone_variance_selects_max` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2634 | `pacf_empty_parcor_selects_zero` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2641 | `pacf_single_significant_lag` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2650 | `pacf_no_significant_lag` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2658 | `pacf_selects_max_significant_lag` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2668 | `pacf_negative_parcor_uses_absolute_value` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2677 | `pacf_zero_observations_selects_zero` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2684 | `pacf_large_sample_low_threshold` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2713 | `periodic_autocorrelation_single_season_basic` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2736 | `periodic_autocorrelation_two_season` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2760 | `periodic_autocorrelation_cross_year_boundary` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2790 | `periodic_autocorrelation_zero_std_returns_zero` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2806 | `periodic_autocorrelation_insufficient_data` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2821 | `periodic_autocorrelation_clamped_to_range` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2833 | `periodic_autocorrelation_population_divisor` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2864 | `periodic_autocorrelation_lag_zero` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2877 | `build_periodic_yw_matrix_order_zero` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2887 | `build_periodic_yw_matrix_single_season_toeplitz` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2910 | `build_periodic_yw_matrix_diagonal_is_one` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2931 | `build_periodic_yw_matrix_symmetry` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2957 | `build_periodic_yw_matrix_rhs_length` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 2975 | `build_periodic_yw_matrix_two_season_not_toeplitz` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3002 | `build_periodic_yw_matrix_forward_prediction_two_season_ar2` | 109 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3117 | `solve_linear_system_1x1` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3127 | `solve_linear_system_2x2` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3139 | `solve_linear_system_3x3` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3153 | `solve_linear_system_singular` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3161 | `solve_linear_system_requires_pivoting` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3172 | `solve_linear_system_diagonal` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3185 | `solve_linear_system_6x6` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3205 | `periodic_autocorrelation_single_season_yw_solve_roundtrip` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3235 | `periodic_autocorrelation_two_obs_per_season` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3251 | `periodic_autocorrelation_large_lag_wraps` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3265 | `periodic_autocorrelation_population_divisor_verification` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3305 | `periodic_yw_matrix_solve_residual_check` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3340 | `periodic_yw_matrix_rhs_matches_extended_matrix` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3382 | `periodic_pacf_empty_for_zero_order` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3392 | `periodic_pacf_single_season_matches_ar1` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3414 | `periodic_pacf_two_season_differs_from_ld` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3437 | `periodic_pacf_length_matches_max_order` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3448 | `periodic_pacf_values_bounded` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3486 | `estimate_periodic_ar_order_zero` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3499 | `estimate_periodic_ar_order_one_known_rho` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3526 | `estimate_periodic_ar_two_season` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3544 | `estimate_periodic_ar_sigma2_per_order_length` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3566 | `estimate_periodic_ar_residual_ratio_bounded` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3588 | `estimate_periodic_ar_sigma2_finite` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3661 | `periodic_pacf_two_season_par2_analytical_verification` | 65 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3732 | `estimate_correlation_min_sample_fallback` | 59 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/fitting.rs` | 3793 | `estimate_correlation_single_season_backward_compat` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 640 | `ar_order_zero_deterministic_base_equals_mean` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 663 | `ar_order_zero_std_zero_gives_zero_sigma` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 673 | `ar_order_1_acceptance_criterion` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 706 | `two_hydros_three_stages_varying_orders` | 68 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 776 | `declaration_order_invariance` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 801 | `psi_slice_padded_for_shorter_order` | 44 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 847 | `missing_model_fills_zero_defaults` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 861 | `deterministic_base_out_of_bounds_panics` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 871 | `sigma_out_of_bounds_panics` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 880 | `acceptance_criterion_ar_order_zero_std_zero` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 927 | `pre_study_lag_resolves_via_season_fallback` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 965 | `pre_study_lag_ar6_all_lags_resolve` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1002 | `pre_study_lag_partial_resolution` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1032 | `pre_study_lag_with_explicit_prestudy_model` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1063 | `season_fallback_deep_negative_wraps_correctly` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1086 | `season_fallback_with_nonzero_offset` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1123 | `integration_ar_conditioning_at_stage_zero` | 94 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1219 | `integration_multiyear_ar_conditioning` | 101 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1364 | `march_start_ar1_resolves_to_february` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1405 | `march_start_ar6_all_lags_use_correct_seasons` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/precompute.rs` | 1442 | `march_start_parity_with_january_start_at_later_stage` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 218 | `empty_input_returns_empty_report` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 224 | `ar_order_zero_produces_no_warnings` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 231 | `ar_order_zero_with_zero_std_is_valid` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 238 | `zero_std_with_nonzero_ar_order_returns_error` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 258 | `valid_ar1_model_produces_no_warnings` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 265 | `low_residual_variance_ratio_triggers_warning` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 291 | `residual_variance_ratio_at_boundary_no_warning` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 298 | `first_fatal_error_stops_iteration` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 306 | `multiple_warnings_accumulated` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/par/validation.rs` | 314 | `mixed_models_accumulate_only_applicable_warnings` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 192 | `provenance_with_user_tree` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 235 | `provenance_without_user_tree` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 272 | `provenance_no_entities` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 303 | `provenance_correlation_from_system` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 340 | `provenance_correlation_empty` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 370 | `provenance_inflow_with_hydros` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 407 | `provenance_inflow_without_hydros` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/provenance.rs` | 437 | `test_provenance_per_class_schemes_populated` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 377 | `test_in_sample_fill_copies_correct_segment` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 416 | `test_in_sample_fill_deterministic` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 451 | `test_out_of_sample_fill_deterministic` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 484 | `test_out_of_sample_fill_stage_idx_out_of_bounds` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 526 | `test_historical_fill_deterministic` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 553 | `test_historical_fill_different_scenarios_may_differ` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 584 | `test_historical_window_stable_across_stages` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 642 | `test_external_fill_copies_eta_slice` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 678 | `test_external_fill_deterministic` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 705 | `test_external_scenario_stable_across_stages` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 778 | `test_historical_apply_initial_state_copies_lags` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 819 | `test_historical_apply_initial_state_consistent_with_fill` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 861 | `test_in_sample_apply_initial_state_noop` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 883 | `test_out_of_sample_apply_initial_state_noop` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 905 | `test_external_apply_initial_state_noop` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 926 | `test_debug_all_variants` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 972 | `test_out_of_sample_same_group_produces_identical_noise` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/class_sampler.rs` | 1012 | `test_out_of_sample_different_group_produces_different_noise` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1033 | `test_inflow_ar0_standardization` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1095 | `test_inflow_ar1_uses_external_lags` | 76 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1185 | `test_inflow_ar1_weekly_frozen_lags` | 131 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1337 | `test_inflow_ar1_spillover_accumulation` | 90 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1434 | `test_load_standardization` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1464 | `test_ncs_standardization` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1494 | `test_std_zero_returns_zero` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1519 | `test_new_allocates_correct_sizes` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1530 | `test_eta_roundtrip` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1538 | `test_entity_class_metadata` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1547 | `test_send_sync` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1553 | `test_zero_initialized` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1565 | `test_eta_roundtrip_multiple_cells` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1582 | `test_clone_is_independent` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1640 | `test_valid_library_passes` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1663 | `test_missing_entity_fails_v3_2` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1701 | `test_nonuniform_divisible_counts_accepted_v3_4` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1728 | `test_nan_eta_fails_v3_7` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1893 | `test_round_trip_weekly_monthly_ar1` | 81 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 1979 | `test_scenario_count_warning_returns_ok` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 2013 | `test_v34_accepts_nonuniform_scenario_counts` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 2040 | `test_v34_still_rejects_indivisible_rows` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 2071 | `test_raw_scenarios_per_stage_uniform` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 2082 | `test_raw_scenarios_per_stage_nonuniform` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 2094 | `test_pad_library_replicates_eta` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/external.rs` | 2123 | `test_pad_library_noop_when_uniform` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 683 | `test_new_allocates_correct_sizes` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 715 | `test_eta_roundtrip` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 741 | `test_lag_roundtrip` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 768 | `test_window_years` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 777 | `test_send_sync` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 864 | `test_ar0_standardization` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 937 | `test_ar1_standardization_uses_raw_lags` | 76 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1025 | `test_multi_hydro_multi_window` | 88 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1131 | `test_pre_study_lags_populated` | 71 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1212 | `test_sigma_zero_returns_zero_eta` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1277 | `test_valid_library_passes` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1301 | `test_neg_infinity_eta_fails_v2_3` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1339 | `test_missing_season_id_fails_v2_1` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1367 | `test_hydro_count_mismatch_fails_v2_9` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1389 | `test_pool_warning_path_returns_ok` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1435 | `test_standardize_monthly_season_map_identical` | 64 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1577 | `test_standardize_quarterly_season_map_correct` | 67 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1681 | `test_standardize_none_season_map_backward_compat` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/historical.rs` | 1742 | `test_quarterly_standardize_historical_windows` | 98 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 57 | `determinism_same_inputs_same_output` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 69 | `different_scenarios_different_indices` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 88 | `all_indices_in_bounds` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 102 | `returned_slice_matches_tree_opening` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 111 | `different_iterations_different_indices` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 125 | `stage_domain_id_affects_seed` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 142 | `single_opening_always_index_zero` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 153 | `slice_length_equals_dim` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/insample.rs` | 163 | `resume_invariant_noise_depends_only_on_iteration_seed` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 875 | `test_build_all_in_sample` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 887 | `test_build_out_of_sample_missing_seed` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 904 | `test_build_out_of_sample_with_seed` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 916 | `test_build_historical_with_library` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 951 | `test_build_historical_missing_library` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 987 | `test_build_external_with_library` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1026 | `test_build_historical_load_unsupported` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1065 | `test_forward_noise_as_slice_newtype` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1072 | `test_forward_noise_as_slice` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1083 | `test_in_sample_sample_returns_noise` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1113 | `test_in_sample_sample_is_deterministic` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1160 | `test_composite_in_sample_fills_correct_segments` | 66 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1230 | `test_composite_out_of_sample_applies_per_class_correlation` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1267 | `test_composite_sample_deterministic` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/mod.rs` | 1319 | `test_sample_request_propagates_noise_group_id` | 67 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 248 | `test_saa_determinism` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 267 | `test_saa_different_seeds_differ` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 295 | `test_lhs_produces_finite_noise` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 315 | `test_sobol_produces_finite_noise` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 335 | `test_sobol_dim_exceeds_capacity` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 367 | `test_halton_produces_finite_noise` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 387 | `test_selective_falls_back_to_saa` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 407 | `test_selective_matches_saa` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 441 | `test_fill_uncorrelated_saa_deterministic` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 457 | `test_fill_uncorrelated_sobol_dim_exceeds_capacity` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/out_of_sample.rs` | 486 | `test_fill_uncorrelated_produces_finite_values` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 314 | `test_auto_discovery_all_valid` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 346 | `test_user_pool_list_filters` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 373 | `test_user_pool_range_expands` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 403 | `test_no_valid_windows_returns_error` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 435 | `test_incomplete_hydro_excludes_window` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 469 | `test_to_years_list` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 479 | `test_to_years_range` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 613 | `test_monthly_season_map_identical_to_month0` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 649 | `test_quarterly_season_map_window_discovery` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 683 | `test_none_season_map_backward_compat` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/sampling/window.rs` | 724 | `test_month0_fallback_matches_monthly_season_map` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 411 | `determinism_same_inputs_produce_identical_trees` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 451 | `opening_0_0_has_correct_length_and_finite_values` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 474 | `seed_sensitivity_different_seeds_produce_different_trees` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 499 | `variable_branching_factors_correct_dimensions` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 527 | `correct_dimensions_uniform_branching` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 553 | `identity_correlation_noise_has_normal_statistics` | 43 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 600 | `all_generated_values_are_finite` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 632 | `correlated_noise_matches_target_correlation` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 690 | `different_openings_and_stages_produce_different_noise` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 722 | `saa_bitwise_compatible_with_pre_refactor_golden_values` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 777 | `selective_returns_error` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 812 | `test_lhs_stage_produces_tree` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 853 | `test_per_stage_method_mixing` | 71 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 935 | `test_sobol_stage_produces_tree` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 978 | `test_sobol_dimension_exceeds_capacity` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1021 | `test_sobol_saa_mixing` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1067 | `test_halton_stage_produces_tree` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1110 | `test_halton_saa_mixing` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1161 | `test_per_class_tree_matches_full_vector_tree` | 116 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1284 | `test_per_class_tree_matches_full_vector_multi_class` | 111 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1411 | `test_per_class_tree_matches_full_vector_tree_lhs` | 106 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1525 | `test_per_class_tree_matches_full_vector_tree_halton` | 106 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1668 | `test_historical_residuals_none_library_returns_error` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1702 | `test_historical_residuals_copies_eta_from_library` | 59 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1766 | `test_historical_residuals_zeros_non_hydro_slots` | 50 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1821 | `test_historical_residuals_clamps_openings_when_windows_lt_branching` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1865 | `test_historical_residuals_deterministic_window_selection` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1927 | `test_external_clamping_reduces_openings` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1963 | `test_external_clamping_none_no_effect` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 1986 | `test_external_and_historical_clamping_combined` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 2031 | `test_external_clamping_counts_length_mismatch_panics` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 2064 | `test_opening_tree_noise_group_none_backward_compat` | 47 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 2115 | `test_opening_tree_same_group_copies_noise` | 45 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 2164 | `test_opening_tree_two_groups` | 56 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/generate.rs` | 2224 | `test_opening_tree_monthly_no_copy` | 53 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 180 | `fisher_yates_is_permutation` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 197 | `fisher_yates_different_states_differ` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 214 | `fisher_yates_edge_cases_do_not_panic` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 227 | `lhs_determinism` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 243 | `lhs_different_seeds_differ` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 255 | `lhs_correct_length` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 267 | `lhs_all_finite` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 294 | `lhs_marginal_stratification` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 330 | `lhs_mean_and_std_within_tolerance` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 354 | `lhs_zero_openings_does_not_panic` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 363 | `lhs_zero_dim_does_not_panic` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 382 | `lhs_point_determinism` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 409 | `lhs_point_different_seeds_differ` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 450 | `lhs_point_all_finite` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/lhs.rs` | 484 | `lhs_point_stratum_coverage` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 278 | `opening_stage0_opening0_returns_first_dim_elements` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 285 | `opening_stage1_returns_correct_slices` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 293 | `n_openings_matches_branching_factors` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 301 | `view_returns_identical_data_to_owned` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 311 | `len_and_size_bytes_uniform_branching` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 318 | `uniform_branching_3_stages_5_openings_2_entities` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 335 | `variable_branching_access` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 359 | `single_stage_single_opening` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 368 | `from_parts_panics_on_wrong_data_length` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 375 | `opening_panics_on_out_of_bounds_stage` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 382 | `opening_panics_on_out_of_bounds_opening_idx` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 389 | `view_accessors_match_owned_for_variable_branching` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 408 | `is_empty_false_for_non_empty_tree` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 414 | `size_bytes_is_8_times_len` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 420 | `view_is_empty_matches_owned` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 427 | `data_returns_full_backing_array` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 435 | `openings_per_stage_slice_matches_input` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/opening_tree.rs` | 442 | `view_data_matches_owned_data` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 361 | `test_sieve_first_10_primes` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 366 | `test_sieve_zero_returns_empty` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 371 | `test_sieve_one_returns_two` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 376 | `test_sieve_100_primes_count` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 383 | `test_radical_inverse_base2_known_values` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 405 | `test_radical_inverse_base3_known_values` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 427 | `test_radical_inverse_base5_n1` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 437 | `test_radical_inverse_range` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 454 | `test_halton_batch_determinism` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 466 | `test_halton_batch_different_seeds_differ` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 478 | `test_halton_batch_all_finite` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 493 | `test_halton_batch_values_in_range` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 509 | `test_halton_batch_correct_length` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 519 | `test_halton_batch_zero_openings` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 527 | `test_halton_batch_zero_dim` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 537 | `test_halton_point_determinism` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 556 | `test_halton_point_different_seeds_differ` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs` | 584 | `test_halton_point_all_finite` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 362 | `test_sobol_batch_determinism` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 374 | `test_sobol_batch_different_seeds_differ` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 386 | `test_sobol_batch_all_finite` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 398 | `test_sobol_batch_correct_length` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 408 | `test_sobol_point_determinism` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 427 | `test_sobol_point_different_seeds_differ` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 455 | `test_sobol_point_all_finite` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 480 | `test_build_direction_matrix_dim1` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 497 | `test_sobol_batch_values_in_range` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 513 | `test_max_sobol_dim_constant` | 3 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 525 | `test_unscrambled_dim1_first_8_points` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs` | 555 | `test_unscrambled_dim2_first_8_points` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/conformance.rs` | 195 | `pipeline_builds_with_correct_dimensions` | 21 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/conformance.rs` | 234 | `par_lp_coefficients_match_hand_computed` | 68 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/conformance.rs` | 310 | `opening_tree_structure_correct` | 36 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/conformance.rs` | 349 | `sample_forward_returns_valid_output` | 47 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/conformance.rs` | 400 | `opening_tree_marginal_statistics` | 71 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 272 | `insample_dispatch_returns_tree_slice_of_correct_dim` | 37 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 311 | `insample_copy_equivalence_matches_direct_call` | 37 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 350 | `out_of_sample_dispatch_returns_fresh_noise_of_correct_dim` | 40 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 392 | `out_of_sample_is_deterministic` | 52 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 446 | `out_of_sample_scenario_changes_noise` | 57 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 505 | `out_of_sample_noise_is_finite` | 41 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 548 | `out_of_sample_correlation_matches_target` | 60 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 610 | `out_of_sample_per_stage_method_mixing` | 52 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 664 | `factory_rejects_out_of_sample_without_seed` | 24 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 690 | `factory_rejects_historical` | 24 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 716 | `factory_rejects_external` | 21 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/forward_sampler.rs` | 739 | `out_of_sample_resume_invariance` | 55 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/halton_integration.rs` | 338 | `halton_2d_star_discrepancy` | 48 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/halton_integration.rs` | 394 | `halton_normal_statistics` | 43 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/halton_integration.rs` | 445 | `halton_correlation_applied` | 52 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/halton_integration.rs` | 507 | `halton_declaration_order_invariant` | 36 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/halton_integration.rs` | 552 | `halton_point_wise_consistency` | 42 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/lhs_integration.rs` | 329 | `lhs_marginal_uniformity` | 52 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/lhs_integration.rs` | 389 | `lhs_no_stratum_collision` | 44 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/lhs_integration.rs` | 441 | `lhs_normal_statistics` | 43 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/lhs_integration.rs` | 492 | `lhs_correlation_applied` | 52 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/lhs_integration.rs` | 554 | `lhs_declaration_order_invariant` | 36 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/lhs_integration.rs` | 600 | `lhs_point_wise_stratum_consistency` | 39 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/reproducibility.rs` | 203 | `deterministic_reproducibility` | 69 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/reproducibility.rs` | 276 | `declaration_order_invariance` | 34 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/reproducibility.rs` | 314 | `seed_sensitivity` | 24 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/reproducibility.rs` | 341 | `infrastructure_genericity_no_sddp_references` | 22 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/saa_golden_value.rs` | 119 | `saa_golden_value_regression` | 48 | `` | `regression` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/sobol_integration.rs` | 336 | `sobol_2d_star_discrepancy` | 48 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/sobol_integration.rs` | 392 | `sobol_normal_statistics` | 43 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/sobol_integration.rs` | 443 | `sobol_correlation_applied` | 52 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/sobol_integration.rs` | 505 | `sobol_declaration_order_invariant` | 36 | `` | `integration` | `generic` |  |
| `cobre-stochastic` | `crates/cobre-stochastic/tests/sobol_integration.rs` | 550 | `sobol_point_wise_consistency` | 42 | `` | `integration` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/bus.rs` | 50 | `test_bus_construction` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/bus.rs` | 79 | `test_deficit_segment_unbounded` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/bus.rs` | 90 | `test_bus_equality` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/bus.rs` | 114 | `test_bus_serde_roundtrip` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/energy_contract.rs` | 58 | `test_import_contract` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/energy_contract.rs` | 76 | `test_export_contract` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/energy_contract.rs` | 94 | `test_contract_type_equality` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 295 | `test_hydro_constant_productivity` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 310 | `test_hydro_fpha` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 316 | `test_hydro_optional_fields_none` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 335 | `test_hydro_optional_fields_some` | 60 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 397 | `test_tailrace_polynomial` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 412 | `test_tailrace_piecewise` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 440 | `test_hydraulic_losses_factor` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 450 | `test_filling_config` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 461 | `test_hydro_penalties_all_fields` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 495 | `test_diversion_channel` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 507 | `test_hydro_serde_roundtrip` | 62 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/hydro.rs` | 571 | `test_hydro_evaporation_reference_volumes` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/line.rs` | 44 | `test_line_construction` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/line.rs` | 71 | `test_line_lifecycle_always` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/line.rs` | 90 | `test_line_lifecycle_bounded` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/line.rs` | 109 | `test_line_equality` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/line.rs` | 136 | `test_line_serde_roundtrip` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/non_controllable.rs` | 44 | `test_non_controllable_construction` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/non_controllable.rs` | 65 | `test_non_controllable_curtailment_cost` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/pumping_station.rs` | 46 | `test_pumping_station_construction` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/thermal.rs` | 55 | `test_thermal_construction` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/thermal.rs` | 80 | `test_thermal_with_gnl` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/thermal.rs` | 99 | `test_thermal_without_gnl` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entities/thermal.rs` | 117 | `test_thermal_serde_roundtrip` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entity_id.rs` | 62 | `test_equality` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entity_id.rs` | 71 | `test_copy` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entity_id.rs` | 79 | `test_hash_consistency` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entity_id.rs` | 92 | `test_display` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entity_id.rs` | 99 | `test_from_i32` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entity_id.rs` | 105 | `test_into_i32` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/entity_id.rs` | 112 | `test_entity_id_serde_roundtrip` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/error.rs` | 142 | `test_display_invalid_reference` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/error.rs` | 157 | `test_display_duplicate_id` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/error.rs` | 168 | `test_display_cascade_cycle` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/error.rs` | 179 | `test_error_trait` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/generic_constraint.rs` | 302 | `test_variable_ref_variants` | 155 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/generic_constraint.rs` | 459 | `test_generic_constraint_construction` | 40 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/generic_constraint.rs` | 501 | `test_slack_config_disabled_has_no_penalty` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/generic_constraint.rs` | 511 | `test_constraint_sense_variants` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/generic_constraint.rs` | 518 | `test_linear_term_with_coefficient` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/generic_constraint.rs` | 532 | `test_variable_ref_block_none_vs_some` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/generic_constraint.rs` | 546 | `test_generic_constraint_serde_roundtrip` | 35 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 162 | `test_initial_conditions_construction` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 200 | `test_initial_conditions_default_is_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 209 | `test_hydro_storage_clone` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 221 | `test_hydro_past_inflows_clone` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 235 | `test_initial_conditions_serde_roundtrip` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 266 | `test_initial_conditions_serde_roundtrip_empty_past_inflows` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 287 | `test_recent_observation_construction_and_clone` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 303 | `test_initial_conditions_construction_with_recent_observations` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 327 | `test_initial_conditions_serde_roundtrip_with_recent_observations` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 362 | `test_initial_conditions_serde_default_recent_observations_absent` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 370 | `test_hydro_past_inflows_with_season_ids` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 386 | `test_hydro_past_inflows_serde_roundtrip_with_season_ids` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/initial_conditions.rs` | 400 | `test_hydro_past_inflows_serde_default_season_ids_absent` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 383 | `test_resolve_bus_excess_cost_global` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 390 | `test_resolve_bus_excess_cost_override` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 397 | `test_resolve_bus_deficit_segments_global` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 406 | `test_resolve_bus_deficit_segments_override` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 419 | `test_resolve_line_exchange_cost_global` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 426 | `test_resolve_line_exchange_cost_override` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 433 | `test_resolve_hydro_penalties_all_global` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 440 | `test_resolve_hydro_penalties_partial_override` | 25 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 467 | `test_resolve_hydro_penalties_all_override` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 508 | `test_resolve_hydro_penalties_default_overrides_equals_global` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 515 | `test_resolve_ncs_curtailment_cost_global` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 522 | `test_resolve_ncs_curtailment_cost_override` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 529 | `test_resolve_hydro_directional_override_pos_only` | 16 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 547 | `test_resolve_hydro_directional_evaporation_asymmetric` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/penalty.rs` | 562 | `test_resolve_hydro_no_overrides_directional_equals_symmetric` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1494 | `test_hydro_stage_penalties_copy` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1503 | `test_all_penalty_structs_are_copy` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1522 | `test_all_bound_structs_are_copy` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1556 | `test_resolved_penalties_construction` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1595 | `test_resolved_penalties_indexed_access` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1626 | `test_resolved_penalties_mutable_update` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1662 | `test_resolved_bounds_construction` | 54 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1718 | `test_resolved_bounds_mutable_update` | 52 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1772 | `test_hydro_stage_bounds_has_eleven_fields` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1793 | `test_resolved_penalties_serde_roundtrip` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1823 | `test_resolved_bounds_serde_roundtrip` | 42 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1871 | `test_generic_bounds_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1882 | `test_generic_bounds_sparse_active` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1902 | `test_generic_bounds_single_block_none` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1914 | `test_generic_bounds_multiple_blocks` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1934 | `test_generic_bounds_unknown_constraint_id_skipped` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1946 | `test_generic_bounds_no_rows` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1957 | `test_generic_bounds_two_stages_one_constraint` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1980 | `test_generic_bounds_serde_roundtrip` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 1998 | `test_load_factors_empty_returns_one` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2005 | `test_load_factors_new_default_is_one` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2018 | `test_load_factors_set_and_get` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2030 | `test_load_factors_out_of_bounds_returns_one` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2041 | `test_exchange_factors_empty_returns_one_one` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2048 | `test_exchange_factors_new_default_is_one_one` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2055 | `test_exchange_factors_set_and_get` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2063 | `test_exchange_factors_out_of_bounds_returns_default` | 4 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2071 | `test_ncs_bounds_empty_is_empty` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2078 | `test_ncs_bounds_new_uses_defaults` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2088 | `test_ncs_bounds_set_and_get` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2098 | `test_ncs_bounds_out_of_bounds_returns_zero` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2107 | `test_ncs_factors_empty_returns_one` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2114 | `test_ncs_factors_new_default_is_one` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2127 | `test_ncs_factors_set_and_get` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/resolved.rs` | 2136 | `test_ncs_factors_out_of_bounds_returns_one` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 738 | `test_inflow_model_construction` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 759 | `test_inflow_model_ar_order_method` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 784 | `test_correlation_model_construction` | 65 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 851 | `test_sampling_scheme_copy` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 871 | `test_scenario_source_serde_roundtrip` | 54 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 927 | `test_scenario_source_default` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 937 | `test_historical_years_list_construction` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 952 | `test_historical_years_range_construction` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 969 | `test_historical_years_list_serde_roundtrip` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 979 | `test_historical_years_range_serde_roundtrip` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 992 | `test_inflow_model_serde_roundtrip` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 1008 | `test_ncs_model_construction` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 1024 | `test_ncs_model_serde_roundtrip` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/scenario.rs` | 1037 | `test_correlation_model_identity_matrix_access` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1442 | `test_empty_system` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1456 | `test_canonical_ordering` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1469 | `test_lookup_by_id` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1483 | `test_lookup_missing_id` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1495 | `test_count_queries` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1517 | `test_slice_accessors` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1531 | `test_duplicate_id_error` | 17 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1550 | `test_multiple_duplicate_errors` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1583 | `test_send_sync` | 5 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1590 | `test_cascade_accessible` | 26 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1618 | `test_network_accessible` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1631 | `test_all_entity_lookups` | 36 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1669 | `test_default_builder` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1679 | `test_invalid_bus_reference_hydro` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1702 | `test_invalid_downstream_reference` | 30 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1734 | `test_invalid_pumping_station_hydro_refs` | 31 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1767 | `test_multiple_invalid_references_collected` | 55 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1824 | `test_valid_cross_references_pass` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1866 | `test_cascade_cycle_detected` | 34 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1902 | `test_cascade_self_loop_detected` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1925 | `test_valid_acyclic_cascade_passes` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1956 | `test_filling_without_entry_stage` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 1990 | `test_filling_negative_inflow` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2024 | `test_valid_filling_config_passes` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2048 | `test_cascade_cycle_and_invalid_filling_both_reported` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2088 | `test_system_serde_roundtrip` | 46 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2169 | `test_system_backward_compat` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2195 | `test_system_resolved_generic_bounds_accessor` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2217 | `test_system_with_stages` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2244 | `test_system_stage_lookup_by_id` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2258 | `test_system_with_initial_conditions` | 20 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2283 | `test_system_serde_roundtrip_with_stages` | 39 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2328 | `test_system_inflow_history_defaults_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2339 | `test_system_inflow_history_stores_rows` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2367 | `test_system_external_scenarios_defaults_empty` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/system.rs` | 2378 | `test_system_external_scenarios_stores_rows` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/temporal.rs` | 632 | `test_block_mode_copy` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/temporal.rs` | 645 | `test_stage_duration` | 29 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/temporal.rs` | 676 | `test_policy_graph_construction` | 38 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/temporal.rs` | 716 | `test_season_map_construction` | 41 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/temporal.rs` | 760 | `test_policy_graph_serde_roundtrip` | 37 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 227 | `test_empty_cascade` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 235 | `test_single_hydro_terminal` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 248 | `test_linear_chain` | 28 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 278 | `test_fork_merge` | 32 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 312 | `test_parallel_chains` | 23 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 337 | `test_all_terminal` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 361 | `test_deterministic_ordering` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 376 | `test_is_headwater` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 390 | `test_is_terminal` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 404 | `test_len` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/cascade.rs` | 416 | `test_topology_serde_roundtrip_cascade` | 12 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 362 | `test_empty_network` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 375 | `test_single_line` | 18 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 395 | `test_multiple_lines_same_bus` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 412 | `test_generators_per_bus` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 433 | `test_loads_per_bus` | 13 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 448 | `test_bus_no_connections` | 14 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 464 | `test_deterministic_ordering` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/topology/network.rs` | 491 | `test_topology_serde_roundtrip_network` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 533 | `all_fourteen_variants_construct` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 543 | `all_variants_clone` | 7 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 552 | `all_variants_debug_non_empty` | 6 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 560 | `forward_pass_complete_fields_accessible` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 586 | `convergence_update_rules_evaluated_field` | 33 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 621 | `stopping_rule_result_fields_accessible` | 11 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 634 | `stopping_rule_result_debug_non_empty` | 10 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 646 | `cut_selection_complete_fields_accessible` | 27 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 675 | `training_started_timestamp_field` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 692 | `simulation_progress_scenario_cost_field_accessible` | 24 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 718 | `simulation_progress_first_scenario_cost_carried` | 15 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 735 | `budget_enforcement_complete_fields_accessible` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/training_event.rs` | 758 | `template_bake_complete_fields_accessible` | 21 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/welford.rs` | 138 | `welford_known_dataset_mean_variance_std` | 22 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/welford.rs` | 163 | `welford_single_value_no_variance` | 19 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/welford.rs` | 185 | `welford_zero_updates` | 9 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/src/welford.rs` | 197 | `welford_count_tracks_updates` | 8 | `tests` | `unit` | `generic` |  |
| `cobre-core` | `crates/cobre-core/tests/integration.rs` | 167 | `test_declaration_order_invariance` | 49 | `` | `integration` | `generic` |  |
| `cobre-core` | `crates/cobre-core/tests/integration.rs` | 219 | `test_realistic_multi_entity_system` | 148 | `` | `integration` | `generic` |  |
| `cobre-core` | `crates/cobre-core/tests/integration.rs` | 370 | `test_invalid_cross_reference_rejected` | 28 | `` | `integration` | `generic` |  |
| `cobre-core` | `crates/cobre-core/tests/integration.rs` | 401 | `test_cascade_cycle_rejected` | 21 | `` | `integration` | `generic` |  |
| `cobre-core` | `crates/cobre-core/tests/integration.rs` | 425 | `test_large_order_invariance` | 31 | `` | `integration` | `generic` |  |
| `cobre-core` | `crates/cobre-core/tests/integration.rs` | 459 | `test_invalid_filling_config_rejected` | 31 | `` | `integration` | `generic` |  |
| `cobre-core` | `crates/cobre-core/tests/integration.rs` | 493 | `test_diversion_invalid_reference_rejected` | 32 | `` | `integration` | `generic` |  |

## 3. Deletion Candidates (by Epic)

### Epic 03 — `AlienOnly` warm-start removal (`alien-only`, `warm-start-config-flag`, `broadcast-warm-start-field`)

**Count:** 7

| Crate | File | Line | Function | Category | Guards |
| ----- | ---- | ---: | -------- | -------- | ------ |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1861 | `test_policy_config_default_boundary_is_none` | `unit` | `warm-start-config-flag` |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1871 | `test_boundary_policy_round_trip` | `unit` | `warm-start-config-flag` |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1921 | `warm_start_basis_mode_returns_migration_error` | `unit` | `warm-start-config-flag,canonical-config-flag` |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1934 | `warm_start_basis_mode_non_alien_also_rejected` | `unit` | `warm-start-config-flag,canonical-config-flag` |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1944 | `config_without_warm_start_basis_mode_parses_cleanly` | `unit` | `warm-start-config-flag,canonical-config-flag` |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 432 | `broadcast_config_propagates_training_enabled` | `unit` | `warm-start-config-flag` |
| `cobre-cli` | `crates/cobre-cli/src/commands/broadcast.rs` | 456 | `broadcast_config_roundtrips_via_postcard_after_warm_start_basis_mode_deletion` | `unit` | `warm-start-config-flag` |

### Epic 04 — Canonical-state strategy removal (`canonical-disabled`, `canonical-config-flag`, `broadcast-canonical-field`, `clear-solver-state-trait`, `solve-with-basis-trait`)

**Count:** 4

| Crate | File | Line | Function | Category | Guards |
| ----- | ---- | ---: | -------- | -------- | ------ |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1921 | `warm_start_basis_mode_returns_migration_error` | `unit` | `warm-start-config-flag,canonical-config-flag` |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1934 | `warm_start_basis_mode_non_alien_also_rejected` | `unit` | `warm-start-config-flag,canonical-config-flag` |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1944 | `config_without_warm_start_basis_mode_parses_cleanly` | `unit` | `warm-start-config-flag,canonical-config-flag` |
| `cobre-io` | `crates/cobre-io/src/config.rs` | 1954 | `canonical_state_key_is_obsolete_but_parses_cleanly` | `unit` | `canonical-config-flag` |

### Epic 05 — Non-baked template removal (`non-baked`, `stored-cut-row-offset`, `add-rows-trait`)

**Count:** 36

| Crate | File | Line | Function | Category | Guards |
| ----- | ---- | ---: | -------- | -------- | ------ |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 783 | `test_slot_lookup_growth_safe_in_release` | `unit` | `stored-cut-row-offset` |
| `cobre-sddp` | `crates/cobre-sddp/src/basis_reconstruct.rs` | 832 | `test_stored_cut_row_offset_skips_baked_rows` | `unit` | `stored-cut-row-offset` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 2309 | `test_penalty_slack_absorbs_negative_inflow` | `unit` | `add-rows-trait` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3482 | `fpha_solve_one_hydro_optimal` | `unit` | `add-rows-trait,fpha-slow` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3537 | `fpha_solve_hyperplane_constraints_hold` | `unit` | `add-rows-trait,fpha-slow` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3647 | `fpha_solve_storage_fixing_dual_differs_from_constant` | `unit` | `add-rows-trait,fpha-slow` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3756 | `fpha_solve_mixed_system_optimal` | `unit` | `add-rows-trait,fpha-slow` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5026 | `evap_lp_solvable_and_q_ev_nonnegative` | `unit` | `add-rows-trait` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5082 | `evap_violation_slacks_near_zero_feasible_constraint` | `unit` | `add-rows-trait` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5143 | `evap_storage_fixing_dual_differs_from_no_evaporation` | `unit` | `add-rows-trait` |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 5217 | `evap_bound_prevents_dump_valve` | `unit` | `add-rows-trait` |
| `cobre-sddp` | `crates/cobre-sddp/src/stage_solve.rs` | 469 | `run_stage_solve_warm_start_excess_basic_demotes` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1388 | `test_highs_load_model_updates_dimensions` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1419 | `test_highs_add_rows_updates_dimensions` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1477 | `test_highs_solve_basic_lp` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1513 | `test_highs_solve_with_cuts` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1549 | `test_highs_solve_after_rhs_patch` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1731 | `test_solve_warm_start_reproduces_cold_objective` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/highs.rs` | 1773 | `test_solve_warm_start_extends_missing_rows_as_basic` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/src/trait_def.rs` | 269 | `test_noop_solver_all_methods` | `unit` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 127 | `test_fixture_stage_template_data` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 150 | `test_fixture_row_batch_data` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 164 | `test_solver_highs_add_rows_tightens` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 195 | `test_solver_highs_add_rows_single_cut` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 231 | `test_solver_highs_set_row_bounds_state_change` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 270 | `test_solver_highs_set_col_bounds_basic` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 356 | `test_solver_highs_solve_dual_values` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 387 | `test_solver_highs_solve_dual_values_with_cuts` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 487 | `test_solver_highs_dual_normalization_sensitivity_check` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 519 | `test_solver_highs_dual_normalization_with_binding_cut` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 659 | `test_solver_highs_name_returns_identifier` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 680 | `test_solver_highs_lifecycle_repeated_patch_solve` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1303 | `solve_borrows_internal_buffers` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1332 | `solve_after_add_rows` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1383 | `basis_dimensions_after_solve` | `integration` | `add-rows-trait` |
| `cobre-solver` | `crates/cobre-solver/tests/conformance.rs` | 1412 | `basis_cut_extension` | `integration` | `add-rows-trait` |

## 4. Parameterization Candidates


_No tests were tagged `parameter-sweep` in this pass. The initial heuristic tagger did not identify clear parameter-sweep groups; this section should be populated by a follow-up manual pass._

## 5. Slow-Test Roster


### `fpha-slow` tests

**Count:** 164

| Crate | File | Line | Function | Feature-gated? |
| ----- | ---- | ---: | -------- | -------------- |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1677 | `valid_five_point_curve_construction_succeeds` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1688 | `interpolation_at_midpoint_segment_0_to_2000` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1697 | `interpolation_at_breakpoints_returns_exact_values` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1714 | `height_clamped_below_v_min` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1724 | `height_clamped_above_v_max` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1734 | `derivative_first_segment_correct` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1743 | `derivative_last_segment_and_at_v_max` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1767 | `derivative_at_interior_breakpoint_uses_right_segment` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1783 | `derivative_clamped_below_v_min_returns_first_segment_slope` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1794 | `insufficient_points_zero_rows` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1807 | `insufficient_points_one_row` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1820 | `non_monotonic_volume_duplicate` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1841 | `non_monotonic_volume_decreasing` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1855 | `non_monotonic_height_decreasing` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1876 | `equal_consecutive_heights_accepted` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1885 | `display_insufficient_points_contains_name_and_count` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1893 | `display_non_monotonic_volume_contains_name_and_index` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1905 | `display_non_monotonic_height_contains_name_and_index` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1917 | `fpha_fitting_error_implements_std_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1925 | `tailrace_polynomial_constant_one_coefficient` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1934 | `tailrace_polynomial_linear_two_coefficients` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1943 | `tailrace_polynomial_quadratic_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1952 | `tailrace_polynomial_quartic_five_coefficients` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1964 | `tailrace_polynomial_derivative_constant_is_zero` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1972 | `tailrace_polynomial_derivative_linear` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1980 | `tailrace_polynomial_derivative_quadratic_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 1989 | `tailrace_polynomial_derivative_quartic` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2020 | `tailrace_piecewise_midpoint_first_segment_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2027 | `tailrace_piecewise_at_breakpoints_exact` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2035 | `tailrace_piecewise_clamp_below_range` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2041 | `tailrace_piecewise_clamp_above_range` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2049 | `tailrace_piecewise_derivative_first_segment_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2056 | `tailrace_piecewise_derivative_second_segment` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2063 | `tailrace_piecewise_derivative_at_q_max_returns_last_segment_slope` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2070 | `tailrace_piecewise_derivative_clamp_above_returns_last_segment_slope` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2077 | `tailrace_piecewise_derivative_clamp_below_returns_first_segment_slope` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2086 | `losses_factor_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2092 | `losses_factor_scales_with_gross_head` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2099 | `losses_factor_turbined_has_no_effect` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2109 | `losses_constant_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2116 | `losses_constant_independent_of_all_inputs` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2127 | `losses_factor_extraction_returns_factor` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2133 | `losses_factor_extraction_constant_returns_zero` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2141 | `two_point_minimum_curve_works` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2155 | `interpolation_second_segment_correct` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2248 | `net_head_no_tailrace_no_losses_equals_h_fore` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2257 | `net_head_polynomial_tailrace_constant_losses_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2278 | `net_head_piecewise_tailrace_factor_losses` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2302 | `net_head_clamped_to_zero_when_losses_exceed_forebay` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2325 | `evaluate_acceptance_criterion` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2348 | `partial_derivatives_no_tailrace_ds_is_zero` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2360 | `partial_derivatives_no_tailrace_dv_is_positive` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2385 | `partial_derivatives_polynomial_tailrace_constant_losses` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2425 | `partial_derivatives_polynomial_tailrace_constant_losses_dv_positive` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2442 | `partial_derivatives_polynomial_tailrace_ds_nonpositive` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2463 | `partial_derivatives_factor_losses_dv_accounts_for_k_factor` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2504 | `partial_derivatives_piecewise_tailrace_factor_losses` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2548 | `partial_derivatives_piecewise_tailrace_ds_negative` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2575 | `finite_difference_cross_check_dv` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2605 | `finite_difference_cross_check_dq` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2635 | `finite_difference_cross_check_ds` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2665 | `finite_difference_cross_check_all_derivatives_factor_losses` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2786 | `no_fitting_window_uses_forebay_defaults` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2806 | `absolute_bounds_both_set` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2829 | `absolute_bounds_only_min` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2852 | `absolute_bounds_only_max` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2876 | `percentile_bounds_both_set` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2901 | `mixed_absolute_min_percentile_max` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2925 | `conflicting_min_bound_returns_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2949 | `conflicting_max_bound_returns_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2973 | `inverted_absolute_bounds_returns_empty_window_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 2995 | `equal_absolute_bounds_returns_empty_window_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3019 | `absolute_min_below_forebay_gets_clamped` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3041 | `absolute_max_above_forebay_gets_clamped` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3065 | `discretization_all_none_defaults_to_five` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3079 | `discretization_explicit_values_passed_through` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3102 | `volume_discretization_one_returns_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3124 | `volume_discretization_zero_returns_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3145 | `turbine_discretization_one_returns_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3166 | `spillage_discretization_one_returns_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3189 | `max_planes_per_hydro_none_defaults_to_ten` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3201 | `max_planes_per_hydro_explicit_value` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3216 | `max_planes_per_hydro_zero_returns_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3270 | `tangent_plane_at_known_operating_point_coefficients` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3313 | `tangent_plane_identity_at_operating_point` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3332 | `tangent_plane_identity_at_second_operating_point` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3351 | `tangent_plane_identity_with_spillage` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3370 | `compute_tangent_plane_zero_flow_returns_none` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3378 | `compute_tangent_plane_negative_flow_returns_none` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3389 | `compute_tangent_plane_zero_production_returns_none` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3410 | `raw_hyperplane_evaluate_linear_combination` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3428 | `gamma_v_positive_for_positive_net_head` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3441 | `gamma_s_nonpositive_with_tailrace` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3454 | `raw_hyperplane_implements_debug_clone_copy` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3506 | `sample_tangent_planes_count_between_100_and_125_for_5x5x5` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3519 | `sample_tangent_planes_count_at_most_n_v_times_n_q_times_n_s` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3540 | `sample_tangent_planes_flow_grid_avoids_zero_q` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3557 | `sample_tangent_planes_spillage_grid_starts_at_zero` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3574 | `eliminate_redundant_strictly_reduces_count_for_non_trivial_geometry` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3592 | `eliminate_redundant_envelope_upper_bounds_production_function` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3641 | `eliminate_redundant_constant_head_produces_one_plane` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3672 | `eliminate_redundant_empty_input_returns_empty` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3682 | `eliminate_redundant_spillage_planes_can_survive` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3704 | `eliminate_redundant_output_is_subset_of_input` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3773 | `select_planes_reduces_to_target_count` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3794 | `select_planes_approximation_error_not_catastrophically_worse` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3815 | `select_planes_passthrough_when_input_is_small` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3857 | `select_planes_preserves_envelope_property` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3902 | `select_planes_empty_input_returns_empty` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3911 | `select_planes_single_plane_returns_unchanged` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3942 | `compute_max_approximation_error_is_zero_for_linear_production_function` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3972 | `compute_max_approximation_error_empty_planes_returns_zero` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3984 | `compute_max_approximation_error_is_non_negative` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 3995 | `select_planes_output_is_subset_of_input` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4016 | `compute_kappa_in_valid_range_for_realistic_geometry` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4038 | `compute_kappa_in_range_for_realistic_geometry` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4076 | `compute_kappa_is_one_for_linear_production_function` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4110 | `compute_kappa_less_than_one_for_nonlinear_production_function` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4132 | `compute_kappa_empty_planes_returns_one` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4146 | `validate_fitted_planes_valid_input_returns_ok` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4167 | `validate_fitted_planes_zero_kappa_returns_invalid_kappa` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4183 | `validate_fitted_planes_kappa_above_one_returns_invalid_kappa` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4199 | `validate_fitted_planes_negative_kappa_returns_invalid_kappa` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4215 | `validate_fitted_planes_empty_planes_returns_no_hyperplanes` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4225 | `validate_fitted_planes_negative_gamma_v_returns_invalid_coefficient` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4245 | `validate_fitted_planes_negative_gamma_q_returns_invalid_coefficient` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4265 | `validate_fitted_planes_positive_gamma_s_returns_invalid_coefficient` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4285 | `validate_fitted_planes_near_zero_gamma_v_within_tolerance_passes` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4301 | `validate_fitted_planes_near_zero_gamma_s_within_tolerance_passes` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4317 | `validate_fitted_planes_kappa_exactly_one_passes` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4382 | `fit_fpha_planes_sobradinho_style_end_to_end` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4430 | `fit_fpha_planes_intercepts_are_kappa_scaled` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4459 | `fit_fpha_planes_linear_function_produces_one_plane_with_kappa_one` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4539 | `fit_fpha_planes_propagates_forebay_error_on_insufficient_rows` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4566 | `display_invalid_kappa_contains_name_and_value` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4577 | `display_no_hyperplanes_produced_contains_name` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4589 | `display_invalid_coefficient_contains_name_and_index_and_detail` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/fpha_fitting.rs` | 4606 | `fit_fpha_planes_result_kappa_in_range_and_intercept_consistent` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 1945 | `fpha_entity_without_config_entry_returns_validation_error` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/hydro_models.rs` | 2643 | `fpha_plane_is_copy` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1651 | `fpha_no_hydros_generation_is_empty` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1685 | `fpha_one_hydro_one_block_three_planes` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1716 | `fpha_two_hydros_two_blocks_different_planes` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1752 | `fpha_generation_contiguous_with_prior_region` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 1772 | `fpha_rows_contiguous_with_load_balance` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/indexer.rs` | 2052 | `fpha_row_range_debug_clone_copy` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1546 | `fpha_turbined_cost_applied_to_fpha_turbine_column` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1600 | `fpha_turbined_cost_multi_block_uses_per_block_hours` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 1636 | `fpha_turbined_cost_mixed_system_only_fpha_hydros_carry_cost` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3139 | `fpha_ac1_dimensions_one_fpha_hydro_five_planes` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3200 | `fpha_ac2_generation_column_entries` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3250 | `fpha_ac3_v_in_column_entries` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3288 | `fpha_ac4_v_out_column_entries` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3335 | `fpha_ac5_mixed_system_load_balance_uses_generation_col` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3482 | `fpha_solve_one_hydro_optimal` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3537 | `fpha_solve_hyperplane_constraints_hold` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3647 | `fpha_solve_storage_fixing_dual_differs_from_constant` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/lp_builder/template.rs` | 3756 | `fpha_solve_mixed_system_optimal` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2512 | `fpha_generation_read_from_lp_column` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2598 | `fpha_productivity_is_none` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/src/simulation/extraction.rs` | 2842 | `fpha_turbined_cost_in_compute_cost_result` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_computed.rs` | 310 | `fpha_computed_case_converges` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_evaporation.rs` | 92 | `fpha_evaporation_case_converges` | **NO — cleanup item for Epic 09** |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_evaporation.rs` | 233 | `test_4ree_fpha_evap_seasonal_ref_provenance` | **NO — cleanup item for Epic 09** |

### E2E pipeline tests (slow by nature)

There are 4 `e2e` tests that run the full training/simulation pipeline.

| Crate | File | Line | Function | Guards |
| ----- | ---- | ---: | -------- | ------ |
| `cobre-sddp` | `crates/cobre-sddp/tests/estimation_integration.rs` | 890 | `test_partial_estimation_end_to_end` | `generic` |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_computed.rs` | 310 | `fpha_computed_case_converges` | `fpha-slow` |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_evaporation.rs` | 92 | `fpha_evaporation_case_converges` | `fpha-slow` |
| `cobre-sddp` | `crates/cobre-sddp/tests/fpha_evaporation.rs` | 233 | `test_4ree_fpha_evap_seasonal_ref_provenance` | `fpha-slow` |

### Parse errors

_No parse errors reported._

---

## 6. Post-epic-05 State (2026-04-19)

This section records the final inventory state after all epic-05 tickets
(ticket-002 through ticket-005, ticket-007) completed. The Section 3
guard-tagged tables above reflect the **pre-epic-05** snapshot and are
preserved as historical reference.

### Summary

**Workspace test count post-epic-05:** 3402
(captured by `cargo nextest run --workspace --all-features`, 2026-04-19)

Net delta from epic-05: -4 tests (four deletions, zero additions).

### Verification grep results (all must be zero hits)

| Check | Command | Result |
| ----- | ------- | ------ |
| `unified_run_path.rs` absent | `test -f crates/cobre-sddp/tests/unified_run_path.rs` | FILE ABSENT |
| `forward_pass_baked_ready` removed | `rg "forward_pass_baked_ready" crates/cobre-sddp/` | 0 hits |
| `test_stored_cut_row_offset_skips_baked_rows` removed | `rg "test_stored_cut_row_offset_skips_baked_rows" crates/` | 0 hits |
| `baked_template: None` removed from stage_solve | `rg "baked_template: None" crates/cobre-sddp/src/stage_solve.rs` | 0 hits |

### Tests deleted by tickets 002 and 003

Four tests were removed across epic-05:

| Ticket | File | Approx. line | Function | Guard |
| ------ | ---- | -----------: | -------- | ----- |
| 002 | `crates/cobre-sddp/tests/unified_run_path.rs` | (entire file) | entire file deleted | `unified-path` |
| 002 | `crates/cobre-sddp/tests/integration.rs` | ~1925 | `forward_pass_baked_ready_non_baked_skips_cut_rebuild` | `non-baked` (not in guard table) |
| 002 | `crates/cobre-sddp/tests/integration.rs` | ~2254 | `forward_pass_baked_ready_baked_performs_cut_rebuild` | `non-baked` (not in guard table) |
| 002 | `crates/cobre-sddp/src/forward.rs` | — | `forward_pass_baked_ready_skips_cut_batches_rebuild` | `non-baked` (in-file) |
| 003 | `crates/cobre-sddp/src/basis_reconstruct.rs` | ~832 | `test_stored_cut_row_offset_skips_baked_rows` | `stored-cut-row-offset` |

Note: the `unified_run_path.rs` file housed the two `forward_pass_baked_ready_*`
integration tests; the three "ticket-002" rows above reflect the two tests in
`integration.rs` plus the one in `forward.rs`, for a total of four deleted tests.

### Inventory mistag: `test_slot_lookup_growth_safe_in_release`

The test at `crates/cobre-sddp/src/basis_reconstruct.rs:783` carries the
`stored-cut-row-offset` guard in the Section 3 Epic-05 table, but this guard
is incorrect. The test exercises the slot-lookup growth path; it passes `0`
as the `stored_cut_row_offset` argument and never exercises the non-zero
offset logic. Ticket-003 retained this test (updating the call site to remove
the now-deleted offset parameter) and flagged the mistag in its Context
section. The mistag is preserved as-is in the Section 3 table above; correcting
the guard label is a future inventory-maintenance pass.

### `add-rows-trait`-tagged tests: retained (34 tests)

The Section 3 Epic-05 table lists 34 tests carrying the `add-rows-trait` guard
(the summary table in Section 1 also records 34; ticket-007 spec cited "33",
reflecting a count taken before `run_stage_solve_warm_start_excess_basic_demotes`
was added to the guard table during ticket-004 planning). All 34 are retained.

**Rationale:** `SolverInterface::add_rows` survives epic-05. Ticket-005
determined that the method has three legitimate callers that cannot be removed:

1. `forward.rs` — `append_new_cuts_to_lp` (incremental cut append during training)
2. `backward.rs` — `load_backward_lp` (backward LP construction)
3. `lower_bound.rs` — test fallback path

Because the trait method and its primary production callers remain, all 34
tests that exercise the `add_rows` API (directly against `HighsSolver` in
`cobre-solver`, via FPHA fitting and evaporation helpers in
`crates/cobre-sddp/src/lp_builder/template.rs`, or against the in-file
`MockSolver` in `stage_solve.rs`) are live tests of a live API and must be
kept.

**Affected files and counts:**

| File | add-rows-trait tests | Notes |
| ---- | -------------------: | ----- |
| `crates/cobre-sddp/src/lp_builder/template.rs` | 9 | 4 carry co-guard `fpha-slow` |
| `crates/cobre-sddp/src/stage_solve.rs` | 1 | `run_stage_solve_warm_start_excess_basic_demotes` |
| `crates/cobre-solver/src/highs.rs` | 7 | direct `HighsSolver` tests |
| `crates/cobre-solver/src/trait_def.rs` | 1 | `test_noop_solver_all_methods` |
| `crates/cobre-solver/tests/conformance.rs` | 16 | integration conformance suite |
| **Total** | **34** | |
