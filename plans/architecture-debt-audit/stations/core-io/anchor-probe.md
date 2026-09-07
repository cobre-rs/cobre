## INGEST ANCHOR PROBE — core-io (2026-09, baseline)

Register-shaped stub: one block per candidate, every anchor rendered so it resolves at the
pinned baseline via check-anchors.py. Symbols are declaration anchors; the single call-site
anchor (pipeline.rs::populate_derived_residual_ratios, declared in scenarios/residual_derivation.rs
and only called here) is rendered by line 225.

### A-architecture-00 · The 16-column hydro penalty vocabulary is declared three times inside cobre-core, two of them field-for-field identical

**Anchors:** `crates/cobre-core/src/entities/hydro.rs::HydroPenalties` `crates/cobre-core/src/model/resolved/penalties.rs::HydroStagePenalties` `crates/cobre-core/src/model/penalty.rs::HydroPenaltyOverrides`

### A-architecture-01 · Two ValidationError variants are constructed nowhere in the workspace and claim an emitter that never imports the type; the parameter that would implement one is dead behind a TODO

**Anchors:** `crates/cobre-core/src/error.rs::ValidationError` `crates/cobre-core/src/error.rs:72` `crates/cobre-core/src/topology/network.rs::build`

### A-architecture-02 · ResolvedBounds open-codes the cell-index arithmetic fourteen times for four entity families while the fifth has a named helper

**Anchors:** `crates/cobre-core/src/model/resolved/bounds.rs::thermal_cell_index` `crates/cobre-core/src/model/resolved/bounds.rs::ResolvedBounds`

### A-architecture-03 · Stage.index encodes the canonical stage order but is assigned in cobre-io and never re-derived or validated by the builder that owns that order

**Anchors:** `crates/cobre-core/src/model/temporal.rs::Stage` `crates/cobre-core/src/system/builder.rs::build` `crates/cobre-core/src/model/temporal/stage_key.rs::StudyPos`

### A-architecture-04 · StageLagTransition is a PAR-lag ring-buffer control block living in L0 with zero cobre-core consumers, documented in terms of a cobre-sddp private function

**Anchors:** `crates/cobre-core/src/model/temporal.rs::StageLagTransition` `crates/cobre-core/src/model/temporal.rs:202`

### A-architecture-05 · SystemBuilder enforces canonical order for nine collections and delegates it by doc comment for the seven scenario/stochastic tables it never sorts or checks

**Anchors:** `crates/cobre-core/src/system/builder.rs::build` `crates/cobre-core/src/system/builder.rs::inflow_models` `crates/cobre-core/src/system/mod.rs::inflow_models` `crates/cobre-core/src/system/mod.rs::with_scenario_models`

### A-architecture-06 · The wire-payload reproducibility rule is enforced by three different bespoke mechanisms and skipped for six HashMap fields that serialize inside the same System payload

**Anchors:** `crates/cobre-core/src/system/mod.rs::System` `crates/cobre-core/src/model/horizon.rs::HorizonGraph` `crates/cobre-core/src/topology/cascade.rs::CascadeTopology` `crates/cobre-core/src/topology/network.rs::NetworkTopology`

### A-over-engineering-00 · Two of the seven ValidationError variants are constructed nowhere in the workspace, and both carry a doc comment attributing them to a cobre-io validation path that uses a different error vocabulary

**Anchors:** `crates/cobre-core/src/error.rs::ValidationError` `crates/cobre-core/src/error.rs:65` `crates/cobre-core/src/error.rs:72`

### A-over-engineering-01 · The whole resolved per-(NCS, stage) penalty axis has no reader: ResolvedPenalties::ncs_penalties is called only from tests while the LP objective reads the entity-level curtailment cost, so the declared penalty_overrides_ncs stage override is resolved and dropped

**Anchors:** `crates/cobre-core/src/model/resolved/penalties.rs::ncs_penalties` `crates/cobre-core/src/model/resolved/penalties.rs::ncs_penalties_mut` `crates/cobre-core/src/model/resolved/penalties.rs::NcsStagePenalties` `crates/cobre-core/src/model/resolved/penalties.rs::ResolvedPenalties`

### A-over-engineering-02 · WelfordAccumulator ships a symmetric population-statistics half that no production caller wants: five of its nine methods have no production call site anywhere in the workspace

**Anchors:** `crates/cobre-core/src/stats/welford.rs::WelfordAccumulator` `crates/cobre-core/src/stats/welford.rs::variance` `crates/cobre-core/src/stats/welford.rs::std_dev` `crates/cobre-core/src/stats/welford.rs::sample_variance` `crates/cobre-core/src/stats/welford.rs::ci_95_half_width` `crates/cobre-core/src/stats/welford.rs::count`

### A-over-engineering-03 · NetworkTopology and its three companion records are a fully unconsumed derived structure: built at System construction, carried on System, serialized into the MPI broadcast payload, re-exported at the crate root, and referenced nowhere outside cobre-core

**Anchors:** `crates/cobre-core/src/topology/network.rs::NetworkTopology` `crates/cobre-core/src/topology/network.rs::build` `crates/cobre-core/src/topology/network.rs:91` `crates/cobre-core/src/system/mod.rs:87` `crates/cobre-core/src/system/mod.rs::network`

### A-perf-00 · The 366-day canonical calendar is heap-built on every call, and the season map's multi-resolution classification — a pure function of an immutable map — is recomputed once per study stage by the loader

**Anchors:** `crates/cobre-core/src/model/temporal.rs::canonical_calendar_days` `crates/cobre-core/src/model/temporal.rs::is_multi_resolution` `crates/cobre-core/src/model/temporal.rs::span_days` `crates/cobre-core/src/model/temporal.rs::resolution_level_of`

### A-perf-01 · `window_period_overlaps` returns an owned, un-reserved `Vec<f64>` with no scalar or predicate entry point, so five of its nine production call sites allocate a vector purely to read one number, a length, or an emptiness test

**Anchors:** `crates/cobre-core/src/model/temporal/overlap.rs::window_period_overlaps` `crates/cobre-core/src/model/temporal/overlap.rs:41` `crates/cobre-core/src/model/temporal/overlap.rs:55`

### A-perf-02 · The MPI broadcast payload carries the two derived topologies (`cascade`, `network`) that every rank could rebuild locally, while the sibling index maps three fields above are skipped and rebuilt

**Anchors:** `crates/cobre-core/src/system/mod.rs::System` `crates/cobre-core/src/system/mod.rs::SystemRepr` `crates/cobre-core/src/system/mod.rs::rebuild_indices` `crates/cobre-core/src/topology/cascade.rs::CascadeTopology` `crates/cobre-core/src/topology/network.rs::NetworkTopology`

### A-test-bloat-00 · `training_event.rs` carries a 463-line inline test module over a data enum with zero methods; every test echoes literals it just wrote or asserts a derived Debug is non-empty

**Anchors:** `crates/cobre-core/src/constraints/training_event.rs::make_all_variants` `crates/cobre-core/src/constraints/training_event.rs::all_variants_construct` `crates/cobre-core/src/constraints/training_event.rs::all_variants_clone` `crates/cobre-core/src/constraints/training_event.rs::all_variants_debug_non_empty` `crates/cobre-core/src/constraints/training_event.rs::forward_pass_complete_fields_accessible` `crates/cobre-core/src/constraints/training_event.rs::stage_row_selection_record_fields_accessible`

### A-test-bloat-01 · Six independent all-same-value `HydroPenalties` test fixtures inside cobre-core, one per test module, each re-enumerating all 16 fields

**Anchors:** `crates/cobre-core/src/entities/hydro.rs::HydroPenalties` `crates/cobre-core/src/entities/hydro.rs::penalties_all` `crates/cobre-core/src/system/builder.rs::zero_penalties` `crates/cobre-core/src/system/mod.rs:636` `crates/cobre-core/src/topology/cascade.rs:145` `crates/cobre-core/src/topology/network.rs:235` `crates/cobre-core/tests/integration.rs::zero_hydro_penalties`

### A-test-bloat-02 · Tests that exercise derived `PartialEq`, `Clone`, `Copy` and `Hash` rather than any code the crate wrote, scattered across `entities/` and `entity_id.rs`

**Anchors:** `crates/cobre-core/src/entity_id.rs::test_equality` `crates/cobre-core/src/entity_id.rs::test_copy` `crates/cobre-core/src/entity_id.rs::test_hash_consistency` `crates/cobre-core/src/entities/bus.rs::test_bus_equality` `crates/cobre-core/src/entities/energy_contract.rs::test_contract_type_equality` `crates/cobre-core/src/entities/pumping_station.rs::test_pumping_station_construction` `crates/cobre-core/src/model/scenario.rs::annual_component_partial_eq_clone` `crates/cobre-core/src/constraints/initial_conditions.rs::test_hydro_storage_clone`

### A-test-bloat-03 · Ad-hoc per-struct bit comparators in `resolved/bounds.rs` with nothing enforcing field exhaustiveness, where the testing yardstick calls for one shared comparator

**Anchors:** `crates/cobre-core/src/model/resolved/bounds.rs::opt_f64_bits_eq` `crates/cobre-core/src/model/resolved/bounds.rs::hydro_stage_bounds_bits_eq` `crates/cobre-core/src/model/resolved/bounds.rs::hydro_block_bounds_bits_eq` `crates/cobre-core/src/model/resolved/bounds.rs::thermal_block_bounds_bits_eq` `crates/cobre-core/src/model/resolved/bounds.rs::line_bounds_bits_eq` `crates/cobre-core/src/model/resolved/bounds.rs::pumping_bounds_bits_eq` `crates/cobre-core/src/model/resolved/bounds.rs::contract_bounds_bits_eq`

### A-test-bloat-04 · The entity fixture-builder family (make_bus / make_line / make_thermal / make_ncs / make_group / make_hydro / make_contract / make_pumping_station) is copy-pasted across five test sites in cobre-core

**Anchors:** `crates/cobre-core/src/topology/network.rs::make_line` `crates/cobre-core/src/system/mod.rs::make_line` `crates/cobre-core/tests/integration.rs::make_line` `crates/cobre-core/src/system/builder.rs::line` `crates/cobre-core/src/topology/cascade.rs::make_hydro`

### B-architecture-00 · Nineteen input-path parsers open-code the same eight-line Parquet reader prologue, and three of them re-declare byte-identical copies of two helpers `parquet_helpers.rs` already exports

**Anchors:** `crates/cobre-io/src/extensions/hydro_geometry.rs::extract_int32_column` `crates/cobre-io/src/extensions/hydro_energy_productivity.rs::extract_int32_column` `crates/cobre-io/src/extensions/tailrace_curves.rs::extract_int32_column` `crates/cobre-io/src/constraints/bounds.rs::parse_line_bounds` `crates/cobre-io/src/scenarios/inflow_stats.rs::parse_inflow_seasonal_stats` `crates/cobre-io/src/parquet_helpers.rs::extract_required_int32`

### B-architecture-01 · The entity-slice ordering precondition is stated three contradictory ways across the seven resolvers in `resolution/`, and two of the wrong statements sit on crate-root-public functions

**Anchors:** `crates/cobre-io/src/resolution/bounds.rs::BoundsEntitySlices` `crates/cobre-io/src/resolution/penalties.rs::PenaltiesEntitySlices` `crates/cobre-io/src/resolution/ncs_bounds.rs::resolve_ncs_bounds` `crates/cobre-io/src/resolution/load_factors.rs::resolve_load_factors` `crates/cobre-io/src/resolution/ncs_factors.rs::resolve_ncs_factors` `crates/cobre-io/src/resolution/group_bounds.rs::resolve_hydro_unit_group_bounds` `crates/cobre-io/src/lib.rs:130`

### B-architecture-02 · `resolve_load_factors` and `resolve_ncs_factors` are the same function twice, algorithm and tests included, differing only in entity type

**Anchors:** `crates/cobre-io/src/resolution/load_factors.rs::resolve_load_factors` `crates/cobre-io/src/resolution/ncs_factors.rs::resolve_ncs_factors`

### B-architecture-03 · Every semantic stage rule the parser already enforces is unreachable: `stages.rs` rejects non-positive block hours and out-of-range CVaR at parse time, and the pipeline bails before Layer 5 ever sees such a deck

**Anchors:** `crates/cobre-io/src/stages.rs::validate_block_hours` `crates/cobre-io/src/stages.rs::validate_risk_measure` `crates/cobre-io/src/stages.rs::validate_raw_stages` `crates/cobre-io/src/stages.rs::convert_stages` `crates/cobre-io/src/validation/semantic/stages.rs::check_stage_structure`

### B-over-engineering-00 · cobre-io's four public postcard broadcast helpers are a parallel MPI serialization facade that the MPI path does not use

**Anchors:** `crates/cobre-io/src/broadcast.rs::serialize_system` `crates/cobre-io/src/broadcast.rs::deserialize_system` `crates/cobre-io/src/broadcast.rs::serialize_parameters` `crates/cobre-io/src/broadcast.rs::deserialize_parameters`

### B-over-engineering-01 · Two zero-consumer public convenience helpers ship from the crate root: a `case_dir`-taking wrapper whose only real consumer open-codes the join it owns, and a stage-to-season map builder nothing builds

**Anchors:** `crates/cobre-io/src/extensions/scalar_parameters.rs::load_scalar_parameters_json` `crates/cobre-io/src/stages.rs::build_season_stage_map`

### B-over-engineering-02 · `load_scenarios` and its 9-field `ScenarioData` result are a zero-consumer public scenario-assembly orchestrator that mirrors the production assembly path and diverges from it by construction

**Anchors:** `crates/cobre-io/src/scenarios/mod.rs::load_scenarios` `crates/cobre-io/src/scenarios/mod.rs::ScenarioData` `crates/cobre-io/src/pipeline.rs:225`

### B-perf-00 · The generic-constraint tokenizer copies each expression into a `Vec<char>` and allocates a fresh `String` for every identifier and numeric literal instead of borrowing the input

**Anchors:** `crates/cobre-io/src/constraints/generic.rs::tokenize` `crates/cobre-io/src/constraints/generic.rs::Token`

### B-perf-01 · Named-expression inlining rebuilds the whole name index on every call, and the relational path pays a linear table scan plus a throwaway one-element vector per reference term

**Anchors:** `crates/cobre-io/src/constraints/named_expression_inline.rs::inline` `crates/cobre-io/src/constraints/generic.rs::resolve_split_side`

### B-perf-02 · The stochastic estimator resolves per-group work by re-scanning a whole table once per group, making history-window projection quadratic in record depth

**Anchors:** `crates/cobre-io/src/scenarios/estimation.rs::resolve_coverage_gated_observations` `crates/cobre-io/src/scenarios/estimation.rs::check_std_ratio_divergence`

### B-perf-03 · Inflow-history parsing materializes a second full copy of the table solely to call the shared windowed-record validator

**Anchors:** `crates/cobre-io/src/scenarios/inflow_history.rs::parse_inflow_history` `crates/cobre-io/src/windowed_history.rs::validate_windowed_records`

### B-test-bloat-00 · test_serialized_size_reasonable asserts an undocumented magic threshold that pins no contract and belongs to no test tier

**Anchors:** `crates/cobre-io/src/broadcast.rs::test_serialized_size_reasonable`

### B-test-bloat-01 · The input path has no shared test-support home: 39 throwaway fixture helpers are copied verbatim across 20 files (18x write_json, 16x write_parquet, 5x make_global)

**Anchors:** `crates/cobre-io/src/constraints/bounds.rs::write_parquet` `crates/cobre-io/src/system/buses.rs::make_global` `crates/cobre-io/src/system/buses.rs::write_json` `crates/cobre-io/src/validation/semantic/mod.rs::test_support` `crates/cobre-io/tests/helpers/mod.rs::write_file`

### B-test-bloat-02 · Unit-test homing in the input path is a coin flip: seven inline modules run 1021-2603 test lines against the section 5.1 extraction threshold, while the single extracted sibling sits in the same directory as a 578-line inline module

**Anchors:** `crates/cobre-io/src/constraints/generic.rs::tests` `crates/cobre-io/src/scenarios/estimation.rs::tests` `crates/cobre-io/src/scenarios/correlation.rs::tests` `crates/cobre-io/src/resolution/bounds.rs::tests` `crates/cobre-io/src/system/hydros.rs::tests` `crates/cobre-io/src/stages.rs::tests`

### B-test-bloat-03 · The three seasonal-stats parsers carry the same six-test template written out three times, down to the shuffle vectors in the order-invariance test

**Anchors:** `crates/cobre-io/src/scenarios/inflow_stats.rs::tests` `crates/cobre-io/src/scenarios/load_stats.rs::tests` `crates/cobre-io/src/scenarios/non_controllable_stats.rs::tests` `crates/cobre-io/src/scenarios/non_controllable_stats.rs::make_batch` `crates/cobre-io/src/scenarios/load_stats.rs::make_batch`

### B-test-bloat-04 · Six input-path integration binaries each re-declare mod helpers, recompiling the 312-line shared harness six times against the one-binary-per-crate target

**Anchors:** `crates/cobre-io/tests/integration.rs::helpers` `crates/cobre-io/tests/invariance.rs::helpers` `crates/cobre-io/tests/resolver_builder_index_alignment.rs::helpers` `crates/cobre-io/tests/post_study_stages.rs::helpers` `crates/cobre-io/tests/load_case_productivity_resolution.rs::helpers` `crates/cobre-io/tests/load_case_scalar_parameters.rs::helpers`

### B-test-bloat-05 · test_bus_ordering_invariance is strictly subsumed by test_full_case_ordering_invariance: same two fixtures, same two load_case calls, same final System-equality assertion

**Anchors:** `crates/cobre-io/tests/invariance.rs::test_bus_ordering_invariance` `crates/cobre-io/tests/invariance.rs::test_full_case_ordering_invariance` `crates/cobre-io/tests/invariance.rs::make_shuffled_multi_entity_case`

### C-architecture-00 · The config admission rules are enforced only as a side effect of an accessor, and every cobre-io caller discards the resulting error on a premise the Layer-2 gate does not satisfy

**Anchors:** `crates/cobre-io/src/config/mod.rs::validate_scenario_source_cfg` `crates/cobre-io/src/config/mod.rs::validate_openings_cfg` `crates/cobre-io/src/config/mod.rs::validate_config` `crates/cobre-io/src/validation/semantic/scenarios.rs::check_external_scheme_has_files`

### C-architecture-01 · Entity referential and filling invariants are implemented twice — cobre-io Layer 3/5a and cobre-core `SystemBuilder::build` — with the cobre-core copy unreachable through the only production build path

**Anchors:** `crates/cobre-io/src/validation/referential.rs::validate_referential_integrity` `crates/cobre-io/src/validation/referential.rs::check_line_references` `crates/cobre-io/src/validation/semantic/hydro.rs::check_filling_guards` `crates/cobre-io/src/error.rs::LoadError`

### C-architecture-02 · The bound-override rule family is split across two validation layers by entity family, and the two halves disagree on both `ErrorKind` and duplicate-key granularity for the same defect

**Anchors:** `crates/cobre-io/src/validation/referential.rs::check_generic_constraint_bounds_validity` `crates/cobre-io/src/validation/semantic/block_bounds.rs::check_bound_block_id_range` `crates/cobre-io/src/validation/semantic/block_bounds.rs::check_duplicate_bound_rows` `crates/cobre-io/src/validation/referential.rs::check_ncs_bounds_and_factors`

### C-architecture-03 · The foreign-key and study-stage idioms are open-coded across the validation layer while a table-driven precedent for the same shape already exists in the same station

**Anchors:** `crates/cobre-io/src/validation/referential.rs::check_scenario_references` `crates/cobre-io/src/validation/referential.rs::check_bounds_references` `crates/cobre-io/src/validation/semantic/block_bounds.rs::FamilyMeta`

### C-architecture-04 · The semantic layer's rule registry is a prose table with no binding to code, and it has already drifted

**Anchors:** `crates/cobre-io/src/validation/semantic/mod.rs:105` `crates/cobre-io/src/validation/semantic/mod.rs::validate_semantic_stages_penalties_scenarios` `crates/cobre-io/src/config/simulation.rs::SimulationConfig`

### C-architecture-05 · An out-of-horizon `stage_id` on a bound-override row is a hard error for two families and a silent drop for the other five

**Anchors:** `crates/cobre-io/src/validation/semantic/thermal.rs::check_thermal_bounds_override_stage_range` `crates/cobre-io/src/validation/semantic/block_bounds.rs:98` `crates/cobre-io/src/validation/referential.rs::check_ncs_bounds_and_factors`

### C-architecture-06 · The input-file registry is four hand-maintained parallel lists joined by position, guarded only by a length assertion

**Anchors:** `crates/cobre-io/src/validation/structural.rs::manifest_fields_mut` `crates/cobre-io/src/validation/structural.rs::FileManifest` `crates/cobre-io/src/validation/structural.rs::validate_structure` `crates/cobre-io/src/validation/schema.rs::ParsedData`

### C-over-engineering-00 · Three hand-written `Deserialize` impls in `config/` reimplement what `#[derive(Deserialize)]` already produces, each duplicating its accepted-value list

**Anchors:** `crates/cobre-io/src/config/estimation.rs::OrderSelectionMethod` `crates/cobre-io/src/config/scenario_source.rs::RawSamplingScheme` `crates/cobre-io/src/config/training.rs::StoppingMode`

### C-over-engineering-01 · `ErrorKind::default_severity` is a public severity classification with zero callers, and it already disagrees with the call sites it purports to describe

**Anchors:** `crates/cobre-io/src/validation/mod.rs::default_severity` `crates/cobre-io/src/validation/semantic/season.rs:177`

### C-over-engineering-02 · Two `#[allow(dead_code)]` attributes on `ParsedData` carry rationales naming consumers that do not exist: `penalties` is genuinely never read, `scalar_parameters` is read and the allow is stale

**Anchors:** `crates/cobre-io/src/validation/schema.rs::ParsedData` `crates/cobre-io/src/validation/schema.rs::ParsedData`

### C-over-engineering-03 · `FileManifest` presence tracking is three hand-maintained 43-element lists zipped positionally, with a hand-rolled `[&mut bool; 43]` reflection helper and only a length guard

**Anchors:** `crates/cobre-io/src/validation/structural.rs::FileManifest` `crates/cobre-io/src/validation/structural.rs::FILE_ENTRIES` `crates/cobre-io/src/validation/structural.rs::manifest_fields_mut`

### C-perf-00 · `merged_windows_for_hydro` filters the whole inflow-history table per hydro, and two independent seeding rules each build the same per-hydro map from scratch

**Anchors:** `crates/cobre-io/src/validation/semantic/inflow_seeding.rs::merged_windows_for_hydro` `crates/cobre-io/src/validation/semantic/inflow_seeding.rs::check_slot_coverage` `crates/cobre-io/src/validation/semantic/inflow_seeding.rs::check_inprogress_partial_coverage`

### C-perf-01 · Prefix-coherence re-walks the entire root prefix for every graph edge, so a repeated column pair is compared once per edge instead of once per pair

**Anchors:** `crates/cobre-io/src/validation/semantic/scenarios.rs::check_prefix_coherence` `crates/cobre-io/src/validation/semantic/scenarios.rs:853`

### C-perf-02 · `extract_class` materializes four full-table-sized structures per external class, one of which carries values no rule reads on the chain dialect

**Anchors:** `crates/cobre-io/src/validation/semantic/scenarios.rs::extract_class` `crates/cobre-io/src/validation/semantic/scenarios.rs::check_external_library_coherence`

### C-perf-03 · Three modules build the same sorted stage-window index and re-run the same per-row season resolution over the inflow-history table under one shared precondition

**Anchors:** `crates/cobre-io/src/validation/semantic/season.rs::check_observation_season_alignment` `crates/cobre-io/src/validation/semantic/season.rs::check_season_observation_coverage` `crates/cobre-io/src/validation/semantic/scenarios.rs::check_estimation_prerequisites`

### C-perf-04 · `slot_occupying_classes` re-scans all three external scenario tables once per graph node and once per staged stage, recomputing a per-stage column count the sibling module already builds in one pass

**Anchors:** `crates/cobre-io/src/validation/semantic/stages.rs::slot_occupying_classes` `crates/cobre-io/src/validation/semantic/stages.rs::check_realization_rules` `crates/cobre-io/src/validation/semantic/stages.rs::check_num_openings_declaration` `crates/cobre-io/src/validation/semantic/stages.rs::check_sampling_method_meaningfulness` `crates/cobre-io/src/validation/semantic/scenarios.rs::ClassExternal`

### C-test-bloat-00 · `error.rs`'s six inline tests assert only that thiserror interpolates the format strings declared a hundred lines above them, and two of the six are the same test twice

**Anchors:** `crates/cobre-io/src/error.rs::LoadError` `crates/cobre-io/src/error.rs::test_load_error_io_display` `crates/cobre-io/src/error.rs::test_load_error_io_helper` `crates/cobre-io/src/error.rs::test_load_error_schema_display` `crates/cobre-io/src/error.rs::test_load_error_is_std_error`

### C-test-bloat-01 · Four of `schema.rs`'s seven inline tests assert nested-subsumed structural properties of an artifact the CI schema gate already pins byte-exactly, and the count assertion tolerates losing a schema

**Anchors:** `crates/cobre-io/src/schema.rs::test_generate_schemas_returns_expected_count` `crates/cobre-io/src/schema.rs::test_all_schema_filenames_and_values_non_empty` `crates/cobre-io/src/schema.rs::test_all_schemas_are_objects` `crates/cobre-io/src/schema.rs::test_all_schemas_have_structure_keys` `crates/cobre-io/src/schema.rs::generate_schemas`

### C-test-bloat-02 · The eight-file valid-case JSON corpus, `write_file` and `make_minimal_case` are triplicated across two inline test modules and the shared integration helper

**Anchors:** `crates/cobre-io/src/validation/schema.rs::VALID_CONFIG_JSON` `crates/cobre-io/src/validation/schema.rs::write_file` `crates/cobre-io/src/validation/referential.rs::make_minimal_case` `crates/cobre-io/src/validation/referential.rs::VALID_CONFIG_JSON` `crates/cobre-io/tests/helpers/mod.rs::make_minimal_case` `crates/cobre-io/tests/helpers/mod.rs::VALID_CONFIG_JSON`

### C-test-bloat-03 · `test_filling_guard_no_exit_no_error` is a byte-for-byte clone of `test_filling_guard_entry_below_horizon_no_error` and cannot exercise the condition its name claims

**Anchors:** `crates/cobre-io/src/validation/semantic/hydro.rs::test_filling_guard_no_exit_no_error` `crates/cobre-io/src/validation/semantic/hydro.rs::test_filling_guard_entry_below_horizon_no_error` `crates/cobre-io/src/validation/semantic/hydro.rs::make_filling_hydro` `crates/cobre-io/src/validation/semantic/hydro.rs::test_filling_guard_exit_on_filling_errors`

### C-test-bloat-04 · `validation/semantic/test_support.rs` is `pub(super)`-scoped one level too deep, so the four sibling validation-phase modules hand-roll byte-identical fixture builders and two more full `ParsedData` skeletons

**Anchors:** `crates/cobre-io/src/validation/semantic/mod.rs::test_support` `crates/cobre-io/src/validation/semantic/test_support.rs::penalties_all` `crates/cobre-io/src/validation/semantic/test_support.rs::base_parsed_data` `crates/cobre-io/src/validation/dimensional.rs::penalties_default` `crates/cobre-io/src/validation/dimensional.rs::base_parsed_data` `crates/cobre-io/src/validation/productivity_resolution.rs::penalties_default` `crates/cobre-io/src/validation/productivity_resolution.rs::base_parsed_data` `crates/cobre-io/src/validation/referential.rs::hydro_penalties` `crates/cobre-io/src/validation/referential.rs::make_unit_group` `crates/cobre-io/src/validation/scalar_parameters.rs::zero_hydro_penalties`

### C-test-bloat-05 · `thermal.rs`'s `boundary_tests` sub-module restates three tests that already exist 140 lines above it with the same fixture arguments and weaker assertions

**Anchors:** `crates/cobre-io/src/validation/semantic/thermal.rs::boundary_tests` `crates/cobre-io/src/validation/semantic/thermal.rs::override_at_t_minus_1_acceptance_boundary` `crates/cobre-io/src/validation/semantic/thermal.rs::test_thermal_bounds_override_stage_within_horizon_accepted` `crates/cobre-io/src/validation/semantic/thermal.rs::test_thermal_bounds_override_stage_equals_n_rejected` `crates/cobre-io/src/validation/semantic/thermal.rs::test_thermal_bounds_override_multiple_offending_rows`

### D-architecture-00 · atomic.rs declares itself sole owner of the write-side crash-safety contract, but the policy checkpoint and two dictionary CSVs bypass it entirely

**Anchors:** `crates/cobre-io/src/output/atomic.rs::write_bytes_atomic` `crates/cobre-io/src/output/policy/checkpoint.rs::write_policy_checkpoint` `crates/cobre-io/src/output/policy/checkpoint.rs:242` `crates/cobre-io/src/output/dictionary.rs::write_entities_csv` `crates/cobre-io/src/output/dictionary.rs::write_variables_csv`

### D-architecture-01 · write_results is documented as the top-level output entry point mirroring load_case, but reaches 5 of the crate's 27 output writers, so cobre-io publishes an aggregate contract it does not honour

**Anchors:** `crates/cobre-io/src/output/mod.rs:6` `crates/cobre-io/src/output/results_writer.rs::write_results` `crates/cobre-io/src/output/results_writer.rs::write_training_results`

### D-architecture-02 · The output Arrow-schema family has no single owner: 34 schemas across 4 modules, and both the shipped data dictionary and the axis-spelling gate enumerate a hand-maintained subset

**Anchors:** `crates/cobre-io/src/output/schemas.rs::costs_schema` `crates/cobre-io/src/output/dictionary.rs::variables_csv_schemas` `crates/cobre-io/src/output/schemas.rs::one_spelling_per_axis_across_every_output_schema` `crates/cobre-io/src/output/stochastic.rs::noise_openings_schema` `crates/cobre-io/src/output/hydro_models.rs::fpha_hyperplanes_schema` `crates/cobre-io/src/output/dictionary.rs::bounds_schema`

### D-architecture-03 · The training-loop vocabulary is baked into the L2 output schema surface as on-disk column names, a coupling the genericity oracle structurally cannot see and no Part-I item names

**Anchors:** `crates/cobre-io/src/output/schemas.rs::convergence_schema` `crates/cobre-io/src/output/schemas.rs::iteration_timing_schema` `crates/cobre-io/src/output/schemas.rs::row_selection_schema` `crates/cobre-io/src/output/mod.rs::IterationRecord` `scripts/ci/check-infra-genericity.sh:79`

### D-architecture-04 · SimulationParquetWriter answers "which entity families does this run emit" twice, with different inputs and contradicting doc comments, inside one file

**Anchors:** `crates/cobre-io/src/output/simulation_writer.rs::new` `crates/cobre-io/src/output/simulation_writer.rs::write_scenario` `crates/cobre-io/src/output/simulation_writer.rs::write_partition`

### D-architecture-05 · The write path has no helper module answering to parquet_helpers.rs: its one shared helper is homed inside a domain writer, two modules re-open-code it, and the identical write prologue repeats ten times

**Anchors:** `crates/cobre-io/src/parquet_helpers.rs::extract_required_date32` `crates/cobre-io/src/output/stochastic.rs::ensure_parent_dir` `crates/cobre-io/src/output/fixed_delivery.rs::write_fixed_delivery` `crates/cobre-io/src/output/scaling_report.rs::write_scaling_report` `crates/cobre-io/src/output/provenance.rs::write_provenance_report`

### D-over-engineering-00 · `write_dictionaries` takes the whole SDDP-shaped study `Config` and never reads it, coupling a `System`-only writer to the engine config type for nothing

**Anchors:** `crates/cobre-io/src/output/dictionary.rs::write_dictionaries` `crates/cobre-io/src/output/dictionary.rs:71` `crates/cobre-io/src/output/dictionary.rs:16` `crates/cobre-io/src/output/results_writer.rs::write_training_results`

### D-over-engineering-01 · `default_bounds` and `default_upper_bound_kind` are serde-attribute helpers published as crate-root public API with no consumer anywhere

**Anchors:** `crates/cobre-io/src/output/manifest.rs::default_bounds` `crates/cobre-io/src/output/manifest.rs::default_upper_bound_kind` `crates/cobre-io/src/output/mod.rs:47`

### D-over-engineering-02 · `IterationRecord.time_bwd_setup_ms` and `time_fwd_setup_ms` are write-only public fields whose doc arrows promise a Parquet column the writer feeds from a different source

**Anchors:** `crates/cobre-io/src/output/mod.rs::IterationRecord` `crates/cobre-io/src/output/mod.rs::IterationRecord` `crates/cobre-io/src/output/training_writer.rs::build_iteration_timing_batch`

### D-over-engineering-03 · `ParquetWriterConfig` is a configurability surface no shipped caller can vary: never constructed except via `Default`, and half the writers rebuild the default locally instead of accepting the parameter

**Anchors:** `crates/cobre-io/src/output/parquet_config.rs::ParquetWriterConfig` `crates/cobre-io/src/output/parquet_config.rs::Default` `crates/cobre-io/src/output/atomic.rs::write_parquet_atomic` `crates/cobre-io/src/output/dictionary.rs:76` `crates/cobre-io/src/output/stochastic.rs:132` `crates/cobre-io/src/output/hydro_models.rs:87` `crates/cobre-io/src/config/exports.rs::ExportsConfig`

### D-over-engineering-04 · `read_f32_vector_as_f64` is a dead codec reader kept by a completeness rationale, with no caller, no test and no f32 field in the schema it mirrors

**Anchors:** `crates/cobre-io/src/output/policy/codec.rs::read_f32_vector_as_f64`

### D-over-engineering-05 · Two of the three functions in `pipeline.rs` are pure one-caller adapters, so the crate's main load entry point crosses three forwarding hops before any work happens

**Anchors:** `crates/cobre-io/src/pipeline.rs::run_pipeline` `crates/cobre-io/src/pipeline.rs::run_pipeline_with_report` `crates/cobre-io/src/pipeline.rs::run_pipeline_with_artifacts`

### D-perf-00 · Every policy artifact is copied out of the FlatBuffers builder into a fresh `Vec` that exists only to be handed to `fs::write`

**Anchors:** `crates/cobre-io/src/output/policy/codec.rs::serialize_stage_cuts` `crates/cobre-io/src/output/policy/checkpoint.rs::write_policy_checkpoint`

### D-perf-01 · Every per-scenario partition write rebuilds a run-invariant Arrow schema and a run-invariant Parquet `WriterProperties`

**Anchors:** `crates/cobre-io/src/output/simulation_writer.rs::build_costs_batch` `crates/cobre-io/src/output/schemas.rs::costs_schema` `crates/cobre-io/src/output/atomic.rs::write_parquet_atomic`

### D-perf-02 · `SimulationParquetWriter` accumulates one heap `String` per written partition for the whole run, and the merged result reaches no output file

**Anchors:** `crates/cobre-io/src/output/simulation_writer.rs::write_partition` `crates/cobre-io/src/output/mod.rs::SimulationOutput` `crates/cobre-io/src/output/mod.rs::merge`

### D-perf-03 · `SolverStatsRow` owns a heap `String` phase and a heap `Vec<u64>` histogram per row, undoing the producer's explicit `&'static str` no-allocation choice

**Anchors:** `crates/cobre-io/src/output/solver_stats_writer.rs::SolverStatsRow` `crates/cobre-io/src/output/solver_stats_writer.rs::SolverStatsRow` `crates/cobre-io/src/output/solver_stats_writer.rs::build_retry_histogram_batch`

### D-perf-04 · `solver_stats_writer` is the only Parquet writer in the crate that materializes an intermediate `Vec` per column instead of appending into a pre-sized Arrow builder

**Anchors:** `crates/cobre-io/src/output/solver_stats_writer.rs::build_iterations_columns`

### D-test-bloat-00 · Byte-identical 73-line fixture block (`make_config` / `make_output_context` / `make_system`) copy-pasted between two sibling inline test modules in `output/`

**Anchors:** `crates/cobre-io/src/output/convergence_reader.rs::make_config` `crates/cobre-io/src/output/results_writer.rs::make_config` `crates/cobre-io/src/output/convergence_reader.rs::make_output_context` `crates/cobre-io/src/output/results_writer.rs::make_output_context` `crates/cobre-io/src/output/convergence_reader.rs::make_system` `crates/cobre-io/src/output/results_writer.rs::make_system`

### D-test-bloat-01 · Three `dictionary.rs` description tests are strict subsets of the exhaustive description test in the same module

**Anchors:** `crates/cobre-io/src/output/dictionary.rs::every_listed_schema_column_has_a_nonempty_description` `crates/cobre-io/src/output/dictionary.rs::new_energy_columns_have_descriptions` `crates/cobre-io/src/output/dictionary.rs::every_hydros_schema_column_has_description` `crates/cobre-io/src/output/dictionary.rs::every_hydro_bus_generation_schema_column_has_description` `crates/cobre-io/src/output/dictionary.rs::variables_csv_schemas`

### D-test-bloat-02 · The Parquet read-back step is open-coded 47 times across nine `output/` inline test modules, including three helpers with the same body under two names

**Anchors:** `crates/cobre-io/src/output/fixed_delivery.rs::read_batch` `crates/cobre-io/src/output/generic_constraints_echo.rs::read_batch` `crates/cobre-io/src/output/solver_stats_writer.rs::read_parquet` `crates/cobre-io/src/output/simulation_writer.rs:2653` `crates/cobre-io/src/output/stochastic.rs:851`

### D-test-bloat-03 · `output/policy/mod.rs` is a 29-line re-export shim carrying a 1315-line inline test module whose 36 tests target the three submodules, which themselves are nearly untested

**Anchors:** `crates/cobre-io/src/output/policy/mod.rs::tests` `crates/cobre-io/src/output/policy/codec.rs::tests` `crates/cobre-io/src/output/policy/checkpoint.rs::read_policy_checkpoint` `crates/cobre-io/src/output/policy/records.rs::tests`

### D-test-bloat-04 · Fourteen output-schema field counts are pinned twice in `schemas.rs`, once in a per-schema test and again in the umbrella test's expected table

**Anchors:** `crates/cobre-io/src/output/schemas.rs::all_schema_functions_return_valid_schemas` `crates/cobre-io/src/output/schemas.rs::thermals_schema_field_count` `crates/cobre-io/src/output/schemas.rs::costs_schema_field_count_and_names` `crates/cobre-io/src/output/schemas.rs::rank_timing_schema_field_count`

### D-test-bloat-05 · `parquet_helpers.rs`, the crate's most reused extraction module, has zero direct tests while all twelve of its error branches are only reachable transitively

**Anchors:** `crates/cobre-io/src/parquet_helpers.rs::extract_required_int32` `crates/cobre-io/src/parquet_helpers.rs::extract_required_date32` `crates/cobre-io/src/parquet_helpers.rs::extract_optional_float64`

