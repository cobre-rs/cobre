## INGEST ANCHOR PROBE — test-corpus (2026-09, baseline)

Register-shaped stub: one block per candidate of the four merged lens files, every anchor rendered so
`check-anchors.py "INGEST ANCHOR PROBE — test-corpus (2026-09, baseline)" --register <this file> --baseline 077dbe2c` resolves it with the register's own parser
and `check-reraise.py "INGEST ANCHOR PROBE — test-corpus (2026-09, baseline)" --register <this file> --baseline 077dbe2c` screens every title against the retired corpus.
Ingest ref `<lens>-<nn>` (nn = zero-based index among the lens file's candidates). Anchor form at this station: `path::symbol`
for a Rust/Python declaration (the checker's declaration regex), `path:line` otherwise. Titles are rendered without backticks so a
path quoted in a title is never mistaken for an anchor. A candidate carrying `dupOf` renders a `Re-raise-of:` line naming the mirror
heading it merges into, so the re-raise checker treats the restatement as justified (dup-of merge, no fresh TD id).

**CD-900 · probe · test-bloat-00**

Byte-identical sixteen-field HydroPenalties fixture literal duplicated between the resolved-penalties and system test modules while the crate's test-support surface already owns the penalty constructors

- **Anchors:** `crates/cobre-core/src/model/resolved/penalties.rs::make_hydro_penalties` `crates/cobre-core/src/system/mod.rs:2409` `crates/cobre-core/src/entities/hydro.rs::uniform` `crates/cobre-core/src/test_support.rs:217` `crates/cobre-core/src/system/builder.rs::zero_penalties` `crates/cobre-core/Cargo.toml:30`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-901 · probe · test-bloat-01**

Debug-non-empty assertions in the training-event test module can only fail if the derive itself is broken, so they cannot fail on the variant data they name

- **Anchors:** `crates/cobre-core/src/constraints/training_event.rs::all_variants_clone` `crates/cobre-core/src/constraints/training_event.rs::all_variants_debug_non_empty` `crates/cobre-core/src/constraints/training_event.rs::stopping_rule_result_debug_non_empty` `crates/cobre-core/src/constraints/training_event.rs::all_variants_construct` `crates/cobre-core/src/constraints/training_event.rs::forward_pass_complete_fields_accessible` `crates/cobre-core/src/constraints/training_event.rs::stage_row_selection_record_fields_accessible`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-902 · probe · test-bloat-02**

Derive-only clone, equality and hash tests across five cobre-core modules assert what the derive macro already guarantees

- **Anchors:** `crates/cobre-core/src/entity_id.rs::test_equality` `crates/cobre-core/src/entity_id.rs::test_hash_consistency` `crates/cobre-core/src/entities/bus.rs::test_bus_equality` `crates/cobre-core/src/entities/energy_contract.rs::test_contract_type_equality` `crates/cobre-core/src/model/scenario.rs::annual_component_partial_eq_clone` `crates/cobre-core/src/constraints/initial_conditions.rs::test_hydro_storage_clone` `crates/cobre-core/src/entity_id.rs::test_copy`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-903 · probe · test-bloat-03**

Every cobre-core test module is homed inline with no sibling tests.rs, and the system and resolved-bounds modules sit far above the threshold the yardstick proposes for extraction

- **Anchors:** `crates/cobre-core/src/system/mod.rs:615` `crates/cobre-core/src/model/resolved/bounds.rs:1193` `docs/design/testing-architecture.md:253`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-904 · probe · test-bloat-04**

The hoist destination the yardstick names for the tolerance comparators already carries the bit-exact comparators but no tolerance helper, so that option extends an existing surface instead of creating one

- **Anchors:** `crates/cobre-core/src/test_support.rs::f64_bits_eq` `crates/cobre-core/src/test_support.rs::opt_f64_bits_eq` `crates/cobre-core/Cargo.toml:30` `docs/design/testing-architecture.md:437`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Re-raise-of:** mirror `Oracle test-harness duplication` — a dup-of merge into the open mirror item (prior-register.md), never a fresh TD id

**CD-905 · probe · test-bloat-05**

The policy directory module is almost entirely test code for submodules it only re-exports, and checkpoint.rs owns none of its own tests

- **Anchors:** `crates/cobre-io/src/output/policy/mod.rs:33` `crates/cobre-io/src/output/policy/mod.rs::write_policy_checkpoint_creates_directory_structure` `crates/cobre-io/src/output/policy/mod.rs::deserialize_stage_cuts_single_cut_all_fields` `crates/cobre-io/src/output/policy/codec.rs:1627`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-906 · probe · test-bloat-06**

One directory holds both homing conventions: the estimation parser extracted its tests to a sibling file while the correlation parser beside it keeps them inline

- **Anchors:** `crates/cobre-io/src/scenarios/estimation.rs:1206` `crates/cobre-io/src/scenarios/estimation/tests.rs:1` `crates/cobre-io/src/scenarios/correlation.rs:404`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-907 · probe · test-bloat-07**

The two richer case builders never crossed into the crate test-support surface, so six integration binaries re-declare a helpers module behind a crate-wide dead-code allow

- **Anchors:** `crates/cobre-io/tests/helpers/mod.rs:2` `crates/cobre-io/tests/helpers/mod.rs:9` `crates/cobre-io/tests/helpers/mod.rs::make_multi_entity_case` `crates/cobre-io/tests/helpers/mod.rs::make_referential_violation_case` `crates/cobre-io/tests/integration.rs:10` `crates/cobre-io/tests/load_case_scalar_parameters.rs:5`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-908 · probe · test-bloat-08**

The schema read-back prelude is repeated verbatim across four output writers while the row read-back has a shared helper in test-support

- **Anchors:** `crates/cobre-io/src/output/hydro_models.rs:515` `crates/cobre-io/src/output/stochastic.rs:787` `crates/cobre-io/src/output/training_writer.rs:753` `crates/cobre-io/src/output/results_writer.rs:507` `crates/cobre-io/src/test_support.rs::read_first_batch`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-909 · probe · test-bloat-09**

The count-named schema test and the public doctest both assert a floor below the produced export count, and the name-pinning test covers only part of the export set

- **Anchors:** `crates/cobre-io/src/schema.rs::test_generate_schemas_returns_expected_count` `crates/cobre-io/src/schema.rs:84` `crates/cobre-io/src/schema.rs::test_all_expected_schema_filenames_present` `crates/cobre-io/src/schema.rs:52`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-910 · probe · test-bloat-10**

Per-schema field-count tests assert only a length while sibling tests in the same module pin the column-name vector

- **Anchors:** `crates/cobre-io/src/output/schemas.rs::thermals_schema_field_count` `crates/cobre-io/src/output/schemas.rs::rank_timing_schema_field_count` `crates/cobre-io/src/output/schemas.rs::costs_schema_field_count_and_names` `crates/cobre-io/src/output/schemas.rs::field_names`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-911 · probe · test-bloat-11**

Four inline fixture builders restate helpers the crate test-support module already provides, two of them byte-identical

- **Anchors:** `crates/cobre-io/src/validation/referential.rs::make_pumping` `crates/cobre-io/src/validation/semantic/pumping.rs::make_pumping` `crates/cobre-io/src/validation/scalar_parameters.rs::zero_hydro_penalties` `crates/cobre-io/src/output/dictionary.rs::hydro_penalties_zero` `crates/cobre-io/src/test_support.rs::penalties_all`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-912 · probe · test-bloat-12**

The broadcast payload-size canary asserts a loose upper bound against a single-bus System

- **Anchors:** `crates/cobre-io/src/broadcast.rs::test_serialized_size_reasonable` `crates/cobre-io/src/broadcast.rs:399`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-913 · probe · test-bloat-13**

Every SS1.1 fixture builder in cobre-solver is the same body modulo indentation, so the seed's in-src versus in-binary split is a feature-reachability boundary and not a code difference

- **Anchors:** `crates/cobre-solver/src/types.rs:736` `crates/cobre-solver/src/freeze.rs:253` `crates/cobre-solver/src/freeze.rs:262` `crates/cobre-solver/src/backends/clp/tests.rs:20` `crates/cobre-solver/src/backends/clp/tests.rs:52` `crates/cobre-solver/src/backends/highs/tests.rs:19` `crates/cobre-solver/src/backends/highs/tests.rs:45` `crates/cobre-solver/tests/conformance.rs:34` `crates/cobre-solver/tests/conformance.rs:57` `crates/cobre-solver/tests/clp_determinism.rs:33` `crates/cobre-solver/tests/ffi_set_basis_non_alien_smoke.rs:29` `crates/cobre-solver/tests/clp_only_smoke.rs:24` `crates/cobre-solver/tests/conformance.rs:4` `crates/cobre-solver/tests/ffi_set_basis_non_alien_smoke.rs:27`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-914 · probe · test-bloat-14**

Three solver probes name the fact they investigate in a doc comment and print it instead of asserting it, while a sibling in the same file asserts exactly that fact

- **Anchors:** `crates/cobre-solver/src/backends/highs/tests.rs:787` `crates/cobre-solver/src/backends/highs/tests.rs:792` `crates/cobre-solver/src/backends/highs/tests.rs:796` `crates/cobre-solver/src/backends/highs/tests.rs:844` `crates/cobre-solver/src/backends/highs/tests.rs:846` `crates/cobre-solver/src/backends/highs/tests.rs:851` `crates/cobre-solver/src/backends/highs/tests.rs:856` `crates/cobre-solver/src/backends/highs/tests.rs:858` `crates/cobre-solver/src/backends/highs/tests.rs:865` `crates/cobre-solver/src/backends/highs/tests.rs:872` `crates/cobre-solver/src/backends/highs/tests.rs:1030` `crates/cobre-solver/src/backends/highs/tests.rs:1044` `crates/cobre-solver/src/backends/highs/tests.rs:1065` `crates/cobre-solver/src/backends/highs/tests.rs:339` `crates/cobre-solver/src/backends/highs/tests.rs:342` `crates/cobre-solver/src/backends/highs/tests.rs:348` `crates/cobre-solver/src/backends/highs/tests.rs:359` `crates/cobre-solver/tests/_q1_sign_convention_probe.rs:1` `crates/cobre-solver/tests/_q1_sign_convention_probe.rs:107` `crates/cobre-solver/tests/_q1_sign_convention_probe.rs:111` `crates/cobre-solver/tests/_q1_sign_convention_probe.rs:126` `crates/cobre-solver/tests/_clp_sign_convention_probe.rs:120`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-915 · probe · test-bloat-15**

The clp-only smoke binary's module-doc guarantee is already satisfied by two sibling clp-gated binaries in the same feature combination

- **Anchors:** `crates/cobre-solver/tests/clp_only_smoke.rs:3` `crates/cobre-solver/tests/clp_only_smoke.rs:8` `crates/cobre-solver/tests/clp_only_smoke.rs:24` `crates/cobre-solver/tests/clp_only_smoke.rs:48` `crates/cobre-solver/tests/clp_only_smoke.rs:56` `crates/cobre-solver/tests/conformance.rs:26` `crates/cobre-solver/tests/conformance.rs:1448` `crates/cobre-solver/tests/conformance.rs:1450` `crates/cobre-solver/tests/clp_determinism.rs:16`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-916 · probe · test-bloat-16**

Both conformance fixture-data tests assert a same-file struct-literal builder against a transcription of its own literals, with no oracle outside the file

- **Anchors:** `crates/cobre-solver/tests/conformance.rs:34` `crates/cobre-solver/tests/conformance.rs:57` `crates/cobre-solver/tests/conformance.rs:138` `crates/cobre-solver/tests/conformance.rs:153` `crates/cobre-solver/tests/conformance.rs:161` `crates/cobre-solver/tests/conformance.rs:72` `crates/cobre-solver/tests/conformance.rs:176`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-917 · probe · test-bloat-17**

The profile and retry composition binary gates its whole test module on the non-default test-support feature while importing nothing that feature gates

- **Anchors:** `crates/cobre-solver/tests/profile_retry_composition.rs:9` `crates/cobre-solver/tests/profile_retry_composition.rs:10` `crates/cobre-solver/tests/profile_retry_composition.rs:21` `crates/cobre-solver/tests/profile_retry_composition.rs:23` `crates/cobre-solver/tests/profile_retry_composition.rs:27` `crates/cobre-solver/tests/profile_retry_composition.rs:431` `crates/cobre-solver/tests/_clp_sign_convention_probe.rs:10` `crates/cobre-solver/tests/_clp_sign_convention_probe.rs:21`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-918 · probe · test-bloat-18**

The freeze module keeps most of its lines in one inline test module while both backend modules of the same crate use extracted siblings

- **Anchors:** `crates/cobre-solver/src/freeze.rs:244` `crates/cobre-solver/src/freeze.rs:245` `crates/cobre-solver/src/freeze.rs:253` `crates/cobre-solver/src/freeze.rs:855` `crates/cobre-solver/src/backends/highs/mod.rs:20` `crates/cobre-solver/src/backends/clp/mod.rs:14` `crates/cobre-solver/src/types.rs:546`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-919 · probe · test-bloat-19**

The five-function undated fixture prelude is re-declared in three inline #[cfg(test)] modules while tests/common/mod.rs already exports every member

- **Anchors:** `crates/cobre-stochastic/src/context.rs::make_stage` `crates/cobre-stochastic/src/provenance.rs::make_stage` `crates/cobre-stochastic/src/sampling/mod.rs::make_bus` `crates/cobre-stochastic/src/tree/generate.rs::make_stage` `crates/cobre-stochastic/tests/common/mod.rs::deficit_bus` `crates/cobre-stochastic/src/lib.rs:32` `crates/cobre-stochastic/src/par/validation.rs::make_model`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-920 · probe · test-bloat-20**

The dated make_hydro preset is byte-identical across four inline test modules and dated_stage across two, none of them reaching the crate's test_support surface

- **Anchors:** `crates/cobre-stochastic/src/seeds.rs::make_hydro` `crates/cobre-stochastic/src/par/lag_transition.rs::make_hydro` `crates/cobre-stochastic/src/sampling/external.rs::make_hydro` `crates/cobre-stochastic/src/sampling/historical.rs::make_hydro` `crates/cobre-stochastic/src/sampling/external.rs::dated_stage` `crates/cobre-stochastic/src/sampling/historical.rs::dated_stage`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-921 · probe · test-bloat-21**

par/precompute.rs and par/evaluate.rs carry a byte-identical 31-line make_stage plus make_model block, not the dummy_date-wrapper variant the register records

- **Anchors:** `crates/cobre-stochastic/src/par/precompute.rs::make_stage` `crates/cobre-stochastic/src/par/precompute.rs::make_model` `crates/cobre-stochastic/src/par/evaluate.rs::make_stage` `crates/cobre-stochastic/src/par/evaluate.rs::make_model` `crates/cobre-stochastic/src/par/validation.rs::make_model`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-922 · probe · test-bloat-22**

simulate_two_season_par2 is duplicated across the crate's only two extracted sibling test files, identical but for one local variable name

- **Anchors:** `crates/cobre-stochastic/src/par/fitting/tests.rs::simulate_two_season_par2` `crates/cobre-stochastic/src/par/fitting/estimation/tests.rs::simulate_two_season_par2`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-923 · probe · test-bloat-23**

pop_mean_std_ann is a rename-only copy of pop_mean_std inside the same sibling test file

- **Anchors:** `crates/cobre-stochastic/src/par/fitting/tests.rs::pop_mean_std` `crates/cobre-stochastic/src/par/fitting/tests.rs::pop_mean_std_ann`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-924 · probe · test-bloat-24**

Seven per-binary stage wrappers across four integration binaries re-declare common::method_stage, and the marginal-moment assertion block repeats in the QMC binaries with no shared counterpart

- **Anchors:** `crates/cobre-stochastic/tests/halton_integration.rs::make_stage_halton_with_block` `crates/cobre-stochastic/tests/sobol_integration.rs::make_stage_sobol_with_block` `crates/cobre-stochastic/tests/lhs_integration.rs::make_stage_lhs` `crates/cobre-stochastic/tests/halton_integration.rs::make_stage_halton` `crates/cobre-stochastic/tests/sobol_integration.rs::make_stage_sobol` `crates/cobre-stochastic/tests/lhs_integration.rs::make_stage_lhs_no_block` `crates/cobre-stochastic/tests/saa_golden_value.rs::make_stage` `crates/cobre-stochastic/tests/halton_integration.rs:207` `crates/cobre-stochastic/tests/common/mod.rs::method_stage`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-925 · probe · test-bloat-25**

The extracted-sibling homing convention is confined to par/fitting/, while inline test tails larger than the smaller extracted sibling stay inline, two of them in the same par/ directory

- **Anchors:** `crates/cobre-stochastic/src/par/fitting/mod.rs:66` `crates/cobre-stochastic/src/par/fitting/estimation.rs:1298` `crates/cobre-stochastic/src/par/fitting/estimation/tests.rs:1` `crates/cobre-stochastic/src/sampling/external.rs:871` `crates/cobre-stochastic/src/sampling/historical.rs:569` `crates/cobre-stochastic/src/tree/generate.rs:382` `crates/cobre-stochastic/src/sampling/mod.rs:764` `crates/cobre-stochastic/src/context.rs:789` `crates/cobre-stochastic/src/par/precompute.rs:589` `crates/cobre-stochastic/src/par/evaluate.rs:467` `docs/design/testing-architecture.md:146`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-926 · probe · test-bloat-26**

test_season_occurrence_matches_pre_refactor_helper_sequence recomputes the delegation it checks from the same three helpers, so it pins no expected value and its single-element stage array hides the anchor-stage rule

- **Anchors:** `crates/cobre-stochastic/src/season_cast/mod.rs::test_season_occurrence_matches_pre_refactor_helper_sequence` `crates/cobre-stochastic/src/season_cast/mod.rs::season_occurrence` `.claude/rules/testing.md:40`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-927 · probe · test-bloat-27**

The crate's in-binary genericity gate greps a single token over its own src while the canonical workspace script it duplicates already scans that directory under the full contract

- **Anchors:** `crates/cobre-stochastic/tests/reproducibility.rs::infrastructure_genericity_no_sddp_references` `crates/cobre-stochastic/tests/reproducibility.rs:4` `CLAUDE.md:39`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-928 · probe · test-bloat-28**

Two owners for the same public LocalBackend Communicator contract: body-identical assertions duplicated between the inline module in src/local.rs and tests/local_conformance.rs

- **Anchors:** `crates/cobre-comm/src/local.rs:292` `crates/cobre-comm/tests/local_conformance.rs:30` `crates/cobre-comm/src/local.rs:379` `crates/cobre-comm/tests/local_conformance.rs:44` `crates/cobre-comm/src/local.rs:406` `crates/cobre-comm/tests/local_conformance.rs:198` `crates/cobre-comm/src/local.rs:453` `crates/cobre-comm/tests/local_conformance.rs:290` `crates/cobre-comm/src/local.rs:444` `crates/cobre-comm/tests/local_conformance.rs:94` `crates/cobre-comm/src/factory.rs:303` `crates/cobre-comm/tests/factory_tests.rs:71` `docs/design/testing-architecture.md:253`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-929 · probe · test-bloat-29**

Send + Sync is asserted in cobre-comm by two different mechanisms: one production const fn versus a family of test-local assert_send_sync redefinitions, one of which is duplicated across the inline and the integration home

- **Anchors:** `crates/cobre-comm/src/factory.rs:74` `crates/cobre-comm/src/factory.rs:420` `crates/cobre-comm/src/ferrompi.rs:489` `crates/cobre-comm/src/ferrompi.rs:524` `crates/cobre-comm/src/local.rs:483` `crates/cobre-comm/src/local.rs:581` `crates/cobre-comm/src/traits.rs:456` `crates/cobre-comm/tests/factory_tests.rs:90` `crates/cobre-comm/src/ferrompi.rs:66`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-930 · probe · test-bloat-30**

cobre-comm declares no test-support feature and mentions the comm harness pair nowhere, while the yardstick states in the present tense that StubComm and Rank0Of2 live in cobre-comm's test-support surface

- **Anchors:** `docs/design/testing-architecture.md:524` `docs/design/testing-architecture.md:415` `crates/cobre-comm/Cargo.toml:20` `crates/cobre-comm/src/traits.rs::Communicator`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-931 · probe · test-bloat-31**

Bucket-topology fixture prelude is declared three times across two src modules and their sibling test file

- **Anchors:** `crates/cobre-sddp/src/setup/bucket_topology.rs:417` `crates/cobre-sddp/src/setup/mod.rs:2981` `crates/cobre-sddp/src/setup/tests.rs:9957` `crates/cobre-sddp/src/test_support.rs:770`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-932 · probe · test-bloat-32**

Eleven distinct helper names build the same zero hydro-penalty fixture crate-wide while cobre-core already exports a test-support constructor

- **Anchors:** `crates/cobre-sddp/src/setup/tests.rs:200` `crates/cobre-sddp/src/setup/tests.rs:7500` `crates/cobre-sddp/src/test_support.rs::fill_consistent_basis`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-933 · probe · test-bloat-33**

Right-boundary ring quartet repeats md5-identical penalties and bounds helpers in four solver-linking integration binaries

- **Anchors:** `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs:164` `crates/cobre-sddp/tests/right_boundary_output.rs:167` `crates/cobre-sddp/tests/right_boundary_pricing.rs:191` `crates/cobre-sddp/tests/right_boundary_validation.rs:200` `crates/cobre-sddp/tests/common/mod.rs:27`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-934 · probe · test-bloat-34**

anticipated_core.rs re-declares its system builder per test group and repeats byte-identical penalty and bound helpers

- **Anchors:** `crates/cobre-sddp/tests/anticipated_core.rs::build_system` `crates/cobre-sddp/tests/anticipated_core.rs:1631` `crates/cobre-sddp/tests/anticipated_scenarios.rs::default_hydro_penalties` `crates/cobre-sddp/tests/hydro_sim.rs:1408`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-935 · probe · test-bloat-35**

Seven integration binaries re-declare state_layout_for and six document it as a test_support symbol that does not exist

- **Anchors:** `crates/cobre-sddp/tests/conformance.rs:797` `crates/cobre-sddp/tests/integration.rs:66` `crates/cobre-sddp/tests/inflow_nonnegativity.rs:74` `crates/cobre-sddp/tests/parity.rs:816` `crates/cobre-sddp/src/test_support.rs:657` `crates/cobre-sddp/tests/mpi_wire.rs:1343`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-936 · probe · test-bloat-36**

A cfg(test) helper sits inside the production region of the stochastic noise module though every caller is in that file's own test module

- **Anchors:** `crates/cobre-sddp/src/stochastic/noise.rs:176` `crates/cobre-sddp/src/stochastic/noise.rs:561` `crates/cobre-sddp/src/stochastic/noise.rs:1347`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-937 · probe · test-bloat-37**

The largest inline test module in the LP-builder entries file spans two thirds of the file with its helpers scattered across it

- **Anchors:** `crates/cobre-sddp/src/lp/builder/entries.rs:3518` `crates/cobre-sddp/src/lp/builder/entries.rs:5948` `crates/cobre-sddp/src/lp/builder/entries.rs:7912` `crates/cobre-sddp/src/lp/builder/entries.rs:8329` `crates/cobre-sddp/src/lp/builder/columns.rs:1305`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Re-raise-of:** mirror `Mega-file / inline-test-giant asymmetry` — a dup-of merge into the open mirror item (prior-register.md), never a fresh TD id

**CD-938 · probe · test-bloat-38**

Two fixture builders in the lag-aware PAR coefficient probe share all but a handful of lines

- **Anchors:** `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs:101` `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs:347` `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs:728`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-939 · probe · test-bloat-39**

template_integration.rs clones near-identical system builders across its anticipated-thermal and one-bus fixture families

- **Anchors:** `crates/cobre-sddp/tests/template_integration.rs:3069` `crates/cobre-sddp/tests/template_integration.rs:3247` `crates/cobre-sddp/tests/template_integration.rs:3430` `crates/cobre-sddp/tests/template_integration.rs:2110` `crates/cobre-sddp/tests/template_integration.rs:2294` `crates/cobre-sddp/tests/common/builders.rs:1`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-940 · probe · test-bloat-40**

lp_builder.rs declares the same deterministic-examples path helper three times under two signatures

- **Anchors:** `crates/cobre-sddp/tests/lp_builder.rs:65` `crates/cobre-sddp/tests/lp_builder.rs:477` `crates/cobre-sddp/tests/lp_builder.rs:573` `crates/cobre-sddp/tests/common/mod.rs:27`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-941 · probe · test-bloat-41**

A two-rank stub communicator is declared identically in two MPI integration binaries beside the strongest copy in the cut-sync unit module

- **Anchors:** `crates/cobre-sddp/tests/test_mpi_allgatherv_nonuniform_workers.rs:22` `crates/cobre-sddp/tests/test_mpi_sync_cuts_invariant.rs:25` `crates/cobre-sddp/src/cut/cut_sync.rs:1672` `crates/cobre-sddp/src/training/backward/tests.rs:5443`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-942 · probe · test-bloat-42**

cut_basis.rs repeats its case-directory and stage-date helpers in every inner module, with one copy drifting to unwrap

- **Anchors:** `crates/cobre-sddp/tests/cut_basis.rs:41` `crates/cobre-sddp/tests/cut_basis.rs:917` `crates/cobre-sddp/tests/cut_basis.rs:1580` `crates/cobre-sddp/tests/cut_basis.rs:23`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-943 · probe · test-bloat-43**

The shared tests/common stub communicator and its rank-zero variant are byte-identical to copies inside two src test modules

- **Anchors:** `crates/cobre-sddp/tests/common/mod.rs:32` `crates/cobre-sddp/tests/common/mod.rs:86` `crates/cobre-sddp/src/training/session/mod.rs:1654` `crates/cobre-sddp/src/training/session/mod.rs:2767` `crates/cobre-sddp/src/training/training/tests.rs:173` `crates/cobre-sddp/src/test_support.rs:3555`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-944 · probe · test-bloat-44**

Nine mock-solver doubles are declared across the crate's src test modules with the same inert configuration

- **Anchors:** `crates/cobre-sddp/src/simulation/pipeline/tests.rs:160` `crates/cobre-sddp/src/training/backward_pass_state.rs:2298` `crates/cobre-sddp/src/training/backward_pass_state.rs:2363` `crates/cobre-sddp/src/training/lower_bound.rs:873` `crates/cobre-sddp/src/workspace/workspace.rs:1155` `crates/cobre-sddp/src/test_support.rs:1859`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-945 · probe · test-bloat-45**

The K1 byte-stability probe prints its verdict without asserting and checks fixture names against literals in its own file

- **Anchors:** `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs:232` `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs:218` `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs:146` `crates/cobre-sddp/tests/anticipated_scenarios.rs:1430` `crates/cobre-sddp/tests/anticipated_scenarios.rs:1540`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-946 · probe · test-bloat-46**

The simulation-pipeline binary's stub communicator is documented single-rank while one construction site gives it a two-rank world

- **Anchors:** `crates/cobre-sddp/tests/simulation_pipeline_integration.rs:77` `crates/cobre-sddp/tests/simulation_pipeline_integration.rs:78` `crates/cobre-sddp/tests/simulation_pipeline_integration.rs:1173` `crates/cobre-sddp/tests/simulation_pipeline_integration.rs:51`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-947 · probe · test-bloat-47**

cobre-cli integration binaries share no harness home: the command builder, the repo-root case-dir resolver and the parquet fixture block are each copied verbatim across run-path binaries

- **Anchors:** `crates/cobre-cli/tests/cli_run_anticipated.rs::read_thermals_parquet` `crates/cobre-cli/tests/cli_run_anticipated_k2.rs::read_thermals_parquet` `crates/cobre-cli/tests/setup_timings_metadata.rs::d01_case_dir` `crates/cobre-cli/tests/output_metadata_active_backend.rs::d01_case_dir` `crates/cobre-cli/tests/cli_run_evaporation.rs::d08_case_dir` `crates/cobre-cli/tests/cli_run_generic_echo.rs::case_dir` `crates/cobre-cli/tests/cli_run.rs::cobre`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-948 · probe · test-bloat-48**

the thread-count unit tests sit in the run module's test module while their subject and its only production caller both live in the setup module, which already carries its own test module

- **Anchors:** `crates/cobre-cli/src/commands/run/mod.rs::test_resolve_thread_count_cli_value` `crates/cobre-cli/src/commands/run/mod.rs::test_resolve_thread_count_default` `crates/cobre-cli/src/commands/run/mod.rs:331` `crates/cobre-cli/src/commands/run/setup.rs::resolve_thread_count` `crates/cobre-cli/src/commands/run/setup.rs:602`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-949 · probe · test-bloat-49**

the validate command's test module renders the report through its own formatter whose header and warning-entry shapes no production path can emit, leaving the shipped renders unasserted

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::format_report_to_string` `crates/cobre-cli/src/commands/validate.rs::format_entry` `crates/cobre-cli/src/commands/validate.rs:532` `crates/cobre-cli/src/commands/validate.rs:436` `crates/cobre-cli/src/commands/validate.rs:446` `crates/cobre-cli/tests/cli_validate.rs:733` `crates/cobre-cli/tests/cli_validate.rs:782`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-950 · probe · test-bloat-50**

the summary printers are asserted only through cfg(test) twins, so the header wrapper, labels, padding and line order of the shipped stderr renders carry no assertion

- **Anchors:** `crates/cobre-cli/src/summary.rs::format_hydro_model_summary_string` `crates/cobre-cli/src/summary.rs::format_setup_summary_string` `crates/cobre-cli/src/summary.rs::format_provenance_summary_string` `crates/cobre-cli/src/summary.rs::format_boundary_summary_string` `crates/cobre-cli/src/summary.rs:1788` `crates/cobre-cli/src/summary.rs:2248` `crates/cobre-cli/src/summary.rs::print_hydro_model_summary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-951 · probe · test-bloat-51**

with no shared fixture home the penalties fixture const is byte-identical across several cobre-cli test binaries and the whole valid-case builder is verbatim identical between the run and colour binaries

- **Anchors:** `crates/cobre-cli/tests/cli_color.rs::make_valid_case` `crates/cobre-cli/tests/cli_run.rs::make_valid_case` `crates/cobre-cli/tests/cli_validate.rs::make_valid_case` `crates/cobre-cli/tests/cli_color.rs::write_file` `crates/cobre-cli/tests/cli_run_anticipated.rs:35` `crates/cobre-cli/tests/cli_run_anticipated_k2.rs:31`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-952 · probe · test-bloat-52**

the colour suite's never-flag case asserts an escape-free banner in an invocation that suppresses the banner, so it cannot fail on the leak it names

- **Anchors:** `crates/cobre-cli/tests/cli_color.rs::color_never_flag_suppresses_ansi_in_banner` `crates/cobre-cli/tests/cli_color.rs:149` `crates/cobre-cli/tests/cli_color.rs:156` `crates/cobre-cli/src/commands/run/setup.rs:209` `crates/cobre-cli/src/banner.rs:70` `crates/cobre-cli/src/commands/init.rs:148`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-953 · probe · test-bloat-53**

progress spawn tests that assert a strict subset of a retained sibling, plus a populated average-LP-time branch and a rendered bar message that no progress test ever reaches

- **Anchors:** `crates/cobre-cli/src/progress.rs:534` `crates/cobre-cli/src/progress.rs:673` `crates/cobre-cli/src/progress.rs:907` `crates/cobre-cli/src/progress.rs:1011` `crates/cobre-cli/src/progress.rs:835` `crates/cobre-cli/src/progress.rs:987` `crates/cobre-cli/src/progress.rs::fmt_avg_lp_time` `crates/cobre-cli/src/progress.rs::make_simulation_progress`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-954 · probe · test-bloat-54**

exit-code-only validate tests duplicate the fixture build and spawn of named siblings whose predicates are strict supersets on identical argv

- **Anchors:** `crates/cobre-cli/tests/cli_validate.rs::valid_case_exits_0` `crates/cobre-cli/tests/cli_validate.rs::missing_buses_json_exits_1` `crates/cobre-cli/tests/cli_validate.rs::nonexistent_path_exits_2` `crates/cobre-cli/tests/cli_validate.rs::fpha_hydro_without_production_models_json_fails_validate` `crates/cobre-cli/tests/cli_validate.rs:109` `crates/cobre-cli/tests/cli_validate.rs:132` `crates/cobre-cli/tests/cli_validate.rs:257` `crates/cobre-cli/tests/cli_validate.rs:802`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-955 · probe · test-bloat-55**

the 1dtoy template test freezes its file count as a hand-maintained literal beside the table it counts, and no test walks the canonical source directory

- **Anchors:** `crates/cobre-cli/src/templates.rs::test_1dtoy_template_has_files` `crates/cobre-cli/src/templates.rs:179` `crates/cobre-cli/src/templates.rs::test_1dtoy_embedded_files_match_canonical_source` `crates/cobre-cli/src/templates.rs:208` `crates/cobre-cli/src/templates.rs::DTOY1_FILES` `crates/cobre-cli/tests/cli_schema.rs::test_schema_export_writes_files`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-956 · probe · test-bloat-56**

the progress module's copy of the scientific-notation formatter is byte-identical to the summary module's and is excluded from its own test module's imports, so a divergence in it ships green

- **Anchors:** `crates/cobre-cli/src/progress.rs:539` `crates/cobre-cli/src/progress.rs::fmt_sci` `crates/cobre-cli/src/summary.rs::fmt_sci` `crates/cobre-cli/src/summary.rs:1121` `crates/cobre-cli/src/summary.rs:1181` `crates/cobre-cli/src/summary.rs::training_summary_lines`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-957 · probe · test-bloat-57**

cobre-cli activates cobre-core's test-support feature in dev-dependencies while no code in the crate references the gated surface

- **Anchors:** `crates/cobre-cli/Cargo.toml:62` `docs/design/testing-architecture.md:425`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-958 · probe · test-bloat-58**

Four read-only Policy assertions in test_study.py each train their own study instead of sharing one trained policy

- **Anchors:** `crates/cobre-python/tests/test_study.py::test_policy_evaluate_matches_cut_matrix_max` `crates/cobre-python/tests/test_study.py::test_policy_cut_matrix_shapes_and_dtype` `crates/cobre-python/tests/test_study.py::test_policy_evaluate_stage_out_of_range_raises_indexerror` `crates/cobre-python/tests/test_study.py::test_policy_evaluate_bad_state_length_raises_valueerror`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-959 · probe · test-bloat-59**

The inline Rust test module in run.rs solves the same 1dtoy deck from three separate tests with a byte-identical call

- **Anchors:** `crates/cobre-python/src/run.rs::python_run_1dtoy_metadata_matches_cli_golden_values` `crates/cobre-python/src/run.rs::reconstruct_policy_from_checkpoint_roundtrips_for_1dtoy` `crates/cobre-python/src/run.rs:2407`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-960 · probe · test-bloat-60**

The case-with-simulation builder and the cell-level float comparator are copied across parity modules with divergent docstrings

- **Anchors:** `crates/cobre-python/tests/test_contract_output_parity.py::_make_case_with_simulation` `crates/cobre-python/tests/test_parity_hydros.py::_make_case_with_simulation` `crates/cobre-python/tests/test_anticipated_lanes_output_parity.py::_lane_values_equal` `crates/cobre-python/tests/test_anticipated_output_parity.py::_anticipated_values_equal` `crates/cobre-python/tests/test_cli_python_file_set_parity.py::_relative_files`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-961 · probe · test-bloat-61**

test_contract_output_parity.py rebuilds the same derived contract deck in every test instead of sharing one prepared case directory

- **Anchors:** `crates/cobre-python/tests/test_contract_output_parity.py:164` `crates/cobre-python/tests/test_contract_output_parity.py:221` `crates/cobre-python/tests/test_contract_output_parity.py:258` `crates/cobre-python/tests/test_contract_output_parity.py::test_run_via_study_emits_contract_output`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-962 · probe · test-bloat-62**

The py_to_json_value converter carries no Rust oracle while its sibling modules do, and its inline rationale names a build the bindings CI step does not use

- **Anchors:** `crates/cobre-python/src/convert.rs::py_to_json_value` `crates/cobre-python/src/convert.rs:104` `crates/cobre-python/src/errors.rs:393`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-963 · probe · test-bloat-63**

test_boundary_load.py borrows a private helper from another test module while every module re-derives the repository root

- **Anchors:** `crates/cobre-python/tests/test_boundary_load.py:24` `crates/cobre-python/tests/test_boundary_load.py:26` `crates/cobre-python/tests/test_policy_load_validation.py::_copy_case_with_renamed_hydro` `crates/cobre-python/tests/conftest.py:20`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-964 · probe · test-bloat-64**

test_boundary_load.py re-solves the same valid deck once per test and test_results.py trains again beside its own module fixture

- **Anchors:** `crates/cobre-python/tests/test_boundary_load.py:59` `crates/cobre-python/tests/test_boundary_load.py:98` `crates/cobre-python/tests/test_boundary_load.py:124` `crates/cobre-python/tests/test_results.py:342`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-965 · probe · test-bloat-65**

The bindings crate declares and consumes no test-support feature and its dev-dependency table names no cobre-sddp

- **Anchors:** `crates/cobre-python/Cargo.toml:34` `docs/design/testing-architecture.md:425`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Re-raise-of:** mirror `Python-binding Rust tests invisible to CI` — a dup-of merge into the open mirror item (prior-register.md), never a fresh TD id

**CD-966 · probe · test-bloat-66**

The shared CLI-helper module's docstring freezes a stale importer count that no guard pins

- **Anchors:** `crates/cobre-python/tests/_cobre_cli.py:4` `crates/cobre-python/tests/conftest.py:14`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-967 · probe · test-bloat-67**

slow-tests rides inside the per-PR feature set, so the declared long-running tier separates nothing in CI

- **Anchors:** `.github/workflows/ci.yml:32` `.github/workflows/ci.yml:114` `.github/workflows/ci.yml:233` `.github/workflows/ci.yml:497`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-968 · probe · test-bloat-68**

The primary-backend test job runs a solver-FFI suite in-process while the same file argues in prose that an FFI suite needs per-process isolation to be a real gate

- **Anchors:** `.github/workflows/ci.yml:114` `.github/workflows/ci.yml:228` `.github/workflows/ci.yml:233`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-969 · probe · test-bloat-69**

No .config/nextest.toml exists, so every nextest invocation runs on tool defaults and the three call sites can drift apart

- **Anchors:** `.github/workflows/ci.yml:233` `.github/workflows/invariance-shuffle.yml:34` `.github/workflows/invariance-shuffle.yml:55`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-970 · probe · test-bloat-70**

Doctests ride on one job's cargo test with no dedicated step, so they never compile under the secondary-backend feature combination

- **Anchors:** `.github/workflows/ci.yml:114` `.github/workflows/ci.yml:233`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-971 · probe · test-bloat-71**

The SLURM cluster's batch-scheduler and multi-node tiers run only on manual dispatch, so every automatic trigger exercises the launcher path alone

- **Anchors:** `.github/workflows/mpi-slurm.yml:4` `.github/workflows/mpi-slurm.yml:121`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-972 · probe · test-bloat-72**

The MPI workflow's path filter omits the module that owns the basis-cache broadcast byte layout and the cut-synchronisation module, so a change to either skips the cluster deck on a pull request

- **Anchors:** `.github/workflows/mpi-slurm.yml:7` `.github/workflows/mpi-slurm.yml:18`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-973 · probe · architecture-00**

The test-support surface has two activation regimes: cobre-core, cobre-io and cobre-stochastic pin the feature with a self dev-dependency while cobre-sddp and cobre-solver leave it to the command line

- **Anchors:** `crates/cobre-core/Cargo.toml:40` `crates/cobre-io/Cargo.toml:54` `crates/cobre-stochastic/Cargo.toml:42` `crates/cobre-sddp/Cargo.toml:74` `crates/cobre-solver/Cargo.toml:51` `crates/cobre-sddp/src/test_support.rs:9`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-974 · probe · architecture-01**

The yardstick's §2.2 map says every crate other than cobre-sddp constructs fixtures per-file, but cobre-stochastic and cobre-io each carry a shared multi-binary harness module at the pin

- **Anchors:** `docs/design/testing-architecture.md:76` `crates/cobre-stochastic/tests/common/mod.rs:13` `crates/cobre-io/tests/helpers/mod.rs:9` `docs/design/testing-architecture.md:417`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-975 · probe · architecture-02**

One shared-harness role carries two directory names and three compositions, and the differently-named cobre-io harness is invisible to a migration that greps for tests/common

- **Anchors:** `crates/cobre-io/tests/helpers/mod.rs:2` `crates/cobre-stochastic/tests/common/mod.rs:6` `crates/cobre-sddp/tests/common/mod.rs:32` `crates/cobre-sddp/tests/common/mod.rs:25` `docs/design/testing-architecture.md:417`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-976 · probe · over-engineering-00**

The test-support fixture surface carries a public GeometryDims builder, eq_with_anticipated, that no caller in the workspace reaches

- **Anchors:** `crates/cobre-sddp/src/test_support.rs::eq_with_anticipated` `crates/cobre-sddp/src/test_support.rs::eq`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-977 · probe · over-engineering-01**

The cobre-sddp tests/common/ aggregator declares every submodule to every consumer binary, so a single-consumer helper module compiles into all of them behind a blanket dead_code allow

- **Anchors:** `crates/cobre-sddp/tests/common/mod.rs:25` `crates/cobre-sddp/tests/common/mod.rs:6` `crates/cobre-sddp/tests/common/anticipated_structural_assertions.rs::assert_training_converged_structurally` `crates/cobre-sddp/tests/permute_helpers.rs:1`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-978 · probe · performance-00**

The whole workspace test surface is linked in three separate pull-request jobs, one per backend-and-profile combination, and no two of them can share a link artifact

- **Anchors:** `.github/workflows/ci.yml:114` `.github/workflows/ci.yml:223` `.github/workflows/ci.yml:233` `.github/workflows/ci.yml:497` `.github/workflows/ci.yml:209` `.github/workflows/ci.yml:495` `crates/cobre-sddp/Cargo.toml:59`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-979 · probe · performance-01**

cobre-sddp concentrates most of the workspace's solver-linking integration binaries, and the consolidation shape the §5.1 target asks for is already proven inside that same test directory

- **Anchors:** `crates/cobre-sddp/tests/template_integration.rs:4400` `crates/cobre-sddp/tests/template_integration.rs:4421` `crates/cobre-sddp/Cargo.toml:59` `docs/design/testing-architecture.md:353`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-980 · probe · performance-02**

The CLP job links the criterion bench executables on every pull request although no workflow at the pin runs a benchmark, while the same job's non-linking passes already compile those bodies

- **Anchors:** `crates/cobre-sddp/Cargo.toml:92` `crates/cobre-sddp/Cargo.toml:108` `crates/cobre-sddp/Cargo.toml:59` `.github/workflows/ci.yml:223` `.github/workflows/ci.yml:219` `.github/workflows/ci.yml:221`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

