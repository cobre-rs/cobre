## INGEST ANCHOR PROBE — solver-comm (2026-09, baseline)

Register-shaped stub: one block per candidate, every anchor of every candidate rendered so
`check-anchors.py --register <this file> --baseline 077dbe2c` resolves it with the register's
own parser and `git show <sha>:<path>` resolver. A declaration anchor is rendered `path::symbol`;
a field, manifest or call-site anchor (no `fn|struct|enum|trait|type|const|static|mod|impl`
declaration for the checker's DECL regex to match) is rendered `path:line` with the symbol
named in prose. Ingest ref `<lens>-<nn>` ↔ attacker ref `SC-<LENS>-<nnn>` (same nn).

### architecture-01 · SC-ARCH-001 · The hand-replicated per-crate `[lints]` tables have already drifted, and the failure mode is silent: a dropped `deny` cannot make CI red

**Anchors:** `crates/cobre-solver/Cargo.toml:41` (lints.clippy) `crates/cobre-solver/Cargo.toml:37` (lints.rust) `crates/cobre-comm/Cargo.toml:39` (lints.clippy) `crates/cobre-comm/Cargo.toml:35` (lints.rust)

### architecture-02 · SC-ARCH-002 · Superseded cut-sync public methods still stand beside the live single path that owns the collective partition contract

**Anchors:** `crates/cobre-sddp/src/cut/cut_sync.rs::sync_cuts` `crates/cobre-sddp/src/cut/cut_sync.rs::pack_local_records` `crates/cobre-sddp/src/cut/cut_sync.rs::sync_packed_records` `crates/cobre-sddp/src/cut/cut_sync.rs::sync_level_records`

### architecture-03 · SC-ARCH-003 · Seven production doc comments in cobre-solver carry caller-loop vocabulary that the genericity gate's pattern cannot see at all

**Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs:66` `crates/cobre-solver/src/backends/clp/retry.rs:74` `crates/cobre-solver/src/backends/highs/solver.rs:320` `crates/cobre-solver/src/backends/highs/solver.rs:477` `crates/cobre-solver/src/trait_def.rs:95` `crates/cobre-solver/src/trait_def.rs:207` `crates/cobre-solver/src/trait_def.rs:212`

### architecture-04 · SC-ARCH-004 · The CLP backend's module facade carries a `pub(crate)` re-export and a lint suppression whose only consumer is its own sibling test module, where the HiGHS facade carries neither

**Anchors:** `crates/cobre-solver/src/backends/clp/mod.rs:24` `crates/cobre-solver/src/backends/clp/mod.rs:25` (LADDER_RUNGS) `crates/cobre-solver/src/backends/clp/interface.rs:6` `crates/cobre-solver/src/backends/highs/mod.rs:23`

### architecture-05 · SC-ARCH-005 · The CLP per-element basis code space has four spellings and no owner, and the comment that forbids a symbolic definition rests on a premise the code contradicts

**Anchors:** `crates/cobre-solver/src/ffi/clp.rs:29` `crates/cobre-solver/src/basis_status.rs::to_clp_code` `crates/cobre-solver/src/basis_status.rs::from_clp_code` `crates/cobre-solver/src/backends/clp/solver.rs::CLP_BASIS_AT_LOWER` `crates/cobre-solver/src/backends/clp/solver.rs::CLP_BASIS_BASIC` `crates/cobre-solver/src/backends/clp/interface.rs:600`

### architecture-06 · SC-ARCH-006 · Part-I item 8 re-verified field by field: all five multistage fields are write-only in cobre-solver, but the handoff's collateral list undercounts the in-src fixture blocks and clips two assertion spans

**Anchors:** `crates/cobre-solver/src/types.rs::StageTemplate` `crates/cobre-solver/src/types.rs:270` (n_state) `crates/cobre-solver/src/types.rs:278` (n_transfer) `crates/cobre-solver/src/types.rs:287` (n_dual_relevant) `crates/cobre-solver/src/types.rs:290` (n_hydro) `crates/cobre-solver/src/types.rs:297` (max_par_order) `crates/cobre-solver/src/freeze.rs::freeze_rows_into_template` `crates/cobre-solver/src/backends/clp/tests.rs:33` `crates/cobre-solver/src/backends/highs/tests.rs:32` `crates/cobre-solver/tests/conformance.rs:153`

### architecture-07 · SC-ARCH-007 · The sealed `ffi` boundary is asymmetric under feature selection: `mod highs` is declared un-gated, so the HiGHS binding module compiles in the CLP-only configuration that CI builds and tests

**Anchors:** `crates/cobre-solver/src/ffi/mod.rs::highs` `crates/cobre-solver/src/ffi/mod.rs::clp` `crates/cobre-solver/src/basis_status.rs:9` `crates/cobre-solver/src/ffi/highs.rs:5`

### architecture-08 · SC-ARCH-008 · L0 vocabulary leakage in the private `FreezeScratch.cut_nz_per_col` field survives the genericity gate because `_` is a word character

**Anchors:** `crates/cobre-solver/src/freeze.rs:22` (cut_nz_per_col) `crates/cobre-solver/src/freeze.rs:137` `crates/cobre-solver/src/freeze.rs:141` `crates/cobre-solver/src/freeze.rs:188`

### architecture-09 · SC-ARCH-009 · Informational: the solver trait has no capability-query surface and exactly one backend compiles, so building the capability trait at this baseline would be a one-consumer abstraction

**Anchors:** `crates/cobre-solver/src/trait_def.rs::SolverInterface` `crates/cobre-solver/src/trait_def.rs::Profile` `crates/cobre-solver/src/lib.rs:44` `crates/cobre-solver/src/lib.rs:50`

### over-engineering-01 · SC-OE-001 · ExecutionTopology::is_homogeneous is a public predicate with zero consumers outside its own four unit tests

**Anchors:** `crates/cobre-comm/src/topology.rs::is_homogeneous`

### over-engineering-02 · SC-OE-002 · Superseded cut-sync public methods still exported beside their live replacement (E5 handoff, re-verified at this baseline)

**Anchors:** `crates/cobre-sddp/src/cut/cut_sync.rs::sync_cuts` `crates/cobre-sddp/src/cut/cut_sync.rs::pack_local_records` `crates/cobre-sddp/src/cut/cut_sync.rs::sync_packed_records`

### over-engineering-03 · SC-OE-003 · CLP hot-start snapshot lifecycle is built end to end (C++ shim, extern decls, safe wrapper, determinism harness) with no production consumer and no reserved-seam registration

**Anchors:** `crates/cobre-solver/src/backends/clp/solver.rs::mark_hot_start` `crates/cobre-solver/src/backends/clp/solver.rs::solve_from_hot_start`

### over-engineering-04 · SC-OE-004 · HiGHS defaults exist twice -- 12 of 14 HighsProfile field defaults hand-mirror the default_options() table, with the bit-for-bit claim stated in prose and pinned by nothing

**Anchors:** `crates/cobre-solver/src/backends/highs/config.rs::default` `crates/cobre-solver/src/backends/highs/config.rs::default_options` `crates/cobre-solver/src/backends/highs/config.rs:10`

### over-engineering-05 · SC-OE-005 · HiGHS retry ladder hand-copies the same tolerance-floor-and-set fragment four times where the CLP ladder is one declarative rung table

**Anchors:** `crates/cobre-solver/src/backends/highs/retry.rs::apply_retry_level_options` `crates/cobre-solver/src/backends/highs/retry.rs::apply_extended_retry_options` `crates/cobre-solver/src/backends/highs/retry.rs::apply_extended_retry_options` `crates/cobre-solver/src/backends/highs/retry.rs::apply_extended_retry_options`

### performance-01 · SC-PERF-001 · Partition helpers hand back freshly allocated owned vectors and the partition is recomputed from scratch at every collective

**Anchors:** `crates/cobre-comm/src/lib.rs::per_rank_counts` `crates/cobre-comm/src/lib.rs::prefix_displs`

### performance-02 · SC-PERF-002 · CLP bound writers push the entire retained bound array across the FFI on every call, while the sibling backend writes only the requested index subset

**Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs::set_row_bounds` `crates/cobre-solver/src/backends/clp/interface.rs::set_col_bounds`

### performance-03 · SC-PERF-003 · CLP add_rows allocates five fresh vectors and replaces three retained buffers wholesale on every append, contradicting the struct's own reusable-buffer contract

**Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs::add_rows` `crates/cobre-solver/src/backends/clp/interface.rs:302` (col_starts)

### performance-04 · SC-PERF-004 · CLP basis capture and cold reset cross the FFI boundary once per column and once per row, where the sibling backend moves the whole status vector in one crossing

**Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs::get_basis` `crates/cobre-solver/src/backends/clp/solver.rs::reset_cold_basis`

### test-bloat-01 · SC-TB-001 · Three #[test] functions restate a Send+Sync monomorphisation that a production const fn beside them already performs on every build

**Anchors:** `crates/cobre-comm/src/factory.rs:74` (_assert_comm_backend_send_sync) `crates/cobre-comm/src/factory.rs::test_comm_backend_send_sync` `crates/cobre-comm/src/ferrompi.rs::test_ferrompi_backend_send_sync` `crates/cobre-comm/tests/factory_tests.rs::test_ferrompi_backend_send_sync`

### test-bloat-02 · SC-TB-002 · cobre-comm tests the same public Communicator contracts twice: six name-identical collective tests exist both inline in src/local.rs and in tests/local_conformance.rs, plus renamed pairs across factory.rs and tests/factory_tests.rs

**Anchors:** `crates/cobre-comm/src/local.rs::test_local_allreduce_buffer_mismatch` `crates/cobre-comm/tests/local_conformance.rs::test_local_allreduce_buffer_mismatch` `crates/cobre-comm/src/local.rs::test_local_allgatherv_recv_too_small` `crates/cobre-comm/tests/local_conformance.rs::test_local_allgatherv_recv_too_small` `crates/cobre-comm/src/local.rs::test_local_broadcast_invalid_root` `crates/cobre-comm/tests/local_conformance.rs::test_local_broadcast_invalid_root` `crates/cobre-comm/src/local.rs::test_local_allreduce_identity_sum` `crates/cobre-comm/tests/local_conformance.rs::test_local_allreduce_identity_sum` `crates/cobre-comm/src/local.rs::test_local_allgatherv_with_offset` `crates/cobre-comm/tests/local_conformance.rs::test_local_allgatherv_with_displacement` `crates/cobre-comm/src/factory.rs::test_available_backends_contains_local` `crates/cobre-comm/tests/factory_tests.rs::test_available_backends_contains_local` `crates/cobre-comm/src/factory.rs::test_create_communicator_no_feature_local` `crates/cobre-comm/tests/factory_tests.rs::test_factory_no_feature_local`

### test-bloat-03 · SC-TB-003 · The SS1.1 stage-template fixture is declared eight times token-identically across src and tests, so every StageTemplate field change costs eight edits

**Anchors:** `crates/cobre-solver/src/lib.rs::test_support` `crates/cobre-solver/src/types.rs::make_fixture_stage_template` `crates/cobre-solver/src/freeze.rs::make_fixture_stage_template` `crates/cobre-solver/src/backends/clp/tests.rs::make_fixture_stage_template` `crates/cobre-solver/src/backends/highs/tests.rs::make_fixture_stage_template` `crates/cobre-solver/tests/conformance.rs::make_fixture_stage_template` `crates/cobre-solver/tests/clp_determinism.rs::make_fixture_stage_template` `crates/cobre-solver/tests/ffi_set_basis_non_alien_smoke.rs::make_fixture_stage_template` `crates/cobre-solver/tests/clp_only_smoke.rs::make_fixture_stage_template` `crates/cobre-solver/src/backends/clp/tests.rs::make_fixture_row_batch` `crates/cobre-solver/src/backends/highs/tests.rs::make_fixture_row_batch` `crates/cobre-solver/tests/conformance.rs::make_fixture_row_batch` `crates/cobre-solver/tests/clp_determinism.rs::make_fixture_row_batch`

### test-bloat-04 · SC-TB-004 · test_research_probe_limit_status_on_ss11_lp is a #[test] with no behavioural assertion; the fact it observes is already prose in the module comment directly above it

**Anchors:** `crates/cobre-solver/src/backends/highs/tests.rs::test_research_probe_limit_status_on_ss11_lp` `crates/cobre-solver/src/backends/highs/tests.rs::research_tests` `crates/cobre-solver/src/backends/highs/tests.rs:787`

### test-bloat-05 · SC-TB-005 · clp_only_smoke.rs is a whole test binary whose stated reason to exist is already met by conformance.rs, and its single test is a tolerance-only clone of the conformance one

**Anchors:** `crates/cobre-solver/tests/clp_only_smoke.rs::clp_only_load_model_and_solve` `crates/cobre-solver/tests/clp_only_smoke.rs:3` `crates/cobre-solver/tests/conformance.rs::test_solver_clp_load_model_and_solve` `crates/cobre-solver/tests/conformance.rs:26`

### test-bloat-06 · SC-TB-006 · The two fixture-contract tests in conformance.rs assert a test-local literal against its own transcription and re-encode the five StageTemplate fields Part-I item 8 sheds

**Anchors:** `crates/cobre-solver/tests/conformance.rs::test_fixture_stage_template_data` `crates/cobre-solver/tests/conformance.rs::test_fixture_row_batch_data` `crates/cobre-solver/tests/conformance.rs:153`

### Anchors outside the harness ANCHOR_RE extension set (resolved directly)

- performance-02 — `crates/cobre-solver/csrc/clp_wrapper.c` `cobre_clp_chg_bounds` L208: `git show 077dbe2c:crates/cobre-solver/csrc/clp_wrapper.c | grep -n cobre_clp_chg_bounds` → 208:static void cobre_clp_chg_bounds(
