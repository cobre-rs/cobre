# Test Inventory Hand-off — Architecture Unification

**Date:** 2026-04-18
**Branch:** `feat/architecture-unification` (after Epic 01 tickets 001–005)
**Inputs:** [`test-inventory.md`](test-inventory.md) · [`test-inventory-taxonomy.md`](test-inventory-taxonomy.md) · [`test-wall-time-baseline.md`](test-wall-time-baseline.md)

---

## 1. Inventory Summary

The committed inventory (`test-inventory.md`) covers 2,680 `#[test]` functions across six crates. The Phase 0b wall-time baseline measured a cold build-and-run at 175.16 s and an incremental median of 10.80 s (Cargo counts 3,676 test cases when fork tests and doc-tests are included). Of the 2,680 inventoried `#[test]` functions, **81 carry at least one deletion-relevant guard** from the taxonomy's Deletion-mapping rows for steps 3, 4, and 5 (Epics 03, 04, and 05 respectively). The remaining 2,599 tests are tagged `generic` or carry a retain-path guard (`baked`, `non-alien-first`, `canonical-clearsolver`, `d-case-determinism`, etc.) and are unaffected by those three epics.

---

## 2. Per-Epic Deletion Scope

### Epic 03 — Drop `WarmStartBasisMode::AlienOnly`

Reference: master plan AD-2.

Epic 03 removes the `AlienOnly` warm-start branch, its config-parsing surface, and its broadcast wire field. The guard labels from the taxonomy's step-3 Deletion-mapping row are `alien-only`, `warm-start-config-flag`, and `broadcast-warm-start-field`.

- guard `alien-only`: 3 tests
  - `crates/cobre-solver/src/highs.rs` (2)
  - `crates/cobre-sddp/tests/basis_reconstruct_churn.rs` (1)
- guard `warm-start-config-flag`: 2 tests
  - `crates/cobre-solver/src/highs.rs` (2)
- guard `broadcast-warm-start-field`: 0 tests

Expected deletion: ~51 body LoC across 1 file (`crates/cobre-solver/src/highs.rs`).

Note: 2 of the 3 `alien-only` tests also carry non-deletion guards (`non-alien-first` or `solve-with-basis-trait`), so those tests survive Epic 03 and are only fully removed when Epics 04 and 05 complete. The one fully-deletable test is `test_cobre_highs_set_basis_non_alien_ffi_contract` (guards: `alien-only`, `warm-start-config-flag`; 51 body LoC).

---

### Epic 04 — Drop `CanonicalStateStrategy::Disabled`

Reference: master plan AD-4.

Epic 04 removes the `Disabled` canonical-state strategy, its config-parsing surface, its broadcast wire field, and the `clear_solver_state` / `solve_with_basis` trait methods that the strategy gates. The guard labels from the taxonomy's step-4 Deletion-mapping row are `canonical-disabled`, `canonical-config-flag`, `broadcast-canonical-field`, `clear-solver-state-trait`, and `solve-with-basis-trait`.

- guard `canonical-disabled`: 5 tests
  - `crates/cobre-sddp/src/setup/params.rs` (3)
  - `crates/cobre-sddp/tests/canonical_state_strategy.rs` (1)
  - `crates/cobre-sddp/tests/integration.rs` (1)
- guard `canonical-config-flag`: 28 tests
  - `crates/cobre-sddp/src/backward.rs` (23)
  - `crates/cobre-sddp/src/setup/params.rs` (4)
  - `crates/cobre-sddp/tests/integration.rs` (1)
- guard `broadcast-canonical-field`: 0 tests
- guard `clear-solver-state-trait`: 9 tests
  - `crates/cobre-solver/src/highs.rs` (5)
  - `crates/cobre-solver/src/trait_def.rs` (4)
- guard `solve-with-basis-trait`: 14 tests
  - `crates/cobre-solver/src/highs.rs` (6)
  - `crates/cobre-solver/tests/conformance.rs` (4)
  - `crates/cobre-solver/src/trait_def.rs` (3)
  - `crates/cobre-solver/tests/ffi_set_basis_non_alien_smoke.rs` (1)

Expected deletion: ~3,148 body LoC across 7 files.

The large LoC figure is driven by `crates/cobre-sddp/src/backward.rs` (23 tests, each constructing a full backward-pass harness to set `canonical_state`). The 41 fully-deletable tests are those whose guard set is a subset of the overall deletion guards from steps 3–5; 7 of the 48 Epic 04 deletion candidates survive this epic because they also carry `non-alien-first` (a retain-path guard).

---

### Epic 05 — Drop non-baked template path

Reference: master plan AD-3.

Epic 05 removes the legacy `add_rows`-based non-baked template path, the `stored_cut_row_offset` parameter in `reconstruct_basis`, and the `add_rows` trait method. The guard labels from the taxonomy's step-5 Deletion-mapping row are `non-baked`, `stored-cut-row-offset`, and `add-rows-trait`.

- guard `non-baked`: 0 tests
- guard `stored-cut-row-offset`: 2 tests
  - `crates/cobre-sddp/src/basis_reconstruct.rs` (2)
- guard `add-rows-trait`: 33 tests
  - `crates/cobre-solver/tests/conformance.rs` (16)
  - `crates/cobre-sddp/src/lp_builder/template.rs` (9)
  - `crates/cobre-solver/src/highs.rs` (7)
  - `crates/cobre-solver/src/trait_def.rs` (1)

Expected deletion: ~1,040 body LoC across 5 files.

The 31 fully-deletable tests exclude 4 `fpha-slow` tests in `lp_builder/template.rs` that carry `add-rows-trait,fpha-slow`; those 4 are Epic 05 deletion candidates (the `add-rows-trait` guard qualifies them) but their `fpha-slow` co-guard makes them also subject to Epic 09's slow-test gating work. The owning epic for the `add-rows-trait` removal is Epic 05; the gating policy change is an Epic 09 concern.

---

## 3. Open Questions

The inventory's Notes column was fully empty at time of commit (no `TODO:` markers and no "verify with owner" annotations were entered during the tagging pass). There are therefore no open questions to hand off from this inventory revision.

Any ambiguities identified during Epic 03/04/05 implementation should be recorded as follow-up comments on the respective epic's opening ticket, not retrofitted here.

---

## 4. Parameterization Queue for Epic 09

The inventory's Section 2 table contains zero rows tagged `parameter-sweep`. The initial automated tagging pass did not identify clear parameter-sweep groups; a manual follow-up pass is needed before Epic 09 opens.

The slow-test roster (`fpha-slow`, 164 tests) was captured in `test-inventory.md` Section 5 and in `test-wall-time-baseline.md` Section 4. Epic 09's ticketer should use those sections as the primary input for the `fpha_*` gating work; no additional parameterization data is surfaced by this inventory pass.
