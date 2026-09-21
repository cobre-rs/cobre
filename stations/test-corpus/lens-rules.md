# Lens rules — test corpus and test-support (baseline `077dbe2c`)

The two mechanically checkable rules every test-corpus lens (attacker, defender, calibration) operates under. Each carries its source anchor at the pin (`077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`) and a candidate test a lens applies without re-reading the source. Read-only: no fix is proposed here and no repo file is written. Figures come from `stations/test-corpus/inventory.json` (E08-1), never from a fresh worktree grep.

## Rule 1 — the DOI tautology rule

**Source.** `.claude/rules/testing.md:40` → `## Contracts`, first bullet (`.claude/rules/testing.md:42-45`). Corroborated by `docs/design/testing-architecture.md:545-547` (§5.9: expand `proptest` to cover the declaration-order-invariance and reduction-order invariants directly — permute → assert identical bits) and by the `permute` helper the yardstick names at `docs/design/testing-architecture.md:74` (§2.2).

**Rule.** Declaration-order invariance (DOI) is a sort contract, tested as a unit test: build a `System` from permuted input and assert identical canonical order. A full training run over already-sorted input is a tautology that cannot detect an ordering bug and must not be passed off as a DOI probe.

**Candidate test a lens applies.** A test whose name, doc comment or module doc claims order or declaration invariance, but whose body reaches no permutation step, is a TD candidate (vacuous coverage: the claim outruns the assertion). A permutation step is any one of:

- a call into `crates/cobre-sddp/tests/common/permute.rs` (`permute_case` at `crates/cobre-sddp/tests/common/permute.rs:83`);
- a `proptest!` block that generates the ordering (the pattern `crates/cobre-stochastic/src/sampling/external.rs` uses for `derive_external_sample_moments_is_declaration_order_invariant`);
- an in-test shuffle of the entity declaration order before the `System` / `StudySetup` is built;
- the shuffle matrix in `.github/workflows/invariance-shuffle.yml:34`, which permutes the deck outside the test body — a test driven only by that workflow is a DOI probe only on manual dispatch (the cron at `.github/workflows/invariance-shuffle.yml:6` is commented out), which is a cadence fact for the §2.3 rows, not a Rule 1 defect.

Mechanical probe at the pin (recorded in E08-2, re-runnable):

```
git grep -il 'invarian' 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c -- 'crates/*/tests/*.rs'   # 40 files claim some invariance
# of which `git show <pin>:<file> | grep -q permute` holds for 9
```

The 31 files without a permute call (29 of the 40 hits are in `cobre-sddp`; the rest split cobre-stochastic 5, cobre-cli 2, cobre-io 2, cobre-core 1, cobre-solver 1) are CANDIDATES, not findings. Most use "invariant" in a non-ordering sense — rank-shape invariance in `crates/cobre-sddp/tests/mpi_wire.rs`, the `sync_cuts invariant violated` error contract and the K-fan `final_lb` rank-invariance gate in `crates/cobre-sddp/tests/test_mpi_sync_cuts_invariant.rs:6` (`k_fan_final_lb_bitwise_invariant_across_world_size` at `crates/cobre-sddp/tests/test_mpi_sync_cuts_invariant.rs:161`). A reproducibility or rank-invariance gate that trains over the canonical deck and compares hashes across rank or worker counts is out of Rule 1's scope by definition (Cobre determinism = reproducibility + order-invariance; the two are separate contracts, and neither is hot == cold). The lens reads each candidate's actual claim and dismisses those whose invariance is not about declaration order.

The 9 permute-carrying files at the pin are proven DOI probes and never candidates: `crates/cobre-solver/tests/clp_determinism.rs`, `crates/cobre-sddp/tests/template_integration/generic_constraints.rs`, `crates/cobre-sddp/tests/scalar_parameters_declaration_order.rs`, `crates/cobre-sddp/tests/parity.rs`, `crates/cobre-sddp/tests/hydro_sim.rs`, `crates/cobre-sddp/tests/deterministic.rs`, `crates/cobre-sddp/tests/cut_basis.rs`, `crates/cobre-sddp/tests/common/permute.rs`, `crates/cobre-sddp/tests/common/parity_hash.rs`.

Two probe hits the attacker lens can dismiss on reading (recorded, not adjudicated): `crates/cobre-cli/tests/cli_run_anticipated.rs:102` and `crates/cobre-cli/tests/cli_run_anticipated_k2.rs:112` mention "declaration-order invariance rule" only inside a doc comment about id ordering ("anticipated id > regular id") — they cite the rule, they do not claim to probe it. They remain the illustration of the structural fact behind Rule 1's reach: `cobre-cli` has no `tests/common/` and cannot call `cobre-sddp`'s `permute_case`, so a genuine CLI-level DOI probe is impossible until the §5.2 collapse of `tests/common/` into `test-support` lands (a roadmap item, never a defect).

## Rule 2 — cost discipline (coverage-neutral fix-shapes only)

**Source.** `.claude/rules/testing.md:113` → `## Cost discipline — Cobre links a solver into every test binary`, first bullet (`.claude/rules/testing.md:115-117`). Corroborated by `docs/design/testing-architecture.md:602-604` (§7 non-goal "Reducing coverage to shrink the suite"), `docs/design/testing-architecture.md:353-354` (§5.1 migration invariant "No test is deleted, skipped, renamed, or weakened") and `docs/design/testing-architecture.md:570-571` (§6: every phase gated on `cargo nextest list` count parity).

**Rule.** `cobre-solver` statically links the C++ solver into every dependent `tests/*.rs`, so cost is per-**binary** and per-**feature-combo**, never per-**test**. At the pin 62 of the 87 integration binaries link the solver (`inventory.json` figures `int-binaries-solver-linking` and `int-binaries`), while the 6,371 listed tests split 5,365 lib-unit / 169 bin-unit / 839 integration (`remeasure.log`, block `nextest-list`). Deleting tests saves nothing structural and forfeits the suite's best asset.

**Consequence for this station.** Deleting, skipping, weakening or re-baselining a test is NEVER a fix-shape in this station. A candidate proposing one — under any rationale that names suite size, build time, link time, CI wall-clock or "bloat" — is dismissed and recorded under `positives`, citing this rule. Admissible fix-shapes are consolidation (grouping binaries through `#[path]` submodules), re-homing (inline ↔ sibling `tests.rs`), feature-surface unification (`test-support`), harness de-duplication (one shared helper replacing byte-identical copies) and cadence tiering — all behavior- and count-neutral.

**Candidate test a lens applies.** For every proposed fix-shape answer two questions: does `cargo nextest list --features test-support` report the same count before and after, and does every retained test assert what it asserted before? If either answer is no, the candidate is dismissed under this rule and recorded under `positives`. A fix-shape that lowers the binary count or the feature-combo count while keeping both answers yes is the only cost-motivated shape this station admits.

**Boundary with the ratified duplication class (recorded, not decided here).** Earlier stations minted, and their owner gates ratified, TD entries whose direction removes a test that duplicates a retained one — `TD-015` (identical bodies), `TD-051` (one deletion), `TD-067` / `TD-068` (tests asserting a proper subset of a named retained test's assertions on the same fixture), `TD-011` (a redundant ordering test) — and those entries are queued to this station. Rule 2 as stated forbids deletion as a fix-shape; the ratified entries treat removal of an exact-duplicate assertion as a duplication fix whose named retained test preserves every claim. The two readings meet only under a stricter test for that class: the entry names the retained test whose assertions are a superset on an identical fixture, the count change equals the removed duplicates and nothing else, and the rationale is duplication, never cost. Whether that carve-out is admitted is an owner decision at the E08 owner gate. Until then a lens records any such candidate under `positives` per this rule and cross-references the queued TD id rather than re-raising it.
