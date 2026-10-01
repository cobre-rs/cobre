---
paths:
  - "crates/**/tests/**/*.rs"
  - "examples/deterministic/**"
---

# Cobre Testing Rules

How Cobre is tested, to balance coverage, precision, and maintainability; this
file is the standing contract. Generic
cross-language testing pillars (test pyramid, few binaries, one builder, property
tests for ordering, benches≠tests, uniform slow-gating) live in the global Rust
rule; the rules below are the Cobre-specific ones.

## Tiers — use the cheapest tier that catches the regression

1. **Golden bit-exact (parity hash).** SHA-256 over **final** output (cut pool,
   simulation primal / dual / equipment / cost). Reserved for a small, deliberately
   feature-combined golden set whose union spans the
   cross-feature interactions that hide dormant bugs (cascade, anticipated, NCS,
   multi-resolution, energy contracts / pumping) and which exercise the shared
   LP-build / scaling / cut / basis machinery. The authoritative membership is the
   set of `parity_hash_<case>` test names, not a list frozen here; every other
   hashed case is demoted to tier 2 (behavioral). NOT the default; promoting a case
   into the golden set needs justification. The hash MUST NOT include the
   convergence trajectory or any per-iteration state — a faster warm-start to the
   same optimum must not break it.
2. **Behavioral (the default for deterministic cases).** `LB == UB` to tolerance,
   known optimum cost, water-balance closure, feature dispatch values. Backend-
   AGNOSTIC: one assertion covers HiGHS and CLP and survives benign refactors with no
   re-baseline. `crates/cobre-sddp/tests/deterministic.rs` is the model.
3. **LP structural.** Column / row counts, CSC validity, objective-coefficient wiring,
   without running a solver (`template_integration.rs`). First-class, not optional: a
   wrong column count produces numerically-smooth wrong bounds a tolerance assertion
   can miss.
4. **Analytical derivation.** Closed-form expected cut / coefficient from the model
   (`anticipated_backward_cut`-style). The gold standard for a NEW correctness claim —
   write one before adding a deterministic case.

## Contracts

- **Declaration-order invariance is a sort contract, tested as a unit test.** Build a
  `System` from permuted input and assert identical canonical order. A full training
  run over already-sorted input is a tautology that cannot detect an ordering bug — do
  not pass it off as a DOI probe.
- **Feature interactions are tested explicitly.** Each feature ships at least one test
  exercising it alongside its nearest neighbours in the LP column layout. Dormant bugs
  come from untested combinations (discount × anticipated; nonuniform-block extraction).
- **Parity baselines have ONE source of truth — the committed `.sha256`.** Do not add a
  second pinned copy (an `EXPECTED_HASHES`-style mirror): git diff records changes and
  the end-to-end test verifies the computation. A moved hash means the numbers changed —
  investigate, do not reflexively re-baseline.
- **Backend parity lives at the behavioral tier.** HiGHS and CLP differ at the
  FP-trajectory level; assert costs / invariants to tolerance (covers both), reserve
  bit-exact for the golden HiGHS path only.
- **Test output goes to `TempDir`.** No test writes into
  `examples/deterministic/*/output/`; generated artifacts self-delete.
- **A determinism gate must have power on the fixture it runs — a bitwise
  comparison proves nothing on a fixture that never exercises the condition
  the gate is meant to guard.** Three invariants apply across
  `crates/cobre-sddp/tests/mpi_wire.rs`'s determinism gates:
  - A gate whose power depends on a runtime-resolved fixture threshold
    self-checks that threshold and fails loudly instead of passing
    vacuously — a forced-retry gate asserts `retry_attempts > 0` per shape,
    an opening-order gate asserts `n_openings >= 3` on some stage, and an
    opening-block-scheduler gate asserts a resolved block count `>= 2`. The
    authoritative set of preconditions is the gates themselves, not a list
    frozen here.
  - A backward-path gate is exercised on both the uniform and the
    non-uniform cut-state-projection fixture axis — a stage whose projected
    cut-state dimension differs from its neighbours' (`d43-storage-only-cut`
    is the non-uniform reference case), not only a fixture where every
    stage projects the same dimension. Uniform-only coverage on this axis
    previously let a per-stage cut-state-projection crash class through
    undetected; a new backward-path gate is incomplete until it covers
    both, or the omission is justified.
  - A scheduler-determinism gate asserts same-scheduler reproducibility and
    thread/rank-shape invariance; it never asserts opening-block-vs-
    trial-point bitwise equality beyond the single-opening case where the
    opening-block chain IS the trial-point chain. The opening-block
    scheduler warm-chains a block from a fresh frozen-LP load while the
    trial-point scheduler warm-chains the whole trial point, so at a
    degenerate optimum a multi-opening comparison may settle on a
    different-but-equally-valid dual vertex — the hot≠cold divergence the
    Cobre determinism contract explicitly permits (CLAUDE.md: "never hot ==
    cold") — so a multi-opening cross-scheduler bit-identity gate is not a
    gateable contract, even where a fixture happens to pass it today.

## Pinned expected values — every numeric oracle states its source

A value recorded from a run pins whatever the code computed, including a defect.
A pinned lower bound taken from a degenerate fixture once passed while the cut mask
was wrong. So every numeric expected value that a test compares a solve or a built LP
against is one of these three:

- **Derived.** The value is written in the test as an expression or closed form over
  the study's named constants. It never comes from the code under test or from the
  builder's own templates.
  - Label the constant or helper with a `/// derived:` doc line naming the closed form.
    `closed_form_total_cost` in `tests/anticipated_core.rs` is the model.
  - A derived oracle also gets a non-degeneracy twin when the fixture could pass by
    symmetry. The twin has the feature off, a zero rate or equal hours, and has its own
    derived value, which must differ.
- **Hybrid.** The derived core is asserted on its own, and the remainder is bounded.
  Label it `/// hybrid:`.
- **Characterization.** The value is recorded from a run. It is allowed only with both
  of these, and is labelled `/// characterization:`:
  - a non-degeneracy guard: a feature-off twin that must give a different value;
  - a companion invariant, such as `LB <= UB` or a binding constraint.

Rules that follow:

- **A recorded value that moves after a formulation change is a finding.** Never
  re-record it. Derive the new value, or route the change as a defect.
- **Literals asserted against a hand-built unit fixture are derived by construction.**
  This covers a coefficient, a column index or a bound the test itself laid out: the
  fixture is the derivation, and no label is needed. Keep an independent literal
  oracle there instead of reading the owner (`StageGeometry`, `StateSpace`), which
  would pin the code against itself.

## Re-baselining parity hashes

- Baselines live in TWO committed dirs under `crates/cobre-sddp/tests/fixtures/`:
  `parity_baselines/` (HiGHS) and `parity_baselines_clp/` (CLP — an independent
  set; CLP's simplex legitimately reaches different-but-valid vertices on
  degenerate optima). A deliberate re-baseline regenerates BOTH dirs in the same
  change — updating only one lets the other backend's slow-gated suite rot
  silently.

  ```bash
  cargo nextest run -p cobre-sddp --features slow-tests --test parity \
    -E 'test(parity_regen)' --run-ignored ignored-only
  cargo nextest run -p cobre-sddp --no-default-features --features "clp slow-tests" \
    --test parity -E 'test(parity_regen)' --run-ignored ignored-only
  ```

- Parity hashes are environment-sensitive (floating-point / solver-build
  divergence). Before trusting a local regen, run the suite WITHOUT regen and
  confirm every case the change should not affect reproduces its committed hash.
  When unchanged cases fail locally, judge the change by result-neutrality —
  capture the actual hashes on the original code and on the changed code; an
  empty diff between the two proves the change bit-for-bit safe — not by
  baseline match.

## Cost discipline — Cobre links a solver into every test binary

- `cobre-solver` statically links HiGHS/CLP into every dependent `tests/*.rs`, so each
  new integration-test **binary** pays a full C++ link. Add a test function to an
  existing domain binary rather than a new file; group related tests with `mod`.
- The shared harness helpers (`StubComm`, `build_setup_in_code`, `build_setup_for_case`,
  `run_simulation`) and entity construction (`Stage`/`Hydro`/`Bus`/`Thermal` through the
  `make_*` builders) live once in `tests/common/` — never re-defined per file. Because
  every `tests/*.rs` construction routes through the shared builder, adding a required
  field to one of those `cobre-core` entities is a one-place change in
  `tests/common/builders.rs` (the `<Entity>Spec` default + the `make_<entity>` mapping);
  no `tests/*.rs` consumer is touched.
- That one-place property is scoped to the integration-test layer and does NOT hold
  workspace-wide: production parquet readers and inline `#[cfg(test)]` `--lib` unit-test
  literals construct these entities directly and must each wire a new field themselves.
