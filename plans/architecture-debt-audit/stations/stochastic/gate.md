# Owner ratification gate — cobre-stochastic

Baseline: `a136840d4f2ea137f685f0af6dac04254b983b60` · station verified 7/7 (see `verification.md`).
Gate run: 2026-09-08 (main session). **Decision: ratified** — 35 accepted, 0 downgraded, 0 rejected, 0 deferred, 0 overridden.

The gate executes no fix. No `conflicts` holds this station (0 fix-shape tripped the L1-purity guardrail) and no dismissals to clear.

## 1. Presentation digest

Confirmed entries: **35** · holds (conflicts): **0** · cleared: **0** · needs-human: **1**.

| ID | Sev | Alignment | Title | Fix-shape (one line) |
| -- | --- | --------- | ----- | -------------------- |
| CD-064 | B | neutral | Decomposition-engine phase vocabulary in par L1-kernel doc comments (forward/backward pas… | Reword the flagged doc comments to consumer-agnostic language that describes the layout/behaviour intrinsically rather than by the SDDP engine's phas… |
| PD-021 | B | neutral | Per-season correlation matrices recompute every hydro-pair date-set intersection twice an… | Two independent, fitting-local reductions with no cross-crate blast radius. (i) Compute each pair's overlap count once: have compute_pearson_correlat… |
| PD-020 | B | neutral | PAR-A initial per-hydro AR estimation runs single-threaded while the byte-identical class… | Lift the initial per-hydro block of estimate_ar_with_pacf_annual (the `for &hydro_id in hydro_ids { ... estimates.push(...) }` region) into the same… |
| TD-025 | B | neutral | Inline/sibling unit-test fixture builders (Stage / InflowModel) re-authored across ~9 par… | Provide one parameterized builder each for Stage and InflowModel (the O(1) field-add builder pattern the yardstick §3.1.4 credits to cobre-sddp/tests… |
| TD-024 | B | neutral | Integration-binary fixture prelude (make_bus / make_hydro / identity_correlation) re-auth… | Introduce a single shared fixture surface for the crate's cobre-core entity builders and mod-declare it once per binary, so make_bus/make_hydro/make_… |
| CD-066 | B (A-risk) | neutral | Entity-class identity degrades to a stringly-typed &str at the build_class_sampler seam,… | Introduce a small closed class discriminant (e.g. an entity-class enum with `Inflow|Load|Ncs`) carried on `ClassSamplerParams` in place of the `&str`… |
| CD-065 | B | advances-1 | ForwardSamplerConfig is both a store handle and a parameter bag, and is the point where t… | Split the one flat bundle into two clearly-named halves at the seam: a persistent store handle (today `&StochasticContext`; the generation-side tree/… |
| OD-029 | B (A-risk) | neutral | standardize_external_inflow's #[allow(too_many_arguments)] rationale ('no natural sub-gro… | Introduce a by-reference aggregate for the stage-0 derived seed, e.g. DerivedSeed<'a> { lag_values: &'a [f64], l_state: usize, accum: &'a [f64], weig… |
| OD-028 | B | neutral | ForwardSamplerConfig encodes a per-class scheme x library validity matrix as four indepen… | Fold each class's scheme selector and its required library into one per-class source enum whose data-bearing variants carry the borrow: e.g. InflowSo… |
| PD-023 | B | neutral | OutOfSample QmcSobol forward draws re-derive the Sobol direction matrix + scramble params… | Give the OutOfSample class sampler (or the caller that owns the per-(iteration,stage) loop) a per-stage SobolPrecomputed cache keyed on (forward_seed… |
| PD-025 | B | neutral | OutOfSample LHS forward draws regenerate a full Fisher-Yates permutation of `total_scenar… | Hoist the per-(iteration,stage) LHS permutation set across the scenario axis: build the `dim` stratification permutations once per (forward_seed, ite… |
| PD-024 | B | neutral | OutOfSample QmcHalton forward draws rebuild the prime sieve and a `total_scenarios`-sized… | Mirror the Sobol hoist: introduce a per-(iteration,stage) precomputed Halton context (a HaltonPrecomputed holding the prime list and the scramble tab… |
| TD-028 | B | neutral | Per-file test fixtures re-declared across the sampling inline test modules: make_hydro x3… | Hoist each shared builder to the crate that owns the type it constructs, behind a test-support feature, per testing-architecture.md 5.2 (helpers live… |
| CD-070 | B (A-risk) | advances-1 | generation seam ingests realized external-scenario tables: external_ar0_inflow_models/ext… | Roadmap-aligned: under Part-1 purification the external-scenario tables leave `System` for the cobre-stochastic uncertainty store; the External-inges… |
| CD-068 | B | neutral | DecomposedCorrelation ships a superseded full-vector correlation seam (test-only) beside… | Collapse to a single correlation applier. Preferred: since production uses only apply_correlation_for_class on the scan path, remove the full-vector… |
| CD-067 | B (A-risk) | neutral | Entity class (inflow/load/ncs) is a stringly-typed closed set matched by `==` across gene… | Introduce an in-crate closed enum EntityClass { Inflow, Load, Ncs }. Parse the cobre-core-sourced CorrelationEntity.entity_type String exactly once,… |
| PD-027 | B | neutral | apply_group_precomputed and apply_group_scan heap-allocate gathered/correlated (and posit… | Give the apply functions caller-owned reusable scratch: allocate gathered/correlated (and, for the scan path, positions) once, sized to the maximum g… |
| PD-026 | B | neutral | generate_opening_tree pays a per-opening O(group*class) linear position scan because the… | Hoist entity-position resolution to once per tree. Either give generate_opening_tree a &mut DecomposedCorrelation and call resolve_class_positions on… |
| TD-030 | B | neutral | Near-verbatim ~230-line fixture prelude copied across the QMC integration binaries (halto… | Hoist the shared fixture prelude out of the per-binary copies into one home. The crate already dev-depends on cobre-core with its `test-support` feat… |
| OD-027 | C | neutral | Bare `estimate_ar_coefficients` / `estimate_correlation` / `estimate_seasonal_stats` are… | Remove the ergonomic overload asymmetry: since `_with_season_map` already takes `Option<&SeasonMap>` and is the sole production entry, either demote… |
| OD-026 | C | neutral | PAR validation warning apparatus (single-variant `ParWarning` enum + single-field `ParVal… | Collapse the single-inhabitant taxonomy: either (a) if the low-residual-variance diagnostic has no intended reader, drop the `warnings` accumulation… |
| PD-022 | C | neutral | Per-hydro observation vectors are deep-copied out of group_obs on every estimate/reduce p… | Where the consumer only reads observations (the two reduction loops and the PAR-A initial estimate all build obs_refs: Vec<&[f64]> and pass borrows i… |
| TD-027 | C | neutral | pop_mean_std and pop_mean_std_ann are identical population mean/std helpers declared twic… | Delete pop_mean_std_ann and point its call sites at the single pop_mean_std (or, if these move under the consolidated par/fitting test helper, keep o… |
| TD-026 | C | neutral | simulate_two_season_par2 PAR(2) Box-Muller scenario simulator duplicated byte-for-byte ac… | Hoist the one simulator into a single shared par/fitting test helper that both sibling files import (a `#[cfg(test)]` helper module under par/fitting… |
| TD-029 | C | neutral | saa_golden_value.rs re-declares the integration-binary fixture prelude: identity_correlat… | When the shared integration prelude is consolidated (testing-architecture.md 5.2/5.1 - a per-crate test-support surface plus the single integration b… |
| CD-071 | C | advances-1 | StochasticContext is both a computed store and a config/layout echo: 5 precomputed compon… | Phase-1 destination note, NOT executed by this spec (lens Q4 is alignmentHint-only). When Part IV/V introduces the Switchable<T> uncertainty store, t… |
| OD-031 | C | neutral | Three StochasticError variants have no production producer — dead public error taxonomy (… | Two roads, owner's pick. (a) Delete the three never-produced variants from the pub StochasticError enum, delete the context.rs:593 doc bullet that pr… |
| PD-029 | C | neutral | Backward season-occurrence walk is quadratic in l_state: derive_inflow_seeds restarts nth… | Walk the backward season-occurrence chain once per hydro instead of restarting per k. Add a StageCalendar method (or a season_cast free helper) that,… |
| PD-030 | C | neutral | derive_inflow_seeds re-scans the entire record and conditioning slices once per hydro ins… | Before the hydro loop, do one pass over record and one over conditioning to bucket rows by hydro_id (e.g. a HashMap<EntityId, Vec<RealizedWindow>>),… |
| TD-032 | C | neutral | Entity-fixture prelude (make_bus/make_stage/make_hydro/make_inflow_model/identity_correla… | Coverage-neutral de-duplication (no test deleted, per yardstick §7): hoist the shared engine-neutral entity builders into a single test-support surfa… |
| TD-033 | C | neutral | Tautological equivalence test: test_season_occurrence_matches_pre_refactor_helper_sequenc… | Remove the test (not a coverage reduction per yardstick §7: it asserts nothing the helper-level unit tests do not already prove, because the wrapper… |
| CD-069 | C | neutral | Three structurally identical point-sample parameter bundles (LhsPointSpec / HaltonPointSp… | Introduce one shared spec type (e.g. QmcPointSpec / PointSampleSpec) in a common noise or tree module carrying sampling_seed/iteration/scenario/stage… |
| OD-030 | C | neutral | SweepDirection is effectively one-valued in production: Ascending is constructed only ins… | Collapse the speculative generality: since every production caller sorts largest-key-first, drop the SweepDirection enum and the direction parameter… |
| PD-028 | C | neutral | apply_correlation_for_class re-resolves the active profile (HashMap<i32,String> + BTreeMa… | Resolve the &[GroupFactor] slice for the stage once, before the opening loop in generate_opening_tree (e.g. a per-stage accessor on DecomposedCorrela… |
| TD-031 | C | neutral | Inline-giant vs extracted-sibling unit-test homing coin-flip inside cobre-stochastic: tre… | Adopt the yardstick §5.1 deterministic homing rule as a crate-wide lint: unit tests stay inline below a fixed threshold (the proposal is ~500 test-LO… |

### Holds (conflicts) — roadmap-consistent alternative

None. No confirmed fix-shape pulls an SDDP/paradigm noun into this L1 crate, makes it depend on an engine crate, or introduces a one-consumer abstraction.

### Cleared (dismissed — do not re-raise)

None. All 35 candidates were confirmed; none was dismissed.

### Phase-1 uncertainty-store notes

Alignment hints only (Epic 9 adjudicates). The `advances-1` seam findings (CD-065 ForwardSamplerConfig store-vs-parameter-bag, CD-070 realized external-scenario ingest, CD-071 StochasticContext store-vs-config echo) mark where the Phase-1 `Switchable<T>` uncertainty store would form; the fitted-process code (PAR estimation, opening-tree generation, seed derivation) stays generation-only.

### Prior-register dispositions

| Prior item | Disposition |
| ---------- | ----------- |
| Stage-calendar crate home | keep |
| External-noise take/fill glue duplication | not-ours |
| Deterministic (σ = 0) AR(p > 0) external inflow stays rejected | keep |
| `LoadModel` conflates physical load with its stochastic model | not-ours |
| Cross-path static-RHS contract not yet in `.claude/rules/sddp.md` | not-ours |
| Oracle test-harness duplication | cross-reference |
| CD-001 — setup config-projection sprawl / CLI non-root reconstruction | resolved |

None is `sharpen`; CD-001 is resolved at the baseline and travels only as an alignment hint.

### Handoff queues

- performance-sweep (Epic 10): **7** Sev-A/B PD entries (UNMEASURED; claim type + 4t layout + profiled symbol; no timing asserted).
- test-corpus (Epic 8): **10** TD entries.
- alignment (Epic 9): **3** `advances-1` entries + the CD-001 L1-homing question.

### Needs-human (aggregated from the candidates envelopes)

- [candidates-seam] Genericity substring confirmation: the lens's unanchored grep `(sddp|benders|cut|...)` matched `cut_points` at crates/cobre-stochastic/src/season_cast/mod.rs:439 (and its uses at 440,442). This is a benign day-boundary variable; the authoritative gate scripts/ci/check-infra-genericity.sh uses `\bcut\b`, which does not match `cut_points`. Recorded per the attacker-prompt rule that any token match…

## 2. AskUserQuestion round plan

Assembled before any question was asked; order and option wording fixed here.

1. **Sev-B batch** (19) — accept all / downgrade / reject / defer.
2. **Sev-C batch** (16) — accept all / downgrade / reject / defer.
3. **Prior-register ratification** (7, none sharpen) — ratify as recorded / amend one.
4. **Needs-human** (1: `cut_points` genericity substring) — not-a-finding / finding.

No Sev-A tier and no conflicts holds, so neither a Sev-A batch nor any per-hold override question was raised.

## 3. Decision record

Owner answers (main-session AskUserQuestion, verbatim): Sev-B tier — "Accept all as recorded"; Sev-C tier — "Accept all as recorded"; prior register — "Ratify all as recorded"; needs-human — "Not-a-finding (benign)".

| ID | Decision | Severity | Alignment | Rationale (owner) | Trigger / override | Queue |
| -- | -------- | -------- | --------- | ----------------- | ------------------ | ----- |
| CD-064 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | none |
| PD-021 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-020 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| TD-025 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| TD-024 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| CD-066 | accept | B (A-risk) | neutral | accepted as recorded (Sev-B batch) | — | none |
| CD-065 | accept | B | advances-1 | accepted as recorded (Sev-B batch) | — | alignment |
| OD-029 | accept | B (A-risk) | neutral | accepted as recorded (Sev-B batch) | — | none |
| OD-028 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | none |
| PD-023 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-025 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-024 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| TD-028 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| CD-070 | accept | B (A-risk) | advances-1 | accepted as recorded (Sev-B batch) | — | alignment |
| CD-068 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | none |
| CD-067 | accept | B (A-risk) | neutral | accepted as recorded (Sev-B batch) | — | none |
| PD-027 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-026 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| TD-030 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| OD-027 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| OD-026 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| PD-022 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none (Sev-C, below sweep threshold) |
| TD-027 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| TD-026 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| TD-029 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| CD-071 | accept | C | advances-1 | accepted as recorded (Sev-C batch) | — | alignment |
| OD-031 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| PD-029 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none (Sev-C, below sweep threshold) |
| PD-030 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none (Sev-C, below sweep threshold) |
| TD-032 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| TD-033 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| CD-069 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| OD-030 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| PD-028 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none (Sev-C, below sweep threshold) |
| TD-031 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |

**Prior-register dispositions ratified (as recorded):**
- Stage-calendar crate home → keep
- External-noise take/fill glue duplication → not-ours
- Deterministic (σ = 0) AR(p > 0) external inflow stays rejected → keep
- `LoadModel` conflates physical load with its stochastic model → not-ours
- Cross-path static-RHS contract not yet in `.claude/rules/sddp.md` → not-ours
- Oracle test-harness duplication → cross-reference
- CD-001 — setup config-projection sprawl / CLI non-root reconstruction → resolved

**Needs-human resolved:** the architecture/seam `cut_points` substring at `season_cast/mod.rs:439` is recorded **not-a-finding** — a benign calendar day-boundary variable; the authoritative `\bcut\b` genericity gate does not flag it.

**Cleared by this gate (do not re-raise):** none.

**Gate: RETURNED 2026-09-08** — baseline `a136840d`; accepted 35, downgraded 0, rejected 0, deferred 0, overridden 0.

Decision: ratified
returned: true
