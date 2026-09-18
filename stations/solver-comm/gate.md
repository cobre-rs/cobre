# Owner gate — cobre-solver + cobre-comm (2026-09, baseline `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`)

Run in the main session as AskUserQuestion rounds over this digest. Inputs: `calibration.json`
(20 calibrated entries), `verdicts.json`, `partI-handoff.json`, `verification.md` (shared verifier
7/7 + station verifier 5/5 at the pin), `candidates.<lens>.json` (`_needsHuman`). Severity is the
HOUSE rating; a downgraded row shows `new (reviewer: original)`.

## 1. Presentation

### 1.1 Actionable entries (Part-I item 8 first, then A/B/C)

| # | ID | Sev | Alignment | Claim (narrowed) | Fix-shape (one line) | Handoff |
| -- | -- | --- | --------- | ---------------- | -------------------- | ------- |
| 1 | CD-079 | B (A-risk) | advances-1 · I.3-8 | Two additions survive, both scoped to the E9 collateral estimate and neither to crate behaviour: (a) the handoff clips exactly two of its three asser… | Delete the five fields from the L0 container so it carries only the CSC arrays and the dimensions the LP itself needs; nothing in either station crate reads them, and th… | alignment |
| 2 | CD-078 | B | neutral | The ffi/clp.rs:27-31 rationale is stale at the baseline — get_basis and install_basis decode and re-encode the CLP codes through BasisStatus with fol… | Give the CLP code space one owner in the binding module, mirroring what the HiGHS side already does: declare all six values as named constants in ffi/clp.rs next to the… | - |
| 3 | OD-033 | B | neutral | Only the acquire half of the CLP hot-start lifecycle - the shim/extern pair cobre_clp_mark_hot_start plus cobre_clp_solve_from_hot_start and the two… | Resolve the seam's status rather than leaving it implicit. | - |
| 4 | OD-035 | B | neutral | The profile-floored tolerance pair in highs/retry.rs (two f64::max bindings plus the two cobre_highs_set_double_option calls) has no single owner: it… | Give the repeated fragment one owner: a private helper that takes the floor and applies both tolerance options in the current fixed order, and express the per-level delt… | - |
| 5 | PD-032 | B | neutral | Narrowed to the allocation only: cobre_clp_chg_bounds (crates/cobre-solver/csrc/clp_wrapper.c:208) heap-allocates, fills and frees a full-dimension s… | Give the CLP path a subset-scoped bound write that mirrors the shape the sibling backend already has. | performance-sweep |
| 6 | PD-033 | B | neutral | Rust side only, and only the CSC trio: on the per-iteration append path (load_backward_lp, append_new_cuts_to_lp) CLP add_rows heap-allocates two scr… | Move the append's working storage into caller-owned scratch on the solver struct, the way the freeze path and the sibling backend already do. | performance-sweep |
| 7 | PD-034 | B | neutral | Only the read side qualifies: ClpSolver::get_basis crosses the FFI once per column and once per row on the per-solve capture path, where the HiGHS si… | Add bulk status transfer to the cobre-owned CLP shim so the trait method crosses once per capture instead of once per element: one shim function that fills a caller-prov… | performance-sweep |
| 8 | TD-037 | B | neutral · I.3-8 | Only the four in-src declarations are unjustified duplication: crates/cobre-solver/src/types.rs:736, src/freeze.rs:262, src/backends/clp/tests.rs:20… | Add a fixtures submodule inside the already-shipped pub mod test_support at crates/cobre-solver/src/lib.rs:164, behind the existing test-support feature, and make it the… | test-corpus, alignment |
| 9 | CD-075 | C (reviewer: B) | neutral | Narrowed to the mechanism at this station only: because a crate-level `[lints]` table replaces rather than overlays `[workspace.lints]`, and no check… | The Cargo constraint is real — a manifest cannot combine `lints.workspace = true` with a per-lint override — so the fix is not deduplication but making the copies checka… | - |
| 10 | CD-076 | C | neutral | Three of the seven put above-L0 vocabulary in the normative sentence itself with no adjacent generic restatement: highs/solver.rs:477 ('the primary w… | Restate each of the seven in terms the solver itself owns, and be explicit about which sentences are load-bearing. | - |
| 11 | CD-077 | C | neutral | At `clp/mod.rs:25` the `pub(crate) use retry::LADDER_RUNGS` re-export has no non-test consumer -- `interface.rs:6` reads the constant through its own… | Have the sibling test module import the constant by its owning path — `super::retry::LADDER_RUNGS`, exactly what interface.rs:6 already does — and delete both the re-exp… | - |
| 12 | CD-080 | C | neutral | Narrowed off ffi/mod.rs and off any harm claim: the CLP-only build's carried HiGHS declarations are benign (no link demand per build.rs:30 and :97, n… | Make the two backends symmetric at the boundary so a single-backend build compiles only its own bindings: once the CLP code space has named constants in its own binding… | - |
| 13 | CD-081 | C | neutral | The private, contract-unpinned FreezeScratch.cut_nz_per_col -- declaration at crates/cobre-solver/src/freeze.rs:22 plus its four production uses at 1… | Rename the private field and its four production uses to name what the vector actually holds — a per-column nonzero census of the rows being appended — so the identifier… | build-ci |
| 14 | OD-032 | C | neutral | At the baseline ExecutionTopology::is_homogeneous (crates/cobre-comm/src/topology.rs:34) has no IN-WORKSPACE production reader and no reserved-seam e… | Drop the predicate and its four unit tests and let the caller that eventually needs a heterogeneity decision express it where the policy lives; today the only consumer o… | - |
| 15 | OD-034 | C | neutral | The 12-value agreement between `HighsProfile::default()` and the `default_options()` table (config.rs:58-77 against 168-238) is a load-bearing invari… | Make one surface the owner of each default. | - |
| 16 | TD-035 | C | neutral | Only crates/cobre-comm/src/factory.rs:419 is a verbatim vacuous restatement, asserting the same `CommBackend: Send + Sync` obligation that the produc… | Keep one mechanism per type and prefer the const-fn form, because it fires without cargo test and is the one already documented as load-bearing. | test-corpus |
| 17 | TD-036 | C | neutral | Only the src/local.rs to tests/local_conformance.rs overlap survives: the six body-identical LocalBackend Communicator assertions (allreduce identity… | Name tests/local_conformance.rs the single owner of the public SS1.1-SS1.8 Communicator contract, which is the role its own module doc already claims, and retire the inl… | test-corpus |
| 18 | TD-038 | C | neutral | Narrower residue: test_research_probe_limit_status_on_ss11_lp (highs/tests.rs:846) asserts neither of the two model_status values it exists to observ… | Retire the #[test] and keep the module-comment sentence that records the finding, which is where the knowledge already lives. | test-corpus |
| 19 | TD-039 | C | neutral | Narrowed to two anchors: the module-doc guarantee at clp_only_smoke.rs:3-4 is false under the very feature combination it names, because conformance.… | Retire the binary and let the clp-gated section of conformance.rs be the clp-only guard it already is. | test-corpus |
| 20 | TD-040 | C | advances-1 · I.3-8 | Only test_fixture_stage_template_data (conformance.rs:137-158) is I.3-8 collateral - its assertions at :153-157 re-encode the five shed fields; test_… | If the fixtures move into the test_support fixtures module, one contract test belongs there beside the builder, asserting once that the shared SS1.1 LP is the LP the bui… | test-corpus, alignment |

Verification: the shared verifier and the station verifier both PASS at `077dbe2c`
(`verification.md`: 7/7 generic rows, 5/5 station rows, test suite 81/81).

### 1.2 Holds (Alignment `conflicts`)

None. The L0 purity test over all 20 fix-shapes produced no hold; the only `conflicts` in the
station is the build-it-now variant of the capability trait, recorded informational (1.4).

### 1.3 Cleared and dup-of — presented read-only, no decision taken

- **Shared-memory communicator trait hierarchy** (`SharedMemoryProvider`, `SharedRegion<T>`,
  `LocalCommunicator`, `LocalCommKind`, `HeapRegion<T>`, `FerrompiBackend::split_local`) — ratified
  reserved seam, `docs/design/reserved-seams-and-deferred-debt.md:79`, register OD-001
  KEEP-RESERVED. Ingest filter: every lens filed it in Positives; no candidate, no defender, no id.
- **Superseded cut-sync public methods** (`sync_cuts` L243, `pack_local_records` L400,
  `sync_packed_records` L495, superseded by `sync_level_records` L581 in
  `crates/cobre-sddp/src/cut/cut_sync.rs`) — dup-of `docs/design/reserved-seams-and-deferred-debt.md:334`
  (register CD-019), handed to E5 with no id (`handoffs.json` E5 + records[]).
- **HiGHS-loud / CLP-silent basis-validation asymmetry** — intended backend behaviour pinned by four
  `conformance.rs` tests; never a finding.

- **Dismissed by its defender — SC-PERF-001 / performance-01** (partition helpers, layout 2x2): The title's load-bearing clause -- the partition is recomputed at every collective -- does not hold at the baseline for the majority of the anchored consumers. cut_sync.rs:182 and :186 are not in the record-sync path the candidate attributes them to; both sit… No seam citation (dismissed on mechanism).

### 1.4 Informational (no severity, no id)

- **SC-ARCH-009 / architecture-09** — `SolverInterface` has no capability / feature query; exactly one backend compiles. Roadmap beyond-sddp-generalization.md § III.6; Alignment neutral. Held variant: one-consumer-abstraction — grow the trait when two backends coexist in one binary (III.6).

### 1.5 Worker needs-human items (all must be answered before the marker)

1. _[attacker/architecture]_ Where does the shed geometry land: the fields' owners today (StateSpace, StageRowLayout) both live in crates/cobre-sddp/src/lp/, which is exactly the engine-neutral region phase 0b carves into cobre-model, so the owner must say whether Part-I item 8 sheds into the engine and moves again at 0b, or waits for the carve and sheds once.
2. _[attacker/architecture]_ Should the LocalCommKind enum live in traits.rs at all: it makes the trait-definition module import both concrete backends (traits.rs:21,23, both behind the shared-memory feature) while the sibling factory.rs:62 CommBackend does the identical enum-dispatch job in the module that already owns concrete-backend enumeration — an ownership question about placement only, distinct from the sanctioned existence-of-the-seam…
3. _[attacker/architecture]_ Does the layering brief's ban on multistage vocabulary in L0 bind production doc comments today, or only at the phase-1 purification: the project's hard genericity rule enumerates sddp/SDDP/Benders and none of the seven solver-side doc sites contains any of them, so the answer decides whether that candidate is a now-fix or a purification rider.
4. _[attacker/architecture]_ Is giving up `unsafe_code = "forbid"` at the workspace root acceptable in exchange for a single lint table: the second fix shape for the lint drift removes the drift surface entirely but trades an unoverridable prohibition for a per-crate audited allowance, which is an owner call and not an engineering preference.
5. _[attacker/over-engineering]_ The HiGHS escalation branches on wall-clock time (overall_budget at highs/retry.rs:40, the elapsed break at :52, the budget_exceeded comparison at :84) while the CLP ladder documents at clp/retry.rs:4-5 that it has no time-dependent branching so results stay bit-for-bit identical across thread and rank counts -- whether the HiGHS ladder is deliberately exempt from that property or this is a live determinism divergen…
6. _[attacker/over-engineering]_ Whether ExecutionTopology::is_homogeneous is intended for a planned heterogeneous-layout guard (in which case it needs a reserved-seam entry with an activating milestone, not deletion) or is simply left over -- the register carries no entry either way.
7. _[attacker/over-engineering]_ Whether the CLP hot-start half should be wired onto the re-solve path or registered as a reserved seam is an owner decision; the candidate deliberately proposes no removal.
8. _[attacker/performance]_ Whether the upstream simplex library exposes an index-scoped bound writer and a bulk basis-status accessor that preserve the factorization the same way the current full-array bound calls do decides whether the first two candidates are cobre-side shim additions or an intrinsic property of that library; the shim already reaches C++-class-only methods, which suggests the former, but the owner should confirm before the…
9. _[attacker/performance]_ Whether the retained column-major mirror on the CLP path must remain a full merged mirror, or whether the append path could keep the base and the appended blocks separately and merge only when the reload path actually needs a contiguous mirror, is an owner call about the retained-mirror contract rather than something the code alone settles.
10. _[attacker/test-bloat]_ crates/cobre-comm/tests/local_conformance.rs:4 cites backend-testing.md for the SS1.1-SS1.8 contract sections, and no file of that name exists anywhere in the tree at the baseline; whether the stale spec citation is this station's to fix or the docs station's (E07) needs the owner.
11. _[attacker/test-bloat]_ Whether clp_only_smoke.rs was created as a deliberate belt-and-braces guard against a future re-gating of conformance.rs, in which case the binary stays despite its assertion being already covered and only its false module doc and its fixture copy need fixing.
12. _[defender/architecture-07]_ Owner picks the residue's direction: keep the canonical mapping unconditional and correct README.md:106-107, or name the CLP codes in ffi::clp and gate each half - the latter also makes BasisStatus::to_highs_code/from_highs_code and the legacy-checkpoint agreement test (basis_status.rs:280-289) HiGHS-only.
13. _[defender/over-engineering-03]_ Owner must supply the activating milestone for the CLP hot-start acquire/solve pair, or rule it retired at the next licensed public-API break; the register admits an entry only with both an owner and a consuming milestone, and this station cannot invent one.
14. _[defender/test-bloat-03]_ Owner call on the four integration binaries: either give cobre-solver a tests/common shared-fixture module (the cobre-sddp idiom, no feature gate) or accept that a test-support-gated fixture makes the documented clp-only invocations (CONTRIBUTING.md:71, crates/cobre-solver/README.md:84) run zero cobre-solver integration tests.

## 2. Round plan

Options are capped at four per question; the recommended option is listed first. Items that get
**no round**: the sanctioned shared-memory hierarchy and the cut-sync dup-of (1.3) — offering a
question would invite a re-raise; the basis-validation asymmetry (intended behaviour) likewise.

| Round | Kind | Subject | Options |
| -- | -- | -- | -- |
| R1 | partI-first | CD-079 StageTemplate sheds n_state/n_transfer/n_dual_relevant/n_hydro/max_par_order (types.rs 270-297), B (A-risk), advances-1, propagates freeze.rs / trait_def.rs / profiled.rs | accept · amend fields · downgrade · defer |
| R2 | severity-batch | remaining Sev-B entries: CD-078, OD-033, OD-035, PD-032, PD-033, PD-034, TD-037 | accept all · downgrade (name) · reject (name) · defer (name + trigger) |
| R3 | severity-batch | Sev-C entries: CD-075, CD-076, CD-077, CD-080, CD-081, OD-032, OD-034, TD-035, TD-036, TD-038, TD-039, TD-040 | accept all · downgrade/reject (name) · defer (name + trigger) · other |
| R4 | informational | capability trait (SC-ARCH-009) | keep informational · promote · drop · defer |
| R5 | needs-human | CD-080 residue direction (defender architecture-07) | keep mapping unconditional + fix README · gate each half · other |
| R6 | needs-human | OD-033 CLP hot-start pair (defender over-engineering-03 + attacker OE#3) | retire at next API break · register with milestone · wire onto re-solve · defer |
| R7 | needs-human | TD-037 four integration binaries (defender test-bloat-03) | tests/common shared module · test-support gate · other |
| R8 | needs-human | OD-032 is_homogeneous (attacker OE#2) | delete · register with milestone |
| R9 | needs-human | CD-075 fix-shape (attacker ARCH#4): keep workspace forbid + checker vs single table with per-crate allow | keep forbid + checker · single table · other |
| R10 | needs-human | CD-076 timing (attacker ARCH#3): now-fix vs purification rider | now-fix · rider · other |
| R11 | needs-human | TD-039 clp_only_smoke (attacker TB#2): retire vs deliberate guard | retire · keep as guard, fix doc + fixture |
| R12 | needs-human | HiGHS wall-clock retry branching vs CLP determinism property (attacker OE#1) | intended, record · determinism follow-up · other |
| R13 | needs-human | I.3-8 landing: shed into engine now vs wait for the 0b carve (attacker ARCH#1) | E9 decides · shed now · wait for carve |
| R14 | needs-human | LocalCommKind placement traits.rs vs factory.rs (attacker ARCH#2) | record informational · drop · other |
| R15 | needs-human | CLP library capabilities + retained-mirror contract (attacker PERF#1, #2) | route to E10 · other |
| R16 | needs-human | stale backend-testing.md citation in local_conformance.rs (attacker TB#1) | route to E07 · fix here · other |

## 3. Decision record

**Decision: ratified** — 2026-09-18, main session, baseline `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`. Presented 20, accepted 20, amended 0, downgraded 0 (the one calibration downgrade CD-075 B→C was accepted as the house rating), rejected 0, deferred 0, overridden 0; informational kept 1; needs-human answered 14/14. Every answer below is the option label chosen verbatim; the rationale column is the chosen option's description.

| ID | Round | Decision | Severity | Alignment | Rationale (owner) | Trigger / override / handoff |
| -- | -- | -- | -- | -- | -- | -- |
| OD-032 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | R8: Delete |
| TD-035 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | E8 test-corpus |
| TD-036 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | E8 test-corpus |
| CD-075 | R3 | accept | C (reviewer: B) | neutral | Accept all (Recommended) — as recorded | R9: Keep forbid + checker |
| CD-076 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | R10: Now-fix reword |
| CD-077 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | - |
| CD-078 | R2 | accept | B | neutral | Accept all (Recommended) — as recorded | - |
| CD-079 | R1 | accept | B (A-risk) | advances-1 | Accept (Recommended) — as recorded | E9: I.3-8 dispositions unchanged |
| CD-080 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | R5: Keep mapping unconditional; fix README |
| CD-081 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | - |
| OD-033 | R2 | accept | B | neutral | Accept all (Recommended) — as recorded | R6: Retire at the next licensed public-API break |
| OD-034 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | - |
| OD-035 | R2 | accept | B | neutral | Accept all (Recommended) — as recorded | - |
| PD-032 | R2 | accept | B | neutral | Accept all (Recommended) — as recorded | E10: 4t, UNMEASURED, kept |
| PD-033 | R2 | accept | B | neutral | Accept all (Recommended) — as recorded | E10: 4t, UNMEASURED, kept |
| PD-034 | R2 | accept | B | neutral | Accept all (Recommended) — as recorded | E10: 4t, UNMEASURED, kept |
| TD-037 | R2 | accept | B | neutral | Accept all (Recommended) — as recorded | E9 cross-ref I.3-8; E8 test-corpus; R7: tests/common shared module |
| TD-038 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | E8 test-corpus |
| TD-039 | R3 | accept | C | neutral | Accept all (Recommended) — as recorded | E8 test-corpus; R11: Retire the binary |
| TD-040 | R3 | accept | C | advances-1 | Accept all (Recommended) — as recorded | E9 cross-ref I.3-8; E8 test-corpus |

### 3.1 Fix-shape directions decided at the gate

| ID | Round | Choice | Direction |
| -- | -- | -- | -- |
| CD-080 | R5 | Keep mapping unconditional; fix README | the canonical BasisStatus mapping stays feature-independent; name the CLP codes in ffi::clp for symmetry and correct README.md:106-107 — smallest blast radius, no test becomes HiGHS-only |
| OD-033 | R6 | Retire at the next licensed public-API break | OD-033 stays accepted at Sev B; the fix-shape becomes 'delete the acquire half (shim, extern, wrapper, harness) at the next licensed API break' — no reserved-seam entry, no wiring |
| TD-037 | R7 | tests/common shared module | the four in-src copies collapse into one in-crate cfg(test) module; the four integration binaries share a tests/common fixture module (the cobre-sddp idiom, testing-architecture §5.1), no feature gate, so the documented clp-only invocations keep running the integration tests; test_support stays the seam for cross-crate consumers |
| OD-032 | R8 | Delete | leftover, not a planned heterogeneous-layout guard; drop the predicate and its four unit tests, the eventual caller re-adds it beside its use |
| CD-075 | R9 | Keep forbid + checker | the workspace `unsafe_code = forbid` stays the unoverridable default (CLAUDE.md hard rule); a scripts/ci checker diffs each per-crate `[lints]` copy against the workspace tables so drift is red CI; the four FFI crates keep their audited overrides |
| CD-076 | R10 | Now-fix reword | same treatment as stochastic CD-064: a pure doc reword in the generic register, no type or behaviour change, independent of the phase-1 shed; CD-076 stays neutral, Sev C |
| TD-039 | R11 | Retire the binary | clp_only_smoke.rs is retired; the clp-gated section of conformance.rs is the clp-only guard (testing-architecture §5.1: a solver-linked binary must earn its link cost) |

### 3.2 Informational

- SC-ARCH-009 / architecture-09 (R4): **Keep informational** — no id promoted; the III.6 cross-reference and the one-consumer hold stand.
- LocalCommKind placement (R14): **Record informational** — added to the section's Informational block; no id.

### 3.3 Worker needs-human answers

| # | Round | From | Question | Answer | Recorded as |
| -- | -- | -- | -- | -- | -- |
| 1 | R13 | attacker/architecture | I.3-8 landing: shed into the engine now vs wait for the 0b carve | **E9 decides** | recorded on the E9 handoff block; the alignment epic owns phase sequencing (Wave-4 setup redesign precedes 0a; the 0b carve); CD-079's retire disposition is unchanged either way |
| 2 | R14 | attacker/architecture | LocalCommKind placement (traits.rs vs factory.rs) | **Record informational** | a no-id informational note in the station section: a placement question for whoever activates the shared-memory seam (mirror :79); the seam itself stays untouched |
| 3 | R10 | attacker/architecture | does the L0 vocabulary ban bind now or at phase 1 (CD-076) | **Now-fix reword** | same treatment as stochastic CD-064: a pure doc reword in the generic register, no type or behaviour change, independent of the phase-1 shed; CD-076 stays neutral, Sev C |
| 4 | R9 | attacker/architecture | give up workspace `unsafe_code = forbid` for a single lint table (CD-075) | **Keep forbid + checker** | the workspace `unsafe_code = forbid` stays the unoverridable default (CLAUDE.md hard rule); a scripts/ci checker diffs each per-crate `[lints]` copy against the workspace tables so drift is red CI; the four FFI crates keep their audited overrides |
| 5 | R12 | attacker/over-engineering | HiGHS retry ladder branches on wall-clock time while the CLP ladder documents no time-dependent branching | **Determinism follow-up** | not a station finding (no attacker candidate was raised): time-dependent branching on the HiGHS retry ladder (highs/retry.rs:40/:52/:84) can change the solve path run-to-run; routed to the E11 reconciliation as a reproducibility follow-up for the HiGHS backend, no id minted here |
| 6 | R8 | attacker/over-engineering | is_homogeneous: planned guard or leftover (OD-032) | **Delete** | leftover, not a planned heterogeneous-layout guard; drop the predicate and its four unit tests, the eventual caller re-adds it beside its use |
| 7 | R6 | attacker/over-engineering | CLP hot-start half: wire or register (OD-033) | **Retire at the next licensed public-API break** | OD-033 stays accepted at Sev B; the fix-shape becomes 'delete the acquire half (shim, extern, wrapper, harness) at the next licensed API break' — no reserved-seam entry, no wiring |
| 8 | R15 | attacker/performance | does the CLP API expose an index-scoped bound writer and a bulk basis-status accessor (PD-032 / PD-034) | **Route to E10** | rides on the perf-queue rows as a shaping question; E10 measures at 4t and the fix ticket answers it against the CLP headers |
| 9 | R15 | attacker/performance | must the retained column-major mirror stay a full merged mirror (PD-033) | **Route to E10** | rides on the perf-queue row as a shaping question; E10 measures at 4t and the fix ticket decides the retained-mirror contract |
| 10 | R16 | attacker/test-bloat | stale backend-testing.md citation in cobre-comm/tests/local_conformance.rs:4 | **Route to E07 docs station** | a doc-drift item for the build-ci/docs station's doc-integrity lens; recorded on the E7 handoff as an observation (like the sddp.md:446 drift), no id here |
| 11 | R11 | attacker/test-bloat | clp_only_smoke.rs: deliberate guard or redundant (TD-039) | **Retire the binary** | clp_only_smoke.rs is retired; the clp-gated section of conformance.rs is the clp-only guard (testing-architecture §5.1: a solver-linked binary must earn its link cost) |
| 12 | R5 | defender/architecture-07 | CD-080 residue direction | **Keep mapping unconditional; fix README** | the canonical BasisStatus mapping stays feature-independent; name the CLP codes in ffi::clp for symmetry and correct README.md:106-107 — smallest blast radius, no test becomes HiGHS-only |
| 13 | R6 | defender/over-engineering-03 | OD-033 activating milestone or retirement | **Retire at the next licensed public-API break** | OD-033 stays accepted at Sev B; the fix-shape becomes 'delete the acquire half (shim, extern, wrapper, harness) at the next licensed API break' — no reserved-seam entry, no wiring |
| 14 | R7 | defender/test-bloat-03 | TD-037 home for the four integration binaries | **tests/common shared module** | the four in-src copies collapse into one in-crate cfg(test) module; the four integration binaries share a tests/common fixture module (the cobre-sddp idiom, testing-architecture §5.1), no feature gate, so the documented clp-only invocations keep running the integration tests; test_support stays the seam for cross-crate consumers |

### 3.4 Presented, no decision taken

- Shared-memory communicator trait hierarchy — ratified reserved seam (mirror :79, OD-001 KEEP-RESERVED); no question offered.
- Superseded cut-sync public methods — dup-of (mirror :334, CD-019), E5 handoff, no id; no question offered.
- HiGHS-loud / CLP-silent basis-validation asymmetry — intended behaviour; no question offered.

## 4. Handoffs after the gate

- **E9** — per-field dispositions unchanged (all five retire); owner question recorded: I.3-8 landing (shed now vs wait for the 0b carve) — E9 decides
- **E10** — PD-032, PD-033, PD-034 kept, layout 4t, UNMEASURED; shaping questions attached (index-scoped bound writer / bulk basis accessor; retained-mirror contract)
- **E7** — cut_nz_per_col blind spot unchanged (CD-081 accepted); observation added: stale backend-testing.md citation in cobre-comm/tests/local_conformance.rs:4
- **E5** — cut-sync dup-of unchanged by the gate
- **E8** — TD-037 home = tests/common shared module (no feature gate); TD-039 = retire the binary
- **E11** — reproducibility follow-up: HiGHS retry ladder wall-clock branching (highs/retry.rs:40/:52/:84), no id

**Gate: RETURNED 2026-09-18** — baseline `077dbe2c`; accepted 20, amended 0, downgraded 0, rejected 0, deferred 0, overridden 0. The cobre-sddp station is unblocked.
