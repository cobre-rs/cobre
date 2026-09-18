# cli-python attacker dispatch log

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (register pin; the ticket text's `a136840d` is the superseded pin — every figure the
workers were handed was re-measured at the pin, see `attacker-prompt.md` § Frozen figures). Dispatched:
2026-09-18. Agent type: `adversarial-attacker` (Opus, read-only). Dispatcher: the main session, sole writer of
every artifact under `stations/cli-python/`.

## Partition check (before dispatch)

The three-way sweep partition is the one proven total and disjoint in `inventory.json` (`coverage`: findCount 30,
assignedCount 30, `unassigned = []`, `doubleAssigned = []`, `phantom = []`), re-asserted by
`InventoryTests.test_partition_is_the_declared_three_sub_surfaces` and rendered verbatim into the prompt's
`## Manifests` section by the prompt generator:

```
S6a 8 files 6,149 lines   S6b 11 files 5,652 lines   S6c 11 files 4,692 lines   = 30 / 16,493
```

Every path a worker may anchor lies in exactly one manifest; the test-bloat lens additionally reads `crates/cobre-cli/tests`
(14 binaries) and `crates/cobre-python/tests` (37 files), split per cell as the lens block states.
The ticket's S6b list named `commands/report.rs` and `commands/summary.rs`; neither exists at the pin (removed by
`797ba443`), so S6b is the 11-file manifest above — recorded as a ticket deviation, not a gap.

## Prompt and validator (before dispatch)

One shared contract, `attacker-prompt.md` (generated from inventory.json / wave-dispositions.json / partI-handoff.json;
adapted from the core-io template through the sddp station's refinement): preamble + fourteen rules (read-only;
scratch-file hand-over `WRITTEN <bytes> <path>`; SYMBOL ANCHORS ONLY with the facade's `{path, line}` exception and
the out-of-station `dupOf` re-route; evidence-as-command; the L2 destination rule with `waveRef` CD-025 / CD-029;
the Part-I I.5 tag; the reserved-seams pre-check on every over-engineering candidate; the station's byte-neutrality
bar; prose fix-shapes; perf layout never a number; empty-is-legitimate with an explicit `cleanVerdict`; never
re-raise; L4 layering → `conflicts`; cross-sub-surface duplication raised once), the alignment vocabulary, the
frozen-figure table (16 rows: ticket figure → pin figure → command), the do-not-raise set (settled with closing
commits, five SUPERSEDED premises, the sanctioned seams, the disowned ids, the four live dispositions as a
sharpen-only table), four lens blocks, the three manifests and the envelope. The gate is
`tools/validate-envelope.py --role attacker --station cli-python` (shape) plus the station rules the shared validator
does not know: symbol-only anchors inside the cell's manifest (line anchors only for `crates/cobre/src/lib.rs` and
`crates/cobre/Cargo.toml`), the `dupOf` re-route for out-of-station candidates, anchors resolved at the pin through
`check-anchors.py` over a rendered stub, the prior-register screen through `check-reraise.py`, and the field rules the
ticket's merge step prescribes (`partIRef: "I.5"` stamped on every cobre_sddp-coupling candidate, `waveRef` stamped
on every output-orchestration / boundary candidate, `reservedSeamsCheck` required on every over-engineering
candidate, `measured: false` + `unmeasured` + `measurementRequest.layout` on every performance candidate).

Gate policies fixed on the first wave (recorded so the verdicts are reproducible):

- **Anchor rejection is per anchor, not per candidate.** A line-only anchor (outside the facade exception), an
  anchor outside the three crates, an anchor outside the cell's manifest (rule 14 exceptions: S6c may anchor
  `cobre-python/src/run.rs::reconcile_boundary_policy` and, for the test-bloat lens, its test modules) or a symbol
  that does not resolve at the pin is moved to the candidate's `citedContext` with its reason; the candidate is
  dropped `anchor-missing` only when no anchor survives.
- **`dupOf` = every real anchor outside the station.** A candidate whose anchors all lie in another station's
  crates (or that the worker pre-tagged `dupOf`) is written to the lens file's `dupOf` array with the target station
  and excluded from `candidates`; the re-route is listed below.
- **Stamps are mechanical, never prose edits.** The gate stamps `partIRef`, `waveRef`, `measured`,
  `unmeasured.tag`, `measurementRequest.layout` (derived from `claimType`) when the worker left them null and the
  rule's vocabulary fires; every stamp is recorded per cell. Fix-shape prose is never edited: a CD-025 fix-shape
  naming a cobre-sddp / cli-local home without `conflicts` is annotated `gateNotes` for ingest.
- **Rule-7 and rule-10 omissions drop the candidate `needs-human`**, not the cell: an over-engineering candidate
  without `reservedSeamsCheck`, a performance candidate without a mechanism.
- **`check-reraise` hits.** An explicit-id or title-overlap hit on a candidate without `reRaiseOf` drops it
  `settled`; an anchor-overlap hit against a mirror section is annotated `retiredOverlaps` and the candidate is kept
  (the two mirror sections that record CD-025 and CD-029 fire on every sharpen of them).
- **Envelope station spelling.** The ticket's `station: "cobre-cli+cobre-python+facade"` is carried as
  `stationLabel`; `station` is the slug `cli-python` the validator and every station artifact use. The merged lens
  file carries `subStation: "S6"` (the validator's token) and `cells` for the three sub-surfaces.

## Dispatch schedule — 12 cells, 6 concurrent workers, pipelined by lens

| wave | lenses | cells | dispatched when |
| ---- | ------ | ----- | --------------- |
| 1 | architecture + performance | S6a S6b S6c × 2 | at start (six concurrent) |
| 2 | over-engineering | S6a S6b S6c | when the performance triple had returned (six running) |
| 2 | test-bloat | S6a S6b S6c | when over-engineering.S6c returned (six running) |

Six at a time (the manifests are 8 / 11 / 11 files and the prompt is a third of the sddp station's, so six
concurrent cells do not starve); a lens is dispatched as a triple as soon as capacity frees, so a lens is reproducible
and a crashed lens is resumable without re-running clean cells; a shape-invalid envelope is re-dispatched once with the validator error quoted (architecture.S6b: two errors, the rewrite passed); no worker sees another worker's output; each cell writes exactly one scratch file
`/tmp/cli-attackers/out/<lens>.<sub>.json`, copied verbatim to `raw/` in scratch and gated into the merged
`candidates-<lens>.json`; a cell whose envelope fails the gate is re-dispatched exactly once with the validator
error quoted, a second failure is `needs-human`, never hand-repaired. No timing command was executed by this
ticket: every performance candidate is queued UNMEASURED to the cross-cutting sweep.

## Coverage matrix (4 lenses × 3 sub-surfaces)

|                  | S6a writer/run boundary | S6b diagnostics + shell | S6c bindings + facade |
| ---------------- | ----------------------- | ----------------------- | --------------------- |
| architecture     | 7/7/3                   | 7/5/4                   | 7/10/2                |
| performance      | 8/8/3                   | 2/8/1                   | 8/10/2                |
| over-engineering | 6/8/2                   | 6/13/4                  | 2/6/2                 |
| test-bloat       | 8/6/3                   | 10/7/2                  | 5/6/2                 |

n/p/h = candidates kept / positives / needs-human. A cell reading 0/0/0 is a FAILED cell; a clean cell reads 0/1+/0 with an explicit clean verdict in the lens file.

## Dispatch records

### Wave 1 — architecture + performance (dispatched 2026-09-18, six concurrent `adversarial-attacker` Opus workers)

| cell | wave | directive focus | scratch path | validator |
| ---- | ---- | --------------- | ------------ | --------- |
| architecture.S6a | 1 | Q1 engine seam on the run path; Q2 run-path projection callers; Q3 non-root MPI path (CD-002 sharpen); Q4 output orchestration (CD-025 sharpen) | /tmp/cli-attackers/out/architecture.S6a.json | pass |
| architecture.S6b | 1 | Q2 validate-path projection callers; Q5 validate mirror (owner; CD-029 sharpen); Q6 shell: error.rs / summary.rs / progress.rs / templates.rs | /tmp/cli-attackers/out/architecture.S6b.json | pass (re-dispatched once; first envelope: $.candidates[3].partIRef: not an I.3-n / I.5 reference (the worker wro…) |
| architecture.S6c | 1 | Q1 bindings' half of the seam (study.rs Study, run.rs run/run_via_study); Q5 io.rs half of the validate mirror; Q6 binding surface twins + facade | /tmp/cli-attackers/out/architecture.S6c.json | pass |
| performance.S6a | 1 | targets 1 output writing, 2 thread resolution, 3 broadcast path (collective 2x2) | /tmp/cli-attackers/out/performance.S6a.json | pass |
| performance.S6b | 1 | target 5 shell: progress.rs rendering, summary.rs formatting, validate.rs prepare_stochastic contract | /tmp/cli-attackers/out/performance.S6b.json | pass |
| performance.S6c | 1 | target 4 Python boundary: results.rs readers, convert.rs, model.rs / policy.rs twins, study.rs GIL | /tmp/cli-attackers/out/performance.S6c.json | pass |

### Wave 2 — over-engineering + test-bloat (dispatched 2026-09-18, six concurrent `adversarial-attacker` Opus workers)

| cell | wave | directive focus | scratch path | validator |
| ---- | ---- | --------------- | ------------ | --------- |
| over-engineering.S6a | 2 | broadcast.rs placement; run-path wrapper layers; *_if_any (CD-025 sharpen only); #[allow] / enum census | /tmp/cli-attackers/out/over-engineering.S6a.json | pass |
| over-engineering.S6b | 2 | wrapper layers validate/schema/version/init; module earning banner/version/templates/summary; census | /tmp/cli-attackers/out/over-engineering.S6b.json | pass |
| over-engineering.S6c | 2 | facade crate (rule 7 + ARCHITECTURE.md:102-106 + needsHuman); module earning version/schema/convert/lib; study.rs / results.rs wrappers; census | /tmp/cli-attackers/out/over-engineering.S6c.json | pass |
| test-bloat.S6a | 2 | Q1 inline vs integration on the run path; Q2 harness duplication across cli_run* + python run/Study suites; Q3 parity triangle | /tmp/cli-attackers/out/test-bloat.S6a.json | pass |
| test-bloat.S6b | 2 | Q1 inline giants summary.rs 61 / error.rs 28 / progress.rs 22; Q2 cli_validate / cli_schema / cli_smoke / init harness; Q3 n/a | /tmp/cli-attackers/out/test-bloat.S6b.json | pass |
| test-bloat.S6c | 2 | Q1 pytest corpus tiers + Rust #[test] census (workspace-excluded, CI-visible); Q2 conftest / _cobre_cli duplication; Q3 parity triangle (python side) | /tmp/cli-attackers/out/test-bloat.S6c.json | pass |

## Prior-register screen and re-routes (what was dropped and why)

| candidate title | cell | action | reason |
| --------------- | ---- | ------ | ------ |
| The validate phase driver is hand-mirrored across the two front ends and has already drifted twice: the CLI an | architecture.S6c | merged | rule 14: the validate mirror is raised once; folded into architecture.S6b 'The pre-solver phase driver is written twice over one engine-owned PrepPhase: th' (anchors + evidence carried) |

Actions are exactly: settled | sanctioned | dup-of | merged | anchor-missing | needs-human.

## Gate stamps (mechanical field fills, per rule)

| cell | field | value | candidate | why |
| ---- | ----- | ----- | --------- | --- |
| architecture.S6a | partIRef | I.5 | The positional carrier is nested: a six-slot LoadedCase alias feeds the ten-slot destructu | cobre_sddp-coupling vocabulary in the title / fix-shape (rule 6) |
| architecture.S6b | waveRef | CD-029 | The pre-solver phase driver is written twice over one engine-owned PrepPhase: the CLI extr | boundary-reconciliation / PrepPhase / validate phase-driver claim (rule 5) |
| architecture.S6c | waveRef | CD-029 | The validate phase driver is hand-mirrored across the two front ends and has already drift | boundary-reconciliation / PrepPhase / validate phase-driver claim (rule 5) |
| architecture.S6c | partIRef | I.5 | write_policy_checkpoint in the bindings re-declares the engine's cost-scale default as a b | cobre_sddp-coupling vocabulary in the title / fix-shape (rule 6) |

## Anchors demoted to citedContext

- **architecture.S6c** — The validate phase driver is hand-mirrored across the two front ends and has already drift: `crates/cobre-cli/src/commands/validate.rs::execute` (anchor outside the S6c manifest: crates/cobre-cli/src/commands/validate.rs (S6b)); `crates/cobre-cli/src/commands/validate.rs::ValidateErrorOutput` (anchor outside the S6c manifest: crates/cobre-cli/src/commands/validate.rs (S6b))

## Needs-human items surfaced for the gate

- **architecture.S6a** — C1: when the phase plan gets a single owner, should the no-op arm's two behaviours converge? The CLI writes 'Training disabled, simulation disabled' to stderr and the bindings return a RunSummary with converged: false, iterations: 0, lower_bound: 0.0. Both are public surface, and neither is covered by the file-set parity test because neither writes a file, so unifying them is a deliberate API deci
- **architecture.S6a** — C6: does the owner accept a cobre-sddp (L3) home for the solver-stats fold? The register's destination rule makes a cobre-sddp home `conflicts` for the WRITER boundary; this fold is outside that boundary (its input types are declared in cobre-sddp `src/solver_stats.rs`, so an L2 home would make cobre-io depend on an engine crate). The distinction is checked and stated, not assumed, and needs confi
- **architecture.S6a** — C5: is renaming `StudySetup::from_broadcast_params` in scope for this station's follow-up, or does it belong to the sddp station that owns cobre-sddp's public API? The caller route through the wire type is this station's; the constructor name is not.
- **architecture.S6b** — CD-029 destination, restated from the disposition record because this station's first candidate assumes it: is cobre-sddp's PrepPhase the confirmed home for the folded boundary check (its fourth phase), rather than cobre-io? The fix shape for the --json boundary gap depends on an engine-owned reconcile function that returns a phase kind, so the answer determines where that kind is defined.
- **architecture.S6b** — Should cobre-cli gain a [lib] target? It would make the crate's pub surface and its four cobre_cli:: doc examples compiler-verified, closing the doctest gap structurally; it would also publish the CLI's internal renderers, error enum and progress types as a contract the workspace must then maintain. The alternative is to de-annotate the examples and accept that no example in this crate is ever com
- **architecture.S6b** — May the CLI's --json kind string "CaseValidationError" be retired in favour of the cobre-io/Python vocabulary ("ConstraintError" etc.)? No test, script or binding references it, so the change is free mechanically, but it is a user-visible field in a documented machine-readable contract (validate.rs:89-93) and some downstream consumer may match on it.
- **architecture.S6b** — Where should the LoadError -> stable kind map live? cobre-io (L2) owns LoadError and both front ends depend on it, which is why the second candidate names it, but the station's committed destination rule is written for output orchestration specifically. Confirm that the same L2 destination binds a shared error-classification map, or name a different owner.
- **architecture.S6c** — The facade crate crates/cobre: keep it as a documented reserved seam, or retire it until the convenience re-export actually exists? Rule 7 pre-check: sanctioned by ARCHITECTURE.md:102-106 ('Currently an empty skeleton ... reserved for a future single-dependency convenience re-export'), not-found in the mirror's reserved-seam register, and absent from the L0-L4 table in target-layering-brief.md § 1
- **architecture.S6c** — Under the Phase-0a Engine seam, does cobre.Study become generic over the engine (its setup field moving behind the enum), or does it stay the SDDP study with a dispatching constructor above it? Both satisfy Part IV.4 and neither is derivable from the code. The choice decides whether cobre.Study is a public API break for existing Python callers, and it also decides where the typed admission gate fo
- **performance.S6a** — Rooted gather at L0: the two simulation gathers need a gatherv the cobre-comm Communicator trait does not expose (it has allgatherv, allreduce, broadcast, barrier, rank, size). Adding it grows the L0 surface for two present consumers, both in this file. Is that acceptable, or should the CLI keep allgatherv and accept that every non-root rank allocates and rebuilds a payload it discards?
- **performance.S6a** — Is the shared-filesystem re-read on non-root ranks a ratified deployment assumption? The setup.rs module doc states the behaviour with no rationale and no docs/design entry covers it, so replacing the per-rank hydro-model refit (and the per-rank policy-checkpoint read in simulation-only mode) with a rank-0 read plus a broadcast is a trade between collective payload size and filesystem load that on
- **performance.S6a** — Deferring the rank-0 training-artefact write so it overlaps rank 0 entering the simulation phase changes the stderr progress ordering (Writing training outputs appears interleaved with simulation progress) and makes the checkpoint write compete for I/O bandwidth with the simulation drain writes. Is the stderr ordering free to change, given the byte-neutrality bar covers the output tree and the val
- **performance.S6b** — Sequencing for the duplicate boundary resolve in validate.rs: patch it now as a cli-local parameter thread, or hold it and let CD-029's approved fold of the boundary check into cobre-sddp PrepPhase subsume it? The fold would make the requirements a phase input resolved once, so taking both in order means touching the same call chain twice. Rule 5 already holds the fold destination itself as a need
- **performance.S6c** — Is the dict-shaped cobre.results.load_simulation still a supported bulk reader, or is load_simulation_arrow now the intended path for large result trees? The answer decides whether the per-cell Arrow dispatch in read_parquet_partition_into is worth fixing or whether the dict reader should instead be documented as a small-result convenience, and it is a user-facing API posture question the station 
- **performance.S6c** — May load_results bind one converted object to both the manifest and metadata keys of the training dict, making them the same Python object and aliasing on mutation, or must they stay two equal-but-distinct dicts? Collapsing the double conversion is only byte-neutral in values, not in object identity, so the owner decides whether identity is part of the published contract.
- **over-engineering.S6a** — BroadcastNodeGraph in crates/cobre-cli/src/commands/broadcast.rs: keep it as a documented reserved seam, or retire it? It has zero consumers workspace-wide plus roughly 110 lines of dedicated round-trip tests, and its doc reserves it for a future caller that would transport the node graph explicitly. The mirror contains the opposite precedent for an identically shaped artifact: at docs/design/rese
- **over-engineering.S6a** — Should the CLI's Config wire mirror move beside crates/cobre-io/src/broadcast.rs, which already owns the identical externally-tagged mirror for ScalarParameter and whose module doc names the cobre-cli copy as the same pattern? The placement candidate above proposes only the minimal in-crate relocation, because the L2 move shares the Config projection with CD-004 (ratified at the sddp gate) and the
- **over-engineering.S6b** — Should `cobre validate --json` emit an error object when boundary reconciliation - the fourth phase the module doc lists - fails? The doc promises an object naming the first failing phase, today that path writes nothing to stdout, and no test pins either behaviour, so adding it is a deliberate extension of the `--json` contract on an untested path rather than a byte-neutral cleanup.
- **over-engineering.S6b** — May the four `#[cfg(test)] pub fn format_*_string` renderers in summary.rs be deleted in favour of private `*_lines` helpers? The rendered output is unchanged either way, but the 61 inline tests that call them have to be repointed, and that test corpus belongs to the sibling test-bloat cell, so the two fixes must be sequenced together.
- **over-engineering.S6b** — The facade crate: keep `crates/cobre` as a documented reserved seam or retire it until the re-export actually exists? It is sanctioned by the crate map and has no mirror entry, and while it re-exports nothing the two `use cobre::...` doc examples in banner.rs and templates.rs cannot be made to resolve.
- **over-engineering.S6b** — Does cobre-cli want a library target and a `cargo test --doc` step so its documented examples are compiled, or should the examples stop presenting themselves as importable code? The manifest declares only `[[bin]]` with `doc = false`, so today the choice is being made by omission.
- **over-engineering.S6c** — The umbrella facade crate crates/cobre: keep it as a documented reserved seam and add the missing row to docs/design/reserved-seams-and-deferred-debt.md as an E11 mirror write-back, or retire the crate from the workspace members list until the convenience re-export it reserves actually exists? Two facts bear on the decision and neither is the attacker's to weigh. First, ARCHITECTURE.md:102-106 doc
- **over-engineering.S6c** — The three cobre.model stub twins (EnergyContract, PumpingStation, NonControllableSource): are they a reserved seam awaiting a full field set, in which case the mirror needs a row and the fix is to fill the getters, or is the Python model surface deliberately limited to the four entity types the SDDP read path needs, in which case the three should leave cobre.model and PySystem should return dicts?
- **test-bloat.S6a** — {"question": "Does section 5.11's instruction to keep the pytest output-parity suite unchanged freeze those files against harness consolidation and against removing a strictly subsumed test, or does it only exempt them from the Layer-1 binary migration while leaving fixture hygiene in scope?", "why": "Candidates S6a-TB-05 and S6a-TB-06 touch files section 5.11 names. Both are assertion-preserving,
- **test-bloat.S6a** — {"question": "Is the run-path binary consolidation sanctioned now, or should it wait for the section 5.1 Layer-1 migration to a single integration-test entry point?", "why": "docs/design/testing-architecture.md carries Proposal status for everything outside section 5.2, including the canonical layout and its test-LOC threshold, while .claude/rules/testing.md section Cost discipline is a standing c
- **test-bloat.S6a** — {"question": "Should the hand-pinned 1dtoy numbers be retired in favour of the live CLI/Python parity derivation, or kept deliberately as an offline regression guard for environments with no built wheel?", "why": "S6a-TB-04 assumes the live derivation is the authority. If the pinned copy is intentional cover for environments where the pytest parity suite cannot run, the right outcome is one pinned
- **test-bloat.S6b** — {"question": "Should cobre-cli grow a library target so its doc examples compile and run, or should the seven inert examples be deleted and their claims kept in the inline test modules?", "why": "Candidate TB-S6b-05 establishes that the seven executable doctest blocks never run and would not compile. Deleting them is the byte-neutral, in-scope fix and is what the candidate recommends. Adding a [li
- **test-bloat.S6b** — {"question": "Is the divergence between the three copies of the valid-case fixture deliberate per suite, or incidental?", "why": "Candidate TB-S6b-03 proposes one parameterised builder in tests/common. The three copies differ in forward passes, iteration limit, stage count and openings. The consolidation is safe either way because the fix keeps today's values at each call site, but knowing which d
- **test-bloat.S6c** — Anchor tooling: rule 3 restricts a symbol anchor to a declared fn, struct, enum, trait, type, const, static, mod or impl name, and tools/check-anchors.py compiles exactly that Rust vocabulary, so a pytest def or class name resolves as symbol-missing. The test-bloat corpus for this cell is mostly pytest files. Every candidate here therefore leads with a Rust symbol that resolves, and the pytest def
- **test-bloat.S6c** — convert.rs carries an in-code comment asserting that a #[cfg(test)] module cannot link an interpreter, and test_convert.py repeats it in a module docstring. Both are contradicted by errors.rs and by the feature-off CI invocation. Correcting an in-code rationale and relocating coverage are separable decisions, so the owner should say whether candidate one lands as a comment correction only, as a co

## Cross-cell overlaps inside a lens (handed to ingest, not merged here)

Pairs of kept candidates from different sub-surfaces under the same lens that share an anchor symbol or whose titles overlap (Jaccard ≥ 0.35). The ingest/defender pass decides whether each pair is one finding or two adjacent ones.

| lens | cell A | title A | cell B | title B | shared symbols | title Jaccard |
| ---- | ------ | ------- | ------ | ------- | -------------- | ------------- |
| test-bloat | test-bloat.S6a | Seven of the nine run-path test binaries hold exactly one #[test], and | test-bloat.S6b | cobre-cli has no tests/common at all: the binary invocation helper is  | cobre | 0.12 |
| test-bloat | test-bloat.S6a | test_study.py and test_run.py declare no fixtures and re-train or re-r | test-bloat.S6c | conftest.py owns one fixture, so copy-the-case and find-the-case are e | cli_binary | 0.05 |
| test-bloat | test-bloat.S6a | test_study.py and test_run.py declare no fixtures and re-train or re-r | test-bloat.S6c | Four redundant full solver runs pay for preconditions a module-scoped  | run_output | 0.15 |
| test-bloat | test-bloat.S6a | The bindings' inline test module runs seven full run_via_study studies | test-bloat.S6c | conftest.py owns one fixture, so copy-the-case and find-the-case are e | copy_dir_all | 0.02 |
| test-bloat | test-bloat.S6a | The pytest parity files each rebuild the same harness: _make_case_with | test-bloat.S6c | conftest.py owns one fixture, so copy-the-case and find-the-case are e | cli_binary | 0.06 |
| test-bloat | test-bloat.S6b | Six of the seven template tests assert compile-time literals or restat | test-bloat.S6c | Seven py_to_json_value unit claims are routed through the eleven-phase | tests | 0.05 |
| test-bloat | test-bloat.S6b | Six of the seven template tests assert compile-time literals or restat | test-bloat.S6c | testing-architecture section 5.11 still asks for the bindings crate's  | tests | 0.05 |
| test-bloat | test-bloat.S6b | Three of the six engine types the summary module re-exports have zero  | test-bloat.S6c | Seven py_to_json_value unit claims are routed through the eleven-phase | tests | 0.00 |
| test-bloat | test-bloat.S6b | Three of the six engine types the summary module re-exports have zero  | test-bloat.S6c | testing-architecture section 5.11 still asks for the bindings crate's  | tests | 0.03 |

## Per-cell records

```json
{
 "performance.S6a": {
  "cell": "performance.S6a",
  "wave": 1,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 8,
  "candidates": 8,
  "dupOf": [],
  "positives": 8,
  "needsHuman": 3,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "performance.S6b": {
  "cell": "performance.S6b",
  "wave": 1,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 2,
  "candidates": 2,
  "dupOf": [],
  "positives": 8,
  "needsHuman": 1,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "performance.S6c": {
  "cell": "performance.S6c",
  "wave": 1,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 8,
  "candidates": 8,
  "dupOf": [],
  "positives": 10,
  "needsHuman": 2,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "architecture.S6a": {
  "cell": "architecture.S6a",
  "wave": 1,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 7,
  "candidates": 7,
  "dupOf": [],
  "positives": 7,
  "needsHuman": 3,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [
   {
    "title": "The positional carrier is nested: a six-slot LoadedCase alias feeds the ten-slot destructure, so one new loaded artifact edits four positional sites",
    "field": "partIRef",
    "value": "I.5",
    "why": "cobre_sddp-coupling vocabulary in the title / fix-shape (rule 6)"
   }
  ],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "architecture.S6b": {
  "cell": "architecture.S6b",
  "wave": 1,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "redispatched": true,
  "firstFailure": [
   "$.candidates[3].partIRef: not an I.3-n / I.5 reference (the worker wrote 'I.3-7, I.5')",
   "$.candidates[4].fixShape: looks like a diff or a code block; prose only",
   "re-dispatched once with both lines quoted; the rewrite passed the validator with the same 7 candidates / 5 positives / 4 needs-human"
  ],
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 7,
  "candidates": 7,
  "dupOf": [],
  "positives": 5,
  "needsHuman": 4,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [
   {
    "title": "The pre-solver phase driver is written twice over one engine-owned PrepPhase: the CLI extracts the (kind, message) pair into a helper, the bindings re-inline the identical two lines three times",
    "field": "waveRef",
    "value": "CD-029",
    "why": "boundary-reconciliation / PrepPhase / validate phase-driver claim (rule 5)"
   }
  ],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "architecture.S6c": {
  "cell": "architecture.S6c",
  "wave": 1,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 8,
  "candidates": 7,
  "dupOf": [],
  "positives": 10,
  "needsHuman": 2,
  "dropped": [
   {
    "title": "The validate phase driver is hand-mirrored across the two front ends and has already drifted twice: the CLI and the bindings run two phases in opposite order, and their machine-readable error contracts share neither key name nor vocabulary",
    "action": "merged",
    "reason": "rule 14: the validate mirror is raised once; folded into architecture.S6b 'The pre-solver phase driver is written twice over one engine-owned PrepPhase: th' (anchors + evidence carried)"
   }
  ],
  "anchorsRejected": [
   {
    "title": "The validate phase driver is hand-mirrored across the two front ends and has already drifted twice: the CLI and the bindings run two phases in opposite order, and their machine-readable error contracts share neither key name nor vocabulary",
    "rejected": [
     "`crates/cobre-cli/src/commands/validate.rs::execute` (anchor outside the S6c manifest: crates/cobre-cli/src/commands/validate.rs (S6b))",
     "`crates/cobre-cli/src/commands/validate.rs::ValidateErrorOutput` (anchor outside the S6c manifest: crates/cobre-cli/src/commands/validate.rs (S6b))"
    ]
   }
  ],
  "stamped": [
   {
    "title": "The validate phase driver is hand-mirrored across the two front ends and has already drifted twice: the CLI and the bindings run two phases in opposite order, and their machine-readable error contracts share neither key name nor vocabulary",
    "field": "waveRef",
    "value": "CD-029",
    "why": "boundary-reconciliation / PrepPhase / validate phase-driver claim (rule 5)"
   },
   {
    "title": "write_policy_checkpoint in the bindings re-declares the engine's cost-scale default as a bare literal while cobre-sddp exports the named constant the bindings already depend on",
    "field": "partIRef",
    "value": "I.5",
    "why": "cobre_sddp-coupling vocabulary in the title / fix-shape (rule 6)"
   }
  ],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false,
  "mergedDupOf": [
   {
    "title": "The validate phase driver is hand-mirrored across the two front ends and has already drifted twice: the CLI and the bindings run two phases in opposite order, and their machine-readable error contracts share neither key name nor vocabulary",
    "into": "architecture.S6b 'The pre-solver phase driver is written twice over one engine-owned PrepPhase: th'"
   }
  ]
 },
 "over-engineering.S6a": {
  "cell": "over-engineering.S6a",
  "wave": 2,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 6,
  "candidates": 6,
  "dupOf": [],
  "positives": 8,
  "needsHuman": 2,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "over-engineering.S6b": {
  "cell": "over-engineering.S6b",
  "wave": 2,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 6,
  "candidates": 6,
  "dupOf": [],
  "positives": 13,
  "needsHuman": 4,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "over-engineering.S6c": {
  "cell": "over-engineering.S6c",
  "wave": 2,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 2,
  "candidates": 2,
  "dupOf": [],
  "positives": 6,
  "needsHuman": 2,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "test-bloat.S6a": {
  "cell": "test-bloat.S6a",
  "wave": 2,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 8,
  "candidates": 8,
  "dupOf": [],
  "positives": 6,
  "needsHuman": 3,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "test-bloat.S6b": {
  "cell": "test-bloat.S6b",
  "wave": 2,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 10,
  "candidates": 10,
  "dupOf": [],
  "positives": 7,
  "needsHuman": 2,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 },
 "test-bloat.S6c": {
  "cell": "test-bloat.S6c",
  "wave": 2,
  "gatedAt": "2026-09-18",
  "promptBytes": 46814,
  "validator": "pass",
  "validatorErrors": [],
  "checkAnchors": "pass",
  "rawCandidates": 5,
  "candidates": 5,
  "dupOf": [],
  "positives": 6,
  "needsHuman": 2,
  "dropped": [],
  "anchorsRejected": [],
  "stamped": [],
  "reraise": {
   "exit": 0,
   "hits": []
  },
  "cleanVerdict": false
 }
}
```
