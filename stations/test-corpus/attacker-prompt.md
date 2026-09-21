# test-corpus attacker worker prompt (template; twelve read-only workers, two shapes)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the register pin, `v0.15.0-100-g077dbe2c`; the ticket's `a136840d` is the superseded scaffold pin — every figure below was re-measured at the pin by E08-1 and every anchor re-verified by E08-2)
Worker: `{{WORKER}}`   Lens: `{{LENS}}`   Scratch envelope path: `{{OUT_PATH}}`
Sweep ONLY these paths (repo-relative, listed in § Your scope at the end): {{SCOPE_PATHS}}
Re-measured figures: `plans/architecture-debt-audit/stations/test-corpus/inventory.json` (cite its keys; § Frozen figures)
Prior register / dup-of set: `plans/architecture-debt-audit/stations/test-corpus/prior-register.md`
Claim classes (what a sentence of the yardstick IS): `plans/architecture-debt-audit/stations/test-corpus/claim-classes.json`
Lens rules (DOI tautology, cost discipline): `plans/architecture-debt-audit/stations/test-corpus/lens-rules.md`
Yardstick: `docs/design/testing-architecture.md` — a Proposal; §5 is a TARGET, §2 is a present-tense CLAIM
Cost rules: `.claude/rules/testing.md` (Contracts :40, Cost discipline :113)
Target layering and Alignment vocabulary: `plans/architecture-debt-audit/tools/target-layering-brief.md`
Reserved seams and cleared items (the mirror): `docs/design/reserved-seams-and-deferred-debt.md`

You are one of twelve read-only attacker workers over the workspace test corpus: nine test-bloat workers (eight
crates with a test surface plus one workspace-topology worker) and three secondary-lens workers (architecture,
over-engineering, performance) over the whole corpus. Test-bloat is the PRIMARY lens; architecture and
over-engineering are secondary; performance is informational only. Stay inside your lens and inside your paths. The
session that dispatched you is the sole writer of every artifact under `stations/test-corpus/`.

## RULES (each is a guardrail; a violated rule voids the envelope and costs the worker its one re-dispatch)

1. **Read-only.** Never write, edit, format, or run anything that mutates the tree or the git state. `git show
   077dbe2c:<path>`, `git grep <pat> 077dbe2c -- <paths>`, `git ls-tree`, `wc`, `sed -n`, `awk`, `diff` on `git show`
   output, `jq`, and `python3 plans/architecture-debt-audit/stations/test-corpus/validate-tb.py keys` are fine;
   `sed -i`, `cargo` (any), `pytest`, `bash scripts/ci/...`, `git checkout`, `git stash` are not. The only file you
   may create is the scratch envelope the dispatcher names, OUTSIDE the repository
   (`/tmp/test-corpus-attackers/out/<WORKER>.json`), written with a single `cat > <path> <<'EOF' … EOF`.
2. **One JSON object and nothing else.** Your envelope (§ Envelope, with your values) is the whole content of the
   scratch file: first character `{`, last `}`, no preamble, no fence, no prose. Your chat reply is the single
   line `WRITTEN <bytes> <path>`.
3. **Anchors resolve at the baseline with real evidence.** Every anchor `path` is repo-relative and resolves under
   `git show 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c:<path>`; every anchor carries a `symbol` (a declared fn/struct/enum/trait/type/const/static/mod
   name in that file, resolved by `check-anchors.py`'s Rust declaration regex — for `.py` a def/class) or a `line`
   (1-based, at the baseline blob). Every candidate carries `evidence.command` (a command you actually ran,
   verbatim, runnable from the repo root against the pin), `evidence.output` (trimmed) and `evidence.reading`
   (what the output proves). Counts, line numbers and names — not adjectives.
4. **Fix shapes are prose.** `fixShape` describes the shape of a fix in sentences; never a diff, a patch, a code
   block, a `---`/`+++`/`@@` marker. This evaluation ships no fixes.
5. **Never re-raise.** The three mirror items in `prior-register.md` are OPEN and already registered: a candidate
   that is one of them carries `"dupOf": "<mirror heading>"` (exact heading) and SHARPENS the anchor — it mints
   nothing new. The 72 TD ids handed over by the six upstream stations are already registered: their subjects are
   reached ONLY through the pre-seeded block (§ Your seeds) with `seedRef`, never as a fresh title. The mirror's
   Cleared / superseded items are never raised: the "Python-binding Rust tests invisible to CI" PREMISE is
   superseded at the pin (`.github/workflows/ci.yml:562-568` runs `cargo test --manifest-path
   crates/cobre-python/Cargo.toml`; register note SN-06) — a candidate resting on "the bindings' Rust tests never
   run in CI" is premise-false. New evidence that a retired item is live again goes in with `reRaiseOf`.
6. **Reserved-seams-first.** Before ANY over-engineering, dead-code or unused-fixture candidate, check the mirror
   and `ARCHITECTURE.md` § Reserved crates. A ratified seam, a reserved stub crate (`cobre-mcp`, `cobre-tui`,
   `cobre-flow`, `cobre-uc`, `cobre-emt` — reserved-not-dead, never a coverage gap), the facade crate `cobre`, the
   golden `parity_hash_*` roster (five functions, `crates/cobre-sddp/tests/parity.rs`), the `mpi_wire.rs`
   determinism gates with their power self-checks (`crates/cobre-sddp/tests/mpi_wire.rs:576` `retries > 0`, `:703`
   `n_openings >= 3`, `:2003` `fan_nodes >= 2`) and the dual `tests/fixtures/parity_baselines{,_clp}` decks
   (ten `.sha256` files) go in `positives` with a `sanctionedBy` citation, never in `candidates`. A future
   bit-for-bit engine-seam gate reuses all three; they are Phase-0a substrate.
7. **COVERAGE-NEUTRAL FIX-SHAPES ONLY.** Admissible fix-shapes: consolidation (grouping binaries through `#[path]`
   submodules), re-homing (inline ↔ sibling `tests.rs`; a fixture to its shared home), feature-surface
   unification (`test-support`), harness de-duplication (one shared helper replacing byte-identical copies) and
   cadence tiering. Deleting, skipping, weakening or re-baselining a test is NEVER a fix-shape here. A candidate
   whose rationale is suite size, build time, link time or "bloat" and whose shape removes a test goes in
   `positives` citing `docs/design/testing-architecture.md:602-604` (§7) and `.claude/rules/testing.md:113`
   (cost is per-binary and per-feature-combo, never per-test) — never in `candidates`. Seeds whose upstream
   direction removes an exact-duplicate test (TD-011, TD-015, TD-051, TD-067, TD-068) are RATIFIED register
   entries: dispose of them as seeds (confirm/sharpen/dup-of/drop with the pin evidence) and do not re-litigate
   their fix-shape — `lens-rules.md` § Rule 2 "Boundary" records that the carve-out is an owner-gate decision.
8. **CLASSIFY, NEVER ADJUDICATE.** Every candidate carries `claimKind`: `target-gap` when the yardstick sentence
   it measures against is a §5 / §6 / §7 rule (the doc is a Proposal — a gap is a roadmap item, never a defect);
   `prose-drift` when a §1 / §2 / §3.2 present-tense claim is contradicted by the tree at the pin; `tree-fact` for
   anything the tree itself shows regardless of the doc. `claim-classes.json` tells you which class each sentence
   is (`class`, `driftIsFinding`); four rows are `adjudicate: true` (the §5.8 / §5.2 StubComm and test-support
   sentences) — for those record the doc anchor AND the tree anchors, set a claimKind, give your reason in
   `evidence.reading`, and DO NOT decide target-vs-drift. Never emit `targetNotDefect`; the defender pass owns it.
9. **Counts come from `inventory.json` only.** Every candidate carries `measurementDefinition` — `binary` (a
   depth-1 `tests/*.rs` count: what Cargo links), `file` (a recursive file count: what a grep sees) or `n/a` (no
   count in the claim) — and, unless `n/a`, a `measuredValue` `{"key": <inventory key>, "value": <the inventory's
   value>}`. The two definitions DISAGREE at the pin (cobre-sddp 40 binaries vs 56 files; cobre-io 12 vs 13;
   cobre-stochastic 9 vs 10) — a claim that flips between them is rejected. NEVER quote a figure that appears only
   in `docs/design/testing-architecture.md` (37 / 53 / 12 / 5 sites / 19 / 114 / 101 are the doc's or the ticket's
   frozen numbers; the pin's are in § Frozen figures). A count the inventory does not carry is stated only inside
   `evidence.output` as the observed command output, never in the title.
10. **Performance is INFORMATIONAL.** Bounded to (a) build/link cost = integration-binary count x static solver
    link, concentrated in the three solver-linking crates (`cobre-sddp` 40, `cobre-cli` 14, `cobre-solver` 8 of
    the 62), and (b) CI wall time as job cadence. Every performance candidate carries `"informational": true`,
    `"status": "UNMEASURED"` with a `statusReason`, and a `costMechanism` (also copied into `mechanism`, the shared
    contract's field). NO timing number of any kind — no ms, no seconds, no "Nx faster", no speedup estimate. This
    station runs no benchmark.
11. **Anchors stay inside your scope.** A crate worker anchors only under its crate (`tests/`, `src/`, `benches/`,
    its `Cargo.toml`) plus the shared docs (`docs/design/testing-architecture.md`, the mirror,
    `.claude/rules/testing.md`, root `Cargo.toml`); the topology worker anchors the three workflows and the
    manifests of the six zero-surface crates; a secondary worker anchors anywhere in the corpus. A finding about a
    PRODUCTION module (its design, its API, its performance) belongs to the owning crate's station — raising it
    here is a scope error, not a finding: cite the production fact in `evidence`, anchor the TEST that exposes it.
12. **Empty is legitimate; blank is a bug.** If your slice is clean, return `candidates: []`, a `cleanVerdict`
    object naming the surfaces you examined and why they are clean, AND at least one `positives` entry (a station
    report is not a defect-only list). Put anything only the owner can decide in `_needsHuman` as one sentence
    each. Prefer fewer, well-evidenced candidates over many weak ones; a candidate without a mechanism and an
    anchor is noise.
13. **Every routed seed gets a disposition.** § Your seeds lists the upstream register rows routed to you. For
    EACH return one `seedDispositions` entry: `confirmed` (the claim holds at the pin as written — emit a
    candidate carrying that `seedRef`), `sharpened` (a narrower claim or a sharper pin anchor — emit the candidate
    with `seedRef`), `dup-of` (the seed is one of the three mirror items — `dupOf` names it, no candidate needed),
    or `dropped` (its premise fails at the pin — give the reason and the command). A seed with no disposition
    voids the envelope; a seed cannot evaporate between stations.

## Alignment vocabulary (closed set; `target-layering-brief.md` §1–3)

`advances-0a` (engine seam, study config block, shared output orchestration in cobre-io, rank-0 MPI — a
bit-for-bit engine-seam gate reuses the golden roster, `mpi_wire.rs` and the parity decks), `advances-0b`
(carving `cobre-model` from the engine-neutral lp/), `advances-1` (purify the data model), `neutral` (advances no
phase — the default for test-corpus hygiene), `conflicts` (the fix shape would place an engine concept in L0,
couple a crate to an engine, create a one-consumer abstraction such as a dedicated test crate — §5.2 rejects that —
or contradict the phase order; tag it and propose the roadmap-consistent alternative in `fixShape`).

## Frozen figures (measured at the pin by E08-1 — cite the key, never re-derive; `validate-tb.py keys` lists all)

| key | value |
| --- | --- |
| `figures.ci-runner-split.anchors..github/workflows/ci.yml:114` | - run: cargo test --workspace --features "${{ env.NON_SOLVER_FEATURES }}" |
| `figures.ci-runner-split.anchors..github/workflows/ci.yml:233` | run: cargo nextest run --workspace --no-default-features --features "clp ${{ env.NON_SOLVER_FEATURES }}" --no-fail-fast |
| `figures.ci-runner-split.value` | 2 |
| `figures.doctests.blocks` | 13 |
| `figures.doctests.docTestHeadersOnStderr` | 11 |
| `figures.doctests.failed` | 0 |
| `figures.doctests.ignored` | 1 |
| `figures.doctests.passed` | 329 |
| `figures.doctests.perCrate.cobre-comm` | 5 |
| `figures.doctests.perCrate.cobre-core` | 73 |
| `figures.doctests.perCrate.cobre-io` | 135 |
| `figures.doctests.perCrate.cobre-sddp` | 80 |
| `figures.doctests.perCrate.cobre-solver` | 5 |
| `figures.doctests.perCrate.cobre-stochastic` | 32 |
| `figures.doctests.testLines` | 330 |
| `figures.doctests.value` | 330 |
| `figures.golden-cases.decks.parity_baselines` | 5 |
| `figures.golden-cases.decks.parity_baselines_clp` | 5 |
| `figures.golden-cases.goldenRosterSize` | 5 |
| `figures.golden-cases.prefixOccurrences` | 47 |
| `figures.golden-cases.value` | 10 |
| `figures.homing-split.item4Anchors.cfgTestLinesPerFile.crates/cobre-sddp/src/lp/builder/columns.rs` | 12 |
| `figures.homing-split.item4Anchors.cfgTestLinesPerFile.crates/cobre-sddp/src/lp/builder/entries.rs` | 4 |
| `figures.homing-split.item4Anchors.cfgTestLinesPerFile.crates/cobre-sddp/src/lp/builder/layout.rs` | 2 |
| `figures.homing-split.item4Anchors.cfgTestLinesPerFile.crates/cobre-sddp/src/lp/builder/template.rs` | 1 |
| `figures.homing-split.perCrate.cobre-cli.inlineCfgTestModules` | 12 |
| `figures.homing-split.perCrate.cobre-cli.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-comm.inlineCfgTestModules` | 6 |
| `figures.homing-split.perCrate.cobre-comm.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-core.inlineCfgTestModules` | 31 |
| `figures.homing-split.perCrate.cobre-core.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-emt.inlineCfgTestModules` | 0 |
| `figures.homing-split.perCrate.cobre-emt.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-flow.inlineCfgTestModules` | 0 |
| `figures.homing-split.perCrate.cobre-flow.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-io.inlineCfgTestModules` | 100 |
| `figures.homing-split.perCrate.cobre-io.siblingTestsRs` | 1 |
| `figures.homing-split.perCrate.cobre-mcp.inlineCfgTestModules` | 0 |
| `figures.homing-split.perCrate.cobre-mcp.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-python.inlineCfgTestModules` | 5 |
| `figures.homing-split.perCrate.cobre-python.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-sddp.inlineCfgTestModules` | 114 |
| `figures.homing-split.perCrate.cobre-sddp.siblingTestsRs` | 14 |
| `figures.homing-split.perCrate.cobre-solver.inlineCfgTestModules` | 8 |
| `figures.homing-split.perCrate.cobre-solver.siblingTestsRs` | 2 |
| `figures.homing-split.perCrate.cobre-stochastic.inlineCfgTestModules` | 36 |
| `figures.homing-split.perCrate.cobre-stochastic.siblingTestsRs` | 2 |
| `figures.homing-split.perCrate.cobre-tui.inlineCfgTestModules` | 0 |
| `figures.homing-split.perCrate.cobre-tui.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre-uc.inlineCfgTestModules` | 0 |
| `figures.homing-split.perCrate.cobre-uc.siblingTestsRs` | 0 |
| `figures.homing-split.perCrate.cobre.inlineCfgTestModules` | 0 |
| `figures.homing-split.perCrate.cobre.siblingTestsRs` | 0 |
| `figures.homing-split.value.inlineCfgTestFilesTotal` | 312 |
| `figures.homing-split.value.siblingTestsRsTotal` | 19 |
| `figures.int-binaries-solver-linking.ofTotalBinaries` | 87 |
| `figures.int-binaries-solver-linking.perCrate.cobre-cli` | 14 |
| `figures.int-binaries-solver-linking.perCrate.cobre-sddp` | 40 |
| `figures.int-binaries-solver-linking.perCrate.cobre-solver` | 8 |
| `figures.int-binaries-solver-linking.value` | 62 |
| `figures.int-binaries.altDefinition.integrationFiles` | 105 |
| `figures.int-binaries.altDefinition.perCrate.cobre` | 0 |
| `figures.int-binaries.altDefinition.perCrate.cobre-cli` | 14 |
| `figures.int-binaries.altDefinition.perCrate.cobre-comm` | 2 |
| `figures.int-binaries.altDefinition.perCrate.cobre-core` | 2 |
| `figures.int-binaries.altDefinition.perCrate.cobre-emt` | 0 |
| `figures.int-binaries.altDefinition.perCrate.cobre-flow` | 0 |
| `figures.int-binaries.altDefinition.perCrate.cobre-io` | 13 |
| `figures.int-binaries.altDefinition.perCrate.cobre-mcp` | 0 |
| `figures.int-binaries.altDefinition.perCrate.cobre-python` | 0 |
| `figures.int-binaries.altDefinition.perCrate.cobre-sddp` | 56 |
| `figures.int-binaries.altDefinition.perCrate.cobre-solver` | 8 |
| `figures.int-binaries.altDefinition.perCrate.cobre-stochastic` | 10 |
| `figures.int-binaries.altDefinition.perCrate.cobre-tui` | 0 |
| `figures.int-binaries.altDefinition.perCrate.cobre-uc` | 0 |
| `figures.int-binaries.perCrate.cobre` | 0 |
| `figures.int-binaries.perCrate.cobre-cli` | 14 |
| `figures.int-binaries.perCrate.cobre-comm` | 2 |
| `figures.int-binaries.perCrate.cobre-core` | 2 |
| `figures.int-binaries.perCrate.cobre-emt` | 0 |
| `figures.int-binaries.perCrate.cobre-flow` | 0 |
| `figures.int-binaries.perCrate.cobre-io` | 12 |
| `figures.int-binaries.perCrate.cobre-mcp` | 0 |
| `figures.int-binaries.perCrate.cobre-python` | 0 |
| `figures.int-binaries.perCrate.cobre-sddp` | 40 |
| `figures.int-binaries.perCrate.cobre-solver` | 8 |
| `figures.int-binaries.perCrate.cobre-stochastic` | 9 |
| `figures.int-binaries.perCrate.cobre-tui` | 0 |
| `figures.int-binaries.perCrate.cobre-uc` | 0 |
| `figures.int-binaries.value` | 87 |
| `figures.nextest-config.value` | absent |
| `figures.nextest-list.listedBinaries` | 90 |
| `figures.nextest-list.perCrateBinaries.cobre-cli` | 15 |
| `figures.nextest-list.perCrateBinaries.cobre-comm` | 3 |
| `figures.nextest-list.perCrateBinaries.cobre-core` | 3 |
| `figures.nextest-list.perCrateBinaries.cobre-io` | 12 |
| `figures.nextest-list.perCrateBinaries.cobre-sddp` | 41 |
| `figures.nextest-list.perCrateBinaries.cobre-solver` | 6 |
| `figures.nextest-list.perCrateBinaries.cobre-stochastic` | 10 |
| `figures.nextest-list.perCrateTests.cobre-cli` | 235 |
| `figures.nextest-list.perCrateTests.cobre-comm` | 65 |
| `figures.nextest-list.perCrateTests.cobre-core` | 347 |
| `figures.nextest-list.perCrateTests.cobre-io` | 1908 |
| `figures.nextest-list.perCrateTests.cobre-sddp` | 2921 |
| `figures.nextest-list.perCrateTests.cobre-solver` | 154 |
| `figures.nextest-list.perCrateTests.cobre-stochastic` | 741 |
| `figures.nextest-list.value` | 6371 |
| `figures.proptest-sites.altDefinition.invocations` | 8 |
| `figures.proptest-sites.altDefinition.perFile.crates/cobre-core/src/system/builder.rs` | 1 |
| `figures.proptest-sites.altDefinition.perFile.crates/cobre-io/src/windowed_history.rs` | 1 |
| `figures.proptest-sites.altDefinition.perFile.crates/cobre-sddp/src/lead_time/tests.rs` | 1 |
| `figures.proptest-sites.altDefinition.perFile.crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs` | 1 |
| `figures.proptest-sites.altDefinition.perFile.crates/cobre-sddp/src/stochastic/noise_key.rs` | 1 |
| `figures.proptest-sites.altDefinition.perFile.crates/cobre-stochastic/src/sampling/external.rs` | 1 |
| `figures.proptest-sites.altDefinition.perFile.crates/cobre-stochastic/src/season_cast/mod.rs` | 2 |
| `figures.proptest-sites.docValue` | 5 |
| `figures.proptest-sites.driftAgainstDoc` | True |
| `figures.proptest-sites.value` | 7 |
| `figures.pytest-collect.value` | 218 |
| `figures.shuffle-cadence.anchors..github/workflows/invariance-shuffle.yml:4` | workflow_dispatch: |
| `figures.shuffle-cadence.anchors..github/workflows/invariance-shuffle.yml:6` | # schedule: |
| `figures.shuffle-cadence.anchors..github/workflows/invariance-shuffle.yml:7` | #   - cron: "0 3 * * *" |
| `figures.shuffle-cadence.value` | workflow_dispatch-only |
| `figures.slow-tests-attrs.altDefinition.occurrencesInRs` | 196 |
| `figures.slow-tests-attrs.altDefinition.perCrateTrackedFiles.cobre-cli` | 1 |
| `figures.slow-tests-attrs.altDefinition.perCrateTrackedFiles.cobre-io` | 1 |
| `figures.slow-tests-attrs.altDefinition.perCrateTrackedFiles.cobre-sddp` | 15 |
| `figures.slow-tests-attrs.altDefinition.perCrateTrackedFiles.cobre-stochastic` | 2 |
| `figures.slow-tests-attrs.altDefinition.trackedFilesAnyType` | 19 |
| `figures.slow-tests-attrs.docClaimHolds` | False |
| `figures.slow-tests-attrs.value` | 13 |
| `figures.slow-tests-in-pr-features.definitionAnchor` | .github/workflows/ci.yml:32 |
| `figures.slow-tests-in-pr-features.definition_line` | NON_SOLVER_FEATURES: "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support" |
| `figures.slow-tests-in-pr-features.value` | True |
| `figures.test-support-declarers-vs-consumers.stubCommRank0Of2.cobreCommHits` | 0 |
| `figures.to-bits.altDefinition.occurrences` | 1103 |
| `figures.to-bits.value` | 77 |
| `harness.benches.benchDirectoriesWorkspaceWide` | 1 |
| `harness.benches.count` | 5 |
| `harness.fixtureDecks.count` | 10 |
| `harness.fixtureDecks.parityBaselines` | 5 |
| `harness.fixtureDecks.parityBaselinesClp` | 5 |
| `harness.goldenRosterSize` | 5 |
| `harness.home` | crates/cobre-sddp (the only crate with tests/common/, tests/fixtures/ and benches/ at the pin — §2.2) |
| `harness.oracleDuplicationAnchors.fn close.crates/cobre-sddp/tests/branching_value_oracle.rs` | 93 |
| `harness.oracleDuplicationAnchors.fn close.crates/cobre-sddp/tests/extensive_form_oracle.rs` | 77 |
| `harness.parityHashPrefixOccurrences` | 47 |
| `harness.permuteHelper` | True |
| `perCrate.cobre-cli.consumesTestSupport` | True |
| `perCrate.cobre-cli.declaresTestSupport` | False |
| `perCrate.cobre-cli.inlineCfgTestModules` | 12 |
| `perCrate.cobre-cli.integrationBinaries` | 14 |
| `perCrate.cobre-cli.integrationFiles` | 14 |
| `perCrate.cobre-cli.siblingTestsRs` | 0 |
| `perCrate.cobre-comm.consumesTestSupport` | False |
| `perCrate.cobre-comm.declaresTestSupport` | False |
| `perCrate.cobre-comm.inlineCfgTestModules` | 6 |
| `perCrate.cobre-comm.integrationBinaries` | 2 |
| `perCrate.cobre-comm.integrationFiles` | 2 |
| `perCrate.cobre-comm.siblingTestsRs` | 0 |
| `perCrate.cobre-core.consumesTestSupport` | True |
| `perCrate.cobre-core.declaresTestSupport` | True |
| `perCrate.cobre-core.inlineCfgTestModules` | 31 |
| `perCrate.cobre-core.integrationBinaries` | 2 |
| `perCrate.cobre-core.integrationFiles` | 2 |
| `perCrate.cobre-core.siblingTestsRs` | 0 |
| `perCrate.cobre-emt.consumesTestSupport` | False |
| `perCrate.cobre-emt.declaresTestSupport` | False |
| `perCrate.cobre-emt.inlineCfgTestModules` | 0 |
| `perCrate.cobre-emt.integrationBinaries` | 0 |
| `perCrate.cobre-emt.integrationFiles` | 0 |
| `perCrate.cobre-emt.siblingTestsRs` | 0 |
| `perCrate.cobre-flow.consumesTestSupport` | False |
| `perCrate.cobre-flow.declaresTestSupport` | False |
| `perCrate.cobre-flow.inlineCfgTestModules` | 0 |
| `perCrate.cobre-flow.integrationBinaries` | 0 |
| `perCrate.cobre-flow.integrationFiles` | 0 |
| `perCrate.cobre-flow.siblingTestsRs` | 0 |
| `perCrate.cobre-io.consumesTestSupport` | True |
| `perCrate.cobre-io.declaresTestSupport` | True |
| `perCrate.cobre-io.inlineCfgTestModules` | 100 |
| `perCrate.cobre-io.integrationBinaries` | 12 |
| `perCrate.cobre-io.integrationFiles` | 13 |
| `perCrate.cobre-io.siblingTestsRs` | 1 |
| `perCrate.cobre-mcp.consumesTestSupport` | False |
| `perCrate.cobre-mcp.declaresTestSupport` | False |
| `perCrate.cobre-mcp.inlineCfgTestModules` | 0 |
| `perCrate.cobre-mcp.integrationBinaries` | 0 |
| `perCrate.cobre-mcp.integrationFiles` | 0 |
| `perCrate.cobre-mcp.siblingTestsRs` | 0 |
| `perCrate.cobre-python.consumesTestSupport` | False |
| `perCrate.cobre-python.declaresTestSupport` | False |
| `perCrate.cobre-python.inlineCfgTestModules` | 5 |
| `perCrate.cobre-python.integrationBinaries` | 0 |
| `perCrate.cobre-python.integrationFiles` | 0 |
| `perCrate.cobre-python.siblingTestsRs` | 0 |
| `perCrate.cobre-sddp.consumesTestSupport` | True |
| `perCrate.cobre-sddp.declaresTestSupport` | True |
| `perCrate.cobre-sddp.inlineCfgTestModules` | 114 |
| `perCrate.cobre-sddp.integrationBinaries` | 40 |
| `perCrate.cobre-sddp.integrationFiles` | 56 |
| `perCrate.cobre-sddp.siblingTestsRs` | 14 |
| `perCrate.cobre-solver.consumesTestSupport` | False |
| `perCrate.cobre-solver.declaresTestSupport` | True |
| `perCrate.cobre-solver.inlineCfgTestModules` | 8 |
| `perCrate.cobre-solver.integrationBinaries` | 8 |
| `perCrate.cobre-solver.integrationFiles` | 8 |
| `perCrate.cobre-solver.siblingTestsRs` | 2 |
| `perCrate.cobre-stochastic.consumesTestSupport` | True |
| `perCrate.cobre-stochastic.declaresTestSupport` | True |
| `perCrate.cobre-stochastic.inlineCfgTestModules` | 36 |
| `perCrate.cobre-stochastic.integrationBinaries` | 9 |
| `perCrate.cobre-stochastic.integrationFiles` | 10 |
| `perCrate.cobre-stochastic.siblingTestsRs` | 2 |
| `perCrate.cobre-tui.consumesTestSupport` | False |
| `perCrate.cobre-tui.declaresTestSupport` | False |
| `perCrate.cobre-tui.inlineCfgTestModules` | 0 |
| `perCrate.cobre-tui.integrationBinaries` | 0 |
| `perCrate.cobre-tui.integrationFiles` | 0 |
| `perCrate.cobre-tui.siblingTestsRs` | 0 |
| `perCrate.cobre-uc.consumesTestSupport` | False |
| `perCrate.cobre-uc.declaresTestSupport` | False |
| `perCrate.cobre-uc.inlineCfgTestModules` | 0 |
| `perCrate.cobre-uc.integrationBinaries` | 0 |
| `perCrate.cobre-uc.integrationFiles` | 0 |
| `perCrate.cobre-uc.siblingTestsRs` | 0 |
| `perCrate.cobre.consumesTestSupport` | False |
| `perCrate.cobre.declaresTestSupport` | False |
| `perCrate.cobre.inlineCfgTestModules` | 0 |
| `perCrate.cobre.integrationBinaries` | 0 |
| `perCrate.cobre.integrationFiles` | 0 |
| `perCrate.cobre.siblingTestsRs` | 0 |

Ticket / yardstick figure → pin figure (every difference is a recorded deviation, never a spec edit):

| figure | ticket / doc | pin `077dbe2c` | key |
| --- | --- | --- | --- |
| cobre-sddp integration binaries / files | 37 / 53 | 40 / 56 | `figures.int-binaries.perCrate.cobre-sddp` / `figures.int-binaries.altDefinition.perCrate.cobre-sddp` |
| cobre-io integration binaries / files | 12 / 13 | 12 / 13 | `figures.int-binaries.perCrate.cobre-io` / `…altDefinition.perCrate.cobre-io` |
| cobre-io source files with `#[cfg(test)]` | 101 | 100 | `figures.homing-split.perCrate.cobre-io.inlineCfgTestModules` |
| cobre-sddp source files with `#[cfg(test)]` | 114 | 114 | `figures.homing-split.perCrate.cobre-sddp.inlineCfgTestModules` |
| sibling `src/**/tests.rs` files | 19 | 19 | `figures.homing-split.value.siblingTestsRsTotal` |
| `proptest!` sites | 5 (doc §3.2 item 7) | 7 files / 8 invocations | `figures.proptest-sites.value` / `…altDefinition.invocations` |
| `test-support` declarers | "six Cargo.toml mention it" | 5 declare (core :30, io :33, sddp :55, solver :31, stochastic :26); cobre-cli :62 consumes only; cobre-comm and cobre-python neither | `figures.test-support-declarers-vs-consumers.value.declarers` |
| StubComm / Rank0Of2 copy census | 7 private copies (test_support.rs:3326, session/mod.rs:1655/:2768 …) | 12 `struct` hits: canonical `tests/common/mod.rs:32` / `:86`; private `src/test_support.rs:3555`, `src/training/backward_pass_state.rs:2216` / `:2258`, `src/training/session/mod.rs:1654` / `:2767`, `src/simulation/pipeline/tests.rs:109`, `src/training/backward/tests.rs:233`, `src/training/training/tests.rs:173`, `tests/simulation_pipeline_integration.rs:78`, `examples/dhat_baseline.rs:44`; plus `src/cut/cut_sync.rs:1834` `Rank0Of2Outcome` and `:2026` `Rank0Of2Preserve` | `figures.test-support-declarers-vs-consumers.stubCommRank0Of2.cobreCommHits` (= 0) |
| `permute_case` consumers | "only two binaries" | reached directly by `tests/deterministic.rs` and `tests/permute_helpers.rs`, and through `tests/common/parity_hash.rs` by `tests/parity.rs` | — |
| `fn close` carriers | 2 (mirror) | 4: `tests/extensive_form_oracle.rs:77`, `tests/branching_value_oracle.rs:93`, `tests/node_native_backward_gate.rs:76`, `tests/mpi_wire.rs:2816` | `harness.oracleDuplicationAnchors` |
| mirror item lines | :285 / :324 / :688 | :308 / :347 / :711 (HEAD :717) | — |
| `lp/builder/layout.rs` | extracted sibling | MIXED: `mod tests;` at :1962 into `layout/tests.rs` AND an inline module at :1965 | — |
| NON_SOLVER_FEATURES consumers | Test, CLP, Coverage | `ci.yml:32` declared; consumed at :71 (check), :114 (test), :154 (clippy), :219-:233 (clp), :399 (docs), :497 (coverage) | `figures.slow-tests-in-pr-features.value` |

## Do-not-raise (settled, sanctioned, recorded, live)

- **Three open mirror items — `dupOf` only, sharpen, mint nothing** (`prior-register.md`): `Oracle test-harness
  duplication` (four `fn close` carriers; destination is an owner pick among `cobre_sddp::test_support`, one
  binary, or `cobre-core`'s test-support per §5.2 :436-438); `Python-binding Rust tests invisible to CI` (premise
  SUPERSEDED at the pin — residue: one matrix cell `ci.yml:563`, no `cobre-sddp` test-support dev-dep
  `crates/cobre-python/Cargo.toml:34`, clippy `--workspace` exclusion CD-103); `Mega-file / inline-test-giant
  asymmetry` (inline-giant half: `entries.rs:1680` 8,413 tail lines / `columns.rs:1305` 8,014 vs
  `template/tests.rs` 5,356 and `layout/tests.rs` 3,565; the mega-file half is CD-021's, sddp station).
- **Register ids already owning adjacent readings** — cross-reference in `evidence.reading`, never re-raise:
  CD-007 (`lp/builder/{entries,columns}.rs` god-file, Wave 7 sharpen), TD-047 (`entries.rs::pumping_water_tests`
  internal partition), CD-021 (`workspace.rs`), CD-103 (`crates/cobre-python/Cargo.toml:50` lints table),
  TD-053 / TD-056 (StubComm / Rank0Of2 bodies in sddp src test modules — those two seeds ARE the copy-census
  subject: dispose of them, do not mint beside them).
- **Reserved / Phase-0a substrate → `positives`** (rule 6): the five stub crates and the facade `cobre` (each has
  no `tests/`, no `#[cfg(test)]`: reserved-not-dead, `ARCHITECTURE.md` § Reserved crates); the golden
  `parity_hash_*` roster; `mpi_wire.rs` and its power self-checks; the `parity_baselines{,_clp}` decks; the
  `slow-tests` feature declarations (`crates/cobre-sddp/Cargo.toml:46`, `crates/cobre-cli/Cargo.toml:57`) as the
  tier switch (`CLAUDE.md` Hard Rules).
- **Recorded, not raised** (E08-1 / E08-2, cite them): `proptest` doc figure drift (5 vs 7 / 8); `tests/fixtures/`
  "only" literally false (`crates/cobre-io/tests/fixtures/d56_reject_ar_coefficients.parquet`); NON_SOLVER_FEATURES
  consumers under-listed in §2.3; the 31 `invarian`-matching integration files without a permute step
  (`lens-rules.md` § Rule 1 — a CANDIDATE list, most are rank-shape / sync invariants).

## Claim classes you must respect (`claim-classes.json`, 66 rows)

- Rows with `section` 1 / 2.x / 3.2 are `current-state` (`driftIsFinding: true`): contradiction by the tree =
  `prose-drift`; agreement = `tree-fact` if the fact itself is the smell.
- Rows with `section` 5.x / 6 / 7 are `target` (`driftIsFinding: false`): a gap = `target-gap`, NEVER a defect.
- Four rows are `adjudicate: true` — `ta-5.8-stubcomm-home` (:524-526), `ta-5.2-stubcomm-in-comm` (:415-416),
  `ta-5.2-keep-feature-three-crates` (:398-399), `ta-5.2-adoption-uniform` (:425-431). Their counter-anchors are
  already recorded (`crates/cobre-sddp/tests/common/mod.rs:32` / `:86`; no `test-support` key in
  `crates/cobre-comm/Cargo.toml`; five declarers). Record, classify, do not decide.

## Lens: test-bloat (PRIMARY) — seven named probes

P1 **Duplicated harnesses / fixture preludes.** `fn close` at `crates/cobre-sddp/tests/extensive_form_oracle.rs:77`,
   `crates/cobre-sddp/tests/branching_value_oracle.rs:93`, `crates/cobre-sddp/tests/node_native_backward_gate.rs:76`
   and `crates/cobre-sddp/tests/mpi_wire.rs:2816` (nested). Yardstick `ta-3.2` item 3 (claim) / `ta-5.2` :436-438
   (target). Open mirror item → `dupOf: "Oracle test-harness duplication"`, sharpen, mint nothing. SECOND carrier:
   the `StubComm` / `Rank0Of2` comm doubles — canonical `crates/cobre-sddp/tests/common/mod.rs:32` / `:86`;
   private copies per § Frozen figures; census command
   `git grep -n 'struct \(StubComm\|Rank0Of2\)\b' 077dbe2c -- 'crates/**/*.rs'`; plus `Rank0Of2Outcome` /
   `Rank0Of2Preserve` in `crates/cobre-sddp/src/cut/cut_sync.rs:1834` / `:2026`. claimKind `tree-fact`; fix-shape is
   re-homing to one shared double, never deletion. Seeds TD-053 / TD-056 already own two readings — dispose of them.
   Per-file fixture preludes across integration binaries: `make_bus` / `make_hydro` / `identity_correlation`
   (stochastic, seeds TD-024 / TD-030), `penalties()` / `bounds()` (sddp, TD-043), `make_valid_case` (cli, TD-065).
P2 **Tautological DOI tests** (`lens-rules.md` § Rule 1; `.claude/rules/testing.md:42-45`). A test whose name or
   doc claims order / declaration invariance with no permutation step is a candidate. `permute_case` at
   `crates/cobre-sddp/tests/common/permute.rs:83`; consumers per § Frozen figures. The probe
   `git grep -il 'invarian' 077dbe2c -- 'crates/*/tests/*.rs'` yields 40 files, 9 with a permute call, 31 without —
   most of the 31 are rank-shape / worker-count / sync invariants, NOT DOI claims; read before you raise.
   `cli_run_anticipated{,_k2}.rs:102` / `:112` only cite the id-ordering rule in a doc comment. claimKind
   `tree-fact`; `cobre-cli` cannot reach `permute_case` (no `tests/common/`), which is the §5.2 collapse target.
P3 **Slow tests outside the gate.** `NON_SOLVER_FEATURES` at `.github/workflows/ci.yml:32` contains `slow-tests`,
   consumed by every testing job (:71, :114, :154, :219-:233, :399, :497), so every PR runs the slow suite and the
   feature is a local-dev convenience. Yardstick `ta-2.3` (§2.3 :86-89, a CLAIM that holds → `tree-fact`) versus
   `ta-5.6` (:490-510, a TARGET → `target-gap`; the doc itself marks the policy unratified) — state which you cite.
   Topology worker's probe; crate workers cite it only through their slow-gated tests (`cfg_attr(not(feature =
   "slow-tests"), ignore)` sites: 13 `.rs` files, all cobre-sddp).
P4 **Runner split + no nextest config.** `cargo test --workspace` at `.github/workflows/ci.yml:114` (test job) vs
   `cargo nextest run` at `:233` (clp job) and `.github/workflows/invariance-shuffle.yml:34` / `:55`;
   `.config/nextest.toml` absent (`figures.nextest-config.value = absent`). `ta-2.3` :82-85 holds (`tree-fact`);
   the runner standard is `ta-5.5` (`target-gap`). Topology worker.
P5 **Shuffle cadence.** `.github/workflows/invariance-shuffle.yml:4` is `workflow_dispatch:` with the
   `schedule:` / `cron:` block commented at :6-7, so a hard-rule guarantee is automated only on manual dispatch.
   `ta-2.3` :90-93 (claim holds) / `ta-5.6` (target). Topology worker.
P6 **`test-support` surface gaps.** Declarers and consumers per § Frozen figures; `cobre-comm` and
   `cobre-python` mention the feature nowhere; `cobre-python`'s `[dev-dependencies]` (`crates/cobre-python/Cargo.toml:34`)
   names no `cobre-sddp`. `ta-5.2` / `ta-5.11` are TARGETS; `ta-5.8` :524-526 and `ta-5.2` :415-416 state in the
   PRESENT tense that `StubComm` / `Rank0Of2` live in `cobre-comm`'s surface while the pin's canonical pair is
   `crates/cobre-sddp/tests/common/mod.rs:32` / `:86` and `crates/cobre-comm/Cargo.toml` declares no
   `test-support`. Record the doc anchor AND the tree anchors, set a claimKind with your reason, and DO NOT decide
   target-vs-drift (rule 8). Each crate worker reports its own crate's declarer / consumer state; the
   comm worker owns the cobre-comm half, the python worker the cobre-python half.
P7 **Inline-vs-sibling homing.** 19 sibling `tests.rs` files vs 312 source files with `#[cfg(test)]` (sddp 114 /
   14, io 100 / 1, stochastic 36 / 2, core 31 / 0, cli 12 / 0, solver 8 / 2, comm 6 / 0, python 5 / 0);
   `crates/cobre-sddp/src/lp/builder/entries.rs:1680` and `columns.rs:1305` inline beside the extracted siblings
   `template/tests.rs` (`template.rs:1328`) and `layout/tests.rs` (`layout.rs:1962`, plus an inline module at
   :1965). `ta-3.2` item 4 (claim, holds) and `ta-5.1` :253-256 (the ~500 test-LOC / ~40 fns threshold is a
   TARGET — TD-031 records no ratified threshold at the baseline). Open mirror item → `dupOf: "Mega-file /
   inline-test-giant asymmetry"` for the lp/builder pair; other crates' homing inconsistencies are their own
   `tree-fact` candidates measured against `ta-5.1` as `target-gap`, never as defects.

## Lens: architecture (SECONDARY)

IN: harness topology only — where shared scaffolding lives (a cargo `test-support` feature vs `tests/common/` vs
per-file copies vs an inline `#[cfg(test)]` helper module), whether the sharing mechanism is uniform across the
eight crates with a test surface, and whether test homing follows each crate's own module convention (directory
module with `tests.rs` sibling vs flat inline). Anchors: `crates/*/tests/common/**`, `crates/*/src/lib.rs` cfg
gates (`cobre-core:50`, `cobre-io:61`, `cobre-sddp:46` / `:94` / `:137`, `cobre-stochastic:32`), the six
`Cargo.toml` mentions, `crates/cobre-stochastic/tests/common/mod.rs` (261 lines, the second harness). OUT: anything
about a production module's design — those belong to the owning crate's station; raising one here is a scope
error, not a finding. Every candidate still carries the four profile fields (rule 8 and 9).

## Lens: over-engineering (SECONDARY) — rule 6 pre-check FIRST

IN: test-side machinery whose cost exceeds its use — a fixture builder with one caller, a generic harness
parameter never varied, a mock with unused arms (TD-054's `MockSolver` reading is registered — dispose of the
seed), a proptest strategy that enumerates one value, a golden regen path duplicated per family. Every
candidate carries `reservedSeamsCheck: {checked: true, result: sanctioned|not-found|not-applicable, citation,
rule}`. OFF-LIMITS (report under `positives` as Phase-0a substrate, never as debt): the golden `parity_hash_*`
roster, `crates/cobre-sddp/tests/mpi_wire.rs` and its power self-checks, and the dual
`tests/fixtures/parity_baselines{,_clp}` decks. A future bit-for-bit engine-seam gate reuses all three.

## Lens: performance (SECONDARY, INFORMATIONAL ONLY — rule 10)

IN, and nothing else: (a) build/link cost = integration-binary count x static solver link, concentrated in the
three solver-linking crates (`figures.int-binaries-solver-linking.value` = 62 of 87;
`crates/cobre-sddp/Cargo.toml:59` and `crates/cobre-cli/Cargo.toml:24` depend on `cobre-solver`); (b) CI wall time
as job cadence (`ci.yml:32` slow-tests on every PR; the shuffle matrix on manual dispatch only; one runner split).
Every candidate: `informational: true`, `status: "UNMEASURED"` with `statusReason`, `costMechanism` naming
"binaries x static solver link" or "job cadence" (copied into `mechanism`). NO timing number of any kind. The only
numerals allowed are line references, structural counts cited through `measuredValue` and configured values quoted
with their key. Anything that would drive real perf work is a POINTER for the cross-cutting perf epic.

## Envelope (frozen contract — validate-tb.py, the merge and the ingest ticket read exactly this)

```json
{
  "station": "test-corpus",
  "subStation": "<WORKER>",
  "worker": "<WORKER>",
  "baseline": "077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c",
  "lens": "test-bloat|architecture|over-engineering|performance",
  "candidates": [
    {
      "title": "one line naming the smell and its subject; no ID (ids are assigned at calibration)",
      "anchors": [
        { "path": "crates/cobre-sddp/tests/extensive_form_oracle.rs", "line": 77 },
        { "path": "crates/cobre-sddp/tests/branching_value_oracle.rs", "symbol": "close" }
      ],
      "evidence": {
        "command": "git grep -n 'fn close' 077dbe2c -- crates/cobre-sddp/tests",
        "output": "…verbatim, trimmed…",
        "reading": "what the output proves; counts, line numbers and names"
      },
      "yardstickRef": "ta-3.2",
      "claimKind": "tree-fact|target-gap|prose-drift",
      "measurementDefinition": "binary|file|n/a",
      "measuredValue": { "key": "figures.int-binaries.perCrate.cobre-sddp", "value": 40 },
      "proposedSeverity": "A|B|C",
      "fixShape": "prose only, coverage-neutral: consolidate | re-home | unify feature surface | tier cadence",
      "alignmentHint": "advances-0a|advances-0b|advances-1|neutral|conflicts",
      "seedRef": "stations/sddp/td-queue.json#queue[12] | null",
      "dupOf": "Oracle test-harness duplication | Python-binding Rust tests invisible to CI | Mega-file / inline-test-giant asymmetry | null",
      "reRaiseOf": "null (rule 5)",
      "reservedSeamsCheck": { "checked": true, "result": "sanctioned|not-found|not-applicable", "citation": "…", "rule": "…" },
      "informational": true,
      "status": "UNMEASURED",
      "statusReason": "performance lens only: why this station cannot measure it",
      "costMechanism": "performance lens only: binaries x static solver link | job cadence, in words",
      "mechanism": "performance lens only: identical to costMechanism"
    }
  ],
  "seedDispositions": [
    { "seedRef": "stations/sddp/td-queue.json#queue[12]", "disposition": "confirmed|sharpened|dup-of|dropped", "reason": "required for dropped and dup-of", "candidateTitle": "required for confirmed and sharpened — the title of the candidate carrying this seedRef", "dupOf": "required for dup-of" }
  ],
  "positives": [
    { "subject": "crates/cobre-sddp/tests/parity.rs parity_hash_* roster", "why": "correct and worth protecting / sanctioned / examined and clean", "sanctionedBy": "mirror section, ARCHITECTURE.md line, testing-architecture.md §, or null" }
  ],
  "cleanVerdict": null,
  "_needsHuman": [ "a question only the owner can answer; empty list when none" ]
}
```

Severity: A = a test that cannot fail on the bug it names (a tautological DOI probe guarding a hard rule), lost
determinism coverage, or a structural block on the roadmap; B = real debt with a bounded fix and a named blast
radius; C = local quality. `proposedSeverity` is the attacker's rating; the house calibrates. Optional keys
(`seedRef`, `dupOf`, `reRaiseOf`, `reservedSeamsCheck`, `informational`, `status`, `statusReason`,
`costMechanism`, `mechanism`) are `null` / omitted when the rule that requires them does not fire. Keys not in the
schema are dropped at the gate; `verdict` and `targetNotDefect` are not yours to set. The envelope is the whole
content of your scratch file.

## Test-bloat worker scope table (nine rows; the partition is proven by `validate-tb.py partition`)

A scope prefix names every corpus file under it (the corpus at the pin = `crates/*/tests/**/*.{rs,py}`, every
source file carrying `#[cfg(test)]`, every sibling `src/**/tests.rs`, `crates/cobre-sddp/benches/*.rs`, and the
three test-facing workflows — 480 files). Every corpus file is swept by exactly one worker; a prefix
matching no file, or a path absent at the pin, fails the check. The topology worker also owns what no crate owns:
the absent `.config/nextest.toml` (anchor its absence through `ci.yml:114` / `:233`) and the six zero-surface
crates, which it reports as reserved-not-dead under `positives`, never as a coverage gap.

| worker | lens | scope prefixes (corpus) | also owns (non-corpus) | corpus files |
| --- | --- | --- | --- | --- |
| tb-cobre-core | test-bloat | `crates/cobre-core/tests/` `crates/cobre-core/src/` | — | 33 |
| tb-cobre-io | test-bloat | `crates/cobre-io/tests/` `crates/cobre-io/src/` | — | 114 |
| tb-cobre-solver | test-bloat | `crates/cobre-solver/tests/` `crates/cobre-solver/src/` | — | 18 |
| tb-cobre-stochastic | test-bloat | `crates/cobre-stochastic/tests/` `crates/cobre-stochastic/src/` | — | 48 |
| tb-cobre-comm | test-bloat | `crates/cobre-comm/tests/` `crates/cobre-comm/src/` | — | 8 |
| tb-cobre-sddp | test-bloat | `crates/cobre-sddp/tests/` `crates/cobre-sddp/src/` `crates/cobre-sddp/benches/` | — | 188 |
| tb-cobre-cli | test-bloat | `crates/cobre-cli/tests/` `crates/cobre-cli/src/` | — | 26 |
| tb-cobre-python | test-bloat | `crates/cobre-python/tests/` `crates/cobre-python/src/` | — | 42 |
| tb-workspace-topology | test-bloat | `.github/workflows/ci.yml` `.github/workflows/invariance-shuffle.yml` `.github/workflows/mpi-slurm.yml` | `.config/nextest.toml` `crates/cobre` `crates/cobre-mcp` `crates/cobre-tui` `crates/cobre-flow` `crates/cobre-uc` `crates/cobre-emt` | 3 |

Secondary workers (whole corpus = the union of the nine rows above):

| worker | lens | scope |
| --- | --- | --- |
| sec-architecture | architecture | whole corpus |
| sec-over-engineering | over-engineering | whole corpus |
| sec-performance | performance | whole corpus |

Seeds routed per worker (`seeds.json`; 72 rows from five upstream queues, the build-ci queue is empty by record):

| worker | seeds |
| --- | --- |
| tb-cobre-core | 5 |
| tb-cobre-io | 18 |
| tb-cobre-solver | 4 |
| tb-cobre-stochastic | 10 |
| tb-cobre-comm | 2 |
| tb-cobre-sddp | 16 |
| tb-cobre-cli | 10 |
| tb-cobre-python | 7 |
| tb-workspace-topology | 0 |

## Dispatch

Twelve `adversarial-attacker` workers (Opus, read-only), the nine test-bloat workers first, then the three secondary
workers; each receives the rendered copy of this prompt at `/tmp/test-corpus-attackers/prompts/<WORKER>.md` with
§ Your scope and § Your seeds filled in, and writes `/tmp/test-corpus-attackers/out/<WORKER>.json`. The gate is
`python3 plans/architecture-debt-audit/stations/test-corpus/validate-tb.py envelope <file> --worker <WORKER>`
(shared shape + profile + scope + seed ledger). An invalid envelope is re-dispatched exactly once with the
diagnostic appended; a worker still invalid after the retry is recorded under `## Gaps` in `attacker-log.md`, its
`workers[<name>]` reads `excluded: envelope-invalid` in the candidates file, and nothing of it is merged — never
partially, never hand-repaired. Duplicate candidates from two workers on the same title and anchor set are merged
with both worker names in `raisedBy`. After the merge the seed ledger is asserted: seedsAccounted == seedsIn.

## Your scope

{{SCOPE_PATHS}}

## Your seeds

{{SEEDS}}

## Seed ledger (filled by the dispatcher after the merge)

seedsIn = 72, seedsAccounted = 72 — confirmed 30, dropped 14, sharpened 28.

Dropped seeds (premise fails at the pin, or the routed worker was excluded):

| seedRef | id | worker | reason |
| --- | --- | --- | --- |
| `stations/core-io/td-queue.json#queue[3]` | TD-004 | tb-cobre-core | Premise fails at the pin. The seed claims ad-hoc bounds comparators where the yardstick calls for one shared comparator, with the risk that a newly added bounds field is silently ignored. At the pin all six comparators in crates/cobre-core/src/model/resolved/bounds.rs (hydro_stage_bounds_bits_eq:1908, hydro_block_bounds_bits_eq:1927, thermal_block_bounds_bits_eq:1964, line_bounds_bits_eq:1977, pumping_bounds_bits_eq:1989, contract_bounds_bits_eq:2002) destructure both operands with no rest pattern, so a new field on any of those structs fails to compile rather than being dropped, and all six already delegate their float comparison to the shared crate::test_support helpers imported at bounds.rs:1205. Command: git show 077dbe2c:crates/cobre-core/src/model/resolved/bounds.rs | sed -n '1905,2020p' ; git grep -n 'use crate::test_support' 077dbe2c -- crates/cobre-core/src/model/resolved/bounds.rs |
| `stations/core-io/td-queue.json#queue[4]` | TD-005 | tb-cobre-core | Premise fails at the pin: the consolidation the seed asks for has already happened. One of the five test sites the seed names, crates/cobre-core/src/topology/network.rs, does not exist at the pin (the topology module declares only the cascade submodule), and each of make_bus, make_line, make_hydro, make_thermal and make_ncs exists exactly once, as a public function in crates/cobre-core/src/test_support.rs, consumed through its spec structs by system/mod.rs, system/builder.rs, topology/cascade.rs and tests/integration.rs. Command: git grep -c 'fn make_bus\|fn make_line\|fn make_hydro\|fn make_thermal\|fn make_ncs' 077dbe2c -- crates/cobre-core/src ; git show 077dbe2c:crates/cobre-core/src/topology/mod.rs ; git log --oneline --diff-filter=D -- crates/cobre-core/src/topology/network.rs |
| `stations/core-io/td-queue.json#queue[6]` | TD-007 | tb-cobre-io | Premise resolved between the seed's mint pin and the register pin. A crate-level test_support.rs was added holding single definitions of write_parquet, write_json, make_global, make_minimal_case, write_file, penalties_all, base_parsed_data and make_unit_group, and validation/semantic/test_support.rs was deleted. The claim that the input path has no shared test-support module while the validation path does is false at the pin; the surviving fixture copies are narrower and are raised separately as the test-support restatement candidate. |
| `stations/core-io/td-queue.json#queue[8]` | TD-009 | tb-cobre-io | Resolved at the pin. The triplicated batch builder and the five common non-determinism cases are now single definitions in the crate test-support module, consumed by all three statistics parsers, and no local make_batch remains. The three per-parser declaration-order tests the seed flagged as load-bearing all survive and each carries an ascending and a descending fixture, so the consolidation preserved the determinism pins. Recorded under positives. |
| `stations/core-io/td-queue.json#queue[10]` | TD-011 | tb-cobre-io | The fact holds at the pin: the bus ordering test's second bus name assertion is the only surface absent from the full-case test, and that test's whole-System comparison implies it. But the only fix shape is removing a test, which this station's cost-discipline rule forbids, and this is one of the ratified removal-direction entries the rule's boundary paragraph names explicitly. Recorded under positives and left for the owner decision on the duplicate-test carve-out rather than re-litigated here. |
| `stations/core-io/td-queue.json#queue[11]` | TD-012 | tb-cobre-io | The fact holds at the pin with shifted lines: test_load_error_io_display sits at 133 and test_load_error_io_helper at 220, both reach the same display facts, and test_load_error_schema_display at 162 re-asserts the bus_id containment the module doctest at line 27 already covers. The only shape that resolves it is removing a test, which the cost-discipline rule forbids, so it is recorded under positives with the other three deletion-only seeds rather than raised as a candidate. |
| `stations/core-io/td-queue.json#queue[13]` | TD-014 | tb-cobre-io | Resolved at the pin. The minimal-case corpus and write_file are now single definitions in the crate test-support module, and tests/helpers/mod.rs re-exports make_minimal_case and write_file from there instead of restating them, so the barrier-forced copy the seed named as its narrowest residue is gone. What remains behind the tests barrier are the two richer case builders, which are raised as their own candidate. |
| `stations/core-io/td-queue.json#queue[14]` | TD-015 | tb-cobre-io | The fact holds at the pin with shifted lines: test_filling_guard_no_exit_no_error is at 1808, test_filling_guard_entry_below_horizon_no_error at 1705, and make_filling_hydro at 1518 sets only entry_stage_id and filling, never exit_stage_id, so the no-exit test cannot exercise the condition it names. The guard's rejection path is covered separately at 1784. The only fix shape is removal, which the cost-discipline rule forbids and which this ratified entry already prescribes, so it is recorded under positives and left to the owner decision. |
| `stations/core-io/td-queue.json#queue[16]` | TD-017 | tb-cobre-io | Every anchor is absent at the pin. thermal.rs is 3784 lines with a single cfg(test) module at 1361 and no nested boundary_tests module, and none of override_at_t_minus_1_acceptance_boundary, test_thermal_bounds_override_stage_within_horizon_accepted, test_thermal_bounds_override_stage_equals_n_rejected or test_thermal_bounds_override_multiple_offending_rows appears anywhere in the file. The unbacked do-not-delete comment the seed relied on has zero matches. The override naming was replaced by a committed_value family, so the premise is dead rather than merely relocated and no anchor is edited into something that resolves. |
| `stations/core-io/td-queue.json#queue[17]` | TD-018 | tb-cobre-io | Premise dead at the pin. output/convergence_reader.rs does not exist, so the byte-identical pair the seed measured cannot be compared. make_config, make_system and make_output_context are now single definitions in the output submodule of the crate test-support module, which is the destination the seed's fix shape implied. |
| `stations/core-io/td-queue.json#queue[18]` | TD-019 | tb-cobre-io | The fact holds at the pin with shifted lines: the exhaustive sweep is at 1336 and the two per-schema description sweeps at 3162 and 3175, with the units test that subsumes the third at 3114 and the descriptions test at 3129. Resolving a strict-subset duplication requires removing a test, which the cost-discipline rule forbids as a fix shape, so it joins the three other deletion-only seeds under positives and the open owner question. |
| `stations/core-io/td-queue.json#queue[22]` | TD-023 | tb-cobre-io | Premise false at the pin. parquet_helpers.rs is 486 lines, not 166, and carries a cfg(test) module at 125 with 33 tests that pin the missing-column and wrong-type message contract directly for all six extractors plus three tests for open_record_batch_reader, so the owner-level coverage the seed said was absent exists. Recorded under positives. |
| `stations/stochastic/td-queue.json#queue[0]` | TD-024 | tb-cobre-stochastic | Premise false at the pin. `git grep -n 'fn make_bus|fn make_hydro' 077dbe2c -- crates/cobre-stochastic/tests` returns nothing: no integration binary declares its own bus or hydro builder any more, they all reach `deficit_bus` at `tests/common/mod.rs:92` and `sized_hydro` at :106. The only per-binary fixture code left in `tests/` is the seven stage wrappers, which the queue[6] candidate raises with its own anchors, so re-raising here would double-claim them. |
| `stations/stochastic/td-queue.json#queue[5]` | TD-029 | tb-cobre-stochastic | Premise false at the pin. `tests/saa_golden_value.rs` no longer declares a local `identity_correlation`; :16 is `use common::{StageSpec, identity_correlation};`, so the copy the seed names has already been replaced by the shared export at `tests/common/mod.rs:74`. Its remaining local declaration is the blockless stage wrapper at :22, raised under the queue[6] candidate. |
