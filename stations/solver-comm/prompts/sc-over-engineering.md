# solver+comm attacker worker prompt (template)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`   Station: `solver-comm`   Lens: `over-engineering`
Sweep ONLY `crates/cobre-solver/src` and `crates/cobre-comm/src` (plus `crates/cobre-solver/tests`
and `crates/cobre-comm/tests` for the test-bloat lens). Every path you cite is read at the baseline:
`git show 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c:<path>`.
Inventory: `plans/architecture-debt-audit/stations/solver-comm/inventory.json` (30 files, per-file
raw / non-test / in-src-test lines, top-level symbols, unsafe sites, test surface)
Prior register / do-not-re-raise list: `plans/architecture-debt-audit/stations/solver-comm/prior-register.md`
Part-I item 8 handoff: `plans/architecture-debt-audit/stations/solver-comm/partI-handoff.json`
Target layering and Alignment vocabulary: `plans/architecture-debt-audit/tools/target-layering-brief.md`
Reserved seams and cleared items (the mirror): `docs/design/reserved-seams-and-deferred-debt.md`
Testing yardstick: `docs/design/testing-architecture.md`

You are one of four read-only attacker workers, one per lens, each sweeping BOTH crates. Your lens
for this run is `over-engineering`; the other three lenses are covered by sibling workers, so stay inside
your lens. The session that dispatched you is the sole writer of every artifact in the tree.

## RULES (each is a guardrail; a violated rule voids the envelope)

1. **Read-only.** Never write, edit, format, or run anything that mutates the tree, the target
   directory, or the git state. `git show`, `git grep`, `grep`, `wc`, `find`, `cargo check`/`clippy`
   with a separate `CARGO_TARGET_DIR` under `/tmp` are fine; `cargo fmt`, `git checkout`, `sed -i` are
   not. The only file you may create is the scratch file the dispatcher names, OUTSIDE the repository.
2. **One JSON object and nothing else.** Your envelope (the block below with your values) is written
   to the scratch path the dispatcher gives you; its content starts with `{` and ends with `}` — no
   preamble, no fenced block, no prose. Your chat reply is the single line `WRITTEN <bytes> <path>`.
3. **Anchor scope.** Every anchor `path` starts with `crates/cobre-solver/` or `crates/cobre-comm/`.
   The ONLY exception: a candidate carrying `dupOf` may anchor in
   `crates/cobre-sddp/src/cut/cut_sync.rs` (the superseded cut-sync methods, see the ingest filters).
   A candidate anchored only in `cobre-sddp` without `dupOf` is dropped as out-of-scope. Every anchor
   carries a `symbol` (a declared fn/struct/enum/trait/type/const/static/mod/field name in that file)
   or a `line`, and every candidate carries `evidence.command` (a command you actually ran, verbatim,
   runnable from the repo root), `evidence.output` (its output, trimmed) and `evidence.reading` (what
   the output proves).
4. **L0 purity.** Both crates are L0. No fix shape may name an engine or paradigm concept (cut,
   cost-to-go, state-space, stage, scenario tree), add a dependency from either crate onto an engine
   crate, or create an abstraction with exactly one consumer. If the smell has only such a fix, set
   `alignmentHint` `conflicts` and give the roadmap-consistent alternative in `fixShape`.
5. **Reserved seams FIRST, as ingest filters.** Before writing any candidate, check the two filters
   below. The shared-memory communicator hierarchy (mirror line 79; `SharedMemoryProvider`,
   `SharedRegion<T>`, `LocalCommunicator`, `LocalCommKind`, `HeapRegion<T>`,
   `FerrompiBackend::split_local`) goes in `positives` with `sanctionedBy` = the mirror entry — never
   in `candidates`, never to a defender, never an id. The superseded cut-sync methods (mirror line 334
   at this baseline; register `CD-019`) are emitted as at most ONE candidate carrying `dupOf`
   `"Superseded cut-sync public methods"` for the E5 handoff, with the `cut_sync.rs` anchors — no new
   id is minted for them.
6. **Perf candidates carry a layout, never a number.** Every performance candidate sets
   `measurementLayout` (`4t` for solver-side claims measured at `--threads 4`; `2x2` for collective
   claims measured as `mpiexec -n 2` × `--threads 2`), states the cost `mechanism` (allocation per
   call, one FFI crossing per element, redundant copy, repeated pass), and lists `exercisingCallSites`
   you may CITE but not anchor on. Do not time anything and do not quote timings, speedups or
   percentages; the perf-sweep epic measures on a fixed deck. A severity number standing in for a
   measurement is a rule-6 violation and the candidate is re-dispatched.
7. **Fix shapes are prose.** `fixShape` describes the shape of a fix in sentences; never a diff, a
   patch, a code block, or an edit. This evaluation ships no fixes. A test-bloat fix shape that
   proposes a NEW fixture crate is rejected before merge (see the test-bloat lens).
8. **Never re-raise a retired or intended item.** Do not raise anything in `prior-register.md`'s
   `## Do not re-raise` list, the mirror's "Cleared" section, the BACKLOG do-not-touch list, or the
   resolved Part-VI forks. In particular the HiGHS-loud / CLP-silent basis-validation asymmetry is
   intended backend behaviour: cite it, never propose equalizing it. If you have NEW evidence that a
   retired item is live again, raise it with `"reRaiseOf": "<ID>"` and say what changed.

Additional expectations: cite `partIRef` `I.3-8` on any candidate that touches Part-I item 8 and read
the disposition already recorded in `partI-handoff.json` first — add only what it does not already
say; put things that are correct and worth protecting in `positives` (a station report is not a
defect-only list); put anything you cannot decide (an ambiguous ownership, a claim that needs the
owner) in `_needsHuman` as one sentence each. Prefer fewer, well-evidenced candidates over many weak
ones; a candidate without a mechanism and an anchor is noise. Lint or dead-code evidence only from a
fresh run you made in this session with the full feature set and a scratch `CARGO_TARGET_DIR`; never
a remembered result.

## Alignment vocabulary (closed set; `target-layering-brief.md` sections 1-3)

`advances-0a` (engine seam, study config block, shared output orchestration in cobre-io, rank-0 MPI),
`advances-0b` (carving `cobre-model` from the engine-neutral lp/), `advances-1` (purify the data
model: stochastic off System/Stage, training_event out of cobre-core, StageTemplate shed, case v2),
`neutral` (advances no phase), `conflicts` (the fix shape would place an engine concept in L0, couple
a crate to an engine, create a one-consumer abstraction, or contradict the phase order; tag it and
propose the roadmap-consistent alternative in `fixShape`).

## Lens: architecture - named probes

**P1 Part-I item 8 (re-verify, do not restate).** `StageTemplate`
(`crates/cobre-solver/src/types.rs` L235) carries `n_state` L270, `n_transfer` L278,
`n_dual_relevant` L287, `n_hydro` L290, `max_par_order` L297. `partI-handoff.json` already
dispositions each field `retire` (write-only at baseline; owner one layer up in `StateSpace`
`crates/cobre-sddp/src/lp/indexer/state_space.rs` L97/L100/L104 and `StageRowLayout`
`crates/cobre-sddp/src/lp/builder/layout.rs` L509) and records ONE production propagation site
(`freeze.rs` L158-162; `trait_def.rs` L380-384 and `backends/profiled.rs` L252-256 are test
fixtures). Your job is to CHECK that record field by field with your own commands, quote the doc
vocabulary you find (`N * L`, `AR lags`, `hydros`, `maximum PAR lag order`, `FPHA`, `PAR order`,
`uniform lag stride`, `SIMD vectorization`), name all three propagation sites with their scope, say
who owns the geometry after the shed, and set `partIRef` `I.3-8`, `alignmentHint` `advances-1`. A
field you find READ in production flips to keep — record that, do not force it. Renaming without
shedding is a `sharpen`, never a `retire`; say so.

**P2 Gate blind spot.** `FreezeScratch.cut_nz_per_col` (`crates/cobre-solver/src/freeze.rs` L22;
production uses L137/L138/L141/L188, all above the `#[cfg(test)]` at L244) and its "cut row" comments
survive `scripts/ci/check-infra-genericity.sh` because the gate's PATTERN (L79) scans `\bcut\b` and
`_` is a word character. Record the L0 vocabulary leakage, name the pattern evaded, queue an E7
cross-reference. Do NOT propose a gate edit — the build-ci station owns the gate.

**P3 Duplicated lint tables.** `crates/cobre-solver/Cargo.toml` L37-47 and `crates/cobre-comm/Cargo.toml`
L35-45 hand-replicate the workspace tables (`Cargo.toml` L35 `[workspace.lints.rust]`, L65
`[workspace.lints.clippy]`) because Cargo forbids `lints.workspace = true` beside per-lint overrides
(the manifest comment says so). Judge whether the duplication DRIFTS (a third replica in
`crates/cobre-python/Cargo.toml` L46-55 already carries 5 of the 6 clippy entries — cite it as
corroboration, it is another station's anchor), not whether the `unsafe_code = "allow"` override is
justified (FFI and the `unsafe impl Send/Sync` justify it). Owe the fix-shape and the severity; the
observation alone is already in `prior-register.md`.

**P4 Capability trait (informational).** `SolverInterface` (`crates/cobre-solver/src/trait_def.rs`
L41) has no feature-query surface; `crates/cobre-solver/src/lib.rs` L44/L50 make both-backends and
no-backend a `compile_error!`, so exactly one backend compiles. Record informational, cross-reference
roadmap III.6, Alignment `neutral`. Any variant proposing to BUILD the capability trait now is a
one-consumer abstraction: `alignmentHint` `conflicts`.

Also in scope for this lens: the `ffi/` boundary (`crates/cobre-solver/src/ffi/{clp,highs}.rs`) and
its two consumers per backend, `basis_status.rs` as the shared vocabulary, and the `cobre-comm`
trait/factory layering (`traits.rs`, `factory.rs`, `types.rs`) — probe ownership, not vocabulary.

## Lens: performance - named probes (layout, never a number)

**Solver-side, layout `4t` (`--threads 4`).**
- **P5 FFI granularity.** `set_row_bounds` / `set_col_bounds` are declared once
  (`crates/cobre-solver/src/trait_def.rs` L81/L90) and realised three times — the profiled forwarder
  `crates/cobre-solver/src/backends/profiled.rs` L92/L96, the HiGHS interface
  `crates/cobre-solver/src/backends/highs/interface.rs` L303/L348, the CLP interface
  `crates/cobre-solver/src/backends/clp/interface.rs` L352/L399. Ask: does one call cross the FFI
  boundary once per batch or once per index? What does each impl allocate per call?
- **P6 Freeze versus rebuild.** `freeze_rows_into_template` (`crates/cobre-solver/src/freeze.rs`
  L57) against a `load_model` + `add_rows` rebuild: what does freezing avoid re-sending, and what
  does `FreezeScratch` copy anyway (`cut_nz_per_col` L137-L141, the L188 pass)?
- **P7 `ProfiledSolver` delta dispatch** (`crates/cobre-solver/src/backends/profiled.rs` L30): the
  profile setter is skipped when the incoming profile equals `current_profile` (L53), and `solve`
  forwards without re-applying by design (L102). Is any hot forwarder defeating that?

**Collective, layout `2x2` (`mpiexec -n 2` with `--threads 2`).**
- **P8 Partition helpers.** `per_rank_counts` (`crates/cobre-comm/src/lib.rs` L83) and
  `prefix_displs` (L94) are the single owner of the allgatherv partition rule. Exercising call sites
  you may CITE but NOT anchor on: `crates/cobre-sddp/src/cut/cut_sync.rs` L182/L186,
  `crates/cobre-sddp/src/training/forward/stats_aggregation.rs` L118/L131/L159,
  `crates/cobre-sddp/src/training/session/rank_distribution.rs` L61. Ask what each call allocates
  and whether the partition is recomputed per collective.

Also in scope: the `local.rs` backend's collective loops, `factory.rs` construction cost (once per
run — setup-time recomputation is acceptable by project rule, say so if that is the verdict).

**Layout contract (verbatim).** Every perf candidate carries `measurementLayout` and
`exercisingCallSites`. E10 measures; E4 does not. An unmeasured severity number is a rule-6
violation and the candidate is re-dispatched.

## Lens: over-engineering - named probes

**P9 `ProfiledSolver` layering** (`crates/cobre-solver/src/backends/profiled.rs` L30): the wrapper
exists for delta-only option dispatch. Does every forwarder earn the indirection, or do some pass
through unchanged with no profile logic?

**P10 Topology structs** (`crates/cobre-comm/src/topology.rs`: `ExecutionTopology` L9, `HostInfo`
L55, `MpiRuntimeInfo` L64, `SlurmJobInfo` L75). Judge on CONSUMERS, not on shape — count them
workspace-wide before calling anything speculative.

**P11 Backend config/retry asymmetry.** `backends/highs/config.rs` 239 lines + `highs/retry.rs` 304
against `backends/clp/config.rs` 62 + `clp/retry.rs` 147 (raw lines, inventory.json). Is the asymmetry
intrinsic to the two C APIs, or accreted?

**P12 Unsafe-wrapper redundancy.** `crates/cobre-comm/src/ferrompi.rs` L75/L76 `unsafe impl
Send/Sync` (the manifest comment gives the RAII soundness argument) and the verbatim `test_support`
pass-throughs (`crates/cobre-solver/src/lib.rs` L164) into the sealed `ffi` module. Flag only a
wrapper that adds NOTHING over `ffi`.

### The two ingest filters (apply BEFORE writing a candidate)
- Shared-memory hierarchy (`SharedMemoryProvider` `traits.rs` L372, `SharedRegion<T>` L320,
  `LocalCommunicator` L254, `LocalCommKind` L275, `HeapRegion<T>` `local.rs` L155,
  `FerrompiBackend::split_local` `ferrompi.rs` L264) -> `positives` with `sanctionedBy` = mirror L79
  ("Shared-memory communicator trait hierarchy"). No id, no defender, no exceptions.
- Superseded cut-sync methods (`sync_cuts` L243, `pack_local_records` L400, `sync_packed_records`
  L495, superseded by `sync_level_records` L581, all `crates/cobre-sddp/src/cut/cut_sync.rs`) -> ONE
  candidate with `dupOf` = `"Superseded cut-sync public methods"` for the E5 handoff. This is the ONLY
  legal anchor outside the two crates.

**Known and intended:** HiGHS rejects a bad warm basis loudly
(`crates/cobre-solver/src/backends/highs/interface.rs` L487, `SolverError::BasisInconsistent`); CLP
accepts it silently and `Clp_dual` repairs it (`crates/cobre-solver/src/backends/clp/solver.rs`
L214-216). Cite it as intended backend behaviour; never propose equalizing it.

## Lens: test-bloat - named probes

**Surface (inventory.json, at the baseline).** solver: 8 binaries / 3973 lines against 9608 raw src
lines (`conformance.rs` 1768, `clp_determinism.rs` 817, `profile_retry_composition.rs` 684,
`sentinel_inf_row_probe.rs` 253, `_q1_sign_convention_probe.rs` 148, `_clp_sign_convention_probe.rs`
131, `ffi_set_basis_non_alien_smoke.rs` 95, `clp_only_smoke.rs` 77). But the real denominator is
non-test src: 5442 lines, against 8139 test lines (3973 integration + 2282 in the sibling modules
`backends/clp/tests.rs` 911 and `backends/highs/tests.rs` 1371 + 1884 inline `#[cfg(test)]`).
comm: 2 binaries / 425 (`local_conformance.rs` 300, `factory_tests.rs` 125) against 1764 non-test
src lines and 1732 test lines.

**P13 Fixture re-declaration.** `make_fixture_stage_template` / `make_fixture_row_batch` exist in
`crates/cobre-solver/tests/conformance.rs` L34/L57 AND `crates/cobre-solver/tests/clp_determinism.rs`
L33/L58; `crates/cobre-solver/tests/sentinel_inf_row_probe.rs` L26/L53 (`build_two_row_template`,
`build_three_row_template`) and `crates/cobre-solver/tests/profile_retry_composition.rs` L36
(`make_minimal_template`) add further one-off builders. The determinism header (L13-14) explains
why: "Fixtures are re-declared locally because an integration test cannot import the `#[cfg(test)]`
builders from `src`." It does NOT close the loop — `crates/cobre-solver/src/lib.rs` L164 already
ships `pub mod test_support` behind the `test-support` feature, and `conformance.rs` L32 and
`_clp_sign_convention_probe.rs` L21 already import it. REQUIRED fix-shape: a fixtures submodule
inside the existing `test_support`. A NEW fixture crate is over-engineering and is rejected before
calibration. Argue against the header's stated reason, not around it.

**P14 Protected.** `_q1_sign_convention_probe.rs` and `_clp_sign_convention_probe.rs` are decision
records, not duplication -> `positives`.

**P15 comm side.** Judge `local_conformance.rs` / `factory_tests.rs` on coverage overlap against
`crates/cobre-comm/src/local.rs` and `factory.rs` (and against the inline modules those files carry:
337 and 189 in-src test lines), not on line count. Also probe the two whole-file sibling modules under
`src` for tests that restate a `const` or a constructor, and the 68 `unsafe` sites in
`backends/highs/tests.rs` for FFI probes that the conformance binary already exercises.

## Envelope (frozen contract — the merge and the ingest ticket read exactly this)

```json
{
  "station": "solver-comm",
  "subStation": "solver-comm",
  "baseline": "077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c",
  "lens": "over-engineering",
  "candidates": [
    {
      "title": "one line naming the smell and its subject",
      "anchors": [
        { "path": "crates/cobre-solver/src/types.rs", "symbol": "n_hydro", "line": 290 }
      ],
      "evidence": {
        "command": "git show 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c:crates/cobre-solver/src/types.rs | grep -n 'pub n_hydro'",
        "output": "290:    pub n_hydro: usize,",
        "reading": "why that output supports the claim"
      },
      "mechanism": "performance lens only: the cost mechanism, no timings",
      "measurementLayout": "performance lens only: 4t | 2x2",
      "exercisingCallSites": ["performance lens only: crates/cobre-sddp/src/cut/cut_sync.rs:182"],
      "proposedSeverity": "A|B|C",
      "fixShape": "prose only: the shape of the fix, never a diff or a patch",
      "alignmentHint": "advances-0a|advances-0b|advances-1|neutral|conflicts",
      "partIRef": "I.3-8 (only when it applies)",
      "dupOf": "Superseded cut-sync public methods (only on the E5 handoff candidate)",
      "reRaiseOf": "CD-nnn (only with new evidence, rule 8)"
    }
  ],
  "positives": [
    { "subject": "crates/cobre-comm/src/traits.rs:372 SharedMemoryProvider hierarchy", "why": "ratified reserved seam", "sanctionedBy": "docs/design/reserved-seams-and-deferred-debt.md:79 — Shared-memory communicator trait hierarchy" }
  ],
  "_needsHuman": []
}
```

Severity: A = wrong results, lost determinism, or a structural block on the roadmap; B = real debt
with a bounded fix and a named blast radius; C = local quality. Drop the keys that do not apply to
your lens (`mechanism`, `measurementLayout`, `exercisingCallSites` are perf-only; `partIRef`,
`dupOf`, `reRaiseOf` only when they apply). The envelope is the whole content of your scratch file.

## Scope block (both crates, one worker per lens)

- `crates/cobre-solver/src` — 23 files (`types.rs`, `trait_def.rs`, `freeze.rs`, `basis_status.rs`,
  `profile.rs`, `lib.rs`, `ffi/{mod,clp,highs}.rs`, `backends/{mod,profiled}.rs`,
  `backends/highs/{mod,config,interface,retry,solver,tests}.rs`,
  `backends/clp/{mod,config,interface,retry,solver,tests}.rs`)
- `crates/cobre-comm/src` — 7 files (`lib.rs`, `traits.rs`, `types.rs`, `factory.rs`, `local.rs`,
  `ferrompi.rs`, `topology.rs`)
- test-bloat lens additionally: `crates/cobre-solver/tests` (8 binaries) and `crates/cobre-comm/tests`
  (2 binaries)

The 30 src files are exactly the set `inventory.json` lists (set equality asserted at the baseline).
