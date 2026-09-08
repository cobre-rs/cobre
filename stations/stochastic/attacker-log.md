# cobre-stochastic attacker dispatch log

Baseline: `a136840d4f2ea137f685f0af6dac04254b983b60` (register pin; HEAD has byte-identical evaluated surfaces under the drift rule) Dispatched: 2026-09-08 Agent type: `adversarial-attacker` (Opus, read-only: no Write/Edit/NotebookEdit tools; every worker returns one JSON object and this session is the sole writer under `plans/`).

## Partition check (before dispatch)

The four-way sweep partition is the one proven total and disjoint in `inventory.json`
(`partitionCheck`: `sum = 51`, `duplicates = []`, `unassigned = []`), re-asserted by the
station test `InventoryTests.test_partition_is_total_and_disjoint`:

```
par 21  sampling 8  tree-noise 16  seam 6   = 51 src files
```

`par` carries its two sibling test files (`par/fitting/tests.rs`,
`par/fitting/estimation/tests.rs`); the test-bloat lens over `par` reaches them.
Sub-station manifests are the `subStations[].files` lists in `inventory.json`; each
worker was handed exactly its cell's manifest.

## Prompts and validator (before dispatch)

One shared worker contract, `attacker-prompt.md`, parameterized per cell by
(lens, sub-station): a do-not-raise table (the 7 `prior-register.md` entries), the
scope walls (cobre-sddp / cobre-io anchors are positives / cross-references, never
findings here), the L1-purity guardrail, the alignment vocabulary, four lens blocks,
and the envelope schema. `tools/validate-envelope.py` was generalized from the core-io
A–D / `perf` / `I.3-n` hard-codings to accept this station's `par|sampling|tree-noise|seam`
sub-stations, the `performance` lens spelling, and the `I.5` Part-I reference, while the
core-io vocabulary still passes. The gate was exercised before dispatch: a well-formed
envelope → exit 0 silent; an out-of-set `alignmentHint` → exit 1 naming
`$.candidates[i].alignmentHint`; a stale-but-well-formed anchor → exit 0 (resolution is
`check-anchors.py`'s job at ingest); prose around the object → exit 1 `$ not parseable JSON`;
a perf candidate quoting a timing → exit 1; a diff-shaped `fixShape` → exit 1.

## Workers (4 lenses × 4 sub-stations)

Dispatched in two waves: the architecture lens first as a 4-worker pilot to confirm the
envelope/anchor shape came back correct, then the remaining 12 once it did.

| worker                          | lens             | sub        | envelope | candidates | note                                                                  |
| ------------------------------- | ---------------- | ---------- | -------- | ---------- | --------------------------------------------------------------------- |
| sto-architecture-par            | architecture     | par        | valid    | 1          | decomposition-vocabulary leakage in L1 doc comments (genericity)      |
| sto-architecture-sampling       | architecture     | sampling   | valid    | 2          | 1 carries I.3-1 (generation/realized straddle)                        |
| sto-architecture-tree-noise     | architecture     | tree-noise | valid    | 3          | stringly-typed entity class across generate/resolve                   |
| sto-architecture-seam           | architecture     | seam       | valid    | 2          | both carry I.3-1; 1 `_needsHuman` (genericity substring `cut_points`) |
| sto-over-engineering-par        | over-engineering | par        | valid    | 2          | single-variant ParWarning; pub forwarding shims                       |
| sto-over-engineering-sampling   | over-engineering | sampling   | valid    | 2          | 4×Option validity matrix; `too_many_arguments` rationale rebutted     |
| sto-over-engineering-tree-noise | over-engineering | tree-noise | valid    | 1          | SweepDirection effectively one-valued in production                   |
| sto-over-engineering-seam       | over-engineering | seam       | valid    | 1          | three dead StochasticError variants                                   |
| sto-performance-par             | performance      | par        | valid    | 3          | rayon determinism rationale quoted and preserved in every fix-shape   |
| sto-performance-sampling        | performance      | sampling   | valid    | 3          | OutOfSample QMC/LHS per-scenario re-derivation                        |
| sto-performance-tree-noise      | performance      | tree-noise | valid    | 3          | per-opening position scan; per-call scratch alloc                     |
| sto-performance-seam            | performance      | seam       | valid    | 2          | quadratic season walk (`reRaiseOf: null` vs stage-calendar entry)     |
| sto-test-bloat-par              | test-bloat       | par        | valid    | 4          | integration-prelude + sibling-test fixture duplication                |
| sto-test-bloat-sampling         | test-bloat       | sampling   | valid    | 2          | inline fixture re-declaration; saa_golden_value prelude               |
| sto-test-bloat-tree-noise       | test-bloat       | tree-noise | valid    | 2          | QMC prelude (cross-refs the Oracle-harness entry, does not restate)   |
| sto-test-bloat-seam             | test-bloat       | seam       | valid    | 2          | seam fixture prelude; tautological season test (`reRaiseOf: null`)    |

All sixteen returned a shape-valid envelope on the first attempt; no re-dispatch was needed.

## Hand-over deviation (recorded, not hidden)

As at core-io, the Agent reply channel is too small to carry an envelope intact, so each
worker wrote its exact final object to the session scratchpad
(`/tmp/claude-1000/…/scratchpad/e03_2/cand.<lens>.<sub>.json`, outside the repository
tree) and replied `WRITTEN <bytes> <path>`. The read-only guardrail over the tree held
(`git status --porcelain --untracked-files=no` names no worker write, and this session is
the sole writer under `plans/`): it copied the sixteen files into `raw/sto-<lens>-<sub>.json`
(re-serialized with `json.dumps(indent=2)`, content unchanged), validated each with
`validate-envelope.py --role attacker --station stochastic`, and merged them.

## Merge (candidates-par|sampling|tree-noise|seam.json)

One merged file per sub-station, the four lenses concatenated. `candidateRef` =
`sto-<lens>-<sub>-<NN>` (`NN` = index in the raw cell envelope). Dedup key = lowercased
title + sorted anchor set; sort key = (lens, first anchor path, title); `positives` and
`_needsHuman` concatenated with a `lens` stamp, never deduped. No exact duplicates were
dropped. The `lenses` map records each worker's candidate count (no `excluded` entry —
every lens returned a valid envelope). Anchor resolution against the baseline is
deliberately not settled here; it is the ingest ticket's `check-anchors.py` pass.

| sub-station | candidates | positives | _needsHuman | severities | partIRef |
| ----------- | ---------- | --------- | ----------- | ---------- | -------- |
| par         | 10         | 24        | 0           | 5×B, 5×C   | —        |
| sampling    | 9          | 16        | 0           | 7×B, 2×C   | I.3-1 ×1 |
| tree-noise  | 9          | 20        | 0           | 5×B, 4×C   | —        |
| seam        | 7          | 22        | 1           | 1×B, 6×C   | I.3-1 ×2 |

Totals: 35 candidates (18×B, 17×C, no Sev-A), 82 positives, 1 `_needsHuman`. All three
`partIRef = I.3-1` carriers are the L1/L2 store-vs-config seam findings (alignment
`advances-1`); they are the only candidates that advance a Part-I generalization fork.

## Prior-register screen

Every candidate anchors inside `crates/cobre-stochastic/` and resolves at the baseline
blob (111/111 anchors, symbol-or-line). A coarse substring pass over the seven
`prior-register.md` `reraiseKey` lists flags four near-collisions; each is adjudicated a
**distinct finding**, not a re-raise of the settled item:

- `sto-architecture-par-00` (decomposition-vocabulary leakage in par doc comments) trips
  the `backward` key of **External-noise take/fill glue duplication** (not-ours, cobre-sddp).
  Distinct: the word is a paradigm-vocabulary term inside an L1 doc comment (a genericity
  concern anchored in `par/`), not the take/fill glue in `cobre-sddp/training/backward`.
- `sto-performance-seam-00` (quadratic season-occurrence walk in `derive_inflow_seeds`)
  trips `backward` and `season_cast`. The worker self-disclosed `reRaiseOf: null`. Distinct
  from **Stage-calendar crate home** (keep): that entry is a _relocation_ question whose
  re-raise bar is L2→L1 layering evidence; this is a performance defect, proposing no move.
- `sto-over-engineering-seam-00` (three dead `StochasticError` variants) trips
  `build_stochastic_context` of **CD-001** via one incidental anchor. Distinct: CD-001 is
  the resolved CLI/pipeline config-projection duplication; this is a dead public error
  taxonomy, and CD-001's own re-raise is not proposed.
- `sto-test-bloat-seam-01` (tautological season-equivalence test) trips `season_cast`. The
  worker self-disclosed `reRaiseOf: null`. Distinct from the relocation entry: a
  test-quality defect in `season_cast/mod.rs`, not a move.

The test-bloat prelude findings over `tests/{halton,sobol,lhs}_integration.rs` and the
`saa_golden_value` / `reproducibility` binaries are anchored in `crates/cobre-stochastic/tests`
and **cross-reference** the **Oracle test-harness duplication** entry (whose anchors are in
`cobre-sddp/tests`) rather than restating it, per that entry's cross-reference disposition.

Every candidate carrying a `reRaiseOf` field set it to `null`; no candidate claims to
re-raise a retired item, and none was found to do so unlabelled.

## Gaps carried forward

None. All sixteen lenses returned a shape-valid envelope with no re-dispatch. Anchor
resolution against the baseline is owed by the ingest ticket's `check-anchors.py` pass, and
the defender ticket owns the per-candidate verdicts; the one `_needsHuman` note
(architecture/seam, the `cut_points` genericity substring) travels to the ingest step for a
human read.
