# cobre-sddp attacker dispatch log

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (register pin; HEAD's evaluated surfaces are
byte-identical under the drift rule; the ticket text's `a136840d` is the superseded pin). Dispatched:
2026-09-18. Agent type: `adversarial-attacker` (Opus, read-only). Dispatcher: the main session, sole
writer of every artifact under `stations/sddp/`.

## Partition check (before dispatch)

The four-way sweep partition is the one proven total and disjoint in `inventory.json`
(`coverage.src`: find_count 163, assigned_count 163, `unassigned = []`, `double_assigned = []`),
re-asserted by `InventoryTests.test_the_four_sub_stations_partition_the_modules` and rendered
verbatim into the prompt's `## Manifests` section by the prompt generator:

```
5a 28 files 41,601 lines   5b 30 files 44,682   5c 58 files 61,104   5d 47 files 42,856   = 163 / 190,243
```

Every path a worker may anchor lies in exactly one manifest; the test-bloat lens additionally reads
`crates/cobre-sddp/tests` (56 files: 40 binaries + 5 `common/` + 11 `template_integration/`).

## Prompt and validator (before dispatch)

One shared contract, `attacker-prompt.md`: preamble + twelve rules (read-only; scratch-file
hand-over `WRITTEN <bytes> <path>`; SYMBOL ANCHORS ONLY with the reason; evidence-as-command;
contract-first against `.claude/rules/sddp.md`; byte-neutrality bar; prose fix-shapes; perf layout
never a number; empty-is-legitimate; never re-raise; L3 layering → `conflicts`; cross-sub-station
duplication raised once), the alignment vocabulary, the do-not-raise set (settled with closing
commits, the four reserved seams with citations, the disowned ids, the 19 live dispositions as a
sharpen-only table, CD-074 pre-listed), four lens blocks (architecture incl. the CD-074 class sweep;
performance with the four pre-resolved targets; over-engineering with the reserved-is-not-dead
precondition and the enum / `#[allow]` census; test-bloat with the census and three questions), the
four manifests, and the envelope. The gate is `tools/validate-envelope.py --role attacker --station
sddp` (shape) plus the station rules the shared validator does not know: symbol-only anchors,
`claimType` on every performance candidate, `sanctionedBy` on every reserved-seam positive, anchors
resolved at the pin through `check-anchors.py` over a rendered stub, and the prior-register screen
through `check-reraise.py` plus explicit id/anchor matching against `wave-dispositions.json`.

Gate policies fixed on the first wave (recorded so the verdicts are reproducible):

- **Anchor rejection is per anchor, not per candidate.** A line-only anchor, an anchor outside
  `crates/cobre-sddp` (rule 3 lets a worker CITE cobre-cli, never anchor it) or an anchor outside the
  cell's manifest is moved to the candidate's `citedContext` with its reason; the candidate is dropped
  `anchor-missing` only when no anchor survives. Exception (rule 12): the two `enumerated.rs` files may
  be anchored from either of 5c/5d. The same per-anchor rule applies to `check-anchors` failures: a
  struct FIELD anchor (the prompt's rule 3 admits field names, but the register checker's DECL regex
  resolves declarations only) is demoted to `citedContext` as "field anchor" and the candidate keeps
  its declaration anchors; a symbol that resolves to nothing at the pin is `check-anchors symbol-missing`.
- **`dupOf` needs real overlap.** A candidate is tagged `dupOf <live id>` when it shares two anchor
  symbols with a live disposition, or one symbol plus title overlap (Jaccard ≥ 0.3) with its claim; a
  single shared god-fn (`compute_one_backward_node`, `run_enumerated_backward`, …) is only annotated
  `relatedTo`. A candidate carrying `reRaiseOf` is a sharpen and is neither.
- **`check-reraise` hits.** An explicit-id or title-overlap hit drops the candidate `settled`; an
  anchor-overlap hit against a file-scoped retired item is annotated `retiredOverlaps` (register
  precedent: `Re-raise-of … NOT a re-raise`) and the candidate is kept.
- **Envelope station spelling.** The ticket's `station: "sddp/5c"` is split into the validator's
  `station: "sddp"` + `subStation: "5c"`; the composite is carried as `cell: "sddp/5c"`.

## Dispatch schedule — 16 cells, 4 concurrent workers, one lens per wave

| wave | lens             | cells          |
| ---- | ---------------- | -------------- |
| 1    | architecture     | 5a 5b 5c 5d    |
| 2    | performance      | 5a 5b 5c 5d    |
| 3    | over-engineering | 5a 5b 5c 5d    |
| 4    | test-bloat       | 5a 5b 5c 5d    |

Rules the scheduler enforces: four at a time, never sixteen (each worker reads the shared prompt plus
a multi-thousand-line manifest; a 16-way fan-out starves every cell); one lens per wave, not one
sub-station per wave (a lens block is identical across its four cells, so a wave is reproducible and a
crashed wave is resumable without re-running clean cells); no worker sees another worker's output
(the 5c/5d enumerated duplication is merged at the gate); each cell writes exactly one scratch file
`/tmp/sddp-attackers/out/<lens>.<sub>.json`, copied verbatim to `raw/` in scratch and gated into
`candidates.<lens>.<sub>.json`; a cell whose envelope fails the gate is re-dispatched exactly once with
the validator error quoted, a second failure is `needs-human`, never hand-repaired.

## Per-cell record shape

```json
{
  "cell": "architecture.5a",
  "wave": 1,
  "dispatchedAt": "2026-09-18T..Z",
  "promptBytes": 0,
  "validator": "pass | fail-redispatched | needs-human",
  "candidates": 0,
  "positives": 0,
  "needsHuman": 0,
  "droppedPriorScreen": [{ "title": "...", "reason": "settled: CD-001 closed by b051c410" }],
  "droppedAnchor": [{ "title": "...", "reason": "anchor-missing: line-only anchor" }],
  "mergedDupOf": [{ "title": "...", "into": "architecture.5c candidate <n>" }]
}
```

## Coverage matrix (4 lenses × 4 sub-stations)

|                  | 5a setup+policy | 5b lp/ | 5c cut+training+solve+workspace | 5d simulation+production |
| ---------------- | --------------- | ------ | ------------------------------- | ------------------------ |
| architecture     | 4/5/0 | 5/10/2 | 5/7/1 | 2/5/2 |
| performance      | 3/14/2 | 3/7/1 | 7/8/2 | 8/18/2 |
| over-engineering | 4/14/1 | 2/9/1 | 2/13/1 | 3/16/1 |
| test-bloat       | 7/8/0 | 8/5/0 | 6/6/2 | 6/5/0 |

n/p/h = candidates kept / positives / needs-human. A cell reading 0/0/0 is a FAILED cell; a clean cell reads 0/1+/0.

## Dispatch records

### Wave 1 — architecture (dispatched 2026-09-18, four concurrent `adversarial-attacker` Opus workers)

Prompt = `attacker-prompt.md` (42,191 bytes, read by the worker) + a per-cell directive (~2 KB) naming
the cell, its manifest, its lens questions and the scratch path `/tmp/sddp-attackers/out/architecture.<sub>.json`.

| cell            | wave | directive focus                                                                   | scratch path                                  |
| --------------- | ---- | --------------------------------------------------------------------------------- | --------------------------------------------- |
| architecture.5a | 1    | Q1 setup lifecycle (CD-005/CD-004 sharpen), Q4 policy seams, Q5 class sweep       | /tmp/sddp-attackers/out/architecture.5a.json |
| architecture.5b | 1    | Q5 CD-074 class sweep over lp/ state families (+ lp/ architecture smells)         | /tmp/sddp-attackers/out/architecture.5b.json |
| architecture.5c | 1    | Q2 backward drivers, Q3 enumerated duplication (both files), Q5 stage_solve_prep  | /tmp/sddp-attackers/out/architecture.5c.json |
| architecture.5d | 1    | Q3 enumerated duplication (both files), Q5 production/simulation families         | /tmp/sddp-attackers/out/architecture.5d.json |

### Wave 2 — performance (pipelined: a cell is dispatched as soon as its wave-1 sibling returns, keeping four workers running)

| cell           | wave | directive focus                                                                              | scratch path                                 |
| -------------- | ---- | -------------------------------------------------------------------------------------------- | -------------------------------------------- |
| performance.5b | 2    | target 3 PatchBuffer fill family + lp/ builder/indexer hot-path allocation                   | /tmp/sddp-attackers/out/performance.5b.json |
| performance.5d | 2    | simulation/production hot loops (pipeline, enumerated, extraction, aggregation); PD-001 refuted | /tmp/sddp-attackers/out/performance.5d.json |
| performance.5c | 2    | targets 1, 2, 4 (gemm sweep, SuccessorOutcomes gather, basis reconstruct) + cut/training/solve hot loops; PD-004 deferred | /tmp/sddp-attackers/out/performance.5c.json |
| performance.5a | 2    | setup pipeline + policy load path allocation / redundant passes (setup-time vs per-solve reach stated) | /tmp/sddp-attackers/out/performance.5a.json |

### Wave 3 — over-engineering (pipelined behind wave 2)

| cell                | wave | directive focus                                                                              | scratch path                                      |
| ------------------- | ---- | -------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| over-engineering.5b | 3    | reserved-seam precondition; enum + #[allow] census over lp/; indexer/builder one-consumer indirections | /tmp/sddp-attackers/out/over-engineering.5b.json |
| over-engineering.5c | 3    | CD-019 positive; enum + #[allow] census (OD-009 sharpen-only); do gemm/claim_scatter/solver_stats/convergence earn their modules | /tmp/sddp-attackers/out/over-engineering.5c.json |
| over-engineering.5d | 3    | anticipated-channel seam positive; enum + #[allow] census; do hull/lead_time/echo modules earn their modules; error.rs + lib.rs surface | /tmp/sddp-attackers/out/over-engineering.5d.json |
| over-engineering.5a | 3    | slot-body + Legacy cost-scale seams as positives; enum + #[allow] census; setup/ new-noun sprawl (scenario_library_set.rs 46 lines), wrappers, ctor-only structs | /tmp/sddp-attackers/out/over-engineering.5a.json |

### Wave 4 — test-bloat (pipelined behind wave 3; yardstick `docs/design/testing-architecture.md`)

| cell          | wave | directive focus                                                                              | scratch path                                |
| ------------- | ---- | -------------------------------------------------------------------------------------------- | ------------------------------------------- |
| test-bloat.5b | 4    | Q1 inline-vs-sibling split in lp/builder (CD-007 sharpen); Q3 harness duplication in the lp-exercising binaries + template_integration/ | /tmp/sddp-attackers/out/test-bloat.5b.json |
| test-bloat.5c | 4    | Q2 parity_hash golden membership (10 cases; determinism/rank-invariance gates informational); Q1 backward/tests.rs vs inline giants; Q3 cut/backward/mpi binaries vs tests/common | /tmp/sddp-attackers/out/test-bloat.5c.json |
| test-bloat.5d | 4    | Q1 simulation/production sibling tests.rs vs inline; test_support.rs (3,652) vs tests/common/builders.rs; Q3 simulation/boundary/anticipated binaries | /tmp/sddp-attackers/out/test-bloat.5d.json |
| test-bloat.5a | 4    | Q1 setup/tests.rs (10,356) vs inline giants (node_graph, policy_load, policy_export, noise); Q3 boundary/anticipated/estimation/load binaries vs tests/common | /tmp/sddp-attackers/out/test-bloat.5a.json |

## Prior-register screen (what was dropped and why)

| candidate title | cell | action | reason |
| --------------- | ---- | ------ | ------ |
| Enumerated forward and enumerated simulation are a twice-declared sweep skeleton: three parallel scratch/visit | architecture.5d | merged | same enumerated-sweep duplication as architecture.5c 'CD-028 sharpened: the shared claim-and-scatter owner already exists and its module doc declares a boundary narrower than the duplication, so what  |
| process_stage_backward allocates a fresh owned staged-cut vector per worker per node at its return boundary, d | performance.5c | dup-of | kept, tagged dupOf CD-016 (shared symbols process_stage_backward, process_stage_backward_by_node) |

Actions are exactly: settled | sanctioned | dup-of | merged | anchor-missing | needs-human.

## Cross-cell overlaps inside a lens (handed to ingest, not merged here)

Pairs of kept candidates from different sub-stations under the same lens that share an anchor symbol or whose titles overlap (Jaccard ≥ 0.35). The ingest/defender pass decides whether each pair is one finding or two adjacent ones; only the 5c/5d enumerated duplication was merged at this gate (rule 12).

| cell A | title A | cell B | title B | shared symbols | title Jaccard |
| ------ | ------- | ------ | ------- | -------------- | ------------- |
| test-bloat.5a | Inline-vs-sibling test homing in setup/, policy/ and stochastic/ is not just una | test-bloat.5d | The sibling-versus-inline test homing split across simulation and production is  | tests | 0.22 |
| test-bloat.5a | HydroPenalties is hand-enumerated field by field 46 times in setup/tests.rs and  | test-bloat.5c | HydroPenalties has no Default, so fixtures spell all sixteen fields; the crate a | neutral_hydro_penalties | 0.06 |
| test-bloat.5a | HydroPenalties is hand-enumerated field by field 46 times in setup/tests.rs and  | test-bloat.5d | anticipated_core.rs re-declares md5-identical hydro-default fixtures six times i | HydroSpec | 0.03 |
| test-bloat.5a | The four right_boundary_* binaries each re-declare the same ResolvedPenalties, R | test-bloat.5d | The four right_boundary binaries carry md5-identical 40-to-41-line fixture prolo | bounds, penalties | 0.11 |
| test-bloat.5a | anticipated_core.rs declares build_config 14 times, build_system 13 times and de | test-bloat.5c | HydroPenalties has no Default, so fixtures spell all sixteen fields; the crate a | default_hydro_penalties | 0.03 |
| test-bloat.5a | anticipated_core.rs declares build_config 14 times, build_system 13 times and de | test-bloat.5d | anticipated_core.rs re-declares md5-identical hydro-default fixtures six times i | build_config, build_system, default_hydro_bounds, default_hydro_penalties | 0.24 |
| test-bloat.5a | state_layout_for is cloned byte-identically into seven integration binaries unde | test-bloat.5d | Five integration binaries hand-copy test_support helpers behind a doc claim that | state_layout_for | 0.18 |
| test-bloat.5a | A test-only reference oracle for the lag-shift kernel is homed in the hot-path p | test-bloat.5d | The sibling-versus-inline test homing split across simulation and production is  | tests | 0.03 |
| test-bloat.5b | The four-line `case_dir` helper is declared twice inside tests/lp_builder.rs alo | test-bloat.5c | cut_basis.rs re-declares five fixture helpers two and three times inside a singl | build_setup_for_case | 0.07 |
| test-bloat.5c | No Communicator test double lives in the shared test_support surface, so src-sid | test-bloat.5d | The shared harness carries two hardcoded single-shape communicators while one bi | Rank0Of2, StubComm | 0.11 |
| test-bloat.5c | HydroPenalties has no Default, so fixtures spell all sixteen fields; the crate a | test-bloat.5d | anticipated_core.rs re-declares md5-identical hydro-default fixtures six times i | default_hydro_penalties | 0.06 |

## Per-cell records

```json
{
 "architecture.5a": {
  "cell": "architecture.5a",
  "wave": 1,
  "validator": "pass",
  "candidates": 4,
  "positives": 5,
  "needsHuman": 0,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [
   {
    "title": "The run-parameter half of StudySetup uses two incompatible conventions (named projections vs raw config types), so train_inner re-inflates TrainingConfig field-by-field out of the god-struct",
    "refs": [
     "mirror:water-travel-time-in-transit-bucket-topology-studysetup-transit_bucket_topology"
    ]
   }
  ],
  "rawCandidates": 4,
  "promptBytes": 42401,
  "anchorsRejected": [
   {
    "title": "The MPI wire config is not a superset-projection of StudyParams: one field is computed on params then dropped and re-read from Config, and two run-mode fields exist only on the wire",
    "rejected": [
     "crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig (anchor outside the station: crates/cobre-cli/src/commands/broadcast.rs)"
    ]
   },
   {
    "title": "scalar_parameters is a caller-patched placeholder rather than a constructor input, so both validate paths build stage templates against an empty parameter table and silently resolve every parameterized coefficient to 0.0",
    "rejected": [
     "crates/cobre-cli/src/commands/validate.rs::reconcile_boundary (anchor outside the station: crates/cobre-cli/src/commands/validate.rs)"
    ]
   },
   {
    "title": "CutManagementConfig::warm_start_cuts is a public field with a capacity contract in its doc, hard-coded to 0 at both production construction sites and read by no production consumer",
    "rejected": [
     "crates/cobre-sddp/src/cut/fcf.rs::pool_capacity (anchor outside the 5a manifest: crates/cobre-sddp/src/cut/fcf.rs (5c))"
    ]
   }
  ]
 },
 "architecture.5b": {
  "cell": "architecture.5b",
  "wave": 1,
  "validator": "pass",
  "candidates": 5,
  "positives": 10,
  "needsHuman": 2,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 5,
  "promptBytes": 42401
 },
 "architecture.5c": {
  "cell": "architecture.5c",
  "wave": 1,
  "validator": "pass",
  "candidates": 5,
  "positives": 7,
  "needsHuman": 1,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [
   {
    "title": "The cut-binding metadata channel is a sampled-traversal-only side duty: the enumerated backward commits cuts but records no binding activity, so DCS resident-set seeding silently degrades to a generation-age filter on a combination no admission gate rejects",
    "refs": [
     "PD-004",
     "mirror:performance-debt-follow-ups-first-performance-pass-2026-08-18"
    ]
   },
   {
    "title": "CD-028 sharpened: the shared claim-and-scatter owner already exists and its module doc declares a boundary narrower than the duplication, so what remains twice-declared between the enumerated forward and the enumerated simulation is the path marking and the worker-result fold that sit outside any stated boundary -- the fix is extending an owner with two consumers, not minting a new one",
    "refs": [
     "mirror:post-lifecycle-walk-findings-fresh-pass-2026-08-18"
    ]
   }
  ],
  "rawCandidates": 5,
  "promptBytes": 42401
 },
 "architecture.5d": {
  "cell": "architecture.5d",
  "wave": 1,
  "validator": "pass",
  "candidates": 2,
  "positives": 5,
  "needsHuman": 2,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [
   {
    "title": "Enumerated forward and enumerated simulation are a twice-declared sweep skeleton: three parallel scratch/visit/params types plus three parallel loops, two of them extracted on the simulation side and inlined on the training side",
    "into": "architecture.5c 'CD-028 sharpened: the shared claim-and-scatter owner already exists and its module doc declares a boundary narrower than the duplication, so what remains twice-declared between the enumerated forward and the enumerated simulation is the path marking and the worker-result fold that sit outside any stated boundary -- the fix is extending an owner with two consumers, not minting a new one'"
   }
  ],
  "dupOf": [],
  "retiredOverlaps": [
   {
    "title": "Enumerated forward and enumerated simulation are a twice-declared sweep skeleton: three parallel scratch/visit/params types plus three parallel loops, two of them extracted on the simulation side and inlined on the training side",
    "refs": [
     "PD-001",
     "mirror:performance-debt-follow-ups-first-performance-pass-2026-08-18"
    ]
   }
  ],
  "rawCandidates": 3,
  "promptBytes": 42401
 },
 "performance.5a": {
  "cell": "performance.5a",
  "wave": 2,
  "validator": "pass",
  "candidates": 3,
  "positives": 14,
  "needsHuman": 2,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [
   {
    "title": "The reverse-topological cut level decomposition is a study invariant but is rescanned and reallocated on every backward pass",
    "refs": [
     "mirror:module-map",
     "mirror:graph-shape-predicate-grep"
    ]
   },
   {
    "title": "Stage frontier resolution scans the entire node array per stage, so an enumerated sweep pays a whole-graph scan for every stage it visits",
    "refs": [
     "mirror:module-map",
     "mirror:graph-shape-predicate-grep"
    ]
   }
  ],
  "rawCandidates": 3,
  "promptBytes": 42401
 },
 "performance.5b": {
  "cell": "performance.5b",
  "wave": 2,
  "validator": "pass",
  "candidates": 3,
  "positives": 7,
  "needsHuman": 1,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 3,
  "promptBytes": 42401
 },
 "performance.5c": {
  "cell": "performance.5c",
  "wave": 2,
  "validator": "pass",
  "candidates": 7,
  "positives": 8,
  "needsHuman": 2,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [
   {
    "title": "process_stage_backward allocates a fresh owned staged-cut vector per worker per node at its return boundary, defeating the reused workspace buffer it drains",
    "dupOf": "CD-016",
    "sharedSymbols": [
     "process_stage_backward",
     "process_stage_backward_by_node"
    ],
    "titleJaccard": 0.08
   }
  ],
  "retiredOverlaps": [
   {
    "title": "The cut-selection value sweep heap-allocates six times per call and twice per rayon fold task, contradicting run_cut_management's stated no-allocation contract",
    "refs": [
     "mirror:performance-debt-follow-ups-first-performance-pass-2026-08-18"
    ]
   },
   {
    "title": "The cut-management projection gather pushes one scalar at a time through the slot-index table with no bulk-copy path for the identity projection",
    "refs": [
     "mirror:performance-debt-follow-ups-first-performance-pass-2026-08-18"
    ]
   },
   {
    "title": "build_slot_lookup clears the whole pool-length slot table on every warm-started solve although only the reconcilable slots are ever written",
    "refs": [
     "mirror:organizational-quality-follow-ups-low-priority"
    ]
   },
   {
    "title": "enforce_basic_count_invariant recounts BASIC statuses with two full filter passes over the status vectors the reconstruction just wrote",
    "refs": [
     "mirror:organizational-quality-follow-ups-low-priority"
    ]
   }
  ],
  "rawCandidates": 7,
  "promptBytes": 42401,
  "anchorsRejected": [
   {
    "title": "build_slot_lookup clears the whole pool-length slot table on every warm-started solve although only the reconcilable slots are ever written",
    "rejected": [
     "crates/cobre-sddp/src/workspace/workspace.rs::recon_slot_lookup (field anchor: check-anchors resolves declarations only (fn|struct|enum|trait|type|const|static|mod|impl); a field is cited, not anchored)"
    ]
   },
   {
    "title": "process_stage_backward allocates a fresh owned staged-cut vector per worker per node at its return boundary, defeating the reused workspace buffer it drains",
    "rejected": [
     "crates/cobre-sddp/src/workspace/workspace.rs::staged_cuts_buf (field anchor: check-anchors resolves declarations only (fn|struct|enum|trait|type|const|static|mod|impl); a field is cited, not anchored)"
    ]
   }
  ]
 },
 "performance.5d": {
  "cell": "performance.5d",
  "wave": 2,
  "validator": "pass",
  "candidates": 8,
  "positives": 18,
  "needsHuman": 2,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 8,
  "promptBytes": 42401
 },
 "over-engineering.5a": {
  "cell": "over-engineering.5a",
  "wave": 3,
  "validator": "pass",
  "candidates": 4,
  "positives": 14,
  "needsHuman": 1,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [
   {
    "title": "Four production #[allow(clippy::...)] sites in the 5a manifest carry no written rationale, so the census's sanctioned-by-construction claim is false at those four",
    "refs": [
     "mirror:module-map",
     "mirror:graph-shape-predicate-grep"
    ]
   }
  ],
  "rawCandidates": 4,
  "promptBytes": 42401
 },
 "over-engineering.5b": {
  "cell": "over-engineering.5b",
  "wave": 3,
  "validator": "pass",
  "candidates": 2,
  "positives": 9,
  "needsHuman": 1,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 2,
  "promptBytes": 42401
 },
 "over-engineering.5c": {
  "cell": "over-engineering.5c",
  "wave": 3,
  "validator": "pass",
  "candidates": 2,
  "positives": 13,
  "needsHuman": 1,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 2,
  "promptBytes": 42401
 },
 "over-engineering.5d": {
  "cell": "over-engineering.5d",
  "wave": 3,
  "validator": "pass",
  "candidates": 3,
  "positives": 16,
  "needsHuman": 1,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 3,
  "promptBytes": 42401
 },
 "test-bloat.5a": {
  "cell": "test-bloat.5a",
  "wave": 4,
  "validator": "pass",
  "candidates": 7,
  "positives": 8,
  "needsHuman": 0,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 7,
  "promptBytes": 42401,
  "anchorsRejected": [
   {
    "title": "The transit-seed fixture family is declared three times byte-identically inside one directory: hydro(), zero_penalties() and date() each exist in setup/bucket_topology.rs, setup/mod.rs and setup/tests.rs",
    "rejected": [
     "crates/cobre-sddp/src/test_support.rs::ymd (anchor outside the 5a manifest: crates/cobre-sddp/src/test_support.rs (5d))"
    ]
   },
   {
    "title": "state_layout_for is cloned byte-identically into seven integration binaries under a doc comment whose stated rationale is false at the pin, while the shared test_support symbol it duplicates is already imported by those same files",
    "rejected": [
     "crates/cobre-sddp/src/test_support.rs::state_layout (anchor outside the 5a manifest: crates/cobre-sddp/src/test_support.rs (5d))",
     "crates/cobre-sddp/src/test_support.rs::state_layout_with_transit_buckets (anchor outside the 5a manifest: crates/cobre-sddp/src/test_support.rs (5d))",
     "crates/cobre-sddp/src/test_support.rs::study_dims (anchor outside the 5a manifest: crates/cobre-sddp/src/test_support.rs (5d))"
    ]
   }
  ]
 },
 "test-bloat.5b": {
  "cell": "test-bloat.5b",
  "wave": 4,
  "validator": "pass",
  "candidates": 8,
  "positives": 5,
  "needsHuman": 0,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 8,
  "promptBytes": 42401
 },
 "test-bloat.5c": {
  "cell": "test-bloat.5c",
  "wave": 4,
  "validator": "pass",
  "candidates": 6,
  "positives": 6,
  "needsHuman": 2,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 6,
  "promptBytes": 42401,
  "anchorsRejected": [
   {
    "title": "cut_basis.rs re-declares five fixture helpers two and three times inside a single binary that already wires mod common",
    "rejected": [
     "crates/cobre-sddp/src/test_support.rs::write_synthetic_boundary (anchor outside the 5c manifest: crates/cobre-sddp/src/test_support.rs (5d))"
    ]
   },
   {
    "title": "No Communicator test double lives in the shared test_support surface, so src-side unit tests re-declare StubComm four times and Rank0Of2 twice while tests/common already exports both",
    "rejected": [
     "crates/cobre-sddp/src/test_support.rs::fill_consistent_basis (anchor outside the 5c manifest: crates/cobre-sddp/src/test_support.rs (5d))"
    ]
   },
   {
    "title": "HydroPenalties has no Default, so fixtures spell all sixteen fields; the crate answers with per-file private constructors including one inside test_support itself",
    "rejected": [
     "crates/cobre-sddp/src/test_support.rs::geometry_zero_penalties (anchor outside the 5c manifest: crates/cobre-sddp/src/test_support.rs (5d))"
    ]
   }
  ]
 },
 "test-bloat.5d": {
  "cell": "test-bloat.5d",
  "wave": 4,
  "validator": "pass",
  "candidates": 6,
  "positives": 5,
  "needsHuman": 0,
  "droppedPriorScreen": [],
  "droppedAnchor": [],
  "mergedDupOf": [],
  "dupOf": [],
  "retiredOverlaps": [],
  "rawCandidates": 6,
  "promptBytes": 42401
 }
}
```
