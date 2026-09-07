# Ingest log — station cobre-core + cobre-io

Baseline: `a136840d4f2ea137f685f0af6dac04254b983b60`  ·  ingested 2026-09

Sources screened: candidates-A.json, candidates-B.json, candidates-C.json, candidates-D.json.
candidateRef re-keyed to `<subStation>-<lens>-<nn>` (nn per sub+lens, array order); the E02-3
worker ref is preserved in verdicts.json as `oldRef`. Every candidate is accounted for exactly
once below and in verdicts.json.

## Received (per source, sub-station and lens)

| source | subStation | lens | candidates |
|---|---|---|---|
| candidates-A.json | A | architecture | 7 |
| candidates-A.json | A | perf | 3 |
| candidates-A.json | A | over-engineering | 4 |
| candidates-A.json | A | test-bloat | 5 |
| candidates-B.json | B | architecture | 4 |
| candidates-B.json | B | perf | 4 |
| candidates-B.json | B | over-engineering | 3 |
| candidates-B.json | B | test-bloat | 6 |
| candidates-C.json | C | architecture | 7 |
| candidates-C.json | C | perf | 5 |
| candidates-C.json | C | over-engineering | 4 |
| candidates-C.json | C | test-bloat | 6 |
| candidates-D.json | D | architecture | 6 |
| candidates-D.json | D | perf | 5 |
| candidates-D.json | D | over-engineering | 6 |
| candidates-D.json | D | test-bloat | 6 |
| **total** |  |  | **81** |

No sub-station or lens is empty; there are no no-finding lines to record.

## Anchor rejections

None. All 319 anchors across the 81 candidates resolve at the baseline through
`check-anchors.py` over `anchor-probe.md` (checked 319, 0 failing). One anchor —
`crates/cobre-io/src/pipeline.rs::populate_derived_residual_ratios` (B-over-engineering-02) —
is a call site, not a declaration, so it is anchored by its line (`pipeline.rs:225`), which
resolves; the candidate is retained.

| candidateRef | failing anchor | checker diagnostic |
|---|---|---|
| _(none)_ |  |  |

## Cleared (sanctioned)

None. The reserved-seams check ran over all 17 over-engineering candidates against the three
sanctioned sources — the reserved-seam register and 'Verified NOT reserved' section of
`docs/design/reserved-seams-and-deferred-debt.md`, the 'Unwired config is reserved, not dead'
rule in `CLAUDE.md`, and the `#[allow(...)]` census (Load-bearing / Reserved-seam /
Symmetry-or-test-retention) at mirror lines 1067-1096. No over-engineering candidate targets an
item named in any of the three: the E02-3 attackers already filed the ratified seams
(`LipschitzConfig.mode`, the hydro storage/filling penalties, `historical_years`) in each
envelope's `positives` with a `sanctionedBy` citation, so none re-entered the candidate set.
The searched sources are recorded here so no defender repeats the search.

| candidateRef | anchor | sanctionedBy |
|---|---|---|
| _(none)_ |  |  |

## Re-raise and dup-of rejections

None. No candidate restates a do-not-touch or Cleared prior item (the retracted CD-008, the
refuted PD-001, or a fix-wave CLEAN verdict), and none re-raises a RESOLVED station item
(CD-010, CD-026, CD-031) or duplicates a registered-and-open entry (`graph_type` stages.rs:134,
`BoundaryPolicy.source_stage` policy.rs:53, stage engine-scoping stages.rs:212). Five candidates
share a *file* with a resolved item but each makes a distinct claim, checked and retained:

| candidateRef | prior ID (file-shared) | why it is NOT a re-raise |
|---|---|---|
| D-architecture-00 | CD-010 | output/policy/checkpoint.rs shared; claim is atomic-write crash-safety, not the untyped state-family primitive |
| D-architecture-04 | CD-026 | simulation_writer.rs::write_partition shared; claim is duplicate entity-family enumeration, not the resolved 12x write repetition |
| D-perf-00 | CD-010 | checkpoint.rs shared; claim is a redundant Vec copy of the FlatBuffers buffer, not state-family typing |
| D-perf-02 | CD-026 | simulation_writer.rs::write_partition shared; claim is a per-partition String retained for the whole run, not write repetition |
| D-test-bloat-03 | CD-010 | records.rs/checkpoint.rs shared; claim is test-module placement, not state-family typing |

## Part-I cross-references

Worker-tagged overlaps with an owned Part-I disposition from `partI-handoff.json`. Each still
goes to a defender; the verdict travels to Epic 9 with its `partIRef`.

| candidateRef | partIRef | title |
|---|---|---|
| A-architecture-04 | I.3-2 | StageLagTransition is a PAR-lag ring-buffer control block living in L0 with zero |
| A-architecture-05 | I.3-1 | SystemBuilder enforces canonical order for nine collections and delegates it by  |
| A-over-engineering-03 | I.3-1 | NetworkTopology and its three companion records are a fully unconsumed derived s |
| B-architecture-03 | I.3-2 | Every semantic stage rule the parser already enforces is unreachable: `stages.rs |
| B-over-engineering-02 | I.3-1 | `load_scenarios` and its 9-field `ScenarioData` result are a zero-consumer publi |
| C-architecture-00 | I.3-7 | The config admission rules are enforced only as a side effect of an accessor, an |
| D-architecture-03 | I.3-6 | The training-loop vocabulary is baked into the L2 output schema surface as on-di |
| D-over-engineering-00 | I.3-7 | `write_dictionaries` takes the whole SDDP-shaped study `Config` and never reads  |
| D-over-engineering-04 | I.3-6 | `read_f32_vector_as_f64` is a dead codec reader kept by a completeness rationale |
| D-perf-00 | I.3-6 | Every policy artifact is copied out of the FlatBuffers builder into a fresh `Vec |
| D-test-bloat-03 | I.3-6 | `output/policy/mod.rs` is a 29-line re-export shim carrying a 1315-line inline t |

## Out-of-station hand-offs

None. No candidate anchors into another crate; the two out-of-station Part-I item-7 anchors
(`crates/cobre-sddp/src/setup/params.rs::from_config`, 
`crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig`) are handled by `partI-handoff.json`,
not raised as candidates here.

## Defender pass

81 candidates survived to the defender pass, dispatched as 16 read-only Opus adversarial-defender
agents (one per sub-station x lens; owner-approved batching of the 'one defender per candidate'
rule), each returning an individual verdict per candidate. Every returned envelope was shape-valid
under `validate-envelope.py --role defender` on the first pass, so no defender was re-dispatched; no
candidate carries a needs-human *in place of* a verdict.

Outcome (see `verdicts.json`, keyed by candidateRef):

| disposition | count |
|---|---|
| defended | 81 |
|   confirmed | 77 |
|   dismissed | 4 |
| anchor-rejected / sanctioned-cleared / re-raise / dup-of / handed-off | 0 |
| **total received** | **81** |

- Every `confirmed` verdict carries a `survivingClaim` strictly narrower than the candidate title;
  every `dismissed` verdict carries an argument (>=120 chars). The four dismissals are decided on the
  merits, not on a reserved seam, so none carries a `sanctionedBy`:
  B-perf-00 (tokenize reached only on the load-time constraint-conversion path, not the training hot
  path), B-perf-01 (per-call index build on a load-time resolver), B-perf-03 (parallel WindowedRecord
  vec consumed only by load-time validation), C-over-engineering-00 (the three hand-written
  `Deserialize` impls are not derive-equivalent — they back `JsonSchema`/rename behaviour a derive
  would not reproduce).
- alignmentHint across the 81: 73 neutral, 4 advances-0a, 4 advances-1, 0 conflicts (no verdict's
  implied fix violates the L0-L4 layering).
- 11 verdicts carry a `partIRef` (the worker-tagged Part-I cross-references), routed to Epic 9 with
  their verdict.
- One needs-human note travels to the owner gate: A-architecture-06 is confirmed, but the owner must
  decide whether byte-level wire-payload reproducibility is a contract (fixing the six unguarded
  HashMap fields) or a non-goal (the defect narrows to an over-stated rationale) — the severity
  depends on that call.

No tracked file under crates/, scripts/, .github/, docs/ or schemas/ was modified; all artifacts are
under gitignored `plans/architecture-debt-audit/`.


## Per-candidate roster

Every one of the 81 candidateRefs, exactly once, so no candidate is silently dropped (mirrors `verdicts.json`).

| candidateRef | disposition | verdict | partIRef |
|---|---|---|---|
| A-architecture-00 | defended | confirmed |  |
| A-architecture-01 | defended | confirmed |  |
| A-architecture-02 | defended | confirmed |  |
| A-architecture-03 | defended | confirmed |  |
| A-architecture-04 | defended | confirmed | I.3-2 |
| A-architecture-05 | defended | confirmed | I.3-1 |
| A-architecture-06 | defended | confirmed |  |
| A-over-engineering-00 | defended | confirmed |  |
| A-over-engineering-01 | defended | confirmed |  |
| A-over-engineering-02 | defended | confirmed |  |
| A-over-engineering-03 | defended | confirmed | I.3-1 |
| A-perf-00 | defended | confirmed |  |
| A-perf-01 | defended | confirmed |  |
| A-perf-02 | defended | confirmed |  |
| A-test-bloat-00 | defended | confirmed |  |
| A-test-bloat-01 | defended | confirmed |  |
| A-test-bloat-02 | defended | confirmed |  |
| A-test-bloat-03 | defended | confirmed |  |
| A-test-bloat-04 | defended | confirmed |  |
| B-architecture-00 | defended | confirmed |  |
| B-architecture-01 | defended | confirmed |  |
| B-architecture-02 | defended | confirmed |  |
| B-architecture-03 | defended | confirmed | I.3-2 |
| B-over-engineering-00 | defended | confirmed |  |
| B-over-engineering-01 | defended | confirmed |  |
| B-over-engineering-02 | defended | confirmed | I.3-1 |
| B-perf-00 | defended | dismissed |  |
| B-perf-01 | defended | dismissed |  |
| B-perf-02 | defended | confirmed |  |
| B-perf-03 | defended | dismissed |  |
| B-test-bloat-00 | defended | confirmed |  |
| B-test-bloat-01 | defended | confirmed |  |
| B-test-bloat-02 | defended | confirmed |  |
| B-test-bloat-03 | defended | confirmed |  |
| B-test-bloat-04 | defended | confirmed |  |
| B-test-bloat-05 | defended | confirmed |  |
| C-architecture-00 | defended | confirmed | I.3-7 |
| C-architecture-01 | defended | confirmed |  |
| C-architecture-02 | defended | confirmed |  |
| C-architecture-03 | defended | confirmed |  |
| C-architecture-04 | defended | confirmed |  |
| C-architecture-05 | defended | confirmed |  |
| C-architecture-06 | defended | confirmed |  |
| C-over-engineering-00 | defended | dismissed |  |
| C-over-engineering-01 | defended | confirmed |  |
| C-over-engineering-02 | defended | confirmed |  |
| C-over-engineering-03 | defended | confirmed |  |
| C-perf-00 | defended | confirmed |  |
| C-perf-01 | defended | confirmed |  |
| C-perf-02 | defended | confirmed |  |
| C-perf-03 | defended | confirmed |  |
| C-perf-04 | defended | confirmed |  |
| C-test-bloat-00 | defended | confirmed |  |
| C-test-bloat-01 | defended | confirmed |  |
| C-test-bloat-02 | defended | confirmed |  |
| C-test-bloat-03 | defended | confirmed |  |
| C-test-bloat-04 | defended | confirmed |  |
| C-test-bloat-05 | defended | confirmed |  |
| D-architecture-00 | defended | confirmed |  |
| D-architecture-01 | defended | confirmed |  |
| D-architecture-02 | defended | confirmed |  |
| D-architecture-03 | defended | confirmed | I.3-6 |
| D-architecture-04 | defended | confirmed |  |
| D-architecture-05 | defended | confirmed |  |
| D-over-engineering-00 | defended | confirmed | I.3-7 |
| D-over-engineering-01 | defended | confirmed |  |
| D-over-engineering-02 | defended | confirmed |  |
| D-over-engineering-03 | defended | confirmed |  |
| D-over-engineering-04 | defended | confirmed | I.3-6 |
| D-over-engineering-05 | defended | confirmed |  |
| D-perf-00 | defended | confirmed | I.3-6 |
| D-perf-01 | defended | confirmed |  |
| D-perf-02 | defended | confirmed |  |
| D-perf-03 | defended | confirmed |  |
| D-perf-04 | defended | confirmed |  |
| D-test-bloat-00 | defended | confirmed |  |
| D-test-bloat-01 | defended | confirmed |  |
| D-test-bloat-02 | defended | confirmed |  |
| D-test-bloat-03 | defended | confirmed | I.3-6 |
| D-test-bloat-04 | defended | confirmed |  |
| D-test-bloat-05 | defended | confirmed |  |
