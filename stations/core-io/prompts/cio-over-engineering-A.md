# core-io attacker worker prompt (template)

Baseline: `a136840d4f2ea137f685f0af6dac04254b983b60`   Sub-station: `A`   Lens: `over-engineering`
Sweep ONLY these paths (repo-relative; a nested directory is listed with its parent and is in scope): `crates/cobre-core/src/constraints`, `crates/cobre-core/src/entities`, `crates/cobre-core/src/model`, `crates/cobre-core/src/model/resolved`, `crates/cobre-core/src/model/temporal`, `crates/cobre-core/src/stats`, `crates/cobre-core/src/system`, `crates/cobre-core/src/topology`, `crates/cobre-core/src/commissioning.rs`, `crates/cobre-core/src/entity_id.rs`, `crates/cobre-core/src/error.rs`, `crates/cobre-core/src/lib.rs`
Inventory: `plans/architecture-debt-audit/stations/core-io/inventory.json` (line budgets, file lists, sub-station partition)
Prior register / do-not-re-raise list: `plans/architecture-debt-audit/stations/core-io/prior-register.md`
Part-I dispositions: `plans/architecture-debt-audit/stations/core-io/partI-handoff.json`
Target layering and Alignment vocabulary: `plans/architecture-debt-audit/tools/target-layering-brief.md`
Reserved seams and cleared items (the mirror): `docs/design/reserved-seams-and-deferred-debt.md`
Testing yardstick: `docs/design/testing-architecture.md`

You are one of sixteen read-only attacker workers (four lenses x four sub-stations). Your lens for this
run is `over-engineering`; the other three lenses are covered by sibling workers, so stay inside your lens and
inside your paths. The session that dispatched you is the sole writer of every artifact.

## RULES (each is a guardrail; a violated rule voids the envelope)

1. **Read-only.** Never write, edit, format, or run anything that mutates the tree, the target
   directory, or the git state. `git show`, `grep`, `wc`, `find`, `cargo check`/`clippy` with a
   separate `CARGO_TARGET_DIR` under `/tmp` are fine; `cargo fmt`, `git checkout`, `sed -i` are not.
2. **One JSON object and nothing else.** Your entire reply is the envelope below: no preamble, no
   fenced code block, no trailing prose, no markdown. The first character of the reply is `{` and the
   last is `}`.
3. **Anchors resolve at the baseline with real evidence.** Every anchor `path` is repo-relative and
   must resolve under `git show a136840d4f2ea137f685f0af6dac04254b983b60:<path>`; every anchor carries a `symbol` (a declared
   fn/struct/enum/trait/type/const/static/mod name in that file) or a `line`. Every candidate carries
   `evidence.command` (a command you actually ran, verbatim), `evidence.output` (its output, trimmed),
   and `evidence.reading` (why the output supports the claim).
4. **Fix shapes are prose.** `fixShape` describes the shape of a fix in sentences; never a diff, a
   patch, a code block, or an edit. This evaluation ships no fixes.
5. **Never re-raise a retired item.** Do not raise CD-008 (retracted), PD-001 (refuted), PD-004
   (deferred pending a profile), anything on the BACKLOG do-not-touch list, anything in the mirror's
   "Cleared" section, the resolved Part-VI forks in `plans/generalizing/refinement-todo.md`, or any
   bullet of the `## Do not re-raise` list in `prior-register.md`. If you have NEW evidence that a
   retired item is live again, raise it with `"reRaiseOf": "<ID>"` and say what changed; otherwise
   leave it out.
6. **Reserved seams first.** Before ANY over-engineering, dead-config or unwired-seam candidate, check
   the mirror's reserved-seam register and prior-register.md's sanctioned list. A ratified seam (for
   example `LipschitzConfig` at `crates/cobre-io/src/config/training.rs:531`, the hydro
   storage/filling penalties, `historical_years`) belongs in `positives` with a `sanctionedBy`
   citation to the mirror section, never in `candidates`.
7. **Perf candidates stay UNMEASURED.** State the suspected cost mechanism (allocation per row,
   repeated pass, redundant copy, lock, syscall) and the deck-independent reason it matters; do not
   time anything and do not quote timings. Measurement is a separate epic on a fixed deck.
8. **Lint or dead-code evidence only from a fresh full-feature run.** If you cite clippy or dead-code
   output, run it yourself in this session with the full feature set
   (`--features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"`) and a
   scratch `CARGO_TARGET_DIR`; never a cached, featureless or remembered result. If you cannot run it,
   do not cite it.

Additional expectations: cite `partIRef` (`I.3-1|I.3-2|I.3-3|I.3-4|I.3-6|I.3-7`) on any candidate that
touches a Part-I item and read the disposition already recorded for it in `partI-handoff.json` first;
put things that are correct and worth protecting in `positives` (a station report is not a
defect-only list); put anything you cannot decide (an ambiguous ownership, a claim that needs the
owner) in `_needsHuman` as one sentence each. Prefer fewer, well-evidenced candidates over many
weak ones; a candidate without a mechanism and an anchor is noise.

## Alignment vocabulary (closed set; `target-layering-brief.md` sections 1-3)

`advances-0a` (engine seam, study config block, shared output orchestration in cobre-io, rank-0 MPI),
`advances-0b` (carving `cobre-model` from the engine-neutral lp/), `advances-1` (purify the data
model: stochastic off System/Stage, training_event out of cobre-core, StageTemplate shed, case v2),
`neutral` (advances no phase), `conflicts` (the fix shape would place an engine concept in L0,
couple a crate to an engine, create a one-consumer abstraction, or contradict the phase order; tag
it and propose the roadmap-consistent alternative in `fixShape`).

## Lens: architecture - named probes

**A (cobre-core).** Paradigm leakage as structure, not vocabulary: `System` stochastic fields
(`crates/cobre-core/src/system/mod.rs`, eight fields at :111-131), `Stage`
`state_config`/`risk_config`/`scenario_config` (`crates/cobre-core/src/model/temporal.rs:309-315`),
`InitialConditions` warm-start shape (`crates/cobre-core/src/constraints/initial_conditions.rs:178`),
`crates/cobre-core/src/constraints/training_event.rs` (`TrainingEvent`, an L0 crate carrying a
training-loop event vocabulary) and `HorizonGraph` (`crates/cobre-core/src/model/horizon.rs:22`,
the discount/cyclic framing). Cross-check each against `partI-handoff.json`, set `partIRef`, and
only add what the disposition there does not already say. Also probe ownership: `system/builder.rs`
canonical `(operational_start_date, id)` ordering (`sort_canonical` at :331-333), `topology/`,
`model/resolved/` — is each invariant owned once, and does `commissioning.rs` / `entity_id.rs`
duplicate anything?

**B (cobre-io input path).** Parser id-order versus builder `(date, id)` canonical order — a known
bug class. The parsers sort on the raw id: `crates/cobre-io/src/system/buses.rs:264`
(`buses.sort_by_key(|b| b.id.0)`), `lines.rs:255`, `non_controllable.rs:233`,
`pumping_stations.rs:246`, `energy_contracts.rs:266`, `hydros.rs:852`, `thermals.rs:319`, while
`crates/cobre-core/src/system/builder.rs:331-333` re-sorts to `(operational_start_date, id)`. For each
parser/consumer pair, prove which order the consumer assumes and whether anything between the
parser sort and the builder sort reads the parser order (index alignment, positional joins,
`resolution/`, `constraints/`, `scenarios/` tables keyed by position). An unproven pair is a
candidate that needs evidence, not a finding. Also probe: `scenarios/` (11,672 lines) and
`constraints/` (10,933) for parsers that re-implement the same schema-read / column-extract shape;
`stages.rs` (2,795) and `initial_conditions.rs` (1,610) for validation logic that belongs in
`validation/`; `resolution/` versus `constraints/` for duplicated bound-resolution rules.

**C (cobre-io validation + config + schema).** Rule sprawl: `crates/cobre-io/src/validation/` is
28,672 raw lines, of which `validation/semantic/` is 20,035 across 14 modules (`thermal.rs` 4,087,
`scenarios.rs` 3,334, `hydro.rs` 2,764, `block_bounds.rs` 2,097, `stages.rs` 1,863; most of it
inline tests, see the inventory's non-test counts). Probe for the same rule implemented in two
modules, for rules that are really an entity's own invariant and belong beside the entity in
cobre-core (`entities/`, `system/validate.rs`), and for `structural.rs` / `referential.rs` /
`dimensional.rs` phases that overlap. Config: `config/` carries the SDDP-shaped `TrainingConfig`
(`partI-handoff.json` I.3-7 already dispositions it — add only new structure), the `schema` feature
(`schema.rs::generate_schemas`) and `exports.rs`.

**D (cobre-io output path).** Writer duplication in `crates/cobre-io/src/output/` (22,530 raw lines;
32 `build_*_batch` functions across the writers, 19 in `simulation_writer.rs` alone) measured
against the dedup precedent already set in `crates/cobre-io/src/parquet_helpers.rs:144`
(`extract_required_date32` and its five siblings at :13-117): where the same column-extract /
column-build / schema-assembly shape is open-coded per writer instead of going through a helper;
`output/schemas.rs` (1,484) and `output/dictionary.rs` (3,250) as parallel vocabularies; the
`output/policy/` codec (`records.rs`, `codec.rs`, `checkpoint.rs`) and the `CheckpointManifest`
provenance path. CD-026 (`write_partition`) is resolved — do not re-raise it; the flat-file size of
`simulation_writer.rs` alone is noted-not-raised (prior-register.md).

## Lens: perf - named probes

Per-row allocation in the parsers (sub-station B: `String` per cell, `Vec` per row, `to_string()`
in hot loops, `collect()` into intermediate vectors that are immediately re-iterated) and in the
write paths (sub-station D: the Parquet writers under `crates/cobre-io/src/output/`, batch builders
that allocate per scenario, the FlatBuffers policy codec
`crates/cobre-io/src/output/policy/{codec,records,checkpoint}.rs`). Validation passes that walk
large scenario tables more than once (sub-station C: `validation/semantic/scenarios.rs`,
`correlation.rs`, `inflow_seeding.rs`). In cobre-core (sub-station A): `system/builder.rs` sort and
validation passes, `topology/` traversal, `model/resolved/` factor tables. State the mechanism and
the deck-independent reason; note whether the path is setup-time (once per run) or per-scenario /
per-stage, because setup-time recomputation is acceptable by project rule and only hot-path cost
justifies cached state. Do not time anything.

## Lens: over-engineering - named probes

Schema-export machinery (`crates/cobre-io/src/schema.rs`, the `schema` feature, `config/exports.rs`),
wrapper enums with a single inhabitant, newtypes that only forward, `pub` surfaces with zero external
consumers (grep the workspace: `grep -rn '<symbol>' crates/ --include='*.rs' | grep -v '<file>'`),
unused generic parameters (the unused-`BuildHasher`-genericity precedent OD-007), speculative
`case_dir`-taking wrappers (the OD-002 precedent), and `#[allow(dead_code)]` "for symmetry" (the
OD-009 sanctioned class — refinement only). Reserved-seam check first, always (rule 6); `Option`
fields loaded-but-unconsumed are reserved config unless the mirror says otherwise.

## Lens: test-bloat - named probes

14 integration binaries across the two crates
(`find crates/cobre-core/tests crates/cobre-io/tests -maxdepth 1 -name '*.rs'`; the inventory names
them), inline `#[cfg(test)]` modules versus sibling `tests.rs` / `test_support.rs`
(`scenarios/estimation/tests.rs`, `validation/semantic/test_support.rs`), the raw-versus-non-test
ratio per module in the inventory (cobre-io `validation/semantic/` and `scenarios/` are majority test
lines), duplicated fixtures and builders across integration binaries and inline modules, tautological
tests (asserting a constructor returns what it was given), tests pinning a wire byte that a
`const` already pins, and the config-spelling reject tests — which are load-bearing, not bloat
(prior-register.md). Yardstick is `docs/design/testing-architecture.md` (its tier taxonomy and the
per-file-binary consolidation in section 5.1).

## Envelope (frozen contract — the merge and the ingest ticket read exactly this)

```json
{
  "station": "core-io",
  "subStation": "A",
  "baseline": "a136840d4f2ea137f685f0af6dac04254b983b60",
  "lens": "over-engineering",
  "candidates": [
    {
      "title": "one line naming the smell and its subject",
      "anchors": [
        { "path": "crates/cobre-io/src/system/buses.rs", "symbol": "parse_buses", "line": 264 }
      ],
      "evidence": {
        "command": "grep -n 'sort_by_key' crates/cobre-io/src/system/buses.rs",
        "output": "264:    buses.sort_by_key(|b| b.id.0);",
        "reading": "why that output supports the claim"
      },
      "mechanism": "perf lens only: the cost mechanism, no timings",
      "proposedSeverity": "A|B|C",
      "fixShape": "prose only: the shape of the fix, never a diff or a patch",
      "alignmentHint": "advances-0a|advances-0b|advances-1|neutral|conflicts",
      "partIRef": "I.3-1|I.3-2|I.3-3|I.3-4|I.3-6|I.3-7 (only when it applies)",
      "reRaiseOf": "CD-nnn (only with new evidence, rule 5)"
    }
  ],
  "positives": [
    { "subject": "crates/cobre-io/src/config/training.rs:531", "why": "ratified reserved seam", "sanctionedBy": "docs/design/reserved-seams-and-deferred-debt.md — `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`" }
  ],
  "_needsHuman": []
}
```

Severity: A = wrong results, lost determinism, or a structural block on the roadmap; B = real debt with
a bounded fix and a named blast radius; C = local quality. The envelope block (the JSON above with
your values) is your whole reply.

## Sub-station scope block (partition proven by the dispatching session)

- **A** cobre-core (L0): `crates/cobre-core/src/{constraints,entities,model,model/resolved,model/temporal,stats,system,topology}` + `commissioning.rs`, `entity_id.rs`, `error.rs`, `lib.rs`
- **B** cobre-io input path (a logical grouping — `crates/cobre-io/src/input` does not exist): `crates/cobre-io/src/{system,scenarios,scenarios/estimation,extensions,constraints,resolution}` + `stages.rs`, `stage_resolve.rs`, `initial_conditions.rs`, `windowed_history.rs`, `post_study_stages.rs`, `penalties.rs`, `broadcast.rs`
- **C** cobre-io validation + config + schema: `crates/cobre-io/src/{config,validation,validation/semantic}` + `schema.rs`, `error.rs`, `lib.rs`
- **D** cobre-io output path: `crates/cobre-io/src/{output,output/policy}` + `parquet_helpers.rs`, `pipeline.rs`, `report.rs`

Union of A-D == every `.rs` under `crates/cobre-core/src` and `crates/cobre-io/src`, pairwise disjoint (checked before dispatch; see `attacker-log.md`).
