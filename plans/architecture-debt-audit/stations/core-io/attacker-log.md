# core-io attacker dispatch log

Baseline: `a136840d4f2ea137f685f0af6dac04254b983b60` (register pin; HEAD `7707fb53` has byte-identical evaluated surfaces)   Dispatched: 2026-09-06   Agent type: `adversarial-attacker` (Opus, read-only: no Write/Edit/NotebookEdit tools; every worker returns one JSON object and this session is the sole writer)

## Partition check (before dispatch)

```
$ find $A $B_DIRS $B_FILES $C $D -name '*.rs' | sort -u > covered.txt
$ find crates/cobre-core/src crates/cobre-io/src -name '*.rs' | sort -u > all.txt
$ diff all.txt covered.txt && echo "PARTITION OK"
PARTITION OK
files=155 A=40 B=58 C=32 D=25 overlap=0
crates/cobre-io/src/input: absent (B is a logical grouping)
```

Prompts rendered: 16 (`prompts/cio-<lens>-<sub>.md`, zero unresolved `{{…}}` placeholders). Envelope validator exercised before dispatch: well-formed → exit 0 silent; out-of-set `alignmentHint` → exit 1 naming `$.candidates[0].alignmentHint`; stale-but-well-formed anchor → exit 0 (resolution is check-anchors.py's job at ingest); prose around the object → exit 1 `$ not parseable JSON`; perf candidate quoting a timing → exit 1; diff-shaped fixShape → exit 1; defender dismissal without argument → exit 1.

## Workers

| worker | lens | sub-station | envelope | candidates | note |
| --- | --- | --- | --- | --- | --- |
| cio-architecture-A | architecture | A cobre-core | valid | 7 | 2 carry partIRef |
| cio-perf-A | perf | A cobre-core | valid | 3 |  |
| cio-over-engineering-A | over-engineering | A cobre-core | valid | 4 | 1 carry partIRef |
| cio-test-bloat-A | test-bloat | A cobre-core | valid | 5 |  |
| cio-architecture-B | architecture | B io input path | attempt 1 INVALID (`$ not parseable JSON`: four `evidence` objects unclosed, one candidate without `alignmentHint`) -> re-dispatched with the diagnostic -> valid (structural repair only, content unchanged, disclosed by the worker) | 4 | 1 carry partIRef |
| cio-perf-B | perf | B io input path | valid | 4 |  |
| cio-over-engineering-B | over-engineering | B io input path | valid | 3 | 1 carry partIRef |
| cio-test-bloat-B | test-bloat | B io input path | valid | 6 |  |
| cio-architecture-C | architecture | C io validation+config+schema | valid | 7 | 1 carry partIRef |
| cio-perf-C | perf | C io validation+config+schema | valid | 5 |  |
| cio-over-engineering-C | over-engineering | C io validation+config+schema | valid | 4 |  |
| cio-test-bloat-C | test-bloat | C io validation+config+schema | valid | 6 |  |
| cio-architecture-D | architecture | D io output path | valid | 6 | 1 carry partIRef |
| cio-perf-D | perf | D io output path | valid | 5 | 1 carry partIRef |
| cio-over-engineering-D | over-engineering | D io output path | valid | 6 | 2 carry partIRef |
| cio-test-bloat-D | test-bloat | D io output path | valid | 6 | 1 carry partIRef |

## Hand-over deviation (recorded, not hidden)

The Agent result channel truncates replies at roughly 4 KB (and the drain at 16 KB), so no envelope reached this session intact through the reply. A first attempt to relay each envelope as numbered 3000-character message parts was abandoned: the parts are not machine-readable and transcribing them by hand would have risked corrupting the evidence. Each worker was instead told to write its exact final object to the session scratchpad (`/tmp/claude-1000/…/scratchpad/raw/<worker>.json`), a location outside the repository tree, and to reply `WRITTEN <bytes>`. The read-only guardrail over the tree held (no worker touched the tree or git state; `git status --porcelain --untracked-files=no` is empty), and this session remains the sole writer under `plans/`: it copied the sixteen files into `stations/core-io/raw/` (re-serialized with `json.dump(indent=2)`), validated each with `tools/validate-envelope.py --role attacker --station core-io`, and merged them.

## Merge (candidates-A..D.json)

Dedup key = lowercased title + sorted anchor set; sort key = (lens, first anchor path, title); positives and `_needsHuman` concatenated with a `lens` stamp, never deduped. No exact duplicates were dropped. Cross-lens near-duplicates left for the defender/ingest ticket to merge on evidence (different titles, overlapping subject): A `cio-architecture-A-03` / `cio-over-engineering-A-02` (the unconstructed `ValidationError` variants); C `cio-architecture-C-04` / `cio-over-engineering-C-02` (the positional input-file registry lists); B `cio-architecture-B-02` (parser prologue duplication) versus `cio-test-bloat-B-00` (test fixture duplication, a different surface).

| sub-station | candidates | positives | _needsHuman | severities |
| --- | --- | --- | --- | --- |
| A | 19 | 26 | 10 | 1×A, 13×B, 5×C |
| B | 17 | 26 | 8 | 5×B, 12×C |
| C | 22 | 21 | 11 | 13×B, 9×C |
| D | 23 | 21 | 11 | 11×B, 12×C |

partIRef coverage across the four files: I.3-1 ×3, I.3-2 ×2, I.3-6 ×4, I.3-7 ×2. The one Sev-A candidate is `cio-over-engineering-A-00` (the resolved per-(NCS, stage) penalty axis has no production reader, so a declared stage override is parsed, validated, resolved and dropped) — a defender verdict and a Sev calibration are owed before it enters the register.

Reserved-seams rule: `LipschitzConfig` / `UpperBoundEvaluationConfig` appears only in `positives` (all four C-lens envelopes and the D over-engineering envelope cite the mirror section); the single candidate mentioning it (`cio-over-engineering-D-00`, `ParquetWriterConfig`) does so only to state that its own subject is NOT in the reserved-seam register. The hydro penalties and `historical_years` were likewise dismissed with citations rather than raised.

## Gaps carried forward

None. All sixteen lenses returned a shape-valid envelope after at most one re-dispatch; no `excluded: envelope-invalid` entry exists in any `lenses` map. Anchor resolution against the baseline was deliberately not checked here and is owed by the ingest ticket's `check-anchors.py` pass.

## Tree assertion

`git status --porcelain --untracked-files=no` -> empty before the commit; every artifact of this ticket sits under `plans/architecture-debt-audit/` (tracked on this branch by owner decision).
