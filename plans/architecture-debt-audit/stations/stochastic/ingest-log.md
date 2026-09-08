# Ingest log — station cobre-stochastic

Baseline: `a136840d4f2ea137f685f0af6dac04254b983b60`  ·  ingested 2026-09

Sources screened: candidates-par.json, candidates-sampling.json, candidates-tree-noise.json, candidates-seam.json. candidateRef re-keyed to `<subStation>-<lens>-<nn>` (nn per sub+lens, array order); the attacker worker ref is preserved in verdicts.json as `candidateRef`. Every candidate is accounted for exactly once below and in verdicts.json.

## Received (per source, sub-station and lens)

| source | subStation | lens | candidates |
|---|---|---|---|
| candidates-par.json | par | architecture | 1 |
| candidates-par.json | par | over-engineering | 2 |
| candidates-par.json | par | performance | 3 |
| candidates-par.json | par | test-bloat | 4 |
| candidates-sampling.json | sampling | architecture | 2 |
| candidates-sampling.json | sampling | over-engineering | 2 |
| candidates-sampling.json | sampling | performance | 3 |
| candidates-sampling.json | sampling | test-bloat | 2 |
| candidates-tree-noise.json | tree-noise | architecture | 3 |
| candidates-tree-noise.json | tree-noise | over-engineering | 1 |
| candidates-tree-noise.json | tree-noise | performance | 3 |
| candidates-tree-noise.json | tree-noise | test-bloat | 2 |
| candidates-seam.json | seam | architecture | 2 |
| candidates-seam.json | seam | over-engineering | 1 |
| candidates-seam.json | seam | performance | 2 |
| candidates-seam.json | seam | test-bloat | 2 |
| **total** |  |  | **35** |

No sub-station or lens is empty; there are no no-finding lines to record.

## Anchor rejections

None. All 111 anchors across the 35 candidates resolve at the baseline (symbol-or-line), rendered in `anchor-probe.md`. Descriptive-symbol anchors (e.g. a module-doc reference) each additionally carry a resolving line, so every anchor resolves through `station_checks.anchor_exists`.

## Cleared (sanctioned)

None. The seven `prior-register.md` entries were screened before dispatch and cited by the attackers as positives / cross-references, never raised as candidates; no candidate targets a ratified reserved seam, so no ingest-time sanction clearance fired (`sanctionedCleared = 0`).

## Re-raise and dup-of rejections

None (`reRaiseRejected = 0`, `dupOf = 0`). Four candidates trip a coarse `reraiseKey` substring and were adjudicated **distinct** at the attacker screen (see `attacker-log.md` → Prior-register screen): `par-architecture-00` (paradigm vocabulary in a doc comment, not the take/fill glue), `seam-performance-00` and `seam-test-bloat-01` (a bounded quadratic and a tautological test in `season_cast`, not the stage-calendar *relocation*; both self-disclosed `reRaiseOf: null`), and `seam-over-engineering-00` (dead error taxonomy, not CD-001). The defender confirmed each on its own merits.

## Part-I cross-references

3 confirmed candidates carry a `partIRef`, all `I.3-1`, all `alignmentHint: advances-1` — the L1 store-vs-config seam findings that travel to the generalization alignment epic:

- `sampling-architecture-01` (sto-architecture-sampling-00) — I.3-1
- `seam-architecture-00` (sto-architecture-seam-01) — I.3-1
- `seam-architecture-01` (sto-architecture-seam-00) — I.3-1

## Out-of-station hand-offs

None (`handedOff = 0`). Every confirmed anchor is inside `crates/cobre-stochastic/`. One attacker `_needsHuman` note travels for a human read at the owner gate: architecture/seam flagged the genericity substring `cut_points` at `season_cast/mod.rs:439` (a `cut_points`/cut-set naming coincidence, not engine vocabulary) — recorded, not a finding.

## Defender pass

Four sub-station defenders (read-only, Opus) adjudicated all 35 candidates at the baseline: **35 confirmed, 0 dismissed**, 0 defender-raised needs-human. Every confirmation carries a `survivingClaim` strictly narrower than the candidate title (the defender's job here was scope-narrowing, not clearance): each names the exact anchor/condition and concedes the over-reach — e.g. `seam-performance-00`'s quadratic is narrowed to a setup-time `O(n_hydros·l_state²·S)` bounded by the PAR lag order, not a per-scenario cost. No verdict's implied fix fights the target layering (`conflicts = 0`). Alignment: {'neutral': 32, 'advances-1': 3}.

## Per-candidate roster

Every one of the 35 candidateRefs, exactly once, so no candidate is silently dropped (mirrors `verdicts.json`).

| candidateRef | disposition | verdict | partIRef |
|---|---|---|---|
| par-architecture-00 | defended | confirmed |  |
| par-over-engineering-00 | defended | confirmed |  |
| par-over-engineering-01 | defended | confirmed |  |
| par-performance-00 | defended | confirmed |  |
| par-performance-01 | defended | confirmed |  |
| par-performance-02 | defended | confirmed |  |
| par-test-bloat-00 | defended | confirmed |  |
| par-test-bloat-01 | defended | confirmed |  |
| par-test-bloat-02 | defended | confirmed |  |
| par-test-bloat-03 | defended | confirmed |  |
| sampling-architecture-00 | defended | confirmed |  |
| sampling-architecture-01 | defended | confirmed | I.3-1 |
| sampling-over-engineering-00 | defended | confirmed |  |
| sampling-over-engineering-01 | defended | confirmed |  |
| sampling-performance-00 | defended | confirmed |  |
| sampling-performance-01 | defended | confirmed |  |
| sampling-performance-02 | defended | confirmed |  |
| sampling-test-bloat-00 | defended | confirmed |  |
| sampling-test-bloat-01 | defended | confirmed |  |
| seam-architecture-00 | defended | confirmed | I.3-1 |
| seam-architecture-01 | defended | confirmed | I.3-1 |
| seam-over-engineering-00 | defended | confirmed |  |
| seam-performance-00 | defended | confirmed |  |
| seam-performance-01 | defended | confirmed |  |
| seam-test-bloat-00 | defended | confirmed |  |
| seam-test-bloat-01 | defended | confirmed |  |
| tree-noise-architecture-00 | defended | confirmed |  |
| tree-noise-architecture-01 | defended | confirmed |  |
| tree-noise-architecture-02 | defended | confirmed |  |
| tree-noise-over-engineering-00 | defended | confirmed |  |
| tree-noise-performance-00 | defended | confirmed |  |
| tree-noise-performance-01 | defended | confirmed |  |
| tree-noise-performance-02 | defended | confirmed |  |
| tree-noise-test-bloat-00 | defended | confirmed |  |
| tree-noise-test-bloat-01 | defended | confirmed |  |
