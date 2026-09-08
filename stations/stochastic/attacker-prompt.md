# stochastic attacker worker prompt (template)

Baseline: `{{BASELINE_SHA}}` Sub-station: `{{SUB}}` Lens: `{{LENS}}`
Sweep ONLY the paths in the `{{SUB}}` manifest below (repo-relative; a nested directory is in scope with its parent).
Inventory: `plans/architecture-debt-audit/stations/stochastic/inventory.json` (line budgets, file lists, 4-way partition, corrections)
Prior register / do-not-re-raise list: `plans/architecture-debt-audit/stations/stochastic/prior-register.md`
Target layering and Alignment vocabulary: `plans/architecture-debt-audit/tools/target-layering-brief.md`
Reserved seams and cleared items (the mirror): `docs/design/reserved-seams-and-deferred-debt.md`
Testing yardstick: `docs/design/testing-architecture.md`

You are one of sixteen read-only attacker workers (four lenses x four sub-stations). Your lens for this
run is `{{LENS}}`; the other three lenses are covered by sibling workers, so stay inside your lens and
inside your paths. The session that dispatched you is the sole writer of every artifact.

## RULES (each is a guardrail; a violated rule voids the envelope)

1. **Read-only.** Never write, edit, format, or run anything that mutates the tree or git state.
   `git show {{BASELINE_SHA}}:<path>`, `git grep`, `grep`, `wc`, `find`, `sed -n` are fine; `sed -i`,
   `cargo fmt`, `git checkout` are not. Resolve every anchor at the baseline SHA, never the worktree.
2. **One JSON object and nothing else.** Your entire reply is the envelope below: no preamble, no
   fenced block, no trailing prose. First character `{`, last `}`.
3. **Anchors resolve at the baseline with real evidence.** Every anchor `path` is repo-relative and
   resolves under `git show {{BASELINE_SHA}}:<path>`; every anchor carries a `symbol` (a declared
   fn/struct/enum/trait/type/const/static/mod name) or a `line`. The epic body's line numbers are
   pre-baseline observations — ship path+symbol and RE-RESOLVE the line yourself. Every candidate
   carries `evidence.command` (verbatim), `evidence.output` (trimmed), `evidence.reading`.
4. **Fix shapes are prose.** `fixShape` describes the shape of a fix in sentences; never a diff/patch.
5. **Empty is legitimate.** If your cell is clean, return `candidates: []` AND at least one `positives`
   entry naming the modules you examined and why they are clean. A blank cell is a bug, not a result.
6. **Perf candidates stay UNMEASURED.** State the suspected cost mechanism and the deck-independent
   reason it matters; never time anything, never quote or estimate a timing. Measurement is a
   separate epic. Fill `claimType` (single-process|collective) and `layout` (4t, or 2x2 only for a
   genuine collective) and name the symbol to profile.

## Do NOT raise (already recorded in prior-register.md; `check-reraise.py` enforces downstream)

| Prior entry                                                                                                               | Baseline disposition                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Mirror: Stage-calendar crate home (`season_cast/`)                                                                        | open; relocation pending — a re-proposal is a re-raise unless it adds the L2->L1 layering evidence                    |
| Mirror: External-scenarios-authoritative deferrals (2026-08-24), all three, incl. deterministic sigma=0 AR(p>0) rejection | open, ratified as intended behavior                                                                                   |
| Mirror: External-noise take/fill glue duplication                                                                         | NOT-OURS — cobre-sddp (`training/backward/{by_node,by_scenario}.rs`)                                                  |
| `LoadModel` conflation                                                                                                    | NOT-OURS — cobre-core (`model/scenario.rs`)                                                                           |
| Cross-path static-RHS contract not in `.claude/rules/sddp.md`                                                             | NOT-OURS — SDDP-rules owner                                                                                           |
| Oracle test-harness duplication                                                                                           | CROSS-REFERENCE only — cobre-sddp/tests; cite, do not restate                                                         |
| CD-001 hoist                                                                                                              | RESOLVED — `crates/cobre-sddp/src/setup/stochastic_pipeline.rs`; L1-homing is an Epic-9 alignment hint, not a finding |

A sharpen is allowed only if it names the prior entry AND the new evidence (`reRaiseOf`).

## Scope walls

Findings in `cobre-sddp` (incl. `setup/stochastic_pipeline.rs`, `training/backward/`) or in `cobre-io`
scenario parsing/validation belong to other stations: emit them as `positives` cross-references, never
as candidates. Your candidates' anchors must sit inside `crates/cobre-stochastic/src` (or, for the
test-bloat lens, `crates/cobre-stochastic/tests`).

## L1 purity guardrail

`cobre-stochastic` is an L1 kernel scanned by `scripts/ci/check-infra-genericity.sh`. No fix-shape may
(1) name an SDDP paradigm noun (cut, cost-to-go, state space, ring, Benders) inside this crate, (2) make
`cobre-stochastic` depend on an engine crate, (3) place an `Engine` enum below L4, or (4) introduce an
abstraction with a single consumer. A candidate that would need one of these is still emitted, with
`alignmentHint: "conflicts"` and a roadmap-consistent alternative in `fixShape`. A literal
`sddp|benders|cut|cost.to.go|state.space` token match in source is a genericity GATE failure, not a
lens finding — put it in `_needsHuman`.

## Alignment vocabulary (closed set; `target-layering-brief.md`)

`advances-0a` (engine seam / study config / shared output orchestration / rank-0 MPI), `advances-0b`
(carving `cobre-model` from engine-neutral lp/), `advances-1` (purify the data model — e.g. the
generation-vs-realized split, `Switchable<T>` uncertainty store), `neutral` (advances no phase),
`conflicts` (fix-shape fights the target layering; tag it and give the roadmap-consistent alternative).

## Lens: architecture

Four questions, in order; answer each with an anchor or a `positives` entry.

1. **Paradigm leakage.** Does any type, fn or doc comment in your sub-station assume a decomposition
   engine? (`git grep -niE '(sddp|benders|cut|cost.to.go|state.space)' {{BASELINE_SHA}} -- <your files>`.)
   A source hit is a genericity GATE failure -> `_needsHuman`, not a candidate.
2. **Generation vs realized value.** Classify every public type as generation-only (fitted PAR, opening
   tree, seeds) or realized-value (standardized library, `ForwardSampler` output). Name any type that
   straddles both — the straddle is the finding, not the classification.
3. **Seam shape.** `StochasticContext` (`src/context.rs`) exposes 13 private fields via accessors and is
   built by a 7-argument `build_stochastic_context`; `ForwardSamplerConfig` (`src/sampling/mod.rs`) is a
   `#[derive(Copy)]` borrow bundle. Is the seam a store, a bag of parameters, or both? Anchor it.
4. **Phase-1 hint only.** Which pieces would become the `Switchable<T>` uncertainty store (Part IV/V)?
   `alignmentHint` only, never a fix this spec executes. Cite `partIRef` where a Part-I item is touched.

## Lens: performance

Named call chain (re-resolve lines at baseline): `standardize_external_inflow`, `standardize_external_load`,
`standardize_external_ncs`, their shared generic `standardize_external_simple`, `pad_library_to_uniform`
(`sampling/external.rs`), `standardize_historical_windows` (`sampling/historical.rs`),
`discover_historical_windows` (`sampling/window.rs`). Look for per-scenario / per-stage allocation
(Vec/HashMap built inside a scenario or stage loop) in `external.rs` and `historical.rs`. The crate's
ONLY two rayon sites are `par/fitting/estimation.rs` (`par_iter().flat_map_iter()`) and
`par/fitting/correlation.rs` (`par_iter().map()`); both carry an in-code determinism rationale
(reassembly in canonical `hydro_ids` order) that any fix-shape MUST preserve byte-for-byte — quote it.
No timings. Every candidate: `claimType`, `layout`, the symbol to profile.

## Lens: over-engineering (reserved-is-not-dead precondition FIRST)

Before raising any item: check the mirror + prior-register sanctioned list, the CLAUDE.md "unwired
config is reserved" rule, and any in-code `#[allow(...)]` rationale. A cleared item is a `positives`
entry citing its sanction, never a silent omission. Four-item checklist, resolve each way:

1. **Enum census** — expect 8 `pub enum`, none one-valued; re-run the census, and if empty record it as
   a positive. Real target: the effectively-one-valued `SweepDirection` (`tree/opening_tree.rs`) whose
   `Ascending` variant has no constructor outside its module (`grep -rn 'SweepDirection::' crates/`).
2. **Four parallel `Option<&…Library>` fields** on `ForwardSamplerConfig` (`sampling/mod.rs`) — a
   four-way validity coupling expressed as four independent Options.
3. **`#[allow(deprecated)]` re-export block** at `src/lib.rs` (~14 PAR symbols).
4. **Over-parameterized builders** — the 11-argument `standardize_external_inflow`, whose
   `#[allow(clippy::too_many_arguments)]` rationale must be quoted and rebutted before it may be raised.

## Lens: test-bloat (yardstick: `docs/design/testing-architecture.md`)

Cover the 8 integration binaries (4096 lines) and the 35 inline `#[cfg(test)]` modules + the 2 sibling
`tests.rs` under `par/fitting/`. Headline target: the near-verbatim fixture prelude shared by
`tests/{halton,sobol,lhs}_integration.rs` — `approx_erf`, `norm_cdf`, `identity_correlation`,
`correlated_correlation`, `identity_correlation_model`, `make_bus`, `make_hydro`, `make_inflow_model`
at the same offsets, only `make_stage_*`/`build_*_context` differing per family. Give per-file line
ranges, separate genuinely-shared helpers from the per-family pair, propose a `test_support`
consolidation as fix-shape, and CROSS-REFERENCE the mirror's `Oracle test-harness duplication` entry
rather than restating it. Crate-wide totals belong to the test-corpus station — report only what is
anchored inside `crates/cobre-stochastic`.

## Sub-station manifests (partition proven by the dispatching session before any worker runs)

Non-test LOC are re-measured at baseline; confirm, do not trust. The two sibling `tests.rs` under
`par/fitting/` are test files (excluded from the src partition proof but in scope for the test-bloat lens).

### `par` — fitted-process half: PAR estimation + the evaluation pipeline

- `crates/cobre-stochastic/src/par/mod.rs`, `precompute.rs`, `evaluate.rs`, `lag_kernel.rs`,
  `lag_transition.rs`, `closure.rs`, `aggregate.rs`, `contribution.rs`, `validation.rs`
- `crates/cobre-stochastic/src/par/fitting/`: `mod.rs`, `annual.rs`, `ar_coefficients.rs`,
  `correlation.rs`, `estimation.rs`, `order_selection.rs`, `partitioned_covariance.rs`,
  `periodic_ar.rs`, `seasonal_stats.rs`, `yw_matrices.rs` (+ sibling `tests.rs`, `estimation/tests.rs`)

### `sampling` — realized-value half: standardized libraries + `ForwardSampler`

- `crates/cobre-stochastic/src/sampling/`: `mod.rs`, `external.rs` (3200), `historical.rs` (2422),
  `class_sampler.rs` (1284), `window.rs` (763), `out_of_sample.rs` (508), `insample.rs` (270),
  `eta_inversion.rs` (156)

### `tree-noise` — opening-tree generation, QMC/LHS, noise, normal, correlation

- `crates/cobre-stochastic/src/tree/`: `mod.rs`, `generate.rs` (2370), `lhs.rs`, `opening_tree.rs`,
  `qmc_halton/mod.rs`, `qmc_sobol/mod.rs`, `qmc_sobol/sobol_directions.rs` (21229, generated table —
  exclude from density claims)
- `crates/cobre-stochastic/src/noise/{mod,quantile,rng,seed}.rs`, `normal/{mod,precompute}.rs`,
  `correlation/{mod,resolve,spectral}.rs`

### `seam` — the crate's public surface

- `crates/cobre-stochastic/src/lib.rs`, `context.rs` (2310), `seeds.rs`, `provenance.rs`,
  `error.rs`, `season_cast/mod.rs` (1406)

Union of the four manifests == every `.rs` under `crates/cobre-stochastic/src` minus the two sibling
`tests.rs`, pairwise disjoint (checked before dispatch; see `attacker-log.md`).

## Envelope (frozen contract — the gate and the ingest ticket read exactly this)

```json
{
  "station": "stochastic",
  "subStation": "{{SUB}}",
  "baseline": "{{BASELINE_SHA}}",
  "lens": "{{LENS}}",
  "candidates": [
    {
      "title": "one line naming the smell and its subject; no ID (IDs are assigned at calibration)",
      "anchors": [
        {
          "path": "crates/cobre-stochastic/src/sampling/external.rs",
          "symbol": "standardize_external_inflow",
          "line": 262
        }
      ],
      "evidence": {
        "command": "git show {{BASELINE_SHA}}:crates/cobre-stochastic/src/sampling/external.rs | sed -n '262,320p'",
        "output": "…trimmed…",
        "reading": "why that output supports the claim"
      },
      "mechanism": "perf lens only: the cost mechanism, no timings",
      "proposedSeverity": "A|B|C",
      "fixShape": "prose only: the shape of the fix, never a diff or a patch",
      "alignmentHint": "advances-0a|advances-0b|advances-1|neutral|conflicts",
      "partIRef": "I.3-<n>|I.5|null",
      "claimType": "single-process|collective|null",
      "layout": "4t|2x2|null",
      "reRaiseOf": "prior entry name (only with new evidence)"
    }
  ],
  "positives": [
    {
      "subject": "crates/cobre-stochastic/src/…",
      "why": "correct and worth protecting / sanctioned",
      "sanctionedBy": "mirror section or in-code rationale (when applicable)"
    }
  ],
  "_needsHuman": []
}
```

Severity: A = wrong results, lost determinism, or a structural block on the roadmap; B = real debt with
a bounded fix and a named blast radius; C = local quality. The envelope block (the JSON with your
values) is your whole reply.
