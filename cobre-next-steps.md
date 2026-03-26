# Cobre — Remaining Work & Priorities

**Date:** 2026-03-26
**Branch:** `feat/backward-pass-performance`
**Current version:** 0.1.11 (unreleased changes on feature branch)

---

## Context

This document replaces the original `cobre-next-steps.md` (2026-03-24). Since
that plan was written, a major backward-pass performance program was executed
(6 epics, 19 tickets — all completed), plus cut selection observability output
was added. The original plan's phases need to be re-assessed against what was
actually done.

### What Was Completed Since the Original Plan

**From the original Phase 1 (Housekeeping):**

- 1.1 CLAUDE.md rewrite + `.claude/` directory + hooks + architecture rules
- 1.2 ADR/DEC cleanup (directory deleted, references removed)
- 1.5 Parameter regression absorption (hot-path signatures at budget)

**Not in the original plan (backward-pass performance program):**

- Instrumentation: solver timing fields (`basis_set_time_ms`, `sync_time_ms`, etc.)
- Structural bugfix: cut sync moved inside per-stage backward loop (MPI correctness)
- Quick wins: skip refactorization in openings 1..N, sparse cut injection (~30% NNZ reduction)
- Cut selection: Level1 + Lml1 strategies, basis reuse after selection changes
- Incremental cut injection: LP persistence, `add_rows` only new cuts, CutRowMap infrastructure
- Solver tuning: simplex strategy benchmarking, pre-allocated coefficient buffers, Vec replaces HashMap
- Cut selection observability: Parquet output pipeline wired in both CLI and Python
- Configurable simplex strategy
- Cut selection bugfix

---

## Priority 1 — Quick Fixes (Low Effort, High Value)

Items that close gaps from the original plan and can each be done in minutes.
Should be completed before merging the current feature branch or cutting a release.

### 1.1 Python Parity — 3 Missing Output Writes

**Status:** NOT DONE
**Effort:** ~20 lines of code
**Files:** `crates/cobre-python/src/run.rs`

The CLI writes these outputs but the Python bindings do not:

| Output                                | CLI location (`run.rs`) | Function                                    |
| ------------------------------------- | ----------------------- | ------------------------------------------- |
| `training/scaling_report.json`        | Line 339                | `cobre_io::write_scaling_report()`          |
| `training/solver/iterations.parquet`  | Line 784                | `cobre_io::write_solver_stats()`            |
| `simulation/solver/scenarios.parquet` | Line 807                | `cobre_io::write_simulation_solver_stats()` |

Mirror the same calls from `cobre-cli/src/commands/run.rs` into `cobre-python/src/run.rs`.
The `write_cut_selection_records` call is already present in Python (line 374).

### 1.2 Book Roadmap — Add v0.1.10 and v0.1.11

**Status:** NOT DONE
**Effort:** ~30 minutes
**File:** `book/src/reference/roadmap.md`

The roadmap page stops at v0.1.9. Add sections for v0.1.10 and v0.1.11
matching the format of existing entries, sourced from `CHANGELOG.md`.

Also consider adding a v0.1.12 (unreleased) section covering the backward-pass
performance work and cut selection observability, since those are significant
deliverables that will ship with the next release.

### 1.3 Pre-Commit Hook Script

**Status:** NOT DONE
**Effort:** ~15 minutes
**File:** `scripts/pre-commit` (create), `.git/hooks/pre-commit` (symlink)

The original plan called for a `scripts/pre-commit` hook that runs
`check_suppressions.py` and `cargo fmt --check` before each commit. The script
was never created. The suppression checker exists at `scripts/check_supressions.py`
(note: filename has typo "supressions" — consider fixing to "suppressions").

Steps:

1. Create `scripts/pre-commit` (runs `cargo fmt --check` + `check_suppressions.py --max 15`)
2. `ln -sf ../../scripts/pre-commit .git/hooks/pre-commit`
3. `chmod +x scripts/pre-commit`

---

## Priority 2 — Public Presence Foundation (Medium Effort)

Items that prepare Cobre for external visibility. These are independent of
each other and can be done in any order. The README and book introduction
are the highest-impact items here since they're what visitors see first.

### 2.1 Slim the README

**Status:** NOT DONE
**Effort:** 1–2 hours
**File:** `README.md`

Reduce from current length to ~60–80 lines. The README becomes a signpost:

**Keep:** Logo, badges, one-line description, 3–4 bullet points, quick install,
"Get Started" / "Convert from NEWAVE" / "Python" links.

**Move to book:** Crate table, architecture diagram, detailed roadmap.

**Remove:** Stale content, feature inventories that belong in docs.

### 2.2 Rewrite the Book Introduction

**Status:** NOT DONE
**Effort:** 2–3 hours
**File:** `book/src/introduction.md`

Replace the dense feature paragraph with:

- Short welcome (2–3 sentences)
- Three audience paths with visual callouts:
  - "Coming from NEWAVE?" → migration guide
  - "New to SDDP?" → conceptual intro
  - "Python user?" → Python quickstart
- Brief "What Cobre does" list (5 bullets, user-facing language)
- Move current feature inventory to a separate Reference page

### 2.3 Apply Brand CSS to mdBook

**Status:** NOT DONE
**Effort:** 1–2 hours
**File:** `book/theme/css/custom.css`

CSS-only changes, no content modifications:

- Headings: copper accent (`#B87333`)
- Links: flow blue (`#4A90B8`)
- Code blocks: surface background (`#1A2028`)
- Sidebar highlights: copper light (`#D4956A`)
- Fonts: IBM Plex Sans (body), JetBrains Mono (code)

### 2.4 Create New Book Pages

**Status:** NOT DONE
**Effort:** 1–2 days total

| Page                            | Audience     | Content                                                |
| ------------------------------- | ------------ | ------------------------------------------------------ |
| `guide/newave-migration.md`     | NEWAVE users | Entity mapping, conversion tutorial, known differences |
| `tutorial/what-cobre-solves.md` | Newcomers    | Conceptual introduction, water balance diagram         |
| `guide/python-quickstart.md`    | Python users | `pip install` → 5 cells → results                      |

These pages can be created incrementally. The NEWAVE migration page can start
as a stub and gain comparison data after Phase 3 validation work.

---

## Priority 3 — Domain & Landing Page (Larger Effort)

These items require work outside the main repository and involve DNS
configuration and a new GitHub repo.

### 3.1 Domain Setup (`cobre-rs.dev`)

**Status:** NOT DONE
**Effort:** 1–2 hours (DNS + GitHub Pages config)

1. Purchase `cobre-rs.dev` on Cloudflare (if not already done)
2. Configure DNS:
   - `cobre-rs.dev` → GitHub Pages (landing page repo)
   - `docs.cobre-rs.dev` → `cobre-rs.github.io/cobre/`
   - `methodology.cobre-rs.dev` → `cobre-rs.github.io/cobre-docs/`
3. Add CNAME files to deployed repos
4. Update cross-references in README and book

### 3.2 Build the Landing Page

**Status:** NOT DONE (prototypes exist)
**Effort:** 1–2 days
**Repo:** `cobre-rs/cobre-rs.dev` (to be created)

Single `index.html`, no framework. Prototypes produced in a previous session:

- `hero-energy-landscape.html` — atmospheric SVG hero with GSAP animation
- `cobre-landing-page.html` — below-fold content (audience cards, metrics, code examples)
- `interactive-hydro-dispatch.html` — interactive dispatch simulator

Design spec is in `cobre-dev-strategy.md` sections 3 and 5.

Tech: inline SVG + GSAP 3.x, IBM Plex Sans + JetBrains Mono, no build step.

---

## Priority 4 — NEWAVE Validation (Significant Effort)

This work produces the credibility artifact — published comparison results
showing how Cobre's bounds match NEWAVE on real cases. Depends on having
`cobre-bridge convert` working on a real NEWAVE case.

### 4.1 Implement `cobre-bridge compare newave`

**Status:** NOT DONE
**Effort:** 2–3 days
**Design:** `compare-command-design.md`

Phased implementation:

1. Hydro bounds comparison (entity alignment, sintetizador reading, terminal summary)
2. Thermal and exchange bounds
3. Parquet report output, CLI flags
4. Tests with fixture Parquets

### 4.2 Run Bounds Comparison

**Status:** NOT DONE
**Effort:** 1–2 days
**Dependency:** 4.1 + a converted NEWAVE case

Three incremental tests:

| Test | Config                           | Validates                                                               |
| ---- | -------------------------------- | ----------------------------------------------------------------------- |
| 1    | Constant productivity, no lags   | Conversion pipeline, entity mapping, block factors, NCS, PAR estimation |
| 2    | Constant productivity, with lags | z-inflow variable, lag state transition, PAR propagation through cuts   |
| 3    | FPHA, with lags                  | Hyperplane fitting, head-dependent production, full methodology         |

### 4.3 Publish Comparison Results

**Status:** NOT DONE
**Effort:** 0.5 days
**File:** `book/src/reference/newave-comparison.md`

Transparent publication: what matches, what doesn't, known modeling differences
and their justification. This page is the primary credibility artifact.

---

## Priority 5 — Ongoing Quality

These are continuous activities, not discrete milestones.

### 5.1 Work Down Suppression Count

**Current:** 15 suppressions (at threshold)
**Target:** 0

Each refactoring PR that absorbs parameters into context structs should lower
the threshold in the pre-commit hook. The suppressions live in infrastructure
crates and generic constructors — not hot-path functions.

### 5.2 Maintain Python Parity

Every PR that adds a new output file in the CLI must include the corresponding
write in `cobre-python/src/run.rs`. The architecture rules document
(`.claude/architecture-rules.md`) has a Python parity checklist.

### 5.3 Keep CLAUDE.md Current

When bumping the workspace version, update the "Current State" section.
Add this to the release checklist in `CONTRIBUTING.md`.

---

## Recommended Execution Order

```
NOW (before merge / next release)
├── 1.1 Python parity (3 output writes)
├── 1.2 Book roadmap v0.1.10 / v0.1.11 / v0.1.12
└── 1.3 Pre-commit hook script

NEXT (public presence foundation)
├── 2.1 Slim README
├── 2.2 Book introduction rewrite
├── 2.3 Brand CSS
└── 2.4 New book pages (can be incremental)

THEN (domain & landing page)
├── 3.1 Domain setup
└── 3.2 Landing page build

WHEN READY (NEWAVE validation)
├── 4.1 compare command
├── 4.2 Run comparison
└── 4.3 Publish results

ALWAYS (ongoing)
├── 5.1 Suppression count reduction
├── 5.2 Python parity enforcement
└── 5.3 CLAUDE.md currency
```

---

## Files to Clean Up

These files in the repo root are planning artifacts that should be removed
or relocated once their content is acted upon:

| File                              | Action                                                     |
| --------------------------------- | ---------------------------------------------------------- |
| `cobre-next-steps.md`             | Replace with this document, then delete after work is done |
| `cobre-dev-strategy.md`           | Move design references to `docs/design/`, delete the rest  |
| `hero-energy-landscape.html`      | Move to landing page repo when created                     |
| `cobre-landing-page.html`         | Move to landing page repo when created                     |
| `interactive-hydro-dispatch.html` | Move to landing page repo when created                     |
| `hero-atmosphere.html`            | Delete (superseded)                                        |
| `hero-animation-prototype.html`   | Delete (rejected)                                          |
| `visualization-guidelines.md`     | Move to `docs/design/`                                     |
| `adr-dec-cleanup-plan.md`         | Delete (completed)                                         |
| `compare-command-design.md`       | Move to `docs/design/`                                     |
