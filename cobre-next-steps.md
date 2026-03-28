# Cobre — Next Steps Action Plan

**Date:** 2026-03-24
**Scope:** Housekeeping, public presence, NEWAVE validation, and ongoing quality

---

## Phase 1 — Housekeeping (Single Claude Code Session)

All five items below are handled in one session, in this order.

### 1.1 Apply CLAUDE.md + Hooks + Pre-Commit System

Replace the current `CLAUDE.md` (311 lines) with the new slimmed version
(106 lines). Add the `.claude/` directory with settings, hooks, and
architecture rules. Add `scripts/check_suppressions.py` and
`scripts/pre-commit`.

**Files to add/replace:**

| File | Action |
|------|--------|
| `CLAUDE.md` | Replace with new version |
| `.claude/settings.json` | Create — hook configuration |
| `.claude/architecture-rules.md` | Create — context struct patterns, signature budgets |
| `.claude/hooks/check-hot-path.sh` | Create — PostToolUse hook for hot-path files |
| `scripts/check_suppressions.py` | Create — production clippy suppression checker |
| `scripts/pre-commit` | Create — git pre-commit hook |

After creating files, install the pre-commit hook:
```
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
chmod +x .claude/hooks/check-hot-path.sh
chmod +x scripts/check_suppressions.py
```

Verify: `python3 scripts/check_suppressions.py --max 15` should pass.

### 1.2 Run ADR/DEC Cleanup

Follow the plan in `adr-dec-cleanup-plan.md` step by step:

1. Move ADRs 010, 011 to `docs/design/`, strip ADR format
2. Replace all ADR references in source code doc comments with plain English
3. Replace all ADR references in book pages
4. Remove DEC reference in `determinism.rs`
5. Clean ADR labels from CHANGELOG (keep the feature descriptions)
6. Delete `docs/adr/` directory entirely

Verify: `grep -rn 'ADR-0\|docs/adr/' crates/ book/ CHANGELOG.md CLAUDE.md`
returns zero matches.

### 1.3 Fix Python Parity Gaps

Add three output writes to `cobre-python/src/run.rs` in `run_inner()`:

1. `cobre_io::write_scaling_report()` — write `training/scaling_report.json`
2. `cobre_io::write_solver_stats()` — write `training/solver/iterations.parquet`
3. `cobre_io::write_simulation_solver_stats()` — write `simulation/solver/scenarios.parquet`

Mirror the same calls from `cobre-cli/src/commands/run.rs` `write_outputs()`.
Each is ~5 lines. Total: ~20 lines of new code.

### 1.4 Update Book Roadmap

Add v0.1.10 and v0.1.11 deliverables sections to `book/src/reference/roadmap.md`,
matching the CHANGELOG entries for those versions.

### 1.5 Absorb Parameter Regression into Context Structs

Refactor hot-path function signatures to absorb parameters added during
v0.1.5–v0.1.11 back into context structs. Target: restore forward/backward
to ≤8 arguments and reduce the production `too_many_arguments` count.

Specific absorptions:

| Parameter | Current location | Target struct |
|-----------|-----------------|---------------|
| `cut_batches: &mut [RowBatch]` | `run_forward_pass`, `run_backward_pass` | `SolverWorkspace` or new `TrainingWorkspace` |
| `stage_bases: &[Option<Basis>]` | `simulate()` | Store on `StudySetup` after training |
| `lb_cut_batch` + `lb_cut_row_map` | `evaluate_lower_bound()` | Bundle into `LbWorkspace` struct |

After refactoring, run `python3 scripts/check_suppressions.py --max 15` and
verify the count decreased. Update the `--max` threshold in `scripts/pre-commit`
to match the new lower count.

**Commit message for Phase 1:**
```
refactor: apply new CLAUDE.md, remove ADR structure, fix Python parity, absorb parameter regression
```

---

## Phase 2 — Public Presence (1–2 weeks)

### 2.1 Slim the README

Reduce `README.md` from 218 lines to ~60–80 lines. Keep: logo, badges,
one-line pitch, 3–4 bullet points of what Cobre does, "Get Started" /
"Convert from NEWAVE" / "Python" links. Move to the book: crate table,
architecture diagram, roadmap section. Remove stale content.

The README's only job: tell a visitor what this is and where to go next.

### 2.2 Build the Landing Page for `cobre-rs.dev`

Create a `cobre-rs/cobre-rs.dev` repository (or use the org `.github` repo)
with a single static `index.html` page. No framework.

Content:
- Busbar logo (from `assets/cobre-logo-dark.svg`)
- Tagline: "Open infrastructure for power system computation"
- Three audience cards:
  - **Coming from NEWAVE?** → link to cobre-bridge, conversion tutorial
  - **Python user?** → `pip install cobre`, link to Jupyter quickstart
  - **Developer?** → GitHub link, contributing guide
- Hero visual: screenshot of the cobre-bridge dashboard or a convergence plot
- Footer: links to docs, methodology, GitHub, PyPI

Style: brand palette (copper primary, midnight background, IBM Plex fonts).
Deploy via GitHub Pages with the `cobre-rs.dev` custom domain.

### 2.3 Rewrite the Book Introduction

Replace the dense feature paragraph with:
- Short welcome (2–3 sentences)
- Two entry paths with visual callouts:
  - "Coming from NEWAVE or other Brazilian power system tools? →"
  - "New to power systems optimization? →"
- A hero image (dashboard screenshot or convergence plot)
- Brief "What Cobre does" list (5 bullets, user-facing language)
- Move the current feature inventory to a separate "What's Implemented" page
  under Reference

### 2.4 Apply Brand to the mdBook

Update `book/theme/css/custom.css` to use the brand palette:
- Headings: copper accent color (`#B87333`)
- Links: flow blue (`#4A90B8`)
- Code blocks: surface background (`#1A2028`)
- Sidebar highlights: copper light (`#D4956A`)
- Font: import IBM Plex Sans from Google Fonts for body text

This is CSS-only work. No content changes.

### 2.5 Set Up `cobre-rs.dev` Domain

1. Purchase `cobre-rs.dev` on Cloudflare
2. Configure DNS:
   - `cobre-rs.dev` → GitHub Pages (landing page repo)
   - `docs.cobre-rs.dev` → GitHub Pages CNAME for `cobre-rs.github.io/cobre/`
   - `methodology.cobre-rs.dev` → GitHub Pages CNAME for `cobre-rs.github.io/cobre-docs/`
3. Add CNAME file to each deployed repo
4. Update all cross-references in README, book, and cobre-docs

---

## Phase 3 — NEWAVE Validation (2–3 weeks, partially parallel with Phase 2)

### 3.1 Implement `cobre-bridge compare newave`

Follow the design in `compare-command-design.md`:

1. Phase 1 (1 day): hydro bounds comparison — entity alignment, sintetizador
   Parquet reading, Cobre `bounds.parquet` reading, terminal summary
2. Phase 2 (0.5 day): thermal and exchange bounds
3. Phase 3 (0.5 day): Parquet report output, CLI flags
4. Phase 4 (0.5 day): tests with fixture Parquets

### 3.2 Run the Bounds Comparison

Three incremental tests on a converted NEWAVE case:

| Test | Config | What it validates |
|------|--------|-------------------|
| 1 | Constant productivity, no inflow lags | Conversion pipeline, entity mapping, deficit segments, generic constraints, block factors, NCS, PAR estimation |
| 2 | Constant productivity, with inflow lags | z-inflow variable, lag state transition, PAR coefficient propagation through cuts |
| 3 | FPHA, with inflow lags | Hyperplane fitting, head-dependent production, full methodology |

Tests 1 and 2 require no converter changes — just config flags on the
converted case. Test 3 requires the converter to emit `hydro_production_models.json`
with computed FPHA mode.

Each test: run NEWAVE with 1 iteration → sintetizador-newave → run Cobre
with 1 iteration → `cobre-bridge compare newave` → analyze diff.

### 3.3 Publish Comparison Results

Add a page to the book: `book/src/reference/newave-comparison.md`.

Content: which case was converted, which configuration was used, summary
table of bounds matches/mismatches by entity type, known modeling differences
and their justification. Link from the book introduction's "Coming from
NEWAVE?" path.

This page is the credibility artifact. Publishing what matches and what
doesn't is more credible than silence.

---

## Phase 4 — Ongoing Quality

These run continuously after Phase 1, not as a separate milestone.

### 4.1 Work Down Suppression Count

The pre-commit hook threshold starts at 15. Each refactoring PR that absorbs
parameters into context structs should lower it. Update `scripts/pre-commit`
threshold after each reduction. Target: 0.

### 4.2 Maintain Python Parity

Every PR that adds a new output file in the CLI must include the
corresponding write in `cobre-python/src/run.rs`. The PostToolUse hook
on `run.rs` edits and the architecture-rules.md Python parity checklist
enforce this.

### 4.3 Keep CLAUDE.md Current

When bumping the workspace version, update the "Current State" section
in CLAUDE.md. This is a 2-minute task that prevents the staleness problem
from recurring. Add it to the release checklist in CONTRIBUTING.md.

---

## Dependency Graph

```
Phase 1 (day 1)
├── 1.1 CLAUDE.md + hooks + pre-commit
├── 1.2 ADR/DEC cleanup
├── 1.3 Python parity fix
├── 1.4 Book roadmap update
└── 1.5 Parameter regression absorption

Phase 2 (week 1–2, after Phase 1)
├── 2.1 Slim README
├── 2.2 Landing page ──→ 2.5 Domain setup
├── 2.3 Book introduction rewrite
└── 2.4 Brand CSS for mdBook

Phase 3 (week 2–3, partially parallel with Phase 2)
├── 3.1 compare command ──→ 3.2 Run comparison ──→ 3.3 Publish results
└── (depends on having a converted NEWAVE case + sintetizador output)

Phase 4 (ongoing)
├── 4.1 Suppression count reduction
├── 4.2 Python parity enforcement
└── 4.3 CLAUDE.md currency
```

---

## Success Criteria

| Milestone | Measurable outcome |
|-----------|-------------------|
| Phase 1 complete | `check_suppressions.py --max 15` passes, zero ADR/DEC references, Python writes all outputs CLI writes |
| Phase 2 complete | `cobre-rs.dev` resolves to a landing page, book intro has two audience paths, README is ≤80 lines |
| Phase 3 complete | `cobre-bridge compare newave` runs on a real case, bounds match ≥95% of hydro/thermal/exchange variables, results page published in book |
| Phase 4 ongoing | Suppression count decreasing over time, no new Python parity gaps, CLAUDE.md version matches workspace version |
