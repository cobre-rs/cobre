# Cobre — Documentation & Public Presence Strategy

**Date:** 2026-03-25
**Scope:** Everything related to how Cobre presents itself to the world:
domain, landing page, mdBook documentation, user onboarding journeys,
visual design, and the relationship between all deliverables.

---

## 1. Domain & Hosting Architecture

**Domain:** `cobre-rs.dev` (purchased on Cloudflare)

| URL | Content | Hosting |
|-----|---------|---------|
| `cobre-rs.dev` | Landing page (single HTML file) | GitHub Pages on `cobre-rs/cobre-rs.dev` repo |
| `docs.cobre-rs.dev` | Software documentation (mdBook) | GitHub Pages CNAME → `cobre-rs.github.io/cobre/` |
| `methodology.cobre-rs.dev` | Methodology reference (mdBook) | GitHub Pages CNAME → `cobre-rs.github.io/cobre-docs/` |

The landing page is the front door. The mdBook is the house. The methodology
reference is the library. Each has a distinct role and audience.

---

## 2. Target Audiences

Three distinct user groups, in order of adoption priority:

### 2.1 NEWAVE users (immediate adoption)

**Who:** Brazilian power systems engineers using NEWAVE, DECOMP, DESSEM
daily. They know the problem domain deeply. They don't know Rust.

**What they want:** Can I convert my NEWAVE case and get similar results?

**Entry point:** "Coming from NEWAVE?" card on landing page → dedicated
migration guide in the book.

**Key content:**
- NEWAVE → Cobre entity mapping table (hidr.dat → hydros.json, etc.)
- `cobre-bridge convert newave` command with output screenshot
- Bounds comparison results (once `cobre-bridge compare` is built)
- What's supported, what's not yet, known modeling differences

**Tone:** Complementary and transparent. Never say "better than NEWAVE."
Let the comparison data speak for itself.

### 2.2 Newcomers — researchers & students (long-term growth)

**Who:** Graduate students studying SDDP, engineers from other countries,
Rust developers interested in scientific computing.

**What they want:** What does this solve, and can I run an example?

**Entry point:** "New to SDDP?" card on landing page → conceptual
introduction → quickstart → understanding results.

**Key content:**
- Water balance conceptual diagram (the tradeoff visualization)
- Quickstart: `cobre init` → `cobre run` → see results in 3 commands
- Annotated convergence plot explaining what lower/upper bounds mean
- Dispatch timeline showing the solver learned merit-order dispatch
- Interactive hydro dispatch simulator

### 2.3 Python users (accessibility)

**Who:** Data scientists, energy analysts who work in Jupyter. May not
know what SDDP is.

**What they want:** `pip install`, run, get data back in polars.

**Entry point:** "Python user?" card on landing page → Python quickstart
in the book.

**Key content:**
- `pip install cobre-python`, 5 cells of Python
- Rendered notebook cells with output visible
- polars DataFrame screenshot showing Arrow zero-copy results

---

## 3. Landing Page Design (`cobre-rs.dev`)

### 3.1 Page structure (top to bottom)

| Section | Height | Content |
|---------|--------|---------|
| **Nav bar** | Fixed | Logo + "Cobre" left, Docs / GitHub / PyPI links right. Transparent with backdrop blur. Always visible. |
| **Hero** | 90vh | Atmospheric energy landscape SVG + GSAP animation. Headline, tagline, `pip install cobre-python` CTA. |
| **Audience cards** | Auto | Three cards with colored top borders: NEWAVE (red), New to SDDP (blue), Python (amber). Each links to the right book page. |
| **Metrics** | Auto | Three numbers: 164K lines, 2,747 tests, 0 proprietary deps. |
| **Code examples** | Auto | Tabbed: CLI / Python / Convert. Syntax-highlighted with brand colors. |
| **Footer** | Auto | Docs, GitHub, PyPI, Apache-2.0. |

### 3.2 Hero animation

An atmospheric dusk scene of a complete energy landscape. Not a diagram —
a mood. Not interactive — ambient.

**Composition (left to right):**
1. Wind turbines on a high ridge (3 turbines, blades rotating at different speeds)
2. Reservoir filling the valley behind the dam (water with shimmer)
3. Dam structure (largest element, visual anchor, penstock, powerhouse with warm windows)
4. Solar field on flat ground downstream (angled panels with glint sweep)
5. Thermal plant (cooling tower silhouettes, smokestack with glow, heat shimmer)
6. Transmission towers receding toward horizon (power lines with spark pulses)
7. City skyline at bottom right (building silhouettes growing from page edge, amber-lit windows)

**Tech:** Inline SVG + GSAP 3.x + MotionPathPlugin. No frameworks.
Self-contained in `index.html`. All animations use `svgOrigin` for SVG
rotation, `prefers-reduced-motion` respected.

**Mood:** Standing near a dam at dusk. Copper glow on the horizon, dark
sky with faint stars, warm amber in the powerhouse and city windows.
The brand palette IS the scene: midnight sky, copper horizon, flow blue
water, spark amber electricity, patina green from the wind farm.

**Current prototype:** `hero-energy-landscape.html`

### 3.3 Tech stack

| Component | Implementation |
|-----------|---------------|
| Landing page | Single `index.html`, no build step |
| Fonts | Google Fonts: IBM Plex Sans + JetBrains Mono |
| Animation | GSAP 3.12.5 + MotionPathPlugin from cdnjs |
| Charts (below fold) | Chart.js 4.4.1 from cdnjs if needed |
| Hosting | GitHub Pages |
| CSS | Inline `<style>`, CSS custom properties for palette |

No React, no Vue, no Tailwind, no bundler. The landing page is one file.

---

## 4. mdBook Documentation Restructure

### 4.1 New introduction page

Replace the current wall-of-text introduction with:
- Short welcome (2–3 sentences)
- Three audience cards with visual callouts and directional links
- Hero image (convergence plot or dashboard screenshot)
- Brief "What Cobre does" list (5 bullets, user-facing language)

Move the current feature inventory to a separate reference page.

**Prototype:** Book introduction mockup (SVG rendered inline earlier in session)

### 4.2 New pages to create

| Page | Audience | Content |
|------|----------|---------|
| `guide/newave-migration.md` | NEWAVE users | Entity mapping, conversion tutorial, comparison results |
| `tutorial/what-cobre-solves.md` | Newcomers | Conceptual introduction with water balance diagram |
| `guide/python-quickstart.md` | Python users | pip install → 5 cells → results, rendered notebook |

### 4.3 Existing pages to improve

| Page | Change |
|------|--------|
| `introduction.md` | Rewrite with three-path routing (see 4.1) |
| `tutorial/quickstart.md` | Already good, keep as-is |
| `tutorial/understanding-results.md` | Add annotated convergence plot and dispatch timeline |
| `reference/roadmap.md` | Add v0.1.10 and v0.1.11 entries |

### 4.4 Brand CSS for mdBook

Update `book/theme/css/custom.css` to apply the brand palette:
- Headings: copper accent (`#B87333`)
- Links: flow blue (`#4A90B8`)
- Code blocks: surface background (`#1A2028`)
- Sidebar highlights: copper light (`#D4956A`)
- Font: IBM Plex Sans for body, JetBrains Mono for code
- Default theme: coal (already set in `book.toml`)

CSS-only work. No content changes.

---

## 5. Visualizations

### 5.1 Design guidelines

Full specification in `visualization-guidelines.md`. Key principles:

- **Power systems visual language.** Reservoirs look like dams, generators
  use domain-standard symbols, buses are thick bars. Not generic charts.
- **Brand palette exclusively.** No Chart.js defaults, no generic blues.
  Fixed generation color mapping: hydro=blue, thermal=amber, deficit=dimmed.
- **Dark-first.** Midnight background is default. Light mode supported
  but secondary.
- **Animation serves understanding.** Water flows because dispatch is
  happening. Turbines spin because generation is active. No decoration.

### 5.2 Diagrams and their placement

| Visualization | Type | Placement | Purpose |
|--------------|------|-----------|---------|
| Water balance tradeoff | Static SVG | `tutorial/what-cobre-solves.md` | Conceptual intro for newcomers |
| NEWAVE → Cobre entity mapping | Static SVG | `guide/newave-migration.md` | Recognition signal for NEWAVE users |
| Annotated convergence plot | Chart.js or static PNG | `tutorial/understanding-results.md` | Teaches what SDDP convergence means |
| Dispatch timeline | Chart.js or static PNG | `tutorial/understanding-results.md` | Shows what the solver learned |
| Interactive dispatch simulator | HTML + CSS + JS | `book/src/interactive/hydro-dispatch.html` | Hands-on exploration for newcomers |
| Energy landscape | SVG + GSAP | `cobre-rs.dev` hero | Atmospheric first impression |

### 5.3 Prototypes produced

| File | Description | Status |
|------|-------------|--------|
| `hero-energy-landscape.html` | Landing page hero with full energy landscape | Ready for refinement |
| `hero-atmosphere.html` | Earlier atmospheric prototype (dam only) | Superseded |
| `hero-animation-prototype.html` | One-line diagram attempt | Rejected — too technical |
| `interactive-hydro-dispatch.html` | Interactive simulator with reservoir, sliders, dispatch bars | Ready for refinement |
| `cobre-landing-page.html` | Full landing page below-the-fold content | Ready for integration |
| Water balance SVG | Conceptual diagram (rendered inline) | Needs export to file |
| NEWAVE mapping SVG | Entity mapping table (rendered inline) | Needs export to file |
| Book intro mockup | Three-card introduction page (rendered inline) | Reference for implementation |
| Convergence plot | Annotated Chart.js plot (rendered inline) | Needs export/recreation |
| Dispatch timeline | Stacked bar chart (rendered inline) | Needs export/recreation |

---

## 6. README Redesign

Slim from 218 lines to ~60–80 lines. The README becomes a signpost:

**Keep:** Logo, badges, one-line description, 3–4 bullet points of what
Cobre does, "Get Started" / "Convert from NEWAVE" / "Python" links.

**Move to book:** Crate table, architecture diagram, roadmap section.

**Remove:** Stale content, detailed feature lists.

The README's job: answer "what is this and where do I go" in 15 seconds.

---

## 7. Relationship to Other Work Streams

This strategy connects to the broader cobre-next-steps.md action plan:

| Phase | Relationship |
|-------|-------------|
| Phase 1 (housekeeping) | Must complete first — CLAUDE.md, ADR cleanup, Python parity |
| Phase 2 (public presence) | **This document drives Phase 2.** Landing page, README, book intro, brand CSS, domain setup. |
| Phase 3 (NEWAVE validation) | Produces the content for the NEWAVE migration page. Bounds comparison results become the credibility artifact. Dashboard screenshot becomes a book visual. |
| Phase 4 (ongoing) | Keep CLAUDE.md current, maintain Python parity, work down suppressions |

### Implementation order within Phase 2

1. Purchase and configure `cobre-rs.dev` domain on Cloudflare
2. Create `cobre-rs/cobre-rs.dev` repo with `index.html` (hero + below-fold content)
3. Configure DNS: `cobre-rs.dev` → GitHub Pages, `docs.cobre-rs.dev` → mdBook
4. Apply brand CSS to mdBook (`custom.css`)
5. Rewrite book introduction with three-path routing
6. Create `guide/newave-migration.md` (initially without comparison data)
7. Create `tutorial/what-cobre-solves.md` with water balance diagram
8. Create `guide/python-quickstart.md` with rendered notebook
9. Slim the README
10. Add annotated convergence plot and dispatch timeline to understanding-results page

---

## 8. Files Produced This Session

### Ready to apply to repo (Phase 1)

| File | Target path in repo |
|------|-------------------|
| `CLAUDE.md` | `CLAUDE.md` (replace) |
| `.claude/settings.json` | `.claude/settings.json` (create) |
| `.claude/architecture-rules.md` | `.claude/architecture-rules.md` (create) |
| `.claude/hooks/check-hot-path.sh` | `.claude/hooks/check-hot-path.sh` (create) |
| `scripts/check_suppressions.py` | `scripts/check_suppressions.py` (create) |
| `scripts/pre-commit` | `scripts/pre-commit` (create, symlink to `.git/hooks/`) |
| `adr-dec-cleanup-plan.md` | Reference for Claude Code to execute cleanup |
| `cobre-next-steps.md` | Reference for sequencing all work |

### Design references (Phase 2)

| File | Purpose |
|------|---------|
| `hero-energy-landscape.html` | Landing page hero prototype — refine and integrate |
| `cobre-landing-page.html` | Below-fold content prototype — integrate with hero |
| `interactive-hydro-dispatch.html` | Tutorial interactive — link from book |
| `visualization-guidelines.md` | Design system for all visual output |
| This document | Master strategy tying everything together |

### Superseded (do not use)

| File | Why |
|------|-----|
| `hero-animation-prototype.html` | Rejected — one-line diagram approach was too technical |
| `hero-atmosphere.html` | Superseded by `hero-energy-landscape.html` |

---

## 9. Resolved Design Decisions

1. **Interactive simulator stays outside the book.** The dispatch simulator
   and any future interactive visualizations are standalone HTML pages
   hosted on `cobre-rs.dev` (e.g. `cobre-rs.dev/interactive/hydro-dispatch.html`).
   The book links to them but does not embed via iframe. This keeps the
   mdBook simple and avoids JS/sizing issues.

2. **Live Chart.js for convergence and dispatch visualizations.** The
   annotated convergence plot and dispatch timeline are built as Chart.js
   widgets in standalone HTML pages, styled with the brand palette. These
   are hosted alongside the interactive simulator and linked from the
   book. This gives interactivity (hover tooltips, responsive sizing)
   without complicating the mdBook build.

3. **Multilingual support (English + Portuguese).** The documentation
   will support English and Portuguese as a medium-term goal. Key
   domain terms in the NEWAVE migration page should use Portuguese
   alongside English immediately (e.g. "subsistema (subsystem)",
   "usina (plant)"). Full i18n for the mdBook and landing page is
   a separate work item — mdBook supports `[output.html]` language
   switching natively. Landing page can use a simple lang toggle.

4. **2D SVG + GSAP hero.** The current atmospheric energy landscape
   approach is the final direction. No 3D / Three.js. The 2D version
   is lighter, works on mobile, loads fast, and the mood is right.