# Cobre — Brand Guidelines

## 1. Name & Identity

**Cobre** (Portuguese: copper) — the metal that conducts electricity. The name captures the project's essence: foundational infrastructure for power systems, built to conduct computational work the way copper conducts current.

**Pronunciation:** KOH-breh (Portuguese), KOH-bray (English approximation)

**Tagline options** (pick one, or rotate by context):
- `Power systems in Rust` — direct, for technical audiences
- `Open infrastructure for energy planning` — for institutional/regulatory audiences
- `The conductor for power system computation` — plays on both meanings

**Namespace:**
- GitHub org: `cobre-rs` (avoids collision with any existing `cobre` orgs; the `-rs` suffix is standard in the Rust ecosystem — see `image-rs`, `tokio-rs`, `compio-rs`)
- Crates.io prefix: `cobre-` (e.g., `cobre-core`, `cobre-sddp`, `cobre-io`)
- Domain candidates: `cobre.energy`, `cobre-rs.org`, `cobresys.dev`
- PyPI (future Python bindings): `cobre` or `pycobre`

> **Note on `copper-rs`:** The `copper-project/copper-rs` GitHub org (1K+ stars) is a robotics OS in Rust. "Cobre" is distinct enough in branding and domain, but using `cobre-rs` rather than `copper-rs` avoids any confusion. The Portuguese name is a feature, not a risk — it signals the project's Brazilian origin.

---

## 2. Visual Identity

### 2.1 Logo Concept

The icon represents **three-phase power conductors** (busbars) connected by a vertical busbar with node points (buses) — a direct visual metaphor for power system topology. A small spark accent adds energy.

Three versions are provided:
- **Dark background** — primary use (README headers, docs, dark-mode UIs)
- **Light background** — for light-mode contexts
- **Icon only** — for GitHub org avatar, favicons, small contexts

The icon is intentionally geometric and minimal — it should read clearly at 16×16 favicon size.

### 2.2 Color Palette

#### Primary Colors
| Name | Hex | Usage |
|------|-----|-------|
| Copper | `#B87333` | Brand primary, icon fills, accent borders |
| Copper Light | `#D4956A` | Gradient highlights, hover states |
| Copper Dark | `#8B5E3C` | Gradient depth, pressed states |
| Patina | `#4A8B6F` | Secondary accent (copper patina/oxidation), success states |

#### Accent Colors
| Name | Hex | Usage |
|------|-----|-------|
| Spark Amber | `#F5A623` | Warnings, energy/active indicators, spark accent in logo |
| Signal Red | `#DC4C4C` | Errors, critical alerts, deficit indicators |
| Flow Blue | `#4A90B8` | Links, informational states, water/hydro associations |

#### Neutral Colors (Dark theme — primary)
| Name | Hex | Usage |
|------|-----|-------|
| Midnight | `#0F1419` | Page/app background |
| Surface | `#1A2028` | Cards, elevated surfaces |
| Border | `#2D3440` | Dividers, subtle borders |
| Muted | `#8B9298` | Secondary text, captions |
| Body | `#C8C6C2` | Primary body text |
| Bright | `#E8E6E3` | Headings, emphasis text |

#### Neutral Colors (Light theme — secondary)
| Name | Hex | Usage |
|------|-----|-------|
| White | `#FAFAF8` | Page background |
| Surface | `#F0EDE8` | Cards |
| Border | `#D4D0CA` | Dividers |
| Muted | `#6B7280` | Secondary text |
| Body | `#374151` | Body text |
| Dark | `#1A2028` | Headings |

### 2.3 Typography

#### Code & CLI (monospace)
- **Primary:** JetBrains Mono
- **Fallback:** `'Fira Code', 'SF Mono', 'Cascadia Code', monospace`
- Used in: terminal output, code examples, crate names, CLI branding

#### Documentation & UI (sans-serif)
- **Headings:** IBM Plex Sans (Medium/Bold)
- **Body:** IBM Plex Sans (Regular)
- **Fallback:** `'Inter', 'Segoe UI', system-ui, sans-serif`
- Rationale: IBM Plex has excellent multilingual support (including Portuguese accented characters), technical/industrial feel without being cold, open-source (SIL OFL)

#### Why not Inter?
Inter is fine but overused in developer tooling. IBM Plex has a more distinctive character that matches the industrial/engineering positioning.

### 2.4 Design Principles

1. **Technical, not trendy.** No gradients for decoration. No rounded-everything aesthetic. Clean lines, grid alignment, purposeful whitespace.
2. **Data-dense when needed.** Power system engineers work with tables of numbers. Don't hide density behind "clean" design — make density legible.
3. **Copper warmth.** The warm palette (copper/amber) differentiates Cobre from the sea of blue developer tools. Use it with restraint — copper accents on neutral backgrounds.
4. **Dark-first.** Engineers and researchers often work long hours. Dark theme is the default, light theme is the alternative.

---

## 3. Voice & Tone

### Writing style for docs, README, blog:
- **Direct.** No marketing fluff. Say what it does, how, and why.
- **Technical-first.** Assume the reader knows what SDDP, power flow, and LP solvers are. Provide context for Rust-specific concepts (not everyone in power systems knows Rust).
- **Bilingual awareness.** Primary docs in English, but key terms in Portuguese should be preserved where they're domain-standard (e.g., "subsistema" when discussing the Brazilian grid structure, "usina" for plants).
- **Honest about maturity.** Mark crates clearly as `experimental`, `alpha`, `beta`, or `stable`. Don't oversell.

### Crate status badges:
```
[experimental] — API will change, not for production
[alpha]        — Core functionality works, gaps remain
[beta]         — Feature-complete, seeking feedback
[stable]       — Production-ready, semver guarantees
```

---

## 4. Application

### GitHub org profile (cobre-rs)
- Avatar: `cobre-icon.svg` (dark background version)
- Bio: "Open infrastructure for power system computation. Built in Rust."
- URL: link to docs site or ecosystem README

### README header pattern
```markdown
<p align="center">
  <img src="assets/cobre-logo-dark.svg" width="360" alt="Cobre — Power Systems in Rust"/>
</p>

<p align="center">
  <strong>Open infrastructure for power system computation</strong>
</p>
```

### Crate README header pattern
```markdown
# cobre-sddp

[![crates.io](https://img.shields.io/crates/v/cobre-sddp.svg)](https://crates.io/crates/cobre-sddp)
[![docs.rs](https://docs.rs/cobre-sddp/badge.svg)](https://docs.rs/cobre-sddp)
[![status: alpha](https://img.shields.io/badge/status-alpha-F5A623.svg)]()

Stochastic Dual Dynamic Programming solver for hydrothermal dispatch.
Part of the [Cobre](https://github.com/cobre-rs) ecosystem.
```

### Custom badge colors (using brand palette):
- Status experimental: `#DC4C4C` (red)
- Status alpha: `#F5A623` (amber)
- Status beta: `#4A90B8` (blue)
- Status stable: `#4A8B6F` (patina green)
