# Bridge Gap Assessment

> **Date:** 2026-03-21
> **Context:** cobre-bridge implementation identified two gaps in cobre that
> block NEWAVE conversion features. This document assesses each gap and
> recommends a resolution path.

---

## Gap 1: `line_exchange` Generic Constraint Term

### Problem

NEWAVE's `restricao-eletrica.csv` uses `ener_interc(from, to)` to represent
net exchange flow between two subsystems. Cobre's generic constraint system
currently supports `line_direct(id)` and `line_reverse(id)` as separate
expression terms, but has no single term for net flow (direct - reverse).

The bridge can technically work around this by emitting multi-term expressions
(`1.0 * line_direct(id) - 1.0 * line_reverse(id)`), but this pushes
canonical-direction bookkeeping into every bridge constraint, making the
mapping fragile and verbose.

### Current State

Cobre's `VariableRef` enum in `cobre-core/src/generic_constraint.rs` defines
19 variable types. The line-related types are:

- `LineDirect { line_id, block_id }` -- forward flow (source -> target)
- `LineReverse { line_id, block_id }` -- reverse flow (target -> source)

No "exchange" or "net flow" concept exists.

### Recommendation

**Add `LineExchange` as the 20th `VariableRef` variant.**

The resolver in `cobre-sddp/src/generic_constraints.rs` returns two LP column
entries: `(line_fwd_col, +1.0)` and `(line_rev_col, -1.0)`, representing net
flow in the source-to-target direction. The bridge then maps `ener_interc`
trivially:

- Canonical direction matches: `+1.0 * line_exchange(line_id)`
- Canonical direction reversed: `-1.0 * line_exchange(line_id)`

### Scope

| File                                    | Change                                                            |
| --------------------------------------- | ----------------------------------------------------------------- |
| `cobre-core/src/generic_constraint.rs`  | Add `LineExchange { line_id, block_id }` variant to `VariableRef` |
| `cobre-io/src/constraints/generic.rs`   | Add `"line_exchange"` string to expression parser                 |
| `cobre-sddp/src/generic_constraints.rs` | `resolve_variable_ref` returns 2 entries (fwd +1, rev -1)         |
| Schema regeneration                     | `cobre schema export` updates `generic_constraints.schema.json`   |

**Estimated size:** ~50 lines across 3 crates. No architectural changes.
No performance impact (resolution runs once at template build time).
Fully backward compatible (no existing variable types change).

---

## Gap 2: Per-Stage Productivity Override

### Problem

NEWAVE allows temporal overrides of tailrace/forebay elevations via MODIF.DAT,
which change the effective head drop at specific stages. This affects the
constant-productivity coefficient for those stages. The bridge currently
evaluates each hydro's productivity once and cannot express per-stage
variations.

### Current State

Cobre's `hydro_production_models.json` supports per-stage **model selection**
(constant-productivity vs FPHA) via `stage_ranges` and `seasonal` modes. The
resolution pipeline in `cobre-sddp/src/hydro_models.rs` calls
`resolve_stage_model()` for each (hydro, stage) pair.

However, the productivity value itself always comes from the entity's base
`productivity_mw_per_m3s` field in `hydros.json`. There is no mechanism to
override this value at specific stages.

### Options Considered

1. **Use `linearized_head` model for affected plants.** Cobre already supports
   this model, but the bridge would need to compute reference head parameters
   from NEWAVE's CFUGA/CMONT data. Significant bridge-side work for a problem
   that is fundamentally just a scalar override.

2. **Add `productivity_override` to the production model config.** A single
   `Option<f64>` field on `StageRange` and `SeasonConfig`. When present,
   `resolve_stage_model()` uses the override instead of the entity's base
   productivity. When absent, behavior is unchanged.

3. **Accept the approximation.** MODIF.DAT overrides affect a minority of
   plants. Viable short-term but will cause validation mismatches against
   NEWAVE reference results.

### Recommendation

**Option 2: Add `productivity_override: Option<f64>` to the production model
config.** This is the cleanest solution -- minimal cobre changes, no
bridge-side head computation, and enables exact NEWAVE reproduction.

### Scope

| File                                           | Change                                                                      |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| `cobre-io/src/extensions/production_models.rs` | Add `productivity_override: Option<f64>` to `StageRange` and `SeasonConfig` |
| `cobre-sddp/src/hydro_models.rs`               | `resolve_stage_model()` applies the override when `Some`                    |
| Schema regeneration                            | `cobre schema export` updates `production_models.schema.json`               |
| Validation                                     | Ensure override > 0 when present                                            |

**Estimated size:** ~50 lines across 2 crates. Fully backward compatible
(field is optional, defaults to `None`).

---

## Implementation Order

1. **`line_exchange` first** -- it unblocks the entire `restricao-eletrica.csv`
   conversion pipeline, which is the higher-value bridge feature.
2. **`productivity_override` second** -- independent, can be done in parallel.

Both fit within a single release cycle and require no architectural changes.

---

## Resolution Status

> **Date resolved:** 2026-03-21
> **Branch:** `feat/v0.2.0-validation-adoption`

Both gaps have been implemented:

- **Gap 1 (`line_exchange`):** `VariableRef::LineExchange` added as the 20th
  variant in `cobre-core`, `"line_exchange"` parser support in `cobre-io`, and
  resolver returning `[(fwd_col, +1.0), (rev_col, -1.0)]` in `cobre-sddp`.
  Referential validation ensures the referenced line ID exists.
- **Gap 2 (`productivity_override`):** `productivity_override: Option<f64>` added
  to `StageRange` and `SeasonConfig` in `cobre-io`, with validation rejecting
  non-positive values and FPHA stages. `resolve_stage_model` in `cobre-sddp`
  applies the override when present.

All workspace tests pass. Schemas, book pages, and case-format reference updated.
