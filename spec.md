# Documentation Overhaul — Implementation Spec

**Target:** cobre v0.3.2 repository at `github.com/cobre-rs/cobre`
**Scope:** Book restructure, README enrichment, broken link fixes, version
updates, CI enforcement, content updates.

Read this entire document before starting. Execute sections in order.

---

## 1. Delete Files

Remove these 4 files from the repository:

- `book/src/reference/roadmap.md` — duplicated changelog, 258 lines
- `book/src/tutorial/next-steps.md` — dead link page, no content value
- `book/src/tutorial/installation.md` — duplicate of `guide/installation.md`
- `book/src/guide/getting-started.md` — wrapper page with single child

---

## 2. Rewrite `book/src/SUMMARY.md`

Replace the entire file. The new structure collapses Tutorial + User Guide into
Getting Started + Guide, renames "Crate Documentation" to "For Developers", and
moves Contributing under For Developers.

```markdown
# Summary

[Introduction](./introduction.md)

---

# Getting Started

- [What Cobre Solves](./tutorial/what-cobre-solves.md)
- [Installation](./guide/installation.md)
- [Quickstart](./tutorial/quickstart.md)
- [Python Quickstart](./guide/python-quickstart.md)

---

# Guide

- [Anatomy of a Case](./tutorial/anatomy-of-a-case.md)
- [Building a Case](./tutorial/building-a-system.md)
- [System Modeling](./guide/system-modeling.md)
  - [Hydro Plants](./guide/hydro-plants.md)
  - [Thermal Units](./guide/thermal-units.md)
  - [Network Topology](./guide/network-topology.md)
  - [Stochastic Modeling](./guide/stochastic-modeling.md)
- [Configuration](./guide/configuration.md)
- [Running Studies](./guide/running-studies.md)
- [Policy Management](./guide/policy-management.md)
- [Case Conversion (cobre-bridge)](./guide/cobre-bridge.md)
- [Understanding Results](./tutorial/understanding-results.md)
- [Convergence & Diagnostics](./guide/interpreting-results.md)
- [CLI Reference](./guide/cli-reference.md)

---

# Examples

- [1dtoy](./examples/1dtoy.md)
- [4ree](./examples/4ree.md)
- [Creating Your Own Case](./examples/creating-your-own.md)

---

# Reference

- [Case Directory Format](./reference/case-format.md)
- [Output Format](./reference/output-format.md)
- [Error Codes](./reference/error-codes.md)
- [Schemas](./reference/schemas.md)

---

# For Developers

- [Crate Overview](./crates/overview.md)
- [cobre-core](./crates/core.md)
- [cobre-io](./crates/io.md)
- [cobre-stochastic](./crates/stochastic.md)
- [cobre-solver](./crates/solver.md)
- [cobre-comm](./crates/comm.md)
- [cobre-sddp](./crates/sddp.md)
- [cobre-cli](./crates/cli.md)
- [ferrompi](./crates/ferrompi.md)
- [Contributing](./contributing.md)
```

---

## 3. Update `book/src/introduction.md`

Add a fourth audience path for users who are ready to run immediately:

```markdown
> **Ready to run?**
> Jump straight to [Installation](./guide/installation.md) and
> [Quickstart](./tutorial/quickstart.md).
```

Insert this after the "Python user?" blockquote.

---

## 4. Fix Stale Version References

Replace `v0.3.1` with `v0.3.2` in these locations:

- `book/src/tutorial/quickstart.md` — two occurrences of `COBRE v0.3.1` in
  banner output blocks
- `book/src/guide/installation.md` — one occurrence of `cobre   v0.3.1` in
  version output block
- `book/src/guide/cli-reference.md` — one occurrence of `cobre   v0.3.1` in
  version output block

Also in `README.md`, replace the sentence:
> "Cobre v0.3.1 is alpha software..."

with:
> "Cobre v0.3.2 is alpha software..."

And in the same sentence, replace the roadmap link:
> "See the [roadmap](https://cobre-rs.github.io/cobre/reference/roadmap.html) for what's next."

with:
> "See the [CHANGELOG](CHANGELOG.md) for release history."

---

## 5. Fix Broken Links

### 5.1 `book/src/reference/error-codes.md`

Find the line containing `../overview/roadmap.md` and replace with:
```
Check the [CHANGELOG](https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md) for the implementation timeline.
```

### 5.2 `book/src/crates/solver.md`

There are 8 links pointing to `../api/cobre_solver/...` (a path that doesn't
exist). Replace all occurrences of `../api/cobre_solver/` with
`https://docs.rs/cobre-solver/latest/cobre_solver/`.

### 5.3 `book/src/tutorial/quickstart.md`

The link `[Installation](./installation.md)` now points to the deleted
tutorial installation page. Replace with `[Installation](../guide/installation.md)`.

---

## 6. Fix Prose References to Old Structure

### 6.1 `book/src/guide/interpreting-results.md`

Change the title from `# Interpreting Results` to `# Convergence & Diagnostics`.

Replace the opening paragraph:
> "The [Understanding Results](../tutorial/understanding-results.md) tutorial
> explains what each output file contains and how to read it. This page goes
> one level deeper: it provides practical analysis patterns for answering
> domain questions from the data. It assumes you have already completed the
> tutorial and are comfortable loading Parquet files in your preferred tool."

with:
> "[Understanding Results](../tutorial/understanding-results.md) explains what
> each output file contains and how to read it. This page goes one level deeper:
> it provides practical analysis patterns for answering domain questions from the
> data. It assumes you are comfortable loading Parquet files in your preferred
> tool."

### 6.2 `book/src/tutorial/understanding-results.md`

Replace the "What's Next" section at the bottom of the file. The current
section references the deleted `next-steps.md`. Replace the entire section
(heading + paragraph + bullet list) with:

```markdown
## See Also

- [Convergence & Diagnostics](../guide/interpreting-results.md) — advanced analysis patterns and convergence assessment
- [CLI Reference](../guide/cli-reference.md) — all flags, subcommands, and exit codes
- [Configuration](../guide/configuration.md) — every `config.json` field documented
```

### 6.3 `book/src/guide/installation.md`

In the "Next Steps" section at the bottom, remove the line:
```
- [Case Directory Format](../reference/case-format.md) — how to structure input data
```
(The remaining 3 links are sufficient and more actionable.)

---

## 7. Create `book/src/guide/policy-management.md`

Create this new file. It documents the v0.3.0 policy warm-start/resume feature.
The page should cover:

- **Three policy modes** (`fresh`, `warm_start`, `resume`) with a description
  of when to use each and a `config.json` snippet for each
- **Simulation-only mode** — `training.enabled = false` with policy loading
- **Checkpointing configuration** — `policy.checkpointing` fields table
  (`enabled`, `initial_iteration`, `interval_iterations`, `store_basis`,
  `compress`)
- **Checkpoint directory contents** — the `policy/` directory structure
  showing `cuts/`, `basis/`, `states/`, and `metadata.json`
- **See Also** links to Configuration, Running Studies, and Output Format

The fresh mode is default and the simplest. Warm-start loads prior cuts but
resets the iteration counter. Resume restores the full FCF state including
iteration counter. All three should have code-fence JSON examples.

---

## 8. Update `book/src/guide/running-studies.md`

In the "Common Workflows" section, add a new subsection between "Training
Only" and "Quiet Mode for Scripts":

```markdown
### Simulation Against a Saved Policy

To evaluate a previously trained policy without re-training:

\```json
{
  "training": { "enabled": false },
  "policy": { "mode": "warm_start", "path": "./policy" }
}
\```

Cobre loads the policy cuts, skips training entirely, and runs simulation.
See [Policy Management](./policy-management.md) for details on warm-start
and resume modes.
```

---

## 9. Update `book/src/guide/hydro-plants.md` Penalties Table

The penalties section shows the old field set. Update the JSON example and
table to include these v0.3.x fields:

**New JSON fields to add to the example block:**
```json
"water_withdrawal_violation_pos_cost": 1200.0,
"water_withdrawal_violation_neg_cost": 800.0,
"evaporation_violation_pos_cost": 5000.0,
"evaporation_violation_neg_cost": 5000.0,
"inflow_nonnegativity_cost": 1000.0
```

**New table rows to add:**

| Field | Unit | Description |
|-------|------|-------------|
| `evaporation_violation_pos_cost` | $/mm | Over-evaporation violation penalty. Overrides `evaporation_violation_cost` for the positive direction. |
| `evaporation_violation_neg_cost` | $/mm | Under-evaporation violation penalty. Overrides `evaporation_violation_cost` for the negative direction. |
| `water_withdrawal_violation_pos_cost` | $/m³/s | Over-withdrawal violation penalty. Overrides `water_withdrawal_violation_cost` for the positive direction. |
| `water_withdrawal_violation_neg_cost` | $/m³/s | Under-withdrawal violation penalty. Overrides `water_withdrawal_violation_cost` for the negative direction. |
| `inflow_nonnegativity_cost` | $/m³/s | Per-plant override for the global inflow non-negativity penalty. Only active when `modeling.inflow_non_negativity.method` is `"penalty"` or `"truncation_with_penalty"`. |

**Update the existing table rows** for `evaporation_violation_cost` and
`water_withdrawal_violation_cost` to clarify they are symmetric defaults:
- `evaporation_violation_cost`: "Symmetric evaporation violation penalty. Applies to both directions unless overridden by directional fields."
- `water_withdrawal_violation_cost`: "Symmetric water withdrawal violation penalty. Applies to both directions unless overridden by directional fields."

**Update existing rows** for `turbined_violation_below_cost`,
`outflow_violation_below_cost`, `outflow_violation_above_cost`, and
`generation_violation_below_cost` to add "Applied per block." to each
description.

**Add a paragraph** after the table explaining the directional override
mechanism and the per-block behavior.

**Update the cascade description:** change "all 11 hydro penalty fields" to
remove the hardcoded count (the number is now 16).

---

## 10. Update `book/src/crates/overview.md`

Change "over 3,000 tests" to "over 3,100 tests".

---

## 11. Enrich All Subcrate READMEs

Replace the contents of each subcrate README. Every README should follow this
template (adapted per crate):

1. **Title** — crate name as `# heading`
2. **One-paragraph description** — what the crate does, accurately
3. **"When to Use" section** — 2-3 sentences on when to depend on this crate directly
4. **"Key Types" section** — 4-6 most important types/functions in bold with one-line descriptions
5. **"Links" table** — Book chapter, API Docs (docs.rs), Repository, CHANGELOG
6. **"Status"** — "Alpha — API is functional but not yet stable."
7. **"License"** — Apache-2.0

**Specific accuracy requirements per crate:**

- **`cobre-io`**: Do NOT claim PSS/E or CSV support. The crate reads JSON +
  Parquet, writes Parquet + FlatBuffers + JSON.
- **`cobre-comm`**: Do NOT claim TCP or shared memory backends. Only MPI (via
  ferrompi) and LocalBackend exist.
- **`cobre-mcp`** and **`cobre-tui`**: Status should say "Not yet implemented —
  this crate is a stub." instead of "Alpha".
- **`cobre-sddp`**: Mention cut selection strategies (Level-1, LML1, dominated),
  policy warm-start/resume, and discount rate support.
- **`cobre-solver`**: Mention the 12-level retry escalation for numerically
  difficult LPs.
- **`cobre` (umbrella)**: Keep the crate table pointing to crates.io. Remove
  the "full roadmap" sentence.

---

## 12. Fix Python README (`crates/cobre-python/README.md`)

### 12.1 Fix `load_policy` signature

The Quick Start example shows:
```python
policy = cobre.results.load_policy("output/policy.fcf")
```
The function takes a directory, not a file. Replace with:
```python
policy = cobre.results.load_policy("output/")
```

### 12.2 Fix `load_simulation` and `load_policy` usage

The Quick Start shows attribute access that doesn't exist:
```python
simulation = cobre.results.load_simulation("output/")
print(f"Stages: {simulation.n_stages}, Scenarios: {simulation.n_scenarios}")

policy = cobre.results.load_policy("output/")
print(f"Policy cuts: {policy.n_cuts}")
```

Both functions return dicts. Replace with correct examples:
```python
result = cobre.run.run("path/to/case", output_dir="output/")
print(f"Converged: {result['converged']}, LB: {result['lower_bound']:.2f}")

convergence = cobre.results.load_convergence("output/")
print(f"Iterations: {len(convergence)}")

simulation = cobre.results.load_simulation("output/")
print(f"Cost records: {len(simulation['costs'])}")

policy = cobre.results.load_policy("output/")
print(f"Iterations completed: {policy['metadata']['completed_iterations']}")
```

---

## 13. Update `lib.rs` and `main.rs` Doc Comments

In all 10 crate root files, replace:
> "See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap."

with:
> "See the [repository](https://github.com/cobre-rs/cobre) for the current status."

Files:
- `crates/cobre/src/lib.rs`
- `crates/cobre-core/src/lib.rs`
- `crates/cobre-sddp/src/lib.rs`
- `crates/cobre-python/src/lib.rs`
- `crates/cobre-stochastic/src/lib.rs`
- `crates/cobre-io/src/lib.rs`
- `crates/cobre-solver/src/lib.rs`
- `crates/cobre-comm/src/lib.rs`
- `crates/cobre-tui/src/lib.rs`
- `crates/cobre-mcp/src/main.rs`

---

## 14. Update `CONTRIBUTING.md`

### 14.1 Project structure

Change `cobre-comm` description from:
> "Communication abstraction (MPI, TCP, shm, local)"

to:
> "Communication abstraction (MPI, local)"

### 14.2 cobre-comm coding guidelines

Change:
> "The `Communicator` trait must remain implementable by all four backends (MPI, TCP, shared-memory, local)."

to:
> "The `Communicator` trait must remain implementable by both backends (MPI and local)."

### 14.3 Documentation contribution guidance

Replace the "Improving Documentation" paragraph to mention the book lives in
this repo (not just in cobre-docs). Add `mdbook serve book --open` as the
local preview command.

### 14.4 Release checklist

Add the new quality scripts to the release checklist. The checklist should
include running all four scripts:
```bash
python3 scripts/check_claudemd_version.py
python3 scripts/check_book_version.py
python3 scripts/check_python_parity.py --max 0
python3 scripts/check_suppressions.py --max 10
```

---

## 15. Create `scripts/check_book_version.py`

Create a Python script that:
1. Reads the workspace version from `Cargo.toml`
2. Scans all `.md` files in `book/src/` for patterns like `COBRE vX.Y.Z` and
   `cobre   vX.Y.Z` (version banners and command output)
3. Reports any that don't match the Cargo.toml version
4. Exits 0 if all match, 1 if any mismatch

Make the script executable. Model it after the existing
`scripts/check_claudemd_version.py`.

---

## 16. Add Book Version Check to CI

In `.github/workflows/ci.yml`, add a step to the `quality-scripts` job:

```yaml
      - name: Check book version references
        run: python3 scripts/check_book_version.py
```

Place it after the existing `Check CLAUDE.md version` step.

---

## 17. Verification

After all changes, verify:

1. `python3 scripts/check_book_version.py` passes
2. `python3 scripts/check_claudemd_version.py` passes
3. `python3 scripts/check_python_parity.py --max 0` passes
4. `python3 scripts/check_suppressions.py --max 10` passes
5. No broken internal links in the book (check all `](./` and `](../`
   relative links resolve to existing files)
6. All entries in `SUMMARY.md` point to existing files
7. No remaining references to `roadmap.md`, `next-steps.md`,
   `tutorial/installation.md`, or `guide/getting-started.md`
8. No remaining occurrences of `v0.3.1` in the book (outside of CHANGELOG
   historical entries)
9. No occurrences of `PSS/E`, `TCP,`, or `shared-memory` in subcrate READMEs
   or CONTRIBUTING.md project structure
10. `grep -r "full roadmap" crates/` returns no results