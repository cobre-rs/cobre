# Prior register — test corpus and test-support (baseline `077dbe2c`)

What the committed ID-free mirror `docs/design/reserved-seams-and-deferred-debt.md` already says about the test corpus, so the attacker and calibration tickets merge into the three existing items instead of minting fresh TD ids. Read-only: no repo file is written and no fix is proposed. Mirror line numbers are the pin's (`git show 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c:docs/design/reserved-seams-and-deferred-debt.md`); the HEAD copy differs only in register-reconciliation sections appended after the pin, so both H3 items sit at the same lines and the bullet moved from :711 to :717 with its text unchanged. Every tree anchor below is re-verified at the pin in the `Anchor verification` section; an anchor that fails there is marked `anchor-missing` and its entry leaves the dup-of set — the anchor is never edited into something that resolves.

## Mirror items (transcribed, then sharpened)

### Oracle test-harness duplication

**Mirror anchor.** `docs/design/reserved-seams-and-deferred-debt.md:308` → `## Deferred-debt register` (`docs/design/reserved-seams-and-deferred-debt.md:240`) → `### Oracle test-harness duplication`

**Claim as written.** "`crates/cobre-sddp/tests/extensive_form_oracle.rs` carries a verbatim copy of the comparison harness in `crates/cobre-sddp/tests/branching_value_oracle.rs` (the `close` tolerance helper and its scaffolding) and pays a second static link of the solver, against the test cost-discipline." The mirror records two resolution options for an owner pick: promote the shared `close` / tolerance helpers into `cobre_sddp::test_support`, or fold both fixtures into one test binary.

**Baseline anchors.** `crates/cobre-sddp/tests/extensive_form_oracle.rs:77` (`fn close`), `crates/cobre-sddp/tests/branching_value_oracle.rs:93` (`fn close`), `crates/cobre-sddp/tests/node_native_backward_gate.rs:76` (`fn close`), `crates/cobre-sddp/tests/mpi_wire.rs:2816` (`fn close`, nested inside a `mod`, not file scope)

**Sharpening at baseline.** `git grep -n 'fn close' 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c -- crates/cobre-sddp/tests` returns four carriers, not the mirror's two: extensive_form_oracle.rs:77, branching_value_oracle.rs:93, node_native_backward_gate.rs:76 and mpi_wire.rs:2816 (full paths in the anchors line above). Definition: a carrier is a `crates/cobre-sddp/tests/**/*.rs` file declaring `fn close` at any nesting depth. The widening is a sharpening of the existing item — the same class with two more instances — not a new finding. The yardstick names a third destination the mirror does not (`docs/design/testing-architecture.md:436-438`, §5.2: hoist the `close` / tolerance helpers into `cobre-core`'s `test-support` surface), so the fix-shape destination is an owner pick among three; recorded for calibration, not adjudicated here. Each of the four carriers is its own solver-linking binary (`stations/test-corpus/inventory.json`, figure `int-binaries-solver-linking`), which is the cost the mirror already names.

**Register cross-references.** No CD/PD/OD/TD id owns this item; it is mirror-only. The stochastic station cross-referenced it three times without re-raising (`plans/architecture-debt-audit/BACKLOG.md` L3719, L3728, L3738; its prior-register disposition `cross-reference` at L3751; handoff note L3763), and the sddp station recorded it under positives as already-registered (L4355). Same-class, distinct-instance ids that are never merge targets: TD-030 (`crates/cobre-stochastic/tests/halton_integration.rs:207`, byte-identical fixture prelude across integration binaries) and TD-043 (`crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::penalties`, md5-identical helper across four binaries). `stations/test-corpus/inventory.json` → `harness.oracleDuplicationAnchors` holds the two mirror-named carriers at pin lines.

**Disposition.** `dup-of-likely` — the calibration ticket merges this into the existing mirror item (sharpened to four carriers, three destination options); do NOT issue a fresh TD id.

**Owner (from mirror).** The test-infrastructure owner. · **Trigger (from mirror).** The next consolidation of the branching oracle test suite.

### Python-binding Rust tests invisible to CI

**Mirror anchor.** `docs/design/reserved-seams-and-deferred-debt.md:347` → `## Deferred-debt register` (`docs/design/reserved-seams-and-deferred-debt.md:240`) → `### Python-binding Rust tests invisible to CI`

**Claim as written.** "The Python-binding crate is excluded from the workspace, and its CI job runs only the Python build plus pytest, so the crate's Rust `#[cfg(test)]` modules are never compiled or run in CI." The mirror's fix: either wire a `cargo check --tests` for the crate, or hoist a shared node-graph test-fixture builder into `cobre_sddp::test_support` so those Rust tests live in a CI-visible crate.

**Baseline anchors.** `Cargo.toml:21` (workspace `exclude`, `crates/cobre-python` listed at `Cargo.toml:22`), `.github/workflows/ci.yml:504` (the `python:` job), `.github/workflows/ci.yml:559` (`maturin develop --release`), `.github/workflows/ci.yml:562` (step "Run Rust tests for the bindings crate"; `cargo test --manifest-path crates/cobre-python/Cargo.toml --no-default-features --features highs` at `.github/workflows/ci.yml:567`, gated to one matrix cell by `if: matrix.python-version == '3.12'` at `.github/workflows/ci.yml:563`), `.github/workflows/ci.yml:572` (pytest), `crates/cobre-python/Cargo.toml:17` (`crate-type = ["cdylib"]`), `crates/cobre-python/Cargo.toml:34` (`[dev-dependencies]`, naming no `cobre-sddp` and no `test-support`)

**Sharpening at baseline.** The workspace exclusion holds (`Cargo.toml:21-22`). The premise "runs only the Python build plus pytest" does NOT hold at the pin: the job runs `cargo test --manifest-path crates/cobre-python/Cargo.toml` (`.github/workflows/ci.yml:562-568`), so the crate's `#[cfg(test)]` modules are compiled and run in CI. The cli-python station recorded this as supersession note SN-06 (`plans/architecture-debt-audit/BACKLOG.md` L5350: 19 tests — errors.rs 3, policy.rs 1, run.rs 13, schema.rs 2) and Cleared the re-raise (L5465, L5495). The residue that survives is narrower than the mirror's claim: (a) the Rust step runs in one matrix cell only (`.github/workflows/ci.yml:563`); (b) the `[dev-dependencies]` table (`crates/cobre-python/Cargo.toml:34`) names neither `cobre-sddp` nor its `test-support` feature, so §5.2's and §5.11's dev-dep bullet is uncarried — the yardstick's own Adoption paragraph says so (`docs/design/testing-architecture.md:425-431`); (c) clippy's `--workspace` runs exclude the crate (CD-103, build-ci station). The doctest half of the yardstick's §2.3 clause ("its Rust unit tests and doctests never compile in CI", `docs/design/testing-architecture.md:100-102`) is vacuous rather than a gap: the crate is `cdylib`-only (`crates/cobre-python/Cargo.toml:17`), which Cargo never builds a doctest target for, and all 13 of its rustdoc fences are tagged `python`, so no Rust doctest exists to run. Deviation from the ticket text: the ticket's step "record that its CI job runs maturin plus pytest only" describes the a136840d-era premise; at the register pin the job also runs the crate's Rust tests.

**Register cross-references.** No CD/PD/OD/TD id owns this item; it is mirror-only. SN-06 (`plans/architecture-debt-audit/BACKLOG.md` L5350) and the two Cleared lines (L5465, L5495) are the cli-python station's record of the supersession. Adjacent, not identical: CD-103 (`crates/cobre-python/Cargo.toml:50`, the crate's `[lints.clippy]` table evaluated by no CI step), TD-059 (`crates/cobre-python/src/run.rs::python_run_1dtoy_metadata_matches_cli_golden_values`, duplication inside the now-CI-visible inline module) and TD-071 (`crates/cobre-python/src/convert.rs::py_to_json_value`).

**Disposition.** `dup-of-likely` — the calibration ticket merges this into the existing mirror item as a supersession note (premise stale at the pin; residue (a)–(c) is the narrowed claim) for the E11 write-back; do NOT issue a fresh TD id.

**Owner (from mirror).** The build / CI owner. · **Trigger (from mirror).** Systemic — the next CI-configuration pass (a Rust test regression in that crate would otherwise ship unseen).

### Mega-file / inline-test-giant asymmetry

**Mirror anchor.** `docs/design/reserved-seams-and-deferred-debt.md:711` → `## Deferred-debt register — whole-lifecycle audit findings` (`docs/design/reserved-seams-and-deferred-debt.md:553`) → bullet `**Mega-file / inline-test-giant asymmetry.**` (HEAD: `docs/design/reserved-seams-and-deferred-debt.md:717`, same text)

**Claim as written.** "The workspace module is a flat mega-file holding many per-worker arena structs, and the LP-builder entries/columns modules carry giant inline test modules while their sibling builder submodules use extracted test files. Split each into a directory module / sibling test file matching the crate's prevailing convention."

**Baseline anchors.** `crates/cobre-sddp/src/lp/builder/entries.rs:1680` (first of four `#[cfg(test)]`: :1680, :1716, :2242, :3510), `crates/cobre-sddp/src/lp/builder/columns.rs:1305` (first of twelve `#[cfg(test)]`: :1305 … :9029), `crates/cobre-sddp/src/lp/builder/template.rs:1328` (`#[cfg(test)] mod tests;`), `crates/cobre-sddp/src/lp/builder/template/tests.rs` (the extracted sibling), `crates/cobre-sddp/src/lp/builder/layout.rs:1962` (`#[cfg(test)] mod tests;`), `crates/cobre-sddp/src/lp/builder/layout.rs:1965` (a residual inline `mod collapse_stage_level_tests`), `crates/cobre-sddp/src/lp/builder/layout/tests.rs` (the extracted sibling), `crates/cobre-sddp/src/workspace/workspace.rs` (the mega-file half, owned elsewhere — see below)

**Sharpening at baseline.** One directory, crates/cobre-sddp/src/lp/builder/ (full paths in the anchors line above), holds both conventions. Inline giants: entries.rs is 10,093 lines with its first `#[cfg(test)]` at :1680, so 8,413 lines (83%) are inline test code across four `#[cfg(test)]` modules — TD-047 records the largest, `pumping_water_tests`, spanning :3518-:10093; columns.rs is 9,319 lines with its first `#[cfg(test)]` at :1305, so 8,014 lines (86%) across twelve modules. Extracted siblings: template.rs is 1,329 lines and declares `mod tests;` at :1328-1329 into template/tests.rs (5,356 lines); layout.rs is 2,090 lines and declares `mod tests;` at :1962-1963 into layout/tests.rs (3,565 lines) while still keeping one inline module at :1965 — so layout.rs is mixed, not purely extracted, which the mirror's wording does not say. Definition: "inline test lines" is lines from the first `#[cfg(test)]` to end of file, valid because every `#[cfg(test)]` module in both files sits after that first attribute; "line counts" are `wc -l` over `git show <pin>:<path>`. The mirror's earlier figures (~7.7k / ~7.1k inline lines; template/tests.rs 4,700, layout/tests.rs 2,979 — BACKLOG L265-266) have grown at the pin; the register's Wave 7 disposition of CD-007 already re-measured entries.rs 10,093 / columns.rs 9,319 (L4223). The mega-file half of the bullet (workspace.rs) is CD-021's subject, sharpened at Wave 7 (L4222: `workspace/` is a directory module at the pin; the surviving claim is nine structs in one file) and owned by the sddp station — outside this station's scope. The inline-giant half is the yardstick's §3.2 item 4 (`docs/design/testing-architecture.md:146-149`) and the object of the §5.1 homing rule (`docs/design/testing-architecture.md:252-256`); TD-031 records that no ratified homing threshold exists at the baseline, so the asymmetry is a fact and the threshold is a target.

**Register cross-references.** CD-007 (`crates/cobre-sddp/src/lp/builder/entries.rs::fill_load_balance_entries`; Wave 7 disposition `sharpen`, L4223) owns the god-file reading of the same two files; TD-047 (`crates/cobre-sddp/src/lp/builder/entries.rs::pumping_water_tests`, L4857) owns the internal partition of the largest inline module; CD-021 (`crates/cobre-sddp/src/workspace/workspace.rs::SolverWorkspace`, L4222) owns the mega-file half. Same-class analogues in other crates, never merge targets: TD-008 (`crates/cobre-io/src/scenarios/estimation.rs::tests`, core-io homing) and TD-031 (`crates/cobre-stochastic/src/tree/generate.rs:341`, stochastic homing; "no RATIFIED homing threshold exists at baseline").

**Disposition.** `dup-of-likely` — the calibration ticket merges the inline-giant half into the existing mirror bullet (cross-referencing CD-007 and TD-047) and leaves the mega-file half with CD-021; do NOT issue a fresh TD id.

**Owner (from mirror).** The training owner. · **Trigger (from mirror).** Navigability-driven, low priority.

## Ticket-figure deviations at the pin (recorded, never spec edits)

The ticket's figures were minted at the superseded scaffold pin `a136840d`; every difference below is a deviation note, not an edit to the ticket or the mirror.

- Mirror line numbers: the ticket's 285 / 324 / 688 are the pin's 308 / 347 / 711 (HEAD 308 / 347 / 717); the two H2 registers sit at :240 and :553.
- The Python item's premise: the ticket says "its CI job runs maturin plus pytest only"; at the pin `.github/workflows/ci.yml:562-568` also runs `cargo test --manifest-path crates/cobre-python/Cargo.toml` (SN-06).
- `test-support` declarers: the ticket says six crates declare the feature; at the pin five do (`crates/cobre-core/Cargo.toml:30`, `crates/cobre-io/Cargo.toml:33`, `crates/cobre-sddp/Cargo.toml:55`, `crates/cobre-solver/Cargo.toml:31`, `crates/cobre-stochastic/Cargo.toml:26`) and `cobre-cli` only consumes it (`crates/cobre-cli/Cargo.toml:62`).
- layout.rs (`crates/cobre-sddp/src/lp/builder/layout.rs:1965`) is mixed — one inline module beside its extracted sibling — not purely extracted as the mirror and the ticket imply.
- `plans/` is tracked; the artifacts are committed, and no `.gitignore` change exists at this pin.

## dup-of set handed to the attacker ticket

Every entry whose anchors all resolve in the `Anchor verification` section below is in the set; an `anchor-missing` entry is out. The attacker screens its candidates against these three mirror items first and records a match as `dup-of-likely: <mirror heading>`, never as a fresh finding.

- `Oracle test-harness duplication` — carriers: the four `fn close` files above.
- `Python-binding Rust tests invisible to CI` — premise superseded (SN-06); residue (a)–(c).
- `Mega-file / inline-test-giant asymmetry` — the inline-giant half only (`lp/builder/{entries,columns}.rs`); the mega-file half is CD-021's.

## Anchor verification

Sweep at the pin `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` over every backticked `path[:line|::symbol]` anchor above (`git show <pin>:<path>`; symbols through `tools/check-anchors.py`'s `decl_pattern`; `plans/` paths excluded as station-internal). 54 resolve, 0 missing.

- ok             `docs/design/reserved-seams-and-deferred-debt.md`
- ok             `docs/design/reserved-seams-and-deferred-debt.md:308`
- ok             `docs/design/reserved-seams-and-deferred-debt.md:240`
- ok             `crates/cobre-sddp/tests/extensive_form_oracle.rs`
- ok             `crates/cobre-sddp/tests/branching_value_oracle.rs`
- ok             `crates/cobre-sddp/tests/extensive_form_oracle.rs:77`
- ok             `crates/cobre-sddp/tests/branching_value_oracle.rs:93`
- ok             `crates/cobre-sddp/tests/node_native_backward_gate.rs:76`
- ok             `crates/cobre-sddp/tests/mpi_wire.rs:2816`
- ok             `docs/design/testing-architecture.md:436-438`
- ok             `crates/cobre-stochastic/tests/halton_integration.rs:207`
- ok             `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::penalties`
- ok             `docs/design/reserved-seams-and-deferred-debt.md:347`
- ok             `Cargo.toml:21`
- ok             `Cargo.toml:22`
- ok             `.github/workflows/ci.yml:504`
- ok             `.github/workflows/ci.yml:559`
- ok             `.github/workflows/ci.yml:562`
- ok             `.github/workflows/ci.yml:567`
- ok             `.github/workflows/ci.yml:563`
- ok             `.github/workflows/ci.yml:572`
- ok             `crates/cobre-python/Cargo.toml:17`
- ok             `crates/cobre-python/Cargo.toml:34`
- ok             `Cargo.toml:21-22`
- ok             `.github/workflows/ci.yml:562-568`
- ok             `docs/design/testing-architecture.md:425-431`
- ok             `docs/design/testing-architecture.md:100-102`
- ok             `crates/cobre-python/Cargo.toml:50`
- ok             `crates/cobre-python/src/run.rs::python_run_1dtoy_metadata_matches_cli_golden_values`
- ok             `crates/cobre-python/src/convert.rs::py_to_json_value`
- ok             `docs/design/reserved-seams-and-deferred-debt.md:711`
- ok             `docs/design/reserved-seams-and-deferred-debt.md:553`
- ok             `docs/design/reserved-seams-and-deferred-debt.md:717`
- ok             `crates/cobre-sddp/src/lp/builder/entries.rs:1680`
- ok             `crates/cobre-sddp/src/lp/builder/columns.rs:1305`
- ok             `crates/cobre-sddp/src/lp/builder/template.rs:1328`
- ok             `crates/cobre-sddp/src/lp/builder/template/tests.rs`
- ok             `crates/cobre-sddp/src/lp/builder/layout.rs:1962`
- ok             `crates/cobre-sddp/src/lp/builder/layout.rs:1965`
- ok             `crates/cobre-sddp/src/lp/builder/layout/tests.rs`
- ok             `crates/cobre-sddp/src/workspace/workspace.rs`
- ok             `docs/design/testing-architecture.md:146-149`
- ok             `docs/design/testing-architecture.md:252-256`
- ok             `crates/cobre-sddp/src/lp/builder/entries.rs::fill_load_balance_entries`
- ok             `crates/cobre-sddp/src/lp/builder/entries.rs::pumping_water_tests`
- ok             `crates/cobre-sddp/src/workspace/workspace.rs::SolverWorkspace`
- ok             `crates/cobre-io/src/scenarios/estimation.rs::tests`
- ok             `crates/cobre-stochastic/src/tree/generate.rs:341`
- ok             `crates/cobre-core/Cargo.toml:30`
- ok             `crates/cobre-io/Cargo.toml:33`
- ok             `crates/cobre-sddp/Cargo.toml:55`
- ok             `crates/cobre-solver/Cargo.toml:31`
- ok             `crates/cobre-stochastic/Cargo.toml:26`
- ok             `crates/cobre-cli/Cargo.toml:62`

**Entry dispositions after the sweep.**

- `Oracle test-harness duplication` — all baseline anchors resolve; stays in the dup-of set.
- `Python-binding Rust tests invisible to CI` — all baseline anchors resolve; stays in the dup-of set.
- `Mega-file / inline-test-giant asymmetry` — all baseline anchors resolve; stays in the dup-of set.
