# Prior register — cobre-solver + cobre-comm (baseline `077dbe2c`)

What the register and the committed mirror already say about these two crates, so the
four attacker passes receive a real do-not-re-raise list instead of re-discovering
ratified work. Every line number below was re-resolved against the baseline blob
(`git show 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c:<path>`), never a remembered one; where
the ticket text carried a line from the superseded pin `a136840d`, the moved line is
noted. Each entry carries a `reraiseKey` token list the attacker screen and
`check-reraise.py` match candidate titles and anchors against.

Four sections, four dispositions: an **ingest filter** clears a matching candidate as
`sanctioned` before the defender runs and assigns no id; a **dup-of handoff** leaves the
station as a typed block for another epic with no new id; **intended behaviour** is cited
to dismiss and never proposed as a fix-shape; a **seeded candidate** is handed to the
attackers as a starting observation they must sharpen rather than re-derive.

---

## Ingest filters (sanctioned)

### Shared-memory communicator trait hierarchy — CLEAR AT INGEST

- **Ratified**: `docs/design/reserved-seams-and-deferred-debt.md:79` ("Shared-memory
  communicator trait hierarchy (`cobre-comm`, `shared-memory` feature)"); BACKLOG `OD-001`
  (Sev B, speculative-generality, `plans/architecture-debt-audit/BACKLOG.md:1770`) recorded
  **KEEP-RESERVED**, owner-ratified 2026-08-22 (`BACKLOG.md:1900`), with the activating
  milestone "intra-node scenario-library sharing AND intra-node cut-archive sharing"; the
  mirror entry was added under that ruling (`BACKLOG.md:1989`).
- **Symbols covered** (all at baseline): `SharedMemoryProvider`
  (`crates/cobre-comm/src/traits.rs:372`), `SharedRegion<T>` (`traits.rs:320`),
  `LocalCommunicator` (`traits.rs:254`), `LocalCommKind` (`traits.rs:275`),
  `HeapRegion<T>` (`crates/cobre-comm/src/local.rs:155`), `FerrompiLocalComm`
  (`crates/cobre-comm/src/ferrompi.rs:207`), and `FerrompiBackend::split_local`
  (`ferrompi.rs:264`, inside `impl crate::SharedMemoryProvider for FerrompiBackend` at
  `ferrompi.rs:239`).
- **Still true at baseline**: both backends resolve `Region<T>` to the same heap-backed
  `HeapRegion<T>` (`crates/cobre-comm/Cargo.toml:23-26` says so in the feature comment), and
  `git grep -E 'SharedMemoryProvider|split_local|create_shared_region|SharedRegion|LocalCommunicator|HeapRegion' 077dbe2c -- crates`
  returns nothing outside `crates/cobre-comm/` — no consumer anywhere else in the workspace.
- **Filter behaviour**: a candidate proposing removal, feature-gating, or collapse of any
  symbol above is Cleared as `sanctioned` with this citation **before** the defender runs.
  No OD id is assigned. Re-raising it is a `check-reraise.py` failure, not a defender
  argument. A candidate about the hierarchy's *implementation* (a defect inside
  `HeapRegion`, an unsound `unsafe impl`) is not filtered — only the existence-of-the-seam
  claim is.
- **reraiseKey**: `shared-memory`, `SharedMemoryProvider`, `SharedRegion`,
  `LocalCommunicator`, `LocalCommKind`, `HeapRegion`, `split_local`, `speculative-generality
  cobre-comm`, `OD-001`

---

## Dup-of handoffs

### Superseded cut-sync public methods — HAND TO E5, NO NEW ID

- **Registered**: mirror `docs/design/reserved-seams-and-deferred-debt.md:334` ("Superseded
  cut-sync public methods"; the ticket text's `:311` is the superseded pin's line — the
  mirror gained Fixed H3s above it); BACKLOG `CD-019` (Sev C, dead-code / retrofit-remnant,
  `BACKLOG.md:668`), deferred to the next licensed public-API break.
- **Anchors VERIFIED at baseline, and they are not ours**: `sync_cuts`
  (`crates/cobre-sddp/src/cut/cut_sync.rs:243`), `pack_local_records` (`:400`),
  `sync_packed_records` (`:495`) — superseded by the live `sync_level_records` (`:581`).
  `git grep -n 'pub fn sync_cuts\|pub fn pack_local_records\|pub fn sync_packed_records' 077dbe2c -- crates/cobre-solver crates/cobre-comm`
  returns nothing: none of the three resolves in either station crate. (The `CD-019` entry
  body quotes `:250`/`:474`/`:598` from its own, older baseline; the methods are the same.)
- **Disposition**: `dup-of` the registered entry, emitted in `handoffs.json` as the `E5`
  block. A `cobre-sddp` anchor is legal in this station **only** on this handoff; any other
  `cobre-sddp` anchor in a candidate is out-of-station.
- **reraiseKey**: `sync_cuts`, `pack_local_records`, `sync_packed_records`, `cut-sync`,
  `cut_sync`, `single-pool`, `CD-019`

---

## Intended behaviour, never a finding

### HiGHS-loud / CLP-silent basis validation

- **HiGHS rejects loudly**: `HighsSolver::solve` returns `SolverError::BasisInconsistent`
  when `isBasisConsistent` rejects the offered basis
  (`crates/cobre-solver/src/backends/highs/interface.rs:487`; documented at `:408`).
- **CLP accepts silently**: `ClpSolver::install_basis`
  (`crates/cobre-solver/src/backends/clp/solver.rs:213`) documents at `:214-216` that
  "CLP's per-element setters silently accept an inconsistent offered basis and `Clp_dual`
  repairs it, so — unlike `HighsSolver::solve` — there is no consistency check and no
  `SolverError::BasisInconsistent` surface here"; it rejects only an undersized row basis via
  `SolverError::BasisRowCountMismatch` (`solver.rs:228`, documented at `:204`).
- **The trait contract already scopes this**: `crates/cobre-solver/src/trait_def.rs:105`
  ("`isBasisConsistent` returns `SolverError::BasisInconsistent`") and `:136-138` (the
  `# Errors` list on `solve`: `BasisInconsistent` ONLY when the backend validates,
  `BasisRowCountMismatch` symmetric on both).
- **Pinned by tests at baseline**: `crates/cobre-solver/tests/conformance.rs`
  `test_solver_highs_solve_rejects_inconsistent_basis_status_combination` (`:1672`),
  `test_solver_clp_solve_accepts_inconsistent_basis_status_combination_silently` (`:1724`),
  and the symmetric pair `test_solver_{highs,clp}_solve_rejects_undersized_row_basis`
  (`:1601`, `:1636`). The SDDP contract that depends on the asymmetry is
  `.claude/rules/sddp.md` § "A stored basis warm-starts only at its own node": the node-tag
  filter in cobre's own apply path is the sole defence precisely because CLP does not
  backstop it.
- **Instruction**: cite as intended backend behaviour when dismissing. Equalizing the two
  backends — adding a consistency check to CLP, or removing HiGHS's — must never be proposed
  as a fix-shape; the asymmetry is a property of the two solver libraries, not of cobre.
- **Doc drift noted, not a station finding**: `.claude/rules/sddp.md:446` still reads "The
  CLP/HiGHS basis-validation asymmetry itself is unpinned by any test" while the four
  conformance tests above pin it at this baseline. The sentence is stale documentation
  owned by the build-ci/docs station (E07), recorded here so no solver attacker re-derives
  it as a test-coverage gap.
- **reraiseKey**: `BasisInconsistent`, `BasisRowCountMismatch`, `isBasisConsistent`,
  `basis validation`, `basis-validation`, `Clp_dual`, `install_basis`, `backend asymmetry`,
  `equalize`

---

## Seeded candidates

### Hand-replicated per-crate `[lints]` tables — SEEDED OD CANDIDATE

- **Observation**: the workspace lint tables `Cargo.toml:35` (`[workspace.lints.rust]`, 2
  entries) and `Cargo.toml:65` (`[workspace.lints.clippy]`, 6 entries) are typed out again
  in both station manifests: `crates/cobre-solver/Cargo.toml:37-47` (`[lints.rust]` at
  `:37`, `[lints.clippy]` at `:41`) and `crates/cobre-comm/Cargo.toml:35-45` (`:35`, `:39`),
  each flipping only `unsafe_code = "forbid"` to `"allow"`.
- **Why Cargo forces it** (the manifest comment, `crates/cobre-solver/Cargo.toml:33-36`;
  the same wording at `crates/cobre-comm/Cargo.toml:29-34`): "All other workspace lints are
  manually replicated here since Cargo does not permit combining `lints.workspace = true`
  with per-lint overrides in the same crate manifest."
- **Drift evidence** (corroboration, another station's anchor): the third replica,
  `crates/cobre-python/Cargo.toml:46-55` (`[lints.rust]` at `:46`, `[lints.clippy]` at `:50`;
  the ticket text's `48-53` is the superseded pin's span), carries only 5 of the 6 clippy
  entries — `too_many_arguments = "deny"` is missing. That instance is the cli-python
  station's (E06) to raise; it is cited here only as proof the hand-copy already drifted.
- **What the attackers must add**: a fix-shape that survives the Cargo constraint (a
  `[workspace.lints]` layout that lets a crate override one lint, a build-time table
  equality check, or an accepted "three copies, one guard" verdict) and the severity
  argument. Restating the observation alone is not a finding.
- **reraiseKey**: `lints`, `lints.rust`, `lints.clippy`, `lints.workspace`, `unsafe_code`,
  `too_many_arguments`, `manifest duplication`, `Cargo.toml duplication`

---

## Do not re-raise

- shared-memory communicator trait hierarchy as dead code or speculative generality
  (sanctioned, OD-001 KEEP-RESERVED, mirror `:79`)
- `sync_cuts` / `pack_local_records` / `sync_packed_records` as dead public API (CD-019;
  anchors in `cobre-sddp`, handed to E5)
- HiGHS-vs-CLP basis-validation divergence as a defect or a test-coverage gap (intended
  behaviour, pinned by `conformance.rs`)
- the per-crate `[lints]` tables as a bare "duplication" observation (seeded; the attackers
  owe the fix-shape and severity, not the observation)
