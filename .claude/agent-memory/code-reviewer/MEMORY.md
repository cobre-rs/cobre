# Code Reviewer Memory

## Project: cobre (Rust SDDP solver ecosystem)

### Key architecture facts

- `System` struct has both `pub` fields AND `pub fn` accessor methods — this dual API is intentional
  (fields for `PartialEq`/`Debug`, methods for slice access). But public fields break encapsulation.
- `SystemBuilder::build()` sorts entities by `id.0`, then checks duplicates via `windows(2)`,
  then validates cross-refs. Error collection is non-short-circuiting per design (DEC-005 style).
- `HasId` trait is private; `check_duplicates` and `build_index` are private fns.
- `CascadeTopology::build()` uses Kahn's algorithm with a `BinaryHeap<Reverse<i32>>` as the min-heap
  ready queue. Push/pop are O(log n). Changed from a sorted `Vec<i32>` (which had O(n) `remove(0)`)
  in commit d5a8c3b. Determinism is preserved: min-heap pops smallest ID first within a topo level.
- `NetworkTopology` uses two `static OnceLock<T>` globals as empty-collection sentinels to
  return `&T` (not `Option<&T>`) from accessor methods.

### Patterns to watch in this codebase

- `Option<usize>` comparisons in tests can silently pass when `None < Some(x)` in Rust's `Ord`.
  Always `.expect()` before comparing positions. Found in `system.rs:1058-1060`.
- `windows(2)` duplicate detection reports N-1 errors for N identical IDs (a triplicate gets 2 errors).
  This is technically correct but may surprise users. Confirm intentional before flagging.
- `pub` fields on `System` allow mutation of entity collections bypassing index invariants. This is
  intentional for Phase 1 (builder is the only constructor; `cobre-io` will own construction) but
  worth noting when reviewing Phase 2.

### Penalty resolution architecture

- `penalty.rs` exports standalone `resolve_*` free functions that are pub but NOT called
  inside `SystemBuilder::build()`. They are Phase 2 utilities: `cobre-io` will call them
  when constructing entities from JSON. `SystemBuilder` is a Phase 1 test-only API that
  accepts already-resolved entities directly (e.g. `Hydro::penalties` field is always set).
- `ValidationError` has two variants that are NEVER constructed in Phase 1:
  `DisconnectedBus` and `InvalidPenalty`. Both have TODO/Phase-2 deferred comments.
  These are pre-declared for Phase 2. Don't flag as dead code.

### Phase 2 cobre-io key facts

- `Stage.index` (0-based positional) ≠ `Stage.id` (user-facing domain ID, can be non-zero/non-contiguous).
  `Stage.index` is documented as the key for penalty/bounds array indexing. Override rows carry `stage_id`.
- `resolve_penalties` and `resolve_bounds` treat `stage_id` from override rows as positional `stage_idx`
  via `usize::try_from(row.stage_id)`. This is correct ONLY if stage IDs happen to equal their 0-based index.
  This is the most important correctness risk identified in Phase 2.
- `n_stages` in pipeline.rs counts only study stages (id >= 0); pre-study stages excluded.
- `dead_code` allowances on `config`, `penalties`, `load_factors`, `exchange_factors` in `ParsedData` are
  intentional: these fields are parsed for Layer 2 validation but not forwarded to `System` yet (Phase 3+).
- `ValidationContext::into_result()` silently discards warnings — this is explicitly documented behavior.
- `ValidationContext.errors()` returns `Vec<&ValidationEntry>` (heap-allocated filter), not a slice.
  This is slightly inefficient but clean. Called at most once per file parse in validate_schema.
- `#[allow(clippy::struct_excessive_bools)]` on `FileManifest` is necessary and correct.
- `unreachable!` in pipeline.rs line 52 is actually sound — it maps the `Ok(())` branch of
  `into_result()` when validate_schema returned `None` (no errors added), which logically cannot happen.
  But in practice it CAN be reached if no new errors were added by validate_schema (pre-existing errors
  from Layer 1 caused validate_schema to return None). This is a logic flaw — see FINDING.

### Phase 3 cobre-solver key facts

- `unsafe_code = "allow"` is set at crate level in `Cargo.toml` (overriding workspace `forbid`)
  because FFI to HiGHS requires unsafe blocks. All other workspace lints are manually replicated.
- `HighsSolver` holds a raw `*mut c_void` handle; `Send` is manually impl'd (not `Sync`).
- `ffi.rs` uses `#![allow(dead_code)]` intentionally — it's a sys-crate-style bindings file.
- `self.reduced_costs` (Vec field in `HighsSolver`) is dead storage — it is resized in `load_model`
  but never written or read. `col_dual` from HiGHS is used as reduced costs in `LpSolution`.
- `as_ptr().cast_mut()` on `&self` methods (`extract_solution`, `get_basis`) creates mutable pointers
  from shared references. Technically UB under stacked borrows; Miri would flag it.
  The correct pattern is `&mut self` or taking `&mut self.field` explicitly.
- Inner-block pointer extraction for `row_indices_ptr` in `load_model` is sound (no mutation between
  extraction and FFI call) but fragile. Same pattern in `add_rows` with `col_indices_ptr`.
- `cobre_highs_get_solution` return is silently discarded in `extract_solution` (not in
  `try_extract_partial_solution` which uses `_status`). Called after OPTIMAL confirmed.
- `restore_default_settings` is private and called unconditionally after retry loop.
- The retry loop (5 levels: clear solver, enable presolve, primal simplex, relax tolerances, IPM)
  only activates on `SOLVE_ERROR` or `UNKNOWN` model status. No test exercises it directly.
- `test-support` feature re-exports raw FFI option-setters for integration tests that need
  time/iteration limits (e.g., `test_solver_highs_solve_time_limit`).
- SAFETY comment in `reset()` says "model is still loaded" (true for HiGHS) but code sets
  `has_model = false`. This is correct behavior but the comment is misleading.

### Phase 4 cobre-comm key facts

- `FerrompiBackend::allreduce` does NOT validate `send.len() == recv.len()` or `send.len() > 0` before calling MPI.
  ferrompi internally validates `send.len() != recv.len()` and returns `Error::InvalidBuffer`, which maps to
  `CommError::InvalidBufferSize { expected: 0, actual: 0 }` (sentinel). The empty-buffer case is NOT validated.
- `FerrompiBackend::allgatherv` does NOT validate `counts.len() == self.size` or `send.len() == counts[self.rank()]`.
  ferrompi's `allgatherv` passes directly to MPI FFI without validating count array lengths. If `counts.len() < size`,
  MPI reads past the end of the counts array (UB).
- `BackendKind` is pub-exported but `create_communicator` doesn't accept it as a parameter — intentional forward-
  declaration for ticket-011. Same pattern as Phase 1/2 pre-declarations.
- `CommBackend` enum (feature-gated) has no `#[derive(Debug)]` — `missing_debug_implementations` is not enforced.
- `FerrompiBackend::new()` calls `world.split_shared()` → stored in `self.shared`. `split_local()` calls
  `world.split_shared()` AGAIN for each invocation — a new collective per call. `is_leader()` uses the cached
  `self.shared` from `new()`. This is consistent (same deterministic split result) but slightly wasteful.
- Unit test `ENV_LOCK` in `factory.rs` and integration test `ENV_LOCK` in `factory_tests.rs` are separate mutexes
  in separate binaries — they do not serialize across binary boundaries. Safe in default `cargo test` (sequential
  binary execution) but not with concurrent harnesses like nextest.
- `LocalBackend` uses `saturating_add` for recv-length validation but plain `+` for the actual slice index on line 95.
  Theoretical inconsistency; not reachable on any real system (no allocation could overflow usize).

### Phase 5 cobre-stochastic key facts

- `StochasticContext` bundles `PrecomputedParLp`, `DecomposedCorrelation`, `OpeningTree`, `base_seed`, `dim`.
  All fields immutable after construction; consumed read-only by the optimization loop.
- `build_stochastic_context` calls `validate_par_parameters` first (fatal check + warnings), then
  `PrecomputedParLp::build`. The `ParValidationReport` returned from the first call is silently discarded
  (`let _report = ...`). The report's warning list is never exposed. This is a design gap — callers cannot
  see low-residual-variance warnings. If warning propagation is needed in Phase 6, refactor to return the report.
- `PrecomputedParLp::build` uses stage-major layout: `array[stage * n_hydros + hydro]`. The 3-D `psi` array
  uses `psi[stage * n_hydros * max_order + hydro * max_order + lag]`. All padded with 0.0.
- `CholeskyFactor` uses packed lower-triangular storage: element (i, j) at index `i*(i+1)/2 + j`.
  Symmetry tolerance is 1e-10. PD check: diagonal element <= 0 during Cholesky → error.
- `DecomposedCorrelation` uses `BTreeMap<String, Vec<GroupFactor>>` (deterministic iteration order).
  `resolve_positions` must be called once before `apply_correlation` to pre-compute entity indices.
  `apply_group_precomputed` uses stack-allocated buffers (128 bytes × 2) for groups ≤ 64 entities.
- `derive_forward_seed` (20 bytes: base+iter+scenario+stage) and `derive_opening_seed` (16 bytes:
  base+opening+stage) use `SipHasher13::new()` with zero key. Domain separation relies on different
  message lengths — SipHash-1-3 incorporates length into state. Golden-value test pins output.
- `sample_forward` uses `rng.random::<u64>() % n` (modulo bias). For small `n` (branching factor),
  bias is negligible in practice but is a correctness defect. Fix: `rng.random_range(0..n)`.
- `StochasticError::SeedDerivationError` and `InsufficientData` are pre-declared but never constructed
  in Phase 5 production code. They lack deferred-phase comments (unlike Phase 1/2 pre-declarations).
- `OpeningTree::from_parts` is `pub(crate)` — only callable by `generate.rs` (tree generation module).
  Public API is `generate_opening_tree` (free function) and `OpeningTree` accessors.
- `infrastructure_genericity_no_sddp_references` test in `reproducibility.rs` spawns `grep` subprocess.
  Linux-only; works in CI but would fail on Windows. Acceptable for this codebase (Linux CI only).
- Broken intra-doc links in Phase 5: `[j]` in `cholesky.rs:22` and `[apply_correlation]` in
  `resolve.rs:178` produce rustdoc warnings. Fix: escape brackets in cholesky.rs, use
  `Self::apply_correlation` in resolve.rs.

### Review workflow notes

- Run `cargo clippy --package cobre-core` and `cargo test --package cobre-core` to baseline.
- This project uses workspace lints: `unsafe_code=forbid`, `unwrap_used=deny`, `missing_docs=warn`,
  `clippy::pedantic`. Clippy clean = confirmed by running.
- All entities are sorted by `id.0` (inner `i32`) not by `EntityId` itself (no `Ord` on `EntityId`).
- `EntityId.0` is intentionally `pub` — used throughout tests and sorting code (`sort_by_key(|e| e.id.0)`).
- `CascadeTopology::build()` is public; callers are responsible for providing sorted input.
  `SystemBuilder` always sorts before calling it. Unit tests calling it directly can pass
  unsorted input but the output remains deterministic because Kahn's ready queue is always sorted.
