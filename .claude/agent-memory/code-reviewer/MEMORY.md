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

### Review workflow notes

- Run `cargo clippy --package cobre-core` and `cargo test --package cobre-core` to baseline.
- This project uses workspace lints: `unsafe_code=forbid`, `unwrap_used=deny`, `missing_docs=warn`,
  `clippy::pedantic`. Clippy clean = confirmed by running.
- All entities are sorted by `id.0` (inner `i32`) not by `EntityId` itself (no `Ord` on `EntityId`).
- `EntityId.0` is intentionally `pub` — used throughout tests and sorting code (`sort_by_key(|e| e.id.0)`).
- `CascadeTopology::build()` is public; callers are responsible for providing sorted input.
  `SystemBuilder` always sorts before calling it. Unit tests calling it directly can pass
  unsorted input but the output remains deterministic because Kahn's ready queue is always sorted.
