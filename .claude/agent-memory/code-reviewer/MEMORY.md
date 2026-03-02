# Code Reviewer Memory

## Project: cobre (Rust SDDP solver ecosystem)

### Key architecture facts

- `System` struct has both `pub` fields AND `pub fn` accessor methods — this dual API is intentional
  (fields for `PartialEq`/`Debug`, methods for slice access). But public fields break encapsulation.
- `SystemBuilder::build()` sorts entities by `id.0`, then checks duplicates via `windows(2)`,
  then validates cross-refs. Error collection is non-short-circuiting per design (DEC-005 style).
- `HasId` trait is private; `check_duplicates` and `build_index` are private fns.
- `CascadeTopology::build()` uses Kahn's algorithm with a sorted `Vec<i32>` as the ready queue.
  The `ready.remove(0)` is O(n) per step — acceptable at build time, not hot path.
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

### Review workflow notes

- Run `cargo clippy --package cobre-core` and `cargo test --package cobre-core` to baseline.
- This project uses workspace lints: `unsafe_code=forbid`, `unwrap_used=deny`, `missing_docs=warn`,
  `clippy::pedantic`. Clippy clean = confirmed by running.
- All entities are sorted by `id.0` (inner `i32`) not by `EntityId` itself (no `Ord` on `EntityId`).
