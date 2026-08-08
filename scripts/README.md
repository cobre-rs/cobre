# `scripts/`

Helper scripts for Cobre, grouped by role.

- **`ci/`** — quality gates and the advisory report. Run in CI
  (`.github/workflows/ci.yml`), and some also from the `pre-commit` hook and from
  Rust integration tests. The shell gates locate the repo root from their own
  path, so they run from anywhere; the Python gates default to the current
  directory (pass `--root` to override).
- **`gen/`** — generators for release/build artifacts.
- **`benchmark_numa.py`** — matched-epoch worker-count/affinity benchmark matrix;
  writes topology, timing, solve-count, memory, and placement data to JSON and
  rejects numerical policy or solver-work drift between arms.
- **`pre-commit`** — the git pre-commit hook. Install with
  `ln -sf ../../scripts/pre-commit .git/hooks/pre-commit`.

## `ci/` — quality gates

Most gates fail the build on violation; the ones marked _advisory_ never fail.
The shared `cfg(test)`-boundary helpers live in `ci/lib/comment_scan.sh` (it is
sourced, not executed; running it directly self-checks that the holdout gates
still carry the canonical boundary regex).

| Script                               | Purpose                                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `check-doc-paths.sh`                 | Repo-relative paths in README/CONTRIBUTING/CLAUDE.md resolve against the tree.                             |
| `check_doc_voice.py`                 | No promotional voice or unpinned "typical" numbers in prose (doc-integrity §5/§2).                         |
| `check-doc-placeholders.sh`          | No placeholder text (TODO/TBD/…) in shipped docs.                                                          |
| `check-docs-examples.sh`             | A fresh `init`→`run`→`report` matches the expected output structure (needs the release binary; `--build`). |
| `check-no-plan-leaks.sh`             | No plan-structure tokens (`Epic`/`ticket`/…) in shipped artifacts.                                         |
| `check-comment-refs.sh`              | No un-rottable `file.rs:NNN`-style references in shipped comments.                                         |
| `check-comment-line-refs.sh`         | No drift-prone line references in shipped comments.                                                        |
| `check-comment-banners.sh`           | Flags in-function banner-divider comments (advisory).                                                      |
| `check-comment-bloat.sh`             | Ranks comment-bloat candidates (advisory; also surfaced by `quality-report.sh`).                           |
| `check-allow-rationale.sh`           | Every `#[allow]` carries a `// Rationale:` (E4; diff-scoped via `BASE_REF`).                               |
| `check-infra-genericity.sh`          | No algorithm-specific vocabulary in the infrastructure crates.                                             |
| `check-cut-selection-determinism.sh` | Cut-selection code stays declaration-order deterministic.                                                  |
| `check_python_parity.py`             | Every output file the CLI writes is also written by the Python bindings.                                   |
| `check_schemas.sh`                   | Exported JSON schemas match the source types (needs the release binary; `--build`).                        |
| `quality-report.sh`                  | Advisory code-quality hotspot report; also runs the comment-bloat advisory.                                |
| `lib/comment_scan.sh`                | Shared `cfg(test)`-boundary helpers, sourced by the comment gates.                                         |
| `allow-rationale-allowlist.txt`      | Allowlist consumed by `check-allow-rationale.sh`.                                                          |

## `gen/` — generators

| Script                     | Purpose                                                                       |
| -------------------------- | ----------------------------------------------------------------------------- |
| `inject_wheel_licenses.py` | Bundle license files into built Python wheels (used by `release-python.yml`). |
