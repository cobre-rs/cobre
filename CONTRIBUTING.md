# Contributing to Cobre

Thanks for your interest in contributing. Cobre is an open-source ecosystem for power system computation, and contributions of all kinds are welcome — code, documentation, bug reports, test cases, and domain expertise.

## Getting Started

### Prerequisites

- **Rust** (stable, latest): https://rustup.rs
- **C compiler** (for HiGHS solver FFI): `gcc` or `clang`
- **CMake** (for building HiGHS from source): `cmake >= 3.15`

Optional (needed for specific crates):

- **MPICH** (for `cobre-comm` MPI backend): `libmpich-dev` on Debian/Ubuntu
- **Python 3.12+** and **maturin** (for `cobre-python` builds): `pip install maturin`

### Building

```bash
git clone https://github.com/cobre-rs/cobre.git
cd cobre

# Build all crates
cargo build --workspace

# Run the test suite (default HiGHS backend)
cargo test --workspace

# Run tests for a specific crate
cargo test -p cobre-sddp

# Build Rust API documentation
cargo doc --workspace --no-deps --open
```

### Solver Backend Selection

`cobre-solver` ships two LP backends behind **mutually exclusive** Cargo features:
**HiGHS** (`highs`, on by default) and **CLP/CoinUtils** (`clp`, off by default).
Exactly one may be enabled per build — enabling both (including via
`--all-features`) is rejected at compile time. Any command that compiles
`cobre-solver` (the whole workspace, or `-p cobre-solver`/`-p cobre-sddp`/`-p
cobre-cli`) must therefore select a single backend instead of passing
`--all-features`:

```bash
# Default HiGHS backend
cargo test --workspace

# CLP backend (requires the Clp and CoinUtils submodules)
cargo test --workspace --no-default-features --features clp
```

To exercise the same surface CI does — every non-solver optional feature, run
once per backend — add the non-solver feature set explicitly (this needs MPICH
and `flatc` installed, just as `--all-features` previously did):

```bash
# HiGHS (default) backend
cargo test --workspace --features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"

# CLP backend
cargo test --workspace --no-default-features \
  --features "clp mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"
```

The same backend split applies to `cargo clippy` and `cargo build`. Crates that
do not depend on `cobre-solver` (e.g. `cobre-core`, `cobre-io`) are unaffected
and may still use `--all-features`.

### Testing cobre-core

```bash
# Run all tests including serde round-trip tests
cargo test -p cobre-core --all-features

# Run only the serde-gated tests
cargo test -p cobre-core --features serde
```

### Testing cobre-io

```bash
# Run all tests (requires --all-features)
cargo test -p cobre-io --all-features

# Run only integration tests
cargo test -p cobre-io --all-features --test '*'
```

Sample case directories are in `tests/data/`. Each follows the standard Cobre layout:
`config.json` at the root, entity files under `buses/`, `hydros/`, `thermals/`, etc.
When adding new parsers, add sample input files to `tests/data/` and reference them
from the integration test.

### Testing cobre-solver

Initialize the solver submodules first:

```bash
git submodule update --init --recursive

# Default HiGHS backend (add `test-support` for the FFI option-setting tests)
cargo test -p cobre-solver --features test-support

# CLP backend
cargo test -p cobre-solver --no-default-features --features "clp test-support"
```

The `highs` feature triggers cmake-based HiGHS compilation (requires cmake >= 3.15);
the `clp` feature builds the vendored CLP/CoinUtils superbuild. The two backends are
mutually exclusive — see [Solver Backend Selection](#solver-backend-selection). The
conformance suite validates the `SolverInterface` contract against hand-computed LP
fixtures for whichever backend is active.

### Testing cobre-comm

**Without MPI** (default):

```bash
cargo test -p cobre-comm
```

**With MPI**:

```bash
cargo test -p cobre-comm --features mpi
```

**With Linux CPU affinity**:

```bash
cargo test -p cobre-comm --features affinity
```

The native integration test is Linux-only; topology-planning tests run on every
platform using synthetic sparse/NUMA layouts.

MPI installation required: Debian/Ubuntu: `sudo apt install libmpich-dev`; Fedora:
`sudo dnf install mpich-devel`; macOS: `brew install mpich`. The conformance suite
validates the `Communicator` contract and runs without the `mpi` feature.

### Testing cobre-stochastic

```bash
cargo test -p cobre-stochastic
```

No external dependencies or feature flags needed. Conformance tests verify PAR(p)
preprocessing (tolerance 1e-10); reproducibility tests verify seed determinism and
declaration-order invariance.

### Testing cobre-sddp

Initialize the HiGHS submodule first:

```bash
git submodule update --init --recursive

# Default HiGHS backend (`slow-tests` un-ignores the D-case parity sweep)
cargo test -p cobre-sddp --features slow-tests

# CLP backend
cargo test -p cobre-sddp --no-default-features --features "clp slow-tests"
```

Pass exactly one backend — `--all-features` enables both and fails to compile (see
[Solver Backend Selection](#solver-backend-selection)). The test suite includes unit
tests (forward/backward pass, cut management, simulation),
conformance tests (algorithm contracts against hand-computed fixtures), and integration
tests (end-to-end pipelines, Parquet output validation). The genericity gate asserts
no algorithm-specific references in infrastructure crates.

### Testing cobre-cli

```bash
# Default HiGHS backend
cargo test -p cobre-cli

# CLP backend
cargo test -p cobre-cli --no-default-features --features clp
```

The CLI forwards the active backend to both `cobre-solver` and `cobre-sddp`, so it
takes one backend at a time — not `--all-features` (see
[Solver Backend Selection](#solver-backend-selection)). Integration tests exercise
the binary via `assert_cmd`, organized by subcommand
(`tests/cli_run.rs`, `tests/cli_validate.rs`, `tests/cli_report.rs`,
`tests/cli_smoke.rs`). Each verifies exit codes, output, and file creation.

### Testing cobre-python

`cobre-python` is excluded from the workspace, so address it with
`--manifest-path crates/cobre-python/Cargo.toml` (not `-p cobre-python`) for any
cargo command — this includes `cargo fmt` and `cargo clippy`, which the
workspace-wide invocations skip. The crate is built with PyO3's
`extension-module` + `abi3` features, which deliberately omit libpython linkage,
so `cargo test` on this crate fails to link by design. The supported test path —
the one CI runs — builds the extension into a virtualenv and runs pytest:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest pyarrow
(cd crates/cobre-python && maturin develop --release)
pytest crates/cobre-python/tests/
```

GIL-bound conversion helpers (Python↔JSON value mapping, result dict shapes) are
covered by the pytest suite rather than Rust unit tests — extend the pytest
suite when touching them.

### Project Structure

```
cobre/
├── crates/
│   ├── cobre-core/         # Entity model (buses, hydros, thermals, lines…)
│   ├── cobre-io/           # JSON/Parquet input, FlatBuffers/Parquet output
│   ├── cobre-stochastic/   # PAR(p) models, scenario generation
│   ├── cobre-solver/       # LP solver abstraction (HiGHS backend)
│   ├── cobre-comm/         # Communication abstraction (MPI, local)
│   ├── cobre-sddp/         # SDDP training loop, simulation, cut management
│   ├── cobre-cli/          # Binary: run/validate/report/init/schema/summary/version
│   ├── cobre-mcp/          # Binary: MCP server for AI agent integration (reserved)
│   ├── cobre-python/       # cdylib: PyO3 Python bindings
│   ├── cobre-tui/          # Library: ratatui terminal UI (reserved)
│   ├── cobre-flow/         # Library: power flow algorithms (reserved)
│   ├── cobre-uc/           # Library: MILP unit commitment for hydrothermal dispatch (reserved)
│   └── cobre-emt/          # Library: electromagnetic transient analysis (reserved)
├── assets/                  # Logos and diagrams
└── docs/                    # Internal project documentation
```

The full specification corpus lives in the separate [cobre-docs](https://github.com/cobre-rs/cobre-docs) repository ([deployed docs](https://docs.cobre-rs.dev/)).

## How to Contribute

### Reporting Bugs

Open an issue with:

1. What you did (steps to reproduce, input data if possible)
2. What you expected
3. What actually happened
4. Cobre version (`cargo --version`, `rustc --version`, and the crate version)

For numerical issues (wrong results, convergence failures), include:

- The study configuration (`config.json`)
- System size (number of hydros, thermals, stages, scenarios)
- Expected values and source (e.g., "produces X for this input")

### Suggesting Features

Open an issue describing:

- The use case — what problem are you trying to solve?
- Which crate(s) it would affect
- Whether you'd be willing to implement it

For algorithmic enhancements, a reference to the relevant paper or implementation is very helpful.

### Submitting Code

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/my-feature`
3. **Make your changes** — see coding guidelines below
4. **Test**: `cargo test --workspace` (and the CLP backend — see [Solver Backend Selection](#solver-backend-selection))
5. **Lint**: `cargo clippy --workspace --all-targets -- -D warnings`
6. **Format**: `cargo fmt --all`
7. **Push** and open a pull request

#### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

Types:

- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `test` — adding or correcting tests
- `perf` — performance improvement
- `ci` — CI/CD changes
- `chore` — maintenance (dependencies, tooling)

Scope is the crate name without the `cobre-` prefix (use `ferrompi` for the MPI bindings):

```
feat(sddp): implement multi-cut strategy
fix(core): correct reservoir volume bounds validation
docs(io): document Parquet schema for hydro inflows
test(stochastic): add PAR(p) coefficient estimation tests
perf(solver): reduce allocations in basis reuse path
refactor(comm): extract MPI backend into separate module
ci: add cargo-deny license check
chore(ferrompi): update to v0.3.0
```

### Improving Documentation

Documentation improvements are always welcome. The Rust API docs live inline
in source code (rustdoc); crate-internal reference docs live in this repo's
per-crate READMEs (`crates/*/README.md`). User-facing guides, tutorials, and
reference pages live on the software docs site,
[docs.cobre-rs.dev](https://docs.cobre-rs.dev/). The methodology specification
corpus lives in the separate
[cobre-docs](https://github.com/cobre-rs/cobre-docs) repository.

#### JSON schemas and the cobre-docs vendored copy

The input JSON schemas are generated from the `cobre-io` Rust types — code is
the source of truth. They are exported with `cobre schema export`, committed
under `schemas/`, and CI guards their freshness (`scripts/ci/check_schemas.sh`,
via the `schemas` job in `.github/workflows/ci.yml`). If a schema-bearing type
changes — including adding or renaming a field, changing a
`#[derive(JsonSchema)]` type, or editing a schemars-visible doc comment —
regenerate them:

```bash
cargo build --release --bin cobre
./target/release/cobre schema export --output-dir schemas
```

The cobre-docs site **vendors** these schemas for its Reference pages. When the
schemas change in a release, refresh the vendored copy in cobre-docs
(`npm run refresh:schemas` there) so the published reference stays in sync. The
freshness gate stays here in `cobre`, next to the generating types; cobre-docs
only consumes the exported output.

## Coding Guidelines

### General

- **Run the full check before pushing** (default HiGHS backend; repeat the
  clippy/test steps with `--no-default-features --features clp` to cover CLP —
  see [Solver Backend Selection](#solver-backend-selection)):
  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace
  ```
- **No `unsafe` without justification.** If you need `unsafe`, add a `// SAFETY:` comment explaining the invariants.
- **No `unwrap()` in library code.** Use `Result` or `Option` with proper error types. `unwrap()` is acceptable in tests and examples.
- **Minimize allocations in hot paths.** The SDDP solver runs millions of LP solves — allocation-heavy code in the inner loop is a performance problem.

### Python Parity

Every output file written by the CLI (`write_training_outputs` / `write_simulation_outputs` in `crates/cobre-cli/src/commands/run/outputs.rs`)
must also be written by the Python bindings (`run_via_study` / `run_training_phase_py` in `crates/cobre-python/src/run.rs`).
When adding a new output:

1. Add the `cobre_io::write_*` call in both the CLI and Python paths
2. Run `python3 scripts/ci/check_python_parity.py` to verify parity
3. The pre-commit hook runs this check automatically

See `.claude/architecture-rules.md` for the full Python parity checklist.

### Crate-Specific Guidelines

#### cobre-core

- Types here are shared across all solvers. Changes require careful consideration of downstream impact.
- All public types must implement `Clone`, `Debug`. Implement `serde::Serialize`/`Deserialize` where appropriate.
- Entity collections must be stored in ID-sorted (canonical) order. **Declaration-order invariance** is a hard requirement: results must be bit-for-bit identical regardless of input file ordering.
- Validation logic lives here. A resolved system should be self-consistent — invalid states should be caught at load time, not at solve time.
- The `serde` feature enables JSON serialization for core types. Enable it with `--features serde` or `--all-features` when running tests that cover serialization round-trips.

#### cobre-io

- Every parser must have round-trip tests: parse → serialize → parse should produce identical data.
- Include sample input files in `tests/data/` for each supported format.
- The layered validation pipeline (structural → schema → referential → dimensional → semantic) must collect all errors before failing; never short-circuit on the first error.
- Always run `cargo test -p cobre-io --all-features` to include the full test suite. Tests gated behind the `serde` feature (from `cobre-core`) are required for integration tests.

#### cobre-sddp

- Algorithmic changes must reference the relevant literature (paper, section, equation number).
- Numerical changes require validation against reference outputs. Include the test case and expected bounds in the PR.
- The four algorithm parameterization points (risk measure, cut formulation, horizon mode, sampling scheme) must remain generic — no hard-coding of specific strategies.

#### cobre-solver

- The `SolverInterface` trait must remain backend-agnostic. HiGHS-specific code stays behind the `highs` feature flag.
- Criterion benchmarks for solver interface changes are planned but not yet configured; they will be added in a future phase.
- Basis warm-starting affects correctness, not only performance — validate it in tests.

#### cobre-comm

- The `Communicator` trait must remain implementable by both backends (MPI and local).
- The local backend's calls compile to no-ops on the hot path — do not add indirection that penalizes single-process users.
- MPI code requires an MPI installation to test; gate MPI tests appropriately.

#### cobre-mcp and cobre-python

- These crates are **single-process only** — they must never initialize MPI or depend on `ferrompi`.
- `cobre-python` must release the GIL (`py.allow_threads()`) during all Rust computation.

### Testing

- **Unit tests** go in the same file as the code (`#[cfg(test)] mod tests`).
- **Integration tests** go in `tests/` at the crate root.
- **Use `approx` for floating-point comparisons:** `assert_relative_eq!(actual, expected, epsilon = 1e-6)`.
- **Property-based tests** (proptest) are encouraged for numerical code.
- **Order-invariance tests:** for any function that processes entity collections, test that reordering the input produces identical output.

### Dependencies

- Prefer well-maintained crates with minimal transitive dependencies.
- New dependencies are checked for license compliance by `cargo-deny` in CI.
- Feature-gate optional heavy dependencies (solver backends, MPI, PyO3, ratatui).

## Domain Knowledge

Cobre sits at the intersection of power systems engineering, stochastic optimization, and systems programming. Not all contributors will have expertise in all three areas. That's fine.

If you're a **power systems engineer** new to Rust:

- The [Rust Book](https://doc.rust-lang.org/book/) is the standard learning resource
- Focus on `cobre-core` and `cobre-io` — these are the most domain-heavy crates

If you're a **Rust developer** new to power systems:

- The [docs site](https://docs.cobre-rs.dev/) has algorithm documentation and a user guide
- Ask questions in [Discussions](https://github.com/cobre-rs/cobre/discussions) — no question is too basic

If you're a **researcher** with algorithmic improvements:

- Open an issue describing the algorithm, with references
- We can help translate the math into Rust

## Decision Making

Cobre is currently maintained by [@rjmalves](https://github.com/rjmalves). Major design decisions (new crates, breaking API changes, new solver backends) are made through GitHub issues with discussion. As the contributor base grows, we'll formalize this with an RFC process.

## Release Checklist

Before tagging a new release:

1. Update `CHANGELOG.md` with the new version's changes
2. If the release changed deterministic numerical output (solver tolerances,
   scaling, cut selection, basis handling, …), regenerate the parity baselines
   and update the guard table **in the same change** — otherwise the `Test`
   job goes red on the stale baselines:
   ```bash
   cargo nextest run -p cobre-sddp --features slow-tests --test parity \
     -E 'test(parity_regen)' --run-ignored ignored-only
   cargo nextest run -p cobre-sddp --no-default-features --features "clp slow-tests" \
     --test parity -E 'test(parity_regen)' --run-ignored ignored-only
   # These rewrite the committed baselines under
   # crates/cobre-sddp/tests/fixtures/parity_baselines/ (HiGHS) and
   # crates/cobre-sddp/tests/fixtures/parity_baselines_clp/ (CLP). The two dirs
   # are the single source of truth and are re-baselined TOGETHER — updating
   # only one lets the other backend's slow-gated suite rot silently. Review
   # the .sha256 diffs and commit them in the same change.
   ```
3. Run quality checks:
   ```bash
   python3 scripts/ci/check_python_parity.py --max 0
   ```
4. If `schemas/` changed since the last release, refresh the vendored copy in
   the `cobre-docs` repository (`npm run refresh:schemas` there) so the
   published reference stays in sync — see
   [JSON schemas and the cobre-docs vendored copy](#json-schemas-and-the-cobre-docs-vendored-copy)
5. Run `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings`,
   then repeat clippy for the CLP backend:
   `cargo clippy --workspace --all-targets --no-default-features --features clp -- -D warnings`
   (the two backends are mutually exclusive — see [Solver Backend Selection](#solver-backend-selection))
6. Tag: `git tag v<version>`

## License

By contributing to Cobre, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
