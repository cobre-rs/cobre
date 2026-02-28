# Contributing to Cobre

Thanks for your interest in contributing. Cobre is an open-source ecosystem for power system computation, and contributions of all kinds are welcome — code, documentation, bug reports, test cases, and domain expertise.

## Getting Started

### Prerequisites

- **Rust** (stable, latest): https://rustup.rs
- **C compiler** (for HiGHS/CLP solver FFI): `gcc` or `clang`
- **CMake** (for building HiGHS from source): `cmake >= 3.15`

Optional (needed for specific crates):

- **MPICH** (for `cobre-comm` MPI backend): `libmpich-dev` on Debian/Ubuntu
- **Python 3.8+** and **maturin** (for `cobre-python` builds): `pip install maturin`
- **mdBook** (for the documentation site): `cargo install mdbook mdbook-katex`

### Building

```bash
git clone https://github.com/cobre-rs/cobre.git
cd cobre

# Build all crates
cargo build --workspace

# Run the full test suite
cargo test --workspace

# Run tests for a specific crate
cargo test -p cobre-sddp

# Build Rust API documentation
cargo doc --workspace --no-deps --open
```

### Project Structure

```
cobre/
├── crates/
│   ├── cobre-core/         # Entity model (buses, hydros, thermals, lines…)
│   ├── cobre-io/           # JSON/Parquet input, FlatBuffers/Parquet output
│   ├── cobre-stochastic/   # PAR(p) models, scenario generation
│   ├── cobre-solver/       # LP solver abstraction (HiGHS, CLP backends)
│   ├── cobre-comm/         # Communication abstraction (MPI, TCP, shm, local)
│   ├── cobre-sddp/         # SDDP training loop, simulation, cut management
│   ├── cobre-cli/          # Binary: run/validate/report/compare/serve
│   ├── cobre-mcp/          # Binary: MCP server for AI agent integration
│   ├── cobre-python/       # cdylib: PyO3 Python bindings
│   └── cobre-tui/          # Library: ratatui terminal UI
├── assets/                  # Logos and diagrams
└── docs/                    # Internal project documentation
```

The full specification corpus lives in the separate [cobre-docs](https://github.com/cobre-rs/cobre-docs) repository ([deployed docs](https://cobre-rs.github.io/cobre-docs/)).

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
4. **Test**: `cargo test --workspace`
5. **Lint**: `cargo clippy --workspace -- -D warnings`
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

Documentation improvements are always welcome. The Rust API docs live inline in source code (rustdoc). The user guide and specification corpus live in [cobre-docs](https://github.com/cobre-rs/cobre-docs).

## Coding Guidelines

### General

- **Run the full check before pushing:**
  ```bash
  cargo fmt --all
  cargo clippy --workspace -- -D warnings
  cargo test --workspace
  ```
- **No `unsafe` without justification.** If you need `unsafe`, add a `// SAFETY:` comment explaining the invariants.
- **No `unwrap()` in library code.** Use `Result` or `Option` with proper error types. `unwrap()` is acceptable in tests and examples.
- **Minimize allocations in hot paths.** The SDDP solver runs millions of LP solves — allocation-heavy code in the inner loop is a performance problem.

### Crate-Specific Guidelines

#### cobre-core

- Types here are shared across all solvers. Changes require careful consideration of downstream impact.
- All public types must implement `Clone`, `Debug`. Implement `serde::Serialize`/`Deserialize` where appropriate.
- Entity collections must be stored in ID-sorted (canonical) order. **Declaration-order invariance** is a hard requirement: results must be bit-for-bit identical regardless of input file ordering.
- Validation logic lives here. A resolved system should be self-consistent — invalid states should be caught at load time, not at solve time.

#### cobre-io

- Every parser must have round-trip tests: parse → serialize → parse should produce identical data.
- Include sample input files in `tests/data/` for each supported format.
- The 5-layer validation pipeline (structural → schema → referential → dimensional → semantic) must collect all errors before failing; never short-circuit on the first error.

#### cobre-sddp

- Algorithmic changes must reference the relevant literature (paper, section, equation number).
- Numerical changes require validation against reference outputs. Include the test case and expected bounds in the PR.
- The four algorithm parameterization points (risk measure, cut formulation, horizon mode, sampling scheme) must remain generic — no hard-coding of specific strategies.

#### cobre-solver

- The `Solver` trait must remain backend-agnostic. HiGHS-specific code stays behind the `highs` feature flag; CLP-specific code behind `clp`.
- Benchmark any solver interface changes with `cargo bench`.
- Basis warm-starting is a correctness feature, not just a performance optimization — validate it in tests.

#### cobre-comm

- The `Communicator` trait must remain implementable by all four backends (MPI, TCP, shared-memory, local).
- Local backend has zero overhead — do not add indirection that penalizes single-process users.
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

- The [cobre-docs site](https://cobre-rs.github.io/cobre-docs/) has algorithm documentation and a user guide
- Ask questions in [Discussions](https://github.com/cobre-rs/cobre/discussions) — no question is too basic

If you're a **researcher** with algorithmic improvements:

- Open an issue describing the algorithm, with references
- We can help translate the math into Rust

## Decision Making

Cobre is currently maintained by [@rjmalves](https://github.com/rjmalves). Major design decisions (new crates, breaking API changes, new solver backends) are made through GitHub issues with discussion. As the contributor base grows, we'll formalize this with an RFC process.

## License

By contributing to Cobre, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
