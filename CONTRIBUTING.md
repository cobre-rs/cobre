# Contributing to Cobre

Thanks for your interest in contributing. Cobre is an open-source ecosystem for power system computation, and contributions of all kinds are welcome — code, documentation, bug reports, test cases, and domain expertise.

## Getting Started

### Prerequisites

- **Rust** (stable, latest): https://rustup.rs
- **C compiler** (for HiGHS solver FFI): `gcc` or `clang`
- **CMake** (for building HiGHS from source): `cmake >= 3.15`

Optional:
- **MPICH** (for ferrompi/distributed execution): `libmpich-dev` on Debian/Ubuntu
- **mdBook** (for building documentation): `cargo install mdbook`

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

# Build documentation locally
cargo doc --workspace --no-deps --open
```

### Project Structure

```
cobre/
├── crates/
│   ├── cobre-core/         # Data model (buses, generators, topology)
│   ├── cobre-io/           # File parsers and serializers
│   ├── cobre-stochastic/   # Stochastic process models (PAR(p), scenarios)
│   ├── cobre-solver/       # LP solver abstraction + HiGHS bindings
│   ├── cobre-sddp/         # SDDP algorithm
│   └── cobre-cli/          # Command-line interface
├── specs/                   # Specification documents
├── examples/                # Example studies and configurations
└── assets/                  # Logos, diagrams
```

## How to Contribute

### Reporting Bugs

Open an issue with:
1. What you did (steps to reproduce, input data if possible)
2. What you expected
3. What actually happened
4. Cobre version (`cargo --version`, `rustc --version`, and the crate version)

For numerical issues (wrong results, convergence failures), include:
- The study configuration (TOML or code snippet)
- System size (number of hydros, thermals, stages, scenarios)
- Expected values and source (e.g., "NEWAVE produces X for the same case")

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

Scope is the crate name without the `cobre-` prefix:

```
feat(sddp): implement multi-cut strategy
fix(core): correct reservoir volume bounds validation
docs(io): document NEWAVE HIDR.DAT field mapping
test(stochastic): add PAR(p) coefficient estimation tests
perf(solver): reduce allocations in basis reuse path
refactor(core): extract topology validation into separate module
```

### Improving Documentation

Documentation improvements are always welcome. This includes:
- Fixing typos or unclear explanations
- Adding examples to rustdoc comments
- Improving the mdBook user guide
- Translating documentation to Portuguese

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
- All public types must implement `Clone`, `Debug`. Implement `Serialize`/`Deserialize` (serde) where appropriate.
- Validation logic lives here. A `System` should be self-consistent when constructed — invalid states should be caught at construction time, not at solve time.

#### cobre-io
- Every parser must have round-trip tests: parse → serialize → parse should produce identical data.
- Include sample input files in `tests/data/` for each supported format.
- Document the source format specification in comments (field widths, units, conventions).

#### cobre-sddp
- Algorithmic changes should reference the relevant literature (paper, section, equation number).
- Numerical changes require validation against reference outputs. Include the test case and expected bounds.

#### cobre-solver
- The solver trait must remain backend-agnostic. HiGHS-specific code stays behind the `highs` feature flag.
- Benchmark any solver interface changes with `cargo bench`.

### Testing

- **Unit tests** go in the same file as the code (`#[cfg(test)] mod tests`).
- **Integration tests** go in `tests/` at the crate root.
- **Use `approx` for floating-point comparisons:** `assert_relative_eq!(actual, expected, epsilon = 1e-6)`.
- **Property-based tests** (proptest) are encouraged for numerical code — if a function should preserve an invariant, test it over random inputs.

### Dependencies

- Prefer well-maintained crates with minimal transitive dependencies.
- New dependencies that add >5s to clean build time need justification.
- Feature-gate optional heavy dependencies (e.g., `serde`, solver backends, MPI).

## Domain Knowledge

Cobre sits at the intersection of power systems engineering, stochastic optimization, and systems programming. Not all contributors will have expertise in all three areas. That's fine.

If you're a **power systems engineer** new to Rust:
- The [Rust Book](https://doc.rust-lang.org/book/) is the standard learning resource
- Focus on `cobre-core` and `cobre-io` — these are the most domain-heavy crates

If you're a **Rust developer** new to power systems:
- The `specs/` directory contains background on the algorithms and domain
- The `cobre-docs` repository has algorithm documentation
- Ask questions in [Discussions](https://github.com/cobre-rs/cobre/discussions) — no question is too basic

If you're a **researcher** with algorithmic improvements:
- Open an issue describing the algorithm, with references
- We can help translate the math into Rust

## Decision Making

Cobre is currently maintained by [@rjmalves](https://github.com/rjmalves). Major design decisions (new crates, breaking API changes, new solver backends) are made through GitHub issues with discussion. As the contributor base grows, we'll formalize this with an RFC process.

## License

By contributing to Cobre, you agree that your contributions will be licensed under the [MIT License](LICENSE).
