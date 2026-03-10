# Getting Started

Cobre is a power system analysis toolkit built around a production-grade SDDP
solver for long-term hydrothermal dispatch planning. It reads a self-contained
case directory of JSON and Parquet input files, trains a stochastic dispatch
policy, simulates that policy over independent scenarios, and writes
Hive-partitioned Parquet output ready for analysis.

This section of the User Guide is the reference path through the software.
If you prefer a hands-on walkthrough starting from a working example, the
[Tutorial](../tutorial/installation.md) is the better starting point.

---

## What You Need

### To use pre-built binaries (recommended for most users)

No Rust toolchain or C compiler required. The pre-built binary is statically
linked and runs on the following platforms out of the box:

| Platform              | Target Triple               |
| --------------------- | --------------------------- |
| macOS (Apple Silicon) | `aarch64-apple-darwin`      |
| macOS (Intel)         | `x86_64-apple-darwin`       |
| Linux (x86-64)        | `x86_64-unknown-linux-gnu`  |
| Linux (ARM64)         | `aarch64-unknown-linux-gnu` |
| Windows (x86-64)      | `x86_64-pc-windows-msvc`    |

### To build from source

| Dependency     | Minimum Version | Notes                                   |
| -------------- | --------------- | --------------------------------------- |
| Rust toolchain | 1.85 (stable)   | Install via [rustup](https://rustup.rs) |
| C compiler     | GCC or Clang    | Required for the HiGHS LP solver        |
| CMake          | 3.15            | Required for the HiGHS build system     |

---

## Next Steps

### Install Cobre

[Installation](./installation.md) covers all three installation methods:
pre-built binary (the fastest path), `cargo install` from crates.io, and
building from source for contributors or unsupported platforms.

### Run Your First Study

[Your First Study](./first-study.md) walks through the end-to-end workflow
for running a study on a case directory you already have: validate inputs,
run the solver, and inspect results with `cobre report`.

---

## For Hands-On Learners

The [Tutorial](../tutorial/installation.md) section provides a step-by-step
learning path that starts by installing Cobre, then scaffolds a complete
example study using the built-in `1dtoy` template, explains the anatomy of a
case directory, and shows how to read the output files.

If you have not used Cobre before, starting with the Tutorial and returning
to the User Guide as a reference is the recommended approach.

---

## Related Pages

- [Installation](./installation.md) — all installation methods and platform table
- [Your First Study](./first-study.md) — validate, run, and inspect a study
- [Running Studies](./running-studies.md) — full workflow reference with exit codes
- [CLI Reference](./cli-reference.md) — all subcommands, flags, and environment variables
