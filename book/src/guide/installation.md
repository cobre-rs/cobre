# Installation

Cobre is a statically linked binary available for five platforms.
Choose the method that best fits your environment.

---

## Pre-built Binaries (Recommended)

Fastest for end users. No Rust toolchain or C compiler required.

### Linux and macOS

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cobre-rs/cobre/releases/latest/download/cobre-cli-installer.sh | sh
```

The installer places the `cobre` binary in `$CARGO_HOME/bin` (typically
`~/.cargo/bin`). Add that directory to your `PATH` if it is not already present.

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/cobre-rs/cobre/releases/latest/download/cobre-cli-installer.ps1 | iex"
```

### Supported Platforms

| Platform              | Target Triple               |
| --------------------- | --------------------------- |
| macOS (Apple Silicon) | `aarch64-apple-darwin`      |
| macOS (Intel)         | `x86_64-apple-darwin`       |
| Linux (x86-64)        | `x86_64-unknown-linux-gnu`  |
| Linux (ARM64)         | `aarch64-unknown-linux-gnu` |
| Windows (x86-64)      | `x86_64-pc-windows-msvc`    |

You can also download individual archives directly from the
[GitHub Releases page](https://github.com/cobre-rs/cobre/releases/latest).

### Verify the Installation

```bash
cobre version
```

Expected output (exact versions and arch will vary):

```
cobre   v0.3.0
solver: HiGHS
comm:   local
zstd:   enabled
arch:   x86_64-linux
build:  release (lto=thin)
```

---

## From crates.io

```bash
cargo install cobre-cli
```

Requires Rust 1.86+ and build prerequisites (see Build from Source below).
Installs to `$CARGO_HOME/bin`.

---

## Build from Source

For contributors or unsupported platforms.

### Prerequisites

| Dependency     | Minimum Version         | Notes                                   |
| -------------- | ----------------------- | --------------------------------------- |
| Rust toolchain | 1.86 (stable)           | Install via [rustup](https://rustup.rs) |
| C compiler     | any recent GCC or Clang | Required for the HiGHS LP solver        |
| CMake          | 3.15                    | Required for the HiGHS build system     |
| Git            | any                     | Required for submodule initialization   |

### Steps

```bash
# Clone the repository
git clone https://github.com/cobre-rs/cobre.git
cd cobre

# Initialize HiGHS submodule (required for the solver backend)
git submodule update --init --recursive

# Build the release binary
cargo build --release -p cobre-cli
```

The binary is written to `target/release/cobre`. Optionally install to `$CARGO_HOME/bin`:

```bash
cargo install --path crates/cobre-cli
```

Verify:

```bash
./target/release/cobre version
cargo test --workspace --all-features
```

---

## Next Steps

- [Quickstart](../tutorial/quickstart.md) — run a complete study end to end using the built-in `1dtoy` template
- [Running Studies](./running-studies.md) — validate, run, and inspect results for any case directory
- [Case Directory Format](../reference/case-format.md) — how to structure input data
- [CLI Reference](./cli-reference.md) — complete flag and subcommand reference
