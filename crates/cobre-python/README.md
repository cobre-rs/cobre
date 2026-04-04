# cobre

Python bindings for the [Cobre](https://github.com/cobre-rs/cobre) power systems solver.

Cobre is a high-performance SDDP (Stochastic Dual Dynamic Programming) solver for hydrothermal dispatch, written in Rust. This package provides Python access to case loading, validation, training, simulation, and result inspection.

## Installation

```bash
pip install cobre-python
```

Pre-built wheels are available for:

- Linux x86_64 and aarch64 (manylinux_2_34), musl x86_64
- macOS Apple Silicon (aarch64) and Intel (x86_64)
- Windows x86_64
- Python 3.12+

## Quick Start

```python
import cobre

# Load and validate a case
system = cobre.io.load_case("path/to/case")
print(f"System: {system.n_buses} buses, {system.n_hydros} hydros, {system.n_thermals} thermals")

# Run training + simulation
result = cobre.run.run("path/to/case", output_dir="output/")
print(f"Converged: {result['converged']}, LB: {result['lower_bound']:.2f}")

convergence = cobre.results.load_convergence("output/")
print(f"Iterations: {len(convergence)}")

simulation = cobre.results.load_simulation("output/")
print(f"Cost records: {len(simulation['costs'])}")

policy = cobre.results.load_policy("output/")
print(f"Iterations completed: {policy['metadata']['completed_iterations']}")
```

## Modules

- **`cobre.io`** — Load and validate case directories
- **`cobre.model`** — Data model classes (System, Bus, Line, Thermal, Hydro, etc.)
- **`cobre.run`** — Execute SDDP training and simulation
- **`cobre.results`** — Load and inspect output artifacts, including convergence
  history, Parquet simulation outputs, and FlatBuffers policy (FCF) checkpoints

## Requirements

- Python >= 3.12
- No runtime dependencies (the Rust solver is statically linked)

## License

Apache-2.0 — see [LICENSE](https://github.com/cobre-rs/cobre/blob/main/LICENSE).

## Links

- [Repository](https://github.com/cobre-rs/cobre)
- [Software Book](https://cobre-rs.github.io/cobre/)
- [Documentation](https://cobre-rs.github.io/cobre-docs/)
- [Bug Tracker](https://github.com/cobre-rs/cobre/issues)
