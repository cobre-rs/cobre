<p align="center">
  <img src="assets/cobre-logo-dark.svg" width="360" alt="Cobre — Power Systems in Rust"/>
</p>

<p align="center">
  <strong>Open infrastructure for power system computation</strong>
</p>

<p align="center">
  <a href="https://github.com/cobre-rs/cobre/actions"><img src="https://github.com/cobre-rs/cobre/workflows/CI/badge.svg" alt="CI status"/></a>
  <a href="https://github.com/cobre-rs/cobre/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache 2.0"/></a>
</p>

---

**Cobre** is an open-source ecosystem of Rust crates for power system analysis and optimization. It provides a shared data model, file format interoperability, and modular solvers — starting with stochastic optimization for hydrothermal dispatch (SDDP) and designed to grow into power flow, dynamic simulation, and beyond.

The name comes from the Portuguese word for **copper** — the metal that conducts electricity.

## Why Cobre?

Power system computation today is split between closed-source commercial tools and fragmented academic projects. Cobre aims to provide:

- **A shared data model** — the same `HydroPlant`, `Bus`, or `ThermalUnit` struct works whether you're running a 10-year stochastic dispatch or a steady-state power flow. Define your system once, analyze it from multiple angles.
- **Production performance** — Rust gives us C/C++-level speed with memory safety guarantees. For software that dispatches national power grids, both matter.
- **Interoperability** — parsers for NEWAVE (CEPEL), PSS/E, and standard formats. Bring your existing data, export results in modern formats (Arrow/Parquet, JSON).
- **Modularity** — pick the crates you need. Use `cobre-core` for data modeling without pulling in solver dependencies. Use `cobre-sddp` without caring about power flow.

## Crates

| Crate | Status | Description |
|-------|--------|-------------|
| [`cobre-core`](crates/cobre-core/) | 🔴 Experimental | Power system data model — buses, branches, generators, loads, topology |
| [`cobre-io`](crates/cobre-io/) | 🔴 Experimental | File parsers and serializers (NEWAVE, CSV, JSON, Arrow) |
| [`cobre-stochastic`](crates/cobre-stochastic/) | 🔴 Experimental | Stochastic processes — PAR(p) models, correlated scenario generation |
| [`cobre-solver`](crates/cobre-solver/) | 🔴 Experimental | LP/MIP solver abstraction with HiGHS backend |
| [`cobre-sddp`](crates/cobre-sddp/) | 🔴 Experimental | Stochastic Dual Dynamic Programming for hydrothermal dispatch |
| [`cobre-cli`](crates/cobre-cli/) | 🔴 Experimental | Command-line interface for running studies |

**Related:**

| Repository | Description |
|-----------|-------------|
| [`ferrompi`](https://github.com/cobre-rs/ferrompi) | MPI 4.x bindings for Rust via FFI |

> **Status key:** 🔴 Experimental — API will change, not for production. 🟡 Alpha — core functionality works, gaps remain. 🔵 Beta — feature-complete, seeking feedback. 🟢 Stable — production-ready, semver guarantees.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Applications                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   CLI    │  │  TUI Monitor │  │  Web Dashboard (future)│ │
│  └────┬─────┘  └──────┬───────┘  └────────┬───────────────┘ │
├───────┼────────────────┼──────────────────┼─────────────────┤
│       │         Solvers│                  │                  │
│  ┌────┴─────┐  ┌──────┴───────┐  ┌───────┴──────────────┐  │
│  │   SDDP   │  │ Power Flow   │  │ Transient Stability  │  │
│  │          │  │   (future)   │  │      (future)        │  │
│  └────┬─────┘  └──────┬───────┘  └───────┬──────────────┘  │
├───────┼────────────────┼──────────────────┼─────────────────┤
│       │        Shared  │Infrastructure    │                  │
│  ┌────┴────────────────┴──────────────────┴──────────────┐  │
│  │                    cobre-core                          │  │
│  │         Data model, topology, validation               │  │
│  └───────────────────────┬───────────────────────────────┘  │
│  ┌──────────┐  ┌─────────┴──────┐  ┌───────────┐           │
│  │ Solver   │  │  Stochastic    │  │    IO     │           │
│  │ (HiGHS)  │  │  (PAR, Monte   │  │ (NEWAVE, │           │
│  │          │  │   Carlo)       │  │  PSS/E)   │           │
│  └──────────┘  └────────────────┘  └───────────┘           │
├─────────────────────────────────────────────────────────────┤
│                 Optional: ferrompi (MPI)                     │
└─────────────────────────────────────────────────────────────┘
```

The key architectural principle: **cobre-core is the foundation.** Every solver and tool shares the same data types. A hydro plant defined for SDDP dispatch is the same struct that would be used in a power flow study. This enables multi-domain analysis on a single system definition.

## Quick Start

> ⚠️ Cobre is under active development. The API is not stable yet.

```bash
# Clone the workspace
git clone https://github.com/cobre-rs/cobre.git
cd cobre

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run an example study
cargo run --bin cobre-cli -- run examples/hydrothermal_3bus.toml
```

### As a library dependency

```toml
# In your Cargo.toml
[dependencies]
cobre-core = "0.1"
cobre-sddp = "0.1"
```

```rust
use cobre_core::{System, HydroPlant, ThermalUnit, Bus};
use cobre_sddp::{SddpConfig, train, simulate};

fn main() {
    // Load system from file or build programmatically
    let system = System::from_toml("my_system.toml").unwrap();
    
    // Configure and run SDDP
    let config = SddpConfig {
        iterations: 200,
        forward_scenarios: 20,
        risk_measure: RiskMeasure::cvar(0.95, 0.5),
        ..Default::default()
    };
    
    let policy = train(&system, &config);
    let results = simulate(&system, &policy, 2000);
    
    println!("Lower bound: {:.2}", policy.lower_bound());
    println!("Simulated cost (mean): {:.2}", results.mean_cost());
    println!("Optimality gap: {:.2}%", results.gap_percent(&policy));
}
```

## Context

Cobre was born from the need for an open, modern alternative to the legacy FORTRAN-based tools used for power system planning in Brazil (NEWAVE, DECOMP, DESSEM) and alongside the commercial PSR SDDP suite. While those tools are mature and production-proven, they present challenges in auditability, extensibility, and integration with modern computational infrastructure.

The project draws inspiration from:
- **NREL Sienna** (Julia) — ecosystem architecture with shared data model
- **PowSyBl** (Java) — modular design, institutional adoption path
- **SDDP.jl** (Julia) — algorithmic reference for SDDP implementation
- **SPARHTACUS** (C++) — auditable pre-processing approach

Cobre is not a replacement for these tools — it's a new entry in the ecosystem, offering the Rust community's strengths (safety, performance, modern tooling) to a domain that can benefit from them.

## Roadmap

### Phase 0 — Foundation (current)
- [ ] Complete SDDP specification
- [ ] Extract core data model into `cobre-core`
- [ ] Implement `cobre-io` with NEWAVE file parsers
- [ ] Implement PAR(p) autoregressive inflow models
- [ ] First working SDDP solver with validation against reference cases
- [ ] CLI for running studies
- [ ] ferrompi integration for distributed execution

### Phase 1 — Ecosystem Hardening
- [ ] Python bindings (PyO3)
- [ ] TUI for real-time study monitoring
- [ ] Benchmark suite with published results
- [ ] Additional IO formats (PSS/E, Arrow/Parquet)
- [ ] Comparison study: Cobre vs. NEWAVE on public test cases

### Phase 2 — Power Flow
- [ ] Newton-Raphson AC power flow
- [ ] DC power flow
- [ ] Optimal Power Flow (OPF)
- [ ] Web-based visualization

### Future
- Dynamic simulation (electromechanical transients)
- Renewable uncertainty modeling
- Battery/storage optimization
- Real-time integration (SCADA protocols, digital twin)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

The project follows [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:
```
feat(sddp): implement multi-cut strategy
fix(core): correct reservoir volume bounds validation  
docs(io): add NEWAVE HIDR.DAT format documentation
```

## License

Cobre is licensed under the [Apache License, Version 2.0](LICENSE).

## Citation

If you use Cobre in academic work, please cite:

```bibtex
@software{cobre,
  author = {Malves, Rogério J.},
  title = {Cobre: Open Infrastructure for Power System Computation},
  url = {https://github.com/cobre-rs/cobre},
  license = {Apache-2.0}
}
```
