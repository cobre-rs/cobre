# cobre-core

Shared data model for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.

This crate defines the fundamental types used across all Cobre tools: buses, branches,
generators (hydro, thermal, renewable), loads, network topology, and the top-level
`System` struct. A power system described with `cobre-core` types can be used for
stochastic optimization, steady-state analysis, and any other procedure in the
ecosystem. The crate carries no solver or algorithm dependencies and enforces
declaration-order invariance so that results are identical regardless of input ordering.

## When to Use

Depend on `cobre-core` directly when you are building a new analysis tool or
algorithm that needs to consume a validated power system description without
pulling in solver or I/O logic. If you are writing test utilities or fixtures
that construct small `System` instances, `cobre-core` is the only dependency
you need.

## Key Types

- **`System`** — Immutable, fully-validated power system description built by `SystemBuilder`
- **`SystemBuilder`** — Validates and assembles all entities into a `System`
- **`Hydro`** — Hydroelectric plant with storage, spillage, and productivity parameters
- **`Thermal`** — Thermal generation unit with cost segments and operational bounds
- **`Bus`** — Network bus carrying load, deficit penalties, and connected generators
- **`GenericConstraint`** — Linear constraint over named variables for custom coupling

## Links

| Resource   | URL                                                      |
| ---------- | -------------------------------------------------------- |
| Book       | https://cobre-rs.github.io/cobre/crates/core.html        |
| API Docs   | https://docs.rs/cobre-core/latest/cobre_core/            |
| Repository | https://github.com/cobre-rs/cobre                        |
| CHANGELOG  | https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md |

## Status

Alpha — API is functional but not yet stable.

## License

Apache-2.0
