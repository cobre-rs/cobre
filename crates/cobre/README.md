# cobre

**Open infrastructure for power system computation.**

Cobre is an ecosystem of Rust crates for power system analysis and optimization.
This umbrella crate re-exports the individual components for convenience.

For most use cases, depend on the specific crates you need:

| Crate                                                           | Purpose                      |
| --------------------------------------------------------------- | ---------------------------- |
| [`cobre-core`](https://crates.io/crates/cobre-core)             | Power system data model      |
| [`cobre-io`](https://crates.io/crates/cobre-io)                 | File parsers and serializers |
| [`cobre-stochastic`](https://crates.io/crates/cobre-stochastic) | Stochastic process models    |
| [`cobre-solver`](https://crates.io/crates/cobre-solver)         | LP/MIP solver abstraction    |
| [`cobre-sddp`](https://crates.io/crates/cobre-sddp)             | SDDP algorithm               |
| [`cobre-cli`](https://crates.io/crates/cobre-cli)               | Command-line interface       |

## When to Use

Use the `cobre` umbrella crate only when you need re-exports from multiple
subcrates in a single dependency. For all other cases, add each subcrate you
actually need as a direct dependency — this keeps compile times lower and
makes the dependency graph explicit.

## Links

| Resource               | URL                                                        |
| ---------------------- | ---------------------------------------------------------- |
| Book — crates overview | <https://cobre-rs.github.io/cobre/crates/overview.html>    |
| Repository             | <https://github.com/cobre-rs/cobre>                        |
| Changelog              | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable. See the [main repository](https://github.com/cobre-rs/cobre) for the current release.

## License

Apache-2.0
