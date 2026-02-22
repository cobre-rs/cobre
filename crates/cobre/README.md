# cobre

**Open infrastructure for power system computation.**

Cobre is an ecosystem of Rust crates for power system analysis and optimization. This umbrella crate re-exports the individual components for convenience.

For most use cases, depend on the specific crates you need:

| Crate                                                           | Purpose                      |
| --------------------------------------------------------------- | ---------------------------- |
| [`cobre-core`](https://crates.io/crates/cobre-core)             | Power system data model      |
| [`cobre-io`](https://crates.io/crates/cobre-io)                 | File parsers and serializers |
| [`cobre-stochastic`](https://crates.io/crates/cobre-stochastic) | Stochastic process models    |
| [`cobre-solver`](https://crates.io/crates/cobre-solver)         | LP/MIP solver abstraction    |
| [`cobre-sddp`](https://crates.io/crates/cobre-sddp)             | SDDP algorithm               |
| [`cobre-cli`](https://crates.io/crates/cobre-cli)               | Command-line interface       |

## Status

**Experimental** — this is a name reservation. The first functional release will be `0.1.0`.

See the [main repository](https://github.com/cobre-rs/cobre) for the full roadmap.

## License

Apache-2.0
