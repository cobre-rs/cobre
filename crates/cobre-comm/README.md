# cobre-comm

Pluggable communication backend abstraction for the [Cobre](https://github.com/cobre-rs/cobre) distributed SDDP solver.

Defines the `Communicator` and `SharedMemoryProvider` traits used by `cobre-sddp`, with feature-gated backends for MPI (via ferrompi), TCP, POSIX shared memory, and single-process local execution.

## Status

**Experimental** — this is a name reservation. The first functional release will be `0.1.0`.

See the [main repository](https://github.com/cobre-rs/cobre) for the full roadmap.

## License

Apache-2.0
