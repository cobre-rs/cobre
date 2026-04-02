# cobre-comm

Pluggable communication backend abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
distributed solver ecosystem.

Defines the `Communicator` and `SharedMemoryProvider` traits that decouple
distributed computations from specific communication technologies. Two backends
are implemented: `LocalBackend`, a single-process no-op that is always
available and carries zero overhead, and `FerrompiBackend`, an MPI 4.x
implementation built on the [ferrompi](https://github.com/cobre-rs/ferrompi)
crate and enabled via the `mpi` Cargo feature. The backend is selected at
build time through Cargo feature flags with an optional runtime override via
the `COBRE_COMM_BACKEND` environment variable. All dispatch over the
`Communicator` trait is static, so there is no dynamic dispatch overhead on
the hot path.

## When to Use

Depend on `cobre-comm` directly when you are writing a distributed algorithm
that needs collective communication (broadcast, reduce, scatter/gather) and
you want to test it locally without an MPI installation. Algorithm crates such
as `cobre-sddp` depend on this crate and accept a generic `Communicator`
parameter; you only need to depend here when adding a new algorithm crate or a
new backend.

## Key Types

- **`Communicator`** — the core trait for collective communication operations
  (broadcast, reduce, barrier) implemented by every backend
- **`LocalBackend`** — single-process no-op backend; always available, no
  external dependencies
- **`FerrompiBackend`** — MPI 4.x backend built on ferrompi; enabled with the
  `mpi` feature flag
- **`BackendKind`** — enum listing the available backends detected at compile
  time, used to select a backend via `create_communicator`
- **`SharedMemoryProvider`** — trait for intra-node memory-region allocation
  used by collocated processes

## Links

| Resource   | URL                                                        |
| ---------- | ---------------------------------------------------------- |
| Book       | <https://cobre-rs.github.io/cobre/crates/comm.html>        |
| API Docs   | <https://docs.rs/cobre-comm/latest/cobre_comm/>            |
| Repository | <https://github.com/cobre-rs/cobre>                        |
| CHANGELOG  | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable.

## License

Apache-2.0
