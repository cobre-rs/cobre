# ferrompi

<span class="status-alpha">alpha</span>

Safe MPI 4.x bindings for Rust, used by `cobre-comm` as the MPI communication backend. This is a separate repository at [github.com/cobre-rs/ferrompi](https://github.com/cobre-rs/ferrompi).

ferrompi provides type-safe wrappers around MPI collective operations (`allgatherv`, `allreduce`, `broadcast`, `barrier`) with RAII-managed `MPI_Init_thread` / `MPI_Finalize` lifecycle. It supports `ThreadLevel::Funneled` initialization, which matches the Cobre execution model where only the main thread issues MPI calls.

See the [ferrompi README](https://github.com/cobre-rs/ferrompi) and the [backend specification](https://cobre-rs.github.io/cobre-docs/specs/hpc/backend-ferrompi.html) for details.
