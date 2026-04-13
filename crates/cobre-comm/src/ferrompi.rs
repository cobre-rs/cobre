//! MPI communication backend powered by [ferrompi](https://github.com/cobre-rs/ferrompi).
//!
//! `FerrompiBackend` implements [`Communicator`](crate::Communicator) using
//! MPI 4.x collective operations via the `ferrompi` crate. It is only available
//! when the `mpi` Cargo feature is enabled.
//!
//! Key characteristics:
//!
//! - Wraps `ferrompi::Communicator` for rank/size queries and collective operations.
//! - Supports persistent collectives (MPI 4.x) for iterative algorithms such as
//!   the forward/backward pass, reducing per-call setup overhead by 10–30 %.
//! - Nonblocking collectives allow overlap of communication with local LP solves.
//!
//! # Feature gate
//!
//! This module is compiled only when `features = ["mpi"]` is specified in
//! `Cargo.toml` (which activates the `ferrompi` optional dependency).

use crate::BackendError;

/// MPI communication backend wrapping ferrompi v0.3.
///
/// Holds the MPI environment handle (`Mpi`), the world communicator, and the
/// intra-node shared communicator. Field declaration order is significant:
/// Rust drops fields in reverse declaration order, so `mpi` is declared first
/// to ensure it is dropped last — after the `Communicator` handles are freed —
/// which satisfies the MPI requirement that `MPI_Finalize` is called only after
/// all communicators have been freed.
///
/// # Thread safety
///
/// `FerrompiBackend` is `Send + Sync`. See the safety note on the `unsafe impl`
/// blocks below for the full rationale.
///
/// # Initialization
///
/// Use [`FerrompiBackend::new`] to initialize MPI and obtain a ready-to-use
/// backend. `new` calls `MPI_Init_thread` with `ThreadLevel::Funneled`, matching
/// the Cobre training loop's model where only the main thread makes MPI calls.
///
/// # Drop
///
/// When `FerrompiBackend` is dropped:
/// 1. `shared` communicator is freed.
/// 2. `world` communicator handle goes out of scope.
/// 3. `mpi` RAII guard calls `MPI_Finalize`.
pub struct FerrompiBackend {
    /// MPI environment RAII guard. Declared first so it is dropped last
    /// (Rust drops in reverse field order), ensuring `MPI_Finalize` is called
    /// only after all communicator handles have been released.
    ///
    /// This field is never read after construction — it is held solely for its
    /// `Drop` side-effect. The suppression is intentional: removing the field
    /// would cause `MPI_Finalize` to be called prematurely.
    #[allow(dead_code)]
    mpi: ferrompi::Mpi,

    /// The `MPI_COMM_WORLD` communicator handle.
    ///
    /// Used for all inter-node collective operations (allreduce, allgatherv,
    /// broadcast, barrier) during distributed execution.
    world: ferrompi::Communicator,

    /// Optional intra-node communicator obtained from `MPI_Comm_split_type`.
    ///
    /// `Some` when this is the top-level backend created by [`FerrompiBackend::new`].
    /// `None` is reserved for sub-communicator instances returned by a future
    /// `split_local` implementation, which represent intra-node
    /// ranks and do not own a shared split.
    shared: Option<ferrompi::Communicator>,

    /// Cached world rank (0-based).
    ///
    /// Cached at construction to avoid repeated FFI calls on the hot path.
    /// Rank is invariant for the lifetime of the backend.
    rank: usize,

    /// Cached world size (total number of MPI ranks).
    ///
    /// Cached at construction for the same reason as `rank`.
    size: usize,

    /// Cached execution topology gathered during initialization.
    ///
    /// Collected once via the collective `world.topology(&mpi)` call during
    /// `FerrompiBackend::new` and cached here. All subsequent queries are
    /// non-collective and allocation-free.
    topology: crate::ExecutionTopology,
}

// SAFETY: ferrompi::Mpi is !Send + !Sync because PhantomData<*const ()> opts out
// of both markers. This restriction exists to ensure that MPI_Init and MPI_Finalize
// are called on the same thread. FerrompiBackend preserves this invariant:
//
//   1. `FerrompiBackend::new` constructs `Mpi` on the calling thread.
//   2. `FerrompiBackend` is the sole owner of `Mpi` and holds it until drop.
//   3. Rust's single-ownership model prevents any other thread from calling
//      `MPI_Finalize` (via `Mpi::drop`) concurrently.
//   4. The Cobre training loop uses ThreadLevel::Funneled, guaranteeing that all
//      MPI calls are made from the main thread — the same thread that constructed
//      this struct.
//
// All actual collective communication goes through `ferrompi::Communicator`, which
// is already `Send + Sync` (it wraps an integer handle into a C-side table).
// Therefore it is sound to implement Send and Sync for FerrompiBackend.
unsafe impl Send for FerrompiBackend {}
unsafe impl Sync for FerrompiBackend {}

impl FerrompiBackend {
    /// Initialize MPI and construct a `FerrompiBackend`.
    ///
    /// Follows the four-step initialization sequence:
    ///
    /// 1. Initialize MPI with `ThreadLevel::Funneled` via `Mpi::init_thread`.
    /// 2. Obtain the world communicator and cache rank and size.
    /// 3. Create the intra-node shared communicator via `world.split_shared()`.
    /// 4. Gather and cache the execution topology via the collective `world.topology(&mpi)`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InitializationFailed`] if:
    /// - `Mpi::init_thread` fails (e.g., MPI runtime not installed, already initialized).
    /// - `world.split_shared()` fails (e.g., MPI communicator split error).
    /// - `world.topology()` fails (e.g., allgather or broadcast error).
    pub fn new() -> Result<Self, BackendError> {
        let mpi = ferrompi::Mpi::init_thread(ferrompi::ThreadLevel::Funneled).map_err(|e| {
            BackendError::InitializationFailed {
                backend: "mpi".to_string(),
                source: Box::new(e),
            }
        })?;

        let world = mpi.world();
        #[allow(clippy::cast_sign_loss)]
        let rank = world.rank() as usize;
        #[allow(clippy::cast_sign_loss)]
        let size = world.size() as usize;

        let shared = world
            .split_shared()
            .map_err(|e| BackendError::InitializationFailed {
                backend: "mpi".to_string(),
                source: Box::new(e),
            })?;

        // Collective: all ranks gather topology. Must be called before returning
        // so all ranks participate (collective operation).
        let ferrompi_topo =
            world
                .topology(&mpi)
                .map_err(|e| BackendError::InitializationFailed {
                    backend: "mpi".to_string(),
                    source: Box::new(e),
                })?;

        #[allow(clippy::cast_sign_loss)]
        let topology = crate::ExecutionTopology {
            backend: crate::BackendKind::Mpi,
            world_size: size,
            hosts: ferrompi_topo
                .hosts()
                .iter()
                .map(|h| crate::HostInfo {
                    hostname: h.hostname.clone(),
                    ranks: h.ranks.iter().map(|&r| r as usize).collect(),
                })
                .collect(),
            mpi: Some(crate::MpiRuntimeInfo {
                library_version: sanitize_library_version(ferrompi_topo.library_version()),
                standard_version: ferrompi_topo.standard_version().to_string(),
                thread_level: format!("{:?}", ferrompi_topo.thread_level()),
            }),
            slurm: convert_slurm_info(&ferrompi_topo),
        };

        Ok(Self {
            mpi,
            world,
            shared: Some(shared),
            rank,
            size,
            topology,
        })
    }

    /// Returns the cached world rank (0-based).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the cached world size (total number of MPI ranks).
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns a reference to the world communicator.
    // Reserved for use by the `SharedMemoryProvider` trait implementation.
    #[allow(dead_code)]
    pub(crate) fn world(&self) -> &ferrompi::Communicator {
        &self.world
    }

    /// Returns a reference to the intra-node shared communicator, if present.
    // Reserved for use by the `SharedMemoryProvider` trait implementation.
    #[allow(dead_code)]
    pub(crate) fn shared(&self) -> Option<&ferrompi::Communicator> {
        self.shared.as_ref()
    }
}

/// Convert SLURM metadata from `ferrompi::TopologyInfo` to `crate::SlurmJobInfo`.
///
/// Extract a concise library identifier from `MPI_Get_library_version`.
///
/// MPI implementations return widely different formats:
/// - **Open MPI**: `"Open MPI v4.1.6"` (already clean)
/// - **MPICH**: `"MPICH Version:      4.3.2\nMPICH Release date: ...\n..."` (multi-line)
/// - **Intel MPI**: `"Intel(R) MPI Library 2021.6 ..."` (single line, long)
///
/// This function extracts the implementation name and version as a single-line
/// string suitable for display and metadata JSON. For MPICH, it parses
/// `"MPICH Version: X.Y.Z"` from the first line. For other implementations,
/// it takes only the first line and trims it.
fn sanitize_library_version(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or(raw).trim();

    // MPICH format: "MPICH Version:      4.3.2"
    if let Some(rest) = first_line.strip_prefix("MPICH Version:") {
        return format!("MPICH {}", rest.trim());
    }

    first_line.to_string()
}

/// With the `numa` feature enabled, reads SLURM job metadata from the topology.
/// Without the feature, always returns `None` (env-var reads are not available).
#[cfg(feature = "numa")]
fn convert_slurm_info(topo: &ferrompi::TopologyInfo) -> Option<crate::SlurmJobInfo> {
    topo.slurm().map(|s| crate::SlurmJobInfo {
        job_id: s.job_id.clone(),
        node_list: s.node_list.clone(),
        #[allow(clippy::cast_sign_loss)]
        cpus_per_task: s.cpus_per_task.map(|v| v as u32),
    })
}

/// Without the `numa` feature, SLURM information is not available.
#[cfg(not(feature = "numa"))]
fn convert_slurm_info(_topo: &ferrompi::TopologyInfo) -> Option<crate::SlurmJobInfo> {
    None
}

#[cfg(feature = "mpi")]
impl crate::TopologyProvider for FerrompiBackend {
    /// Returns the cached execution topology gathered during [`FerrompiBackend::new`].
    ///
    /// Non-collective and allocation-free. The topology was collected once via
    /// `world.topology(&mpi)` during initialization.
    fn topology(&self) -> &crate::ExecutionTopology {
        &self.topology
    }
}

/// Intra-node communicator wrapping a ferrompi shared communicator.
///
/// Implements [`crate::LocalCommunicator`] only (not full [`crate::Communicator`]).
/// Returned by [`FerrompiBackend::split_local`] as `Box<dyn LocalCommunicator>`.
///
/// The concrete type is not re-exported — it is an implementation detail of
/// `FerrompiBackend`. Callers receive a `Box<dyn LocalCommunicator>` and are
/// not aware of the underlying type.
///
/// # Thread safety
///
/// `FerrompiLocalComm` is `Send + Sync` because `ferrompi::Communicator` is
/// already `Send + Sync` (it wraps an integer handle into a C-side table).
/// No unsafe impl is needed.
#[cfg(feature = "mpi")]
struct FerrompiLocalComm(ferrompi::Communicator);

#[cfg(feature = "mpi")]
impl crate::LocalCommunicator for FerrompiLocalComm {
    fn rank(&self) -> usize {
        #[allow(clippy::cast_sign_loss)]
        {
            self.0.rank() as usize
        }
    }

    fn size(&self) -> usize {
        #[allow(clippy::cast_sign_loss)]
        {
            self.0.size() as usize
        }
    }

    /// Block until all intra-node ranks have called barrier.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::CollectiveFailed`] if the underlying MPI
    /// barrier call fails.
    fn barrier(&self) -> Result<(), crate::CommError> {
        self.0
            .barrier()
            .map_err(|e| map_ferrompi_error(&e, "barrier"))
    }
}

#[cfg(feature = "mpi")]
impl crate::SharedMemoryProvider for FerrompiBackend {
    /// Heap-fallback region type.
    ///
    /// Per spec SS4.7, true `MPI_Win` shared windows are deferred to post-profiling.
    /// The minimal viable phase uses `HeapRegion<T>` on all ranks — each rank holds
    /// its own `Vec<T>` copy with no memory shared across ranks.
    type Region<T: crate::CommData> = crate::HeapRegion<T>;

    /// Allocate a `HeapRegion` with `count` zero-initialized elements.
    ///
    /// # Errors
    ///
    /// Always returns `Ok`. Heap allocation failure follows Rust's standard
    /// behavior (process abort on OOM before returning `Err`).
    fn create_shared_region<T: crate::CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, crate::CommError> {
        Ok(crate::local::HeapRegion::new(count))
    }

    /// Create an intra-node communicator via `MPI_Comm_split_type SHARED`.
    ///
    /// Calls `self.world.split_shared()` to obtain a communicator containing
    /// only the ranks that share the same physical node as the calling rank,
    /// then wraps it in `FerrompiLocalComm` and returns it as
    /// `Box<dyn LocalCommunicator>`.
    ///
    /// Each call to `split_local` issues a new `MPI_Comm_split_type` collective.
    /// Callers should call this once during startup and cache the result.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::CollectiveFailed`] if `split_shared()` fails.
    fn split_local(&self) -> Result<Box<dyn crate::LocalCommunicator>, crate::CommError> {
        self.world
            .split_shared()
            .map(|c| Box::new(FerrompiLocalComm(c)) as Box<dyn crate::LocalCommunicator>)
            .map_err(|e| map_ferrompi_error(&e, "split_local"))
    }

    /// Return whether the calling rank is the intra-node leader (local rank 0).
    ///
    /// Returns `self.shared.as_ref().map(|c| c.rank() == 0).unwrap_or(true)`.
    /// When `shared` is `None` (should not occur for a fully initialized backend),
    /// returns `true` as a safe default per spec SS3.1.
    fn is_leader(&self) -> bool {
        self.shared.as_ref().is_none_or(|c| c.rank() == 0)
    }
}

/// Convert a `ferrompi::Error` to the most specific `CommError` variant.
///
/// Used by all [`crate::Communicator`] trait method implementations on
/// [`FerrompiBackend`]. The mapping follows the error classification table in
/// the spec (backend-ferrompi.md SS5.2):
///
/// | ferrompi variant              | `CommError` variant                                              |
/// |-------------------------------|------------------------------------------------------------------|
/// | `Error::Mpi { class: Comm }`  | `InvalidCommunicator`                                            |
/// | `Error::Mpi { class: Root }`  | `InvalidRoot { root: 0, size: 0 }` (sentinel; message has detail)|
/// | `Error::Mpi { class: Buffer \| Count }` | `InvalidBufferSize { expected: 0, actual: 0 }`          |
/// | `Error::Mpi { class: _ }`     | `CollectiveFailed` with the MPI error code and message           |
/// | `Error::InvalidBuffer`        | `InvalidBufferSize { expected: 0, actual: 0 }`                   |
/// | `Error::AlreadyInitialized`   | `InvalidCommunicator`                                            |
/// | `Error::NotSupported(_)`      | `CollectiveFailed { mpi_error_code: -1, .. }`                    |
/// | `Error::Internal(_)`          | `CollectiveFailed { mpi_error_code: -1, .. }`                    |
#[cfg(feature = "mpi")]
fn map_ferrompi_error(e: &ferrompi::Error, operation: &'static str) -> crate::CommError {
    match e {
        ferrompi::Error::Mpi {
            class,
            code,
            message,
        } => match class {
            ferrompi::MpiErrorClass::Comm => crate::CommError::InvalidCommunicator,
            ferrompi::MpiErrorClass::Root => crate::CommError::InvalidRoot {
                // ferrompi::Error does not carry root/size context; use sentinel
                // values. The message string provides the diagnostic detail.
                root: 0,
                size: 0,
            },
            ferrompi::MpiErrorClass::Buffer | ferrompi::MpiErrorClass::Count => {
                crate::CommError::InvalidBufferSize {
                    operation,
                    // ferrompi::Error does not carry expected/actual counts;
                    // use sentinel values. The message string provides detail.
                    expected: 0,
                    actual: 0,
                }
            }
            _ => crate::CommError::CollectiveFailed {
                operation,
                mpi_error_code: *code,
                message: message.clone(),
            },
        },
        ferrompi::Error::InvalidBuffer => crate::CommError::InvalidBufferSize {
            operation,
            expected: 0,
            actual: 0,
        },
        ferrompi::Error::AlreadyInitialized => crate::CommError::InvalidCommunicator,
        // NotSupported and Internal: no MPI error code available, use -1.
        _ => crate::CommError::CollectiveFailed {
            operation,
            mpi_error_code: -1,
            message: e.to_string(),
        },
    }
}

/// Map a `cobre_comm::ReduceOp` to the corresponding `ferrompi::ReduceOp`.
///
/// The mapping is one-to-one for the three variants that Cobre currently uses.
/// `ferrompi::ReduceOp::Prod` is not exposed in the Cobre trait.
#[cfg(feature = "mpi")]
fn map_reduce_op(op: crate::ReduceOp) -> ferrompi::ReduceOp {
    match op {
        crate::ReduceOp::Sum => ferrompi::ReduceOp::Sum,
        crate::ReduceOp::Min => ferrompi::ReduceOp::Min,
        crate::ReduceOp::Max => ferrompi::ReduceOp::Max,
    }
}

/// Convert a slice of `usize` values to a `Vec<i32>`.
///
/// Returns `Err(CommError::InvalidBufferSize)` if any value exceeds `i32::MAX`,
/// because ferrompi's collective APIs use `i32` for counts and displacements.
///
/// # Errors
///
/// Returns [`crate::CommError::InvalidBufferSize`] if any element in `values`
/// exceeds `i32::MAX` (2 147 483 647).
#[cfg(feature = "mpi")]
fn to_i32_vec(values: &[usize], operation: &'static str) -> Result<Vec<i32>, crate::CommError> {
    values
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| crate::CommError::InvalidBufferSize {
                operation,
                expected: i32::MAX as usize,
                actual: v,
            })
        })
        .collect()
}

#[cfg(feature = "mpi")]
impl crate::Communicator for FerrompiBackend {
    /// Gather variable-length data from all ranks into all ranks.
    ///
    /// # Errors
    ///
    /// - [`crate::CommError::InvalidBufferSize`] if `counts.len() != size`,
    ///   `displs.len() != size`, `send.len() != counts[rank]`, or any element
    ///   overflows `i32`.
    /// - [`crate::CommError::CollectiveFailed`] if the underlying MPI call fails.
    fn allgatherv<T: crate::CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), crate::CommError> {
        if counts.len() != self.size {
            return Err(crate::CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: self.size,
                actual: counts.len(),
            });
        }
        if displs.len() != self.size {
            return Err(crate::CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: self.size,
                actual: displs.len(),
            });
        }
        if send.len() != counts[self.rank] {
            return Err(crate::CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: counts[self.rank],
                actual: send.len(),
            });
        }
        let i32_counts = to_i32_vec(counts, "allgatherv")?;
        let i32_displs = to_i32_vec(displs, "allgatherv")?;
        self.world
            .allgatherv(send, recv, &i32_counts, &i32_displs)
            .map_err(|e| map_ferrompi_error(&e, "allgatherv"))
    }

    /// Reduce data element-wise from all ranks, with the result on all ranks.
    ///
    /// # Errors
    ///
    /// - [`crate::CommError::InvalidBufferSize`] if `send.len() != recv.len()`
    ///   or `send.is_empty()`.
    /// - [`crate::CommError::CollectiveFailed`] if the underlying MPI call fails.
    fn allreduce<T: crate::CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: crate::ReduceOp,
    ) -> Result<(), crate::CommError> {
        if send.len() != recv.len() {
            return Err(crate::CommError::InvalidBufferSize {
                operation: "allreduce",
                expected: send.len(),
                actual: recv.len(),
            });
        }
        if send.is_empty() {
            return Err(crate::CommError::InvalidBufferSize {
                operation: "allreduce",
                expected: 1,
                actual: 0,
            });
        }
        let mpi_op = map_reduce_op(op);
        self.world
            .allreduce(send, recv, mpi_op)
            .map_err(|e| map_ferrompi_error(&e, "allreduce"))
    }

    /// Broadcast data from `root` rank to all other ranks.
    ///
    /// # Errors
    ///
    /// - [`crate::CommError::InvalidRoot`] if `root >= self.size`.
    /// - [`crate::CommError::CollectiveFailed`] if the underlying MPI call fails.
    fn broadcast<T: crate::CommData>(
        &self,
        buf: &mut [T],
        root: usize,
    ) -> Result<(), crate::CommError> {
        if root >= self.size {
            return Err(crate::CommError::InvalidRoot {
                root,
                size: self.size,
            });
        }
        // root < self.size is guaranteed above; self.size fits in i32 because it
        // was obtained from ferrompi::Communicator::size() which returns i32.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let root_i32 = root as i32;
        self.world
            .broadcast(buf, root_i32)
            .map_err(|e| map_ferrompi_error(&e, "broadcast"))
    }

    /// Block until all ranks have called barrier.
    ///
    /// # Errors
    ///
    /// - [`crate::CommError::CollectiveFailed`] if the underlying MPI barrier fails.
    fn barrier(&self) -> Result<(), crate::CommError> {
        self.world
            .barrier()
            .map_err(|e| map_ferrompi_error(&e, "barrier"))
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn abort(&self, error_code: i32) -> ! {
        self.world.abort(error_code)
    }
}

#[cfg(test)]
mod tests {
    use super::FerrompiBackend;

    #[test]
    fn test_ferrompi_backend_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FerrompiBackend>();
    }

    #[test]
    fn sanitize_mpich_multiline() {
        let raw = "MPICH Version:      4.3.2\n\
                    MPICH Release date: Mon Oct  6 11:14:20 AM CDT 2025\n\
                    MPICH ABI:          17:2:5\n\
                    MPICH Device:       ch4:ofi";
        assert_eq!(super::sanitize_library_version(raw), "MPICH 4.3.2");
    }

    #[test]
    fn sanitize_openmpi_clean() {
        assert_eq!(
            super::sanitize_library_version("Open MPI v4.1.6"),
            "Open MPI v4.1.6"
        );
    }

    #[test]
    fn sanitize_intel_mpi() {
        let raw = "Intel(R) MPI Library 2021.6 for Linux* OS";
        assert_eq!(super::sanitize_library_version(raw), raw);
    }

    #[test]
    fn sanitize_empty() {
        assert_eq!(super::sanitize_library_version(""), "");
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn test_ferrompi_local_comm_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::FerrompiLocalComm>();
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn test_ferrompi_local_comm_is_object_safe() {
        fn assert_object_safe(_comm: &dyn crate::LocalCommunicator) {}
        let _ = assert_object_safe as fn(&dyn crate::LocalCommunicator);
    }

    #[cfg(feature = "mpi")]
    mod mpi_helpers {
        use super::super::{map_ferrompi_error, map_reduce_op, to_i32_vec};
        use crate::{CommError, ReduceOp};

        #[test]
        fn test_map_reduce_op_exhaustive() {
            assert!(matches!(
                map_reduce_op(ReduceOp::Sum),
                ferrompi::ReduceOp::Sum
            ));
            assert!(matches!(
                map_reduce_op(ReduceOp::Min),
                ferrompi::ReduceOp::Min
            ));
            assert!(matches!(
                map_reduce_op(ReduceOp::Max),
                ferrompi::ReduceOp::Max
            ));
        }

        #[test]
        fn test_to_i32_vec_valid() {
            let result = to_i32_vec(&[0, 1, 100], "test").expect("valid values should convert");
            assert_eq!(result, vec![0i32, 1, 100]);
        }

        #[test]
        fn test_to_i32_vec_overflow() {
            let overflow = usize::try_from(i32::MAX).expect("i32::MAX fits in usize") + 1;
            let err =
                to_i32_vec(&[overflow], "allgatherv").expect_err("overflow should return error");
            assert!(
                matches!(
                    err,
                    CommError::InvalidBufferSize {
                        operation: "allgatherv",
                        ..
                    }
                ),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_to_i32_vec_empty() {
            let result = to_i32_vec(&[], "test").expect("empty slice should convert");
            assert!(result.is_empty());
        }

        #[test]
        fn test_map_ferrompi_error_invalid_buffer() {
            let err = map_ferrompi_error(&ferrompi::Error::InvalidBuffer, "allreduce");
            assert!(
                matches!(
                    err,
                    CommError::InvalidBufferSize {
                        operation: "allreduce",
                        ..
                    }
                ),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_already_initialized() {
            let err = map_ferrompi_error(&ferrompi::Error::AlreadyInitialized, "barrier");
            assert!(
                matches!(err, CommError::InvalidCommunicator),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_internal() {
            let err = map_ferrompi_error(
                &ferrompi::Error::Internal("internal msg".into()),
                "allgatherv",
            );
            assert!(
                matches!(
                    err,
                    CommError::CollectiveFailed {
                        operation: "allgatherv",
                        mpi_error_code: -1,
                        ..
                    }
                ),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_not_supported() {
            let err = map_ferrompi_error(&ferrompi::Error::NotSupported("op".into()), "broadcast");
            assert!(
                matches!(
                    err,
                    CommError::CollectiveFailed {
                        operation: "broadcast",
                        mpi_error_code: -1,
                        ..
                    }
                ),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_mpi_comm_class() {
            let mpi_err = ferrompi::Error::Mpi {
                class: ferrompi::MpiErrorClass::Comm,
                code: 5,
                message: "invalid comm".into(),
            };
            let err = map_ferrompi_error(&mpi_err, "barrier");
            assert!(
                matches!(err, CommError::InvalidCommunicator),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_mpi_root_class() {
            let mpi_err = ferrompi::Error::Mpi {
                class: ferrompi::MpiErrorClass::Root,
                code: 8,
                message: "invalid root".into(),
            };
            let err = map_ferrompi_error(&mpi_err, "broadcast");
            assert!(
                matches!(err, CommError::InvalidRoot { root: 0, size: 0 }),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_mpi_buffer_class() {
            let mpi_err = ferrompi::Error::Mpi {
                class: ferrompi::MpiErrorClass::Buffer,
                code: 1,
                message: "bad buffer".into(),
            };
            let err = map_ferrompi_error(&mpi_err, "allgatherv");
            assert!(
                matches!(
                    err,
                    CommError::InvalidBufferSize {
                        operation: "allgatherv",
                        ..
                    }
                ),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_mpi_count_class() {
            let mpi_err = ferrompi::Error::Mpi {
                class: ferrompi::MpiErrorClass::Count,
                code: 2,
                message: "bad count".into(),
            };
            let err = map_ferrompi_error(&mpi_err, "allgatherv");
            assert!(
                matches!(
                    err,
                    CommError::InvalidBufferSize {
                        operation: "allgatherv",
                        ..
                    }
                ),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_mpi_other_class() {
            let mpi_err = ferrompi::Error::Mpi {
                class: ferrompi::MpiErrorClass::Rank,
                code: 6,
                message: "bad rank".into(),
            };
            let err = map_ferrompi_error(&mpi_err, "allreduce");
            assert!(
                matches!(
                    err,
                    CommError::CollectiveFailed {
                        operation: "allreduce",
                        mpi_error_code: 6,
                        ..
                    }
                ),
                "unexpected error: {err:?}"
            );
        }
    }
}
