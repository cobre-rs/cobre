//! MPI communication backend powered by [ferrompi](https://github.com/cobre-rs/ferrompi).
//!
//! `FerrompiBackend` implements [`Communicator`](crate::Communicator) using
//! MPI 4.x blocking collectives via the `ferrompi` crate, managing the MPI
//! environment lifecycle through an `Mpi` RAII guard. Only available with the
//! `mpi` Cargo feature.

use crate::BackendError;
use crate::BackendKind::Mpi;
use crate::CommData;
use crate::CommError;
use crate::CommError::CollectiveFailed;
use crate::CommError::InvalidBufferSize;
use crate::CommError::InvalidCommunicator;
use crate::CommError::InvalidRoot;
use crate::ExecutionTopology;
#[cfg(feature = "shared-memory")]
use crate::HeapRegion;
use crate::HostInfo;
#[cfg(feature = "shared-memory")]
use crate::LocalCommKind;
#[cfg(feature = "shared-memory")]
use crate::LocalCommKind::Ferrompi;
#[cfg(feature = "shared-memory")]
use crate::LocalCommunicator;
use crate::MpiRuntimeInfo;
use crate::ReduceOp;
use crate::ReduceOp::BitwiseOr;
use crate::ReduceOp::Max;
use crate::ReduceOp::Min;
use crate::ReduceOp::Sum;
use crate::SlurmJobInfo;

/// MPI communication backend wrapping ferrompi.
///
/// `Send + Sync` (see the SAFETY note on the `unsafe impl` blocks below).
/// Construct with [`FerrompiBackend::new`].
///
/// Field declaration order is load-bearing: Rust drops fields in reverse order,
/// so `mpi` is declared first to be dropped last — `MPI_Finalize` must run only
/// after all communicator handles are freed. Reordering finalizes prematurely.
pub struct FerrompiBackend {
    /// MPI environment RAII guard; held solely for its `Drop` side-effect.
    // Rationale: removing this field finalises MPI before the communicator
    // handles derived from it are released (see the field-order note above).
    #[allow(dead_code)]
    mpi: ferrompi::Mpi,

    /// The `MPI_COMM_WORLD` communicator handle for inter-node collectives.
    world: ferrompi::Communicator,

    /// Intra-node communicator (`MPI_Comm_split_type`) for `is_leader`.
    #[cfg(feature = "shared-memory")]
    shared: Option<ferrompi::Communicator>,

    /// World rank (0-based), cached at construction to avoid hot-path FFI calls.
    rank: usize,

    /// World size (total MPI ranks), cached at construction.
    size: usize,

    /// Execution topology, gathered once during `new` and cached.
    topology: ExecutionTopology,
}

// SAFETY: ferrompi::Mpi is !Send + !Sync to force MPI_Init/MPI_Finalize onto the
// same thread. FerrompiBackend upholds that invariant:
//   1. `new` constructs `Mpi` on the calling thread.
//   2. The backend is the sole owner of `Mpi` until drop, so single-ownership
//      bars any other thread from calling `MPI_Finalize` (via `Mpi::drop`).
//   3. Initialization uses `ThreadLevel::Funneled`, so all MPI calls must come
//      from the same thread that constructed this struct.
// All collective communication goes through `ferrompi::Communicator`, which is
// already Send + Sync (an integer handle into a C-side table).
unsafe impl Send for FerrompiBackend {}
unsafe impl Sync for FerrompiBackend {}

impl FerrompiBackend {
    /// Initialize MPI (`ThreadLevel::Funneled`) and construct a `FerrompiBackend`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InitializationFailed`] if:
    /// - `Mpi::init_thread` fails (e.g., MPI runtime not installed, already initialized).
    /// - `world.topology()` fails (e.g., allgather or broadcast error).
    /// - (`shared-memory` feature only) `world.split_shared()` fails.
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

        #[cfg(feature = "shared-memory")]
        let shared = world
            .split_shared()
            .map_err(|e| BackendError::InitializationFailed {
                backend: "mpi".to_string(),
                source: Box::new(e),
            })?;

        // Collective: every rank must reach this before returning.
        let ferrompi_topo =
            world
                .topology(&mpi)
                .map_err(|e| BackendError::InitializationFailed {
                    backend: "mpi".to_string(),
                    source: Box::new(e),
                })?;

        #[allow(clippy::cast_sign_loss)]
        let topology = ExecutionTopology {
            backend: Mpi,
            world_size: size,
            hosts: ferrompi_topo
                .hosts()
                .iter()
                .map(|h| HostInfo {
                    hostname: h.hostname.clone(),
                    ranks: h.ranks.iter().map(|&r| r as usize).collect(),
                })
                .collect(),
            mpi: Some(MpiRuntimeInfo {
                library_version: sanitize_library_version(ferrompi_topo.library_version()),
                standard_version: ferrompi_topo.standard_version().to_string(),
                thread_level: format!("{:?}", ferrompi_topo.thread_level()),
            }),
            slurm: convert_slurm_info(&ferrompi_topo),
        };

        Ok(Self {
            mpi,
            world,
            #[cfg(feature = "shared-memory")]
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
}

/// Extract a concise library identifier from `MPI_Get_library_version`.
///
/// MPI implementations return widely different formats, normalized here to a
/// single-line display string.
fn sanitize_library_version(raw: &str) -> String {
    let first_line = raw.lines().next().unwrap_or(raw).trim();

    if let Some(rest) = first_line.strip_prefix("MPICH Version:") {
        return format!("MPICH {}", rest.trim());
    }

    first_line.to_string()
}

/// With the `numa` feature enabled, reads SLURM job metadata from the topology.
/// Without the feature, always returns `None` (env-var reads are not available).
#[cfg(feature = "numa")]
fn convert_slurm_info(topo: &ferrompi::TopologyInfo) -> Option<SlurmJobInfo> {
    topo.slurm().map(|s| SlurmJobInfo {
        job_id: s.job_id.clone(),
        node_list: s.node_list.clone(),
        #[allow(clippy::cast_sign_loss)]
        cpus_per_task: s.cpus_per_task.map(|v| v as u32),
    })
}

/// Without the `numa` feature, SLURM information is not available.
#[cfg(not(feature = "numa"))]
fn convert_slurm_info(_topo: &ferrompi::TopologyInfo) -> Option<SlurmJobInfo> {
    None
}

impl crate::TopologyProvider for FerrompiBackend {
    /// Returns the cached execution topology (non-collective, allocation-free).
    fn topology(&self) -> &ExecutionTopology {
        &self.topology
    }
}

/// Intra-node communicator wrapping a ferrompi shared communicator, returned by
/// `FerrompiBackend::split_local` inside [`crate::traits::LocalCommKind::Ferrompi`].
///
/// Implements [`crate::LocalCommunicator`] only, not full [`crate::Communicator`].
#[cfg(feature = "shared-memory")]
pub struct FerrompiLocalComm(ferrompi::Communicator);

#[cfg(feature = "shared-memory")]
impl LocalCommunicator for FerrompiLocalComm {
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
    fn barrier(&self) -> Result<(), CommError> {
        self.0
            .barrier()
            .map_err(|e| map_ferrompi_error(&e, "barrier"))
    }
}

#[cfg(feature = "shared-memory")]
impl crate::SharedMemoryProvider for FerrompiBackend {
    /// Heap-fallback region type: every rank holds its own `Vec<T>`, no memory
    /// shared across ranks (true `MPI_Win` windows deferred, spec SS4.7).
    type Region<T: CommData> = HeapRegion<T>;

    /// Allocate a `HeapRegion` with `count` zero-initialized elements.
    ///
    /// # Errors
    ///
    /// Always returns `Ok`. Heap allocation failure follows Rust's standard
    /// behavior (process abort on OOM before returning `Err`).
    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError> {
        Ok(HeapRegion::new(count))
    }

    /// Create an intra-node communicator via `MPI_Comm_split_type SHARED`.
    ///
    /// Each call issues a fresh collective; call once at startup and cache.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CommError::CollectiveFailed`] if `split_shared()` fails.
    fn split_local(&self) -> Result<LocalCommKind, CommError> {
        self.world
            .split_shared()
            .map(|c| Ferrompi(FerrompiLocalComm(c)))
            .map_err(|e| map_ferrompi_error(&e, "split_local"))
    }

    /// Return whether the calling rank is the intra-node leader (local rank 0);
    /// defaults to `true` when `shared` is `None` (safe default, spec SS3.1).
    fn is_leader(&self) -> bool {
        self.shared.as_ref().is_none_or(|c| c.rank() == 0)
    }
}

/// Convert a `ferrompi::Error` to the most specific `CommError` variant,
/// following the classification in spec backend-ferrompi.md SS5.2.
fn map_ferrompi_error(e: &ferrompi::Error, operation: &'static str) -> CommError {
    match e {
        ferrompi::Error::Mpi {
            class,
            code,
            message,
            ..
        } => match class {
            ferrompi::MpiErrorClass::Comm => InvalidCommunicator,
            ferrompi::MpiErrorClass::Root => InvalidRoot {
                // Sentinels: ferrompi carries no root/size; detail is in the message.
                root: 0,
                size: 0,
            },
            ferrompi::MpiErrorClass::Buffer | ferrompi::MpiErrorClass::Count => {
                InvalidBufferSize {
                    operation,
                    // Sentinels: ferrompi carries no counts; detail is in the message.
                    expected: 0,
                    actual: 0,
                }
            }
            _ => CollectiveFailed {
                operation,
                mpi_error_code: *code,
                message: message.clone(),
            },
        },
        ferrompi::Error::InvalidBuffer => InvalidBufferSize {
            operation,
            expected: 0,
            actual: 0,
        },
        ferrompi::Error::AlreadyInitialized | ferrompi::Error::Finalized => InvalidCommunicator,
        // ThreadLevelViolation / NotSupported / Internal and any future variant
        // (`Error` is #[non_exhaustive]) carry no MPI error code; use -1.
        _ => CollectiveFailed {
            operation,
            mpi_error_code: -1,
            message: e.to_string(),
        },
    }
}

/// Map a `cobre_comm::ReduceOp` to the corresponding `ferrompi::ReduceOp`.
///
/// `ferrompi::ReduceOp::Prod` is not exposed in the Cobre trait.
fn map_reduce_op(op: ReduceOp) -> ferrompi::ReduceOp {
    match op {
        Sum => ferrompi::ReduceOp::Sum,
        Min => ferrompi::ReduceOp::Min,
        Max => ferrompi::ReduceOp::Max,
        BitwiseOr => ferrompi::ReduceOp::BitwiseOr,
    }
}

/// Convert a slice of `usize` to `Vec<i32>` (ferrompi collectives use `i32`
/// counts/displacements).
///
/// # Errors
///
/// Returns [`crate::CommError::InvalidBufferSize`] if any element in `values`
/// exceeds `i32::MAX`.
fn to_i32_vec(values: &[usize], operation: &'static str) -> Result<Vec<i32>, CommError> {
    values
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| InvalidBufferSize {
                operation,
                expected: i32::MAX as usize,
                actual: v,
            })
        })
        .collect()
}

impl crate::Communicator for FerrompiBackend {
    /// Gather variable-length data from all ranks into all ranks.
    ///
    /// # Errors
    ///
    /// - [`crate::CommError::InvalidBufferSize`] if `counts.len() != size`,
    ///   `displs.len() != size`, `send.len() != counts[rank]`, or any element
    ///   overflows `i32`.
    /// - [`crate::CommError::CollectiveFailed`] if the underlying MPI call fails.
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        if counts.len() != self.size {
            return Err(InvalidBufferSize {
                operation: "allgatherv",
                expected: self.size,
                actual: counts.len(),
            });
        }
        if displs.len() != self.size {
            return Err(InvalidBufferSize {
                operation: "allgatherv",
                expected: self.size,
                actual: displs.len(),
            });
        }
        if send.len() != counts[self.rank] {
            return Err(InvalidBufferSize {
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
    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError> {
        if send.len() != recv.len() {
            return Err(InvalidBufferSize {
                operation: "allreduce",
                expected: send.len(),
                actual: recv.len(),
            });
        }
        if send.is_empty() {
            return Err(InvalidBufferSize {
                operation: "allreduce",
                expected: 1,
                actual: 0,
            });
        }

        self.world
            .allreduce(send, recv, map_reduce_op(op))
            .map_err(|e| map_ferrompi_error(&e, "allreduce"))
    }

    /// Broadcast data from `root` rank to all other ranks.
    ///
    /// # Errors
    ///
    /// - [`crate::CommError::InvalidRoot`] if `root >= self.size`.
    /// - [`crate::CommError::CollectiveFailed`] if the underlying MPI call fails.
    fn broadcast<T: CommData>(&self, buf: &mut [T], root: usize) -> Result<(), CommError> {
        if root >= self.size {
            return Err(InvalidRoot {
                root,
                size: self.size,
            });
        }
        // root < self.size (checked above) and self.size came from a ferrompi i32,
        // so the cast cannot truncate or wrap.
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
    fn barrier(&self) -> Result<(), CommError> {
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

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_ferrompi_local_comm_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::FerrompiLocalComm>();
    }

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
            assert!(matches!(
                map_reduce_op(ReduceOp::BitwiseOr),
                ferrompi::ReduceOp::BitwiseOr
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
        fn test_map_ferrompi_error_finalized() {
            let err = map_ferrompi_error(&ferrompi::Error::Finalized, "barrier");
            assert!(
                matches!(err, CommError::InvalidCommunicator),
                "unexpected error: {err:?}"
            );
        }

        #[test]
        fn test_map_ferrompi_error_thread_level_violation() {
            let err = map_ferrompi_error(&ferrompi::Error::ThreadLevelViolation, "allreduce");
            assert!(
                matches!(
                    err,
                    CommError::CollectiveFailed {
                        operation: "allreduce",
                        mpi_error_code: -1,
                        ..
                    }
                ),
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
                operation: None,
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
                operation: None,
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
                operation: None,
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
                operation: None,
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
                operation: None,
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
