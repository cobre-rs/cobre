//! Local (single-process) communication backend.
//!
//! `LocalBackend` is always available without any feature flags. It implements
//! [`Communicator`](crate::Communicator) and [`LocalCommunicator`](crate::LocalCommunicator)
//! with identity copy semantics for data-moving operations and no-op semantics for
//! synchronization operations, formalizing single-process execution mode:
//!
//! - `rank()` always returns `0`.
//! - `size()` always returns `1`.
//! - `allgatherv` copies `send` to `recv[displs[0]..displs[0]+counts[0]]` (identity copy).
//! - `allreduce` copies `send` to `recv` unchanged (reduction of a single operand is identity).
//! - `broadcast` is a no-op (data is already at the only rank).
//! - `barrier` is a no-op (nothing to synchronize with a single rank).
//!
//! This backend imposes zero overhead and zero external dependencies. As a zero-sized
//! type (ZST), it occupies zero bytes at runtime and has no construction cost.

use crate::{
    CommData, CommError, Communicator, LocalCommunicator, ReduceOp, SharedMemoryProvider,
    SharedRegion,
};

/// Single-process communication backend with identity collective semantics.
///
/// Zero-sized type with no runtime state. All collective operations are
/// identity copies or no-ops, compiling to zero instructions after inlining
/// in single-feature builds.
///
/// # Examples
///
/// ```rust
/// use cobre_comm::{LocalBackend, Communicator, ReduceOp};
///
/// let comm = LocalBackend;
/// assert_eq!(comm.rank(), 0);
/// assert_eq!(comm.size(), 1);
///
/// let send = vec![1.0_f64, 2.0, 3.0];
/// let mut recv = vec![0.0_f64; 3];
/// comm.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
/// assert_eq!(recv, send);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LocalBackend;

impl Communicator for LocalBackend {
    /// Copies `send` to `recv[displs[0]..displs[0]+counts[0]]`.
    ///
    /// With a single rank, `allgatherv` is an identity copy.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::InvalidBufferSize`] if:
    /// - `counts.len() != 1`
    /// - `displs.len() != 1`
    /// - `send.len() != counts[0]`
    /// - `recv.len() < displs[0] + counts[0]`
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        if counts.len() != 1 {
            return Err(CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: 1,
                actual: counts.len(),
            });
        }
        if displs.len() != 1 {
            return Err(CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: 1,
                actual: displs.len(),
            });
        }
        if send.len() != counts[0] {
            return Err(CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: counts[0],
                actual: send.len(),
            });
        }
        let required = displs[0].saturating_add(counts[0]);
        if recv.len() < required {
            return Err(CommError::InvalidBufferSize {
                operation: "allgatherv",
                expected: required,
                actual: recv.len(),
            });
        }

        recv[displs[0]..displs[0] + counts[0]].copy_from_slice(send);
        Ok(())
    }

    /// Copies `send` to `recv` unchanged.
    ///
    /// With a single rank, reduction of a single operand is the identity.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::InvalidBufferSize`] if:
    /// - `send.len() != recv.len()`
    /// - `send.len() == 0`
    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        if send.len() != recv.len() {
            return Err(CommError::InvalidBufferSize {
                operation: "allreduce",
                expected: send.len(),
                actual: recv.len(),
            });
        }
        if send.is_empty() {
            return Err(CommError::InvalidBufferSize {
                operation: "allreduce",
                expected: 1,
                actual: 0,
            });
        }

        recv.copy_from_slice(send);
        Ok(())
    }

    /// No-op for valid `root == 0`.
    ///
    /// # Errors
    ///
    /// Returns [`CommError::InvalidRoot`] if `root >= 1`.
    fn broadcast<T: CommData>(&self, _buf: &mut [T], root: usize) -> Result<(), CommError> {
        if root >= 1 {
            return Err(CommError::InvalidRoot { root, size: 1 });
        }
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code);
    }
}

/// Shared memory region backed by a heap-allocated [`Vec<T>`].
///
/// Used by backends without true intra-node shared memory (local, tcp).
/// Lifecycle phases from the `SharedRegion` trait degenerate to simple
/// `Vec` operations:
///
/// - **Allocation**: `Vec` is allocated with `count` zero-initialized elements.
/// - **Population**: [`as_mut_slice`](HeapRegion::as_mut_slice) returns the full slice.
/// - **Synchronization**: [`fence`](HeapRegion::fence) is a no-op (single process).
/// - **Read-only**: [`as_slice`](HeapRegion::as_slice) returns the full slice.
/// - **Deallocation**: inner `Vec<T>` is dropped via standard RAII.
///
/// # Thread safety
///
/// `HeapRegion<T>` is `Send + Sync` when `T: Send + Sync`, which is guaranteed
/// by the `CommData` bound.
///
/// # Examples
///
/// ```rust
/// use cobre_comm::{LocalBackend, SharedMemoryProvider, SharedRegion};
///
/// let backend = LocalBackend;
/// let mut region = backend.create_shared_region::<f64>(5).unwrap();
/// region.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!(region.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
pub struct HeapRegion<T: CommData> {
    data: Vec<T>,
}

impl<T: CommData> HeapRegion<T> {
    /// Construct a `HeapRegion` with `count` zero-initialized elements.
    ///
    /// Used by backends other than `LocalBackend` (e.g., `FerrompiBackend`)
    /// that reuse `HeapRegion` as their `Region<T>` type but cannot access the
    /// private `data` field directly.
    #[cfg(any(feature = "mpi", feature = "tcp", feature = "shm"))]
    pub(crate) fn new(count: usize) -> Self {
        Self {
            data: vec![T::default(); count],
        }
    }
}

impl<T: CommData> SharedRegion<T> for HeapRegion<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn fence(&self) -> Result<(), CommError> {
        Ok(())
    }
}

impl SharedMemoryProvider for LocalBackend {
    type Region<T: CommData> = HeapRegion<T>;

    /// Allocates a `HeapRegion` with `count` zero-initialized elements.
    ///
    /// # Errors
    ///
    /// Always returns `Ok`. Heap allocation failure follows Rust's standard behavior (abort on OOM).
    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError> {
        Ok(HeapRegion {
            data: vec![T::default(); count],
        })
    }

    /// Returns a single-rank intra-node communicator wrapping `LocalBackend`.
    ///
    /// # Errors
    ///
    /// Always returns `Ok(...)`.
    fn split_local(&self) -> Result<Box<dyn LocalCommunicator>, CommError> {
        Ok(Box::new(LocalBackend))
    }
    fn is_leader(&self) -> bool {
        true
    }
}

impl LocalCommunicator for LocalBackend {
    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }
}

impl crate::TopologyProvider for LocalBackend {
    /// Return the cached single-host, single-rank execution topology.
    ///
    /// Because `LocalBackend` is a ZST with no per-instance storage, the
    /// topology is stored in a process-wide `OnceLock` and initialized on
    /// first call. The topology is always a single host with a single rank.
    fn topology(&self) -> &crate::ExecutionTopology {
        use std::sync::OnceLock;

        use crate::BackendKind;
        use crate::topology::{ExecutionTopology, HostInfo};

        static TOPOLOGY: OnceLock<ExecutionTopology> = OnceLock::new();
        TOPOLOGY.get_or_init(|| {
            let hostname = std::env::var("HOSTNAME")
                .or_else(|_| std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()))
                .unwrap_or_else(|_| "localhost".to_string());
            ExecutionTopology {
                backend: BackendKind::Local,
                world_size: 1,
                hosts: vec![HostInfo {
                    hostname,
                    ranks: vec![0],
                }],
                mpi: None,
                slurm: None,
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::{HeapRegion, LocalBackend};
    use crate::{
        CommError, Communicator, LocalCommunicator, ReduceOp, SharedMemoryProvider, SharedRegion,
    };

    #[test]
    fn test_local_backend_is_zst() {
        assert_eq!(std::mem::size_of::<LocalBackend>(), 0);
    }

    #[test]
    fn test_local_allgatherv_identity() {
        let comm = LocalBackend;
        let send = [1.0_f64, 2.0, 3.0];
        let mut recv = [0.0_f64; 3];
        comm.allgatherv(&send, &mut recv, &[3], &[0]).unwrap();
        assert_eq!(recv, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_local_allgatherv_with_offset() {
        let comm = LocalBackend;
        let send = [7.0_f64, 8.0];
        let mut recv = [0.0_f64; 5];
        comm.allgatherv(&send, &mut recv, &[2], &[2]).unwrap();
        assert_eq!(recv, [0.0, 0.0, 7.0, 8.0, 0.0]);
    }

    #[test]
    fn test_local_allgatherv_invalid_counts_len() {
        let comm = LocalBackend;
        let send = [1.0_f64];
        let mut recv = [0.0_f64; 2];
        let err = comm
            .allgatherv(&send, &mut recv, &[1, 1], &[0])
            .unwrap_err();
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
    fn test_local_allgatherv_invalid_displs_len() {
        let comm = LocalBackend;
        let send = [1.0_f64];
        let mut recv = [0.0_f64; 2];
        let err = comm
            .allgatherv(&send, &mut recv, &[1], &[0, 0])
            .unwrap_err();
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
    fn test_local_allgatherv_send_count_mismatch() {
        let comm = LocalBackend;
        let send = [1.0_f64, 2.0];
        let mut recv = [0.0_f64; 3];
        let err = comm.allgatherv(&send, &mut recv, &[3], &[0]).unwrap_err();
        assert!(
            matches!(
                err,
                CommError::InvalidBufferSize {
                    operation: "allgatherv",
                    expected: 3,
                    actual: 2,
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_local_allgatherv_recv_too_small() {
        let comm = LocalBackend;
        let send = [1.0_f64, 2.0, 3.0];
        let mut recv = [0.0_f64; 4];
        let err = comm.allgatherv(&send, &mut recv, &[3], &[2]).unwrap_err();
        assert!(
            matches!(
                err,
                CommError::InvalidBufferSize {
                    operation: "allgatherv",
                    expected: 5,
                    actual: 4,
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_local_allreduce_identity_sum() {
        let comm = LocalBackend;
        let send = [42.0_f64, 99.0];
        let mut recv = [0.0_f64; 2];
        comm.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
        assert_eq!(recv, [42.0, 99.0]);
    }

    #[test]
    fn test_local_allreduce_identity_min() {
        let comm = LocalBackend;
        let send = [5.0_f64, 3.0, 7.0];
        let mut recv = [0.0_f64; 3];
        comm.allreduce(&send, &mut recv, ReduceOp::Min).unwrap();
        assert_eq!(recv, [5.0, 3.0, 7.0]);
    }

    #[test]
    fn test_local_allreduce_identity_max() {
        let comm = LocalBackend;
        let send = [10.0_f64, 20.0];
        let mut recv = [0.0_f64; 2];
        comm.allreduce(&send, &mut recv, ReduceOp::Max).unwrap();
        assert_eq!(recv, [10.0, 20.0]);
    }

    #[test]
    fn test_local_allreduce_buffer_mismatch() {
        let comm = LocalBackend;
        let send = [1.0_f64, 2.0, 3.0];
        let mut recv = [0.0_f64; 2];
        let err = comm.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap_err();
        assert!(
            matches!(
                err,
                CommError::InvalidBufferSize {
                    operation: "allreduce",
                    expected: 3,
                    actual: 2,
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_local_allreduce_empty() {
        let comm = LocalBackend;
        let send: [f64; 0] = [];
        let mut recv: [f64; 0] = [];
        let err = comm.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap_err();
        assert!(
            matches!(
                err,
                CommError::InvalidBufferSize {
                    operation: "allreduce",
                    expected: 1,
                    actual: 0,
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_local_broadcast_root0_noop() {
        let comm = LocalBackend;
        let mut buf = [1.0_f64, 2.0];
        let result = comm.broadcast(&mut buf, 0);
        assert!(result.is_ok());
        assert_eq!(buf, [1.0, 2.0]);
    }

    #[test]
    fn test_local_broadcast_invalid_root() {
        let comm = LocalBackend;
        let mut buf = [1.0_f64, 2.0];
        let err = comm.broadcast(&mut buf, 1).unwrap_err();
        assert!(
            matches!(err, CommError::InvalidRoot { root: 1, size: 1 }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_local_barrier_noop() {
        let comm = LocalBackend;
        assert!(Communicator::barrier(&comm).is_ok());
    }

    #[test]
    fn test_local_rank() {
        let comm = LocalBackend;
        assert_eq!(Communicator::rank(&comm), 0);
    }

    #[test]
    fn test_local_size() {
        let comm = LocalBackend;
        assert_eq!(Communicator::size(&comm), 1);
    }

    #[test]
    fn test_local_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LocalBackend>();
    }

    #[test]
    fn test_local_communicator_rank() {
        let comm = LocalBackend;
        assert_eq!(LocalCommunicator::rank(&comm), 0);
    }

    #[test]
    fn test_local_communicator_size() {
        let comm = LocalBackend;
        assert_eq!(LocalCommunicator::size(&comm), 1);
    }

    #[test]
    fn test_local_communicator_barrier_noop() {
        let comm = LocalBackend;
        assert!(LocalCommunicator::barrier(&comm).is_ok());
    }

    #[test]
    fn test_local_communicator_as_dyn() {
        let comm = LocalBackend;
        let dyn_comm: &dyn LocalCommunicator = &comm;
        assert_eq!(dyn_comm.rank(), 0);
        assert_eq!(dyn_comm.size(), 1);
        assert!(dyn_comm.barrier().is_ok());
    }

    #[test]
    fn test_heap_region_create() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(10).unwrap();
        assert_eq!(region.as_slice().len(), 10);
    }

    #[test]
    fn test_heap_region_write_read() {
        let backend = LocalBackend;
        let mut region = backend.create_shared_region::<f64>(5).unwrap();
        region
            .as_mut_slice()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(region.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_heap_region_fence_noop() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(4).unwrap();
        assert!(region.fence().is_ok());
    }

    #[test]
    fn test_heap_region_zero_count() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(0).unwrap();
        assert_eq!(region.as_slice().len(), 0);
    }

    #[test]
    fn test_local_create_shared_region() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(100).unwrap();
        assert_eq!(region.as_slice().len(), 100);
        assert!(region.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_local_split_local() {
        let backend = LocalBackend;
        let local_comm = backend.split_local().unwrap();
        assert_eq!(local_comm.rank(), 0);
        assert_eq!(local_comm.size(), 1);
    }

    #[test]
    fn test_local_is_leader() {
        let backend = LocalBackend;
        assert!(backend.is_leader());
    }

    #[test]
    fn test_heap_region_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HeapRegion<f64>>();
    }

    #[test]
    #[cfg(any(feature = "mpi", feature = "tcp", feature = "shm"))]
    fn test_heap_region_new_crate_visible() {
        let region = HeapRegion::<f64>::new(5);
        assert_eq!(region.as_slice().len(), 5);
        assert!(region.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_heap_region_lifecycle() {
        let backend = LocalBackend;
        let mut region = backend.create_shared_region::<f64>(3).unwrap();
        region.as_mut_slice().copy_from_slice(&[10.0, 20.0, 30.0]);
        region.fence().unwrap();
        assert_eq!(region.as_slice(), &[10.0, 20.0, 30.0]);
    }
}
