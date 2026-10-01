//! Local (single-process) communication backend.
//!
//! `LocalBackend` is a zero-sized type, always available without feature flags.
//! It implements [`Communicator`] and
//! [`LocalCommunicator`] with identity-copy semantics
//! for data-moving operations and no-op semantics for synchronization.

use crate::ExecutionTopology;
use crate::{CommData, CommError, Communicator, ReduceOp};

#[cfg(feature = "shared-memory")]
use crate::{SharedMemoryProvider, SharedRegion};

#[cfg(feature = "shared-memory")]
use crate::traits::{LocalCommKind, LocalCommunicator};

/// Single-process communication backend with identity collective semantics.
///
/// Zero-sized type with no runtime state; collectives are identity copies or
/// no-ops, compiling to zero instructions after inlining.
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
    /// Identity copy of `send` into `recv[displs[0]..displs[0]+counts[0]]`.
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

    /// Identity copy of `send` into `recv` (single-operand reduction).
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

/// Shared memory region backed by a heap-allocated [`Vec<T>`], for backends
/// without true intra-node shared memory; [`SharedRegion`] phases degenerate to
/// plain `Vec` operations and `fence` is a no-op.
#[cfg(feature = "shared-memory")]
pub struct HeapRegion<T: CommData> {
    data: Vec<T>,
}

#[cfg(feature = "shared-memory")]
impl<T: CommData> HeapRegion<T> {
    /// Construct a `HeapRegion` with `count` zero-initialized elements, for
    /// backends that reuse it as their `Region<T>` but cannot reach `data`.
    #[cfg(feature = "mpi")]
    pub(crate) fn new(count: usize) -> Self {
        Self {
            data: vec![T::default(); count],
        }
    }
}

#[cfg(feature = "shared-memory")]
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

#[cfg(feature = "shared-memory")]
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
    fn split_local(&self) -> Result<LocalCommKind, CommError> {
        Ok(LocalCommKind::Local(LocalBackend))
    }

    fn is_leader(&self) -> bool {
        true
    }
}

#[cfg(feature = "shared-memory")]
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
    /// Stored in a process-wide `OnceLock` because `LocalBackend` is a ZST with
    /// no per-instance storage to hold it.
    fn topology(&self) -> &ExecutionTopology {
        use std::sync::OnceLock;

        use crate::BackendKind;
        use crate::topology::HostInfo;

        static TOPOLOGY: OnceLock<ExecutionTopology> = OnceLock::new();
        TOPOLOGY.get_or_init(|| {
            let hostname = gethostname::gethostname().to_string_lossy().into_owned();
            let hostname = if hostname.is_empty() {
                "localhost".to_string()
            } else {
                hostname
            };
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
    use super::LocalBackend;
    use crate::{CommError, Communicator, ReduceOp};

    #[cfg(feature = "shared-memory")]
    use super::HeapRegion;
    #[cfg(feature = "shared-memory")]
    use crate::{LocalCommunicator, SharedMemoryProvider, SharedRegion};

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

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_communicator_rank() {
        let comm = LocalBackend;
        assert_eq!(LocalCommunicator::rank(&comm), 0);
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_communicator_size() {
        let comm = LocalBackend;
        assert_eq!(LocalCommunicator::size(&comm), 1);
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_communicator_barrier_noop() {
        let comm = LocalBackend;
        assert!(LocalCommunicator::barrier(&comm).is_ok());
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_communicator_as_trait_ref() {
        use crate::traits::LocalCommunicator as LC;
        let comm = LocalBackend;
        assert_eq!(LC::rank(&comm), 0);
        assert_eq!(LC::size(&comm), 1);
        assert!(LC::barrier(&comm).is_ok());
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_heap_region_create() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(10).unwrap();
        assert_eq!(region.as_slice().len(), 10);
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_heap_region_write_read() {
        let backend = LocalBackend;
        let mut region = backend.create_shared_region::<f64>(5).unwrap();
        region
            .as_mut_slice()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(region.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_heap_region_fence_noop() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(4).unwrap();
        assert!(region.fence().is_ok());
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_heap_region_zero_count() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(0).unwrap();
        assert_eq!(region.as_slice().len(), 0);
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_create_shared_region() {
        let backend = LocalBackend;
        let region = backend.create_shared_region::<f64>(100).unwrap();
        assert_eq!(region.as_slice().len(), 100);
        assert!(region.as_slice().iter().all(|&x| x == 0.0));
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_split_local() {
        let backend = LocalBackend;
        let local_comm = backend.split_local().unwrap();
        assert_eq!(local_comm.rank(), 0);
        assert_eq!(local_comm.size(), 1);
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_local_is_leader() {
        let backend = LocalBackend;
        assert!(backend.is_leader());
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_heap_region_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HeapRegion<f64>>();
    }

    #[cfg(all(feature = "shared-memory", feature = "mpi"))]
    #[test]
    fn test_heap_region_new_crate_visible() {
        let region = HeapRegion::<f64>::new(5);
        assert_eq!(region.as_slice().len(), 5);
        assert!(region.as_slice().iter().all(|&x| x == 0.0));
    }

    #[cfg(feature = "shared-memory")]
    #[test]
    fn test_heap_region_lifecycle() {
        let backend = LocalBackend;
        let mut region = backend.create_shared_region::<f64>(3).unwrap();
        region.as_mut_slice().copy_from_slice(&[10.0, 20.0, 30.0]);
        region.fence().unwrap();
        assert_eq!(region.as_slice(), &[10.0, 20.0, 30.0]);
    }
}
