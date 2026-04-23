//! Type definitions for the cobre-comm abstraction layer.
//!
//! This module provides the shared types used across all communication backends:
//!
//! - [`ReduceOp`] — enumeration of supported reduction operations (sum, min, max)
//!   passed to `Communicator::allreduce`.
//! - [`CommError`] — top-level error type for the communicator API, covering
//!   collective operation errors, buffer size mismatches, and shared memory failures.
//! - [`BackendError`] — error type for backend selection and initialization failures.

/// Element-wise reduction operations for `allreduce`.
///
/// These map directly to MPI reduction operations used during distributed execution.
/// The `Sum` and `Min` variants are the two operations required by the backward
/// pass: `MPI_SUM` for upper bound statistics and `MPI_MIN` for the lower bound.
/// Because MPI may not support mixed reduction operations in a single `allreduce`
/// call, the training loop issues two separate calls — one with [`ReduceOp::Min`]
/// for the lower bound scalar and one with [`ReduceOp::Sum`] for the remaining
/// statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise summation.
    ///
    /// Used for upper bound statistics: total cost sum, sum of squares, and
    /// trajectory count aggregated across all ranks after the forward pass.
    Sum,

    /// Element-wise minimum.
    ///
    /// Used for lower bound aggregation: the minimum first-stage LP objective
    /// across all ranks is the global lower bound for the current iteration.
    Min,

    /// Element-wise maximum.
    ///
    /// Reserved for future use, for example maximum per-rank solve time for
    /// load balance diagnostics.
    Max,

    /// Element-wise bitwise OR.
    ///
    /// Used for bitmap reductions where any rank observing a condition should
    /// set the corresponding bit in the global result. Semantically equivalent
    /// to `MPI_BOR`. OR is commutative and associative, so the result is
    /// deterministic under any rank count and message-ordering.
    ///
    /// In the MPI backend this is implemented via `allgatherv` followed by a
    /// local fold, because ferrompi v0.3 does not yet expose `MPI_BOR`.
    /// In the local backend (single rank) the identity copy is correct.
    ///
    /// Used for the per-row `active_window` sliding-window bitmap reduction
    /// across ranks.
    BitwiseOr,
}

/// Errors that can occur during collective communication or shared memory operations.
///
/// This type is returned by all fallible methods on `Communicator`
/// and `SharedMemoryProvider`.
#[derive(Debug, thiserror::Error)]
pub enum CommError {
    /// An MPI collective operation failed at the library level.
    ///
    /// Contains the MPI error code and a human-readable description.
    #[error(
        "collective operation '{operation}' failed with MPI error code {mpi_error_code}: {message}"
    )]
    CollectiveFailed {
        /// Name of the collective operation that failed (e.g., `"allgatherv"`).
        operation: &'static str,
        /// MPI error code returned by the MPI library.
        mpi_error_code: i32,
        /// Human-readable description of the failure.
        message: String,
    },

    /// The buffer sizes provided to a collective operation are inconsistent.
    ///
    /// For example, `recv.len() < sum(counts)` in `allgatherv`, or
    /// `send.len() != recv.len()` in `allreduce`.
    #[error("invalid buffer size for '{operation}': expected {expected} elements, got {actual}")]
    InvalidBufferSize {
        /// Name of the collective operation (e.g., `"allreduce"`).
        operation: &'static str,
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length supplied by the caller.
        actual: usize,
    },

    /// The `root` rank argument is out of range (`root >= size()`).
    #[error("invalid root rank {root}: communicator has only {size} rank(s)")]
    InvalidRoot {
        /// The out-of-range root rank value provided by the caller.
        root: usize,
        /// Total number of ranks in the communicator.
        size: usize,
    },

    /// The communicator has been finalized or is in an invalid state.
    ///
    /// This typically occurs if `MPI_Finalize` has been called before all
    /// collective operations have completed.
    #[error("communicator is in an invalid state (MPI may have been finalized)")]
    InvalidCommunicator,

    /// A shared memory allocation request was rejected by the OS.
    ///
    /// This can occur if the requested size exceeds system shared memory limits
    /// (`/proc/sys/kernel/shmmax` on Linux), if the process lacks permissions for
    /// shared memory operations, or if the system is out of shared memory resources.
    #[error("shared memory allocation of {requested_bytes} bytes failed: {message}")]
    AllocationFailed {
        /// Number of bytes that were requested.
        requested_bytes: usize,
        /// Human-readable description of why the allocation failed.
        message: String,
    },
}

/// Errors that can occur during backend selection and initialization.
///
/// This type is returned by the factory function `create_communicator` when the
/// requested backend cannot be selected or initialized.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The requested backend is not compiled into this binary.
    ///
    /// The user or environment requested a backend that was not enabled via Cargo
    /// feature flags at compile time.
    #[error(
        "communication backend '{requested}' is not available in this build (available: {available})",
        available = available.join(", ")
    )]
    BackendNotAvailable {
        /// The backend name that was requested (e.g., `"mpi"`).
        requested: String,
        /// List of backend names that are compiled into this binary.
        available: Vec<String>,
    },

    /// The requested backend name is not recognized.
    ///
    /// The value set in `COBRE_COMM_BACKEND` does not match any known backend name.
    #[error(
        "unknown communication backend '{requested}' (known backends: {available})",
        available = available.join(", ")
    )]
    InvalidBackend {
        /// The unrecognized backend name that was requested.
        requested: String,
        /// List of all known backend names (compiled in or not).
        available: Vec<String>,
    },

    /// The backend initialization failed.
    ///
    /// The backend was correctly selected but failed to initialize, for example
    /// because the MPI runtime is not installed, the TCP coordinator is unreachable,
    /// or the shared memory segment does not exist.
    #[error("'{backend}' backend initialization failed: {source}")]
    InitializationFailed {
        /// Name of the backend that failed to initialize (e.g., `"mpi"`).
        backend: String,
        /// The underlying error from the backend initialization.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Required environment variables for the selected backend are not set.
    ///
    /// The TCP and shared memory backends require additional configuration via
    /// environment variables. This error lists the variables that are missing.
    #[error(
        "backend '{backend}' requires missing configuration: {missing_vars}",
        missing_vars = missing_vars.join(", ")
    )]
    MissingConfiguration {
        /// Name of the backend requiring configuration (e.g., `"tcp"`).
        backend: String,
        /// List of environment variable names that are not set.
        missing_vars: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::{BackendError, CommError, ReduceOp};

    #[test]
    fn test_reduce_op_debug_format() {
        assert_eq!(format!("{:?}", ReduceOp::Sum), "Sum");
        assert_eq!(format!("{:?}", ReduceOp::Min), "Min");
        assert_eq!(format!("{:?}", ReduceOp::Max), "Max");
        assert_eq!(format!("{:?}", ReduceOp::BitwiseOr), "BitwiseOr");
    }

    #[test]
    fn test_reduce_op_copy_eq() {
        let op = ReduceOp::Sum;
        let cloned = op;
        assert_eq!(op, cloned);
        let copied: ReduceOp = op;
        assert_eq!(op, copied);
        assert_ne!(ReduceOp::Sum, ReduceOp::Min);
        assert_ne!(ReduceOp::Min, ReduceOp::Max);
        assert_ne!(ReduceOp::Sum, ReduceOp::Max);
        assert_ne!(ReduceOp::Sum, ReduceOp::BitwiseOr);
        assert_ne!(ReduceOp::BitwiseOr, ReduceOp::Max);
    }

    #[test]
    fn test_comm_error_display() {
        let err = CommError::CollectiveFailed {
            operation: "allgatherv",
            mpi_error_code: 5,
            message: "test".into(),
        };
        let display = format!("{err}");
        assert!(display.contains("allgatherv"), "display was: {display}");
        assert!(display.contains("test"), "display was: {display}");

        let err = CommError::InvalidBufferSize {
            operation: "allreduce",
            expected: 4,
            actual: 3,
        };
        let display = format!("{err}");
        assert!(display.contains("allreduce"), "display was: {display}");
        assert!(display.contains('4'), "display was: {display}");
        assert!(display.contains('3'), "display was: {display}");

        let err = CommError::InvalidRoot { root: 5, size: 4 };
        let display = format!("{err}");
        assert!(display.contains('5'), "display was: {display}");
        assert!(display.contains('4'), "display was: {display}");

        let display = format!("{}", CommError::InvalidCommunicator);
        assert!(!display.is_empty(), "display was empty");

        let err = CommError::AllocationFailed {
            requested_bytes: 1024,
            message: "permission denied".into(),
        };
        let display = format!("{err}");
        assert!(display.contains("1024"), "display was: {display}");
        assert!(
            display.contains("permission denied"),
            "display was: {display}"
        );
    }

    #[test]
    fn test_comm_error_debug() {
        let err = CommError::CollectiveFailed {
            operation: "broadcast",
            mpi_error_code: 1,
            message: "rank died".into(),
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("CollectiveFailed"), "debug was: {debug}");

        let debug = format!("{:?}", CommError::InvalidCommunicator);
        assert!(debug.contains("InvalidCommunicator"), "debug was: {debug}");
    }

    #[test]
    fn test_backend_error_display() {
        let err = BackendError::BackendNotAvailable {
            requested: "mpi".into(),
            available: vec!["local".into()],
        };
        let display = format!("{err}");
        assert!(display.contains("mpi"), "display was: {display}");

        let err = BackendError::InvalidBackend {
            requested: "foobar".into(),
            available: vec!["mpi".into(), "local".into()],
        };
        let display = format!("{err}");
        assert!(display.contains("foobar"), "display was: {display}");

        let err = BackendError::InitializationFailed {
            backend: "mpi".into(),
            source: "MPI runtime not found".into(),
        };
        let display = format!("{err}");
        assert!(display.contains("mpi"), "display was: {display}");

        let err = BackendError::MissingConfiguration {
            backend: "tcp".into(),
            missing_vars: vec!["COBRE_TCP_COORDINATOR".into(), "COBRE_TCP_RANK".into()],
        };
        let display = format!("{err}");
        assert!(display.contains("tcp"), "display was: {display}");
        assert!(
            display.contains("COBRE_TCP_COORDINATOR"),
            "display was: {display}"
        );
    }

    #[test]
    fn test_comm_error_std_error() {
        fn accepts_std_error(_e: &dyn std::error::Error) {}
        accepts_std_error(&CommError::InvalidCommunicator);
        accepts_std_error(&CommError::InvalidRoot { root: 2, size: 1 });
    }

    #[test]
    fn test_backend_error_std_error() {
        fn accepts_std_error(_e: &dyn std::error::Error) {}
        accepts_std_error(&BackendError::BackendNotAvailable {
            requested: "mpi".into(),
            available: vec!["local".into()],
        });
        accepts_std_error(&BackendError::MissingConfiguration {
            backend: "tcp".into(),
            missing_vars: vec!["COBRE_TCP_RANK".into()],
        });
    }
}
