//! Factory function for creating the active communication backend.
//!
//! [`create_communicator`] is the single entry point for constructing a
//! [`Communicator`](crate::Communicator) instance at runtime. It selects the
//! backend according to:
//!
//! 1. The `COBRE_COMM_BACKEND` environment variable (runtime override).
//! 2. The Cargo feature flags compiled into the binary (`mpi`).
//! 3. A fallback to [`LocalBackend`](crate::LocalBackend) when no distributed backend
//!    is available.
//!
//! This indirection keeps algorithm crates decoupled from any specific backend: the
//! calling code only sees the `Communicator` trait, never a concrete type.

/// Programmatic backend selector for library-mode callers.
///
/// Used by `cobre-python` (`PyO3` bindings) and `cobre-mcp` (MCP server) to
/// select a backend explicitly without relying on environment variables.
/// The factory function [`create_communicator`] accepts a
/// `BackendKind` argument when called from library code.
///
/// # Variants
///
/// - [`Auto`](BackendKind::Auto) — let the factory choose the best available
///   backend (same priority order as the environment-variable path).
/// - [`Mpi`](BackendKind::Mpi) — request the MPI backend. Returns an error
///   if the `mpi` feature is not compiled in.
/// - [`Local`](BackendKind::Local) — always use the single-process
///   [`LocalBackend`](crate::LocalBackend), even when MPI is available.
///
/// # Future variants
///
/// `Tcp` and `Shm` will be added when the corresponding backend crates are
/// implemented. They are deliberately excluded because no
/// concrete `TcpBackend` or `ShmBackend` types exist yet.
///
/// # Examples
///
/// ```rust
/// use cobre_comm::BackendKind;
///
/// let kind = BackendKind::Auto;
/// let copy = kind;  // Copy trait is derived
/// assert_eq!(copy, kind);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Automatically select the best available backend.
    ///
    /// Priority order (highest to lowest):
    /// 1. MPI (`mpi` feature enabled and MPI runtime detectable)
    /// 2. Local (single-process fallback, always available)
    Auto,
    /// Use the MPI backend (`FerrompiBackend`).
    ///
    /// Fails at runtime if the `mpi` feature is not compiled in.
    Mpi,
    /// Use the single-process local backend (`LocalBackend`).
    ///
    /// Always succeeds regardless of feature flags.
    Local,
}

/// Enum-dispatched communicator that wraps any available concrete backend.
///
/// # Design rationale
///
/// [`crate::Communicator`] carries generic methods that make the trait not object-safe.
/// Enum dispatch for closed variant sets avoids `Box<dyn>`: a `match` arm delegates
/// each method call to the inner concrete type. The dispatch overhead is negligible
/// compared to the MPI collective or LP solve it wraps.
///
/// # Availability
///
/// This enum is only present in builds where the `mpi` feature is compiled in.
/// In no-feature builds, callers use [`crate::LocalBackend`] directly.
///
/// # Variants
///
/// - `Mpi(FerrompiBackend)` — available when `features = ["mpi"]`
/// - `Local(LocalBackend)` — always present inside this enum
///
/// # Thread safety
///
/// `CommBackend: Send + Sync` because all inner backend types are `Send + Sync`.
#[cfg(feature = "mpi")]
pub enum CommBackend {
    /// MPI backend powered by ferrompi.
    Mpi(Box<crate::FerrompiBackend>),

    /// Single-process local backend.
    ///
    /// Always present as a fallback. Selected when no distributed runtime is
    /// available or when the caller explicitly requests `BackendKind::Local`.
    Local(crate::LocalBackend),
}

#[cfg(feature = "mpi")]
#[allow(dead_code)]
const fn _assert_comm_backend_send_sync() {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CommBackend>();
}

#[cfg(feature = "mpi")]
impl crate::Communicator for CommBackend {
    fn allgatherv<T: crate::CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.allgatherv(send, recv, counts, displs),
            Self::Local(backend) => backend.allgatherv(send, recv, counts, displs),
        }
    }

    fn allreduce<T: crate::CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: crate::ReduceOp,
    ) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.allreduce(send, recv, op),
            Self::Local(backend) => backend.allreduce(send, recv, op),
        }
    }

    fn broadcast<T: crate::CommData>(
        &self,
        buf: &mut [T],
        root: usize,
    ) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.broadcast(buf, root),
            Self::Local(backend) => backend.broadcast(buf, root),
        }
    }

    fn barrier(&self) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.barrier(),
            Self::Local(backend) => backend.barrier(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.rank(),
            Self::Local(backend) => backend.rank(),
        }
    }

    fn size(&self) -> usize {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.size(),
            Self::Local(backend) => backend.size(),
        }
    }

    fn abort(&self, error_code: i32) -> ! {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.abort(error_code),
            Self::Local(backend) => backend.abort(error_code),
        }
    }
}

#[cfg(feature = "mpi")]
impl crate::SharedMemoryProvider for CommBackend {
    /// Both `LocalBackend` and `FerrompiBackend` use `HeapRegion<T>` as their
    /// `Region<T>` type (`HeapFallback` semantics per spec SS4.7). Using
    /// `HeapRegion<T>` as the concrete GAT here avoids an additional
    /// enum wrapper around region types, since both inner backends already
    /// unify on the same concrete region type.
    type Region<T: crate::CommData> = crate::HeapRegion<T>;

    fn create_shared_region<T: crate::CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.create_shared_region(count),
            Self::Local(backend) => backend.create_shared_region(count),
        }
    }

    fn split_local(&self) -> Result<Box<dyn crate::LocalCommunicator>, crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.split_local(),
            Self::Local(backend) => backend.split_local(),
        }
    }

    fn is_leader(&self) -> bool {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.is_leader(),
            Self::Local(backend) => backend.is_leader(),
        }
    }
}

#[cfg(feature = "mpi")]
impl crate::TopologyProvider for CommBackend {
    fn topology(&self) -> &crate::ExecutionTopology {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.topology(),
            Self::Local(backend) => backend.topology(),
        }
    }
}

/// Returns all backend names compiled into this binary.
///
/// Always includes `"local"`. Conditionally includes `"mpi"` (feature `mpi`).
///
/// # Examples
///
/// ```rust
/// use cobre_comm::available_backends;
///
/// let backends = available_backends();
/// assert!(backends.contains(&"local".to_string()));
/// ```
#[must_use]
#[allow(clippy::vec_init_then_push)] // cfg-gated push pattern
pub fn available_backends() -> Vec<String> {
    let mut backends = Vec::new();
    #[cfg(feature = "mpi")]
    backends.push("mpi".to_string());
    backends.push("local".to_string());
    backends
}

/// Returns `true` if any MPI launcher environment variable is present.
///
/// Checks the following variables (without inspecting their values):
/// `PMI_RANK`, `PMI_SIZE`, `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE`,
/// `MPI_LOCALRANKID`, `SLURM_PROCID`.
///
/// Uses `std::env::var_os` so that non-UTF-8 values are still detected
/// correctly. Always compiled in (no cfg gate) so that auto-detection
/// logic is testable in no-feature builds.
///
/// In no-feature builds this function is not called from production code
/// (only from tests), so dead-code lint is suppressed for that configuration.
#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
fn mpi_launch_detected() -> bool {
    const MPI_ENV_VARS: [&str; 6] = [
        "PMI_RANK",
        "PMI_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "MPI_LOCALRANKID",
        "SLURM_PROCID",
    ];
    MPI_ENV_VARS
        .iter()
        .any(|var| std::env::var_os(var).is_some())
}

/// Construct the active communication backend (no-feature build).
///
/// When the `mpi` feature is not compiled in, this function always returns a
/// [`crate::LocalBackend`] or an error:
///
/// - `COBRE_COMM_BACKEND` unset, `"auto"`, or `"local"` → `Ok(LocalBackend)`
/// - A known distributed backend name (`"mpi"`) →
///   `Err(BackendError::BackendNotAvailable)`
/// - An unknown name → `Err(BackendError::InvalidBackend)`
///
/// # Errors
///
/// - [`crate::BackendError::BackendNotAvailable`]: a known backend was requested
///   but not compiled into this binary.
/// - [`crate::BackendError::InvalidBackend`]: `COBRE_COMM_BACKEND` contains an
///   unrecognized value.
///
/// # Examples
///
/// ```rust
/// # #[cfg(not(feature = "mpi"))]
/// # {
/// use cobre_comm::create_communicator;
///
/// // With no distributed features, the factory always returns LocalBackend.
/// let backend = create_communicator().expect("local backend must succeed");
/// # use cobre_comm::Communicator;
/// assert_eq!(backend.rank(), 0);
/// assert_eq!(backend.size(), 1);
/// # }
/// ```
#[cfg(not(feature = "mpi"))]
pub fn create_communicator() -> Result<crate::LocalBackend, crate::BackendError> {
    let requested = std::env::var("COBRE_COMM_BACKEND").unwrap_or_else(|_| "auto".to_string());
    match requested.as_str() {
        "auto" | "local" => Ok(crate::LocalBackend),
        "mpi" => Err(crate::BackendError::BackendNotAvailable {
            requested,
            available: available_backends(),
        }),
        _ => Err(crate::BackendError::InvalidBackend {
            requested,
            available: vec!["auto", "mpi", "local"]
                .into_iter()
                .map(String::from)
                .collect(),
        }),
    }
}

/// Construct the active communication backend (MPI build).
///
/// When the `mpi` feature is compiled in, this function returns a
/// [`CommBackend`] selected according to the `COBRE_COMM_BACKEND` environment
/// variable:
///
/// - Unset or `"auto"` → auto-detect priority chain
/// - `"mpi"` → `CommBackend::Mpi(FerrompiBackend::new()?)`
/// - `"local"` → `CommBackend::Local(LocalBackend)`
/// - Unknown name → `Err(BackendError::InvalidBackend)`
///
/// # Errors
///
/// - [`crate::BackendError::InvalidBackend`]: `COBRE_COMM_BACKEND` contains an
///   unrecognized value.
/// - [`crate::BackendError::InitializationFailed`]: the selected backend failed
///   to initialize (propagated from [`crate::FerrompiBackend::new`]).
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "mpi")]
/// # {
/// use cobre_comm::{create_communicator, Communicator};
///
/// // With COBRE_COMM_BACKEND unset or "local", returns CommBackend::Local.
/// // std::env::remove_var is unsafe in multi-threaded contexts (Rust 2024).
/// // SAFETY: this doctest runs single-threaded; no concurrent env mutation.
/// unsafe { std::env::set_var("COBRE_COMM_BACKEND", "local") };
/// let backend = create_communicator().expect("local backend must succeed");
/// assert_eq!(backend.rank(), 0);
/// unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
/// # }
/// ```
#[cfg(feature = "mpi")]
pub fn create_communicator() -> Result<CommBackend, crate::BackendError> {
    let requested = std::env::var("COBRE_COMM_BACKEND").unwrap_or_else(|_| "auto".to_string());
    match requested.as_str() {
        "auto" => auto_detect(),
        "mpi" => Ok(CommBackend::Mpi(Box::new(crate::FerrompiBackend::new()?))),
        "local" => Ok(CommBackend::Local(crate::LocalBackend)),
        _ => Err(crate::BackendError::InvalidBackend {
            requested,
            available: vec!["auto", "mpi", "local"]
                .into_iter()
                .map(String::from)
                .collect(),
        }),
    }
}

/// Auto-detect the best available backend using a priority chain.
///
/// Priority order (highest to lowest):
/// 1. MPI — if the `mpi` feature is compiled in and [`mpi_launch_detected`]
///    returns `true`.
/// 2. Local — unconditional fallback.
///
/// # Errors
///
/// Returns [`crate::BackendError::InitializationFailed`] if the MPI backend is
/// selected but [`crate::FerrompiBackend::new`] fails.
#[cfg(feature = "mpi")]
fn auto_detect() -> Result<CommBackend, crate::BackendError> {
    #[cfg(feature = "mpi")]
    if mpi_launch_detected() {
        return Ok(CommBackend::Mpi(Box::new(crate::FerrompiBackend::new()?)));
    }
    Ok(CommBackend::Local(crate::LocalBackend))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::BackendKind;

    use super::{available_backends, mpi_launch_detected};

    /// Serialises tests that mutate `COBRE_COMM_BACKEND`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// `available_backends()` always contains `"local"` regardless of features.
    #[test]
    fn test_available_backends_contains_local() {
        let backends = available_backends();
        assert!(
            backends.contains(&"local".to_string()),
            "expected 'local' in {backends:?}"
        );
    }

    /// In a no-feature build `available_backends()` returns exactly `["local"]`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_available_backends_no_feature_exact() {
        assert_eq!(available_backends(), vec!["local".to_string()]);
    }

    /// `mpi_launch_detected()` returns `false` when none of the 6 MPI env vars
    /// are set.
    ///
    /// This test is not perfectly hermetic in a multi-threaded test run because
    /// other tests may set env vars concurrently. However, since none of the
    /// *MPI-specific* vars (`PMI_RANK`, `PMI_SIZE`, etc.) are set by other
    /// tests in this crate, the risk is negligible.
    #[test]
    fn test_mpi_launch_detected_false_by_default() {
        const MPI_VARS: [&str; 6] = [
            "PMI_RANK",
            "PMI_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "MPI_LOCALRANKID",
            "SLURM_PROCID",
        ];
        // Hold ENV_LOCK to prevent races with tests that set/remove MPI vars.
        let _guard = ENV_LOCK.lock().unwrap();
        let any_set = MPI_VARS.iter().any(|v| std::env::var_os(v).is_some());
        if any_set {
            // Running inside a real MPI launch; skip rather than fail.
            return;
        }
        assert!(!mpi_launch_detected());
    }

    /// `mpi_launch_detected()` returns `true` when `PMI_RANK` is set.
    ///
    /// # Safety (env var mutation in tests)
    ///
    /// `std::env::set_var` and `remove_var` are unsafe in Rust 2024 when called
    /// from a multi-threaded process because concurrent reads of the environment
    /// (e.g., from `std::env::var`) are not thread-safe on some platforms. These
    /// tests access only a purpose-specific env var that no other test in this
    /// crate reads, which makes concurrent mutation safe in practice. The
    /// `unsafe` block is required by the compiler and documents this invariant.
    #[test]
    fn test_mpi_launch_detected_pmi_rank() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialised by ENV_LOCK; no concurrent env var access.
        unsafe { std::env::set_var("PMI_RANK", "0") };
        let result = mpi_launch_detected();
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("PMI_RANK") };
        assert!(
            result,
            "expected mpi_launch_detected() == true when PMI_RANK is set"
        );
    }

    /// `mpi_launch_detected()` returns `true` when `OMPI_COMM_WORLD_RANK` is set.
    #[test]
    fn test_mpi_launch_detected_ompi() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialised by ENV_LOCK.
        unsafe { std::env::set_var("OMPI_COMM_WORLD_RANK", "0") };
        let result = mpi_launch_detected();
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("OMPI_COMM_WORLD_RANK") };
        assert!(
            result,
            "expected mpi_launch_detected() == true when OMPI_COMM_WORLD_RANK is set"
        );
    }

    /// No-feature build: unset `COBRE_COMM_BACKEND` → `Ok(LocalBackend)` with
    /// rank 0 and size 1.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_auto() {
        use crate::Communicator;

        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let backend = super::create_communicator().expect("LocalBackend construction must succeed");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    /// No-feature build: `COBRE_COMM_BACKEND=foobar` → `Err(InvalidBackend)`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_invalid() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "foobar") };
        let err = super::create_communicator().expect_err("unknown backend must return Err");
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        assert!(
            matches!(
                err,
                crate::BackendError::InvalidBackend { ref requested, .. }
                    if requested == "foobar"
            ),
            "unexpected error: {err:?}"
        );
    }

    /// No-feature build: `COBRE_COMM_BACKEND=mpi` → `Err(BackendNotAvailable)`
    /// where `requested == "mpi"` and `available` contains `"local"`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_unavailable() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "mpi") };
        let err = super::create_communicator().expect_err("unavailable backend must return Err");
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        assert!(
            matches!(err, crate::BackendError::BackendNotAvailable { .. }),
            "expected BackendNotAvailable, got {err:?}"
        );
        if let crate::BackendError::BackendNotAvailable {
            ref requested,
            ref available,
        } = err
        {
            assert_eq!(requested, "mpi");
            assert!(
                available.contains(&"local".to_string()),
                "available should contain 'local', got {available:?}"
            );
        }
    }

    /// Verify that `BackendKind` derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`.
    #[test]
    fn test_backend_kind_derives() {
        let kind = BackendKind::Auto;

        // Debug
        let s = format!("{kind:?}");
        assert!(s.contains("Auto"), "Debug output should contain 'Auto'");

        // Clone — explicit call to Clone::clone to prove the trait is derived
        #[allow(clippy::clone_on_copy)]
        let cloned = kind.clone();
        assert_eq!(cloned, kind);

        // Copy
        let copied = kind;
        assert_eq!(copied, kind);

        // PartialEq + Eq
        assert_eq!(BackendKind::Mpi, BackendKind::Mpi);
        assert_ne!(BackendKind::Mpi, BackendKind::Local);
        assert_eq!(BackendKind::Local, BackendKind::Local);
    }

    #[cfg(feature = "mpi")]
    #[allow(clippy::float_cmp)]
    mod comm_backend {
        use super::super::CommBackend;
        use crate::{Communicator, LocalBackend, ReduceOp, SharedMemoryProvider, SharedRegion};

        /// Compile-time assertion that `CommBackend: Send + Sync`.
        #[test]
        fn test_comm_backend_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<CommBackend>();
        }

        /// `CommBackend::Local` delegates `rank()` → 0 and `size()` → 1.
        #[test]
        fn test_comm_backend_local_rank_size() {
            let backend = CommBackend::Local(LocalBackend);
            assert_eq!(backend.rank(), 0);
            assert_eq!(backend.size(), 1);
        }

        /// `CommBackend::Local` delegates `barrier()` → `Ok(())`.
        #[test]
        fn test_comm_backend_local_barrier() {
            let backend = CommBackend::Local(LocalBackend);
            assert!(backend.barrier().is_ok());
        }

        /// `CommBackend::Local` delegates `allreduce` with identity-copy semantics.
        #[test]
        fn test_comm_backend_local_allreduce() {
            let backend = CommBackend::Local(LocalBackend);
            let send = [1.0_f64, 2.0, 3.0];
            let mut recv = [0.0_f64; 3];
            backend.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
            assert_eq!(recv, [1.0, 2.0, 3.0]);
        }

        /// `CommBackend::Local` delegates `allgatherv` with identity-copy semantics.
        #[test]
        fn test_comm_backend_local_allgatherv() {
            let backend = CommBackend::Local(LocalBackend);
            let send = [7.0_f64, 8.0, 9.0];
            let mut recv = [0.0_f64; 3];
            backend.allgatherv(&send, &mut recv, &[3], &[0]).unwrap();
            assert_eq!(recv, [7.0, 8.0, 9.0]);
        }

        /// `CommBackend::Local` delegates `broadcast` as a no-op for root 0.
        #[test]
        fn test_comm_backend_local_broadcast() {
            let backend = CommBackend::Local(LocalBackend);
            let mut buf = [1.0_f64, 2.0];
            assert!(backend.broadcast(&mut buf, 0).is_ok());
            assert_eq!(buf, [1.0, 2.0]);
        }

        /// `CommBackend::Local` delegates `SharedMemoryProvider` methods correctly.
        ///
        /// Covers: `create_shared_region`, `split_local`, `is_leader`.
        #[test]
        fn test_comm_backend_local_shared_memory() {
            let backend = CommBackend::Local(LocalBackend);

            // create_shared_region: returns a region with the correct element count
            let mut region = backend.create_shared_region::<f64>(10).unwrap();
            assert_eq!(region.as_slice().len(), 10);
            // Verify the region is writable (population phase)
            region.as_mut_slice().fill(42.0);
            assert_eq!(region.as_slice(), &[42.0_f64; 10]);
            assert!(region.fence().is_ok());

            // split_local: returns a local communicator with rank 0 and size 1
            let local_comm = backend.split_local().unwrap();
            assert_eq!(local_comm.rank(), 0);
            assert_eq!(local_comm.size(), 1);
            assert!(local_comm.barrier().is_ok());

            // is_leader: LocalBackend is always the leader
            assert!(backend.is_leader());
        }
    }
}
