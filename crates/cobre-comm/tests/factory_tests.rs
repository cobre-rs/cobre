//! Integration tests for the `create_communicator()` factory and public APIs.
//!
//! Tests env var handling, backend availability, and trait bounds across
//! different feature compilation scenarios.
// Allow expect/unwrap in test code — these guard test setup paths that must not fail.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Mutex;

/// Global mutex serialising all tests that mutate `COBRE_COMM_BACKEND`.
/// Rust runs tests in parallel within a binary; without this lock, concurrent
/// `set_var` / `remove_var` calls produce non-deterministic results.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Factory tests for builds where no distributed backend feature is compiled in.
///
/// All tests in this module are compiled only when neither `mpi`, `tcp`, nor `shm`
/// features are enabled, because in those builds `create_communicator()` returns
/// `Result<LocalBackend, BackendError>`.
#[cfg(not(any(feature = "mpi", feature = "tcp", feature = "shm")))]
mod no_feature_factory {
    use cobre_comm::{BackendError, Communicator, create_communicator};

    /// Unset `COBRE_COMM_BACKEND` → `Ok(LocalBackend)` with rank=0, size=1.
    #[test]
    fn test_factory_no_feature_auto() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let backend = create_communicator().expect("must succeed");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    /// `COBRE_COMM_BACKEND=local` → `Ok(LocalBackend)` with rank=0, size=1.
    #[test]
    fn test_factory_no_feature_explicit_local() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "local") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let backend = result.expect("must succeed");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    /// `COBRE_COMM_BACKEND=auto` → `Ok(LocalBackend)`.
    #[test]
    fn test_factory_no_feature_explicit_auto() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "auto") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        result.expect("must succeed");
    }

    /// `COBRE_COMM_BACKEND=mpi` → `Err(BackendNotAvailable)` with `requested=="mpi"`.
    #[test]
    fn test_factory_no_feature_mpi_unavailable() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "mpi") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let err = result.expect_err("must fail");
        assert!(
            matches!(
                err,
                BackendError::BackendNotAvailable {
                    ref requested,
                    ..
                } if requested == "mpi"
            ),
            "got {err:?}"
        );
        if let BackendError::BackendNotAvailable { ref available, .. } = err {
            assert!(available.contains(&"local".to_string()));
        }
    }

    /// `COBRE_COMM_BACKEND=tcp` → `Err(BackendNotAvailable)`.
    #[test]
    fn test_factory_no_feature_tcp_unavailable() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "tcp") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let err = result.expect_err("must fail");
        assert!(matches!(err, BackendError::BackendNotAvailable { .. }));
    }

    /// `COBRE_COMM_BACKEND=shm` → `Err(BackendNotAvailable)`.
    #[test]
    fn test_factory_no_feature_shm_unavailable() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "shm") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let err = result.expect_err("must fail");
        assert!(matches!(err, BackendError::BackendNotAvailable { .. }));
    }

    /// `COBRE_COMM_BACKEND=foobar` → `Err(InvalidBackend)` with `requested=="foobar"`.
    #[test]
    fn test_factory_no_feature_invalid_name() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "foobar") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let err = result.expect_err("must fail");
        assert!(
            matches!(
                err,
                BackendError::InvalidBackend {
                    ref requested,
                    ..
                } if requested == "foobar"
            ),
            "got {err:?}"
        );
    }
}

// ── any-feature factory tests ─────────────────────────────────────────────────

/// Factory tests for builds with distributed backend features enabled.
#[cfg(any(feature = "mpi", feature = "tcp", feature = "shm"))]
mod any_feature_factory {
    use cobre_comm::{BackendError, CommBackend, Communicator, create_communicator};

    /// `COBRE_COMM_BACKEND=local` → `Ok(CommBackend::Local(...))` with rank=0.
    #[test]
    fn test_factory_any_feature_local() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "local") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let backend = result.expect("must succeed");
        assert!(matches!(backend, CommBackend::Local(_)));
        assert_eq!(backend.rank(), 0);
    }

    /// `COBRE_COMM_BACKEND=tcp` → `Err(BackendNotAvailable)`.
    #[test]
    fn test_factory_any_feature_tcp_unavailable() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "tcp") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        match result {
            Ok(_) => panic!("must fail"),
            Err(err) => assert!(matches!(err, BackendError::BackendNotAvailable { .. })),
        }
    }

    /// `COBRE_COMM_BACKEND=shm` → `Err(BackendNotAvailable)`.
    #[test]
    fn test_factory_any_feature_shm_unavailable() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "shm") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        match result {
            Ok(_) => panic!("must fail"),
            Err(err) => assert!(matches!(err, BackendError::BackendNotAvailable { .. })),
        }
    }

    /// `COBRE_COMM_BACKEND=foobar` → `Err(InvalidBackend)` with `requested=="foobar"`.
    #[test]
    fn test_factory_any_feature_invalid_name() {
        let _guard = crate::ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "foobar") };
        let result = create_communicator();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        match result {
            Ok(_) => panic!("must fail"),
            Err(err) => assert!(
                matches!(
                    err,
                    BackendError::InvalidBackend {
                        ref requested,
                        ..
                    } if requested == "foobar"
                ),
                "got {err:?}"
            ),
        }
    }
}

// ── available_backends() tests ────────────────────────────────────────────────

/// Tests for the `available_backends()` public function.
mod available_backends_tests {
    use cobre_comm::available_backends;

    /// `available_backends()` always includes `"local"`.
    #[test]
    fn test_available_backends_contains_local() {
        let backends = available_backends();
        assert!(backends.contains(&"local".to_string()));
    }

    /// When the `mpi` feature is enabled, `available_backends()` includes `"mpi"`.
    #[test]
    #[cfg(feature = "mpi")]
    fn test_available_backends_mpi_feature() {
        let backends = available_backends();
        assert!(backends.contains(&"mpi".to_string()));
    }
}

// ── compile-time checks ───────────────────────────────────────────────────────

/// Compile-time trait-bound verification for `FerrompiBackend`.
mod compile_time_checks {
    /// `FerrompiBackend: Send + Sync` compiles.
    #[test]
    #[cfg(feature = "mpi")]
    fn test_ferrompi_backend_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<cobre_comm::FerrompiBackend>();
    }

    /// `FerrompiBackend: Communicator` compiles.
    #[test]
    #[cfg(feature = "mpi")]
    fn test_ferrompi_backend_communicator() {
        fn assert_communicator<T: cobre_comm::Communicator>() {}
        assert_communicator::<cobre_comm::FerrompiBackend>();
    }

    /// `FerrompiBackend: SharedMemoryProvider` compiles.
    #[test]
    #[cfg(feature = "mpi")]
    fn test_ferrompi_backend_shared_memory_provider() {
        fn assert_shared_memory_provider<T: cobre_comm::SharedMemoryProvider>() {}
        assert_shared_memory_provider::<cobre_comm::FerrompiBackend>();
    }
}

// ── error type checks ─────────────────────────────────────────────────────────

/// Compile-time verification that error types implement standard traits.
mod error_type_checks {
    use cobre_comm::{BackendError, CommError};

    /// `CommError: std::error::Error + Send + Sync` compiles.
    #[test]
    fn test_comm_error_std_error_send_sync() {
        fn assert_error_send_sync<T: std::error::Error + Send + Sync>() {}
        assert_error_send_sync::<CommError>();
    }

    /// `BackendError: std::error::Error + Send + Sync` compiles.
    #[test]
    fn test_backend_error_std_error_send_sync() {
        fn assert_error_send_sync<T: std::error::Error + Send + Sync>() {}
        assert_error_send_sync::<BackendError>();
    }
}
