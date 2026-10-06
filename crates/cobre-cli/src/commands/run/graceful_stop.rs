//! SIGTERM and SIGINT handling for `cobre run`.

use std::ffi::c_int;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use cobre_sddp::config::ShutdownSource;
use signal_hook::consts::signal::{SIGINT, SIGTERM};
use signal_hook::{flag, low_level};

use crate::error::CliError;

/// The graceful window: while it is open, SIGTERM and SIGINT request a stop at
/// the next iteration boundary instead of taking their default action.
pub(super) struct SignalWindow {
    shutdown: Arc<AtomicUsize>,
    immediate: Arc<AtomicBool>,
    sigint_armed: Arc<AtomicBool>,
    last_signal: Arc<AtomicUsize>,
}

static SIGNAL_WINDOW: OnceLock<io::Result<SignalWindow>> = OnceLock::new();

/// Registers the SIGTERM and SIGINT handlers once per process, with the window
/// closed; `arm_second_sigint` makes a second SIGINT inside the window take its
/// default action.
pub(super) fn install(arm_second_sigint: bool) -> Result<&'static SignalWindow, CliError> {
    SIGNAL_WINDOW
        .get_or_init(|| SignalWindow::register(arm_second_sigint))
        .as_ref()
        .map_err(|e| CliError::Internal {
            message: format!("signal handler registration failed: {e}"),
        })
}

impl SignalWindow {
    fn register(arm_second_sigint: bool) -> io::Result<Self> {
        let window = Self {
            shutdown: Arc::new(AtomicUsize::new(0)),
            immediate: Arc::new(AtomicBool::new(true)),
            sigint_armed: Arc::new(AtomicBool::new(false)),
            last_signal: Arc::new(AtomicUsize::new(0)),
        };
        let signal_level = ShutdownSource::Signal.level();
        // A signal's actions run in registration order: the default-action check
        // goes first so a partly registered chain still dies by the signal,
        // `sigint_armed` is checked before it is set, and `last_signal` is stored
        // before `shutdown`.
        flag::register_conditional_default(SIGTERM, Arc::clone(&window.immediate))?;
        flag::register_usize(
            SIGTERM,
            Arc::clone(&window.last_signal),
            signal_number(SIGTERM)?,
        )?;
        flag::register_usize(SIGTERM, Arc::clone(&window.shutdown), signal_level)?;
        flag::register_conditional_default(SIGINT, Arc::clone(&window.immediate))?;
        flag::register_conditional_default(SIGINT, Arc::clone(&window.sigint_armed))?;
        flag::register_usize(
            SIGINT,
            Arc::clone(&window.last_signal),
            signal_number(SIGINT)?,
        )?;
        flag::register_usize(SIGINT, Arc::clone(&window.shutdown), signal_level)?;
        if arm_second_sigint {
            flag::register(SIGINT, Arc::clone(&window.sigint_armed))?;
        }
        Ok(window)
    }

    pub(super) fn shutdown_flag(&self) -> &Arc<AtomicUsize> {
        &self.shutdown
    }

    pub(super) fn open(&self) {
        // The resets run while a signal still takes its default action, so none
        // of them can erase a request.
        self.shutdown.store(0, Ordering::SeqCst);
        self.last_signal.store(0, Ordering::SeqCst);
        self.sigint_armed.store(false, Ordering::SeqCst);
        self.immediate.store(false, Ordering::SeqCst);
    }

    /// Gives each signal its default action back, then re-raises a signal that
    /// arrived inside the window, so the process dies by it.
    pub(super) fn close(&self) -> Result<(), CliError> {
        self.immediate.store(true, Ordering::SeqCst);
        if self.shutdown.load(Ordering::SeqCst) != ShutdownSource::Signal.level() {
            return Ok(());
        }
        let last_signal = self.last_signal.load(Ordering::SeqCst);
        c_int::try_from(last_signal)
            .map_err(io::Error::other)
            .and_then(low_level::raise)
            .map_err(|e| CliError::Internal {
                message: format!("re-raising signal {last_signal} failed: {e}"),
            })
    }
}

fn signal_number(signal: c_int) -> io::Result<usize> {
    usize::try_from(signal).map_err(io::Error::other)
}
