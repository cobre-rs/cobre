//! SIGTERM and SIGINT handling for `cobre run`, and the cross-rank agreement
//! that settles a signal stop after the training writes.

use std::ffi::c_int;
use std::fmt::Display;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use cobre_comm::{Communicator, ReduceOp};
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

    pub(super) fn sample_level(&self) -> usize {
        self.shutdown.load(Ordering::SeqCst)
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

/// Rank 0's write failure code (`0` for none) and the shutdown level, each the
/// largest over every rank.
pub(super) struct PostWriteAgreement {
    pub(super) failure_code: i32,
    pub(super) level: usize,
}

pub(super) fn failure_code(local: &Result<(), CliError>) -> i32 {
    local.as_ref().err().map_or(0, CliError::exit_code)
}

/// Agrees the failure code and the shutdown level with a `Max` reduction, which
/// is independent of rank order and count, so every rank takes the same exit
/// code and the same stop decision.
pub(super) fn agree_post_write<C: Communicator>(
    comm: &C,
    local_failure_code: i32,
    local_level: usize,
) -> Result<PostWriteAgreement, CliError> {
    if comm.size() == 1 {
        return Ok(PostWriteAgreement {
            failure_code: local_failure_code,
            level: local_level,
        });
    }
    let reconcile_error = |e: &dyn Display| CliError::Internal {
        message: format!("post-write reconcile error: {e}"),
    };
    let level = i32::try_from(local_level).map_err(|e| reconcile_error(&e))?;
    let mut agreed = [0_i32; 2];
    comm.allreduce(&[local_failure_code, level], &mut agreed, ReduceOp::Max)
        .map_err(|e| reconcile_error(&e))?;
    Ok(PostWriteAgreement {
        failure_code: agreed[0],
        level: usize::try_from(agreed[1]).map_err(|e| reconcile_error(&e))?,
    })
}

pub(super) fn into_agreed_result(
    local: Result<(), CliError>,
    agreed: &PostWriteAgreement,
) -> Result<(), CliError> {
    match local {
        _ if agreed.failure_code == 0 => Ok(()),
        Err(e) if e.exit_code() == agreed.failure_code => Err(e),
        _ => Err(CliError::for_peer_failure(agreed.failure_code)),
    }
}

#[cfg(test)]
mod tests {
    use cobre_comm::LocalBackend;

    use super::{PostWriteAgreement, agree_post_write, into_agreed_result};

    #[test]
    fn single_rank_post_write_agreement_is_the_local_value() {
        for (code, level) in [(2, 0), (0, 2)] {
            let agreed = agree_post_write(&LocalBackend, code, level).expect("no collective runs");
            assert_eq!((agreed.failure_code, agreed.level), (code, level));
        }

        let peer = into_agreed_result(
            Ok(()),
            &PostWriteAgreement {
                failure_code: 2,
                level: 0,
            },
        );
        assert_eq!(peer.map_err(|e| e.exit_code()), Err(2));
    }
}
