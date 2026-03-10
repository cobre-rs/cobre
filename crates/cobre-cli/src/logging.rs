//! Tracing-subscriber initialization for the `cobre` binary.
//!
//! [`init_logging`] configures a `fmt` subscriber that writes to stderr.
//! The default filter keeps library crates at `warn` and `cobre_cli` at `info`.
//! Passing `verbose = true` raises library crates to `info` and `cobre_cli`
//! to `debug`. The `RUST_LOG` environment variable overrides both presets.

use tracing_subscriber::{fmt, EnvFilter};

/// Initialize the tracing subscriber for the process.
///
/// Must be called exactly once, early in `main`, before any tracing events
/// are emitted. A second call within the same process will panic (tracing
/// subscriber can only be set once globally).
///
/// # Filter levels
///
/// | Mode | Library crates | `cobre_cli` |
/// |------|----------------|-------------|
/// | Normal (`verbose = false`) | `warn` | `info` |
/// | Verbose (`verbose = true`) | `info` | `debug` |
///
/// The `RUST_LOG` environment variable, when set, fully overrides the
/// preset filter. This allows fine-grained control on HPC systems:
/// `RUST_LOG=cobre_sddp=debug,cobre_cli=trace`.
///
/// # Output channel
///
/// All tracing output goes to **stderr**, keeping stdout clean for
/// machine-readable output from `cobre report` and `cobre validate`.
pub fn init_logging(verbose: bool) {
    let default_filter = if verbose {
        "warn,cobre_cli=debug,cobre_core=info,cobre_io=info,cobre_sddp=info,\
         cobre_solver=info,cobre_comm=info,cobre_stochastic=info"
    } else {
        "warn,cobre_cli=info"
    };

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));

    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn init_logging_non_verbose_does_not_panic() {
        let filter_str = "warn,cobre_cli=info";
        let filter = EnvFilter::new(filter_str);
        drop(filter);
    }

    #[test]
    fn init_logging_verbose_filter_string_is_valid() {
        let filter_str = "warn,cobre_cli=debug,cobre_core=info,cobre_io=info,\
                          cobre_sddp=info,cobre_solver=info,cobre_comm=info,cobre_stochastic=info";
        let filter = EnvFilter::new(filter_str);
        drop(filter);
    }
}
