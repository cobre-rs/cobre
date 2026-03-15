//! Subcommand implementations for the `cobre` binary.
//!
//! Each module contains an `Args` struct (clap derive) and an `execute` function
//! that performs the subcommand's work and returns a [`crate::error::CliError`]
//! on failure.

pub(crate) mod broadcast;
pub mod init;
pub mod report;
pub mod run;
pub mod schema;
pub mod summary;
pub mod validate;
pub mod version;
