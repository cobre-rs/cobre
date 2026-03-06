//! Opening scenario tree generation.
//!
//! Constructs the opening scenario tree from pre-computed PAR coefficient
//! matrices and Cholesky-factorised correlation matrices. Deterministic
//! seeds derived via SipHash-1-3 are used to initialise per-scenario
//! PRNGs, ensuring reproducibility without inter-process communication.
