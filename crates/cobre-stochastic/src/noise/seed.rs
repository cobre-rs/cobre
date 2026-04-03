//! Deterministic seed derivation via SipHash-1-3.
//!
//! Derives unique `u64` seeds from a global base seed and context tuple
//! using SipHash-1-3. The three variants use different wire format lengths
//! to prevent hash domain collisions:
//! - [`derive_forward_seed`]: 20 bytes (base seed + iteration + scenario + stage)
//! - [`derive_opening_seed`]: 16 bytes (base seed + opening index + stage)
//! - [`derive_stage_seed`]: 12 bytes (base seed + stage)

use siphasher::sip::SipHasher13;
use std::hash::Hasher;

/// Derive a deterministic seed for forward-pass noise generation.
///
/// Returns the same output for identical `(base_seed, iteration, scenario, stage)`
/// tuples regardless of MPI rank, thread ID, or process restart.
#[must_use]
pub fn derive_forward_seed(base_seed: u64, iteration: u32, scenario: u32, stage: u32) -> u64 {
    let mut hasher = SipHasher13::new();
    hasher.write(&base_seed.to_le_bytes());
    hasher.write(&iteration.to_le_bytes());
    hasher.write(&scenario.to_le_bytes());
    hasher.write(&stage.to_le_bytes());
    hasher.finish()
}

/// Derive a deterministic seed for stage-level batch generation.
///
/// Returns the same output for identical `(base_seed, stage_id)` tuples
/// regardless of MPI rank or thread ID. Intended for batch noise methods
/// (LHS, QMC) that require all openings at a stage simultaneously.
#[must_use]
pub fn derive_stage_seed(base_seed: u64, stage_id: u32) -> u64 {
    let mut hasher = SipHasher13::new();
    hasher.write(&base_seed.to_le_bytes());
    hasher.write(&stage_id.to_le_bytes());
    hasher.finish()
}

/// Derive a deterministic seed for opening tree generation.
///
/// Returns the same output for identical `(base_seed, opening_index, stage)`
/// tuples regardless of MPI rank or thread ID.
#[must_use]
pub fn derive_opening_seed(base_seed: u64, opening_index: u32, stage: u32) -> u64 {
    let mut hasher = SipHasher13::new();
    hasher.write(&base_seed.to_le_bytes());
    hasher.write(&opening_index.to_le_bytes());
    hasher.write(&stage.to_le_bytes());
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::{derive_forward_seed, derive_opening_seed, derive_stage_seed};

    // -------------------------------------------------------------------------
    // derive_forward_seed: determinism
    // -------------------------------------------------------------------------

    #[test]
    fn forward_seed_is_deterministic() {
        assert_eq!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 0, 0, 0),
        );
    }

    #[test]
    fn forward_seed_varies_with_stage() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 0, 0, 1),
        );
    }

    #[test]
    fn forward_seed_varies_with_scenario() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 0, 1, 0),
        );
    }

    #[test]
    fn forward_seed_varies_with_iteration() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(42, 1, 0, 0),
        );
    }

    #[test]
    fn forward_seed_varies_with_base_seed() {
        assert_ne!(
            derive_forward_seed(42, 0, 0, 0),
            derive_forward_seed(43, 0, 0, 0),
        );
    }

    // -------------------------------------------------------------------------
    // derive_opening_seed: determinism
    // -------------------------------------------------------------------------

    #[test]
    fn opening_seed_is_deterministic() {
        assert_eq!(derive_opening_seed(42, 0, 0), derive_opening_seed(42, 0, 0),);
    }

    #[test]
    fn opening_seed_varies_with_stage() {
        assert_ne!(derive_opening_seed(42, 0, 0), derive_opening_seed(42, 0, 1),);
    }

    #[test]
    fn opening_seed_varies_with_opening_index() {
        assert_ne!(derive_opening_seed(42, 0, 0), derive_opening_seed(42, 1, 0),);
    }

    #[test]
    fn opening_seed_varies_with_base_seed() {
        assert_ne!(derive_opening_seed(42, 0, 0), derive_opening_seed(43, 0, 0),);
    }

    // -------------------------------------------------------------------------
    // Cross-function differentiation: 16-byte vs 20-byte wire format
    // -------------------------------------------------------------------------

    /// `derive_opening_seed(base, 0, 0)` feeds 16 bytes;
    /// `derive_forward_seed(base, 0, 0, 0)` feeds 20 bytes.
    /// SipHash-1-3 incorporates message length into its state, so the two
    /// outputs must differ even when the numeric arguments overlap.
    #[test]
    fn forward_and_opening_seeds_differ_for_same_partial_inputs() {
        assert_ne!(
            derive_opening_seed(42, 0, 0),
            derive_forward_seed(42, 0, 0, 0),
        );
    }

    // -------------------------------------------------------------------------
    // Golden value regression: pin the output to catch algorithm changes
    // -------------------------------------------------------------------------

    /// This value was computed by running the implementation and recording the
    /// output. If this test fails, the SipHash-1-3 wire format or the
    /// `siphasher` crate version has changed in a breaking way.
    ///
    /// Input bytes (little-endian):
    ///   42u64  = [2a 00 00 00 00 00 00 00]
    ///   0u32   = [00 00 00 00]  (iteration)
    ///   0u32   = [00 00 00 00]  (scenario)
    ///   0u32   = [00 00 00 00]  (stage)
    #[test]
    fn forward_seed_golden_value() {
        let seed = derive_forward_seed(42, 0, 0, 0);
        // Golden value recorded from siphasher 1.0.2 with zero key.
        assert_eq!(seed, 4_418_977_803_187_233_897_u64);
    }

    // -------------------------------------------------------------------------
    // derive_stage_seed: determinism
    // -------------------------------------------------------------------------

    #[test]
    fn stage_seed_is_deterministic() {
        assert_eq!(derive_stage_seed(42, 0), derive_stage_seed(42, 0));
    }

    #[test]
    fn stage_seed_varies_with_stage() {
        assert_ne!(derive_stage_seed(42, 0), derive_stage_seed(42, 1));
    }

    #[test]
    fn stage_seed_varies_with_base_seed() {
        assert_ne!(derive_stage_seed(42, 0), derive_stage_seed(43, 0));
    }

    // -------------------------------------------------------------------------
    // Cross-function differentiation: 12-byte vs 16-byte and 20-byte wire formats
    // -------------------------------------------------------------------------

    /// `derive_stage_seed(base, 0)` feeds 12 bytes;
    /// `derive_opening_seed(base, 0, 0)` feeds 16 bytes.
    /// SipHash-1-3 incorporates message length into its state, so the two
    /// outputs must differ even when the numeric arguments overlap.
    #[test]
    fn stage_seed_differs_from_opening_seed() {
        assert_ne!(derive_stage_seed(42, 0), derive_opening_seed(42, 0, 0));
    }

    /// `derive_stage_seed(base, 0)` feeds 12 bytes;
    /// `derive_forward_seed(base, 0, 0, 0)` feeds 20 bytes.
    /// SipHash-1-3 incorporates message length into its state, so the two
    /// outputs must differ even when the numeric arguments overlap.
    #[test]
    fn stage_seed_differs_from_forward_seed() {
        assert_ne!(derive_stage_seed(42, 0), derive_forward_seed(42, 0, 0, 0));
    }

    // -------------------------------------------------------------------------
    // Golden value regression: pin derive_stage_seed output
    // -------------------------------------------------------------------------

    /// This value was computed by running the implementation and recording the
    /// output. If this test fails, the SipHash-1-3 wire format or the
    /// `siphasher` crate version has changed in a breaking way.
    ///
    /// Input bytes (little-endian):
    ///   42u64  = [2a 00 00 00 00 00 00 00]
    ///   0u32   = [00 00 00 00]  (stage_id)
    #[test]
    fn stage_seed_golden_value() {
        let seed = derive_stage_seed(42, 0);
        // Golden value recorded from siphasher 1.0.2 with zero key.
        assert_eq!(seed, 983_776_962_555_776_753_u64);
    }
}
