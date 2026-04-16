//! Wire format for MPI cut exchange.
//!
//! During the SDDP backward pass, each MPI rank broadcasts its newly generated
//! cuts to all other ranks via `allgatherv`. Because the coefficient count
//! (`n_state`) is a runtime value, `allgatherv` is called with `T = u8` and
//! cuts are packed into a contiguous byte buffer using the layout below.
//!
//! ## Wire format layout
//!
//! Each cut occupies `24 + n_state * 8` bytes, laid out as:
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//!  0- 3     4   slot_index          (u32, native-endian)
//!  4- 7     4   iteration           (u32, native-endian)
//!  8-11     4   forward_pass_index  (u32, native-endian)
//! 12-15     4   padding             (zeroed, reserved)
//! 16-23     8   intercept           (f64, native-endian)
//! 24 ...    8*n coefficients[0..n]  (f64 each, native-endian)
//! ```
//!
//! The 4-byte padding at offset 12–15 aligns `intercept` to an 8-byte
//! boundary, matching the `#[repr(C)]` layout described in the spec (SS4.2a).
//! All multi-rank executions use the same binary, so native-endian byte order
//! is sufficient and avoids unnecessary byte-swapping.
//!
//! ## Functions
//!
//! - [`cut_wire_size`] — compute the byte size for one cut record.
//! - [`serialize_cut`] — write one cut record into a byte buffer.
//! - [`deserialize_cut`] — read one cut record from a byte buffer.
//! - [`serialize_cuts_to_buffer`] — pack multiple cuts into a new buffer.
//! - [`deserialize_cuts_from_buffer`] — unpack multiple cuts from a buffer.

// ---------------------------------------------------------------------------
// CutWireHeader
// ---------------------------------------------------------------------------

/// Parsed header from a [`cut wire record`](self).
///
/// This struct holds the decoded header fields of a cut wire record.
/// It is a plain Rust struct (not `#[repr(C)]`); byte conversion is handled
/// explicitly by [`serialize_cut`] and [`deserialize_cut`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutWireHeader {
    /// Deterministic slot index in the target [`CutPool`].
    ///
    /// [`CutPool`]: crate::cut::CutPool
    pub slot_index: u32,

    /// Training iteration counter when this cut was generated.
    pub iteration: u32,

    /// Forward pass index within the iteration when this cut was generated.
    pub forward_pass_index: u32,

    /// Intercept of the Benders cut (`α` in `α + β · x`).
    pub intercept: f64,
}

// ---------------------------------------------------------------------------
// cut_wire_size
// ---------------------------------------------------------------------------

/// Return the byte size of one cut wire record with `n_state` coefficients.
///
/// The layout is a 24-byte fixed header followed by `n_state * 8` bytes for
/// the coefficient array:
///
/// ```
/// use cobre_sddp::cut::wire::cut_wire_size;
///
/// assert_eq!(cut_wire_size(0), 24);
/// assert_eq!(cut_wire_size(1), 32);
/// assert_eq!(cut_wire_size(9), 96);
/// assert_eq!(cut_wire_size(2080), 16664);
/// ```
#[inline]
#[must_use]
pub fn cut_wire_size(n_state: usize) -> usize {
    24 + n_state * 8
}

/// Serialize one cut record into `buf` starting at offset 0.
///
/// The header is written as three `u32` values (12 bytes), then 4 bytes of
/// zero padding, then one `f64` intercept (8 bytes). Coefficients follow
/// immediately as native-endian `f64` bytes.
///
/// # Panics (debug builds only)
///
/// Panics if `buf.len() < cut_wire_size(coefficients.len())`.
pub fn serialize_cut(
    buf: &mut [u8],
    slot_index: u32,
    iteration: u32,
    forward_pass_index: u32,
    intercept: f64,
    coefficients: &[f64],
) {
    debug_assert!(
        buf.len() >= cut_wire_size(coefficients.len()),
        "buffer too small: {} < {}",
        buf.len(),
        cut_wire_size(coefficients.len())
    );

    buf[0..4].copy_from_slice(&slot_index.to_ne_bytes());
    buf[4..8].copy_from_slice(&iteration.to_ne_bytes());
    buf[8..12].copy_from_slice(&forward_pass_index.to_ne_bytes());
    buf[12..16].copy_from_slice(&0u32.to_ne_bytes());
    buf[16..24].copy_from_slice(&intercept.to_ne_bytes());

    for (i, &coeff) in coefficients.iter().enumerate() {
        let start = 24 + i * 8;
        buf[start..start + 8].copy_from_slice(&coeff.to_ne_bytes());
    }
}

/// Deserialize one cut record from `buf`, expecting `n_state` coefficients.
///
/// Reads the 24-byte header from fixed offsets and recovers `n_state` `f64`
/// values starting at offset 24.
///
/// After the length `debug_assert`, all slice-to-array conversions use direct
/// fixed-length indexing, which is infallible for the exact sizes used here.
///
/// # Panics (debug builds only)
///
/// Panics if `buf.len() < cut_wire_size(n_state)`.
#[must_use]
pub fn deserialize_cut(buf: &[u8], n_state: usize) -> (CutWireHeader, Vec<f64>) {
    debug_assert!(
        buf.len() >= cut_wire_size(n_state),
        "buffer too small: {} < {}",
        buf.len(),
        cut_wire_size(n_state)
    );

    // All slice-to-array conversions below are infallible: the debug_assert
    // above guarantees buf.len() >= 24 + n_state*8, so the fixed offsets 0..4,
    // 4..8, 8..12, and 16..24 are all within bounds.
    let slot_index = u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let iteration = u32::from_ne_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let forward_pass_index = u32::from_ne_bytes([buf[8], buf[9], buf[10], buf[11]]);
    // bytes 12-15 are padding — intentionally ignored
    let intercept = f64::from_ne_bytes([
        buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
    ]);

    let header = CutWireHeader {
        slot_index,
        iteration,
        forward_pass_index,
        intercept,
    };

    let coefficients: Vec<f64> = (0..n_state)
        .map(|i| {
            let s = 24 + i * 8;
            f64::from_ne_bytes([
                buf[s],
                buf[s + 1],
                buf[s + 2],
                buf[s + 3],
                buf[s + 4],
                buf[s + 5],
                buf[s + 6],
                buf[s + 7],
            ])
        })
        .collect();

    (header, coefficients)
}

/// Serialize multiple cuts into a freshly allocated contiguous byte buffer.
///
/// Each element of `cuts` is a tuple `(slot_index, iteration,
/// forward_pass_index, intercept, coefficients)`.  All cuts must have the
/// same `n_state` coefficient count; `n_state` is passed explicitly so the
/// caller controls the layout without iterating over the slice.
///
/// Returns a `Vec<u8>` of length `cuts.len() * cut_wire_size(n_state)`.
///
/// # Panics (debug builds only)
///
/// Panics if any coefficient slice has length != `n_state`.
#[must_use]
pub fn serialize_cuts_to_buffer(cuts: &[(u32, u32, u32, f64, &[f64])], n_state: usize) -> Vec<u8> {
    let record_size = cut_wire_size(n_state);
    let mut buf = vec![0u8; cuts.len() * record_size];

    for (i, &(slot_index, iteration, forward_pass_index, intercept, coefficients)) in
        cuts.iter().enumerate()
    {
        debug_assert!(
            coefficients.len() == n_state,
            "cut {i} coefficient length {} != n_state {n_state}",
            coefficients.len()
        );
        let start = i * record_size;
        serialize_cut(
            &mut buf[start..start + record_size],
            slot_index,
            iteration,
            forward_pass_index,
            intercept,
            coefficients,
        );
    }

    buf
}

/// Deserialize all cuts from a contiguous byte buffer.
///
/// The buffer must contain a whole number of cut records: its length must be
/// `0` or a multiple of `cut_wire_size(n_state)`. Returns a `Vec` of
/// `(header, coefficients)` pairs in the same order they appear in the buffer.
///
/// # Panics (debug builds only)
///
/// Panics if `buf.len()` is not a multiple of `cut_wire_size(n_state)` (when
/// `n_state > 0`).
#[must_use]
pub fn deserialize_cuts_from_buffer(buf: &[u8], n_state: usize) -> Vec<(CutWireHeader, Vec<f64>)> {
    if buf.is_empty() {
        return Vec::new();
    }

    let record_size = cut_wire_size(n_state);
    debug_assert!(
        buf.len() % record_size == 0,
        "buffer length {} is not a multiple of record size {record_size}",
        buf.len()
    );

    let n_cuts = buf.len() / record_size;
    (0..n_cuts)
        .map(|i| {
            let start = i * record_size;
            deserialize_cut(&buf[start..start + record_size], n_state)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation, // loop indices are small constants in tests
        clippy::cast_precision_loss,      // usize cast to f64 is intentional in tests
        clippy::cast_lossless,            // i32→f64 is lossless but clippy prefers From
    )]

    use super::{
        CutWireHeader, cut_wire_size, deserialize_cut, deserialize_cuts_from_buffer, serialize_cut,
        serialize_cuts_to_buffer,
    };

    #[test]
    fn cut_wire_size_zero_state_returns_24() {
        assert_eq!(cut_wire_size(0), 24);
    }

    #[test]
    fn cut_wire_size_one_state_returns_32() {
        assert_eq!(cut_wire_size(1), 32);
    }

    #[test]
    fn cut_wire_size_three_hydro_ar2_returns_96() {
        // 3-hydro AR(2) system: n_state = 9 → 24 + 9 * 8 = 96
        assert_eq!(cut_wire_size(9), 96);
    }

    #[test]
    fn cut_wire_size_production_scale_returns_16664() {
        // Production-scale: n_state = 2080 → 24 + 2080 * 8 = 16664
        assert_eq!(cut_wire_size(2080), 16664);
    }

    #[test]
    fn round_trip_all_fields_match_exactly() {
        let n_state = 3;
        let coefficients = [1.0_f64, 2.0, 3.0];
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 5, 3, 7, 42.0, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state);

        assert_eq!(header.slot_index, 5);
        assert_eq!(header.iteration, 3);
        assert_eq!(header.forward_pass_index, 7);
        assert_eq!(header.intercept, 42.0);
        assert_eq!(recovered, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn round_trip_verifies_bit_for_bit_coefficient_integrity() {
        // Use values that are not exactly representable in f64 to verify
        // that round-trip preserves the bit pattern exactly.
        let n_state = 4;
        let val = 1.0_f64 / 3.0;
        let coefficients = [val, -val, val * 2.0, f64::MIN_POSITIVE];
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 1, 10, 2, f64::MAX, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state);

        assert_eq!(header.intercept.to_bits(), f64::MAX.to_bits());
        for (orig, got) in coefficients.iter().zip(&recovered) {
            assert_eq!(orig.to_bits(), got.to_bits(), "coefficient mismatch");
        }
    }

    #[test]
    fn byte_offsets_match_wire_format_spec() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let mut buf = vec![0u8; cut_wire_size(3)];

        serialize_cut(&mut buf, 5, 3, 7, 42.0, &coefficients);

        // slot_index at offset 0-3
        assert_eq!(
            u32::from_ne_bytes(buf[0..4].try_into().unwrap()),
            5u32,
            "slot_index at offset 0"
        );
        // iteration at offset 4-7
        assert_eq!(
            u32::from_ne_bytes(buf[4..8].try_into().unwrap()),
            3u32,
            "iteration at offset 4"
        );
        // forward_pass_index at offset 8-11
        assert_eq!(
            u32::from_ne_bytes(buf[8..12].try_into().unwrap()),
            7u32,
            "forward_pass_index at offset 8"
        );
        // padding at offset 12-15 must be zero
        assert_eq!(&buf[12..16], &[0u8; 4], "padding at offset 12 must be zero");
        // intercept at offset 16-23
        assert_eq!(
            f64::from_ne_bytes(buf[16..24].try_into().unwrap()),
            42.0_f64,
            "intercept at offset 16"
        );
        // first coefficient at offset 24
        assert_eq!(
            f64::from_ne_bytes(buf[24..32].try_into().unwrap()),
            1.0_f64,
            "coefficient[0] at offset 24"
        );
    }

    #[test]
    fn round_trip_production_scale_n_state_2080() {
        let n_state = 2080;
        let coefficients: Vec<f64> = (0..n_state).map(|i| i as f64 * 0.001).collect();
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 100, 50, 3, 999.0, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state);

        assert_eq!(header.slot_index, 100);
        assert_eq!(header.iteration, 50);
        assert_eq!(header.forward_pass_index, 3);
        assert_eq!(header.intercept, 999.0);
        assert_eq!(recovered.len(), n_state);
        for (i, (orig, got)) in coefficients.iter().zip(&recovered).enumerate() {
            assert_eq!(orig.to_bits(), got.to_bits(), "mismatch at coefficient {i}");
        }
    }

    #[test]
    fn edge_case_n_state_zero_header_only_24_bytes() {
        let mut buf = vec![0u8; cut_wire_size(0)];
        assert_eq!(buf.len(), 24);

        serialize_cut(&mut buf, 1, 2, 3, -1.0, &[]);
        let (header, coefficients) = deserialize_cut(&buf, 0);

        assert_eq!(header.slot_index, 1);
        assert_eq!(header.iteration, 2);
        assert_eq!(header.forward_pass_index, 3);
        assert_eq!(header.intercept, -1.0);
        assert!(coefficients.is_empty());
    }

    #[test]
    fn edge_case_n_state_one_produces_32_byte_record() {
        let mut buf = vec![0u8; cut_wire_size(1)];
        assert_eq!(buf.len(), 32);

        // Use 2.5 (exactly representable in f64) as a non-PI coefficient.
        let coeff = 2.5_f64;
        serialize_cut(&mut buf, 0, 0, 0, 7.0, &[coeff]);
        let (header, coefficients) = deserialize_cut(&buf, 1);

        assert_eq!(header.intercept, 7.0);
        assert_eq!(coefficients.len(), 1);
        assert_eq!(coefficients[0].to_bits(), coeff.to_bits());
    }

    #[test]
    fn padding_bytes_at_offset_12_to_15_are_zero() {
        let mut buf = vec![0xFFu8; cut_wire_size(2)]; // Pre-fill with 0xFF
        serialize_cut(&mut buf, 1, 1, 1, 1.0, &[1.0, 2.0]);
        assert_eq!(&buf[12..16], &[0u8; 4], "padding bytes must be zero");
    }

    #[test]
    fn multi_cut_five_cuts_round_trip_all_match() {
        let n_state = 3;
        let coefficients: Vec<[f64; 3]> = (0..5u32).map(|i| [f64::from(i); 3]).collect();
        let cuts: Vec<(u32, u32, u32, f64, &[f64])> = coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let idx = i as u32;
                (idx, idx * 2, idx, f64::from(idx) * 10.0, c.as_slice())
            })
            .collect();

        let buf = serialize_cuts_to_buffer(&cuts, n_state);
        assert_eq!(buf.len(), 5 * cut_wire_size(n_state));

        let recovered = deserialize_cuts_from_buffer(&buf, n_state);
        assert_eq!(recovered.len(), 5);

        for (i, (header, coeffs)) in recovered.iter().enumerate() {
            let idx = i as u32;
            assert_eq!(header.slot_index, idx, "slot_index mismatch at cut {i}");
            assert_eq!(header.iteration, idx * 2, "iteration mismatch at cut {i}");
            assert_eq!(
                header.forward_pass_index, idx,
                "forward_pass_index mismatch at cut {i}"
            );
            assert_eq!(
                header.intercept,
                f64::from(idx) * 10.0,
                "intercept mismatch at cut {i}"
            );
            for (j, &c) in coeffs.iter().enumerate() {
                assert_eq!(c, f64::from(idx), "coefficient[{j}] mismatch at cut {i}");
            }
        }
    }

    #[test]
    fn multi_cut_ten_cuts_round_trip_order_preserved() {
        let n_state = 2;
        let all_coefficients: Vec<Vec<f64>> = (0..10u32)
            .map(|i| vec![f64::from(i), -f64::from(i)])
            .collect();
        let cuts: Vec<(u32, u32, u32, f64, &[f64])> = all_coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let idx = i as u32;
                (idx, 0u32, idx, f64::from(idx), c.as_slice())
            })
            .collect();

        let buf = serialize_cuts_to_buffer(&cuts, n_state);
        let recovered = deserialize_cuts_from_buffer(&buf, n_state);

        assert_eq!(recovered.len(), 10);
        for (i, (header, coeffs)) in recovered.iter().enumerate() {
            let idx = i as u32;
            assert_eq!(header.slot_index, idx);
            assert_eq!(coeffs[0].to_bits(), f64::from(idx).to_bits());
            assert_eq!(coeffs[1].to_bits(), (-f64::from(idx)).to_bits());
        }
    }

    #[test]
    fn deserialize_cuts_from_empty_buffer_returns_empty_vec() {
        let result = deserialize_cuts_from_buffer(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn cut_wire_header_derives_debug_clone_copy_partialeq() {
        let h = CutWireHeader {
            slot_index: 1,
            iteration: 2,
            forward_pass_index: 3,
            intercept: 4.0,
        };
        let cloned = h;
        assert_eq!(h, cloned);
        let debug_str = format!("{h:?}");
        assert!(!debug_str.is_empty());
    }
}
