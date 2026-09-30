//! Microbenchmark for Dynamic Cut Selection candidate scoring.
//!
//! Compares the production batched `score_violated_candidates` (one N-row GEMM
//! per pass) against a per-candidate baseline scoring each candidate with its
//! own single-row `dgemm`. Both use the same single-threaded
//! `matrixmultiply::dgemm` kernel, so the comparison isolates the call-shape
//! change (one N-row GEMM vs N individual 1-row GEMMs).
//!
//! Parameterized by candidate count (incl. a ≥ 10,000 case) and `n_state`.

#![allow(
    missing_docs,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::doc_markdown
)]

use cobre_core::StageStateConfig;
use cobre_sddp::cut::{CutPool, CutRowMap};
use cobre_sddp::cut_selection::CutMetadata;
use cobre_sddp::dcs::{DcsParams, DcsScoringScratch, score_violated_candidates};
use cobre_sddp::indexer::{CutStateProjection, StateDim, StateSpace};
use cobre_sddp::lead_time::AnticipatedResolution;
use cobre_sddp::setup::NodeId;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// --- deterministic PRNG (no external rand dep) ------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn draw_f64(state: &mut u64) -> f64 {
    let r = splitmix64(state);
    let bits = (r >> 12) & ((1_u64 << 52) - 1);
    f64::from_bits((1023_u64 << 52) | bits) - 1.5
}

fn fill_f64(buf: &mut [f64], seed: u64) {
    let mut state = seed;
    for slot in buf.iter_mut() {
        *slot = draw_f64(&mut state);
    }
}

// Inline copy of `gemm::gemm_block` (`pub(crate)`, unreachable from a bench):
// identical `matrixmultiply::dgemm` call and strides, so a single-row
// (`k_rows == 1`) invocation reproduces the per-candidate dispatch exactly.
// Must stay faithful to the original — a hand-rolled dot would break the witness.
#[allow(clippy::cast_possible_wrap)]
fn gemm_block(
    coef: &[f64],
    state_block: &[f64],
    k_rows: usize,
    d: usize,
    m_len: usize,
    out: &mut [f64],
) {
    if k_rows == 0 || d == 0 || m_len == 0 {
        return;
    }
    // SAFETY: dimensions are non-zero (checked above); `coef.len() == k_rows*d`,
    // `state_block.len() == m_len*d`, `out.len() == k_rows*m_len` by the call
    // sites below; the three slices come from distinct owners so the raw
    // pointers cannot alias; strides match the row-major layout (transpose
    // access for `state_block`). dgemm is single-threaded in this build.
    unsafe {
        matrixmultiply::dgemm(
            k_rows,
            d,
            m_len,
            1.0,
            coef.as_ptr(),
            d as isize,
            1,
            state_block.as_ptr(),
            1,
            d as isize,
            0.0,
            out.as_mut_ptr(),
            m_len as isize,
            1,
        );
    }
}

/// Per-candidate baseline: replays the filter/gather/score logic of
/// `score_violated_candidates` faithfully, but scores each surviving candidate
/// with its own single-row `gemm_block` call. Returns the violated count.
#[allow(clippy::too_many_arguments)]
fn score_per_candidate_baseline(
    pool: &CutPool,
    state: &StateSpace,
    primal: &[f64],
    col_scale: &[f64],
    resident: &CutRowMap,
    params: &DcsParams,
    current_iteration: u64,
    unscaled_state: &mut Vec<f64>,
    violations: &mut Vec<(f64, u32)>,
    out_selected: &mut Vec<u32>,
) -> usize {
    let n_state = state.n_state;
    let theta = state.theta;

    out_selected.clear();
    violations.clear();

    let theta_raw = if col_scale.is_empty() {
        primal[theta]
    } else {
        col_scale[theta] * primal[theta]
    };

    unscaled_state.clear();
    for j in 0..n_state {
        let c = state.state_to_lp_column(StateDim::new(j)).get();
        let x_raw = if col_scale.is_empty() {
            primal[c]
        } else {
            col_scale[c] * primal[c]
        };
        unscaled_state.push(x_raw);
    }

    for (slot, intercept, coefficients) in pool.active_cuts() {
        if let Some(k1) = params.k1 {
            let age = current_iteration.saturating_sub(pool.metadata(slot).iteration_generated);
            if age >= u64::from(k1) {
                continue;
            }
        }
        if resident.lp_row_for_slot(slot).is_some() {
            continue;
        }
        let mut dot = [0.0_f64; 1];
        gemm_block(coefficients, unscaled_state, 1, n_state, 1, &mut dot);
        let alpha = intercept + dot[0];
        let v = alpha - theta_raw;
        if v > params.epsilon_viol {
            violations.push((v, slot as u32));
        }
    }

    let violated_count = violations.len();
    violations.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let take = (params.nadic as usize).min(violated_count);
    out_selected.extend(violations[..take].iter().map(|&(_, slot)| slot));
    violated_count
}

// --- fixtures ---------------------------------------------------------------

/// Build a pool of `k` active, non-resident, in-window cuts (each one a
/// candidate) with `n_state` coefficients, all generated at iteration 1.
fn make_candidate_pool(k: usize, n_state: usize, seed: u64) -> CutPool {
    let mut pool = CutPool::new(k.max(1), n_state, k.max(1) as u32, 0);
    let mut state = seed;
    let mut coeffs = vec![0.0_f64; n_state];
    for slot in 0..k {
        let intercept = draw_f64(&mut state);
        fill_f64(&mut coeffs, state.wrapping_add(0xABCD_0000 + slot as u64));
        pool.add_cut(NodeId(0), 0, slot as u32, intercept, &coeffs);
    }
    let metadata: Vec<CutMetadata> = (0..k)
        .map(|slot| {
            let mut meta = pool.metadata(slot).clone();
            meta.iteration_generated = 1;
            meta
        })
        .collect();
    let active: Vec<bool> = (0..k).map(|slot| pool.is_active(slot)).collect();
    pool.replace_selection(&metadata, &active);
    pool
}

/// Primal vector: state columns 0..n_state,
/// theta at column 3*n_state, driven strongly negative so most candidates are
/// violated (so the bench exercises the sort + selection alongside scoring).
fn make_primal(n_state: usize, seed: u64) -> Vec<f64> {
    let theta = 3 * n_state;
    let mut primal = vec![0.0_f64; theta + 1];
    fill_f64(&mut primal[..n_state], seed);
    primal[theta] = -1.0e3;
    primal
}

fn bench_one(c: &mut Criterion, k: usize, n_state: usize) {
    let state = StateSpace::new(
        n_state,
        0,
        Vec::new(),
        vec![],
        AnticipatedResolution::default(),
        &vec![0; n_state],
    );
    let cut_state = CutStateProjection::new(
        &state,
        StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    );
    let pool = make_candidate_pool(k, n_state, 0xDEAD_BEEF_CAFE_F00D);
    let primal = make_primal(n_state, 0xFEED_FACE_1234_5678);
    let resident = CutRowMap::new(k.max(1), 0); // empty: every cut a candidate
    // k1 None selects the exactness path (scores all cuts, no age cutoff).
    let params = DcsParams {
        k1: None,
        nadic: 10,
        epsilon_viol: 1e-10,
        ..DcsParams::default()
    };
    let current_iteration = 10;

    let mut group = c.benchmark_group("dcs_score");
    let id = format!("k{k}/n_state{n_state}");

    {
        let mut scratch = DcsScoringScratch::default();
        scratch.reserve(n_state, k);
        let mut out = Vec::with_capacity(k);
        group.bench_with_input(BenchmarkId::new("batched", &id), &id, |b, _| {
            b.iter(|| {
                let count = score_violated_candidates(
                    black_box(&pool),
                    black_box(&state),
                    black_box(&cut_state),
                    black_box(&primal),
                    black_box(&[]),
                    black_box(&resident),
                    black_box(&params),
                    black_box(current_iteration),
                    &mut scratch,
                    &mut out,
                );
                black_box(count);
            });
        });
    }

    {
        let mut unscaled_state = Vec::with_capacity(n_state);
        let mut violations: Vec<(f64, u32)> = Vec::with_capacity(k);
        let mut out = Vec::with_capacity(k);
        group.bench_with_input(BenchmarkId::new("per_candidate", &id), &id, |b, _| {
            b.iter(|| {
                let count = score_per_candidate_baseline(
                    black_box(&pool),
                    black_box(&state),
                    black_box(&primal),
                    black_box(&[]),
                    black_box(&resident),
                    black_box(&params),
                    black_box(current_iteration),
                    &mut unscaled_state,
                    &mut violations,
                    &mut out,
                );
                black_box(count);
            });
        });
    }

    group.finish();
}

fn dcs_score_small_state(c: &mut Criterion) {
    // Aggregated-style n_state, growing candidate counts.
    bench_one(c, 1_000, 155);
    bench_one(c, 10_000, 155);
    bench_one(c, 60_000, 155);
}

fn dcs_score_large_state(c: &mut Criterion) {
    // Disaggregated-style n_state.
    bench_one(c, 10_000, 2_080);
}

criterion_group!(benches, dcs_score_small_state, dcs_score_large_state);
criterion_main!(benches);
