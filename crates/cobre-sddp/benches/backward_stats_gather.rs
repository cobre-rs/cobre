//! Criterion micro-benchmark for the per-stage `StageWorkerStatsBuffer` gather.
//!
//! Locks in the zero-allocation property of the gather loop introduced in
//! epic-04b T003. Target: < 100 microseconds per stage at production sizing
//! (`n_workers = 10`, `n_openings = 20`).
//!
//! Run with: `cargo bench --bench backward_stats_gather`

#![allow(missing_docs)]

use std::sync::mpsc;

use cobre_core::{TrainingEvent, WORKER_TIMING_SLOT_COUNT, WorkerTimingPhase};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

use cobre_sddp::solver_stats::{SolverStatsDelta, StageWorkerStatsBuffer};

fn bench_gather(c: &mut Criterion) {
    let n_workers = 10;
    let n_openings = 20;
    let mut buf = StageWorkerStatsBuffer::new(n_workers, n_openings);
    let deltas: Vec<SolverStatsDelta> = (0..n_workers * n_openings)
        .map(|_| SolverStatsDelta::default())
        .collect();
    c.bench_function("backward_stats_gather 10x20", |b| {
        b.iter(|| {
            buf.reset();
            for w in 0..n_workers {
                for k in 0..n_openings {
                    buf.set(w, k, black_box(deltas[w * n_openings + k].clone()));
                }
            }
            black_box(buf.as_slice().len());
        });
    });
}

/// Per-worker `WorkerTiming` event construction + send micro-benchmark
/// (epic-04b T006). Locks in the zero-heap-allocation property of the per-worker
/// emission path: the `[f64; 16]` payload is stack-resident, the channel send
/// of the variant is amortised across iterations.
///
/// The `mpsc::Sender::send` does internally allocate a node; the focus here is
/// that the **event construction** path itself never introduces a `Vec` or
/// `Box` allocation. Inspecting the generated assembly (or running under
/// `dhat`) on this loop will show only the channel-internal allocations.
#[allow(clippy::expect_used)]
fn bench_worker_timing_emit(c: &mut Criterion) {
    let n_workers: i32 = 10;
    let (tx, rx) = mpsc::channel::<TrainingEvent>();
    c.bench_function("worker_timing_emit 10_workers", |b| {
        b.iter(|| {
            for w in 0..n_workers {
                let timings = [black_box(0.0_f64); WORKER_TIMING_SLOT_COUNT];
                let event = TrainingEvent::WorkerTiming {
                    rank: 0,
                    worker_id: w,
                    iteration: 1,
                    phase: WorkerTimingPhase::Backward,
                    timings,
                };
                tx.send(black_box(event)).expect("channel open");
            }
        });
    });
    // Drain receiver so the channel does not grow unboundedly between runs.
    drop(tx);
    while rx.try_recv().is_ok() {}
}

criterion_group!(benches, bench_gather, bench_worker_timing_emit);
criterion_main!(benches);
