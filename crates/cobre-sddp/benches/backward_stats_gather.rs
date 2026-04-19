//! Criterion micro-benchmark for the per-stage `StageWorkerStatsBuffer` gather.
//!
//! Locks in the zero-allocation property of the gather loop introduced in
//! epic-04b T003. Target: < 100 microseconds per stage at production sizing
//! (`n_workers = 10`, `n_openings = 20`).
//!
//! Run with: `cargo bench --bench backward_stats_gather`

#![allow(missing_docs)]

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

criterion_group!(benches, bench_gather);
criterion_main!(benches);
