//! Criterion micro-benchmark for the per-solve `PatchBuffer::fill_load_patches`
//! + `fill_z_inflow_patches` pair, at two shapes: (a) 5 buses x 3 blocks x 165
//! hydros, and (b) 64 buses x 24 blocks x 400 hydros. Bus positions are in
//! reverse (non-slot) order, matching the runtime caller's `load_bus_indices`.

#![allow(missing_docs, clippy::unwrap_used, clippy::expect_used)]

use cobre_sddp::indexer::BlockGrid;
use cobre_sddp::lp::builder::PatchBuffer;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const SHAPES: &[(&str, usize, usize, usize)] =
    &[("5x3x165", 5, 3, 165), ("64x24x400", 64, 24, 400)];

fn bench_patch_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("patch_fill");
    for &(name, n_buses, n_blocks, n_hydros) in SHAPES {
        let mut buf = PatchBuffer::new(n_hydros, 0, n_buses, n_blocks, 0, 0, 0);
        let load_rhs = vec![100.0_f64; n_buses * n_blocks];
        let bus_positions: Vec<usize> = (0..n_buses).rev().collect();
        let z_inflow_rhs = vec![0.5_f64; n_hydros];
        let row_scale: [f64; 0] = [];
        let load_row_start = 0_usize;
        let z_inflow_row_start = n_buses * n_blocks;

        group.bench_function(name, |b| {
            b.iter(|| {
                buf.fill_load_patches(
                    black_box(load_row_start),
                    BlockGrid::new(n_blocks, 0),
                    black_box(&load_rhs),
                    black_box(&bus_positions),
                    black_box(&row_scale),
                );
                buf.fill_z_inflow_patches(
                    black_box(z_inflow_row_start),
                    black_box(&z_inflow_rhs),
                    black_box(&row_scale),
                );
                black_box(buf.forward_patch_count());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_patch_fill);
criterion_main!(benches);
