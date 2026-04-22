//! DHAT heap allocation baseline for the cobre-sddp backward pass.
//!
//! This example runs the D19 test case (multi-hydro PAR, 4 plants, 12 stages)
//! under DHAT heap instrumentation to capture a quantitative allocation
//! baseline for the backward pass hot path.
//!
//! # Usage
//!
//! ```text
//! cargo run --example dhat_baseline --features dhat-heap -p cobre-sddp --release
//! ```
//!
//! The `--release` flag is mandatory: debug builds have different allocation
//! behaviour due to optimizer differences and are not representative of the
//! production hot path.
//!
//! # Feature gate
//!
//! The DHAT global allocator is only installed when the `dhat-heap` feature is
//! enabled. Without the feature, the example still compiles and runs, but no
//! profiling is performed and no `dhat-heap.json` is written.
//!
//! # Output
//!
//! On exit the DHAT profiler writes `dhat-heap.json` in the current working
//! directory. Open it in the DHAT viewer at
//! <https://nnethercote.github.io/dh_view/dh_view.html> to inspect per-site
//! allocation counts.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stdout
)]

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use std::path::Path;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_io::{config::StoppingRuleConfig, parse_config};
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
use cobre_solver::highs::HighsSolver;

/// Single-rank stub communicator — mirrors the one in `tests/deterministic.rs`.
struct StubComm;

impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].clone_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let case_dir = Path::new("examples/deterministic/d19-multi-hydro-par");

    println!("Loading D19 case from {:?}", case_dir.canonicalize().ok());

    let config_path = case_dir.join("config.json");
    let mut config = parse_config(&config_path).expect("config must parse");

    config.training.forward_passes = Some(3);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]);

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let cobre_sddp::setup::PrepareStochasticResult {
        system, stochastic, ..
    } = prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
        .expect("prepare_stochastic must succeed");

    let hydro_models =
        prepare_hydro_models(&system, case_dir).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    println!("Starting training (3 forward passes, 10 iterations)...");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");

    let result = outcome.result;

    println!(
        "Training complete: {} iterations, final_lb = {:.4}, reason = {}",
        result.iterations, result.final_lb, result.reason
    );

    #[cfg(feature = "dhat-heap")]
    println!("DHAT profile written to dhat-heap.json");

    #[cfg(not(feature = "dhat-heap"))]
    println!(
        "Warning: dhat-heap feature not enabled. Re-run with --features dhat-heap to produce a profile."
    );
}
