//! MPI-integration-job gate for the cut exchange:
//!
//! - The real-MPI DECOMP K-fan branching rank-invariance gate: `final_lb` is
//!   bitwise identical across world sizes, driving the genuine per-`(rank, pool)`
//!   cut-count exchange under `mpiexec -n 2`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic
)]

/// Real-MPI DECOMP K-fan branching rank-invariance under the default by-scenario
/// scheduler. Launched under `mpiexec -n 2` (the MPI integration job) the real
/// communicator drives the genuine per-`(rank, pool)` cut-count exchange at the
/// fan stage's multi-node backward level; under plain `cargo test` the world is
/// size 1 and the comparison is the single-rank identity. Either way the
/// real-communicator training must match the `LocalBackend` single-rank reference
/// to the bit, and must not return the (now removed) multi-node rejection.
///
/// `create_communicator(BackendKind::Auto)` resolves to `LocalBackend` with no
/// launcher and to the MPI world under `mpiexec`, so the same source compiles and
/// runs in both the no-`mpi` and `mpi` builds without a per-crate feature gate;
/// its genuine 2-rank pass is verified by the real-MPI integration job.
#[cfg(feature = "test-support")]
mod k_fan_branching_rank_invariance {
    use cobre_comm::{BackendKind, Communicator, LocalBackend, create_communicator};
    use cobre_sddp::setup::NodePos;
    use cobre_sddp::test_support::k_fan_setup;
    use cobre_solver::ActiveSolver;

    const K: usize = 8;
    const FORWARD_PASSES: u32 = 6;
    const MAX_ITERATIONS: u32 = 3;

    /// Train a fresh K-fan fixture at world size `comm.size()`, single-threaded,
    /// under the default by-scenario scheduler; return the full converged result
    /// `(final_lb, final_ub, final_ub_std)` — the lower bound and the statistical
    /// upper bound's mean/std, all of which must be rank-shape-invariant.
    fn train_result<C: Communicator>(comm: &C) -> (f64, f64, f64) {
        let mut fixture = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = fixture
            .setup
            .train(&mut solver, comm, 1, ActiveSolver::new, None, None)
            .expect("k_fan training must return Ok (no multi-node rejection)");
        assert!(
            outcome.error.is_none(),
            "k_fan training must not error: {:?}",
            outcome.error
        );
        (
            outcome.result.final_lb,
            outcome.result.final_ub,
            outcome.result.final_ub_std,
        )
    }

    #[test]
    fn k_fan_final_lb_bitwise_invariant_across_world_size() {
        let world =
            create_communicator(BackendKind::Auto).expect("communicator construction must succeed");
        let world_size = world.size();

        // Power self-check: the fan level carries >= 2 cut-generating pools, and
        // at world size >= 2 a single rank's forward passes are fewer than those
        // pools — so at least one pool draws cuts from a strict subset of ranks,
        // the genuine per-(rank, pool) divergence the fix addresses.
        let probe = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let node_graph = &probe.setup.inputs.node_graph;
        let cut_generating = (0..node_graph.nodes.len())
            .filter(|&pos| !node_graph.successors[NodePos(pos)].is_empty())
            .count();
        let fan_nodes = cut_generating - 1;
        assert!(
            fan_nodes >= 2,
            "power: the K-fan must carry >= 2 cut-generating fan pools (got {fan_nodes})"
        );
        const { assert!(FORWARD_PASSES >= 2, "power: forward_passes must be >= 2") };
        if world_size >= 2 {
            let per_rank_max =
                FORWARD_PASSES.div_ceil(u32::try_from(world_size).expect("world size fits u32"));
            assert!(
                usize::try_from(per_rank_max).expect("per_rank_max fits usize") < fan_nodes,
                "power: a single rank's forward passes ({per_rank_max}) must be fewer than the \
                 fan pools ({fan_nodes}) so some pool draws cuts from a strict subset of ranks"
            );
        }

        let (lb_n, ub_n, ub_std_n) = train_result(&world);
        let (lb_1, ub_1, ub_std_1) = train_result(&LocalBackend);
        // The full converged result — lower bound, and the statistical upper
        // bound's mean and std — must not change with workload distribution: the
        // cut set, the canonical-order LB root evaluation, and the canonical-order
        // forward-cost merge are all rank-shape-invariant.
        assert_eq!(
            lb_n.to_bits(),
            lb_1.to_bits(),
            "final_lb at world size {world_size} (real MPI) must be bitwise identical to the \
             single-rank reference"
        );
        assert_eq!(
            ub_n.to_bits(),
            ub_1.to_bits(),
            "final_ub at world size {world_size} (real MPI) must be bitwise identical to the \
             single-rank reference"
        );
        assert_eq!(
            ub_std_n.to_bits(),
            ub_std_1.to_bits(),
            "final_ub_std at world size {world_size} (real MPI) must be bitwise identical to the \
             single-rank reference"
        );
    }
}
