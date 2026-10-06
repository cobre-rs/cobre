//! Deterministic cross-rank reconciliation of error flags before lockstep
//! collectives, and of the rank-variant stop inputs at the iteration boundary.

use cobre_comm::{CommError, Communicator, ReduceOp};

use crate::SddpError;
use crate::config::ShutdownSource;

/// Reconcile whether every rank's local step succeeded before a lockstep
/// collective: returns `true` iff no rank reported a failure, so all ranks enter
/// or skip the next collective together.
///
/// `ReduceOp::Max` over a 0/non-zero flag is order-independent, so the global flag
/// is identical on every rank regardless of rank count (the declaration-order/
/// bit-for-bit invariant). Reports only the flag, not an [`SddpError`], so the CLI
/// can reconcile with its own `CliError`; `reconcile_error_flag` wraps it for the
/// `SddpError` path.
///
/// # Errors
///
/// Returns [`CommError`] when the reconciling `allreduce` transport fails.
pub fn reconcile_global_ok<C: Communicator>(
    local_ok: bool,
    comm: &C,
    scratch: &mut [i32; 1],
) -> Result<bool, CommError> {
    if comm.size() == 1 {
        return Ok(local_ok);
    }

    scratch[0] = i32::from(!local_ok);

    let mut reduced = [0_i32];
    comm.allreduce(&scratch[..], &mut reduced, ReduceOp::Max)?;

    Ok(reduced[0] == 0)
}

/// Reduce every rank's local outcome to one global outcome that all ranks return
/// identically, so they enter or skip the next collective in lockstep.
///
/// # Errors
///
/// Returns [`SddpError::Communication`] when the `allreduce` transport fails or
/// when a peer failed while this rank did not; returns this rank's own error when
/// it was the failing rank.
pub(crate) fn reconcile_error_flag<C: Communicator>(
    local_ok: Result<(), SddpError>,
    comm: &C,
    scratch: &mut [i32; 1],
) -> Result<(), SddpError> {
    if reconcile_global_ok(local_ok.is_ok(), comm, scratch)? {
        Ok(())
    } else {
        Err(local_ok.err().unwrap_or_else(|| {
            SddpError::Communication(CommError::CollectiveFailed {
                operation: "reconcile_error_flag",
                mpi_error_code: 0,
                message: "a peer rank reported a local failure; failing this \
                          phase on every rank in lockstep"
                    .to_string(),
            })
        }))
    }
}

/// Reconcile a rank-local `Result<T, SddpError>` across ranks before a lockstep
/// collective: every rank returns `Ok(payload)` iff no rank failed, otherwise
/// every rank returns `Err` in lockstep.
///
/// # Errors
///
/// Returns whatever [`reconcile_error_flag`] returns, or
/// [`SddpError::Communication`] if the flag agreed but this rank held no payload.
pub(crate) fn reconcile_result<T, C: Communicator>(
    local: Result<T, SddpError>,
    comm: &C,
    scratch: &mut [i32; 1],
) -> Result<T, SddpError> {
    let (payload, local_ok) = match local {
        Ok(value) => (Some(value), Ok(())),
        Err(e) => (None, Err(e)),
    };
    reconcile_error_flag(local_ok, comm, scratch)?;
    payload.ok_or_else(|| {
        SddpError::Communication(CommError::CollectiveFailed {
            operation: "reconcile_result",
            mpi_error_code: 0,
            message: "cross-rank reconcile agreed but this rank held no payload".to_string(),
        })
    })
}

/// The stop-decision inputs each rank reads locally: its shutdown request and
/// its elapsed training time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StopInputs {
    pub(crate) shutdown: Option<ShutdownSource>,
    pub(crate) wall_time_seconds: f64,
}

impl StopInputs {
    pub(crate) fn vote(self) -> [f64; 2] {
        // Rationale: a shutdown level is 0, 1 or 2, exact in an f64.
        #[allow(clippy::cast_precision_loss)]
        let level = self.shutdown.map_or(0, ShutdownSource::level) as f64;
        [level, self.wall_time_seconds]
    }

    pub(crate) fn from_vote(vote: [f64; 2]) -> Self {
        // Rationale: the agreed level is the largest of the exact levels 0, 1 or 2.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level = vote[0] as usize;
        Self {
            shutdown: ShutdownSource::from_level(level),
            wall_time_seconds: vote[1],
        }
    }
}

/// Agree the stop inputs across ranks: every rank receives the strongest
/// shutdown source and the largest elapsed time, so every rank evaluates the
/// same stop decision and leaves the training loop at the same iteration.
///
/// `ReduceOp::Max` over exact values is order-independent, so the agreed inputs
/// are bit-identical on every rank regardless of rank count.
///
/// # Errors
///
/// Returns [`CommError`] when the agreeing `allreduce` transport fails.
pub(crate) fn agree_stop_inputs<C: Communicator>(
    local: StopInputs,
    comm: &C,
) -> Result<StopInputs, CommError> {
    if comm.size() == 1 {
        return Ok(local);
    }

    let mut agreed = [0.0_f64; 2];
    comm.allreduce(&local.vote(), &mut agreed, ReduceOp::Max)?;

    Ok(StopInputs::from_vote(agreed))
}

#[cfg(test)]
mod tests {
    use super::{
        StopInputs, agree_stop_inputs, reconcile_error_flag, reconcile_global_ok, reconcile_result,
    };
    use crate::SddpError;
    use crate::config::ShutdownSource;
    use crate::convergence::convergence::ConvergenceMonitor;
    use crate::forward::{ForwardResult, SyncResult};
    use crate::stopping_rule::{StopMask, StoppingMode, StoppingRule, StoppingRuleSet};
    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};

    enum Mode {
        Reduce,
        Transport,
        Forbidden,
    }

    struct ReconcileStub {
        size: usize,
        peer_flag: i32,
        mode: Mode,
    }

    impl Communicator for ReconcileStub {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("reconcile_error_flag does not call allgatherv")
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            match self.mode {
                Mode::Forbidden => {
                    unreachable!("single-rank fast path must not enter a collective")
                }
                Mode::Transport => Err(CommError::CollectiveFailed {
                    operation: "allreduce",
                    mpi_error_code: 7,
                    message: "simulated transport failure".to_string(),
                }),
                Mode::Reduce => {
                    assert_eq!(op, ReduceOp::Max, "reconcile must reduce with Max");
                    assert_eq!(send.len(), recv.len());
                    for (s, r) in send.iter().zip(recv.iter_mut()) {
                        *r = upcast_i32(downcast_i32(*s).max(self.peer_flag));
                    }
                    Ok(())
                }
            }
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("reconcile_error_flag does not call broadcast")
        }

        fn barrier(&self) -> Result<(), CommError> {
            unreachable!("reconcile_error_flag does not call barrier")
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            self.size
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    // forbid-unsafe crate: `&dyn Any` dispatch, not a transmute; the stub only
    // reduces the i32 flag `reconcile_error_flag` sends.
    fn downcast_i32<T: CommData>(value: T) -> i32 {
        use std::any::Any;
        *(&value as &dyn Any)
            .downcast_ref::<i32>()
            .expect("reconcile stub only reduces i32 flags")
    }

    fn upcast_i32<T: CommData>(value: i32) -> T {
        use std::any::Any;
        *(&value as &dyn Any)
            .downcast_ref::<T>()
            .expect("reconcile stub only reduces i32 flags")
    }

    fn downcast_f64<T: CommData>(value: T) -> f64 {
        use std::any::Any;
        *(&value as &dyn Any)
            .downcast_ref::<f64>()
            .expect("stop-vote stub only reduces f64 votes")
    }

    fn upcast_f64<T: CommData>(value: f64) -> T {
        use std::any::Any;
        *(&value as &dyn Any)
            .downcast_ref::<T>()
            .expect("stop-vote stub only reduces f64 votes")
    }

    /// A two-rank peer whose `allreduce` combines this rank's vote with the
    /// other rank's `peer` vote.
    struct StopVoteStub {
        size: usize,
        peer: [f64; 2],
        mode: Mode,
    }

    impl Communicator for StopVoteStub {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("agree_stop_inputs does not call allgatherv")
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            match self.mode {
                Mode::Forbidden => {
                    unreachable!("single-rank fast path must not enter a collective")
                }
                Mode::Transport => Err(CommError::CollectiveFailed {
                    operation: "allreduce",
                    mpi_error_code: 7,
                    message: "simulated transport failure".to_string(),
                }),
                Mode::Reduce => {
                    assert_eq!(op, ReduceOp::Max, "the stop inputs must reduce with Max");
                    assert_eq!(send.len(), 2);
                    assert_eq!(recv.len(), 2);
                    for ((s, r), peer) in send.iter().zip(recv.iter_mut()).zip(self.peer) {
                        *r = upcast_f64(downcast_f64(*s).max(peer));
                    }
                    Ok(())
                }
            }
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("agree_stop_inputs does not call broadcast")
        }

        fn barrier(&self) -> Result<(), CommError> {
            unreachable!("agree_stop_inputs does not call barrier")
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            self.size
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    #[test]
    fn single_rank_ok_passes_through_without_collective() {
        let comm = ReconcileStub {
            size: 1,
            peer_flag: 0,
            mode: Mode::Forbidden,
        };
        let mut scratch = [0_i32];
        assert!(reconcile_error_flag(Ok(()), &comm, &mut scratch).is_ok());
    }

    #[test]
    fn single_rank_err_passes_through_without_collective() {
        let comm = ReconcileStub {
            size: 1,
            peer_flag: 0,
            mode: Mode::Forbidden,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(
            Err(SddpError::Validation("local".to_string())),
            &comm,
            &mut scratch,
        );
        assert!(matches!(out, Err(SddpError::Validation(_))));
    }

    #[test]
    fn multi_rank_all_ok_returns_ok() {
        let comm = ReconcileStub {
            size: 4,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(reconcile_error_flag(Ok(()), &comm, &mut scratch).is_ok());
    }

    #[test]
    fn multi_rank_failing_rank_returns_its_own_error() {
        let comm = ReconcileStub {
            size: 4,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(
            Err(SddpError::Infeasible {
                stage: 2,
                iteration: 3,
                scenario: 1,
            }),
            &comm,
            &mut scratch,
        );
        assert!(matches!(out, Err(SddpError::Infeasible { .. })));
    }

    #[test]
    fn multi_rank_healthy_peer_of_failing_rank_returns_err() {
        let comm = ReconcileStub {
            size: 4,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(Ok(()), &comm, &mut scratch);
        assert!(matches!(out, Err(SddpError::Communication(_))));
    }

    #[test]
    fn lockstep_agreement_every_rank_gets_err_when_one_fails() {
        let ranks: [(Result<(), SddpError>, i32); 3] = [
            (Ok(()), 1),
            (Err(SddpError::Validation("rank1 failed".to_string())), 0),
            (Ok(()), 1),
        ];
        for (local_ok, peer_flag) in ranks {
            let comm = ReconcileStub {
                size: 3,
                peer_flag,
                mode: Mode::Reduce,
            };
            let mut scratch = [0_i32];
            assert!(
                reconcile_error_flag(local_ok, &comm, &mut scratch).is_err(),
                "every rank must return Err in lockstep when one rank fails"
            );
        }
    }

    #[test]
    fn transport_failure_surfaces_as_communication_error() {
        let comm = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Transport,
        };
        let mut scratch = [0_i32];
        let out = reconcile_error_flag(Ok(()), &comm, &mut scratch);
        assert!(matches!(out, Err(SddpError::Communication(_))));
    }

    fn dummy_forward_result() -> ForwardResult {
        ForwardResult {
            scenario_costs: Vec::new(),
            elapsed_ms: 0,
            lp_solves: 0,
            setup_time_ms: 0,
            load_imbalance_ms: 0,
            scheduling_overhead_ms: 0,
            stage_stats: Vec::new(),
        }
    }

    /// Forward-phase deadlock-freedom: both ranks return `Err` before `sync_forward`'s collective (avoiding deadlock).
    #[test]
    fn reconcile_result_fails_both_ranks_before_forward_collective() {
        let failing = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let rank1 = reconcile_result::<ForwardResult, _>(
            Err(SddpError::Infeasible {
                stage: 1,
                iteration: 3,
                scenario: 0,
            }),
            &failing,
            &mut scratch,
        );
        assert!(matches!(rank1, Err(SddpError::Infeasible { .. })));

        let healthy = ReconcileStub {
            size: 2,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let rank0 = reconcile_result(Ok(dummy_forward_result()), &healthy, &mut scratch);
        assert!(matches!(rank0, Err(SddpError::Communication(_))));
    }

    /// Backward-phase deadlock-freedom: both ranks return `Err` before `sync_level_records`' collective (avoiding deadlock).
    #[test]
    fn reconcile_result_fails_both_ranks_before_backward_collective() {
        let failing = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let rank1 = reconcile_result::<usize, _>(
            Err(SddpError::Infeasible {
                stage: 4,
                iteration: 2,
                scenario: 1,
            }),
            &failing,
            &mut scratch,
        );
        assert!(matches!(rank1, Err(SddpError::Infeasible { .. })));

        let healthy = ReconcileStub {
            size: 2,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        let rank0 = reconcile_result::<usize, _>(Ok(7), &healthy, &mut scratch);
        assert!(matches!(rank0, Err(SddpError::Communication(_))));
    }

    /// Finalize deadlock-freedom: all ranks skip `broadcast_basis_cache` in lockstep when any rank errors.
    #[test]
    fn finalize_reconcile_makes_all_ranks_skip_broadcast_in_lockstep() {
        let clean_peer_failed = ReconcileStub {
            size: 2,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_error_flag(Ok(()), &clean_peer_failed, &mut scratch),
            Err(SddpError::Communication(_))
        ));

        let erroring = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_error_flag(
                Err(SddpError::Validation("boom".to_string())),
                &erroring,
                &mut scratch,
            ),
            Err(SddpError::Validation(_))
        ));
    }

    /// Single-rank `reconcile_global_ok` returns `local_ok` unchanged and enters no
    /// collective (`Mode::Forbidden` panics if `allreduce` is reached).
    #[test]
    fn reconcile_global_ok_single_rank_skips_collective() {
        let comm = ReconcileStub {
            size: 1,
            peer_flag: 0,
            mode: Mode::Forbidden,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_global_ok(true, &comm, &mut scratch),
            Ok(true)
        ));
        assert!(matches!(
            reconcile_global_ok(false, &comm, &mut scratch),
            Ok(false)
        ));
    }

    /// CLI simulation-phase deadlock-freedom: `reconcile_global_ok` makes both ranks fail before `merge_simulation_metadata`'s collective (avoiding deadlock).
    #[test]
    fn reconcile_global_ok_fails_both_ranks_before_post_sim_collective() {
        let failing = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_global_ok(false, &failing, &mut scratch),
            Ok(false)
        ));

        let healthy = ReconcileStub {
            size: 2,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_global_ok(true, &healthy, &mut scratch),
            Ok(false)
        ));
    }

    /// CLI setup-phase deadlock-freedom: `reconcile_global_ok` makes all ranks fail before the post-export barrier (avoiding deadlock).
    #[test]
    fn reconcile_global_ok_fails_both_ranks_before_post_export_barrier() {
        let root_failed = ReconcileStub {
            size: 2,
            peer_flag: 0,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_global_ok(false, &root_failed, &mut scratch),
            Ok(false)
        ));

        let peer = ReconcileStub {
            size: 2,
            peer_flag: 1,
            mode: Mode::Reduce,
        };
        let mut scratch = [0_i32];
        assert!(matches!(
            reconcile_global_ok(true, &peer, &mut scratch),
            Ok(false)
        ));
    }

    fn input(shutdown: Option<ShutdownSource>, wall_time_seconds: f64) -> StopInputs {
        StopInputs {
            shutdown,
            wall_time_seconds,
        }
    }

    fn reduce_against(peer: StopInputs) -> StopVoteStub {
        StopVoteStub {
            size: 2,
            peer: peer.vote(),
            mode: Mode::Reduce,
        }
    }

    /// One rank's first stopping iteration, reason and mask, agreeing each
    /// iteration's inputs with the other rank's; `(0, None, empty)` if it never stops.
    fn rank_stop(
        rules: &StoppingRuleSet,
        own: &[StopInputs],
        peer: &[StopInputs],
    ) -> (u64, Option<&'static str>, StopMask) {
        let mut monitor = ConvergenceMonitor::new(rules.clone());
        let sync = SyncResult {
            global_ub_mean: 110.0,
            global_ub_std: 5.0,
            ci_95_half_width: 2.0,
            sync_time_ms: 10,
        };
        for (&local, &other) in own.iter().zip(peer) {
            let agreed = agree_stop_inputs(local, &reduce_against(other))
                .expect("the stub reduces without a transport failure");
            if let Some(source) = agreed.shutdown {
                monitor.set_shutdown(source);
            }
            let decision = monitor.update(100.0, &sync, agreed.wall_time_seconds);
            if decision.should_stop() {
                return (
                    monitor.iteration_count(),
                    decision.termination_reason(),
                    decision.mask(),
                );
            }
        }
        (0, None, StopMask::default())
    }

    fn two_ranks(
        rules: Vec<StoppingRule>,
        inputs_a: &[StopInputs],
        inputs_b: &[StopInputs],
    ) -> [(u64, Option<&'static str>, StopMask); 2] {
        assert_eq!(inputs_a.len(), inputs_b.len());
        let rules = StoppingRuleSet {
            rules,
            mode: StoppingMode::Any,
        };
        [
            rank_stop(&rules, inputs_a, inputs_b),
            rank_stop(&rules, inputs_b, inputs_a),
        ]
    }

    #[test]
    fn stop_inputs_agreement_on_one_rank_returns_the_local_inputs_without_a_collective() {
        let comm = StopVoteStub {
            size: 1,
            peer: [2.0, 99.0],
            mode: Mode::Forbidden,
        };
        for local in [
            input(None, 0.0),
            input(Some(ShutdownSource::Cooperative), 3.5),
            input(Some(ShutdownSource::Signal), 12.25),
        ] {
            assert_eq!(
                agree_stop_inputs(local, &comm).expect("no collective, no transport"),
                local
            );
        }
    }

    #[test]
    fn stop_inputs_agreement_takes_the_strongest_shutdown_source_across_ranks() {
        use ShutdownSource::{Cooperative, Signal};
        let cases = [
            (None, None, None),
            (None, Some(Cooperative), Some(Cooperative)),
            (Some(Cooperative), None, Some(Cooperative)),
            (Some(Cooperative), Some(Signal), Some(Signal)),
            (Some(Signal), Some(Cooperative), Some(Signal)),
            (None, Some(Signal), Some(Signal)),
        ];
        for (local, peer, expected) in cases {
            let agreed = agree_stop_inputs(input(local, 1.0), &reduce_against(input(peer, 1.0)))
                .expect("the stub reduces without a transport failure");
            assert_eq!(
                agreed,
                input(expected, 1.0),
                "local {local:?}, peer {peer:?}"
            );
        }
    }

    #[test]
    fn stop_inputs_agreement_takes_the_largest_elapsed_time_across_ranks() {
        for (local, peer, expected) in [(9.9, 10.1, 10.1), (10.1, 9.9, 10.1), (4.25, 4.25, 4.25)] {
            let agreed = agree_stop_inputs(input(None, local), &reduce_against(input(None, peer)))
                .expect("the stub reduces without a transport failure");
            assert_eq!(agreed, input(None, expected), "local {local}, peer {peer}");
        }
    }

    #[test]
    fn stop_inputs_agreement_reports_a_transport_failure() {
        let comm = StopVoteStub {
            size: 2,
            peer: [0.0; 2],
            mode: Mode::Transport,
        };
        assert!(matches!(
            agree_stop_inputs(input(Some(ShutdownSource::Signal), 1.0), &comm),
            Err(CommError::CollectiveFailed { .. })
        ));
    }

    #[test]
    fn ranks_with_a_divergent_shutdown_flag_stop_at_the_same_iteration_with_the_same_reason() {
        let cooperative = Some(ShutdownSource::Cooperative);
        let rank_a = [
            input(None, 0.0),
            input(cooperative, 0.0),
            input(cooperative, 0.0),
        ];
        let rank_b = [input(None, 0.0); 3];
        let [a, b] = two_ranks(
            vec![StoppingRule::IterationLimit { limit: 100 }],
            &rank_a,
            &rank_b,
        );
        assert_eq!((a.0, a.1), (2, Some("graceful_shutdown")));
        assert_eq!(a, b);
    }

    #[test]
    fn ranks_with_divergent_clocks_stop_on_the_time_limit_at_the_same_iteration() {
        let rank_a = [input(None, 1.0), input(None, 5.0), input(None, 9.9)];
        let rank_b = [input(None, 1.2), input(None, 5.1), input(None, 10.1)];
        let [a, b] = two_ranks(
            vec![
                StoppingRule::TimeLimit { seconds: 10.0 },
                StoppingRule::IterationLimit { limit: 100 },
            ],
            &rank_a,
            &rank_b,
        );
        assert_eq!((a.0, a.1), (3, Some("time_limit")));
        assert_eq!(a, b);
    }

    #[test]
    fn a_signal_on_one_rank_sets_the_signal_source_on_every_rank() {
        let rank_a = [
            input(None, 0.0),
            input(None, 0.0),
            input(Some(ShutdownSource::Signal), 0.0),
        ];
        let rank_b = [input(None, 0.0); 3];
        let [a, b] = two_ranks(
            vec![StoppingRule::IterationLimit { limit: 3 }],
            &rank_a,
            &rank_b,
        );
        for (rank, (iteration, reason, mask)) in [("A", a), ("B", b)] {
            assert_eq!(
                (iteration, reason),
                (3, Some("iteration_limit")),
                "rank {rank}"
            );
            assert!(mask.contains(StopMask::SIGNAL), "rank {rank}");
            assert!(mask.contains(StopMask::SHUTDOWN), "rank {rank}");
        }
        assert_eq!(a, b);
    }
}
