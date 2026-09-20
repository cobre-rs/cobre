# Performance sweep log (register pin 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c; the tickets quote the scaffold pin a136840d)

Appended by the sweep tickets in order; every figure a later ticket records is admissible only under the binary and host below.

## Binary

head_sha	376c3acee8a0feab5b2b30e16cc9b3bc42ccb4de
register_pin	077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c
pin_check	crates/ + Cargo.toml + Cargo.lock at HEAD are byte-identical to the register pin (git diff --quiet 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c HEAD -- crates Cargo.toml Cargo.lock); HEAD != pin because plans/ is tracked and the evaluation commits under it
surface_drift_outside_crates	1 file changed, 45 insertions(+), 3 deletions(-) (docs/design/reserved-seams-and-deferred-debt.md, reconciliation commits; not compiled)
ticket_pin	a136840d4f2ea137f685f0af6dac04254b983b60 (scaffold pin the tickets quote; crates/ differ from it, so every figure is a register-pin figure)
build_command	cargo build --profile profiling --features mpi --bin cobre
binary	target/profiling/cobre
binary_sha256	aa82c78c50cd94885a9a5c4db973fb4af3e4648a0ace6470b57f8438a3fdec2d
build_id	de4953977d7744db4709abfd6ff9dee49381a399
cargo_profile	profiling (inherits release, debug = 1, strip = "none"; no Cargo.toml edit)
features	default(highs)+mpi (cobre-cli mpi = ["cobre-comm/mpi"]; one binary serves 4t, 2t and 2x2; --comm-backend local keeps 4t/2t single-process)
solver_backend	highs (default feature)
unstripped_check	file: not stripped; readelf -S: .debug_line present
note	release strips symbols and dist adds thin LTO, so neither is substituted; a stripped binary reaching the profiler is the symbolless-binary error

## Host discipline

cpu_model	12th Gen Intel(R) Core(TM) i7-12700KF
governor	powersave (disclosed, not changed)
topology	lscpu -e: cpus 0-15 are the eight SMT P-cores (cores 0-7, 4.9-5.0 GHz max), cpus 16-19 are E-cores (cores 8-11, 3.8 GHz max); E-cores never enter a timed run
core_set_4t	0,2,4,6 (four distinct physical P-cores, one SMT sibling each)
core_set_2t	0,2
core_set_2x2	rank0=0,2 rank1=4,6 (applied by the wrapper measurements/_wrap/rank-wrapper.sh, never around mpiexec)
perf_run_pcores	perf-run.sh already pins 4t to 0,2,4,6, 2t to 0,2 and 2x2 per rank (the ticket's 'default PCORES=0-15 overridden' premise is stale; nothing is edited in the harness)
perf	perf version 7.2.5-200.fc44.x86_64
perf_event_paranoid	2
rule	paranoid=2: user-space samples of our own process resolve, kernel symbols do not; no claim may attribute cost to a kernel frame
flamegraph_renderer	none (nothing is installed by this epic)
perf_deliverable	perf report --stdio --no-children text (SVG is optional garnish, never a gate)
mpiexec	/opt/mpich/bin/mpiexec (HYDRA build details:)
rank_wrapper	ABSENT at measurements/_wrap/rank-wrapper.sh — perf-run.sh needs it for 2x2; the collective ticket creates it before its first run
protocol_bound_4t	228.319 s (BACKLOG 'Protocol bound:' == measurements/CAL/median.txt); timeout 3x = 685 s
protocol_bound_2t	30.434 s (BACKLOG 'Protocol bound (enumerated):' == measurements/CAL-ENUM/median.txt; measured, not UNMEASURED timeout-3x); timeout 3x = 92 s
unmeasured_reasons	timeout / unexercised-path / mpi-unavailable / case-infeasible
materiality	>= 3% of the median phase wall, or an allocation site holding >= 1% of user-space samples; below that a claim closes not-material

## Decks (staged once; source digests fence both corpora)

sampled_deck	/home/rogerio/git/cobre-bridge/example/cobre_reduzido -> measurements/_case/deck (case id reduzido). The tickets name cobre_reduzido_2, which was lost from the gitignored cobre-bridge example/ tree; the owner re-sanctioned cobre_reduzido and the harness (perf-run.sh DECK_SAMPLED) and the CAL run already use it
enumerated_deck	/home/rogerio/git/cobre-bridge/example/cobre-mar-26-rv2-reduced -> measurements/_case/deck-enumerated (case id mar-26-enumerated; its dated output/ tree is included in the digest)
digest_file	measurements/_case/source-sha256.txt (sha256_before == sha256_after == sha256_staged per deck; the verification ticket re-digests both sources against it)
git_state	plans/ is tracked in this repository, so measurements/_case/.gitignore keeps the deck copies out of git while source-sha256.txt is committed
harness_reads	perf-run.sh copies its DECK_* source into a per-run scratch dir and re-digests it (exit 6 on mutation); the staged copies are the sweep-frozen reference the digests bind to
cal_deck_sha256	CAL env.txt deck_sha256 deca695f5aeffac36778e0220e3a32d951cf2a982da752b096ba4ad45c1c086c; CAL-ENUM 3eae6cb5c36149b66fe38a8ed151c4d1efdcc3b263713a4ce1a3489295c88598

## Claim table

rows	32 (32 schedulable; 2t 3, 2x2 2, 4t 27)
station_counts	core-io 8, stochastic 7, solver-comm 3, sddp 10, cli-python 3, build-ci 0, test-corpus 0
parked	PD-004 (pending-a-profile exemption; the sddp queue's existence row is folded into it), PD-005-residual (Wave-2 residual)
layout_allowance	PD-047 (sddp): station stated 4t, derived 2t accepted by the owner 2026-09-20; the stated-layout slip is returned to the station for correction

## Bounced claims (anchor-missing at the pin; returned to their station; no measurements/<ID>/ directory)

none	every queued anchor resolves at the register pin

## Preflight answers inherited by the measuring tickets

mpiexec	present — the collective ticket runs 2x2; a 2x2 row that cannot run is case-infeasible, not mpi-unavailable
rank_wrapper	measurements/_wrap/rank-wrapper.sh is absent; perf-run.sh requires it for 2x2 — the collective ticket creates it (taskset rank 0 -> 0,2, rank 1 -> 4,6) before its first run
enumerated_bound	measured (30.434 s), so no 2t row is pre-tagged UNMEASURED timeout

## Guard demonstrations (scratch copies of the queues under /tmp; the real queues untouched)

A-layout-contradiction (core-io PD-008 collective stated 4t)	exit=1	written=no	PD-008: stated layout 4t contradicts claimType collective / requires [] (derived 2x2)
B-anchor-past-eof (stochastic PD-020 anchor :99999)	exit=0	written=yes	claim-table.json: 32 rows (31 schedulable; {'4t': 26, '2x2': 2, '2t': 3}), stations {'core-io': 8, 'stochastic': 7, 'solver-comm': 3, 'sddp': 10, 'cli-python': 
C-anchor-path-missing (solver-comm PD-032 anchor to a path absent at the pin)	exit=0	written=yes	claim-table.json: 32 rows (31 schedulable; {'4t': 26, '2x2': 2, '2t': 3}), stations {'core-io': 8, 'stochastic': 7, 'solver-comm': 3, 'sddp': 10, 'cli-python': 
D-timing-assertion (cli-python PD-049 claim quotes 12.5 s)	exit=1	written=no	PD-049: station asserted a timing figure (12.5 s); queues carry claims only
E-no-allowance (PD-047 without --accept-derived)	exit=1	written=no	PD-047: stated layout 4t contradicts claimType single-process / requires ['enumerated'] (derived 2t)
B-detail	PD-020 written with status anchor-missing, listed in bounced (reason line-past-eof), schedulable 31 of 32, no measurements/PD-020/ directory created; C behaves identically with reason path-missing
conclusion	a stated/derived layout contradiction and a timing assertion abort without writing claim-table.json; an unresolvable anchor is kept as evidence, excluded from the schedulable set and listed for its station

## PD-004 preflight (2026-09-20, register pin 077dbe2c)

anchors	run_enumerated_backward at crates/cobre-sddp/src/training/backward_pass_state.rs:721; stage_stats Vec<(usize, Vec<StageWorkerOpeningDelta>)> + delta.clone() at :919-931; sibling run_sampled_backward at :531 with its Vec at :632; StageWorkerOpeningDelta alias at crates/cobre-sddp/src/training/backward/mod.rs:92; SolverStatsDelta at crates/cobre-sddp/src/solver_stats.rs:15; Traversal::resolve at crates/cobre-sddp/src/setup/node_graph.rs:1384 (the ticket quotes :1386 at the scaffold pin); dispatch Traversal::Enumerated -> run_enumerated_backward at backward_pass_state.rs:508 — all resolved via git show at the register pin
binary	target/profiling/cobre not stripped, .debug_line present (build id de4953977d7744db4709abfd6ff9dee49381a399)
bound	Protocol bound (enumerated): 30.434 s == measurements/CAL-ENUM/median.txt; timeout 3x = 92 s; measured, not UNMEASURED timeout-3x
traversal	staged deck measurements/_case/deck-enumerated/config.json training.selection.method = enumerated -> Traversal::resolve(is_enumerated = true) -> the Traversal::Enumerated arm at backward_pass_state.rs:508; an unexercised-path verdict on this deck would be a finding about the dispatch, not a deck mismatch
claim_table_row	PD-004: single-process, requires enumerated, layout 2t, case mar-26-enumerated, profiledSymbol run_enumerated_backward, status queued
register_lines	PD-004 entry BACKLOG.md:1316 (the ticket quotes :1255/:1696), do-not-touch list :1942 (ticket :1881), PD-005 entry :1499, Wave-2 PD-005 topology-precompute bullet :2097
harness	perf-run.sh --perf records the FIRST timed run (perf record -F 99 -g --call-graph dwarf) and writes perf.txt via perf report --stdio --no-children; perf.data lives in the scratch dir the harness deletes on exit (no perf.data digest survives); runs are --quiet so the phase split comes from one separate un-timed run whose output is kept

## PD-004 measurement (2026-09-20)

layout	2t (--threads 2 --comm-backend local, taskset -c 0,2) on the enumerated deck
runs	warmup 32.514 s; timed 46.822 (perf-instrumented run-1) / 31.023 / 32.607; median 32.607 s, min 31.023 s
phase_split	training 30.0 s / simulation 1.9 s (forward-solve 30.5 s, backward-solve 2.7 s, 18 iterations)
attribution	LBR call graph: run_enumerated_backward inclusive 2.98% (all HiGHS LP-solve under run_backward_node_replicated, 0 self); alloc site 0.139%; the DWARF recording perf-run.sh captured did not unwind on this binary (4939/5710 single-frame), so measurements/PD-004/perf.txt is a self view + an appended LBR-inclusive supplement
verdict	not-material (alloc 0.139% < 1% sample-share; < 3% phase-wall on the claimed cost)
fix_shape	none promoted; the symmetric-fold premise preserved as recorded, not re-argued
pd005	residual re-confirmed by absence (nested_ub_recursion/NestedUbTopology/walk_leaf_to_root = 0 samples); closed, not re-opened
do_not_touch	BACKLOG.md do-not-touch line amended: PD-004 parenthetical now reads not-material with the profile path; CD-008/PD-001/reserved-seam census byte-identical
deviations	register pin 077dbe2c measured (not the ticket's scaffold pin a136840d); the block template quotes 4t but PD-004 is 2t on the enumerated deck; no perf.data digest or flamegraph SVG survives (scratch dir deleted; no renderer on PATH) — the perf-run.sh artifact set (cmd/runs/median/env/perf/run-N) is authoritative
