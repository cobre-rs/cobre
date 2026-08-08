# NUMA-aware execution plan

> **Status:** CPU-topology observability and opt-in worker affinity are
> implemented. Target-host performance validation remains open, and NUMA memory
> placement remains gated on those measurements. CPU topology and binding are
> portable concerns; AMD EPYC is the first target platform, not a
> vendor-specific implementation target.

## Objective

Reduce wall-clock time on multi-core NUMA hosts by keeping worker threads and
their memory traffic local to the CPU resources assigned to the Cobre process.
The first target is a single-host production study on a 48-core AMD EPYC server.
The design must remain useful on other NUMA machines, including Intel Xeon and
ARM servers.

The optimization must not change:

- the SDDP algorithm, solve order, backward scheduler, or solver profile;
- the number or order of LP solves;
- cuts, bounds, policies, and simulation trajectories;
- run-to-run and thread/rank-shape determinism.

Timing records, host topology, and execution-distribution metadata necessarily
differ when the execution shape changes. They are operational telemetry, not
part of the numerical parity set. The parity gate compares all deterministic
model-result payloads and explicitly excludes only those operational fields.

## Current state

`RunArgs::threads` controls a process-wide Rayon pool in
`setup_communicator`. When omitted, the CLI resolves the worker count to one.
The Python binding creates a scoped Rayon pool with the same one-thread default.
Both accept an opt-in `none|core|numa` CPU-binding policy and use the same
`cobre-comm::WorkerAffinity` planner.

The optional `cobre-comm/affinity` feature discovers the Linux CPU set,
processing units, physical cores, packages, NUMA nodes, and the current memory
policy/nodemask without a native library dependency. A blocked or unavailable
memory-policy query is retained as non-fatal telemetry. `core` fills physical
cores before SMT siblings; `numa` spreads physical-core assignments across
visible nodes before SMT. Explicit binding fails on unsupported builds instead
of silently continuing unbound. The existing `numa` feature now implies
`affinity` alongside MPI topology.

Every run persists per-rank topology and worker mappings in
`distribution.rank_affinity`. The CLI gathers variable-length rank reports
collectively; Python writes the equivalent rank-zero local report. Timing and
topology remain operational metadata outside the numerical parity set.

MPI launchers and batch schedulers already know the allocation boundary. Cobre
must not silently override their rank placement. In an MPI run it should inspect
and validate the resources visible to each rank, then bind only that rank's
Rayon workers within those resources.

## Decision sequence

Implementation proceeds only through the following gates. A failed gate closes
the work at that phase rather than promoting an unmeasured optimization.

### Phase 0 — establish the external-binding baseline

**Status:** the benchmark harness records the required phase timing, topology,
placement, memory, solver-work, and numerical-parity evidence; execution on the
target EPYC host is still required.

No Cobre code changes are required. On the target host:

1. Record the physical topology, NUMA domains, SMT state, current CPU affinity,
   and memory policy using the host's standard topology tools.
2. Benchmark one process with several worker counts spanning the physical-core
   count and, when SMT is enabled, selected logical-thread counts.
3. On multi-domain or multi-socket allocations, compare the local backend with
   MPI layouts such as one rank per NUMA domain. Let the MPI launcher or
   scheduler bind ranks and CPUs.
4. Compare default, local, and interleaved memory policies using external tools.
5. Run each arm repeatedly on matched epochs of the same case and binary. Avoid
   simultaneous unrelated workloads.

Primary metric: training wall time, split into forward, backward, cut-selection,
and simulation time. Corroborating metrics: LP solves, simplex iterations,
effective CPU utilization, memory bandwidth, and local versus remote NUMA
traffic. Solver work counts must remain identical between arms.

**Go criterion:** an externally bound configuration improves median training
wall time by at least 5% and the confidence interval or run spread does not
overlap the unbound baseline materially. All numerical parity checks must pass.

**No-go criterion:** close the implementation if the gain is below the threshold
or is explained entirely by an incorrect worker count, oversubscription, or
unrelated host contention. Document the recommended launch command instead.

### Phase 1 — topology and observability

**Status:** implemented, including memory-policy observability, synthetic
sparse/asymmetric topology tests, and a Linux inherited-cpuset integration test.

Extend `cobre-comm` with a platform-neutral description of:

- packages/sockets;
- NUMA nodes and their memory capacity;
- physical cores and processing units;
- the CPU set available to the process or MPI rank;
- current CPU and memory binding, when the platform exposes them.

Keep topology discovery behind an optional feature and return an explicit
unsupported result on platforms without the capability. The existing behavior
must remain available without extra native dependencies.

At startup, report the effective worker count, visible physical/logical CPUs,
NUMA-node count, and binding policy. Warn when workers exceed the visible CPU set
or when all workers are confined unintentionally to a strict subset. Do not make
an automatic placement change in this phase.

Tests use synthetic topologies to cover asymmetric NUMA domains, cpusets,
non-contiguous CPU identifiers, SMT, and unavailable topology. A Linux
integration test verifies that reported affinity is a subset of the process
cpuset.

### Phase 2 — CPU affinity

**Status:** implemented for CLI and Python. Linux binding, metadata parity, and
bound-versus-unbound numerical parity gates are included; target-host performance
validation remains open.

Add an opt-in affinity policy shared by the CLI and Python binding:

```text
none       Preserve the operating system or launcher behavior.
core       Assign one worker per physical core before using SMT siblings.
numa       Spread workers across visible NUMA domains, then across physical cores.
```

The precise public flag and Python parameter names are chosen with the CLI/API
change, but the policy must default to `none` in its first release. The resolved
placement is included in execution metadata.

Build the Rayon pools with a worker-start hook that maps the stable Rayon worker
index to a CPU set and applies affinity before the worker handles study work.
Both the CLI global pool and Python scoped pool must call the same
`cobre-comm` placement implementation. Binding failure is a startup error when
the user explicitly requested binding; unsupported automatic discovery must not
silently claim success.

For MPI, intersect the discovered topology with the CPU set already assigned to
the rank. Never bind a worker outside that set. Rank-to-socket placement remains
the launcher's responsibility.

**Acceptance criteria:**

- every worker reports a CPU set contained in the process/rank allocation;
- physical cores are used before SMT siblings under `core`;
- synthetic topology tests pin the mapping independently of host enumeration
  order;
- existing scheduler determinism tests pass across worker and rank shapes;
- production numerical payloads match the unbound baseline without a parity
  re-baseline;
- the target-host gain remains at or above the Phase 0 threshold.

### Phase 3 — NUMA memory placement, separately gated

**Status:** not implemented. The Phase 2 target-host measurement must show
material remote-memory traffic before this phase is authorized.

CPU affinity alone can expose remote-memory traffic when large structures were
allocated by the main thread before workers start. Do not add memory policy on
speculation. Measure again after Phase 2.

Proceed only if remote traffic remains material and correlates with lost wall
time. Evaluate two policies independently:

- **interleave:** distribute read-mostly shared structures across visible NUMA
  domains to balance aggregate bandwidth;
- **local first-touch:** initialize worker-owned solver and workspace state on
  the NUMA node where its worker is bound.

Prefer first-touch initialization of already worker-owned state over relocating
shared model data. Do not duplicate the cut pool, stochastic model, or other
shared algorithm state merely for locality without a separate memory and
coherency design.

Memory placement remains opt-in until measurements cover at least one
single-socket multi-domain host and one multi-socket or scheduler-partitioned
host. Allocation failure or an unsupported policy must be reported; it must not
fall through to a misleading "NUMA enabled" status.

**Go criterion:** retain a memory policy only if it adds at least another 5%
median wall-time improvement over CPU affinity alone, stays within the agreed
memory overhead, and passes the numerical parity suite.

## Benchmark protocol

Use the same release binary, solver backend, case, seed, algorithm configuration,
and stopping point for every arm. Benchmark a short calibration run and a long
run that includes the larger late-iteration cut pool; early iterations alone do
not represent steady-state solver pressure.

The result record for each arm contains:

- CPU model, firmware-visible topology, SMT and NUMA configuration;
- operating system, kernel, MPI runtime, and scheduler allocation;
- rank count, workers per rank, CPU map, and memory policy;
- per-phase wall time and total wall time;
- LP solves and simplex iterations by phase;
- peak resident memory and available NUMA traffic counters;
- hashes of the deterministic numerical payload set.

Compare matched epochs rather than a single whole-run total. Report the median,
dispersion, and worst regression; do not promote a policy from one winning run.

## Determinism gates

Affinity may change which worker claims a work unit first, but it must not change
the canonical result path. The opening-block scheduler already requires results
to be independent of worker count and claim order. This implementation must not
alter solve order, warm-start ancestry, result scattering, or canonical
aggregation.

The test plan adds:

1. same worker count, unbound versus bound, repeated run;
2. one worker versus multiple bound workers;
3. local backend versus multiple MPI rank shapes;
4. declaration-order permutation under a bound execution;
5. CLI versus Python parity for the same local execution policy.

Compare the committed deterministic payload hashes and the relevant
thread/rank-shape gates. A changed numerical hash is a regression to investigate,
not a baseline to refresh. Timing and topology fields are asserted structurally
rather than byte-compared.

## Delivery slices and effort

| Slice | Deliverable | Estimated engineering effort |
| --- | --- | ---: |
| Baseline | Reproducible external-binding harness and report | 1–2 days |
| Observability | Portable topology model, diagnostics, tests | 3–5 days |
| CPU affinity | Shared CLI/Python binding policy and determinism gates | 4–7 days |
| EPYC validation | Repeated production benchmarks and tuning | 3–5 days |
| Memory placement | First-touch/interleave prototype and evaluation | 2–4 weeks |

The first production-worthy CPU-affinity increment is therefore approximately
two to three engineering weeks, including access to the target server. Deep
NUMA-aware allocation is a separate four-to-eight-week total program and is not
authorized by the CPU-affinity result alone.

## Risks and stop conditions

- **No measurable locality problem:** stop after Phase 0 and publish launcher
  guidance.
- **SMT or worker-count effect mistaken for NUMA:** separate those benchmark
  axes before attributing the gain.
- **Launcher conflict:** never expand a rank's inherited CPU set; explicit Cobre
  binding must be an intersection with it.
- **Platform dependency cost:** keep the feature optional and preserve a
  dependency-free fallback.
- **Python/CLI drift:** the pool-building paths differ, so both must consume one
  placement API and share mapping tests.
- **Misleading bitwise claim:** exclude only operational telemetry from parity;
  no numerical result may require re-baselining.
- **Memory replication:** reject a placement design whose memory growth erases
  the throughput gain or limits rank count.

## Completion definition

The CPU-affinity work is complete when an opt-in, portable policy is available
from both CLI and Python, respects scheduler/MPI allocations, diagnoses its
effective placement, preserves numerical payload hashes across thread/rank
shapes, and demonstrates the pre-registered wall-time gain on the target host.

The NUMA memory-placement work is complete only if its separate gate passes. If
it does not, the supported outcome is CPU affinity plus documented external
memory-policy guidance, not an unmeasured automatic allocator.
