//! Linux integration coverage for native CPU discovery and Rayon binding.

#![cfg(all(feature = "affinity", target_os = "linux"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeSet;

use cobre_comm::{AffinityPolicy, CpuTopology, WorkerAffinity, current_thread_cpu_set};

#[test]
fn discovered_cpus_match_the_process_affinity_mask() {
    let current = current_thread_cpu_set().expect("sched_getaffinity must succeed");
    let topology = CpuTopology::discover().expect("Linux CPU topology discovery must succeed");

    assert_eq!(topology.visible_cpus(), current);
    assert!(topology.visible_processing_units() > 0);
    assert!(topology.physical_cores() > 0);
    assert!(topology.online_processing_units() >= topology.visible_processing_units());
    assert!(!topology.memory_binding().allowed_nodes.is_empty());
    assert!(
        topology.memory_binding().policy.is_some()
            || topology.memory_binding().discovery_error.is_some()
    );
}

#[test]
fn rayon_workers_are_bound_inside_the_inherited_cpuset() {
    let inherited = current_thread_cpu_set().expect("sched_getaffinity must succeed");
    let worker_count = inherited.len().min(4);
    let affinity = WorkerAffinity::prepare(AffinityPolicy::Core, worker_count)
        .expect("core binding plan must resolve");
    let worker_affinity = affinity.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .start_handler(move |worker_index| worker_affinity.bind_worker(worker_index))
        .build()
        .expect("rayon pool must build");

    let mut observed = pool.broadcast(|context| {
        (
            context.index(),
            current_thread_cpu_set().expect("bound worker affinity must be readable"),
        )
    });
    affinity
        .verify()
        .expect("every worker binding must succeed");
    observed.sort_unstable_by_key(|(worker, _)| *worker);

    let inherited = inherited.into_iter().collect::<BTreeSet<_>>();
    assert_eq!(observed.len(), worker_count);
    for (worker, cpus) in observed {
        assert_eq!(cpus, vec![affinity.report().worker_cpus[worker]]);
        assert!(cpus.iter().all(|cpu| inherited.contains(cpu)));
    }
}
