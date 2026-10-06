//! Pins the stop decision and the convergence monitor's per-iteration update to
//! zero heap allocation, for both stopping modes and a rule set holding every
//! configurable rule kind, and the monitor of a resumed run to the same bound.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use cobre_sddp::{
    ConvergenceMonitor, MonitorState, StoppingMode, StoppingRule, StoppingRuleSet, SyncResult,
};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method is a verbatim forward to `System`, which upholds
// `GlobalAlloc`'s contract on its own; this wrapper only adds a counter
// increment around the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` are forwarded unchanged to `System::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr`/`layout`/`new_size` are forwarded unchanged to
        // `System::realloc`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

fn reset_alloc_count() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

fn rule_set(mode: StoppingMode) -> StoppingRuleSet {
    StoppingRuleSet {
        rules: vec![
            StoppingRule::IterationLimit { limit: 64 },
            StoppingRule::TimeLimit { seconds: 3600.0 },
            StoppingRule::BoundStalling {
                tolerance: 1e-9,
                iterations: 5,
            },
            StoppingRule::Gap {
                tolerance: Some(1e-6),
                relative_tolerance: Some(1e-6),
            },
        ],
        mode,
    }
}

#[test]
fn stop_decision_and_convergence_update_allocate_nothing() {
    let any_set = rule_set(StoppingMode::Any);
    let all_set = rule_set(StoppingMode::All);
    let state = MonitorState {
        iteration: 10,
        wall_time_seconds: 1.0,
        lower_bound: 100.0,
        upper_bound: 110.0,
        lower_bound_history: (0..10).map(f64::from).collect(),
        shutdown_requested: false,
    };
    let sync = SyncResult {
        global_ub_mean: 110.0,
        global_ub_std: 5.0,
        ci_95_half_width: 2.0,
        sync_time_ms: 10,
    };
    let mut monitor = ConvergenceMonitor::with_iteration_budget(any_set.clone(), 64);

    reset_alloc_count();
    for i in 0..50 {
        black_box(any_set.evaluate(black_box(&state)));
        black_box(all_set.evaluate(black_box(&state)));
        black_box(monitor.update(100.0 + f64::from(i), &sync, 0.0));
    }

    assert_eq!(alloc_count(), 0, "the stop decision must not allocate");

    let recorded: Vec<f64> = (0..10).map(f64::from).collect();
    let mut resumed = ConvergenceMonitor::with_iteration_budget(any_set.clone(), 64);

    reset_alloc_count();
    resumed.resume_at(10, &recorded);
    for i in 0..40 {
        black_box(resumed.update(100.0 + f64::from(i), &sync, 0.0));
    }

    assert_eq!(alloc_count(), 0, "a resumed monitor must not allocate");
}
