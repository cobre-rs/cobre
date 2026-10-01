//! CLP determinism harness — declaration-order invariance and bit-for-bit
//! re-solve reproducibility for the CLP backend.
//!
//! Determinism is the property under test, so every comparison is **bit-for-bit**
//! via [`f64::to_bits`], never a tolerance — a tolerance would mask exactly the
//! drift this harness exists to catch.
//!
//! Two preconditions hold this together: the active solver disables perturbation
//! (without which CLP injects pseudo-random bound perturbations and no re-solve is
//! bit-reproducible), and it forwards row prices with no negation (so duals
//! compare directly).
//!
//! Fixtures are re-declared locally because an integration test cannot import the
//! `#[cfg(test)]` builders from `src`.

#![cfg(feature = "clp")]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::panic
    )
)]

use cobre_solver::{ClpProfile, ClpSolver, RowBatch, SolverInterface, StageTemplate};

// ─── Shared fixtures (re-declared locally; see module docs) ──────────────────

/// `SS1.1`: 3 columns, 2 equality rows. Optimal: objective `100`, primals
/// `(6, 0, 2)`, row duals `[-100, 50]`.
fn make_fixture_stage_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 3,
        col_starts: vec![0_i32, 2, 2, 3],
        row_indices: vec![0_i32, 1, 1],
        values: vec![1.0, 2.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![10.0, f64::INFINITY, 8.0],
        objective: vec![0.0, 1.0, 50.0],
        row_lower: vec![6.0, 14.0],
        row_upper: vec![6.0, 14.0],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// `SS1.2`: two appended `>=` rows over the `SS1.1` columns. After `add_rows`,
/// the merged LP solves to objective `162`, primals `(6, 62, 2)`.
fn make_fixture_row_batch() -> RowBatch {
    RowBatch {
        num_rows: 2,
        row_starts: vec![0_i32, 2, 4],
        col_indices: vec![0_i32, 1, 0, 1],
        values: vec![-5.0, 1.0, 3.0, 1.0],
        row_lower: vec![20.0, 80.0],
        row_upper: vec![f64::INFINITY, f64::INFINITY],
    }
}

// ─── Reusable mutate-then-solve vocabulary ───────────────────────────────────

/// A single step in a mutate-then-solve sequence fed to
/// [`assert_solve_sequence_deterministic`].
#[derive(Clone)]
pub enum MutationStep {
    /// Append a batch of constraint rows (CSR form) via `add_rows`.
    AddRows(RowBatch),
    /// Patch the bounds of the given rows via `set_row_bounds`.
    SetRowBounds {
        /// Row indices to patch.
        indices: Vec<usize>,
        /// New lower bounds, aligned with `indices`.
        lower: Vec<f64>,
        /// New upper bounds, aligned with `indices`.
        upper: Vec<f64>,
    },
    /// Patch the bounds of the given columns via `set_col_bounds`.
    SetColBounds {
        /// Column indices to patch.
        indices: Vec<usize>,
        /// New lower bounds, aligned with `indices`.
        lower: Vec<f64>,
        /// New upper bounds, aligned with `indices`.
        upper: Vec<f64>,
    },
    /// Solve the current model (cold, no offered basis) and capture the result.
    Solve,
}

/// The bit-pattern fingerprint of one solve: objective + every solution buffer,
/// stored as raw `u64` bit patterns so equality is exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveBits {
    /// Raw bits of the objective value.
    pub objective: u64,
    /// Raw bits of the primal column-value vector.
    pub primal: Vec<u64>,
    /// Raw bits of the reduced-cost vector.
    pub reduced_costs: Vec<u64>,
    /// Raw bits of the row-dual vector.
    pub dual: Vec<u64>,
}

impl SolveBits {
    fn capture(objective: f64, primal: &[f64], reduced_costs: &[f64], dual: &[f64]) -> Self {
        Self {
            objective: objective.to_bits(),
            primal: primal.iter().map(|v| v.to_bits()).collect(),
            reduced_costs: reduced_costs.iter().map(|v| v.to_bits()).collect(),
            dual: dual.iter().map(|v| v.to_bits()).collect(),
        }
    }
}

fn run_sequence(
    mut solver: ClpSolver,
    base: &StageTemplate,
    steps: &[MutationStep],
) -> Vec<SolveBits> {
    run_sequence_in_place(&mut solver, base, steps)
}

fn assert_bits_eq(label: &str, a: &[SolveBits], b: &[SolveBits]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: solve count differs ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (lhs, rhs)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            lhs.objective, rhs.objective,
            "{label}: objective bits diverged at solve #{i}"
        );
        assert_eq!(
            lhs.primal, rhs.primal,
            "{label}: primal bits diverged at solve #{i}"
        );
        assert_eq!(
            lhs.reduced_costs, rhs.reduced_costs,
            "{label}: reduced-cost bits diverged at solve #{i}"
        );
        assert_eq!(
            lhs.dual, rhs.dual,
            "{label}: row-dual bits diverged at solve #{i}"
        );
    }
}

/// Asserts that running `steps` over a CLP solver built by `build` is
/// bit-for-bit reproducible: the same instance run twice and a fresh instance
/// run once all yield identical objective / primal / reduced-cost / dual
/// buffers at every `Solve` step. Returns the first run's fingerprints.
///
/// The property under test is self-consistency — a path reproducing itself and
/// staying order-invariant — never a cross-check that two different solve methods
/// agree.
///
/// # Panics
///
/// Panics (failing the test) if any run diverges bit-for-bit from the first, if
/// any `Solve` step fails to reach optimality, or if `steps` contains no `Solve`
/// step.
#[must_use]
pub fn assert_solve_sequence_deterministic(
    build: impl Fn() -> ClpSolver,
    base: &StageTemplate,
    steps: &[MutationStep],
) -> Vec<SolveBits> {
    let run1 = run_sequence(build(), base, steps);
    assert!(
        !run1.is_empty(),
        "step sequence must contain at least one Solve step"
    );

    let run2 = run_sequence(build(), base, steps);
    assert_bits_eq("fresh-instance re-run", &run1, &run2);

    // CLP retains internal factorization/basis state across solves, so reusing
    // one instance exercises the "warm internal state must not perturb results"
    // path that a two-instance comparison cannot reach.
    let mut solver = build();
    let pass_a = run_sequence_in_place(&mut solver, base, steps);
    let pass_b = run_sequence_in_place(&mut solver, base, steps);
    assert_bits_eq("same-instance pass A vs run1", &pass_a, &run1);
    assert_bits_eq("same-instance pass B vs pass A", &pass_b, &pass_a);

    run1
}

fn run_sequence_in_place(
    solver: &mut ClpSolver,
    base: &StageTemplate,
    steps: &[MutationStep],
) -> Vec<SolveBits> {
    solver.load_model(base);
    let mut captured = Vec::new();
    for step in steps {
        match step {
            MutationStep::AddRows(batch) => solver.add_rows(batch),
            MutationStep::SetRowBounds {
                indices,
                lower,
                upper,
            } => solver.set_row_bounds(indices, lower, upper),
            MutationStep::SetColBounds {
                indices,
                lower,
                upper,
            } => solver.set_col_bounds(indices, lower, upper),
            MutationStep::Solve => {
                let view = solver.solve(None).expect("solve must reach optimality");
                captured.push(SolveBits::capture(
                    view.objective,
                    view.primal,
                    view.reduced_costs,
                    view.dual,
                ));
            }
        }
    }
    captured
}

// ─── Row permutation (declaration-order invariance) ──────────────────────────

/// Reorders the rows of a `StageTemplate`'s CSC matrix and row bounds according
/// to `perm`, where `perm[new_row] == old_row`. The column structure is
/// unchanged; only each non-zero's row index is remapped.
///
/// `perm` must be a permutation of `0..template.num_rows`.
///
/// # Panics
///
/// Panics if `perm.len() != template.num_rows`, or if a stored row index does
/// not fit back into `i32` after remapping.
#[must_use]
pub fn permute_rows(template: &StageTemplate, perm: &[usize]) -> StageTemplate {
    assert_eq!(
        perm.len(),
        template.num_rows,
        "perm length {} must equal num_rows {}",
        perm.len(),
        template.num_rows
    );

    let inverse = inverse_permutation(perm);

    // col_starts is untouched; only each stored row index moves old → new.
    let mut new_row_indices = template.row_indices.clone();
    for ri in &mut new_row_indices {
        let old = usize::try_from(*ri).expect("row index must be non-negative");
        *ri = i32::try_from(inverse[old]).expect("row index fits in i32");
    }

    let mut new_row_lower = vec![0.0_f64; template.num_rows];
    let mut new_row_upper = vec![0.0_f64; template.num_rows];
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        new_row_lower[new_pos] = template.row_lower[old_pos];
        new_row_upper[new_pos] = template.row_upper[old_pos];
    }

    StageTemplate {
        row_indices: new_row_indices,
        row_lower: new_row_lower,
        row_upper: new_row_upper,
        col_starts: template.col_starts.clone(),
        values: template.values.clone(),
        col_lower: template.col_lower.clone(),
        col_upper: template.col_upper.clone(),
        objective: template.objective.clone(),
        col_scale: template.col_scale.clone(),
        row_scale: template.row_scale.clone(),
        ..*template
    }
}

/// Computes the inverse of a permutation: `inverse[perm[i]] == i`.
#[must_use]
pub fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0_usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    inverse
}

// ─── Declaration-order invariance ────────────────────────────────────────────

/// Declaration-order invariance: solving an LP and a row-permuted copy yields
/// bit-for-bit identical objective, primals, and reduced costs, and the
/// permuted solve's row duals, mapped back through the inverse permutation,
/// match the original solve's row duals bit-for-bit.
///
/// Run on the `SS1.2` fixture (`SS1.1` + the two appended `>=` rows) so the
/// permutation spans four rows of two distinct kinds (equality + inequality).
#[test]
fn declaration_order_invariance_ss1_2() {
    let base = merged_ss1_2_template();

    let mut solver = ClpSolver::new().expect("ClpSolver::new must succeed");
    solver.load_model(&base);
    let ref_view = solver.solve(None).expect("reference solve must succeed");
    let ref_objective = ref_view.objective.to_bits();
    let ref_primal: Vec<u64> = ref_view.primal.iter().map(|v| v.to_bits()).collect();
    let ref_reduced: Vec<u64> = ref_view.reduced_costs.iter().map(|v| v.to_bits()).collect();
    let ref_dual: Vec<u64> = ref_view.dual.iter().map(|v| v.to_bits()).collect();
    drop(solver);

    // perm[new_row] = old_row.
    let perm = [3_usize, 2, 1, 0];
    let inverse = inverse_permutation(&perm);
    let permuted = permute_rows(&base, &perm);

    let mut psolver = ClpSolver::new().expect("ClpSolver::new must succeed");
    psolver.load_model(&permuted);
    let pview = psolver.solve(None).expect("permuted solve must succeed");

    // Objective / primals / reduced costs are row-order-independent → identical bits.
    assert_eq!(
        pview.objective.to_bits(),
        ref_objective,
        "objective bits changed under row permutation"
    );
    let p_primal: Vec<u64> = pview.primal.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        p_primal, ref_primal,
        "primal bits changed under row permutation"
    );
    let p_reduced: Vec<u64> = pview.reduced_costs.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        p_reduced, ref_reduced,
        "reduced-cost bits changed under row permutation"
    );

    // Row duals must match after inverse-permuting the permuted solve's duals
    // back into the original row order: permuted_dual[new] corresponds to
    // original row `perm[new]`, so original row `r` reads from new position
    // `inverse[r]`.
    let p_dual: Vec<u64> = pview.dual.iter().map(|v| v.to_bits()).collect();
    let unpermuted_dual: Vec<u64> = (0..base.num_rows).map(|r| p_dual[inverse[r]]).collect();
    assert_eq!(
        unpermuted_dual, ref_dual,
        "row-dual bits diverged after inverse permutation (declaration-order leak)"
    );
}

/// Cross-instance re-solve reproducibility: the same mutate-then-solve sequence
/// run on a fresh instance, a second fresh instance, and a single reused
/// instance (twice) all yield bit-for-bit identical objective / primal /
/// reduced-cost / dual buffers at every solve.
///
/// The sequence loads `SS1.1`, solves, appends `SS1.2`, solves again, then
/// patches a column bound and solves a third time — exercising every mutation
/// variant and three solves through the reusable helper.
#[test]
fn cross_instance_resolve_reproducibility() {
    let base = make_fixture_stage_template();
    let steps = vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
    ];

    let captured = assert_solve_sequence_deterministic(
        || ClpSolver::new().expect("ClpSolver::new must succeed"),
        &base,
        &steps,
    );

    // Anchor to the known optima so a refactor that silently changes the solved
    // model is caught here, not only by the reproducibility helper.
    assert_eq!(captured.len(), 3, "expected three Solve captures");
    assert_eq!(
        f64::from_bits(captured[0].objective),
        100.0,
        "first solve objective must be the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(captured[1].objective),
        162.0,
        "second solve objective must be the SS1.2 optimum"
    );
}

/// Native incremental-mutation determinism: a mutate-then-solve sequence that
/// drives every native-mutation entry point (`add_rows`, `set_row_bounds`,
/// `set_col_bounds`) interleaved with solves must be bit-for-bit reproducible
/// (run-to-run, cross-instance, and same-instance-reused) AND declaration-order
/// invariant.
///
/// Reproducibility is checked through the reusable
/// [`assert_solve_sequence_deterministic`] helper (three runs: two fresh
/// instances plus one reused instance driven twice). Declaration-order
/// invariance is checked by replaying the identical mutation sequence on the
/// row-permuted base `SS1.1` and inverse-permuting the resulting row duals back
/// into declaration order, then comparing bit-for-bit against the reference run.
///
/// The native-mutation path appends rows via `Clp_addRows` and patches bounds
/// via `Clp_chgRowLower/Upper` / `Clp_chgColumnLower/Upper` in place — preserving
/// CLP's factorization across each solve — so this is the determinism gate for
/// that path.
#[test]
fn native_mutation_sequence_is_deterministic_and_order_invariant() {
    let base = make_fixture_stage_template();
    let steps = vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
        MutationStep::SetRowBounds {
            indices: vec![0],
            lower: vec![6.0],
            upper: vec![6.0],
        },
        MutationStep::Solve,
    ];

    let reference = assert_solve_sequence_deterministic(
        || ClpSolver::new().expect("ClpSolver::new must succeed"),
        &base,
        &steps,
    );
    assert_eq!(reference.len(), 4, "expected four Solve captures");
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "cold SS1.1 solve must hit the known optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "post-add_rows solve must hit the SS1.2 optimum"
    );

    // Replay on a row-permuted base. `add_rows` appends after the existing rows
    // regardless of base order, so the SetRowBounds indices must be permuted to
    // keep patching the same logical rows.
    let perm = [1_usize, 0];
    let inverse = inverse_permutation(&perm);
    let permuted_base = permute_rows(&base, &perm);
    let permuted_steps = vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
        MutationStep::SetRowBounds {
            indices: vec![inverse[0]],
            lower: vec![6.0],
            upper: vec![6.0],
        },
        MutationStep::Solve,
    ];
    let permuted = assert_solve_sequence_deterministic(
        || ClpSolver::new().expect("ClpSolver::new must succeed"),
        &permuted_base,
        &permuted_steps,
    );

    // Old→new map over the full 4-row model: base rows follow `inverse`,
    // appended rows keep positions 2 and 3.
    let full_inverse = [inverse[0], inverse[1], 2_usize, 3];
    for (i, (r, p)) in reference.iter().zip(&permuted).enumerate() {
        assert_eq!(
            r.objective, p.objective,
            "objective bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.primal, p.primal,
            "primal bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.reduced_costs, p.reduced_costs,
            "reduced-cost bits diverged under row permutation at solve #{i}"
        );
        // Inverse-permute the row duals back into declaration order before
        // comparing (solve #0 sees only the 2 base rows; later solves see all 4).
        let unpermuted: Vec<u64> = (0..r.dual.len())
            .map(|row| p.dual[full_inverse[row]])
            .collect();
        assert_eq!(
            r.dual, unpermuted,
            "row-dual bits diverged after inverse permutation at solve #{i} \
             (declaration-order leak in the native-mutation path)"
        );
    }
}

/// Builds the SS1.2 LP (SS1.1 + the two appended `>=` rows) as one
/// `StageTemplate`, so the full 4-row set can be permuted atomically.
fn merged_ss1_2_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 4,
        num_nz: 7,
        col_starts: vec![0_i32, 4, 6, 7],
        row_indices: vec![0_i32, 1, 2, 3, 2, 3, 1],
        values: vec![1.0, 2.0, -5.0, 3.0, 1.0, 1.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![10.0, f64::INFINITY, 8.0],
        objective: vec![0.0, 1.0, 50.0],
        row_lower: vec![6.0, 14.0, 20.0, 80.0],
        row_upper: vec![6.0, 14.0, f64::INFINITY, f64::INFINITY],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

// ─── Wired-knob determinism (DSE pricing, refactor cadence) ──────────────────

/// The mutate-then-solve sequence shared by the wired-knob determinism tests:
/// solve `SS1.1` cold, append the two `SS1.2` rows, solve, tighten a column
/// bound, solve — exercising the knob across a re-solve sequence, not one solve.
fn knob_step_sequence() -> Vec<MutationStep> {
    vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
    ]
}

/// **Wired knob — dual-steepest-edge pricing.** A profile pinning full DSE
/// (`dual_pricing_mode = 1`, the backward-pass tuned value) driven through
/// `apply_profile` must yield a bit-for-bit reproducible, declaration-order
/// invariant solve sequence — never a hot-vs-cold check.
#[test]
fn dse_pricing_profile_is_deterministic_and_order_invariant() {
    let dse_profile = ClpProfile {
        dual_pricing_mode: 1,
        ..ClpProfile::default()
    };
    let build = move || {
        let mut s = ClpSolver::new().expect("ClpSolver::new must succeed");
        s.apply_profile(&dse_profile);
        s
    };

    let base = make_fixture_stage_template();
    let steps = knob_step_sequence();
    let reference = assert_solve_sequence_deterministic(build, &base, &steps);
    assert_eq!(reference.len(), 3, "expected three Solve captures");
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "DSE cold solve must hit the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "DSE post-add_rows solve must hit the SS1.2 optimum"
    );

    assert_knob_sequence_order_invariant(&build, &reference, &steps);
}

/// **Wired knob — refactorization cadence.** A profile setting
/// `factorization_frequency = 200` (the backward-pass tuned value) is driven
/// through `apply_profile`, which now issues the shim factorization-frequency
/// setter. The solve sequence must be bit-for-bit reproducible and
/// declaration-order invariant.
#[test]
fn factorization_frequency_profile_is_deterministic_and_order_invariant() {
    let factfreq_profile = ClpProfile {
        factorization_frequency: 200,
        ..ClpProfile::default()
    };
    let build = move || {
        let mut s = ClpSolver::new().expect("ClpSolver::new must succeed");
        s.apply_profile(&factfreq_profile);
        s
    };

    let base = make_fixture_stage_template();
    let steps = knob_step_sequence();
    let reference = assert_solve_sequence_deterministic(build, &base, &steps);
    assert_eq!(reference.len(), 3, "expected three Solve captures");
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "refactor-cadence cold solve must hit the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "refactor-cadence post-add_rows solve must hit the SS1.2 optimum"
    );

    assert_knob_sequence_order_invariant(&build, &reference, &steps);
}

/// **Wired knob — the backward-pass tuned profile (DSE + cadence together).**
/// The `BACKWARD` CLP profile pins both `dual_pricing_mode = 1` and
/// `factorization_frequency = 200`; this drives both shim setters in one
/// `apply_profile` and proves the combination is deterministic and
/// order-invariant — the configuration the backward pass actually applies.
#[test]
fn backward_tuned_profile_is_deterministic_and_order_invariant() {
    let backward_profile = ClpProfile {
        dual_pricing_mode: 1,
        factorization_frequency: 200,
        ..ClpProfile::default()
    };
    let build = move || {
        let mut s = ClpSolver::new().expect("ClpSolver::new must succeed");
        s.apply_profile(&backward_profile);
        s
    };

    let base = make_fixture_stage_template();
    let steps = knob_step_sequence();
    let reference = assert_solve_sequence_deterministic(build, &base, &steps);
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "backward-tuned cold solve must hit the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "backward-tuned post-add_rows solve must hit the SS1.2 optimum"
    );

    assert_knob_sequence_order_invariant(&build, &reference, &steps);
}

/// Shared declaration-order-invariance check for the knob sequences: replays the
/// sequence on a row-permuted `SS1.1` base and asserts objective / primals /
/// reduced costs match bit-for-bit, and row duals match after inverse-permuting
/// back into declaration order. The two appended `SS1.2` rows keep positions 2
/// and 3 (`add_rows` appends after the existing rows regardless of base order);
/// the two base rows follow the permutation.
fn assert_knob_sequence_order_invariant(
    build: &impl Fn() -> ClpSolver,
    reference: &[SolveBits],
    steps: &[MutationStep],
) {
    let base = make_fixture_stage_template();
    let perm = [1_usize, 0];
    let inverse = inverse_permutation(&perm);
    let permuted_base = permute_rows(&base, &perm);
    let permuted = assert_solve_sequence_deterministic(build, &permuted_base, steps);

    let full_inverse = [inverse[0], inverse[1], 2_usize, 3];
    for (i, (r, p)) in reference.iter().zip(&permuted).enumerate() {
        assert_eq!(
            r.objective, p.objective,
            "objective bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.primal, p.primal,
            "primal bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.reduced_costs, p.reduced_costs,
            "reduced-cost bits diverged under row permutation at solve #{i}"
        );
        let unpermuted: Vec<u64> = (0..r.dual.len())
            .map(|row| p.dual[full_inverse[row]])
            .collect();
        assert_eq!(
            r.dual, unpermuted,
            "row-dual bits diverged after inverse permutation at solve #{i} \
             (declaration-order leak in the wired-knob path)"
        );
    }
}
