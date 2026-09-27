//! Per-(stage, canonical-ω) backward solve-order keys.
//!
//! `noise_key[stage][ω]` is a pure function of setup-constant data (the synced
//! tree's `raw_noise` and the fixed per-(hydro, stage) `std_m3s`), precomputed
//! once at setup and never reaching the hot path. The backward solve order
//! sorts each stage's openings by this key (see [`crate::setup`] /
//! `OpeningTree::set_solve_order`); because the key is setup-constant, every
//! rank computes the identical permutation and cuts stay bit-identical across
//! thread/rank counts.
//!
//! ## σ layout alignment
//!
//! The opening noise vector's first `n_hydros` components are the per-hydro
//! inflow residuals η, in canonical `System::hydros()` order, weighted by the
//! seasonal `InflowModel::std_m3s` for that `(hydro, stage)` pair.
//! `build_sigma_table` keys the inflow models on `(hydro_id, stage_id)` exactly
//! as the PAR precompute does, indexed `[stage * n_hydros + h]`; a `(hydro,
//! stage)` pair with no inflow model contributes σ = 0 (the PAR's
//! deterministic-zero-inflow fallback).
//!
//! ## Opening order
//!
//! Each stage with 3 or more openings is ordered along its shortest chain: a
//! nearest-neighbor + 2-opt minimum-distance path over unweighted L2 distance
//! between openings' hydro-prefix noise vectors, shortening the warm-start
//! chain the backward pass walks. A stage with fewer than 3 openings keeps its
//! σ-key from above (there is no path to improve), so the σ computation is a
//! live fallback, not dead machinery.

use std::ops::Range;

use cobre_core::{EntityId, System};
use cobre_stochastic::{OpeningTreeView, StochasticContext};

use crate::error::SddpError;

/// Build the per-(stage, canonical-ω) backward solve-order key table from
/// setup-constant data.
///
/// Always computes the σ-weighted key first (also performs the noise-dimension
/// validation every path relies on), then overwrites each stage's keys with
/// its winning shortest-chain path's reversed positions (see the module docs).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when the per-(hydro,stage) σ slice is longer
/// than an opening's noise vector (the σ-layout vs noise-dim mismatch), naming
/// both lengths and never silently truncating or zero-padding.
pub(crate) fn build_noise_key_table(
    system: &System,
    stochastic: &StochasticContext,
) -> Result<Vec<Vec<f64>>, SddpError> {
    let n_hydros = stochastic.n_hydros();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let study_stage_ids: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let n_stages = study_stage_ids.len();

    let sigma = build_sigma_table(system, &hydro_ids, &study_stage_ids, n_hydros);

    let tree = stochastic.tree_view();
    let mut keys = Vec::with_capacity(n_stages);
    for stage in 0..n_stages {
        let n_openings = tree.n_openings(stage);
        let sigma_stage = &sigma[stage * n_hydros..stage * n_hydros + n_hydros];
        let mut stage_keys = Vec::with_capacity(n_openings);
        for omega in 0..n_openings {
            let raw_noise = tree.opening(stage, omega);
            stage_keys.push(noise_key(sigma_stage, raw_noise)?);
        }
        keys.push(stage_keys);
    }

    apply_chain_order(
        &mut keys,
        &tree,
        stochastic.class_dimensions().hydro_range(),
    );

    Ok(keys)
}

/// Compute the σ-weighted aggregate noise key `Σ_h σ_h · raw_noise[h]`.
///
/// Only the first `sigma.len()` components of `raw_noise` (the inflow η) are
/// weighted; trailing load/NCS noise components are ignored.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when `raw_noise` is shorter than `sigma`
/// (the σ-layout vs noise-dim mismatch), naming both lengths. The key is never
/// computed over a truncated or zero-padded slice.
pub(crate) fn noise_key(sigma: &[f64], raw_noise: &[f64]) -> Result<f64, SddpError> {
    if raw_noise.len() < sigma.len() {
        return Err(SddpError::Validation(format!(
            "noise_key σ-layout mismatch: σ length {} exceeds opening noise dimension {}; \
             refusing to truncate or zero-pad",
            sigma.len(),
            raw_noise.len(),
        )));
    }
    Ok(sigma.iter().zip(raw_noise.iter()).map(|(s, n)| s * n).sum())
}

/// Build the per-(stage, hydro) seasonal `std_m3s` table indexed
/// `[stage * n_hydros + h]`; a `(hydro, stage)` pair with no inflow model
/// contributes σ = 0.
fn build_sigma_table(
    system: &System,
    hydro_ids: &[EntityId],
    study_stage_ids: &[i32],
    n_hydros: usize,
) -> Vec<f64> {
    use std::collections::HashMap;

    let model_std: HashMap<(i32, i32), f64> = system
        .inflow_models()
        .iter()
        .map(|m| ((m.hydro_id.0, m.stage_id), m.std_m3s))
        .collect();

    let mut sigma = vec![0.0_f64; study_stage_ids.len() * n_hydros];
    for (s_idx, &stage_id) in study_stage_ids.iter().enumerate() {
        for (h_idx, hydro_id) in hydro_ids.iter().enumerate() {
            if let Some(&std) = model_std.get(&(hydro_id.0, stage_id)) {
                sigma[s_idx * n_hydros + h_idx] = std;
            }
        }
    }
    sigma
}

/// Total pairwise-distance cost of visiting `tour` in order.
fn tour_cost(tour: &[usize], distances: &[f64], n_o: usize) -> f64 {
    tour.windows(2).map(|w| distances[w[0] * n_o + w[1]]).sum()
}

/// Nearest-neighbor tour from `start` over the row-major `n_o x n_o` distance
/// matrix, breaking ties by keeping the lowest already-found candidate index
/// (`total_cmp`, never `partial_cmp`/`<`, to stay a deterministic function of
/// `distances` alone).
fn nearest_neighbor_tour(
    distances: &[f64],
    n_o: usize,
    start: usize,
    tour: &mut Vec<usize>,
    visited: &mut [bool],
) {
    tour.clear();
    visited.fill(false);
    tour.push(start);
    visited[start] = true;

    for _ in 1..n_o {
        let last = tour[tour.len() - 1];
        let mut nearest: Option<usize> = None;
        for candidate in 0..n_o {
            if visited[candidate] {
                continue;
            }
            let better = match nearest {
                None => true,
                Some(best) => distances[last * n_o + candidate]
                    .total_cmp(&distances[last * n_o + best])
                    .is_lt(),
            };
            if better {
                nearest = Some(candidate);
            }
        }
        if let Some(next) = nearest {
            tour.push(next);
            visited[next] = true;
        }
    }
}

/// Improve `tour` in place with full 2-opt (every segment reversal), keeping a
/// reversal only when it strictly lowers `*cost` (`total_cmp`).
fn two_opt_improve(tour: &mut [usize], cost: &mut f64, distances: &[f64], n_o: usize) {
    let mut improved = true;
    while improved {
        improved = false;
        for i in 1..n_o.saturating_sub(1) {
            for j in (i + 1)..n_o {
                tour[i..=j].reverse();
                let candidate_cost = tour_cost(tour, distances, n_o);
                if candidate_cost.total_cmp(cost).is_lt() {
                    *cost = candidate_cost;
                    improved = true;
                } else {
                    tour[i..=j].reverse();
                }
            }
        }
    }
}

/// Lowest-cost Hamiltonian path over the row-major `n_o x n_o` distance
/// matrix: a nearest-neighbor tour from every start index, each improved by
/// full 2-opt, keeping the winning tour. All comparisons use `total_cmp` — a
/// `partial_cmp`/`<`/`min_by` reintroduces a NaN/`-0.0`-dependent order that
/// breaks run-to-run reproducibility.
fn shortest_chain_path(distances: &[f64], n_o: usize) -> Vec<usize> {
    debug_assert_eq!(
        distances.len(),
        n_o * n_o,
        "shortest_chain_path: distance matrix must be n_o x n_o"
    );

    let mut best: Vec<usize> = (0..n_o).collect();
    let mut best_cost = tour_cost(&best, distances, n_o);

    let mut tour = Vec::with_capacity(n_o);
    let mut visited = vec![false; n_o];
    for start in 0..n_o {
        nearest_neighbor_tour(distances, n_o, start, &mut tour, &mut visited);
        let mut cost = tour_cost(&tour, distances, n_o);
        two_opt_improve(&mut tour, &mut cost, distances, n_o);
        if cost.total_cmp(&best_cost).is_lt() {
            best_cost = cost;
            best.clone_from(&tour);
        }
    }

    best
}

/// Row-major pairwise unweighted-L2 distance matrix over the canonical
/// hydro-segment noise prefix of stage `stage`'s `n_o` openings.
fn l2_distance_matrix(
    tree: &OpeningTreeView<'_>,
    stage: usize,
    n_o: usize,
    hydro_segment: Range<usize>,
) -> Vec<f64> {
    let mut distances = vec![0.0_f64; n_o * n_o];
    for i in 0..n_o {
        let opening_i = &tree.opening(stage, i)[hydro_segment.clone()];
        for j in (i + 1)..n_o {
            let opening_j = &tree.opening(stage, j)[hydro_segment.clone()];
            let d = opening_i
                .iter()
                .zip(opening_j)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            distances[i * n_o + j] = d;
            distances[j * n_o + i] = d;
        }
    }
    distances
}

#[allow(clippy::cast_precision_loss)]
fn chain_position_key(n_o: usize, pos: usize) -> f64 {
    (n_o - pos) as f64
}

/// Overwrite each stage's keys with its shortest-chain path's reversed
/// positions (`keys[stage][ω] = n_o - pos`, where `pos` is ω's index in the
/// winning path); a stage below 3 openings keeps its σ-key — there is no path
/// to improve.
fn apply_chain_order(
    keys: &mut [Vec<f64>],
    tree: &OpeningTreeView<'_>,
    hydro_segment: Range<usize>,
) {
    for (stage, stage_keys) in keys.iter_mut().enumerate() {
        let n_o = stage_keys.len();
        if n_o < 3 {
            continue;
        }
        let distances = l2_distance_matrix(tree, stage, n_o, hydro_segment.clone());
        let path = shortest_chain_path(&distances, n_o);
        for (pos, &omega) in path.iter().enumerate() {
            stage_keys[omega] = chain_position_key(n_o, pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, Bus, DeficitSegment, EntityId, Hydro, HydroGenerationModel,
        HydroPenalties, InflowModel, NoiseMethod, SamplingScheme, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig, System, SystemBuilder,
    };
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use super::{StochasticContext, build_noise_key_table, noise_key};

    #[test]
    fn test_noise_key_sums_sigma_weighted_components() {
        let sigma = [30.0, 20.0, 10.0];
        let raw_noise = [1.5, -2.0, 0.5];
        // 30*1.5 + 20*(-2.0) + 10*0.5 = 45 - 40 + 5 = 10.
        let key = noise_key(&sigma, &raw_noise).expect("dims aligned");
        assert!((key - 10.0).abs() < 1e-12, "expected 10.0, got {key}");
    }

    #[test]
    fn test_noise_key_ignores_trailing_noise_components() {
        let sigma = [2.0, 4.0];
        let raw_noise = [1.0, 1.0, 100.0, -50.0];
        // 2*1 + 4*1 = 6; the 100.0 and -50.0 tail is ignored.
        let key = noise_key(&sigma, &raw_noise).expect("dims aligned");
        assert!((key - 6.0).abs() < 1e-12, "expected 6.0, got {key}");
    }

    #[test]
    fn test_noise_key_hard_errors_on_sigma_longer_than_noise() {
        let sigma = [1.0, 2.0, 3.0];
        let raw_noise = [1.0, 1.0];
        let err = noise_key(&sigma, &raw_noise).expect_err("must reject mismatch");
        let msg = err.to_string();
        assert!(msg.contains('3'), "message must name σ length 3: {msg}");
        assert!(msg.contains('2'), "message must name noise dim 2: {msg}");
    }

    // ------------------------------------------------------------------
    // shortest_chain_path: pure-kernel tests over hand-built distance matrices.
    // ------------------------------------------------------------------

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn shortest_chain_path_orders_collinear_points() {
        use super::shortest_chain_path;

        let n_o = 4;
        let mut distances = vec![0.0_f64; n_o * n_o];
        for i in 0..n_o {
            for j in 0..n_o {
                distances[i * n_o + j] = (i as f64 - j as f64).abs();
            }
        }

        let path = shortest_chain_path(&distances, n_o);
        assert_eq!(path.len(), n_o);
        let ascending = path.windows(2).all(|w| w[1] == w[0] + 1);
        let descending = path.windows(2).all(|w| w[0] == w[1] + 1);
        assert!(
            ascending || descending,
            "expected a monotone path over collinear points, got {path:?}"
        );
    }

    mod shortest_chain_path_proptests {
        use proptest::prelude::*;
        use proptest::test_runner::RngSeed;

        use super::super::shortest_chain_path;

        /// Fixed seed so a failing shrink is reproducible run-to-run.
        fn fixed_config() -> ProptestConfig {
            ProptestConfig {
                cases: 64,
                rng_seed: RngSeed::Fixed(7),
                ..ProptestConfig::default()
            }
        }

        fn size_and_matrix() -> impl Strategy<Value = (usize, Vec<f64>)> {
            (3usize..=8).prop_flat_map(|n_o| {
                prop::collection::vec(0.0..1000.0_f64, n_o * n_o).prop_map(move |d| (n_o, d))
            })
        }

        proptest! {
            #![proptest_config(fixed_config())]

            #[test]
            fn shortest_chain_path_is_always_a_permutation((n_o, distances) in size_and_matrix()) {
                let path = shortest_chain_path(&distances, n_o);
                let mut sorted = path.clone();
                sorted.sort_unstable();
                prop_assert_eq!(sorted, (0..n_o).collect::<Vec<_>>());
            }
        }
    }

    // ------------------------------------------------------------------
    // build_noise_key_table fixture tests.
    // ------------------------------------------------------------------

    fn bus_fixture(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn hydro_fixture(id: i32, bus_id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(EntityId(bus_id));
        hydro
    }

    fn stage_fixture(index: usize, id: i32, branching_factor: usize) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "SINGLE".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn inflow_model_fixture(hydro_id: i32, stage_id: i32, std_m3s: f64) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }
    }

    /// Build a `System` with one hydro per id in `hydro_order` (declared to
    /// `SystemBuilder` in that Vec order; `System::hydros()` re-sorts
    /// canonically regardless), `n_stages` stages each with `branching_factor`
    /// openings.
    fn build_system(hydro_order: &[i32], n_stages: usize, branching_factor: usize) -> System {
        let bus = bus_fixture(1);
        let hydros: Vec<Hydro> = hydro_order.iter().map(|&id| hydro_fixture(id, 1)).collect();
        let stages: Vec<Stage> = (0..n_stages)
            .map(|s| {
                stage_fixture(
                    s,
                    i32::try_from(s).expect("small stage count"),
                    branching_factor,
                )
            })
            .collect();
        let mut inflow_models: Vec<InflowModel> = hydro_order
            .iter()
            .flat_map(|&hydro_id| {
                stages.iter().map(move |stage| {
                    inflow_model_fixture(hydro_id, stage.id, 10.0 + f64::from(hydro_id))
                })
            })
            .collect();
        inflow_models.sort_by_key(|m| (m.hydro_id.0, m.stage_id));

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .build()
            .expect("fixture system must build")
    }

    fn build_ctx(system: &System, seed: u64) -> StochasticContext {
        build_stochastic_context(
            system,
            seed,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("fixture stochastic context must build")
    }

    #[test]
    fn chain_order_is_declaration_order_invariant() {
        let system_a = build_system(&[10, 20, 30], 2, 4);
        let system_b = build_system(&[30, 10, 20], 2, 4);

        let ctx_a = build_ctx(&system_a, 99);
        let ctx_b = build_ctx(&system_b, 99);

        let table_a = build_noise_key_table(&system_a, &ctx_a).expect("table must build");
        let table_b = build_noise_key_table(&system_b, &ctx_b).expect("table must build");

        assert_eq!(table_a.len(), table_b.len());
        for (stage_a, stage_b) in table_a.iter().zip(&table_b) {
            assert_eq!(stage_a.len(), stage_b.len());
            for (&ka, &kb) in stage_a.iter().zip(stage_b) {
                assert_eq!(
                    ka.to_bits(),
                    kb.to_bits(),
                    "shortest-chain key tables must be bit-identical across hydro declaration order"
                );
            }
        }
    }

    /// A stage with fewer than 3 openings keeps its σ-weighted key — the σ
    /// computation is the live small-stage fallback, not dead machinery. The
    /// fixture's single hydro (id 10) carries `std_m3s = 20.0` on every stage,
    /// so each expected key is `20.0 · η_ω[0]` verbatim.
    #[test]
    fn small_stages_keep_sigma_keys() {
        let system = build_system(&[10], 2, 2);
        let ctx = build_ctx(&system, 7);

        let table = build_noise_key_table(&system, &ctx).expect("table must build");

        let tree = ctx.tree_view();
        for (stage, stage_keys) in table.iter().enumerate() {
            assert_eq!(stage_keys.len(), 2, "fixture branches 2 openings per stage");
            for (omega, &key) in stage_keys.iter().enumerate() {
                let expected = 20.0 * tree.opening(stage, omega)[0];
                assert_eq!(
                    key.to_bits(),
                    expected.to_bits(),
                    "a below-3-openings stage must keep its σ-key"
                );
            }
        }
    }
}
