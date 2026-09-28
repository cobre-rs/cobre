//! `water_balance_coupling` section tests.
//!
//! Every water-balance row reads the realized inflow through the `z_h`
//! column (`push_z_inflow_coupling` in `lp/builder/entries.rs`) rather than
//! an AR-lag column or a PAR base baked into the row's own RHS.

use super::*;

use cobre_sddp::test_support::decks::{SLOW_DECKS, committed_decks};

use super::common::fresh_setup_with;

/// Nonzero-entry count per row across the whole template (every column), used
/// to detect a frozen water row (the `v_h - v_h_in = 0` identity, exactly two
/// nonzero entries) without depending on the builder's private phase state.
#[allow(clippy::cast_sign_loss)]
fn row_nnz_counts(t: &StageTemplate) -> Vec<usize> {
    let mut counts = vec![0usize; t.num_rows];
    for &r in &t.row_indices {
        counts[r as usize] += 1;
    }
    counts
}

#[test]
fn every_committed_deck_reads_z_inflow_on_its_water_rows() {
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    let slow_tests_enabled = cfg!(feature = "slow-tests");

    for deck in committed_decks() {
        if !slow_tests_enabled && SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let deck_key = deck.key.as_str();
        let setup = fresh_setup_with(&deck.dir, |_| {});
        let n_hydros = setup.stage_state().hydro_count;
        if n_hydros == 0 {
            continue;
        }
        let z_inflow_start = setup.stage_state().z_inflow.start;
        let inflow_lags = setup.stage_state().inflow_lags.clone();
        let templates = &setup.stage_data.stage_templates;

        for (s, t) in templates.templates.iter().enumerate() {
            let geom = &templates.geometry_per_stage[s];
            let water = geom.water_balance.range();
            let stride = water.len() / n_hydros;
            let block_hours = &templates.block_hours_per_stage[s];
            let zeta = block_hours.iter().sum::<f64>() * M3S_TO_HM3;
            let nnz = row_nnz_counts(t);
            let is_frozen =
                |d: usize| -> bool { (0..stride).all(|k| nnz[water.start + d * stride + k] == 2) };
            let unscale = |r: usize, c: usize, v: f64| -> f64 {
                let rs = t.row_scale.get(r).copied().unwrap_or(1.0);
                let cs = t.col_scale.get(c).copied().unwrap_or(1.0);
                v / (rs * cs)
            };

            for h in 0..n_hydros {
                let col_z = z_inflow_start + h;
                let water_entries: Vec<(usize, f64)> = entries_for_col(t, col_z)
                    .into_iter()
                    .filter(|&(r, _)| water.contains(&r))
                    .collect();

                if water_entries.is_empty() {
                    assert!(
                        is_frozen(h),
                        "{deck_key}: stage {s} hydro {h}: z_{h} has no water-row entry, but \
                         hydro {h} is not frozen"
                    );
                    continue;
                }

                let owners: std::collections::BTreeSet<usize> = water_entries
                    .iter()
                    .map(|&(r, _)| (r - water.start) / stride)
                    .collect();
                assert_eq!(
                    owners.len(),
                    1,
                    "{deck_key}: stage {s} hydro {h}: z_{h} has water-row entries on more than \
                     one hydro's rows: {owners:?}"
                );
                let d = *owners
                    .iter()
                    .next()
                    .expect("owners has exactly one element");

                assert_eq!(
                    water_entries.len(),
                    stride,
                    "{deck_key}: stage {s} hydro {h}: z_{h} must have exactly one entry per row \
                     of hydro {d} ({stride} rows), got {}",
                    water_entries.len()
                );

                for &(r, v) in &water_entries {
                    let unscaled = unscale(r, col_z, v);
                    let expected = if stride == 1 {
                        -zeta
                    } else {
                        let k = r - (water.start + d * stride);
                        -(block_hours[k] * M3S_TO_HM3)
                    };
                    assert!(
                        (unscaled - expected).abs() < 1e-12 * zeta.abs(),
                        "{deck_key}: stage {s} hydro {h}: z_{h} coefficient at row {r} = \
                         {unscaled}, expected {expected}"
                    );
                }

                if is_frozen(h) {
                    assert!(
                        d == h || !is_frozen(d),
                        "{deck_key}: stage {s} hydro {h}: z_{h} routes to hydro {d}'s water \
                         rows, but hydro {d} is also frozen"
                    );
                } else {
                    assert_eq!(
                        d, h,
                        "{deck_key}: stage {s} hydro {h}: hydro {h} is not frozen, but z_{h} \
                         routes to hydro {d}'s water rows"
                    );
                }
            }

            for c in inflow_lags.clone() {
                for &(r, _) in &entries_for_col(t, c) {
                    assert!(
                        !water.contains(&r),
                        "{deck_key}: stage {s}: inflow-lag column {c} has an entry on water \
                         row {r}"
                    );
                }
            }
        }
    }
}

/// Builder-level structural check on the PAR fixture of
/// `max_par_order_z_inflow_row_has_twelve_lag_entries`: the water rows carry
/// no PAR base (row bounds both 0.0), and the z-inflow row for each hydro
/// carries exactly `par_lp.deterministic_base`.
#[test]
fn water_rows_carry_no_par_base() {
    use cobre_core::scenario::AnnualComponent;

    let ar_coeffs: Vec<f64> = vec![0.3, 0.2];
    let ann = AnnualComponent {
        coefficient: 0.5,
        mean_m3s: 80.0,
        std_m3s: 20.0,
    };
    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: Some(ann),
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 60.0,
            std_m3s: 15.0,
            ar_coefficients: ar_coeffs.clone(),
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let system = two_hydro_par_system(2, inflow_models.clone());
    let stages = system.stages().to_vec();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let par_lp =
        PrecomputedPar::build(&inflow_models, &stages, &hydro_ids, None).expect("par build ok");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &par_lp,
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout ok");

    let t = &result.templates[0];
    let water = result.geometry_per_stage[0].water_balance.range();
    for r in water {
        assert_eq!(
            t.row_lower[r], 0.0,
            "water row {r} row_lower must be 0.0, got {}",
            t.row_lower[r]
        );
        assert_eq!(
            t.row_upper[r], 0.0,
            "water row {r} row_upper must be 0.0, got {}",
            t.row_upper[r]
        );
    }

    let n_h = 2_usize;
    for h in 0..n_h {
        let expected = par_lp.deterministic_base(0, h);
        assert_eq!(
            t.row_lower[h], expected,
            "z row {h} row_lower must equal par_lp.deterministic_base(0, {h}) = {expected}, got {}",
            t.row_lower[h]
        );
    }
}
