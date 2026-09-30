//! `water_balance_coupling` section tests.
//!
//! Every water-balance row reads the realized inflow through the `z_h`
//! column (`push_z_inflow_coupling` in `lp/builder/entries.rs`) rather than
//! an AR-lag column or a PAR base baked into the row's own RHS.

use std::collections::HashSet;

use super::*;

use cobre_sddp::indexer::{BlockIdx, HydroSys};
use cobre_sddp::test_support::template_structure::{
    RowOwner, UnscaledMatrix, hours_to_hm3, row_owners, z_inflow_column,
};

use super::common;

#[test]
fn every_study_reads_z_inflow_on_its_water_rows() {
    let count = common::for_each_study(|key, system, setup| {
        let state = setup.stage_state();
        let n_hydros = state.hydro_count;
        if n_hydros == 0 {
            return;
        }
        let inflow_lags = state.inflow_lags.clone();
        let templates = &setup.inputs.stage_data.stage_templates;

        for (s, t) in templates.templates.iter().enumerate() {
            let geom = &templates.geometry_per_stage[s];
            let owners = row_owners(system, geom, state);
            let matrix = UnscaledMatrix::of(t);
            let rows_per_entity = geom.water_balance.rows_per_entity(geom.n_blks);
            let block_hours = &templates.block_hours_per_stage[s];
            let zeta = hours_to_hm3(block_hours.iter().sum());
            let water_hydro_of = |r: usize| -> Option<HydroSys> {
                match owners.get(&r) {
                    Some(&RowOwner::Water { hydro, .. }) => Some(hydro),
                    _ => None,
                }
            };
            let is_frozen = |d: usize| -> bool {
                (0..rows_per_entity).all(|k| {
                    matrix
                        .row(geom.water_balance_row(HydroSys::new(d), BlockIdx::new(k)))
                        .len()
                        == 2
                })
            };

            for h in 0..n_hydros {
                let h_sys = HydroSys::new(h);
                let col_z = z_inflow_column(state, h_sys);
                let water_entries: Vec<(usize, f64)> = matrix
                    .col(col_z)
                    .iter()
                    .copied()
                    .filter(|&(r, _)| water_hydro_of(r).is_some())
                    .collect();

                if water_entries.is_empty() {
                    assert!(
                        is_frozen(h),
                        "{key}: stage {s} hydro {h}: z_{h} has no water-row entry, but \
                         hydro {h} is not frozen"
                    );
                    continue;
                }

                let hydros_hit: std::collections::BTreeSet<usize> = water_entries
                    .iter()
                    .map(|&(r, _)| water_hydro_of(r).expect("filtered above").get())
                    .collect();
                assert_eq!(
                    hydros_hit.len(),
                    1,
                    "{key}: stage {s} hydro {h}: z_{h} has water-row entries on more than \
                     one hydro's rows: {hydros_hit:?}"
                );
                let d = *hydros_hit
                    .iter()
                    .next()
                    .expect("hydros_hit has exactly one element");

                assert_eq!(
                    water_entries.len(),
                    rows_per_entity,
                    "{key}: stage {s} hydro {h}: z_{h} must have exactly one entry per row \
                     of hydro {d} ({rows_per_entity} rows), got {}",
                    water_entries.len()
                );

                for &(r, v) in &water_entries {
                    let RowOwner::Water { blk, .. } = owners[&r] else {
                        unreachable!("filtered to Water rows above")
                    };
                    let expected = if rows_per_entity == 1 {
                        -zeta
                    } else {
                        -hours_to_hm3(block_hours[blk.get()])
                    };
                    assert!(
                        (v - expected).abs() < 1e-12 * zeta.abs(),
                        "{key}: stage {s} hydro {h}: z_{h} coefficient at row {r} = \
                         {v}, expected {expected}"
                    );
                }

                // (e) the sum of z_h's water coefficients is -zeta whenever z_h
                // has any water-row entry at all (0.0, the empty-entries branch
                // above, only when h is frozen).
                let sum: f64 = water_entries.iter().map(|&(_, v)| v).sum();
                let expected_sum = -zeta;
                assert!(
                    (sum - expected_sum).abs() <= 1e-12 * zeta.abs(),
                    "{key}: stage {s} hydro {h}: sum of z_{h}'s water coefficients = {sum}, \
                     expected {expected_sum}"
                );

                if is_frozen(h) {
                    assert!(
                        d == h || !is_frozen(d),
                        "{key}: stage {s} hydro {h}: z_{h} routes to hydro {d}'s water \
                         rows, but hydro {d} is also frozen"
                    );
                } else {
                    assert_eq!(
                        d, h,
                        "{key}: stage {s} hydro {h}: hydro {h} is not frozen, but z_{h} \
                         routes to hydro {d}'s water rows"
                    );
                }
            }

            for c in inflow_lags.clone() {
                for &(r, _) in matrix.col(c) {
                    assert!(
                        water_hydro_of(r).is_none(),
                        "{key}: stage {s}: inflow-lag column {c} has an entry on water \
                         row {r}"
                    );
                }
            }

            // (d) no (row, col) pair occurs twice on any water row, for any column.
            for (&r, owner) in &owners {
                if matches!(owner, RowOwner::Water { .. }) {
                    let mut seen = HashSet::new();
                    for &(c, _) in matrix.row(r) {
                        assert!(
                            seen.insert(c),
                            "{key}: stage {s}: water row {r} has a duplicate entry for \
                             column {c}"
                        );
                    }
                }
            }
        }
    });

    assert_eq!(
        count,
        common::expected_study_count(),
        "every committed deck (minus SLOW_DECKS skips) plus every in-code and structural \
         study must be swept"
    );
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
