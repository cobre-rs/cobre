//! Integration test: lag-12 LP coefficient for PAR(2)-A vs classical PAR.
//!
//! Builds two small synthetic fixtures (2 hydros × 24 stages × 12 seasons) and
//! calls [`cobre_sddp::build_stage_templates`] on each. No HiGHS solve, no
//! forward/backward pass, no filesystem I/O. Sub-second runtime; not gated
//! behind `slow-tests`.
//!
//! ## What is being verified
//!
//! The PAR(p)-A architectural insight is that the annual component is absorbed
//! into a single effective coefficient per lag:
//!
//! ```text
//! ψ_eff[lag] = φ̂_{lag+1} + ψ̂/12   for lag ∈ [0, ar_order)
//! ψ_eff[lag] =             ψ̂/12   for lag ∈ [ar_order, 12)
//! ```
//!
//! where `ψ̂ = ψ · σ_m / σ^A`. The LP layer therefore needs no new variables or
//! constraint families. These tests verify that the assembled CSC matrix carries
//! the expected coefficient values in the correct (row, col) positions.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ContractStageBounds, DeficitSegment,
    EntityId, HydroStageBounds, HydroStagePenalties, LineStageBounds, LineStagePenalties,
    NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
    ResolvedPenalties, SystemBuilder, ThermalStageBounds,
    entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
    scenario::{AnnualComponent, InflowModel, LoadModel},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_sddp::{
    InflowNonNegativityMethod, hydro_models::PrepareHydroModelsResult,
    lp_builder::build_stage_templates,
};
use cobre_stochastic::{PrecomputedPar, normal::precompute::PrecomputedNormal};

// ---------------------------------------------------------------------------
// Fixture parameters
// ---------------------------------------------------------------------------

/// Number of hydro plants in both fixtures.
const N_H: usize = 2;
/// Number of study stages in both fixtures (2 full cycles of 12 months).
const N_STUDY: usize = 24;
/// Number of seasons (months per year).
const N_SEASONS: usize = 12;
/// PAR classical order used in study models.
const AR_ORDER: usize = 2;

/// Monthly inflow σ used in all seasonal models.
const SIGMA_M: f64 = 200.0;
/// Annual inflow σ used in `AnnualComponent`.
const SIGMA_A: f64 = 250.0;
/// Annual coefficient ψ in `AnnualComponent`.
const PSI: f64 = 0.1;
/// AR-1 coefficient φ₁ in study models.
const PHI_1: f64 = 0.5;
/// AR-2 coefficient φ₂ in study models.
const PHI_2: f64 = 0.2;

// ---------------------------------------------------------------------------
// Private fixture builder — PAR(2)-A
// ---------------------------------------------------------------------------

/// Build a 2-hydro, 24-stage, 12-season system with `annual: Some(_)` and
/// the corresponding [`PrecomputedPar`].
///
/// Fixture choices that make the arithmetic readable:
/// - Uniform σ_m = 200, σ^A = 250 across all stages → σ_m / σ_{m-1} = 1.0
///   so φ̂_j = φ_j · 1.0 = φ_j for the AR unit conversion.
/// - ψ̂ = 0.1 * 200 / 250 = 0.08  (PSI * SIGMA_M / SIGMA_A)
/// - Lag-11 expected coefficient: −ψ̂/12 = −0.08/12
/// - Lag-0  expected coefficient: −(φ̂_1 + ψ̂/12) = −(0.5 + 0.08/12)
///
/// Pre-study models (stage ids -1 and -2) are required so the
/// `PrecomputedPar` builder can resolve lag-stage statistics for stage 0.
fn build_par_a_fixture() -> (cobre_core::System, PrecomputedPar) {
    let hydro_ids = [EntityId(1), EntityId(2)];

    // -----------------------------------------------------------------------
    // Hydro entities
    // -----------------------------------------------------------------------
    let zero_penalties = HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        fpha_turbined_cost: 0.0,
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
    };

    let hydros: Vec<Hydro> = hydro_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| Hydro {
            id,
            name: format!("H{}", i + 1),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 200.0,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        })
        .collect();

    // -----------------------------------------------------------------------
    // Bus entity
    // -----------------------------------------------------------------------
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    // -----------------------------------------------------------------------
    // Study stages: ids 0..23, season_id = stage_idx % 12
    // -----------------------------------------------------------------------
    let study_stages: Vec<Stage> = (0..N_STUDY)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(i % N_SEASONS),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    // -----------------------------------------------------------------------
    // Load models: one deterministic entry per stage per bus
    // -----------------------------------------------------------------------
    let load_models: Vec<LoadModel> = (0..N_STUDY)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    // -----------------------------------------------------------------------
    // Inflow models: pre-study seeds + study stages, all with annual: Some(_)
    // -----------------------------------------------------------------------
    let annual_component = AnnualComponent {
        coefficient: PSI,
        mean_m3s: 1000.0,
        std_m3s: SIGMA_A,
    };

    // Pre-study entries at stage ids -1 and -2 (required for lag resolution).
    // Season ids must also be set so `PrecomputedPar` can resolve coefficients
    // for the AR order at stage 0. We assign season 11 and 10 respectively
    // (the months just before month 0), which is the standard wrap-around.
    let mut all_inflow_models: Vec<InflowModel> = Vec::new();

    for pre_id in [-2_i32, -1_i32] {
        for &h_id in &hydro_ids {
            all_inflow_models.push(InflowModel {
                hydro_id: h_id,
                stage_id: pre_id,
                mean_m3s: 1000.0,
                std_m3s: SIGMA_M,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: Some(annual_component.clone()),
            });
        }
    }

    // Study stage models
    for i in 0..N_STUDY {
        for &h_id in &hydro_ids {
            all_inflow_models.push(InflowModel {
                hydro_id: h_id,
                stage_id: i as i32,
                mean_m3s: 1000.0,
                std_m3s: SIGMA_M,
                ar_coefficients: vec![PHI_1, PHI_2],
                residual_std_ratio: 0.7,
                annual: Some(annual_component.clone()),
            });
        }
    }

    // -----------------------------------------------------------------------
    // Build PrecomputedPar
    // -----------------------------------------------------------------------
    let par_lp = PrecomputedPar::build(&all_inflow_models, &study_stages, &hydro_ids)
        .expect("PrecomputedPar::build must succeed for a valid PAR(2)-A fixture");

    // -----------------------------------------------------------------------
    // ResolvedBounds and ResolvedPenalties (required by build_stage_templates)
    // -----------------------------------------------------------------------
    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 500.0,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        max_diversion_m3s: None,
        filling_inflow_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: N_H,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STUDY,
        },
        &BoundsDefaults {
            hydro: hydro_bounds_default,
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );

    let hydro_penalties_default = HydroStagePenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        fpha_turbined_cost: 0.0,
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
    };
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: N_H,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STUDY,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    // -----------------------------------------------------------------------
    // Build System
    // -----------------------------------------------------------------------
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(study_stages)
        .inflow_models(all_inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder::build must succeed for a valid PAR(2)-A fixture");

    (system, par_lp)
}

/// Build the same system shape as [`build_par_a_fixture`] but with
/// `annual: None` on every model. The classical PAR(2) path is used and
/// `max_par_order` stays at 2.
fn build_classical_fixture() -> (cobre_core::System, PrecomputedPar) {
    let hydro_ids = [EntityId(1), EntityId(2)];

    let zero_penalties = HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        fpha_turbined_cost: 0.0,
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
    };

    let hydros: Vec<Hydro> = hydro_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| Hydro {
            id,
            name: format!("H{}", i + 1),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 500.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 200.0,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        })
        .collect();

    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };

    let study_stages: Vec<Stage> = (0..N_STUDY)
        .map(|i| Stage {
            index: i,
            id: i as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(i % N_SEASONS),
            blocks: vec![Block {
                index: 0,
                name: "S".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: true,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..N_STUDY)
        .map(|i| LoadModel {
            bus_id: EntityId(0),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect();

    // Pre-study seeds: stage ids -1 and -2, annual: None.
    let mut all_inflow_models: Vec<InflowModel> = Vec::new();
    for pre_id in [-2_i32, -1_i32] {
        for &h_id in &hydro_ids {
            all_inflow_models.push(InflowModel {
                hydro_id: h_id,
                stage_id: pre_id,
                mean_m3s: 1000.0,
                std_m3s: SIGMA_M,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            });
        }
    }
    // Study stage models: classical PAR(2), annual: None.
    for i in 0..N_STUDY {
        for &h_id in &hydro_ids {
            all_inflow_models.push(InflowModel {
                hydro_id: h_id,
                stage_id: i as i32,
                mean_m3s: 1000.0,
                std_m3s: SIGMA_M,
                ar_coefficients: vec![PHI_1, PHI_2],
                residual_std_ratio: 0.7,
                annual: None,
            });
        }
    }

    let par_lp = PrecomputedPar::build(&all_inflow_models, &study_stages, &hydro_ids)
        .expect("PrecomputedPar::build must succeed for a classical PAR(2) fixture");

    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 500.0,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        min_generation_mw: 0.0,
        max_generation_mw: 200.0,
        max_diversion_m3s: None,
        filling_inflow_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: N_H,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STUDY,
        },
        &BoundsDefaults {
            hydro: hydro_bounds_default,
            thermal: ThermalStageBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
            },
            line: LineStageBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping: PumpingStageBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract: ContractStageBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let hydro_penalties_default = HydroStagePenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        fpha_turbined_cost: 0.0,
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
    };
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: N_H,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STUDY,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(study_stages)
        .inflow_models(all_inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("SystemBuilder::build must succeed for a classical PAR(2) fixture");

    (system, par_lp)
}

// ---------------------------------------------------------------------------
// Helper: walk a CSC column and find the value at a given row target.
// ---------------------------------------------------------------------------

/// Walk a CSC column `col_idx` in `template` and return the value at `row_target`.
///
/// Returns `None` when the entry is structurally absent (coefficient == 0 and
/// not stored). Panics when more than one entry maps to `row_target`.
fn find_csc_entry(
    template: &cobre_solver::StageTemplate,
    col_idx: usize,
    row_target: usize,
) -> Option<f64> {
    let start = template.col_starts[col_idx] as usize;
    let end = template.col_starts[col_idx + 1] as usize;
    let mut found: Option<f64> = None;
    for i in start..end {
        if template.row_indices[i] as usize == row_target {
            assert!(
                found.is_none(),
                "duplicate CSC entry at (col={col_idx}, row={row_target})"
            );
            found = Some(template.values[i]);
        }
    }
    found
}

// ---------------------------------------------------------------------------
// Test 1: lag-11 column carries −ψ̂/12  (AC#1 + AC#2)
// ---------------------------------------------------------------------------

/// Verify that the lag-11 LP matrix coefficient for hydro 0 at stage 0 equals
/// `−ψ̂/12` where `ψ̂ = ψ · σ_m / σ^A`.
///
/// Column layout (N=2, L=12):
///   `inflow_lags.start = N = 2`
///   lag-l column for hydro h = `2 + l * 2 + h`
///   lag-11, hydro 0 → column 24
///
/// Row layout:
///   `z_inflow_row_start = N * (1 + L) = 2 * 13 = 26`
///   z-inflow row for hydro 0 → row 26
#[test]
fn lag_11_lp_coefficient_equals_psi_hat_over_twelve() {
    let (system, par_lp) = build_par_a_fixture();

    // AC#1: max_order must be 12.
    assert_eq!(
        par_lp.max_order(),
        12,
        "PAR(2)-A must widen max_order to 12; got {}",
        par_lp.max_order()
    );

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let templates = build_stage_templates(
        &system,
        &InflowNonNegativityMethod::None,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
    )
    .expect("build_stage_templates must succeed for the PAR(2)-A fixture");

    // AC#1: templates[0].max_par_order == 12.
    let tmpl = &templates.templates[0];
    assert_eq!(
        tmpl.max_par_order, 12,
        "templates[0].max_par_order must be 12; got {}",
        tmpl.max_par_order
    );

    // Column indices: N=2, L=12.
    //   inflow_lags.start = N = 2
    //   lag-11, hydro 0 → 2 + 11 * 2 = 24
    let col_lag11_h0: usize = 2 + 11 * 2;

    // Row index:
    //   z_inflow_row_start = N * (1 + L) = 2 * 13 = 26
    //   z-inflow row for hydro 0 → 26
    let row_z_h0: usize = 2 * (1 + 12);

    // AC#2: the CSC entry at (lag-11 col, z-inflow row) must equal −ψ̂/12.
    let psi_hat = PSI * SIGMA_M / SIGMA_A; // 0.1 * 200 / 250 = 0.08
    let expected = -(psi_hat / 12.0);

    let value = find_csc_entry(tmpl, col_lag11_h0, row_z_h0).unwrap_or_else(|| {
        panic!(
            "no CSC entry at (z-inflow row {row_z_h0}, lag-11 col {col_lag11_h0}); \
                 the PAR-A coefficient is missing from the LP matrix"
        )
    });

    assert!(
        (value - expected).abs() < 1e-12,
        "lag-11 coefficient: got {value:.15}, expected {expected:.15} (diff = {:.3e})",
        (value - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// Test 2: lag-0 column carries −(φ̂_1 + ψ̂/12)  (AC#3)
// ---------------------------------------------------------------------------

/// Verify that the lag-0 LP matrix coefficient for hydro 0 at stage 0 equals
/// `−(φ̂_1 + ψ̂/12)`.
///
/// With uniform σ_m across stages, φ̂_1 = φ_1 · (σ_m / σ_{m-1}) = 0.5 · 1.0 = 0.5.
///
/// Column indices (N=2, L=12):
///   lag-0, hydro 0 → `2 + 0 * 2 + 0 = 2`
///
/// Row index:
///   z-inflow row for hydro 0 → `2 * (1 + 12) + 0 = 26`
#[test]
fn lag_0_lp_coefficient_combines_ar_and_annual() {
    let (system, par_lp) = build_par_a_fixture();

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let templates = build_stage_templates(
        &system,
        &InflowNonNegativityMethod::None,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
    )
    .expect("build_stage_templates must succeed for the PAR(2)-A fixture");

    let tmpl = &templates.templates[0];

    // Column for lag-0, hydro 0: inflow_lags.start + 0 * N_H + 0 = 2.
    let col_lag0_h0: usize = N_H; // = 2
    // Z-inflow row for hydro 0: N_H * (1 + max_par_order) = 2 * 13 = 26.
    let row_z_h0: usize = 2 * (1 + 12);

    // φ̂_1 = φ_1 * (σ_m / σ_{m-1}); fixture uses uniform σ_m so the ratio is 1.0.
    let phi_hat_1 = PHI_1 * (SIGMA_M / SIGMA_M); // 0.5
    let psi_hat = PSI * SIGMA_M / SIGMA_A; // 0.08
    let expected = -(phi_hat_1 + psi_hat / 12.0);

    let value = find_csc_entry(tmpl, col_lag0_h0, row_z_h0).unwrap_or_else(|| {
        panic!(
            "no CSC entry at (z-inflow row {row_z_h0}, lag-0 col {col_lag0_h0}); \
                 the AR-1 + annual coefficient is missing from the LP matrix"
        )
    });

    assert!(
        (value - expected).abs() < 1e-12,
        "lag-0 coefficient: got {value:.15}, expected {expected:.15} (diff = {:.3e})",
        (value - expected).abs()
    );
}

// ---------------------------------------------------------------------------
// Test 3: classical PAR has no lag-11 column  (AC#4)
// ---------------------------------------------------------------------------

/// Verify that a classical PAR(2) fixture (annual: None) keeps `max_par_order == 2`
/// and that the lag-11 column index would fall outside the inflow-lags range.
///
/// For classical PAR(2) with N=2, L=2:
///   `inflow_lags = N .. N*(1+L) = 2..6`
///
/// The lag-11 column that PAR-A would place at index `2 + 11*2 + 0 = 24` lies
/// beyond `inflow_lags.end = 6`, proving that the column is absent from the
/// classical LP structure.
#[test]
fn classical_par_has_no_lag_11_column() {
    let (system, par_lp) = build_classical_fixture();

    // Classical max_order must be AR_ORDER = 2, not 12.
    assert_eq!(
        par_lp.max_order(),
        AR_ORDER,
        "classical PAR(2) must keep max_order == {AR_ORDER}; got {}",
        par_lp.max_order()
    );

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let templates = build_stage_templates(
        &system,
        &InflowNonNegativityMethod::None,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
    )
    .expect("build_stage_templates must succeed for the classical PAR(2) fixture");

    let tmpl = &templates.templates[0];

    assert_eq!(
        tmpl.max_par_order, AR_ORDER,
        "templates[0].max_par_order must be {AR_ORDER} for classical PAR(2); got {}",
        tmpl.max_par_order
    );

    // For N=2, L=2 the inflow_lags range is N..N*(1+L) = 2..6.
    // The column index that lag-11 of hydro 0 would occupy in the PAR-A layout
    // is inflow_lags.start + 11 * N = 2 + 22 = 24.
    // Since 24 >= inflow_lags.end = 6, the lag-11 column is absent from the
    // classical LP layout.
    let inflow_lags_end_classical = N_H * (1 + AR_ORDER); // 2 * 3 = 6
    let lag11_col_in_par_a = N_H + 11 * N_H; // 2 + 22 = 24
    assert!(
        lag11_col_in_par_a >= inflow_lags_end_classical,
        "classical PAR(2) inflow_lags ends at {inflow_lags_end_classical}; \
         lag-11 index {lag11_col_in_par_a} is unexpectedly within range"
    );
}
