//! Consolidated anticipated-scenarios integration tests for `cobre-sddp`.
//!
//! Grouped into inner `mod`s in one binary so the statically-linked solver links
//! once, not once per file.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

mod common;

use chrono::{NaiveDate, TimeDelta};
use cobre_core::{AnticipatedCommitmentHistory, EntityId};

/// One windowed commitment per value, tiling stage `i`'s
/// `[anchor + i*days_per_stage, anchor + (i+1)*days_per_stage)` span for `i`
/// in `0..values.len()`.
///
/// `StageCalendar::coverage` (`cobre-stochastic`) resolves fractional overlap
/// against each stage's own real `[start_date, end_date)` calendar span, so a
/// window's `days_per_stage` must equal the matching `daily_stage_dates` call
/// that built the fixture's `Stage`s — the two must derive from the same
/// `anchor`/`days_per_stage`, or the window and the stage it should fully
/// cover disagree and `build_initial_state`'s commitment-hold seed silently never
/// writes the value (`fraction != 1.0`).
fn windowed_commitments_daily(
    thermal_id: EntityId,
    anchor: NaiveDate,
    days_per_stage: i64,
    values: &[f64],
) -> Vec<AnticipatedCommitmentHistory> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
            thermal_id,
            start_date: anchor + TimeDelta::days(days_per_stage * i as i64),
            end_date: anchor + TimeDelta::days(days_per_stage * (i as i64 + 1)),
            value_mw,
        })
        .collect()
}

/// Sequential stage boundary dates for `n_stages` stages of
/// `days_per_stage` days each, starting at `anchor` — pass the SAME
/// `anchor`/`days_per_stage` to [`windowed_commitments_daily`] so a
/// commitment window's span matches a `Stage`'s own span exactly.
fn daily_stage_dates(
    anchor: NaiveDate,
    n_stages: usize,
    days_per_stage: i64,
) -> Vec<(NaiveDate, NaiveDate)> {
    (0..n_stages)
        .map(|i| {
            (
                anchor + TimeDelta::days(days_per_stage * i as i64),
                anchor + TimeDelta::days(days_per_stage * (i as i64 + 1)),
            )
        })
        .collect()
}

mod anticipated_5stage_k2_smoke {
    //! Smoke test for K=2 anticipated thermal dispatch, 5 stages: structural
    //! assertions only. No `EXPECTED_LB` is pinned — this fixture has no closed form,
    //! so a pinned value would certify stability, not correctness. Value-correctness
    //! is the `anticipated_closed_form_lb_k1_single_thermal` canary in
    //! `anticipated_core.rs`; this defends multi-stage state propagation, the K=2
    //! commitment-hold carry, and basis-cache capture.

    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage, InitialConditions,
        LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::SolverStatsDelta;
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::anticipated_structural_assertions::{
        assert_anticipated_delivery_slots_populated, assert_basis_cache_fully_populated,
        assert_training_converged_structurally,
    };
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// The LP is always feasible: the backup thermal alone covers the load.
    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(2);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 500.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
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
                ..Default::default()
            },
        );

        let n_stages = 5_usize;
        let calendar_anchor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let stage_dates = super::daily_stage_dates(calendar_anchor, n_stages, 31);
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_dates[i].0,
                        end_date: stage_dates[i].1,
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 744.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        let k_max: usize = 2;

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            }
        }

        fn default_hydro_penalties() -> HydroPenalties {
            HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: super::windowed_commitments_daily(
                anticipated_id,
                calendar_anchor,
                31,
                &[100.0, 50.0],
            ),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 8 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Integration test
    // ---------------------------------------------------------------------------

    #[test]
    fn test_anticipated_5stage_k2_analytical_lb() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 8, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        let state = setup.stage_state();
        let commit_out_start = state.commit_out.start;
        let n_anticipated = state.n_anticipated;

        assert_training_converged_structurally(result, &[], 8);
        assert_basis_cache_fully_populated(result, 5);
        // K=2 strict predicate (t + K < n_stages): decisions at t in {0,1,2} deliver
        // at t+K in {2,3,4}.
        assert_anticipated_delivery_slots_populated(
            result,
            commit_out_start,
            n_anticipated,
            &[2, 3, 4],
        );
    }

    /// Warm-start regression: the anticipated ring's `commit_out` block
    /// shifts every downstream column by `n_anticipated * k_max`. `reconstruct_basis`
    /// matches stored cut rows by `CutPool` slot identity, never absolute column index,
    /// so the shift must stay transparent — zero `basis_consistency_failures`.
    #[test]
    fn test_anticipated_5stage_k2_warm_start_zero_basis_rejections() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 8, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        assert_eq!(
            result.iterations, 8,
            "warm-start regression needs the full 8-iteration run to exercise \
         reconstruct_basis on iterations 2..8"
        );

        let total_rejections =
            SolverStatsDelta::aggregate(result.solver_stats_log.iter().map(|entry| &entry.delta))
                .basis_consistency_failures;

        assert_eq!(
            total_rejections, 0,
            "anticipated warm-start: expected 0 basis rejections with the \
         anticipated ring in the state region, got {total_rejections} \
         (reconstruct_basis must match cut rows by CutPool slot identity, not by \
         absolute column index, so the column shift is transparent)"
        );
    }
}

mod anticipated_two_plants_smoke {
    //! Training lower bound for a 6-stage system with 2 anticipated thermals
    //! (K_1=2, K_2=4), 1 backup thermal, and 1 hydro.
    //!
    //! With `n_anticipated=2` and `K_max=4`, the commitment-hold block has
    //! `2 * 4 = 8` columns in slot-major, plant-minor order
    //! (`ant_start + slot * n_anticipated + plant`) — the index arithmetic the
    //! assertions below depend on:
    //!
    //! ```text
    //! ant_start + 0 = slot 0, plant 0  (K=2)
    //! ant_start + 1 = slot 0, plant 1  (K=4)
    //! ant_start + 2 = slot 1, plant 0  (K=2)
    //! ant_start + 3 = slot 1, plant 1  (K=4)
    //! ant_start + 4 = slot 2, plant 0  (K=2)
    //! ant_start + 5 = slot 2, plant 1  (K=4)
    //! ant_start + 6 = slot 3, plant 0  (K=2)
    //! ant_start + 7 = slot 3, plant 1  (K=4)
    //! ```
    //!
    //! Under the HOLD geometry each commitment is held at its delivery-target modular
    //! slot (`m mod k_max`) until it matures — the ring never shifts slots. The carry
    //! invariant asserts a not-yet-matured commitment (plant 1's slot-3 delivery, held
    //! across stages 0-2) keeps the SAME slot across the consecutive forward-outgoing
    //! captures `basis_cache[1]` (stage 0) and `basis_cache[2]` (stage 1). It uses
    //! stage 0->1, not the trivial `basis_cache[0] == basis_cache[1]` identity, so it
    //! exercises a genuine forward-to-forward hold advancement.

    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage, InitialConditions,
        LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // Closed form over 744 h stages: each stage's residual (150 MW load minus its
    // delivered commitments; seeded ones are sunk) at the cheapest source the
    // lead times still reach — stage 0: 70 MW @500; stage 1: 95 @500; stage 2:
    // 100 K2 @50 + 20 @500; stage 3: 100 K2 @50 + 15 @500; stages 4 and 5:
    // 80 K4 @40 + 70 K2 @50 each.
    const EXPECTED_LB: f64 = 91_809_600.0_f64;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 6-stage two-anticipated-plant system. `SystemBuilder::build()` sorts
    /// thermals by id into `[id=2 (ant K=2), id=4 (backup), id=5 (ant K=4)]`, so the
    /// anticipated-local indices the assertions use are plant 0 → id=2, plant 1 → id=5.
    /// The bus's deficit segment is unbounded, so the LP is always feasible
    /// regardless of thermal capacity.
    fn build_system_two_anticipated() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let ant_id_k2 = EntityId(2);
        let thermal_ant_k2 = make_thermal(
            ant_id_k2,
            ThermalSpec {
                name: "T_ant_k2".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 50.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 5).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 500.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let ant_id_k4 = EntityId(5);
        let thermal_ant_k4 = make_thermal(
            ant_id_k4,
            ThermalSpec {
                name: "T_ant_k4".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 6).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 80.0,
                cost_per_mwh: 40.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(4)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
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
                ..Default::default()
            },
        );

        let n_stages = 6_usize;
        let calendar_anchor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let stage_dates = super::daily_stage_dates(calendar_anchor, n_stages, 31);
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_dates[i].0,
                        end_date: stage_dates[i].1,
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 744.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 20.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        let k_max: usize = 4;

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            }
        }

        fn default_hydro_penalties() -> HydroPenalties {
            HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            }
        }

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 3,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        for (t_idx, thermal) in [&thermal_ant_k2, &thermal_backup, &thermal_ant_k4]
            .into_iter()
            .enumerate()
        {
            for stage_idx in 0..n_stages {
                *bounds.thermal_bounds_mut(t_idx, stage_idx) = ThermalStageBounds {
                    cost_per_mwh: thermal.cost_per_mwh,
                };
                *bounds.thermal_block_base_mut(t_idx, stage_idx) = ThermalBlockBounds {
                    min_generation_mw: thermal.min_generation_mw,
                    max_generation_mw: thermal.max_generation_mw,
                };
            }
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Seed lengths force locked deliveries: plant 0 at stages 0..=1, plant 1 at
        // stages 0..=3 — sunk, so the lower bound prices only the residual load.
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: [
                super::windowed_commitments_daily(ant_id_k2, calendar_anchor, 31, &[60.0, 30.0]),
                super::windowed_commitments_daily(
                    ant_id_k4,
                    calendar_anchor,
                    31,
                    &[20.0, 25.0, 30.0, 35.0],
                ),
            ]
            .concat(),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant_k2, thermal_backup, thermal_ant_k4])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system_two_anticipated: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 12 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Integration test
    // ---------------------------------------------------------------------------

    #[test]
    fn test_two_anticipated_plants_k1_2_k2_4_convergence() {
        let ant_id_k2: u32 = 2;
        let ant_id_k4: u32 = 5;
        let backup_id: u32 = 4;
        assert!(ant_id_k2 < ant_id_k4);
        assert_ne!(backup_id, ant_id_k4);

        let system = build_system_two_anticipated();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 12, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let result = &outcome.result;
        assert_eq!(result.iterations, 12);

        let actual = result.final_lb;
        let expected = EXPECTED_LB;
        let rel_diff = if expected.abs() > f64::EPSILON {
            (actual - expected).abs() / expected.abs()
        } else {
            actual.abs()
        };
        assert!(
            rel_diff < 1e-6,
            "final_lb mismatch: {actual} vs {expected} (rel_diff={rel_diff}). \
         If intentional, update EXPECTED_LB."
        );

        // Convergence certificate. This fixture is a single deterministic path
        // (branching_factor 1, one forward pass), so the forward simulation is a
        // feasible policy whose cost bounds the optimum from above while the cuts
        // bound it from below: final_lb <= optimum <= final_ub. A closed gap
        // therefore certifies final_lb IS the extensive-form optimum, independent
        // of its magnitude — and covers the heterogeneous-lead ring (a K=2 and a
        // K=4 plant sharing one k_max=4 ring) the single-plant closed-form anchors
        // do not reach. final_ub_std == 0 confirms final_ub is that exact point
        // cost, not a Monte Carlo estimate, so the gap is a point certificate.
        let lb = result.final_lb;
        let ub = result.final_ub;
        let convergence_gap = (lb - ub).abs() / ub.abs().max(1.0);
        assert!(
            convergence_gap < 1e-9,
            "LB==UB convergence certificate failed: final_lb={lb}, final_ub={ub}, \
         gap={convergence_gap} (must be < 1e-9 for a converged deterministic run)."
        );
        assert!(
            result.final_ub_std < 1e-6,
            "single deterministic forward path must yield ~zero UB std; got {}",
            result.final_ub_std
        );

        let state = setup.stage_state();
        assert_eq!(state.n_anticipated, 2);
        assert_eq!(state.k_max, 4);

        let n_anticipated = state.n_anticipated;
        let k_max = state.k_max;
        let ant_start = state.commit_out.start;
        let ant_block_len = n_anticipated * k_max;

        let basis_cache = &result.basis_cache;
        assert_eq!(basis_cache.len(), 6);

        let s0 = basis_cache[0]
            .as_ref()
            .expect("stage 0 basis must be Some")
            .state_at_capture
            .as_slice();

        let ant_slice = &s0[ant_start..ant_start + ant_block_len];
        assert_eq!(ant_slice.len(), 8);
        for &v in ant_slice {
            assert!(v.is_finite(), "anticipated state must be finite");
        }

        let s1 = basis_cache[1]
            .as_ref()
            .expect("stage 1 basis must be Some")
            .state_at_capture
            .as_slice();
        let s2 = basis_cache[2]
            .as_ref()
            .expect("stage 2 basis must be Some")
            .state_at_capture
            .as_slice();

        // Plant 1 (K=4, id 5) holds its delivery-3 commitment in slot `3 mod 4 = 3`
        // across stages 0-2, maturing only at stage 3, so the SAME slot carries the
        // same value across the consecutive forward-outgoing captures bc[1] (stage
        // 0) and bc[2] (stage 1).
        let plant1 = 1_usize;
        let slot3_p1 = ant_start + 3 * n_anticipated + plant1;
        assert!(
            s1[slot3_p1].abs() > 1e-9,
            "carry check must exercise a non-zero held commitment; got {}",
            s1[slot3_p1]
        );
        assert!(
            (s1[slot3_p1] - s2[slot3_p1]).abs() < 1e-9,
            "commitment-hold carry violated: plant 1's slot-3 commitment must be held \
         in place across the forward transition (bc[1]={}, bc[2]={}), not shifted",
            s1[slot3_p1],
            s2[slot3_p1]
        );
    }
}

mod anticipated_simulation_ring_buffer {
    //! Regression: simulation advances the anticipated ring like the forward pass
    //! (`StateSpace::state_to_lp_column`). With the anticipated thermal cheaper than
    //! backup, the matured commitment equals the decision made `K` stages earlier,
    //!
    //! `anticipated_committed_mw(t = K) == anticipated_decision_mw(t = 0)`,
    //!
    //! not the seeded `past_anticipated_commitments` a broken ring would surface.

    use cobre_io::config::{SimulationSelection, TrainingSelection};
    use std::sync::mpsc;

    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage, InitialConditions,
        LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build a deterministic K-stage system: cheap anticipated thermal (id 2),
    /// expensive backup (id 4), 150 MW load, ring seeded with `past_commitments_mw`.
    ///
    /// The non-zero seed is intentional: constructing `System` directly via
    /// `SystemBuilder::new()` bypasses `cobre-io`'s semantic validation of
    /// `past_anticipated_commitments` (coverage tiling, commissioning-window
    /// checks) — those rules apply to JSON through `load_case`, not here.
    fn build_system(
        k: usize,
        past_commitments_mw: Vec<f64>,
        n_stages: usize,
    ) -> cobre_core::System {
        use chrono::NaiveDate;

        assert_eq!(
            past_commitments_mw.len(),
            k,
            "past_commitments_mw length must equal K (lead_stages)",
        );

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(2);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(k as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
                cost_per_mwh: 5000.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
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
                ..Default::default()
            },
        );

        let calendar_anchor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let stage_dates = super::daily_stage_dates(calendar_anchor, n_stages, 31);
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_dates[i].0,
                        end_date: stage_dates[i].1,
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 744.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 0.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 1.0,
                max_generation_mw: 1.0,
                ..Default::default()
            }
        }

        fn default_hydro_penalties() -> HydroPenalties {
            HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            }
        }

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: k,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 200.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        // ResolvedBounds::new applies one default to ALL thermals; without these
        // per-thermal overrides the cheap anticipated (index 0) and expensive backup
        // (index 1) are indistinguishable and decision_at(t) collapses to zero,
        // neutering the regression. The override must span the padding region
        // `[n_stages, n_stages + k_max)` too — `fill_anticipated_columns` reads the
        // delivery-stage axis there, so the decision column needs its cost out to K.
        let thermal_axis = n_stages + k;
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 100.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 200.0;
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: super::windowed_commitments_daily(
                anticipated_id,
                calendar_anchor,
                31,
                &past_commitments_mw,
            ),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    fn build_config(training_iters: u32) -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: training_iters,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 8,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// K=1 case of the module-doc invariant: stage-1 matured commitment equals the
    /// stage-0 decision, not the non-zero seed a missing ring carry-advance surfaces.
    /// (K=1 byte-stable across the shift-to-hold switchover: depth-1 ring, so hold and
    /// shift render identical rows; the fn name predates the switchover.)
    #[test]
    fn simulation_ring_buffer_shifts_anticipated_state_k1() {
        let k: usize = 1;
        let n_stages: usize = 5;
        // Seed (7 MW) has no relationship to load/bounds, so the LP will not pick it
        // as a decision — a seed equal to a plausible decision would let the buggy and
        // fixed paths agree and neuter the test.
        let seed: Vec<f64> = vec![7.0];

        let system = build_system(k, seed, n_stages);
        let config = build_config(50);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 50, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let sim_run = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            sim_run.costs.len(),
            1,
            "simulation must produce exactly one scenario cost",
        );
        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );

        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            n_stages,
            "scenario must contain one stage record per study stage",
        );

        let anticipated_thermal_id: i32 = 2;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };
        let committed_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_committed_mw)
        };

        // Under always-active fishing, stage-0 committed reads the seed at slot 0.
        let d0 = decision_at(0)
            .expect("anticipated_decision_mw must exist at stage 0 (t + K < n_stages)");
        let c0 = committed_at(0)
            .expect("anticipated_committed_mw must be Some at stage 0 under always-active fishing");
        assert!(
            (c0 - 7.0).abs() < 1e-6,
            "committed_at(0) must equal the K=1 seed window[0].value_mw=7.0; got {c0}",
        );
        assert!(
            d0.abs() > 1e-6,
            "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
        );

        let c1 = committed_at(1).expect("committed at stage 1 must exist (K <= 1)");
        assert!(
            (c1 - d0).abs() < 1e-6,
            "REGRESSION (commitment-hold carry): stage 1 committed ({c1}) must equal \
         stage 0 decision ({d0}). On the buggy code path the ring buffer was \
         not advanced in simulation, so stage 1's fishing RHS carried the \
         residual `seed - d_0` (negative when d_0 > seed) and the LP was \
         infeasible. With the carry advance, the fishing RHS = d_0 and gt_anticipated \
         at stage 1 saturates at d_0 (cost zeroed at delivery).",
        );
    }

    /// K=2 case of the module-doc invariant: the two pre-horizon stages fish the seed
    /// at their maturing modular slots, and from stage 2 the matured commitment equals
    /// the decision made K=2 stages earlier (held at its slot until maturity), never the
    /// seed, never zero.
    #[test]
    fn simulation_commitment_hold_carries_anticipated_state_k2() {
        let k: usize = 2;
        let n_stages: usize = 6;
        // Seed slots are distinct from d_0 (which saturates near the thermal max of
        // 100), so neither slot can coincide with a decision and mask a missing carry.
        let seed: Vec<f64> = vec![50.0, 30.0];

        let system = build_system(k, seed, n_stages);
        let config = build_config(10);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 10, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let _sim_run = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");
        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(scenario_results.len(), 1);
        let scenario = &scenario_results[0];
        assert_eq!(scenario.stages.len(), n_stages);

        let anticipated_thermal_id: i32 = 2;
        let decision_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_decision_mw)
        };
        let committed_at = |t: usize| -> Option<f64> {
            scenario.stages[t]
                .thermals
                .iter()
                .find(|th| th.thermal_id == anticipated_thermal_id)
                .and_then(|th| th.anticipated_committed_mw)
        };

        // Pre-horizon: stage 0 fishes its maturing slot `0 mod 2 = 0` (seed[0]);
        // stage 1 fishes its maturing slot `1 mod 2 = 1` (seed[1]) — each seed is
        // held at its own modular slot, never shifted.
        let c0 = committed_at(0)
            .expect("committed_at(0) must be Some under always-active fishing with K=2");
        assert!(
            (c0 - 50.0).abs() < 1e-6,
            "committed_at(0) must equal K=2 seed window[0].value_mw=50.0; got {c0}",
        );
        let c1 = committed_at(1)
            .expect("committed_at(1) must be Some under always-active fishing with K=2");
        assert!(
            (c1 - 30.0).abs() < 1e-6,
            "committed_at(1) must equal K=2 seed window[1].value_mw=30.0 (held at slot 1 mod 2); got {c1}",
        );

        let d0 = decision_at(0).expect("decision at stage 0 must exist (0 + K < n_stages)");

        assert!(
            d0.abs() > 1e-6,
            "stage-0 decision must be non-zero for the test to be meaningful; got {d0}",
        );

        let c2 = committed_at(2).expect("committed at stage 2 must exist (K <= 2)");
        assert!(
            (c2 - d0).abs() < 1e-6,
            "REGRESSION (commitment-hold carry, K=2): stage 2 committed ({c2}) must \
         equal stage 0 decision ({d0}). On the buggy code path the ring \
         buffer was not advanced in simulation, so stage 2's fishing RHS \
         carried a stale residual instead of the d_0 that was held at its \
         modular slot from stage 0 until it matures at stage 2.",
        );
    }
}

mod anticipated_generic_constraint_e2e {
    //! End-to-end integration tests for generic constraints referencing
    //! `anticipated_decision(N)`: one pins that a binding cap raises the lower bound,
    //! one that the validator rejects the reference on a non-anticipated thermal.

    use cobre_io::config::TrainingSelection;
    use std::path::Path;

    use chrono::NaiveDate;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ConstraintExpression,
        ContractBlockBounds, EntityId, GenericConstraint, HydroBlockBounds, HydroPenalties,
        HydroStageBounds, InitialConditions, LineBlockBounds, LineStagePenalties, LinearTerm,
        NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds,
        ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties, SlackConfig,
        SystemBuilder, ThermalBlockBounds, ThermalStageBounds, VariableRef,
        entities::{bus::DeficitSegment, thermal::AnticipatedConfig},
        scenario::LoadModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // Fixture parameters
    // ---------------------------------------------------------------------------

    const N_STAGES: usize = 4;
    const K_MAX: usize = 2;
    /// 1 hour keeps cost magnitudes small and hand-derivable.
    const BLOCK_HOURS: f64 = 1.0;
    const LOAD_MW: f64 = 50.0;
    const ANT_MAX_MW: f64 = 100.0;
    /// Must be less than BACKUP_COST (anticipated is the cheap plant).
    const ANT_COST: f64 = 10.0;
    const BACKUP_MAX_MW: f64 = 200.0;
    const BACKUP_COST: f64 = 100.0;
    /// Well above BACKUP_COST so deficit is never optimal.
    const DEFICIT_COST: f64 = 1000.0;
    /// Strictly below LOAD_MW, so the unconstrained optimum (d_ant = LOAD_MW) is
    /// infeasible under the constraint.
    const CONSTRAINT_BOUND_MW: f64 = 20.0;

    const ANT_THERMAL_ID: EntityId = EntityId(2);
    /// Non-anticipated — the AC rejection target.
    const BACKUP_THERMAL_ID: EntityId = EntityId(3);
    const BUS_ID: EntityId = EntityId(1);

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the `N_STAGES`-stage no-hydro system (1 bus, 1 anticipated thermal, 1
    /// backup thermal) with optional generic constraints + resolved stage bounds.
    /// Always feasible: the backup thermal alone covers `LOAD_MW`.
    fn build_system(
        generic_constraints: Vec<GenericConstraint>,
        generic_bounds: ResolvedGenericConstraintBounds,
    ) -> cobre_core::System {
        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: DEFICIT_COST,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANT_THERMAL_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: ANT_MAX_MW,
                cost_per_mwh: ANT_COST,
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_THERMAL_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 4).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_MAX_MW,
                cost_per_mwh: BACKUP_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let calendar_anchor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let stage_dates = super::daily_stage_dates(calendar_anchor, N_STAGES, 1);
        let stages: Vec<Stage> = (0..N_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_dates[i].0,
                        end_date: stage_dates[i].1,
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: BLOCK_HOURS,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: false,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..N_STAGES)
            .map(|i| LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: LOAD_MW,
                std_mw: 0.0,
            })
            .collect();

        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        // SystemBuilder sorts thermals by EntityId ascending, so thermal_idx 0 = id=2
        // (anticipated) and thermal_idx 1 = id=3 (backup); these indices feed
        // thermal_bounds_mut below. The thermal stage axis runs N_STAGES + K_MAX to
        // cover delivery-stage lookups in fill_anticipated_columns.
        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(0, s) = ThermalStageBounds {
                cost_per_mwh: ANT_COST,
            };
            *bounds.thermal_block_base_mut(0, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: ANT_MAX_MW,
            };
            *bounds.thermal_bounds_mut(1, s) = ThermalStageBounds {
                cost_per_mwh: BACKUP_COST,
            };
            *bounds.thermal_block_base_mut(1, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_MAX_MW,
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: super::windowed_commitments_daily(
                ANT_THERMAL_ID,
                calendar_anchor,
                1,
                &[0.0, 0.0],
            ),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let mut builder = SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions);

        if !generic_constraints.is_empty() {
            builder = builder
                .generic_constraints(generic_constraints)
                .resolved_generic_bounds(generic_bounds);
        }

        builder.build().expect("build_system: valid")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    /// Build a minimal [`Config`] for this fixture.
    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::None,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Constrained training lower-bound is strictly worse than unconstrained
    // ---------------------------------------------------------------------------

    /// A 4-stage, K=2, no-hydro fixture with `anticipated_decision(2) <= 20.0`.
    ///
    /// K=2, N=4: decisions active at stages 0 (delivers at 2) and 1 (delivers at 3),
    /// zero past commitments. Unconstrained optimum is `d0 = d1 = 50 MW`, eliminating
    /// backup at the delivery stages. Capped at 20 MW, backup must cover the remaining
    /// 30 MW at stages 2 and 3 at BACKUP_COST — so constrained LB exceeds baseline.
    #[test]
    fn anticipated_decision_constraint_raises_lb() {
        let constraint = GenericConstraint {
            id: EntityId(1),
            name: "cap_ant_decision".to_string(),
            description: Some(format!(
                "Cap anticipated commitment for T_ant at {CONSTRAINT_BOUND_MW} MW"
            )),
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::AnticipatedDecision {
                        thermal_id: ANT_THERMAL_ID,
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        };

        let config = build_config();
        let comm = StubComm;

        let id_map: std::collections::HashMap<i32, usize> =
            [(1_i32, 0_usize)].into_iter().collect();
        let raw_bounds = (0..N_STAGES as i32)
            .map(|stage_id| {
                (
                    1_i32,
                    stage_id,
                    None::<i32>,
                    None,
                    Some(CONSTRAINT_BOUND_MW),
                )
            })
            .collect::<Vec<_>>();
        let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, raw_bounds.into_iter());

        let constrained_system = build_system(vec![constraint], generic_bounds);
        let mut constrained_setup = build_setup_in_code(constrained_system, &config);
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let constrained_outcome = constrained_setup
            .train(&mut solver, &comm, 10, ActiveSolver::new, None, None)
            .expect("constrained train must not return Err");

        assert!(
            constrained_outcome.error.is_none(),
            "constrained training error: {:?}",
            constrained_outcome.error
        );
        let constrained_lb = constrained_outcome.result.final_lb;

        let baseline_system = build_system(vec![], ResolvedGenericConstraintBounds::empty());
        let mut baseline_setup = build_setup_in_code(baseline_system, &config);
        let mut baseline_solver = ActiveSolver::new().expect("ActiveSolver::new baseline");

        let baseline_outcome = baseline_setup
            .train(
                &mut baseline_solver,
                &comm,
                10,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("baseline train must not return Err");

        assert!(
            baseline_outcome.error.is_none(),
            "baseline training error: {:?}",
            baseline_outcome.error
        );
        let baseline_lb = baseline_outcome.result.final_lb;

        assert!(
            constrained_lb > baseline_lb,
            "constrained LB ({constrained_lb:.6}) must be strictly greater than \
         baseline LB ({baseline_lb:.6}) — constraint is not binding"
        );

        let lb_delta = constrained_lb - baseline_lb;
        assert!(
            lb_delta > 1.0,
            "LB delta ({lb_delta:.6}) is too small to confirm economically meaningful binding"
        );
    }

    // ---------------------------------------------------------------------------
    // Semantic validator rejects anticipated_decision on non-anticipated thermal
    // ---------------------------------------------------------------------------

    /// A case with a generic constraint `anticipated_decision(3)` where id=3 is NOT
    /// anticipated. `cobre_io::validate_case` must reject it via
    /// `check_anticipated_decision_target_is_anticipated` — the error contains
    /// "not an anticipated thermal".
    #[test]
    fn anticipated_decision_on_non_anticipated_thermal_rejected_by_validator() {
        use std::fs;

        let tmp = tempfile::tempdir().expect("tempdir");
        let case_dir = tmp.path();

        // ── system/ ───────────────────────────────────────────────────────────────
        let system_dir = case_dir.join("system");
        fs::create_dir_all(&system_dir).expect("create system dir");

        fs::write(
        system_dir.join("buses.json"),
        r#"{"buses":[{"id":1,"name":"B1","operational_start_date":"2024-01-02","deficit_segments":[{"depth_mw":null,"cost":1000.0}]}]}"#,
    )
    .expect("write buses.json");

        fs::write(system_dir.join("hydros.json"), r#"{"hydros":[]}"#).expect("write hydros.json");

        fs::write(system_dir.join("lines.json"), r#"{"lines":[]}"#).expect("write lines.json");

        fs::write(
            system_dir.join("thermals.json"),
            r#"{
  "thermals": [
    {
      "id": 2,
      "name": "T_ant",
      "operational_start_date": "2024-01-03",
      "bus_id": 1,
      "generation": { "min_mw": 0.0, "max_mw": 100.0 },
      "cost_per_mwh": 10.0,
      "anticipated_config": { "lead_stages": 2 }
    },
    {
      "id": 3,
      "name": "T_backup",
      "operational_start_date": "2024-01-04",
      "bus_id": 1,
      "generation": { "min_mw": 0.0, "max_mw": 200.0 },
      "cost_per_mwh": 100.0
    }
  ]
}"#,
        )
        .expect("write thermals.json");

        // ── constraints/ ─────────────────────────────────────────────────────────
        let constraints_dir = case_dir.join("constraints");
        fs::create_dir_all(&constraints_dir).expect("create constraints dir");

        fs::write(
            constraints_dir.join("generic_constraints.json"),
            r#"{
  "constraints": [
    {
      "id": 1,
      "name": "bad_constraint",
      "expression": "anticipated_decision(3)",
      "slack": { "enabled": false }
    }
  ]
}"#,
        )
        .expect("write generic_constraints.json");

        // The pipeline requires a constraint-bounds parquet whenever
        // generic_constraints.json is present.
        write_constraint_bounds_parquet(
            &constraints_dir.join("generic_constraint_bounds.parquet"),
            1,    // constraint_id
            0,    // stage_id
            25.0, // bound_upper
        )
        .expect("write generic_constraint_bounds.parquet");

        // ── stages.json ───────────────────────────────────────────────────────────
        fs::write(
            case_dir.join("stages.json"),
            r#"{
  "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.0 },
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 744 }],
      "num_openings": 1
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 672 }],
      "num_openings": 1
    },
    {
      "id": 2,
      "start_date": "2024-03-01",
      "end_date": "2024-04-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 744 }],
      "num_openings": 1
    },
    {
      "id": 3,
      "start_date": "2024-04-01",
      "end_date": "2024-05-01",
      "blocks": [{ "id": 0, "name": "S", "hours": 720 }],
      "num_openings": 1
    }
  ]
}"#,
        )
        .expect("write stages.json");

        // ── initial_conditions.json ───────────────────────────────────────────────
        // Anticipated thermal (id=2, K=2) requires past_anticipated_commitments.
        fs::write(
            case_dir.join("initial_conditions.json"),
            r#"{
  "storage": [],
  "filling_storage": [],
  "past_anticipated_commitments": [
    { "thermal_id": 2, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 },
    { "thermal_id": 2, "start_date": "2024-02-01", "end_date": "2024-03-01", "value_mw": 0.0 }
  ]
}"#,
        )
        .expect("write initial_conditions.json");

        // ── penalties.json ────────────────────────────────────────────────────────
        fs::write(
            case_dir.join("penalties.json"),
            r#"{
  "bus": {
    "deficit_segments": [{ "depth_mw": null, "cost": 1000.0 }],
    "excess_cost": 0.01
  },
  "line": { "exchange_cost": 0.01 },
  "hydro": {
    "spillage_cost": 0.01,
    "turbined_cost": 0.01,
    "diversion_cost": 0.01,
    "storage_violation_below_cost": 10000.0,
    "filling_target_violation_cost": 10000.0,
    "turbined_violation_below_cost": 10000.0,
    "outflow_violation_below_cost": 10000.0,
    "outflow_violation_above_cost": 10000.0,
    "generation_violation_below_cost": 10000.0,
    "evaporation_violation_cost": 10000.0,
    "water_withdrawal_violation_cost": 10000.0
  },
  "non_controllable_source": { "curtailment_cost": 0.005 }
}"#,
        )
        .expect("write penalties.json");

        // ── config.json ───────────────────────────────────────────────────────────
        fs::write(
            case_dir.join("config.json"),
            r#"{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 1 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 2 }]
  },
  "simulation": { "enabled": false },
  "modeling": { "inflow_non_negativity": { "method": "none" } }
}"#,
        )
        .expect("write config.json");

        let result = cobre_io::validate_case(case_dir);

        assert!(
            result.is_err(),
            "validate_case should fail when anticipated_decision references a non-anticipated thermal"
        );

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("not an anticipated thermal"),
            "error message must contain 'not an anticipated thermal', got: {err_msg}"
        );
    }

    // ---------------------------------------------------------------------------
    // Helper: write a minimal `generic_constraint_bounds.parquet`
    // ---------------------------------------------------------------------------

    /// Write a single-row constraint-bounds parquet; `block_id` is null (all blocks).
    fn write_constraint_bounds_parquet(
        path: &Path,
        constraint_id: i32,
        stage_id: i32,
        bound_upper: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use arrow::array::{Float64Array, Int32Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("constraint_id", DataType::Int32, false),
            Field::new("stage_id", DataType::Int32, false),
            Field::new("block_id", DataType::Int32, true),
            Field::new("bound_lower", DataType::Float64, true),
            Field::new("bound_upper", DataType::Float64, true),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![constraint_id])),
                Arc::new(Int32Array::from(vec![stage_id])),
                Arc::new(Int32Array::new_null(1)),
                Arc::new(Float64Array::new_null(1)),
                Arc::new(Float64Array::from(vec![bound_upper])),
            ],
        )?;

        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }
}

mod d34_anticipated_varying_blocks_shape {
    //! Case-shape assertions for the D34 deterministic fixture.
    //!
    //! `parity_hash_d34` exercises the `anticipated_state_out` column's independence
    //! from `n_blks` only if D34 combines two shapes no other shipped case does:
    //!
    //! 1. an anticipated thermal whose `lead_stages` `K_i` matures **strictly inside**
    //!    the horizon (`stage + K_i < n_stages`) at an interior delivery stage, and
    //! 2. a per-stage-varying block schedule, with the maturation stage landing on an
    //!    off-stage-0 block count.
    //!
    //! This test pins both, so an edit that flattens the block schedule, drops the
    //! anticipated thermal, or pushes maturation outside the horizon is caught here
    //! rather than silently degrading `parity_hash_d34` to a no-op.

    use std::path::Path;

    fn d34_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic")
            .join("d34-anticipated-varying-blocks")
    }

    #[test]
    fn d34_combines_anticipated_thermal_with_non_uniform_block_schedule() {
        let system = cobre_io::load_case(&d34_dir()).expect("D34 case must load");

        let block_counts: Vec<usize> = system.stages().iter().map(|s| s.blocks.len()).collect();
        assert_eq!(
            block_counts,
            vec![1, 3, 2],
            "D34 must ship the d33-style non-uniform [1, 3, 2] block schedule; \
         got {block_counts:?}"
        );
        let stage0_blocks = block_counts[0];
        assert!(
            block_counts.iter().any(|&c| c != stage0_blocks),
            "at least one interior stage must differ from stage 0's block count, \
         or the case cannot exercise an off-stage-0 maturation"
        );

        let n_stages = system.stages().len();

        let anticipated: Vec<_> = system
            .thermals()
            .iter()
            .filter_map(|t| {
                t.anticipated_config
                    .and_then(|cfg| cfg.lead_stages().map(|k| (t.id, k)))
            })
            .collect();
        assert!(
            !anticipated.is_empty(),
            "D34 must declare at least one anticipated thermal"
        );

        let exercises_off_stage0_maturation = anticipated.iter().any(|&(_, k_i)| {
            let k = k_i as usize;
            (0..n_stages).any(|decision_stage| {
                let delivery_stage = decision_stage + k;
                delivery_stage < n_stages && block_counts[delivery_stage] != stage0_blocks
            })
        });
        assert!(
            exercises_off_stage0_maturation,
            "no anticipated commitment matures strictly inside the horizon at an \
         interior stage whose block count differs from stage 0's; \
         anticipated K_i = {anticipated:?}, block_counts = {block_counts:?}, \
         n_stages = {n_stages} — the case would not exercise the relocated \
         anticipated_state_out column"
        );

        // Pin the K=1 maturation coordinates: decision at stage 0 delivers at stage 1
        // (3 blocks), decision at stage 1 delivers at stage 2 (2 blocks) — both off
        // stage 0's single-block stride.
        let (_, k1) = anticipated
            .iter()
            .find(|&&(_, k)| k == 1)
            .expect("D34's anticipated thermal uses lead_stages = 1");
        assert_eq!(*k1, 1);
        assert_eq!(block_counts[1], 3, "stage-1 delivery lands on 3 blocks");
        assert_eq!(block_counts[2], 2, "stage-2 delivery lands on 2 blocks");
    }
}

mod d37_anticipated_commissioning_simulation {
    //! Anticipated-thermal commissioning-window gating must reach the simulation
    //! output, and warm-start must survive the dormancy boundary.
    //!
    //! T1 (`K=2`, cheap, `max 150 MW`) carries window `[entry=2, exit=4)` over a
    //! 6-stage horizon (per-stage block schedule 1/3/2/3/1/2). The decision gate
    //! (`StateSpace::is_anticipated_decision_active`) conjoins the strict horizon
    //! clause (`t + K < n_stages`) with the operation-window clause keyed on the
    //! DELIVERY stage `id(t + K)`:
    //!
    //! - `t = 0` → delivers at id 2 ∈ `[2, 4)` → ACTIVE (pre-entry, priced at `entry − K`),
    //! - `t = 1` → delivers at id 3 ∈ `[2, 4)` → ACTIVE,
    //! - `t = 2, 3` → delivers at id 4, 5 ∉ `[2, 4)` → INACTIVE,
    //! - `t = 4, 5` → `t + K ≥ 6` → horizon-INACTIVE.
    //!
    //! The parity baseline hashes hydro storage/water/cuts/convergence, NOT thermal
    //! generation, the decision, or the ring — so a gating bug (T1 running out of
    //! window, an undelivered pre-entry commitment, an un-drained ring) could still
    //! hash-match. This test exercises those paths through train + simulate.

    use cobre_io::config::SimulationSelection;
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::Config;
    use cobre_io::config::SimulationConfig;
    use cobre_sddp::simulation::SimulationThermalResult;
    use cobre_sddp::{
        SimulationScenarioResult, SolverStatsDelta, StudySetup, hydro_models::prepare_hydro_models,
        setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    fn case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d37-anticipated-commissioning")
    }

    fn build_setup(case_dir: &Path) -> (StudySetup, Config) {
        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        // The shipped case disables simulation (parity trains only); enable one
        // deterministic scenario so the thermal extraction paths run.
        config.simulation = SimulationConfig {
            enabled: true,
            io_channel_capacity: 8,
            selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
            ..SimulationConfig::default()
        };

        let system = cobre_io::load_case(case_dir).expect("load_case must succeed");
        let prepare_result = prepare_stochastic(
            system,
            case_dir,
            &config,
            42,
            &ScenarioSource::default(),
            None,
        )
        .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
            .expect("StudySetup::new");
        (setup, config)
    }

    const T1_ID: i32 = 1;
    const ENTRY: usize = 2;
    const EXIT: usize = 4;
    const K: usize = 2;
    const N_STAGES: usize = 6;
    const TOL: f64 = 1e-6;

    fn t1_at(scenario: &SimulationScenarioResult, stage: usize) -> &SimulationThermalResult {
        scenario.stages[stage]
            .thermals
            .iter()
            .find(|t| t.thermal_id == T1_ID)
            .unwrap_or_else(|| panic!("T1 (id={T1_ID}) missing at stage {stage}"))
    }

    #[test]
    fn anticipated_commissioning_window_gates_simulation_output() {
        let dir = case_dir();
        let (mut setup, _config) = build_setup(&dir);

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "training must run at least one iteration",
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect(
                "simulate must not return Err (a windowed anticipated thermal must stay feasible)",
            );

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");
        assert_eq!(scenario_results.len(), 1, "one deterministic scenario");
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            N_STAGES,
            "one record per study stage"
        );

        for stage in 0..N_STAGES {
            if !(ENTRY..EXIT).contains(&stage) {
                let g = t1_at(scenario, stage).generation_mw;
                assert!(
                    g.abs() <= TOL,
                    "stage {stage} is outside the operation window [{ENTRY}, {EXIT}): \
                 T1 generation must be 0, got {g}",
                );
            }
        }

        // Generation at the first operating stage `entry` proves the pre-entry
        // decision at `entry − K` delivered: the decision column at `entry` itself
        // delivers at `entry + K`, outside the window, so this generation can only
        // come from the commitment placed K stages earlier.
        let g_entry = t1_at(scenario, ENTRY).generation_mw;
        assert!(
            g_entry > TOL,
            "T1 must generate at the first operating stage {ENTRY} (the pre-entry \
         decision at stage {} delivered here); got {g_entry}",
            ENTRY - K,
        );

        // The ring drains within K stages after exit: no in-window decision is made
        // after the last in-window delivery (stage EXIT-1), so by stage EXIT + K - 1
        // the buffer has drained all residual commitments and committed MW reads 0.
        let drain_stage = EXIT + K - 1;
        let committed_drain = t1_at(scenario, drain_stage)
            .anticipated_committed_mw
            .expect("anticipated thermal must report committed MW");
        assert!(
            committed_drain.abs() <= TOL,
            "the ring buffer must drain to 0 within K={K} stages after exit={EXIT}: \
         committed MW at stage {drain_stage} must be 0, got {committed_drain}",
        );

        for stage in ENTRY..N_STAGES {
            let decision = t1_at(scenario, stage).anticipated_decision_mw;
            assert!(
                decision.is_none(),
                "T1 decision must be inactive (None) at stage {stage} (delivery at \
             {} is outside [{ENTRY}, {EXIT}) or beyond the horizon); got {decision:?}",
                stage + K,
            );
        }
        for stage in 0..ENTRY {
            let decision = t1_at(scenario, stage).anticipated_decision_mw;
            assert!(
                decision.is_some(),
                "T1 decision must be active at the pre-entry stage {stage} (delivers \
             at {} ∈ [{ENTRY}, {EXIT})); got None",
                stage + K,
            );
        }
    }

    /// Warm-start regression across the dormancy boundary: the commissioning window
    /// toggles which stages emit an `anticipated_state_out_def` row, changing per-stage
    /// row/column counts at the entry/exit boundaries. `reconstruct_basis` matches cut
    /// rows by `CutPool` slot identity, never absolute index, so the change must stay
    /// transparent — zero `basis_consistency_failures`.
    #[test]
    fn anticipated_commissioning_warm_start_zero_basis_rejections() {
        let dir = case_dir();
        let (mut setup, _config) = build_setup(&dir);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );
        let result = &outcome.result;
        // reconstruct_basis runs on every iteration after the first, so the regression
        // needs >= 2 iterations to exercise it across the dormancy boundary.
        assert!(
            result.iterations >= 2,
            "warm-start regression needs >= 2 iterations to exercise reconstruct_basis \
         on iteration 2+, got {}",
            result.iterations,
        );

        let total_rejections =
            SolverStatsDelta::aggregate(result.solver_stats_log.iter().map(|entry| &entry.delta))
                .basis_consistency_failures;
        assert_eq!(
            total_rejections, 0,
            "windowed anticipated warm-start: expected 0 basis rejections across the \
         dormancy boundary, got {total_rejections} (reconstruct_basis must match \
         cut rows by CutPool slot identity, not by absolute index)",
        );
    }
}

mod anticipated_commitment_at_cap {
    //! Regression: the must-generate fishing equality (`Σ_b h_b·gen_b =
    //! H·commitment`) pins a delivery-stage anticipated generation column to the
    //! ring-carried commitment with no slack, so feasibility requires the
    //! commitment lie within the delivery stage's own `[min_gen, max_gen]`.
    //!
    //! Two seeds carried through a `K = 2` in-LP ring pin the contract: at the
    //! cap (feasible unrelaxed — no patch, so parity holds) and a hair over it
    //! by less than the solver's own primal feasibility tolerance (feasible only
    //! because the incoming state is canonicalized within the delivery stage's
    //! own admissible box before it is ever pinned).
    //!
    //! Seeding exactly at the cap alone is NOT sufficient coverage: zero drift
    //! exercises no canonicalization, so an at-cap-only suite stays green while
    //! studies whose commitments drift abort on a false infeasibility.

    use cobre_core::entities::{
        bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, EntityId,
        HydroBlockBounds, HydroPenalties, HydroStageBounds, HydroStorage, InitialConditions,
        LineBlockBounds, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    const CAP_MW: f64 = 100.0;
    const AT_CAP_SEED_MW: f64 = CAP_MW;
    /// `1e-12` (relative) over the cap: above the ring's own round-trip noise
    /// (`~1e-16`), so the fail-without/pass-with split is deterministic.
    const DRIFTED_SEED_MW: f64 = CAP_MW * (1.0 + 1e-12);

    fn build_system(seed_mw: f64) -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 500.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let anticipated_id = EntityId(2);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: CAP_MW,
                cost_per_mwh: 50.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(4),
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: CAP_MW,
                cost_per_mwh: 500.0,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let hydro = make_hydro(
            EntityId(3),
            HydroSpec {
                name: "H1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
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
                ..Default::default()
            },
        );

        let n_stages = 4_usize;
        let calendar_anchor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let stage_dates = super::daily_stage_dates(calendar_anchor, n_stages, 31);
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: stage_dates[i].0,
                        end_date: stage_dates[i].1,
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: 744.0,
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: true,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| InflowModel {
                hydro_id: EntityId(3),
                stage_id: i as i32,
                mean_m3s: 80.0,
                std_m3s: 0.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 80.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            }
        }

        let bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 2,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: CAP_MW,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
                    spillage_cost: 0.01,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 500.0,
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
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(3),
                value_hm3: 100.0,
            }],
            filling_storage: vec![],
            // Two pre-study commitment windows at `seed_mw`: the stage-0 delivery
            // matures immediately, the stage-1 delivery is held at its own K=2 modular
            // slot until it matures at stage 1.
            past_anticipated_commitments: super::windowed_commitments_daily(
                anticipated_id,
                calendar_anchor,
                31,
                &[seed_mw, seed_mw],
            ),
            recent_observations: vec![],
            past_defluences: vec![],
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal_ant, thermal_backup])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .build()
            .expect("build_system: valid")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    #[test]
    fn anticipated_commitment_at_cap_survives_ring_carry() {
        let system = build_system(AT_CAP_SEED_MW);
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 4, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "anticipated commitment at its delivery cap must stay feasible through \
             the ring carry, got training error: {:?}",
            outcome.error
        );
    }

    /// A commitment carried a hair past its delivery cap trains, not
    /// `Infeasible`, over a drift the solver itself would report as feasible.
    #[test]
    fn anticipated_commitment_drifted_over_cap_is_absorbed() {
        let system = build_system(DRIFTED_SEED_MW);
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 4, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "a commitment carried a sub-tolerance hair past its delivery cap must \
             train: the overshoot is numerical drift, not an over-commitment, and \
             refusing it aborts training on a physically meaningless quantity. Got: {:?}",
            outcome.error
        );
    }
}

mod faithful_resolution {
    //! The headline resolution regression: a reference-deck-shaped case (one
    //! anticipated plant, `LeadStages(7)` over four study stages plus a
    //! seven-stage post-study calendar) asserting each carried post-study
    //! delivery resolves to its real calendar window and duration — structural
    //! only, read off the built setup through the two public terminal
    //! accessors with no training and no solve.

    use chrono::NaiveDate;
    use cobre_core::entities::bus::DeficitSegment;
    use cobre_core::entities::thermal::AnticipatedConfig;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PostStudyStage, PostStudyStages,
        PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
        ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, SimulationSelection,
        StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
        UpperBoundEvaluationConfig,
    };
    use cobre_io::{ENTITY_SLOT_DATE_SENTINEL, StateFamily, decode_slot_date};
    use cobre_sddp::study_horizon_end;

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    const BUS_ID: EntityId = EntityId(1);
    const ANTICIPATED_ID: EntityId = EntityId(2);
    const N_STUDY_STAGES: usize = 4;
    const LEAD_STAGES: u32 = 7;
    const N_POST_STUDY_STAGES: usize = 7;

    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("hardcoded fixture date is valid")
    }

    /// Study-stage boundaries (five dates for four stages): three operative
    /// weeks (168 h each) then one month (648 h) — the reference-deck shape.
    fn study_boundaries() -> [NaiveDate; N_STUDY_STAGES + 1] {
        [
            date(2026, 3, 14),
            date(2026, 3, 21),
            date(2026, 3, 28),
            date(2026, 4, 4),
            date(2026, 5, 1),
        ]
    }
    const STUDY_DURATION_HOURS: [f64; N_STUDY_STAGES] = [168.0, 168.0, 168.0, 648.0];

    fn study_stages() -> Vec<Stage> {
        let bounds = study_boundaries();
        (0..N_STUDY_STAGES)
            .map(|i| {
                make_stage(
                    i,
                    StageSpec {
                        start_date: bounds[i],
                        end_date: bounds[i + 1],
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: STUDY_DURATION_HOURS[i],
                        }],
                        block_mode: BlockMode::Parallel,
                        state_config: StageStateConfig {
                            storage: false,
                            inflow_lags: false,
                        },
                        risk_config: StageRiskConfig::Expectation,
                        scenario_config: ScenarioSourceConfig {
                            branching_factor: 1,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    /// Post-study calendar (seven stages, delivery indices 4–10): the fixed
    /// window (24 h stub, then Sem 8/9) followed by the four class-3 windows
    /// (Sem 10/11/12, then the June remainder) study stages 0–3 decide into.
    const POST_STUDY_STARTS: [(i32, u32, u32); N_POST_STUDY_STAGES] = [
        (2026, 5, 1),
        (2026, 5, 2),
        (2026, 5, 9),
        (2026, 5, 16),
        (2026, 5, 23),
        (2026, 5, 30),
        (2026, 6, 6),
    ];
    const POST_STUDY_DURATION_HOURS: [f64; N_POST_STUDY_STAGES] =
        [24.0, 168.0, 168.0, 168.0, 168.0, 168.0, 576.0];

    fn post_study_stages() -> PostStudyStages {
        let stages = POST_STUDY_STARTS
            .iter()
            .zip(POST_STUDY_DURATION_HOURS)
            .map(|(&(y, m, d), duration_hours)| PostStudyStage {
                start_date: date(y, m, d),
                duration_hours,
            })
            .collect();
        PostStudyStages {
            stages,
            thermal_bounds: vec![],
        }
    }

    fn bus() -> cobre_core::entities::bus::Bus {
        make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: date(2026, 3, 14),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 100_000.0,
                }],
                excess_cost: 0.0,
            },
        )
    }

    fn anticipated_thermal() -> cobre_core::entities::thermal::Thermal {
        make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: date(2026, 3, 14),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(LEAD_STAGES)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        )
    }

    fn load_models() -> Vec<cobre_core::scenario::LoadModel> {
        (0..N_STUDY_STAGES)
            .map(|i| cobre_core::scenario::LoadModel {
                bus_id: BUS_ID,
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect()
    }

    /// Thermal bounds/cost over the full delivery axis (`n_stages + lead`), the
    /// same extended stride `fill_anticipated_columns` reads a delivery-anchored
    /// decision's bounds from.
    fn bounds() -> ResolvedBounds {
        let k_max = LEAD_STAGES as usize;
        let mut b = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STUDY_STAGES,
                k_max,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds::default(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        for s in 0..(N_STUDY_STAGES + k_max) {
            *b.thermal_bounds_mut(0, s) = ThermalStageBounds { cost_per_mwh: 10.0 };
            *b.thermal_block_base_mut(0, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 500.0,
            };
        }
        b
    }

    fn penalties() -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STUDY_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroPenalties {
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
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// One `past_anticipated_commitments` window per class-2 (in-study,
    /// deliveries 0–3) study stage, tiling the study calendar.
    const CLASS2_SEEDS_MW: [f64; N_STUDY_STAGES] = [100.0, 150.0, 200.0, 250.0];

    fn class2_seed_windows() -> Vec<AnticipatedCommitmentHistory> {
        let bounds = study_boundaries();
        (0..N_STUDY_STAGES)
            .map(|i| AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                start_date: bounds[i],
                end_date: bounds[i + 1],
                value_mw: CLASS2_SEEDS_MW[i],
            })
            .collect()
    }

    /// One `past_anticipated_commitments` window per class-4 (fixed
    /// post-horizon, deliveries 4–6) post-study stage: an explicit 0 MW stub at
    /// delivery 4, non-zero at deliveries 5 and 6.
    const CLASS4_SEEDS_MW: [f64; 3] = [0.0, 60.0, 80.0];

    fn class4_seed_windows() -> Vec<AnticipatedCommitmentHistory> {
        (0..3)
            .map(|i| {
                let (y, m, d) = POST_STUDY_STARTS[i];
                let (ny, nm, nd) = POST_STUDY_STARTS[i + 1];
                AnticipatedCommitmentHistory {
                    thermal_id: ANTICIPATED_ID,
                    start_date: date(y, m, d),
                    end_date: date(ny, nm, nd),
                    value_mw: CLASS4_SEEDS_MW[i],
                }
            })
            .collect()
    }

    fn build_system() -> cobre_core::System {
        let mut past_anticipated_commitments = class2_seed_windows();
        past_anticipated_commitments.extend(class4_seed_windows());
        SystemBuilder::new()
            .buses(vec![bus()])
            .thermals(vec![anticipated_thermal()])
            .stages(study_stages())
            .load_models(load_models())
            .bounds(bounds())
            .penalties(penalties())
            .initial_conditions(InitialConditions {
                storage: vec![],
                filling_storage: vec![],
                past_anticipated_commitments,
                recent_observations: vec![],
                past_defluences: vec![],
            })
            .post_study_stages(Some(post_study_stages()))
            .build()
            .expect("build_system: valid system")
    }

    fn build_config() -> Config {
        Config {
            schema: None,
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },
                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 8,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// The reference calendar's real post-study delivery windows: each carried
    /// post-study-targeted ring slot's real `(start, end)` window, ascending by
    /// start date.
    fn expected_class3_intervals() -> [(NaiveDate, NaiveDate); 4] {
        [
            (date(2026, 5, 16), date(2026, 5, 23)),
            (date(2026, 5, 23), date(2026, 5, 30)),
            (date(2026, 5, 30), date(2026, 6, 6)),
            (date(2026, 6, 6), date(2026, 6, 30)),
        ]
    }

    #[test]
    fn study_stage_deliveries_resolve_to_their_real_post_study_calendar_window() {
        let config = build_config();
        let setup = build_setup_in_code(build_system(), &config);
        let system = build_system();

        let manifest = setup.build_terminal_entity_manifest(&system);
        let boundary_date =
            study_horizon_end(&system).expect("a non-negative stage exists in this study");

        // A `LeadStages` plant's ring `k_max` is clamped up to its own declared
        // lead rather than the resolver's true occupancy, so the ring's own
        // slot order does not match ascending delivery-target order — compare
        // the carried set by value (sorted by start), never by vector position.
        let mut carried: Vec<(usize, (NaiveDate, NaiveDate))> = manifest
            .iter()
            .enumerate()
            .filter_map(|(j, slot)| {
                if slot.entity_type != StateFamily::AnticipatedThermalState.code()
                    || slot.interval_start == ENTITY_SLOT_DATE_SENTINEL
                {
                    return None;
                }
                let start = decode_slot_date(slot.interval_start)
                    .expect("a live interval_start must decode");
                let end =
                    decode_slot_date(slot.interval_end).expect("a live interval_end must decode");
                (end > boundary_date).then_some((j, (start, end)))
            })
            .collect();
        carried.sort_by_key(|&(_, (start, _))| start);

        let expected = expected_class3_intervals();
        assert_eq!(
            carried.len(),
            expected.len(),
            "expected exactly {} carried post-study-targeted ring slots, got {:?}",
            expected.len(),
            carried,
        );

        for (&(slot, window), expected_window) in carried.iter().zip(expected.iter()) {
            assert_eq!(
                window, *expected_window,
                "a carried delivery interval must equal the literal reference window \
                 {expected_window:?}, got {window:?} (slot {slot})",
            );
        }
    }

    #[test]
    fn carried_delivery_durations_equal_their_declared_post_study_stage_span() {
        let config = build_config();
        let setup = build_setup_in_code(build_system(), &config);
        let system = build_system();

        let manifest = setup.build_terminal_entity_manifest(&system);
        let boundary_date =
            study_horizon_end(&system).expect("a non-negative stage exists in this study");

        let mut carried: Vec<(NaiveDate, NaiveDate)> = manifest
            .iter()
            .filter_map(|slot| {
                if slot.entity_type != StateFamily::AnticipatedThermalState.code()
                    || slot.interval_start == ENTITY_SLOT_DATE_SENTINEL
                {
                    return None;
                }
                let start = decode_slot_date(slot.interval_start)
                    .expect("a live interval_start must decode");
                let end =
                    decode_slot_date(slot.interval_end).expect("a live interval_end must decode");
                (end > boundary_date).then_some((start, end))
            })
            .collect();
        carried.sort_by_key(|&(start, _)| start);

        let expected_days = [7_i64, 7, 7, 24];
        assert_eq!(carried.len(), expected_days.len());
        for ((start, end), &days) in carried.iter().zip(expected_days.iter()) {
            assert_eq!(
                (*end - *start).num_days(),
                days,
                "carried window {start}..{end} must span its literal declared \
                 post-study stage width of {days} days",
            );
        }
    }
}
