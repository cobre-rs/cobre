//! Stochastic preprocessing pipeline: PAR estimation, opening tree loading, and stochastic context construction.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use cobre_core::{
    EntityId, System,
    scenario::{SamplingScheme, ScenarioSource},
};
use cobre_stochastic::{OpeningTreeInputs, StochasticContext, context::OpeningTree};

use crate::{EstimationPath, EstimationReport, SddpError};

// ---------------------------------------------------------------------------
// PrepareStochasticResult + prepare_stochastic
// ---------------------------------------------------------------------------

/// Result of the stochastic preprocessing pipeline.
///
/// Bundles outputs from [`prepare_stochastic`].
#[derive(Debug)]
pub struct PrepareStochasticResult {
    /// Updated system with estimated PAR models (if estimation ran).
    pub system: System,
    /// Built stochastic context, ready to pass to [`crate::setup::StudySetup::new`].
    pub stochastic: StochasticContext,
    /// Estimation report (`Some` if `inflow_history.parquet` was present and
    /// `inflow_seasonal_stats.parquet` was absent, triggering auto-estimation).
    pub estimation_report: Option<EstimationReport>,
    /// Which of the 7 estimation path rows was taken during preprocessing.
    pub estimation_path: EstimationPath,
}

/// Load and validate a user-supplied opening tree from `scenarios/noise_openings.parquet`.
///
/// Returns `Ok(None)` if the file is absent.
///
/// # Errors
///
/// Returns [`SddpError::Io`] if the file exists but cannot be read or fails validation.
fn load_user_opening_tree_inner(
    case_dir: &Path,
    system: &System,
) -> Result<Option<OpeningTree>, SddpError> {
    let mut ctx = cobre_io::ValidationContext::new();
    let manifest = cobre_io::validate_structure(case_dir, &mut ctx);

    if !manifest.scenarios_noise_openings_parquet {
        return Ok(None);
    }

    let path = case_dir.join("scenarios").join("noise_openings.parquet");

    let rows = cobre_io::scenarios::load_noise_openings(Some(&path))?;

    let n_hydros = system.hydros().len();
    let mut load_bus_ids: Vec<EntityId> = system
        .load_models()
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    load_bus_ids.sort_unstable_by_key(|id| id.0);
    load_bus_ids.dedup();
    let n_load_buses = load_bus_ids.len();
    let expected_dim = n_hydros + n_load_buses;

    let expected_stages = system.stages().iter().filter(|s| s.id >= 0).count();
    let mut openings_by_stage: BTreeMap<i32, BTreeSet<u32>> = BTreeMap::new();
    for row in &rows {
        openings_by_stage
            .entry(row.stage_id)
            .or_default()
            .insert(row.opening_index);
    }
    let openings_per_stage: Vec<usize> = openings_by_stage.values().map(BTreeSet::len).collect();

    cobre_io::scenarios::validate_noise_openings(
        &rows,
        expected_dim,
        expected_stages,
        &openings_per_stage,
    )?;

    let tree = cobre_io::scenarios::assemble_opening_tree(rows, expected_dim);
    Ok(Some(tree))
}

/// Build NCS entity factor entries from `System::resolved_ncs_factors()`.
///
/// Converts the dense 3D table into `(entity_id, stage_id, block_pairs)` tuples
/// for `PrecomputedNormal::build`.
#[must_use]
pub fn build_ncs_factor_entries(
    system: &System,
) -> Vec<(
    cobre_core::EntityId,
    i32,
    Vec<cobre_stochastic::normal::precompute::BlockFactorPair>,
)> {
    use cobre_stochastic::normal::precompute::BlockFactorPair;
    use std::collections::BTreeSet;

    // Collect NCS entity IDs that have model entries.
    let stochastic_ncs: BTreeSet<cobre_core::EntityId> =
        system.ncs_models().iter().map(|m| m.ncs_id).collect();

    if stochastic_ncs.is_empty() {
        return Vec::new();
    }

    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let ncs_ids: Vec<cobre_core::EntityId> = system
        .non_controllable_sources()
        .iter()
        .map(|n| n.id)
        .collect();

    let mut entries = Vec::new();
    for (ncs_idx, ncs_id) in ncs_ids.iter().enumerate() {
        if !stochastic_ncs.contains(ncs_id) {
            continue;
        }
        for (stage_idx, stage) in study_stages.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let block_pairs: Vec<BlockFactorPair> = stage
                .blocks
                .iter()
                .enumerate()
                .map(|(block_idx, _)| {
                    let factor = system
                        .resolved_ncs_factors()
                        .factor(ncs_idx, stage_idx, block_idx);
                    // block_idx is a small count (< 1000 in practice); fits in i32.
                    (block_idx as i32, factor)
                })
                .collect();
            entries.push((*ncs_id, stage.id, block_pairs));
        }
    }
    entries
}

/// Load `scenarios/load_factors.json` from the case directory, returning an
/// empty vec when the file is absent. This is consumed by the stochastic
/// context builder for per-block noise scaling.
///
/// # Errors
///
/// Returns [`SddpError`] if the file exists but cannot be read or parsed.
pub fn load_load_factors_for_stochastic(
    case_dir: &Path,
) -> Result<Vec<cobre_io::scenarios::LoadFactorEntry>, SddpError> {
    let path = case_dir.join("scenarios").join("load_factors.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    cobre_io::scenarios::parse_load_factors(&path).map_err(SddpError::from)
}

/// Prepare the stochastic pipeline: PAR estimation, opening tree loading, and context build.
///
/// Estimates PAR from history if `inflow_history.parquet` is present and
/// `inflow_seasonal_stats.parquet` is absent. Loads user opening tree from
/// `scenarios/noise_openings.parquet` if present. Builds [`StochasticContext`].
///
/// **MPI note**: Call on rank 0 only; broadcast the opening tree to other ranks.
///
/// # Errors
///
/// Returns [`SddpError::Io`] on file read/parse/validation failure,
/// or [`SddpError::Stochastic`] on PAR/decomposition failure.
#[allow(clippy::too_many_lines)]
pub fn prepare_stochastic(
    system: System,
    case_dir: &Path,
    config: &cobre_io::Config,
    seed: u64,
    training_source: &ScenarioSource,
) -> Result<PrepareStochasticResult, SddpError> {
    let (system, estimation_report, estimation_path) =
        crate::estimation::estimate_from_history(system, case_dir, config)?;

    let user_opening_tree = load_user_opening_tree_inner(case_dir, &system)?;

    // Load block-level load factors (optional). When present, these scale the
    // stochastic noise realization per block, mirroring how the LP builder
    // scales the deterministic load balance RHS.
    let load_factor_entries = load_load_factors_for_stochastic(case_dir)?;

    // Convert LoadFactorEntry -> Vec<BlockFactorPair> per entry. The pairs
    // vec must outlive the entity_factor_entries references.
    let block_pairs: Vec<Vec<cobre_stochastic::normal::precompute::BlockFactorPair>> =
        load_factor_entries
            .iter()
            .map(|e| {
                e.block_factors
                    .iter()
                    .map(|bf| (bf.block_id, bf.factor))
                    .collect()
            })
            .collect();

    let entity_factor_entries: Vec<cobre_stochastic::normal::precompute::EntityFactorEntry<'_>> =
        load_factor_entries
            .iter()
            .zip(block_pairs.iter())
            .map(|(e, pairs)| (e.bus_id, e.stage_id, pairs.as_slice()))
            .collect();

    // Build NCS block factor entries from ResolvedNcsFactors, mirroring the
    // load factor conversion above. NCS entities consume their block factors
    // from the resolved NCS factors table.
    let ncs_factor_entries = build_ncs_factor_entries(&system);
    let ncs_entity_factor_entries: Vec<
        cobre_stochastic::normal::precompute::EntityFactorEntry<'_>,
    > = ncs_factor_entries
        .iter()
        .map(|(ncs_id, stage_id, pairs)| (*ncs_id, *stage_id, pairs.as_slice()))
        .collect();

    // Build a HistoricalScenarioLibrary for the opening tree when any study
    // stage uses NoiseMethod::HistoricalResiduals. This must be done before
    // build_stochastic_context because generate_opening_tree consumes the
    // library reference. The forward-pass Historical library (built in
    // StudySetup::new) is separate and remains unchanged.
    let opening_tree_library = {
        use cobre_core::temporal::NoiseMethod;

        let needs_historical_tree = system.stages().iter().any(|s| {
            s.id >= 0 && s.scenario_config.noise_method == NoiseMethod::HistoricalResiduals
        });

        if needs_historical_tree {
            let study_stages: Vec<_> = system
                .stages()
                .iter()
                .filter(|s| s.id >= 0)
                .cloned()
                .collect();
            let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
            // Build PAR cache directly — before the stochastic context exists.
            let par = cobre_stochastic::PrecomputedPar::build(
                system.inflow_models(),
                &study_stages,
                &hydro_ids,
            )?;
            let max_order = par.max_order();
            let user_pool = training_source.historical_years.as_ref();
            let window_years = cobre_stochastic::discover_historical_windows(
                system.inflow_history(),
                &hydro_ids,
                &study_stages,
                max_order,
                user_pool,
                system.policy_graph().season_map.as_ref(),
                1, // forward_passes not relevant for opening tree windows
            )?;
            let mut lib = cobre_stochastic::HistoricalScenarioLibrary::new(
                window_years.len(),
                study_stages.len(),
                hydro_ids.len(),
                max_order,
                window_years.clone(),
            );
            cobre_stochastic::standardize_historical_windows(
                &mut lib,
                system.inflow_history(),
                &hydro_ids,
                &study_stages,
                &par,
                &window_years,
                system.policy_graph().season_map.as_ref(),
            );
            Some(lib)
        } else {
            None
        }
    };

    // Compute per-stage external scenario counts for opening tree clamping.
    //
    // When any entity class uses External sampling, the external library is
    // padded to a uniform scenario count after loading. The opening tree
    // generator must clamp per-stage openings to the pre-padding raw count to
    // avoid redundant LP solves for stages with fewer distinct scenarios.
    //
    // The raw count for inflow is `rows_per_stage / n_hydros`.
    // For load it is `rows_per_stage / n_buses`.
    // For ncs it is `rows_per_stage / n_ncs_entities`.
    // When multiple classes use External, the element-wise minimum is used.
    let external_scenario_counts: Option<Vec<usize>> = {
        let study_stages: Vec<_> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        let n_stages = study_stages.len();

        let inflow_counts: Option<Vec<usize>> =
            if training_source.inflow_scheme == SamplingScheme::External && n_stages > 0 {
                let external_rows = system.external_scenarios();
                let n_hydros = system.hydros().len();
                let mut rows_per_stage = vec![0usize; n_stages];
                #[allow(clippy::cast_sign_loss)]
                for row in external_rows {
                    let s = row.stage_id as usize;
                    if s < n_stages {
                        rows_per_stage[s] += 1;
                    }
                }
                Some(if n_hydros > 0 {
                    rows_per_stage.iter().map(|&r| r / n_hydros).collect()
                } else {
                    vec![0usize; n_stages]
                })
            } else {
                None
            };

        let load_counts: Option<Vec<usize>> =
            if training_source.load_scheme == SamplingScheme::External && n_stages > 0 {
                let external_rows = system.external_load_scenarios();
                let mut bus_ids: Vec<EntityId> = system
                    .load_models()
                    .iter()
                    .filter(|m| m.std_mw > 0.0)
                    .map(|m| m.bus_id)
                    .collect();
                bus_ids.sort_unstable_by_key(|id| id.0);
                bus_ids.dedup();
                let n_buses = bus_ids.len();
                let mut rows_per_stage = vec![0usize; n_stages];
                #[allow(clippy::cast_sign_loss)]
                for row in external_rows {
                    let s = row.stage_id as usize;
                    if s < n_stages {
                        rows_per_stage[s] += 1;
                    }
                }
                Some(if n_buses > 0 {
                    rows_per_stage.iter().map(|&r| r / n_buses).collect()
                } else {
                    vec![0usize; n_stages]
                })
            } else {
                None
            };

        let ncs_counts: Option<Vec<usize>> =
            if training_source.ncs_scheme == SamplingScheme::External && n_stages > 0 {
                let external_rows = system.external_ncs_scenarios();
                let mut ncs_ids: Vec<EntityId> =
                    system.ncs_models().iter().map(|m| m.ncs_id).collect();
                ncs_ids.sort_unstable_by_key(|id| id.0);
                ncs_ids.dedup();
                let n_ncs = ncs_ids.len();
                let mut rows_per_stage = vec![0usize; n_stages];
                #[allow(clippy::cast_sign_loss)]
                for row in external_rows {
                    let s = row.stage_id as usize;
                    if s < n_stages {
                        rows_per_stage[s] += 1;
                    }
                }
                Some(if n_ncs > 0 {
                    rows_per_stage.iter().map(|&r| r / n_ncs).collect()
                } else {
                    vec![0usize; n_stages]
                })
            } else {
                None
            };

        // Combine class counts via element-wise minimum.
        match (inflow_counts, load_counts, ncs_counts) {
            (None, None, None) => None,
            (Some(a), None, None) => Some(a),
            (None, Some(b), None) => Some(b),
            (None, None, Some(c)) => Some(c),
            (Some(a), Some(b), None) => {
                Some(a.iter().zip(b.iter()).map(|(&x, &y)| x.min(y)).collect())
            }
            (Some(a), None, Some(c)) => {
                Some(a.iter().zip(c.iter()).map(|(&x, &y)| x.min(y)).collect())
            }
            (None, Some(b), Some(c)) => {
                Some(b.iter().zip(c.iter()).map(|(&x, &y)| x.min(y)).collect())
            }
            (Some(a), Some(b), Some(c)) => Some(
                a.iter()
                    .zip(b.iter())
                    .zip(c.iter())
                    .map(|((&x, &y), &z)| x.min(y).min(z))
                    .collect(),
            ),
        }
    };

    // Compute noise group IDs for Pattern C noise sharing (Epic 2).
    // Groups stages with the same (season_id, year) so weekly stages within
    // the same monthly bucket share noise draws in the opening tree.
    // For uniform monthly studies each stage has a unique group ID, so no
    // sharing is triggered and the opening tree is identical to the pre-noise-
    // sharing baseline (modulo the seed domain change from ticket-003).
    let opening_tree_noise_group_ids: Vec<u32> = {
        let study_stages: Vec<_> = system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect();
        crate::lag_transition::precompute_noise_groups(&study_stages)
    };

    let forward_seed = training_source.seed.map(i64::unsigned_abs);
    let stochastic = cobre_stochastic::build_stochastic_context(
        &system,
        seed,
        forward_seed,
        &entity_factor_entries,
        &ncs_entity_factor_entries,
        OpeningTreeInputs {
            user_tree: user_opening_tree,
            historical_library: opening_tree_library.as_ref(),
            external_scenario_counts,
            noise_group_ids: Some(opening_tree_noise_group_ids),
        },
        cobre_stochastic::ClassSchemes {
            inflow: Some(training_source.inflow_scheme),
            load: Some(training_source.load_scheme),
            ncs: Some(training_source.ncs_scheme),
        },
    )?;

    Ok(PrepareStochasticResult {
        system,
        stochastic,
        estimation_report,
        estimation_path,
    })
}
