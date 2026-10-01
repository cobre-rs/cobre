//! Pre-materialised lookup table: `(parameter_id, stage_idx, block_idx)` → `f64`.
//!
//! ## Basis-cache invariance
//!
//! Resolved values freeze into [`StageTemplate`](cobre_solver::StageTemplate)
//! entries at construction and are never rebuilt; the hot-path solver patches
//! only row bounds, so LP matrix coefficients stay identical across iterations
//! and the warm-start basis cache needs no invalidation. A future change that
//! mutates parameter values across iterations (adaptive updates) breaks this and
//! must rebuild or invalidate the affected `StageTemplate` entries.

use std::collections::HashMap;
use std::fmt;

use cobre_core::{ComputedParameter, EntityId, Hydro, ParameterKind, ScalarParameter, StageId};
use thiserror::Error;

use crate::energy_conversion::{EnergyConversionSet, HydroEnergyProductivityOverride};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`build_resolved_parameters`].
#[derive(Debug, Error, PartialEq)]
pub enum ResolvedParametersError {
    /// A `PerStage` parameter vector has the wrong length.
    #[error("parameter '{name}': PerStage vector has {got} values but n_stages = {expected}")]
    PerStageLengthMismatch {
        /// Parameter name from the source record.
        name: String,
        /// Required length (`n_stages`).
        expected: usize,
        /// Actual vector length.
        got: usize,
    },

    /// A `Seasonal` parameter has no entry for the season required by a stage.
    #[error(
        "parameter '{name}': no seasonal value for season_id={season_id} (needed by stage {stage_idx})"
    )]
    MissingSeason {
        /// Parameter name from the source record.
        name: String,
        /// The season ID that had no matching entry.
        season_id: i32,
        /// The stage index that needed the season.
        stage_idx: usize,
    },

    /// A `Computed` parameter references a hydro plant that is not in `hydros`.
    #[error("parameter '{name}': computed variant references unknown hydro_id={hydro_id:?}")]
    UnknownHydro {
        /// Parameter name from the source record.
        name: String,
        /// The hydro entity ID that was not found.
        hydro_id: EntityId,
    },

    /// A `SpecificProductivity` variant could not be resolved because neither
    /// the override table nor the hydro entity record has a value.
    #[error(
        "parameter '{name}': no specific productivity (`ρ_esp`) available for hydro_id={hydro_id:?}"
    )]
    MissingSpecificProductivity {
        /// Parameter name from the source record.
        name: String,
        /// The hydro entity ID whose `ρ_esp` could not be found.
        hydro_id: EntityId,
    },

    /// A `PerStageBlock` parameter's triples do not tile every `(stage, block)`
    /// cell of the study grid exactly once.
    #[error("parameter '{name}': PerStageBlock cell (stage={stage}, block={block}) {issue}")]
    PerStageBlockCoverage {
        /// Parameter name from the source record.
        name: String,
        /// The offending stage index (raw triple value).
        stage: i32,
        /// The offending block index (raw triple value).
        block: i32,
        /// Why the cell failed coverage.
        issue: PerStageBlockCoverageIssue,
    },
}

/// Why a [`ParameterKind::PerStageBlock`] cell failed the exact-coverage check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerStageBlockCoverageIssue {
    /// No triple supplied a value for this cell.
    Missing,
    /// More than one triple targeted this in-range cell.
    Duplicated,
    /// The triple lies outside `0..n_stages × 0..blocks(stage)`.
    OutOfRange,
}

impl fmt::Display for PerStageBlockCoverageIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Missing => "is not covered by any triple",
            Self::Duplicated => "is covered by more than one triple",
            Self::OutOfRange => "lies outside the study's (stage, block) grid",
        })
    }
}

// ---------------------------------------------------------------------------
// ResolvedParameters
// ---------------------------------------------------------------------------

/// Dense lookup table mapping `(parameter_id, stage_idx, block_idx)` → `f64`.
///
/// `id_to_slot` is sorted ascending by key — for declaration-order invariance
/// (a `HashMap` would not be deterministic) and `O(log n)` binary-search lookup.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolvedParameters {
    /// Jagged storage indexed slot → `stage_idx` → block. A stage-/block-invariant
    /// parameter stores a length-1 inner vector per stage that broadcasts to every
    /// block; a `PerStageBlock` parameter stores one entry per block of that stage.
    pub per_param: Vec<Vec<Vec<f64>>>,
    /// Maps `EntityId.0` of the parameter to its slot in `per_param`.
    pub id_to_slot: Vec<(i32, usize)>,
    /// Resolved `modeling.cost_scale_factor` (default `1_000_000.0`): the
    /// divisor applied to every non-theta objective coefficient at template
    /// build time, multiplied back at every cost-domain reporting boundary.
    pub cost_scale_factor: f64,
}

impl Default for ResolvedParameters {
    /// `cost_scale_factor` defaults to [`crate::DEFAULT_COST_SCALE_FACTOR`],
    /// never `#[derive(Default)]`'s `0.0` — a template built against a
    /// zero factor divides every non-theta objective coefficient by zero.
    fn default() -> Self {
        Self {
            per_param: Vec::new(),
            id_to_slot: Vec::new(),
            cost_scale_factor: crate::DEFAULT_COST_SCALE_FACTOR,
        }
    }
}

impl ResolvedParameters {
    /// Return the resolved `f64` value for `(id, stage_idx, block_idx)`.
    ///
    /// A length-1 inner block vector broadcasts to every `block_idx`; otherwise
    /// `block_idx` indexes it. On a miss returns `0.0` — mirroring the LP-build
    /// site sentinel — and `debug_assert`s in debug builds, with a distinct
    /// message per miss class (unknown id, out-of-range stage, out-of-range block).
    #[must_use]
    pub fn get(&self, id: EntityId, stage_idx: usize, block_idx: usize) -> f64 {
        let Ok(pos) = self.id_to_slot.binary_search_by_key(&id.0, |(k, _)| *k) else {
            debug_assert!(
                false,
                "ResolvedParameters miss: unknown id={id:?} (stage={stage_idx}, block={block_idx})"
            );
            return 0.0;
        };
        let slot = self.id_to_slot[pos].1;
        let Some(blocks) = self.per_param[slot].get(stage_idx) else {
            debug_assert!(
                false,
                "ResolvedParameters miss: id={id:?}, stage={stage_idx} out of range (n_stages={})",
                self.per_param[slot].len()
            );
            return 0.0;
        };
        if blocks.len() == 1 {
            return blocks[0];
        }
        if block_idx < blocks.len() {
            blocks[block_idx]
        } else {
            debug_assert!(
                false,
                "ResolvedParameters block miss: id={id:?}, stage={stage_idx}, block={block_idx} out of range (n_blocks={})",
                blocks.len()
            );
            0.0
        }
    }

    /// Whether parameter `id` resolves to per-block values at any stage — i.e. some
    /// stage stores more than one block value. A stage-/block-invariant parameter
    /// (length-1 inner everywhere) and an unknown `id` both return `false`.
    #[must_use]
    pub fn is_block_varying(&self, id: EntityId) -> bool {
        self.id_to_slot
            .binary_search_by_key(&id.0, |(k, _)| *k)
            .is_ok_and(|pos| {
                let slot = self.id_to_slot[pos].1;
                self.per_param[slot].iter().any(|blocks| blocks.len() > 1)
            })
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Build a [`ResolvedParameters`] table from assembled scalar parameters and
/// energy-conversion data.
///
/// # Errors
///
/// Returns a [`ResolvedParametersError`] on the first failure encountered:
/// - [`PerStageLengthMismatch`](ResolvedParametersError::PerStageLengthMismatch)
///   when a `PerStage` vector length ≠ `n_stages`.
/// - [`MissingSeason`](ResolvedParametersError::MissingSeason) when a
///   `Seasonal` entry is absent for a required season.
/// - [`UnknownHydro`](ResolvedParametersError::UnknownHydro) when a `Computed`
///   variant references a `hydro_id` not present in `hydros`.
/// - [`MissingSpecificProductivity`](ResolvedParametersError::MissingSpecificProductivity)
///   when neither the override table nor the hydro entity record has a `ρ_esp`
///   value.
///
/// The constructor never panics; `n_stages == 0` produces an empty table.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ParameterKind, ScalarParameter, StageId};
/// use cobre_sddp::energy_conversion::{
///     EnergyConversionSet, HydroEnergyProductivityOverride,
/// };
/// use cobre_sddp::resolved_parameters::build_resolved_parameters;
///
/// let params = vec![ScalarParameter {
///     id: EntityId(1),
///     name: "constant_coeff".to_string(),
///     kind: ParameterKind::Constant { value: 3.6 },
/// }];
/// let ec = EnergyConversionSet::new(vec![], vec![], &[], 4);
/// let overrides = HydroEnergyProductivityOverride::default();
/// let stage_ids = [StageId(0), StageId(1), StageId(2), StageId(3)];
///
/// let table = build_resolved_parameters(
///     &params, &ec, &overrides, &[], &[0, 0, 1, 1], &stage_ids, &[1, 1, 1, 1],
///     1_000_000.0,
/// )
///     .unwrap();
///
/// // A block-invariant kind broadcasts to every block.
/// assert!((table.get(EntityId(1), 0, 0) - 3.6).abs() < 1e-12);
/// assert!((table.get(EntityId(1), 3, 0) - 3.6).abs() < 1e-12);
/// ```
pub fn build_resolved_parameters(
    parameters: &[ScalarParameter],
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    stage_to_season: &[i32],
    stage_ids: &[StageId],
    stage_block_counts: &[usize],
    cost_scale_factor: f64,
) -> Result<ResolvedParameters, ResolvedParametersError> {
    debug_assert_eq!(
        stage_block_counts.len(),
        stage_ids.len(),
        "stage_block_counts must carry one block count per study stage"
    );
    let hydro_index: HashMap<EntityId, usize> = hydros
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.id, idx))
        .collect();
    let stage_axis = StageAxis {
        ids: stage_ids,
        to_season: stage_to_season,
        block_counts: stage_block_counts,
    };

    let mut per_param: Vec<Vec<Vec<f64>>> = Vec::with_capacity(parameters.len());
    let mut id_to_slot: Vec<(i32, usize)> = Vec::with_capacity(parameters.len());

    for (slot, param) in parameters.iter().enumerate() {
        let values = resolve_kind(
            &param.kind,
            &param.name,
            &stage_axis,
            energy_conversion,
            override_table,
            hydros,
            &hydro_index,
        )?;
        per_param.push(values);
        id_to_slot.push((param.id.0, slot));
    }

    // Uniqueness is enforced upstream (the JSON reader); the debug_assert makes
    // a duplicate observable at the resolver boundary.
    id_to_slot.sort_by_key(|(k, _)| *k);
    debug_assert!(
        id_to_slot.windows(2).all(|w| w[0].0 != w[1].0),
        "duplicate EntityId.0 values in parameters: {:?}",
        id_to_slot
            .windows(2)
            .filter(|w| w[0].0 == w[1].0)
            .map(|w| w[0].0)
            .collect::<Vec<_>>()
    );

    Ok(ResolvedParameters {
        per_param,
        id_to_slot,
        cost_scale_factor,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Per-stage inputs shared by [`resolve_kind`]'s match arms: the domain
/// `StageId`, season id, and block count for each study stage. `n_stages` is
/// always `stage_ids.len()` — never stored separately, so it cannot drift
/// from the slice it counts.
struct StageAxis<'a> {
    ids: &'a [StageId],
    to_season: &'a [i32],
    block_counts: &'a [usize],
}

/// Resolve a single [`ParameterKind`] into the jagged `stage → block` storage.
///
/// The four stage-/block-invariant kinds each produce a length-1 inner block
/// vector per stage (it broadcasts to every block in [`ResolvedParameters::get`]);
/// `PerStageBlock` produces one entry per block of each stage.
fn resolve_kind(
    kind: &ParameterKind,
    name: &str,
    stage_axis: &StageAxis<'_>,
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
) -> Result<Vec<Vec<f64>>, ResolvedParametersError> {
    let n_stages = stage_axis.ids.len();
    match kind {
        ParameterKind::Constant { value: c } => Ok(vec![vec![*c]; n_stages]),

        ParameterKind::PerStage { values: v } => {
            if v.len() != n_stages {
                return Err(ResolvedParametersError::PerStageLengthMismatch {
                    name: name.to_string(),
                    expected: n_stages,
                    got: v.len(),
                });
            }
            Ok(v.iter().map(|&x| vec![x]).collect())
        }

        ParameterKind::Seasonal { values: pairs } => {
            let mut values = Vec::with_capacity(n_stages);
            for (t, &season_id) in stage_axis.to_season.iter().enumerate() {
                // `pairs` is sorted ascending by season_id (invariant from
                // `ParameterKind::new_seasonal`).
                match pairs.binary_search_by_key(&season_id, |(k, _)| *k) {
                    Ok(pos) => values.push(vec![pairs[pos].1]),
                    Err(_) => {
                        return Err(ResolvedParametersError::MissingSeason {
                            name: name.to_string(),
                            season_id,
                            stage_idx: t,
                        });
                    }
                }
            }
            Ok(values)
        }

        ParameterKind::Computed { computed_spec: cp } => {
            let per_stage = resolve_computed(
                *cp,
                name,
                stage_axis.ids,
                energy_conversion,
                override_table,
                hydros,
                hydro_index,
            )?;
            Ok(per_stage.into_iter().map(|x| vec![x]).collect())
        }

        ParameterKind::PerStageBlock { values } => {
            resolve_per_stage_block(values, name, n_stages, stage_axis.block_counts)
        }
    }
}

/// Resolve a [`ParameterKind::PerStageBlock`] into per-`(stage, block)` values.
///
/// The triples must tile `0..n_stages × 0..stage_block_counts[stage]` exactly
/// once: a gap, a duplicate, or an out-of-range cell (no sensible default for a
/// missing cell) returns a
/// [`PerStageBlockCoverage`](ResolvedParametersError::PerStageBlockCoverage)
/// naming the parameter and the offending cell.
fn resolve_per_stage_block(
    values: &[(i32, i32, f64)],
    name: &str,
    n_stages: usize,
    stage_block_counts: &[usize],
) -> Result<Vec<Vec<f64>>, ResolvedParametersError> {
    let mut grid: Vec<Vec<Option<f64>>> = (0..n_stages)
        .map(|s| vec![None; stage_block_counts[s]])
        .collect();

    for &(stage, block, value) in values {
        let out_of_range = || ResolvedParametersError::PerStageBlockCoverage {
            name: name.to_string(),
            stage,
            block,
            issue: PerStageBlockCoverageIssue::OutOfRange,
        };
        let Ok(s) = usize::try_from(stage) else {
            return Err(out_of_range());
        };
        if s >= n_stages {
            return Err(out_of_range());
        }
        let Ok(b) = usize::try_from(block) else {
            return Err(out_of_range());
        };
        if b >= stage_block_counts[s] {
            return Err(out_of_range());
        }
        if grid[s][b].is_some() {
            return Err(ResolvedParametersError::PerStageBlockCoverage {
                name: name.to_string(),
                stage,
                block,
                issue: PerStageBlockCoverageIssue::Duplicated,
            });
        }
        grid[s][b] = Some(value);
    }

    let mut resolved: Vec<Vec<f64>> = Vec::with_capacity(n_stages);
    for (s, row) in grid.into_iter().enumerate() {
        let mut inner: Vec<f64> = Vec::with_capacity(row.len());
        for (b, cell) in row.into_iter().enumerate() {
            let Some(v) = cell else {
                return Err(ResolvedParametersError::PerStageBlockCoverage {
                    name: name.to_string(),
                    stage: i32::try_from(s).unwrap_or(i32::MAX),
                    block: i32::try_from(b).unwrap_or(i32::MAX),
                    issue: PerStageBlockCoverageIssue::Missing,
                });
            };
            inner.push(v);
        }
        resolved.push(inner);
    }
    Ok(resolved)
}

/// Resolve a [`ComputedParameter`] into a `Vec<f64>` of length `stage_ids.len()`.
fn resolve_computed(
    cp: ComputedParameter,
    name: &str,
    stage_ids: &[StageId],
    energy_conversion: &EnergyConversionSet,
    override_table: &HydroEnergyProductivityOverride,
    hydros: &[Hydro],
    hydro_index: &HashMap<EntityId, usize>,
) -> Result<Vec<f64>, ResolvedParametersError> {
    let hydro_id = match cp {
        ComputedParameter::EquivalentProductivity { hydro_id }
        | ComputedParameter::AccumulatedProductivity { hydro_id }
        | ComputedParameter::ReferenceVolume { hydro_id }
        | ComputedParameter::ReferenceTurbine { hydro_id }
        | ComputedParameter::MinStorage { hydro_id }
        | ComputedParameter::MaxStorage { hydro_id }
        | ComputedParameter::SpecificProductivity { hydro_id }
        | ComputedParameter::IntegratedEquivalentProductivity { hydro_id }
        | ComputedParameter::IntegratedAccumulatedProductivity { hydro_id }
        | ComputedParameter::MaxStoredEnergy { hydro_id } => hydro_id,
    };

    let hydro_idx = hydro_index.get(&hydro_id).copied().ok_or_else(|| {
        ResolvedParametersError::UnknownHydro {
            name: name.to_string(),
            hydro_id,
        }
    })?;

    let hydro = &hydros[hydro_idx];

    // SpecificProductivity is resolved per-stage below, not replicated once: it may
    // carry stage-specific overrides.
    match cp {
        ComputedParameter::MinStorage { .. } => {
            return Ok(vec![hydro.min_storage_hm3; stage_ids.len()]);
        }
        ComputedParameter::MaxStorage { .. } => {
            return Ok(vec![hydro.max_storage_hm3; stage_ids.len()]);
        }
        _ => {}
    }

    // Stage-varying resolution for the remaining five variants.
    let mut values = Vec::with_capacity(stage_ids.len());
    for (t, &stage_id) in stage_ids.iter().enumerate() {
        let value = match cp {
            ComputedParameter::EquivalentProductivity { .. } => {
                energy_conversion
                    .conversion(hydro_idx, t)
                    .equivalent_productivity_mw_per_m3s
            }
            ComputedParameter::AccumulatedProductivity { .. } => {
                energy_conversion.accumulated_productivity(hydro_idx, t)
            }
            ComputedParameter::IntegratedEquivalentProductivity { .. } => {
                energy_conversion.integrated_equivalent_productivity(hydro_idx, t)
            }
            ComputedParameter::IntegratedAccumulatedProductivity { .. } => {
                energy_conversion.integrated_accumulated_productivity(hydro_idx, t)
            }
            ComputedParameter::ReferenceVolume { .. } => {
                // Intentionally NO parquet `v_ref` override tier — unlike the
                // sibling `ReferenceTurbine` below, which consults the `q_ref`
                // override. Re-adding one "for symmetry" resurrects a retired
                // second source of truth that silently shadows the JSON value.
                energy_conversion
                    .conversion(hydro_idx, t)
                    .reference_volume_hm3
            }
            ComputedParameter::ReferenceTurbine { .. } => {
                // Keyed by the domain StageId (matches how the override table is
                // built and validated), never by the study position `t` used for
                // the position-indexed `energy_conversion` fallback below.
                override_table
                    .reference_outflow(hydro_id, stage_id)
                    .unwrap_or(
                        energy_conversion
                            .conversion(hydro_idx, t)
                            .reference_outflow_m3s,
                    )
            }
            ComputedParameter::MinStorage { .. } => {
                // Handled above via early return; unreachable here.
                hydro.min_storage_hm3
            }
            ComputedParameter::MaxStorage { .. } => {
                // Handled above via early return; unreachable here.
                hydro.max_storage_hm3
            }
            ComputedParameter::SpecificProductivity { .. } => override_table
                .resolve_specific_productivity(
                    hydro_id,
                    stage_id,
                    hydro.specific_productivity_mw_per_m3s_per_m,
                )
                .ok_or_else(|| ResolvedParametersError::MissingSpecificProductivity {
                    name: name.to_string(),
                    hydro_id,
                })?,
            ComputedParameter::MaxStoredEnergy { .. } => {
                energy_conversion.integrated_accumulated_productivity(hydro_idx, t)
                    * (hydro.max_storage_hm3 - hydro.min_storage_hm3)
            }
        };
        values.push(value);
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use cobre_core::{
        CascadeTopology, ComputedParameter, EntityId, ParameterKind, ScalarParameter, StudyPos,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
    };

    use cobre_io::{
        HydroEnergyProductivityRow, HydroGeometryRow, build_hydro_reference_volumes_resolved,
    };

    use super::*;
    use crate::energy_conversion::{
        EnergyConversion, EnergyConversionSet, build_energy_conversion_set,
        build_hydro_energy_productivity_override,
    };
    use crate::fpha_fitting::ForebayTable;
    use crate::test_support;

    // -------------------------------------------------------------------------
    // Shared test helpers
    // -------------------------------------------------------------------------

    /// Build a minimal [`Hydro`] with the given storage limits and optional
    /// `specific_productivity_mw_per_m3s_per_m`.
    fn make_hydro(
        id: i32,
        min_storage: f64,
        max_storage: f64,
        specific_productivity: Option<f64>,
    ) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("h{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_storage_hm3: min_storage,
            max_storage_hm3: max_storage,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: specific_productivity,
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
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    /// Build an [`EnergyConversionSet`] with `n_hydros` hydros, each with
    /// `n_stages` stages. Every `(hydro, stage)` cell is populated with
    /// predictable values derived from the hydro and stage indices.
    fn make_energy_conversion(n_hydros: usize, n_stages: usize) -> EnergyConversionSet {
        let per_hydro_stage: Vec<Vec<EnergyConversion>> = (0..n_hydros)
            .map(|h| {
                (0..n_stages)
                    .map(|t| EnergyConversion {
                        equivalent_productivity_mw_per_m3s: 0.42 + h as f64 + t as f64 * 0.01,
                        reference_volume_hm3: 1000.0 + h as f64 * 100.0 + t as f64,
                        reference_outflow_m3s: 500.0 + h as f64 * 50.0 + t as f64 * 0.1,
                    })
                    .collect()
            })
            .collect();
        let accumulated: Vec<Vec<f64>> = (0..n_hydros)
            .map(|h| {
                (0..n_stages)
                    .map(|t| 0.90 + h as f64 * 0.05 + t as f64 * 0.001)
                    .collect()
            })
            .collect();
        EnergyConversionSet::new(
            per_hydro_stage,
            accumulated,
            &test_support::minimal_hydros(n_hydros),
            n_stages,
        )
    }

    /// `StageId(0)..StageId(n_stages - 1)`: the 0-based domain ids every test
    /// fixture in this module uses (no pre-study-stage offset), so study
    /// position and domain id coincide.
    fn stage_ids_0_based(n_stages: usize) -> Vec<StageId> {
        (0..n_stages)
            .map(|s| StageId(i32::try_from(s).expect("test stage count fits in i32")))
            .collect()
    }

    /// One block per stage: the block counts every block-invariant fixture uses,
    /// under which `get(id, stage, b)` broadcasts identically for every `b`.
    fn one_block_per_stage(n_stages: usize) -> Vec<usize> {
        vec![1; n_stages]
    }

    /// Return `(hydros, energy_conversion, override_table, stage_to_season, stage_ids)`
    /// for tests that need a consistent set of inputs.
    fn make_setup_inputs(
        n_stages: usize,
    ) -> (
        Vec<Hydro>,
        EnergyConversionSet,
        HydroEnergyProductivityOverride,
        Vec<i32>,
        Vec<StageId>,
    ) {
        let hydros = vec![
            make_hydro(0, 50.0, 2000.0, Some(0.0085)),
            make_hydro(1, 100.0, 5000.0, Some(0.0090)),
        ];
        let energy_conversion = make_energy_conversion(2, n_stages);
        let override_table = HydroEnergyProductivityOverride::default();
        let stage_to_season: Vec<i32> = (0..n_stages).map(|t| (t % 4) as i32).collect();
        let stage_ids = stage_ids_0_based(n_stages);
        (
            hydros,
            energy_conversion,
            override_table,
            stage_to_season,
            stage_ids,
        )
    }

    fn make_param(id: i32, kind: ParameterKind) -> ScalarParameter {
        ScalarParameter {
            id: EntityId(id),
            name: format!("param_{id}"),
            kind,
        }
    }

    // -------------------------------------------------------------------------
    // AC-1: Constant kind fills all stages
    // -------------------------------------------------------------------------

    #[test]
    fn constant_kind_fills_all_stages() {
        let params = vec![make_param(0, ParameterKind::Constant { value: 3.6 })];
        let ec = EnergyConversionSet::new(vec![], vec![], &[], 4);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; 4];
        let stage_ids = stage_ids_0_based(4);

        let table = build_resolved_parameters(
            &params,
            &ec,
            &overrides,
            &[],
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(4),
            1_000_000.0,
        )
        .unwrap();

        assert!((table.get(EntityId(0), 0, 0) - 3.6).abs() < 1e-12);
        assert!((table.get(EntityId(0), 1, 0) - 3.6).abs() < 1e-12);
        assert!((table.get(EntityId(0), 2, 0) - 3.6).abs() < 1e-12);
        assert!((table.get(EntityId(0), 3, 0) - 3.6).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // AC-2: PerStage length mismatch errors
    // -------------------------------------------------------------------------

    #[test]
    fn per_stage_kind_length_mismatch_errors() {
        let params = vec![make_param(
            0,
            ParameterKind::PerStage {
                values: vec![1.0, 2.0],
            },
        )];
        let ec = EnergyConversionSet::new(vec![], vec![], &[], 3);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; 3];
        let stage_ids = stage_ids_0_based(3);

        let result = build_resolved_parameters(
            &params,
            &ec,
            &overrides,
            &[],
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(3),
            1_000_000.0,
        );

        assert!(matches!(
            result,
            Err(ResolvedParametersError::PerStageLengthMismatch {
                expected: 3,
                got: 2,
                ..
            })
        ));
    }

    // -------------------------------------------------------------------------
    // AC-3: Seasonal kind maps stage to value
    // -------------------------------------------------------------------------

    #[test]
    fn seasonal_kind_maps_stage_to_value() {
        let params = vec![make_param(
            0,
            ParameterKind::Seasonal {
                values: vec![(0, 0.5), (1, 1.5)],
            },
        )];
        let ec = EnergyConversionSet::new(vec![], vec![], &[], 3);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32, 1, 0];
        let stage_ids = stage_ids_0_based(3);

        let table = build_resolved_parameters(
            &params,
            &ec,
            &overrides,
            &[],
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(3),
            1_000_000.0,
        )
        .unwrap();

        assert!((table.get(EntityId(0), 0, 0) - 0.5).abs() < 1e-12);
        assert!((table.get(EntityId(0), 1, 0) - 1.5).abs() < 1e-12);
        assert!((table.get(EntityId(0), 2, 0) - 0.5).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // AC-4: Computed EquivalentProductivity reads EnergyConversionSet
    // -------------------------------------------------------------------------

    #[test]
    fn computed_equivalent_productivity_reads_energy_conversion() {
        let n_stages = 4;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(0),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        let expected = energy_conversion
            .conversion(0, 0)
            .equivalent_productivity_mw_per_m3s;
        assert!((table.get(EntityId(0), 0, 0) - expected).abs() < 1e-12);
        // Validate all stages
        for t in 0..n_stages {
            let exp_t = energy_conversion
                .conversion(0, t)
                .equivalent_productivity_mw_per_m3s;
            assert!(
                (table.get(EntityId(0), t, 0) - exp_t).abs() < 1e-12,
                "stage {t}: expected {exp_t}, got {}",
                table.get(EntityId(0), t, 0)
            );
        }
    }

    // -------------------------------------------------------------------------
    // Integrated productivity reads the integrated grids, not the reference-point
    // cells — installed distinct so a resolve arm that read `conversion` /
    // `accumulated_productivity` instead would fail bit-for-bit.
    // -------------------------------------------------------------------------

    #[test]
    fn computed_integrated_productivity_reads_integrated_grids() {
        let n_stages = 4;
        let (hydros, base_ec, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let integrated_equivalent: Vec<Vec<f64>> = (0..2usize)
            .map(|h| {
                (0..n_stages)
                    .map(|t| 7.0 + h as f64 + t as f64 * 0.1)
                    .collect()
            })
            .collect();
        let integrated_accumulated: Vec<Vec<f64>> = (0..2usize)
            .map(|h| {
                (0..n_stages)
                    .map(|t| 3.0 + h as f64 + t as f64 * 0.2)
                    .collect()
            })
            .collect();
        let energy_conversion =
            base_ec.with_integrated(integrated_equivalent, integrated_accumulated);

        let params = vec![
            make_param(
                0,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::IntegratedEquivalentProductivity {
                        hydro_id: EntityId(0),
                    },
                },
            ),
            make_param(
                1,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::IntegratedAccumulatedProductivity {
                        hydro_id: EntityId(1),
                    },
                },
            ),
        ];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        for t in 0..n_stages {
            let exp_eq = energy_conversion.integrated_equivalent_productivity(0, t);
            assert_eq!(
                table.get(EntityId(0), t, 0).to_bits(),
                exp_eq.to_bits(),
                "stage {t}: IntegratedEquivalentProductivity must equal the integrated grid"
            );
            let exp_acum = energy_conversion.integrated_accumulated_productivity(1, t);
            assert_eq!(
                table.get(EntityId(1), t, 0).to_bits(),
                exp_acum.to_bits(),
                "stage {t}: IntegratedAccumulatedProductivity must equal the integrated grid"
            );
        }

        // The installed integrated grids genuinely differ from the reference-point
        // grids the non-integrated tags read, so the arms cannot pass by aliasing.
        assert_ne!(
            energy_conversion
                .integrated_equivalent_productivity(0, 0)
                .to_bits(),
            energy_conversion
                .conversion(0, 0)
                .equivalent_productivity_mw_per_m3s
                .to_bits(),
        );
        assert_ne!(
            energy_conversion
                .integrated_accumulated_productivity(1, 0)
                .to_bits(),
            energy_conversion.accumulated_productivity(1, 0).to_bits(),
        );
    }

    // -------------------------------------------------------------------------
    // ReferenceVolume reads the JSON-sourced energy-conversion cell directly
    // (no parquet `v_ref` override tier) — the surviving half of the asymmetry.
    // -------------------------------------------------------------------------

    #[test]
    fn reference_volume_reads_energy_conversion_cell() {
        let n_stages = 4;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::ReferenceVolume {
                    hydro_id: EntityId(1),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        // For every stage the resolved value is bit-for-bit the energy-conversion
        // cell — there is no override branch that could shadow it.
        for t in 0..n_stages {
            let expected = energy_conversion.conversion(1, t).reference_volume_hm3;
            assert_eq!(
                table.get(EntityId(0), t, 0).to_bits(),
                expected.to_bits(),
                "stage {t}: ReferenceVolume must equal the energy-conversion cell"
            );
        }
    }

    // -------------------------------------------------------------------------
    // ReferenceTurbine STILL consults the parquet `q_ref` override (the other
    // half of the intentional asymmetry documented at the resolution site).
    // -------------------------------------------------------------------------

    #[test]
    fn reference_turbine_consults_parquet_override() {
        let n_stages = 3;
        let (hydros, energy_conversion, _default_override, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        // A per-hydro-default q_ref override that differs from every
        // energy-conversion `reference_outflow_m3s` cell.
        let override_q_ref = 12_345.0;
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: None,
                reference_outflow_m3s: Some(override_q_ref),
                specific_productivity_mw_per_m3s_per_m: None,
            }])
            .unwrap();

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::ReferenceTurbine {
                    hydro_id: EntityId(1),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        for t in 0..n_stages {
            assert_eq!(
                table.get(EntityId(0), t, 0).to_bits(),
                override_q_ref.to_bits(),
                "stage {t}: ReferenceTurbine must honor the parquet q_ref override"
            );
        }
    }

    /// A stage-specific override (not the per-hydro default) resolves by
    /// domain `StageId`, not by the study position `t`. The study has a
    /// single stage whose position is 0 but whose domain id is 60; keying
    /// the lookup at position 0 would miss the `(hydro, 60)` entry and fall
    /// through to the `energy_conversion` / hydro-entity fallback instead.
    #[test]
    fn reference_turbine_and_specific_productivity_key_by_domain_stage_id_not_position() {
        let hydros = vec![make_hydro(1, 100.0, 5000.0, Some(0.0090))];
        let energy_conversion = make_energy_conversion(1, 1);
        let stage_to_season = vec![0i32];
        let stage_ids = vec![StageId(60)];

        let override_q_ref = 777.0;
        let override_esp = 0.0123;
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(60),
                equivalent_productivity_mw_per_m3s: None,
                reference_outflow_m3s: Some(override_q_ref),
                specific_productivity_mw_per_m3s_per_m: Some(override_esp),
            }])
            .unwrap();

        let params = vec![
            make_param(
                0,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::ReferenceTurbine {
                        hydro_id: EntityId(1),
                    },
                },
            ),
            make_param(
                1,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::SpecificProductivity {
                        hydro_id: EntityId(1),
                    },
                },
            ),
        ];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(1),
            1_000_000.0,
        )
        .unwrap();

        assert_eq!(
            table.get(EntityId(0), 0, 0).to_bits(),
            override_q_ref.to_bits(),
            "ReferenceTurbine must resolve the StageId(60) override at position 0"
        );
        assert_eq!(
            table.get(EntityId(1), 0, 0).to_bits(),
            override_esp.to_bits(),
            "SpecificProductivity must resolve the StageId(60) override at position 0"
        );
    }

    // -------------------------------------------------------------------------
    // AC-5: SpecificProductivity missing returns error
    // -------------------------------------------------------------------------

    #[test]
    fn computed_specific_productivity_missing_returns_error() {
        let n_stages = 2;
        // Hydro with no specific_productivity
        let hydros = vec![make_hydro(0, 50.0, 2000.0, None)];
        let energy_conversion = make_energy_conversion(1, n_stages);
        let override_table = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::SpecificProductivity {
                    hydro_id: EntityId(0),
                },
            },
        )];

        let result = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        );

        assert!(matches!(
            result,
            Err(ResolvedParametersError::MissingSpecificProductivity {
                hydro_id: EntityId(0),
                ..
            })
        ));
    }

    /// The `SpecificProductivity` tag and the energy-conversion builder's mean
    /// own term resolve the override table through the same
    /// `resolve_specific_productivity` method: recomputing the mean own term as
    /// `tag_value · mean_head` from an independently built `ForebayTable`
    /// (identical VHA rows) recovers the builder's grid cell bit-for-bit.
    #[test]
    fn specific_productivity_tag_and_builder_resolve_the_same_rho_esp() {
        let hydro = make_hydro(1, 100.0, 300.0, None);
        let hydro_id = hydro.id;
        let hydros = vec![hydro];
        let cascade = CascadeTopology::build(&hydros);
        let resolver =
            build_hydro_reference_volumes_resolved(&[(hydro_id, StudyPos(0), 200.0)], 0.0);
        let rows = vec![
            HydroGeometryRow {
                hydro_id,
                volume_hm3: 0.0,
                height_m: 100.0,
                area_km2: 1.0,
            },
            HydroGeometryRow {
                hydro_id,
                volume_hm3: 1000.0,
                height_m: 200.0,
                area_km2: 1.0,
            },
        ];
        let mut vha = HashMap::new();
        vha.insert(hydro_id, rows.clone());
        let override_esp = 0.0123;
        let override_table =
            build_hydro_energy_productivity_override(&[HydroEnergyProductivityRow {
                hydro_id,
                stage_id: None,
                equivalent_productivity_mw_per_m3s: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: Some(override_esp),
            }])
            .expect("override builds");
        let stage_ids = stage_ids_0_based(1);

        let energy_conversion = build_energy_conversion_set(
            &hydros,
            &stage_ids,
            &cascade,
            &resolver,
            &vha,
            Some(&override_table),
            None,
        )
        .expect("builder succeeds");

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::SpecificProductivity { hydro_id },
            },
        )];
        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &[0i32],
            &stage_ids,
            &one_block_per_stage(1),
            1_000_000.0,
        )
        .expect("tag resolves");
        let tag_value = table.get(EntityId(0), 0, 0);
        assert_eq!(
            tag_value.to_bits(),
            override_esp.to_bits(),
            "the tag resolves the same override the builder consults"
        );

        let recomputed_table = ForebayTable::new(&rows, "coherence").expect("table builds");
        let mean_head = recomputed_table.mean_height(100.0, 300.0);
        let expected_integrated = tag_value * mean_head;
        assert_eq!(
            expected_integrated.to_bits(),
            energy_conversion
                .integrated_equivalent_productivity(0, 0)
                .to_bits(),
            "the builder must apply the same resolved ρ_esp the tag resolves"
        );
    }

    // -------------------------------------------------------------------------
    // AC-6: Declaration-order invariance
    // -------------------------------------------------------------------------

    #[test]
    fn declaration_order_invariance() {
        let n_stages = 3;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let param_a = make_param(10, ParameterKind::Constant { value: 1.0 });
        let param_b = make_param(
            20,
            ParameterKind::PerStage {
                values: vec![2.0, 3.0, 4.0],
            },
        );
        let param_c = make_param(
            30,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::AccumulatedProductivity {
                    hydro_id: EntityId(0),
                },
            },
        );

        let params_abc = vec![param_a.clone(), param_b.clone(), param_c.clone()];
        let params_cab = vec![param_c.clone(), param_a.clone(), param_b.clone()];

        let table_abc = build_resolved_parameters(
            &params_abc,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();
        let table_cab = build_resolved_parameters(
            &params_cab,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        for id in [EntityId(10), EntityId(20), EntityId(30)] {
            for t in 0..n_stages {
                assert_eq!(
                    table_abc.get(id, t, 0).to_bits(),
                    table_cab.get(id, t, 0).to_bits(),
                    "bit mismatch for id={id:?}, stage={t}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // UnknownHydro variant
    // -------------------------------------------------------------------------

    #[test]
    fn unknown_hydro_in_computed_returns_error() {
        let n_stages = 2;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        // Use a hydro_id that does not exist in the hydros slice (ids 0 and 1 exist).
        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::MinStorage {
                    hydro_id: EntityId(99),
                },
            },
        )];

        let result = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        );

        assert!(matches!(
            result,
            Err(ResolvedParametersError::UnknownHydro {
                hydro_id: EntityId(99),
                ..
            })
        ));
    }

    // -------------------------------------------------------------------------
    // MinStorage / MaxStorage: stage-invariant hydro field
    // -------------------------------------------------------------------------

    #[test]
    fn min_max_storage_use_stage_invariant_hydro_field() {
        let n_stages = 5;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![
            make_param(
                0,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::MinStorage {
                        hydro_id: EntityId(0),
                    },
                },
            ),
            make_param(
                1,
                ParameterKind::Computed {
                    computed_spec: ComputedParameter::MaxStorage {
                        hydro_id: EntityId(1),
                    },
                },
            ),
        ];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        // hydro 0: min_storage = 50.0
        for t in 0..n_stages {
            assert!(
                (table.get(EntityId(0), t, 0) - 50.0).abs() < 1e-12,
                "min_storage mismatch at stage {t}"
            );
        }
        // hydro 1: max_storage = 5000.0
        for t in 0..n_stages {
            assert!(
                (table.get(EntityId(1), t, 0) - 5000.0).abs() < 1e-12,
                "max_storage mismatch at stage {t}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Empty parameter slice: n_stages = 0 does not panic
    // -------------------------------------------------------------------------

    #[test]
    fn empty_parameters_n_stages_zero_is_ok() {
        let ec = EnergyConversionSet::new(vec![], vec![], &[], 0);
        let overrides = HydroEnergyProductivityOverride::default();

        let table =
            build_resolved_parameters(&[], &ec, &overrides, &[], &[], &[], &[], 1_000_000.0)
                .unwrap();
        // Nothing to query — just verify it doesn't panic.
        let _ = table;
    }

    // -------------------------------------------------------------------------
    // PerStageBlock resolves to distinct per-(stage, block) values
    // -------------------------------------------------------------------------

    /// Build a table over `n_stages` stages with `block_counts` blocks per stage,
    /// carrying only the supplied `params` (no hydros / energy conversion).
    fn build_block_table(
        params: &[ScalarParameter],
        block_counts: &[usize],
    ) -> Result<ResolvedParameters, ResolvedParametersError> {
        let n_stages = block_counts.len();
        let ec = EnergyConversionSet::new(vec![], vec![], &[], n_stages);
        let overrides = HydroEnergyProductivityOverride::default();
        let stage_to_season = vec![0i32; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);
        build_resolved_parameters(
            params,
            &ec,
            &overrides,
            &[],
            &stage_to_season,
            &stage_ids,
            block_counts,
            1_000_000.0,
        )
    }

    #[test]
    fn per_stage_block_resolves_to_distinct_block_values() {
        let params = vec![make_param(
            0,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 10.0), (0, 1, 20.0), (1, 0, 30.0), (1, 1, 40.0)],
            },
        )];

        let table = build_block_table(&params, &[2, 2]).unwrap();

        assert_eq!(table.get(EntityId(0), 1, 0).to_bits(), 30.0f64.to_bits());
        assert_eq!(table.get(EntityId(0), 1, 1).to_bits(), 40.0f64.to_bits());
        assert_ne!(
            table.get(EntityId(0), 1, 0).to_bits(),
            table.get(EntityId(0), 1, 1).to_bits(),
            "stage-1 block-0 and block-1 must resolve distinct values"
        );
    }

    #[test]
    fn block_invariant_kinds_broadcast_to_every_block() {
        // Each of the four stage-/block-invariant kinds stores a length-1 inner
        // vector that must resolve identically for any block index.
        let params = vec![
            make_param(0, ParameterKind::Constant { value: 3.6 }),
            make_param(
                1,
                ParameterKind::PerStage {
                    values: vec![1.0, 2.0],
                },
            ),
            make_param(
                2,
                ParameterKind::Seasonal {
                    values: vec![(0, 0.5)],
                },
            ),
        ];

        let table = build_block_table(&params, &[1, 1]).unwrap();

        for id in [EntityId(0), EntityId(1), EntityId(2)] {
            for stage in 0..2 {
                assert_eq!(
                    table.get(id, stage, 0).to_bits(),
                    table.get(id, stage, 3).to_bits(),
                    "id={id:?} stage={stage}: length-1 inner must broadcast to every block"
                );
            }
        }
    }

    #[test]
    fn per_stage_block_missing_cell_is_a_coverage_error() {
        let params = vec![make_param(
            0,
            ParameterKind::PerStageBlock {
                // stage 0 covers only block 0; the (0, 1) cell is left uncovered.
                values: vec![(0, 0, 1.0), (1, 0, 2.0), (1, 1, 3.0)],
            },
        )];

        let result = build_block_table(&params, &[2, 2]);

        assert!(
            matches!(
                &result,
                Err(ResolvedParametersError::PerStageBlockCoverage {
                    name,
                    stage: 0,
                    block: 1,
                    issue: PerStageBlockCoverageIssue::Missing,
                }) if name.as_str() == "param_0"
            ),
            "expected a Missing coverage error at (0, 1) naming param_0, got {result:?}"
        );
    }

    #[test]
    fn per_stage_block_duplicate_cell_is_a_coverage_error() {
        let params = vec![make_param(
            0,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0), (0, 0, 9.0), (0, 1, 2.0)],
            },
        )];

        let result = build_block_table(&params, &[2]);

        assert!(
            matches!(
                &result,
                Err(ResolvedParametersError::PerStageBlockCoverage {
                    stage: 0,
                    block: 0,
                    issue: PerStageBlockCoverageIssue::Duplicated,
                    ..
                })
            ),
            "expected a Duplicated coverage error at (0, 0), got {result:?}"
        );
    }

    #[test]
    fn per_stage_block_out_of_range_cell_is_a_coverage_error() {
        let params = vec![make_param(
            0,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0), (0, 5, 2.0)],
            },
        )];

        let result = build_block_table(&params, &[2]);

        assert!(
            matches!(
                &result,
                Err(ResolvedParametersError::PerStageBlockCoverage {
                    stage: 0,
                    block: 5,
                    issue: PerStageBlockCoverageIssue::OutOfRange,
                    ..
                })
            ),
            "expected an OutOfRange coverage error at (0, 5), got {result:?}"
        );
    }

    #[test]
    #[should_panic(expected = "block miss")]
    fn out_of_range_block_query_fires_the_block_specific_assert() {
        let params = vec![make_param(
            0,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0), (0, 1, 2.0)],
            },
        )];
        let table = build_block_table(&params, &[2]).unwrap();
        // Block 5 is out of range for a stage that stores two blocks; the miss is
        // a block-index miss, distinct from the unknown-id miss.
        let _ = table.get(EntityId(0), 0, 5);
    }

    #[test]
    fn is_block_varying_truth_table() {
        let params = vec![
            make_param(
                0,
                ParameterKind::PerStageBlock {
                    values: vec![(0, 0, 1.0), (0, 1, 2.0)],
                },
            ),
            make_param(1, ParameterKind::Constant { value: 3.6 }),
        ];

        let table = build_block_table(&params, &[2]).unwrap();

        assert!(
            table.is_block_varying(EntityId(0)),
            "PerStageBlock is block-varying"
        );
        assert!(
            !table.is_block_varying(EntityId(1)),
            "Constant broadcasts and is not block-varying"
        );
        assert!(
            !table.is_block_varying(EntityId(99)),
            "an unknown id is not block-varying"
        );
    }

    // -------------------------------------------------------------------------
    // MaxStoredEnergy: per-stage oracle on integrated_accumulated_productivity,
    // physical-bounds source, and declaration-order invariance.
    // -------------------------------------------------------------------------

    #[test]
    fn max_stored_energy_matches_oracle_on_integrated_accumulated_grid() {
        let n_stages = 4;
        let (hydros, energy_conversion, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::MaxStoredEnergy {
                    hydro_id: EntityId(1),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        let physical_range = hydros[1].max_storage_hm3 - hydros[1].min_storage_hm3;
        for t in 0..n_stages {
            let expected =
                energy_conversion.integrated_accumulated_productivity(1, t) * physical_range;
            assert_eq!(
                table.get(EntityId(0), t, 0).to_bits(),
                expected.to_bits(),
                "stage {t}: MaxStoredEnergy must equal \
                 integrated_accumulated_productivity * (V_hi - V_lo)"
            );
        }

        // The resolved value still varies across stages here — driven entirely by
        // the integrated_accumulated_productivity grid; the physical range factor
        // (hydro.{min,max}_storage_hm3) is stage-invariant.
        assert_ne!(
            table.get(EntityId(0), 0, 0).to_bits(),
            table.get(EntityId(0), n_stages - 1, 0).to_bits(),
            "MaxStoredEnergy must vary across stages when the integrated productivity grid varies"
        );
    }

    /// `MaxStoredEnergy` reads the hydro entity's own `[min, max]_storage_hm3` —
    /// stage-invariant by construction. Pinning `integrated_accumulated_productivity`
    /// stage-invariant too isolates the range factor: a stage-varying resolved value
    /// here could only come from reading something other than the entity fields.
    #[test]
    fn max_stored_energy_reads_physical_bounds_not_operative_hydro_fields() {
        let n_stages = 3;
        let (hydros, base_ec, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        let integrated_accumulated: Vec<Vec<f64>> =
            (0..hydros.len()).map(|_| vec![2.0; n_stages]).collect();
        let integrated_equivalent = integrated_accumulated.clone();
        let energy_conversion =
            base_ec.with_integrated(integrated_equivalent, integrated_accumulated);

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::MaxStoredEnergy {
                    hydro_id: EntityId(0),
                },
            },
        )];

        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        let expected = 2.0 * (hydros[0].max_storage_hm3 - hydros[0].min_storage_hm3);
        for t in 0..n_stages {
            assert_eq!(
                table.get(EntityId(0), t, 0).to_bits(),
                expected.to_bits(),
                "stage {t}: MaxStoredEnergy must read hydro.{{min,max}}_storage_hm3"
            );
        }
    }

    #[test]
    fn max_stored_energy_is_invariant_to_hydro_declaration_order() {
        let n_stages = 3;
        let stage_to_season = vec![0i32; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);
        let override_table = HydroEnergyProductivityOverride::default();

        // Every input is keyed by hydro ENTITY id, never by position: an
        // indexing bug that conflates the two surfaces as a mismatch between
        // the two orderings built below.
        let integrated_for = |id: i32, t: usize| 10.0 + f64::from(id) + t as f64 * 0.3;
        let min_storage_for = |id: i32| 5.0 * f64::from(id);
        let max_storage_for = |id: i32| 100.0 * f64::from(id) + 50.0;

        let build_for_order = |order: &[i32]| -> (Vec<Hydro>, EnergyConversionSet) {
            let hydros: Vec<Hydro> = order
                .iter()
                .map(|&id| make_hydro(id, min_storage_for(id), max_storage_for(id), Some(0.0085)))
                .collect();
            let n_hydros = hydros.len();

            let zero_conversion = EnergyConversion {
                equivalent_productivity_mw_per_m3s: 0.0,
                reference_volume_hm3: 0.0,
                reference_outflow_m3s: 0.0,
            };
            let per_hydro_stage = vec![vec![zero_conversion; n_stages]; n_hydros];
            let accumulated = vec![vec![0.0; n_stages]; n_hydros];
            let integrated_equivalent = vec![vec![0.0; n_stages]; n_hydros];
            let integrated_accumulated: Vec<Vec<f64>> = order
                .iter()
                .map(|&id| (0..n_stages).map(|t| integrated_for(id, t)).collect())
                .collect();
            let energy_conversion =
                EnergyConversionSet::new(per_hydro_stage, accumulated, &hydros, n_stages)
                    .with_integrated(integrated_equivalent, integrated_accumulated);

            (hydros, energy_conversion)
        };

        let order_abc = [1_i32, 2, 3];
        let order_cab = [3_i32, 1, 2];

        let (hydros_abc, ec_abc) = build_for_order(&order_abc);
        let (hydros_cab, ec_cab) = build_for_order(&order_cab);

        // Same parameters — referencing hydro identity, not position — resolved
        // against each self-consistent (hydros, energy_conversion) pair.
        let params: Vec<ScalarParameter> = order_abc
            .iter()
            .enumerate()
            .map(|(slot, &id)| {
                make_param(
                    i32::try_from(slot).unwrap(),
                    ParameterKind::Computed {
                        computed_spec: ComputedParameter::MaxStoredEnergy {
                            hydro_id: EntityId(id),
                        },
                    },
                )
            })
            .collect();

        let table_abc = build_resolved_parameters(
            &params,
            &ec_abc,
            &override_table,
            &hydros_abc,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();
        let table_cab = build_resolved_parameters(
            &params,
            &ec_cab,
            &override_table,
            &hydros_cab,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        for (slot, &id) in order_abc.iter().enumerate() {
            let param_id = EntityId(i32::try_from(slot).unwrap());
            for t in 0..n_stages {
                let a = table_abc.get(param_id, t, 0);
                let b = table_cab.get(param_id, t, 0);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "hydro_id={id}, stage={t}: MaxStoredEnergy must not depend on hydro \
                     declaration order"
                );
                let expected = integrated_for(id, t) * (max_storage_for(id) - min_storage_for(id));
                assert_eq!(
                    a.to_bits(),
                    expected.to_bits(),
                    "hydro_id={id}, stage={t}: resolved value must match the entity-keyed oracle"
                );
            }
        }
    }

    /// On a hydro with a genuine `[V_lo, V_hi]` range whose integrated grid differs
    /// from the reference-point grid, dividing `pct * MaxStoredEnergy` by that
    /// hydro's integrated cascade productivity leaves `pct * (V_hi - V_lo)`, a pure
    /// volume quantity.
    #[test]
    fn normalized_max_stored_energy_bound_cancels_the_grid_on_a_real_range() {
        let n_stages = 1;
        let t = 0;
        let (hydros, base_ec, override_table, stage_to_season, stage_ids) =
            make_setup_inputs(n_stages);

        // Distinct from the reference-point grid, so the real-range precondition
        // below is genuinely exercised, not a collapsed-range triviality.
        let int_rho = 0.625_f64;
        let integrated_equivalent = vec![vec![int_rho; n_stages]; hydros.len()];
        let integrated_accumulated = integrated_equivalent.clone();
        let energy_conversion =
            base_ec.with_integrated(integrated_equivalent, integrated_accumulated);

        let rho_acum = energy_conversion.accumulated_productivity(0, t);
        assert_ne!(
            int_rho.to_bits(),
            rho_acum.to_bits(),
            "fixture must exercise a real range, not the collapsed-range triviality"
        );

        let params = vec![make_param(
            0,
            ParameterKind::Computed {
                computed_spec: ComputedParameter::MaxStoredEnergy {
                    hydro_id: EntityId(0),
                },
            },
        )];
        let table = build_resolved_parameters(
            &params,
            &energy_conversion,
            &override_table,
            &hydros,
            &stage_to_season,
            &stage_ids,
            &one_block_per_stage(n_stages),
            1_000_000.0,
        )
        .unwrap();

        let v_lo = hydros[0].min_storage_hm3;
        let v_hi = hydros[0].max_storage_hm3;
        let pct = 0.25_f64;
        let expected = pct * (v_hi - v_lo);
        let rel_tol = 1e-9 * expected.abs().max(1.0);

        let max_stored_energy = table.get(EntityId(0), t, 0);
        let normalized_integrated = pct * max_stored_energy / int_rho;
        assert!(
            (normalized_integrated - expected).abs() <= rel_tol,
            "normalized bound on the integrated grid: got {normalized_integrated}, expected {expected}"
        );
    }
}
