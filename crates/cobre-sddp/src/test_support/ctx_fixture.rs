//! One test-only fixture for every hand-built [`TemplateBuildCtx`].

use std::collections::{BTreeMap, HashMap};

use cobre_core::{
    Bus, CascadeTopology, EnergyContract, EntityId, GenericConstraint, Hydro, Line, LoadModel,
    NonControllableSource, PumpingStation, ResolvedBounds, ResolvedGenericConstraintBounds,
    ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, Thermal,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::bucket_topology::TransitBucketTopology;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
use crate::indexer::{
    AnticipatedPlants, EntityPositions, HydroCellIndex, StateSpace, StudyDimensions,
};
use crate::lead_time::AnticipatedResolution;
use crate::lp::builder::{ResolvedTables, TemplateBuildCtx};
use crate::resolved_parameters::ResolvedParameters;
use crate::test_support::constant_lead_resolution;
use crate::time_value::{PostStudyResolved, TimeValue};

/// Owns every value a [`TemplateBuildCtx`] borrows, or holds by value, so a
/// test builds one through [`Self::ctx`] instead of hand-writing its own
/// field-by-field construction; larger builder fixtures embed it as their
/// `base`. `ctx()` derives `positions` from its own slices, the way
/// [`EntityPositions::build`] does; every other field is copied through
/// unchanged.
pub(crate) struct CtxFixture {
    /// Derived fresh, from the slices below, on every [`Self::ctx`] call — the
    /// backing store [`TemplateBuildCtx::positions`] borrows.
    pub(crate) positions: EntityPositions,
    pub(crate) hydros: Vec<Hydro>,
    pub(crate) thermals: Vec<Thermal>,
    pub(crate) lines: Vec<Line>,
    pub(crate) buses: Vec<Bus>,
    pub(crate) load_models: Vec<LoadModel>,
    pub(crate) cascade: CascadeTopology,
    pub(crate) hydro_cell_index: HydroCellIndex,
    pub(crate) bounds: ResolvedBounds,
    pub(crate) penalties: ResolvedPenalties,
    pub(crate) resolved_generic_bounds: ResolvedGenericConstraintBounds,
    pub(crate) resolved_load_factors: ResolvedLoadFactors,
    pub(crate) resolved_ncs_bounds: ResolvedNcsBounds,
    pub(crate) resolved_ncs_factors: ResolvedNcsFactors,
    pub(crate) resolved_parameters: ResolvedParameters,
    pub(crate) par_lp: PrecomputedPar,
    pub(crate) production_models: ProductionModelSet,
    pub(crate) evaporation_models: EvaporationModelSet,
    pub(crate) generic_constraints: Vec<GenericConstraint>,
    pub(crate) non_controllable_sources: Vec<NonControllableSource>,
    pub(crate) pumping_stations: Vec<PumpingStation>,
    pub(crate) contracts: Vec<EnergyContract>,
    pub(crate) diversion_upstream: HashMap<EntityId, Vec<usize>>,
    pub(crate) anticipated_lead_stages: Vec<usize>,
    /// Non-default only when a test set it explicitly — [`Self::ctx`] then
    /// attaches it to `self.state` as-is, in place of the saturating-default
    /// [`constant_lead_resolution`] every other fixture gets.
    pub(crate) anticipated_resolution: AnticipatedResolution,
    pub(crate) anticipated_plants: AnticipatedPlants,
    pub(crate) has_penalty: bool,
    /// Derived fresh, from `anticipated_plants`/`has_penalty` and the slices
    /// below, on every [`Self::ctx`] call — the backing store
    /// [`TemplateBuildCtx::study_dims`] borrows.
    pub(crate) study_dims: StudyDimensions,
    pub(crate) time_value: TimeValue,
    pub(crate) filling_v_target: BTreeMap<(usize, i32), f64>,
    /// The resolved bucket topology. Its fields are `pub(crate)` and mutable
    /// after construction (only [`TransitBucketTopology::empty`] can build
    /// one outside `bucket_topology.rs`, since its `arcs` field is private
    /// there) — a test needing a non-empty one mutates this in place.
    pub(crate) topology: TransitBucketTopology,
    /// Derived fresh, from this fixture's own `hydros/par_lp/topology`/
    /// `anticipated_lead_stages`, on every [`Self::ctx`] call — the backing
    /// store [`TemplateBuildCtx::state`] borrows. See [`Self::build_state`]
    /// for a test that needs a differently-resolved one.
    pub(crate) state: StateSpace,
}

impl Default for CtxFixture {
    fn default() -> Self {
        let topology = TransitBucketTopology::empty();
        let state = StateSpace::build(
            &[],
            0,
            &[],
            &topology,
            Vec::new(),
            AnticipatedResolution::default(),
        );
        Self {
            positions: EntityPositions::from_slices([], [], [], [], [], []),
            hydros: Vec::new(),
            thermals: Vec::new(),
            lines: Vec::new(),
            buses: Vec::new(),
            load_models: Vec::new(),
            cascade: CascadeTopology::build(&[]),
            hydro_cell_index: HydroCellIndex::build(&[]),
            bounds: ResolvedBounds::empty(),
            penalties: ResolvedPenalties::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            resolved_parameters: ResolvedParameters::default(),
            par_lp: PrecomputedPar::default(),
            production_models: ProductionModelSet::new(Vec::new(), &[], 0),
            evaporation_models: EvaporationModelSet::new(Vec::new()),
            generic_constraints: Vec::new(),
            non_controllable_sources: Vec::new(),
            pumping_stations: Vec::new(),
            contracts: Vec::new(),
            diversion_upstream: HashMap::new(),
            anticipated_lead_stages: Vec::new(),
            anticipated_resolution: AnticipatedResolution::default(),
            anticipated_plants: AnticipatedPlants::default(),
            has_penalty: false,
            study_dims: StudyDimensions::default(),
            time_value: TimeValue::from_parts(
                Vec::new(),
                vec![1.0],
                vec![744.0],
                vec![0],
                PostStudyResolved::default(),
            ),
            filling_v_target: BTreeMap::new(),
            topology,
            state,
        }
    }
}

impl CtxFixture {
    /// Derives `positions`, `state`, and `study_dims` from this fixture's own
    /// slices, the way [`EntityPositions::build`]/`build_study_dimensions` do,
    /// and `time_value`'s one-step discount factors as one `1.0` per `bounds`
    /// stage; every other field is copied through unchanged. `&mut self`: all
    /// four are recomputed into `self`'s own fields on every call, so
    /// [`TemplateBuildCtx::positions`]/`state`/`study_dims` can borrow a
    /// backing store with `self`'s own lifetime. `state` attaches
    /// `self.anticipated_resolution` as-is when a test set it explicitly (it
    /// is no longer `AnticipatedResolution::default`), else the
    /// saturating-default [`constant_lead_resolution`] over this fixture's
    /// own `anticipated_lead_stages`. A test that needs a derived field's
    /// value to disagree with its own slices sets the fixture's owner field
    /// (e.g. `self.has_penalty`) before calling [`Self::ctx`], never the
    /// returned context's field.
    pub(crate) fn ctx(&mut self) -> TemplateBuildCtx<'_> {
        self.positions = EntityPositions::from_slices(
            self.hydros.iter().map(|h| h.id),
            self.thermals.iter().map(|t| t.id),
            self.lines.iter().map(|l| l.id),
            self.buses.iter().map(|b| b.id),
            self.pumping_stations.iter().map(|p| p.id),
            self.contracts.iter().map(|c| c.id),
        );
        let resolution = if self.anticipated_resolution == AnticipatedResolution::default() {
            constant_lead_resolution(&self.anticipated_lead_stages, self.bounds.n_stages())
        } else {
            self.anticipated_resolution.clone()
        };
        self.state = self.build_state(resolution);
        let (_, cumulative, calendar, post_study) = self.time_value.canonical_fields();
        self.time_value = TimeValue::from_parts(
            vec![1.0; self.bounds.n_stages()],
            cumulative.to_vec(),
            calendar.total_hours().to_vec(),
            calendar.stage_ids().to_vec(),
            post_study.clone(),
        );
        self.study_dims = StudyDimensions {
            max_deficit_segments: self
                .buses
                .iter()
                .map(|b| b.deficit_segments.len())
                .max()
                .unwrap_or(0),
            inflow_method: if self.has_penalty {
                crate::InflowNonNegativityMethod::Penalty
            } else {
                crate::InflowNonNegativityMethod::None
            },
            anticipated_plants: self.anticipated_plants.clone(),
            ..StudyDimensions::default()
        };
        TemplateBuildCtx {
            hydros: &self.hydros,
            thermals: &self.thermals,
            lines: &self.lines,
            buses: &self.buses,
            load_models: &self.load_models,
            cascade: &self.cascade,
            hydro_cell_index: &self.hydro_cell_index,
            resolved: ResolvedTables {
                bounds: &self.bounds,
                penalties: &self.penalties,
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                resolved_parameters: &self.resolved_parameters,
            },
            positions: &self.positions,
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &self.generic_constraints,
            non_controllable_sources: &self.non_controllable_sources,
            pumping_stations: &self.pumping_stations,
            contracts: &self.contracts,
            diversion_upstream: &self.diversion_upstream,
            state: &self.state,
            study_dims: &self.study_dims,
            time_value: &self.time_value,
            filling_v_target: &self.filling_v_target,
            topology: &self.topology,
        }
    }

    /// Build a [`StateSpace`] from this fixture's current
    /// `hydros/par_lp/topology/anticipated_lead_stages`, attaching `resolution`.
    /// [`Self::ctx`] calls this for its own `self.state`, with either the
    /// saturating-default [`constant_lead_resolution`] or `self`'s own
    /// explicitly-set `anticipated_resolution` (see [`Self::ctx`]). A test
    /// needing a non-empty bucket topology sets `self.topology`'s fields
    /// before calling [`Self::ctx`]; both flow into the built state
    /// automatically.
    pub(crate) fn build_state(&self, resolution: AnticipatedResolution) -> StateSpace {
        let max_par_order = self.par_lp.max_order();
        let effective_lag_counts: Vec<usize> = if max_par_order > 0 {
            (0..self.hydros.len())
                .map(|h| {
                    if h < self.par_lp.n_hydros() {
                        self.par_lp.effective_lag_count(h)
                    } else {
                        max_par_order
                    }
                })
                .collect()
        } else {
            vec![0; self.hydros.len()]
        };
        StateSpace::build(
            &self.hydros,
            max_par_order,
            &effective_lag_counts,
            &self.topology,
            self.anticipated_lead_stages.clone(),
            resolution,
        )
    }
}
