//! One test-only fixture for every hand-built [`TemplateBuildCtx`].

use std::collections::{BTreeMap, HashMap};

use cobre_core::{
    Bus, CascadeTopology, ContractType, EnergyContract, EntityId, GenericConstraint, Hydro, Line,
    LoadModel, NonControllableSource, PumpingStation, ResolvedBounds,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, Thermal,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
use crate::indexer::{AnticipatedPlants, HydroCellIndex};
use crate::lead_time::{AnticipatedResolution, SpreadResolution};
use crate::lp::builder::{ResolvedTables, TemplateBuildCtx};
use crate::resolved_parameters::ResolvedParameters;
use crate::time_value::{PostStudyResolved, TimeValue};

/// Owns every value a [`TemplateBuildCtx`] borrows, or holds by value, so a
/// test builds one through [`Self::ctx`] instead of hand-writing its own
/// field-by-field construction — the same shape `columns.rs`'s
/// `InteriorStorageFixtures` and `generic_constraints::tests::ResolverFixture`
/// each duplicate independently. `ctx()` derives every position map and
/// entity count from its own slices, the way `build_template_build_ctx` does;
/// every other field is copied through unchanged.
pub(crate) struct CtxFixture {
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
    pub(crate) max_par_order: usize,
    pub(crate) anticipated_lead_stages: Vec<usize>,
    pub(crate) anticipated_plants: AnticipatedPlants,
    pub(crate) anticipated_resolution: AnticipatedResolution,
    pub(crate) has_penalty: bool,
    pub(crate) time_value: TimeValue,
    pub(crate) filling_v_target: BTreeMap<(usize, i32), f64>,
    pub(crate) arc_stage_weights: HashMap<usize, Vec<Vec<f64>>>,
    pub(crate) arc_spread_chrono: HashMap<usize, Vec<Option<SpreadResolution>>>,
    pub(crate) arc_arrival_density: HashMap<usize, Vec<Option<Vec<f64>>>>,
    pub(crate) per_stage_mask: Vec<Vec<usize>>,
}

impl Default for CtxFixture {
    fn default() -> Self {
        Self {
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
            production_models: ProductionModelSet::new(Vec::new(), 0, 0),
            evaporation_models: EvaporationModelSet::new(Vec::new()),
            generic_constraints: Vec::new(),
            non_controllable_sources: Vec::new(),
            pumping_stations: Vec::new(),
            contracts: Vec::new(),
            diversion_upstream: HashMap::new(),
            max_par_order: 0,
            anticipated_lead_stages: Vec::new(),
            anticipated_plants: AnticipatedPlants::default(),
            anticipated_resolution: AnticipatedResolution::default(),
            has_penalty: false,
            time_value: TimeValue::from_parts(
                Vec::new(),
                vec![1.0],
                vec![744.0],
                vec![0],
                PostStudyResolved::default(),
            ),
            filling_v_target: BTreeMap::new(),
            arc_stage_weights: HashMap::new(),
            arc_spread_chrono: HashMap::new(),
            arc_arrival_density: HashMap::new(),
            per_stage_mask: Vec::new(),
        }
    }
}

impl CtxFixture {
    /// Derives every position map and entity count from this fixture's own
    /// slices, the way `build_template_build_ctx` does; every other field is
    /// copied through unchanged. A test whose original literal set one of the
    /// derived fields to a value the slices disagree with restores it by
    /// mutating the returned context's field.
    pub(crate) fn ctx(&self) -> TemplateBuildCtx<'_> {
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
            hydro_pos: self
                .hydros
                .iter()
                .enumerate()
                .map(|(i, h)| (h.id, i))
                .collect(),
            thermal_pos: self
                .thermals
                .iter()
                .enumerate()
                .map(|(i, t)| (t.id, i))
                .collect(),
            line_pos: self
                .lines
                .iter()
                .enumerate()
                .map(|(i, l)| (l.id, i))
                .collect(),
            bus_pos: self
                .buses
                .iter()
                .enumerate()
                .map(|(i, b)| (b.id, i))
                .collect(),
            par_lp: &self.par_lp,
            production_models: &self.production_models,
            evaporation_models: &self.evaporation_models,
            generic_constraints: &self.generic_constraints,
            non_controllable_sources: &self.non_controllable_sources,
            pumping_stations: &self.pumping_stations,
            pumping_pos: self
                .pumping_stations
                .iter()
                .enumerate()
                .map(|(i, p)| (p.id, i))
                .collect(),
            n_pumping: self.pumping_stations.len(),
            contracts: &self.contracts,
            contract_pos: self
                .contracts
                .iter()
                .enumerate()
                .map(|(i, c)| (c.id, i))
                .collect(),
            n_contract_import: self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Import)
                .count(),
            n_contract_export: self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Export)
                .count(),
            diversion_upstream: self.diversion_upstream.clone(),
            n_hydros: self.hydros.len(),
            n_thermals: self.thermals.len(),
            n_lines: self.lines.len(),
            n_buses: self.buses.len(),
            max_par_order: self.max_par_order,
            n_anticipated: self.anticipated_plants.len(),
            anticipated_lead_stages: self.anticipated_lead_stages.clone(),
            anticipated_plants: &self.anticipated_plants,
            anticipated_resolution: self.anticipated_resolution.clone(),
            has_penalty: self.has_penalty,
            time_value: &self.time_value,
            filling_v_target: self.filling_v_target.clone(),
            arc_stage_weights: self.arc_stage_weights.clone(),
            arc_spread_chrono: self.arc_spread_chrono.clone(),
            arc_arrival_density: self.arc_arrival_density.clone(),
            per_stage_mask: self.per_stage_mask.clone(),
        }
    }
}
