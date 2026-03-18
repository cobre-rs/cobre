//! Top-level system struct and builder.
//!
//! The `System` struct is the top-level in-memory representation of a fully loaded,
//! validated, and resolved case. It is produced by `cobre-io::load_case()` and consumed
//! by solvers and analysis tools (e.g., optimization, simulation, power flow).
//!
//! All entity collections in `System` are stored in canonical ID-sorted order to ensure
//! declaration-order invariance: results are bit-for-bit identical regardless of input
//! entity ordering. See the design principles spec for details.

use std::collections::{HashMap, HashSet};

use crate::{
    Bus, CascadeTopology, CorrelationModel, EnergyContract, EntityId, GenericConstraint, Hydro,
    InflowModel, InitialConditions, Line, LoadModel, NetworkTopology, NonControllableSource,
    PolicyGraph, PumpingStation, ResolvedBounds, ResolvedPenalties, ScenarioSource, Stage, Thermal,
    ValidationError,
};

/// Top-level system representation.
///
/// Produced by `cobre-io::load_case()` or [`SystemBuilder`] in tests.
/// Consumed by solvers and analysis tools via shared reference.
/// Immutable and thread-safe after construction.
///
/// Entity collections are in canonical order (sorted by [`EntityId`]'s inner `i32`).
/// Lookup indices provide O(1) access by [`EntityId`].
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "Main Bus".to_string(),
///     deficit_segments: vec![],
///     excess_cost: 0.0,
/// };
///
/// let system = SystemBuilder::new()
///     .buses(vec![bus])
///     .build()
///     .expect("valid system");
///
/// assert_eq!(system.n_buses(), 1);
/// assert!(system.bus(EntityId(1)).is_some());
/// ```
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct System {
    // Entity collections (canonical ordering by ID)
    buses: Vec<Bus>,
    lines: Vec<Line>,
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    pumping_stations: Vec<PumpingStation>,
    contracts: Vec<EnergyContract>,
    non_controllable_sources: Vec<NonControllableSource>,

    // O(1) lookup indices (entity ID -> position in collection) -- private.
    // Per spec SS6.2: HashMap lookup indices are NOT serialized. After deserialization
    // the caller must invoke `rebuild_indices()` to restore O(1) lookup capability.
    #[cfg_attr(feature = "serde", serde(skip))]
    bus_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    line_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    hydro_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    thermal_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pumping_station_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    contract_index: HashMap<EntityId, usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    non_controllable_source_index: HashMap<EntityId, usize>,

    // Topology
    /// Resolved hydro cascade graph.
    cascade: CascadeTopology,
    /// Resolved transmission network topology.
    network: NetworkTopology,

    // Temporal domain
    /// Ordered list of stages (study + pre-study), sorted by `id` (canonical order).
    stages: Vec<Stage>,
    /// Policy graph defining stage transitions, horizon type, and discount rate.
    policy_graph: PolicyGraph,

    // Stage O(1) lookup index (stage ID -> position in stages vec).
    // Stage IDs are `i32` (pre-study stages have negative IDs).
    // Not serialized; rebuilt via `rebuild_indices()`.
    #[cfg_attr(feature = "serde", serde(skip))]
    stage_index: HashMap<i32, usize>,

    // Resolved tables (populated by cobre-io after penalty/bound cascade)
    /// Pre-resolved penalty values for all entities across all stages.
    penalties: ResolvedPenalties,
    /// Pre-resolved bound values for all entities across all stages.
    bounds: ResolvedBounds,

    // Scenario pipeline data (raw parameters loaded by cobre-io)
    /// PAR(p) inflow model parameters, one entry per (hydro, stage) pair.
    inflow_models: Vec<InflowModel>,
    /// Seasonal load statistics, one entry per (bus, stage) pair.
    load_models: Vec<LoadModel>,
    /// Correlation model for stochastic inflow/load generation.
    correlation: CorrelationModel,

    // Study state
    /// Initial reservoir storage levels at the start of the study.
    initial_conditions: InitialConditions,
    /// User-defined generic linear constraints, sorted by `id`.
    generic_constraints: Vec<GenericConstraint>,
    /// Top-level scenario source configuration (sampling scheme, seed).
    scenario_source: ScenarioSource,
}

// Compile-time check that System is Send + Sync.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    const fn check() {
        assert_send_sync::<System>();
    }
    let _ = check;
};

impl System {
    /// Returns all buses in canonical ID order.
    #[must_use]
    pub fn buses(&self) -> &[Bus] {
        &self.buses
    }

    /// Returns all lines in canonical ID order.
    #[must_use]
    pub fn lines(&self) -> &[Line] {
        &self.lines
    }

    /// Returns all hydro plants in canonical ID order.
    #[must_use]
    pub fn hydros(&self) -> &[Hydro] {
        &self.hydros
    }

    /// Returns all thermal plants in canonical ID order.
    #[must_use]
    pub fn thermals(&self) -> &[Thermal] {
        &self.thermals
    }

    /// Returns all pumping stations in canonical ID order.
    #[must_use]
    pub fn pumping_stations(&self) -> &[PumpingStation] {
        &self.pumping_stations
    }

    /// Returns all energy contracts in canonical ID order.
    #[must_use]
    pub fn contracts(&self) -> &[EnergyContract] {
        &self.contracts
    }

    /// Returns all non-controllable sources in canonical ID order.
    #[must_use]
    pub fn non_controllable_sources(&self) -> &[NonControllableSource] {
        &self.non_controllable_sources
    }

    /// Returns the number of buses in the system.
    #[must_use]
    pub fn n_buses(&self) -> usize {
        self.buses.len()
    }

    /// Returns the number of lines in the system.
    #[must_use]
    pub fn n_lines(&self) -> usize {
        self.lines.len()
    }

    /// Returns the number of hydro plants in the system.
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.hydros.len()
    }

    /// Returns the number of thermal plants in the system.
    #[must_use]
    pub fn n_thermals(&self) -> usize {
        self.thermals.len()
    }

    /// Returns the number of pumping stations in the system.
    #[must_use]
    pub fn n_pumping_stations(&self) -> usize {
        self.pumping_stations.len()
    }

    /// Returns the number of energy contracts in the system.
    #[must_use]
    pub fn n_contracts(&self) -> usize {
        self.contracts.len()
    }

    /// Returns the number of non-controllable sources in the system.
    #[must_use]
    pub fn n_non_controllable_sources(&self) -> usize {
        self.non_controllable_sources.len()
    }

    /// Returns the bus with the given ID, or `None` if not found.
    #[must_use]
    pub fn bus(&self, id: EntityId) -> Option<&Bus> {
        self.bus_index.get(&id).map(|&i| &self.buses[i])
    }

    /// Returns the line with the given ID, or `None` if not found.
    #[must_use]
    pub fn line(&self, id: EntityId) -> Option<&Line> {
        self.line_index.get(&id).map(|&i| &self.lines[i])
    }

    /// Returns the hydro plant with the given ID, or `None` if not found.
    #[must_use]
    pub fn hydro(&self, id: EntityId) -> Option<&Hydro> {
        self.hydro_index.get(&id).map(|&i| &self.hydros[i])
    }

    /// Returns the thermal plant with the given ID, or `None` if not found.
    #[must_use]
    pub fn thermal(&self, id: EntityId) -> Option<&Thermal> {
        self.thermal_index.get(&id).map(|&i| &self.thermals[i])
    }

    /// Returns the pumping station with the given ID, or `None` if not found.
    #[must_use]
    pub fn pumping_station(&self, id: EntityId) -> Option<&PumpingStation> {
        self.pumping_station_index
            .get(&id)
            .map(|&i| &self.pumping_stations[i])
    }

    /// Returns the energy contract with the given ID, or `None` if not found.
    #[must_use]
    pub fn contract(&self, id: EntityId) -> Option<&EnergyContract> {
        self.contract_index.get(&id).map(|&i| &self.contracts[i])
    }

    /// Returns the non-controllable source with the given ID, or `None` if not found.
    #[must_use]
    pub fn non_controllable_source(&self, id: EntityId) -> Option<&NonControllableSource> {
        self.non_controllable_source_index
            .get(&id)
            .map(|&i| &self.non_controllable_sources[i])
    }

    /// Returns a reference to the hydro cascade topology.
    #[must_use]
    pub fn cascade(&self) -> &CascadeTopology {
        &self.cascade
    }

    /// Returns a reference to the transmission network topology.
    #[must_use]
    pub fn network(&self) -> &NetworkTopology {
        &self.network
    }

    /// Returns all stages in canonical ID order (study and pre-study stages).
    #[must_use]
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Returns the number of stages (study and pre-study) in the system.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// Returns the stage with the given stage ID, or `None` if not found.
    ///
    /// Stage IDs are `i32`. Study stages have non-negative IDs; pre-study
    /// stages (used only for PAR model lag initialization) have negative IDs.
    #[must_use]
    pub fn stage(&self, id: i32) -> Option<&Stage> {
        self.stage_index.get(&id).map(|&i| &self.stages[i])
    }

    /// Returns a reference to the policy graph.
    #[must_use]
    pub fn policy_graph(&self) -> &PolicyGraph {
        &self.policy_graph
    }

    /// Returns a reference to the pre-resolved penalty table.
    #[must_use]
    pub fn penalties(&self) -> &ResolvedPenalties {
        &self.penalties
    }

    /// Returns a reference to the pre-resolved bounds table.
    #[must_use]
    pub fn bounds(&self) -> &ResolvedBounds {
        &self.bounds
    }

    /// Returns all PAR(p) inflow models in canonical order (by hydro ID, then stage ID).
    #[must_use]
    pub fn inflow_models(&self) -> &[InflowModel] {
        &self.inflow_models
    }

    /// Returns all load models in canonical order (by bus ID, then stage ID).
    #[must_use]
    pub fn load_models(&self) -> &[LoadModel] {
        &self.load_models
    }

    /// Returns a reference to the correlation model.
    #[must_use]
    pub fn correlation(&self) -> &CorrelationModel {
        &self.correlation
    }

    /// Returns a reference to the initial conditions.
    #[must_use]
    pub fn initial_conditions(&self) -> &InitialConditions {
        &self.initial_conditions
    }

    /// Returns all generic constraints in canonical ID order.
    #[must_use]
    pub fn generic_constraints(&self) -> &[GenericConstraint] {
        &self.generic_constraints
    }

    /// Returns a reference to the scenario source configuration.
    #[must_use]
    pub fn scenario_source(&self) -> &ScenarioSource {
        &self.scenario_source
    }

    /// Replace the scenario models and correlation on this `System`, returning a new
    /// `System` with updated fields and all other fields preserved.
    ///
    /// This is the only supported way to update `inflow_models` and `correlation`
    /// after a `System` has been constructed — the fields are not public outside
    /// this crate. All entity collections, topology, stages, penalties, bounds, and
    /// study state are preserved unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::{EntityId, SystemBuilder};
    /// use cobre_core::scenario::{InflowModel, CorrelationModel};
    ///
    /// let system = SystemBuilder::new().build().expect("valid system");
    /// let model = InflowModel {
    ///     hydro_id: EntityId(1),
    ///     stage_id: 0,
    ///     mean_m3s: 100.0,
    ///     std_m3s: 10.0,
    ///     ar_coefficients: vec![],
    ///     residual_std_ratio: 1.0,
    /// };
    /// let updated = system.with_scenario_models(vec![model], CorrelationModel::default());
    /// assert_eq!(updated.inflow_models().len(), 1);
    /// ```
    #[must_use]
    pub fn with_scenario_models(
        mut self,
        inflow_models: Vec<InflowModel>,
        correlation: CorrelationModel,
    ) -> Self {
        self.inflow_models = inflow_models;
        self.correlation = correlation;
        self
    }

    /// Rebuild all O(1) lookup indices from the entity collections.
    ///
    /// Required after deserialization: the `HashMap` lookup indices are not serialized
    /// (per spec SS6.2 — they are derived from the entity collections). After
    /// deserializing a `System` from JSON or any other format, call this method once
    /// to restore O(1) access via [`bus`](Self::bus), [`hydro`](Self::hydro), etc.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "serde")]
    /// # {
    /// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
    ///
    /// let system = SystemBuilder::new()
    ///     .buses(vec![Bus {
    ///         id: EntityId(1),
    ///         name: "A".to_string(),
    ///         deficit_segments: vec![],
    ///         excess_cost: 0.0,
    ///     }])
    ///     .build()
    ///     .expect("valid system");
    ///
    /// let json = serde_json::to_string(&system).unwrap();
    /// let mut deserialized: cobre_core::System = serde_json::from_str(&json).unwrap();
    /// deserialized.rebuild_indices();
    ///
    /// // O(1) lookup now works after index rebuild.
    /// assert!(deserialized.bus(EntityId(1)).is_some());
    /// # }
    /// ```
    pub fn rebuild_indices(&mut self) {
        self.bus_index = build_index(&self.buses);
        self.line_index = build_index(&self.lines);
        self.hydro_index = build_index(&self.hydros);
        self.thermal_index = build_index(&self.thermals);
        self.pumping_station_index = build_index(&self.pumping_stations);
        self.contract_index = build_index(&self.contracts);
        self.non_controllable_source_index = build_index(&self.non_controllable_sources);
        self.stage_index = build_stage_index(&self.stages);
    }
}

/// Builder for constructing a validated, immutable [`System`].
///
/// Accepts entity collections, sorts entities by ID, checks for duplicate IDs,
/// builds topology, and returns the [`System`]. All entity collections default to
/// empty; only supply the collections your test case requires.
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
///
/// let system = SystemBuilder::new()
///     .buses(vec![
///         Bus { id: EntityId(2), name: "B".to_string(), deficit_segments: vec![], excess_cost: 0.0 },
///         Bus { id: EntityId(1), name: "A".to_string(), deficit_segments: vec![], excess_cost: 0.0 },
///     ])
///     .build()
///     .expect("valid system");
///
/// // Canonical ordering: id=1 comes before id=2.
/// assert_eq!(system.buses()[0].id, EntityId(1));
/// assert_eq!(system.buses()[1].id, EntityId(2));
/// ```
pub struct SystemBuilder {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    pumping_stations: Vec<PumpingStation>,
    contracts: Vec<EnergyContract>,
    non_controllable_sources: Vec<NonControllableSource>,
    // New fields from tickets 004-007
    stages: Vec<Stage>,
    policy_graph: PolicyGraph,
    penalties: ResolvedPenalties,
    bounds: ResolvedBounds,
    inflow_models: Vec<InflowModel>,
    load_models: Vec<LoadModel>,
    correlation: CorrelationModel,
    initial_conditions: InitialConditions,
    generic_constraints: Vec<GenericConstraint>,
    scenario_source: ScenarioSource,
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemBuilder {
    /// Create a new empty builder. All entity collections start empty.
    ///
    /// New fields default to empty/default values so that
    /// pre-existing tests continue to work without modification.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buses: Vec::new(),
            lines: Vec::new(),
            hydros: Vec::new(),
            thermals: Vec::new(),
            pumping_stations: Vec::new(),
            contracts: Vec::new(),
            non_controllable_sources: Vec::new(),
            stages: Vec::new(),
            policy_graph: PolicyGraph::default(),
            penalties: ResolvedPenalties::empty(),
            bounds: ResolvedBounds::empty(),
            inflow_models: Vec::new(),
            load_models: Vec::new(),
            correlation: CorrelationModel::default(),
            initial_conditions: InitialConditions::default(),
            generic_constraints: Vec::new(),
            scenario_source: ScenarioSource::default(),
        }
    }

    /// Set the bus collection.
    #[must_use]
    pub fn buses(mut self, buses: Vec<Bus>) -> Self {
        self.buses = buses;
        self
    }

    /// Set the line collection.
    #[must_use]
    pub fn lines(mut self, lines: Vec<Line>) -> Self {
        self.lines = lines;
        self
    }

    /// Set the hydro plant collection.
    #[must_use]
    pub fn hydros(mut self, hydros: Vec<Hydro>) -> Self {
        self.hydros = hydros;
        self
    }

    /// Set the thermal plant collection.
    #[must_use]
    pub fn thermals(mut self, thermals: Vec<Thermal>) -> Self {
        self.thermals = thermals;
        self
    }

    /// Set the pumping station collection.
    #[must_use]
    pub fn pumping_stations(mut self, stations: Vec<PumpingStation>) -> Self {
        self.pumping_stations = stations;
        self
    }

    /// Set the energy contract collection.
    #[must_use]
    pub fn contracts(mut self, contracts: Vec<EnergyContract>) -> Self {
        self.contracts = contracts;
        self
    }

    /// Set the non-controllable source collection.
    #[must_use]
    pub fn non_controllable_sources(mut self, sources: Vec<NonControllableSource>) -> Self {
        self.non_controllable_sources = sources;
        self
    }

    /// Set the stage collection (study and pre-study stages).
    ///
    /// Stages are sorted by `id` in [`build`](Self::build) to canonical order.
    #[must_use]
    pub fn stages(mut self, stages: Vec<Stage>) -> Self {
        self.stages = stages;
        self
    }

    /// Set the policy graph.
    #[must_use]
    pub fn policy_graph(mut self, policy_graph: PolicyGraph) -> Self {
        self.policy_graph = policy_graph;
        self
    }

    /// Set the pre-resolved penalty table.
    ///
    /// Populated by `cobre-io` after the three-tier penalty cascade is applied.
    #[must_use]
    pub fn penalties(mut self, penalties: ResolvedPenalties) -> Self {
        self.penalties = penalties;
        self
    }

    /// Set the pre-resolved bounds table.
    ///
    /// Populated by `cobre-io` after base bounds are overlaid with stage overrides.
    #[must_use]
    pub fn bounds(mut self, bounds: ResolvedBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Set the PAR(p) inflow model collection.
    #[must_use]
    pub fn inflow_models(mut self, inflow_models: Vec<InflowModel>) -> Self {
        self.inflow_models = inflow_models;
        self
    }

    /// Set the load model collection.
    #[must_use]
    pub fn load_models(mut self, load_models: Vec<LoadModel>) -> Self {
        self.load_models = load_models;
        self
    }

    /// Set the correlation model.
    #[must_use]
    pub fn correlation(mut self, correlation: CorrelationModel) -> Self {
        self.correlation = correlation;
        self
    }

    /// Set the initial conditions.
    #[must_use]
    pub fn initial_conditions(mut self, initial_conditions: InitialConditions) -> Self {
        self.initial_conditions = initial_conditions;
        self
    }

    /// Set the generic constraint collection.
    ///
    /// Constraints are sorted by `id` in [`build`](Self::build) to canonical order.
    #[must_use]
    pub fn generic_constraints(mut self, generic_constraints: Vec<GenericConstraint>) -> Self {
        self.generic_constraints = generic_constraints;
        self
    }

    /// Set the scenario source configuration.
    #[must_use]
    pub fn scenario_source(mut self, scenario_source: ScenarioSource) -> Self {
        self.scenario_source = scenario_source;
        self
    }

    /// Build the [`System`].
    ///
    /// Sorts all entity collections by [`EntityId`] (canonical ordering).
    /// Checks for duplicate IDs within each collection.
    /// Validates all cross-reference fields (e.g., `bus_id`, `downstream_id`) against
    /// the appropriate index to ensure every referenced entity exists.
    /// Builds [`CascadeTopology`] and [`NetworkTopology`].
    /// Validates the cascade graph for cycles and checks hydro filling configurations.
    /// Constructs lookup indices.
    ///
    /// Returns `Err` with a list of all validation errors found across all collections.
    /// All invalid references across all entity types are collected before returning —
    /// no short-circuiting on first error.
    ///
    /// # Errors
    ///
    /// Returns `Err(Vec<ValidationError>)` if:
    /// - Duplicate IDs are detected in any entity collection.
    /// - Any cross-reference field refers to an entity ID that does not exist.
    /// - The hydro cascade graph contains a cycle.
    /// - Any hydro filling configuration is invalid (non-positive inflow or missing
    ///   `entry_stage_id`).
    ///
    /// All errors across all collections are reported together.
    pub fn build(mut self) -> Result<System, Vec<ValidationError>> {
        self.buses.sort_by_key(|e| e.id.0);
        self.lines.sort_by_key(|e| e.id.0);
        self.hydros.sort_by_key(|e| e.id.0);
        self.thermals.sort_by_key(|e| e.id.0);
        self.pumping_stations.sort_by_key(|e| e.id.0);
        self.contracts.sort_by_key(|e| e.id.0);
        self.non_controllable_sources.sort_by_key(|e| e.id.0);
        self.stages.sort_by_key(|s| s.id);
        self.generic_constraints.sort_by_key(|c| c.id.0);

        let mut errors: Vec<ValidationError> = Vec::new();
        check_duplicates(&self.buses, "Bus", &mut errors);
        check_duplicates(&self.lines, "Line", &mut errors);
        check_duplicates(&self.hydros, "Hydro", &mut errors);
        check_duplicates(&self.thermals, "Thermal", &mut errors);
        check_duplicates(&self.pumping_stations, "PumpingStation", &mut errors);
        check_duplicates(&self.contracts, "EnergyContract", &mut errors);
        check_duplicates(
            &self.non_controllable_sources,
            "NonControllableSource",
            &mut errors,
        );

        if !errors.is_empty() {
            return Err(errors);
        }

        let bus_index = build_index(&self.buses);
        let line_index = build_index(&self.lines);
        let hydro_index = build_index(&self.hydros);
        let thermal_index = build_index(&self.thermals);
        let pumping_station_index = build_index(&self.pumping_stations);
        let contract_index = build_index(&self.contracts);
        let non_controllable_source_index = build_index(&self.non_controllable_sources);

        validate_cross_references(
            &self.lines,
            &self.hydros,
            &self.thermals,
            &self.pumping_stations,
            &self.contracts,
            &self.non_controllable_sources,
            &bus_index,
            &hydro_index,
            &mut errors,
        );

        if !errors.is_empty() {
            return Err(errors);
        }

        let cascade = CascadeTopology::build(&self.hydros);

        if cascade.topological_order().len() < self.hydros.len() {
            let in_topo: HashSet<EntityId> = cascade.topological_order().iter().copied().collect();
            let mut cycle_ids: Vec<EntityId> = self
                .hydros
                .iter()
                .map(|h| h.id)
                .filter(|id| !in_topo.contains(id))
                .collect();
            cycle_ids.sort_by_key(|id| id.0);
            errors.push(ValidationError::CascadeCycle { cycle_ids });
        }

        validate_filling_configs(&self.hydros, &mut errors);

        if !errors.is_empty() {
            return Err(errors);
        }

        let network = NetworkTopology::build(
            &self.buses,
            &self.lines,
            &self.hydros,
            &self.thermals,
            &self.non_controllable_sources,
            &self.contracts,
            &self.pumping_stations,
        );

        let stage_index = build_stage_index(&self.stages);

        Ok(System {
            buses: self.buses,
            lines: self.lines,
            hydros: self.hydros,
            thermals: self.thermals,
            pumping_stations: self.pumping_stations,
            contracts: self.contracts,
            non_controllable_sources: self.non_controllable_sources,
            bus_index,
            line_index,
            hydro_index,
            thermal_index,
            pumping_station_index,
            contract_index,
            non_controllable_source_index,
            cascade,
            network,
            stages: self.stages,
            policy_graph: self.policy_graph,
            stage_index,
            penalties: self.penalties,
            bounds: self.bounds,
            inflow_models: self.inflow_models,
            load_models: self.load_models,
            correlation: self.correlation,
            initial_conditions: self.initial_conditions,
            generic_constraints: self.generic_constraints,
            scenario_source: self.scenario_source,
        })
    }
}

trait HasId {
    fn entity_id(&self) -> EntityId;
}

impl HasId for Bus {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Line {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Hydro {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for Thermal {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for PumpingStation {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for EnergyContract {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}
impl HasId for NonControllableSource {
    fn entity_id(&self) -> EntityId {
        self.id
    }
}

fn build_index<T: HasId>(entities: &[T]) -> HashMap<EntityId, usize> {
    let mut index = HashMap::with_capacity(entities.len());
    for (i, entity) in entities.iter().enumerate() {
        index.insert(entity.entity_id(), i);
    }
    index
}

/// Build a stage lookup index from the canonical-ordered stages vec.
///
/// Keys are `i32` stage IDs (which can be negative for pre-study stages).
fn build_stage_index(stages: &[Stage]) -> HashMap<i32, usize> {
    let mut index = HashMap::with_capacity(stages.len());
    for (i, stage) in stages.iter().enumerate() {
        index.insert(stage.id, i);
    }
    index
}

fn check_duplicates<T: HasId>(
    entities: &[T],
    entity_type: &'static str,
    errors: &mut Vec<ValidationError>,
) {
    for window in entities.windows(2) {
        if window[0].entity_id() == window[1].entity_id() {
            errors.push(ValidationError::DuplicateId {
                entity_type,
                id: window[0].entity_id(),
            });
        }
    }
}

/// Validate all cross-reference fields across entity collections.
///
/// Checks every entity field that references another entity by [`EntityId`]
/// against the appropriate index. All invalid references are appended to
/// `errors` — no short-circuiting on first error.
///
/// This function runs after duplicate checking passes and after indices are
/// built, but before topology construction.
#[allow(clippy::too_many_arguments)]
fn validate_cross_references(
    lines: &[Line],
    hydros: &[Hydro],
    thermals: &[Thermal],
    pumping_stations: &[PumpingStation],
    contracts: &[EnergyContract],
    non_controllable_sources: &[NonControllableSource],
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    validate_line_refs(lines, bus_index, errors);
    validate_hydro_refs(hydros, bus_index, hydro_index, errors);
    validate_thermal_refs(thermals, bus_index, errors);
    validate_pumping_station_refs(pumping_stations, bus_index, hydro_index, errors);
    validate_contract_refs(contracts, bus_index, errors);
    validate_ncs_refs(non_controllable_sources, bus_index, errors);
}

fn validate_line_refs(
    lines: &[Line],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for line in lines {
        if !bus_index.contains_key(&line.source_bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Line",
                source_id: line.id,
                field_name: "source_bus_id",
                referenced_id: line.source_bus_id,
                expected_type: "Bus",
            });
        }
        if !bus_index.contains_key(&line.target_bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Line",
                source_id: line.id,
                field_name: "target_bus_id",
                referenced_id: line.target_bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_hydro_refs(
    hydros: &[Hydro],
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for hydro in hydros {
        if !bus_index.contains_key(&hydro.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                source_id: hydro.id,
                field_name: "bus_id",
                referenced_id: hydro.bus_id,
                expected_type: "Bus",
            });
        }
        if let Some(downstream_id) = hydro.downstream_id {
            if !hydro_index.contains_key(&downstream_id) {
                errors.push(ValidationError::InvalidReference {
                    source_entity_type: "Hydro",
                    source_id: hydro.id,
                    field_name: "downstream_id",
                    referenced_id: downstream_id,
                    expected_type: "Hydro",
                });
            }
        }
        if let Some(ref diversion) = hydro.diversion {
            if !hydro_index.contains_key(&diversion.downstream_id) {
                errors.push(ValidationError::InvalidReference {
                    source_entity_type: "Hydro",
                    source_id: hydro.id,
                    field_name: "diversion.downstream_id",
                    referenced_id: diversion.downstream_id,
                    expected_type: "Hydro",
                });
            }
        }
    }
}

fn validate_thermal_refs(
    thermals: &[Thermal],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for thermal in thermals {
        if !bus_index.contains_key(&thermal.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Thermal",
                source_id: thermal.id,
                field_name: "bus_id",
                referenced_id: thermal.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_pumping_station_refs(
    pumping_stations: &[PumpingStation],
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for ps in pumping_stations {
        if !bus_index.contains_key(&ps.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "bus_id",
                referenced_id: ps.bus_id,
                expected_type: "Bus",
            });
        }
        if !hydro_index.contains_key(&ps.source_hydro_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "source_hydro_id",
                referenced_id: ps.source_hydro_id,
                expected_type: "Hydro",
            });
        }
        if !hydro_index.contains_key(&ps.destination_hydro_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "PumpingStation",
                source_id: ps.id,
                field_name: "destination_hydro_id",
                referenced_id: ps.destination_hydro_id,
                expected_type: "Hydro",
            });
        }
    }
}

fn validate_contract_refs(
    contracts: &[EnergyContract],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for contract in contracts {
        if !bus_index.contains_key(&contract.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "EnergyContract",
                source_id: contract.id,
                field_name: "bus_id",
                referenced_id: contract.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

fn validate_ncs_refs(
    non_controllable_sources: &[NonControllableSource],
    bus_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for ncs in non_controllable_sources {
        if !bus_index.contains_key(&ncs.bus_id) {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "NonControllableSource",
                source_id: ncs.id,
                field_name: "bus_id",
                referenced_id: ncs.bus_id,
                expected_type: "Bus",
            });
        }
    }
}

/// Validate filling configurations for all hydros that have one.
///
/// For each hydro with `filling: Some(config)`:
/// - `filling_inflow_m3s` must be positive (> 0.0).
/// - `entry_stage_id` must be set (`Some`), since filling requires a known start stage.
///
/// All violations are appended to `errors` — no short-circuiting on first error.
fn validate_filling_configs(hydros: &[Hydro], errors: &mut Vec<ValidationError>) {
    for hydro in hydros {
        if let Some(filling) = &hydro.filling {
            if filling.filling_inflow_m3s.is_nan() || filling.filling_inflow_m3s <= 0.0 {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling_inflow_m3s must be positive".to_string(),
                });
            }
            if hydro.entry_stage_id.is_none() {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling requires entry_stage_id to be set".to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{ContractType, HydroGenerationModel, HydroPenalties, ThermalCostSegment};

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("bus-{id}"),
            deficit_segments: vec![],
            excess_cost: 0.0,
        }
    }

    fn make_line(id: i32, source_bus_id: i32, target_bus_id: i32) -> Line {
        crate::Line {
            id: EntityId(id),
            name: format!("line-{id}"),
            source_bus_id: EntityId(source_bus_id),
            target_bus_id: EntityId(target_bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    fn make_hydro_on_bus(id: i32, bus_id: i32) -> Hydro {
        let zero_penalties = HydroPenalties {
            spillage_cost: 0.0,
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
        };
        Hydro {
            id: EntityId(id),
            name: format!("hydro-{id}"),
            bus_id: EntityId(bus_id),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s: 1.0,
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        }
    }

    /// Creates a hydro on bus 0. Caller must supply `make_bus(0)`.
    fn make_hydro(id: i32) -> Hydro {
        make_hydro_on_bus(id, 0)
    }

    fn make_thermal_on_bus(id: i32, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("thermal-{id}"),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_segments: vec![ThermalCostSegment {
                capacity_mw: 100.0,
                cost_per_mwh: 50.0,
            }],
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            gnl_config: None,
        }
    }

    /// Creates a thermal on bus 0. Caller must supply `make_bus(0)`.
    fn make_thermal(id: i32) -> Thermal {
        make_thermal_on_bus(id, 0)
    }

    fn make_pumping_station_full(
        id: i32,
        bus_id: i32,
        source_hydro_id: i32,
        destination_hydro_id: i32,
    ) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("ps-{id}"),
            bus_id: EntityId(bus_id),
            source_hydro_id: EntityId(source_hydro_id),
            destination_hydro_id: EntityId(destination_hydro_id),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 10.0,
        }
    }

    fn make_pumping_station(id: i32) -> PumpingStation {
        make_pumping_station_full(id, 0, 0, 1)
    }

    fn make_contract_on_bus(id: i32, bus_id: i32) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("contract-{id}"),
            bus_id: EntityId(bus_id),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 100.0,
        }
    }

    fn make_contract(id: i32) -> EnergyContract {
        make_contract_on_bus(id, 0)
    }

    fn make_ncs_on_bus(id: i32, bus_id: i32) -> NonControllableSource {
        NonControllableSource {
            id: EntityId(id),
            name: format!("ncs-{id}"),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 50.0,
            curtailment_cost: 0.0,
        }
    }

    fn make_ncs(id: i32) -> NonControllableSource {
        make_ncs_on_bus(id, 0)
    }

    #[test]
    fn test_empty_system() {
        let system = SystemBuilder::new().build().expect("empty system is valid");
        assert_eq!(system.n_buses(), 0);
        assert_eq!(system.n_lines(), 0);
        assert_eq!(system.n_hydros(), 0);
        assert_eq!(system.n_thermals(), 0);
        assert_eq!(system.n_pumping_stations(), 0);
        assert_eq!(system.n_contracts(), 0);
        assert_eq!(system.n_non_controllable_sources(), 0);
        assert!(system.buses().is_empty());
        assert!(system.cascade().is_empty());
    }

    #[test]
    fn test_canonical_ordering() {
        // Provide buses in reverse order: id=2, id=1, id=0
        let system = SystemBuilder::new()
            .buses(vec![make_bus(2), make_bus(1), make_bus(0)])
            .build()
            .expect("valid system");

        assert_eq!(system.buses()[0].id, EntityId(0));
        assert_eq!(system.buses()[1].id, EntityId(1));
        assert_eq!(system.buses()[2].id, EntityId(2));
    }

    #[test]
    fn test_lookup_by_id() {
        // Hydros reference bus id=0; supply it so cross-reference validation passes.
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![make_hydro(10), make_hydro(5), make_hydro(20)])
            .build()
            .expect("valid system");

        assert_eq!(system.hydro(EntityId(5)).map(|h| h.id), Some(EntityId(5)));
        assert_eq!(system.hydro(EntityId(10)).map(|h| h.id), Some(EntityId(10)));
        assert_eq!(system.hydro(EntityId(20)).map(|h| h.id), Some(EntityId(20)));
    }

    #[test]
    fn test_lookup_missing_id() {
        // Hydros reference bus id=0; supply it so cross-reference validation passes.
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![make_hydro(1), make_hydro(2)])
            .build()
            .expect("valid system");

        assert!(system.hydro(EntityId(999)).is_none());
    }

    #[test]
    fn test_count_queries() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1)])
            .lines(vec![make_line(0, 0, 1)])
            .hydros(vec![make_hydro(0), make_hydro(1), make_hydro(2)])
            .thermals(vec![make_thermal(0)])
            .pumping_stations(vec![make_pumping_station(0)])
            .contracts(vec![make_contract(0), make_contract(1)])
            .non_controllable_sources(vec![make_ncs(0)])
            .build()
            .expect("valid system");

        assert_eq!(system.n_buses(), 2);
        assert_eq!(system.n_lines(), 1);
        assert_eq!(system.n_hydros(), 3);
        assert_eq!(system.n_thermals(), 1);
        assert_eq!(system.n_pumping_stations(), 1);
        assert_eq!(system.n_contracts(), 2);
        assert_eq!(system.n_non_controllable_sources(), 1);
    }

    #[test]
    fn test_slice_accessors() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1), make_bus(2)])
            .build()
            .expect("valid system");

        let buses = system.buses();
        assert_eq!(buses.len(), 3);
        assert_eq!(buses[0].id, EntityId(0));
        assert_eq!(buses[1].id, EntityId(1));
        assert_eq!(buses[2].id, EntityId(2));
    }

    #[test]
    fn test_duplicate_id_error() {
        // Two buses with the same id=0 must yield an Err.
        let result = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(0)])
            .build();

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::DuplicateId {
                entity_type: "Bus",
                id: EntityId(0),
            }
        )));
    }

    #[test]
    fn test_multiple_duplicate_errors() {
        // Duplicates in both buses (id=0) and thermals (id=5) must both be reported.
        let result = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(0)])
            .thermals(vec![make_thermal(5), make_thermal(5)])
            .build();

        assert!(result.is_err());
        let errors = result.unwrap_err();

        let has_bus_dup = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::DuplicateId {
                    entity_type: "Bus",
                    ..
                }
            )
        });
        let has_thermal_dup = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::DuplicateId {
                    entity_type: "Thermal",
                    ..
                }
            )
        });
        assert!(has_bus_dup, "expected Bus duplicate error");
        assert!(has_thermal_dup, "expected Thermal duplicate error");
    }

    #[test]
    fn test_send_sync() {
        fn require_send_sync<T: Send + Sync>(_: T) {}
        let system = SystemBuilder::new().build().expect("valid system");
        require_send_sync(system);
    }

    #[test]
    fn test_cascade_accessible() {
        // Hydros reference bus id=0; supply it so cross-reference validation passes.
        let mut h0 = make_hydro_on_bus(0, 0);
        h0.downstream_id = Some(EntityId(1));
        let mut h1 = make_hydro_on_bus(1, 0);
        h1.downstream_id = Some(EntityId(2));
        let h2 = make_hydro_on_bus(2, 0);

        let system = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .hydros(vec![h0, h1, h2])
            .build()
            .expect("valid system");

        let order = system.cascade().topological_order();
        assert!(!order.is_empty(), "topological order must be non-empty");
        let pos_0 = order
            .iter()
            .position(|&id| id == EntityId(0))
            .expect("EntityId(0) must be in topological order");
        let pos_2 = order
            .iter()
            .position(|&id| id == EntityId(2))
            .expect("EntityId(2) must be in topological order");
        assert!(pos_0 < pos_2, "EntityId(0) must precede EntityId(2)");
    }

    #[test]
    fn test_network_accessible() {
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1)])
            .lines(vec![make_line(0, 0, 1)])
            .build()
            .expect("valid system");

        let connections = system.network().bus_lines(EntityId(0));
        assert!(!connections.is_empty(), "bus 0 must have connections");
        assert_eq!(connections[0].line_id, EntityId(0));
    }

    #[test]
    fn test_all_entity_lookups() {
        // Provide all buses and hydros that the other entities reference.
        // - Buses 0 and 1 are needed by all entities (lines, hydros, thermals, etc.)
        // - Hydros 0 and 1 are needed by the pumping station (source/destination)
        // - Hydro 3 is the entity under test (lookup by id=3), on bus 0
        let system = SystemBuilder::new()
            .buses(vec![make_bus(0), make_bus(1)])
            .lines(vec![make_line(2, 0, 1)])
            .hydros(vec![
                make_hydro_on_bus(0, 0),
                make_hydro_on_bus(1, 0),
                make_hydro_on_bus(3, 0),
            ])
            .thermals(vec![make_thermal(4)])
            .pumping_stations(vec![make_pumping_station(5)])
            .contracts(vec![make_contract(6)])
            .non_controllable_sources(vec![make_ncs(7)])
            .build()
            .expect("valid system");

        assert!(system.bus(EntityId(1)).is_some());
        assert!(system.line(EntityId(2)).is_some());
        assert!(system.hydro(EntityId(3)).is_some());
        assert!(system.thermal(EntityId(4)).is_some());
        assert!(system.pumping_station(EntityId(5)).is_some());
        assert!(system.contract(EntityId(6)).is_some());
        assert!(system.non_controllable_source(EntityId(7)).is_some());

        assert!(system.bus(EntityId(999)).is_none());
        assert!(system.line(EntityId(999)).is_none());
        assert!(system.hydro(EntityId(999)).is_none());
        assert!(system.thermal(EntityId(999)).is_none());
        assert!(system.pumping_station(EntityId(999)).is_none());
        assert!(system.contract(EntityId(999)).is_none());
        assert!(system.non_controllable_source(EntityId(999)).is_none());
    }

    #[test]
    fn test_default_builder() {
        let system = SystemBuilder::default()
            .build()
            .expect("default builder produces valid empty system");
        assert_eq!(system.n_buses(), 0);
    }

    // ---- Cross-reference validation tests -----------------------------------

    #[test]
    fn test_invalid_bus_reference_hydro() {
        // Hydro references bus id=99 which does not exist.
        let hydro = make_hydro_on_bus(1, 99);
        let result = SystemBuilder::new().hydros(vec![hydro]).build();

        assert!(result.is_err(), "expected Err for missing bus reference");
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "Hydro",
                    source_id: EntityId(1),
                    field_name: "bus_id",
                    referenced_id: EntityId(99),
                    expected_type: "Bus",
                }
            )),
            "expected InvalidReference for Hydro bus_id=99, got: {errors:?}"
        );
    }

    #[test]
    fn test_invalid_downstream_reference() {
        // Hydro references downstream hydro id=50 which does not exist.
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.downstream_id = Some(EntityId(50));

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_err(),
            "expected Err for missing downstream reference"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "Hydro",
                    source_id: EntityId(1),
                    field_name: "downstream_id",
                    referenced_id: EntityId(50),
                    expected_type: "Hydro",
                }
            )),
            "expected InvalidReference for Hydro downstream_id=50, got: {errors:?}"
        );
    }

    #[test]
    fn test_invalid_pumping_station_hydro_refs() {
        // Pumping station references source hydro id=77 which does not exist.
        let bus = make_bus(0);
        let dest_hydro = make_hydro(1);
        let ps = make_pumping_station_full(10, 0, 77, 1);

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![dest_hydro])
            .pumping_stations(vec![ps])
            .build();

        assert!(
            result.is_err(),
            "expected Err for missing source_hydro_id reference"
        );
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "PumpingStation",
                    source_id: EntityId(10),
                    field_name: "source_hydro_id",
                    referenced_id: EntityId(77),
                    expected_type: "Hydro",
                }
            )),
            "expected InvalidReference for PumpingStation source_hydro_id=77, got: {errors:?}"
        );
    }

    #[test]
    fn test_multiple_invalid_references_collected() {
        // A line with bad source_bus_id AND a thermal with bad bus_id.
        // Both errors must be reported (no short-circuiting).
        let line = make_line(1, 99, 0);
        let thermal = make_thermal_on_bus(2, 88);

        let result = SystemBuilder::new()
            .buses(vec![make_bus(0)])
            .lines(vec![line])
            .thermals(vec![thermal])
            .build();

        assert!(
            result.is_err(),
            "expected Err for multiple invalid references"
        );
        let errors = result.unwrap_err();

        let has_line_error = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "Line",
                    field_name: "source_bus_id",
                    referenced_id: EntityId(99),
                    ..
                }
            )
        });
        let has_thermal_error = errors.iter().any(|e| {
            matches!(
                e,
                ValidationError::InvalidReference {
                    source_entity_type: "Thermal",
                    field_name: "bus_id",
                    referenced_id: EntityId(88),
                    ..
                }
            )
        });

        assert!(
            has_line_error,
            "expected Line source_bus_id=99 error, got: {errors:?}"
        );
        assert!(
            has_thermal_error,
            "expected Thermal bus_id=88 error, got: {errors:?}"
        );
        assert!(
            errors.len() >= 2,
            "expected at least 2 errors, got {}: {errors:?}",
            errors.len()
        );
    }

    #[test]
    fn test_valid_cross_references_pass() {
        // All cross-references point to entities that exist — build must succeed.
        let bus_0 = make_bus(0);
        let bus_1 = make_bus(1);
        let h0 = make_hydro_on_bus(0, 0);
        let h1 = make_hydro_on_bus(1, 1);
        let mut h2 = make_hydro_on_bus(2, 0);
        h2.downstream_id = Some(EntityId(1));
        let line = make_line(10, 0, 1);
        let thermal = make_thermal_on_bus(20, 0);
        let ps = make_pumping_station_full(30, 0, 0, 1);
        let contract = make_contract_on_bus(40, 1);
        let ncs = make_ncs_on_bus(50, 0);

        let result = SystemBuilder::new()
            .buses(vec![bus_0, bus_1])
            .lines(vec![line])
            .hydros(vec![h0, h1, h2])
            .thermals(vec![thermal])
            .pumping_stations(vec![ps])
            .contracts(vec![contract])
            .non_controllable_sources(vec![ncs])
            .build();

        assert!(
            result.is_ok(),
            "expected Ok for all valid cross-references, got: {:?}",
            result.unwrap_err()
        );
        let system = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(system.n_buses(), 2);
        assert_eq!(system.n_hydros(), 3);
        assert_eq!(system.n_lines(), 1);
        assert_eq!(system.n_thermals(), 1);
        assert_eq!(system.n_pumping_stations(), 1);
        assert_eq!(system.n_contracts(), 1);
        assert_eq!(system.n_non_controllable_sources(), 1);
    }

    // ---- Cascade cycle detection tests --------------------------------------

    #[test]
    fn test_cascade_cycle_detected() {
        // Three-node cycle: A(0)->B(1)->C(2)->A(0).
        // All three reference a common bus (bus 0).
        let bus = make_bus(0);
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(1));
        let mut h1 = make_hydro(1);
        h1.downstream_id = Some(EntityId(2));
        let mut h2 = make_hydro(2);
        h2.downstream_id = Some(EntityId(0));

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0, h1, h2])
            .build();

        assert!(result.is_err(), "expected Err for 3-node cycle");
        let errors = result.unwrap_err();
        let cycle_error = errors
            .iter()
            .find(|e| matches!(e, ValidationError::CascadeCycle { .. }));
        assert!(
            cycle_error.is_some(),
            "expected CascadeCycle error, got: {errors:?}"
        );
        let ValidationError::CascadeCycle { cycle_ids } = cycle_error.unwrap() else {
            unreachable!()
        };
        assert_eq!(
            cycle_ids,
            &[EntityId(0), EntityId(1), EntityId(2)],
            "cycle_ids must be sorted ascending, got: {cycle_ids:?}"
        );
    }

    #[test]
    fn test_cascade_self_loop_detected() {
        // Single hydro pointing to itself: A(0)->A(0).
        let bus = make_bus(0);
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(0));

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0])
            .build();

        assert!(result.is_err(), "expected Err for self-loop");
        let errors = result.unwrap_err();
        let has_cycle = errors
            .iter()
            .any(|e| matches!(e, ValidationError::CascadeCycle { cycle_ids } if cycle_ids.contains(&EntityId(0))));
        assert!(
            has_cycle,
            "expected CascadeCycle containing EntityId(0), got: {errors:?}"
        );
    }

    #[test]
    fn test_valid_acyclic_cascade_passes() {
        // Linear acyclic cascade A(0)->B(1)->C(2).
        // Verifies that a valid cascade produces Ok with correct topological_order length.
        let bus = make_bus(0);
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(1));
        let mut h1 = make_hydro(1);
        h1.downstream_id = Some(EntityId(2));
        let h2 = make_hydro(2);

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0, h1, h2])
            .build();

        assert!(
            result.is_ok(),
            "expected Ok for acyclic cascade, got: {:?}",
            result.unwrap_err()
        );
        let system = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(
            system.cascade().topological_order().len(),
            system.n_hydros(),
            "topological_order must contain all hydros"
        );
    }

    // ---- Filling config validation tests ------------------------------------

    #[test]
    fn test_filling_without_entry_stage() {
        // Filling config present but entry_stage_id is None.
        use crate::entities::FillingConfig;
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = None;
        hydro.filling = Some(FillingConfig {
            start_stage_id: 10,
            filling_inflow_m3s: 100.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_err(),
            "expected Err for filling without entry_stage_id"
        );
        let errors = result.unwrap_err();
        let has_error = errors.iter().any(|e| match e {
            ValidationError::InvalidFillingConfig { hydro_id, reason } => {
                *hydro_id == EntityId(1) && reason.contains("entry_stage_id")
            }
            _ => false,
        });
        assert!(
            has_error,
            "expected InvalidFillingConfig with entry_stage_id reason, got: {errors:?}"
        );
    }

    #[test]
    fn test_filling_negative_inflow() {
        // Filling config with filling_inflow_m3s <= 0.0.
        use crate::entities::FillingConfig;
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = Some(10);
        hydro.filling = Some(FillingConfig {
            start_stage_id: 10,
            filling_inflow_m3s: -5.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_err(),
            "expected Err for negative filling_inflow_m3s"
        );
        let errors = result.unwrap_err();
        let has_error = errors.iter().any(|e| match e {
            ValidationError::InvalidFillingConfig { hydro_id, reason } => {
                *hydro_id == EntityId(1) && reason.contains("filling_inflow_m3s must be positive")
            }
            _ => false,
        });
        assert!(
            has_error,
            "expected InvalidFillingConfig with positive inflow reason, got: {errors:?}"
        );
    }

    #[test]
    fn test_valid_filling_config_passes() {
        // Valid filling config: entry_stage_id set and filling_inflow_m3s positive.
        use crate::entities::FillingConfig;
        let bus = make_bus(0);
        let mut hydro = make_hydro(1);
        hydro.entry_stage_id = Some(10);
        hydro.filling = Some(FillingConfig {
            start_stage_id: 10,
            filling_inflow_m3s: 100.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .build();

        assert!(
            result.is_ok(),
            "expected Ok for valid filling config, got: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_cascade_cycle_and_invalid_filling_both_reported() {
        // Both a cascade cycle (A->A self-loop) AND an invalid filling config
        // must produce both error variants.
        use crate::entities::FillingConfig;
        let bus = make_bus(0);

        // Hydro 0: self-loop (cycle)
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(0));

        // Hydro 1: valid cycle participant? No -- use a separate hydro with invalid filling.
        let mut h1 = make_hydro(1);
        h1.entry_stage_id = None; // no entry_stage_id
        h1.filling = Some(FillingConfig {
            start_stage_id: 5,
            filling_inflow_m3s: 50.0,
        });

        let result = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![h0, h1])
            .build();

        assert!(result.is_err(), "expected Err for cycle + invalid filling");
        let errors = result.unwrap_err();
        let has_cycle = errors
            .iter()
            .any(|e| matches!(e, ValidationError::CascadeCycle { .. }));
        let has_filling = errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidFillingConfig { .. }));
        assert!(has_cycle, "expected CascadeCycle error, got: {errors:?}");
        assert!(
            has_filling,
            "expected InvalidFillingConfig error, got: {errors:?}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_system_serde_roundtrip() {
        // Build a system with a bus, a hydro, a line, and a thermal.
        let bus_a = make_bus(1);
        let bus_b = make_bus(2);
        let hydro = make_hydro_on_bus(10, 1);
        let thermal = make_thermal_on_bus(20, 2);
        let line = make_line(1, 1, 2);

        let system = SystemBuilder::new()
            .buses(vec![bus_a, bus_b])
            .hydros(vec![hydro])
            .thermals(vec![thermal])
            .lines(vec![line])
            .build()
            .expect("valid system");

        let json = serde_json::to_string(&system).unwrap();

        // Deserialize and rebuild indices.
        let mut deserialized: System = serde_json::from_str(&json).unwrap();
        deserialized.rebuild_indices();

        // Entity collections must match.
        assert_eq!(system.buses(), deserialized.buses());
        assert_eq!(system.hydros(), deserialized.hydros());
        assert_eq!(system.thermals(), deserialized.thermals());
        assert_eq!(system.lines(), deserialized.lines());

        // O(1) lookup must work after index rebuild.
        assert_eq!(
            deserialized.bus(EntityId(1)).map(|b| b.id),
            Some(EntityId(1))
        );
        assert_eq!(
            deserialized.hydro(EntityId(10)).map(|h| h.id),
            Some(EntityId(10))
        );
        assert_eq!(
            deserialized.thermal(EntityId(20)).map(|t| t.id),
            Some(EntityId(20))
        );
        assert_eq!(
            deserialized.line(EntityId(1)).map(|l| l.id),
            Some(EntityId(1))
        );
    }

    // ---- Extended System tests ----------------------------------------------

    fn make_stage(id: i32) -> Stage {
        use crate::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };
        use chrono::NaiveDate;
        Stage {
            index: usize::try_from(id.max(0)).unwrap_or(0),
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
                branching_factor: 50,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Verify that `SystemBuilder::new().build()` still works correctly.
    /// New fields must default to empty/default values.
    #[test]
    fn test_system_backward_compat() {
        let system = SystemBuilder::new().build().expect("empty system is valid");
        // Entity counts unchanged
        assert_eq!(system.n_buses(), 0);
        assert_eq!(system.n_hydros(), 0);
        // New fields default to empty
        assert_eq!(system.n_stages(), 0);
        assert!(system.stages().is_empty());
        assert!(system.initial_conditions().storage.is_empty());
        assert!(system.generic_constraints().is_empty());
        assert!(system.inflow_models().is_empty());
        assert!(system.load_models().is_empty());
        assert_eq!(system.penalties().n_stages(), 0);
        assert_eq!(system.bounds().n_stages(), 0);
    }

    /// Build a System with 2 stages and verify `n_stages()` and `stage(id)` lookup.
    #[test]
    fn test_system_with_stages() {
        let s0 = make_stage(0);
        let s1 = make_stage(1);

        let system = SystemBuilder::new()
            .stages(vec![s1.clone(), s0.clone()]) // supply in reverse order
            .build()
            .expect("valid system");

        // Canonical ordering: id=0 comes before id=1
        assert_eq!(system.n_stages(), 2);
        assert_eq!(system.stages()[0].id, 0);
        assert_eq!(system.stages()[1].id, 1);

        // O(1) lookup by stage id
        let found = system.stage(0).expect("stage 0 must be found");
        assert_eq!(found.id, s0.id);

        let found1 = system.stage(1).expect("stage 1 must be found");
        assert_eq!(found1.id, s1.id);

        // Missing stage returns None
        assert!(system.stage(99).is_none());
    }

    /// Build a System with 3 stages having IDs 0, 1, 2 and verify `stage()` lookups.
    #[test]
    fn test_system_stage_lookup_by_id() {
        let stages: Vec<Stage> = [0i32, 1, 2].iter().map(|&id| make_stage(id)).collect();

        let system = SystemBuilder::new()
            .stages(stages)
            .build()
            .expect("valid system");

        assert_eq!(system.stage(1).map(|s| s.id), Some(1));
        assert!(system.stage(99).is_none());
    }

    /// Build a System with `InitialConditions` containing 1 storage entry and verify accessor.
    #[test]
    fn test_system_with_initial_conditions() {
        let ic = InitialConditions {
            storage: vec![crate::HydroStorage {
                hydro_id: EntityId(0),
                value_hm3: 15_000.0,
            }],
            filling_storage: vec![],
            past_inflows: vec![],
        };

        let system = SystemBuilder::new()
            .initial_conditions(ic)
            .build()
            .expect("valid system");

        assert_eq!(system.initial_conditions().storage.len(), 1);
        assert_eq!(system.initial_conditions().storage[0].hydro_id, EntityId(0));
        assert!((system.initial_conditions().storage[0].value_hm3 - 15_000.0).abs() < f64::EPSILON);
    }

    /// Verify serde round-trip of a System with stages and `policy_graph`,
    /// including that `stage_index` is correctly rebuilt after deserialization.
    #[cfg(feature = "serde")]
    #[test]
    fn test_system_serde_roundtrip_with_stages() {
        use crate::temporal::PolicyGraphType;

        let stages = vec![make_stage(0), make_stage(1)];
        let policy_graph = PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            season_map: None,
        };

        let system = SystemBuilder::new()
            .stages(stages)
            .policy_graph(policy_graph)
            .build()
            .expect("valid system");

        let json = serde_json::to_string(&system).unwrap();
        let mut deserialized: System = serde_json::from_str(&json).unwrap();

        // stage_index is skipped during serde; rebuild before querying
        deserialized.rebuild_indices();

        // Collections must match after round-trip
        assert_eq!(system.n_stages(), deserialized.n_stages());
        assert_eq!(system.stages()[0].id, deserialized.stages()[0].id);
        assert_eq!(system.stages()[1].id, deserialized.stages()[1].id);

        // O(1) lookup must work after index rebuild
        assert_eq!(deserialized.stage(0).map(|s| s.id), Some(0));
        assert_eq!(deserialized.stage(1).map(|s| s.id), Some(1));
        assert!(deserialized.stage(99).is_none());

        // policy_graph fields must round-trip
        assert_eq!(
            deserialized.policy_graph().graph_type,
            system.policy_graph().graph_type
        );
    }
}
