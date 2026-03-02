//! Top-level system struct and builder.
//!
//! The `System` struct is the top-level in-memory representation of a fully loaded,
//! validated, and resolved case. It is produced by `cobre-io::load_case()` and consumed
//! by `cobre-sddp::train()`, `cobre-sddp::simulate()`, and `cobre-stochastic` scenario
//! generation.
//!
//! All entity collections in `System` are stored in canonical ID-sorted order to ensure
//! declaration-order invariance: results are bit-for-bit identical regardless of input
//! entity ordering. See the design principles spec for details.

use std::collections::HashMap;

use crate::{
    Bus, CascadeTopology, EnergyContract, EntityId, Hydro, Line, NetworkTopology,
    NonControllableSource, PumpingStation, Thermal, ValidationError,
};

/// Top-level system representation.
///
/// Produced by `cobre-io` (Phase 2) or [`SystemBuilder`] (Phase 1 tests).
/// Consumed by `cobre-sddp` and `cobre-stochastic` by shared reference.
/// Immutable after construction. Shared read-only across threads.
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
#[derive(Debug)]
pub struct System {
    // Entity collections (canonical ordering by ID) -- public for read access
    /// All bus entities, sorted by `EntityId` inner `i32`.
    pub buses: Vec<Bus>,
    /// All line entities, sorted by `EntityId` inner `i32`.
    pub lines: Vec<Line>,
    /// All hydro plant entities, sorted by `EntityId` inner `i32`.
    pub hydros: Vec<Hydro>,
    /// All thermal plant entities, sorted by `EntityId` inner `i32`.
    pub thermals: Vec<Thermal>,
    /// All pumping station entities, sorted by `EntityId` inner `i32`.
    pub pumping_stations: Vec<PumpingStation>,
    /// All energy contract entities, sorted by `EntityId` inner `i32`.
    pub contracts: Vec<EnergyContract>,
    /// All non-controllable source entities, sorted by `EntityId` inner `i32`.
    pub non_controllable_sources: Vec<NonControllableSource>,

    // O(1) lookup indices (entity ID -> position in collection) -- private
    bus_index: HashMap<EntityId, usize>,
    line_index: HashMap<EntityId, usize>,
    hydro_index: HashMap<EntityId, usize>,
    thermal_index: HashMap<EntityId, usize>,
    pumping_station_index: HashMap<EntityId, usize>,
    contract_index: HashMap<EntityId, usize>,
    non_controllable_source_index: HashMap<EntityId, usize>,

    // Topology
    /// Resolved hydro cascade graph.
    cascade: CascadeTopology,
    /// Resolved transmission network topology.
    network: NetworkTopology,
}

// Compile-time check that System is Send + Sync.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
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
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemBuilder {
    /// Create a new empty builder. All entity collections start empty.
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

    /// Build the [`System`].
    ///
    /// Sorts all entity collections by [`EntityId`] (canonical ordering).
    /// Checks for duplicate IDs within each collection.
    /// Builds [`CascadeTopology`] and [`NetworkTopology`].
    /// Constructs lookup indices.
    ///
    /// Returns `Err` with a list of all validation errors found across all collections.
    /// Currently only checks for duplicate IDs within each collection. Cross-reference
    /// and topology validation is added in Epic 3.
    ///
    /// # Errors
    ///
    /// Returns `Err(Vec<ValidationError>)` if duplicate IDs are detected in any
    /// entity collection. All duplicates across all collections are reported together.
    pub fn build(mut self) -> Result<System, Vec<ValidationError>> {
        self.buses.sort_by_key(|e| e.id.0);
        self.lines.sort_by_key(|e| e.id.0);
        self.hydros.sort_by_key(|e| e.id.0);
        self.thermals.sort_by_key(|e| e.id.0);
        self.pumping_stations.sort_by_key(|e| e.id.0);
        self.contracts.sort_by_key(|e| e.id.0);
        self.non_controllable_sources.sort_by_key(|e| e.id.0);

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

        let cascade = CascadeTopology::build(&self.hydros);
        let network = NetworkTopology::build(
            &self.buses,
            &self.lines,
            &self.hydros,
            &self.thermals,
            &self.non_controllable_sources,
            &self.contracts,
            &self.pumping_stations,
        );

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

fn check_duplicates<T: HasId>(
    entities: &[T],
    entity_type: &'static str,
    errors: &mut Vec<ValidationError>,
) {
    for window in entities.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if a.entity_id() == b.entity_id() {
            errors.push(ValidationError::DuplicateId {
                entity_type,
                id: a.entity_id(),
            });
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

    fn make_hydro(id: i32) -> Hydro {
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
            bus_id: EntityId(0),
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
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        }
    }

    fn make_thermal(id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("thermal-{id}"),
            bus_id: EntityId(0),
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

    fn make_pumping_station(id: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("ps-{id}"),
            bus_id: EntityId(0),
            source_hydro_id: EntityId(0),
            destination_hydro_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 10.0,
        }
    }

    fn make_contract(id: i32) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("contract-{id}"),
            bus_id: EntityId(0),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 100.0,
        }
    }

    fn make_ncs(id: i32) -> NonControllableSource {
        NonControllableSource {
            id: EntityId(id),
            name: format!("ncs-{id}"),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 50.0,
            curtailment_cost: 0.0,
        }
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
        let system = SystemBuilder::new()
            .hydros(vec![make_hydro(10), make_hydro(5), make_hydro(20)])
            .build()
            .expect("valid system");

        assert_eq!(system.hydro(EntityId(5)).map(|h| h.id), Some(EntityId(5)));
        assert_eq!(system.hydro(EntityId(10)).map(|h| h.id), Some(EntityId(10)));
        assert_eq!(system.hydro(EntityId(20)).map(|h| h.id), Some(EntityId(20)));
    }

    #[test]
    fn test_lookup_missing_id() {
        let system = SystemBuilder::new()
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
        let mut h0 = make_hydro(0);
        h0.downstream_id = Some(EntityId(1));
        let mut h1 = make_hydro(1);
        h1.downstream_id = Some(EntityId(2));
        let h2 = make_hydro(2);

        let system = SystemBuilder::new()
            .hydros(vec![h0, h1, h2])
            .build()
            .expect("valid system");

        let order = system.cascade().topological_order();
        assert!(!order.is_empty(), "topological order must be non-empty");
        let pos_0 = order.iter().position(|&id| id == EntityId(0));
        let pos_2 = order.iter().position(|&id| id == EntityId(2));
        assert!(pos_0 < pos_2);
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
        let system = SystemBuilder::new()
            .buses(vec![make_bus(1)])
            .lines(vec![make_line(2, 0, 1)])
            .hydros(vec![make_hydro(3)])
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
}
