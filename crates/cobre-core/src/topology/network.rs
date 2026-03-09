//! Resolved electrical transmission network topology.
//!
//! `NetworkTopology` holds the validated adjacency structure derived from the
//! `Line` entity collection. It is built during case loading after all `Bus` and
//! `Line` entities have been validated and their cross-references verified.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::entities::Bus;
use crate::{
    EnergyContract, EntityId, Hydro, Line, NonControllableSource, PumpingStation, Thermal,
};

/// A line connection from a bus perspective.
///
/// Describes whether a bus is the source or target end of a transmission line,
/// and which line it refers to. Used in bus-line incidence lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusLineConnection {
    /// The line's entity ID.
    pub line_id: EntityId,
    /// True if this bus is the line's source (direct flow direction).
    /// False if this bus is the line's target (reverse flow direction).
    pub is_source: bool,
}

/// Generator entities connected to a bus.
///
/// Groups hydro, thermal, and non-controllable source IDs by type. All ID
/// lists are in canonical ascending-`i32` order for determinism.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusGenerators {
    /// Hydro plant IDs connected to this bus.
    pub hydro_ids: Vec<EntityId>,
    /// Thermal plant IDs connected to this bus.
    pub thermal_ids: Vec<EntityId>,
    /// Non-controllable source IDs connected to this bus.
    pub ncs_ids: Vec<EntityId>,
}

/// Load/demand entities connected to a bus.
///
/// Groups energy contract and pumping station IDs. All ID lists are in
/// canonical ascending-`i32` order for determinism.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusLoads {
    /// Energy contract IDs at this bus.
    pub contract_ids: Vec<EntityId>,
    /// Pumping station IDs consuming power at this bus.
    pub pumping_station_ids: Vec<EntityId>,
}

/// Resolved transmission network topology for buses and lines.
///
/// Provides O(1) lookup for bus-line incidence, bus-to-generator maps,
/// and bus-to-load maps. Built from entity collections during System
/// construction and immutable thereafter.
///
/// Used for power balance constraint generation.
// The three private fields all share the `bus_` prefix intentionally: they form a
// cohesive group keyed by bus identity. The prefix mirrors the public accessor names
// and the spec's field names, making the code self-documenting.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NetworkTopology {
    /// Bus-line incidence: `bus_id` -> list of (`line_id`, `is_source`).
    /// `is_source` is true when the bus is the source (direct flow direction).
    bus_lines: HashMap<EntityId, Vec<BusLineConnection>>,

    /// Bus generation map: `bus_id` -> list of generator IDs by type.
    bus_generators: HashMap<EntityId, BusGenerators>,

    /// Bus load map: `bus_id` -> list of load/demand entity IDs.
    bus_loads: HashMap<EntityId, BusLoads>,
}

/// Global default for buses with no generators. Used as fallback in `bus_generators`.
static DEFAULT_BUS_GENERATORS: OnceLock<BusGenerators> = OnceLock::new();

/// Global default for buses with no loads. Used as fallback in `bus_loads`.
static DEFAULT_BUS_LOADS: OnceLock<BusLoads> = OnceLock::new();

impl NetworkTopology {
    /// Build network topology from entity collections.
    ///
    /// Constructs bus-line incidence, bus generation maps, and bus load maps
    /// from the entity collections. Does not validate (no bus existence checks) --
    /// validation is separate.
    ///
    /// All entity slices are assumed to be in canonical ID order.
    ///
    /// # Arguments
    ///
    /// * `buses` - All bus entities in canonical ID order.
    /// * `lines` - All line entities in canonical ID order.
    /// * `hydros` - All hydro plant entities in canonical ID order.
    /// * `thermals` - All thermal plant entities in canonical ID order.
    /// * `non_controllable_sources` - All NCS entities in canonical ID order.
    /// * `contracts` - All energy contract entities in canonical ID order.
    /// * `pumping_stations` - All pumping station entities in canonical ID order.
    #[must_use]
    pub fn build(
        buses: &[Bus],
        lines: &[Line],
        hydros: &[Hydro],
        thermals: &[Thermal],
        non_controllable_sources: &[NonControllableSource],
        contracts: &[EnergyContract],
        pumping_stations: &[PumpingStation],
    ) -> Self {
        let mut bus_lines: HashMap<EntityId, Vec<BusLineConnection>> = HashMap::new();
        let mut bus_generators: HashMap<EntityId, BusGenerators> = HashMap::new();
        let mut bus_loads: HashMap<EntityId, BusLoads> = HashMap::new();

        // TODO: use `buses` for disconnected-bus validation (ValidationError::DisconnectedBus)
        let _ = buses;

        for line in lines {
            bus_lines
                .entry(line.source_bus_id)
                .or_default()
                .push(BusLineConnection {
                    line_id: line.id,
                    is_source: true,
                });
            bus_lines
                .entry(line.target_bus_id)
                .or_default()
                .push(BusLineConnection {
                    line_id: line.id,
                    is_source: false,
                });
        }
        for connections in bus_lines.values_mut() {
            connections.sort_by_key(|c| c.line_id.0);
        }

        for hydro in hydros {
            bus_generators
                .entry(hydro.bus_id)
                .or_default()
                .hydro_ids
                .push(hydro.id);
        }

        for thermal in thermals {
            bus_generators
                .entry(thermal.bus_id)
                .or_default()
                .thermal_ids
                .push(thermal.id);
        }

        for ncs in non_controllable_sources {
            bus_generators
                .entry(ncs.bus_id)
                .or_default()
                .ncs_ids
                .push(ncs.id);
        }

        for generators in bus_generators.values_mut() {
            generators.hydro_ids.sort_by_key(|id| id.0);
            generators.thermal_ids.sort_by_key(|id| id.0);
            generators.ncs_ids.sort_by_key(|id| id.0);
        }

        for contract in contracts {
            bus_loads
                .entry(contract.bus_id)
                .or_default()
                .contract_ids
                .push(contract.id);
        }

        for station in pumping_stations {
            bus_loads
                .entry(station.bus_id)
                .or_default()
                .pumping_station_ids
                .push(station.id);
        }

        for loads in bus_loads.values_mut() {
            loads.contract_ids.sort_by_key(|id| id.0);
            loads.pumping_station_ids.sort_by_key(|id| id.0);
        }

        Self {
            bus_lines,
            bus_generators,
            bus_loads,
        }
    }

    /// Returns the lines connected to a bus.
    ///
    /// Returns an empty slice if the bus has no connected lines.
    #[must_use]
    pub fn bus_lines(&self, bus_id: EntityId) -> &[BusLineConnection] {
        self.bus_lines.get(&bus_id).map_or(&[], Vec::as_slice)
    }

    /// Returns the generators connected to a bus.
    ///
    /// Returns a reference to an empty `BusGenerators` if the bus has no generators.
    #[must_use]
    pub fn bus_generators(&self, bus_id: EntityId) -> &BusGenerators {
        self.bus_generators
            .get(&bus_id)
            .unwrap_or_else(|| DEFAULT_BUS_GENERATORS.get_or_init(BusGenerators::default))
    }

    /// Returns the loads connected to a bus.
    ///
    /// Returns a reference to an empty `BusLoads` if the bus has no loads.
    #[must_use]
    pub fn bus_loads(&self, bus_id: EntityId) -> &BusLoads {
        self.bus_loads
            .get(&bus_id)
            .unwrap_or_else(|| DEFAULT_BUS_LOADS.get_or_init(BusLoads::default))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::{ContractType, HydroGenerationModel, HydroPenalties, ThermalCostSegment};

    fn make_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: String::new(),
            deficit_segments: vec![],
            excess_cost: 0.0,
        }
    }

    fn make_line(id: i32, source_bus_id: i32, target_bus_id: i32) -> Line {
        Line {
            id: EntityId(id),
            name: String::new(),
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

    fn make_hydro(id: i32, bus_id: i32) -> Hydro {
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
            name: String::new(),
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
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        }
    }

    fn make_thermal(id: i32, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: String::new(),
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

    fn make_ncs(id: i32, bus_id: i32) -> NonControllableSource {
        NonControllableSource {
            id: EntityId(id),
            name: String::new(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 50.0,
            curtailment_cost: 0.0,
        }
    }

    fn make_contract(id: i32, bus_id: i32) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: String::new(),
            bus_id: EntityId(bus_id),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 100.0,
        }
    }

    fn make_pumping_station(id: i32, bus_id: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: String::new(),
            bus_id: EntityId(bus_id),
            source_hydro_id: EntityId(0),
            destination_hydro_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 10.0,
        }
    }

    #[test]
    fn test_empty_network() {
        let topo = NetworkTopology::build(&[], &[], &[], &[], &[], &[], &[]);

        // Any bus ID returns empty collections.
        assert_eq!(topo.bus_lines(EntityId(0)), &[]);
        assert!(topo.bus_generators(EntityId(0)).hydro_ids.is_empty());
        assert!(topo.bus_generators(EntityId(0)).thermal_ids.is_empty());
        assert!(topo.bus_generators(EntityId(0)).ncs_ids.is_empty());
        assert!(topo.bus_loads(EntityId(0)).contract_ids.is_empty());
        assert!(topo.bus_loads(EntityId(0)).pumping_station_ids.is_empty());
    }

    #[test]
    fn test_single_line() {
        // Line 0 connects bus 0 (source) -> bus 1 (target).
        let buses = vec![make_bus(0), make_bus(1)];
        let lines = vec![make_line(0, 0, 1)];
        let topo = NetworkTopology::build(&buses, &lines, &[], &[], &[], &[], &[]);

        // Bus 0 is the source.
        let conns_0 = topo.bus_lines(EntityId(0));
        assert_eq!(conns_0.len(), 1);
        assert_eq!(conns_0[0].line_id, EntityId(0));
        assert!(conns_0[0].is_source);

        // Bus 1 is the target.
        let conns_1 = topo.bus_lines(EntityId(1));
        assert_eq!(conns_1.len(), 1);
        assert_eq!(conns_1[0].line_id, EntityId(0));
        assert!(!conns_1[0].is_source);
    }

    #[test]
    fn test_multiple_lines_same_bus() {
        // Bus 0 is source of lines 0, 1, 2; each targeting a different bus.
        let buses = vec![make_bus(0), make_bus(1), make_bus(2), make_bus(3)];
        let lines = vec![make_line(0, 0, 1), make_line(1, 0, 2), make_line(2, 0, 3)];
        let topo = NetworkTopology::build(&buses, &lines, &[], &[], &[], &[], &[]);

        let conns = topo.bus_lines(EntityId(0));
        assert_eq!(conns.len(), 3);
        // All connections belong to bus 0 as source.
        assert!(conns.iter().all(|c| c.is_source));
        // Sorted by line_id inner i32.
        assert_eq!(conns[0].line_id, EntityId(0));
        assert_eq!(conns[1].line_id, EntityId(1));
        assert_eq!(conns[2].line_id, EntityId(2));
    }

    #[test]
    fn test_generators_per_bus() {
        // Bus 0: hydro 0, hydro 1, thermal 0.  Bus 1: NCS 0.
        let buses = vec![make_bus(0), make_bus(1)];
        let hydros = vec![make_hydro(0, 0), make_hydro(1, 0)];
        let thermals = vec![make_thermal(0, 0)];
        let ncs = vec![make_ncs(0, 1)];
        let topo = NetworkTopology::build(&buses, &[], &hydros, &thermals, &ncs, &[], &[]);

        let gen0 = topo.bus_generators(EntityId(0));
        assert_eq!(gen0.hydro_ids.len(), 2);
        assert_eq!(gen0.thermal_ids.len(), 1);
        assert!(gen0.ncs_ids.is_empty());

        let gen1 = topo.bus_generators(EntityId(1));
        assert!(gen1.hydro_ids.is_empty());
        assert!(gen1.thermal_ids.is_empty());
        assert_eq!(gen1.ncs_ids.len(), 1);
        assert_eq!(gen1.ncs_ids[0], EntityId(0));
    }

    #[test]
    fn test_loads_per_bus() {
        // Bus 0: contract 0, pumping station 0.
        let buses = vec![make_bus(0)];
        let contracts = vec![make_contract(0, 0)];
        let stations = vec![make_pumping_station(0, 0)];
        let topo = NetworkTopology::build(&buses, &[], &[], &[], &[], &contracts, &stations);

        let loads0 = topo.bus_loads(EntityId(0));
        assert_eq!(loads0.contract_ids.len(), 1);
        assert_eq!(loads0.contract_ids[0], EntityId(0));
        assert_eq!(loads0.pumping_station_ids.len(), 1);
        assert_eq!(loads0.pumping_station_ids[0], EntityId(0));
    }

    #[test]
    fn test_bus_no_connections() {
        // Bus 0 exists but nothing is connected to it.
        let buses = vec![make_bus(0)];
        let topo = NetworkTopology::build(&buses, &[], &[], &[], &[], &[], &[]);

        assert_eq!(topo.bus_lines(EntityId(0)), &[]);
        let generators = topo.bus_generators(EntityId(0));
        assert!(generators.hydro_ids.is_empty());
        assert!(generators.thermal_ids.is_empty());
        assert!(generators.ncs_ids.is_empty());
        let loads = topo.bus_loads(EntityId(0));
        assert!(loads.contract_ids.is_empty());
        assert!(loads.pumping_station_ids.is_empty());
    }

    #[test]
    fn test_deterministic_ordering() {
        // Insert generators in reverse ID order; expect canonical ID-ascending order.
        let buses = vec![make_bus(0)];
        // Hydros with IDs 5, 3, 1 connected to bus 0 — supplied in reverse order.
        let hydros = vec![make_hydro(5, 0), make_hydro(3, 0), make_hydro(1, 0)];
        // Thermals with IDs 4, 2 connected to bus 0 — supplied in reverse order.
        let thermals = vec![make_thermal(4, 0), make_thermal(2, 0)];
        // Contracts with IDs 10, 7 connected to bus 0 — supplied in reverse order.
        let contracts = vec![make_contract(10, 0), make_contract(7, 0)];
        let topo = NetworkTopology::build(&buses, &[], &hydros, &thermals, &[], &contracts, &[]);

        let generators = topo.bus_generators(EntityId(0));
        // Hydro IDs must be sorted ascending: 1, 3, 5.
        assert_eq!(
            generators.hydro_ids,
            vec![EntityId(1), EntityId(3), EntityId(5)]
        );
        // Thermal IDs must be sorted ascending: 2, 4.
        assert_eq!(generators.thermal_ids, vec![EntityId(2), EntityId(4)]);

        let loads = topo.bus_loads(EntityId(0));
        // Contract IDs must be sorted ascending: 7, 10.
        assert_eq!(loads.contract_ids, vec![EntityId(7), EntityId(10)]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_topology_serde_roundtrip_network() {
        // Build a network with buses, lines, hydros, and thermals.
        let buses = vec![make_bus(0), make_bus(1)];
        let lines = vec![make_line(0, 0, 1)];
        let hydros = vec![make_hydro(0, 0)];
        let thermals = vec![make_thermal(0, 1)];
        let topo = NetworkTopology::build(&buses, &lines, &hydros, &thermals, &[], &[], &[]);
        let json = serde_json::to_string(&topo).unwrap();
        let deserialized: NetworkTopology = serde_json::from_str(&json).unwrap();
        assert_eq!(topo, deserialized);
    }
}
