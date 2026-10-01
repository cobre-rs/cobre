//! Construction-time validation helpers for [`SystemBuilder::build()`](crate::SystemBuilder::build).

use std::collections::HashMap;

use crate::{
    Bus, EnergyContract, EntityId, Hydro, Line, NonControllableSource, PumpingStation, Stage,
    Thermal, ValidationError,
};

pub(crate) trait HasId {
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

pub(crate) fn build_index<T: HasId>(entities: &[T]) -> HashMap<EntityId, usize> {
    let mut index = HashMap::with_capacity(entities.len());
    for (i, entity) in entities.iter().enumerate() {
        index.insert(entity.entity_id(), i);
    }
    index
}

/// Keys are `i32` stage IDs, negative for pre-study stages.
pub(crate) fn build_stage_index(stages: &[Stage]) -> HashMap<i32, usize> {
    let mut index = HashMap::with_capacity(stages.len());
    for (i, stage) in stages.iter().enumerate() {
        index.insert(stage.id, i);
    }
    index
}

/// Detect duplicate entity IDs in a **sorted** slice by scanning adjacent pairs.
///
/// Contract: the caller must sort `entities` by `entity_id()` ascending before
/// calling (the builder sorts every collection in `build`). On unsorted input
/// the adjacent-pair scan silently misses non-adjacent duplicates, letting a
/// colliding id slip past validation.
pub(crate) fn check_duplicates<T: HasId>(
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

/// Parallel i32-keyed duplicate scan for stages, which key on a domain `i32` id
/// (negative for pre-study) and so cannot reuse the [`HasId`]-generic [`check_duplicates`].
///
/// Contract: `stages` must be sorted ascending by `id` before calling. The
/// adjacent-pair scan misses non-adjacent duplicates on unsorted input, letting
/// `build_stage_index` silently overwrite a colliding stage with no validation error.
pub(crate) fn check_duplicate_stages(stages: &[Stage], errors: &mut Vec<ValidationError>) {
    for window in stages.windows(2) {
        if window[0].id == window[1].id {
            errors.push(ValidationError::DuplicateId {
                entity_type: "Stage",
                id: EntityId(window[0].id),
            });
        }
    }
}

/// Bundles entity slices needed for cross-reference validation.
pub(crate) struct CrossRefEntities<'a> {
    pub(crate) lines: &'a [Line],
    pub(crate) hydros: &'a [Hydro],
    pub(crate) thermals: &'a [Thermal],
    pub(crate) pumping_stations: &'a [PumpingStation],
    pub(crate) contracts: &'a [EnergyContract],
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
}

/// Validate every [`EntityId`] cross-reference field against the appropriate index,
/// appending all violations to `errors` — no short-circuiting on the first error.
pub(crate) fn validate_cross_references(
    entities: &CrossRefEntities<'_>,
    bus_index: &HashMap<EntityId, usize>,
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    validate_line_refs(entities.lines, bus_index, errors);
    validate_hydro_refs(entities.hydros, hydro_index, errors);
    validate_thermal_refs(entities.thermals, bus_index, errors);
    validate_pumping_station_refs(entities.pumping_stations, bus_index, hydro_index, errors);
    validate_contract_refs(entities.contracts, bus_index, errors);
    validate_ncs_refs(entities.non_controllable_sources, bus_index, errors);
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
    hydro_index: &HashMap<EntityId, usize>,
    errors: &mut Vec<ValidationError>,
) {
    for hydro in hydros {
        if let Some(downstream_id) = hydro.downstream_id
            && !hydro_index.contains_key(&downstream_id)
        {
            errors.push(ValidationError::InvalidReference {
                source_entity_type: "Hydro",
                source_id: hydro.id,
                field_name: "downstream_id",
                referenced_id: downstream_id,
                expected_type: "Hydro",
            });
        }
        if let Some(ref diversion) = hydro.diversion
            && !hydro_index.contains_key(&diversion.downstream_id)
        {
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

/// Validate the filling config of every hydro that has one, appending all
/// violations to `errors` (no short-circuiting on the first).
///
/// For each `filling: Some(config)`:
/// - `filling_min_rate_m3s` must be non-negative: zero is valid (no minimum
///   accumulation required); only a negative or NaN rate is rejected.
/// - `entry_stage_id` must be `Some`.
/// - `start_stage_id < entry_stage_id`. Mirrors the cobre-io ordering guard here so
///   a `System` built directly via the public [`SystemBuilder`](super::SystemBuilder)
///   still rejects an inverted config, which otherwise mis-phases the reservoir in
///   the solver with no error. Validating `start_stage_id` against the study stage
///   set stays in cobre-io, which alone holds the horizon.
pub(crate) fn validate_filling_configs(hydros: &[Hydro], errors: &mut Vec<ValidationError>) {
    for hydro in hydros {
        if let Some(filling) = &hydro.filling {
            if filling.filling_min_rate_m3s.is_nan() || filling.filling_min_rate_m3s < 0.0 {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling_min_rate_m3s must be non-negative".to_string(),
                });
            }
            if hydro.entry_stage_id.is_none() {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling requires entry_stage_id to be set".to_string(),
                });
            }
            if let Some(entry) = hydro.entry_stage_id
                && filling.start_stage_id >= entry
            {
                errors.push(ValidationError::InvalidFillingConfig {
                    hydro_id: hydro.id,
                    reason: "filling start_stage_id must be less than entry_stage_id".to_string(),
                });
            }
        }
    }
}
