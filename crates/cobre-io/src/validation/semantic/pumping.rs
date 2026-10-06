//! Layer 5a — pumping-station-domain semantic validation.
//!
//! Referential integrity is discharged in Layer 3 (`check_pumping_references`);
//! this module covers two invariants: a station must move water between two
//! *distinct* reservoirs (rule 19), and it may be active only at stages where both
//! endpoint hydros are `Operating` (rule 52).

use cobre_core::commissioning::{commissioning_active, hydro_operating_active};
use cobre_core::{Hydro, PumpingStation};

use super::super::{ValidationContext, rules, schema::ParsedData};

/// Rule 19: rejects pumping stations whose source and destination hydros are identical.
///
/// A `source_hydro_id == destination_hydro_id` station passes referential
/// validation (the ID resolves) yet models a degenerate self-transfer: a
/// self-cancelling `+τ`/`−τ` pair on one water-balance row while still drawing
/// power — a silent modeling error, not a dangling reference.
pub(super) fn check_pumping_semantics(data: &ParsedData, ctx: &mut ValidationContext) {
    for station in &data.pumping_stations {
        if station.source_hydro_id == station.destination_hydro_id {
            let entity_str = format!("PumpingStation {}", station.id.0);
            ctx.emit(
                &rules::SEMANTIC_PUMPING_SAME_ENDPOINTS,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} has source_hydro_id == destination_hydro_id ({}): a station must transfer water between two distinct reservoirs",
                    station.source_hydro_id.0
                ),
            );
        }
    }
}

/// Rule 52: rejects a pumping station that is active at a study stage where its
/// source or destination hydro is not `Operating`.
///
/// An active pumping column writes `±τ` onto both endpoints' water-balance rows,
/// and a `PreFilling` row is a frozen identity that cannot absorb it. `Filling`
/// is rejected too, mirroring the travel-time arc into a plant that has not
/// reached `Operating`. One error per offending side, naming the first such stage.
pub(super) fn check_pumping_operating_window(data: &ParsedData, ctx: &mut ValidationContext) {
    for station in &data.pumping_stations {
        if station.source_hydro_id == station.destination_hydro_id {
            continue;
        }
        for (side, hydro_id) in [
            ("source", station.source_hydro_id),
            ("destination", station.destination_hydro_id),
        ] {
            let Some(hydro) = data.hydros.iter().find(|h| h.id == hydro_id) else {
                continue;
            };
            let Some(stage_id) = first_stage_outside_operating_window(data, station, hydro) else {
                continue;
            };
            let entity_str = format!("PumpingStation {}", station.id.0);
            ctx.emit(
                &rules::SEMANTIC_PUMPING_ENDPOINT_NOT_OPERATING,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: active at stage {stage_id} while its {side} hydro {} is not Operating there \
                     (before entry_stage_id, at or after exit_stage_id, or Filling); a pumping station \
                     may operate only inside both endpoints' Operating windows",
                    hydro_id.0
                ),
            );
        }
    }
}

fn first_stage_outside_operating_window(
    data: &ParsedData,
    station: &PumpingStation,
    hydro: &Hydro,
) -> Option<i32> {
    data.stages
        .stages
        .iter()
        .map(|stage| stage.id)
        .filter(|&id| {
            id >= 0
                && commissioning_active(station.entry_stage_id, station.exit_stage_id, id)
                && !hydro_operating_active(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    id,
                )
        })
        .min()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::doc_markdown)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{EntityId, FillingConfig, Hydro, entities::PumpingStation};

    use super::super::validate_semantic_hydro_thermal;
    use crate::test_support::{make_data, make_hydro, make_stages};
    use crate::validation::{ErrorKind, ValidationContext, schema::ParsedData};

    fn make_pumping(id: i32, bus_id: i32, src_hydro: i32, dst_hydro: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pump_{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(bus_id),
            source_hydro_id: EntityId::from(src_hydro),
            destination_hydro_id: EntityId::from(dst_hydro),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    /// A station whose source and destination hydros are identical (hydro 10)
    /// yields exactly one `InvalidValue` mentioning the degenerate condition and
    /// the shared hydro ID.
    #[test]
    fn test_pumping_source_equals_destination_invalid_value() {
        let mut data = make_data(
            vec![make_hydro(10, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![make_pumping(1, 1, 10, 10)];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidValue error, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        let msg = &inv[0].message;
        assert!(
            msg.contains("source_hydro_id == destination_hydro_id"),
            "message should describe the degenerate transfer, got: {msg}"
        );
        assert!(
            msg.contains("10"),
            "message should contain the shared hydro id '10', got: {msg}"
        );
    }

    /// A station with distinct existing source/destination hydros produces no error.
    #[test]
    fn test_pumping_distinct_endpoints_no_error() {
        let mut data = make_data(
            vec![make_hydro(10, None), make_hydro(20, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![make_pumping(1, 1, 10, 20)];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "distinct endpoints should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// A run-of-river endpoint (an existing hydro with no reservoir capacity) is
    /// accepted: only the source-equals-destination condition is rejected here.
    /// `make_hydro` yields a hydro with no geometry/FPHA — an RoR-like endpoint
    /// from the semantic check's perspective; its distinct ID resolves.
    #[test]
    fn test_pumping_run_of_river_endpoint_no_error() {
        let mut ror = make_hydro(30, None);
        ror.max_storage_hm3 = 0.0; // run-of-river: no reservoir storage
        let mut data = make_data(
            vec![make_hydro(10, None), ror],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![make_pumping(1, 1, 10, 30)];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "run-of-river endpoint should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    fn hydro_with_window(id: i32, entry: Option<i32>, exit: Option<i32>) -> Hydro {
        let mut hydro = make_hydro(id, None);
        hydro.entry_stage_id = entry;
        hydro.exit_stage_id = exit;
        hydro
    }

    fn station_with_window(
        src_hydro: i32,
        dst_hydro: i32,
        entry: Option<i32>,
        exit: Option<i32>,
    ) -> PumpingStation {
        let mut station = make_pumping(1, 1, src_hydro, dst_hydro);
        station.entry_stage_id = entry;
        station.exit_stage_id = exit;
        station
    }

    fn operating_window_messages(hydros: Vec<Hydro>, station: PumpingStation) -> Vec<String> {
        let mut data: ParsedData = make_data(
            hydros,
            vec![],
            vec![],
            make_stages(vec![0, 1, 2, 3]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![station];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        ctx.errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("PumpingStation")
            })
            .map(|e| e.message.clone())
            .collect()
    }

    #[test]
    fn pumping_station_active_while_its_source_hydro_has_exited_is_rejected() {
        let msgs = operating_window_messages(
            vec![
                hydro_with_window(10, None, Some(2)),
                hydro_with_window(20, None, None),
            ],
            station_with_window(10, 20, None, None),
        );
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        assert!(msgs[0].contains("PumpingStation 1"), "got: {}", msgs[0]);
        assert!(msgs[0].contains("source hydro 10"), "got: {}", msgs[0]);
        assert!(msgs[0].contains("stage 2"), "got: {}", msgs[0]);
    }

    #[test]
    fn pumping_station_active_while_its_destination_hydro_is_not_yet_built_is_rejected() {
        let msgs = operating_window_messages(
            vec![
                hydro_with_window(10, None, None),
                hydro_with_window(20, Some(2), None),
            ],
            station_with_window(10, 20, Some(1), None),
        );
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        assert!(msgs[0].contains("destination hydro 20"), "got: {}", msgs[0]);
        assert!(msgs[0].contains("stage 1"), "got: {}", msgs[0]);
    }

    #[test]
    fn pumping_station_with_both_endpoints_outside_their_windows_reports_each_side() {
        let msgs = operating_window_messages(
            vec![
                hydro_with_window(10, None, Some(2)),
                hydro_with_window(20, Some(1), None),
            ],
            station_with_window(10, 20, None, None),
        );
        assert_eq!(msgs.len(), 2, "got: {msgs:?}");
        assert!(
            msgs[0].contains("source hydro 10") && msgs[0].contains("stage 2"),
            "source error must come first, got: {}",
            msgs[0]
        );
        assert!(
            msgs[1].contains("destination hydro 20") && msgs[1].contains("stage 0"),
            "got: {}",
            msgs[1]
        );
    }

    #[test]
    fn pumping_station_window_ending_at_the_hydro_exit_is_accepted() {
        let msgs = operating_window_messages(
            vec![
                hydro_with_window(10, None, Some(3)),
                hydro_with_window(20, None, None),
            ],
            station_with_window(10, 20, Some(1), Some(3)),
        );
        assert!(msgs.is_empty(), "got: {msgs:?}");
    }

    #[test]
    fn pumping_station_active_one_stage_past_the_hydro_exit_names_that_stage() {
        let msgs = operating_window_messages(
            vec![
                hydro_with_window(10, None, Some(2)),
                hydro_with_window(20, None, None),
            ],
            station_with_window(10, 20, Some(1), Some(3)),
        );
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        assert!(msgs[0].contains("stage 2"), "got: {}", msgs[0]);
    }

    fn filling_destination() -> Hydro {
        let mut hydro = hydro_with_window(20, Some(3), None);
        hydro.filling = Some(FillingConfig {
            start_stage_id: 1,
            filling_min_rate_m3s: 1.0,
        });
        hydro
    }

    #[test]
    fn pumping_station_active_while_its_destination_hydro_is_filling_is_rejected() {
        let msgs = operating_window_messages(
            vec![hydro_with_window(10, None, None), filling_destination()],
            station_with_window(10, 20, Some(1), None),
        );
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        assert!(msgs[0].contains("destination hydro 20"), "got: {}", msgs[0]);
        assert!(msgs[0].contains("stage 1"), "got: {}", msgs[0]);
    }

    #[test]
    fn pumping_station_starting_at_the_filling_hydro_entry_is_accepted() {
        let msgs = operating_window_messages(
            vec![hydro_with_window(10, None, None), filling_destination()],
            station_with_window(10, 20, Some(3), None),
        );
        assert!(msgs.is_empty(), "got: {msgs:?}");
    }

    #[test]
    fn pumping_station_with_an_unresolved_endpoint_is_checked_on_the_resolved_side_only() {
        let msgs = operating_window_messages(
            vec![hydro_with_window(10, None, Some(2))],
            station_with_window(10, 99, None, None),
        );
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        assert!(msgs[0].contains("source hydro 10"), "got: {}", msgs[0]);
    }

    #[test]
    fn pumping_self_transfer_is_left_to_the_distinct_endpoints_rule() {
        let msgs = operating_window_messages(
            vec![hydro_with_window(10, None, Some(2))],
            station_with_window(10, 10, None, None),
        );
        assert!(msgs.is_empty(), "got: {msgs:?}");
    }
}
