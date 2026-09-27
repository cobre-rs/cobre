//! Pins `StochasticContext`'s noise-vector segment layout to the entity
//! order `noise_entity_order` derives, so the two cannot silently diverge.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use cobre_core::{EntityId, LoadModel, NcsModel, SamplingScheme, SystemBuilder};
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, build_stochastic_context, noise_entity_order,
};

mod common;
use common::{default_inflow_model, deficit_bus, saa_stage, sized_hydro};

fn build_system(with_load: bool) -> cobre_core::System {
    let mut builder = SystemBuilder::new()
        .buses(vec![deficit_bus(0)])
        .hydros(vec![sized_hydro(1), sized_hydro(2)])
        .stages(vec![saa_stage(0, 0, 2)])
        .inflow_models(vec![default_inflow_model(1, 0), default_inflow_model(2, 0)])
        .ncs_models(vec![NcsModel {
            ncs_id: EntityId(20),
            stage_id: 0,
            mean: 0.7,
            std: 0.1,
        }]);
    if with_load {
        builder = builder.load_models(vec![LoadModel {
            bus_id: EntityId(0),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 10.0,
        }]);
    }
    builder.build().unwrap()
}

fn in_sample_schemes() -> ClassSchemes {
    ClassSchemes {
        inflow: Some(SamplingScheme::InSample),
        load: Some(SamplingScheme::InSample),
        ncs: Some(SamplingScheme::InSample),
    }
}

#[test]
fn every_class_present_layout_matches_entity_order() {
    let system = build_system(true);
    let schemes = in_sample_schemes();
    let order = noise_entity_order(&system, &schemes);
    assert_eq!(order.hydro_ids.len(), 2, "class precondition: 2 hydros");
    assert_eq!(
        order.load_bus_ids.len(),
        1,
        "class precondition: 1 stochastic load bus"
    );
    assert_eq!(order.ncs_entity_ids.len(), 1, "class precondition: 1 NCS");

    let ctx = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        schemes,
    )
    .unwrap();

    let (hydro, rest) = ctx.entity_order().split_at(ctx.n_hydros());
    let (load, ncs) = rest.split_at(ctx.n_load_buses());

    assert_eq!(hydro, order.hydro_ids.as_slice());
    assert_eq!(load, order.load_bus_ids.as_slice());
    assert_eq!(ncs, order.ncs_entity_ids.as_slice());
    assert_eq!(ncs, ctx.ncs_entity_ids());
    assert_eq!(ncs.len(), ctx.n_stochastic_ncs());
    assert_eq!(ctx.dim(), ctx.entity_order().len());

    let dims = ctx.class_dimensions();
    assert_eq!(&ctx.entity_order()[dims.hydro_range()], hydro);
    assert_eq!(&ctx.entity_order()[dims.load_bus_range()], load);
    assert_eq!(&ctx.entity_order()[dims.ncs_range()], ncs);
    assert_eq!(dims.total(), ctx.dim());
}

#[test]
fn empty_load_class_layout_matches_entity_order() {
    let system = build_system(false);
    let schemes = in_sample_schemes();
    let order = noise_entity_order(&system, &schemes);
    assert_eq!(order.hydro_ids.len(), 2, "class precondition: 2 hydros");
    assert_eq!(
        order.load_bus_ids.len(),
        0,
        "class precondition: empty load class"
    );
    assert_eq!(order.ncs_entity_ids.len(), 1, "class precondition: 1 NCS");

    let ctx = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        schemes,
    )
    .unwrap();

    let (hydro, rest) = ctx.entity_order().split_at(ctx.n_hydros());
    let (load, ncs) = rest.split_at(ctx.n_load_buses());

    assert_eq!(hydro, order.hydro_ids.as_slice());
    assert_eq!(load, order.load_bus_ids.as_slice());
    assert!(load.is_empty());
    assert_eq!(ncs, order.ncs_entity_ids.as_slice());
    assert_eq!(ncs, ctx.ncs_entity_ids());
    assert_eq!(ncs.len(), ctx.n_stochastic_ncs());
    assert_eq!(ctx.dim(), ctx.entity_order().len());

    let dims = ctx.class_dimensions();
    assert_eq!(&ctx.entity_order()[dims.hydro_range()], hydro);
    assert_eq!(&ctx.entity_order()[dims.load_bus_range()], load);
    assert_eq!(&ctx.entity_order()[dims.ncs_range()], ncs);
    assert_eq!(dims.total(), ctx.dim());
}
