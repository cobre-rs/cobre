//! Canonical little-endian byte encoding of the stage-LP builder's facts,
//! exhaustively destructured so an added field cannot escape the digest.

use std::collections::BTreeMap;
use std::ops::Range;

use cobre_core::{BlockMode, EntityId, PostStudyThermalBound};
use cobre_solver::StageTemplate;

use crate::lp::builder::{GenericConstraintRowEntry, StageGeometry, StateBox};
use crate::lp::indexer::{
    BlockRowFamily, EvaporationIndices, HydroSys, StateSpace, StorageBoundaryGrid,
};
use crate::time_value::TimeValue;

use super::StageTemplates;

pub(crate) type FactGroups = BTreeMap<&'static str, Vec<u8>>;

fn group<'g>(groups: &'g mut FactGroups, key: &'static str) -> &'g mut Vec<u8> {
    groups.entry(key).or_default()
}

fn put_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_usize(buf: &mut Vec<u8>, value: usize) {
    put_u64(buf, value as u64);
}

fn put_i32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_f64(buf: &mut Vec<u8>, value: f64) {
    put_u64(buf, value.to_bits());
}

fn put_i32_slice(buf: &mut Vec<u8>, values: &[i32]) {
    put_u64(buf, values.len() as u64);
    for &v in values {
        put_i32(buf, v);
    }
}

fn put_f64_slice(buf: &mut Vec<u8>, values: &[f64]) {
    put_u64(buf, values.len() as u64);
    for &v in values {
        put_f64(buf, v);
    }
}

fn put_usize_slice(buf: &mut Vec<u8>, values: &[usize]) {
    put_u64(buf, values.len() as u64);
    for &v in values {
        put_usize(buf, v);
    }
}

fn put_bool(buf: &mut Vec<u8>, value: bool) {
    buf.push(u8::from(value));
}

fn put_option_f64(buf: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(v) => {
            buf.push(1);
            put_f64(buf, v);
        }
        None => buf.push(0),
    }
}

fn put_option_usize(buf: &mut Vec<u8>, value: Option<usize>) {
    match value {
        Some(v) => {
            buf.push(1);
            put_usize(buf, v);
        }
        None => buf.push(0),
    }
}

fn put_range(buf: &mut Vec<u8>, range: &Range<usize>) {
    put_usize(buf, range.start);
    put_usize(buf, range.end);
}

fn put_block_row_family(buf: &mut Vec<u8>, family: BlockRowFamily) {
    let (start, end, per_block) = family.canonical_fields();
    put_usize(buf, start);
    put_usize(buf, end);
    put_bool(buf, per_block);
}

/// Group keys [`encode_lp_facts`] writes, guaranteed present even for zero
/// stages — the fixed key set [`encode_stage_templates_facts`]'s contract
/// depends on.
const LP_FACT_GROUP_KEYS: [&str; 7] = [
    "lp.dims",
    "lp.sparsity",
    "lp.values",
    "lp.col_bounds",
    "lp.row_bounds",
    "lp.objective",
    "lp.scaling",
];

pub(crate) fn encode_lp_facts(templates: &[StageTemplate], groups: &mut FactGroups) {
    for key in LP_FACT_GROUP_KEYS {
        group(groups, key);
    }
    for (stage, template) in templates.iter().enumerate() {
        let StageTemplate {
            num_cols,
            num_rows,
            num_nz,
            col_starts,
            row_indices,
            values,
            col_lower,
            col_upper,
            objective,
            row_lower,
            row_upper,
            n_state,
            col_scale,
            row_scale,
        } = template;

        let dims = group(groups, "lp.dims");
        put_usize(dims, stage);
        put_usize(dims, *num_cols);
        put_usize(dims, *num_rows);
        put_usize(dims, *n_state);

        let sparsity = group(groups, "lp.sparsity");
        put_usize(sparsity, stage);
        put_usize(sparsity, *num_nz);
        put_i32_slice(sparsity, col_starts);
        put_i32_slice(sparsity, row_indices);

        let vals = group(groups, "lp.values");
        put_usize(vals, stage);
        put_f64_slice(vals, values);

        let col_bounds = group(groups, "lp.col_bounds");
        put_usize(col_bounds, stage);
        put_f64_slice(col_bounds, col_lower);
        put_f64_slice(col_bounds, col_upper);

        let row_bounds = group(groups, "lp.row_bounds");
        put_usize(row_bounds, stage);
        put_f64_slice(row_bounds, row_lower);
        put_f64_slice(row_bounds, row_upper);

        let objective_group = group(groups, "lp.objective");
        put_usize(objective_group, stage);
        put_f64_slice(objective_group, objective);

        let scaling = group(groups, "lp.scaling");
        put_usize(scaling, stage);
        put_f64_slice(scaling, col_scale);
        put_f64_slice(scaling, row_scale);
    }
}

fn put_evap_indices(buf: &mut Vec<u8>, indices: &EvaporationIndices) {
    let EvaporationIndices {
        evaporation_flow_col,
        f_evap_plus_col,
        f_evap_minus_col,
        evap_row,
    } = indices;
    put_usize(buf, *evaporation_flow_col);
    put_usize(buf, *f_evap_plus_col);
    put_usize(buf, *f_evap_minus_col);
    put_usize(buf, *evap_row);
}

fn put_hydro_sys_slice(buf: &mut Vec<u8>, values: &[HydroSys]) {
    put_usize(buf, values.len());
    for &v in values {
        put_usize(buf, v.get());
    }
}

fn put_gc_entry(buf: &mut Vec<u8>, entry: &GenericConstraintRowEntry) {
    let GenericConstraintRowEntry {
        constraint_idx,
        entity_id,
        block_idx,
        is_stage_level,
        bound_lower,
        bound_upper,
        slack_enabled,
        slack_penalty,
        slack_plus_col,
        slack_minus_col,
    } = entry;
    put_usize(buf, *constraint_idx);
    put_i32(buf, *entity_id);
    put_usize(buf, *block_idx);
    put_bool(buf, *is_stage_level);
    put_option_f64(buf, *bound_lower);
    put_option_f64(buf, *bound_upper);
    put_bool(buf, *slack_enabled);
    put_f64(buf, *slack_penalty);
    put_option_usize(buf, *slack_plus_col);
    put_option_usize(buf, *slack_minus_col);
}

fn entities_per_block(
    geometry_per_stage: &[StageGeometry],
    family: fn(&StageGeometry) -> &Range<usize>,
) -> usize {
    geometry_per_stage
        .first()
        .map_or(0, |g| family(g).len() / g.n_blks)
}

fn put_geometry(buf: &mut Vec<u8>, geometry: &StageGeometry, state: &StateSpace) {
    let StageGeometry {
        turbine,
        spillage,
        diversion,
        thermal,
        anticipated_decision,
        line_fwd,
        line_rev,
        deficit,
        excess,
        generation,
        // Encoded under layout.ncs_cols / layout.pumping_cols instead — see below.
        ncs_generation: _ncs_generation,
        pumping_flow: _pumping_flow,
        evap_indices,
        inflow_slack,
        withdrawal_slack_neg,
        withdrawal_slack_pos,
        outflow_below_slack,
        outflow_above_slack,
        turbine_below_slack,
        generation_below_slack,
        contract_import,
        contract_export,
        water_balance,
        load_balance,
        fpha,
        filling_target,
        filling_target_col,
        filled_min_storage_floor,
        filled_min_storage_floor_col,
        n_blks,
        storage_internal_start,
        block_mode,
        fpha_hydro_indices,
        evap_hydro_indices,
        filling_target_hydro_indices,
        filled_min_storage_floor_hydro_indices,
    } = geometry;

    put_usize(buf, state.theta);
    put_range(buf, turbine);
    put_range(buf, spillage);
    put_range(buf, diversion);
    put_range(buf, thermal);
    put_range(buf, anticipated_decision);
    put_range(buf, line_fwd);
    put_range(buf, line_rev);
    put_range(buf, deficit);
    put_range(buf, excess);
    put_range(buf, generation);

    put_usize(buf, evap_indices.len());
    for indices in evap_indices {
        put_evap_indices(buf, indices);
    }

    put_range(buf, inflow_slack);
    put_range(buf, withdrawal_slack_neg);
    put_range(buf, withdrawal_slack_pos);
    put_range(buf, outflow_below_slack);
    put_range(buf, outflow_above_slack);
    put_range(buf, turbine_below_slack);
    put_range(buf, generation_below_slack);
    put_range(buf, contract_import);
    put_range(buf, contract_export);
    put_block_row_family(buf, *water_balance);
    put_block_row_family(buf, *load_balance);
    put_range(buf, fpha);
    put_range(buf, filling_target);
    put_range(buf, filling_target_col);
    put_range(buf, filled_min_storage_floor);
    put_range(buf, filled_min_storage_floor_col);

    put_usize(buf, state.z_inflow_rows().start);
    put_usize(buf, *n_blks);

    put_usize(buf, state.storage_in.start);
    put_usize(buf, state.storage.start);
    for field in StorageBoundaryGrid::new(*storage_internal_start, *n_blks).canonical_fields() {
        put_usize(buf, field);
    }

    buf.push(match block_mode {
        BlockMode::Parallel => 0,
        BlockMode::Chronological => 1,
    });

    put_hydro_sys_slice(buf, fpha_hydro_indices);
    put_hydro_sys_slice(buf, evap_hydro_indices);
    put_hydro_sys_slice(buf, filling_target_hydro_indices);
    put_hydro_sys_slice(buf, filled_min_storage_floor_hydro_indices);
}

/// Encode every fact of [`StageTemplates`] — the LP templates plus every
/// other field of the struct and the nested types it holds, destructured
/// exhaustively so an added field fails to compile rather than
/// silently escaping the digest.
pub(crate) fn encode_stage_templates_facts(
    templates: &StageTemplates,
    state: &StateSpace,
    groups: &mut FactGroups,
) {
    let StageTemplates {
        templates,
        state_boxes,
        block_hours_per_stage,
        cost_scale_factor,
        load_bus_indices,
        generic_constraint_row_entries,
        geometry_per_stage,
        diversion_upstream,
        hydro_productivities_per_stage,
    } = templates;

    encode_lp_facts(templates, groups);

    let state_boxes_buf = group(groups, "state_boxes");
    for (stage, state_box) in state_boxes.iter().enumerate() {
        let StateBox { lower, upper } = state_box;
        put_usize(state_boxes_buf, stage);
        put_f64_slice(state_boxes_buf, lower);
        put_f64_slice(state_boxes_buf, upper);
    }

    let buf = group(groups, "layout.ncs_cols");
    put_u64(buf, geometry_per_stage.len() as u64);
    for g in geometry_per_stage {
        put_usize(buf, g.ncs_generation.start);
    }
    put_usize(
        buf,
        entities_per_block(geometry_per_stage, |g| &g.ncs_generation),
    );

    let buf = group(groups, "layout.pumping_cols");
    put_u64(buf, geometry_per_stage.len() as u64);
    for g in geometry_per_stage {
        put_usize(buf, g.pumping_flow.start);
    }
    put_usize(
        buf,
        entities_per_block(geometry_per_stage, |g| &g.pumping_flow),
    );

    let geometry_buf = group(groups, "layout.geometry");
    for (stage, geometry) in geometry_per_stage.iter().enumerate() {
        put_usize(geometry_buf, stage);
        put_geometry(geometry_buf, geometry, state);
    }

    let buf = group(groups, "stochastic.load_buses");
    put_usize(buf, load_bus_indices.len());
    put_usize_slice(buf, load_bus_indices);

    put_f64(
        group(groups, "reporting.cost_scale_factor"),
        *cost_scale_factor,
    );

    let block_hours_buf = group(groups, "reporting.block_hours_per_stage");
    for (stage, hours) in block_hours_per_stage.iter().enumerate() {
        put_usize(block_hours_buf, stage);
        put_f64_slice(block_hours_buf, hours);
    }

    let productivities_buf = group(groups, "reporting.hydro_productivities_per_stage");
    for (stage, productivities) in hydro_productivities_per_stage.iter().enumerate() {
        put_usize(productivities_buf, stage);
        put_f64_slice(productivities_buf, productivities);
    }

    let gc_buf = group(groups, "reporting.generic_constraint_row_entries");
    for (stage, entries) in generic_constraint_row_entries.iter().enumerate() {
        put_usize(gc_buf, stage);
        put_usize(gc_buf, entries.len());
        for entry in entries {
            put_gc_entry(gc_buf, entry);
        }
    }

    let buf = group(groups, "reporting.diversion_upstream");
    let mut sorted: Vec<(&EntityId, &Vec<usize>)> = diversion_upstream.iter().collect();
    sorted.sort_by_key(|(id, _)| id.0);
    put_usize(buf, sorted.len());
    for (id, values) in sorted {
        put_i32(buf, id.0);
        put_usize_slice(buf, values);
    }
}

fn put_post_study_thermal_bound(buf: &mut Vec<u8>, bound: &PostStudyThermalBound) {
    let PostStudyThermalBound {
        thermal_id,
        post_study_stage_index,
        cost_per_mwh,
        min_mw,
        max_mw,
    } = bound;
    put_i32(buf, thermal_id.0);
    put_usize(buf, *post_study_stage_index);
    put_f64(buf, *cost_per_mwh);
    put_f64(buf, *min_mw);
    put_f64(buf, *max_mw);
}

fn put_option_triple(buf: &mut Vec<u8>, value: Option<(f64, f64, f64)>) {
    match value {
        Some((a, b, c)) => {
            buf.push(1);
            put_f64(buf, a);
            put_f64(buf, b);
            put_f64(buf, c);
        }
        None => buf.push(0),
    }
}

/// Encode every fact of [`TimeValue`], destructured exhaustively via
/// [`TimeValue::canonical_fields`] (and its nested
/// [`crate::time_value::PostStudyResolved::canonical_fields`]/
/// [`crate::time_value::PostStudyThermalLookup::canonical_fields`]) so an added
/// field fails to compile rather than silently escaping the digest.
/// `time_value.cumulative_discount_factors` is written through
/// [`TimeValue::cumulative_discount_factors`] (the prefix accessor), never
/// recomputed, so it stays the single derivation.
pub(crate) fn encode_time_value_facts(time_value: &TimeValue, groups: &mut FactGroups) {
    let (discount_factors, delivery_cumulative_discount_factors, calendar, post_study) =
        time_value.canonical_fields();

    put_f64_slice(
        group(groups, "time_value.discount_factors"),
        discount_factors,
    );
    put_f64_slice(
        group(groups, "time_value.cumulative_discount_factors"),
        time_value.cumulative_discount_factors(),
    );
    put_f64_slice(
        group(groups, "time_value.delivery_cumulative_discount_factors"),
        delivery_cumulative_discount_factors,
    );

    // Dates are not LP facts; their sole LP effect, the post-study discount
    // continuation, is already digested via `post_study` below.
    let (total_hours, stage_ids, _post_study_stages) = calendar.canonical_fields();
    put_f64_slice(
        group(groups, "time_value.delivery_total_hours"),
        total_hours,
    );
    put_i32_slice(group(groups, "time_value.delivery_stage_ids"), stage_ids);

    let (
        post_study_cumulative_discount_factors,
        thermal_bounds,
        anticipated_bounds,
        anticipated_bounds_stride,
    ) = post_study.canonical_fields();
    let buf = group(groups, "time_value.post_study");
    put_f64_slice(buf, calendar.post_study_total_hours());
    put_f64_slice(buf, post_study_cumulative_discount_factors);
    let bounds = thermal_bounds.canonical_fields();
    put_usize(buf, bounds.len());
    for bound in bounds {
        put_post_study_thermal_bound(buf, bound);
    }
    put_usize(buf, anticipated_bounds.len());
    for &bound in anticipated_bounds {
        put_option_triple(buf, bound);
    }
    put_usize(buf, anticipated_bounds_stride);
}

#[cfg(test)]
mod tests {
    use super::{
        EntityId, FactGroups, StageTemplate, StageTemplates, encode_lp_facts,
        encode_stage_templates_facts, encode_time_value_facts,
    };
    use crate::test_support::state_layout;
    use crate::time_value::{PostStudyResolved, TimeValue};

    fn one_stage(template: StageTemplate) -> FactGroups {
        let mut groups = FactGroups::new();
        encode_lp_facts(&[template], &mut groups);
        groups
    }

    #[test]
    fn lp_groups_are_the_documented_keys() {
        let groups = one_stage(StageTemplate::empty());
        let mut keys: Vec<&str> = groups.keys().copied().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "lp.col_bounds",
                "lp.dims",
                "lp.objective",
                "lp.row_bounds",
                "lp.scaling",
                "lp.sparsity",
                "lp.values",
            ]
        );
    }

    #[test]
    fn signed_zero_moves_only_the_values_group() {
        let mut positive_template = StageTemplate::empty();
        positive_template.values = vec![0.0];
        let mut negative_template = StageTemplate::empty();
        negative_template.values = vec![-0.0];

        let positive = one_stage(positive_template);
        let negative = one_stage(negative_template);

        for (&key, value) in &positive {
            let other = &negative[key];
            if key == "lp.values" {
                assert_ne!(value, other, "lp.values must differ on signed zero");
            } else {
                assert_eq!(value, other, "{key} must not move on a values-only change");
            }
        }
    }

    #[test]
    fn encoding_is_a_pure_function_of_the_templates() {
        let mut stage_0 = StageTemplate::empty();
        stage_0.values = vec![1.0, 2.0];
        let mut stage_1 = StageTemplate::empty();
        stage_1.col_starts = vec![0, 1];
        let templates = vec![stage_0, stage_1];

        let mut a = FactGroups::new();
        encode_lp_facts(&templates, &mut a);
        let mut b = FactGroups::new();
        encode_lp_facts(&templates, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn stage_templates_groups_are_the_documented_keys() {
        let templates = StageTemplates::empty(1.0);
        let mut groups = FactGroups::new();
        encode_stage_templates_facts(&templates, &state_layout(0, 0), &mut groups);
        let mut keys: Vec<&str> = groups.keys().copied().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "layout.geometry",
                "layout.ncs_cols",
                "layout.pumping_cols",
                "lp.col_bounds",
                "lp.dims",
                "lp.objective",
                "lp.row_bounds",
                "lp.scaling",
                "lp.sparsity",
                "lp.values",
                "reporting.block_hours_per_stage",
                "reporting.cost_scale_factor",
                "reporting.diversion_upstream",
                "reporting.generic_constraint_row_entries",
                "reporting.hydro_productivities_per_stage",
                "state_boxes",
                "stochastic.load_buses",
            ]
        );
    }

    #[test]
    fn time_value_groups_are_the_documented_keys() {
        let tv =
            TimeValue::from_parts(vec![], vec![], vec![], vec![], PostStudyResolved::default());
        let mut groups = FactGroups::new();
        encode_time_value_facts(&tv, &mut groups);
        let mut keys: Vec<&str> = groups.keys().copied().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "time_value.cumulative_discount_factors",
                "time_value.delivery_cumulative_discount_factors",
                "time_value.delivery_stage_ids",
                "time_value.delivery_total_hours",
                "time_value.discount_factors",
                "time_value.post_study",
            ]
        );
    }

    #[test]
    fn diversion_order_does_not_change_the_bytes() {
        let mut a = StageTemplates::empty(1.0);
        let mut b = StageTemplates::empty(1.0);
        a.diversion_upstream.insert(EntityId(1), vec![10, 11]);
        a.diversion_upstream.insert(EntityId(2), vec![20]);
        b.diversion_upstream.insert(EntityId(2), vec![20]);
        b.diversion_upstream.insert(EntityId(1), vec![10, 11]);

        let mut groups_a = FactGroups::new();
        encode_stage_templates_facts(&a, &state_layout(0, 0), &mut groups_a);
        let mut groups_b = FactGroups::new();
        encode_stage_templates_facts(&b, &state_layout(0, 0), &mut groups_b);
        assert_eq!(groups_a, groups_b);
    }

    #[test]
    fn one_discount_factor_moves_only_its_group() {
        let a = TimeValue::from_parts(
            vec![1.0],
            vec![1.0],
            vec![10.0],
            vec![0],
            PostStudyResolved::default(),
        );
        let b = TimeValue::from_parts(
            vec![0.5],
            vec![1.0],
            vec![10.0],
            vec![0],
            PostStudyResolved::default(),
        );

        let mut groups_a = FactGroups::new();
        encode_time_value_facts(&a, &mut groups_a);
        let mut groups_b = FactGroups::new();
        encode_time_value_facts(&b, &mut groups_b);

        for (&key, value) in &groups_a {
            let other = &groups_b[key];
            if key == "time_value.discount_factors" {
                assert_ne!(value, other, "time_value.discount_factors must differ");
            } else {
                assert_eq!(
                    value, other,
                    "{key} must not move on a discount-only change"
                );
            }
        }
    }
}
