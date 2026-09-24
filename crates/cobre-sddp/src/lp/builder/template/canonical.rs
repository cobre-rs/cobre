//! Canonical little-endian byte encoding of the stage-LP builder's facts,
//! exhaustively destructured so an added field cannot escape the digest.

use std::collections::BTreeMap;
use std::ops::Range;

use cobre_core::{BlockMode, EntityId};
use cobre_solver::StageTemplate;

use crate::lp::builder::{GenericConstraintRowEntry, StateBox};
use crate::lp::indexer::{EvaporationIndices, HydroSys};

use super::{StageGeometry, StageTemplates};

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

/// Group keys [`encode_lp_facts`] writes, guaranteed present even for zero
/// stages — the fixed schema [`encode_stage_templates_facts`]'s 28-key
/// contract depends on.
const LP_FACT_GROUP_KEYS: [&str; 11] = [
    "lp.dims",
    "lp.sparsity",
    "lp.values",
    "lp.col_bounds",
    "lp.row_bounds",
    "lp.objective",
    "lp.scaling",
    "solver_meta.n_transfer",
    "solver_meta.n_dual_relevant",
    "solver_meta.n_hydro",
    "solver_meta.max_par_order",
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
            n_transfer,
            n_dual_relevant,
            n_hydro,
            max_par_order,
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

        let n_transfer_group = group(groups, "solver_meta.n_transfer");
        put_usize(n_transfer_group, stage);
        put_usize(n_transfer_group, *n_transfer);

        let n_dual_relevant_group = group(groups, "solver_meta.n_dual_relevant");
        put_usize(n_dual_relevant_group, stage);
        put_usize(n_dual_relevant_group, *n_dual_relevant);

        let n_hydro_group = group(groups, "solver_meta.n_hydro");
        put_usize(n_hydro_group, stage);
        put_usize(n_hydro_group, *n_hydro);

        let max_par_order_group = group(groups, "solver_meta.max_par_order");
        put_usize(max_par_order_group, stage);
        put_usize(max_par_order_group, *max_par_order);
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

fn put_geometry(buf: &mut Vec<u8>, geometry: &StageGeometry) {
    let StageGeometry {
        theta_col,
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
        z_inflow_row_start,
        n_blks,
        storage_boundary_grid,
        block_mode,
        fpha_hydro_indices,
        evap_hydro_indices,
        filling_target_hydro_indices,
        filled_min_storage_floor_hydro_indices,
    } = geometry;

    put_usize(buf, *theta_col);
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
    put_range(buf, water_balance);
    put_range(buf, load_balance);
    put_range(buf, fpha);
    put_range(buf, filling_target);
    put_range(buf, filling_target_col);
    put_range(buf, filled_min_storage_floor);
    put_range(buf, filled_min_storage_floor_col);

    put_usize(buf, *z_inflow_row_start);
    put_usize(buf, *n_blks);

    for field in storage_boundary_grid.canonical_fields() {
        put_usize(buf, field);
    }

    let block_mode_tag: u8 = match block_mode {
        BlockMode::Parallel => 0,
        BlockMode::Chronological => 1,
    };
    buf.push(block_mode_tag);

    put_hydro_sys_slice(buf, fpha_hydro_indices);
    put_hydro_sys_slice(buf, evap_hydro_indices);
    put_hydro_sys_slice(buf, filling_target_hydro_indices);
    put_hydro_sys_slice(buf, filled_min_storage_floor_hydro_indices);
}

/// Encode every fact of [`StageTemplates`] — the LP templates plus every
/// other field of the struct and the nested types it holds, destructured
/// exhaustively so an added field fails to compile rather than
/// silently escaping the digest.
pub(crate) fn encode_stage_templates_facts(templates: &StageTemplates, groups: &mut FactGroups) {
    let StageTemplates {
        templates,
        state_boxes,
        base_rows,
        noise_scale,
        zeta_per_stage,
        block_hours_per_stage,
        n_hydros,
        cost_scale_factor,
        load_balance_row_starts,
        n_load_buses,
        load_bus_indices,
        generic_constraint_row_entries,
        ncs_col_starts,
        n_ncs,
        pumping_col_starts,
        n_pumping,
        geometry_per_stage,
        diversion_upstream,
        hydro_productivities_per_stage,
        discount_factors,
        cumulative_discount_factors,
    } = templates;

    encode_lp_facts(templates, groups);

    let state_boxes_buf = group(groups, "state_boxes");
    for (stage, state_box) in state_boxes.iter().enumerate() {
        let StateBox { lower, upper } = state_box;
        put_usize(state_boxes_buf, stage);
        put_f64_slice(state_boxes_buf, lower);
        put_f64_slice(state_boxes_buf, upper);
    }

    put_usize_slice(group(groups, "layout.base_rows"), base_rows);
    put_usize_slice(
        group(groups, "layout.load_balance_row_starts"),
        load_balance_row_starts,
    );

    let buf = group(groups, "layout.ncs_cols");
    put_usize_slice(buf, ncs_col_starts);
    put_usize(buf, *n_ncs);

    let buf = group(groups, "layout.pumping_cols");
    put_usize_slice(buf, pumping_col_starts);
    put_usize(buf, *n_pumping);

    let geometry_buf = group(groups, "layout.geometry");
    for (stage, geometry) in geometry_per_stage.iter().enumerate() {
        put_usize(geometry_buf, stage);
        put_geometry(geometry_buf, geometry);
    }

    put_f64_slice(group(groups, "stochastic.noise_scale"), noise_scale);

    let buf = group(groups, "stochastic.load_buses");
    put_usize(buf, *n_load_buses);
    put_usize_slice(buf, load_bus_indices);

    put_f64_slice(
        group(groups, "time_value.discount_factors"),
        discount_factors,
    );
    put_f64_slice(
        group(groups, "time_value.cumulative_discount_factors"),
        cumulative_discount_factors,
    );

    put_usize(group(groups, "reporting.n_hydros"), *n_hydros);
    put_f64(
        group(groups, "reporting.cost_scale_factor"),
        *cost_scale_factor,
    );
    put_f64_slice(group(groups, "reporting.zeta_per_stage"), zeta_per_stage);

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

#[cfg(test)]
mod tests {
    use super::{
        EntityId, FactGroups, StageTemplate, StageTemplates, encode_lp_facts,
        encode_stage_templates_facts,
    };

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
                "solver_meta.max_par_order",
                "solver_meta.n_dual_relevant",
                "solver_meta.n_hydro",
                "solver_meta.n_transfer",
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
        let templates = StageTemplates::empty(2, 1.0);
        let mut groups = FactGroups::new();
        encode_stage_templates_facts(&templates, &mut groups);
        let mut keys: Vec<&str> = groups.keys().copied().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "layout.base_rows",
                "layout.geometry",
                "layout.load_balance_row_starts",
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
                "reporting.n_hydros",
                "reporting.zeta_per_stage",
                "solver_meta.max_par_order",
                "solver_meta.n_dual_relevant",
                "solver_meta.n_hydro",
                "solver_meta.n_transfer",
                "state_boxes",
                "stochastic.load_buses",
                "stochastic.noise_scale",
                "time_value.cumulative_discount_factors",
                "time_value.discount_factors",
            ]
        );
    }

    #[test]
    fn diversion_order_does_not_change_the_bytes() {
        let mut a = StageTemplates::empty(2, 1.0);
        let mut b = StageTemplates::empty(2, 1.0);
        a.diversion_upstream.insert(EntityId(1), vec![10, 11]);
        a.diversion_upstream.insert(EntityId(2), vec![20]);
        b.diversion_upstream.insert(EntityId(2), vec![20]);
        b.diversion_upstream.insert(EntityId(1), vec![10, 11]);

        let mut groups_a = FactGroups::new();
        encode_stage_templates_facts(&a, &mut groups_a);
        let mut groups_b = FactGroups::new();
        encode_stage_templates_facts(&b, &mut groups_b);
        assert_eq!(groups_a, groups_b);
    }

    #[test]
    fn one_discount_factor_moves_only_its_group() {
        let mut a = StageTemplates::empty(2, 1.0);
        let mut b = StageTemplates::empty(2, 1.0);
        a.discount_factors = vec![1.0];
        b.discount_factors = vec![0.5];

        let mut groups_a = FactGroups::new();
        encode_stage_templates_facts(&a, &mut groups_a);
        let mut groups_b = FactGroups::new();
        encode_stage_templates_facts(&b, &mut groups_b);

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
