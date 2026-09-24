//! Canonical little-endian byte encoding of the stage-LP builder's facts,
//! exhaustively destructured so an added field cannot escape the digest.

use std::collections::BTreeMap;

use cobre_solver::StageTemplate;

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

pub(crate) fn encode_lp_facts(templates: &[StageTemplate], groups: &mut FactGroups) {
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

#[cfg(test)]
mod tests {
    use super::{FactGroups, StageTemplate, encode_lp_facts};

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
}
