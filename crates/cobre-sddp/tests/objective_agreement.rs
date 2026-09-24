//! Per-LP objective baseline for the water-balance reformulation gate: the
//! optimal objective of every production-patched stage LP (every opening of
//! every node, at two trial incoming states) plus the no-cut root lower
//! bound, on `HiGHS` only. A row operation on the water-balance rows (the
//! z-inflow reformulation) is exact in exact arithmetic, so this baseline
//! must reproduce across that change.
//!
//! Temporary safety net for the stage-LP builder consolidation: this baseline
//! and its two tests are deleted once the reformulation lands.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

#[cfg(feature = "highs")]
mod highs {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use cobre_sddp::StudySetup;
    use cobre_sddp::setup::NodePos;
    use cobre_sddp::test_support::{
        capture_patched_node_template_at, no_cut_root_lower_bound, node_opening_noise,
        oracle_initial_state, stage_state_box_bounds,
    };
    use cobre_solver::highs::HighsSolver;
    use cobre_solver::{SolverError, SolverInterface, StageTemplate};

    use super::common::decks::{Deck, committed_decks};
    use super::common::fresh_setup_with;

    fn baseline_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/objective_agreement_baseline.tsv")
    }

    /// The two trial incoming states at `stage`: `a` is
    /// [`oracle_initial_state`] clamped elementwise into the stage's box; `b`
    /// is the box midpoint where both bounds are finite, else falls back to
    /// `a`'s value in that dimension.
    fn trial_states(setup: &StudySetup, stage: usize) -> [(&'static str, Vec<f64>); 2] {
        let (lo, hi) = stage_state_box_bounds(setup, stage);
        let initial_state = oracle_initial_state(setup);
        let trial_a: Vec<f64> = initial_state
            .iter()
            .zip(&lo)
            .zip(&hi)
            .map(|((&x, &l), &h)| x.max(l).min(h))
            .collect();
        let trial_b: Vec<f64> = lo
            .iter()
            .zip(&hi)
            .zip(&trial_a)
            .map(|((&l, &h), &a)| {
                if l.is_finite() && h.is_finite() {
                    f64::midpoint(l, h)
                } else {
                    a
                }
            })
            .collect();
        [("a", trial_a), ("b", trial_b)]
    }

    /// Cold-solves `template` on a fresh `HighsSolver`: the objective bits, or
    /// `"infeasible"` on `SolverError::Infeasible`. Any other solver error
    /// panics naming `record_key`, so a real solver failure never silently
    /// becomes a baseline entry.
    fn solve_objective_record(
        deck_key: &str,
        record_key: &str,
        template: &StageTemplate,
    ) -> String {
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        solver.load_model(template);
        match solver.solve(None) {
            Ok(view) => format!("{:016x}", view.objective.to_bits()),
            Err(SolverError::Infeasible) => "infeasible".to_string(),
            Err(other) => panic!("{deck_key}\t{record_key}: unexpected solver error {other:?}"),
        }
    }

    /// Every `node{pos}/trial{a|b}/opening{j}` record plus the deck's
    /// `lower_bound` record, unsorted.
    fn deck_records(deck: &Deck) -> Vec<String> {
        let setup = fresh_setup_with(&deck.dir, |_| {});
        let mut records = Vec::new();

        for pos in (0..setup.node_graph.nodes.len()).map(NodePos) {
            let node = setup.node_graph.nodes[pos];
            let stage = node.stage.0;
            for (trial_label, incoming_state) in trial_states(&setup, stage.saturating_sub(1)) {
                for opening in 0..node.openings.len {
                    let record_key = format!("node{pos}/trial{trial_label}/opening{opening}");
                    let raw_noise = node_opening_noise(&setup, pos, opening);
                    let template =
                        capture_patched_node_template_at(&setup, pos, &raw_noise, &incoming_state);
                    let value = solve_objective_record(&deck.key, &record_key, &template);
                    records.push(format!("{}\t{record_key}\t{value}", deck.key));
                }
            }
        }

        let mut lb_solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        let lb = no_cut_root_lower_bound(&setup, &mut lb_solver)
            .expect("no_cut_root_lower_bound must succeed on a freshly built StudySetup");
        let scaled_lb = lb / setup.stage_data.stage_templates.cost_scale_factor;
        records.push(format!(
            "{}\tlower_bound\t{:016x}",
            deck.key,
            scaled_lb.to_bits()
        ));

        records
    }

    /// Every committed deck but `examples/4ree`, sorted by key.
    fn baseline_decks() -> Vec<Deck> {
        committed_decks()
            .into_iter()
            .filter(|deck| deck.key != "examples/4ree")
            .collect()
    }

    fn recompute_lines() -> Vec<String> {
        let mut lines: Vec<String> = baseline_decks().iter().flat_map(deck_records).collect();
        lines.sort();
        lines
    }

    fn parse_lines(lines: &[String]) -> BTreeMap<(String, String), String> {
        lines
            .iter()
            .map(|line| {
                let mut fields = line.split('\t');
                let deck = fields.next().expect("deck field");
                let key = fields.next().expect("record key field");
                let value = fields.next().expect("value field");
                assert!(
                    fields.next().is_none(),
                    "unexpected extra field in line: {line}"
                );
                ((deck.to_string(), key.to_string()), value.to_string())
            })
            .collect()
    }

    fn agrees(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-8 * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn bits_to_f64(hex: &str) -> f64 {
        f64::from_bits(
            u64::from_str_radix(hex, 16)
                .unwrap_or_else(|e| panic!("{hex} is not 16 hex digits: {e}")),
        )
    }

    /// `None` when `committed`/`computed` agree; otherwise a one-line mismatch
    /// description naming which of the three failure modes fired.
    fn describe_mismatch(
        deck: &str,
        key: &str,
        committed: Option<&str>,
        computed: Option<&str>,
    ) -> Option<String> {
        match (committed, computed) {
            (None, Some(d)) => Some(format!(
                "{deck}\t{key}: present only in the recomputed set (value {d})"
            )),
            (Some(c), None) => Some(format!(
                "{deck}\t{key}: present only in the committed baseline (value {c})"
            )),
            (None, None) => {
                unreachable!("describe_mismatch is only called with at least one side present")
            }
            (Some(c), Some(d)) => {
                let c_infeasible = c == "infeasible";
                let d_infeasible = d == "infeasible";
                if c_infeasible != d_infeasible {
                    return Some(format!(
                        "{deck}\t{key}: infeasible mismatch (baseline={c}, recomputed={d})"
                    ));
                }
                if c_infeasible {
                    return None;
                }
                let a = bits_to_f64(c);
                let b = bits_to_f64(d);
                if agrees(a, b) {
                    None
                } else {
                    Some(format!(
                        "{deck}\t{key}: objective mismatch (baseline={a}, recomputed={b})"
                    ))
                }
            }
        }
    }

    #[test]
    #[ignore = "temporary objective-agreement gate; run explicitly with --ignored"]
    fn objective_agreement_matches_baseline() {
        let committed_text =
            std::fs::read_to_string(baseline_path()).expect("read objective agreement baseline");
        let committed_lines: Vec<String> = committed_text.lines().map(str::to_string).collect();
        let committed = parse_lines(&committed_lines);

        let computed_lines = recompute_lines();
        let computed = parse_lines(&computed_lines);

        let mut keys: Vec<(String, String)> =
            committed.keys().chain(computed.keys()).cloned().collect();
        keys.sort();
        keys.dedup();

        let mismatches: Vec<String> = keys
            .iter()
            .filter_map(|key| {
                describe_mismatch(
                    &key.0,
                    &key.1,
                    committed.get(key).map(String::as_str),
                    computed.get(key).map(String::as_str),
                )
            })
            .collect();

        assert!(
            mismatches.is_empty(),
            "objective-agreement mismatches ({}):\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }

    #[test]
    #[ignore = "rewrites the committed objective-agreement baseline; run explicitly"]
    fn objective_agreement_regen() {
        let lines = recompute_lines();
        let mut content = lines.join("\n");
        content.push('\n');

        let path = baseline_path();
        let tmp_path = path.with_extension("tsv.tmp");
        std::fs::write(&tmp_path, content).expect("write temporary baseline");
        std::fs::rename(&tmp_path, &path).expect("rename temporary baseline into place");
    }
}
