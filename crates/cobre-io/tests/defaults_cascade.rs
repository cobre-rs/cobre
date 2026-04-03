//! Integration tests for the config/stages defaults cascade.
//!
//! Each test exercises `parse_config` or `parse_stages` with various
//! combinations of present/absent optional fields and asserts that the
//! resulting structs reflect documented default values.  Error paths are
//! deliberately excluded here — they are covered by the unit tests in
//! `crates/cobre-io/src/config.rs` and `crates/cobre-io/src/stages.rs`.
#![allow(clippy::unwrap_used, clippy::panic, clippy::doc_markdown)]

use cobre_core::scenario::SamplingScheme;
use cobre_io::config::parse_config;
use cobre_io::stages::parse_stages;
use std::io::Write;
use tempfile::NamedTempFile;

// ── helper ────────────────────────────────────────────────────────────────────

fn write_json(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f
}

// ── config.json defaults cascade ─────────────────────────────────────────────

/// Given a minimal `config.json` with only `training.forward_passes` and
/// `training.stopping_rules` present, every optional section must fall back to
/// its documented default.
///
/// Acceptance criterion: modeling, training flags, simulation, policy, and
/// exports all resolve to their specification defaults.
#[test]
fn test_minimal_config_all_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 50,
            "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    // modeling defaults
    assert_eq!(
        cfg.modeling.inflow_non_negativity.method, "penalty",
        "inflow_non_negativity.method should default to 'penalty'"
    );
    assert!(
        (cfg.modeling.inflow_non_negativity.penalty_cost - 1000.0).abs() < f64::EPSILON,
        "inflow_non_negativity.penalty_cost should default to 1000.0"
    );

    // training optional flags
    assert!(
        cfg.training.enabled,
        "training.enabled should default to true"
    );
    assert_eq!(
        cfg.training.stopping_mode, "any",
        "training.stopping_mode should default to 'any'"
    );
    assert!(
        cfg.training.tree_seed.is_none(),
        "training.tree_seed should default to None when absent"
    );

    // simulation defaults
    assert!(
        !cfg.simulation.enabled,
        "simulation.enabled should default to false"
    );
    assert_eq!(
        cfg.simulation.num_scenarios, 2000,
        "simulation.num_scenarios should default to 2000"
    );

    // policy defaults
    assert_eq!(
        cfg.policy.mode,
        cobre_io::PolicyMode::Fresh,
        "policy.mode should default to 'fresh'"
    );
    assert_eq!(
        cfg.policy.path, "./policy",
        "policy.path should default to './policy'"
    );

    // exports defaults
    assert!(
        cfg.exports.training,
        "exports.training should default to true"
    );
    assert!(
        !cfg.exports.forward_detail,
        "exports.forward_detail should default to false"
    );
}

/// Given a `config.json` with `training.tree_seed: 99`, `parse_config` must
/// preserve the seed as `Some(99)`.
#[test]
fn test_config_explicit_seed_preserved() {
    let f = write_json(
        r#"{
          "training": {
            "seed": 99,
            "forward_passes": 50,
            "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert_eq!(
        cfg.training.tree_seed,
        Some(99),
        "training.tree_seed should be Some(99) when explicitly set"
    );
}

/// Given a `config.json` without `training.tree_seed`, `parse_config` must return
/// `training.tree_seed == None`.
#[test]
fn test_config_absent_seed_is_none() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 50,
            "stopping_rules": [{"type": "iteration_limit", "limit": 10}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert!(
        cfg.training.tree_seed.is_none(),
        "training.tree_seed must be None when not present in JSON"
    );
}

/// Given a full `config.json` with non-default values in every section,
/// no field should fall through to its default — explicit values are preserved.
#[test]
fn test_config_all_sections_explicit_no_defaults_applied() {
    let f = write_json(
        r#"{
          "modeling": {
            "inflow_non_negativity": {
              "method": "truncation",
              "penalty_cost": 500.0
            }
          },
          "training": {
            "enabled": false,
            "seed": 7,
            "forward_passes": 192,
            "stopping_rules": [{"type": "iteration_limit", "limit": 200}],
            "stopping_mode": "all"
          },
          "simulation": {
            "enabled": true,
            "num_scenarios": 500,
            "policy_type": "outer"
          },
          "policy": {
            "path": "./my_policy",
            "mode": "warm_start",
            "validate_compatibility": false
          },
          "exports": {
            "training": false,
            "cuts": false,
            "states": false,
            "vertices": false,
            "simulation": false,
            "forward_detail": true,
            "backward_detail": true
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    // modeling: non-default values preserved
    assert_eq!(cfg.modeling.inflow_non_negativity.method, "truncation");
    assert!(
        (cfg.modeling.inflow_non_negativity.penalty_cost - 500.0).abs() < f64::EPSILON,
        "explicit penalty_cost 500.0 should be preserved"
    );

    // training: non-default values preserved
    assert!(!cfg.training.enabled, "enabled: false should be preserved");
    assert_eq!(cfg.training.tree_seed, Some(7));
    assert_eq!(cfg.training.forward_passes, Some(192));
    assert_eq!(cfg.training.stopping_mode, "all");

    // simulation: non-default values preserved
    assert!(
        cfg.simulation.enabled,
        "simulation.enabled: true should be preserved"
    );
    assert_eq!(cfg.simulation.num_scenarios, 500);

    // policy: non-default values preserved
    assert_eq!(cfg.policy.path, "./my_policy");
    assert_eq!(cfg.policy.mode, cobre_io::PolicyMode::WarmStart);
    assert!(!cfg.policy.validate_compatibility);

    // exports: non-default values preserved
    assert!(!cfg.exports.training);
    assert!(cfg.exports.forward_detail);
    assert!(cfg.exports.backward_detail);
}

/// Given a `config.json` where the `modeling` section is absent, the
/// `inflow_non_negativity` defaults must be applied: method = `"penalty"`,
/// penalty_cost = `1000.0`.
#[test]
fn test_config_absent_modeling_uses_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 10,
            "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert_eq!(
        cfg.modeling.inflow_non_negativity.method, "penalty",
        "absent modeling section must default method to 'penalty'"
    );
    assert!(
        (cfg.modeling.inflow_non_negativity.penalty_cost - 1000.0).abs() < f64::EPSILON,
        "absent modeling section must default penalty_cost to 1000.0"
    );
}

/// Given a `config.json` where the `simulation` section is absent, the
/// defaults must be: `enabled = false`, `num_scenarios = 2000`.
#[test]
fn test_config_absent_simulation_uses_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 10,
            "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert!(
        !cfg.simulation.enabled,
        "absent simulation section must default enabled to false"
    );
    assert_eq!(
        cfg.simulation.num_scenarios, 2000,
        "absent simulation section must default num_scenarios to 2000"
    );
}

/// Given a `config.json` where the `exports` section is absent, the defaults
/// must be: `training = true`, `forward_detail = false`.
#[test]
fn test_config_absent_exports_uses_defaults() {
    let f = write_json(
        r#"{
          "training": {
            "forward_passes": 10,
            "stopping_rules": [{"type": "iteration_limit", "limit": 5}]
          }
        }"#,
    );
    let cfg = parse_config(f.path()).unwrap();

    assert!(
        cfg.exports.training,
        "absent exports section must default exports.training to true"
    );
    assert!(
        !cfg.exports.forward_detail,
        "absent exports section must default exports.forward_detail to false"
    );
}

// ── stages.json defaults cascade ─────────────────────────────────────────────

/// Minimal stages.json fixture used across scenario-source tests.
///
/// Includes the mandatory `policy_graph` and one study stage.
fn minimal_stages_json_with(scenario_source_fragment: &str) -> String {
    format!(
        r#"{{
          "policy_graph": {{
            "type": "finite_horizon",
            "annual_discount_rate": 0.0,
            "transitions": []
          }},
          {scenario_source_fragment}
          "stages": [{{
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{{"id": 0, "name": "LEVE", "hours": 744.0}}],
            "num_scenarios": 10
          }}]
        }}"#
    )
}

/// Given a `stages.json` without a `scenario_source` field, `parse_stages`
/// must return `ScenarioSource::default()`: InSample, seed = None, selection_mode = None.
#[test]
fn test_stages_absent_scenario_source_uses_defaults() {
    let json = minimal_stages_json_with("");
    let f = write_json(&json);
    let data = parse_stages(f.path()).unwrap();

    assert_eq!(
        data.scenario_source.sampling_scheme,
        SamplingScheme::InSample,
        "absent scenario_source must default sampling_scheme to InSample"
    );
    assert!(
        data.scenario_source.seed.is_none(),
        "absent scenario_source must default seed to None"
    );
    assert!(
        data.scenario_source.selection_mode.is_none(),
        "absent scenario_source must default selection_mode to None"
    );
}

/// Given a `stages.json` with `scenario_source.seed: 42`, `parse_stages`
/// must preserve the seed as `Some(42)`.
#[test]
fn test_stages_scenario_source_seed_preserved() {
    let json = minimal_stages_json_with(
        r#""scenario_source": { "sampling_scheme": "in_sample", "seed": 42 },"#,
    );
    let f = write_json(&json);
    let data = parse_stages(f.path()).unwrap();

    assert_eq!(
        data.scenario_source.seed,
        Some(42),
        "scenario_source.seed must be Some(42) when explicitly set"
    );
    assert_eq!(
        data.scenario_source.sampling_scheme,
        SamplingScheme::InSample,
        "sampling_scheme must be InSample"
    );
    assert!(
        data.scenario_source.selection_mode.is_none(),
        "selection_mode must be None for InSample"
    );
}

/// Given a `stages.json` where stages have different `num_scenarios` values,
/// `parse_stages` must preserve the per-stage branching factors.
#[test]
fn test_stages_variable_branching_factor_preserved() {
    let json = r#"{
      "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.0,
        "transitions": []
      },
      "stages": [
        {
          "id": 0,
          "start_date": "2024-01-01",
          "end_date": "2024-02-01",
          "blocks": [{"id": 0, "name": "A", "hours": 744.0}],
          "num_scenarios": 5
        },
        {
          "id": 1,
          "start_date": "2024-02-01",
          "end_date": "2024-03-01",
          "blocks": [{"id": 0, "name": "A", "hours": 672.0}],
          "num_scenarios": 20
        },
        {
          "id": 2,
          "start_date": "2024-03-01",
          "end_date": "2024-04-01",
          "blocks": [{"id": 0, "name": "A", "hours": 744.0}],
          "num_scenarios": 50
        }
      ]
    }"#;
    let f = write_json(json);
    let data = parse_stages(f.path()).unwrap();

    // Stages are sorted by id ascending; only study stages (id >= 0) appear.
    let study_stages: Vec<_> = data.stages.iter().filter(|s| s.id >= 0).collect();
    assert_eq!(study_stages.len(), 3, "should have exactly 3 study stages");

    assert_eq!(
        study_stages[0].scenario_config.branching_factor, 5,
        "stage 0 branching_factor must be 5"
    );
    assert_eq!(
        study_stages[1].scenario_config.branching_factor, 20,
        "stage 1 branching_factor must be 20"
    );
    assert_eq!(
        study_stages[2].scenario_config.branching_factor, 50,
        "stage 2 branching_factor must be 50"
    );
}
