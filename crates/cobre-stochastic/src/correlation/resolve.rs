//! Resolution of correlation profiles from case input data.
//!
//! Maps correlation profile specifications from the loaded case into
//! dense matrices indexed by the internal hydro plant registry order.
//! Validates that all referenced hydro plants exist in the registry and
//! that correlation entries are in the valid range `[-1.0, 1.0]`.
//!
//! Returns [`StochasticError::InvalidCorrelation`] when validation fails.
//!
//! [`StochasticError::InvalidCorrelation`]: crate::StochasticError::InvalidCorrelation

use std::collections::{BTreeMap, HashMap};

use cobre_core::{CorrelationModel, EntityId};

use crate::{correlation::cholesky::CholeskyFactor, StochasticError};

/// Maximum group dimension for stack-allocated buffers in `apply_correlation`.
/// Groups with more entities than this threshold use heap-allocated buffers.
const MAX_STACK_DIM: usize = 64;

/// A single correlation group's Cholesky factor with entity ID mapping.
#[derive(Debug)]
pub struct GroupFactor {
    /// The Cholesky factor for this group.
    pub factor: CholeskyFactor,
    /// Entity IDs in this group, in the order matching the factor rows/columns.
    pub entity_ids: Vec<EntityId>,
    /// Pre-computed positions of this group's entities within the canonical
    /// entity order. Filled by [`DecomposedCorrelation::resolve_positions`].
    /// When `Some`, `apply_correlation` skips the per-call linear scan.
    positions: Option<Box<[usize]>>,
}

/// Pre-decomposed correlation data for all profiles, with stage-to-profile mapping.
///
/// Built once during initialization. At runtime, the noise generator looks up
/// the active profile for the current stage via `profile_for_stage()` and
/// applies the Cholesky transform using the cached factor.
#[derive(Debug)]
pub struct DecomposedCorrelation {
    /// Cholesky factors keyed by profile name.
    /// `BTreeMap` preserves deterministic iteration order.
    factors: BTreeMap<String, Vec<GroupFactor>>,

    /// Stage-to-profile-name mapping. For stages not in this map,
    /// the "default" profile is used.
    schedule: HashMap<i32, String>,

    /// Name of the default profile.
    default_profile: String,
}

impl DecomposedCorrelation {
    /// Builds a `DecomposedCorrelation` from a [`CorrelationModel`].
    ///
    /// Decomposes each profile's correlation groups into Cholesky factors and
    /// builds the stage-to-profile schedule. Validates that a `"default"`
    /// profile exists, or that exactly one profile exists (which then serves
    /// as the default).
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InvalidCorrelation`] if no `"default"` profile exists
    ///   and there is more than one profile (ambiguous default).
    /// - [`StochasticError::InvalidCorrelation`] if the model has no profiles.
    /// - [`StochasticError::InvalidCorrelation`] if a correlation matrix is not
    ///   square or not symmetric.
    /// - [`StochasticError::CholeskyDecompositionFailed`] if a matrix is not
    ///   positive-definite.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use cobre_core::{EntityId, scenario::{
    ///     CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
    /// }};
    /// use cobre_stochastic::correlation::resolve::DecomposedCorrelation;
    ///
    /// let mut profiles = BTreeMap::new();
    /// profiles.insert("default".to_string(), CorrelationProfile {
    ///     groups: vec![CorrelationGroup {
    ///         name: "g1".to_string(),
    ///         entities: vec![
    ///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(1) },
    ///             CorrelationEntity { entity_type: "inflow".to_string(), id: EntityId(2) },
    ///         ],
    ///         matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
    ///     }],
    /// });
    /// let model = CorrelationModel { method: "cholesky".to_string(), profiles, schedule: vec![] };
    /// let dc = DecomposedCorrelation::build(&model).unwrap();
    /// ```
    pub fn build(model: &CorrelationModel) -> Result<Self, StochasticError> {
        if model.profiles.is_empty() {
            return Err(StochasticError::InvalidCorrelation {
                profile_name: String::new(),
                reason: "correlation model contains no profiles".into(),
            });
        }

        // Determine the default profile name.
        let default_profile = if model.profiles.contains_key("default") {
            "default".to_string()
        } else if model.profiles.len() == 1 {
            // Use the single profile as implicit default (empty map ruled out above).
            model.profiles.keys().next().cloned().unwrap_or_default()
        } else {
            return Err(StochasticError::InvalidCorrelation {
                profile_name: String::new(),
                reason: format!(
                    "no 'default' profile found and {} profiles exist; \
                     add a profile named 'default' or reduce to a single profile",
                    model.profiles.len()
                ),
            });
        };

        // Decompose each profile.
        let mut factors: BTreeMap<String, Vec<GroupFactor>> = BTreeMap::new();
        for (profile_name, profile) in &model.profiles {
            let mut group_factors: Vec<GroupFactor> = Vec::with_capacity(profile.groups.len());
            for group in &profile.groups {
                let factor = CholeskyFactor::decompose(&group.matrix).map_err(|e| match e {
                    StochasticError::CholeskyDecompositionFailed { reason, .. } => {
                        StochasticError::CholeskyDecompositionFailed {
                            profile_name: profile_name.clone(),
                            reason,
                        }
                    }
                    StochasticError::InvalidCorrelation { reason, .. } => {
                        StochasticError::InvalidCorrelation {
                            profile_name: profile_name.clone(),
                            reason,
                        }
                    }
                    other => other,
                })?;

                let entity_ids: Vec<EntityId> = group.entities.iter().map(|e| e.id).collect();

                group_factors.push(GroupFactor {
                    factor,
                    entity_ids,
                    positions: None,
                });
            }
            factors.insert(profile_name.clone(), group_factors);
        }

        // Build the stage-to-profile schedule.
        let schedule: HashMap<i32, String> = model
            .schedule
            .iter()
            .map(|entry| (entry.stage_id, entry.profile_name.clone()))
            .collect();

        Ok(Self {
            factors,
            schedule,
            default_profile,
        })
    }

    /// Returns the profile name active for the given stage.
    ///
    /// Looks up `stage_id` in the schedule; falls back to the default profile.
    pub fn profile_for_stage(&self, stage_id: i32) -> &str {
        self.schedule
            .get(&stage_id)
            .map_or(self.default_profile.as_str(), String::as_str)
    }

    /// Pre-computes entity position indices for all correlation groups.
    ///
    /// Call this once after building the correlation data, before entering
    /// the hot loop that calls [`Self::apply_correlation`]. This eliminates the
    /// per-call O(n) linear scan over `entity_order`.
    ///
    /// Each `GroupFactor` stores the positions of its entity IDs within the
    /// canonical `entity_order` slice. Groups whose entities do not appear
    /// in `entity_order` get an empty position array and are skipped during
    /// correlation application.
    pub fn resolve_positions(&mut self, entity_order: &[EntityId]) {
        let id_to_pos: HashMap<EntityId, usize> = entity_order
            .iter()
            .enumerate()
            .map(|(i, &eid)| (eid, i))
            .collect();

        for group_factors in self.factors.values_mut() {
            for gf in group_factors.iter_mut() {
                let positions: Vec<usize> = gf
                    .entity_ids
                    .iter()
                    .filter_map(|eid| id_to_pos.get(eid).copied())
                    .collect();
                gf.positions = Some(positions.into_boxed_slice());
            }
        }
    }

    /// Applies spatial correlation to `independent_noise` for the given stage.
    ///
    /// Looks up the active correlation profile for `stage_id`, then for each
    /// correlation group in that profile:
    ///
    /// 1. Finds the positions of the group's entity IDs within `entity_order`.
    /// 2. Gathers the independent noise values for those positions.
    /// 3. Applies the group's Cholesky factor in-place.
    /// 4. Scatters the correlated values back to the matching positions.
    ///
    /// Entities that do not appear in any correlation group retain their
    /// independent noise values unchanged.
    ///
    /// For best performance, call [`resolve_positions`] once before entering
    /// the hot loop. When positions are pre-computed, this method performs
    /// zero heap allocations for groups of up to 64 entities.
    ///
    /// [`resolve_positions`]: Self::resolve_positions
    pub fn apply_correlation(
        &self,
        stage_id: i32,
        independent_noise: &mut [f64],
        entity_order: &[EntityId],
    ) {
        let profile_name = self.profile_for_stage(stage_id);
        let Some(group_factors) = self.factors.get(profile_name) else {
            // Profile not found — leave noise unchanged (defensive).
            return;
        };

        for gf in group_factors {
            // Use pre-computed positions if available; fall back to linear scan.
            if let Some(ref precomputed) = gf.positions {
                Self::apply_group_precomputed(&gf.factor, precomputed, independent_noise);
            } else {
                Self::apply_group_scan(&gf.factor, &gf.entity_ids, independent_noise, entity_order);
            }
        }
    }

    /// Fast path: positions are pre-computed. Uses stack buffers for groups ≤ 64.
    fn apply_group_precomputed(factor: &CholeskyFactor, positions: &[usize], noise: &mut [f64]) {
        let n = positions.len();
        if n == 0 || n != factor.dim() {
            return;
        }

        if n <= MAX_STACK_DIM {
            let mut gathered = [0.0_f64; MAX_STACK_DIM];
            let mut correlated = [0.0_f64; MAX_STACK_DIM];
            for (i, &pos) in positions.iter().enumerate() {
                gathered[i] = noise[pos];
            }
            factor.transform(&gathered[..n], &mut correlated[..n]);
            for (i, &pos) in positions.iter().enumerate() {
                noise[pos] = correlated[i];
            }
        } else {
            let gathered: Vec<f64> = positions.iter().map(|&pos| noise[pos]).collect();
            let mut correlated = vec![0.0_f64; n];
            factor.transform(&gathered, &mut correlated);
            for (i, &pos) in positions.iter().enumerate() {
                noise[pos] = correlated[i];
            }
        }
    }

    /// Slow path: linear scan for positions (backward compatibility without `resolve_positions`).
    fn apply_group_scan(
        factor: &CholeskyFactor,
        entity_ids: &[EntityId],
        noise: &mut [f64],
        entity_order: &[EntityId],
    ) {
        let positions: Vec<usize> = entity_ids
            .iter()
            .filter_map(|eid| entity_order.iter().position(|e| e == eid))
            .collect();

        let n = positions.len();
        if n == 0 || n != factor.dim() {
            return;
        }

        let gathered: Vec<f64> = positions.iter().map(|&pos| noise[pos]).collect();
        let mut correlated = vec![0.0_f64; n];
        factor.transform(&gathered, &mut correlated);
        for (i, &pos) in positions.iter().enumerate() {
            noise[pos] = correlated[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use cobre_core::{
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            CorrelationScheduleEntry,
        },
        EntityId,
    };

    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_entity(id: i32) -> CorrelationEntity {
        CorrelationEntity {
            entity_type: "inflow".to_string(),
            id: EntityId(id),
        }
    }

    fn identity_group(name: &str, entity_ids: &[i32]) -> CorrelationGroup {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        CorrelationGroup {
            name: name.to_string(),
            entities: entity_ids.iter().copied().map(make_entity).collect(),
            matrix,
        }
    }

    fn correlated_group(name: &str, entity_ids: &[i32], rho: f64) -> CorrelationGroup {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        CorrelationGroup {
            name: name.to_string(),
            entities: entity_ids.iter().copied().map(make_entity).collect(),
            matrix,
        }
    }

    fn single_profile_model(profile_name: &str, groups: Vec<CorrelationGroup>) -> CorrelationModel {
        let mut profiles = BTreeMap::new();
        profiles.insert(profile_name.to_string(), CorrelationProfile { groups });
        CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // Build tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_single_default_profile() {
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();
        assert_eq!(dc.default_profile, "default");
        assert!(dc.factors.contains_key("default"));
    }

    #[test]
    fn build_single_non_default_profile_used_as_default() {
        // Exactly one profile named "wet" — should become the implicit default.
        let model = single_profile_model("wet", vec![identity_group("g1", &[1])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();
        assert_eq!(dc.default_profile, "wet");
    }

    #[test]
    fn build_fails_with_no_profiles() {
        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles: BTreeMap::new(),
            schedule: vec![],
        };
        let result = DecomposedCorrelation::build(&model);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn build_fails_with_multiple_profiles_and_no_default() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "wet".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1])],
            },
        );
        profiles.insert(
            "dry".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1])],
            },
        );
        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![],
        };
        let result = DecomposedCorrelation::build(&model);
        assert!(
            matches!(result, Err(StochasticError::InvalidCorrelation { .. })),
            "Expected InvalidCorrelation, got: {result:?}"
        );
    }

    #[test]
    fn build_with_schedule_mapping() {
        // Acceptance criterion: profiles "default" and "wet", schedule maps stage 0 -> "wet".
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1, 2])],
            },
        );
        profiles.insert(
            "wet".to_string(),
            CorrelationProfile {
                groups: vec![correlated_group("g1", &[1, 2], 0.8)],
            },
        );
        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![CorrelationScheduleEntry {
                stage_id: 0,
                profile_name: "wet".to_string(),
            }],
        };
        let dc = DecomposedCorrelation::build(&model).unwrap();

        // Stage 0 should use "wet".
        assert_eq!(dc.profile_for_stage(0), "wet");
        // Stage 1 (not in schedule) should use "default".
        assert_eq!(dc.profile_for_stage(1), "default");
    }

    // -----------------------------------------------------------------------
    // apply_correlation tests
    // -----------------------------------------------------------------------

    #[test]
    fn apply_correlation_with_identity_factor_leaves_noise_unchanged() {
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2)];
        let mut noise = [3.0_f64, 5.0];
        dc.apply_correlation(0, &mut noise, &entity_order);

        assert!((noise[0] - 3.0).abs() < 1e-12, "noise[0]={}", noise[0]);
        assert!((noise[1] - 5.0).abs() < 1e-12, "noise[1]={}", noise[1]);
    }

    #[test]
    fn apply_correlation_with_known_factor() {
        // Use [[1, 0.8],[0.8, 1]] => L[1][0]=0.8, L[1][1]=0.6.
        // z=[1.0, 0.0] => eta=[1.0, 0.8].
        let group = CorrelationGroup {
            name: "g1".to_string(),
            entities: vec![make_entity(1), make_entity(2)],
            matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
        };
        let model = single_profile_model("default", vec![group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2)];
        let mut noise = [1.0_f64, 0.0];
        dc.apply_correlation(0, &mut noise, &entity_order);

        assert!((noise[0] - 1.0).abs() < 1e-12, "noise[0]={}", noise[0]);
        assert!((noise[1] - 0.8).abs() < 1e-12, "noise[1]={}", noise[1]);
    }

    #[test]
    fn apply_correlation_leaves_unmatched_entities_unchanged() {
        // entity_order has entities 1, 2, 3; group only covers 1 and 2.
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2])]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2), EntityId(3)];
        let mut noise = [1.0_f64, 2.0, 99.0];
        dc.apply_correlation(0, &mut noise, &entity_order);

        // Entity 3 (index 2) must be untouched.
        assert!(
            (noise[2] - 99.0).abs() < 1e-12,
            "unmatched entity changed: {}",
            noise[2]
        );
    }

    #[test]
    fn apply_correlation_with_reordered_entities() {
        // The canonical order is [2, 1] but the group defines entities as [1, 2].
        // Independent samples: noise[0] (entity 2) = 0.5, noise[1] (entity 1) = 1.0.
        // Group matrix [[1,0.8],[0.8,1]] -- entities in group order [1, 2]:
        //   entity 1 is at position 1 in entity_order
        //   entity 2 is at position 0 in entity_order
        // gathered = [noise[pos(1)]=noise[1]=1.0, noise[pos(2)]=noise[0]=0.5]
        // correlated = L * [1.0, 0.5]:
        //   correlated[0] = 1.0 * 1.0 = 1.0
        //   correlated[1] = 0.8 * 1.0 + 0.6 * 0.5 = 1.1
        // scattered: noise[1] = 1.0, noise[0] = 1.1
        let group = CorrelationGroup {
            name: "g1".to_string(),
            entities: vec![make_entity(1), make_entity(2)],
            matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
        };
        let model = single_profile_model("default", vec![group]);
        let dc = DecomposedCorrelation::build(&model).unwrap();

        // Canonical order: entity 2 first, entity 1 second.
        let entity_order = [EntityId(2), EntityId(1)];
        let mut noise = [0.5_f64, 1.0]; // noise[0]=entity2, noise[1]=entity1
        dc.apply_correlation(0, &mut noise, &entity_order);

        // entity 1 at position 1: correlated[0]=1.0
        assert!((noise[1] - 1.0).abs() < 1e-12, "noise[1]={}", noise[1]);
        // entity 2 at position 0: correlated[1]=0.8*1.0+0.6*0.5=1.1
        assert!((noise[0] - 1.1).abs() < 1e-12, "noise[0]={}", noise[0]);
    }

    #[test]
    fn apply_correlation_uses_correct_profile_for_stage() {
        // Stage 0 -> "wet" (rho=0.8), stage 1 -> "default" (identity).
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![identity_group("g1", &[1, 2])],
            },
        );
        profiles.insert(
            "wet".to_string(),
            CorrelationProfile {
                groups: vec![correlated_group("g1", &[1, 2], 0.8)],
            },
        );
        let model = CorrelationModel {
            method: "cholesky".to_string(),
            profiles,
            schedule: vec![CorrelationScheduleEntry {
                stage_id: 0,
                profile_name: "wet".to_string(),
            }],
        };
        let dc = DecomposedCorrelation::build(&model).unwrap();

        let entity_order = [EntityId(1), EntityId(2)];

        // Stage 0 uses "wet": z=[1,0] -> [1.0, 0.8].
        let mut noise0 = [1.0_f64, 0.0];
        dc.apply_correlation(0, &mut noise0, &entity_order);
        assert!(
            (noise0[0] - 1.0).abs() < 1e-12,
            "stage0 noise0[0]={}",
            noise0[0]
        );
        assert!(
            (noise0[1] - 0.8).abs() < 1e-12,
            "stage0 noise0[1]={}",
            noise0[1]
        );

        // Stage 1 uses "default" (identity): z=[1,0] -> [1.0, 0.0].
        let mut noise1 = [1.0_f64, 0.0];
        dc.apply_correlation(1, &mut noise1, &entity_order);
        assert!(
            (noise1[0] - 1.0).abs() < 1e-12,
            "stage1 noise1[0]={}",
            noise1[0]
        );
        assert!(
            (noise1[1] - 0.0).abs() < 1e-12,
            "stage1 noise1[1]={}",
            noise1[1]
        );
    }
}
