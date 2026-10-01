//! Spectral decomposition and runtime application of spatial correlation.
//!
//! Decomposes each correlation profile's groups into spectral factors and builds
//! a stage-to-profile schedule for runtime lookup, then transforms independent
//! standard-normal noise into spatially correlated noise.
//!
//! Non-positive-definite matrices do not error; negative eigenvalues are clipped
//! to 0.0 (nearest PSD approximation).

use std::collections::{BTreeMap, HashMap, HashSet};

use cobre_core::{CorrelationModel, EntityId};

use crate::{ClassDimensions, StochasticError, correlation::spectral::SpectralFactor};

/// Group dimension at or below which `apply_groups_for_class` buffers are
/// stack-allocated; larger groups use the caller's scratch buffer instead.
const MAX_STACK_DIM: usize = 64;

/// Correlation-group entity class, parsed from the wire strings `"inflow"`,
/// `"load"` and `"ncs"` by [`EntityClass::from_wire`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityClass {
    /// Hydro inflow entities (`"inflow"`).
    Inflow,
    /// Stochastic load-bus entities (`"load"`).
    Load,
    /// Non-controllable source entities (`"ncs"`).
    Ncs,
}

impl EntityClass {
    /// Parses the wire string into a class; `None` for anything other than
    /// exactly `"inflow"`, `"load"` or `"ncs"`.
    #[must_use]
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "inflow" => Some(Self::Inflow),
            "load" => Some(Self::Load),
            "ncs" => Some(Self::Ncs),
            _ => None,
        }
    }

    /// Returns the wire spelling that [`EntityClass::from_wire`] parses back
    /// into this variant: `"inflow"`, `"load"` or `"ncs"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inflow => "inflow",
            Self::Load => "load",
            Self::Ncs => "ncs",
        }
    }
}

/// A single correlation group's spectral factor with entity ID mapping.
#[derive(Debug)]
pub struct GroupFactor {
    /// Spectral factor for this group.
    pub factor: SpectralFactor,
    /// Entity IDs in the order matching the factor rows/columns.
    pub entity_ids: Vec<EntityId>,
    /// Entity class shared by all entities in this group.
    pub entity_type: EntityClass,
    /// Positions within this group's class segment, resolved once at construction.
    class_positions: Box<[usize]>,
}

/// Pre-decomposed correlation data for all profiles, with stage-to-profile mapping.
#[derive(Debug)]
pub struct DecomposedCorrelation {
    /// Spectral factors keyed by profile name; `BTreeMap` for deterministic
    /// iteration order (upholds declaration-order invariance).
    factors: BTreeMap<String, Vec<GroupFactor>>,

    /// Stage-to-profile mapping; stages absent here use the default profile.
    schedule: HashMap<i32, String>,

    /// Name of the default profile.
    default_profile: String,
}

impl DecomposedCorrelation {
    /// Constructs an empty `DecomposedCorrelation` for systems with no stochastic
    /// entities. [`groups_for_stage`] then returns an empty slice; [`profile_for_stage`]
    /// returns an empty string.
    ///
    /// [`groups_for_stage`]: Self::groups_for_stage
    /// [`profile_for_stage`]: Self::profile_for_stage
    #[must_use]
    pub fn empty() -> Self {
        Self {
            factors: BTreeMap::new(),
            schedule: HashMap::new(),
            default_profile: String::new(),
        }
    }

    /// Builds a `DecomposedCorrelation` from a [`CorrelationModel`].
    ///
    /// `entity_order` is the canonical full entity order; `dims` splits it into
    /// `[hydros | load buses | NCS]` segments used to resolve each group's
    /// per-class positions.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InvalidCorrelation`] if no `"default"` profile exists
    ///   and there is more than one profile (ambiguous default).
    /// - [`StochasticError::InvalidCorrelation`] if the model has no profiles.
    /// - [`StochasticError::InvalidCorrelation`] if a correlation matrix is not
    ///   square or not symmetric.
    ///
    /// Non-positive-definite matrices do not cause an error; negative eigenvalues
    /// are clipped to 0.0 to produce the nearest positive-semidefinite approximation.
    ///
    /// # Panics
    ///
    /// Panics if `entity_order` fails [`ClassDimensions::assert_partitions`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use cobre_core::{EntityId, scenario::{
    ///     CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
    /// }};
    /// use cobre_stochastic::{ClassDimensions, correlation::resolve::DecomposedCorrelation};
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
    /// let model = CorrelationModel { method: "spectral".to_string(), profiles, schedule: vec![] };
    /// let entity_order = [EntityId(1), EntityId(2)];
    /// let dims = ClassDimensions { n_hydros: 2, n_load_buses: 0, n_ncs: 0 };
    /// let dc = DecomposedCorrelation::build(&model, &entity_order, dims).unwrap();
    /// ```
    pub fn build(
        model: &CorrelationModel,
        entity_order: &[EntityId],
        dims: ClassDimensions,
    ) -> Result<Self, StochasticError> {
        if model.profiles.is_empty() {
            return Err(StochasticError::InvalidCorrelation {
                profile_name: String::new(),
                reason: "correlation model contains no profiles".into(),
            });
        }

        let default_profile = if model.profiles.contains_key("default") {
            "default".to_string()
        } else if model.profiles.len() == 1 {
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

        let mut factors: BTreeMap<String, Vec<GroupFactor>> = BTreeMap::new();

        // Accumulate clipping across all matrices to emit one summary, not one line per matrix.
        let mut clip_matrices_total = 0_usize;
        let mut clip_matrices_affected = 0_usize;
        let mut clip_eigenvalues_total = 0_usize;
        let mut clip_largest_magnitude = 0.0_f64;

        for (profile_name, profile) in &model.profiles {
            for group in &profile.groups {
                if group.entities.len() > 1 {
                    let first_type = &group.entities[0].entity_type;
                    if let Some(mixed) =
                        group.entities.iter().find(|e| e.entity_type != *first_type)
                    {
                        return Err(StochasticError::InvalidCorrelation {
                            profile_name: profile_name.clone(),
                            reason: format!(
                                "correlation group '{}' contains mixed entity types: \
                                 found '{}' and '{}'; \
                                 all entities in a group must share the same entity_type",
                                group.name, first_type, mixed.entity_type,
                            ),
                        });
                    }
                }
            }

            // Overlapping groups (an entity ID in more than one group) produce incorrect covariance.
            let mut seen: HashSet<EntityId> = HashSet::new();
            for group in &profile.groups {
                for entity in &group.entities {
                    if !seen.insert(entity.id) {
                        return Err(StochasticError::InvalidCorrelation {
                            profile_name: profile_name.clone(),
                            reason: format!(
                                "entity ID {} appears in more than one correlation group \
                                 within profile '{}'; groups must be disjoint",
                                entity.id.0, profile_name,
                            ),
                        });
                    }
                }
            }

            let mut group_factors: Vec<GroupFactor> = Vec::with_capacity(profile.groups.len());
            for group in &profile.groups {
                let (factor, clip) = SpectralFactor::decompose_with_diagnostics(&group.matrix)
                    .map_err(|e| match e {
                        StochasticError::InvalidCorrelation { reason, .. } => {
                            StochasticError::InvalidCorrelation {
                                profile_name: profile_name.clone(),
                                reason,
                            }
                        }
                        other => other,
                    })?;
                clip_matrices_total += 1;
                if clip.clipped_count > 0 {
                    clip_matrices_affected += 1;
                    clip_eigenvalues_total += clip.clipped_count;
                    clip_largest_magnitude = clip_largest_magnitude.max(clip.largest_magnitude);
                }

                let entity_ids: Vec<EntityId> = group.entities.iter().map(|e| e.id).collect();
                let entity_type_str = &group
                    .entities
                    .first()
                    .ok_or_else(|| StochasticError::InvalidCorrelation {
                        profile_name: profile_name.clone(),
                        reason: format!("correlation group '{}' has no entities", group.name),
                    })?
                    .entity_type;
                let entity_type = EntityClass::from_wire(entity_type_str).ok_or_else(|| {
                    StochasticError::InvalidCorrelation {
                        profile_name: profile_name.clone(),
                        reason: format!(
                            "correlation group '{}' has unknown entity_type '{}'; \
                             valid types are: inflow, load, ncs",
                            group.name, entity_type_str,
                        ),
                    }
                })?;

                group_factors.push(GroupFactor {
                    factor,
                    entity_ids,
                    entity_type,
                    class_positions: Box::default(),
                });
            }
            factors.insert(profile_name.clone(), group_factors);
        }

        // Indefinite input is routine here, not exceptional — log at debug, not
        // warn; `largest_negative_magnitude` separates round-off from a
        // meaningfully indefinite matrix.
        if clip_matrices_affected > 0 {
            tracing::debug!(
                matrices_affected = clip_matrices_affected,
                matrices_total = clip_matrices_total,
                clipped_eigenvalues = clip_eigenvalues_total,
                largest_negative_magnitude = clip_largest_magnitude,
                "spectral decomposition clipped negative eigenvalues to 0.0 \
                 (nearest PSD projection)"
            );
        }

        let schedule: HashMap<i32, String> = model
            .schedule
            .iter()
            .map(|entry| (entry.stage_id, entry.profile_name.clone()))
            .collect();

        dims.assert_partitions(entity_order);

        let inflow_order = &entity_order[dims.hydro_range()];
        let load_order = &entity_order[dims.load_bus_range()];
        let ncs_order = &entity_order[dims.ncs_range()];
        for group_factors in factors.values_mut() {
            Self::resolve_into(group_factors, inflow_order, EntityClass::Inflow);
            Self::resolve_into(group_factors, load_order, EntityClass::Load);
            Self::resolve_into(group_factors, ncs_order, EntityClass::Ncs);
        }

        Ok(Self {
            factors,
            schedule,
            default_profile,
        })
    }

    fn resolve_into(
        group_factors: &mut [GroupFactor],
        class_order: &[EntityId],
        class: EntityClass,
    ) {
        let id_to_pos: HashMap<EntityId, usize> = class_order
            .iter()
            .enumerate()
            .map(|(i, &eid)| (eid, i))
            .collect();

        for gf in group_factors.iter_mut() {
            if gf.entity_type != class {
                continue;
            }
            let positions: Vec<usize> = gf
                .entity_ids
                .iter()
                .filter_map(|eid| id_to_pos.get(eid).copied())
                .collect();
            gf.class_positions = positions.into_boxed_slice();
        }
    }

    /// Returns the profile name active for `stage_id`, falling back to the default profile.
    pub fn profile_for_stage(&self, stage_id: i32) -> &str {
        self.schedule
            .get(&stage_id)
            .map_or(self.default_profile.as_str(), String::as_str)
    }

    /// Groups active for `stage_id`'s correlation profile; empty when the
    /// schedule has no matching profile.
    #[must_use]
    pub fn groups_for_stage(&self, stage_id: i32) -> &[GroupFactor] {
        let profile_name = self.profile_for_stage(stage_id);
        self.factors
            .get(profile_name)
            .map_or(&[], |gf| gf.as_slice())
    }

    /// Applies every `groups` entry whose class matches `class`, gathering from
    /// and scattering into `class_noise` at each entry's `class_positions`.
    /// Groups above `MAX_STACK_DIM` entities use `scratch`; smaller ones stay on
    /// the stack. Allocates nothing at any width.
    ///
    /// # Panics
    ///
    /// Panics when `scratch` is smaller than twice the widest matching
    /// group's entity count; a debug build panics eagerly with a sized
    /// message, a release build panics from the unconditional slice-bounds
    /// check inside the split/index below.
    pub fn apply_groups_for_class(
        groups: &[GroupFactor],
        class: EntityClass,
        class_noise: &mut [f64],
        scratch: &mut [f64],
    ) {
        for gf in groups {
            if gf.entity_type != class {
                continue;
            }
            let positions = &gf.class_positions;
            let n = positions.len();
            if n == 0 || n != gf.factor.dim() {
                continue;
            }

            if n <= MAX_STACK_DIM {
                let mut gathered = [0.0_f64; MAX_STACK_DIM];
                let mut correlated = [0.0_f64; MAX_STACK_DIM];
                for (i, &pos) in positions.iter().enumerate() {
                    gathered[i] = class_noise[pos];
                }
                gf.factor.transform(&gathered[..n], &mut correlated[..n]);
                for (i, &pos) in positions.iter().enumerate() {
                    class_noise[pos] = correlated[i];
                }
            } else {
                debug_assert!(
                    scratch.len() >= 2 * n,
                    "correlation scratch too small: have {}, need {}",
                    scratch.len(),
                    2 * n,
                );
                let (gathered, correlated) = scratch.split_at_mut(n);
                for (i, &pos) in positions.iter().enumerate() {
                    gathered[i] = class_noise[pos];
                }
                gf.factor.transform(gathered, &mut correlated[..n]);
                for (i, &pos) in positions.iter().enumerate() {
                    class_noise[pos] = correlated[i];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use cobre_core::{
        EntityId,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            CorrelationScheduleEntry,
        },
    };

    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_entity_of_type(id: i32, entity_type: &str) -> CorrelationEntity {
        CorrelationEntity {
            entity_type: entity_type.to_string(),
            id: EntityId(id),
        }
    }

    fn identity_group(name: &str, entity_ids: &[i32]) -> CorrelationGroup {
        make_group_with_type(name, entity_ids, 0.0, "inflow")
    }

    fn correlated_group(name: &str, entity_ids: &[i32], rho: f64) -> CorrelationGroup {
        make_group_with_type(name, entity_ids, rho, "inflow")
    }

    fn make_group_with_type(
        name: &str,
        entity_ids: &[i32],
        rho: f64,
        entity_type: &str,
    ) -> CorrelationGroup {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        CorrelationGroup {
            name: name.to_string(),
            entities: entity_ids
                .iter()
                .copied()
                .map(|id| make_entity_of_type(id, entity_type))
                .collect(),
            matrix,
        }
    }

    fn single_profile_model(profile_name: &str, groups: Vec<CorrelationGroup>) -> CorrelationModel {
        let mut profiles = BTreeMap::new();
        profiles.insert(profile_name.to_string(), CorrelationProfile { groups });
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    /// Builds against the two-entity inflow-only order shared by most fixtures
    /// in this module.
    fn build_two_entity_inflow(
        model: &CorrelationModel,
    ) -> Result<DecomposedCorrelation, StochasticError> {
        DecomposedCorrelation::build(
            model,
            &[EntityId(1), EntityId(2)],
            ClassDimensions {
                n_hydros: 2,
                n_load_buses: 0,
                n_ncs: 0,
            },
        )
    }

    // -----------------------------------------------------------------------
    // Build tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_single_default_profile() {
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2])]);
        let dc = build_two_entity_inflow(&model).unwrap();
        assert_eq!(dc.default_profile, "default");
        assert!(dc.factors.contains_key("default"));
    }

    #[test]
    fn build_single_non_default_profile_used_as_default() {
        let model = single_profile_model("wet", vec![identity_group("g1", &[1])]);
        let dc = DecomposedCorrelation::build(
            &model,
            &[EntityId(1)],
            ClassDimensions {
                n_hydros: 1,
                n_load_buses: 0,
                n_ncs: 0,
            },
        )
        .unwrap();
        assert_eq!(dc.default_profile, "wet");
    }

    #[test]
    fn build_fails_with_no_profiles() {
        let model = CorrelationModel {
            method: "spectral".to_string(),
            profiles: BTreeMap::new(),
            schedule: vec![],
        };
        let result = build_two_entity_inflow(&model);
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
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let result = build_two_entity_inflow(&model);
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
            method: "spectral".to_string(),
            profiles,
            schedule: vec![CorrelationScheduleEntry {
                stage_id: 0,
                profile_name: "wet".to_string(),
            }],
        };
        let dc = build_two_entity_inflow(&model).unwrap();

        // Stage 0 should use "wet".
        assert_eq!(dc.profile_for_stage(0), "wet");
        // Stage 1 (not in schedule) should use "default".
        assert_eq!(dc.profile_for_stage(1), "default");
    }

    #[test]
    fn apply_groups_for_class_differs_between_scheduled_and_default_profile() {
        // Same profiles/schedule as `build_with_schedule_mapping`: stage 0 -> "wet"
        // (rho=0.8), stage 1 (unscheduled) -> "default" (identity).
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
            method: "spectral".to_string(),
            profiles,
            schedule: vec![CorrelationScheduleEntry {
                stage_id: 0,
                profile_name: "wet".to_string(),
            }],
        };
        let dc = build_two_entity_inflow(&model).unwrap();

        let mut scratch = vec![0.0_f64; 4];
        let mut stage0_noise = [1.0_f64, 0.0];
        DecomposedCorrelation::apply_groups_for_class(
            dc.groups_for_stage(0),
            EntityClass::Inflow,
            &mut stage0_noise,
            &mut scratch,
        );
        let mut stage1_noise = [1.0_f64, 0.0];
        DecomposedCorrelation::apply_groups_for_class(
            dc.groups_for_stage(1),
            EntityClass::Inflow,
            &mut stage1_noise,
            &mut scratch,
        );

        // Stage 0 ("wet", rho=0.8): z=[1,0] -> spectral D*[1,0] = [D[0][0], D[1][0]].
        let d00 = f64::midpoint(f64::sqrt(1.8), f64::sqrt(0.2));
        let d10 = (f64::sqrt(1.8) - f64::sqrt(0.2)) / 2.0;
        assert!(
            (stage0_noise[0] - d00).abs() < 1e-8,
            "stage0_noise[0]={} (expected {d00})",
            stage0_noise[0]
        );
        assert!(
            (stage0_noise[1] - d10).abs() < 1e-8,
            "stage0_noise[1]={} (expected {d10})",
            stage0_noise[1]
        );

        // Stage 1 ("default", identity): z=[1,0] -> [1.0, 0.0], unchanged.
        assert_eq!(stage1_noise, [1.0, 0.0]);

        assert_ne!(
            stage0_noise, stage1_noise,
            "a stage-scheduled non-default profile must apply a different transform \
             than the default profile"
        );
    }

    // -----------------------------------------------------------------------
    // Same-type entity validation tests
    // -----------------------------------------------------------------------

    fn mixed_type_group(name: &str) -> CorrelationGroup {
        CorrelationGroup {
            name: name.to_string(),
            entities: vec![
                CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                },
                CorrelationEntity {
                    entity_type: "load".to_string(),
                    id: EntityId(2),
                },
            ],
            matrix: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
        }
    }

    #[test]
    fn test_build_rejects_mixed_entity_types() {
        let model = single_profile_model("default", vec![mixed_type_group("mixed_group")]);
        let result = build_two_entity_inflow(&model);
        match result {
            Err(StochasticError::InvalidCorrelation { reason, .. }) => {
                assert!(
                    reason.contains("mixed entity types"),
                    "expected 'mixed entity types' in reason, got: {reason}"
                );
            }
            other => panic!("expected InvalidCorrelation, got: {other:?}"),
        }
    }

    #[test]
    fn test_build_accepts_same_type_entities() {
        let model = single_profile_model("default", vec![identity_group("g1", &[1, 2, 3])]);
        assert!(
            DecomposedCorrelation::build(
                &model,
                &[EntityId(1), EntityId(2), EntityId(3)],
                ClassDimensions {
                    n_hydros: 3,
                    n_load_buses: 0,
                    n_ncs: 0,
                },
            )
            .is_ok(),
            "same-type group should be accepted"
        );
    }

    #[test]
    fn test_build_accepts_single_entity_group() {
        // A single-entity group is trivially homogeneous.
        let model = single_profile_model("default", vec![identity_group("g1", &[1])]);
        assert!(
            DecomposedCorrelation::build(
                &model,
                &[EntityId(1)],
                ClassDimensions {
                    n_hydros: 1,
                    n_load_buses: 0,
                    n_ncs: 0,
                },
            )
            .is_ok(),
            "single-entity group should be accepted"
        );
    }

    #[test]
    fn test_build_mixed_type_error_includes_group_name() {
        let model = single_profile_model("default", vec![mixed_type_group("mixed_group")]);
        let result = build_two_entity_inflow(&model);
        match result {
            Err(StochasticError::InvalidCorrelation {
                profile_name,
                reason,
            }) => {
                assert_eq!(
                    profile_name, "default",
                    "expected profile_name 'default', got: {profile_name}"
                );
                assert!(
                    reason.contains("mixed_group"),
                    "expected group name 'mixed_group' in reason, got: {reason}"
                );
            }
            other => panic!("expected InvalidCorrelation, got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Entity class parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_rejects_unknown_entity_type() {
        let group = make_group_with_type("bad_group", &[1], 0.0, "hydro_unit_group");
        let model = single_profile_model("default", vec![group]);
        let result = build_two_entity_inflow(&model);
        match result {
            Err(StochasticError::InvalidCorrelation { reason, .. }) => {
                assert!(
                    reason.contains("bad_group")
                        && reason.contains("hydro_unit_group")
                        && reason.contains("inflow")
                        && reason.contains("load")
                        && reason.contains("ncs"),
                    "expected group name, offending string and the three valid values \
                     in reason, got: {reason}"
                );
            }
            other => panic!("expected InvalidCorrelation, got: {other:?}"),
        }
    }

    #[test]
    fn test_build_parses_entity_type_to_class() {
        let inflow_group = make_group_with_type("g_inflow", &[1], 0.0, "inflow");
        let load_group = make_group_with_type("g_load", &[2], 0.0, "load");
        let ncs_group = make_group_with_type("g_ncs", &[3], 0.0, "ncs");
        let model = single_profile_model("default", vec![inflow_group, load_group, ncs_group]);
        let dc = DecomposedCorrelation::build(
            &model,
            &[EntityId(1), EntityId(2), EntityId(3)],
            ClassDimensions {
                n_hydros: 1,
                n_load_buses: 1,
                n_ncs: 1,
            },
        )
        .unwrap();

        // BTreeMap preserves insertion order of the profile vec — the groups
        // appear in the order they were pushed during build().
        let group_factors = dc.factors.get("default").unwrap();
        assert_eq!(
            group_factors
                .iter()
                .map(|gf| gf.entity_type)
                .collect::<Vec<_>>(),
            vec![EntityClass::Inflow, EntityClass::Load, EntityClass::Ncs],
        );
    }

    #[test]
    fn test_entity_class_from_wire_rejects_unknown() {
        assert_eq!(EntityClass::from_wire(""), None);
        assert_eq!(EntityClass::from_wire("Inflow"), None);
        assert_eq!(EntityClass::from_wire("hydro"), None);
    }

    #[test]
    fn test_entity_class_as_str() {
        assert_eq!(EntityClass::Inflow.as_str(), "inflow");
        assert_eq!(EntityClass::Load.as_str(), "load");
        assert_eq!(EntityClass::Ncs.as_str(), "ncs");
    }

    #[test]
    fn test_entity_class_from_wire_as_str_round_trip() {
        for class in [EntityClass::Inflow, EntityClass::Load, EntityClass::Ncs] {
            assert_eq!(EntityClass::from_wire(class.as_str()), Some(class));
        }
    }

    // -----------------------------------------------------------------------
    // apply_groups_for_class tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_groups_for_class_inflow_only() {
        // Inflow group [EntityId(1), EntityId(2)] with rho=0.8 and a single-entity
        // load group [EntityId(3)].
        // Spectral factor of [[1,0.8],[0.8,1]]:
        //   D[0][0] = D[1][1] = (sqrt(1.8) + sqrt(0.2)) / 2 ~= 0.894427
        //   D[0][1] = D[1][0] = (sqrt(1.8) - sqrt(0.2)) / 2 ~= 0.447214
        // z=[1.0, 0.0] => result=[D[0][0], D[1][0]] ~= [0.894427, 0.447214].
        let inflow_group = make_group_with_type("inflow_g", &[1, 2], 0.8, "inflow");
        let load_group = make_group_with_type("load_g", &[3], 0.0, "load");
        let model = single_profile_model("default", vec![inflow_group, load_group]);
        let dc = DecomposedCorrelation::build(
            &model,
            &[EntityId(1), EntityId(2), EntityId(3)],
            ClassDimensions {
                n_hydros: 2,
                n_load_buses: 1,
                n_ncs: 0,
            },
        )
        .unwrap();

        let mut inflow_noise = [1.0_f64, 0.0];
        let groups = dc.groups_for_stage(0);
        let mut scratch = vec![0.0_f64; 2 * inflow_noise.len()];
        DecomposedCorrelation::apply_groups_for_class(
            groups,
            EntityClass::Inflow,
            &mut inflow_noise,
            &mut scratch,
        );

        let d00 = f64::midpoint(f64::sqrt(1.8), f64::sqrt(0.2));
        let d10 = (f64::sqrt(1.8) - f64::sqrt(0.2)) / 2.0;
        assert!(
            (inflow_noise[0] - d00).abs() < 1e-8,
            "inflow_noise[0]={} (expected {d00})",
            inflow_noise[0]
        );
        assert!(
            (inflow_noise[1] - d10).abs() < 1e-8,
            "inflow_noise[1]={} (expected {d10})",
            inflow_noise[1]
        );
    }

    #[test]
    fn test_apply_groups_for_class_skips_other_types() {
        // Same setup as above: inflow group [1,2] with rho=0.8, load group [3].
        // Calling with entity_type="load" and load_noise=[5.0] should leave noise
        // unchanged (the single-entity identity group is a no-op AND this also
        // verifies that the inflow group is skipped entirely).
        let inflow_group = make_group_with_type("inflow_g", &[1, 2], 0.8, "inflow");
        let load_group = make_group_with_type("load_g", &[3], 0.0, "load");
        let model = single_profile_model("default", vec![inflow_group, load_group]);
        let dc = DecomposedCorrelation::build(
            &model,
            &[EntityId(1), EntityId(2), EntityId(3)],
            ClassDimensions {
                n_hydros: 2,
                n_load_buses: 1,
                n_ncs: 0,
            },
        )
        .unwrap();

        let mut load_noise = [5.0_f64];
        let groups = dc.groups_for_stage(0);
        let mut scratch = vec![0.0_f64; 2 * load_noise.len()];
        DecomposedCorrelation::apply_groups_for_class(
            groups,
            EntityClass::Load,
            &mut load_noise,
            &mut scratch,
        );

        // Identity group on a single entity leaves noise unchanged.
        assert!(
            (load_noise[0] - 5.0).abs() < 1e-12,
            "load_noise[0]={} (expected 5.0)",
            load_noise[0]
        );
    }

    #[test]
    fn test_apply_groups_for_class_no_matching_groups() {
        // Only inflow groups exist; calling with entity_type="ncs" must be a no-op.
        let inflow_group = make_group_with_type("inflow_g", &[1, 2], 0.8, "inflow");

        let model = single_profile_model("default", vec![inflow_group]);
        let dc = build_two_entity_inflow(&model).unwrap();

        let mut noise: [f64; 0] = [];
        let groups = dc.groups_for_stage(0);
        let mut scratch: Vec<f64> = vec![];
        DecomposedCorrelation::apply_groups_for_class(
            groups,
            EntityClass::Ncs,
            &mut noise,
            &mut scratch,
        );
        // No panic, no modification — test passes if we reach here.
    }

    #[test]
    fn test_group_factor_stores_entity_type() {
        let inflow_group = make_group_with_type("g_inflow", &[1, 2], 0.0, "inflow");
        let load_group = make_group_with_type("g_load", &[3], 0.0, "load");
        let ncs_group = make_group_with_type("g_ncs", &[4], 0.0, "ncs");
        let model = single_profile_model("default", vec![inflow_group, load_group, ncs_group]);
        let dc = DecomposedCorrelation::build(
            &model,
            &[EntityId(1), EntityId(2), EntityId(3), EntityId(4)],
            ClassDimensions {
                n_hydros: 2,
                n_load_buses: 1,
                n_ncs: 1,
            },
        )
        .unwrap();

        let group_factors = dc.factors.get("default").unwrap();
        assert_eq!(group_factors.len(), 3);

        // BTreeMap preserves insertion order of the profile vec — the groups
        // appear in the order they were pushed during build().
        let types: Vec<EntityClass> = group_factors.iter().map(|gf| gf.entity_type).collect();
        assert!(
            types.contains(&EntityClass::Inflow),
            "expected Inflow group factor, got: {types:?}"
        );
        assert!(
            types.contains(&EntityClass::Load),
            "expected Load group factor, got: {types:?}"
        );
        assert!(
            types.contains(&EntityClass::Ncs),
            "expected Ncs group factor, got: {types:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Per-class position resolution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_class_positions_match_linear_scan_three_class_model() {
        let inflow_group = make_group_with_type("inflow_g", &[10, 30, 50], 0.5, "inflow");
        let load_group = make_group_with_type("load_g", &[200, 100], 0.3, "load");
        let ncs_group = make_group_with_type("ncs_g", &[2000], 0.0, "ncs");
        let model = single_profile_model("default", vec![inflow_group, load_group, ncs_group]);

        let entity_order = [
            EntityId(10),
            EntityId(20),
            EntityId(30),
            EntityId(40),
            EntityId(50),
            EntityId(100),
            EntityId(200),
            EntityId(300),
            EntityId(1000),
            EntityId(2000),
        ];
        let dims = ClassDimensions {
            n_hydros: 5,
            n_load_buses: 3,
            n_ncs: 2,
        };
        let dc = DecomposedCorrelation::build(&model, &entity_order, dims).unwrap();

        let inflow_order = &entity_order[..5];
        let load_order = &entity_order[5..8];
        let ncs_order = &entity_order[8..];

        for gf in dc.factors.get("default").unwrap() {
            let class_order = match gf.entity_type {
                EntityClass::Inflow => inflow_order,
                EntityClass::Load => load_order,
                EntityClass::Ncs => ncs_order,
            };
            let expected: Vec<usize> = gf
                .entity_ids
                .iter()
                .filter_map(|eid| class_order.iter().position(|e| e == eid))
                .collect();
            assert_eq!(
                gf.class_positions.as_ref(),
                expected.as_slice(),
                "entity_type={:?}",
                gf.entity_type
            );
        }
    }

    #[test]
    fn test_group_absent_from_class_slice_resolves_empty_and_is_noop() {
        // Entity 99 does not appear in the two-entity inflow class slice.
        let orphan_group = make_group_with_type("orphan_g", &[99], 0.0, "inflow");
        let model = single_profile_model("default", vec![orphan_group]);
        let dc = build_two_entity_inflow(&model).unwrap();

        let group_factors = dc.factors.get("default").unwrap();
        assert!(
            group_factors[0].class_positions.is_empty(),
            "expected empty class_positions for an entity absent from the class slice"
        );

        let mut class_noise = [7.0_f64, 9.0];
        let groups = dc.groups_for_stage(0);
        let mut scratch = vec![0.0_f64; 2 * class_noise.len()];
        DecomposedCorrelation::apply_groups_for_class(
            groups,
            EntityClass::Inflow,
            &mut class_noise,
            &mut scratch,
        );
        assert_eq!(
            class_noise,
            [7.0, 9.0],
            "class_noise must be untouched when the group resolves empty"
        );
    }

    #[test]
    #[should_panic(expected = "must equal")]
    fn build_panics_when_class_dimensions_mismatch_entity_order_len() {
        let model = single_profile_model("default", vec![identity_group("g1", &[1])]);
        let _ = DecomposedCorrelation::build(
            &model,
            &[EntityId(1), EntityId(2)],
            ClassDimensions {
                n_hydros: 1,
                n_load_buses: 0,
                n_ncs: 0,
            },
        );
    }
}
