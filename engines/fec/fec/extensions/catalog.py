from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ExtensionSpec:
    name: str
    purpose: str
    required_inputs: tuple[str,...]
    safe_default: str
    core_change: bool
    forbidden_shortcut: str
    maturity: str = 'validated_extension'


CATALOG={
    'class_score_conformal_bridge':ExtensionSpec(
        'class_score_conformal_bridge','persist/apply mask-conditional conformal prediction sets for external selected-classifier probability score packs',
        ('genuine_class_probability_scores','independent_labeled_calibration','shared_class_frame'),
        'fit only on an independent calibration split; require score rows to sum to one; unseen/sparse modality masks abstain',False,
        'do not reuse model-selection validation as calibration, normalize arbitrary logits silently, or promote prediction sets to root-cause diagnosis','production_diagnostic_control_plane'),
    'context_reference_binding':ExtensionSpec(
        'context_reference_binding','bind a commissioned calibration Reference to an externally declared operating/task context identity',
        ('external_operating_context','reference_compatibility_contract'),
        'new context-aware adapters bind references to an exact declared context; missing or mismatched runtime context returns reference_unsupported',False,
        'do not reuse a Reference across operating/task domains merely because modality, provider, physical asset, and real/simulation provenance still match','release_candidate_control_plane'),
    'asset_reference_scope':ExtensionSpec(
        'asset_reference_scope','bind a commissioned Reference to one physical asset or an explicitly admitted fleet group',
        ('physical_asset_id','reference_compatibility_contract'),
        'new adapters default to asset-local; unseen assets return reference_unsupported',False,
        'do not infer cross-asset exchangeability from a shared sensor schema, provider version, or model family','release_candidate_control_plane'),
    'fleet_asset_admission_gate':ExtensionSpec(
        'fleet_asset_admission_gate','audit a candidate asset with an independent externally trusted normal sentinel before widening a Reference to a fleet group',
        ('active_reference_bundle','candidate_asset_id','trusted_normal_sentinel','matching_non_asset_reference_contract'),
        'admit only healthy asset-level calibration audits; anti-conservative, conservative/power-limited, sparse-support, or unverified assets remain blocked by default',False,
        'do not add an asset to a fleet scope because its schema matches or because fault-detection performance looks good','release_candidate_control_plane'),
    'fleet_reference_robustness_gate':ExtensionSpec(
        'fleet_reference_robustness_gate','block fleet scope widening when a candidate asset qualification is sensitive to finite Reference construction',
        ('predeclared_reference_variants','variant_level_reference_validation','candidate_asset_admission_per_valid_variant'),
        'require sufficient predeclared Reference-variant coverage and unanimous healthy admission by default; any anti-conservative valid variant blocks widening',False,
        'do not cherry-pick the one Reference split/bootstrap that admits the candidate and do not count reused folds as new independent data','release_candidate_control_plane'),
    'fleet_reference_commissioning_gate':ExtensionSpec(
        'fleet_reference_commissioning_gate','atomically validate a multi-asset fleet Reference against one independent trusted-normal sentinel per declared member asset',
        ('staged_fleet_reference_bundle','fleet_asset_membership','trusted_normal_sentinel_per_asset','runtime_contract_per_asset'),
        'fleet_group bundles reject generic candidate validation; promotion is allowed only when every declared fleet member passes its own compatibility/support/calibration-health audit through the asset-stratified fleet path; conservative is blocked by default',False,
        'do not use generic single-sentinel bundle validation for fleet scope, do not validate a mixed-asset sentinel as if it came from one runtime asset, and do not let a later passing asset overwrite an earlier rejection','release_candidate_control_plane'),
    'fleet_reference_promotion_guard':ExtensionSpec(
        'fleet_reference_promotion_guard','bind a predeclared multi-construction fleet robustness decision to the actual Reference promotion call',
        ('reference_construction_plan_id','planned_variant_count','variant_level_atomic_commissioning','aggregate_robustness_decision'),
        'an individually passing variant remains robustness_pending; only a matching accepted aggregate robustness attestation unlocks promotion',False,
        'do not call ordinary promote on one favorable Reference construction after predeclaring multiple constructions whose aggregate robustness gate is HOLD/BLOCK','release_candidate_control_plane'),
    'lineage_availability_resolution':ExtensionSpec(
        'lineage_availability_resolution','propagate physical modality loss through virtual/learned Evidence lineage before reference selection',
        ('evidence_provenance_graph','observed_modality_state','explicit_provider_selection'),
        'resolve the lineage DAG fail-closed; block derived sources whose required upstream source is missing; select failover providers explicitly',False,
        'do not trust stale/precomputed derived values after an upstream modality disappears and do not auto-select the provider that preserves normality','production_control_plane'),
    'evidence_provenance_lineage':ExtensionSpec(
        'evidence_provenance_lineage','make observed/virtual/simulated/learned Evidence origin and shared ancestry auditable',
        ('provider_identity','asset_acquisition_identity','source_kind','modality_identity'),
        'record provenance as sidecar metadata and audit shared ancestors before treating sources as independent',False,
        'do not convert provenance categories or shared lineage into extra anomaly Evidence; do not auto-assign semantic equivalence','production_control_plane'),
    'modality_reference_compatibility':ExtensionSpec(
        'modality_reference_compatibility','bind commissioned references to modality availability and provider fingerprints',
        ('modality_availability_contract','provider_fingerprint','versioned_reference_bundle'),
        'require exact declared modality/provider compatibility; otherwise return reference_unsupported',False,
        'do not borrow the reference that yields the largest normality score and do not treat derived/simulated modality as observed','production_control_plane'),
    'mask_conditional_class_conformal':ExtensionSpec(
        'mask_conditional_class_conformal','optional class-level Mondrian conformal prediction sets under missing modalities',
        ('nonnegative_class_evidence','shared_class_frame','modality_presence_mask','labeled_calibration_per_supported_mask'),
        'experimental only; absent modalities are vacuous and sparse/unseen masks abstain',False,
        'do not convert scalar anomaly E into Dirichlet evidence and do not silently pool incompatible masks','experimental_diagnostic_adapter'),
    'durable_reference_bundle_store':ExtensionSpec(
        'durable_reference_bundle_store','persist commissioned active/retired Reference Bundles across process restart',
        ('commissioned_reference_bundle','stable_empirical_leaf_provider','stable_empirical_root_backend','durable_state_path'),
        'persist only promoted active/retired bundles using the explicit JSON schema; verify SHA-256 before restore',False,
        'do not pickle arbitrary providers/backends and do not persist unvalidated shadow candidates','production_control_plane'),
    'reference_bundle_lifecycle':ExtensionSpec(
        'reference_bundle_lifecycle','atomically version leaf normal reference and root calibration reference',
        ('trusted_leaf_normal','trusted_root_normal','D_cal_claims','disjoint_shadow_sentinel','versioned_reference_key'),
        'stage leaf+root together from trusted raw Contact rows; validate end-to-end on a disjoint raw sentinel; block anti-conservative candidates and protect a healthy active bundle from silent power-limited replacement',False,
        'do not refresh only the root after leaf Evidence has drifted/saturated; do not auto-promote conservative refreshes over a healthy active bundle','production_control_plane'),
    'production_calibration_lifecycle':ExtensionSpec(
        'production_calibration_lifecycle','validity-first reference lifecycle under non-exchangeability',
        ('trusted_normal_sentinel','D_cal_claims','disjoint_shadow_sentinel','versioned_reference_key'),
        'unverified/anti-conservative/unsupported references cannot drive Policy; refreshes require shadow validation before promotion',False,
        'do not auto-promote a reference fitted and validated on the same data; do not adapt from arbitrary low-score online samples','production_control_plane'),
    'mondrian_calibration':ExtensionSpec(
        'mondrian_calibration','context/presence-conditioned root calibration',
        ('external_group_key','trusted_normal_per_group','matched_G_A_topology'),
        'enable only when every relevant group is adequately populated',False,
        'do not fall back to the group that maximizes normality'),
    'weighted_conformal':ExtensionSpec(
        'weighted_conformal','correct calibration under covariate shift',
        ('source_covariates','target_unlabeled_covariates','valid_density_ratio_model','stable_conditional_mechanism'),
        'disabled unless a defensible shift covariate exists',False,
        'do not use fault-responsive variables merely because they improve FPR','experimental_adapter'),
    'frozen_reserve_admission':ExtensionSpec(
        'frozen_reserve_admission','reduce normal-reference self-poisoning',
        ('immutable_trusted_reserve','D_cal_claims','external_reserve_key'),
        'deny on missing key or incomplete dependency provenance',False,
        'do not update the reserve with the candidates it judges'),
    'credibility_overlay':ExtensionSpec(
        'credibility_overlay','add learned/static source credibility without redefining Applicability',
        ('credibility_provider','provenance'),
        'keep Q separate from A; product weighting is opt-in experimental',False,
        'do not rename learned attention/credibility as contract Applicability'),
    'opinion_fusion':ExtensionSpec(
        'opinion_fusion','class/opinion-level DS or conflict-discounted fusion',
        ('nonnegative_class_evidence','shared_class_frame'),
        'disabled for scalar anomaly-E Contacts',False,
        'do not invent a scalar-E to Dirichlet-belief mapping'),
    'calibration_health_guard':ExtensionSpec(
        'calibration_health_guard','audit deployed p-values on externally trusted normal sentinel batches',
        ('trusted_normal_sentinel',),
        'always audit after deployment or reference changes; never infer sentinel health from the same anomaly score',False,
        'do not treat arbitrary low-score online samples as trusted normal sentinels'),
    'localized_conformal':ExtensionSpec(
        'localized_conformal','continuous-context localized conformal root p-values',
        ('external_continuous_context','trusted_normal_reference','stable_conditional_mechanism','effective_sample_size_audit'),
        'disabled when the external context is not defensible or local ESS is too small',False,
        'do not localize on fault-responsive features merely to improve FPR','validated_research_backend'),
    'sparse_group_policy':ExtensionSpec(
        'sparse_group_policy','safe handling of sparse or conservative conditional-calibration strata',
        ('group_reference_counts','trusted_group_health_audit'),
        'abstain on sparse groups unless an explicitly declared compatible parent exists; keep conservative groups power-limited',False,
        'do not merge groups simply to make empirical FPR closer to alpha','validated_extension'),
    'active_reference_cleaning':ExtensionSpec(
        'active_reference_cleaning','limited-budget cleaning of a contaminated normal reference',
        ('reference_scores','external_review_budget','review_labels'),
        'remove only externally confirmed outliers; preserve reviewed inliers',False,
        'do not replace labels with automatic top-score trimming','validated_extension'),
    'edl_class_evidence':ExtensionSpec(
        'edl_class_evidence','genuine non-negative class evidence provider for opinion fusion',
        ('labeled_fault_data','shared_class_frame','evidential_model'),
        'keep outside scalar anomaly-E Core and compare against ordinary probabilities',False,
        'do not convert scalar anomaly E into class evidence','experimental_diagnostic_adapter'),
    'contact_bank_audit':ExtensionSpec(
        'contact_bank_audit','flag branch-capacity imbalance and redundant Contacts',
        ('normal_contact_matrix','contact_semantics'),
        'review-only; no automatic equivalence assignment',False,
        'high correlation is not semantic equivalence'),
}


def assess_extension_readiness(profile: Mapping[str,bool]):
    out={}
    for name,spec in CATALOG.items():
        missing=[x for x in spec.required_inputs if not bool(profile.get(x,False))]
        out[name]={
            'ready':not missing,
            'missing':missing,
            'safe_default':spec.safe_default,
            'forbidden_shortcut':spec.forbidden_shortcut,
            'maturity':spec.maturity,
        }
    return out
