from .catalog import CATALOG, ExtensionSpec, assess_extension_readiness
from .conditional_calibration import MondrianRootRegistry, CalibrationGroupUnavailable
from .calibration_audit import audit_group_pvalues, CalibrationAuditSummary, GroupCalibrationAudit
from .safe_admission import FrozenReserveAdmissionGate, FrozenReserveRegistry, AdmissionDecision
from .credibility import CredibilityOverlay, combine_contract_and_credibility
from .contact_bank_audit import audit_contact_bank
from .weighted_conformal import weighted_upper_p, LogisticDensityRatio
from .opinion_fusion import (
    DirichletOpinion,
    OpinionFusionResult,
    pairwise_degree_of_conflict,
    discounted_belief_fusion,
    dempster_shafer_pair,
    opinion_from_scalar_anomaly_evidence,
)

__all__ = [
    'CATALOG','ExtensionSpec','assess_extension_readiness',
    'MondrianRootRegistry','CalibrationGroupUnavailable',
    'audit_group_pvalues','CalibrationAuditSummary','GroupCalibrationAudit',
    'FrozenReserveAdmissionGate','FrozenReserveRegistry','AdmissionDecision',
    'CredibilityOverlay','combine_contract_and_credibility','audit_contact_bank',
    'weighted_upper_p','LogisticDensityRatio',
    'DirichletOpinion','OpinionFusionResult','pairwise_degree_of_conflict',
    'discounted_belief_fusion','dempster_shafer_pair','opinion_from_scalar_anomaly_evidence',
    'CalibrationHealth','audit_p_values','calibration_strategy_decision',
    'ClassEvidenceBatch','validate_shared_frame',
]

from .drift_guard import CalibrationHealth,audit_p_values,calibration_strategy_decision
from .class_evidence import ClassEvidenceBatch,validate_shared_frame

from .conformal_backend import (
    ConformalResult, EmpiricalRootBackend, MondrianRootBackend,
    LocalizedRootBackend, WeightedRootBackend,
)
from .sparse_group_policy import SparseGroupPolicy, GroupRouteDecision
from .reference_cleaning import ActiveReferenceCleaner, CleaningPlan, CleaningResult, naive_unlabelled_trim
from .edl_provider import EvidentialMLPProvider, EDLTrainingReport

__all__ += [
    'ConformalResult','EmpiricalRootBackend','MondrianRootBackend','LocalizedRootBackend','WeightedRootBackend',
    'SparseGroupPolicy','GroupRouteDecision',
    'ActiveReferenceCleaner','CleaningPlan','CleaningResult','naive_unlabelled_trim',
    'EvidentialMLPProvider','EDLTrainingReport',
]

from .reference_bundle import (
    LeafEvidenceProvider, CoreEmpiricalLeafProvider,
    ReferenceBundleMeta, ManagedReferenceBundle, ReferenceBundleManager,
)

__all__ += [
    'LeafEvidenceProvider','CoreEmpiricalLeafProvider',
    'ReferenceBundleMeta','ManagedReferenceBundle','ReferenceBundleManager',
]

# v2.0 durable commissioned Reference Bundle state
from .reference_bundle_store import (
    ReferenceBundleStoreError,
    ReferenceBundleIntegrityError,
    export_reference_bundle_manager,
    save_reference_bundle_manager,
    load_reference_bundle_manager,
)


# v2.1 provenance / missing-modality update
from .evidence_provenance import (
    SourceKind, SourceDomain, ModalityStatus, AssetScopeKind, ConstraintType,
    ProviderIdentity, PhysicsKnowledgeProfile, EvidenceProvenance,
    ModalityAvailability, OperatingContextIdentity, AssetReferenceScope, ReferenceCompatibilityContract,
    LineageOverlap, EvidenceLineageAudit, provider_fingerprint, audit_evidence_lineage,
    SourceAvailabilityState, EvidenceAvailabilityResolution, resolve_evidence_availability,
)
from .mask_conditional_class import (
    MaskCalibrationUnavailable, MaskedFusionBatch, MaskCalibrationInfo,
    fuse_class_evidence_by_mask, MaskConditionalClassConformal,
)

__all__ += [
    'SourceKind','SourceDomain','ModalityStatus','AssetScopeKind','ConstraintType',
    'ProviderIdentity','PhysicsKnowledgeProfile','EvidenceProvenance',
    'ModalityAvailability','OperatingContextIdentity','AssetReferenceScope','ReferenceCompatibilityContract',
    'LineageOverlap','EvidenceLineageAudit','provider_fingerprint','audit_evidence_lineage',
    'SourceAvailabilityState','EvidenceAvailabilityResolution','resolve_evidence_availability',
    'MaskCalibrationUnavailable','MaskedFusionBatch','MaskCalibrationInfo',
    'fuse_class_evidence_by_mask','MaskConditionalClassConformal',
]

from .fleet_reference import (
    FleetAssetAdmissionDecision, FleetReferenceRobustnessDecision,
    FleetReferenceAssetValidation, FleetReferenceCommissioningDecision,
    FleetReferenceCommissioningRobustnessDecision,
    audit_asset_for_fleet_reference, robust_fleet_admission_from_variants,
    validate_fleet_reference_candidate, robust_fleet_commissioning_from_variants,
    attest_fleet_reference_robustness,
    fleet_scope_from_admissions, fleet_scope_from_robust_admissions,
)

__all__ += [
    'FleetAssetAdmissionDecision','FleetReferenceRobustnessDecision',
    'FleetReferenceAssetValidation','FleetReferenceCommissioningDecision',
    'FleetReferenceCommissioningRobustnessDecision',
    'audit_asset_for_fleet_reference','robust_fleet_admission_from_variants',
    'validate_fleet_reference_candidate','robust_fleet_commissioning_from_variants',
    'attest_fleet_reference_robustness','fleet_scope_from_admissions','fleet_scope_from_robust_admissions',
]

# v2.3 class-score persistence/CLI bridge (Extension/Control Plane only)
from .class_score_bridge import (
    ClassConformalReference, fit_class_score_reference, infer_class_prediction_sets,
)

__all__ += [
    'ClassConformalReference','fit_class_score_reference','infer_class_prediction_sets',
]

# v2.4 context-conditioned external score bridges (Extension/Control Plane)
from .anomaly_score_bridge import AnomalyContextReference, fit_anomaly_score_reference, infer_anomaly_decisions
from .context_class_score_bridge import ContextClassConformalReference, fit_context_class_score_reference, infer_context_class_prediction_sets
from .mask_calibration_audit import MaskCoverageAudit, MaskCalibrationAuditReport, audit_mask_conditional_class_reference, write_mask_calibration_audit
__all__ += [
    'AnomalyContextReference','fit_anomaly_score_reference','infer_anomaly_decisions',
    'ContextClassConformalReference','fit_context_class_score_reference','infer_context_class_prediction_sets',
    'MaskCoverageAudit','MaskCalibrationAuditReport','audit_mask_conditional_class_reference','write_mask_calibration_audit',
]
