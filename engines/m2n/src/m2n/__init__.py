"""M2N Practitioner Toolkit.

Frozen practitioner-facing API derived from M2N research release v6.3.
The toolkit is intentionally conservative: it exposes mechanism knowledge
contracts, IR lowering/compilation, architecture selection, finite portfolios,
and model-evidence utilities without the historical experiment harnesses.
"""

__version__ = "2.16.1+timer.1"
__method_baseline__ = "M2N v2.16 forecast-horizon/representation qualification plus DOR finite diagnostic orchestration, safe symbolic compilation and executable UDE; adopted methods do not bypass qualification"

from .canonical_ir import Dim, CanonicalIR, canonical_signature, compile_canonical
from .knowledge_contract import (
    KnowledgeLevel,
    KnowledgeSource,
    MechanismSlot,
    SlotEvidence,
    SlotPlan,
    NeuralArchitecturePlan,
    plan_mechanism_slot,
    plan_neural_architecture,
)
from .architecture_lowering import (
    FieldAssignment,
    FamilyPatch,
    SlotBinding,
    ExecutableArchitectureCandidate,
    ExecutableArchitectureSpace,
    lower_architecture_plan,
    compile_executable_ir,
    compile_architecture_space,
)
from .selection import ArchitectureSelectionPolicy, ArchitectureSelectionPlan, select_architecture
from .design_session import DesignSessionResult, run_design_session
from .architecture_portfolio import PortfolioContract, ArchitecturePortfolio
from .model_evidence import RobustSupportReference, portfolio_derivative_evidence, summarize_portfolio_segment
from .mechanics_ir import (
    CoordinateSpec,
    InputSpec,
    DriveSpec,
    ConstantLoadSpec,
    ResistanceSpec,
    SecondOrderMechanicsIR,
    compile_second_order_ir,
)
from .peaked_response_ir import PeakAxisSpec, SeparablePeakIR, compile_separable_peak_ir
from .valve_ir import FourWayValveIR, compile_four_way_valve_ir
from .conservation_ir import ConservationIR, compile_conservation_ir
from .block_ir import BlockLNLIR, compile_block_lnl_ir
from .observation_ir import ObservationContract, ObservedStateModel
from .observation_planner import ObservationDesignPolicy, ObservationPlan, plan_observation
from .decision_contract import ArchitectureDecision

__all__ = [
    "Dim", "CanonicalIR", "canonical_signature", "compile_canonical",
    "KnowledgeLevel", "KnowledgeSource", "MechanismSlot", "SlotEvidence", "SlotPlan",
    "NeuralArchitecturePlan", "plan_mechanism_slot", "plan_neural_architecture",
    "FieldAssignment", "FamilyPatch", "SlotBinding", "ExecutableArchitectureCandidate",
    "ExecutableArchitectureSpace", "lower_architecture_plan", "compile_executable_ir",
    "compile_architecture_space", "ArchitectureSelectionPolicy", "ArchitectureSelectionPlan",
    "select_architecture", "DesignSessionResult", "run_design_session",
    "PortfolioContract", "ArchitecturePortfolio", "RobustSupportReference",
    "portfolio_derivative_evidence", "summarize_portfolio_segment",
    "CoordinateSpec", "InputSpec", "DriveSpec", "ConstantLoadSpec", "ResistanceSpec",
    "SecondOrderMechanicsIR", "compile_second_order_ir", "PeakAxisSpec", "SeparablePeakIR",
    "compile_separable_peak_ir", "FourWayValveIR", "compile_four_way_valve_ir",
    "ConservationIR", "compile_conservation_ir", "BlockLNLIR", "compile_block_lnl_ir",
    "ObservationContract", "ObservedStateModel", "ObservationDesignPolicy", "ObservationPlan",
    "plan_observation", "ArchitectureDecision",
]

from .backends import BackendCapability, BackendDescriptor, BackendRuntimeProbe, list_backends, get_backend
from .discovery import (
    BackendUnavailableError, DiscoveryGateDecision, DynamicsDiscoveryRequest,
    CandidateProposal, CandidateProposalBatch, ProposalAdmissionCheck, AdmittedCandidateProposal,
    admit_candidate_proposal, gate_candidate_discovery, PySINDyCandidateBackend,
)
from .novelty_kill import (
    NoveltyStatus, NoveltyClaim, BehavioralNKTResult,
    default_claim_ledger, nkt2_claim_ledger, run_behavioral_nkt,
)

__all__ += [
    "BackendCapability", "BackendDescriptor", "BackendRuntimeProbe", "list_backends", "get_backend",
    "BackendUnavailableError", "DiscoveryGateDecision", "DynamicsDiscoveryRequest",
    "CandidateProposal", "CandidateProposalBatch", "ProposalAdmissionCheck", "AdmittedCandidateProposal",
    "admit_candidate_proposal", "gate_candidate_discovery", "PySINDyCandidateBackend", "NoveltyStatus", "NoveltyClaim",
    "BehavioralNKTResult", "default_claim_ledger", "nkt2_claim_ledger", "run_behavioral_nkt",
]

from .symmatika_adapter import (
    SYMMATIKA_OPERATOR_ORDER, SymMatikaExplicitRequest, SymMatikaSourceProbe, SymMatikaPreparedJob,
    probe_symmatika_source, prepare_symmatika_job, parse_symmatika_driver_output, run_symmatika_from_source,
)
from .model_set_inference import (
    ModelComponentSlot, ParameterPrior, SBMIModelSetRequest, SBMIFeasibilityAssessment,
    assess_sbmi_feasibility, export_sbmi_contract,
)

__all__ += [
    "SYMMATIKA_OPERATOR_ORDER", "SymMatikaExplicitRequest", "SymMatikaSourceProbe", "SymMatikaPreparedJob",
    "probe_symmatika_source", "prepare_symmatika_job", "parse_symmatika_driver_output", "run_symmatika_from_source",
    "ModelComponentSlot", "ParameterPrior", "SBMIModelSetRequest", "SBMIFeasibilityAssessment",
    "assess_sbmi_feasibility", "export_sbmi_contract",
]

from .goal_design import (
    SUPPORTED_TEMPLATES, GoalDrivenDesignSummary, run_goal_driven_design, predict_model_artifact, generate_model_evidence,
)
__all__ += [
    "SUPPORTED_TEMPLATES", "GoalDrivenDesignSummary", "run_goal_driven_design", "predict_model_artifact", "generate_model_evidence",
]

from .generic_graph import build_generic_canonical_ir, build_generic_problem
from .auto_qualification import AutoQualificationResult, qualify_tabular_design
from .backend_router import BackendAction, run_backend_requests, assess_model_set_routing
from .qualified_k3 import K3QualificationPolicy, K3GateDecision, finite_gate, compare_k3
from .phm_handoff import EvidenceContract, export_fec_contact_pack, fit_support_reference, load_support_reference
from .fec_runtime import calibrate_fec_from_contacts, infer_fec_contacts

__all__ += [
    "build_generic_canonical_ir", "build_generic_problem", "AutoQualificationResult", "qualify_tabular_design",
    "BackendAction", "run_backend_requests", "assess_model_set_routing",
    "K3QualificationPolicy", "K3GateDecision", "finite_gate", "compare_k3",
    "EvidenceContract", "export_fec_contact_pack", "fit_support_reference", "load_support_reference",
    "calibrate_fec_from_contacts", "infer_fec_contacts",
]

from .practitioner import (
    PractitionerCheckResult, validate_device_card, inspect_tabular_data,
    suggested_next_actions, write_next_actions, seal_design_bundle, verify_design_bundle,
    predict_design_bundle,
)
__all__ += [
    "PractitionerCheckResult", "validate_device_card", "inspect_tabular_data",
    "suggested_next_actions", "write_next_actions", "seal_design_bundle", "verify_design_bundle",
    "predict_design_bundle",
]

from .bias_routing import BiasRoute, PhysicsKnowledgeItem, BiasRoutingDecision, BiasRoutePlan, route_physics_knowledge, build_bias_route_plan
from .representation_policy import RepresentationKind, RepresentationEligibility, RepresentationPlan, assess_representation, build_representation_plan
from .exact_physics import ExactPhysicsContract, audit_exact_physics
from .symbolic_compiler import (
    SymbolicParameter, SymbolicCompilationContract, SymbolicCompilationError,
    SymbolicDomainError, CompiledSymbolicModule, SymbolicResidualModel,
    compile_symbolic_expression,
)
from .ude import UDERuntimeContract, CompiledVectorField, HybridUDE, integrate_ude

from .diagnostic_modeling import (
    DIAGNOSTIC_TASKS, FEATURE_VIEWS, CLASSIFIER_FAMILIES, DETECTOR_FAMILIES,
    DiagnosticFamilyDescriptor, DiagnosticQualification, DiagnosticCandidate,
    DiagnosticSelectionPolicy, DiagnosticSelectionDecision, DiagnosticDesignSummary,
    list_diagnostic_families, qualify_diagnostic_request, build_diagnostic_candidates,
    run_diagnostic_model_design, predict_diagnostic_design,
)
from .model_family import (
    ModelingObjective, InputRepresentation, MechanismUsage, ModelFamily,
    ModelingContract, FamilyEligibility, ModelFamilyPlan,
    profile_modeling_data, qualify_model_families, load_and_plan_model_families,
)
from .temporal_design import (
    TemporalModelKind, TemporalCandidate, TemporalQualification, TemporalDesignSummary,
    qualify_temporal_design, run_temporal_design, predict_temporal_design,
)

__all__ += [
    "SymbolicParameter", "SymbolicCompilationContract", "SymbolicCompilationError",
    "SymbolicDomainError", "CompiledSymbolicModule", "SymbolicResidualModel",
    "compile_symbolic_expression", "UDERuntimeContract", "CompiledVectorField",
    "HybridUDE", "integrate_ude",
    "DIAGNOSTIC_TASKS", "FEATURE_VIEWS", "CLASSIFIER_FAMILIES", "DETECTOR_FAMILIES",
    "DiagnosticFamilyDescriptor", "DiagnosticQualification", "DiagnosticCandidate",
    "DiagnosticSelectionPolicy", "DiagnosticSelectionDecision", "DiagnosticDesignSummary",
    "list_diagnostic_families", "qualify_diagnostic_request", "build_diagnostic_candidates",
    "run_diagnostic_model_design", "predict_diagnostic_design",
    "ModelingObjective", "InputRepresentation", "MechanismUsage", "ModelFamily",
    "ModelingContract", "FamilyEligibility", "ModelFamilyPlan",
    "profile_modeling_data", "qualify_model_families", "load_and_plan_model_families",
    "TemporalModelKind", "TemporalCandidate", "TemporalQualification", "TemporalDesignSummary",
    "qualify_temporal_design", "run_temporal_design", "predict_temporal_design",
]
