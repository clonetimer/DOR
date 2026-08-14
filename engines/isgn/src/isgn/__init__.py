"""ISGN Toolkit.

Core scope: recover and complete missing industrial mechanisms from partial physics and data,
then certify or refuse the recovered Operational Mechanism Model. Atlas, architecture, PHM,
and closed-loop modules are optional downstream/auxiliary capabilities.
"""
__version__="2.3.0+timer.1"
from .api import (RecoveryResult, RecoverySummary, recover_graph, complete_graph, certify_completion, calibrate_completion_coverage, fit_normal_phm, extract_mechanism_features, save_checkpoint, load_checkpoint)
from .phm import PHMHealthCalibration, anchored_parameter_drift, mechanism_health_evidence, calibrate_phm_health, score_phm_health, slice_time_window
from .io import from_arrays, save_npz, load_npz, save_algebraic_npz, load_algebraic_npz
from .contracts import PublicDataset, AlgebraicDataset, ObservationContract, ParameterPolicy
from .backends import backend_status, backend_manifest, recommend_backend
from .fusion import FusionOutcome, build_operational_model
from .consensus import ConsensusOutcome, ConsensusReport, build_consensus_model, graph_jaccard
from .fallbacks import FallbackResult, FallbackSummary, fit_linear_state_space_fallback, save_fallback, load_fallback
__all__=["RecoveryResult","RecoverySummary","recover_graph","complete_graph","certify_completion","calibrate_completion_coverage","fit_normal_phm","extract_mechanism_features","save_checkpoint","load_checkpoint","from_arrays","save_npz","load_npz","save_algebraic_npz","load_algebraic_npz","PublicDataset","backend_status","backend_manifest","recommend_backend","FusionOutcome","build_operational_model","ConsensusOutcome","ConsensusReport","build_consensus_model","graph_jaccard","FallbackResult","FallbackSummary","fit_linear_state_space_fallback","save_fallback","load_fallback","ObservationGateConfig","ObservationGateReport","assess_observation","PHMHealthCalibration","anchored_parameter_drift","mechanism_health_evidence","calibrate_phm_health","score_phm_health","slice_time_window"]

from .observation_gate import ObservationGateConfig, ObservationGateReport, assess_observation


from .mechanism_scope import project_mechanism_scope
__all__.append("project_mechanism_scope")

from .completion import CoverageCalibration
__all__.append("CoverageCalibration")

from .friction import SmoothStribeckProfile, smooth_stribeck_transform, profile_smooth_stribeck
__all__ += ["SmoothStribeckProfile","smooth_stribeck_transform","profile_smooth_stribeck"]

from .adapters import AdapterDescriptor, available_adapters, adapter_manifest, get_adapter
from .atlas import MechanismAtlas, MechanismScopeRecord, ScopeInterface, UncoveredRegion, save_atlas, load_atlas
__all__ += ["ObservationContract","ParameterPolicy","AdapterDescriptor","available_adapters","adapter_manifest","get_adapter","MechanismAtlas","MechanismScopeRecord","ScopeInterface","UncoveredRegion","save_atlas","load_atlas"]

from .algebraic import AlgebraicRelationResult, discover_algebraic_relation, discover_algebraic_relation_subset
__all__ += ["AlgebraicDataset","AlgebraicRelationResult","discover_algebraic_relation","discover_algebraic_relation_subset"]

from .evidence_router import HealthEvidenceEvent, EvidenceRoute, route_health_evidence, route_health_evidence_batch
__all__ += ["HealthEvidenceEvent","EvidenceRoute","route_health_evidence","route_health_evidence_batch"]

from .atlas_synthesis import VariableEvidence, RelationEvidence, ScopeProposal, AtlasSynthesisResult, synthesize_mechanism_atlas
from .architecture import TargetArchitectureSpec, ScopeArchitectureSpec, ArchitectureBlueprint, CompiledAtlasNetwork, compile_mechanism_atlas, build_compiled_network
__all__ += ["VariableEvidence","RelationEvidence","ScopeProposal","AtlasSynthesisResult","synthesize_mechanism_atlas","TargetArchitectureSpec","ScopeArchitectureSpec","ArchitectureBlueprint","CompiledAtlasNetwork","compile_mechanism_atlas","build_compiled_network"]

from .closed_loop import InformationGap, ClosedLoopAction, ClosedLoopPlan, ActionEvaluation, ClosedLoopEvaluation, ClosedLoopCycle, plan_closed_loop_atlas, evaluate_closed_loop_plan, run_closed_loop_cycle
__all__ += ["InformationGap","ClosedLoopAction","ClosedLoopPlan","ActionEvaluation","ClosedLoopEvaluation","ClosedLoopCycle","plan_closed_loop_atlas","evaluate_closed_loop_plan","run_closed_loop_cycle"]

from .mechanism_recovery import MissingTermRecoveryResult, evaluate_term, target_derivative, recover_missing_terms
from .physics_reducer import DimensionalCandidate, PairwiseReduction, PhysicsReductionReport, dimensionally_consistent_monomials, probe_pairwise_reductions, reduce_algebraic_search_space
from .trajectory_recovery import TrajectoryMissingDynamicsResult, recover_missing_dynamics_trajectory
__all__ += ["MissingTermRecoveryResult","evaluate_term","target_derivative","recover_missing_terms","DimensionalCandidate","PairwiseReduction","PhysicsReductionReport","dimensionally_consistent_monomials","probe_pairwise_reductions","reduce_algebraic_search_space","TrajectoryMissingDynamicsResult","recover_missing_dynamics_trajectory"]

from .hybrid import (RegimePartition, RegimeRelationEvidence, HybridAlgebraicRecoveryResult, threshold_regime_partition, project_algebraic_regime, project_public_regime_windows, discover_regime_algebraic_relations)
__all__ += ["RegimePartition","RegimeRelationEvidence","HybridAlgebraicRecoveryResult","threshold_regime_partition","project_algebraic_regime","project_public_regime_windows","discover_regime_algebraic_relations"]

from .discovery_policy import DiscoveryRouteDecision, recommend_discovery_route
__all__ += ["DiscoveryRouteDecision", "recommend_discovery_route"]
from .tabulated import Tabulated1DResult, fit_tabulated_1d
__all__ += ["Tabulated1DResult", "fit_tabulated_1d"]
