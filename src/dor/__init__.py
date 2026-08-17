"""DOR — Diagnostic Orchestration & Reasoning runtime."""
__version__ = "1.0.3+timer.1"
from .arbiter import arbitrate_anomaly_csv, arbitrate_anomaly_portfolio, arbitrate_class_portfolio
from .control import build_plan, formalize_manifest, qualify_task
from .reference import build_reference_applicability, normalize_context_levels, select_hierarchical_fec_output
from .executor import execute_chain
from .ontology import validate_fault_ontology, validate_mechanism_component_map
from .localization import localize_faults, localize_from_files, localize_batch_from_files
from .rootcause import infer_root_causes, infer_root_causes_from_files
from .auto_design import predict_pipeline, run_auto_design
from .modality_calibration import ModalityMaskCalibrationPlan, assess_modality_mask_calibration
from .m2n_forecast_bridge import bind_m2n_forecast_contract
from .isgn_active_discovery_bridge import build_active_discovery_advisory
from .identification import (
    build_fault_identification_request,
    build_identification_request_from_files,
    identify_fault_changes,
    identify_from_files,
    validate_isgn_identification_evidence,
)
from .isgn_identification_bridge import (
    build_isgn_identification_evidence,
    build_isgn_identification_evidence_from_files,
)
from .multitask import arbitrate_multitask, arbitrate_multitask_from_files
from .vertical import evaluate_vertical, evaluate_vertical_from_file, validate_vertical_registry
from .freeze import evaluate_freeze_gate, evaluate_freeze_gate_from_files

__all__ = [
    "arbitrate_anomaly_csv",
    "arbitrate_anomaly_portfolio",
    "arbitrate_class_portfolio",
    "build_plan",
    "formalize_manifest",
    "qualify_task",
    "execute_chain",
    "build_reference_applicability",
    "normalize_context_levels",
    "select_hierarchical_fec_output",
    "validate_fault_ontology",
    "validate_mechanism_component_map",
    "localize_faults",
    "localize_from_files",
    "localize_batch_from_files",
    "infer_root_causes",
    "infer_root_causes_from_files",
    "run_auto_design",
    "predict_pipeline",
    "ModalityMaskCalibrationPlan",
    "assess_modality_mask_calibration",
    "bind_m2n_forecast_contract",
    "build_active_discovery_advisory",
    "build_fault_identification_request",
    "build_identification_request_from_files",
    "identify_fault_changes",
    "identify_from_files",
    "validate_isgn_identification_evidence",
    "build_isgn_identification_evidence",
    "build_isgn_identification_evidence_from_files",
    "arbitrate_multitask",
    "arbitrate_multitask_from_files",
    "evaluate_vertical",
    "evaluate_vertical_from_file",
    "validate_vertical_registry",
    "evaluate_freeze_gate",
    "evaluate_freeze_gate_from_files",
]
