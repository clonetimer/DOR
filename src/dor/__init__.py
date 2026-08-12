"""DOR — Diagnostic Orchestration & Reasoning runtime."""
__version__ = "0.7.0.dev0"
from .arbiter import arbitrate_anomaly_csv, arbitrate_anomaly_portfolio, arbitrate_class_portfolio
from .control import build_plan, formalize_manifest, qualify_task
from .reference import build_reference_applicability, normalize_context_levels, select_hierarchical_fec_output
from .executor import execute_chain
from .ontology import validate_fault_ontology, validate_mechanism_component_map
from .localization import localize_faults, localize_from_files, localize_batch_from_files
from .rootcause import infer_root_causes, infer_root_causes_from_files
from .auto_design import predict_pipeline, run_auto_design

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
]
