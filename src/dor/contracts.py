from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

TASK_STATES = {"QUALIFIED", "CONDITIONALLY_QUALIFIED", "UNSUPPORTED"}
DIAGNOSIS_STATES = {"SUPPORTED", "ALTERNATIVES", "UNKNOWN", "ABSTAIN"}
ENGINE_STATES = {"RUN", "SKIP", "BLOCKED"}
STEP_STATES = {"PLANNED", "RUNNING", "COMPLETED", "FAILED", "SKIPPED", "BLOCKED"}
SUPPORTED_RUNTIME_TASKS = {"ANOMALY_DETECTION", "FAULT_CLASSIFICATION"}


@dataclass(frozen=True)
class TaskQualification:
    task: str
    status: str
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    required_next_actions: tuple[str, ...] = ()
    checks: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in TASK_STATES:
            raise ValueError(f"invalid task qualification status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineAdmission:
    engine: str
    status: str
    reasons: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ENGINE_STATES:
            raise ValueError(f"invalid engine admission status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    engine: str
    action: str
    status: str = "PLANNED"
    depends_on: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    command_template: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STEP_STATES:
            raise ValueError(f"invalid plan step status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticPlan:
    plan_id: str
    task: str
    qualification: TaskQualification
    engine_admissions: tuple[EngineAdmission, ...]
    steps: tuple[PlanStep, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    contract: str = "DOR-DIAGNOSTIC-PLAN/0.4"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["qualification"] = self.qualification.to_dict()
        d["engine_admissions"] = [x.to_dict() for x in self.engine_admissions]
        d["steps"] = [x.to_dict() for x in self.steps]
        return d


@dataclass(frozen=True)
class DiagnosisBundleMeta:
    task: str
    status: str
    model_selection_status: str
    calibration_state: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in DIAGNOSIS_STATES:
            raise ValueError(f"invalid diagnosis status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
