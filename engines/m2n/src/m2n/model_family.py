from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np


class ModelingObjective(str, Enum):
    STATIC_REGRESSION = 'STATIC_REGRESSION'
    NORMALITY_MODELING = 'NORMALITY_MODELING'
    CATEGORICAL_PREDICTION = 'CATEGORICAL_PREDICTION'
    DYNAMICS_IDENTIFICATION = 'DYNAMICS_IDENTIFICATION'
    OPERATOR_LEARNING = 'OPERATOR_LEARNING'


class InputRepresentation(str, Enum):
    TABULAR = 'TABULAR'
    MULTIVARIATE_SEQUENCE = 'MULTIVARIATE_SEQUENCE'
    WAVEFORM_SEQUENCE = 'WAVEFORM_SEQUENCE'
    FUNCTION = 'FUNCTION'
    REGULAR_GRID_FIELD = 'REGULAR_GRID_FIELD'


class MechanismUsage(str, Enum):
    REQUIRED = 'REQUIRED'
    PREFERRED = 'PREFERRED'
    OPTIONAL = 'OPTIONAL'
    NONE = 'NONE'


class ModelFamily(str, Enum):
    PHYSICS_PARAMETRIC = 'PHYSICS_PARAMETRIC'
    SYSTEM_IDENTIFICATION = 'SYSTEM_IDENTIFICATION'
    STATISTICAL = 'STATISTICAL'
    CLASSICAL_ML = 'CLASSICAL_ML'
    DEEP_LEARNING = 'DEEP_LEARNING'
    SCIML_HYBRID = 'SCIML_HYBRID'
    OPERATOR_FIELD = 'OPERATOR_FIELD'


ELIGIBILITY_STATUS = {'ELIGIBLE', 'CONDITIONAL', 'NOT_ELIGIBLE', 'NOT_APPLICABLE'}
PLAN_STATUS = {'READY', 'CONDITIONAL', 'UNSUPPORTED'}


@dataclass(frozen=True)
class ModelingContract:
    """A modeling-only contract.

    It intentionally does not encode fault ontology, diagnosis semantics, evidence
    calibration or maintenance actions.  Those belong outside M2N.
    """
    objective: ModelingObjective
    input_representation: InputRepresentation
    mechanism_usage: MechanismUsage = MechanismUsage.OPTIONAL
    prediction_unit: str = 'ROW'
    input_columns: tuple[str, ...] = ()
    target_columns: tuple[str, ...] = ()
    label_column: str | None = None
    time_column: str | None = None
    required_outputs: tuple[str, ...] = ('prediction',)
    constraints: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> 'ModelingContract':
        if not isinstance(d, Mapping):
            raise ValueError('modeling_contract must be an object')
        try:
            objective = ModelingObjective(str(d['objective']).upper())
        except KeyError:
            raise ValueError('modeling_contract.objective is required') from None
        try:
            representation = InputRepresentation(str(d.get('input_representation', 'TABULAR')).upper())
            mechanism_usage = MechanismUsage(str(d.get('mechanism_usage', 'OPTIONAL')).upper())
        except ValueError as e:
            raise ValueError(str(e)) from None
        inputs = tuple(map(str, d.get('input_columns') or ()))
        targets = tuple(map(str, d.get('target_columns') or ()))
        label = d.get('label_column')
        time_col = d.get('time_column')
        outputs = tuple(map(str, d.get('required_outputs') or ('prediction',)))
        c = cls(
            objective=objective,
            input_representation=representation,
            mechanism_usage=mechanism_usage,
            prediction_unit=str(d.get('prediction_unit', 'ROW')).upper(),
            input_columns=inputs,
            target_columns=targets,
            label_column=None if label is None else str(label),
            time_column=None if time_col is None else str(time_col),
            required_outputs=outputs,
            constraints=dict(d.get('constraints') or {}),
        )
        c.validate()
        return c

    def validate(self) -> 'ModelingContract':
        if self.objective == ModelingObjective.CATEGORICAL_PREDICTION and not self.label_column:
            raise ValueError('CATEGORICAL_PREDICTION requires modeling_contract.label_column')
        if self.objective == ModelingObjective.DYNAMICS_IDENTIFICATION and self.prediction_unit == 'ROW':
            raise ValueError('DYNAMICS_IDENTIFICATION requires a trajectory/context prediction_unit, not ROW')
        if self.objective == ModelingObjective.OPERATOR_LEARNING and self.input_representation not in (
            InputRepresentation.FUNCTION, InputRepresentation.REGULAR_GRID_FIELD
        ):
            raise ValueError('OPERATOR_LEARNING requires FUNCTION or REGULAR_GRID_FIELD representation')
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['objective'] = self.objective.value
        d['input_representation'] = self.input_representation.value
        d['mechanism_usage'] = self.mechanism_usage.value
        return d


@dataclass(frozen=True)
class FamilyEligibility:
    family: ModelFamily
    status: str
    reasons: tuple[str, ...]
    capability_note: str

    def to_dict(self) -> dict[str, Any]:
        if self.status not in ELIGIBILITY_STATUS:
            raise ValueError(f'unsupported eligibility status: {self.status}')
        return {
            'family': self.family.value,
            'status': self.status,
            'reasons': list(self.reasons),
            'capability_note': self.capability_note,
        }


@dataclass(frozen=True)
class ModelFamilyPlan:
    status: str
    contract: ModelingContract
    families: tuple[FamilyEligibility, ...]
    diagnostics: Mapping[str, Any]
    boundary: str = (
        'This plan qualifies model families only. It does not discover missing mechanism, '
        'calibrate evidence, assign fault meaning, or perform root-cause diagnosis. Some qualified temporal backends are implemented separately by temporal-design.'
    )

    def to_dict(self) -> dict[str, Any]:
        if self.status not in PLAN_STATUS:
            raise ValueError(f'unsupported plan status: {self.status}')
        return {
            'status': self.status,
            'modeling_contract': self.contract.to_dict(),
            'families': [x.to_dict() for x in self.families],
            'diagnostics': dict(self.diagnostics),
            'boundary': self.boundary,
        }


def _knowledge_levels(spec: Mapping[str, Any]) -> tuple[str, ...]:
    k = spec.get('knowledge') or {}
    levels: list[str] = []
    if isinstance(k, Mapping):
        for v in k.values():
            if isinstance(v, Mapping) and v.get('level'):
                levels.append(str(v['level']).upper())
        for slot in k.get('slots', []) or []:
            if isinstance(slot, Mapping) and slot.get('level'):
                levels.append(str(slot['level']).upper())
    return tuple(levels)


def _k2_allows_k3(spec: Mapping[str, Any]) -> bool:
    k = spec.get('knowledge') or {}
    if not isinstance(k, Mapping):
        return False
    for slot in k.get('slots', []) or []:
        if isinstance(slot, Mapping) and str(slot.get('level', '')).upper() == 'K2' and bool(slot.get('allow_k3_escape', False)):
            return True
    for v in k.values():
        if isinstance(v, Mapping) and str(v.get('level', '')).upper() == 'K2' and bool(v.get('allow_k3_escape', False)):
            return True
    return False


def profile_modeling_data(spec: Mapping[str, Any], contract: ModelingContract, data: Mapping[str, np.ndarray] | None) -> dict[str, Any]:
    diag: dict[str, Any] = {'data_supplied': data is not None}
    if data is None:
        return diag
    n = len(next(iter(data.values()))) if data else 0
    diag['rows'] = int(n)
    diag['columns'] = list(data.keys())
    required = list(contract.input_columns) + list(contract.target_columns)
    if contract.label_column:
        required.append(contract.label_column)
    if contract.time_column:
        required.append(contract.time_column)
    missing = [c for c in dict.fromkeys(required) if c not in data]
    diag['missing_contract_columns'] = missing
    q = spec.get('qualification') or {}
    context_col = q.get('independent_context_column') if isinstance(q, Mapping) else None
    if context_col:
        diag['independent_context_column'] = str(context_col)
        if context_col in data:
            ctx = np.asarray(data[context_col])
            diag['independent_contexts'] = int(len(np.unique(ctx)))
        else:
            diag['independent_context_column_missing'] = True
    if contract.label_column and contract.label_column in data:
        labels = np.asarray(data[contract.label_column]).astype(str)
        diag['label_classes'] = int(len(np.unique(labels)))
    return diag


def qualify_model_families(
    spec: Mapping[str, Any],
    *,
    data: Mapping[str, np.ndarray] | None = None,
) -> ModelFamilyPlan:
    contract = ModelingContract.from_mapping(spec.get('modeling_contract') or {})
    levels = _knowledge_levels(spec)
    has_mechanism = bool(spec.get('mechanism')) and bool(levels)
    has_k3 = 'K3' in levels
    has_finite_physics = 'K2' in levels
    k2_escape = _k2_allows_k3(spec)
    diag = profile_modeling_data(spec, contract, data)
    diag.update({
        'knowledge_levels': list(levels),
        'has_admitted_mechanism': bool(has_mechanism),
        'has_k3_slot': bool(has_k3),
        'has_k2_slot': bool(has_finite_physics),
        'k2_allows_k3_escape': bool(k2_escape),
    })
    missing = tuple(diag.get('missing_contract_columns') or ())

    out: list[FamilyEligibility] = []
    obj = contract.objective
    rep = contract.input_representation
    mech = contract.mechanism_usage

    # PHYSICS_PARAMETRIC
    if obj in (ModelingObjective.STATIC_REGRESSION, ModelingObjective.DYNAMICS_IDENTIFICATION) and mech != MechanismUsage.NONE:
        if not has_mechanism:
            st, rs = 'NOT_ELIGIBLE', ('modeling contract requests mechanism use but no admitted K0/K1/K2/K3 mechanism is declared',)
        elif has_k3 and not any(x in levels for x in ('K0', 'K1', 'K2')):
            st, rs = 'CONDITIONAL', ('mechanism is dominated by K3 unknowns; a purely parametric physical family is not yet sufficient by itself',)
        else:
            st, rs = 'ELIGIBLE', ('admitted mechanism contains fixed/parametric/finite physical structure suitable for a physics-preserving candidate',)
    elif mech == MechanismUsage.PREFERRED and obj in (ModelingObjective.NORMALITY_MODELING, ModelingObjective.CATEGORICAL_PREDICTION):
        st, rs = 'CONDITIONAL', ('physics-parametric models may supply residual/state features, but are not automatically the direct target model for this contract',)
    else:
        st, rs = 'NOT_APPLICABLE', ('the modeling objective does not require a direct parametric physics predictor',)
    out.append(FamilyEligibility(ModelFamily.PHYSICS_PARAMETRIC, st, rs, 'grey-box equations, parameterized mechanism models, observer-like physical predictors'))

    # SYSTEM_IDENTIFICATION
    if obj == ModelingObjective.DYNAMICS_IDENTIFICATION:
        time_col = contract.time_column or (spec.get('data') or {}).get('time_column')
        if rep not in (InputRepresentation.MULTIVARIATE_SEQUENCE, InputRepresentation.TABULAR):
            st, rs = 'NOT_ELIGIBLE', ('dynamics identification currently requires a temporal tabular/sequence observation contract',)
        elif not time_col:
            st, rs = 'NOT_ELIGIBLE', ('no local time column is declared for dynamics identification',)
        elif data is not None and time_col not in data:
            st, rs = 'NOT_ELIGIBLE', (f'declared time column {time_col!r} is missing from supplied data',)
        else:
            st, rs = 'ELIGIBLE', ('temporal semantics are declared; state-space/system-identification families may participate subject to trajectory qualification',)
    elif obj == ModelingObjective.NORMALITY_MODELING and rep == InputRepresentation.MULTIVARIATE_SEQUENCE:
        st, rs = 'CONDITIONAL', ('state-space normal-behavior models are plausible, but require application-specific state/trajectory qualification',)
    else:
        st, rs = 'NOT_APPLICABLE', ('the current modeling objective is not a dynamics-identification problem',)
    out.append(FamilyEligibility(ModelFamily.SYSTEM_IDENTIFICATION, st, rs, 'state-space, ARX/NARX and sparse dynamics identification families; v2.10 temporal-design includes deterministic linear-AR plus K1, bounded-integrator, quaternion-kinematics and affine state-space mechanism adapters, but not a Kalman/continuous-time/nonlinear-operator zoo'))

    # STATISTICAL
    if obj == ModelingObjective.NORMALITY_MODELING and rep in (InputRepresentation.TABULAR, InputRepresentation.MULTIVARIATE_SEQUENCE, InputRepresentation.WAVEFORM_SEQUENCE):
        st, rs = 'ELIGIBLE', ('normality modeling with observed numeric variables is directly compatible with statistical residual/subspace families',)
    else:
        st, rs = 'NOT_APPLICABLE', ('statistical normality/subspace family is not the direct target family for this contract',)
    out.append(FamilyEligibility(ModelFamily.STATISTICAL, st, rs, 'PCA/DPCA, Mahalanobis and related statistical normal-behavior models'))

    # CLASSICAL_ML
    if obj in (ModelingObjective.STATIC_REGRESSION, ModelingObjective.CATEGORICAL_PREDICTION, ModelingObjective.NORMALITY_MODELING) and rep == InputRepresentation.TABULAR:
        if missing:
            st, rs = 'NOT_ELIGIBLE', (f'contract-required data columns are missing: {list(missing)}',)
        elif mech == MechanismUsage.REQUIRED:
            st, rs = 'NOT_ELIGIBLE', ('black-box classical ML does not by itself satisfy a REQUIRED mechanism-preservation contract',)
        elif mech == MechanismUsage.PREFERRED:
            st, rs = 'CONDITIONAL', ('classical ML may participate only with an explicit mechanism-feature/residual interface; mechanism use must not be silently discarded',)
        else:
            st, rs = 'ELIGIBLE', ('tabular modeling objective and supervision are compatible with classical ML candidates',)
    elif obj in (ModelingObjective.CATEGORICAL_PREDICTION, ModelingObjective.STATIC_REGRESSION) and rep == InputRepresentation.WAVEFORM_SEQUENCE:
        if missing:
            st, rs = 'NOT_ELIGIBLE', (f'contract-required data columns are missing: {list(missing)}',)
        elif mech == MechanismUsage.REQUIRED:
            st, rs = 'NOT_ELIGIBLE', ('generic feature-based classifiers do not by themselves satisfy a REQUIRED mechanism-preservation contract',)
        else:
            st, rs = 'CONDITIONAL', ('waveform modeling requires an explicit qualified feature representation before classical ML can compete; v2.15 provides deterministic time-domain waveform classification/regression, engineering-declared order-domain classification, and explicit competition among multiple candidate-specific waveform representations when their configurations are declared before selection',)
    elif obj == ModelingObjective.DYNAMICS_IDENTIFICATION:
        st, rs = 'CONDITIONAL', ('classical ML requires an explicitly qualified lag/state representation before competing on a dynamics contract',)
    else:
        st, rs = 'NOT_APPLICABLE', ('the current representation/objective is outside first-class classical ML eligibility',)
    out.append(FamilyEligibility(ModelFamily.CLASSICAL_ML, st, rs, 'linear/logistic models, SVM, tree ensembles and one-class learners; v2.15 supports deterministic time-domain and declared order-domain waveform ridge classification, including multiple candidate-specific representation configurations, plus time-domain waveform regression; broader backend choice remains application-triggered'))

    # DEEP_LEARNING
    if obj == ModelingObjective.OPERATOR_LEARNING:
        st, rs = 'NOT_APPLICABLE', ('operator-specific deep families are handled by OPERATOR_FIELD rather than generic deep learning',)
    elif mech == MechanismUsage.REQUIRED and obj in (ModelingObjective.STATIC_REGRESSION, ModelingObjective.DYNAMICS_IDENTIFICATION):
        st, rs = 'NOT_ELIGIBLE', ('generic black-box deep learning does not by itself satisfy a REQUIRED mechanism-preservation contract',)
    elif obj in (
        ModelingObjective.STATIC_REGRESSION,
        ModelingObjective.CATEGORICAL_PREDICTION,
        ModelingObjective.NORMALITY_MODELING,
        ModelingObjective.DYNAMICS_IDENTIFICATION,
    ):
        st, rs = 'CONDITIONAL', ('deep models require application-specific evidence of independent-data adequacy and representation need; v2.10 intentionally defines no universal sample threshold',)
    else:
        st, rs = 'NOT_APPLICABLE', ('generic deep learning is not applicable to this contract',)
    out.append(FamilyEligibility(ModelFamily.DEEP_LEARNING, st, rs, 'generic deep family; v2.10 provides native TCN/GRU/Transformer competition for qualified regular sequence contracts and a deterministic feature-based baseline for declared WAVEFORM_SEQUENCE regions of interest, while other deep backends remain application-triggered'))

    # SCIML_HYBRID
    if obj in (ModelingObjective.STATIC_REGRESSION, ModelingObjective.DYNAMICS_IDENTIFICATION) and mech in (MechanismUsage.REQUIRED, MechanismUsage.PREFERRED):
        if has_k3:
            st, rs = 'ELIGIBLE', ('a declared K3 unknown region can justify a constrained hybrid/SciML candidate subject to representation eligibility',)
        elif has_finite_physics and k2_escape:
            st, rs = 'CONDITIONAL', ('K2 finite physical hypotheses must be sufficiently rejected before the declared K3 escape may participate',)
        else:
            st, rs = 'NOT_ELIGIBLE', ('no qualified K3 unknown region or K2-to-K3 escape is declared; neural augmentation would add unsupported freedom',)
    else:
        st, rs = 'NOT_APPLICABLE', ('the modeling contract does not call for a mechanism-preserving hybrid predictor',)
    out.append(FamilyEligibility(ModelFamily.SCIML_HYBRID, st, rs, 'local constrained neuralization and Scientific-ML hybrids governed by existing K3/representation contracts'))

    # OPERATOR_FIELD
    if obj == ModelingObjective.OPERATOR_LEARNING:
        if rep in (InputRepresentation.FUNCTION, InputRepresentation.REGULAR_GRID_FIELD):
            st, rs = 'ELIGIBLE', ('operator-learning objective and function/field representation are explicitly declared',)
        else:
            st, rs = 'NOT_ELIGIBLE', ('operator learning requires FUNCTION or REGULAR_GRID_FIELD representation',)
    else:
        st, rs = 'NOT_APPLICABLE', ('the current modeling objective is not an operator-learning problem',)
    out.append(FamilyEligibility(ModelFamily.OPERATOR_FIELD, st, rs, 'DeepONet/FNO-style operator families; runtime remains application-triggered unless already integrated'))

    eligible = sum(x.status == 'ELIGIBLE' for x in out)
    conditional = sum(x.status == 'CONDITIONAL' for x in out)
    status = 'READY' if eligible else ('CONDITIONAL' if conditional else 'UNSUPPORTED')
    diag['eligible_family_count'] = int(eligible)
    diag['conditional_family_count'] = int(conditional)
    return ModelFamilyPlan(status, contract, tuple(out), diag)


def load_and_plan_model_families(spec_path: str | Path, data_path: str | Path | None = None) -> ModelFamilyPlan:
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    data = None
    if data_path is not None:
        from .goal_design import _read_csv
        data = _read_csv(Path(data_path))
    return qualify_model_families(spec, data=data)
