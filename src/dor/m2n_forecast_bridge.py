from __future__ import annotations

"""Bind M2N v2.16 forecast-horizon qualification to a DOR deployment contract.

DOR owns the operational question "how far must this model remain useful?".
M2N owns model qualification/selection at that declared horizon.  This adapter
never invents a horizon and never upgrades rollout performance into RUL or
maintenance meaning.
"""

from copy import deepcopy
from typing import Any

BINDING_CONTRACT = "DOR-M2N-FORECAST-BINDING/1.0"


def bind_m2n_forecast_contract(manifest: dict[str, Any], device_card: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(device_card)
    deployment = dict(manifest.get("deployment") or {})
    declared = deployment.get("forecast_evaluation")
    constraints = out.setdefault("contract", {}).setdefault("constraints", {})
    existing = constraints.get("forecast_evaluation")

    if declared is None:
        if existing is not None and str(existing.get("mode", "ONE_STEP")).upper() == "RECURSIVE_ROLLOUT":
            raise ValueError("M2N recursive-rollout forecast_evaluation is present but DOR DatasetManifest.deployment.forecast_evaluation is not declared")
        return {
            "contract": BINDING_CONTRACT,
            "status": "NOT_REQUIRED",
            "source": None,
            "forecast_evaluation": existing,
            "device_card": out,
            "safety": {"horizon_invented": False, "rul_semantics_added": False},
        }

    mode = str(declared.get("mode", "ONE_STEP")).upper()
    try:
        horizon = int(declared.get("horizon_steps", 1))
    except Exception as exc:
        raise ValueError("DOR deployment forecast_evaluation.horizon_steps must be an integer") from exc
    if mode == "ONE_STEP" and horizon != 1:
        raise ValueError("DOR ONE_STEP deployment contract requires horizon_steps=1")
    if mode == "RECURSIVE_ROLLOUT" and horizon < 2:
        raise ValueError("DOR RECURSIVE_ROLLOUT deployment contract requires horizon_steps>=2")
    if mode not in {"ONE_STEP", "RECURSIVE_ROLLOUT"}:
        raise ValueError(f"unsupported DOR deployment forecast mode {mode!r}")
    resolved = {"mode": mode, "horizon_steps": horizon}
    if existing is not None:
        emode = str(existing.get("mode", "ONE_STEP")).upper()
        eh = int(existing.get("horizon_steps", 1))
        if emode != mode or eh != horizon:
            raise ValueError("M2N device-card forecast_evaluation disagrees with the DOR deployment contract")
    constraints["forecast_evaluation"] = resolved
    return {
        "contract": BINDING_CONTRACT,
        "status": "BOUND",
        "source": "DatasetManifest.deployment.forecast_evaluation",
        "forecast_evaluation": resolved,
        "device_card": out,
        "safety": {"horizon_invented": False, "future_exogenous_inputs_authorized": False, "rul_semantics_added": False},
    }
