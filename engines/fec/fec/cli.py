from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from fec.runtime import (
    ag_from_config,
    build_model,
    fixed_policy,
    infer_rows,
    read_config,
    read_csv_matrix,
    specs_from_config,
    write_states,
)
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta
from fec.extensions.reference_bundle_store import (
    load_reference_bundle_manager,
    save_reference_bundle_manager,
)
from fec.extensions.evidence_provenance import (
    AssetReferenceScope,
    OperatingContextIdentity,
    ModalityAvailability,
    ReferenceCompatibilityContract,
    SourceDomain,
)


def _claims(value: str | None) -> set[str]:
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def _reference_compatibility_from_config(
    cfg, *, commissioned: bool = False
) -> ReferenceCompatibilityContract | None:
    rc = cfg.get("reference_contract")
    if not rc:
        return None
    modalities = rc.get("modalities", {})
    availability = None if not modalities else ModalityAvailability.from_mapping(modalities)
    domain = rc.get("source_domain")
    source_domain = None if domain is None else SourceDomain(str(domain))
    ctx_cfg = rc.get("operating_context")
    operating_context = None
    if ctx_cfg is not None:
        if not isinstance(ctx_cfg, dict) or not ctx_cfg.get("namespace"):
            raise ValueError("reference_contract.operating_context requires namespace and attributes")
        operating_context = OperatingContextIdentity.from_mapping(
            str(ctx_cfg["namespace"]), ctx_cfg.get("attributes", {})
        )
    if commissioned:
        return ReferenceCompatibilityContract.commissioned(
            availability=availability,
            provider_fingerprints=rc.get("provider_fingerprints", []),
            source_domain=source_domain,
            asset_scope=AssetReferenceScope.from_dict(rc.get("asset_scope")),
            operating_context=operating_context,
        )
    return ReferenceCompatibilityContract.from_runtime(
        availability=availability,
        provider_fingerprints=rc.get("provider_fingerprints", []),
        source_domain=source_domain,
        operating_context=operating_context,
        asset_id=rc.get("runtime_asset_id"),
        fleet_group_id=rc.get("runtime_fleet_group_id"),
    )


def _require_paths(args, names: Iterable[str]) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise ValueError("missing required commissioning/refresh arguments: " + ", ".join(missing))


def _write_production_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_run(args) -> int:
    cfg = read_config(args.config)
    cols, specs = specs_from_config(cfg)
    A, G = ag_from_config(cfg)
    leaf, root = build_model(
        read_csv_matrix(args.normal_leaf, cols),
        read_csv_matrix(args.normal_root, cols),
        specs,
        A,
        G,
        cfg,
    )
    states = infer_rows(read_csv_matrix(args.input, cols), leaf, root, specs, A, G, cfg)
    write_states(args.output, states)
    counts: dict[str, int] = {}
    for state in states:
        counts[state.decision] = counts.get(state.decision, 0) + 1
    print(json.dumps({"rows": len(states), "decisions": counts, "output": str(Path(args.output).resolve())}, indent=2))
    return 0


def _commission_manager(args, cfg, cols, specs, A, G) -> tuple[ReferenceBundleManager, dict]:
    _require_paths(
        args,
        (
            "commission_leaf",
            "commission_root",
            "commission_sentinel",
            "bundle_id",
            "leaf_data_id",
            "root_data_id",
            "sentinel_data_id",
        ),
    )
    X_leaf = read_csv_matrix(args.commission_leaf, cols)
    X_root = read_csv_matrix(args.commission_root, cols)
    X_sentinel = read_csv_matrix(args.commission_sentinel, cols)
    dep = frozenset(str(x) for x in cfg.get("calibration", {}).get("dependency_set", []))
    compatibility = _reference_compatibility_from_config(cfg, commissioned=True)
    sentinel_compatibility = _reference_compatibility_from_config(cfg)
    manager = ReferenceBundleManager(
        alpha=float(cfg.get("policy", {}).get("alpha", 0.05)),
        min_sentinel_samples=int(cfg.get("production_calibration", {}).get("min_sentinel_samples", 20)),
        sentinel_block_size=int(cfg.get("production_calibration", {}).get("sentinel_block_size", 1)),
    )
    meta = ReferenceBundleMeta(
        bundle_id=args.bundle_id,
        key=args.key,
        provenance=args.provenance,
        dependency_set=dep,
        leaf_data_ids=frozenset({args.leaf_data_id}),
        root_data_ids=frozenset({args.root_data_id}),
        notes="commissioned by fec.cli production",
        compatibility=compatibility,
    )
    staged = manager.stage_candidate(
        meta,
        X_leaf,
        X_root,
        specs,
        trusted_claims=_claims(args.trusted_claims),
        A=A,
        G=G,
        operators=cfg.get("operators"),
        expected_contact_count=int(cfg.get("expected_contact_count", len(specs))),
    )
    if not staged:
        return manager, {"staged": False, "promoted": False, "reason": "D_cal/trusted-claims rejected candidate"}
    validation = manager.validate_candidate(
        args.key,
        X_sentinel,
        sentinel_data_ids={args.sentinel_data_id},
        allow_conservative=bool(args.allow_conservative_commissioning),
        sentinel_compatibility=sentinel_compatibility,
    )
    promoted = False
    if validation.accepted_for_promotion:
        manager.promote(args.key)
        promoted = True
    return manager, {
        "staged": True,
        "promoted": promoted,
        "validation": {
            "accepted": validation.accepted_for_promotion,
            "power_limited": validation.power_limited,
            "supported_fraction": validation.supported_fraction,
            "reason": validation.reason,
            "health": None if validation.health is None else asdict(validation.health),
        },
    }


def _refresh_manager(args, manager, cfg, cols, specs, A, G) -> dict | None:
    requested = any((args.refresh_leaf, args.refresh_root, args.refresh_sentinel))
    if not requested:
        return None
    _require_paths(
        args,
        (
            "refresh_leaf",
            "refresh_root",
            "refresh_sentinel",
            "refresh_bundle_id",
            "refresh_leaf_data_id",
            "refresh_root_data_id",
            "refresh_sentinel_data_id",
        ),
    )
    dep = frozenset(str(x) for x in cfg.get("calibration", {}).get("dependency_set", []))
    compatibility = _reference_compatibility_from_config(cfg, commissioned=True)
    sentinel_compatibility = _reference_compatibility_from_config(cfg)
    meta = ReferenceBundleMeta(
        bundle_id=args.refresh_bundle_id,
        key=args.key,
        provenance=args.refresh_provenance,
        dependency_set=dep,
        leaf_data_ids=frozenset({args.refresh_leaf_data_id}),
        root_data_ids=frozenset({args.refresh_root_data_id}),
        notes="refresh staged by fec.cli production",
        compatibility=compatibility,
    )
    staged = manager.stage_candidate(
        meta,
        read_csv_matrix(args.refresh_leaf, cols),
        read_csv_matrix(args.refresh_root, cols),
        specs,
        trusted_claims=_claims(args.trusted_claims),
        A=A,
        G=G,
        operators=cfg.get("operators"),
        expected_contact_count=int(cfg.get("expected_contact_count", len(specs))),
    )
    if not staged:
        return {"staged": False, "promoted": False, "reason": "D_cal/trusted-claims rejected candidate"}
    validation = manager.validate_candidate(
        args.key,
        read_csv_matrix(args.refresh_sentinel, cols),
        sentinel_data_ids={args.refresh_sentinel_data_id},
        allow_conservative=bool(args.allow_conservative_refresh),
        protect_active_power=True,
        sentinel_compatibility=sentinel_compatibility,
    )
    promoted = False
    if validation.accepted_for_promotion:
        manager.promote(args.key)
        promoted = True
    return {
        "staged": True,
        "promoted": promoted,
        "validation": {
            "accepted": validation.accepted_for_promotion,
            "power_limited": validation.power_limited,
            "supported_fraction": validation.supported_fraction,
            "reason": validation.reason,
            "health": None if validation.health is None else asdict(validation.health),
        },
    }


def cmd_production(args) -> int:
    cfg = read_config(args.config)
    cols, specs = specs_from_config(cfg)
    A, G = ag_from_config(cfg)
    state_path = Path(args.state)
    commissioning = None
    changed = False
    if state_path.exists():
        manager = load_reference_bundle_manager(state_path)
        source = "restored"
    else:
        manager, commissioning = _commission_manager(args, cfg, cols, specs, A, G)
        source = "commissioned"
        changed = bool(commissioning.get("promoted"))
        if not changed:
            report = {"state": "COMMISSIONING_BLOCKED", "commissioning": commissioning}
            print(json.dumps(report, indent=2, default=str))
            return 2
    refresh = _refresh_manager(args, manager, cfg, cols, specs, A, G)
    if refresh is not None and refresh.get("promoted"):
        changed = True
    if manager.active_info(args.key) is None:
        print(json.dumps({"state": "REFERENCE_UNSUPPORTED", "key": args.key}, indent=2))
        return 3
    if changed or not state_path.exists():
        state_digest = save_reference_bundle_manager(manager, state_path)
    else:
        # Verify/read path was already integrity-checked by load; keep storage immutable when state did not change.
        state_digest = None

    X = read_csv_matrix(args.input, cols)
    bundle = manager._active[args.key]
    runtime_compatibility = _reference_compatibility_from_config(cfg)
    cal = manager.evaluate(args.key, X, runtime_compatibility=runtime_compatibility)
    if cal.state == "reference_unsupported":
        report = {
            "state": "REFERENCE_UNSUPPORTED",
            "manager_source": source,
            "active": manager.active_info(args.key),
            "runtime_reference_compatibility": None if runtime_compatibility is None else runtime_compatibility.to_runtime_dict(),
            "calibration_reason": cal.reason,
            "calibration_diagnostics": dict(cal.diagnostics),
        }
        print(json.dumps(report, indent=2, default=str))
        return 3
    root_E = manager._root_scores(bundle, X)
    present = np.asarray(bundle.G) > 0
    A_mean = float(np.mean(np.asarray(bundle.A)[present])) if np.any(present) else 0.0
    C_presence = float(np.sum(present) / bundle.expected_contact_count)
    C_effective = float(np.sum(np.asarray(bundle.A) * present) / bundle.expected_contact_count)
    rows = []
    for i in range(len(X)):
        if not cal.usable_for_policy[i]:
            p = float("nan")
            decision = "CALIBRATION_UNSUPPORTED"
        else:
            p = float(cal.p_values[i])
            decision = fixed_policy(
                p_root=p,
                A=A_mean,
                C=C_presence,
                S_ctx=1.0,
                S_ref=1.0,
                cfg=cfg,
            )
        rows.append(
            {
                "root_E": float(root_E[i]),
                "p_root": p,
                "A_mean_present": A_mean,
                "C_presence": C_presence,
                "C_effective": C_effective,
                "calibration_state": cal.state,
                "calibration_supported": bool(cal.supported[i]),
                "calibration_reason": cal.reason,
                "bundle_id": bundle.meta.bundle_id,
                "decision": decision,
            }
        )
    output = Path(args.output)
    _write_production_rows(output, rows)
    decisions = {d: sum(r["decision"] == d for r in rows) for d in sorted({r["decision"] for r in rows})}
    report = {
        "state": "OK",
        "manager_source": source,
        "commissioning": commissioning,
        "refresh": refresh,
        "active": manager.active_info(args.key),
        "runtime_reference_compatibility": None if runtime_compatibility is None else runtime_compatibility.to_runtime_dict(),
        "calibration_state": cal.state,
        "calibration_reason": cal.reason,
        "calibration_diagnostics": dict(cal.diagnostics),
        "state_file": str(state_path.resolve()),
        "state_payload_sha256": state_digest,
        "rows": len(rows),
        "decisions": decisions,
    }
    Path(str(output) + ".report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_verify_state(args) -> int:
    manager = load_reference_bundle_manager(args.state)
    print(json.dumps({"state": "VALID", "manifest": manager.manifest()}, indent=2, default=str))
    return 0


def cmd_class_calibrate(args) -> int:
    from fec.extensions.class_score_bridge import fit_class_score_reference
    out = fit_class_score_reference(
        args.calibration_scores, args.reference,
        alpha=args.alpha,
        lambda_vacuity=args.lambda_vacuity,
        min_group_size=args.min_group_size,
        score_contract=args.score_contract,
        provenance=args.provenance,
    )
    print(json.dumps({
        "state": "OK",
        "reference": str(Path(out).resolve()),
        "mode": "MASK_CONDITIONAL_CLASS_CONFORMAL",
        "note": "This calibrates class prediction sets only; it is not scalar anomaly Evidence and not DOR root-cause reasoning."
    }, indent=2))
    return 0


def cmd_class_infer(args) -> int:
    from fec.extensions.class_score_bridge import infer_class_prediction_sets
    out = infer_class_prediction_sets(args.scores, args.reference, args.output)
    print(json.dumps({
        "state": "OK",
        "output": str(Path(out).resolve()),
        "mode": "MASK_CONDITIONAL_CLASS_CONFORMAL",
        "note": "Rows contain supported prediction sets or abstention; final diagnostic arbitration remains downstream."
    }, indent=2))
    return 0


def cmd_anomaly_calibrate(args) -> int:
    from fec.extensions.anomaly_score_bridge import fit_anomaly_score_reference
    out = fit_anomaly_score_reference(
        args.calibration_scores, args.reference,
        context_columns=args.context_columns,
        alpha=args.alpha,
        min_group_size=args.min_group_size,
        score_contract=args.score_contract,
        provenance=args.provenance,
    )
    print(json.dumps({
        "state": "OK",
        "reference": str(Path(out).resolve()),
        "mode": "CONTEXT_CONDITIONAL_ANOMALY_SCORE",
        "note": "This calibrates scalar detector scores only; it does not produce fault classes or root cause."
    }, indent=2))
    return 0


def cmd_anomaly_infer(args) -> int:
    from fec.extensions.anomaly_score_bridge import infer_anomaly_decisions
    out = infer_anomaly_decisions(args.scores, args.reference, args.output)
    print(json.dumps({
        "state": "OK",
        "output": str(Path(out).resolve()),
        "mode": "CONTEXT_CONDITIONAL_ANOMALY_SCORE",
        "note": "Rows with unseen/sparse context fail closed as CONTEXT_UNSUPPORTED."
    }, indent=2))
    return 0


def cmd_class_context_calibrate(args) -> int:
    from fec.extensions.context_class_score_bridge import fit_context_class_score_reference
    out = fit_context_class_score_reference(
        args.calibration_scores, args.reference,
        context_columns=args.context_columns,
        alpha=args.alpha,
        min_group_size=args.min_group_size,
        score_contract=args.score_contract,
        provenance=args.provenance,
    )
    print(json.dumps({
        "state":"OK", "reference":str(Path(out).resolve()),
        "mode":"CONTEXT_CONDITIONAL_CLASS_CONFORMAL",
        "note":"Exact declared context/mask groups only; unseen/sparse groups abstain."
    }, indent=2))
    return 0


def cmd_class_context_infer(args) -> int:
    from fec.extensions.context_class_score_bridge import infer_context_class_prediction_sets
    out = infer_context_class_prediction_sets(args.scores, args.reference, args.output)
    print(json.dumps({
        "state":"OK", "output":str(Path(out).resolve()),
        "mode":"CONTEXT_CONDITIONAL_CLASS_CONFORMAL",
        "note":"Prediction sets are statistical support sets, not physical root-cause probabilities."
    }, indent=2))
    return 0


def cmd_mask_calibration_audit(args) -> int:
    from fec.extensions.mask_calibration_audit import write_mask_calibration_audit
    out = write_mask_calibration_audit(
        args.scores, args.reference, args.output,
        min_test_group_size=args.min_test_group_size,
        coverage_tolerance=args.coverage_tolerance,
    )
    report = json.loads(Path(out).read_text(encoding="utf-8"))
    print(json.dumps({
        "state": "OK", "output": str(Path(out).resolve()),
        "supported_fraction": report["supported_fraction"],
        "all_observed_masks_supported": report["all_observed_masks_supported"],
        "all_supported_groups_pass": report["all_supported_groups_pass"],
        "note": report["claim_boundary"],
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fec", description="FEC v2.4.2 Stable-derived DOR-integrated toolkit CLI")
    sp = p.add_subparsers(dest="command", required=True)

    run = sp.add_parser("run", help="minimal non-production run on precomputed Contact columns")
    run.add_argument("--config", required=True)
    run.add_argument("--normal-leaf", required=True)
    run.add_argument("--normal-root", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--output", default="fec_output.csv")
    run.set_defaults(func=cmd_run)

    prod = sp.add_parser("production", help="commission/restore a durable leaf+root Reference Bundle and infer")
    prod.add_argument("--config", required=True)
    prod.add_argument("--state", required=True, help="durable commissioned Reference Bundle JSON")
    prod.add_argument("--key", default="default")
    prod.add_argument("--input", required=True)
    prod.add_argument("--output", required=True)
    prod.add_argument("--trusted-claims", default="")
    prod.add_argument("--provenance", default="trusted-commissioning")
    prod.add_argument("--commission-leaf")
    prod.add_argument("--commission-root")
    prod.add_argument("--commission-sentinel")
    prod.add_argument("--bundle-id")
    prod.add_argument("--leaf-data-id")
    prod.add_argument("--root-data-id")
    prod.add_argument("--sentinel-data-id")
    prod.add_argument("--allow-conservative-commissioning", action="store_true")
    prod.add_argument("--refresh-leaf")
    prod.add_argument("--refresh-root")
    prod.add_argument("--refresh-sentinel")
    prod.add_argument("--refresh-bundle-id")
    prod.add_argument("--refresh-leaf-data-id")
    prod.add_argument("--refresh-root-data-id")
    prod.add_argument("--refresh-sentinel-data-id")
    prod.add_argument("--refresh-provenance", default="trusted-refresh")
    prod.add_argument("--allow-conservative-refresh", action="store_true")
    prod.set_defaults(func=cmd_production)

    verify = sp.add_parser("verify-state", help="verify and inspect a durable Reference Bundle state file")
    verify.add_argument("--state", required=True)
    verify.set_defaults(func=cmd_verify_state)

    cc = sp.add_parser("class-calibrate", help="fit/persist mask-conditional conformal reference from independent labelled class-score calibration data")
    cc.add_argument("--calibration-scores", required=True)
    cc.add_argument("--reference", required=True)
    cc.add_argument("--score-contract", default=None)
    cc.add_argument("--alpha", type=float, default=0.10)
    cc.add_argument("--lambda-vacuity", type=float, default=0.0)
    cc.add_argument("--min-group-size", type=int, default=20)
    cc.add_argument("--provenance", default="external-selected-classifier")
    cc.set_defaults(func=cmd_class_calibrate)

    ci = sp.add_parser("class-infer", help="apply a persisted class conformal reference to class-score rows and emit prediction sets/abstentions")
    ci.add_argument("--scores", required=True)
    ci.add_argument("--reference", required=True)
    ci.add_argument("--output", required=True)
    ci.set_defaults(func=cmd_class_infer)

    ac = sp.add_parser("anomaly-calibrate", help="fit/persist context-conditioned trusted-normal reference for external scalar anomaly scores")
    ac.add_argument("--calibration-scores", required=True)
    ac.add_argument("--reference", required=True)
    ac.add_argument("--score-contract", default=None)
    ac.add_argument("--context-columns", default="", help="comma-separated externally declared context columns; empty means global")
    ac.add_argument("--alpha", type=float, default=0.10)
    ac.add_argument("--min-group-size", type=int, default=20)
    ac.add_argument("--provenance", default="external-selected-detector")
    ac.set_defaults(func=cmd_anomaly_calibrate)

    ai = sp.add_parser("anomaly-infer", help="apply persisted context-conditioned anomaly score reference")
    ai.add_argument("--scores", required=True)
    ai.add_argument("--reference", required=True)
    ai.add_argument("--output", required=True)
    ai.set_defaults(func=cmd_anomaly_infer)

    ccc = sp.add_parser("class-context-calibrate", help="fit/persist class conformal reference keyed by declared operating context plus modality mask")
    ccc.add_argument("--calibration-scores", required=True)
    ccc.add_argument("--reference", required=True)
    ccc.add_argument("--score-contract", default=None)
    ccc.add_argument("--context-columns", required=True, help="comma-separated externally declared context columns")
    ccc.add_argument("--alpha", type=float, default=0.10)
    ccc.add_argument("--min-group-size", type=int, default=20)
    ccc.add_argument("--provenance", default="external-selected-classifier")
    ccc.set_defaults(func=cmd_class_context_calibrate)

    cci = sp.add_parser("class-context-infer", help="apply persisted context-conditioned class conformal reference")
    cci.add_argument("--scores", required=True)
    cci.add_argument("--reference", required=True)
    cci.add_argument("--output", required=True)
    cci.set_defaults(func=cmd_class_context_infer)

    mca = sp.add_parser("mask-calibration-audit", help="audit held-out coverage and efficiency for every observed modality mask")
    mca.add_argument("--scores", required=True, help="held-out class score CSV containing true_label")
    mca.add_argument("--reference", required=True)
    mca.add_argument("--output", required=True)
    mca.add_argument("--min-test-group-size", type=int, default=20)
    mca.add_argument("--coverage-tolerance", type=float, default=None)
    mca.set_defaults(func=cmd_mask_calibration_audit)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
