from pathlib import Path
import csv
import json

from fec.cli import main
from fec.extensions.class_score_bridge import fit_class_score_reference
from fec.extensions.mask_calibration_audit import audit_mask_conditional_class_reference


def _write(path: Path, masks, *, labels=True):
    fields = ["sample_id", "score__NORMAL", "score__FAULT", "mask__imu", "mask__power"]
    if labels:
        fields.append("true_label")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, mask in enumerate(masks):
            label = "NORMAL" if i % 2 == 0 else "FAULT"
            confidence = 0.91 if i % 7 else 0.72
            row = {
                "sample_id": i,
                "score__NORMAL": confidence if label == "NORMAL" else 1 - confidence,
                "score__FAULT": confidence if label == "FAULT" else 1 - confidence,
                "mask__imu": mask[0], "mask__power": mask[1],
            }
            if labels: row["true_label"] = label
            writer.writerow(row)


def test_mask_audit_reports_supported_and_unseen_groups(tmp_path: Path):
    cal = tmp_path / "cal.csv"
    _write(cal, [(1, 1)] * 40 + [(1, 0)] * 40)
    reference = tmp_path / "reference.json"
    fit_class_score_reference(cal, reference, alpha=.1, min_group_size=20)
    test = tmp_path / "test.csv"
    _write(test, [(1, 1)] * 20 + [(1, 0)] * 20 + [(0, 1)] * 5)
    report = audit_mask_conditional_class_reference(test, reference, min_test_group_size=10)
    assert report.supported_fraction == 40 / 45
    assert report.unsupported_rows == 5
    assert report.all_observed_masks_supported is False
    by_mask = {row.mask: row for row in report.groups}
    assert by_mask[(1, 1)].pass_coverage
    assert by_mask[(1, 0)].mean_set_size is not None
    assert by_mask[(0, 1)].calibration_supported is False


def test_mask_audit_cli_persists_claim_boundary(tmp_path: Path, monkeypatch):
    cal = tmp_path / "cal.csv"; _write(cal, [(1, 1)] * 40)
    reference = tmp_path / "reference.json"; fit_class_score_reference(cal, reference, min_group_size=20)
    test = tmp_path / "test.csv"; _write(test, [(1, 1)] * 20)
    output = tmp_path / "audit.json"
    monkeypatch.setattr("sys.argv", ["fec", "mask-calibration-audit", "--scores", str(test), "--reference", str(reference), "--output", str(output), "--min-test-group-size", "10"])
    assert main() == 0
    payload = json.loads(output.read_text())
    assert payload["contract"] == "FEC-MASK-CALIBRATION-AUDIT/1.0"
    assert "exchangeability" in payload["claim_boundary"]
