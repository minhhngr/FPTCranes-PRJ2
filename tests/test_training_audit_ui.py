from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pages.model_evidence import load_evidence
from pages.model_training_presentation import (
    load_compatible_training_audit,
    load_evidence_download,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _audit_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "workspace"
    raw = root / "data/raw/ai_jobs_market_2025_2026.csv"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"job_id,salary\n1,100\n")

    logs = root / "outputs/08_full_pipeline/logs"
    run_id = "20260918T152345Z-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    log = logs / f"training-{run_id}.logs"
    run_dir = log.with_suffix("")
    run_dir.mkdir(parents=True)
    event = {
        "schema_version": 1,
        "audit_run_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "pipeline_run_id": "20260918T152345Z",
        "sequence": 1,
        "event": "selection_recorded",
        "operation": "create_final_estimator",
        "status": "completed",
        "selection_source": "provided_tuning_table_first_row",
    }
    log.write_text(json.dumps(event) + "\n", encoding="utf-8")
    export = run_dir / "tuning-step1.csv"
    export.write_bytes(b"candidate,CV_R2\n1,0.8\n")
    manifest = {
        "schema_version": 1,
        "audit_run_id": event["audit_run_id"],
        "pipeline_run_id": event["pipeline_run_id"],
        "training_status": "completed",
        "log_complete": True,
        "console_complete": True,
        "coverage_complete": True,
        "source_path": str(raw),
        "source_sha256": _sha256(raw.read_bytes()),
        "exports": [
            {
                "evidence_id": "tuning-step1",
                "kind": "tuning_trials",
                "path": export.name,
                "status": "complete",
                "rows": 1,
                "columns": ["candidate", "CV_R2"],
                "sha256": _sha256(export.read_bytes()),
            }
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, log, manifest_path


def test_load_compatible_training_audit_validates_and_exposes_downloads(tmp_path):
    root, log, manifest_path = _audit_workspace(tmp_path)

    audit = load_compatible_training_audit(root)

    assert audit["available"] is True
    assert audit["audit_run_id"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert audit["pipeline_run_id"] == "20260918T152345Z"
    assert audit["selection_source"] == "provided_tuning_table_first_row"
    assert {item["kind"] for item in audit["downloads"]} == {
        "audit_manifest",
        "complete_jsonl",
        "tuning_trials",
    }
    downloads = {item["kind"]: item for item in audit["downloads"]}
    assert downloads["complete_jsonl"]["content"] == log.read_bytes()
    assert downloads["audit_manifest"]["content"] == manifest_path.read_bytes()
    assert all(item["sha256"] == _sha256(item["content"]) for item in audit["downloads"])
    assert audit["event_count"] == 1
    assert audit["event_preview"] == [
        {
            "sequence": 1,
            "timestamp": None,
            "operation": "create_final_estimator",
            "status": "completed",
            "model": None,
            "fold_id": None,
            "trial_id": None,
            "message": None,
            "evidence_ref": log.name + "#sequence=1",
        }
    ]


def test_training_audit_rejects_incomplete_source_mismatch_and_bad_checksum(tmp_path):
    root, _, manifest_path = _audit_workspace(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    manifest["coverage_complete"] = False
    manifest_path.write_text(json.dumps(manifest))
    audit = load_compatible_training_audit(root)
    assert audit["available"] is False
    assert "complete" in audit["reason"].lower()

    manifest["coverage_complete"] = True
    manifest["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    audit = load_compatible_training_audit(root)
    assert audit["available"] is False
    assert "source" in audit["reason"].lower()

    raw = root / "data/raw/ai_jobs_market_2025_2026.csv"
    manifest["source_sha256"] = _sha256(raw.read_bytes())
    manifest["exports"][0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    audit = load_compatible_training_audit(root)
    assert audit["available"] is False
    assert "checksum" in audit["reason"].lower()


def test_active_evidence_downloads_are_byte_exact_and_checksum_verified():
    manifest, _ = load_evidence(ROOT)

    for stem in [
        "candidate_fold_metrics",
        "fold_membership",
        "manual_tuning_step1_gridsearch",
        "manual_tuning_step2_n_estimators",
        "manual_tuning_step3_max_depth",
        "manual_tuning_step4_min_samples_leaf",
        "manual_tuning_step5_max_features",
        "manual_tuning_4params_summary",
    ]:
        download = load_evidence_download(ROOT, manifest, stem)
        source = ROOT / download["path"]
        assert download["content"] == source.read_bytes()
        assert download["sha256"] == _sha256(download["content"])


def test_training_audit_event_preview_is_bounded_and_rejects_non_objects(tmp_path):
    root, log, manifest_path = _audit_workspace(tmp_path)
    event = json.loads(log.read_text())
    log.write_text(
        "".join(
            json.dumps(event | {"sequence": index, "operation": "evaluate_rf_tuning_trial"})
            + "\n"
            for index in range(1, 251)
        ),
        encoding="utf-8",
    )
    audit = load_compatible_training_audit(root)
    assert audit["available"] is True
    assert audit["event_count"] == 250
    assert len(audit["event_preview"]) == 200
    assert audit["event_preview"][0]["sequence"] == 1
    assert audit["event_preview"][-1]["sequence"] == 200

    log.write_text(json.dumps(["not", "an", "event"]) + "\n", encoding="utf-8")
    audit = load_compatible_training_audit(root)
    assert audit["available"] is False
    assert "event" in audit["reason"].lower()


def test_training_audit_rejects_symlinked_export(tmp_path):
    root, _, manifest_path = _audit_workspace(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    external = tmp_path / "external.csv"
    external.write_bytes(b"candidate,CV_R2\n1,0.8\n")
    export = run_dir / manifest["exports"][0]["path"]
    export.unlink()
    export.symlink_to(external)
    manifest["exports"][0]["sha256"] = _sha256(external.read_bytes())
    manifest_path.write_text(json.dumps(manifest))

    audit = load_compatible_training_audit(root)

    assert audit["available"] is False
    assert "safe" in audit["reason"].lower()
