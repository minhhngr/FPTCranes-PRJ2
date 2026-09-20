from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_job_market.ui_evidence_io import (
    EvidenceContractError,
    atomic_publish,
    load_current_evidence,
    safe_workspace_path,
    sha256_file,
)


def test_safe_workspace_path_rejects_escape_and_symlink(tmp_path):
    with pytest.raises(EvidenceContractError):
        safe_workspace_path(tmp_path, "../outside")
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(EvidenceContractError):
        safe_workspace_path(tmp_path, "link/file.json")


def test_atomic_publish_keeps_previous_pointer_on_invalid_manifest(tmp_path):
    out = tmp_path / "outputs/ui_evidence"
    out.mkdir(parents=True)
    old = {
        "schema_version": 1,
        "evidence_id": "old",
        "manifest_path": "outputs/ui_evidence/old/manifest.json",
        "manifest_sha256": "x" * 64,
    }
    (out / "current.json").write_text(json.dumps(old))
    with pytest.raises(EvidenceContractError):
        atomic_publish(tmp_path, {"schema_version": 1, "evidence_id": "new", "files": []})
    assert json.loads((out / "current.json").read_text())["evidence_id"] == "old"


def test_load_current_validates_hash_and_required_fields(tmp_path):
    run = tmp_path / "outputs/ui_evidence/run-1"
    run.mkdir(parents=True)
    payload = run / "table.csv"
    payload.write_text("a\n1\n")
    manifest = {
        "schema_version": 1,
        "evidence_id": "run-1",
        "target": "annual_salary_usd",
        "full_features": ["a"],
        "top2_features": ["job_category", "years_of_experience"],
        "models": {},
        "components": {},
        "limitations": [],
        "files": [
            {
                "path": "outputs/ui_evidence/run-1/table.csv",
                "sha256": sha256_file(payload),
                "size": payload.stat().st_size,
            }
        ],
    }
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    pointer = {
        "schema_version": 1,
        "evidence_id": "run-1",
        "manifest_path": "outputs/ui_evidence/run-1/manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
    }
    (tmp_path / "outputs/ui_evidence/current.json").write_text(json.dumps(pointer))
    loaded = load_current_evidence(tmp_path)
    assert loaded["evidence_id"] == "run-1"
    payload.write_text("a\n2\n")
    with pytest.raises(EvidenceContractError, match="checksum"):
        load_current_evidence(tmp_path)
