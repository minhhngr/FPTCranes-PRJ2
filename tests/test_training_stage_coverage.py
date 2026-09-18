import json
from io import StringIO

import pytest

from ai_job_market.training_audit import start_training_audit


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_observer_receives_copy_and_cannot_corrupt_file_event(tmp_path):
    observed = []

    def observer(event):
        observed.append(event)
        event["status"] = "corrupted"

    with start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=1,
        target="target",
        features=[],
        stream=StringIO(),
        on_event=observer,
    ) as audit:
        audit.event(
            "stage_started",
            step_id="stage.B4",
            operation="model_comparison",
            status="started",
            message="Compare models.",
            detail_level=1,
            extra={"stage_id": "B4"},
        )
        audit.event(
            "stage_completed",
            step_id="stage.B4",
            operation="model_comparison",
            status="completed",
            message="Compared models.",
            detail_level=1,
            extra={"stage_id": "B4"},
        )
        log_path = audit.log_path

    saved = _events(log_path)
    assert observed
    assert [e["status"] for e in saved if e["event"].startswith("stage_")] == [
        "started",
        "completed",
    ]


def test_active_stage_is_failed_before_run_failure(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with start_training_audit(
            workspace_root=tmp_path,
            raw_path=tmp_path / "input.csv",
            seed=1,
            target="target",
            features=[],
            stream=StringIO(),
        ) as audit:
            audit.start_stage("B4", "Compare models")
            log_path = audit.log_path
            raise RuntimeError("boom")

    events = _events(log_path)
    assert [e["event"] for e in events[-2:]] == ["stage_failed", "run_failed"]
    assert events[-2]["stage_id"] == "B4"


def test_stage_completion_requires_matching_open_stage(tmp_path):
    with start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=1,
        target="target",
        features=[],
        stream=StringIO(),
    ) as audit:
        audit.start_stage("C1", "Read data")
        with pytest.raises(ValueError, match="not active"):
            audit.complete_stage("C2", "Wrong stage")
        audit.complete_stage("C1", "Read data")
