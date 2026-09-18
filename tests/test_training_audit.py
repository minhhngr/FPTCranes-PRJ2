import json
from io import StringIO
from pathlib import Path

import pytest

from ai_job_market import training_audit


def _file_events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_audit_session_writes_strict_json_to_file_and_readable_terminal(tmp_path):
    stream = StringIO()
    with training_audit.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target="annual_salary_usd",
        features=["years_of_experience"],
        stream=stream,
    ) as audit:
        audit.event(
            "operation_started",
            step_id="unit.step",
            operation="unit_operation",
            status="started",
            message="Starting unit operation.",
            extra={"model_id": "unit", "fold_id": 1},
        )
        log_path = audit.log_path

    file_events = _file_events(log_path)
    assert file_events[0]["event"] == "run_started"
    assert file_events[1]["message"] == "Starting unit operation."
    assert file_events[-1]["event"] == "run_completed"
    assert [event["sequence"] for event in file_events] == list(range(1, len(file_events) + 1))
    assert all(event["schema_version"] == 1 for event in file_events)
    assert all(event["audit_run_id"] == file_events[0]["audit_run_id"] for event in file_events)
    terminal = stream.getvalue()
    assert "Training audit run started" in terminal
    assert not any(line.lstrip().startswith("{") for line in terminal.splitlines())


def test_audit_session_validates_workspace_containment(tmp_path):
    unsafe_link = tmp_path / "workspace" / "outputs" / "08_full_pipeline" / "logs"
    unsafe_link.parent.mkdir(parents=True)
    unsafe_link.symlink_to(tmp_path)

    with pytest.raises(ValueError, match="Unsafe log directory"):
        training_audit.start_training_audit(
            workspace_root=tmp_path / "workspace",
            raw_path=tmp_path / "input.csv",
            seed=42,
            target="annual_salary_usd",
            features=[],
        ).__enter__()


def test_audit_event_outside_session_is_noop(tmp_path, capsys):
    training_audit.emit_event(
        "operation_started",
        step_id="outside",
        operation="noop",
        status="started",
        message="This should not be emitted.",
    )

    assert capsys.readouterr().out == ""
    assert not list(tmp_path.rglob("*.logs"))


def test_audit_file_write_failure_marks_degraded_but_does_not_raise(tmp_path):
    stream = StringIO()
    with training_audit.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target="annual_salary_usd",
        features=[],
        stream=stream,
    ) as audit:
        assert audit._fh is not None
        audit._fh.close()
        audit.event(
            "operation_started",
            step_id="degraded.write",
            operation="unit_operation",
            status="started",
            message="Starting operation after file sink failure.",
        )
        assert audit.log_complete is False

    assert "evidence write failed" in stream.getvalue().lower()
    assert not any(line.lstrip().startswith("{") for line in stream.getvalue().splitlines())


def test_broken_console_does_not_prevent_file_events(tmp_path):
    class BrokenStream:
        def write(self, value):
            raise OSError("terminal unavailable")

        def flush(self):
            raise OSError("terminal unavailable")

    with training_audit.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target="annual_salary_usd",
        features=[],
        stream=BrokenStream(),
    ) as audit:
        audit.event(
            "data_summary",
            step_id="unit.summary",
            operation="unit",
            status="completed",
            message="Summary recorded.",
        )
        log_path = audit.log_path

    events = _file_events(log_path)
    assert any(event["event"] == "data_summary" for event in events)
    assert audit.console_complete is False


def test_audit_records_cancelled_run(tmp_path):
    stream = StringIO()
    with pytest.raises(KeyboardInterrupt):
        with training_audit.start_training_audit(
            workspace_root=tmp_path,
            raw_path=tmp_path / "input.csv",
            seed=42,
            target="annual_salary_usd",
            features=[],
            stream=stream,
        ) as audit:
            log_path = audit.log_path
            raise KeyboardInterrupt

    events = _file_events(log_path)
    assert events[-1]["event"] == "run_cancelled"
    assert events[-1]["status"] == "cancelled"


def test_audit_serializes_non_finite_numbers_as_null_with_valid_json(tmp_path):
    with training_audit.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target="annual_salary_usd",
        features=[],
        stream=StringIO(),
    ) as audit:
        audit.event(
            "candidate_completed",
            step_id="nan.value",
            operation="serialize_candidate",
            status="completed",
            message="Candidate with non-finite value recorded.",
            extra={"max_depth": float("nan"), "score": float("inf")},
        )
        log_path = audit.log_path

    candidate = [e for e in _file_events(log_path) if e["event"] == "candidate_completed"][0]
    assert candidate["max_depth"] is None
    assert candidate["score"] is None


def test_nested_sessions_restore_parent_context(tmp_path):
    with training_audit.start_training_audit(
        workspace_root=tmp_path / "outer",
        raw_path=tmp_path / "input.csv",
        seed=1,
        target="target",
        features=[],
        stream=StringIO(),
    ) as outer:
        assert training_audit.active_audit() is outer
        with training_audit.start_training_audit(
            workspace_root=tmp_path / "inner",
            raw_path=tmp_path / "input.csv",
            seed=2,
            target="target",
            features=[],
            stream=StringIO(),
        ) as inner:
            assert training_audit.active_audit() is inner
        assert training_audit.active_audit() is outer
    assert training_audit.active_audit() is None
