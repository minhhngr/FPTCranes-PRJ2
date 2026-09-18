import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("training_pipeline_cli", ROOT / "pipeline.py")
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def test_parse_args_supports_debuglog(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--debuglog", "--no-heartbeat"])
    args = pipeline.parse_args()
    assert args.debuglog is True
    assert args.no_heartbeat is True


def test_terminal_progress_records_only_observed_stage_completion(tmp_path, capsys):
    progress = pipeline.TerminalProgress(
        pipeline.RunContext(root=tmp_path, raw_path=tmp_path / "raw.csv"),
        pipeline.STAGES,
        heartbeat_enabled=False,
    )
    progress.start()
    progress.observe(
        {
            "event": "stage_started",
            "stage_id": "C1",
            "message": "Read input.",
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )
    progress.observe(
        {
            "event": "stage_completed",
            "stage_id": "C1",
            "message": "Input read.",
            "timestamp": "2026-01-01T00:00:01Z",
        }
    )
    progress.finalize_unknown()
    progress.stop()

    assert progress.records[0]["stage_code"] == "C1"
    assert progress.records[0]["status"] == "PASS"
    assert all(row["status"] == "UNKNOWN" for row in progress.records[1:])
    output = capsys.readouterr().out
    assert "UNKNOWN" not in output  # UNKNOWN is timing evidence, not fabricated terminal events.


def test_all_core_stage_ids_match_cli_stage_ids(tmp_path):
    progress = pipeline.TerminalProgress(
        pipeline.RunContext(root=tmp_path, raw_path=tmp_path / "raw.csv"),
        pipeline.STAGES,
        heartbeat_enabled=False,
    )
    for stage in pipeline.STAGES:
        progress.observe({"event": "stage_started", "stage_id": stage.code, "message": stage.title})
        progress.observe({"event": "stage_completed", "stage_id": stage.code, "message": stage.title})
    progress.finalize_unknown()

    assert len(progress.records) == 12
    assert {row["status"] for row in progress.records} == {"PASS"}
    assert [stage.code for stage in pipeline.STAGES] == [
        "C1", "C2", "C3", "C4", "C5", "B1-B3", "A1-A8", "B4", "B5-B6", "B7", "I1", "F"
    ]


def test_main_passes_debuglog_and_observer_without_output_hooks(monkeypatch, tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {"data": raw, "heartbeat": 20.0, "no_heartbeat": True, "debuglog": True},
        )(),
    )
    monkeypatch.setattr(pipeline, "_preflight", lambda path: None)
    called = {}

    def fake_run_pipeline(**kwargs):
        called.update(kwargs)
        observer = kwargs["on_audit_event"]
        for stage in pipeline.STAGES:
            observer({"event": "stage_started", "stage_id": stage.code, "message": stage.title})
            observer({"event": "stage_completed", "stage_id": stage.code, "message": stage.title})
        return {
            "run_id": "test",
            "raw_shape": [1, 1],
            "clean_shape": [1, 1],
            "development_rows": 1,
            "locked_test_rows": 1,
            "segmentation": {},
            "best_model": "Dummy",
            "locked_test_metrics": {},
        }

    monkeypatch.setattr(pipeline.core, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(pipeline, "_save_terminal_timing", lambda *args: None)

    assert pipeline.main() == 0
    assert called["debuglog"] is True
    assert callable(called["on_audit_event"])


def _args(raw, *, debuglog=False):
    return type(
        "Args",
        (),
        {"data": raw, "heartbeat": 20.0, "no_heartbeat": True, "debuglog": debuglog},
    )()


def test_main_returns_preflight_exit_2(monkeypatch, tmp_path):
    raw = tmp_path / "missing.csv"
    monkeypatch.setattr(pipeline, "parse_args", lambda: _args(raw))
    monkeypatch.setattr(
        pipeline, "_preflight", lambda path: (_ for _ in ()).throw(FileNotFoundError(path))
    )
    assert pipeline.main() == 2


@pytest.mark.parametrize(
    ("error", "expected_exit"),
    [(RuntimeError("training failed"), 1), (KeyboardInterrupt(), 130)],
)
def test_main_preserves_training_failure_and_cancel_exits(
    monkeypatch, tmp_path, error, expected_exit
):
    raw = tmp_path / "raw.csv"
    raw.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "parse_args", lambda: _args(raw, debuglog=True))
    monkeypatch.setattr(pipeline, "_preflight", lambda path: None)
    monkeypatch.setattr(
        pipeline.core,
        "run_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(pipeline, "_save_terminal_timing", lambda *args: None)

    assert pipeline.main() == expected_exit
