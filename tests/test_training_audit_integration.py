import json

import pandas as pd

from ai_job_market import core


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_pipeline_wrapper_preserves_return_and_creates_log(monkeypatch, tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "project.yaml").write_text(
        "project:\n  random_seed: 77\n  target: annual_salary_usd\n",
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text("x\n1\n", encoding="utf-8")

    def fake_impl(raw_path, root=None, workspace_root=None):
        audit = core.active_audit()
        assert audit is not None
        audit.set_pipeline_run_id("fake-pipeline-id")
        return {"run_id": "fake-pipeline-id", "ok": True}

    monkeypatch.setattr(core, "_run_pipeline_impl", fake_impl)

    result = core.run_pipeline(raw_path=raw_path, root=tmp_path, workspace_root=tmp_path)

    assert result == {"run_id": "fake-pipeline-id", "ok": True}
    logs = list((tmp_path / "outputs" / "08_full_pipeline" / "logs").glob("*.logs"))
    assert len(logs) == 1
    terminal = capsys.readouterr().out
    file_events = _events(logs[0])
    assert not any(line.lstrip().startswith("{") for line in terminal.splitlines())
    assert "Training audit run completed" in terminal
    assert [event["event"] for event in file_events] == [
        "run_started",
        "run_context",
        "run_completed",
    ]


def test_evaluate_model_cv_emits_fold_fit_predict_and_metric_events(tmp_path):
    rows = []
    for i in range(800):
        rows.append(
            {
                "posting_year": 2025 + (i // 12),
                "posting_month": (i % 12) + 1,
                "years_of_experience": float(i),
                core.TARGET: float(i),
            }
        )
    dev = pd.DataFrame(rows)

    with core.start_training_audit(
        workspace_root=tmp_path,
        raw_path=tmp_path / "input.csv",
        seed=42,
        target=core.TARGET,
        features=["years_of_experience"],
        stream=__import__("io").StringIO(),
    ) as audit:
        fold, summary = core.evaluate_model_cv(
            dev,
            ["years_of_experience"],
            "Linear Regression",
            core.LinearRegression(),
            n_splits=2,
        )
        log_path = audit.log_path

    assert summary["folds"] == len(fold) == 2
    events = _events(log_path)
    names = [event["event"] for event in events]
    assert "split_defined" in names
    assert names.count("candidate_fold_scored") == 2
    assert any(event["operation"] == "preprocess_and_fit" for event in events)
    assert any(event["operation"] == "predict_validation" for event in events)
