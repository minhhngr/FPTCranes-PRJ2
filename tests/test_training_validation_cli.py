from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from ai_job_market.core import MODEL_FEATURES, TARGET
from ai_job_market.training_partitions import (
    build_expanding_monthly_folds,
    build_partition_membership,
)
from ai_job_market.training_validation import (
    benchmark_fitted_contexts,
    build_approval_template,
    compare_training_partition,
    inspect_workspace,
    main,
)


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    raw = root / "outputs/01_data_basic_clean/basic_clean.csv"
    raw.parent.mkdir(parents=True)
    rows = []
    for period in pd.period_range("2025-01", periods=12, freq="M"):
        for value in range(period.month + 1):
            rows.append(
                {
                    "posting_year": period.year,
                    "posting_month": period.month,
                    "job_category": "Data Science",
                    "years_of_experience": value,
                    "annual_salary_usd": 100_000 + value,
                }
            )
    pd.DataFrame(rows).to_csv(raw, index=False)
    return root


def _authorized_workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "authorized-workspace"
    raw = root / "outputs/01_data_basic_clean/basic_clean.csv"
    raw.parent.mkdir(parents=True)
    rows = []
    for period in pd.period_range("2024-01", periods=15, freq="M"):
        for item in range(3):
            years = float(item + period.month / 10)
            rows.append(
                {
                    "posting_year": period.year,
                    "posting_month": period.month,
                    "job_title": "Data Scientist" if item else "ML Engineer",
                    "job_category": "Data Science" if item else "AI Engineering",
                    "years_of_experience": years,
                    "education_required": "Bachelor's",
                    "city": "Hanoi",
                    "country": "Vietnam",
                    "remote_work": "Hybrid",
                    "company_size": "Mid-size (501-5000)",
                    "industry": "Technology",
                    "demand_score": 40 + period.month,
                    "benefits_score_10": 6.0 + item / 10,
                    "required_skills": "Python|SQL",
                    "skill_count": 2,
                    TARGET: 50_000 + 3_000 * years + 100 * period.month,
                }
            )
    pd.DataFrame(rows).to_csv(raw, index=False)
    approval = build_approval_template(root)
    approval.update(
        {
            "reviewer": "fixture-custodian",
            "approved_at": "2026-09-20T00:00:00Z",
            "exposure": {
                "holdout": "attested-unexposed",
                "reserve": "attested-unexposed",
            },
            "max_holdout_mae_usd": 100_000,
            "max_holdout_rmse_usd": 100_000,
            "max_run_wall_s": 120,
            "max_final_batch1_p95_ms": 1000,
            "max_bundle_bytes": 10_000_000,
        }
    )
    approval_path = root / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    return root, approval_path


def test_inspect_is_read_only_and_reports_exact_counts(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    result = inspect_workspace(root)
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    assert before == after
    assert result["execution_status"] == "blocked"
    assert result["reason_code"] == "RESERVE_EXPOSURE_UNKNOWN"
    assert sum(result["proposal"]["actual_counts"].values()) == result["dataset"]["rows"]
    assert result["proposal"]["train_end"] < result["proposal"]["holdout_end"]
    assert result["proposal"]["holdout_end"] < result["dataset"]["period_max"]
    assert result["fit_budget"]["maximum_fits"] == 540
    assert result["fit_budget"]["maximum_benchmark_predict_calls"] == 1784


def test_repository_wrapper_runs_inspect_without_pythonpath(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    checkout = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(checkout / "training_validation.py"),
            "inspect",
            "--workspace",
            str(root),
            "--policy",
            str(checkout / "config/training_validation.json"),
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "ModuleNotFoundError" not in completed.stderr
    assert json.loads(completed.stdout)["execution_status"] == "blocked"


def test_inspect_cli_prints_json_stdout_and_english_blocker_stderr(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    checkout = Path(__file__).resolve().parents[1]
    env = os.environ | {"PYTHONPATH": str(checkout / "src")}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_job_market.training_validation",
            "inspect",
            "--workspace",
            str(root),
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    payload = json.loads(completed.stdout)
    assert payload["reason_code"] == "RESERVE_EXPOSURE_UNKNOWN"
    assert "No fitting or prediction was performed" in completed.stderr
    assert "inference reserve" in completed.stderr.lower()
    assert not (root / "outputs/training_validation").exists()


def test_comparison_stage_freezes_family_from_complete_training_only_evidence() -> None:
    rows = []
    for month in range(1, 13):
        for item in range(3):
            rows.append(
                {
                    "posting_year": 2025,
                    "posting_month": month,
                    "job_title": "Data Scientist",
                    "job_category": "Data Science",
                    "years_of_experience": item + month / 10,
                    "education_required": "Bachelor's",
                    "city": "Hanoi",
                    "country": "Vietnam",
                    "remote_work": "Hybrid",
                    "company_size": "Mid-size (501-5000)",
                    "industry": "Technology",
                    "demand_score": 50 + month,
                    "benefits_score_10": 7.0,
                    "required_skills": "Python|SQL",
                    "skill_count": 2,
                    TARGET: 60000 + month * 500 + item * 2000,
                }
            )
    frame = pd.DataFrame(rows)
    membership = build_partition_membership(
        frame, dataset_id="fixture", train_end="2025-10", holdout_end="2025-11"
    )
    train = membership[membership["partition"] == "TRAIN"]
    folds = build_expanding_monthly_folds(
        train, n_splits=5, scope="outer", parent_population_id="TRAIN:fixture"
    )
    models = {
        "Dummy Median": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "Random Forest": RandomForestRegressor(n_estimators=3, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=3, random_state=42),
    }
    result = compare_training_partition(frame, membership, folds, models=models)
    assert len(result["evaluation"].fold_metrics) == 25
    assert result["selection"]["selected_family"] in models
    assert result["selection"]["frozen"] is True

    with pytest.raises(ValueError, match="five declared"):
        compare_training_partition(
            frame,
            membership,
            folds,
            models={"Dummy Median": DummyRegressor(strategy="median")},
        )


def test_benchmark_integration_uses_27_fitted_contexts_without_refit() -> None:
    class FittedSpy:
        def __init__(self):
            self.predict_calls = 0

        def fit(self, *_args, **_kwargs):
            raise AssertionError("benchmark must not fit")

        def predict(self, frame):
            self.predict_calls += 1
            return [0.0] * len(frame)

    contexts = {}
    spies = []
    frame = pd.DataFrame({"feature": range(4)})
    for index in range(25):
        spy = FittedSpy()
        spies.append(spy)
        contexts[f"candidate-{index}"] = (spy, frame)
    for role in ("final_full", "final_top2"):
        spy = FittedSpy()
        spies.append(spy)
        contexts[role] = (spy, frame)
    samples, metadata = benchmark_fitted_contexts(contexts, environment_id="test-env")
    assert len(samples) == 27 * 2 * 30
    assert sum(spy.predict_calls for spy in spies) == 27 * 2 * 33
    assert metadata["benchmark_predict_calls"] == 1782
    assert metadata["remaining_first_load_predict_calls"] == 2


def test_check_command_is_read_only_and_prints_validated_manifest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    run_id = "tv-" + "a" * 32
    seen = []

    def validate(workspace, identity):
        seen.append((workspace, identity))
        return {"schema_version": "training-validation/v1", "run_id": identity, "execution_status": "complete"}

    monkeypatch.setattr("ai_job_market.training_validation.validate_complete_pack", validate)
    assert main(["check", "--workspace", str(tmp_path), "--run-id", run_id]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_status"] == "complete"
    assert seen == [(tmp_path, run_id)]


def test_run_rejects_stale_approval_before_creating_a_namespace(tmp_path: Path) -> None:
    root, approval = _authorized_workspace(tmp_path)
    raw = root / "outputs/01_data_basic_clean/basic_clean.csv"
    frame = pd.read_csv(raw)
    frame.loc[0, TARGET] += 1
    frame.to_csv(raw, index=False)
    result = main(
        [
            "run",
            "--workspace",
            str(root),
            "--approval",
            str(approval),
        ]
    )
    assert result == 3
    validation_root = root / "outputs/training_validation"
    assert not validation_root.exists()
    assert not (root / "artifacts/training_validation").exists()


def test_stage_failure_removes_staging_without_opening_holdout(
    tmp_path: Path, monkeypatch
) -> None:
    root, approval = _authorized_workspace(tmp_path)

    def fail_comparison(*_args, **_kwargs):
        raise RuntimeError("fixture stage failure")

    monkeypatch.setattr(
        "ai_job_market.training_validation.compare_training_partition", fail_comparison
    )
    result = main(
        [
            "run",
            "--workspace",
            str(root),
            "--approval",
            str(approval),
        ]
    )
    assert result == 4
    validation_root = root / "outputs/training_validation"
    assert not any(validation_root.glob("tv-*"))
    assert not any(validation_root.glob(".*.staging"))
    assert not (validation_root / "holdout_access").exists()


def test_full_run_cli_publishes_checks_and_reuses_complete_pack(tmp_path: Path) -> None:
    root, approval = _authorized_workspace(tmp_path)
    checkout = Path(__file__).resolve().parents[1]
    env = os.environ | {
        "PYTHONPATH": str(checkout / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        "-m",
        "ai_job_market.training_validation",
        "run",
        "--workspace",
        str(root),
        "--approval",
        str(approval),
    ]
    completed = subprocess.run(
        command,
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["execution_status"] == "complete"
    assert payload["fits_performed"] <= 540
    assert payload["predictions_performed"] == 1784
    run_id = payload["run_id"]
    run_dir = root / "outputs/training_validation" / run_id
    assert run_dir.is_dir()
    predictions = pd.read_csv(run_dir / "holdout_predictions.csv")
    assert set(predictions["partition"]) == {"EVALUATION_HOLDOUT"}
    fold_summary = pd.read_csv(run_dir / "fold_summary.csv")
    assert (fold_summary["scope"] == "outer").sum() == 5
    assert (fold_summary["scope"] == "inner").sum() == 18
    assert fold_summary.query("fold_id == 'outer-3:inner-2'").iloc[0]["parent_fold_id"] == "outer-3"
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Parent-labelled inner tuning folds" in report_text
    assert "outer-3:inner-2" in report_text
    ui_summary = json.loads((run_dir / "ui_summary.json").read_text(encoding="utf-8"))
    assert sum(
        row["scope"] == "inner"
        for row in ui_summary["page04"]["fold_method"]["rows"]
    ) == 18
    runtime_summary = pd.read_csv(run_dir / "runtime_summary.csv")
    assert set(runtime_summary.dropna(subset=["first_load_ms"])["model_id"]) == {
        "final_full",
        "final_top2",
    }
    reserve_ids = set(
        pd.read_csv(run_dir / "partition_membership.csv")
        .query("partition == 'INFERENCE_RESERVE'")["row_id"]
    )
    assert not reserve_ids & set(predictions["row_id"])

    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_job_market.training_validation",
            "check",
            "--workspace",
            str(root),
            "--run-id",
            run_id,
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr

    before = {path.relative_to(root): path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    reused = subprocess.run(
        command,
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert reused.returncode == 0, reused.stderr
    reused_payload = json.loads(reused.stdout)
    assert reused_payload["execution_status"] == "reused"
    assert reused_payload["fits_performed"] == 0
    assert reused_payload["predictions_performed"] == 0
    after = {
        path.relative_to(root): path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before

    bundle = root / "artifacts/training_validation" / run_id / "full.joblib"
    bundle.write_bytes(bundle.read_bytes() + b"corrupt")
    corrupt = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_job_market.training_validation",
            "check",
            "--workspace",
            str(root),
            "--run-id",
            run_id,
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert corrupt.returncode == 4
    assert "bundle checksum mismatch" in corrupt.stderr


def test_inspect_rejects_oversized_row_count_before_any_fit(tmp_path: Path, monkeypatch) -> None:
    root = _workspace(tmp_path)
    monkeypatch.setattr("ai_job_market.training_validation.MAX_CSV_ROWS", 1)
    result = inspect_workspace(root)
    assert result["execution_status"] == "blocked"
    assert result["reason_code"] == "INPUT_RESOURCE_LIMIT"
