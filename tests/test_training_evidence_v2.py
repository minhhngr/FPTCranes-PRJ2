from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_job_market.training_evidence_io import (
    EvidenceContractError,
    TrainingEventWriter,
    claim_holdout_access,
    complete_holdout_access,
    fail_holdout_access,
    holdout_identity,
    load_approval,
    load_policy,
    make_run_id,
    publish_staged_pack,
    required_evidence_files,
    safe_workspace_path,
    validate_complete_pack,
    validate_metric,
    validate_run_id,
)


def _policy() -> dict:
    return {
        "schema_version": "training-validation-policy/v1",
        "seed": 42,
        "split_targets": {"train": 0.8, "evaluation_holdout": 0.19, "inference_reserve": 0.01},
        "outer_folds": 5,
        "inner_folds": 3,
        "top2_features": ["job_category", "years_of_experience"],
        "n_jobs": 1,
        "permutation_repeats": 12,
        "diagnostic_gap": 0.2,
        "small_subgroup_rows": 20,
        "runtime_protocol": {
            "method_version": "warm-train-predict/v1",
            "warmups": 3,
            "measured_calls": 30,
            "batch_sizes": [1, 100],
            "max_benchmark_predict_calls": 1784,
            "numerical_threads": 1,
            "execution": "sequential",
        },
    }


def test_safe_workspace_path_rejects_escape_and_symlink(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)

    with pytest.raises(EvidenceContractError, match="unsafe"):
        safe_workspace_path(tmp_path, "../outside")
    with pytest.raises(EvidenceContractError, match="symlink"):
        safe_workspace_path(tmp_path, "link/file.json")


def test_run_id_is_stable_and_strict() -> None:
    run_id = make_run_id("a" * 64)
    assert run_id == f"tv-{'a' * 32}"
    validate_run_id(run_id)
    for bad in ("tv-short", "TV-" + "a" * 32, "tv-" + "g" * 32):
        with pytest.raises(EvidenceContractError, match="run ID"):
            validate_run_id(bad)


def test_metric_rejects_nonfinite_and_preserves_explicit_null_reason() -> None:
    assert validate_metric(12.5, None) == (12.5, None)
    assert validate_metric(None, "constant_target") == (None, "constant_target")
    with pytest.raises(EvidenceContractError, match="finite"):
        validate_metric(float("nan"), None)
    with pytest.raises(EvidenceContractError, match="reason"):
        validate_metric(None, None)


def test_policy_is_strict_and_rejects_expanded_or_unknown_values(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    policy = load_policy(policy_path)
    assert policy["runtime_protocol"]["max_benchmark_predict_calls"] == 1784

    bad = _policy()
    bad["outer_folds"] = 4
    policy_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="outer_folds"):
        load_policy(policy_path)

    bad = _policy()
    bad["surprise"] = True
    policy_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="unknown"):
        load_policy(policy_path)


def test_approval_parsing_rejects_stale_shapes_and_invalid_limits(tmp_path: Path) -> None:
    approval = {
        "schema_version": "training-validation-approval/v1",
        "source_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "proposal_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
        "train_end": "2025-10",
        "holdout_end": "2025-11",
        "actual_counts": {"TRAIN": 80, "EVALUATION_HOLDOUT": 19, "INFERENCE_RESERVE": 1},
        "actual_shares": {"TRAIN": 0.8, "EVALUATION_HOLDOUT": 0.19, "INFERENCE_RESERVE": 0.01},
        "reviewer": "data-owner",
        "approved_at": "2026-09-19T00:00:00Z",
        "exposure": {"holdout": "attested-unexposed", "reserve": "attested-unexposed"},
        "max_holdout_mae_usd": 20000.0,
        "max_holdout_rmse_usd": 30000.0,
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval), encoding="utf-8")
    assert load_approval(path)["train_end"] == "2025-10"

    approval["max_run_wall_s"] = -1
    path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="max_run_wall_s"):
        load_approval(path)


def test_repository_policy_locks_feature_families_and_search_bounds(tmp_path: Path) -> None:
    repository_policy = Path(__file__).resolve().parents[1] / "config/training_validation.json"
    policy = load_policy(repository_policy)
    assert len(policy["feature_families"]) == 6
    assert policy["rf_search"]["method_version"] == "gridsearchcv-temporal/v1"
    assert policy["rf_search"]["scoring"] == "neg_mean_absolute_error"
    assert len(policy["rf_search"]["n_estimators"]) * len(policy["rf_search"]["max_depth"]) == 36

    bad = json.loads(repository_policy.read_text(encoding="utf-8"))
    bad["feature_families"]["experience_education"].append("experience_level")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="feature_families"):
        load_policy(path)


def test_policy_file_size_is_bounded(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(EvidenceContractError, match="1 MiB"):
        load_policy(policy_path)


def test_event_writer_emits_correlated_terminal_event_and_redacts_secrets(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    with TrainingEventWriter(events, "tv-" + "a" * 32, invocation_id="inv-test") as writer:
        operation_id = writer.start("candidate_fold", message="Started fold", planned_units=1)
        writer.complete(
            operation_id,
            message="Completed fold",
            metrics=[{"metric_id": "mae_usd", "value": 123.0, "unit": "USD/year"}],
            details={"password": "secret", "fold_id": "outer-1"},
        )

    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["started", "completed"]
    assert rows[0]["operation_id"] == rows[1]["operation_id"] == operation_id
    assert rows[1]["details"]["password"] == "[REDACTED]"
    assert rows[1]["event_id"] == "inv-test:2"
    assert rows[1]["progress"]["completed_units"] == 1


def test_complete_pack_validator_checks_required_files_hashes_and_schema(tmp_path: Path) -> None:
    run_id = "tv-" + "6" * 32
    run_dir = tmp_path / "outputs/training_validation" / run_id
    run_dir.mkdir(parents=True)
    files = []
    for relative in sorted(required_evidence_files()):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")
        from ai_job_market.training_evidence_io import sha256_file

        files.append(
            {
                "path": f"outputs/training_validation/{run_id}/{relative}",
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": "training-validation/v1",
        "training_method_version": "gridsearchcv-temporal/v1",
        "run_id": run_id,
        "execution_status": "complete",
        "files": files,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_complete_pack(tmp_path, run_id)["run_id"] == run_id

    old_manifest = manifest | {"training_method_version": "manual-stepwise/v1"}
    (run_dir / "manifest.json").write_text(json.dumps(old_manifest), encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="training method"):
        validate_complete_pack(tmp_path, run_id)

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "fold_summary.csv").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="checksum"):
        validate_complete_pack(tmp_path, run_id)


def test_staged_pack_publication_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    run_id = "tv-" + "8" * 32
    staging = tmp_path / "outputs/training_validation" / f".{run_id}.staging"
    staging.mkdir(parents=True)
    for relative in required_evidence_files():
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")
    final = publish_staged_pack(tmp_path, run_id, staging)
    assert final.name == run_id
    assert not staging.exists()
    assert validate_complete_pack(tmp_path, run_id)["execution_status"] == "complete"
    with pytest.raises(EvidenceContractError, match="already exists"):
        publish_staged_pack(tmp_path, run_id, staging)


def test_pack_validator_rejects_staging_and_unknown_schema(tmp_path: Path) -> None:
    run_id = "tv-" + "7" * 32
    staging = tmp_path / "outputs/training_validation" / f".{run_id}.staging"
    staging.mkdir(parents=True)
    with pytest.raises(EvidenceContractError, match="complete manifest"):
        validate_complete_pack(tmp_path, run_id)


def test_holdout_identity_survives_changed_container_and_run_id_cannot_bypass_lock(
    tmp_path: Path,
) -> None:
    holdout_id = holdout_identity(["row-b", "row-a", "row-c"])
    assert holdout_id == holdout_identity(["row-c", "row-a", "row-b"])
    run1 = "tv-" + "1" * 32
    run2 = "tv-" + "2" * 32
    first = claim_holdout_access(tmp_path, holdout_id, "a" * 64, run1)
    assert first["state"] == "started"
    with pytest.raises(EvidenceContractError, match="already started"):
        claim_holdout_access(tmp_path, holdout_id, "a" * 64, run2)
    fail_holdout_access(tmp_path, holdout_id, reason_code="EVALUATION_FAILED")
    with pytest.raises(EvidenceContractError, match="previously exposed"):
        claim_holdout_access(tmp_path, holdout_id, "a" * 64, run2)


def test_completed_holdout_is_reused_only_for_matching_declaration(tmp_path: Path) -> None:
    holdout_id = holdout_identity(["row-a"])
    run_id = "tv-" + "3" * 32
    claim_holdout_access(tmp_path, holdout_id, "b" * 64, run_id)
    complete_holdout_access(
        tmp_path,
        holdout_id,
        evidence_path="outputs/training_validation/tv-result/holdout_metrics.csv",
        evidence_sha256="c" * 64,
    )
    reused = claim_holdout_access(tmp_path, holdout_id, "b" * 64, "tv-" + "4" * 32)
    assert reused["state"] == "complete"
    assert reused["reuse"] is True
    with pytest.raises(EvidenceContractError, match="different frozen declaration"):
        claim_holdout_access(tmp_path, holdout_id, "d" * 64, "tv-" + "5" * 32)


def test_event_writer_records_failure_on_context_error(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        with TrainingEventWriter(events, "tv-" + "b" * 32, invocation_id="inv-fail") as writer:
            writer.start("run", message="Start")
            raise RuntimeError("boom")
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["reason_code"] == "UNHANDLED_RUNTIME_ERROR"
    assert "boom" not in json.dumps(rows[-1])
