"""Validation and structured evidence helpers for training-validation/v1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "training-validation/v1"
POLICY_SCHEMA_VERSION = "training-validation-policy/v1"
MAX_CONTROL_FILE_BYTES = 1024 * 1024
_RUN_ID = re.compile(r"^tv-[0-9a-f]{32}$")
_REDACTED_FIELDS = {"password", "secret", "token", "api_key", "credential"}
_APPROVAL_KEYS = {
    "schema_version",
    "source_sha256",
    "policy_sha256",
    "proposal_sha256",
    "partition_sha256",
    "train_end",
    "holdout_end",
    "actual_counts",
    "actual_shares",
    "reviewer",
    "approved_at",
    "exposure",
    "max_holdout_mae_usd",
    "max_holdout_rmse_usd",
    "max_run_wall_s",
    "max_final_batch1_p95_ms",
    "max_bundle_bytes",
}
_EXPECTED_FEATURE_FAMILIES = {
    "job_domain": ["job_title", "job_category"],
    "experience_education": ["years_of_experience", "education_required"],
    "geography": ["city", "country"],
    "company_work": ["remote_work", "company_size", "industry"],
    "demand_benefits": ["demand_score", "benefits_score_10"],
    "skills": ["required_skills", "skill_count"],
}
_EXPECTED_RF_SEARCH = {
    "anchors": [
        {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.7, "max_depth": 20},
        {"n_estimators": 100, "min_samples_leaf": 2, "max_features": 0.8, "max_depth": None},
        {"n_estimators": 80, "min_samples_leaf": 1, "max_features": 0.8, "max_depth": None},
        {"n_estimators": 200, "min_samples_leaf": 1, "max_features": 0.7, "max_depth": 20},
    ],
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}
_POLICY_KEYS = {
    "schema_version",
    "seed",
    "split_targets",
    "outer_folds",
    "inner_folds",
    "top2_features",
    "n_jobs",
    "permutation_repeats",
    "diagnostic_gap",
    "small_subgroup_rows",
    "runtime_protocol",
    "feature_families",
    "rf_search",
}


_REQUIRED_EVIDENCE_FILES = {
    "partition_declaration.json",
    "partition_membership.csv",
    "fold_membership.csv",
    "fold_summary.csv",
    "monthly_row_counts.csv",
    "fold_explanation.md",
    "model_conclusions.json",
    "events.jsonl",
    "training.log",
    "report.md",
    "metric_catalog.json",
    "agent_summary.json",
    "ui_summary.json",
    "oof_predictions.csv",
    "runtime.csv",
    "runtime_samples.csv",
    "runtime_summary.csv",
    "performance_comparison.csv",
    "accuracy_runtime_tradeoff.csv",
    "operational_assessment.json",
    "candidate_fold_metrics.csv",
    "candidate_summary.csv",
    "fit_diagnostics.csv",
    "ablation_metrics.csv",
    "fold_importance.csv",
    "importance_drift.csv",
    "family_selection.json",
    "tuning_trials.csv",
    "tuning_summary.json",
    "variant_fold_metrics.csv",
    "evaluation_declaration.json",
    "holdout_metrics.csv",
    "holdout_predictions.csv",
    "encoded_importance.csv",
    "permutation_importance.csv",
    "subgroup_metrics.csv",
    "uncertainty.json",
    "conclusion.json",
    "charts/model_comparison.html",
    "charts/accuracy_runtime_tradeoff.html",
    "charts/holdout_diagnostics.html",
    "charts/feature_importance.html",
}


class EvidenceContractError(ValueError):
    """Training-validation evidence failed boundary or schema validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_workspace_path(workspace: Path | str, relative: str | Path) -> Path:
    """Resolve a relative path without following symlinks or escaping workspace."""
    root = Path(workspace).resolve(strict=True)
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise EvidenceContractError(f"unsafe workspace path: {relative}")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceContractError(f"symlink paths are not allowed: {relative}")
    resolved = current.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise EvidenceContractError(f"unsafe path escapes workspace: {relative}")
    return resolved


def validate_run_id(run_id: str) -> None:
    if not _RUN_ID.fullmatch(run_id):
        raise EvidenceContractError("invalid training-validation run ID")


def make_run_id(experiment_digest: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", experiment_digest):
        raise EvidenceContractError("experiment digest must be 64 lowercase hexadecimal characters")
    return f"tv-{experiment_digest[:32]}"


def validate_metric(value: float | int | None, reason: str | None) -> tuple[float | None, str | None]:
    if value is None:
        if not reason:
            raise EvidenceContractError("null metric requires an explicit reason")
        return None, reason
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EvidenceContractError("metric value must be finite or explicit null with reason")
    return numeric, reason


def _require_exact(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise EvidenceContractError(f"invalid {name}: expected {expected!r}, received {actual!r}")


def _load_control_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceContractError(f"control file is missing: {path.name}")
    if path.is_symlink():
        raise EvidenceContractError(f"control file cannot be a symlink: {path.name}")
    if path.stat().st_size > MAX_CONTROL_FILE_BYTES:
        raise EvidenceContractError("control JSON exceeds 1 MiB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceContractError(f"invalid JSON control file: {path.name}") from error
    if not isinstance(value, dict):
        raise EvidenceContractError("control JSON root must be an object")
    return value


def load_policy(path: Path | str) -> dict[str, Any]:
    policy = _load_control_json(Path(path))
    unknown = sorted(set(policy) - _POLICY_KEYS)
    if unknown:
        raise EvidenceContractError(f"policy contains unknown fields: {unknown}")
    required = _POLICY_KEYS - {"feature_families", "rf_search"}
    missing = sorted(required - set(policy))
    if missing:
        raise EvidenceContractError(f"policy missing fields: {missing}")
    _require_exact("schema_version", policy["schema_version"], POLICY_SCHEMA_VERSION)
    seed = policy["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise EvidenceContractError("invalid seed")
    _require_exact("outer_folds", policy["outer_folds"], 5)
    _require_exact("inner_folds", policy["inner_folds"], 3)
    _require_exact(
        "split_targets",
        policy["split_targets"],
        {"train": 0.8, "evaluation_holdout": 0.19, "inference_reserve": 0.01},
    )
    _require_exact(
        "top2_features", policy["top2_features"], ["job_category", "years_of_experience"]
    )
    _require_exact("n_jobs", policy["n_jobs"], 1)
    _require_exact("permutation_repeats", policy["permutation_repeats"], 12)
    _require_exact("diagnostic_gap", policy["diagnostic_gap"], 0.2)
    _require_exact("small_subgroup_rows", policy["small_subgroup_rows"], 20)
    runtime = policy["runtime_protocol"]
    if not isinstance(runtime, dict):
        raise EvidenceContractError("invalid runtime_protocol")
    expected_runtime = {
        "method_version": "warm-train-predict/v1",
        "warmups": 3,
        "measured_calls": 30,
        "batch_sizes": [1, 100],
        "max_benchmark_predict_calls": 1784,
        "numerical_threads": 1,
        "execution": "sequential",
    }
    _require_exact("runtime_protocol", runtime, expected_runtime)
    if "feature_families" in policy:
        _require_exact("feature_families", policy["feature_families"], _EXPECTED_FEATURE_FAMILIES)
    if "rf_search" in policy:
        _require_exact("rf_search", policy["rf_search"], _EXPECTED_RF_SEARCH)
    return policy


def load_approval(path: Path | str) -> dict[str, Any]:
    approval = _load_control_json(Path(path))
    unknown = sorted(set(approval) - _APPROVAL_KEYS)
    if unknown:
        raise EvidenceContractError(f"approval contains unknown fields: {unknown}")
    required = {
        "schema_version",
        "source_sha256",
        "policy_sha256",
        "proposal_sha256",
        "partition_sha256",
        "train_end",
        "holdout_end",
        "actual_counts",
        "actual_shares",
        "reviewer",
        "approved_at",
        "exposure",
    }
    missing = sorted(required - set(approval))
    if missing:
        raise EvidenceContractError(f"approval missing fields: {missing}")
    _require_exact(
        "schema_version", approval["schema_version"], "training-validation-approval/v1"
    )
    for field in ("source_sha256", "policy_sha256", "proposal_sha256", "partition_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(approval[field])):
            raise EvidenceContractError(f"invalid {field}")
    for field in ("train_end", "holdout_end"):
        if not re.fullmatch(r"(?:19|20|21)\d{2}-(?:0[1-9]|1[0-2])", str(approval[field])):
            raise EvidenceContractError(f"invalid {field}")
    if approval["train_end"] >= approval["holdout_end"]:
        raise EvidenceContractError("approval month boundaries are not chronological")
    if not str(approval["reviewer"]).strip() or not str(approval["approved_at"]).endswith("Z"):
        raise EvidenceContractError("approval reviewer and UTC timestamp are required")
    expected_partitions = {"TRAIN", "EVALUATION_HOLDOUT", "INFERENCE_RESERVE"}
    if set(approval["actual_counts"]) != expected_partitions or set(
        approval["actual_shares"]
    ) != expected_partitions:
        raise EvidenceContractError("approval partition counts/shares have an invalid shape")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in approval["actual_counts"].values()
    ):
        raise EvidenceContractError("approval partition counts must be positive integers")
    shares = [float(approval["actual_shares"][name]) for name in expected_partitions]
    if any(not math.isfinite(value) or value <= 0 for value in shares) or not math.isclose(
        sum(shares), 1.0, abs_tol=1e-9
    ):
        raise EvidenceContractError("approval partition shares must be finite and total 1")
    for field in (
        "max_holdout_mae_usd",
        "max_holdout_rmse_usd",
        "max_run_wall_s",
        "max_final_batch1_p95_ms",
        "max_bundle_bytes",
    ):
        if field in approval:
            try:
                value = float(approval[field])
            except (TypeError, ValueError) as error:
                raise EvidenceContractError(f"invalid {field}") from error
            if not math.isfinite(value) or value <= 0:
                raise EvidenceContractError(f"invalid {field}")
    return approval


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise EvidenceContractError("evidence is not finite JSON") from error
    path.write_text(text, encoding="utf-8")


def required_evidence_files() -> frozenset[str]:
    return frozenset(_REQUIRED_EVIDENCE_FILES)


def publish_staged_pack(
    workspace: Path | str,
    run_id: str,
    staging: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    validate_run_id(run_id)
    root = Path(workspace).resolve(strict=True)
    final = safe_workspace_path(root, Path("outputs/training_validation") / run_id)
    if final.exists():
        raise EvidenceContractError("immutable run namespace already exists")
    expected_staging = safe_workspace_path(
        root, Path("outputs/training_validation") / f".{run_id}.staging"
    )
    if not staging.exists() or staging.resolve(strict=True) != expected_staging.resolve(strict=True):
        raise EvidenceContractError("staging directory is missing or has an invalid identity")
    if staging.is_symlink() or not staging.is_dir():
        raise EvidenceContractError("staging path must be a real directory")
    paths = sorted(path for path in staging.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in paths):
        raise EvidenceContractError("staged evidence cannot contain symlinks")
    relatives = {path.relative_to(staging).as_posix() for path in paths}
    relatives.discard("manifest.json")
    missing = sorted(_REQUIRED_EVIDENCE_FILES - relatives)
    if missing:
        raise EvidenceContractError(f"staging is missing required files: {missing}")
    files = []
    for path in paths:
        relative = path.relative_to(staging).as_posix()
        if relative == "manifest.json":
            continue
        files.append(
            {
                "path": f"outputs/training_validation/{run_id}/{relative}",
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    reserved = {"schema_version", "run_id", "execution_status", "generated_at", "files"}
    if metadata and reserved & set(metadata):
        raise EvidenceContractError("publication metadata contains reserved manifest fields")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "execution_status": "complete",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "files": files,
        **(metadata or {}),
    }
    write_json(staging / "manifest.json", manifest)
    staging.rename(final)
    return final


def validate_complete_pack(workspace: Path | str, run_id: str) -> dict[str, Any]:
    validate_run_id(run_id)
    root = Path(workspace).resolve(strict=True)
    manifest_path = safe_workspace_path(
        root, Path("outputs/training_validation") / run_id / "manifest.json"
    )
    if not manifest_path.is_file():
        raise EvidenceContractError("complete manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceContractError("complete manifest is invalid JSON") from error
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceContractError("unsupported training-validation manifest schema")
    if manifest.get("run_id") != run_id or manifest.get("execution_status") != "complete":
        raise EvidenceContractError("manifest identity/status is not complete")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise EvidenceContractError("manifest files must be a list")
    prefix = f"outputs/training_validation/{run_id}/"
    by_relative: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not str(entry.get("path", "")).startswith(prefix):
            raise EvidenceContractError("manifest file escapes the immutable run namespace")
        relative = str(entry["path"])[len(prefix) :]
        if not relative or relative == "manifest.json" or ".." in Path(relative).parts:
            raise EvidenceContractError("manifest contains an invalid evidence path")
        if relative in by_relative:
            raise EvidenceContractError("manifest contains duplicate evidence paths")
        path = safe_workspace_path(root, entry["path"])
        if not path.is_file():
            raise EvidenceContractError(f"manifest file is missing: {relative}")
        if entry.get("sha256") != sha256_file(path):
            raise EvidenceContractError(f"manifest checksum mismatch: {relative}")
        if entry.get("size") != path.stat().st_size:
            raise EvidenceContractError(f"manifest size mismatch: {relative}")
        by_relative[relative] = entry
    missing = sorted(_REQUIRED_EVIDENCE_FILES - set(by_relative))
    if missing:
        raise EvidenceContractError(f"complete manifest is missing required files: {missing}")
    bundles = manifest.get("artifact_bundles", {})
    if bundles and set(bundles) != {"final_full", "final_top2"}:
        raise EvidenceContractError("manifest bundle roles are incomplete")
    for role, bundle in bundles.items():
        if not isinstance(bundle, dict):
            raise EvidenceContractError(f"invalid bundle metadata: {role}")
        path = safe_workspace_path(root, str(bundle.get("path", "")))
        metadata_path = safe_workspace_path(root, str(bundle.get("metadata_path", "")))
        if not path.is_file() or not metadata_path.is_file():
            raise EvidenceContractError(f"bundle or metadata is missing: {role}")
        if sha256_file(path) != bundle.get("sha256"):
            raise EvidenceContractError(f"bundle checksum mismatch: {role}")
        if path.stat().st_size != bundle.get("bytes"):
            raise EvidenceContractError(f"bundle size mismatch: {role}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EvidenceContractError(f"bundle metadata is invalid: {role}") from error
        if metadata != bundle:
            raise EvidenceContractError(f"bundle metadata mismatch: {role}")
    return manifest


def holdout_identity(row_fingerprints: list[str] | tuple[str, ...]) -> str:
    values = sorted(row_fingerprints)
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise EvidenceContractError("holdout identity requires unique nonempty row fingerprints")
    digest = hashlib.sha256("|".join(values).encode()).hexdigest()
    return f"holdout-{digest}"


def _access_path(workspace: Path | str, holdout_id: str) -> Path:
    if not re.fullmatch(r"holdout-[0-9a-f]{64}", holdout_id):
        raise EvidenceContractError("invalid holdout identity")
    return safe_workspace_path(
        workspace, Path("outputs/training_validation/holdout_access") / holdout_id / "access.json"
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".access-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def claim_holdout_access(
    workspace: Path | str, holdout_id: str, declaration_sha256: str, run_id: str
) -> dict[str, Any]:
    validate_run_id(run_id)
    if not re.fullmatch(r"[0-9a-f]{64}", declaration_sha256):
        raise EvidenceContractError("invalid evaluation declaration identity")
    path = _access_path(workspace, holdout_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not path.is_file():
            raise EvidenceContractError("holdout access directory is incomplete")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("state") == "complete":
            if existing.get("declaration_sha256") != declaration_sha256:
                raise EvidenceContractError(
                    "holdout was completed under a different frozen declaration"
                )
            return existing | {"reuse": True}
        if existing.get("state") == "started":
            raise EvidenceContractError("holdout evaluation is already started")
        raise EvidenceContractError("holdout was previously exposed by a failed evaluation")
    record = {
        "schema_version": SCHEMA_VERSION,
        "holdout_id": holdout_id,
        "declaration_sha256": declaration_sha256,
        "run_id": run_id,
        "state": "started",
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "reuse": False,
    }
    try:
        write_json(path, record)
    except BaseException:
        path.parent.rmdir()
        raise
    return record


def complete_holdout_access(
    workspace: Path | str,
    holdout_id: str,
    *,
    evidence_path: str,
    evidence_sha256: str,
) -> None:
    path = _access_path(workspace, holdout_id)
    if not path.is_file():
        raise EvidenceContractError("holdout access was not claimed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("state") != "started":
        raise EvidenceContractError("only a started holdout access can complete")
    if not evidence_path.startswith("outputs/training_validation/") or ".." in Path(
        evidence_path
    ).parts:
        raise EvidenceContractError("invalid holdout evidence path")
    if not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256):
        raise EvidenceContractError("invalid holdout evidence checksum")
    record.update(
        {
            "state": "complete",
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_sha256,
        }
    )
    _atomic_json(path, record)


def fail_holdout_access(workspace: Path | str, holdout_id: str, *, reason_code: str) -> None:
    path = _access_path(workspace, holdout_id)
    if not path.is_file():
        raise EvidenceContractError("holdout access was not claimed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("state") != "started":
        raise EvidenceContractError("only a started holdout access can fail")
    record.update(
        {
            "state": "failed",
            "failed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "reason_code": reason_code,
        }
    )
    _atomic_json(path, record)


def _redact(value: Any, key: str | None = None) -> Any:
    if key and key.lower() in _REDACTED_FIELDS:
        return "[REDACTED]"
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(item) for item in value]
    return str(value)


class TrainingEventWriter:
    """Small append-only JSONL event writer for one CLI invocation."""

    def __init__(
        self, path: Path, run_id: str, *, invocation_id: str | None = None, max_details: int = 16_384
    ) -> None:
        validate_run_id(run_id)
        self.path = path
        self.run_id = run_id
        self.invocation_id = invocation_id or f"inv-{uuid.uuid4().hex}"
        self.max_details = max_details
        self.sequence = 0
        self._open: list[str] = []
        self._handle = None

    def __enter__(self) -> "TrainingEventWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            for operation_id in reversed(self._open):
                self._emit(
                    operation_id=operation_id,
                    event_type="operation_failed",
                    status="failed",
                    level="error",
                    reason_code="UNHANDLED_RUNTIME_ERROR",
                    message="Operation failed with an unhandled runtime error.",
                )
            self._open.clear()
        if self._handle is not None:
            self._handle.close()
        return False

    def start(self, event_type: str, *, message: str, planned_units: int = 0) -> str:
        operation_id = f"op-{uuid.uuid4().hex}"
        self._open.append(operation_id)
        self._emit(
            operation_id=operation_id,
            event_type=f"{event_type}_started",
            status="started",
            level="info",
            reason_code="OPERATION_STARTED",
            message=message,
            progress={
                "planned_units": int(planned_units),
                "completed_units": 0,
                "failed_units": 0,
                "skipped_units": 0,
                "reused_units": 0,
            },
        )
        return operation_id

    def complete(
        self,
        operation_id: str,
        *,
        message: str,
        metrics: list[dict[str, Any]] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if operation_id not in self._open:
            raise EvidenceContractError("cannot complete unknown operation")
        self._open.remove(operation_id)
        self._emit(
            operation_id=operation_id,
            event_type="operation_completed",
            status="completed",
            level="info",
            reason_code="OPERATION_COMPLETED",
            message=message,
            metrics=metrics or [],
            details=details or {},
            progress={
                "planned_units": 1,
                "completed_units": 1,
                "failed_units": 0,
                "skipped_units": 0,
                "reused_units": 0,
            },
        )

    def _emit(self, **payload: Any) -> None:
        if self._handle is None:
            raise EvidenceContractError("event writer is not open")
        metrics = payload.get("metrics", [])
        if len(metrics) > 100:
            raise EvidenceContractError("event contains more than 100 inline metrics")
        details = _redact(payload.get("details", {}))
        if len(json.dumps(details, allow_nan=False).encode("utf-8")) > self.max_details:
            raise EvidenceContractError("event details exceed 16 KiB")
        self.sequence += 1
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "invocation_id": self.invocation_id,
            "event_id": f"{self.invocation_id}:{self.sequence}",
            "sequence": self.sequence,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "operation_id": payload["operation_id"],
            "parent_operation_id": payload.get("parent_operation_id"),
            "event_type": payload["event_type"],
            "level": payload["level"],
            "status": payload["status"],
            "reason_code": payload["reason_code"],
            "message": payload["message"],
            "progress": payload.get("progress"),
            "metrics": _redact(metrics),
            "details": details,
            "evidence_refs": payload.get("evidence_refs", []),
            "next_action": payload.get("next_action"),
        }
        self._handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
