from __future__ import annotations

import contextvars
import csv
import hashlib
import json
import math
import re
import sys
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from .training_console import ConsoleRenderer, event_detail_level, write_block

_ACTIVE_AUDIT: contextvars.ContextVar["TrainingAuditSession | None"] = contextvars.ContextVar(
    "training_audit_session", default=None
)
_SAFE_STATUSES = {"started", "completed", "skipped", "failed", "cancelled"}
_EXPECTED_STAGES = {"C1", "C2", "C3", "C4", "C5", "B1-B3", "A1-A8", "B4", "B5-B6", "B7", "I1", "F"}
_PROTECTED_FIELDS = {
    "schema_version",
    "timestamp",
    "audit_run_id",
    "pipeline_run_id",
    "sequence",
    "level",
    "event",
    "message",
    "step_id",
    "operation",
    "status",
    "detail_level",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _safe_log_dir(workspace_root: Path) -> Path:
    workspace = workspace_root.resolve()
    log_dir = workspace / "outputs" / "08_full_pipeline" / "logs"
    current = log_dir
    existing_parts: list[Path] = []
    while current != workspace.parent:
        if current.exists():
            existing_parts.append(current)
        current = current.parent
    for part in existing_parts:
        if part.is_symlink():
            raise ValueError(f"Unsafe log directory: {part} is a symlink")
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved = log_dir.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Unsafe log directory escapes workspace: {resolved}")
    return resolved


def _safe_evidence_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return name or "evidence"


def dataframe_identity(frame: Any) -> str:
    """Return a content-and-order identity without changing the frame."""
    try:
        hashed = frame.__class__.__module__.startswith("pandas")
    except AttributeError:
        hashed = False
    if hashed:
        import pandas as pd

        values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        columns = "|".join(map(str, frame.columns)).encode("utf-8")
        return hashlib.sha256(columns + b"\0" + values).hexdigest()
    return hashlib.sha256(repr(frame).encode("utf-8")).hexdigest()


@dataclass
class TrainingAuditSession:
    workspace_root: Path
    raw_path: Path
    seed: int
    target: str
    features: list[str]
    stream: TextIO | None = None
    debuglog: bool = False
    on_event: Callable[[dict[str, Any]], None] | None = None
    audit_run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root)
        self.raw_path = Path(self.raw_path)
        self.stream = self.stream or sys.stdout
        self.sequence = 0
        self.log_complete = True
        self.console_complete = True
        self.coverage_complete = True
        self.training_status = "started"
        self.pipeline_run_id: str | None = None
        self.recorded_split_ids: set[str] = set()
        self.exports: list[dict[str, Any]] = []
        self._token: contextvars.Token | None = None
        self._fh: TextIO | None = None
        self._active_stage: tuple[str, str] | None = None
        self._completed_stages: set[str] = set()
        self.renderer = ConsoleRenderer(debuglog=self.debuglog)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir = _safe_log_dir(self.workspace_root)
        stem = f"training-{timestamp}-{self.audit_run_id}"
        self.log_path = log_dir / f"{stem}.logs"
        self.evidence_dir = log_dir / stem
        try:
            self._fh = self.log_path.open("x", encoding="utf-8")
        except OSError as exc:
            self.log_complete = False
            self._fh = None
            self._console_warning(
                f"Structured training log initialization failed: {type(exc).__name__}: {exc}"
            )
        try:
            self.evidence_dir.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            self.log_complete = False
            self._console_warning(
                f"Training evidence directory initialization failed: {type(exc).__name__}: {exc}"
            )

    def __enter__(self) -> "TrainingAuditSession":
        self._token = _ACTIVE_AUDIT.set(self)
        self.event(
            "run_started",
            step_id="run.start",
            operation="run",
            status="started",
            message="Training audit run started.",
            detail_level=1,
            extra={
                "raw_path": str(self.raw_path),
                "raw_sha256": _sha256_file(self.raw_path),
                "seed": self.seed,
                "target": self.target,
                "features": list(self.features),
                "log_path": str(self.log_path),
                "evidence_directory": str(self.evidence_dir),
                "debuglog": self.debuglog,
                "log_complete": self.log_complete,
            },
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._active_stage is not None:
            stage_id, title = self._active_stage
            if exc_type is KeyboardInterrupt:
                self._end_stage(stage_id, title, "cancelled", exc_type)
            elif exc_type is not None:
                self._end_stage(stage_id, title, "failed", exc_type)
            else:
                self.coverage_complete = False
                self._warning_event(
                    "Stage observation remained open; completion was not assumed.",
                    extra={"stage_id": stage_id},
                )
                self._active_stage = None
        if not _EXPECTED_STAGES.issubset(self._completed_stages):
            self.coverage_complete = False
        if exc_type is None:
            self.training_status = "completed"
            end_event, status, message = (
                "run_completed",
                "completed",
                "Training audit run completed.",
            )
        elif exc_type is KeyboardInterrupt:
            self.training_status = "cancelled"
            end_event, status, message = (
                "run_cancelled",
                "cancelled",
                "Training audit run cancelled.",
            )
        else:
            self.training_status = "failed"
            end_event, status, message = "run_failed", "failed", "Training audit run failed."
        self.event(
            end_event,
            step_id="run.end",
            operation="run",
            status=status,
            message=message,
            detail_level=1,
            extra={
                "error_type": exc_type.__name__ if exc_type else None,
                "log_complete": self.log_complete,
                "console_complete": self.console_complete,
                "coverage_complete": self.coverage_complete,
                "log_path": str(self.log_path),
            },
        )
        if self._token is not None:
            _ACTIVE_AUDIT.reset(self._token)
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError as close_error:
                self.log_complete = False
                self._console_warning(
                    f"Training evidence close failed: {type(close_error).__name__}: {close_error}"
                )
        self._write_manifest()
        return False

    def _write_manifest(self) -> None:
        if not self.evidence_dir.is_dir():
            self.log_complete = False
            return
        manifest = {
            "schema_version": 1,
            "audit_run_id": self.audit_run_id,
            "pipeline_run_id": self.pipeline_run_id,
            "training_status": self.training_status,
            "source_path": str(self.raw_path),
            "source_sha256": _sha256_file(self.raw_path),
            "seed": self.seed,
            "target": self.target,
            "log_complete": self.log_complete,
            "console_complete": self.console_complete,
            "coverage_complete": self.coverage_complete,
            "exports": self.exports,
        }
        try:
            (self.evidence_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as error:
            self.log_complete = False
            self._console_warning(
                f"Training evidence manifest failed: {type(error).__name__}: {error}"
            )

    def should_emit_split_definition(self, split_id: str) -> bool:
        if split_id in self.recorded_split_ids:
            return False
        self.recorded_split_ids.add(split_id)
        return True

    def set_pipeline_run_id(self, pipeline_run_id: str) -> None:
        self.pipeline_run_id = pipeline_run_id
        self.event(
            "run_context",
            step_id="run.context",
            operation="run",
            status="completed",
            message="Pipeline run context recorded.",
            detail_level=2,
            extra={"pipeline_run_id": pipeline_run_id},
        )

    def start_stage(self, stage_id: str, title: str) -> None:
        if self._active_stage is not None:
            raise ValueError(f"Stage {self._active_stage[0]} is already active")
        self._active_stage = (stage_id, title)
        self.event(
            "stage_started",
            step_id=f"stage.{stage_id}.start",
            operation="pipeline_stage",
            status="started",
            message=title,
            detail_level=1,
            extra={"stage_id": stage_id},
        )

    def complete_stage(self, stage_id: str, title: str, *, extra: dict[str, Any] | None = None) -> None:
        if self._active_stage is None or self._active_stage[0] != stage_id:
            raise ValueError(f"Stage {stage_id} is not active")
        self._active_stage = None
        self._completed_stages.add(stage_id)
        self.event(
            "stage_completed",
            step_id=f"stage.{stage_id}.end",
            operation="pipeline_stage",
            status="completed",
            message=title,
            detail_level=1,
            extra={"stage_id": stage_id, **(extra or {})},
        )

    def _end_stage(self, stage_id: str, title: str, status: str, exc_type: type) -> None:
        self._active_stage = None
        self.event(
            f"stage_{status}",
            step_id=f"stage.{stage_id}.end",
            operation="pipeline_stage",
            status=status,
            message=title,
            detail_level=1,
            extra={"stage_id": stage_id, "error_type": exc_type.__name__},
        )

    def event(
        self,
        event: str,
        *,
        step_id: str,
        operation: str,
        status: str,
        message: str,
        detail_level: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if status not in _SAFE_STATUSES:
            raise ValueError(f"Unsupported audit status: {status}")
        self.sequence += 1
        safe_extra = {k: _json_safe(v) for k, v in (extra or {}).items() if k not in _PROTECTED_FIELDS}
        payload: dict[str, Any] = {
            **safe_extra,
            "schema_version": 1,
            "timestamp": _utc_now(),
            "audit_run_id": self.audit_run_id,
            "pipeline_run_id": self.pipeline_run_id,
            "sequence": self.sequence,
            "level": "ERROR" if status == "failed" else "INFO",
            "event": event,
            "message": message,
            "step_id": step_id,
            "operation": operation,
            "status": status,
        }
        payload["detail_level"] = detail_level or event_detail_level(payload)
        try:
            line = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            self.log_complete = False
            self._warning_event(
                f"Training event serialization failed: {type(error).__name__}",
                extra={"failed_event": event},
            )
            return

        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except (OSError, ValueError) as error:
                self.log_complete = False
                self._fh = None
                self._console_warning(
                    f"Training evidence write failed: {type(error).__name__}: {error}"
                )
        else:
            self.log_complete = False

        try:
            rendered = self.renderer.render(payload)
            if rendered is not None:
                write_block(self.stream, rendered)
        except (OSError, ValueError) as error:
            self.console_complete = False
            if self._fh is not None:
                self._write_warning_line(
                    f"Training console rendering failed: {type(error).__name__}: {error}"
                )

        if self.on_event is not None:
            try:
                self.on_event(deepcopy(payload))
            except Exception as error:  # observer is explicitly non-scientific
                self._warning_event(
                    f"Training observer failed: {type(error).__name__}: {error}",
                    extra={"failed_event": event},
                )

    def _write_warning_line(self, message: str, extra: dict[str, Any] | None = None) -> None:
        self.sequence += 1
        payload = {
            **(extra or {}),
            "schema_version": 1,
            "timestamp": _utc_now(),
            "audit_run_id": self.audit_run_id,
            "pipeline_run_id": self.pipeline_run_id,
            "sequence": self.sequence,
            "level": "WARNING",
            "event": "logging_degraded",
            "message": message,
            "step_id": "logging.degraded",
            "operation": "logging",
            "status": "failed",
            "detail_level": 2,
        }
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
                self._fh.flush()
            except (OSError, ValueError):
                self.log_complete = False

    def _warning_event(self, message: str, extra: dict[str, Any] | None = None) -> None:
        self._write_warning_line(message, extra)
        self._console_warning(message)

    def _console_warning(self, message: str) -> None:
        try:
            write_block(self.stream, f"  [WARNING] {message}")
        except (OSError, ValueError):
            self.console_complete = False

    def export_table(
        self,
        evidence_id: str,
        rows: Iterable[dict[str, Any]] | Any,
        *,
        kind: str,
        columns: list[str] | None = None,
        detail_level: int = 4,
    ) -> dict[str, Any]:
        safe_id = _safe_evidence_name(evidence_id)
        path = self.evidence_dir / f"{safe_id}.csv"
        if hasattr(rows, "to_dict"):
            records = rows.to_dict("records")
        else:
            records = list(rows)
        normalized = [{str(k): _json_safe(v) for k, v in row.items()} for row in records]
        fieldnames = list(columns or (normalized[0].keys() if normalized else []))
        shown = min(20, len(normalized))
        result: dict[str, Any] = {
            "evidence_id": safe_id,
            "kind": kind,
            "path": path.name,
            "rows": len(normalized),
            "columns": fieldnames,
            "shown_rows": shown,
            "omitted_rows": max(0, len(normalized) - shown),
        }
        try:
            self.evidence_dir.mkdir(parents=False, exist_ok=True)
            with path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(normalized)
            result.update({"status": "complete", "sha256": _sha256_file(path)})
            self.exports.append({k: result[k] for k in ["evidence_id", "kind", "path", "status", "rows", "columns", "sha256"]})
            self.event(
                "evidence_exported",
                step_id=f"evidence.{safe_id}",
                operation="write_evidence_csv",
                status="completed",
                message="Complete evidence table exported.",
                detail_level=detail_level,
                extra={**result, "evidence_path": str(path)},
            )
        except (OSError, ValueError, TypeError) as error:
            self.log_complete = False
            result.update({"status": "failed", "sha256": None, "reason": type(error).__name__})
            self.exports.append({k: result.get(k) for k in ["evidence_id", "kind", "path", "status", "rows", "columns", "sha256", "reason"]})
            self.event(
                "evidence_export_failed",
                step_id=f"evidence.{safe_id}",
                operation="write_evidence_csv",
                status="failed",
                message=f"Evidence table export failed: {type(error).__name__}.",
                detail_level=2,
                extra=result,
            )
        return result


def start_training_audit(
    *,
    workspace_root: Path,
    raw_path: Path,
    seed: int,
    target: str,
    features: list[str],
    stream: TextIO | None = None,
    debuglog: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingAuditSession:
    return TrainingAuditSession(
        workspace_root=workspace_root,
        raw_path=raw_path,
        seed=seed,
        target=target,
        features=features,
        stream=stream,
        debuglog=debuglog,
        on_event=on_event,
    )


def active_audit() -> TrainingAuditSession | None:
    return _ACTIVE_AUDIT.get()


def emit_event(
    event: str,
    *,
    step_id: str,
    operation: str,
    status: str,
    message: str,
    detail_level: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    audit = active_audit()
    if audit is not None:
        audit.event(
            event,
            step_id=step_id,
            operation=operation,
            status=status,
            message=message,
            detail_level=detail_level,
            extra=extra,
        )


def export_evidence_table(
    evidence_id: str,
    rows: Iterable[dict[str, Any]] | Any,
    *,
    kind: str,
    columns: list[str] | None = None,
    detail_level: int = 4,
) -> dict[str, Any] | None:
    audit = active_audit()
    if audit is None:
        return None
    return audit.export_table(
        evidence_id,
        rows,
        kind=kind,
        columns=columns,
        detail_level=detail_level,
    )
