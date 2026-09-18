from __future__ import annotations

import math
import shutil
import textwrap
import threading
from typing import Any, TextIO

_OUTPUT_LOCK = threading.RLock()

_DEFAULT_DEPTHS = {
    "run_started": 1,
    "run_context": 2,
    "run_completed": 1,
    "run_failed": 1,
    "run_cancelled": 1,
    "stage_started": 1,
    "stage_completed": 1,
    "stage_failed": 1,
    "stage_cancelled": 1,
    "stage_skipped": 1,
    "data_summary": 2,
    "preprocessing_summary": 2,
    "evaluation_started": 1,
    "evaluation_completed": 2,
    "candidate_completed": 2,
    "selection_recorded": 2,
    "tuning_skipped": 2,
    "evidence_exported": 2,
    "evidence_export_failed": 2,
    "trial_started": 2,
    "trial_completed": 2,
    "trial_failed": 2,
    "candidate_fold_scored": 3,
    "operation_started": 3,
    "operation_completed": 3,
    "split_defined": 4,
    "split_used": 4,
    "artifact_write_started": 4,
    "artifact_written": 4,
    "logging_degraded": 2,
}


def event_detail_level(event: dict[str, Any]) -> int:
    value = event.get("detail_level")
    if isinstance(value, int) and 1 <= value <= 4:
        return value
    return _DEFAULT_DEPTHS.get(str(event.get("event", "")), 2)


def write_block(stream: TextIO, text: str) -> None:
    """Write one terminal block under a shared lock to avoid heartbeat interleaving."""
    if not text:
        return
    with _OUTPUT_LOCK:
        stream.write(text.rstrip("\n") + "\n")
        stream.flush()


def _number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not math.isfinite(number):
        return "unavailable"
    return f"{number:,.{digits}f}"


def _readable_reason(value: Any) -> str:
    return str(value or "reason unavailable").replace("_", " ").replace(";", "; ")


class ConsoleRenderer:
    """Render allowlisted event fields; structured payloads remain file-only."""

    def __init__(self, *, debuglog: bool = False, width: int | None = None) -> None:
        self.debuglog = bool(debuglog)
        terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
        self.width = max(50, int(width or terminal_width))

    def render(self, event: dict[str, Any]) -> str | None:
        depth = event_detail_level(event)
        severity = str(event.get("level", "INFO")).upper()
        if depth > 2 and not self.debuglog and severity not in {"WARNING", "ERROR"}:
            return None

        indent = "  " * (depth - 1)
        body = self._body(event)
        prefix = self._prefix(event, severity)
        line = f"{indent}{prefix} {body}".rstrip()
        return textwrap.fill(
            line,
            width=self.width,
            subsequent_indent=indent + "  ",
            break_long_words=False,
            break_on_hyphens=False,
        )

    def _prefix(self, event: dict[str, Any], severity: str) -> str:
        if severity in {"WARNING", "ERROR"}:
            return f"[{severity}]"
        status = str(event.get("status", "completed"))
        return {
            "started": "[RUNNING]",
            "completed": "[COMPLETED]",
            "skipped": "[SKIPPED]",
            "failed": "[FAILED]",
            "cancelled": "[CANCELLED]",
        }.get(status, "[INFO]")

    def _body(self, event: dict[str, Any]) -> str:
        name = str(event.get("event", ""))
        if name == "evaluation_completed":
            model = event.get("model_name", "Model")
            metrics = (
                f"MAE={_number(event.get('MAE_mean'))} USD | "
                f"RMSE={_number(event.get('RMSE_mean'))} USD | "
                f"R²={_number(event.get('R2_mean'))} (unitless) | "
                f"MedAE={_number(event.get('MedAE_mean'))} USD"
            )
            assessment = str(event.get("fit_assessment", "insufficient_evidence")).replace(
                "_", " "
            )
            reason = _readable_reason(event.get("fit_reason"))
            return f"{model} CV: {metrics}. Fit assessment: {assessment} — {reason}."
        if name == "candidate_fold_scored":
            return (
                f"{event.get('model_name', 'Model')} fold {event.get('fold_id', '?')}: "
                f"MAE={_number(event.get('MAE'))} USD | RMSE={_number(event.get('RMSE'))} USD | "
                f"R²={_number(event.get('R2'))} (unitless) | "
                f"MedAE={_number(event.get('MedAE'))} USD; "
                f"validation={event.get('validation_period', 'unavailable')}."
            )
        if name in {"split_defined", "split_used"}:
            return (
                f"Temporal split {str(event.get('split_id', 'unavailable'))[:12]}: "
                f"requested={event.get('requested_folds', '?')}, "
                f"effective={event.get('effective_folds', '?')}. "
                f"{event.get('message', '')}"
            ).strip()
        if name == "evidence_exported":
            return (
                f"Evidence {event.get('evidence_id', 'table')}: {event.get('rows', 0)} rows; "
                f"showing {event.get('shown_rows', 0)}, omitted {event.get('omitted_rows', 0)}; "
                f"file={event.get('evidence_path', 'unavailable')}."
            )
        stage = event.get("stage_id")
        message = str(event.get("message") or event.get("operation") or name or "Training event")
        if stage:
            return f"{stage} — {message}"
        return message
