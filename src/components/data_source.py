from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd

from ai_job_market.core import run_pipeline, validate_input_schema
from components.language import get_translator


def _sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def _copy_runtime_support(project_root: Path, workspace: Path) -> None:
    """Copy only presentation assets needed when a run workspace becomes active."""
    (workspace / "data" / "raw").mkdir(parents=True, exist_ok=True)
    src_pipes = project_root / "docs" / "pipelines"
    dst_pipes = workspace / "docs" / "pipelines"
    if src_pipes.exists():
        dst_pipes.mkdir(parents=True, exist_ok=True)
        for p in src_pipes.glob("*.png"):
            shutil.copy2(p, dst_pipes / p.name)


def discover_runs(project_root: Path, i18n) -> list[tuple[str, Path]]:
    rows = []
    runs_dir = project_root / "runs"
    if not runs_dir.exists():
        return rows
    for p in sorted(runs_dir.iterdir(), reverse=True):
        summary = p / "outputs" / "08_full_pipeline" / "run_summary.json"
        if summary.exists():
            try:
                obj = json.loads(summary.read_text(encoding="utf-8"))
                label = i18n.text(
                    "source.run_label", run_id=p.name,
                    rows=obj.get("raw_shape", ["?"])[0], model=obj.get("best_model", "?"),
                )
            except Exception:
                label = p.name
            rows.append((label, p))
    return rows


def render_data_source_control(st, project_root: Path, role: str) -> Path:
    """Render upload / schema / full-process controls and return active evidence root."""
    i18n = get_translator(st)
    t = i18n.text
    st.session_state.setdefault("analysis_root", str(project_root))
    st.markdown(t("source.title"))
    existing = discover_runs(project_root, i18n)
    # Stable path identities, not localized labels, determine the active workspace.
    labels = {str(project_root): t("source.baseline")}
    labels.update({str(path): label for label, path in existing})
    current = st.session_state.analysis_root
    pending = st.session_state.pop("pending_workspace", None)
    if pending in labels:
        st.session_state.workspace_selector = pending
    if st.session_state.get("workspace_selector") not in labels:
        st.session_state.workspace_selector = current if current in labels else str(project_root)
    chosen = st.selectbox(
        t("source.workspace"), list(labels), format_func=labels.__getitem__,
        key="workspace_selector",
    )
    if chosen not in labels:
        chosen = str(project_root)
    st.session_state.analysis_root = chosen
    st.session_state.active_run_label = labels[chosen]

    active_root = Path(st.session_state.analysis_root)
    if role != "admin":
        st.caption(t("source.standard_user"))
        return active_root

    with st.expander(t("source.upload_title"), expanded=False):
        st.caption(t("source.upload_caption"))
        upload = st.file_uploader(
            t("source.upload_label"), type=["csv"], key="global_dataset_upload"
        )
        if upload is None:
            st.info(t("source.no_file"))
            return active_root
        data = upload.getvalue()
        try:
            df = pd.read_csv(BytesIO(data))
        except Exception as exc:
            st.error(t("source.parse_failed", error=str(exc)))
            return active_root

        report = validate_input_schema(df)
        c1, c2, c3 = st.columns(3)
        c1.metric(t("source.rows"), f"{report['rows']:,}")
        c2.metric(t("source.columns"), report["columns"])
        c3.metric(t("source.schema_gate"), t("status.pass" if report["valid"] else "status.fail"))

        issues = pd.DataFrame(report["issues"])
        if report["valid"]:
            st.success(t("source.schema_passed"))
        else:
            st.error(t("source.schema_failed"))
        if not issues.empty:
            st.dataframe(i18n.frame(issues), width="stretch", hide_index=True)
        with st.expander(t("source.expected_columns"), expanded=not report["valid"]):
            st.dataframe(i18n.frame(report["field_table"]), width="stretch", hide_index=True, height=430)

        process = st.button(
            t("source.process"),
            type="primary",
            width="stretch",
            disabled=not report["valid"],
            key="process_uploaded_dataset",
        )
        if process:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{stamp}_{_sha8(data)}"
            workspace = project_root / "runs" / run_id
            _copy_runtime_support(project_root, workspace)
            raw_path = workspace / "data" / "raw" / "ai_jobs_market_2025_2026.csv"
            raw_path.write_bytes(data)
            with st.spinner(t("source.running")):
                try:
                    summary = run_pipeline(raw_path, root=project_root, workspace_root=workspace)
                except Exception as exc:
                    st.exception(exc)
                    st.error(t("source.stopped"))
                    return active_root
            st.session_state.analysis_root = str(workspace)
            st.session_state.active_run_label = t(
                "source.run_label", run_id=run_id,
                rows=summary["raw_shape"][0], model=summary["best_model"],
            )
            st.session_state.pending_workspace = str(workspace)
            st.session_state.pop("prediction_queue", None)
            st.session_state.pop("last_prediction_batch", None)
            st.success(t("source.completed", run_id=run_id))
            st.rerun()

    return Path(st.session_state.analysis_root)
