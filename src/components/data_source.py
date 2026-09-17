from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd

from ai_job_market.core import run_pipeline, validate_input_schema


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


def discover_runs(project_root: Path) -> list[tuple[str, Path]]:
    rows = []
    runs_dir = project_root / "runs"
    if not runs_dir.exists():
        return rows
    for p in sorted(runs_dir.iterdir(), reverse=True):
        summary = p / "outputs" / "08_full_pipeline" / "run_summary.json"
        if summary.exists():
            try:
                obj = json.loads(summary.read_text(encoding="utf-8"))
                label = f"{p.name} · {obj.get('raw_shape', ['?'])[0]} rows · {obj.get('best_model', '?')}"
            except Exception:
                label = p.name
            rows.append((label, p))
    return rows


def render_data_source_control(st, project_root: Path, role: str) -> Path:
    """Render upload / schema / full-process controls and return active evidence root."""
    if "analysis_root" not in st.session_state:
        st.session_state.analysis_root = str(project_root)
    if "active_run_label" not in st.session_state:
        st.session_state.active_run_label = "Baseline release"

    st.markdown("### Data source")
    existing = discover_runs(project_root)
    options = ["Baseline release"] + [label for label, _ in existing]
    current = st.session_state.get("active_run_label", "Baseline release")
    idx = options.index(current) if current in options else 0
    chosen = st.selectbox("Evidence workspace", options, index=idx, key="workspace_selector")
    if chosen == "Baseline release":
        st.session_state.analysis_root = str(project_root)
        st.session_state.active_run_label = chosen
    else:
        run_path = dict(existing)[chosen]
        st.session_state.analysis_root = str(run_path)
        st.session_state.active_run_label = chosen

    active_root = Path(st.session_state.analysis_root)
    if role != "admin":
        st.caption(
            "Standard-user mode uses the active validated workspace selected by the administrator."
        )
        return active_root

    with st.expander("Upload & process a new dataset", expanded=False):
        st.caption(
            "Upload a raw CSV → validate the 25-column source contract → run the full Common + Branch A + Branch B workflow in an isolated workspace."
        )
        upload = st.file_uploader(
            "Raw AI job-market CSV", type=["csv"], key="global_dataset_upload"
        )
        if upload is None:
            st.info("No file selected. The current workspace remains unchanged.")
            return active_root
        data = upload.getvalue()
        try:
            df = pd.read_csv(BytesIO(data))
        except Exception as exc:
            st.error(f"CSV parsing failed: {exc}")
            return active_root

        report = validate_input_schema(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{report['rows']:,}")
        c2.metric("Columns", report["columns"])
        c3.metric("Schema gate", "PASS" if report["valid"] else "FAIL")

        issues = pd.DataFrame(report["issues"])
        if report["valid"]:
            st.success("Schema gate passed. The file can run through the full offline workflow.")
        else:
            st.error("Schema gate failed. Fix all ERROR items before processing.")
        if not issues.empty:
            st.dataframe(issues, use_container_width=True, hide_index=True)
        with st.expander("Expected vs detected columns", expanded=not report["valid"]):
            st.dataframe(
                report["field_table"], use_container_width=True, hide_index=True, height=430
            )

        process = st.button(
            "Process full pipeline on this dataset",
            type="primary",
            use_container_width=True,
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
            with st.spinner(
                "Running full data-quality, segmentation and salary-prediction workflow…"
            ):
                try:
                    summary = run_pipeline(raw_path, root=project_root, workspace_root=workspace)
                except Exception as exc:
                    st.exception(exc)
                    st.error("Processing stopped. The baseline workspace has not been overwritten.")
                    return active_root
            st.session_state.analysis_root = str(workspace)
            st.session_state.active_run_label = (
                f"{run_id} · {summary['raw_shape'][0]} rows · {summary['best_model']}"
            )
            st.session_state.pop("prediction_queue", None)
            st.session_state.pop("last_prediction_batch", None)
            st.success(f"Full pipeline completed. Active run: {run_id}")
            st.rerun()

    return Path(st.session_state.analysis_root)
