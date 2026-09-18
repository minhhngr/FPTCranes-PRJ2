from __future__ import annotations

"""
Terminal-first runner for the AI Job Market project.

Purpose
-------
Run the existing offline pipeline WITHOUT changing its modelling logic, while
showing in the terminal:

* which stage is currently running;
* the most important result from each completed stage;
* stage elapsed time;
* periodic heartbeat messages during long stages;
* total run time and final model / segmentation summary.

This file is intentionally a thin orchestration layer over
``src.ai_job_market.core.run_pipeline``.  It does not refit anything outside
the existing pipeline and it does not change the persisted modelling contract.

Run:
    python pipeline.py

Optional:
    python pipeline.py --data data/raw/ai_jobs_market_2025_2026.csv
    python pipeline.py --heartbeat 15
    python pipeline.py --no-heartbeat
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from src.ai_job_market import core
from src.ai_job_market.training_console import write_block

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "ai_jobs_market_2025_2026.csv"


# ---------------------------------------------------------------------------
# Small terminal helpers
# ---------------------------------------------------------------------------


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return os.name != "nt" or bool(
        os.getenv("WT_SESSION")
        or os.getenv("ANSICON")
        or os.getenv("ConEmuANSI")
        or os.getenv("TERM_PROGRAM")
    )


_USE_COLOR = _supports_color()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _blue(text: str) -> str:
    return _c(text, "96")


def _green(text: str) -> str:
    return _c(text, "92")


def _yellow(text: str) -> str:
    return _c(text, "93")


def _red(text: str) -> str:
    return _c(text, "91")


def _bold(text: str) -> str:
    return _c(text, "1")


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:,.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes):02d}:{sec:04.1f}"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{sec:04.1f}"


def _hr(char: str = "─", width: int = 92) -> str:
    return char * width


def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _num(x, digits: int = 3, default: str = "—") -> str:
    try:
        if pd.isna(x):
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default


def _money(x, default: str = "—") -> str:
    try:
        if pd.isna(x):
            return default
        return f"${float(x):,.0f}"
    except Exception:
        return default


def _pct(x, digits: int = 1, default: str = "—") -> str:
    try:
        if pd.isna(x):
            return default
        value = float(x)
        # Accept either 0..1 or 0..100 input.
        if abs(value) <= 1.000001:
            value *= 100.0
        return f"{value:.{digits}f}%"
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Stage summaries
# ---------------------------------------------------------------------------


def _summary_ingestion(ctx: RunContext) -> str:
    df = ctx.raw_preview
    if df is None or df.empty:
        return "Raw CSV loaded."
    period = ""
    if {"posting_year", "posting_month"}.issubset(df.columns):
        ym = pd.to_numeric(df["posting_year"], errors="coerce") * 100 + pd.to_numeric(
            df["posting_month"], errors="coerce"
        )
        if ym.notna().any():
            lo, hi = int(ym.min()), int(ym.max())
            period = f" | Period {lo // 100:04d}-{lo % 100:02d} → {hi // 100:04d}-{hi % 100:02d}"
    return f"Raw data: {len(df):,} rows × {df.shape[1]} columns{period}"


def _summary_basic_clean(ctx: RunContext) -> str:
    a = _read_json(ctx.root / "outputs/01_data_basic_clean/basic_clean_audit.json")
    if not a:
        return "Basic-clean artifacts written."
    return (
        f"Clean: {a.get('clean_rows', '—'):,} rows × {a.get('clean_columns', '—')} columns"
        if isinstance(a.get("clean_rows"), int)
        else "Basic-clean artifacts written."
    ) + (
        f" | invalid category removed={a.get('invalid_category_rows_removed', '—')}"
        f" | duplicates removed={a.get('duplicate_rows_removed', '—')}"
        f" | job_id removed={a.get('identifier_removed', '—')}"
    )


def _summary_quality(ctx: RunContext) -> str:
    q = _read_csv(ctx.root / "outputs/01_data_basic_clean/contradiction_summary.csv")
    if q.empty:
        return "Data-quality and contradiction evidence written."
    row = q.sort_values("affected_pct", ascending=False).iloc[0]
    return (
        f"Top issue: {row.get('issue', '—')} | affected={_pct(row.get('affected_pct'))}"
        f" ({int(row.get('affected_rows', 0)):,} rows)"
    )


def _summary_feature_governance(ctx: RunContext) -> str:
    p = _read_csv(ctx.root / "outputs/02_data_ready_for_ml/feature_policy.csv")
    if p.empty or "policy" not in p.columns:
        return "Feature-governance policy persisted."
    counts = p["policy"].astype(str).value_counts()
    allow = int(counts.get("ALLOW", 0))
    block = int(counts.get("BLOCK", 0))
    target = int(counts.get("TARGET", 0))
    return f"Feature policy: ALLOW={allow} | BLOCK={block} | TARGET={target}"


def _summary_shared_base(ctx: RunContext) -> str:
    p = ctx.root / "outputs/02_data_ready_for_ml/shared_prepared_feature_base.csv"
    d = _read_csv(p)
    if d.empty:
        return "Shared prepared feature base persisted."
    skill_msg = ""
    if "skill_count" in d.columns:
        skill_msg = f" | mean skill_count={d['skill_count'].mean():.1f}"
    return f"Prepared base: {len(d):,} rows × {d.shape[1]} columns{skill_msg}"


def _summary_b123(ctx: RunContext) -> str:
    split = _read_json(ctx.root / "outputs/02_data_ready_for_ml/temporal_split_summary.json")
    prep = _read_json(ctx.root / "outputs/02_data_ready_for_ml/preprocessing_contract.json")
    dev = split.get("development_rows", "—")
    test = split.get("locked_test_rows", "—")
    period = split.get("locked_test_period_label", "—")
    enc = prep.get("encoded_feature_count", "—")
    return (
        f"DEV={dev:,} | Locked test={test:,} | Locked period={period} | Encoded features={enc}"
        if isinstance(dev, int) and isinstance(test, int)
        else f"Temporal split / TRAIN-only preprocessing complete | Locked period={period} | Encoded features={enc}"
    )


def _summary_branch_a(ctx: RunContext) -> str:
    meta = _read_json(ctx.root / "outputs/03_ai_job_market_segmentation/segmentation_metadata.json")
    rat = _read_json(ctx.root / "outputs/03_ai_job_market_segmentation/k_selection_rationale.json")
    rep = (
        meta.get("representation_id")
        or meta.get("selected_representation")
        or meta.get("representation_decision", {}).get("selected_representation")
        or "—"
    )
    alg = rat.get("selected_algorithm", meta.get("algorithm", "—"))
    k = rat.get("selected_k", meta.get("k", "—"))
    sil = rat.get("selected_silhouette", meta.get("silhouette"))
    seed_ari = rat.get("selected_stability_ari", meta.get("stability_ari"))
    sub_ari = rat.get(
        "selected_resample_stability_ari_mean",
        meta.get("resample_stability_ari_mean"),
    )
    min_share = rat.get("selected_min_cluster_share", meta.get("min_cluster_share"))
    msg = (
        f"Selected {rep} | {alg} K={k} | silhouette={_num(sil)}"
        f" | seed ARI={_num(seed_ari)} | subsample ARI={_num(sub_ari)}"
    )
    if min_share is not None:
        msg += f" | min cluster={_pct(min_share)}"
    return msg


def _summary_b4(ctx: RunContext) -> str:
    m = _read_csv(ctx.root / "outputs/04_model_comparison/09_model_comparison_temporal_cv.csv")
    if m.empty:
        m = _read_csv(ctx.root / "outputs/04_model_comparison/model_comparison.csv")
    if m.empty:
        return "Temporal-CV model comparison complete."
    row = m.sort_values("MAE_mean").iloc[0]
    return (
        f"Best CV model: {row.get('model', '—')} | MAE={_money(row.get('MAE_mean'))}"
        f" | RMSE={_money(row.get('RMSE_mean'))} | R²={_num(row.get('R2_mean'))}"
        f" | candidates={len(m)}"
    )


def _summary_b56(ctx: RunContext) -> str:
    met = _read_json(ctx.root / "outputs/05_best_model/locked_test_metrics.json")
    if not met:
        d = _read_csv(ctx.root / "outputs/05_best_model/10_final_locked_test_metrics.csv")
        if not d.empty:
            met = d.iloc[0].to_dict()
    if not met:
        return "Best-model locked-test evaluation complete."
    return (
        f"Locked test: MAE={_money(met.get('MAE'))} | RMSE={_money(met.get('RMSE'))}"
        f" | R²={_num(met.get('R2'))} | MedAE={_money(met.get('MedAE'))}"
        f" | q90 error band=±{_money(met.get('prediction_interval_abs_error_q90')).lstrip('$')}"
    )


def _summary_b7(ctx: RunContext) -> str:
    eq = _read_json(ctx.root / "outputs/06_salary_prediction/serialization_check.json")
    pred = _read_csv(ctx.root / "outputs/06_salary_prediction/12_prediction_summary.csv")
    eq_pass = eq.get("passed", "—")
    if pred.empty:
        return f"Deployment bundle written | serialization equivalence={eq_pass}"
    r = pred.iloc[0]
    return (
        f"Bundle equivalence={eq_pass} | model={r.get('model', '—')}"
        f" | prediction mean={_money(r.get('prediction_mean'))}"
        f" | locked rows={int(r.get('locked_test_rows', 0)):,}"
    )


def _summary_integrated(ctx: RunContext) -> str:
    seg = _read_csv(ctx.root / "outputs/07_integrated_insight/segment_salary_summary.csv")
    pred = _read_csv(ctx.root / "outputs/07_integrated_insight/predicted_salary_by_segment.csv")
    clusters = int(seg["cluster"].nunique()) if not seg.empty and "cluster" in seg.columns else "—"
    rows = int(seg["records"].sum()) if not seg.empty and "records" in seg.columns else "—"
    pred_rows = int(pred["records"].sum()) if not pred.empty and "records" in pred.columns else "—"
    return f"Integrated insight: clusters={clusters} | profiled rows={rows} | locked prediction rows={pred_rows}"


def _summary_final(ctx: RunContext) -> str:
    p = _read_csv(ctx.root / "outputs/08_full_pipeline/pipeline_status.csv")
    if p.empty:
        return "Run summary written."
    passed = int((p["status"].astype(str) == "PASS").sum()) if "status" in p.columns else len(p)
    return f"Pipeline status: {passed}/{len(p)} stages PASS"


@dataclass
class StageSpec:
    code: str
    title: str
    marker_suffix: str
    summary: Callable[[RunContext], str]


@dataclass
class RunContext:
    root: Path
    raw_path: Path
    raw_preview: pd.DataFrame | None = None


STAGES = [
    StageSpec(
        "C1",
        "Common 1 — Project Scope & Raw Data Ingestion",
        "outputs/01_data_basic_clean/raw_profile.csv",
        _summary_ingestion,
    ),
    StageSpec(
        "C2",
        "Common 2 — Basic Clean",
        "outputs/01_data_basic_clean/basic_clean_audit.json",
        _summary_basic_clean,
    ),
    StageSpec(
        "C3",
        "Common 3 — Data Quality & Contradiction Check",
        "outputs/01_data_basic_clean/target_summary.json",
        _summary_quality,
    ),
    StageSpec(
        "C4",
        "Common 4 — Feature Governance",
        "outputs/02_data_ready_for_ml/feature_policy.csv",
        _summary_feature_governance,
    ),
    StageSpec(
        "C5",
        "Common 5 — Shared Prepared Feature Base",
        "outputs/02_data_ready_for_ml/shared_prepared_feature_base.csv",
        _summary_shared_base,
    ),
    StageSpec(
        "B1-B3",
        "Branch B — Temporal Split & TRAIN-only Preprocessing",
        "outputs/02_data_ready_for_ml/08_training_readiness.json",
        _summary_b123,
    ),
    StageSpec(
        "A1-A8",
        "Branch A — AI Job Market Segmentation",
        "outputs/03_ai_job_market_segmentation/segmentation_insights.json",
        _summary_branch_a,
    ),
    StageSpec(
        "B4",
        "Branch B — Model Training & Temporal-CV Comparison",
        "outputs/04_model_comparison/09_fold_stability_mae.csv",
        _summary_b4,
    ),
    StageSpec(
        "B5-B6",
        "Branch B — Best Model, Explainability & Locked Test",
        "outputs/05_best_model/locked_test_metrics.json",
        _summary_b56,
    ),
    StageSpec(
        "B7",
        "Branch B — Deployment Bundle & Prediction Output",
        "outputs/06_salary_prediction/12_prediction_summary.csv",
        _summary_b7,
    ),
    StageSpec(
        "I1",
        "Integrated Insight — Segments + Salary Prediction",
        "outputs/07_integrated_insight/predicted_salary_by_segment.csv",
        _summary_integrated,
    ),
    StageSpec(
        "F",
        "Full Pipeline — Status & Reproducibility Summary",
        "outputs/08_full_pipeline/run_summary.json",
        _summary_final,
    ),
]


class TerminalProgress:
    def __init__(
        self,
        context: RunContext,
        stages: list[StageSpec],
        heartbeat_seconds: float = 20.0,
        heartbeat_enabled: bool = True,
    ) -> None:
        self.context = context
        self.stages = stages
        self.heartbeat_seconds = max(5.0, float(heartbeat_seconds))
        self.heartbeat_enabled = bool(heartbeat_enabled)
        self.index = 0
        self.stage_started = time.perf_counter()
        self.pipeline_started = self.stage_started
        self.records: list[dict] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def current(self) -> StageSpec | None:
        if 0 <= self.index < len(self.stages):
            return self.stages[self.index]
        return None

    def start(self) -> None:
        self.pipeline_started = time.perf_counter()
        self.stage_started = self.pipeline_started
        self._print_header()
        if self.heartbeat_enabled:
            self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.25)

    def _print_header(self) -> None:
        print()
        print(_blue(_hr("═")))
        print(_bold(" AI JOB MARKET — OFFLINE ML PIPELINE"))
        print(_blue(_hr("═")))
        print(f" Project root : {self.context.root}")
        print(f" Source       : {self.context.raw_path}")
        print(f" Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Stages       : {len(self.stages)}")
        print(_blue(_hr()))
        print(flush=True)

    def _announce_current(self) -> None:
        stage = self.current
        if stage is None:
            return
        self.stage_started = time.perf_counter()
        print(_blue(f"[RUNNING] {stage.code:<6} {stage.title}"), flush=True)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                stage = self.current
                if stage is None:
                    return
                elapsed = time.perf_counter() - self.stage_started
                total = time.perf_counter() - self.pipeline_started
                write_block(
                    sys.stdout,
                    _yellow(
                        f"[WORKING] {stage.code:<6} still running "
                        f"| stage={_fmt_elapsed(elapsed)} | total={_fmt_elapsed(total)}"
                    ),
                )

    def observe(self, event: dict) -> None:
        """Record stage timing from explicit core lifecycle events without printing twice."""
        event_name = event.get("event")
        stage_id = event.get("stage_id")
        if event_name not in {"stage_started", "stage_completed", "stage_failed", "stage_cancelled"}:
            return
        matches = [i for i, stage in enumerate(self.stages) if stage.code == stage_id]
        if not matches:
            return
        with self._lock:
            self.index = matches[0]
            stage = self.stages[self.index]
            if event_name == "stage_started":
                self.stage_started = time.perf_counter()
                return
            elapsed = time.perf_counter() - self.stage_started
            status = {
                "stage_completed": "PASS",
                "stage_failed": "FAIL",
                "stage_cancelled": "CANCELLED",
            }[event_name]
            self.records.append(
                {
                    "order": len(self.records) + 1,
                    "stage_code": stage.code,
                    "stage": stage.title,
                    "status": status,
                    "elapsed_seconds": round(elapsed, 6),
                    "elapsed_display": _fmt_elapsed(elapsed),
                    "summary": event.get("message", ""),
                    "marker_artifact": stage.marker_suffix,
                }
            )
            self.index += 1
            self.stage_started = time.perf_counter()

    def artifact_written(self, path: Path) -> None:
        """Legacy marker observer retained for compatibility; main() no longer installs it."""
        norm = Path(path).as_posix().lower()
        with self._lock:
            stage = self.current
            if stage is None:
                return
            marker = stage.marker_suffix.replace("\\", "/").lower()
            if not norm.endswith(marker):
                return

            elapsed = time.perf_counter() - self.stage_started
            try:
                summary = stage.summary(self.context)
            except Exception as exc:
                summary = f"Stage completed; summary unavailable ({type(exc).__name__})."

            self.records.append(
                {
                    "order": len(self.records) + 1,
                    "stage_code": stage.code,
                    "stage": stage.title,
                    "status": "PASS",
                    "elapsed_seconds": round(elapsed, 6),
                    "elapsed_display": _fmt_elapsed(elapsed),
                    "summary": summary,
                    "marker_artifact": stage.marker_suffix,
                }
            )

            print(_green(f"[PASS]    {stage.code:<6} {stage.title}"), flush=True)
            print(f"          Result : {summary}", flush=True)
            print(f"          Time   : {_fmt_elapsed(elapsed)}", flush=True)
            print(_hr(), flush=True)

            self.index += 1
            if self.current is not None:
                self._announce_current()

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            stage = self.current
            elapsed = time.perf_counter() - self.stage_started
            if stage is not None:
                self.records.append(
                    {
                        "order": len(self.records) + 1,
                        "stage_code": stage.code,
                        "stage": stage.title,
                        "status": "FAIL",
                        "elapsed_seconds": round(elapsed, 6),
                        "elapsed_display": _fmt_elapsed(elapsed),
                        "summary": f"{type(exc).__name__}: {exc}",
                        "marker_artifact": stage.marker_suffix,
                    }
                )
                print(_red(f"[FAIL]    {stage.code:<6} {stage.title}"), flush=True)
                print(_red(f"          {type(exc).__name__}: {exc}"), flush=True)
                print(f"          Time   : {_fmt_elapsed(elapsed)}", flush=True)
                print(_hr(), flush=True)

    def finalize_unknown(self) -> None:
        """Record missing observations as UNKNOWN; never infer PASS from a successful return."""
        with self._lock:
            observed = {row["stage_code"] for row in self.records}
            for stage in self.stages:
                if stage.code in observed:
                    continue
                self.records.append(
                    {
                        "order": len(self.records) + 1,
                        "stage_code": stage.code,
                        "stage": stage.title,
                        "status": "UNKNOWN",
                        "elapsed_seconds": None,
                        "elapsed_display": "unavailable",
                        "summary": "No explicit stage completion event was observed.",
                        "marker_artifact": stage.marker_suffix,
                    }
                )


def _install_output_hooks(progress: TerminalProgress):
    """Observe existing core save functions without changing modelling logic."""
    original_save_csv = core.save_csv
    original_save_json = core.save_json

    def save_csv_with_progress(df, path):
        original_save_csv(df, path)
        progress.artifact_written(Path(path))

    def save_json_with_progress(obj, path):
        original_save_json(obj, path)
        progress.artifact_written(Path(path))

    core.save_csv = save_csv_with_progress
    core.save_json = save_json_with_progress

    def restore():
        core.save_csv = original_save_csv
        core.save_json = original_save_json

    return restore


def _preflight(raw_path: Path) -> pd.DataFrame:
    print(_blue("[PREFLIGHT] Validating source file and schema..."), flush=True)
    t0 = time.perf_counter()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    raw = pd.read_csv(raw_path)
    validation = core.validate_input_schema(raw)

    errors = [x for x in validation.get("issues", []) if x.get("level") == "ERROR"]
    warnings = [x for x in validation.get("issues", []) if x.get("level") == "WARNING"]

    print(
        f"            Rows={len(raw):,} | Columns={raw.shape[1]} "
        f"| Schema={'PASS' if validation.get('valid') else 'FAIL'} "
        f"| Warnings={len(warnings)} | Errors={len(errors)}",
        flush=True,
    )
    if warnings:
        for item in warnings[:5]:
            print(
                _yellow(f"            WARNING [{item.get('field')}]: {item.get('message')}"),
                flush=True,
            )

    if errors:
        for item in errors:
            print(
                _red(f"            ERROR   [{item.get('field')}]: {item.get('message')}"),
                flush=True,
            )
        raise ValueError("Input schema failed preflight validation.")

    print(
        _green(f"[PREFLIGHT PASS] {_fmt_elapsed(time.perf_counter() - t0)}"),
        flush=True,
    )
    print(_hr(), flush=True)
    return raw


def _save_terminal_timing(root: Path, records: list[dict], total_seconds: float) -> None:
    out_dir = root / "outputs" / "08_full_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    timing = pd.DataFrame(records)
    if not timing.empty:
        timing.to_csv(out_dir / "pipeline_terminal_timing.csv", index=False)

    payload = {
        "total_elapsed_seconds": round(float(total_seconds), 6),
        "total_elapsed_display": _fmt_elapsed(total_seconds),
        "stage_count": len(records),
        "stages": records,
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "pipeline_terminal_timing.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _print_final_summary(summary: dict, progress: TerminalProgress, total_seconds: float) -> None:
    seg = summary.get("segmentation", {}) or {}
    rat = seg.get("rationale", {}) or {}
    met = summary.get("locked_test_metrics", {}) or {}

    rep = (
        seg.get("representation_id")
        or seg.get("selected_representation")
        or seg.get("representation_decision", {}).get("selected_representation")
        or "—"
    )
    alg = rat.get("selected_algorithm", seg.get("algorithm", "—"))
    kval = rat.get("selected_k", seg.get("k", "—"))
    sil = rat.get("selected_silhouette", seg.get("silhouette"))
    subari = rat.get(
        "selected_resample_stability_ari_mean",
        seg.get("resample_stability_ari_mean"),
    )

    print()
    print(_green(_hr("═")))
    print(_bold(" PIPELINE COMPLETED SUCCESSFULLY"))
    print(_green(_hr("═")))
    print(f" Run ID             : {summary.get('run_id', '—')}")
    print(f" Raw shape          : {summary.get('raw_shape', '—')}")
    print(f" Clean shape        : {summary.get('clean_shape', '—')}")
    print(
        f" DEV / Locked Test  : {summary.get('development_rows', '—')} / "
        f"{summary.get('locked_test_rows', '—')}"
    )
    print(
        f" Branch A           : {rep} | {alg} K={kval} "
        f"| Silhouette={_num(sil)} | Subsample ARI={_num(subari)}"
    )
    print(f" Best salary model  : {summary.get('best_model', '—')}")
    print(
        f" Locked test        : MAE={_money(met.get('MAE'))} "
        f"| RMSE={_money(met.get('RMSE'))} | R²={_num(met.get('R2'))} "
        f"| MedAE={_money(met.get('MedAE'))}"
    )
    print(f" Total elapsed      : {_fmt_elapsed(total_seconds)}")
    print(" Timing evidence    : outputs/08_full_pipeline/pipeline_terminal_timing.csv")
    print(_green(_hr("═")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full AI Job Market pipeline with terminal progress, stage summaries and timing."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Raw input CSV. Default: data/raw/ai_jobs_market_2025_2026.csv",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=20.0,
        help="Seconds between 'still running' messages for long stages. Default: 20.",
    )
    parser.add_argument(
        "--no-heartbeat",
        action="store_true",
        help="Disable periodic long-stage heartbeat messages.",
    )
    parser.add_argument(
        "--debuglog",
        action="store_true",
        help="Show detailed fold, trial, feature and split evidence (levels 3–4).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = args.data
    if not raw_path.is_absolute():
        raw_path = (PROJECT_ROOT / raw_path).resolve()

    try:
        raw = _preflight(raw_path)
    except Exception as exc:
        print(_red(f"\nPipeline not started: {type(exc).__name__}: {exc}"), flush=True)
        return 2

    context = RunContext(
        root=PROJECT_ROOT,
        raw_path=raw_path,
        raw_preview=raw,
    )
    progress = TerminalProgress(
        context,
        STAGES,
        heartbeat_seconds=args.heartbeat,
        heartbeat_enabled=not args.no_heartbeat,
    )

    progress.start()
    total_start = time.perf_counter()

    try:
        summary = core.run_pipeline(
            raw_path=raw_path,
            root=PROJECT_ROOT,
            workspace_root=None,
            debuglog=args.debuglog,
            on_audit_event=progress.observe,
        )
        progress.finalize_unknown()

        total_seconds = time.perf_counter() - total_start
        _save_terminal_timing(PROJECT_ROOT, progress.records, total_seconds)
        _print_final_summary(summary, progress, total_seconds)
        return 0

    except KeyboardInterrupt:
        progress.fail(KeyboardInterrupt("Run cancelled by user."))
        print(_red("\nPipeline cancelled."), flush=True)
        return 130

    except Exception as exc:
        progress.fail(exc)
        total_seconds = time.perf_counter() - total_start
        _save_terminal_timing(PROJECT_ROOT, progress.records, total_seconds)
        print(_red("\nPipeline failed. Full traceback:"), flush=True)
        traceback.print_exc()
        return 1

    finally:
        progress.stop()


if __name__ == "__main__":
    raise SystemExit(main())
