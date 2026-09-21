from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go

from ai_job_market.ui_evidence_io import EvidenceContractError, load_current_evidence

EVIDENCE_COLORS = {
    "validation": "#1976D2",
    "training": "#F28E2B",
    "development": "#F28E2B",
    "historical_test": "#D32F2F",
    "threshold": "#B71C1C",
    "prediction": "#00897B",
    "uncertainty": "#F9A825",
    "actual": "#C62828",
    "neutral": "#6B7280",
}
CHART_WIDTHS = {"P1": "stretch", "P2": 820, "P3": 680}
PREDICTION_SERIES_COLORS = ["#00897B", "#1976D2", "#00796B", "#1565C0"]
METRIC_GLOSSARY = {
    "MAE": "Mean absolute error: the average absolute dollar difference between prediction and actual salary; lower is better.",
    "MedAE": "Median absolute error: the middle absolute dollar error, which is less influenced by extreme misses.",
    "RMSE": "Root mean squared error: a dollar error measure that gives larger misses more weight; lower is better.",
    "R²": "Explained variance relative to a mean baseline. One is ideal, zero matches the baseline, and negative values are worse than it.",
    "CV": "Cross-validation: repeated development-period evaluation. Here the folds follow recorded temporal order.",
    "Residual": "Actual salary minus predicted salary. Positive means the model predicted below the actual value.",
    "q90": "The 90th percentile of historical absolute error. It is an empirical reference, not a guaranteed confidence interval.",
}


def chart_container_width(priority: str) -> str | int:
    """Return the native Streamlit container width for a visual-priority tier."""
    try:
        return CHART_WIDTHS[priority]
    except KeyError as exc:
        raise ValueError(f"unknown chart priority: {priority}") from exc


def render_page_brief(
    st,
    *,
    question: str,
    evidence_scope: str,
    takeaway: str,
    limitation: str,
) -> None:
    with st.container(border=True):
        st.markdown("**Report question**")
        st.markdown(question)
        st.markdown(f"**Current takeaway:** {takeaway}")
        st.caption(f"Evidence: {evidence_scope}")
        st.caption(f"Main limitation: {limitation}")


def render_metric_glossary(st, terms: list[str]) -> None:
    with st.expander("How to read these metrics", expanded=False):
        for term in terms:
            definition = METRIC_GLOSSARY.get(term)
            if definition:
                st.markdown(f"**{term}:** {definition}")


def render_conclusion(
    st, conclusion: dict[str, Any], *, title: str = "Evidence conclusion"
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(f"**Finding:** {conclusion['finding']}")
        st.markdown(f"**Why it matters:** {conclusion['why_it_matters']}")
        st.markdown(f"**Limit:** {conclusion['limit']}")
        if conclusion.get("decision_or_use"):
            st.markdown(f"**Decision/use:** {conclusion['decision_or_use']}")


def load_evidence(root: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest = load_current_evidence(root)
    tables: dict[str, pd.DataFrame] = {}
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.suffix == ".csv":
            tables[path.stem] = pd.read_csv(path)
    return manifest, tables


def load_json_file(root: Path, manifest: dict[str, Any], stem: str) -> dict[str, Any]:
    entry = next((item for item in manifest["files"] if Path(item["path"]).stem == stem), None)
    if not entry:
        raise EvidenceContractError(f"evidence file is unavailable: {stem}")
    return json.loads((root / entry["path"]).read_text(encoding="utf-8"))


def load_top2_bundle(root: Path, manifest: dict[str, Any]):
    entry = next(
        (item for item in manifest["files"] if Path(item["path"]).name == "top2_model.joblib"),
        None,
    )
    if not entry:
        raise EvidenceContractError("Top-2 model is unavailable")
    return joblib.load(root / entry["path"])


def baseline_status(row: pd.Series, dummy_mae: float | None) -> str:
    if row.get("model") == "Dummy Median":
        return "Reference baseline"
    value = row.get("validation_MAE_mean")
    if dummy_mae is None or pd.isna(value):
        return "Insufficient evidence"
    return "Below baseline" if float(value) < float(dummy_mae) else "At/above baseline"


def _key_value_labels(values: pd.Series, formatter) -> list[str]:
    """Label extrema only when a comparison line would otherwise become crowded."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    selected = {valid.idxmin(), valid.idxmax()} if len(valid) else set()
    return [formatter(value) if index in selected else "" for index, value in numeric.items()]


def _unavailable_conclusion(missing: str) -> dict[str, Any]:
    return {
        "available": False,
        "finding": f"Conclusion unavailable — missing {missing}.",
        "why_it_matters": "The report will not infer a result from incomplete evidence.",
        "limit": "Restore or regenerate evidence that matches the current data format before interpretation.",
        "decision_or_use": None,
    }


def ranking_conclusion(summary: pd.DataFrame) -> dict[str, Any]:
    required = {"model", "validation_MAE_mean"}
    if (
        summary.empty
        or not required <= set(summary)
        or summary.validation_MAE_mean.notna().sum() < 2
    ):
        return _unavailable_conclusion("at least two candidate validation-MAE rows")
    ordered = summary.dropna(subset=["validation_MAE_mean"])
    ordered = ordered[
        ordered.validation_MAE_mean.map(lambda value: math.isfinite(float(value)))
    ].sort_values("validation_MAE_mean")
    if len(ordered) < 2:
        return _unavailable_conclusion("at least two finite candidate validation-MAE rows")
    winner, runner = ordered.iloc[0], ordered.iloc[1]
    gap = float(runner.validation_MAE_mean - winner.validation_MAE_mean)
    dummy = ordered[ordered.model == "Dummy Median"]
    if len(dummy) and float(dummy.iloc[0].validation_MAE_mean) != 0:
        dummy_mae = float(dummy.iloc[0].validation_MAE_mean)
        improvement = 100 * (dummy_mae - float(winner.validation_MAE_mean)) / abs(dummy_mae)
        baseline_text = f"{improvement:.1f}% lower than the Dummy reference"
    else:
        baseline_text = "Dummy-relative improvement is unavailable"
    return {
        "available": True,
        "finding": (
            f"{winner.model} has the lowest mean temporal-validation MAE at "
            f"${float(winner.validation_MAE_mean):,.0f}."
        ),
        "why_it_matters": (f"It leads {runner.model} by ${gap:,.0f} and is {baseline_text}."),
        "limit": "This ranks frozen candidate configurations on DEV folds; it does not replace the saved-model identity or prove a causal fit diagnosis.",
        "decision_or_use": "Use this ranking to compare candidate-family evidence, not to auto-promote a model.",
    }


def fold_stability_conclusion(folds: pd.DataFrame, model: str) -> dict[str, Any]:
    required = {"model", "fold_id", "validation_MAE"}
    data = folds[folds.model == model] if required <= set(folds) else pd.DataFrame()
    data = data.dropna(subset=["validation_MAE"]) if len(data) else data
    if len(data):
        data = data[data.validation_MAE.map(lambda value: math.isfinite(float(value)))]
    if data.empty:
        return _unavailable_conclusion(f"fold validation MAE for {model}")
    worst = data.loc[data.validation_MAE.idxmax()]
    period = worst.get("validation_period")
    period_text = f" ({period})" if pd.notna(period) else ""
    spread = float(data.validation_MAE.max() - data.validation_MAE.min())
    return {
        "available": True,
        "finding": (
            f"Fold {int(worst.fold_id)}{period_text} is the weakest {model} period at "
            f"${float(worst.validation_MAE):,.0f} validation MAE."
        ),
        "why_it_matters": f"Validation MAE spans ${spread:,.0f} across the recorded folds, so performance is not temporally uniform.",
        "limit": "Fold variation is descriptive and does not identify the economic or modeling cause.",
        "decision_or_use": "Review the weakest period before relying on the mean alone.",
    }


def generalization_conclusion(cv: pd.Series, test: pd.Series) -> dict[str, Any]:
    required = {"MAE", "MedAE", "RMSE", "R2"}
    if not required <= set(cv.index) or not required <= set(test.index):
        return _unavailable_conclusion("matching CV and historical-test metrics")
    if any(
        pd.isna(cv[key])
        or pd.isna(test[key])
        or not math.isfinite(float(cv[key]))
        or not math.isfinite(float(test[key]))
        for key in required
    ):
        return _unavailable_conclusion("finite matching CV and historical-test metrics")
    mae_delta = float(test.MAE - cv.MAE)
    mae_direction = (
        "decreased" if mae_delta < 0 else "increased" if mae_delta > 0 else "was unchanged"
    )
    med_delta = float(test.MedAE - cv.MedAE)
    r2_delta = float(test.R2 - cv.R2)
    r2_direction = "increased" if r2_delta > 0 else "decreased" if r2_delta < 0 else "was unchanged"
    return {
        "available": True,
        "finding": f"Historical-test MAE {mae_direction} by ${abs(mae_delta):,.0f} versus final-configuration DEV CV.",
        "why_it_matters": f"Median absolute error changed by {med_delta:+,.0f} USD; test RMSE is ${float(test.RMSE):,.0f}.",
        "limit": f"R² {r2_direction} by {abs(r2_delta):.3f}; the historical test was already exposed to scoring and is not pristine.",
        "decision_or_use": "Use the paired metrics as retrospective generalization diagnostics, not a future guarantee.",
    }


def uncertainty_conclusion(test: pd.Series) -> dict[str, Any]:
    required = {"q90_abs_error_usd", "coverage", "row_count", "RMSE", "MedAE"}
    if not required <= set(test.index) or any(
        pd.isna(test[key]) or not math.isfinite(float(test[key])) for key in required
    ):
        return _unavailable_conclusion("q90, coverage, population, RMSE and MedAE")
    tail_ratio = float(test.RMSE / test.MedAE) if float(test.MedAE) else None
    return {
        "available": True,
        "finding": (
            f"The ±${float(test.q90_abs_error_usd):,.0f} historical-error band covers "
            f"{float(test.coverage):.1%} of {int(test.row_count)} scored rows."
        ),
        "why_it_matters": (
            f"RMSE is {tail_ratio:.1f}× MedAE, indicating that larger errors materially affect the average squared error."
            if tail_ratio is not None
            else "The RMSE/MedAE tail ratio is unavailable because MedAE is zero."
        ),
        "limit": "Coverage was measured on the same historically scored population; it is not a calibrated future confidence interval.",
        "decision_or_use": "Use the band as an empirical risk reference alongside the point estimate.",
    }


def variant_conclusion(variants: pd.DataFrame) -> dict[str, Any]:
    required = {"feature_variant", "evaluation", "MAE", "row_count"}
    data = (
        variants[variants.evaluation == "historical_test"]
        if required <= set(variants)
        else pd.DataFrame()
    )
    indexed = data.set_index("feature_variant") if len(data) else data
    if data.empty or not {"full", "top2"} <= set(indexed.index):
        return _unavailable_conclusion("matched full and Top-2 historical-test rows")
    full, top2 = indexed.loc["full"], indexed.loc["top2"]
    if not all(math.isfinite(float(value)) for value in [full.MAE, top2.MAE]):
        return _unavailable_conclusion("finite full and Top-2 historical-test MAE")
    if int(full.row_count) != int(top2.row_count):
        return _unavailable_conclusion("equal full and Top-2 historical-test populations")
    delta = float(top2.MAE - full.MAE)
    if delta < 0:
        finding = f"Top-2 historical-test MAE is ${abs(delta):,.0f} lower than the full model on {int(full.row_count)} matched rows."
    elif delta > 0:
        finding = f"Top-2 historical-test MAE is ${delta:,.0f} higher than the full model on {int(full.row_count)} matched rows."
    else:
        finding = (
            f"Top-2 and full historical-test MAE are equal on {int(full.row_count)} matched rows."
        )
    return {
        "available": True,
        "finding": finding,
        "why_it_matters": "The comparison isolates the fixed feature-set change on the same historical population.",
        "limit": "Top-2 reused frozen full-model settings and was not independently tuned or promoted from test results.",
        "decision_or_use": "Use Top-2 for the controlled two-input scenario UI; do not infer causal feature sufficiency.",
    }


def tuning_conclusion(
    summary: pd.DataFrame | None, applied_parameters: dict[str, Any]
) -> dict[str, Any]:
    required = {"hyperparameter", "optimal_value", "best_cv_r2", "best_cv_mae"}
    if summary is None or summary.empty or not required <= set(summary):
        return _unavailable_conclusion("validated tuning sensitivity summary")
    parameter_names = [str(value).split(". ", 1)[-1] for value in summary.hyperparameter]
    applied = ", ".join(
        f"{name}={applied_parameters.get(name, 'unavailable')}" for name in parameter_names
    )
    return {
        "available": True,
        "finding": f"The inherited evidence records {len(summary)} one-parameter sensitivity sweeps; the saved model applies {applied}.",
        "why_it_matters": "The displayed sweeps show local sensitivity around the inherited workflow rather than a new UI-time search.",
        "limit": "The search was non-nested DEV evaluation ranked by R²; sweep winners do not independently establish future optimality.",
        "decision_or_use": "Read applied settings from the saved model contract and use sweeps only as supporting evidence.",
    }


def prediction_batch_conclusion(results: pd.DataFrame, q90_abs_error_usd: float) -> dict[str, Any]:
    required = {"predicted_salary_usd", "validation_mode"}
    if results.empty or not required <= set(results):
        return _unavailable_conclusion("a current nonempty prediction result snapshot")
    predictions = pd.to_numeric(results.predicted_salary_usd, errors="coerce")
    if predictions.isna().any() or not predictions.map(math.isfinite).all():
        return _unavailable_conclusion("finite batch predictions")
    if not math.isfinite(float(q90_abs_error_usd)) or float(q90_abs_error_usd) < 0:
        return _unavailable_conclusion("a finite nonnegative q90 error reference")
    actual_count = int(results.get("absolute_error_usd", pd.Series(dtype=float)).notna().sum())
    exception_count = int((results.validation_mode == "experience_exception").sum())
    scenario_word = "scenario" if len(results) == 1 else "scenarios"
    actual_word = "actual" if actual_count == 1 else "actuals"
    return {
        "available": True,
        "finding": (
            f"{len(results)} {scenario_word} range from ${predictions.min():,.0f} to "
            f"${predictions.max():,.0f} in predicted annual salary."
        ),
        "why_it_matters": (
            f"The batch has {actual_count} known {actual_word}; the empirical historical-error reference is "
            f"±${float(q90_abs_error_usd):,.0f}."
        ),
        "limit": (
            f"{exception_count} extrapolated scenario(s) fall outside observed DEV experience bounds; "
            "historical q90 coverage is unvalidated there."
            if exception_count
            else "All scenarios use strict observed DEV bounds, but historical q90 still does not guarantee future coverage."
        ),
        "decision_or_use": "Compare controlled scenarios; do not treat point estimates or bands as compensation guarantees.",
    }


def candidate_comparison_figure(summary: pd.DataFrame) -> go.Figure:
    ordered = summary.sort_values("validation_MAE_mean").reset_index(drop=True)
    figure = go.Figure()
    figure.add_bar(
        x=ordered["model"],
        y=ordered["validation_MAE_mean"],
        name="Validation MAE",
        text=[f"${value:,.0f}" for value in ordered["validation_MAE_mean"]],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    if "train_MAE_mean" in ordered:
        figure.add_scatter(
            x=ordered["model"],
            y=ordered["train_MAE_mean"],
            name="Train MAE",
            mode="lines+markers+text",
            text=_key_value_labels(ordered["train_MAE_mean"], lambda value: f"${value:,.0f}"),
            textposition="top center",
            line_color=EVIDENCE_COLORS["training"],
            marker_color=EVIDENCE_COLORS["training"],
        )
    dummy = ordered.loc[ordered.model == "Dummy Median", "validation_MAE_mean"]
    if len(dummy):
        figure.add_hline(
            y=float(dummy.iloc[0]),
            line_dash="dash",
            line_color=EVIDENCE_COLORS["threshold"],
            annotation_text="Dummy median",
        )
    figure.update_layout(
        title="Train and temporal-validation error",
        xaxis=dict(title="Frozen candidate configuration", automargin=True),
        yaxis=dict(title="MAE (USD, lower is better)", tickformat=",.0f", automargin=True),
        barmode="group",
        height=470,
        margin=dict(l=80, r=35, t=75, b=105),
        uniformtext=dict(minsize=11, mode="hide"),
    )
    return figure


def ratio_r2_figure(summary: pd.DataFrame) -> go.Figure:
    ordered = summary.sort_values("validation_MAE_mean").copy()
    ratio = ordered["validation_MAE_mean"] / ordered["train_MAE_mean"].replace(0, pd.NA)
    figure = go.Figure()
    figure.add_bar(
        x=ordered.model,
        y=ratio,
        name="Validation / train MAE",
        text=["—" if pd.isna(v) else f"{v:.1f}×" for v in ratio],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    figure.add_scatter(
        x=ordered.model,
        y=ordered.validation_R2_mean,
        name="Validation R²",
        yaxis="y2",
        mode="lines+markers+text",
        text=_key_value_labels(ordered.validation_R2_mean, lambda value: f"{value:.3f}"),
        textposition="top center",
        line_color=EVIDENCE_COLORS["training"],
        marker_color=EVIDENCE_COLORS["training"],
    )
    figure.update_layout(
        title="Error ratio and explained variance",
        xaxis=dict(automargin=True),
        yaxis=dict(title="MAE ratio", automargin=True),
        yaxis2=dict(
            title="Validation R²", overlaying="y", side="right", zeroline=True, automargin=True
        ),
        height=430,
        margin=dict(l=70, r=80, t=75, b=105),
        uniformtext=dict(minsize=11, mode="hide"),
    )
    return figure


def fold_combo_figure(folds: pd.DataFrame, model: str) -> go.Figure:
    data = folds[folds.model == model].sort_values("fold_id")
    figure = go.Figure()
    figure.add_bar(
        x=data.fold_id,
        y=data.validation_MAE,
        name="Validation MAE",
        text=[f"${v:,.0f}" for v in data.validation_MAE],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    figure.add_scatter(
        x=data.fold_id,
        y=data.train_MAE,
        name="Train MAE",
        mode="lines+markers+text",
        text=_key_value_labels(data.train_MAE, lambda value: f"${value:,.0f}"),
        textposition="top center",
        line_color=EVIDENCE_COLORS["training"],
        marker_color=EVIDENCE_COLORS["training"],
    )
    figure.update_layout(
        title=f"{model}: fold-by-fold MAE",
        xaxis=dict(title="Fold", automargin=True),
        yaxis=dict(title="USD", tickformat=",.0f", automargin=True),
        height=450,
        margin=dict(l=80, r=35, t=75, b=65),
        uniformtext=dict(minsize=11, mode="hide"),
    )
    return figure


def actual_predicted_figure(predictions: pd.DataFrame, model_id: str) -> go.Figure:
    data = predictions[predictions.model_id == model_id]
    lo = min(data.annual_salary_usd.min(), data.predicted_salary_usd.min())
    hi = max(data.annual_salary_usd.max(), data.predicted_salary_usd.max())
    figure = go.Figure()
    figure.add_scatter(
        x=data.annual_salary_usd,
        y=data.predicted_salary_usd,
        mode="markers",
        name="Historical test rows",
        text=data.get("job_title"),
        hovertemplate="Actual $%{x:,.0f}<br>Predicted $%{y:,.0f}<extra></extra>",
        marker_color=EVIDENCE_COLORS["prediction"],
    )
    figure.add_scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        name="Ideal y=x",
        line=dict(dash="dash", color=EVIDENCE_COLORS["neutral"]),
    )
    figure.update_layout(
        title="Historical test: actual vs predicted",
        xaxis=dict(title="Actual USD", tickformat=",.0f", automargin=True),
        yaxis=dict(title="Predicted USD", tickformat=",.0f", automargin=True),
        height=500,
        margin=dict(l=85, r=35, t=75, b=70),
    )
    return figure


def generalization_error_figure(cv: pd.Series, test: pd.Series) -> go.Figure:
    metrics = ["MAE", "MedAE", "RMSE"]
    figure = go.Figure()
    figure.add_bar(
        x=metrics,
        y=[test[metric] for metric in metrics],
        name="Historical test",
        text=[f"${test[metric]:,.0f}" for metric in metrics],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["historical_test"],
    )
    figure.add_scatter(
        x=metrics,
        y=[cv[metric] for metric in metrics],
        name="DEV CV",
        mode="lines+markers+text",
        text=[f"${cv[metric]:,.0f}" for metric in metrics],
        textposition="top center",
        line_color=EVIDENCE_COLORS["development"],
        marker_color=EVIDENCE_COLORS["development"],
    )
    figure.update_layout(
        title="Frozen full configuration: dollar-error metrics",
        xaxis=dict(automargin=True),
        yaxis=dict(title="USD", tickformat=",.0f", automargin=True),
        height=440,
        margin=dict(l=80, r=35, t=75, b=65),
        uniformtext=dict(minsize=11, mode="hide"),
    )
    return figure


def prediction_interval_figure(results: pd.DataFrame) -> go.Figure:
    data = results.copy()
    hover_text = [
        " · ".join(
            part
            for part in [
                str(row.get("job_title", "")),
                str(row.get("job_category", "")),
                f"{row.get('years_of_experience'):g} years"
                if pd.notna(row.get("years_of_experience"))
                else "",
            ]
            if part
        )
        for _, row in data.iterrows()
    ]
    figure = go.Figure()
    figure.add_bar(
        x=data.scenario_id,
        y=data.predicted_salary_usd,
        name="Prediction",
        text=[f"${v:,.0f}" for v in data.predicted_salary_usd],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["prediction"],
        hovertext=hover_text,
        hovertemplate="%{x}<br>%{hovertext}<br>Prediction $%{y:,.0f}<extra></extra>",
        error_y=dict(
            type="data",
            symmetric=False,
            array=data.upper_bound_usd - data.predicted_salary_usd,
            arrayminus=data.predicted_salary_usd - data.lower_bound_usd,
            color=EVIDENCE_COLORS["uncertainty"],
            thickness=2,
            width=6,
        ),
    )
    if "actual_salary_usd" in data:
        actual = data[data.actual_salary_usd.notna()]
        figure.add_scatter(
            x=actual.scenario_id,
            y=actual.actual_salary_usd,
            name="Known historical actual",
            mode="markers+text",
            marker=dict(color=EVIDENCE_COLORS["actual"], size=11, symbol="diamond"),
            text=[f"${v:,.0f}" for v in actual.actual_salary_usd],
            textposition="top center",
        )
    figure.update_layout(
        title="Salary predictions with empirical historical-error whiskers",
        xaxis=dict(title="Scenario ID", automargin=True),
        yaxis=dict(title="Annual salary (USD)", tickformat=",.0f", automargin=True),
        height=480,
        margin=dict(l=90, r=45, t=80, b=80),
        uniformtext=dict(minsize=11, mode="hide"),
    )
    return figure


def importance_figure(importance: pd.DataFrame, model_id: str) -> go.Figure:
    data = importance[importance.model_id == model_id].sort_values("mae_increase_usd").tail(13)
    figure = go.Figure(
        go.Bar(
            x=data.mae_increase_usd,
            y=data.raw_feature,
            orientation="h",
            error_x=dict(type="data", array=data.std_usd),
            text=[f"${v:,.0f}" for v in data.mae_increase_usd],
            textposition="outside",
            marker_color=EVIDENCE_COLORS["validation"],
        )
    )
    visible_rows = len(data)
    figure.update_layout(
        title=f"Historical-test raw-feature permutation reliance · Top {visible_rows}",
        xaxis=dict(title="MAE increase (USD)", tickformat=",.0f", automargin=True),
        yaxis=dict(automargin=True),
        height=max(480, 32 * visible_rows + 150),
        margin=dict(l=220, r=45, t=80, b=65),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return figure
