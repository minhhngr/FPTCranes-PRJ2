from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go

from ai_job_market.ui_evidence_io import EvidenceContractError, load_current_evidence
from components.presentation import (
    _display,
    _label,
    _src,
    _tr,
    display_column_config,
    display_frame,
    translate_figure,
)

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
    "MAE": _src("model_evidence.mean_absolute_error_the_average_absolute_dollar_4a9db47"),
    _src("model_evidence.medae_ad7020a"): _src(
        "model_evidence.median_absolute_error_the_middle_absolute_dollar_fcc02d6"
    ),
    "RMSE": _src("model_evidence.root_mean_squared_error_a_dollar_error_7d20ab7"),
    "R²": _src("model_evidence.explained_variance_relative_to_a_mean_baseline_adb2a96"),
    "CV": _src(
        "model_evidence.cross_validation_repeated_development_period_evaluation_here_7774177"
    ),
    _src("model_evidence.residual_0662144"): _src(
        "model_evidence.actual_salary_minus_predicted_salary_positive_means_31daa37"
    ),
    "q90": _src("model_evidence.the_90th_percentile_of_historical_absolute_error_534050d"),
}


def chart_container_width(priority: str) -> str | int:
    """Return the native Streamlit container width for a visual-priority tier."""
    try:
        return CHART_WIDTHS[priority]
    except KeyError as exc:
        raise ValueError(
            _tr("model_evidence.unknown_chart_priority_value0_518b0c6", value0=f"{priority}")
        ) from exc


def render_page_brief(
    st,
    *,
    question: str,
    evidence_scope: str,
    takeaway: str,
    limitation: str,
) -> None:
    with st.container(border=True):
        st.markdown(_tr("model_evidence.report_question_c2567a4"))
        st.markdown(_display(question))
        st.markdown(
            _tr("model_evidence.current_takeaway_value0_93ed113", value0=_display(takeaway))
        )
        st.caption(_tr("model_evidence.evidence_value0_4b668dc", value0=_display(evidence_scope)))
        st.caption(
            _tr("model_evidence.main_limitation_value0_731ef95", value0=_display(limitation))
        )


def render_metric_glossary(st, terms: list[str]) -> None:
    with st.expander(_tr("model_evidence.how_to_read_these_metrics_4271ae0"), expanded=False):
        for term in terms:
            definition = METRIC_GLOSSARY.get(term)
            if definition:
                st.markdown(f"**{_label(term)}:** {_display(definition)}")


def render_conclusion(
    st,
    conclusion: dict[str, Any],
    *,
    title: str = _src("model_evidence.evidence_conclusion_d451c7a"),
) -> None:
    with st.container(border=True):
        st.markdown(_display(f"**{title}**"))
        st.markdown(_tr("report.finding", value=_display(conclusion["finding"])))
        st.markdown(
            _tr(
                "model_evidence.why_it_matters_value0_325039d",
                value0=_display(conclusion["why_it_matters"]),
            )
        )
        st.markdown(_tr("report.limit", value=_display(conclusion["limit"])))
        if conclusion.get("decision_or_use"):
            st.markdown(_tr("report.decision", value=_display(conclusion["decision_or_use"])))


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
        raise EvidenceContractError(
            _tr("model_evidence.evidence_file_is_unavailable_value0_6f0b474", value0=f"{stem}")
        )
    return json.loads((root / entry["path"]).read_text(encoding="utf-8"))


def load_top2_bundle(root: Path, manifest: dict[str, Any]):
    entry = next(
        (item for item in manifest["files"] if Path(item["path"]).name == "top2_model.joblib"),
        None,
    )
    if not entry:
        raise EvidenceContractError(_src("model_evidence.top_2_model_is_unavailable_7badb7f"))
    return joblib.load(root / entry["path"])


def baseline_status(row: pd.Series, dummy_mae: float | None) -> str:
    if row.get("model") == _src("model_evidence.dummy_median_accc4f4"):
        return _src("model_evidence.reference_baseline_58e7d63")
    value = row.get("validation_MAE_mean")
    if dummy_mae is None or pd.isna(value):
        return _src("model_evidence.insufficient_evidence_d7584d9")
    return (
        _src("model_evidence.below_baseline_46a3099")
        if float(value) < float(dummy_mae)
        else _src("model_evidence.at_above_baseline_6beb6a9")
    )


def _key_value_labels(values: pd.Series, formatter) -> list[str]:
    """Label extrema only when a comparison line would otherwise become crowded."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    selected = {valid.idxmin(), valid.idxmax()} if len(valid) else set()
    return [formatter(value) if index in selected else "" for index, value in numeric.items()]


def _unavailable_conclusion(missing: str) -> dict[str, Any]:
    return {
        "available": False,
        "finding": _tr(
            "model_evidence.conclusion_unavailable_missing_value0_11de0e3", value0=f"{missing}"
        ),
        "why_it_matters": _src("model_evidence.the_report_will_not_infer_a_result_7cd5589"),
        "limit": _src("model_evidence.restore_or_regenerate_evidence_that_matches_the_9491de9"),
        "decision_or_use": None,
    }


def ranking_conclusion(summary: pd.DataFrame) -> dict[str, Any]:
    required = {"model", "validation_MAE_mean"}
    if (
        summary.empty
        or not required <= set(summary)
        or summary.validation_MAE_mean.notna().sum() < 2
    ):
        return _unavailable_conclusion(
            _src("model_evidence.at_least_two_candidate_validation_mae_rows_02861b6")
        )
    ordered = summary.dropna(subset=["validation_MAE_mean"])
    ordered = ordered[
        ordered.validation_MAE_mean.map(lambda value: math.isfinite(float(value)))
    ].sort_values("validation_MAE_mean")
    if len(ordered) < 2:
        return _unavailable_conclusion(
            _src("model_evidence.at_least_two_finite_candidate_validation_mae_92bf110")
        )
    winner, runner = ordered.iloc[0], ordered.iloc[1]
    gap = float(runner.validation_MAE_mean - winner.validation_MAE_mean)
    dummy = ordered[ordered.model == _src("model_evidence.dummy_median_accc4f4")]
    if len(dummy) and float(dummy.iloc[0].validation_MAE_mean) != 0:
        dummy_mae = float(dummy.iloc[0].validation_MAE_mean)
        improvement = 100 * (dummy_mae - float(winner.validation_MAE_mean)) / abs(dummy_mae)
        baseline_text = _tr(
            "model_evidence.value0_lower_than_the_dummy_reference_5ff3605",
            value0=f"{improvement:.1f}",
        )
    else:
        baseline_text = _src("model_evidence.dummy_relative_improvement_is_unavailable_5c6be90")
    return {
        "available": True,
        "finding": (
            _tr(
                "model_evidence.value0_has_the_lowest_mean_temporal_validation_c79ce6b",
                value0=f"{winner.model}",
                value1=f"{float(winner.validation_MAE_mean):,.0f}",
            )
        ),
        "why_it_matters": (
            _tr(
                "model_evidence.it_leads_value0_by_value1_and_is_86f93e5",
                value0=f"{runner.model}",
                value1=f"{gap:,.0f}",
                value2=f"{baseline_text}",
            )
        ),
        "limit": _tr("model_evidence.this_ranks_frozen_candidate_configurations_on_dev_c3d6f95"),
        "decision_or_use": _src(
            "model_evidence.use_this_ranking_to_compare_candidate_family_4a9c673"
        ),
    }


def fold_stability_conclusion(folds: pd.DataFrame, model: str) -> dict[str, Any]:
    required = {"model", "fold_id", "validation_MAE"}
    data = folds[folds.model == model] if required <= set(folds) else pd.DataFrame()
    data = data.dropna(subset=["validation_MAE"]) if len(data) else data
    if len(data):
        data = data[data.validation_MAE.map(lambda value: math.isfinite(float(value)))]
    if data.empty:
        return _unavailable_conclusion(
            _tr("model_evidence.fold_validation_mae_for_value0_4a39ee8", value0=f"{model}")
        )
    worst = data.loc[data.validation_MAE.idxmax()]
    period = worst.get("validation_period")
    period_text = f" ({period})" if pd.notna(period) else ""
    spread = float(data.validation_MAE.max() - data.validation_MAE.min())
    return {
        "available": True,
        "finding": (
            _tr(
                "model_evidence.fold_value0_value1_is_the_weakest_value2_c92bb39",
                value0=f"{int(worst.fold_id)}",
                value1=f"{period_text}",
                value2=f"{model}",
                value3=f"{float(worst.validation_MAE):,.0f}",
            )
        ),
        "why_it_matters": _tr(
            "model_evidence.validation_mae_spans_value0_across_the_recorded_b83efe1",
            value0=f"{spread:,.0f}",
        ),
        "limit": _src("model_evidence.fold_variation_is_descriptive_and_does_not_ec71bd0"),
        "decision_or_use": _src(
            "model_evidence.review_the_weakest_period_before_relying_on_06cb6c8"
        ),
    }


def generalization_conclusion(cv: pd.Series, test: pd.Series) -> dict[str, Any]:
    required = {"MAE", _src("model_evidence.medae_ad7020a"), "RMSE", "R2"}
    if not required <= set(cv.index) or not required <= set(test.index):
        return _unavailable_conclusion(
            _src("model_evidence.matching_cv_and_historical_test_metrics_0913809")
        )
    if any(
        pd.isna(cv[key])
        or pd.isna(test[key])
        or not math.isfinite(float(cv[key]))
        or not math.isfinite(float(test[key]))
        for key in required
    ):
        return _unavailable_conclusion(
            _src("model_evidence.finite_matching_cv_and_historical_test_metrics_72c5260")
        )
    mae_delta = float(test.MAE - cv.MAE)
    mae_direction = (
        "decreased"
        if mae_delta < 0
        else "increased"
        if mae_delta > 0
        else _src("model_evidence.was_unchanged_799289c")
    )
    med_delta = float(test.MedAE - cv.MedAE)
    r2_delta = float(test.R2 - cv.R2)
    r2_direction = (
        "increased"
        if r2_delta > 0
        else "decreased"
        if r2_delta < 0
        else _src("model_evidence.was_unchanged_799289c")
    )
    return {
        "available": True,
        "finding": _tr(
            "model_evidence.historical_test_mae_value0_by_value1_versus_2f1fce8",
            value0=f"{mae_direction}",
            value1=f"{abs(mae_delta):,.0f}",
        ),
        "why_it_matters": _tr(
            "model_evidence.median_absolute_error_changed_by_value0_usd_a30a49b",
            value0=f"{med_delta:+,.0f}",
            value1=f"{float(test.RMSE):,.0f}",
        ),
        "limit": _tr(
            "model_evidence.r_value0_by_value1_the_historical_test_f51305d",
            value0=f"{r2_direction}",
            value1=f"{abs(r2_delta):.3f}",
        ),
        "decision_or_use": _src(
            "model_evidence.use_the_paired_metrics_as_retrospective_generalization_bf6ed8b"
        ),
    }


def uncertainty_conclusion(test: pd.Series) -> dict[str, Any]:
    required = {
        "q90_abs_error_usd",
        "coverage",
        "row_count",
        "RMSE",
        _src("model_evidence.medae_ad7020a"),
    }
    if not required <= set(test.index) or any(
        pd.isna(test[key]) or not math.isfinite(float(test[key])) for key in required
    ):
        return _unavailable_conclusion(
            _src("model_evidence.q90_coverage_population_rmse_and_medae_c256449")
        )
    tail_ratio = float(test.RMSE / test.MedAE) if float(test.MedAE) else None
    return {
        "available": True,
        "finding": (
            _tr(
                "model_evidence.the_value0_historical_error_band_covers_value1_aee47b4",
                value0=f"{float(test.q90_abs_error_usd):,.0f}",
                value1=f"{float(test.coverage):.1%}",
                value2=f"{int(test.row_count)}",
            )
        ),
        "why_it_matters": (
            _tr(
                "model_evidence.rmse_is_value0_medae_indicating_that_larger_bf0c293",
                value0=f"{tail_ratio:.1f}",
            )
            if tail_ratio is not None
            else _src("model_evidence.the_rmse_medae_tail_ratio_is_unavailable_0b00493")
        ),
        "limit": _tr("model_evidence.coverage_was_measured_on_the_same_historically_6eba4ef"),
        "decision_or_use": _src("model_evidence.use_the_band_as_an_empirical_risk_06e484b"),
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
        return _unavailable_conclusion(
            _src("model_evidence.matched_full_and_top_2_historical_test_bbf181d")
        )
    full, top2 = indexed.loc["full"], indexed.loc["top2"]
    if not all(math.isfinite(float(value)) for value in [full.MAE, top2.MAE]):
        return _unavailable_conclusion(
            _src("model_evidence.finite_full_and_top_2_historical_test_8127295")
        )
    if int(full.row_count) != int(top2.row_count):
        return _unavailable_conclusion(
            _src("model_evidence.equal_full_and_top_2_historical_test_ad6bfa2")
        )
    delta = float(top2.MAE - full.MAE)
    if delta < 0:
        finding = _tr(
            "model_evidence.top_2_historical_test_mae_is_value0_bac7f08",
            value0=f"{abs(delta):,.0f}",
            value1=f"{int(full.row_count)}",
        )
    elif delta > 0:
        finding = _tr(
            "model_evidence.top_2_historical_test_mae_is_value0_b5860ea",
            value0=f"{delta:,.0f}",
            value1=f"{int(full.row_count)}",
        )
    else:
        finding = _tr(
            "model_evidence.top_2_and_full_historical_test_mae_0b4a59d",
            value0=f"{int(full.row_count)}",
        )
    return {
        "available": True,
        "finding": finding,
        "why_it_matters": _src(
            "model_evidence.the_comparison_isolates_the_fixed_feature_set_5120309"
        ),
        "limit": _tr("model_evidence.top_2_reused_frozen_full_model_settings_3b25f97"),
        "decision_or_use": _src("model_evidence.use_top_2_for_the_controlled_two_2cc75ca"),
    }


def tuning_conclusion(
    summary: pd.DataFrame | None, applied_parameters: dict[str, Any]
) -> dict[str, Any]:
    required = {"hyperparameter", "optimal_value", "best_cv_r2", "best_cv_mae"}
    if summary is None or summary.empty or not required <= set(summary):
        return _unavailable_conclusion(
            _src("model_evidence.validated_tuning_sensitivity_summary_a0c4a86")
        )
    parameter_names = [str(value).split(". ", 1)[-1] for value in summary.hyperparameter]
    applied = ", ".join(
        f"{name}={applied_parameters.get(name, 'unavailable')}" for name in parameter_names
    )
    return {
        "available": True,
        "finding": _tr(
            "model_evidence.the_inherited_evidence_records_value0_one_parameter_da34b4b",
            value0=f"{len(summary)}",
            value1=f"{applied}",
        ),
        "why_it_matters": _tr(
            "model_evidence.the_displayed_sweeps_show_local_sensitivity_around_a34ddee"
        ),
        "limit": _tr("model_evidence.the_search_was_non_nested_dev_evaluation_5886832"),
        "decision_or_use": _src(
            "model_evidence.read_applied_settings_from_the_saved_model_aac0acc"
        ),
    }


def prediction_batch_conclusion(results: pd.DataFrame, q90_abs_error_usd: float) -> dict[str, Any]:
    required = {"predicted_salary_usd", "validation_mode"}
    if results.empty or not required <= set(results):
        return _unavailable_conclusion(
            _src("model_evidence.a_current_nonempty_prediction_result_snapshot_1b1ff36")
        )
    predictions = pd.to_numeric(results.predicted_salary_usd, errors="coerce")
    if predictions.isna().any() or not predictions.map(math.isfinite).all():
        return _unavailable_conclusion(_src("model_evidence.finite_batch_predictions_46ae58f"))
    if not math.isfinite(float(q90_abs_error_usd)) or float(q90_abs_error_usd) < 0:
        return _unavailable_conclusion(
            _src("model_evidence.a_finite_nonnegative_q90_error_reference_0488285")
        )
    actual_count = int(results.get("absolute_error_usd", pd.Series(dtype=float)).notna().sum())
    exception_count = int((results.validation_mode == "experience_exception").sum())
    scenario_word = "scenario" if len(results) == 1 else "scenarios"
    actual_word = "actual" if actual_count == 1 else "actuals"
    return {
        "available": True,
        "finding": (
            _tr(
                "model_evidence.value0_value1_range_from_value2_to_value3_a5e00a0",
                value0=f"{len(results)}",
                value1=f"{scenario_word}",
                value2=f"{predictions.min():,.0f}",
                value3=f"{predictions.max():,.0f}",
            )
        ),
        "why_it_matters": (
            _tr(
                "model_evidence.the_batch_has_value0_known_value1_the_921efc6",
                value0=f"{actual_count}",
                value1=f"{actual_word}",
                value2=f"{float(q90_abs_error_usd):,.0f}",
            )
        ),
        "limit": (
            _tr(
                "model_evidence.value0_extrapolated_scenario_s_fall_outside_observed_8fdbba2",
                value0=f"{exception_count}",
            )
            if exception_count
            else _tr("model_evidence.all_scenarios_use_strict_observed_dev_bounds_7430f17")
        ),
        "decision_or_use": _src(
            "model_evidence.compare_controlled_scenarios_do_not_treat_point_46897f4"
        ),
    }


def candidate_comparison_figure(summary: pd.DataFrame) -> go.Figure:
    ordered = summary.sort_values("validation_MAE_mean").reset_index(drop=True)
    figure = go.Figure()
    figure.add_bar(
        x=ordered["model"],
        y=ordered["validation_MAE_mean"],
        name=_src("model_evidence.validation_mae_c5a1c85"),
        text=[f"${value:,.0f}" for value in ordered["validation_MAE_mean"]],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    if "train_MAE_mean" in ordered:
        figure.add_scatter(
            x=ordered["model"],
            y=ordered["train_MAE_mean"],
            name=_src("model_evidence.train_mae_8ffd76c"),
            mode="lines+markers+text",
            text=_key_value_labels(ordered["train_MAE_mean"], lambda value: f"${value:,.0f}"),
            textposition=_src("model_evidence.top_center_24b3167"),
            line_color=EVIDENCE_COLORS["training"],
            marker_color=EVIDENCE_COLORS["training"],
        )
    dummy = ordered.loc[
        ordered.model == _src("model_evidence.dummy_median_accc4f4"), "validation_MAE_mean"
    ]
    if len(dummy):
        figure.add_hline(
            y=float(dummy.iloc[0]),
            line_dash="dash",
            line_color=EVIDENCE_COLORS["threshold"],
            annotation_text=_src("model_evidence.dummy_median_b8dcf68"),
        )
    figure.update_layout(
        title=_src("model_evidence.train_and_temporal_validation_error_5d99f86"),
        xaxis=dict(
            title=_src("model_evidence.frozen_candidate_configuration_b23ed7e"), automargin=True
        ),
        yaxis=dict(
            title=_src("model_evidence.mae_usd_lower_is_better_78e001a"),
            tickformat=",.0f",
            automargin=True,
        ),
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
        name=_src("model_evidence.validation_train_mae_22adea5"),
        text=["—" if pd.isna(v) else f"{v:.1f}×" for v in ratio],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    figure.add_scatter(
        x=ordered.model,
        y=ordered.validation_R2_mean,
        name=_src("model_evidence.validation_r_538b15c"),
        yaxis="y2",
        mode="lines+markers+text",
        text=_key_value_labels(ordered.validation_R2_mean, lambda value: f"{value:.3f}"),
        textposition=_src("model_evidence.top_center_24b3167"),
        line_color=EVIDENCE_COLORS["training"],
        marker_color=EVIDENCE_COLORS["training"],
    )
    figure.update_layout(
        title=_src("model_evidence.error_ratio_and_explained_variance_6dc3cea"),
        xaxis=dict(automargin=True),
        yaxis=dict(title=_src("model_evidence.mae_ratio_5fa0fd9"), automargin=True),
        yaxis2=dict(
            title=_src("model_evidence.validation_r_538b15c"),
            overlaying="y",
            side="right",
            zeroline=True,
            automargin=True,
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
        name=_src("model_evidence.validation_mae_c5a1c85"),
        text=[f"${v:,.0f}" for v in data.validation_MAE],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    figure.add_scatter(
        x=data.fold_id,
        y=data.train_MAE,
        name=_src("model_evidence.train_mae_8ffd76c"),
        mode="lines+markers+text",
        text=_key_value_labels(data.train_MAE, lambda value: f"${value:,.0f}"),
        textposition=_src("model_evidence.top_center_24b3167"),
        line_color=EVIDENCE_COLORS["training"],
        marker_color=EVIDENCE_COLORS["training"],
    )
    figure.update_layout(
        title=_tr("model_evidence.value0_fold_by_fold_mae_c77b25b", value0=f"{model}"),
        xaxis=dict(title=_src("model_evidence.fold_92c122b"), automargin=True),
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
        name=_src("model_evidence.historical_test_rows_4312b04"),
        text=data.get("job_title"),
        hovertemplate=_src("model_evidence.actual_x_0f_br_predicted_y_0f_0414ed0"),
        marker_color=EVIDENCE_COLORS["prediction"],
    )
    figure.add_scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        name=_src("model_evidence.ideal_y_x_2022943"),
        line=dict(dash="dash", color=EVIDENCE_COLORS["neutral"]),
    )
    figure.update_layout(
        title=_src("model_evidence.historical_test_actual_vs_predicted_2fd71ac"),
        xaxis=dict(
            title=_src("model_evidence.actual_usd_4ac8326"), tickformat=",.0f", automargin=True
        ),
        yaxis=dict(
            title=_src("model_evidence.predicted_usd_b999c81"), tickformat=",.0f", automargin=True
        ),
        height=500,
        margin=dict(l=85, r=35, t=75, b=70),
    )
    return figure


def generalization_error_figure(cv: pd.Series, test: pd.Series) -> go.Figure:
    metrics = ["MAE", _src("model_evidence.medae_ad7020a"), "RMSE"]
    figure = go.Figure()
    figure.add_bar(
        x=metrics,
        y=[test[metric] for metric in metrics],
        name=_src("model_evidence.historical_test_4241834"),
        text=[f"${test[metric]:,.0f}" for metric in metrics],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["historical_test"],
    )
    figure.add_scatter(
        x=metrics,
        y=[cv[metric] for metric in metrics],
        name=_src("model_evidence.dev_cv_2c5366a"),
        mode="lines+markers+text",
        text=[f"${cv[metric]:,.0f}" for metric in metrics],
        textposition=_src("model_evidence.top_center_24b3167"),
        line_color=EVIDENCE_COLORS["development"],
        marker_color=EVIDENCE_COLORS["development"],
    )
    figure.update_layout(
        title=_src("model_evidence.frozen_full_configuration_dollar_error_metrics_a5e581e"),
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
                _tr("report.years", value=f"{row.get('years_of_experience'):g}")
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
        name=_src("model_evidence.prediction_6f2ddd1"),
        text=[f"${v:,.0f}" for v in data.predicted_salary_usd],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["prediction"],
        hovertext=hover_text,
        hovertemplate=_src("model_evidence.x_br_hovertext_br_prediction_y_0f_c0b6ac3"),
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
            name=_src("model_evidence.known_historical_actual_b05a0de"),
            mode="markers+text",
            marker=dict(color=EVIDENCE_COLORS["actual"], size=11, symbol="diamond"),
            text=[f"${v:,.0f}" for v in actual.actual_salary_usd],
            textposition=_src("model_evidence.top_center_24b3167"),
        )
    figure.update_layout(
        title=_src(
            "model_evidence.salary_predictions_with_empirical_historical_error_whiskers_10acc98"
        ),
        xaxis=dict(title=_src("model_evidence.scenario_id_a8571ee"), automargin=True),
        yaxis=dict(
            title=_src("model_evidence.annual_salary_usd_46de225"),
            tickformat=",.0f",
            automargin=True,
        ),
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
        title=_tr(
            "model_evidence.historical_test_raw_feature_permutation_reliance_top_aa482cd",
            value0=f"{visible_rows}",
        ),
        xaxis=dict(
            title=_src("model_evidence.mae_increase_usd_7841f5c"),
            tickformat=",.0f",
            automargin=True,
        ),
        yaxis=dict(automargin=True),
        height=max(480, 32 * visible_rows + 150),
        margin=dict(l=220, r=45, t=80, b=65),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return figure
