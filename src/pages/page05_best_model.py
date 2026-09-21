from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ai_job_market.ui_evidence_io import EvidenceContractError

from .common import money, show_plot, style_page
from .model_evidence import (
    EVIDENCE_COLORS,
    actual_predicted_figure,
    chart_container_width,
    generalization_conclusion,
    generalization_error_figure,
    importance_figure,
    load_evidence,
    render_conclusion,
    render_metric_glossary,
    render_page_brief,
    tuning_conclusion,
    uncertainty_conclusion,
    variant_conclusion,
)
from .model_training_presentation import (
    TABLE_EMPHASIS_LEGEND,
    load_compatible_training_audit,
    load_evidence_download,
    temporal_validation_guide,
    tuning_decision_table,
    tuning_method_guide,
)
from .training_validation_presentation import render_training_log_footer


def _metrics(st, values):
    with st.container(horizontal=True):
        for label, value in values:
            st.metric(label, value, border=True)


def _evidence_download(st, root, manifest, label: str, stem: str, *, key: str):
    item = load_evidence_download(root, manifest, stem)
    st.download_button(
        label,
        item["content"],
        file_name=item["filename"],
        mime=item["mime"],
        key=key,
        width="stretch",
    )


def _audit_downloads(st, audit: dict):
    if not audit.get("available"):
        st.caption(f"Historical primary-pipeline audit unavailable: {audit.get('reason')}")
        return
    st.caption(
        "Historical primary-pipeline audit · "
        f"pipeline `{audit['pipeline_run_id']}` · audit `{audit['audit_run_id']}` · "
        f"selection source `{audit.get('selection_source') or 'unavailable'}`. "
        "This trace is separate from the active supplemental metrics."
    )
    for index, item in enumerate(audit["downloads"]):
        st.download_button(
            f"Download {item['label'].lower()}",
            item["content"],
            file_name=item["filename"],
            mime=item["mime"],
            key=f"p5_audit_{index}",
            width="stretch",
        )


def _tuning_figure(frame: pd.DataFrame, parameter: str, title: str) -> go.Figure:
    ordered = frame.sort_values(parameter, key=lambda values: values.astype(str))
    figure = go.Figure()
    figure.add_bar(
        x=ordered[parameter].astype(str),
        y=ordered.CV_MAE,
        name="CV MAE",
        text=[f"${v:,.0f}" for v in ordered.CV_MAE],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    figure.add_scatter(
        x=ordered[parameter].astype(str),
        y=ordered.CV_R2,
        name="CV R²",
        yaxis="y2",
        mode="lines+markers+text",
        text=[f"{value:.3f}" if value == ordered.CV_R2.max() else "" for value in ordered.CV_R2],
        textposition="top center",
        line_color=EVIDENCE_COLORS["development"],
        marker_color=EVIDENCE_COLORS["development"],
    )
    figure.update_layout(
        title=title,
        xaxis=dict(automargin=True),
        yaxis=dict(title="CV MAE (USD)", tickformat=",.0f", automargin=True),
        yaxis2=dict(title="CV R²", overlaying="y", side="right", automargin=True),
        legend=dict(orientation="h"),
        height=410,
        margin=dict(l=85, r=85, t=80, b=70),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return figure


def _branch_b_fold_table(folds: pd.DataFrame) -> pd.DataFrame:
    """Build one evidence-backed row per historical Branch B temporal fold."""
    required = {
        "fold_id",
        "train_period",
        "validation_period",
        "train_rows",
        "validation_rows",
    }
    if folds.empty or required - set(folds.columns):
        raise ValueError("candidate fold evidence is incomplete")
    definitions = (
        folds[
            [
                "fold_id",
                "train_period",
                "validation_period",
                "train_rows",
                "validation_rows",
            ]
        ]
        .drop_duplicates()
        .sort_values("fold_id", kind="stable")
        .reset_index(drop=True)
    )
    if definitions["fold_id"].duplicated().any():
        raise ValueError("candidate models disagree on temporal fold definitions")
    definitions.insert(0, "Step", range(1, len(definitions) + 1))
    definitions.insert(2, "Flow", "Train → Validation")
    return definitions.rename(
        columns={
            "fold_id": "Fold",
            "train_period": "Train period",
            "validation_period": "Validation period",
            "train_rows": "Train rows",
            "validation_rows": "Validation rows",
        }
    )


def _render_branch_b_fold_log(st, manifest, tables):
    folds = tables.get("candidate_fold_metrics", pd.DataFrame())
    membership = tables.get("fold_membership", pd.DataFrame())
    try:
        guide = temporal_validation_guide(
            folds,
            membership,
            full_feature_count=len(manifest["full_features"]),
        )
        fold_table = _branch_b_fold_table(folds)
        guide_error = None
    except (KeyError, TypeError, ValueError) as exc:
        guide = None
        fold_table = pd.DataFrame()
        guide_error = str(exc)

    with st.expander("Branch B · How the temporal DEV folds are split", expanded=False):
        st.caption(
            "Historical Branch B evidence. This is the adjacent-block DEV protocol used by the "
            "saved workflow; it is separate from the strict training-validation/v1 expanding-fold protocol."
        )
        if guide is None:
            st.warning(f"Branch B fold guide unavailable: {guide_error}")
            return

        with st.container(horizontal=True):
            st.metric("Temporal folds", guide["fold_count"], border=True)
            st.metric("Candidate models", guide["candidate_count"], border=True)
            st.metric("Train/validation overlap", guide["row_overlap_count"], border=True)

        st.markdown("**How the fold split works**")
        st.markdown(guide["method"])
        st.dataframe(
            fold_table,
            hide_index=True,
            width="stretch",
            column_config={
                "Step": st.column_config.NumberColumn(format="%d"),
                "Train rows": st.column_config.NumberColumn(format="%d"),
                "Validation rows": st.column_config.NumberColumn(format="%d"),
            },
        )

        st.markdown("**Step-by-step fold execution**")
        for row in fold_table.to_dict("records"):
            st.markdown(
                f"**Step {row['Step']} · {row['Fold']}:** fit preprocessing + model on "
                f"`{row['Train period']}` (**{int(row['Train rows']):,} rows**) → validate on "
                f"`{row['Validation period']}` (**{int(row['Validation rows']):,} rows**)."
            )

        st.markdown("**Limits**")
        st.markdown(guide["limitations"])
        st.caption(
            "Calendar labels can meet at block boundaries while record identities remain disjoint. "
            "The historical test was not used to choose fold membership or tuning settings, but it has been exposed to final scoring."
        )


def render(st, root, role="admin"):
    style_page(st)
    st.title("5. Best model, diagnostics and uncertainty")
    st.caption("Saved full model · frozen-configuration CV · historically scored test evidence")
    try:
        manifest, tables = load_evidence(root)
        variants = tables["variant_metrics"]
        predictions = tables["variant_test_predictions"]
    except (EvidenceContractError, KeyError, OSError, ValueError) as exc:
        st.error(f"Supplemental diagnostic evidence is unavailable: {exc}")
        st.code(
            f'PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "{root}"'
        )
        render_training_log_footer(
            st, load_compatible_training_audit(root), page="page05"
        )
        return

    audit = load_compatible_training_audit(root)

    test = variants[
        (variants.feature_variant == "full") & (variants.evaluation == "historical_test")
    ].iloc[0]
    cv = variants[
        (variants.feature_variant == "full") & (variants.evaluation == "dev_cv_mean")
    ].iloc[0]
    r2_gap = float(test.R2 - cv.R2)
    generalization = generalization_conclusion(cv, test)
    render_page_brief(
        st,
        question="How does the saved full model generalize from temporal DEV validation to the historically scored test period?",
        evidence_scope=f"Final full configuration · {int(test.row_count)} historical-test rows · {manifest['evidence_id']}",
        takeaway=generalization["finding"],
        limitation=generalization["limit"],
    )
    _metrics(
        st,
        [
            ("Saved model", f"Random Forest · full {len(manifest['full_features'])}"),
            ("Historical test R²", f"{test.R2:.3f}"),
            ("Historical test MAE", money(test.MAE)),
            ("Historical test MedAE", money(test.MedAE)),
            ("Test − CV R²", f"{r2_gap:+.3f}"),
            ("Empirical q90", f"±{money(test.q90_abs_error_usd)}"),
        ],
    )
    test_period = manifest.get("datasets", {}).get("historical_test", {}).get("period", "unknown")
    st.caption(
        f"{int(test.row_count)} historically scored rows · period {test_period} · evidence `{manifest['evidence_id']}`"
    )
    render_metric_glossary(st, ["MAE", "MedAE", "RMSE", "R²", "CV", "Residual", "q90"])

    general, diagnostics, tuning, reliance, trust = st.tabs(
        ["Generalization", "Residual diagnostics", "Tuning", "Feature reliance", "Trust and audit"]
    )
    with general:
        st.markdown(
            "**Question:** Did dollar error and explained variance improve or deteriorate from final-configuration DEV CV to the historical test period?"
        )
        show_plot(st, generalization_error_figure(cv, test), "p5_general_dollars")
        render_conclusion(st, generalization, title="Generalization conclusion")
        r2_figure = go.Figure()
        r2_figure.add_bar(
            x=["Historical test"],
            y=[test.R2],
            name="Historical test",
            text=[f"{test.R2:.3f}"],
            textposition="outside",
            marker_color=EVIDENCE_COLORS["historical_test"],
        )
        r2_figure.add_scatter(
            x=["DEV CV"],
            y=[cv.R2],
            name="DEV CV",
            mode="markers+text",
            text=[f"{cv.R2:.3f}"],
            textposition="top center",
            marker=dict(color=EVIDENCE_COLORS["development"], size=12, symbol="circle"),
        )
        r2_figure.update_layout(
            title="R² on a separate unitless scale",
            xaxis=dict(automargin=True),
            yaxis=dict(title="R²", zeroline=True, tickformat=".3f", automargin=True),
            height=390,
            margin=dict(l=75, r=40, t=80, b=70),
        )
        with st.container(width=chart_container_width("P2")):
            show_plot(st, r2_figure, "p5_general_r2")
        scorecard = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "DEV CV": float(cv[metric]),
                    "Historical test": float(test[metric]),
                    "absolute_delta_test_minus_cv": float(test[metric] - cv[metric]),
                    "relative_delta_pct": 100
                    * float(test[metric] - cv[metric])
                    / abs(float(cv[metric]))
                    if float(cv[metric]) != 0
                    else None,
                }
                for metric in ["R2", "MAE", "MedAE", "RMSE"]
            ]
        )
        scorecard["Unit"] = scorecard.metric.map(
            {"R2": "unitless", "MAE": "USD", "MedAE": "USD", "RMSE": "USD"}
        )
        for column in ["DEV CV", "Historical test", "absolute_delta_test_minus_cv"]:
            scorecard[column] = [
                f"{value:.3f}" if metric == "R2" else f"${value:,.0f}"
                for metric, value in zip(scorecard.metric, scorecard[column])
            ]
        scorecard["relative_delta_pct"] = scorecard.relative_delta_pct.map(
            lambda value: "Unavailable" if pd.isna(value) else f"{value:+.1f}%"
        )
        scorecard = scorecard.rename(
            columns={
                "metric": "Metric",
                "absolute_delta_test_minus_cv": "Test − CV delta",
                "relative_delta_pct": "Relative delta",
            }
        )
        scorecard["DEV CV"] = scorecard["DEV CV"].map(lambda value: f"*{value}*")
        for column in ["Historical test", "Test − CV delta", "Relative delta"]:
            scorecard[column] = scorecard[column].map(lambda value: f"**{value}**")
        scorecard["Unit"] = scorecard["Unit"].map(lambda value: f"*{value}*")
        st.markdown(TABLE_EMPHASIS_LEGEND)
        st.table(scorecard, hide_index=True, border="horizontal")
        st.info(
            "Lower error is an error reduction—not a percentage accuracy gain. Historical test results were not used to retune this saved model."
        )

    with diagnostics:
        st.markdown(
            "**Question:** Where do historical predictions depart from actual salaries, and are errors concentrated in a tail?"
        )
        show_plot(st, actual_predicted_figure(predictions, "full:selected"), "p5_actual_predicted")
        full = predictions[predictions.model_id == "full:selected"]
        median_residual = float(full.residual_usd.median())
        render_conclusion(
            st,
            {
                "available": True,
                "finding": f"The median historical-test residual (actual − predicted) is ${median_residual:,.0f} across {len(full)} rows.",
                "why_it_matters": "Residual sign shows the typical direction of error, while the scatter shows how errors vary across salary levels.",
                "limit": "A centered median or visual pattern does not prove unbiasedness, no overfitting or future stability.",
                "decision_or_use": "Inspect the residual distribution and tail metrics together rather than relying on one center statistic.",
            },
            title="Residual diagnostic conclusion",
        )
        residual = go.Figure(
            go.Histogram(
                x=full.residual_usd,
                nbinsx=30,
                name="Historical-test residuals",
                marker_color=EVIDENCE_COLORS["historical_test"],
                opacity=0.82,
            )
        )
        residual.add_vline(x=0, line_dash="dash", annotation_text="Zero")
        residual.update_layout(
            title="Historical-test residual distribution",
            xaxis=dict(title="Actual − predicted (USD)", tickformat=",.0f", automargin=True),
            yaxis=dict(title="Records", automargin=True),
            height=420,
            margin=dict(l=75, r=40, t=80, b=70),
        )
        with st.container(width=chart_container_width("P3")):
            show_plot(st, residual, "p5_residual")
        st.caption(
            "Scatter alignment and residual shape are diagnostics; neither proves unbiasedness, no overfitting, or future-market validity."
        )

    with tuning:
        st.markdown(
            "**Question:** Which parameter values were applied, and what does the inherited sensitivity evidence actually support?"
        )
        _render_branch_b_fold_log(st, manifest, tables)
        applied_parameters = manifest["models"]["full:selected"]["parameters"]
        try:
            tuning_guide = tuning_method_guide(
                tables,
                applied_parameters=applied_parameters,
                effective_folds=int(cv.effective_folds),
            )
        except (KeyError, TypeError, ValueError) as exc:
            tuning_guide = None
            tuning_guide_error = str(exc)
        with st.expander("Priority guide: How hyperparameter tuning works", expanded=False):
            if tuning_guide is None:
                st.warning(f"Tuning methodology guide unavailable: {tuning_guide_error}")
            else:
                counts = tuning_guide["stage_trial_counts"]
                st.markdown("**What is tuned?**")
                st.markdown(
                    "The inherited workflow tunes four Random Forest controls: number of trees "
                    "(`n_estimators`), tree depth, minimum leaf size and feature subsampling. "
                    "Every trial keeps the same full feature contract and random seed."
                )
                st.markdown("**How does the search run?**")
                st.markdown(
                    f"It first compares **{tuning_guide['initial_trial_count']} anchor configurations**, "
                    f"then runs sequential one-parameter sweeps: **{counts['n_estimators']}** tree-count, "
                    f"**{counts['max_depth']}** depth, **{counts['min_samples_leaf']}** leaf-size and "
                    f"**{counts['max_features']}** feature-subsampling trials "
                    f"(**{tuning_guide['total_trial_count']} trials total**). Each trial is scored "
                    f"over {tuning_guide['effective_folds']} temporal DEV folds. Values selected or "
                    "fixed by the inherited workflow are carried into later sweeps."
                )
                st.markdown("**Why tune on temporal DEV folds?**")
                st.markdown(
                    "The procedure compares settings on later chronological blocks without using the "
                    "locked test to choose parameters. Reusing identical folds makes trial scores comparable."
                )
                st.markdown("**Selection rule**")
                st.markdown(
                    f"Tuning ranks trials by **{tuning_guide['ranking_metric']} "
                    f"({tuning_guide['ranking_direction']})**. This differs from the Page 04 family "
                    "comparison, which ranks candidates by mean validation **MAE (lower is better)**."
                )
                st.markdown("**Applied configuration**")
                match_text = (
                    "matches"
                    if tuning_guide["saved_matches_initial_winner"]
                    else "does not fully match"
                )
                st.markdown(
                    f"The saved model {match_text} the highest-R² row in the recorded initial anchor "
                    "table. The primary audit records selection from the first row of that sorted table; "
                    "later coordinate sweeps are sensitivity evidence and do not silently redefine the model."
                )
                st.markdown("**Limitations**")
                st.markdown(
                    "This is an inherited **non-nested DEV search**: the same development-fold system "
                    "supports tuning and reporting, so sweep gains are not an independent outer-fold estimate. "
                    "The locked test was not used to choose these settings, but it was historically exposed "
                    "to final scoring. Parameter associations do not prove why performance changed."
                )
            st.markdown("**Current tuning evidence downloads**")
            active_downloads = [
                ("Download initial anchor grid", "manual_tuning_step1_gridsearch"),
                ("Download n_estimators sweep", "manual_tuning_step2_n_estimators"),
                ("Download max_depth sweep", "manual_tuning_step3_max_depth"),
                ("Download min_samples_leaf sweep", "manual_tuning_step4_min_samples_leaf"),
                ("Download max_features sweep", "manual_tuning_step5_max_features"),
                ("Download tuning summary", "manual_tuning_4params_summary"),
            ]
            for index, (label, stem) in enumerate(active_downloads):
                if tables.get(stem) is not None:
                    _evidence_download(
                        st,
                        root,
                        manifest,
                        label,
                        stem,
                        key=f"p5_tuning_{index}",
                    )
            st.markdown("**Optional historical primary-pipeline audit**")
            _audit_downloads(st, audit)
        stages = [
            ("manual_tuning_step2_n_estimators", "n_estimators", "Number of trees"),
            ("manual_tuning_step3_max_depth", "max_depth", "Maximum depth"),
            ("manual_tuning_step4_min_samples_leaf", "min_samples_leaf", "Minimum leaf samples"),
            ("manual_tuning_step5_max_features", "max_features", "Feature subsampling"),
        ]
        available = 0
        for stem, parameter, title in stages:
            frame = tables.get(stem)
            if frame is None or frame.empty or parameter not in frame:
                st.warning(f"{title} sensitivity evidence is unavailable.")
                continue
            with st.container(width=chart_container_width("P2")):
                show_plot(st, _tuning_figure(frame, parameter, title), f"p5_tune_{parameter}")
            available += 1
        summary = tables.get("manual_tuning_4params_summary")
        render_conclusion(
            st,
            tuning_conclusion(summary, applied_parameters),
            title="Tuning evidence conclusion",
        )
        if summary is not None:
            st.markdown(TABLE_EMPHASIS_LEGEND)
            st.table(
                tuning_decision_table(summary, applied_parameters),
                hide_index=True,
                border="horizontal",
            )
            tuning_table = summary[
                ["hyperparameter", "search_space", "optimal_value", "best_cv_r2", "best_cv_mae"]
            ].rename(
                columns={
                    "hyperparameter": "Hyperparameter",
                    "search_space": "Recorded sweep",
                    "optimal_value": "Stage winner",
                    "best_cv_r2": "Best CV R²",
                    "best_cv_mae": "Best CV MAE (USD)",
                }
            )
            st.dataframe(
                tuning_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Best CV R²": st.column_config.NumberColumn(format="%.3f"),
                    "Best CV MAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                },
            )
        st.caption(
            f"{available}/4 sensitivity charts available. Sweep winners are evidence; the model's applied parameters are the manifest contract."
        )

    with reliance:
        st.markdown(
            "**Question:** Which raw inputs does the fitted model rely on, and what changes when serving only category and experience?"
        )
        permutation = tables.get("variant_permutation_importance", pd.DataFrame())
        if permutation.empty:
            st.warning("Raw permutation evidence is unavailable.")
        else:
            show_plot(st, importance_figure(permutation, "full:selected"), "p5_full_importance")
        encoded = tables.get("variant_encoded_importance", pd.DataFrame())
        if not encoded.empty:
            top = (
                encoded[encoded.model_id == "full:selected"]
                .nlargest(15, "importance")
                .sort_values("importance")
            )
            figure = go.Figure(
                go.Bar(
                    x=top.importance,
                    y=top.encoded_feature,
                    orientation="h",
                    text=[f"{v:.3f}" for v in top.importance],
                    textposition="outside",
                    marker_color=EVIDENCE_COLORS["validation"],
                )
            )
            figure.update_layout(
                title="Full model: encoded estimator reliance · Top 15",
                xaxis=dict(title="Impurity importance", tickformat=".3f", automargin=True),
                yaxis=dict(automargin=True),
                height=max(520, 30 * len(top) + 140),
                margin=dict(l=250, r=45, t=80, b=65),
                uniformtext=dict(minsize=10, mode="hide"),
            )
            with st.container(width=chart_container_width("P3")):
                show_plot(st, figure, "p5_encoded_importance")
        comparison = variants[variants.evaluation.isin(["dev_cv_mean", "historical_test"])][
            ["feature_variant", "evaluation", "row_count", "MAE", "RMSE", "R2", "MedAE"]
        ]
        st.subheader("Fixed full vs Top-2 feature comparison")
        comparison_display = comparison.rename(
            columns={
                "feature_variant": "Feature set",
                "evaluation": "Evaluation",
                "row_count": "Rows",
                "MAE": "MAE (USD)",
                "RMSE": "RMSE (USD)",
                "R2": "R²",
                "MedAE": "MedAE (USD)",
            }
        )
        st.dataframe(
            comparison_display,
            hide_index=True,
            width="stretch",
            column_config={
                "Rows": st.column_config.NumberColumn(format="%d"),
                "MAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "RMSE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
                "MedAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
            },
        )
        render_conclusion(st, variant_conclusion(variants), title="Feature-set conclusion")
        st.warning(
            "Importance and feature ablation describe this fitted dataset/model. They are not causal wage effects, fairness evidence, or proof of production fitness."
        )

    with trust:
        st.markdown(
            "**Question:** How wide is the empirical historical-error reference, how often did it cover this same population, and what tail risk remains?"
        )
        coverage = float(test.coverage)
        full = predictions[predictions.model_id == "full:selected"]
        tail_ratio = float(test.RMSE / test.MedAE) if test.MedAE else None
        _metrics(
            st,
            [
                ("Empirical q90", f"±{money(test.q90_abs_error_usd)}"),
                ("Same-population coverage", f"{coverage:.1%}"),
                ("Coverage denominator", f"{int(test.row_count)}"),
                ("RMSE / MedAE", f"{tail_ratio:.1f}×" if tail_ratio is not None else "Unavailable"),
            ],
        )
        sorted_errors = full.absolute_error_usd.sort_values()
        ecdf = go.Figure(
            go.Scatter(
                x=sorted_errors,
                y=[(i + 1) / len(sorted_errors) for i in range(len(sorted_errors))],
                mode="lines",
                name="Historical-test empirical CDF",
                line_color=EVIDENCE_COLORS["historical_test"],
            )
        )
        ecdf.add_vline(x=float(test.q90_abs_error_usd), line_dash="dash", annotation_text="q90")
        ecdf.update_layout(
            title="Historical-test absolute-error distribution",
            xaxis=dict(title="Absolute error (USD)", tickformat=",.0f", automargin=True),
            yaxis=dict(title="Cumulative share", tickformat=".0%", automargin=True),
            height=430,
            margin=dict(l=80, r=40, t=80, b=70),
        )
        with st.container(width=chart_container_width("P3")):
            show_plot(st, ecdf, "p5_ecdf")
        render_conclusion(st, uncertainty_conclusion(test), title="Uncertainty conclusion")
        examples = tables.get("benchmark_examples", pd.DataFrame())
        st.subheader("Three traceable historical benchmark examples")
        if len(examples):
            example_display = examples[
                [
                    "record_id",
                    "job_title",
                    "job_category",
                    "years_of_experience",
                    "annual_salary_usd",
                    "predicted_full_usd",
                    "predicted_top2_usd",
                    "historically_exposed",
                    "pristine",
                ]
            ].rename(
                columns={
                    "record_id": "Record ID",
                    "job_title": "Job title",
                    "job_category": "Job category",
                    "years_of_experience": "Experience (years)",
                    "annual_salary_usd": "Actual salary (USD)",
                    "predicted_full_usd": "Full prediction (USD)",
                    "predicted_top2_usd": "Top-2 prediction (USD)",
                    "historically_exposed": "Historically exposed",
                    "pristine": "Pristine",
                }
            )
            st.dataframe(
                example_display,
                hide_index=True,
                width="stretch",
                column_config={
                    "Record ID": st.column_config.TextColumn(pinned=True),
                    "Experience (years)": st.column_config.NumberColumn(format="%.1f"),
                    "Actual salary (USD)": st.column_config.NumberColumn(format="$%.0f"),
                    "Full prediction (USD)": st.column_config.NumberColumn(format="$%.0f"),
                    "Top-2 prediction (USD)": st.column_config.NumberColumn(format="$%.0f"),
                },
            )
        else:
            st.warning("No strict-policy historical examples are available.")
        st.warning(
            "The q90 band is computed and checked on the same historical test population. It is not a formal confidence interval or guaranteed coverage for future or extrapolated scenarios."
        )

    render_training_log_footer(st, audit, page="page05")
