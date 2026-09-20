from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ai_job_market.ui_evidence_io import EvidenceContractError

from .common import money, show_plot, style_page
from .model_evidence import (
    EVIDENCE_COLORS,
    baseline_status,
    candidate_comparison_figure,
    chart_container_width,
    fold_combo_figure,
    fold_stability_conclusion,
    load_evidence,
    ranking_conclusion,
    ratio_r2_figure,
    render_conclusion,
    render_metric_glossary,
    render_page_brief,
)
from .model_training_presentation import (
    TABLE_EMPHASIS_LEGEND,
    candidate_decision_table,
    load_compatible_training_audit,
    load_evidence_download,
    temporal_validation_guide,
)
from .training_validation_presentation import (
    render_historical_training_report,
    render_page04_training_validation,
)


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
        st.caption(f"Historical primary-pipeline audit unavailable: {audit.get('reason')} ")
        return
    st.caption(
        "Historical primary-pipeline audit · "
        f"pipeline `{audit['pipeline_run_id']}` · audit `{audit['audit_run_id']}`. "
        "This trace is separate from the active supplemental metrics."
    )
    for index, item in enumerate(audit["downloads"]):
        st.download_button(
            f"Download {item['label'].lower()}",
            item["content"],
            file_name=item["filename"],
            mime=item["mime"],
            key=f"p4_audit_{index}",
            width="stretch",
        )


def render(st, root, role="admin"):
    style_page(st)
    st.title("4. Model comparison and temporal validation")
    st.caption("Frozen candidates · identical chronological DEV folds · lower MAE is better")
    render_page04_training_validation(st, root)
    try:
        manifest, tables = load_evidence(root)
        summary = tables["candidate_summary"]
        folds = tables["candidate_fold_metrics"]
    except (EvidenceContractError, KeyError, OSError, ValueError) as exc:
        st.error(f"Supplemental model evidence is unavailable: {exc}")
        st.code(
            f'PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "{root}"'
        )
        return

    audit = load_compatible_training_audit(root)
    render_historical_training_report(st, manifest, tables, audit, page="page04")

    ranked = summary.sort_values("validation_MAE_mean").reset_index(drop=True)
    winner = ranked.iloc[0]
    dummy_rows = summary[summary.model == "Dummy Median"]
    dummy_mae = float(dummy_rows.iloc[0].validation_MAE_mean) if len(dummy_rows) else None
    ranking = ranking_conclusion(summary)
    render_page_brief(
        st,
        question="Which frozen candidate performs best on chronological DEV validation, and how stable is that result?",
        evidence_scope=f"{len(summary)} candidates across {int(winner.effective_folds)} temporal folds · {manifest['evidence_id']}",
        takeaway=ranking["finding"],
        limitation=ranking["limit"],
    )
    _metrics(
        st,
        [
            ("Candidates", f"{len(summary)}"),
            ("Temporal folds", f"{int(winner.effective_folds)}"),
            ("Lowest CV MAE", str(winner.model)),
            ("Best mean CV MAE", money(winner.validation_MAE_mean)),
            ("Dummy reference", money(dummy_mae) if dummy_mae is not None else "Unavailable"),
        ],
    )
    st.caption(
        f"Supplemental evidence `{manifest['evidence_id']}`. Candidate CV rank does not replace the saved selected model."
    )
    render_metric_glossary(st, ["MAE", "RMSE", "MedAE", "R²", "CV"])

    membership = tables.get("fold_membership", pd.DataFrame())
    try:
        validation_guide = temporal_validation_guide(
            folds,
            membership,
            full_feature_count=len(manifest["full_features"]),
        )
    except ValueError as exc:
        validation_guide = None
        validation_guide_error = str(exc)
    with st.expander("Priority guide: How temporal validation works", expanded=False):
        if validation_guide is None:
            st.warning(f"Temporal-validation guide unavailable: {validation_guide_error}")
        else:
            st.markdown("**What is trained?**")
            st.markdown(
                f"Each of the **{validation_guide['candidate_count']} frozen candidate pipelines** "
                f"uses the same {validation_guide['full_feature_count']} raw salary inputs. Within "
                "every fold, preprocessing and the estimator are fitted only on that fold's training rows."
            )
            st.markdown("**How are folds constructed?**")
            st.markdown(validation_guide["method"])
            st.markdown(
                f"The active pack records **{validation_guide['fold_count']} folds** and "
                f"**{validation_guide['row_overlap_count']} train/validation row overlaps** within folds. "
                "The final validation block may contain the chronological remainder."
            )
            st.markdown("**Why use temporal validation?**")
            st.markdown(
                "Training on an earlier row block and evaluating on the next block better represents "
                "out-of-time salary prediction than randomly mixing records. Candidates are ranked by "
                "mean validation MAE—lower is better—while RMSE, MedAE and R² provide supporting views."
            )
            st.markdown("**Limits of this design**")
            st.markdown(validation_guide["limitations"])
            st.caption(
                "Shared calendar months across adjacent blocks do not imply shared records. This design "
                "does not establish future-market stability or causal salary relationships."
            )
        st.markdown("**Current supplemental evidence downloads**")
        _evidence_download(
            st,
            root,
            manifest,
            "Download fold metrics",
            "candidate_fold_metrics",
            key="p4_fold_metrics",
        )
        if len(membership):
            _evidence_download(
                st,
                root,
                manifest,
                "Download fold membership",
                "fold_membership",
                key="p4_fold_membership",
            )
        st.markdown("**Optional historical primary-pipeline audit**")
        _audit_downloads(st, audit)

    overall, rf_tab, gb_tab, baseline_tab = st.tabs(
        ["Overall comparison", "Random Forest", "Gradient Boosting", "Baseline controls"]
    )
    with overall:
        st.markdown(
            "**Question:** Which candidate has the lowest temporal-validation error, and how does it compare with training error and the Dummy reference?"
        )
        show_plot(st, candidate_comparison_figure(summary), "p4_train_validation")
        render_conclusion(st, ranking, title="Candidate ranking conclusion")
        st.markdown(TABLE_EMPHASIS_LEGEND)
        st.table(candidate_decision_table(summary), hide_index=True, border="horizontal")
        with st.container(width=chart_container_width("P2")):
            show_plot(st, ratio_r2_figure(summary), "p4_ratio_r2")
        table = ranked.copy()
        table.insert(0, "rank", range(1, len(table) + 1))
        table["role"] = [
            "Lowest CV MAE"
            if index == 0
            else "Reference floor"
            if row.model == "Dummy Median"
            else "Candidate"
            for index, row in table.iterrows()
        ]
        table["MAE_gap_usd"] = table.validation_MAE_mean - table.train_MAE_mean
        table["status"] = [baseline_status(row, dummy_mae) for _, row in table.iterrows()]
        columns = [
            "rank",
            "model",
            "role",
            "validation_MAE_mean",
            "train_MAE_mean",
            "MAE_gap_usd",
            "validation_R2_mean",
            "train_R2_mean",
            "validation_RMSE_mean",
            "validation_MedAE_mean",
            "fit_time_mean_s",
            "status",
        ]
        detail = table[columns].rename(
            columns={
                "rank": "Rank",
                "model": "Model",
                "role": "Decision role",
                "validation_MAE_mean": "Validation MAE (USD)",
                "train_MAE_mean": "Train MAE (USD)",
                "MAE_gap_usd": "MAE gap (USD)",
                "validation_R2_mean": "Validation R²",
                "train_R2_mean": "Train R²",
                "validation_RMSE_mean": "Validation RMSE (USD)",
                "validation_MedAE_mean": "Validation MedAE (USD)",
                "fit_time_mean_s": "Mean fit time (seconds)",
                "status": "Baseline status",
            }
        )
        st.dataframe(
            detail,
            hide_index=True,
            width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Model": st.column_config.TextColumn(pinned=True),
                "Validation MAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Train MAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "MAE gap (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Validation R²": st.column_config.NumberColumn(format="%.3f"),
                "Train R²": st.column_config.NumberColumn(format="%.3f"),
                "Validation RMSE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Validation MedAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Mean fit time (seconds)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        timeline = (
            folds[["fold_id", "validation_period", "train_rows", "validation_rows"]]
            .drop_duplicates()
            .sort_values("fold_id")
            .rename(
                columns={
                    "fold_id": "Fold",
                    "validation_period": "Validation period",
                    "train_rows": "Training rows",
                    "validation_rows": "Validation rows",
                }
            )
        )
        with st.container(width=chart_container_width("P3")):
            st.subheader("Recorded sliding-window timeline")
            st.dataframe(timeline, hide_index=True, width="stretch")
            st.caption(
                "Adjacent row blocks may share calendar months; row identities—not month labels alone—define each fold."
            )
        with st.expander("Additional comparison download"):
            _evidence_download(
                st,
                root,
                manifest,
                "Download candidate summary",
                "candidate_summary",
                key="p4_candidate_summary",
            )

    with rf_tab:
        st.markdown(
            "**Question:** How does Random Forest error change across the recorded temporal folds, and which period is weakest?"
        )
        if "Random Forest" not in set(folds.model):
            st.warning("Random Forest fold evidence is unavailable in this pack.")
        else:
            show_plot(st, fold_combo_figure(folds, "Random Forest"), "p4_rf_folds")
            render_conclusion(
                st,
                fold_stability_conclusion(folds, "Random Forest"),
                title="Temporal stability conclusion",
            )
            drift = tables.get("rf_importance_drift", pd.DataFrame())
            rf_drift = (
                drift[
                    (drift.model_id == "candidate:Random Forest") & (drift.scope == "raw_family")
                ].nlargest(12, "importance_mean")
                if not drift.empty
                else drift
            )
            if len(rf_drift):
                figure = go.Figure(
                    go.Scatter(
                        x=rf_drift.importance_mean,
                        y=rf_drift.importance_std,
                        mode="markers+text",
                        text=[
                            feature
                            if feature in set(rf_drift.nlargest(3, "importance_mean").feature)
                            else ""
                            for feature in rf_drift.feature
                        ],
                        textposition="top center",
                        marker=dict(size=10, color=EVIDENCE_COLORS["validation"]),
                        name="Raw-feature family",
                    )
                )
                figure.update_layout(
                    title="Raw-family reliance mean vs fold variation · Top 3 labelled",
                    xaxis=dict(title="Mean importance", automargin=True),
                    yaxis=dict(title="Population SD", automargin=True),
                    height=450,
                    margin=dict(l=85, r=40, t=80, b=70),
                )
                with st.container(width=chart_container_width("P3")):
                    show_plot(st, figure, "p4_rf_drift")
            rf = ranked[ranked.model == "Random Forest"].iloc[0]
            _metrics(
                st,
                [
                    ("Raw inputs", str(len(manifest["full_features"]))),
                    (
                        "Mean encoded columns",
                        f"{folds.loc[folds.model == 'Random Forest', 'encoded_feature_count'].mean():.0f}",
                    ),
                    ("Validation R²", f"{rf.validation_R2_mean:.3f}"),
                    ("Fold MAE SD", money(rf.validation_MAE_SD)),
                ],
            )
            st.info(
                "Fold gaps and reliance drift are diagnostics, not proof of no memorization or production fitness."
            )

    with gb_tab:
        st.markdown(
            "**Question:** Does Gradient Boosting match Random Forest across the same validation folds?"
        )
        needed = {"Gradient Boosting", "Random Forest"}
        if not needed <= set(folds.model):
            st.warning(
                "Gradient Boosting and Random Forest fold evidence are both required for this view."
            )
        else:
            data = folds[folds.model.isin(needed)].sort_values(["fold_id", "model"])
            figure = go.Figure()
            for model in ["Random Forest", "Gradient Boosting"]:
                current = data[data.model == model]
                figure.add_bar(
                    x=current.fold_id,
                    y=current.validation_MAE,
                    name=f"{model} validation",
                    text=[f"${v:,.0f}" for v in current.validation_MAE],
                    textposition="outside",
                    marker_color=(
                        EVIDENCE_COLORS["validation"]
                        if model == "Random Forest"
                        else EVIDENCE_COLORS["prediction"]
                    ),
                )
            gb = data[data.model == "Gradient Boosting"]
            figure.add_scatter(
                x=gb.fold_id,
                y=gb.train_MAE,
                name="Gradient Boosting train",
                mode="lines+markers+text",
                text=[
                    f"${value:,.0f}" if value == gb.train_MAE.max() else ""
                    for value in gb.train_MAE
                ],
                textposition="top center",
                line_color=EVIDENCE_COLORS["training"],
                marker_color=EVIDENCE_COLORS["training"],
            )
            figure.update_layout(
                title="Frozen candidates across identical folds",
                barmode="group",
                xaxis=dict(title="Fold", automargin=True),
                yaxis=dict(title="MAE (USD)", tickformat=",.0f", automargin=True),
                height=470,
                margin=dict(l=85, r=40, t=80, b=70),
                uniformtext=dict(minsize=10, mode="hide"),
            )
            show_plot(st, figure, "p4_gb_head")
            render_conclusion(
                st,
                fold_stability_conclusion(folds, "Gradient Boosting"),
                title="Gradient Boosting temporal conclusion",
            )
            med = ranked[ranked.model.isin(needed)][["model", "validation_MedAE_mean"]].rename(
                columns={
                    "model": "Model",
                    "validation_MedAE_mean": "Validation MedAE (USD)",
                }
            )
            st.dataframe(
                med,
                hide_index=True,
                width="stretch",
                column_config={
                    "Model": st.column_config.TextColumn(pinned=True),
                    "Validation MedAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                },
            )
            st.caption(
                "The chart compares recorded errors; it does not establish a single causal reason for either model's rank."
            )

    with baseline_tab:
        st.markdown(
            "**Question:** Which candidates clear the Dummy reference, and what do the linear controls show without assuming a root cause?"
        )
        names = [
            name
            for name in ["Linear Regression", "Ridge Regression", "Dummy Median", str(winner.model)]
            if name in set(summary.model)
        ]
        data = summary[summary.model.isin(names)].copy().sort_values("validation_MAE_mean")
        figure = go.Figure()
        figure.add_bar(
            x=data.model,
            y=data.validation_MAE_mean,
            name="Validation MAE",
            text=[f"${v:,.0f}" for v in data.validation_MAE_mean],
            textposition="outside",
            marker_color=EVIDENCE_COLORS["validation"],
        )
        figure.add_scatter(
            x=data.model,
            y=data.train_MAE_mean,
            name="Train MAE",
            mode="markers+lines+text",
            text=[
                f"${value:,.0f}"
                if value in {data.train_MAE_mean.min(), data.train_MAE_mean.max()}
                else ""
                for value in data.train_MAE_mean
            ],
            textposition="top center",
            line_color=EVIDENCE_COLORS["training"],
            marker_color=EVIDENCE_COLORS["training"],
        )
        if dummy_mae is not None:
            figure.add_hline(
                y=dummy_mae,
                line_dash="dash",
                line_color=EVIDENCE_COLORS["threshold"],
                annotation_text="Dummy reference",
            )
        figure.update_layout(
            title="Baseline controls and lowest-CV-MAE reference",
            xaxis=dict(automargin=True),
            yaxis=dict(title="MAE (USD)", tickformat=",.0f", automargin=True),
            height=460,
            margin=dict(l=85, r=40, t=80, b=105),
            uniformtext=dict(minsize=10, mode="hide"),
        )
        show_plot(st, figure, "p4_baselines")
        render_conclusion(st, ranking, title="Baseline comparison conclusion")
        baseline_display = data[
            ["model", "train_MAE_mean", "validation_MAE_mean", "validation_R2_mean"]
        ].rename(
            columns={
                "model": "Model",
                "train_MAE_mean": "Train MAE (USD)",
                "validation_MAE_mean": "Validation MAE (USD)",
                "validation_R2_mean": "Validation R²",
            }
        )
        st.dataframe(
            baseline_display,
            hide_index=True,
            width="stretch",
            column_config={
                "Model": st.column_config.TextColumn(pinned=True),
                "Train MAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Validation MAE (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Validation R²": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.warning(
            "A large train–validation gap is descriptive evidence. It does not by itself prove matrix singularity, memorization, or a universal overfit diagnosis."
        )
