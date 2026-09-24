from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ai_job_market.ui_evidence_io import EvidenceContractError
from components.presentation import (
    _display,
    _label,
    _src,
    _tr,
    display_column_config,
    display_frame,
    translate_figure,
)

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
from .training_validation_presentation import render_training_log_footer


def _metrics(st, values):
    with st.container(horizontal=True):
        for label, value in values:
            st.metric(_display(label), _display(value), border=True)


def _evidence_download(st, root, manifest, label: str, stem: str, *, key: str):
    item = load_evidence_download(root, manifest, stem)
    st.download_button(
        _display(label),
        item["content"],
        file_name=item["filename"],
        mime=item["mime"],
        key=key,
        width="stretch",
    )


def _audit_downloads(st, audit: dict):
    if not audit.get("available"):
        st.caption(
            _tr(
                "page04_model_comparison.historical_primary_pipeline_audit_unavailable_value0_68e1139",
                value0=f"{audit.get('reason')}",
            )
        )
        return
    st.caption(
        _tr(
            "page04_model_comparison.historical_primary_pipeline_audit_pipeline_value0_audit_66d3725",
            value0=f"{audit['pipeline_run_id']}",
            value1=f"{audit['audit_run_id']}",
        )
    )
    for index, item in enumerate(audit["downloads"]):
        st.download_button(
            _tr(
                "page04_model_comparison.download_value0_0f5a93c",
                value0=f"{item['label'].lower()}",
            ),
            item["content"],
            file_name=item["filename"],
            mime=item["mime"],
            key=f"p4_audit_{index}",
            width="stretch",
        )


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("page04_model_comparison.4_model_comparison_and_temporal_validation_e1a441a"))
    st.caption(
        _tr(
            "page04_model_comparison.frozen_candidates_identical_chronological_dev_folds_lower_811f5c3"
        )
    )
    try:
        manifest, tables = load_evidence(root)
        summary = tables["candidate_summary"]
        folds = tables["candidate_fold_metrics"]
    except (EvidenceContractError, KeyError, OSError, ValueError) as exc:
        st.error(
            _tr(
                "page04_model_comparison.supplemental_model_evidence_is_unavailable_value0_77d852a",
                value0=f"{exc}",
            )
        )
        st.code(
            f'PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "{root}"'
        )
        render_training_log_footer(st, load_compatible_training_audit(root), page="page04")
        return

    audit = load_compatible_training_audit(root)

    ranked = summary.sort_values("validation_MAE_mean").reset_index(drop=True)
    winner = ranked.iloc[0]
    dummy_rows = summary[summary.model == _src("model_evidence.dummy_median_accc4f4")]
    dummy_mae = float(dummy_rows.iloc[0].validation_MAE_mean) if len(dummy_rows) else None
    ranking = ranking_conclusion(summary)
    render_page_brief(
        st,
        question=_tr(
            "page04_model_comparison.which_frozen_candidate_performs_best_on_chronological_cc4ea33"
        ),
        evidence_scope=_tr(
            "page04_model_comparison.value0_candidates_across_value1_temporal_folds_value2_02be45f",
            value0=f"{len(summary)}",
            value1=f"{int(winner.effective_folds)}",
            value2=f"{manifest['evidence_id']}",
        ),
        takeaway=ranking["finding"],
        limitation=ranking["limit"],
    )
    _metrics(
        st,
        [
            (_tr("page04_model_comparison.candidates_fdc9c0f"), f"{len(summary)}"),
            (
                _tr("page04_model_comparison.temporal_folds_5228b93"),
                f"{int(winner.effective_folds)}",
            ),
            (_tr("page04_model_comparison.lowest_cv_mae_c484200"), str(winner.model)),
            (
                _tr("page04_model_comparison.best_mean_cv_mae_51c940d"),
                money(winner.validation_MAE_mean),
            ),
            (
                _tr("page04_model_comparison.dummy_reference_3ae2fb6"),
                money(dummy_mae)
                if dummy_mae is not None
                else _tr("page04_model_comparison.unavailable_ca18449"),
            ),
        ],
    )
    st.caption(
        _tr(
            "page04_model_comparison.supplemental_evidence_value0_candidate_cv_rank_does_4472d18",
            value0=f"{manifest['evidence_id']}",
        )
    )
    render_metric_glossary(st, ["MAE", "RMSE", _src("model_evidence.medae_ad7020a"), "R²", "CV"])

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
    with st.expander(
        _tr("page04_model_comparison.priority_guide_how_temporal_validation_works_21b2cd0"),
        expanded=False,
    ):
        if validation_guide is None:
            st.warning(
                _tr(
                    "page04_model_comparison.temporal_validation_guide_unavailable_value0_2ee1e7c",
                    value0=f"{validation_guide_error}",
                )
            )
        else:
            st.markdown(_tr("page04_model_comparison.what_is_trained_129b215"))
            st.markdown(
                _tr(
                    "page04_model_comparison.each_of_the_value0_frozen_candidate_pipelines_65c3ad9",
                    value0=f"{validation_guide['candidate_count']}",
                    value1=f"{validation_guide['full_feature_count']}",
                )
            )
            st.markdown(_tr("page04_model_comparison.how_are_folds_constructed_660b8a1"))
            st.markdown(_display(validation_guide["method"]))
            st.markdown(
                _tr(
                    "page04_model_comparison.the_active_pack_records_value0_folds_and_a5bf2b4",
                    value0=f"{validation_guide['fold_count']}",
                    value1=f"{validation_guide['row_overlap_count']}",
                )
            )
            st.markdown(_tr("page04_model_comparison.why_use_temporal_validation_f226986"))
            st.markdown(_tr("page04_model_comparison.training_on_an_earlier_row_block_and_622a729"))
            st.markdown(_tr("page04_model_comparison.limits_of_this_design_7d621c4"))
            st.markdown(_display(validation_guide["limitations"]))
            st.caption(
                _tr(
                    "page04_model_comparison.shared_calendar_months_across_adjacent_blocks_do_3579a54"
                )
            )
        st.markdown(_tr("page04_model_comparison.current_supplemental_evidence_downloads_13e3246"))
        _evidence_download(
            st,
            root,
            manifest,
            _src("page04_model_comparison.download_fold_metrics_5f1e42c"),
            "candidate_fold_metrics",
            key="p4_fold_metrics",
        )
        if len(membership):
            _evidence_download(
                st,
                root,
                manifest,
                _src("page04_model_comparison.download_fold_membership_12f5669"),
                "fold_membership",
                key="p4_fold_membership",
            )
        st.markdown(
            _tr("page04_model_comparison.optional_historical_primary_pipeline_audit_6c2c311")
        )
        _audit_downloads(st, audit)

    overall, rf_tab, gb_tab, baseline_tab = st.tabs(
        [
            _tr("page04_model_comparison.overall_comparison_07e5f1e"),
            _tr("page04_model_comparison.random_forest_4c8e7b7"),
            _tr("page04_model_comparison.gradient_boosting_a8b554e"),
            _tr("page04_model_comparison.baseline_controls_00b0773"),
        ]
    )
    with overall:
        st.markdown(
            _tr("page04_model_comparison.question_which_candidate_has_the_lowest_temporal_53be758")
        )
        timeline = (
            folds[["fold_id", "validation_period", "train_rows", "validation_rows"]]
            .drop_duplicates()
            .sort_values("fold_id")
            .rename(
                columns={
                    "fold_id": _src("model_evidence.fold_92c122b"),
                    "validation_period": _src("page04_model_comparison.validation_period_0073795"),
                    "train_rows": _src("page04_model_comparison.training_rows_3447c69"),
                    "validation_rows": _src("page04_model_comparison.validation_rows_830d733"),
                }
            )
        )
        with st.container(width=chart_container_width("P3")):
            st.subheader(_tr("page04_model_comparison.recorded_sliding_window_timeline_976bf7c"))
            st.dataframe(display_frame(timeline), hide_index=True, width="stretch")
            st.caption(
                _tr("page04_model_comparison.adjacent_row_blocks_may_share_calendar_months_d0d97f3")
            )
        show_plot(st, candidate_comparison_figure(summary), "p4_train_validation")
        render_conclusion(
            st, ranking, title=_tr("page04_model_comparison.candidate_ranking_conclusion_4117e1d")
        )
        st.markdown(_display(TABLE_EMPHASIS_LEGEND))
        st.table(
            display_frame(candidate_decision_table(summary)), hide_index=True, border="horizontal"
        )
        with st.container(width=chart_container_width("P2")):
            show_plot(st, ratio_r2_figure(summary), "p4_ratio_r2")
        table = ranked.copy()
        table.insert(0, "rank", range(1, len(table) + 1))
        table["role"] = [
            _src("page04_model_comparison.lowest_cv_mae_c484200")
            if index == 0
            else _src("page04_model_comparison.reference_floor_0161736")
            if row.model == _src("model_evidence.dummy_median_accc4f4")
            else _src("model_training_presentation.candidate_b2452d1")
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
                "rank": _src("model_training_presentation.rank_a4130d7"),
                "model": _src("model_training_presentation.model_5e2c614"),
                "role": _src("model_training_presentation.decision_role_b8dd1f9"),
                "validation_MAE_mean": _src("page04_model_comparison.validation_mae_usd_7ea40f0"),
                "train_MAE_mean": _src("page04_model_comparison.train_mae_usd_6583524"),
                "MAE_gap_usd": _src("page04_model_comparison.mae_gap_usd_52685dc"),
                "validation_R2_mean": _src("model_evidence.validation_r_538b15c"),
                "train_R2_mean": _src("page04_model_comparison.train_r_f7e3d67"),
                "validation_RMSE_mean": _src("page04_model_comparison.validation_rmse_usd_16f223d"),
                "validation_MedAE_mean": _src(
                    "page04_model_comparison.validation_medae_usd_86e1f99"
                ),
                "fit_time_mean_s": _src("page04_model_comparison.mean_fit_time_seconds_6d6846f"),
                "status": _src("page04_model_comparison.baseline_status_78aec7d"),
            }
        )
        st.dataframe(
            display_frame(detail),
            hide_index=True,
            width="stretch",
            column_config=display_column_config(
                {
                    _src("model_training_presentation.rank_a4130d7"): st.column_config.NumberColumn(
                        format="%d"
                    ),
                    _src("model_training_presentation.model_5e2c614"): st.column_config.TextColumn(
                        pinned=True
                    ),
                    _src(
                        "page04_model_comparison.validation_mae_usd_7ea40f0"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page04_model_comparison.train_mae_usd_6583524"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page04_model_comparison.mae_gap_usd_52685dc"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src("model_evidence.validation_r_538b15c"): st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    _src("page04_model_comparison.train_r_f7e3d67"): st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                    _src(
                        "page04_model_comparison.validation_rmse_usd_16f223d"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page04_model_comparison.validation_medae_usd_86e1f99"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page04_model_comparison.mean_fit_time_seconds_6d6846f"
                    ): st.column_config.NumberColumn(format="%.3f"),
                }
            ),
        )
        with st.expander(_tr("page04_model_comparison.additional_comparison_download_83bd7e5")):
            _evidence_download(
                st,
                root,
                manifest,
                _src("page04_model_comparison.download_candidate_summary_03e008a"),
                "candidate_summary",
                key="p4_candidate_summary",
            )

    with rf_tab:
        st.markdown(
            _tr("page04_model_comparison.question_how_does_random_forest_error_change_b7e566d")
        )
        if _src("page04_model_comparison.random_forest_4c8e7b7") not in set(folds.model):
            st.warning(
                _tr("page04_model_comparison.random_forest_fold_evidence_is_unavailable_in_717271f")
            )
        else:
            show_plot(
                st,
                fold_combo_figure(folds, _src("page04_model_comparison.random_forest_4c8e7b7")),
                "p4_rf_folds",
            )
            render_conclusion(
                st,
                fold_stability_conclusion(
                    folds, _src("page04_model_comparison.random_forest_4c8e7b7")
                ),
                title=_tr("page04_model_comparison.temporal_stability_conclusion_7cde15d"),
            )
            drift = tables.get("rf_importance_drift", pd.DataFrame())
            rf_drift = (
                drift[
                    (
                        drift.model_id
                        == _src("page04_model_comparison.candidate_random_forest_30b624d")
                    )
                    & (drift.scope == "raw_family")
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
                        textposition=_src("model_evidence.top_center_24b3167"),
                        marker=dict(size=10, color=EVIDENCE_COLORS["validation"]),
                        name=_src("page04_model_comparison.raw_feature_family_5a3f5f1"),
                    )
                )
                figure.update_layout(
                    title=_src(
                        "page04_model_comparison.raw_family_reliance_mean_vs_fold_variation_0155282"
                    ),
                    xaxis=dict(
                        title=_src("page04_model_comparison.mean_importance_7f3dcab"),
                        automargin=True,
                    ),
                    yaxis=dict(
                        title=_src("page04_model_comparison.population_sd_ebb7610"), automargin=True
                    ),
                    height=450,
                    margin=dict(l=85, r=40, t=80, b=70),
                )
                with st.container(width=chart_container_width("P3")):
                    show_plot(st, figure, "p4_rf_drift")
            rf = ranked[ranked.model == _src("page04_model_comparison.random_forest_4c8e7b7")].iloc[
                0
            ]
            _metrics(
                st,
                [
                    (
                        _tr("page03_segmentation.raw_inputs_fca9033"),
                        str(len(manifest["full_features"])),
                    ),
                    (
                        _tr("page04_model_comparison.mean_encoded_columns_df262e5"),
                        f"{folds.loc[folds.model == 'Random Forest', 'encoded_feature_count'].mean():.0f}",
                    ),
                    (_tr("model_evidence.validation_r_538b15c"), f"{rf.validation_R2_mean:.3f}"),
                    (
                        _tr("page04_model_comparison.fold_mae_sd_4c9a096"),
                        money(rf.validation_MAE_SD),
                    ),
                ],
            )
            st.info(
                _tr("page04_model_comparison.fold_gaps_and_reliance_drift_are_diagnostics_66b69c9")
            )

    with gb_tab:
        st.markdown(
            _tr(
                "page04_model_comparison.question_does_gradient_boosting_match_random_forest_8136d1c"
            )
        )
        needed = {
            _src("page04_model_comparison.gradient_boosting_a8b554e"),
            _src("page04_model_comparison.random_forest_4c8e7b7"),
        }
        if not needed <= set(folds.model):
            st.warning(
                _tr(
                    "page04_model_comparison.gradient_boosting_and_random_forest_fold_evidence_7077361"
                )
            )
        else:
            data = folds[folds.model.isin(needed)].sort_values(["fold_id", "model"])
            figure = go.Figure()
            for model in [
                _src("page04_model_comparison.random_forest_4c8e7b7"),
                _src("page04_model_comparison.gradient_boosting_a8b554e"),
            ]:
                current = data[data.model == model]
                figure.add_bar(
                    x=current.fold_id,
                    y=current.validation_MAE,
                    name=_tr("report.validation", value=_label(model)),
                    text=[f"${v:,.0f}" for v in current.validation_MAE],
                    textposition="outside",
                    marker_color=(
                        EVIDENCE_COLORS["validation"]
                        if model == _src("page04_model_comparison.random_forest_4c8e7b7")
                        else EVIDENCE_COLORS["prediction"]
                    ),
                )
            gb = data[data.model == _src("page04_model_comparison.gradient_boosting_a8b554e")]
            figure.add_scatter(
                x=gb.fold_id,
                y=gb.train_MAE,
                name=_src("page04_model_comparison.gradient_boosting_train_2f1a65e"),
                mode="lines+markers+text",
                text=[
                    f"${value:,.0f}" if value == gb.train_MAE.max() else ""
                    for value in gb.train_MAE
                ],
                textposition=_src("model_evidence.top_center_24b3167"),
                line_color=EVIDENCE_COLORS["training"],
                marker_color=EVIDENCE_COLORS["training"],
            )
            figure.update_layout(
                title=_src(
                    "page04_model_comparison.frozen_candidates_across_identical_folds_92f42c6"
                ),
                barmode="group",
                xaxis=dict(title=_src("model_evidence.fold_92c122b"), automargin=True),
                yaxis=dict(
                    title=_src("page04_model_comparison.mae_usd_b27cf24"),
                    tickformat=",.0f",
                    automargin=True,
                ),
                height=470,
                margin=dict(l=85, r=40, t=80, b=70),
                uniformtext=dict(minsize=10, mode="hide"),
            )
            show_plot(st, figure, "p4_gb_head")
            render_conclusion(
                st,
                fold_stability_conclusion(
                    folds, _src("page04_model_comparison.gradient_boosting_a8b554e")
                ),
                title=_tr("page04_model_comparison.gradient_boosting_temporal_conclusion_c17b90c"),
            )
            med = ranked[ranked.model.isin(needed)][["model", "validation_MedAE_mean"]].rename(
                columns={
                    "model": _src("model_training_presentation.model_5e2c614"),
                    "validation_MedAE_mean": _src(
                        "page04_model_comparison.validation_medae_usd_86e1f99"
                    ),
                }
            )
            st.dataframe(
                display_frame(med),
                hide_index=True,
                width="stretch",
                column_config=display_column_config(
                    {
                        _src(
                            "model_training_presentation.model_5e2c614"
                        ): st.column_config.TextColumn(pinned=True),
                        _src(
                            "page04_model_comparison.validation_medae_usd_86e1f99"
                        ): st.column_config.NumberColumn(format="$%.0f"),
                    }
                ),
            )
            st.caption(
                _tr("page04_model_comparison.the_chart_compares_recorded_errors_it_does_035355d")
            )

    with baseline_tab:
        st.markdown(
            _tr(
                "page04_model_comparison.question_which_candidates_clear_the_dummy_reference_f6bb21e"
            )
        )
        names = [
            name
            for name in [
                _src("page04_model_comparison.linear_regression_5c0d967"),
                _src("page04_model_comparison.ridge_regression_1be6652"),
                _src("model_evidence.dummy_median_accc4f4"),
                str(winner.model),
            ]
            if name in set(summary.model)
        ]
        data = summary[summary.model.isin(names)].copy().sort_values("validation_MAE_mean")
        figure = go.Figure()
        figure.add_bar(
            x=data.model,
            y=data.validation_MAE_mean,
            name=_src("model_evidence.validation_mae_c5a1c85"),
            text=[f"${v:,.0f}" for v in data.validation_MAE_mean],
            textposition="outside",
            marker_color=EVIDENCE_COLORS["validation"],
        )
        figure.add_scatter(
            x=data.model,
            y=data.train_MAE_mean,
            name=_src("model_evidence.train_mae_8ffd76c"),
            mode="markers+lines+text",
            text=[
                f"${value:,.0f}"
                if value in {data.train_MAE_mean.min(), data.train_MAE_mean.max()}
                else ""
                for value in data.train_MAE_mean
            ],
            textposition=_src("model_evidence.top_center_24b3167"),
            line_color=EVIDENCE_COLORS["training"],
            marker_color=EVIDENCE_COLORS["training"],
        )
        if dummy_mae is not None:
            figure.add_hline(
                y=dummy_mae,
                line_dash="dash",
                line_color=EVIDENCE_COLORS["threshold"],
                annotation_text=_src("page04_model_comparison.dummy_reference_3ae2fb6"),
            )
        figure.update_layout(
            title=_src(
                "page04_model_comparison.baseline_controls_and_lowest_cv_mae_reference_d1ff6d8"
            ),
            xaxis=dict(automargin=True),
            yaxis=dict(
                title=_src("page04_model_comparison.mae_usd_b27cf24"),
                tickformat=",.0f",
                automargin=True,
            ),
            height=460,
            margin=dict(l=85, r=40, t=80, b=105),
            uniformtext=dict(minsize=10, mode="hide"),
        )
        show_plot(st, figure, "p4_baselines")
        render_conclusion(
            st, ranking, title=_tr("page04_model_comparison.baseline_comparison_conclusion_625c3c6")
        )
        baseline_display = data[
            ["model", "train_MAE_mean", "validation_MAE_mean", "validation_R2_mean"]
        ].rename(
            columns={
                "model": _src("model_training_presentation.model_5e2c614"),
                "train_MAE_mean": _src("page04_model_comparison.train_mae_usd_6583524"),
                "validation_MAE_mean": _src("page04_model_comparison.validation_mae_usd_7ea40f0"),
                "validation_R2_mean": _src("model_evidence.validation_r_538b15c"),
            }
        )
        st.dataframe(
            display_frame(baseline_display),
            hide_index=True,
            width="stretch",
            column_config=display_column_config(
                {
                    _src("model_training_presentation.model_5e2c614"): st.column_config.TextColumn(
                        pinned=True
                    ),
                    _src(
                        "page04_model_comparison.train_mae_usd_6583524"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page04_model_comparison.validation_mae_usd_7ea40f0"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src("model_evidence.validation_r_538b15c"): st.column_config.NumberColumn(
                        format="%.3f"
                    ),
                }
            ),
        )
        st.warning(
            _tr("page04_model_comparison.a_large_train_validation_gap_is_descriptive_c0b3c8e")
        )

    render_training_log_footer(st, audit, page="page04")
