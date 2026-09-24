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
                "page05_best_model.historical_primary_pipeline_audit_unavailable_value0_92206dd",
                value0=f"{audit.get('reason')}",
            )
        )
        return
    st.caption(
        _tr(
            "page05_best_model.historical_primary_pipeline_audit_pipeline_value0_audit_3f218ce",
            value0=f"{audit['pipeline_run_id']}",
            value1=f"{audit['audit_run_id']}",
            value2=f"{audit.get('selection_source') or 'unavailable'}",
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
            key=f"p5_audit_{index}",
            width="stretch",
        )


def _tuning_figure(frame: pd.DataFrame, parameter: str, title: str) -> go.Figure:
    ordered = frame.sort_values(
        parameter, key=lambda x: pd.to_numeric(x, errors="coerce").fillna(float("inf"))
    )
    figure = go.Figure()
    figure.add_bar(
        x=ordered[parameter].astype(str),
        y=ordered.CV_MAE,
        name=_src("page05_best_model.cv_mae_c21e027"),
        text=[f"${v:,.0f}" for v in ordered.CV_MAE],
        textposition="outside",
        marker_color=EVIDENCE_COLORS["validation"],
    )
    figure.add_scatter(
        x=ordered[parameter].astype(str),
        y=ordered.CV_R2,
        name=_src("model_training_presentation.cv_r_c40ab4a"),
        yaxis="y2",
        mode="lines+markers+text",
        text=[f"{value:.3f}" if value == ordered.CV_R2.max() else "" for value in ordered.CV_R2],
        textposition=_src("model_evidence.top_center_24b3167"),
        line_color=EVIDENCE_COLORS["development"],
        marker_color=EVIDENCE_COLORS["development"],
    )
    figure.update_layout(
        title=title,
        xaxis=dict(automargin=True),
        yaxis=dict(
            title=_src("page05_best_model.cv_mae_usd_32b4ebb"), tickformat=",.0f", automargin=True
        ),
        yaxis2=dict(
            title=_src("model_training_presentation.cv_r_c40ab4a"),
            overlaying="y",
            side="right",
            automargin=True,
        ),
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
        raise ValueError(
            _src("model_training_presentation.candidate_fold_evidence_is_incomplete_a76324e")
        )
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
        raise ValueError(
            _src("page05_best_model.candidate_models_disagree_on_temporal_fold_definitions_250c995")
        )
    definitions.insert(0, _src("page05_best_model.step_8e6a6cc"), range(1, len(definitions) + 1))
    definitions.insert(
        2,
        _src("page05_best_model.flow_f1273dc"),
        _src("page05_best_model.train_validation_abbf43d"),
    )
    return definitions.rename(
        columns={
            "fold_id": _src("model_evidence.fold_92c122b"),
            "train_period": _src("page05_best_model.train_period_bf75e4d"),
            "validation_period": _src("page04_model_comparison.validation_period_0073795"),
            "train_rows": _src("page05_best_model.train_rows_28a20a5"),
            "validation_rows": _src("page04_model_comparison.validation_rows_830d733"),
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

    with st.expander(
        _tr("page05_best_model.branch_b_how_the_temporal_dev_folds_5fd9c47"),
        expanded=False,
    ):
        st.caption(_tr("page05_best_model.historical_branch_b_evidence_this_is_the_7eed668"))
        if guide is None:
            st.warning(
                _tr(
                    "page05_best_model.branch_b_fold_guide_unavailable_value0_a1d97e1",
                    value0=f"{guide_error}",
                )
            )
            return

        with st.container(horizontal=True):
            st.metric(
                _tr("page04_model_comparison.temporal_folds_5228b93"),
                _display(guide["fold_count"]),
                border=True,
            )
            st.metric(
                _tr("page05_best_model.candidate_models_1ef330f"),
                _display(guide["candidate_count"]),
                border=True,
            )
            st.metric(
                _tr("page05_best_model.train_validation_overlap_a7cde9d"),
                _display(guide["row_overlap_count"]),
                border=True,
            )

        st.markdown(_tr("page05_best_model.how_the_fold_split_works_02228cf"))
        st.markdown(_display(guide["method"]))
        st.dataframe(
            display_frame(fold_table),
            hide_index=True,
            width="stretch",
            column_config=display_column_config(
                {
                    _src("page05_best_model.step_8e6a6cc"): st.column_config.NumberColumn(
                        format="%d"
                    ),
                    _src("page05_best_model.train_rows_28a20a5"): st.column_config.NumberColumn(
                        format="%d"
                    ),
                    _src(
                        "page04_model_comparison.validation_rows_830d733"
                    ): st.column_config.NumberColumn(format="%d"),
                }
            ),
        )

        st.markdown(_tr("page05_best_model.step_by_step_fold_execution_4e7b238"))
        for row in fold_table.to_dict("records"):
            st.markdown(
                _tr(
                    "page05_best_model.step_value0_value1_fit_preprocessing_model_on_2c029cc",
                    value0=f"{row['Step']}",
                    value1=f"{row['Fold']}",
                    value2=f"{row['Train period']}",
                    value3=f"{int(row['Train rows']):,}",
                    value4=f"{row['Validation period']}",
                    value5=f"{int(row['Validation rows']):,}",
                )
            )

        st.markdown(_display("**Limits**"))
        st.markdown(_display(guide["limitations"]))
        st.caption(_tr("page05_best_model.calendar_labels_can_meet_at_block_boundaries_af5a5a3"))


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("page05_best_model.5_best_model_diagnostics_and_uncertainty_64849bd"))
    st.caption(
        _tr("page05_best_model.saved_full_model_frozen_configuration_cv_historically_5e0ca70")
    )
    try:
        manifest, tables = load_evidence(root)
        variants = tables["variant_metrics"]
        predictions = tables["variant_test_predictions"]
    except (EvidenceContractError, KeyError, OSError, ValueError) as exc:
        st.error(
            _tr(
                "page05_best_model.supplemental_diagnostic_evidence_is_unavailable_value0_633c78a",
                value0=f"{exc}",
            )
        )
        st.code(
            f'PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "{root}"'
        )
        render_training_log_footer(st, load_compatible_training_audit(root), page="page05")
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
        question=_tr("page05_best_model.how_does_the_saved_full_model_generalize_1fcfb42"),
        evidence_scope=_tr(
            "page05_best_model.final_full_configuration_value0_historical_test_rows_37cde38",
            value0=f"{int(test.row_count)}",
            value1=f"{manifest['evidence_id']}",
        ),
        takeaway=generalization["finding"],
        limitation=generalization["limit"],
    )
    _metrics(
        st,
        [
            (
                _tr("page05_best_model.saved_model_c1a3996"),
                _tr(
                    "page05_best_model.random_forest_full_value0_4d98175",
                    value0=f"{len(manifest['full_features'])}",
                ),
            ),
            (_tr("page05_best_model.historical_test_r_ee278ef"), f"{test.R2:.3f}"),
            (_tr("page05_best_model.historical_test_mae_c62b6e1"), money(test.MAE)),
            (_tr("page05_best_model.historical_test_medae_3d1be05"), money(test.MedAE)),
            (_tr("page05_best_model.test_cv_r_52478f4"), f"{r2_gap:+.3f}"),
            (_tr("page05_best_model.empirical_q90_382dadb"), f"±{money(test.q90_abs_error_usd)}"),
        ],
    )
    test_period = manifest.get("datasets", {}).get("historical_test", {}).get("period", "unknown")
    st.caption(
        _tr(
            "page05_best_model.value0_historically_scored_rows_period_value1_evidence_bf87d6c",
            value0=f"{int(test.row_count)}",
            value1=f"{test_period}",
            value2=f"{manifest['evidence_id']}",
        )
    )
    render_metric_glossary(
        st,
        [
            "MAE",
            _src("model_evidence.medae_ad7020a"),
            "RMSE",
            "R²",
            "CV",
            _src("model_evidence.residual_0662144"),
            "q90",
        ],
    )

    general, diagnostics, tuning, reliance, trust = st.tabs(
        [
            _tr("page05_best_model.generalization_74faa5f"),
            _tr("page05_best_model.residual_diagnostics_e861f5c"),
            _tr("page05_best_model.tuning_2abeefb"),
            _tr("page05_best_model.feature_reliance_e220f83"),
            _tr("page05_best_model.trust_and_audit_c14675d"),
        ]
    )
    with general:
        st.markdown(
            _tr("page05_best_model.question_did_dollar_error_and_explained_variance_45765de")
        )
        show_plot(st, generalization_error_figure(cv, test), "p5_general_dollars")
        render_conclusion(
            st, generalization, title=_tr("page05_best_model.generalization_conclusion_c4a3994")
        )
        r2_figure = go.Figure()
        r2_figure.add_bar(
            x=[_src("model_evidence.historical_test_4241834")],
            y=[test.R2],
            name=_src("model_evidence.historical_test_4241834"),
            text=[f"{test.R2:.3f}"],
            textposition="outside",
            marker_color=EVIDENCE_COLORS["historical_test"],
        )
        r2_figure.add_scatter(
            x=[_src("model_evidence.dev_cv_2c5366a")],
            y=[cv.R2],
            name=_src("model_evidence.dev_cv_2c5366a"),
            mode="markers+text",
            text=[f"{cv.R2:.3f}"],
            textposition=_src("model_evidence.top_center_24b3167"),
            marker=dict(color=EVIDENCE_COLORS["development"], size=12, symbol="circle"),
        )
        r2_figure.update_layout(
            title=_src("page05_best_model.r_on_a_separate_unitless_scale_c8b51bd"),
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
                    _src("model_evidence.dev_cv_2c5366a"): float(cv[metric]),
                    _src("model_evidence.historical_test_4241834"): float(test[metric]),
                    "absolute_delta_test_minus_cv": float(test[metric] - cv[metric]),
                    "relative_delta_pct": 100
                    * float(test[metric] - cv[metric])
                    / abs(float(cv[metric]))
                    if float(cv[metric]) != 0
                    else None,
                }
                for metric in ["R2", "MAE", _src("model_evidence.medae_ad7020a"), "RMSE"]
            ]
        )
        scorecard[_src("page05_best_model.unit_4e54596")] = scorecard.metric.map(
            {
                "R2": "unitless",
                "MAE": "USD",
                _src("model_evidence.medae_ad7020a"): "USD",
                "RMSE": "USD",
            }
        )
        for column in [
            _src("model_evidence.dev_cv_2c5366a"),
            _src("model_evidence.historical_test_4241834"),
            "absolute_delta_test_minus_cv",
        ]:
            scorecard[column] = [
                f"{value:.3f}" if metric == "R2" else f"${value:,.0f}"
                for metric, value in zip(scorecard.metric, scorecard[column])
            ]
        scorecard["relative_delta_pct"] = scorecard.relative_delta_pct.map(
            lambda value: (
                _src("page04_model_comparison.unavailable_ca18449")
                if pd.isna(value)
                else f"{value:+.1f}%"
            )
        )
        scorecard = scorecard.rename(
            columns={
                "metric": _src("page01_data_basic_clean.metric_2d275a7"),
                "absolute_delta_test_minus_cv": _src("page05_best_model.test_cv_delta_8e4cc28"),
                "relative_delta_pct": _src("page05_best_model.relative_delta_9871ba4"),
            }
        )
        scorecard[_src("model_evidence.dev_cv_2c5366a")] = scorecard[
            _src("model_evidence.dev_cv_2c5366a")
        ].map(lambda value: f"*{value}*")
        for column in [
            _src("model_evidence.historical_test_4241834"),
            _src("page05_best_model.test_cv_delta_8e4cc28"),
            _src("page05_best_model.relative_delta_9871ba4"),
        ]:
            scorecard[column] = scorecard[column].map(lambda value: f"**{value}**")
        scorecard[_src("page05_best_model.unit_4e54596")] = scorecard[
            _src("page05_best_model.unit_4e54596")
        ].map(lambda value: f"*{value}*")
        st.markdown(_display(TABLE_EMPHASIS_LEGEND))
        st.table(display_frame(scorecard), hide_index=True, border="horizontal")
        st.info(_tr("page05_best_model.lower_error_is_an_error_reduction_not_a0f707a"))

    with diagnostics:
        st.markdown(
            _tr("page05_best_model.question_where_do_historical_predictions_depart_from_5feb308")
        )
        show_plot(st, actual_predicted_figure(predictions, "full:selected"), "p5_actual_predicted")
        full = predictions[predictions.model_id == "full:selected"]
        median_residual = float(full.residual_usd.median())
        render_conclusion(
            st,
            {
                "available": True,
                "finding": _tr(
                    "page05_best_model.the_median_historical_test_residual_actual_predicted_9e54749",
                    value0=f"{median_residual:,.0f}",
                    value1=f"{len(full)}",
                ),
                "why_it_matters": _tr(
                    "page05_best_model.residual_sign_shows_the_typical_direction_of_3fec370"
                ),
                "limit": _tr("page05_best_model.a_centered_median_or_visual_pattern_does_db1098f"),
                "decision_or_use": _tr(
                    "page05_best_model.inspect_the_residual_distribution_and_tail_metrics_c6329f0"
                ),
            },
            title=_tr("page05_best_model.residual_diagnostic_conclusion_fca1aec"),
        )
        residual = go.Figure(
            go.Histogram(
                x=full.residual_usd,
                nbinsx=30,
                name=_src("page05_best_model.historical_test_residuals_05add0a"),
                marker_color=EVIDENCE_COLORS["historical_test"],
                opacity=0.82,
            )
        )
        residual.add_vline(
            x=0, line_dash="dash", annotation_text=_src("page05_best_model.zero_973d0c6")
        )
        residual.update_layout(
            title=_src("page05_best_model.historical_test_residual_distribution_2305831"),
            xaxis=dict(
                title=_src("page05_best_model.actual_predicted_usd_953fe41"),
                tickformat=",.0f",
                automargin=True,
            ),
            yaxis=dict(title=_src("page01_data_basic_clean.records_47a84e9"), automargin=True),
            height=420,
            margin=dict(l=75, r=40, t=80, b=70),
        )
        with st.container(width=chart_container_width("P3")):
            show_plot(st, residual, "p5_residual")
        st.caption(
            _tr("page05_best_model.scatter_alignment_and_residual_shape_are_diagnostics_37b9020")
        )

    with tuning:
        st.markdown(
            _tr("page05_best_model.question_which_parameter_values_were_applied_and_ea12653")
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
        with st.expander(
            _tr("page05_best_model.priority_guide_how_hyperparameter_tuning_works_0d15d9c"),
            expanded=False,
        ):
            if tuning_guide is None:
                st.warning(
                    _tr(
                        "page05_best_model.tuning_methodology_guide_unavailable_value0_8382b8d",
                        value0=f"{tuning_guide_error}",
                    )
                )
            else:
                counts = tuning_guide["stage_trial_counts"]
                st.markdown(_tr("page05_best_model.what_is_tuned_c618f0c"))
                st.markdown(
                    _tr("page05_best_model.the_inherited_workflow_tunes_four_random_forest_8b467b5")
                )
                st.markdown(_tr("page05_best_model.how_does_the_search_run_31cd059"))
                st.markdown(
                    _tr(
                        "page05_best_model.it_first_compares_value0_anchor_configurations_then_7681f97",
                        value0=f"{tuning_guide['initial_trial_count']}",
                        value1=f"{counts['n_estimators']}",
                        value2=f"{counts['max_depth']}",
                        value3=f"{counts['min_samples_leaf']}",
                        value4=f"{counts['max_features']}",
                        value5=f"{tuning_guide['total_trial_count']}",
                        value6=f"{tuning_guide['effective_folds']}",
                    )
                )
                st.markdown(_tr("page05_best_model.why_tune_on_temporal_dev_folds_519af40"))
                st.markdown(
                    _tr(
                        "page05_best_model.the_procedure_compares_settings_on_later_chronological_c2a6edc"
                    )
                )
                st.markdown(_tr("page05_best_model.selection_rule_cd97737"))
                st.markdown(
                    _tr(
                        "page05_best_model.tuning_ranks_trials_by_value0_value1_this_c86362d",
                        value0=f"{tuning_guide['ranking_metric']}",
                        value1=f"{tuning_guide['ranking_direction']}",
                    )
                )
                st.markdown(_tr("page05_best_model.applied_configuration_c3ee3a3"))
                match_text = (
                    "matches"
                    if tuning_guide["saved_matches_initial_winner"]
                    else _src("page05_best_model.does_not_fully_match_4a0165d")
                )
                st.markdown(
                    _tr(
                        "page05_best_model.the_saved_model_value0_the_highest_r_a5eb22c",
                        value0=f"{match_text}",
                    )
                )
                st.markdown(_display("**Limitations**"))
                st.markdown(_tr("page05_best_model.this_is_an_inherited_non_nested_dev_b6822e1"))
            st.markdown(_tr("page05_best_model.current_tuning_evidence_downloads_a210ab9"))
            active_downloads = [
                (
                    _src("page05_best_model.download_initial_anchor_grid_1728d9f"),
                    "manual_tuning_step1_gridsearch",
                ),
                (
                    _src("page05_best_model.download_n_estimators_sweep_04ef92a"),
                    "manual_tuning_step2_n_estimators",
                ),
                (
                    _src("page05_best_model.download_max_depth_sweep_b8c7492"),
                    "manual_tuning_step3_max_depth",
                ),
                (
                    _src("page05_best_model.download_min_samples_leaf_sweep_28fd5bf"),
                    "manual_tuning_step4_min_samples_leaf",
                ),
                (
                    _src("page05_best_model.download_max_features_sweep_e324d73"),
                    "manual_tuning_step5_max_features",
                ),
                (
                    _src("page05_best_model.download_tuning_summary_d863330"),
                    "manual_tuning_4params_summary",
                ),
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
            st.markdown(
                _tr("page04_model_comparison.optional_historical_primary_pipeline_audit_6c2c311")
            )
            _audit_downloads(st, audit)
        stages = [
            (
                "manual_tuning_step1_gridsearch",
                "candidate",
                _src("page05_best_model.initial_grid_search_a1b2c3d"),
            ),
            (
                "manual_tuning_step2_n_estimators",
                "n_estimators",
                _src("page05_best_model.number_of_trees_40cd852"),
            ),
            (
                "manual_tuning_step3_max_depth",
                "max_depth",
                _src("page05_best_model.maximum_depth_5c34c5f"),
            ),
            (
                "manual_tuning_step4_min_samples_leaf",
                "min_samples_leaf",
                _src("page05_best_model.minimum_leaf_samples_1775f41"),
            ),
            (
                "manual_tuning_step5_max_features",
                "max_features",
                _src("page05_best_model.feature_subsampling_a17d9a7"),
            ),
        ]
        available = 0
        for stem, parameter, title in stages:
            frame = tables.get(stem)
            if frame is None or frame.empty or parameter not in frame:
                st.warning(
                    _tr(
                        "page05_best_model.value0_sensitivity_evidence_is_unavailable_f5cca02",
                        value0=f"{title}",
                    )
                )
                continue
            if stem == "manual_tuning_step1_gridsearch":
                st.subheader(_tr("page05_best_model.initial_gridsearch_title_a1b2c3e"))
            with st.container(width=chart_container_width("P2")):
                show_plot(st, _tuning_figure(frame, parameter, title), f"p5_tune_{parameter}")
            if stem == "manual_tuning_step1_gridsearch":
                st.dataframe(
                    display_frame(frame),
                    hide_index=True,
                    width="stretch",
                    column_config=display_column_config(
                        {
                            "candidate": st.column_config.NumberColumn(format="%d"),
                            "n_estimators": st.column_config.NumberColumn(format="%d"),
                            "min_samples_leaf": st.column_config.NumberColumn(format="%d"),
                            "max_features": st.column_config.NumberColumn(format="%.2f"),
                            "max_depth": st.column_config.NumberColumn(format="%.0f"),
                            "CV_R2": st.column_config.NumberColumn(format="%.3f"),
                            "CV_MAE": st.column_config.NumberColumn(format="$%.0f"),
                            "CV_MAE_SD": st.column_config.NumberColumn(format="$%.0f"),
                            "CV_RMSE": st.column_config.NumberColumn(format="$%.0f"),
                            "CV_MedAE": st.column_config.NumberColumn(format="$%.0f"),
                        }
                    ),
                )
            available += 1
        summary = tables.get("manual_tuning_4params_summary")
        render_conclusion(
            st,
            tuning_conclusion(summary, applied_parameters),
            title=_tr("page05_best_model.tuning_evidence_conclusion_362f96e"),
        )
        if summary is not None:
            st.markdown(_display(TABLE_EMPHASIS_LEGEND))
            st.table(
                display_frame(tuning_decision_table(summary, applied_parameters)),
                hide_index=True,
                border="horizontal",
            )
            tuning_table = summary[
                ["hyperparameter", "search_space", "optimal_value", "best_cv_r2", "best_cv_mae"]
            ].rename(
                columns={
                    "hyperparameter": _src("model_training_presentation.hyperparameter_5bc576c"),
                    "search_space": _src("page05_best_model.recorded_sweep_926115d"),
                    "optimal_value": _src("model_training_presentation.stage_winner_6d40f65"),
                    "best_cv_r2": _src("model_training_presentation.best_cv_r_2dc7204"),
                    "best_cv_mae": _src("page05_best_model.best_cv_mae_usd_f86366a"),
                }
            )
            st.dataframe(
                display_frame(tuning_table),
                hide_index=True,
                width="stretch",
                column_config=display_column_config(
                    {
                        _src(
                            "model_training_presentation.best_cv_r_2dc7204"
                        ): st.column_config.NumberColumn(format="%.3f"),
                        _src(
                            "page05_best_model.best_cv_mae_usd_f86366a"
                        ): st.column_config.NumberColumn(format="$%.0f"),
                    }
                ),
            )
        st.caption(
            _tr(
                "page05_best_model.value0_4_sensitivity_charts_available_sweep_winners_b040465",
                value0=f"{available}",
            )
        )

    with reliance:
        st.markdown(_tr("page05_best_model.question_which_raw_inputs_does_the_fitted_ac4e0a2"))
        permutation = tables.get("variant_permutation_importance", pd.DataFrame())
        if permutation.empty:
            st.warning(_tr("page05_best_model.raw_permutation_evidence_is_unavailable_e289181"))
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
                title=_src(
                    "page05_best_model.full_model_encoded_estimator_reliance_top_15_38acb0b"
                ),
                xaxis=dict(
                    title=_src("page05_best_model.impurity_importance_4581059"),
                    tickformat=".3f",
                    automargin=True,
                ),
                yaxis=dict(automargin=True),
                height=max(520, 30 * len(top) + 140),
                margin=dict(l=250, r=45, t=80, b=65),
                uniformtext=dict(minsize=10, mode="hide"),
            )
            with st.container(width=chart_container_width("P3")):
                show_plot(st, figure, "p5_encoded_importance")
        comparison = variants[variants.evaluation.isin(["dev_cv_mean", "historical_test"])][
            [
                "feature_variant",
                "evaluation",
                "row_count",
                "MAE",
                "RMSE",
                "R2",
                _src("model_evidence.medae_ad7020a"),
            ]
        ]
        st.subheader(_tr("page05_best_model.fixed_full_vs_top_2_feature_comparison_bf9b6f3"))
        comparison_display = comparison.rename(
            columns={
                "feature_variant": _src("page05_best_model.feature_set_94cdc96"),
                "evaluation": _src("page05_best_model.evaluation_163e44b"),
                "row_count": _src("source.rows"),
                "MAE": _src("page04_model_comparison.mae_usd_b27cf24"),
                "RMSE": _src("page05_best_model.rmse_usd_45938ee"),
                "R2": "R²",
                _src("model_evidence.medae_ad7020a"): _src("page05_best_model.medae_usd_bb89b91"),
            }
        )
        st.dataframe(
            display_frame(comparison_display),
            hide_index=True,
            width="stretch",
            column_config=display_column_config(
                {
                    _src("source.rows"): st.column_config.NumberColumn(format="%d"),
                    _src("page04_model_comparison.mae_usd_b27cf24"): st.column_config.NumberColumn(
                        format="$%.0f"
                    ),
                    _src("page05_best_model.rmse_usd_45938ee"): st.column_config.NumberColumn(
                        format="$%.0f"
                    ),
                    "R²": st.column_config.NumberColumn(format="%.3f"),
                    _src("page05_best_model.medae_usd_bb89b91"): st.column_config.NumberColumn(
                        format="$%.0f"
                    ),
                }
            ),
        )
        render_conclusion(
            st,
            variant_conclusion(variants),
            title=_tr("page05_best_model.feature_set_conclusion_115dfcb"),
        )
        st.warning(
            _tr("page05_best_model.importance_and_feature_ablation_describe_this_fitted_6f9aed8")
        )

    with trust:
        st.markdown(_tr("page05_best_model.question_how_wide_is_the_empirical_historical_72791dd"))
        coverage = float(test.coverage)
        full = predictions[predictions.model_id == "full:selected"]
        tail_ratio = float(test.RMSE / test.MedAE) if test.MedAE else None
        _metrics(
            st,
            [
                (
                    _tr("page05_best_model.empirical_q90_382dadb"),
                    f"±{money(test.q90_abs_error_usd)}",
                ),
                (_tr("page05_best_model.same_population_coverage_3a11089"), f"{coverage:.1%}"),
                (_tr("page05_best_model.coverage_denominator_1da322b"), f"{int(test.row_count)}"),
                (
                    _tr("page05_best_model.rmse_medae_9def26d"),
                    f"{tail_ratio:.1f}×"
                    if tail_ratio is not None
                    else _tr("page04_model_comparison.unavailable_ca18449"),
                ),
            ],
        )
        sorted_errors = full.absolute_error_usd.sort_values()
        ecdf = go.Figure(
            go.Scatter(
                x=sorted_errors,
                y=[(i + 1) / len(sorted_errors) for i in range(len(sorted_errors))],
                mode="lines",
                name=_src("page05_best_model.historical_test_empirical_cdf_d42fb3d"),
                line_color=EVIDENCE_COLORS["historical_test"],
            )
        )
        ecdf.add_vline(x=float(test.q90_abs_error_usd), line_dash="dash", annotation_text="q90")
        ecdf.update_layout(
            title=_src("page05_best_model.historical_test_absolute_error_distribution_bd1fdcf"),
            xaxis=dict(
                title=_src("page05_best_model.absolute_error_usd_c68c45a"),
                tickformat=",.0f",
                automargin=True,
            ),
            yaxis=dict(
                title=_src("page05_best_model.cumulative_share_153bff0"),
                tickformat=".0%",
                automargin=True,
            ),
            height=430,
            margin=dict(l=80, r=40, t=80, b=70),
        )
        with st.container(width=chart_container_width("P3")):
            show_plot(st, ecdf, "p5_ecdf")
        render_conclusion(
            st,
            uncertainty_conclusion(test),
            title=_tr("page05_best_model.uncertainty_conclusion_4c55330"),
        )
        examples = tables.get("benchmark_examples", pd.DataFrame())
        st.subheader(_tr("page05_best_model.three_traceable_historical_benchmark_examples_bc33163"))
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
                    "record_id": _src("page05_best_model.record_id_a36acea"),
                    "job_title": _src("page05_best_model.job_title_86db80a"),
                    "job_category": _src("page01_data_basic_clean.job_category_0175eef"),
                    "years_of_experience": _src("page05_best_model.experience_years_3453141"),
                    "annual_salary_usd": _src("page05_best_model.actual_salary_usd_442bdcb"),
                    "predicted_full_usd": _src("page05_best_model.full_prediction_usd_31e30a4"),
                    "predicted_top2_usd": _src("page05_best_model.top_2_prediction_usd_eef2a2b"),
                    "historically_exposed": _src("page05_best_model.historically_exposed_ae41cd2"),
                    "pristine": _src("page05_best_model.pristine_3ba5d85"),
                }
            )
            st.dataframe(
                display_frame(example_display),
                hide_index=True,
                width="stretch",
                column_config=display_column_config(
                    {
                        _src("page05_best_model.record_id_a36acea"): st.column_config.TextColumn(
                            pinned=True
                        ),
                        _src(
                            "page05_best_model.experience_years_3453141"
                        ): st.column_config.NumberColumn(format="%.1f"),
                        _src(
                            "page05_best_model.actual_salary_usd_442bdcb"
                        ): st.column_config.NumberColumn(format="$%.0f"),
                        _src(
                            "page05_best_model.full_prediction_usd_31e30a4"
                        ): st.column_config.NumberColumn(format="$%.0f"),
                        _src(
                            "page05_best_model.top_2_prediction_usd_eef2a2b"
                        ): st.column_config.NumberColumn(format="$%.0f"),
                    }
                ),
            )
        else:
            st.warning(
                _tr("page05_best_model.no_strict_policy_historical_examples_are_available_aa307ba")
            )
        st.warning(_tr("page05_best_model.the_q90_band_is_computed_and_checked_2b35fdd"))

    render_training_log_footer(st, audit, page="page05")
