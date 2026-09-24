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
)
from .model_training_presentation import (
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


def _sliding_window_diagram_html() -> str:
    """Illustrative 5-fold sliding-window layout (Train / Validation / Test).

    All block columns and the Test column render at the same width so the grid
    stays visually balanced regardless of cell text length.
    """
    train = '<span style="color:#2ea44f;">●</span> Train'
    valid = '<span style="color:#f38020;">●</span> Validation'
    empty = '<span style="color:#c9c3d6;">●</span> —'
    test = '<span style="color:#e5484d;">●</span> Test'
    rows = [
        ("Fold 1", [train, valid, empty, empty, empty, empty]),
        ("Fold 2", [empty, train, valid, empty, empty, empty]),
        ("Fold 3", [empty, empty, train, valid, empty, empty]),
        ("Fold 4", [empty, empty, empty, train, valid, empty]),
        ("Fold 5", [empty, empty, empty, empty, train, valid]),
    ]
    # Column widths: Fold label column narrow, then 7 equal columns for
    # Block 1..6 + Test so each cell renders at the same size.
    block_columns = 7
    fold_label_pct = 12
    block_pct = round((100 - fold_label_pct) / block_columns, 4)
    colgroup = (
        "<colgroup>"
        f'<col style="width:{fold_label_pct}%;">'
        + f'<col style="width:{block_pct}%;">' * block_columns
        + "</colgroup>"
    )
    th_style = (
        "padding:10px 8px;text-align:center;font-weight:700;"
        "border-bottom:2px solid #d0d0d8;background:#fafafa;"
    )
    th_label_style = th_style + "text-align:left;"
    td_style = (
        "padding:10px 8px;border-bottom:1px solid #eee;"
        "text-align:center;overflow:hidden;text-overflow:ellipsis;"
    )
    td_label_style = td_style + "text-align:left;"
    head = (
        "<tr>"
        f'<th style="{th_label_style}">Fold</th>'
        + "".join(f'<th style="{th_style}">Block {i}</th>' for i in range(1, 7))
        + f'<th style="{th_style}">Test</th>'
        + "</tr>"
    )
    body = "".join(
        "<tr>"
        + f'<td style="{td_label_style}"><b>{label}</b></td>'
        + "".join(f'<td style="{td_style}">{cell}</td>' for cell in cells)
        + f'<td style="{td_style}">{test}</td>'
        + "</tr>"
        for label, cells in rows
    )
    return (
        '<div style="margin:6px 0 14px 0;">'
        '<table style="border-collapse:collapse;font-size:13px;'
        'width:100%;table-layout:fixed;">'
        f"{colgroup}<thead>{head}</thead><tbody>{body}</tbody>"
        "</table></div>"
    )


def _render_sliding_window_block(st, folds, model_name=None):
    """Render the sliding-window subheader + diagram + timeline evidence table.

    When ``model_name`` is provided, the timeline evidence dataframe is filtered
    to that model's folds AND enriched with per-fold train/validation metrics
    (Train MAE, Validation MAE, Train R², Validation R²). When ``model_name`` is
    ``None`` (General/Overall or Baseline multi-model view), only the shared
    fold definition columns are shown.
    """
    if folds is None or getattr(folds, "empty", True):
        return
    df = folds if model_name is None else folds[folds.model == model_name]
    if df.empty:
        return
    st.subheader(_tr("page04_model_comparison.recorded_sliding_window_timeline_976bf7c"))
    col_diagram, col_table = st.columns(2, gap="medium")
    with col_diagram:
        st.markdown(_sliding_window_diagram_html(), unsafe_allow_html=True)
    with col_table:
        base_cols = ["fold_id", "validation_period", "train_rows", "validation_rows"]
        extra_cols = (
            [c for c in ("train_MAE", "validation_MAE", "train_R2", "validation_R2") if c in df.columns]
            if model_name is not None
            else []
        )
        select_cols = base_cols + extra_cols
        timeline_df = (
            df[select_cols]
            .drop_duplicates(subset=["fold_id"])
            .sort_values("fold_id")
            .rename(
                columns={
                    "fold_id": _src("model_evidence.fold_92c122b"),
                    "validation_period": _src("page04_model_comparison.validation_period_0073795"),
                    "train_rows": _src("page04_model_comparison.training_rows_3447c69"),
                    "validation_rows": _src("page04_model_comparison.validation_rows_830d733"),
                    "train_MAE": "Train MAE",
                    "validation_MAE": "Validation MAE",
                    "train_R2": "Train R²",
                    "validation_R2": "Validation R²",
                }
            )
        )
        column_config = {}
        for name, fmt in (
            ("Train MAE", "$%.0f"),
            ("Validation MAE", "$%.0f"),
            ("Train R²", "%.3f"),
            ("Validation R²", "%.3f"),
        ):
            if name in timeline_df.columns:
                column_config[name] = st.column_config.NumberColumn(format=fmt)
        st.dataframe(
            display_frame(timeline_df),
            hide_index=True,
            width="stretch",
            column_config=display_column_config(column_config) if column_config else None,
        )
    st.caption(
        _tr("page04_model_comparison.adjacent_row_blocks_may_share_calendar_months_d0d97f3")
    )


def _percent_ranking(summary, ranking):
    """Rewrite ranking['why_it_matters'] so the runner-up gap is expressed as a %."""
    if not ranking.get("available"):
        return ranking
    ordered = summary.dropna(subset=["validation_MAE_mean"]).sort_values("validation_MAE_mean")
    if len(ordered) < 2:
        return ranking
    winner_row = ordered.iloc[0]
    runner_row = ordered.iloc[1]
    runner_mae = float(runner_row.validation_MAE_mean)
    winner_mae = float(winner_row.validation_MAE_mean)
    if runner_mae <= 0:
        return ranking
    gap_pct = (runner_mae - winner_mae) / runner_mae * 100
    dummy_rows = summary[summary.model == _src("model_evidence.dummy_median_accc4f4")]
    dummy_pct = None
    if len(dummy_rows):
        dummy_mae = float(dummy_rows.iloc[0].validation_MAE_mean)
        if dummy_mae > 0:
            dummy_pct = (dummy_mae - winner_mae) / dummy_mae * 100
    parts = [f"It leads {runner_row.model} by {gap_pct:.1f}%"]
    if dummy_pct is not None:
        parts.append(f"and is {dummy_pct:.1f}% lower than the Dummy reference")
    ranking = dict(ranking)
    ranking["why_it_matters"] = " ".join(parts) + "."
    return ranking


def _render_ranking_conclusion_slim(st, ranking, title):
    """Compact conclusion box — finding + why-it-matters only (no limit/decision)."""
    with st.container(border=True):
        st.markdown(_display(f"**{title}**"))
        st.markdown(_tr("report.finding", value=_display(ranking["finding"])))
        st.markdown(
            _tr(
                "model_evidence.why_it_matters_value0_325039d",
                value0=_display(ranking["why_it_matters"]),
            )
        )


def _workflow_intro(st, manifest, summary):
    """Sections 1–2 rendered before the detail tabs — backed by live evidence."""
    st.divider()
    st.subheader("Workflow")

    # ---- Section 1: Target & Feature Engineering ----
    with st.expander(
        "1. " + _tr("page04_model_comparison.target_feature_engineering_title"),
        expanded=False,
    ):
        datasets = manifest.get("datasets", {}) or {}
        dev_rows = (datasets.get("development") or {}).get("rows")
        test_rows = (datasets.get("historical_test") or {}).get("rows")
        test_period = (datasets.get("historical_test") or {}).get("period")
        target_name = manifest.get("target") or "target"

        # Metric cards — target definition summary (TOP)
        with st.container(horizontal=True):
            st.metric("Target column", str(target_name), border=True)
            st.metric("Transformation", f"log1p({target_name})", border=True)
            if dev_rows is not None:
                st.metric("Development pool", f"{int(dev_rows):,} rows", border=True)
            if test_rows is not None:
                delta = f"period {test_period}" if test_period else None
                st.metric(
                    "Test set (out-of-sample)",
                    f"{int(test_rows):,} rows",
                    delta=delta,
                    delta_color="off",
                    border=True,
                )

        # Illustrative chart: right-skewed distribution vs log1p mapping (TOP)
        import math
        buckets = 24
        raw_bins = [10_000 + i * 10_000 for i in range(buckets)]
        raw_counts = [
            int(220 * math.exp(-((i - 3) ** 2) / 18) + 8) for i in range(buckets)
        ]
        log_bins = [round(math.log1p(x), 2) for x in raw_bins]
        skew_fig = go.Figure()
        skew_fig.add_bar(
            x=raw_bins,
            y=raw_counts,
            name="salary (raw USD)",
            marker_color="#f38020",
            yaxis="y1",
        )
        skew_fig.add_scatter(
            x=raw_bins,
            y=log_bins,
            name="log1p(salary)",
            mode="lines+markers",
            marker_color="#2ea44f",
            yaxis="y2",
        )
        skew_fig.update_layout(
            title="Right-skewed target and its log1p transform (illustrative)",
            xaxis=dict(title="salary (USD)", tickformat=",.0f", automargin=True),
            yaxis=dict(title="count of jobs", automargin=True),
            yaxis2=dict(
                title="log1p(salary)",
                overlaying="y",
                side="right",
                automargin=True,
            ),
            legend=dict(orientation="h", y=-0.25),
            height=340,
            margin=dict(l=70, r=70, t=60, b=70),
        )
        st.plotly_chart(skew_fig, use_container_width=True)
        st.caption(
            "Left axis (bars): synthetic right-skewed salary counts. "
            "Right axis (line): log1p mapping compresses the long tail into a near-linear range "
            "where residuals stabilise."
        )

        # Narrative (1. Target definition · 2. Scaling · 3. 5-Fold split) BELOW
        st.markdown(_tr("page04_model_comparison.target_feature_engineering_body"))

        st.info(
            "**Evidence · `"
            + str(manifest.get("evidence_id", ""))
            + "`** — target "
            + f"= `log1p({target_name})` "
            + (f"· Test period `{test_period}`" if test_period else "")
        )

    # ---- Section 2: 5 Models used ----
    with st.expander(
        "2. " + _tr("page04_model_comparison.models_roster_title"),
        expanded=False,
    ):
        st.markdown(_tr("page04_model_comparison.models_roster_body"))
        model_names = [str(m) for m in summary.model.tolist()]
        st.info(
            "**Evidence** — "
            + f"`candidate_summary` contains **{len(model_names)} candidates**: "
            + ", ".join(f"`{name}`" for name in model_names)
        )


def _workflow_reading_and_tuning(st, manifest, summary):
    """Deprecated — sections 4–5 removed; kept as no-op for call-site compatibility."""
    return


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
    ranking = _percent_ranking(summary, ranking_conclusion(summary))
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

    # Workflow (Target & FE + 5 candidate models)
    _workflow_intro(st, manifest, summary)

    overall, rf_tab, gb_tab, baseline_tab = st.tabs(
        [
            _tr("page04_model_comparison.overall_comparison_07e5f1e"),
            _tr("page04_model_comparison.random_forest_4c8e7b7"),
            _tr("page04_model_comparison.gradient_boosting_a8b554e"),
            _tr("page04_model_comparison.baseline_controls_00b0773"),
        ]
    )
    with overall:
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
        st.subheader(_tr("page04_model_comparison.recorded_sliding_window_timeline_976bf7c"))
        col_diagram, col_table = st.columns(2, gap="medium")
        with col_diagram:
            st.markdown(_sliding_window_diagram_html(), unsafe_allow_html=True)
        with col_table:
            st.dataframe(display_frame(timeline), hide_index=True, width="stretch")
        st.caption(
            _tr("page04_model_comparison.adjacent_row_blocks_may_share_calendar_months_d0d97f3")
        )
        col_error, col_ratio = st.columns(2, gap="medium")
        _row_height = 470
        with col_error:
            _fig_error = candidate_comparison_figure(summary)
            _fig_error.update_layout(height=_row_height)
            show_plot(st, _fig_error, "p4_train_validation")
        with col_ratio:
            _fig_ratio = ratio_r2_figure(summary)
            _fig_ratio.update_layout(height=_row_height)
            show_plot(st, _fig_ratio, "p4_ratio_r2")
        _render_ranking_conclusion_slim(
            st, ranking, title=_tr("page04_model_comparison.candidate_ranking_conclusion_4117e1d")
        )
        st.table(
            display_frame(candidate_decision_table(summary)), hide_index=True, border="horizontal"
        )
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
        if dummy_mae is not None and dummy_mae > 0:
            table["pct_lower_vs_dummy"] = (
                (dummy_mae - table.validation_MAE_mean) / dummy_mae * 100
            )
        else:
            table["pct_lower_vs_dummy"] = float("nan")
        table["status"] = [baseline_status(row, dummy_mae) for _, row in table.iterrows()]
        columns = [
            "rank",
            "model",
            "role",
            "validation_MAE_mean",
            "train_MAE_mean",
            "MAE_gap_usd",
            "validation_R2_mean",
            "pct_lower_vs_dummy",
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
                "pct_lower_vs_dummy": "% lower vs Dummy",
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
                    "% lower vs Dummy": st.column_config.NumberColumn(format="%.1f%%"),
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
            _render_sliding_window_block(
                st, folds, model_name=_src("page04_model_comparison.random_forest_4c8e7b7")
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
                with st.container(width=chart_container_width("P1")):
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
            _render_sliding_window_block(
                st, folds, model_name=_src("page04_model_comparison.gradient_boosting_a8b554e")
            )
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
        _render_ranking_conclusion_slim(
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
        _render_sliding_window_block(st, folds)

    # Workflow sections 4–5 (Reading results · RF tuning) with live evidence
    _workflow_reading_and_tuning(st, manifest, summary)

    render_training_log_footer(st, audit, page="page04")
