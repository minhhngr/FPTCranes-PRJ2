from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.presentation import (
    _display,
    _label,
    _option_label,
    _src,
    _tr,
    display_column_config,
    display_frame,
    translate_figure,
)

from .common import (
    downloadable_table,
    interpretation_card,
    metric_cols,
    read_csv,
    read_json,
    show_plot,
    style_page,
)

SEG_DIR = "03_ai_job_market_segmentation"

# Branch-A output contract.  Streamlit is an evidence renderer only: it reads
# these persisted artifacts and never refits PCA / KMeans / GMM in the UI.
EVIDENCE_NAMES = [
    "family_dimensions",
    "family_balance_diagnostics",
    "pca_variance",
    "cluster_evaluation",
    "resample_stability_runs",
    "candidate_cluster_assignments",
    "candidate_cluster_coordinates",
    "candidate_cluster_balance",
    "cluster_assignments",
    "cluster_profiles",
    "cluster_job_category_profile",
    "cluster_skill_profile",
    "cluster_experience_level_profile",
    "cluster_years_distribution",
    "cluster_company_profile",
    "cluster_geography_profile",
    "cluster_market_signal_values",
    "country_cluster_profile",
    "city_cluster_profile",
    # O1 vs O2 feature-space construction evidence
    "feature_space_option_summary",
    "feature_space_candidate_metrics",
    "feature_space_pairwise_ari",
    "correlation_feature_selection",
    "correlation_redundancy_pairs",
    "correlation_family_summary",
    "o2_candidate_metrics",
    "o2_resample_stability_runs",
    "o2_candidate_assignments",
    "o2_visual_coordinates",
    # Representation-robustness evidence (R0/R1/R2/R3/R4 when available)
    "representation_summary",
    "representation_comparison",
    "representation_pairwise_ari",
    "representation_candidate_metrics",
    "representation_pca_variance",
    "representation_top_loadings",
    "representation_candidate_assignments",
    "representation_resample_stability_runs",
    "representation_selected_assignments",
    "selected_representation_family_pca_summary",
    "feature_dependency_summary",
]


FAMILY_CONTRACT = [
    {
        "family": _src("common.job_domain_9e966d1"),
        "raw_fields": _src("page03_segmentation.job_title_job_category_771e8f1"),
        "representation": _src("page03_segmentation.nominal_one_hot_block_dcf92bd"),
    },
    {
        "family": _src("page03_segmentation.experience_education_5265686"),
        "raw_fields": _src("page03_segmentation.years_of_experience_education_required_d12e3c2"),
        "representation": _src("page03_segmentation.scaled_numeric_nominal_one_hot_9f5ce01"),
    },
    {
        "family": _src("page03_segmentation.company_work_mode_4ec2677"),
        "raw_fields": _src("page03_segmentation.remote_work_company_size_industry_5788ff6"),
        "representation": _src("page03_segmentation.nominal_one_hot_block_dcf92bd"),
    },
    {
        "family": _src("common.geography_f3c7380"),
        "raw_fields": _src("page03_segmentation.city_country_c86a9ba"),
        "representation": _src("page03_segmentation.nominal_one_hot_block_dcf92bd"),
    },
    {
        "family": _src("common.demand_benefits_e58ed2c"),
        "raw_fields": _src("page03_segmentation.demand_score_benefits_score_10_e8d88d5"),
        "representation": _src("page03_segmentation.scaled_numeric_block_8688162"),
    },
    {
        "family": _src("common.skills_66d0f52"),
        "raw_fields": _src("page03_segmentation.required_skills_skill_count_6ec1f63"),
        "representation": _src(
            "page03_segmentation.normalized_multi_hot_skill_tokens_scaled_skill_99acba5"
        ),
    },
]


def _read_all_evidence(root) -> dict[str, pd.DataFrame]:
    """Load persisted Branch-A evidence and keep optional artifacts non-fatal."""
    out: dict[str, pd.DataFrame] = {}
    for name in EVIDENCE_NAMES:
        try:
            out[name] = read_csv(root, f"{SEG_DIR}/{name}.csv")
        except Exception:
            out[name] = pd.DataFrame()
    return out


def _filter_rows(df: pd.DataFrame, clusters, categories, countries) -> pd.DataFrame:
    """UI-only row filtering; no model fitting or feature derivation is performed."""
    out = df.copy()
    if out.empty:
        return out
    if clusters and "cluster" in out.columns:
        out = out[out["cluster"].astype(int).isin(list(map(int, clusters)))]
    if categories and "job_category" in out.columns:
        out = out[out["job_category"].astype(str).isin(categories)]
    if countries and "country" in out.columns:
        out = out[out["country"].astype(str).isin(countries)]
    return out


def _show_saved_insight(st, insights: dict, key: str, title: str = _src("interpretation.title")):
    card = insights.get(key, {})
    interpretation_card(
        st,
        card.get(
            "observed",
            _src("page03_segmentation.no_run_level_interpretation_was_persisted_3323e1c"),
        ),
        card.get(
            "interpretation",
            _src("page03_segmentation.re_run_the_offline_pipeline_to_refresh_98cfbe3"),
        ),
        card.get("action"),
        card.get("tone", "info"),
        title=title,
    )


def _available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _fmt_value(value, digits: int = 3, pct: bool = False, default: str = "—") -> str:
    try:
        if pd.isna(value):
            return default
        if pct:
            return f"{100 * float(value):.{digits}f}%"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value) if value is not None else default


def _render_missing(st, name: str) -> None:
    st.warning(
        _tr(
            "page03_segmentation.value0_is_not_available_in_the_active_b02be72",
            value0=f"{name}",
        )
    )


def _render_representation_detail(
    st,
    rep_id: str,
    rep_row: pd.Series,
    candidate_metrics: pd.DataFrame,
    pca_all: pd.DataFrame,
    loadings_all: pd.DataFrame,
    rep_assignments: pd.DataFrame,
    rationale: dict,
) -> None:
    """Show complete persisted evidence for one representation R0/R1/R2/R3."""
    st.markdown(f"#### {rep_id} — {_display(rep_row.get('representation_label', ''))}")
    purpose = rep_row.get("purpose", "")
    if pd.notna(purpose) and str(purpose).strip():
        st.caption(_display(str(purpose)))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(
        _tr("page03_segmentation.algorithm_d704d8a"),
        _display(str(rep_row.get("algorithm", "—"))),
    )
    c2.metric(
        _tr("page03_segmentation.selected_k_df75501"),
        _display(str(int(rep_row["k"])) if pd.notna(rep_row.get("k")) else "—"),
    )
    c3.metric(
        _tr("page03_segmentation.silhouette_af465c1"),
        _display(_fmt_value(rep_row.get("silhouette"))),
    )
    c4.metric(
        _tr("page03_segmentation.seed_ari_dd87010"),
        _display(_fmt_value(rep_row.get("stability_ari"))),
    )
    c5.metric(
        _tr("page03_segmentation.subsample_ari_05527db"),
        _display(_fmt_value(rep_row.get("resample_stability_ari_mean"))),
    )
    c6.metric(
        _tr("page03_segmentation.min_cluster_3ed65ac"),
        _display(_fmt_value(rep_row.get("min_cluster_share"), digits=1, pct=True)),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        _tr("page03_segmentation.raw_inputs_fca9033"),
        _display(
            str(int(rep_row["raw_input_count"]))
            if pd.notna(rep_row.get("raw_input_count"))
            else "—"
        ),
    )
    c2.metric(
        _tr("page03_segmentation.latent_dimensions_251c35d"),
        _display(
            str(int(rep_row["latent_dimensions"]))
            if pd.notna(rep_row.get("latent_dimensions"))
            else "—"
        ),
    )
    c3.metric(
        _tr("page03_segmentation.variance_summary_a016821"),
        _display(_fmt_value(rep_row.get("variance_retained_summary"), digits=1, pct=True)),
    )
    c4.metric(
        _tr("page03_segmentation.cross_rep_ari_69f85f3"),
        _display(_fmt_value(rep_row.get("mean_cross_representation_ari"))),
    )

    excluded = str(
        rep_row.get("excluded_features", _src("model_training_presentation.none_dc937b5"))
    )
    st.markdown(
        _tr("page03_segmentation.excluded_raw_feature_s_value0_48374b3", value0=f"{excluded}")
    )
    selected_inputs = rep_row.get("selected_input_features", "")
    if pd.notna(selected_inputs) and str(selected_inputs).strip():
        with st.expander(
            _tr("page03_segmentation.raw_input_contract_for_this_representation_53bc89d"),
            expanded=False,
        ):
            st.write(_display(str(selected_inputs)))

    # ---- Full K=2..8 candidate evidence for this representation ----
    cm = candidate_metrics.copy()
    if not cm.empty and "representation_id" in cm.columns:
        cm = cm[cm["representation_id"].astype(str) == str(rep_id)].copy()
    if not cm.empty:
        st.markdown(
            _tr("page03_segmentation.kmeans_gmm_candidate_evidence_for_this_representation_f1b98e2")
        )
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(
                cm.sort_values(["algorithm", "k"]),
                x="k",
                y="silhouette",
                color="algorithm",
                markers=True,
                text="silhouette",
                title=_tr("page03_segmentation.value0_silhouette_by_k_f6f8dae", value0=f"{rep_id}"),
            )
            fig.update_traces(
                texttemplate="%{text:.3f}", textposition=_src("model_evidence.top_center_24b3167")
            )
            show_plot(st, fig, f"p3_{rep_id}_sil")
        with c2:
            y = (
                "resample_stability_ari_mean"
                if "resample_stability_ari_mean" in cm.columns
                else "stability_ari"
            )
            fig = px.line(
                cm.sort_values(["algorithm", "k"]),
                x="k",
                y=y,
                color="algorithm",
                markers=True,
                text=y,
                title=_tr(
                    "page03_segmentation.value0_value1_stability_8b69e76",
                    value0=f"{rep_id}",
                    value1=f"{('subsample' if y.startswith('resample') else 'seed')}",
                ),
            )
            gate = rationale.get(
                "resample_stability_threshold"
                if y.startswith("resample")
                else "stability_threshold"
            )
            if gate is not None:
                fig.add_hline(y=float(gate), line_dash="dash", annotation_text="gate")
            fig.update_traces(
                texttemplate="%{text:.3f}", textposition=_src("model_evidence.top_center_24b3167")
            )
            show_plot(st, fig, f"p3_{rep_id}_stab")

        c1, c2 = st.columns(2)
        with c1:
            if "min_cluster_share" in cm.columns:
                fig = px.line(
                    cm.sort_values(["algorithm", "k"]),
                    x="k",
                    y="min_cluster_share",
                    color="algorithm",
                    markers=True,
                    text="min_cluster_share",
                    title=_tr(
                        "page03_segmentation.value0_minimum_cluster_share_4bcf741",
                        value0=f"{rep_id}",
                    ),
                )
                gate = rationale.get("minimum_cluster_share_threshold")
                if gate is not None:
                    fig.add_hline(
                        y=float(gate),
                        line_dash="dash",
                        annotation_text=_src("page03_segmentation.balance_gate_fd74cc7"),
                    )
                fig.update_yaxes(tickformat=".0%")
                fig.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition=_src("model_evidence.top_center_24b3167"),
                )
                show_plot(st, fig, f"p3_{rep_id}_balance")
        with c2:
            support = next(
                (
                    x
                    for x in ["calinski_harabasz", "davies_bouldin", "selection_score"]
                    if x in cm.columns
                ),
                None,
            )
            if support:
                fig = px.line(
                    cm.sort_values(["algorithm", "k"]),
                    x="k",
                    y=support,
                    color="algorithm",
                    markers=True,
                    text=support,
                    title=_tr(
                        "page03_segmentation.value0_supporting_diagnostic_value1_ae78fd7",
                        value0=f"{rep_id}",
                        value1=f"{support}",
                    ),
                )
                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition=_src("model_evidence.top_center_24b3167"),
                )
                show_plot(st, fig, f"p3_{rep_id}_support")

        show_cols = _available(
            cm,
            [
                "representation_id",
                "algorithm",
                "k",
                "silhouette",
                "stability_ari",
                "resample_stability_ari_mean",
                "resample_stability_ari_p10",
                "min_cluster_share",
                "eligible",
                "selected_within_representation",
                "calinski_harabasz",
                "davies_bouldin",
                "inertia",
                "gmm_bic",
                "gmm_aic",
            ],
        )
        downloadable_table(
            st,
            cm.sort_values([c for c in ["algorithm", "k"] if c in cm.columns])[show_cols],
            _tr("page03_segmentation.value0_all_candidate_metrics_f2c866b", value0=f"{rep_id}"),
            f"p3_{rep_id}_candidate_metrics",
            height=360,
        )
    else:
        _render_missing(st, "representation_candidate_metrics.csv")

    # ---- Representation-specific PCA evidence ----
    pv = pca_all.copy()
    if not pv.empty and "representation_id" in pv.columns:
        pv = pv[pv["representation_id"].astype(str) == str(rep_id)].copy()
    if not pv.empty:
        st.markdown(
            _tr("page03_segmentation.pca_latent_space_evidence_for_this_representation_4f17d60")
        )
        if (
            "family" in pv.columns
            and pv["family"].notna().any()
            and pv["family"].astype(str).nunique() > 1
        ):
            families = sorted(pv["family"].dropna().astype(str).unique())
            fam = st.selectbox(
                _tr("page03_segmentation.family_pca_60ec911"),
                families,
                key=f"p3_{rep_id}_family_pca",
                format_func=_option_label(),
            )
            pv_plot = pv[pv["family"].astype(str) == fam].copy()
        else:
            pv_plot = pv.copy()

        xcol = "component_number" if "component_number" in pv_plot.columns else "component"
        if xcol in pv_plot.columns and "explained_variance_ratio" in pv_plot.columns:
            fig = go.Figure()
            fig.add_bar(
                x=pv_plot[xcol],
                y=100 * pd.to_numeric(pv_plot["explained_variance_ratio"], errors="coerce"),
                name=_src("page03_segmentation.individual_variance_618539d"),
            )
            if "cumulative_variance" in pv_plot.columns:
                fig.add_scatter(
                    x=pv_plot[xcol],
                    y=100 * pd.to_numeric(pv_plot["cumulative_variance"], errors="coerce"),
                    mode="lines+markers",
                    name=_src("page03_segmentation.cumulative_variance_38fdf7f"),
                )
            fig.update_layout(
                title=_tr(
                    "page03_segmentation.value0_pca_explained_variance_513bd2a", value0=f"{rep_id}"
                ),
                xaxis_title=_src("page03_segmentation.component_ce54f0e"),
                yaxis_title=_src("page03_segmentation.variance_0e11bb2"),
            )
            show_plot(st, fig, f"p3_{rep_id}_pca")
        downloadable_table(
            st,
            pv,
            _tr("page03_segmentation.value0_pca_component_evidence_d2ca8af", value0=f"{rep_id}"),
            f"p3_{rep_id}_pca_table",
            height=340,
        )

    # ---- Representation-specific top loadings ----
    ld = loadings_all.copy()
    if not ld.empty and "representation_id" in ld.columns:
        ld = ld[ld["representation_id"].astype(str) == str(rep_id)].copy()
    if not ld.empty and "absolute_loading" in ld.columns:
        st.markdown(
            _tr("page03_segmentation.top_encoded_contributors_to_the_representation_a009d69")
        )
        if "component" in ld.columns:
            components = list(dict.fromkeys(ld["component"].astype(str).tolist()))
            component = st.selectbox(
                _tr("page03_segmentation.component_ce54f0e"),
                components,
                key=f"p3_{rep_id}_loading_component",
                format_func=_option_label(),
            )
            ld = ld[ld["component"].astype(str) == component].copy()
        top = ld.sort_values("absolute_loading", ascending=False).head(12)
        ycol = "encoded_feature" if "encoded_feature" in top.columns else top.columns[0]
        fig = px.bar(
            top.sort_values("absolute_loading"),
            y=ycol,
            x="absolute_loading",
            orientation="h",
            text="absolute_loading",
            hover_data=_available(top, ["signed_loading", "family", "explained_variance_ratio"]),
            title=_tr(
                "page03_segmentation.value0_top_absolute_loadings_e715be9", value0=f"{rep_id}"
            ),
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        show_plot(st, fig, f"p3_{rep_id}_loadings")

    # Raw candidate assignments are intentionally shown as evidence, not recomputed.
    ra = rep_assignments.copy()
    if not ra.empty and "representation_id" in ra.columns:
        ra = ra[ra["representation_id"].astype(str) == str(rep_id)].copy()
    if not ra.empty:
        with st.expander(
            _tr(
                "page03_segmentation.value0_persisted_candidate_assignments_813b206",
                value0=f"{rep_id}",
            ),
            expanded=False,
        ):
            downloadable_table(
                st,
                ra,
                _tr("page03_segmentation.value0_candidate_assignments_512e4b9", value0=f"{rep_id}"),
                f"p3_{rep_id}_assignments",
                height=420,
            )


def _render_representation_design_detail(
    st,
    rep_id: str,
    rep_row: pd.Series,
    pca_all: pd.DataFrame,
    loadings_all: pd.DataFrame,
) -> None:
    """Render representation construction evidence only (no clustering decision metrics)."""
    st.markdown(f"#### {rep_id} — {_display(rep_row.get('representation_label', ''))}")
    purpose = rep_row.get("purpose", "")
    if pd.notna(purpose) and str(purpose).strip():
        st.caption(_display(str(purpose)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        _tr("page03_segmentation.raw_inputs_fca9033"),
        _display(
            str(int(rep_row["raw_input_count"]))
            if pd.notna(rep_row.get("raw_input_count"))
            else "—"
        ),
    )
    c2.metric(
        _tr("page03_segmentation.latent_dimensions_251c35d"),
        _display(
            str(int(rep_row["latent_dimensions"]))
            if pd.notna(rep_row.get("latent_dimensions"))
            else "—"
        ),
    )
    c3.metric(
        _tr("page03_segmentation.variance_retained_2c2524b"),
        _display(_fmt_value(rep_row.get("variance_retained_summary"), digits=1, pct=True)),
    )
    c4.metric(
        _tr("page03_segmentation.excluded_fields_0d29a80"),
        _display(
            str(rep_row.get("excluded_features", _src("model_training_presentation.none_dc937b5")))
        ),
    )

    selected_inputs = rep_row.get("selected_input_features", "")
    if pd.notna(selected_inputs) and str(selected_inputs).strip():
        with st.expander(
            _tr("page03_segmentation.raw_input_contract_for_this_representation_53bc89d"),
            expanded=False,
        ):
            st.write(_display(str(selected_inputs)))

    pv = pca_all.copy()
    if not pv.empty and "representation_id" in pv.columns:
        pv = pv[pv["representation_id"].astype(str) == str(rep_id)].copy()
    if not pv.empty:
        if (
            "family" in pv.columns
            and pv["family"].notna().any()
            and pv["family"].astype(str).nunique() > 1
        ):
            families = sorted(pv["family"].dropna().astype(str).unique())
            fam = st.selectbox(
                _tr("page03_segmentation.pca_family_13d345e"),
                families,
                key=f"p3_design_{rep_id}_family",
                format_func=_option_label(),
            )
            pv_plot = pv[pv["family"].astype(str) == fam].copy()
        else:
            pv_plot = pv.copy()
        xcol = (
            "component_number"
            if "component_number" in pv_plot.columns
            else ("component" if "component" in pv_plot.columns else None)
        )
        if xcol and "explained_variance_ratio" in pv_plot.columns:
            fig = go.Figure()
            fig.add_bar(
                x=pv_plot[xcol],
                y=100 * pd.to_numeric(pv_plot["explained_variance_ratio"], errors="coerce"),
                name=_src("page03_segmentation.individual_variance_618539d"),
            )
            if "cumulative_variance" in pv_plot.columns:
                fig.add_scatter(
                    x=pv_plot[xcol],
                    y=100 * pd.to_numeric(pv_plot["cumulative_variance"], errors="coerce"),
                    mode="lines+markers",
                    name=_src("page03_segmentation.cumulative_variance_38fdf7f"),
                )
            fig.update_layout(
                title=_tr(
                    "page03_segmentation.value0_explained_variance_7f93857", value0=f"{rep_id}"
                ),
                xaxis_title=_src("page03_segmentation.component_ce54f0e"),
                yaxis_title=_src("page03_segmentation.variance_0e11bb2"),
            )
            show_plot(st, fig, f"p3_design_{rep_id}_pca")

    ld = loadings_all.copy()
    if not ld.empty and "representation_id" in ld.columns:
        ld = ld[ld["representation_id"].astype(str) == str(rep_id)].copy()
    if not ld.empty and "absolute_loading" in ld.columns:
        if "component" in ld.columns:
            components = list(dict.fromkeys(ld["component"].astype(str).tolist()))
            component = st.selectbox(
                _tr("page03_segmentation.component_loading_faaa8fe"),
                components,
                key=f"p3_design_{rep_id}_component",
                format_func=_option_label(),
            )
            ld = ld[ld["component"].astype(str) == component].copy()
        top = ld.sort_values("absolute_loading", ascending=False).head(12)
        ycol = "encoded_feature" if "encoded_feature" in top.columns else top.columns[0]
        fig = px.bar(
            top.sort_values("absolute_loading"),
            y=ycol,
            x="absolute_loading",
            orientation="h",
            text="absolute_loading",
            hover_data=_available(top, ["signed_loading", "family", "explained_variance_ratio"]),
            title=_tr(
                "page03_segmentation.value0_top_encoded_contributors_5f846fa", value0=f"{rep_id}"
            ),
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        show_plot(st, fig, f"p3_design_{rep_id}_loadings")


def _safe_options(df: pd.DataFrame, col: str) -> list[str]:
    if df.empty or col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def _render_feature_space_option_compare(
    st,
    option_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    rationale: dict,
    insights: dict,
) -> None:
    """Visual comparison of O1 PCA vs O2 correlation-selected X."""
    if option_summary.empty or "option_id" not in option_summary.columns:
        _render_missing(st, "feature_space_option_summary.csv")
        return

    st.markdown(_tr("page03_segmentation.o1_vs_o2_feature_space_option_comparison_9a7d466"))
    st.caption(_tr("page03_segmentation.o1_is_feature_extraction_clustering_uses_retained_0893fc4"))

    opt = option_summary.copy()
    for col in [
        "silhouette",
        "stability_ari",
        "resample_stability_ari_mean",
        "min_cluster_share",
        "clustering_dimensions",
        "cross_option_ari",
    ]:
        if col in opt.columns:
            opt[col] = pd.to_numeric(opt[col], errors="coerce")

    selected_opt = (
        opt.loc[_truthy(opt["selected_option"]), "option_id"].astype(str).iloc[0]
        if "selected_option" in opt.columns and _truthy(opt["selected_option"]).any()
        else str(rationale.get("selected_feature_space_option", "—"))
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(_tr("page03_segmentation.selected_feature_space_9c88064"), _display(selected_opt))
    if "cross_option_ari" in opt.columns and opt["cross_option_ari"].notna().any():
        c2.metric(
            _tr("page03_segmentation.o1_o2_assignment_ari_e262340"),
            _display(_fmt_value(opt["cross_option_ari"].dropna().iloc[0])),
        )
    else:
        c2.metric(_tr("page03_segmentation.o1_o2_assignment_ari_e262340"), _display("—"))
    c3.metric(
        _tr("page03_segmentation.o2_correlation_threshold_a15cc7d"),
        _display(_fmt_value(rationale.get("correlation_threshold"))),
    )
    c4.metric(
        _tr("page03_segmentation.decision_tolerance_4907293"),
        _display(_fmt_value(rationale.get("representation_silhouette_tolerance"))),
    )

    metric_cols_to_plot = [
        c
        for c in [
            "silhouette",
            "stability_ari",
            "resample_stability_ari_mean",
            "min_cluster_share",
        ]
        if c in opt.columns
    ]
    c1, c2 = st.columns(2)

    with c1:
        if metric_cols_to_plot:
            long = opt[["option_id"] + metric_cols_to_plot].melt(
                id_vars="option_id",
                var_name="metric",
                value_name="value",
            )
            label_map = {
                "silhouette": _src("page03_segmentation.silhouette_af465c1"),
                "stability_ari": _src("page03_segmentation.seed_ari_dd87010"),
                "resample_stability_ari_mean": _src("page03_segmentation.subsample_ari_05527db"),
                "min_cluster_share": _src("page03_segmentation.min_cluster_share_bd3e8c3"),
            }
            long["metric"] = long["metric"].map(label_map).fillna(long["metric"])
            fig = px.bar(
                long,
                x="option_id",
                y="value",
                color="metric",
                barmode="group",
                text="value",
                title=_src("page03_segmentation.o1_vs_o2_common_clustering_evidence_358dfc7"),
                labels={
                    "option_id": _src("page03_segmentation.feature_space_option_9181318"),
                    "value": _src("page01_data_basic_clean.metric_2d275a7"),
                    "metric": _src("page03_segmentation.evidence_03867ae"),
                },
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_yaxes(range=[0, 1.05])
            show_plot(st, fig, "p3_o1_o2_common_metrics")

    with c2:
        if {"silhouette", "resample_stability_ari_mean"}.issubset(opt.columns):
            fig = px.scatter(
                opt,
                x="resample_stability_ari_mean",
                y="silhouette",
                size="min_cluster_share" if "min_cluster_share" in opt.columns else None,
                color="option_id",
                text="option_id",
                hover_data=_available(
                    opt,
                    [
                        "algorithm",
                        "k",
                        "clustering_dimensions",
                        "interpretability",
                        "internal_representation",
                    ],
                ),
                title=_src("page03_segmentation.decision_view_separation_vs_robustness_d0765a3"),
                labels={
                    "resample_stability_ari_mean": _src(
                        "page03_segmentation.subsample_stability_ari_5f96836"
                    ),
                    "silhouette": _src("page03_segmentation.silhouette_af465c1"),
                },
            )
            if rationale.get("resample_stability_threshold") is not None:
                fig.add_vline(
                    x=float(rationale["resample_stability_threshold"]),
                    line_dash="dash",
                    annotation_text=_src("page03_segmentation.stability_gate_6b89a27"),
                )
            sil = opt["silhouette"].dropna()
            tol = rationale.get("representation_silhouette_tolerance", 0.02)
            if len(sil):
                fig.add_hline(
                    y=float(sil.max()) - float(tol),
                    line_dash="dot",
                    annotation_text=_src("page03_segmentation.near_best_floor_aed09bf"),
                )
            fig.update_traces(textposition=_src("model_evidence.top_center_24b3167"))
            show_plot(st, fig, "p3_o1_o2_sep_vs_robustness")

    c1, c2 = st.columns(2)
    with c1:
        if {"clustering_dimensions", "silhouette"}.issubset(opt.columns):
            fig = px.scatter(
                opt,
                x="clustering_dimensions",
                y="silhouette",
                color="selected_option" if "selected_option" in opt.columns else "option_id",
                text="option_id",
                hover_data=_available(
                    opt,
                    ["algorithm", "k", "resample_stability_ari_mean", "interpretability"],
                ),
                title=_src("page03_segmentation.parsimony_view_dimensions_vs_separation_7a095f9"),
                labels={
                    "clustering_dimensions": _src(
                        "page03_segmentation.clustering_dimensions_lower_is_simpler_5d0eef1"
                    ),
                    "silhouette": _src("page03_segmentation.silhouette_af465c1"),
                },
            )
            fig.update_traces(textposition=_src("model_evidence.top_center_24b3167"))
            show_plot(st, fig, "p3_o1_o2_parsimony")

    with c2:
        if (
            pairwise is not None
            and not pairwise.empty
            and {"option_a", "option_b", "ari"}.issubset(pairwise.columns)
        ):
            matrix = pairwise.pivot(index="option_a", columns="option_b", values="ari")
            fig = px.imshow(
                matrix,
                text_auto=".3f",
                zmin=0,
                zmax=1,
                aspect="auto",
                title=_src("page03_segmentation.o1_vs_o2_assignment_agreement_ari_34d4a66"),
            )
            show_plot(st, fig, "p3_o1_o2_pairwise_ari")

    # Gate matrix at option level
    st.markdown(_tr("page03_segmentation.option_level_gate_matrix_8dcd2ee"))
    seed_thr = rationale.get("stability_threshold")
    sub_thr = rationale.get("resample_stability_threshold")
    bal_thr = rationale.get("minimum_cluster_share_threshold")
    tol = rationale.get("representation_silhouette_tolerance", 0.02)
    best_sil = float(opt["silhouette"].max()) if "silhouette" in opt.columns else np.nan

    gate_rows = []
    for _, row in opt.iterrows():
        gate_rows.append(
            {
                "option_id": str(row["option_id"]),
                _src("page03_segmentation.seed_stability_3a7a9e2"): int(
                    True
                    if seed_thr is None
                    else (
                        pd.notna(row.get("stability_ari"))
                        and float(row.get("stability_ari")) >= float(seed_thr)
                    )
                ),
                _src("page03_segmentation.subsample_stability_0805ba1"): int(
                    True
                    if sub_thr is None
                    else (
                        pd.notna(row.get("resample_stability_ari_mean"))
                        and float(row.get("resample_stability_ari_mean")) >= float(sub_thr)
                    )
                ),
                _src("page03_segmentation.cluster_balance_37f6519"): int(
                    True
                    if bal_thr is None
                    else (
                        pd.notna(row.get("min_cluster_share"))
                        and float(row.get("min_cluster_share")) >= float(bal_thr)
                    )
                ),
                _src("page03_segmentation.near_best_separation_8af640b"): int(
                    pd.notna(row.get("silhouette"))
                    and pd.notna(best_sil)
                    and float(row.get("silhouette")) >= best_sil - float(tol)
                ),
                _src("page03_segmentation.eligible_1889cf7"): int(bool(row.get("eligible", False))),
            }
        )
    gate_df = pd.DataFrame(gate_rows)
    if not gate_df.empty:
        gate_cols = [
            _src("page03_segmentation.seed_stability_3a7a9e2"),
            _src("page03_segmentation.subsample_stability_0805ba1"),
            _src("page03_segmentation.cluster_balance_37f6519"),
            _src("page03_segmentation.near_best_separation_8af640b"),
            _src("page03_segmentation.eligible_1889cf7"),
        ]
        z = gate_df[gate_cols].to_numpy(dtype=float)
        txt = np.where(z >= 1, "PASS", "FAIL")
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=gate_cols,
                y=gate_df["option_id"],
                text=txt,
                texttemplate="%{text}",
                zmin=0,
                zmax=1,
                showscale=False,
            )
        )
        fig.update_layout(
            title=_src("page03_segmentation.o1_o2_decision_gates_834ed1e"),
            xaxis_title=_src("page03_segmentation.decision_check_b036ea0"),
            yaxis_title=_src("page03_segmentation.option_45aaacb"),
        )
        show_plot(st, fig, "p3_o1_o2_gate_matrix")

    show_cols = _available(
        opt,
        [
            "option_id",
            "option_label",
            "method",
            "internal_representation",
            "algorithm",
            "k",
            "silhouette",
            "stability_ari",
            "resample_stability_ari_mean",
            "min_cluster_share",
            "eligible",
            "within_option_tolerance",
            "clustering_dimensions",
            "encoded_features_before",
            "selected_encoded_features",
            "correlation_threshold",
            "interpretability",
            "selected_option",
        ],
    )
    downloadable_table(
        st,
        opt[show_cols],
        _tr("page03_segmentation.o1_vs_o2_feature_space_decision_evidence_0ad3a69"),
        "p3_o1_o2_option_summary",
        height=330,
    )

    _show_saved_insight(
        st,
        insights,
        "feature_space_selection",
        _src("page03_segmentation.why_this_feature_space_option_is_selected_32da98a"),
    )



def _render_representation_overview(
    st,
    rep: pd.DataFrame,
    comparison: pd.DataFrame,
    rationale: dict,
    selected_representation: str,
) -> None:
    """Always-visible R0-R4 summary for the A6 decision page.

    The offline pipeline is the source of truth.  This renderer only reads the
    persisted representation evidence and makes the official R decision visible
    without requiring the user to open an expander.
    """
    if rep.empty or "representation_id" not in rep.columns:
        _render_missing(st, "representation_summary.csv")
        return

    short_map = {
        "R0_GLOBAL_PCA": "R0",
        "R1_FAMILYWISE_PCA": "R1",
        "R2_NO_JOB_CATEGORY": "R2",
        "R3_NO_YEARS_EXPERIENCE": "R3",
        "R4_NO_JOB_CATEGORY_NO_YEARS": "R4",
    }

    r = rep.copy()
    r["representation_id"] = r["representation_id"].astype(str)
    r["R"] = r["representation_id"].map(short_map).fillna(r["representation_id"])
    for col in [
        "silhouette",
        "stability_ari",
        "resample_stability_ari_mean",
        "min_cluster_share",
        "mean_cross_representation_ari",
        "latent_dimensions",
    ]:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")

    if "selected_representation" in r.columns:
        r["_selected"] = _truthy(r["selected_representation"])
    else:
        r["_selected"] = r["representation_id"].eq(str(selected_representation))

    if "eligible" in r.columns:
        r["_eligible"] = _truthy(r["eligible"])
    else:
        r["_eligible"] = False

    tolerance = float(
        rationale.get(
            "representation_silhouette_tolerance",
            rationale.get("silhouette_tolerance", 0.02),
        )
    )
    if "within_representation_tolerance" in r.columns:
        r["_near_best"] = _truthy(r["within_representation_tolerance"])
    elif "silhouette" in r.columns:
        r["_near_best"] = r["silhouette"] >= r["silhouette"].max() - tolerance
    else:
        r["_near_best"] = False

    st.markdown(_tr("page03_segmentation.overview_title"))
    st.caption(_tr("page03_segmentation.overview_caption"))

    # ------------------------------------------------------------------
    # 1) Scorecards — one card per R
    # ------------------------------------------------------------------
    cols = st.columns(min(5, max(1, len(r))))
    ordered = r.sort_values("R")
    for i, (_, row) in enumerate(ordered.iterrows()):
        with cols[i % len(cols)]:
            selected = bool(row.get("_selected", False))
            title = f"★ {row['R']}" if selected else str(row["R"])
            sil = row.get("silhouette")
            st.metric(title, f"Sil {sil:.3f}" if pd.notna(sil) else "Sil —")
            seed = row.get("stability_ari")
            subs = row.get("resample_stability_ari_mean")
            share = row.get("min_cluster_share")
            cross = row.get("mean_cross_representation_ari")
            dims = row.get("latent_dimensions")
            eligibility = _tr("status.pass" if bool(row.get("_eligible", False)) else "status.fail")
            near_best = _tr("status.pass" if bool(row.get("_near_best", False)) else "status.fail")
            st.caption(_tr(
                "page03_segmentation.overview_scorecard",
                seed=_fmt_value(seed), subsample=_fmt_value(subs),
                share=_fmt_value(share, digits=1, pct=True), cross=_fmt_value(cross),
                dimensions=_fmt_value(dims, digits=0), eligible=eligibility, near_best=near_best,
            ))
            if selected:
                st.success(_tr("page03_segmentation.overview_selected"))

    # ------------------------------------------------------------------
    # 2) Metric heatmap + latent-dimension chart
    # ------------------------------------------------------------------
    st.markdown(_tr("page03_segmentation.overview_metrics"))
    c1, c2 = st.columns([1.55, 1.0])

    with c1:
        metric_map = {
            "silhouette": "Silhouette",
            "stability_ari": "Seed ARI",
            "resample_stability_ari_mean": "Subsample ARI",
            "min_cluster_share": "Min cluster share",
            "mean_cross_representation_ari": "Cross-R ARI",
        }
        metric_cols_present = [c for c in metric_map if c in r.columns]
        if metric_cols_present:
            heat = ordered.set_index("R")[metric_cols_present].copy()
            heat.columns = [metric_map[c] for c in metric_cols_present]
            display_rows = [
                (f"★ {idx}" if bool(ordered.loc[ordered["R"].eq(idx), "_selected"].iloc[0]) else idx)
                for idx in heat.index
            ]
            z = heat.to_numpy(dtype=float)
            text_values = np.empty_like(z, dtype=object)
            for rr in range(z.shape[0]):
                for cc in range(z.shape[1]):
                    val = z[rr, cc]
                    col_name = heat.columns[cc]
                    if pd.isna(val):
                        text_values[rr, cc] = "—"
                    elif col_name == "Min cluster share":
                        text_values[rr, cc] = f"{val:.1%}"
                    else:
                        text_values[rr, cc] = f"{val:.3f}"
            fig = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=list(heat.columns),
                    y=display_rows,
                    text=text_values,
                    texttemplate="%{text}",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title=_tr("page03_segmentation.overview_metric_label")),
                    hovertemplate=(f"R=%{{y}}<br>{_label('Metric')}=%{{x}}<br>"
                                   f"{_label('Value')}=%{{text}}<extra></extra>"),
                )
            )
            fig.update_layout(
                title=_tr("page03_segmentation.overview_heatmap"),
                xaxis_title="",
                yaxis_title="Representation",
            )
            show_plot(st, fig, "p3_rcompare_metric_heatmap_visible")
        else:
            st.info(_tr("page03_segmentation.overview_no_metrics"))

    with c2:
        if "latent_dimensions" in ordered.columns:
            chart_df = ordered[["R", "latent_dimensions", "_selected"]].copy()
            chart_df["Status"] = np.where(
                chart_df["_selected"], _tr("page03_segmentation.overview_selected"),
                _tr("page03_segmentation.overview_alternative"),
            )
            fig = px.bar(
                chart_df,
                x="R",
                y="latent_dimensions",
                color="Status",
                text="latent_dimensions",
                title=_tr("page03_segmentation.overview_dimensions"),
                labels={"latent_dimensions": "Latent dimensions"},
            )
            fig.update_traces(textposition="outside")
            show_plot(st, fig, "p3_rcompare_latent_dims_visible")
        else:
            st.info(_tr("page03_segmentation.overview_no_dimensions"))

    # ------------------------------------------------------------------
    # 3) Exact decision table requested for presentation / export
    # ------------------------------------------------------------------
    st.markdown(_tr("page03_segmentation.overview_exact"))
    if comparison is not None and not comparison.empty:
        table = comparison.copy()
    else:
        table = pd.DataFrame(
            {
                "R": ordered["R"],
                "Silhouette": ordered.get("silhouette"),
                "Seed ARI": ordered.get("stability_ari"),
                "Subsample ARI": ordered.get("resample_stability_ari_mean"),
                "Min cluster share": ordered.get("min_cluster_share"),
                "Cross-R ARI": ordered.get("mean_cross_representation_ari"),
                "Latent dimensions": ordered.get("latent_dimensions"),
                "Eligible": np.where(ordered["_eligible"], "PASS", "FAIL"),
                "Near-best separation": np.where(ordered["_near_best"], "PASS", "FAIL"),
                "Selected": np.where(ordered["_selected"], "★ SELECTED", ""),
            }
        )

    # Keep exactly the requested headline fields first, with useful traceability after them.
    requested = [
        "R",
        "Silhouette",
        "Seed ARI",
        "Subsample ARI",
        "Min cluster share",
        "Cross-R ARI",
        "Latent dimensions",
        "Eligible",
        "Near-best separation",
        "Selected",
    ]
    requested = [c for c in requested if c in table.columns]
    trailing = [
        c
        for c in ["representation_id", "representation_label", "algorithm", "k", "excluded_features"]
        if c in table.columns
    ]
    table = table[requested + trailing]
    downloadable_table(
        st,
        table,
        _tr("page03_segmentation.overview_export"),
        "p3_r0_r4_actual_comparison",
        file_name="representation_comparison.csv",
        height=300,
    )

    selected_row = ordered[ordered["_selected"]]
    if not selected_row.empty:
        sr = selected_row.iloc[0]
        st.success(_tr(
            "page03_segmentation.overview_summary",
            representation=sr["R"],
            label=_display(sr.get("representation_label", sr["representation_id"])),
            silhouette=_fmt_value(sr.get("silhouette")),
            seed=_fmt_value(sr.get("stability_ari")),
            subsample=_fmt_value(sr.get("resample_stability_ari_mean")),
            cross=_fmt_value(sr.get("mean_cross_representation_ari")),
            dimensions=_fmt_value(sr.get("latent_dimensions"), digits=0),
        ))


def _render_representation_compare(
    st,
    rep: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    pairwise_ari: pd.DataFrame,
    rationale: dict,
    selected_representation: str,
) -> None:
    """Compare persisted R alternatives before the official representation is used.

    Streamlit remains an evidence renderer only:
    - no PCA is refit;
    - no KMeans/GMM model is refit;
    - no new weighted composite score is invented.

    The comparison follows the same scientific decision logic used by the offline
    pipeline: fit validity/convergence -> seed stability -> subsample stability ->
    cluster balance -> near-best separation -> cross-representation robustness ->
    parsimony.
    """
    if rep.empty or "representation_id" not in rep.columns:
        _render_missing(st, "representation_summary.csv")
        return

    st.markdown(_tr("page03_segmentation.r_compare_representation_decision_evidence_81e4b92"))
    st.caption(
        _tr("page03_segmentation.compare_every_persisted_representation_side_by_side_0a7e1f7")
    )

    r = rep.copy()
    r["representation_id"] = r["representation_id"].astype(str)

    numeric_cols = [
        "silhouette",
        "stability_ari",
        "resample_stability_ari_mean",
        "min_cluster_share",
        "mean_cross_representation_ari",
        "latent_dimensions",
        "raw_input_count",
    ]
    for col in numeric_cols:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")

    if "selected_representation" in r.columns:
        r["_selected"] = _truthy(r["selected_representation"])
    else:
        r["_selected"] = r["representation_id"].eq(str(selected_representation))

    cm = candidate_metrics.copy()
    if not cm.empty:
        if "representation_id" in cm.columns:
            cm["representation_id"] = cm["representation_id"].astype(str)
        if "k" in cm.columns:
            cm["k"] = pd.to_numeric(cm["k"], errors="coerce")
        for col in [
            "silhouette",
            "stability_ari",
            "resample_stability_ari_mean",
            "min_cluster_share",
        ]:
            if col in cm.columns:
                cm[col] = pd.to_numeric(cm[col], errors="coerce")

    # ------------------------------------------------------------------
    # 1. Candidate landscape across all R
    # ------------------------------------------------------------------
    st.markdown(_tr("page03_segmentation.1_candidate_landscape_across_all_representations_c548875"))
    c1, c2 = st.columns(2)

    with c1:
        if not cm.empty and {"representation_id", "algorithm", "k", "silhouette"}.issubset(
            cm.columns
        ):
            fig = px.line(
                cm.sort_values(["representation_id", "algorithm", "k"]),
                x="k",
                y="silhouette",
                color="representation_id",
                line_dash="algorithm",
                markers=True,
                hover_data=_available(
                    cm,
                    [
                        "algorithm",
                        "stability_ari",
                        "resample_stability_ari_mean",
                        "min_cluster_share",
                    ],
                ),
                title=_src("page03_segmentation.all_r_silhouette_across_algorithm_k_a1777bd"),
            )
            fig.update_layout(
                legend_title_text=_src("page03_segmentation.representation_algorithm_3c7a682")
            )
            show_plot(st, fig, "p3_rcompare_all_silhouette")
        else:
            st.info(
                _tr(
                    "page03_segmentation.candidate_level_silhouette_evidence_is_not_available_4907cd1"
                )
            )

    with c2:
        stability_col = None
        if not cm.empty:
            if "resample_stability_ari_mean" in cm.columns:
                stability_col = "resample_stability_ari_mean"
            elif "stability_ari" in cm.columns:
                stability_col = "stability_ari"

        if stability_col and {"representation_id", "algorithm", "k"}.issubset(cm.columns):
            fig = px.line(
                cm.sort_values(["representation_id", "algorithm", "k"]),
                x="k",
                y=stability_col,
                color="representation_id",
                line_dash="algorithm",
                markers=True,
                hover_data=_available(cm, ["silhouette", "min_cluster_share"]),
                title=(
                    _src(
                        "page03_segmentation.all_r_subsample_robustness_across_algorithm_k_e1944e7"
                    )
                    if stability_col == "resample_stability_ari_mean"
                    else _src("page03_segmentation.all_r_seed_stability_across_algorithm_k_2c84db2")
                ),
            )
            gate_key = (
                "resample_stability_threshold"
                if stability_col == "resample_stability_ari_mean"
                else "stability_threshold"
            )
            if rationale.get(gate_key) is not None:
                fig.add_hline(
                    y=float(rationale[gate_key]),
                    line_dash="dash",
                    annotation_text="gate",
                )
            fig.update_layout(
                legend_title_text=_src("page03_segmentation.representation_algorithm_3c7a682")
            )
            show_plot(st, fig, "p3_rcompare_all_stability")
        else:
            st.info(
                _tr(
                    "page03_segmentation.candidate_level_stability_evidence_is_not_available_e4619e4"
                )
            )

    # ------------------------------------------------------------------
    # 2. Winner inside each R: common comparable metrics
    # ------------------------------------------------------------------
    st.markdown(_tr("page03_segmentation.2_best_candidate_inside_each_r_common_5c669bc"))
    comparable = [
        c
        for c in [
            "silhouette",
            "stability_ari",
            "resample_stability_ari_mean",
            "min_cluster_share",
            "mean_cross_representation_ari",
        ]
        if c in r.columns
    ]

    if comparable:
        long = r[["representation_id"] + comparable].melt(
            id_vars="representation_id",
            var_name="metric",
            value_name="value",
        )
        label_map = {
            "silhouette": _src("page03_segmentation.silhouette_af465c1"),
            "stability_ari": _src("page03_segmentation.seed_ari_dd87010"),
            "resample_stability_ari_mean": _src("page03_segmentation.subsample_ari_05527db"),
            "min_cluster_share": _src("page03_segmentation.min_cluster_share_bd3e8c3"),
            "mean_cross_representation_ari": _src("page03_segmentation.cross_r_ari_377f0df"),
        }
        long["metric"] = long["metric"].map(label_map).fillna(long["metric"])
        fig = px.bar(
            long,
            x="representation_id",
            y="value",
            color="metric",
            barmode="group",
            text="value",
            title=_src("page03_segmentation.r_level_comparison_higher_is_better_649d219"),
            labels={
                "representation_id": _src("page03_segmentation.representation_15ecda5"),
                "value": _src("page03_segmentation.metric_value_0d5bc43"),
                "metric": _src("page03_segmentation.evidence_03867ae"),
            },
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_yaxes(range=[0, 1.05])
        show_plot(st, fig, "p3_rcompare_grouped_metrics")

    # ------------------------------------------------------------------
    # 3. Decision views
    # ------------------------------------------------------------------
    st.markdown(
        _tr("page03_segmentation.3_decision_views_robustness_separation_and_parsimony_6633eec")
    )
    c1, c2 = st.columns(2)

    with c1:
        if {"silhouette", "resample_stability_ari_mean"}.issubset(r.columns):
            fig = px.scatter(
                r,
                x="resample_stability_ari_mean",
                y="silhouette",
                size="min_cluster_share" if "min_cluster_share" in r.columns else None,
                color="representation_id",
                text="representation_id",
                hover_data=_available(
                    r,
                    [
                        "algorithm",
                        "k",
                        "stability_ari",
                        "min_cluster_share",
                        "mean_cross_representation_ari",
                        "latent_dimensions",
                        "excluded_features",
                    ],
                ),
                title=_src("page03_segmentation.separation_vs_subsample_robustness_aecccbb"),
                labels={
                    "resample_stability_ari_mean": _src(
                        "page03_segmentation.subsample_stability_ari_5f96836"
                    ),
                    "silhouette": _src("page03_segmentation.silhouette_af465c1"),
                },
            )

            if rationale.get("resample_stability_threshold") is not None:
                fig.add_vline(
                    x=float(rationale["resample_stability_threshold"]),
                    line_dash="dash",
                    annotation_text=_src("page03_segmentation.stability_gate_6b89a27"),
                )

            sil = pd.to_numeric(r["silhouette"], errors="coerce").dropna()
            tolerance = rationale.get(
                "representation_silhouette_tolerance",
                rationale.get("silhouette_tolerance", 0.02),
            )
            if len(sil):
                near_best_floor = float(sil.max()) - float(tolerance)
                fig.add_hline(
                    y=near_best_floor,
                    line_dash="dot",
                    annotation_text=_src("page03_segmentation.near_best_floor_aed09bf"),
                )

            fig.update_traces(textposition=_src("model_evidence.top_center_24b3167"))
            show_plot(st, fig, "p3_rcompare_sep_vs_stability")
        else:
            st.info(
                _tr(
                    "page03_segmentation.r_level_separation_stability_evidence_is_incomplete_2733a5a"
                )
            )

    with c2:
        if {"latent_dimensions", "silhouette"}.issubset(r.columns):
            fig = px.scatter(
                r,
                x="latent_dimensions",
                y="silhouette",
                size="raw_input_count" if "raw_input_count" in r.columns else None,
                color="_selected",
                text="representation_id",
                hover_data=_available(
                    r,
                    [
                        "algorithm",
                        "k",
                        "resample_stability_ari_mean",
                        "mean_cross_representation_ari",
                        "excluded_features",
                    ],
                ),
                title=_src("page03_segmentation.parsimony_vs_separation_e92cd6f"),
                labels={
                    "latent_dimensions": _src(
                        "page03_segmentation.latent_dimensions_lower_is_simpler_7639bec"
                    ),
                    "silhouette": _src("page03_segmentation.silhouette_af465c1"),
                    "_selected": _src("page03_segmentation.official_r_ba4de7f"),
                },
            )
            fig.update_traces(textposition=_src("model_evidence.top_center_24b3167"))
            show_plot(st, fig, "p3_rcompare_parsimony")
        else:
            st.info(
                _tr("page03_segmentation.latent_dimension_evidence_is_not_available_for_622744e")
            )

    # ------------------------------------------------------------------
    # 4. Gate matrix for each selected-within-R candidate
    # ------------------------------------------------------------------
    st.markdown(_tr("page03_segmentation.4_representation_gate_matrix_99e7943"))

    seed_gate = rationale.get("stability_threshold")
    subsample_gate = rationale.get("resample_stability_threshold")
    balance_gate = rationale.get("minimum_cluster_share_threshold")
    tolerance = rationale.get(
        "representation_silhouette_tolerance",
        rationale.get("silhouette_tolerance", 0.02),
    )

    best_sil = np.nan
    if "silhouette" in r.columns:
        sil_values = pd.to_numeric(r["silhouette"], errors="coerce").dropna()
        if len(sil_values):
            best_sil = float(sil_values.max())

    gate_rows = []

    for _, rr in r.iterrows():
        rid = str(rr["representation_id"])
        selected_candidate = pd.DataFrame()

        if not cm.empty and "representation_id" in cm.columns:
            q = cm[cm["representation_id"].astype(str) == rid].copy()

            if not q.empty and "selected_within_representation" in q.columns:
                q_selected = q[_truthy(q["selected_within_representation"])]
                if not q_selected.empty:
                    selected_candidate = q_selected.head(1)

            if selected_candidate.empty and not q.empty and {"algorithm", "k"}.issubset(q.columns):
                rr_alg = str(rr.get("algorithm", ""))
                rr_k = pd.to_numeric(pd.Series([rr.get("k")]), errors="coerce").iloc[0]
                if pd.notna(rr_k):
                    q_selected = q[
                        q["algorithm"].astype(str).eq(rr_alg)
                        & pd.to_numeric(q["k"], errors="coerce").eq(float(rr_k))
                    ]
                    if not q_selected.empty:
                        selected_candidate = q_selected.head(1)

        candidate = selected_candidate.iloc[0] if not selected_candidate.empty else rr

        seed_val = pd.to_numeric(
            pd.Series([candidate.get("stability_ari")]),
            errors="coerce",
        ).iloc[0]
        sub_val = pd.to_numeric(
            pd.Series([candidate.get("resample_stability_ari_mean")]),
            errors="coerce",
        ).iloc[0]
        share_val = pd.to_numeric(
            pd.Series([candidate.get("min_cluster_share")]),
            errors="coerce",
        ).iloc[0]
        sil_val = pd.to_numeric(
            pd.Series([rr.get("silhouette")]),
            errors="coerce",
        ).iloc[0]

        algorithm = str(candidate.get("algorithm", rr.get("algorithm", "")))
        convergence_pass = True

        if algorithm.upper() == "GMM":
            if "convergence_pass" in candidate.index:
                convergence_pass = bool(
                    _truthy(pd.Series([candidate.get("convergence_pass")])).iloc[0]
                )
            elif "gmm_converged" in candidate.index:
                convergence_pass = bool(
                    _truthy(pd.Series([candidate.get("gmm_converged")])).iloc[0]
                )
            elif "convergence_warning" in candidate.index:
                convergence_pass = not bool(
                    _truthy(pd.Series([candidate.get("convergence_warning")])).iloc[0]
                )

        seed_pass = (
            True if seed_gate is None else bool(pd.notna(seed_val) and seed_val >= float(seed_gate))
        )
        subsample_pass = (
            True
            if subsample_gate is None
            else bool(pd.notna(sub_val) and sub_val >= float(subsample_gate))
        )
        balance_pass = (
            True
            if balance_gate is None
            else bool(pd.notna(share_val) and share_val >= float(balance_gate))
        )
        near_best = bool(
            pd.notna(sil_val) and pd.notna(best_sil) and sil_val >= best_sil - float(tolerance)
        )

        gate_rows.append(
            {
                "representation_id": rid,
                _src("page03_segmentation.convergence_096b170"): int(convergence_pass),
                _src("page03_segmentation.seed_stability_3a7a9e2"): int(seed_pass),
                _src("page03_segmentation.subsample_stability_0805ba1"): int(subsample_pass),
                _src("page03_segmentation.cluster_balance_37f6519"): int(balance_pass),
                _src("page03_segmentation.near_best_separation_8af640b"): int(near_best),
            }
        )

    gate_df = pd.DataFrame(gate_rows)

    if not gate_df.empty:
        gate_cols = [
            _src("page03_segmentation.convergence_096b170"),
            _src("page03_segmentation.seed_stability_3a7a9e2"),
            _src("page03_segmentation.subsample_stability_0805ba1"),
            _src("page03_segmentation.cluster_balance_37f6519"),
            _src("page03_segmentation.near_best_separation_8af640b"),
        ]
        z = gate_df[gate_cols].to_numpy(dtype=float)
        labels = np.where(z >= 1, "PASS", "FAIL")

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=gate_cols,
                y=gate_df["representation_id"],
                text=labels,
                texttemplate="%{text}",
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate=(
                    _src("page03_segmentation.representation_y_br_check_x_br_status_155f6c6")
                ),
            )
        )
        fig.update_layout(
            title=_src("page03_segmentation.gate_status_by_representation_057161e"),
            xaxis_title=_src("page03_segmentation.decision_check_b036ea0"),
            yaxis_title=_src("page03_segmentation.representation_15ecda5"),
        )
        show_plot(st, fig, "p3_rcompare_gate_matrix")

    # ------------------------------------------------------------------
    # 5. Cross-R assignment agreement
    # ------------------------------------------------------------------
    if (
        pairwise_ari is not None
        and not pairwise_ari.empty
        and {"representation_a", "representation_b", "ari"}.issubset(pairwise_ari.columns)
    ):
        st.markdown(_tr("page03_segmentation.5_cross_representation_assignment_agreement_7b0313c"))
        matrix = pairwise_ari.pivot(
            index="representation_a",
            columns="representation_b",
            values="ari",
        )

        all_ids = sorted(set(matrix.index.astype(str)).union(set(matrix.columns.astype(str))))
        matrix = matrix.reindex(index=all_ids, columns=all_ids)

        for rid in all_ids:
            matrix.loc[rid, rid] = 1.0

        for a in all_ids:
            for b in all_ids:
                if pd.isna(matrix.loc[a, b]) and b in matrix.index and a in matrix.columns:
                    matrix.loc[a, b] = matrix.loc[b, a]

        fig = px.imshow(
            matrix,
            text_auto=".2f",
            zmin=0,
            zmax=1,
            aspect="auto",
            title=_src(
                "page03_segmentation.cross_r_ari_assignment_similarity_after_representation_bd05e80"
            ),
        )
        show_plot(st, fig, "p3_rcompare_cross_ari")

    # ------------------------------------------------------------------
    # 6. Transparent decision table
    # ------------------------------------------------------------------
    st.markdown(_tr("page03_segmentation.6_transparent_r_decision_table_no_composite_b564fa4"))

    decision = r.copy()

    if "silhouette" in decision.columns:
        decision["rank_separation"] = decision["silhouette"].rank(
            method="min",
            ascending=False,
        )

    if "resample_stability_ari_mean" in decision.columns:
        decision["rank_subsample_stability"] = decision["resample_stability_ari_mean"].rank(
            method="min",
            ascending=False,
        )

    if "mean_cross_representation_ari" in decision.columns:
        decision["rank_cross_R_agreement"] = decision["mean_cross_representation_ari"].rank(
            method="min",
            ascending=False,
        )

    if "latent_dimensions" in decision.columns:
        decision["rank_parsimony"] = decision["latent_dimensions"].rank(
            method="min",
            ascending=True,
        )

    near_best_lookup = {}
    if not gate_df.empty:
        near_best_lookup = {
            str(row["representation_id"]): bool(
                row[_src("page03_segmentation.near_best_separation_8af640b")]
            )
            for _, row in gate_df.iterrows()
        }

    decision["near_best_separation"] = (
        decision["representation_id"].map(near_best_lookup).fillna(False)
    )

    if "eligible" in decision.columns:
        eligible_flag = _truthy(decision["eligible"])
    else:
        hard_gate_lookup = {}
        if not gate_df.empty:
            for _, row in gate_df.iterrows():
                hard_gate_lookup[str(row["representation_id"])] = bool(
                    row[_src("page03_segmentation.convergence_096b170")]
                    and row[_src("page03_segmentation.seed_stability_3a7a9e2")]
                    and row[_src("page03_segmentation.subsample_stability_0805ba1")]
                    and row[_src("page03_segmentation.cluster_balance_37f6519")]
                )
        eligible_flag = decision["representation_id"].map(hard_gate_lookup).fillna(False)

    decision["decision_cue"] = _src("page03_segmentation.secondary_alternative_b35c089")

    decision.loc[
        ~eligible_flag,
        "decision_cue",
    ] = _src("page03_segmentation.reject_review_hard_gate_failed_93d1468")

    decision.loc[
        eligible_flag & decision["near_best_separation"],
        "decision_cue",
    ] = _src("page03_segmentation.strong_candidate_eligible_and_near_best_d95116e")

    decision.loc[
        decision["representation_id"].eq(str(selected_representation)),
        "decision_cue",
    ] = _src("page03_segmentation.official_representation_7d8b068")

    decision_cols = _available(
        decision,
        [
            "representation_id",
            "representation_label",
            "algorithm",
            "k",
            "silhouette",
            "stability_ari",
            "resample_stability_ari_mean",
            "min_cluster_share",
            "mean_cross_representation_ari",
            "latent_dimensions",
            "raw_input_count",
            "excluded_features",
            "rank_separation",
            "rank_subsample_stability",
            "rank_cross_R_agreement",
            "rank_parsimony",
            "near_best_separation",
            "eligible",
            "selected_representation",
            "decision_cue",
        ],
    )

    downloadable_table(
        st,
        decision[decision_cols],
        _tr("page03_segmentation.r_comparison_decision_evidence_02e1163"),
        "p3_rcompare_decision_table",
        height=430,
    )

    # ------------------------------------------------------------------
    # 7. Dynamic conclusion
    # ------------------------------------------------------------------
    selected = decision[decision["representation_id"].astype(str) == str(selected_representation)]

    if not selected.empty:
        s = selected.iloc[0]
        observed = _tr(
            "page03_segmentation.official_representation_value0_within_this_r_the_dd9c228",
            value0=f"{selected_representation}",
            value1=f"{s.get('algorithm', '—')}",
            value2=f"{s.get('k', '—')}",
            value3=f"{_fmt_value(s.get('silhouette'))}",
            value4=f"{_fmt_value(s.get('resample_stability_ari_mean'))}",
            value5=f"{_fmt_value(s.get('min_cluster_share'), digits=1, pct=True)}",
            value6=f"{_fmt_value(s.get('mean_cross_representation_ari'))}",
            value7=f"{(int(s['latent_dimensions']) if pd.notna(s.get('latent_dimensions')) else '—')}",
        )

        interpretation = _tr("page03_segmentation.the_r_decision_is_not_based_on_48773e5")

        action = _tr("page03_segmentation.use_the_official_r_as_the_frozen_09cd513")

        interpretation_card(
            st,
            observed,
            interpretation,
            action,
            "success",
            title=_tr("page03_segmentation.r_selection_conclusion_41952b2"),
        )


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("navigation.page03"))
    st.caption(_tr("page03_segmentation.branch_a_pipeline_a1_shared_prepared_feature_eccb803"))

    meta = read_json(root, f"{SEG_DIR}/segmentation_metadata.json")
    rationale = read_json(root, f"{SEG_DIR}/k_selection_rationale.json")
    insights = read_json(root, f"{SEG_DIR}/segmentation_insights.json")
    e = _read_all_evidence(root)

    assign = e["cluster_assignments"]
    ev = e["cluster_evaluation"]
    prof = e["cluster_profiles"]
    rep = e["representation_summary"].copy()
    rep_metrics = e["representation_candidate_metrics"].copy()
    option_summary = e["feature_space_option_summary"].copy()
    option_metrics = e["feature_space_candidate_metrics"].copy()

    if assign.empty or ev.empty or prof.empty:
        st.error(_tr("page03_segmentation.core_branch_a_outputs_are_missing_in_b49b9ac"))
        return

    metric_cols(
        st,
        [
            (_tr("page03_segmentation.records_clustered_eb549f3"), f"{len(assign):,}"),
            (_tr("page03_segmentation.feature_families_7b08dc2"), "6"),
            (
                _tr("page03_segmentation.encoded_dimensions_a06381b"),
                meta.get("encoded_dimensions", "—"),
            ),
            (
                _tr("page03_segmentation.feature_space_option_9181318"),
                meta.get("feature_space_option_id", "—"),
            ),
            (
                _tr("page03_segmentation.selected_representation_c11a9ce"),
                meta.get("representation_id", "—"),
            ),
            (
                _tr("page03_segmentation.selected_solution_9cc5bdc"),
                f"{meta.get('algorithm', '—')} · K={meta.get('k', '—')}",
            ),
            (
                _tr("page03_segmentation.subsample_ari_05527db"),
                _fmt_value(meta.get("resample_stability_ari_mean")),
            ),
        ],
    )
    st.info(_tr("page03_segmentation.interpret_the_page_from_left_to_right_8d18a73"))

    with st.expander(
        _tr("page03_segmentation.display_filters_outputs_only_1a814ea"), expanded=False
    ):
        c1, c2, c3, c4 = st.columns(4)
        cluster_opts = sorted(assign["cluster"].astype(int).unique().tolist())
        clusters = c1.multiselect(
            _tr("page03_segmentation.official_clusters_4e3e17f"),
            cluster_opts,
            default=cluster_opts,
            key="p3_clusters",
            format_func=_option_label(),
        )
        categories = c2.multiselect(
            _tr("page03_segmentation.job_categories_1fd1537"),
            _safe_options(assign, "job_category"),
            key="p3_categories",
            format_func=_option_label(),
        )
        countries = c3.multiselect(
            _tr("page03_segmentation.countries_8faf7ec"),
            _safe_options(assign, "country"),
            key="p3_countries",
            format_func=_option_label(),
        )
        family = c4.selectbox(
            _tr("page03_segmentation.feature_family_focus_7cb486d"),
            [
                _src("common.job_domain_9e966d1"),
                _src("common.skills_66d0f52"),
                _src("common.experience_8eab0f0"),
                _src("common.company_de4743c"),
                _src("common.geography_f3c7380"),
                _src("common.demand_benefits_e58ed2c"),
            ],
            key="p3_family",
            format_func=_option_label(),
        )
        st.caption(
            _tr("page03_segmentation.these_filters_change_displayed_evidence_only_the_d91fd52")
        )

    focused_rows = _filter_rows(assign, clusters, categories, countries)
    if focused_rows.empty:
        st.warning(_tr("page03_segmentation.no_persisted_evidence_rows_match_the_current_2f4cdb2"))
        return

    tabs = st.tabs(
        [
            _tr("page03_segmentation.a1_prepared_base_e3ce71a"),
            _tr("page03_segmentation.a2_feature_families_d3872da"),
            _tr("page03_segmentation.a3_family_balancing_9c81206"),
            _tr("page03_segmentation.a4_o1_vs_o2_design_3baf0b4"),
            _tr("page03_segmentation.a5_same_candidate_search_77d8ab2"),
            _tr("page03_segmentation.a6_option_compare_robustness_b1b19f2"),
            _tr("page03_segmentation.a7_official_selection_f86d94a"),
            _tr("page03_segmentation.a8_final_outputs_67bd085"),
        ]
    )

    # ------------------------------------------------------------------
    # A1 — Shared Prepared Feature Base
    # ------------------------------------------------------------------
    with tabs[0]:
        st.markdown(_tr("page03_segmentation.a1_shared_prepared_feature_base_aef6153"))
        st.caption(_tr("page03_segmentation.start_from_the_cleaned_leakage_safe_feature_583137d"))

        selected_inputs = (
            meta.get("representation_input_features") or meta.get("selected_input_features") or []
        )
        excluded = (
            meta.get("representation_excluded_features") or meta.get("excluded_raw_features") or []
        )

        r0_inputs = None
        if (
            not rep.empty
            and "representation_id" in rep.columns
            and "selected_input_features" in rep.columns
        ):
            r0 = rep[rep["representation_id"].astype(str) == "R0_GLOBAL_PCA"]
            if not r0.empty:
                r0_inputs = r0.iloc[0].get("selected_input_features")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(_tr("page01_data_basic_clean.records_47a84e9"), _display(f"{len(assign):,}"))
        c2.metric(
            _tr("page03_segmentation.target_used_for_clustering_ee5be77"),
            _display(
                _src("page03_segmentation.no_1ea442a")
                if not bool(meta.get("target_used_for_clustering", False))
                else _src("page03_segmentation.yes_85a39ab")
            ),
        )
        c3.metric(
            _tr("page03_segmentation.selected_raw_inputs_932aeda"),
            _display(len(selected_inputs) if isinstance(selected_inputs, list) else "—"),
        )
        c4.metric(
            _tr("page03_segmentation.ablation_exclusions_d83d216"),
            _display(len(excluded) if isinstance(excluded, list) else "—"),
        )

        st.markdown(
            _tr(
                "page03_segmentation.prepared_feature_contract_before_representation_selection_1c6e94a"
            )
        )
        if r0_inputs is not None and pd.notna(r0_inputs):
            st.code(str(r0_inputs), language=None)
        else:
            contract = pd.DataFrame(FAMILY_CONTRACT)
            st.dataframe(
                display_frame(contract[["family", "raw_fields"]]), width="stretch", hide_index=True
            )

        st.markdown(
            _tr(
                "page03_segmentation.official_feature_contract_after_representation_selection_3b9b907"
            )
        )
        st.code(str(selected_inputs), language=None)
        if excluded:
            st.warning(
                _display(
                    _tr("page03_segmentation.official_representation_excludes_58fb600")
                    + ", ".join(map(str, excluded))
                )
            )
        st.success(_tr("page03_segmentation.next_organize_the_prepared_features_into_six_58673a5"))

    # ------------------------------------------------------------------
    # A2 — Feature Families / encoding
    # ------------------------------------------------------------------
    with tabs[1]:
        st.markdown(_tr("page03_segmentation.a2_six_feature_families_aa8bc02"))
        st.caption(
            _tr("page03_segmentation.group_related_variables_before_clustering_so_the_3ac0fce")
        )

        contract = pd.DataFrame(FAMILY_CONTRACT)
        dims = e["family_dimensions"].copy()
        if not dims.empty and "family" in dims.columns:
            contract = contract.merge(dims, on="family", how="left")
        st.dataframe(display_frame(contract), width="stretch", hide_index=True)

        if not dims.empty and "encoded_dimensions" in dims.columns:
            fig = px.bar(
                dims.sort_values("encoded_dimensions"),
                y="family",
                x="encoded_dimensions",
                orientation="h",
                text="encoded_dimensions",
                hover_data=_available(dims, ["balance_weight"]),
                title=_src("page03_segmentation.encoded_dimensions_by_feature_family_e53c17b"),
            )
            fig.update_traces(textposition="outside")
            show_plot(st, fig, "p3_a2_family_dims")
        st.info(
            _tr(
                "page03_segmentation.interpretation_encoded_dimension_count_is_descriptive_not_a963dbb"
            )
        )

    # ------------------------------------------------------------------
    # A3 — Family Balancing
    # ------------------------------------------------------------------
    with tabs[2]:
        st.markdown(_tr("page03_segmentation.a3_family_balancing_b04270c"))
        st.caption(_tr("page03_segmentation.high_dimensional_blocks_such_as_skills_or_8163768"))
        dims = e["family_dimensions"].copy()
        diag = e["family_balance_diagnostics"].copy()
        if dims.empty or diag.empty:
            _render_missing(
                st,
                _src(
                    "page03_segmentation.family_dimensions_csv_family_balance_diagnostics_csv_4b57e23"
                ),
            )
        else:
            dims = dims.sort_values("encoded_dimensions")
            diag = diag.sort_values("encoded_dimensions")
            c1, c2 = st.columns(2)
            with c1:
                if "balance_weight" in dims.columns:
                    fig = px.bar(
                        dims.sort_values("balance_weight"),
                        y="family",
                        x="balance_weight",
                        orientation="h",
                        text="balance_weight",
                        title=_src("page03_segmentation.applied_family_balance_weight_ab59184"),
                    )
                    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                    show_plot(st, fig, "p3_a3_weights")
            with c2:
                fig = go.Figure()
                if "mean_row_l2_before" in diag.columns:
                    fig.add_bar(
                        name=_src("page03_segmentation.before_balancing_f581e17"),
                        y=diag["family"],
                        x=diag["mean_row_l2_before"],
                        orientation="h",
                        text=diag["mean_row_l2_before"],
                        texttemplate="%{text:.2f}",
                    )
                if "mean_row_l2_after" in diag.columns:
                    fig.add_bar(
                        name=_src("page03_segmentation.after_balancing_d071998"),
                        y=diag["family"],
                        x=diag["mean_row_l2_after"],
                        orientation="h",
                        text=diag["mean_row_l2_after"],
                        texttemplate="%{text:.2f}",
                    )
                fig.update_layout(
                    barmode="group",
                    title=_src("page03_segmentation.mean_row_l2_norm_before_vs_after_731cbe9"),
                )
                show_plot(st, fig, "p3_a3_l2")

            if {"frobenius_norm_before", "frobenius_norm_after"}.issubset(diag.columns):
                norm_long = diag[["family", "frobenius_norm_before", "frobenius_norm_after"]].melt(
                    id_vars="family", var_name="state", value_name="frobenius_norm"
                )
                fig = px.bar(
                    norm_long,
                    x="family",
                    y="frobenius_norm",
                    color="state",
                    barmode="group",
                    text="frobenius_norm",
                    title=_src(
                        "page03_segmentation.family_block_frobenius_norm_before_vs_after_fdd55fa"
                    ),
                )
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                show_plot(st, fig, "p3_a3_frobenius")

            downloadable_table(
                st,
                diag,
                _tr("page03_segmentation.family_balance_diagnostics_0d129e8"),
                "p3_a3_family_balance",
                height=360,
            )
        st.success(
            _tr(
                "page03_segmentation.next_build_alternative_latent_representations_and_test_4062a0d"
            )
        )

    # ------------------------------------------------------------------
    # A4 — Feature-space construction: O1 PCA vs O2 Correlation
    # ------------------------------------------------------------------
    with tabs[3]:
        st.markdown(_tr("page03_segmentation.a4_feature_space_construction_o1_pca_vs_cfc2b80"))
        st.caption(
            _tr("page03_segmentation.two_independent_feature_space_strategies_are_built_bbc80c9")
        )

        o1_tab, o2_tab = st.tabs(
            [
                _tr("page03_segmentation.o1_pca_based_latent_representation_31e357c"),
                _tr("page03_segmentation.o2_correlation_based_feature_selection_8330959"),
            ]
        )

        with o1_tab:
            st.markdown(_tr("page03_segmentation.o1_pca_based_latent_representation_3869ae2"))
            st.info(_tr("page03_segmentation.o1_does_not_force_clustering_to_use_3b78ccd"))

            pca_all = e["representation_pca_variance"].copy()
            loadings_all = e["representation_top_loadings"].copy()

            if rep.empty:
                _render_missing(st, "representation_summary.csv")
            else:
                if "representation_id" in rep.columns:
                    rep = rep.sort_values("representation_id")

                design_cols = _available(
                    rep,
                    [
                        "representation_id",
                        "representation_label",
                        "purpose",
                        "excluded_features",
                        "raw_input_count",
                        "latent_dimensions",
                        "variance_retained_summary",
                        "selected_input_features",
                    ],
                )
                downloadable_table(
                    st,
                    rep[design_cols],
                    _tr("page03_segmentation.o1_pca_representation_design_r0_to_r4_16694be"),
                    "p3_a4_o1_rep_design",
                    height=390,
                )

                rep_ids = (
                    rep["representation_id"].astype(str).tolist()
                    if "representation_id" in rep.columns
                    else []
                )
                if rep_ids:
                    rep_tabs = st.tabs(rep_ids)
                    for rt, rid in zip(rep_tabs, rep_ids):
                        with rt:
                            rr = rep[rep["representation_id"].astype(str) == rid].iloc[0]
                            _render_representation_design_detail(
                                st,
                                rid,
                                rr,
                                pca_all,
                                loadings_all,
                            )

            st.caption(
                _tr("page03_segmentation.o1_output_a_lower_dimensional_latent_matrix_566bfe1")
            )

        with o2_tab:
            st.markdown(_tr("page03_segmentation.o2_correlation_based_feature_selection_ce7c4ac"))
            st.info(
                _tr(
                    "page03_segmentation.o2_measures_feature_to_feature_pearson_correlation_afffbb3"
                )
            )

            sel = e["correlation_feature_selection"].copy()
            pairs = e["correlation_redundancy_pairs"].copy()
            fam = e["correlation_family_summary"].copy()

            threshold = meta.get(
                "correlation_threshold",
                rationale.get("correlation_threshold"),
            )
            before = meta.get("o2_encoded_features_before", "—")
            kept = meta.get("o2_selected_encoded_features", "—")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                _tr("page03_segmentation.correlation_threshold_449fae3"),
                _display(_fmt_value(threshold)),
            )
            c2.metric(
                _tr("page03_segmentation.encoded_features_before_fe6a796"),
                _display(str(before)),
            )
            c3.metric(
                _tr("page03_segmentation.selected_encoded_features_f300200"),
                _display(str(kept)),
            )
            if isinstance(before, (int, float)) and isinstance(kept, (int, float)) and before:
                c4.metric(
                    _tr("page03_segmentation.reduction_0b92322"),
                    _display(f"{100 * (1 - float(kept) / float(before)):.1f}%"),
                )
            else:
                c4.metric(_tr("page03_segmentation.reduction_0b92322"), _display("—"))

            if not fam.empty:
                fam_plot = fam.copy()
                fig = go.Figure()
                if "encoded_features_before" in fam_plot.columns:
                    fig.add_bar(
                        x=fam_plot["family"],
                        y=fam_plot["encoded_features_before"],
                        name=_src("page03_segmentation.before_correlation_filter_c91e6f9"),
                        text=fam_plot["encoded_features_before"],
                    )
                if "selected_encoded_features" in fam_plot.columns:
                    fig.add_bar(
                        x=fam_plot["family"],
                        y=fam_plot["selected_encoded_features"],
                        name=_src("page03_segmentation.selected_for_o2_cb5a2c1"),
                        text=fam_plot["selected_encoded_features"],
                    )
                fig.update_layout(
                    barmode="group",
                    title=_src("page03_segmentation.o2_feature_retention_by_family_01dec3d"),
                    xaxis_title=_src("page03_segmentation.feature_family_81bb690"),
                    yaxis_title=_src("page03_segmentation.encoded_features_746683f"),
                )
                show_plot(st, fig, "p3_a4_o2_family_retention")

            if not pairs.empty and "abs_r" in pairs.columns:
                top_pairs = pairs.sort_values("abs_r", ascending=False).head(30).copy()
                top_pairs["pair"] = (
                    top_pairs["feature_a"].astype(str)
                    + "  ↔  "
                    + top_pairs["feature_b"].astype(str)
                )
                fig = px.bar(
                    top_pairs.sort_values("abs_r"),
                    y="pair",
                    x="abs_r",
                    orientation="h",
                    text="abs_r",
                    hover_data=_available(
                        top_pairs,
                        [
                            "pearson_r",
                            "family_a",
                            "family_b",
                            "feature_a_selected",
                            "feature_b_selected",
                        ],
                    ),
                    title=_src(
                        "page03_segmentation.top_redundant_encoded_feature_pairs_by_pearson_39771e3"
                    ),
                )
                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )
                show_plot(st, fig, "p3_a4_o2_top_corr_pairs")

            if not sel.empty:
                decision_counts = (
                    sel.value_counts(["family", "decision"]).rename("features").reset_index()
                )
                fig = px.bar(
                    decision_counts,
                    x="family",
                    y="features",
                    color="decision",
                    barmode="stack",
                    text="features",
                    title=_src(
                        "page03_segmentation.o2_keep_drop_decisions_by_feature_family_d7b9d0f"
                    ),
                )
                show_plot(st, fig, "p3_a4_o2_decisions")

                selected_only = sel[_truthy(sel["selected"])] if "selected" in sel.columns else sel
                downloadable_table(
                    st,
                    selected_only,
                    _tr("page03_segmentation.o2_selected_encoded_features_f5b8591"),
                    "p3_a4_o2_selected_features",
                    height=390,
                )

            if not pairs.empty:
                downloadable_table(
                    st,
                    pairs,
                    _tr("page03_segmentation.o2_high_correlation_pair_evidence_47dfcf4"),
                    "p3_a4_o2_corr_pairs",
                    height=390,
                )

            st.caption(
                _tr("page03_segmentation.o2_output_selected_original_encoded_x_dimensions_efaa39a")
            )

        st.success(_tr("page03_segmentation.next_apply_the_same_kmeans_gmm_k_024e150"))

    # ------------------------------------------------------------------
    # A5 — Same clustering candidate search for O1 and O2
    # ------------------------------------------------------------------
    with tabs[4]:
        st.markdown(_tr("page03_segmentation.a5_same_clustering_candidate_search_for_o1_8a14734"))
        st.caption(
            _tr("page03_segmentation.both_feature_space_options_are_evaluated_under_dd11c68")
        )

        cm = option_metrics.copy()
        if cm.empty:
            _render_missing(st, "feature_space_candidate_metrics.csv")
        else:
            option_choices = (
                sorted(cm["option_id"].dropna().astype(str).unique().tolist())
                if "option_id" in cm.columns
                else []
            )
            default_option = str(meta.get("feature_space_option_id", "O1_PCA"))
            idx = option_choices.index(default_option) if default_option in option_choices else 0
            chosen_option = st.selectbox(
                _tr("page03_segmentation.feature_space_option_to_inspect_3d16ef5"),
                option_choices,
                index=idx,
                key="p3_a5_option",
                format_func=_option_label(),
            )
            view = cm[cm["option_id"].astype(str) == chosen_option].copy()

            if "internal_representation" in view.columns:
                internal = ", ".join(
                    sorted(view["internal_representation"].dropna().astype(str).unique())
                )
                st.caption(
                    _tr(
                        "page03_segmentation.internal_representation_value0_7462bd0",
                        value0=f"{internal}",
                    )
                )

            c1, c2 = st.columns(2)
            with c1:
                if "silhouette" in view.columns:
                    fig = px.line(
                        view.sort_values(["algorithm", "k"]),
                        x="k",
                        y="silhouette",
                        color="algorithm",
                        markers=True,
                        text="silhouette",
                        title=_tr(
                            "page03_segmentation.value0_silhouette_by_k_f6f8dae",
                            value0=f"{chosen_option}",
                        ),
                    )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition=_src("model_evidence.top_center_24b3167"),
                    )
                    show_plot(st, fig, "p3_a5_option_silhouette")

            with c2:
                support_choices = [
                    x for x in ["calinski_harabasz", "davies_bouldin"] if x in view.columns
                ]
                if support_choices:
                    support = st.selectbox(
                        _tr("page03_segmentation.supporting_fit_metric_01d6d6b"),
                        support_choices,
                        key="p3_a5_option_support",
                        format_func=_option_label(),
                    )
                    fig = px.line(
                        view.sort_values(["algorithm", "k"]),
                        x="k",
                        y=support,
                        color="algorithm",
                        markers=True,
                        text=support,
                        title=f"{chosen_option} — {support}",
                    )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition=_src("model_evidence.top_center_24b3167"),
                    )
                    show_plot(st, fig, "p3_a5_option_support_chart")

            c1, c2 = st.columns(2)
            with c1:
                km = (
                    view[
                        view["algorithm"].astype(str) == _src("page03_segmentation.kmeans_cf52793")
                    ].copy()
                    if "algorithm" in view.columns
                    else pd.DataFrame()
                )
                if not km.empty and "inertia" in km.columns:
                    fig = px.line(
                        km.sort_values("k"),
                        x="k",
                        y="inertia",
                        markers=True,
                        text="inertia",
                        title=_tr(
                            "page03_segmentation.value0_kmeans_inertia_9e67419",
                            value0=f"{chosen_option}",
                        ),
                    )
                    fig.update_traces(
                        texttemplate="%{text:.1f}",
                        textposition=_src("model_evidence.top_center_24b3167"),
                    )
                    show_plot(st, fig, "p3_a5_option_inertia")

            with c2:
                gm = (
                    view[view["algorithm"].astype(str) == "GMM"].copy()
                    if "algorithm" in view.columns
                    else pd.DataFrame()
                )
                gmm_metric = next(
                    (x for x in ["gmm_bic", "gmm_aic"] if x in gm.columns),
                    None,
                )
                if not gm.empty and gmm_metric:
                    fig = px.line(
                        gm.sort_values("k"),
                        x="k",
                        y=gmm_metric,
                        markers=True,
                        text=gmm_metric,
                        title=(f"{chosen_option} — {gmm_metric.replace('gmm_', '').upper()} ↓"),
                    )
                    fig.update_traces(
                        texttemplate="%{text:.0f}",
                        textposition=_src("model_evidence.top_center_24b3167"),
                    )
                    show_plot(st, fig, "p3_a5_option_gmm")

            show_cols = _available(
                view,
                [
                    "option_id",
                    "internal_representation",
                    "algorithm",
                    "k",
                    "silhouette",
                    "calinski_harabasz",
                    "davies_bouldin",
                    "inertia",
                    "gmm_bic",
                    "gmm_aic",
                    "gmm_converged",
                    "convergence_warning",
                    "convergence_pass",
                ],
            )
            downloadable_table(
                st,
                view[show_cols],
                _tr(
                    "page03_segmentation.value0_candidate_search_evidence_1fdc8fa",
                    value0=f"{chosen_option}",
                ),
                "p3_a5_option_candidates",
                height=430,
            )

            # Side-by-side best-candidate silhouette profile across O1 and O2.
            if {"option_id", "algorithm", "k", "silhouette"}.issubset(cm.columns):
                fig = px.line(
                    cm.sort_values(["option_id", "algorithm", "k"]),
                    x="k",
                    y="silhouette",
                    color="option_id",
                    line_dash="algorithm",
                    markers=True,
                    title=_src("page03_segmentation.direct_o1_vs_o2_silhouette_landscape_f459d3d"),
                )
                show_plot(st, fig, "p3_a5_o1_o2_silhouette_landscape")

    # ------------------------------------------------------------------
    # A6 — Robustness, option comparison and O1 internal R comparison
    # ------------------------------------------------------------------
    with tabs[5]:
        st.markdown(_tr("page03_segmentation.a6_robustness_o1_vs_o2_decision_de17f5f"))
        st.caption(_tr("page03_segmentation.feature_space_choice_is_made_only_after_78d68ed"))

        cm = option_metrics.copy()
        if cm.empty:
            _render_missing(st, "feature_space_candidate_metrics.csv")
        else:
            option_choices = sorted(cm["option_id"].dropna().astype(str).unique().tolist())
            default_option = str(meta.get("feature_space_option_id", option_choices[0]))
            idx = option_choices.index(default_option) if default_option in option_choices else 0
            chosen_option = st.selectbox(
                _tr("page03_segmentation.option_for_detailed_robustness_review_6dfed5d"),
                option_choices,
                index=idx,
                key="p3_a6_option",
                format_func=_option_label(),
            )
            view = cm[cm["option_id"].astype(str) == chosen_option].copy()

            c1, c2 = st.columns(2)
            with c1:
                if "stability_ari" in view.columns:
                    fig = px.line(
                        view.sort_values(["algorithm", "k"]),
                        x="k",
                        y="stability_ari",
                        color="algorithm",
                        markers=True,
                        text="stability_ari",
                        title=_tr(
                            "page03_segmentation.value0_seed_stability_ari_7eebaa8",
                            value0=f"{chosen_option}",
                        ),
                    )
                    if rationale.get("stability_threshold") is not None:
                        fig.add_hline(
                            y=float(rationale["stability_threshold"]),
                            line_dash="dash",
                            annotation_text=_src("page03_segmentation.seed_gate_9f5f40c"),
                        )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition=_src("model_evidence.top_center_24b3167"),
                    )
                    show_plot(st, fig, "p3_a6_option_seed")

            with c2:
                if "resample_stability_ari_mean" in view.columns:
                    fig = px.line(
                        view.sort_values(["algorithm", "k"]),
                        x="k",
                        y="resample_stability_ari_mean",
                        color="algorithm",
                        markers=True,
                        text="resample_stability_ari_mean",
                        title=_tr(
                            "page03_segmentation.value0_subsample_stability_ari_52dcb2e",
                            value0=f"{chosen_option}",
                        ),
                    )
                    if rationale.get("resample_stability_threshold") is not None:
                        fig.add_hline(
                            y=float(rationale["resample_stability_threshold"]),
                            line_dash="dash",
                            annotation_text=_src("page03_segmentation.subsample_gate_9c4805a"),
                        )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition=_src("model_evidence.top_center_24b3167"),
                    )
                    show_plot(st, fig, "p3_a6_option_subsample")

            c1, c2 = st.columns(2)
            with c1:
                if "min_cluster_share" in view.columns:
                    fig = px.line(
                        view.sort_values(["algorithm", "k"]),
                        x="k",
                        y="min_cluster_share",
                        color="algorithm",
                        markers=True,
                        text="min_cluster_share",
                        title=_tr(
                            "page03_segmentation.value0_minimum_cluster_share_95f156b",
                            value0=f"{chosen_option}",
                        ),
                    )
                    if rationale.get("minimum_cluster_share_threshold") is not None:
                        fig.add_hline(
                            y=float(rationale["minimum_cluster_share_threshold"]),
                            line_dash="dash",
                            annotation_text=_src("page03_segmentation.balance_gate_fd74cc7"),
                        )
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_traces(
                        texttemplate="%{text:.1%}",
                        textposition=_src("model_evidence.top_center_24b3167"),
                    )
                    show_plot(st, fig, "p3_a6_option_balance")

            with c2:
                convergence_col = next(
                    (x for x in ["convergence_pass", "gmm_converged"] if x in view.columns),
                    None,
                )
                if convergence_col and "algorithm" in view.columns:
                    gm = view[view["algorithm"].astype(str) == "GMM"].copy()
                    if not gm.empty:
                        gm["convergence_status"] = gm[convergence_col].map(
                            lambda x: (
                                _src("page03_segmentation.pass_ebdf8cc")
                                if str(x).lower() in {"true", "1", "yes"}
                                else _src("page03_segmentation.fail_09230b3")
                            )
                        )
                        fig = px.bar(
                            gm.sort_values("k"),
                            x="k",
                            y=[1] * len(gm),
                            color="convergence_status",
                            text="convergence_status",
                            title=_tr(
                                "page03_segmentation.value0_gmm_convergence_gate_b48a161",
                                value0=f"{chosen_option}",
                            ),
                        )
                        fig.update_yaxes(visible=False)
                        show_plot(st, fig, "p3_a6_option_convergence")

        st.divider()
        _render_feature_space_option_compare(
            st,
            option_summary=option_summary,
            pairwise=e["feature_space_pairwise_ari"].copy(),
            rationale=rationale,
            insights=insights,
        )

        # R0-R4 official robustness overview is intentionally visible by default.
        # Detailed candidate-level diagnostics remain inside the expander below.
        _render_representation_overview(
            st,
            rep=rep,
            comparison=e["representation_comparison"].copy(),
            rationale=rationale,
            selected_representation=str(
                meta.get(
                    "o1_selected_representation",
                    rationale.get("o1_selected_representation", ""),
                )
            ),
        )

        with st.expander(
            _tr("page03_segmentation.overview_advanced"),
            expanded=False,
        ):
            _render_representation_compare(
                st,
                rep=rep,
                candidate_metrics=rep_metrics,
                pairwise_ari=e["representation_pairwise_ari"].copy(),
                rationale=rationale,
                selected_representation=str(
                    meta.get(
                        "o1_selected_representation",
                        rationale.get("o1_selected_representation", ""),
                    )
                ),
            )

            dep = e["feature_dependency_summary"].copy()
            if not dep.empty:
                downloadable_table(
                    st,
                    dep,
                    _tr("page03_segmentation.o1_dominant_feature_dependency_summary_993e882"),
                    "p3_a6_o1_dependency",
                    height=280,
                )

        _show_saved_insight(
            st,
            insights,
            "feature_space_selection",
            _src("page03_segmentation.feature_space_decision_interpretation_37b1e35"),
        )

    # ------------------------------------------------------------------
    # A7 — Official Feature Space + Algorithm + K selection
    # ------------------------------------------------------------------
    with tabs[6]:
        st.markdown(_tr("page03_segmentation.a7_official_selection_424b82f"))
        st.caption(_tr("page03_segmentation.a7_freezes_the_complete_branch_a_contract_7ebd583"))

        selected_option = str(
            rationale.get(
                "selected_feature_space_option",
                meta.get("feature_space_option_id", "—"),
            )
        )
        selected_rep = str(
            rationale.get(
                "selected_representation",
                meta.get("representation_id", "—"),
            )
        )
        selected_alg = str(
            rationale.get(
                "selected_algorithm",
                meta.get("algorithm", "—"),
            )
        )
        selected_k = rationale.get(
            "selected_k",
            meta.get("k", "—"),
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(
            _tr("page03_segmentation.feature_space_option_9181318"),
            _display(selected_option),
        )
        c2.metric(_tr("page03_segmentation.representation_15ecda5"), _display(selected_rep))
        c3.metric(_tr("page03_segmentation.algorithm_d704d8a"), _display(selected_alg))
        c4.metric(_display("K"), _display(str(selected_k)))
        c5.metric(
            _tr("page03_segmentation.silhouette_af465c1"),
            _display(
                _fmt_value(
                    rationale.get(
                        "selected_silhouette",
                        meta.get("silhouette"),
                    )
                )
            ),
        )
        c6.metric(
            _tr("page03_segmentation.subsample_ari_05527db"),
            _display(
                _fmt_value(
                    rationale.get(
                        "selected_resample_stability_ari_mean",
                        meta.get("resample_stability_ari_mean"),
                    )
                )
            ),
        )

        if selected_option == "O1_PCA":
            st.info(
                _tr(
                    "page03_segmentation.o1_won_the_feature_space_comparison_its_eb367d0",
                    value0=f"{selected_rep}",
                )
            )
        elif selected_option == "O2_CORRELATION":
            st.info(
                _tr("page03_segmentation.o2_won_the_feature_space_comparison_clustering_27573b2")
            )

        st.markdown(
            _tr(
                "page03_segmentation.decision_sequence_same_candidate_search_convergence_fit_d01ca17"
            )
        )

        gate_rows = [
            [
                _src("page03_segmentation.optimizer_seed_stability_fd69f90"),
                "stability_ari",
                rationale.get("stability_threshold"),
                "≥",
            ],
            [
                _src("page03_segmentation.subsample_stability_0805ba1"),
                "resample_stability_ari_mean",
                rationale.get("resample_stability_threshold"),
                "≥",
            ],
            [
                _src("page03_segmentation.minimum_cluster_share_0b3a5c7"),
                "min_cluster_share",
                rationale.get("minimum_cluster_share_threshold"),
                "≥",
            ],
            [
                _src("page03_segmentation.feature_space_practical_tie_79fe4ce"),
                _src("page03_segmentation.silhouette_gap_c1cf076"),
                rationale.get("representation_silhouette_tolerance"),
                "≤",
            ],
            [
                _src("page03_segmentation.within_option_k_tie_67c8508"),
                _src("page03_segmentation.silhouette_gap_c1cf076"),
                rationale.get("silhouette_tolerance"),
                "≤",
            ],
        ]
        st.dataframe(
            display_frame(
                pd.DataFrame(
                    gate_rows,
                    columns=[
                        _src("page03_segmentation.gate_rule_79fe099"),
                        _src("page01_data_basic_clean.metric_2d275a7"),
                        _src("page03_segmentation.threshold_0da627a"),
                        _src("page03_segmentation.direction_9c8a957"),
                    ],
                )
            ),
            width="stretch",
            hide_index=True,
        )

        if not option_summary.empty:
            option_cols = _available(
                option_summary,
                [
                    "option_id",
                    "option_label",
                    "internal_representation",
                    "algorithm",
                    "k",
                    "silhouette",
                    "stability_ari",
                    "resample_stability_ari_mean",
                    "min_cluster_share",
                    "eligible",
                    "within_option_tolerance",
                    "clustering_dimensions",
                    "interpretability",
                    "selected_option",
                ],
            )
            downloadable_table(
                st,
                option_summary[option_cols],
                _tr("page03_segmentation.official_feature_space_selection_summary_28b61af"),
                "p3_a7_option_selection",
                height=320,
            )

        feature_decision = rationale.get("feature_space_decision", {})
        runner = feature_decision.get("runner_up") if isinstance(feature_decision, dict) else None
        if isinstance(runner, dict) and runner:
            selected_row = (
                option_summary[_truthy(option_summary["selected_option"])].iloc[0].to_dict()
                if (
                    not option_summary.empty
                    and "selected_option" in option_summary.columns
                    and _truthy(option_summary["selected_option"]).any()
                )
                else {}
            )
            comp = pd.DataFrame(
                [
                    {
                        "candidate": _src("page03_segmentation.selected_option_84abbce"),
                        **selected_row,
                    },
                    {"candidate": _src("page03_segmentation.runner_up_option_6cf27d0"), **runner},
                ]
            )
            st.markdown(
                _tr("page03_segmentation.selected_feature_space_option_vs_runner_up_358e340")
            )
            st.dataframe(
                display_frame(comp),
                width="stretch",
                hide_index=True,
            )

        decision_note = rationale.get("decision_note")
        if decision_note:
            st.success(_display(str(decision_note)))

        # Official Algorithm × K candidates after the option is frozen.
        eligible = ev[_truthy(ev["eligible"])] if "eligible" in ev.columns else ev.copy()
        if not eligible.empty:
            elig_cols = _available(
                eligible,
                [
                    "algorithm",
                    "k",
                    "silhouette",
                    "stability_ari",
                    "resample_stability_ari_mean",
                    "min_cluster_share",
                    "within_primary_tolerance",
                    "selected",
                    "gmm_converged",
                    "convergence_pass",
                ],
            )
            downloadable_table(
                st,
                eligible[elig_cols].sort_values(
                    "silhouette",
                    ascending=False,
                ),
                _tr("page03_segmentation.official_feature_space_eligible_k_candidates_aad0972"),
                "p3_a7_eligible",
                height=330,
            )

        coords = e["candidate_cluster_coordinates"].copy()
        balance = e["candidate_cluster_balance"].copy()
        try:
            kval = int(selected_k)
        except Exception:
            kval = int(meta.get("k", 0) or 0)

        if not coords.empty and {"algorithm", "k"}.issubset(coords.columns):
            chosen = coords[
                (coords["algorithm"].astype(str) == selected_alg)
                & (coords["k"].astype(int) == kval)
            ].copy()
            chosen = _filter_rows(chosen, [], categories, countries)

            if not chosen.empty and {"PC1", "PC2", "cluster"}.issubset(chosen.columns):
                chosen["cluster_label"] = _src("page03_segmentation.cluster_cac75ce") + chosen[
                    "cluster"
                ].astype(int).astype(str)
                c1, c2 = st.columns([2, 1])

                with c1:
                    fig = px.scatter(
                        chosen,
                        x="PC1",
                        y="PC2",
                        color="cluster_label",
                        hover_data=_available(
                            chosen,
                            [
                                "job_title",
                                "job_category",
                                "country",
                                "years_of_experience",
                                "demand_score",
                            ],
                        ),
                        title=(
                            _tr(
                                "page03_segmentation.official_solution_value0_value1_value2_k_value3_c84578e",
                                value0=f"{selected_option}",
                                value1=f"{selected_rep}",
                                value2=f"{selected_alg}",
                                value3=f"{kval}",
                            )
                        ),
                    )
                    show_plot(st, fig, "p3_a7_selected_scatter")

                with c2:
                    bal = (
                        balance[
                            (balance["algorithm"].astype(str) == selected_alg)
                            & (balance["k"].astype(int) == kval)
                        ].copy()
                        if not balance.empty
                        else pd.DataFrame()
                    )
                    if not bal.empty:
                        fig = px.bar(
                            bal,
                            x="cluster",
                            y="records",
                            text="share",
                            title=_src("page03_segmentation.official_cluster_balance_d5d2d9d"),
                        )
                        fig.update_traces(
                            texttemplate="%{text:.1%}",
                            textposition="outside",
                        )
                        show_plot(st, fig, "p3_a7_selected_balance")

        _show_saved_insight(
            st,
            insights,
            "feature_space_selection",
            _src("page03_segmentation.why_this_feature_space_is_selected_4388aa9"),
        )
        _show_saved_insight(
            st,
            insights,
            "k_selection",
            _src("page03_segmentation.why_this_algorithm_k_is_selected_3511999"),
        )

    # ------------------------------------------------------------------
    # A8 — Final Outputs
    # ------------------------------------------------------------------
    with tabs[7]:
        st.markdown(_tr("page03_segmentation.a8_final_outputs_interpretation_9c4d89f"))
        st.caption(
            _tr("page03_segmentation.after_the_official_clustering_solution_is_frozen_5b0cf05")
        )

        out_tabs = st.tabs(
            [
                _tr("page03_segmentation.cluster_labels_map_289c1ea"),
                _tr("page03_segmentation.cluster_profiles_a36c7f4"),
                _tr("page03_segmentation.feature_family_eda_b150f8e"),
                _tr("page03_segmentation.market_geography_7a551ac"),
                _tr("page01_data_basic_clean.evidence_tables_21295b1"),
            ]
        )

        with out_tabs[0]:
            scatter = focused_rows.copy()
            if {"PC1", "PC2", "cluster"}.issubset(scatter.columns):
                scatter["cluster_label"] = _src("page03_segmentation.cluster_cac75ce") + scatter[
                    "cluster"
                ].astype(int).astype(str)
                fig = px.scatter(
                    scatter,
                    x="PC1",
                    y="PC2",
                    color="cluster_label",
                    hover_data=_available(
                        scatter,
                        [
                            "job_title",
                            "job_category",
                            "country",
                            "years_of_experience",
                            "demand_score",
                        ],
                    ),
                    title=_src("page03_segmentation.official_segmentation_2d_pca_display_c2f884e"),
                )
                show_plot(st, fig, "p3_a8_map")
            downloadable_table(
                st,
                focused_rows,
                _tr("page03_segmentation.filtered_official_row_level_assignments_f68871e"),
                "p3_a8_assignments",
                height=420,
            )

        with out_tabs[1]:
            view = (
                prof[prof["cluster"].astype(int).isin(list(map(int, clusters)))]
                if clusters
                else prof.copy()
            )
            st.dataframe(display_frame(view), width="stretch", hide_index=True)
            profile_cols = [
                c
                for c in ["years_mean", "demand_mean", "benefits_mean", "skill_count_mean"]
                if c in view.columns
            ]
            if profile_cols:
                fig = go.Figure()
                for col in profile_cols:
                    fig.add_bar(
                        x=view["cluster"].astype(str),
                        y=view[col],
                        name=col.replace("_mean", "").replace("_", " ").title(),
                        text=view[col],
                        texttemplate="%{text:.1f}",
                    )
                fig.update_layout(
                    barmode="group",
                    title=_src("page03_segmentation.numeric_cluster_profile_89986f3"),
                )
                show_plot(st, fig, "p3_a8_profile")
            if "annual_salary_usd" in focused_rows.columns:
                fig = px.box(
                    focused_rows,
                    x=focused_rows["cluster"].astype(str),
                    y="annual_salary_usd",
                    points="outliers",
                    title=_src(
                        "page03_segmentation.observed_salary_distribution_by_segment_post_hoc_2a1f3d4"
                    ),
                    labels={
                        "x": _src("page03_segmentation.cluster_c137a7f"),
                        "annual_salary_usd": _src("model_evidence.annual_salary_usd_46de225"),
                    },
                )
                show_plot(st, fig, "p3_a8_salary")
                st.caption(_tr("page03_segmentation.salary_is_descriptive_here_it_was_not_2c70aae"))
            _show_saved_insight(
                st,
                insights,
                "profiles",
                _src("page03_segmentation.cluster_profile_interpretation_1076a90"),
            )

        with out_tabs[2]:
            st.markdown(
                _tr("page03_segmentation.feature_family_eda_value0_3ec82b9", value0=f"{family}")
            )
            fig = None
            if family == _src("common.job_domain_9e966d1"):
                d = e["cluster_job_category_profile"].copy()
                if clusters and not d.empty:
                    d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                if categories and not d.empty:
                    d = d[d["job_category"].astype(str).isin(categories)]
                if not d.empty:
                    metric_options = [
                        x
                        for x in ["records", "salary_mean_posthoc", "demand_mean"]
                        if x in d.columns
                    ]
                    if metric_options:
                        metric = st.radio(
                            _tr("page03_segmentation.job_domain_view_9030ef6"),
                            metric_options,
                            horizontal=True,
                            key="p3_a8_job_metric",
                            format_func=_option_label(),
                        )
                        focus = st.multiselect(
                            _tr("page03_segmentation.focus_job_categories_b5d51dc"),
                            sorted(d["job_category"].astype(str).unique()),
                            key="p3_a8_job_focus",
                            format_func=_option_label(),
                        )
                        if focus:
                            d = d[d["job_category"].astype(str).isin(focus)]
                        fig = px.bar(
                            d,
                            x="job_category",
                            y=metric,
                            color=d["cluster"].astype(str),
                            barmode="group",
                            text=metric,
                            title=_src("page03_segmentation.job_domain_profile_by_cluster_58772a8"),
                            labels={"color": _src("page03_segmentation.cluster_c137a7f")},
                        )

            elif family == _src("common.skills_66d0f52"):
                d = e["cluster_skill_profile"].copy()
                if clusters and not d.empty:
                    d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                if not d.empty:
                    topn = st.slider(
                        _tr("page03_segmentation.top_skill_rows_23457c4"),
                        10,
                        50,
                        20,
                        key="p3_a8_skill_topn",
                    )
                    d = d.sort_values("records", ascending=False).head(topn)
                    fig = px.bar(
                        d,
                        x="skill",
                        y="records",
                        color=d["cluster"].astype(str),
                        barmode="group",
                        text="records",
                        title=_src(
                            "page03_segmentation.normalized_skill_frequency_by_cluster_c54a7b0"
                        ),
                        labels={"color": _src("page03_segmentation.cluster_c137a7f")},
                    )

            elif family == _src("common.experience_8eab0f0"):
                mode = st.radio(
                    _tr("page03_segmentation.experience_chart_1c32c79"),
                    [
                        _src("page03_segmentation.years_distribution_824c692"),
                        _src("page03_segmentation.experience_level_salary_629a6fd"),
                    ],
                    horizontal=True,
                    key="p3_a8_exp_mode",
                    format_func=_option_label(),
                )
                if mode == _src("page03_segmentation.years_distribution_824c692"):
                    d = e["cluster_years_distribution"].copy()
                    if clusters and not d.empty:
                        d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                    if not d.empty:
                        fig = px.bar(
                            d,
                            x="years_of_experience",
                            y="records",
                            color=d["cluster"].astype(str),
                            barmode="group",
                            text="records",
                            title=_src(
                                "page03_segmentation.years_of_experience_distribution_2a3094f"
                            ),
                            labels={"color": _src("page03_segmentation.cluster_c137a7f")},
                        )
                else:
                    d = e["cluster_experience_level_profile"].copy()
                    if clusters and not d.empty:
                        d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                    if not d.empty and "salary_mean_posthoc" in d.columns:
                        fig = px.bar(
                            d,
                            x="experience_level",
                            y="salary_mean_posthoc",
                            color=d["cluster"].astype(str),
                            barmode="group",
                            text="salary_mean_posthoc",
                            title=_src(
                                "page03_segmentation.post_hoc_salary_by_experience_level_and_4fdd75c"
                            ),
                            labels={"color": _src("page03_segmentation.cluster_c137a7f")},
                        )
                        fig.update_traces(texttemplate="$%{text:,.0f}")

            elif family == _src("common.company_de4743c"):
                d = e["cluster_company_profile"].copy()
                if not d.empty and "feature" in d.columns:
                    options = [
                        x
                        for x in ["company_size", "industry", "remote_work"]
                        if x in d["feature"].astype(str).unique()
                    ]
                    if options:
                        feature = st.selectbox(
                            _tr("page03_segmentation.company_feature_522213a"),
                            options,
                            key="p3_a8_company_feature",
                            format_func=_option_label(),
                        )
                        d = d[d["feature"] == feature].copy()
                        if clusters:
                            d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                        focus = st.multiselect(
                            _tr("page03_segmentation.focus_categories_91d2bfa"),
                            sorted(d["category"].astype(str).unique()),
                            key="p3_a8_company_focus",
                            format_func=_option_label(),
                        )
                        if focus:
                            d = d[d["category"].astype(str).isin(focus)]
                        fig = px.bar(
                            d,
                            x="category",
                            y="records",
                            color=d["cluster"].astype(str),
                            barmode="group",
                            text="records",
                            title=_tr("report.profile", value=_label(feature)),
                            labels={"color": _src("page03_segmentation.cluster_c137a7f")},
                        )

            elif family == _src("common.geography_f3c7380"):
                d = e["cluster_geography_profile"].copy()
                if not d.empty and "geo_level" in d.columns:
                    options = [
                        x for x in ["country", "city"] if x in d["geo_level"].astype(str).unique()
                    ]
                    if options:
                        level = st.selectbox(
                            _tr("page03_segmentation.geography_level_3abb8fc"),
                            options,
                            key="p3_a8_geo_level",
                            format_func=_option_label(),
                        )
                        d = d[d["geo_level"] == level].copy()
                        if clusters:
                            d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                        d = d.sort_values("records", ascending=False).head(30)
                        fig = px.bar(
                            d,
                            x="category",
                            y="records",
                            color=d["cluster"].astype(str),
                            barmode="group",
                            text="records",
                            title=_tr(
                                "page03_segmentation.value0_profile_by_cluster_e392671",
                                value0=f"{level.title()}",
                            ),
                            labels={"color": _src("page03_segmentation.cluster_c137a7f")},
                        )

            else:
                d = e["cluster_market_signal_values"].copy()
                if not d.empty:
                    market_options = [
                        x
                        for x in [
                            "demand_score",
                            "demand_growth_yoy_pct",
                            "benefits_score_10",
                            "ai_salary_premium_pct",
                        ]
                        if x in d.columns
                    ]
                    if market_options:
                        market_metric = st.selectbox(
                            _tr("page03_segmentation.market_signal_5e9adbe"),
                            market_options,
                            key="p3_a8_market_metric",
                            format_func=_option_label(),
                        )
                        d = _filter_rows(d, clusters, categories, countries)
                        if not d.empty:
                            fig = px.box(
                                d,
                                x=d["cluster"].astype(str),
                                y=market_metric,
                                points="outliers",
                                title=_tr(
                                    "page03_segmentation.value0_by_cluster_a7a96cf",
                                    value0=f"{market_metric.replace('_', ' ')}",
                                ),
                                labels={"x": _src("page03_segmentation.cluster_c137a7f")},
                            )

            if fig is not None:
                show_plot(st, fig, "p3_a8_family_chart")
            else:
                st.info(
                    _tr("page03_segmentation.no_persisted_evidence_is_available_for_this_02ca7e8")
                )
            _show_saved_insight(
                st, insights, family, _tr("report.interpretation", value=_label(family))
            )

        with out_tabs[3]:
            geo = e["country_cluster_profile"].copy()
            if clusters and not geo.empty:
                geo = geo[geo["cluster"].astype(int).isin(list(map(int, clusters)))]
            if countries and not geo.empty:
                geo = geo[geo["country"].astype(str).isin(countries)]
            if not geo.empty:
                geo["cluster_label"] = _src("page03_segmentation.cluster_cac75ce") + geo[
                    "cluster"
                ].astype(int).astype(str)
                fig = px.scatter_geo(
                    geo,
                    locations="country",
                    locationmode=_src("page03_segmentation.country_names_28a73cb"),
                    size="records",
                    color="cluster_label",
                    hover_name="country",
                    hover_data={"salary_mean_posthoc": ":,.0f", "records": True}
                    if "salary_mean_posthoc" in geo.columns
                    else {"records": True},
                    projection=_src("page03_segmentation.natural_earth_c8d4369"),
                    title=_src("page03_segmentation.country_level_segment_footprint_952f75c"),
                )
                show_plot(st, fig, "p3_a8_geo_map")

            city = e["city_cluster_profile"].copy()
            if clusters and not city.empty:
                city = city[city["cluster"].astype(int).isin(list(map(int, clusters)))]
            if countries and not city.empty:
                city = city[city["country"].astype(str).isin(countries)]
            if not city.empty:
                city = city.sort_values("records", ascending=False).head(30)
                ycol = "salary_mean_posthoc" if "salary_mean_posthoc" in city.columns else "records"
                fig = px.scatter(
                    city,
                    x="records",
                    y=ycol,
                    size="records",
                    color=city["cluster"].astype(str),
                    text="city",
                    hover_data=["country"],
                    title=_src(
                        "page03_segmentation.city_market_view_posting_volume_vs_post_933d90c"
                    )
                    if ycol == "salary_mean_posthoc"
                    else _src("page03_segmentation.city_market_view_posting_volume_e51da31"),
                    labels={
                        "color": _src("page03_segmentation.cluster_c137a7f"),
                        ycol: _src("page03_segmentation.mean_salary_usd_5e08331")
                        if ycol == "salary_mean_posthoc"
                        else _src("page01_data_basic_clean.records_47a84e9"),
                    },
                )
                fig.update_traces(textposition=_src("model_evidence.top_center_24b3167"))
                show_plot(st, fig, "p3_a8_city_scatter")
            _show_saved_insight(
                st,
                insights,
                "geography",
                _src("page03_segmentation.geography_interpretation_e7d51ed"),
            )

        with out_tabs[4]:
            st.caption(
                _tr(
                    "page03_segmentation.raw_persisted_evidence_for_auditability_and_reproducibility_47d7be0"
                )
            )
            table_specs = [
                (
                    "feature_space_option_summary",
                    _src("page03_segmentation.feature_space_option_summary_o1_vs_o2_537dc2e"),
                    "p3_a8_option_summary",
                    360,
                ),
                (
                    "feature_space_candidate_metrics",
                    _src(
                        "page03_segmentation.feature_space_candidate_metrics_o1_o2_algorithm_091a01c"
                    ),
                    "p3_a8_option_metrics",
                    480,
                ),
                (
                    "feature_space_pairwise_ari",
                    _src("page03_segmentation.o1_vs_o2_pairwise_ari_35e2f04"),
                    "p3_a8_option_ari",
                    260,
                ),
                (
                    "correlation_feature_selection",
                    _src("page03_segmentation.o2_correlation_feature_selection_7ba3d81"),
                    "p3_a8_o2_selection",
                    480,
                ),
                (
                    "correlation_redundancy_pairs",
                    _src("page03_segmentation.o2_redundancy_pairs_eafe230"),
                    "p3_a8_o2_pairs",
                    480,
                ),
                (
                    "correlation_family_summary",
                    _src("page03_segmentation.o2_retention_by_family_a1ad72d"),
                    "p3_a8_o2_family",
                    320,
                ),
                (
                    "representation_summary",
                    _src("page03_segmentation.o1_representation_summary_r0_to_r4_6ece7c3"),
                    "p3_a8_ev_rep_summary",
                    360,
                ),
                (
                    "representation_candidate_metrics",
                    _src(
                        "page03_segmentation.representation_candidate_metrics_all_r_algorithm_k_c42d928"
                    ),
                    "p3_a8_ev_rep_metrics",
                    480,
                ),
                (
                    "representation_pairwise_ari",
                    _src("page03_segmentation.representation_pairwise_ari_43531b6"),
                    "p3_a8_ev_rep_ari",
                    320,
                ),
                (
                    "representation_pca_variance",
                    _src("page03_segmentation.representation_pca_component_variance_fe939a3"),
                    "p3_a8_ev_rep_pca",
                    480,
                ),
                (
                    "representation_top_loadings",
                    _src("page03_segmentation.representation_top_loadings_4d17c02"),
                    "p3_a8_ev_rep_loadings",
                    480,
                ),
                (
                    "representation_resample_stability_runs",
                    _src("page03_segmentation.representation_resample_stability_runs_1e433d0"),
                    "p3_a8_ev_rep_resample",
                    480,
                ),
                (
                    "feature_dependency_summary",
                    _src("page03_segmentation.feature_dependency_summary_bdcff48"),
                    "p3_a8_ev_dependency",
                    300,
                ),
                (
                    "cluster_evaluation",
                    _src("page03_segmentation.official_cluster_candidate_evaluation_40f3584"),
                    "p3_a8_eval",
                    440,
                ),
                (
                    "resample_stability_runs",
                    _src("page03_segmentation.official_subsample_stability_runs_c3a7d81"),
                    "p3_a8_resample_runs",
                    420,
                ),
                (
                    "pca_variance",
                    _src("page03_segmentation.selected_pca_variance_contract_76938a6"),
                    "p3_a8_pca_variance",
                    420,
                ),
                (
                    "family_balance_diagnostics",
                    _src("page03_segmentation.family_balance_diagnostics_0d129e8"),
                    "p3_a8_family_balance",
                    360,
                ),
                (
                    "cluster_profiles",
                    _src("page03_segmentation.official_cluster_profiles_81202b1"),
                    "p3_a8_profiles",
                    360,
                ),
                (
                    "candidate_cluster_balance",
                    _src("page03_segmentation.candidate_cluster_balance_bca9db5"),
                    "p3_a8_candidate_balance",
                    420,
                ),
                (
                    "candidate_cluster_coordinates",
                    _src("page03_segmentation.candidate_cluster_coordinates_850520c"),
                    "p3_a8_candidate_coords",
                    480,
                ),
                (
                    "cluster_skill_profile",
                    _src("page03_segmentation.offline_skill_profile_5f64d12"),
                    "p3_a8_skills",
                    450,
                ),
            ]
            for key, title, dl_key, height in table_specs:
                df = e.get(key, pd.DataFrame())
                if df is not None and not df.empty:
                    downloadable_table(st, df, title, dl_key, height=height)
