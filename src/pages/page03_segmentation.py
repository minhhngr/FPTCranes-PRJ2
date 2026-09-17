from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
        "family": "Job Domain",
        "raw_fields": "job_title, job_category",
        "representation": "Nominal one-hot block",
    },
    {
        "family": "Experience & Education",
        "raw_fields": "years_of_experience, education_required",
        "representation": "Scaled numeric + nominal one-hot",
    },
    {
        "family": "Company & Work Mode",
        "raw_fields": "remote_work, company_size, industry",
        "representation": "Nominal one-hot block",
    },
    {
        "family": "Geography",
        "raw_fields": "city, country",
        "representation": "Nominal one-hot block",
    },
    {
        "family": "Demand / Benefits",
        "raw_fields": "demand_score, benefits_score_10",
        "representation": "Scaled numeric block",
    },
    {
        "family": "Skills",
        "raw_fields": "required_skills, skill_count",
        "representation": "Normalized multi-hot skill tokens + scaled skill_count",
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


def _show_saved_insight(st, insights: dict, key: str, title: str = "Data-driven interpretation"):
    card = insights.get(key, {})
    interpretation_card(
        st,
        card.get("observed", "No run-level interpretation was persisted."),
        card.get("interpretation", "Re-run the offline pipeline to refresh this evidence."),
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
        f"`{name}` is not available in the active run. Run `python pipeline.py` once with the current codebase, "
        "then refresh Streamlit so Branch A can read the persisted evidence."
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
    st.markdown(f"#### {rep_id} — {rep_row.get('representation_label', '')}")
    purpose = rep_row.get("purpose", "")
    if pd.notna(purpose) and str(purpose).strip():
        st.caption(str(purpose))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Algorithm", str(rep_row.get("algorithm", "—")))
    c2.metric("Selected K", str(int(rep_row["k"])) if pd.notna(rep_row.get("k")) else "—")
    c3.metric("Silhouette", _fmt_value(rep_row.get("silhouette")))
    c4.metric("Seed ARI", _fmt_value(rep_row.get("stability_ari")))
    c5.metric("Subsample ARI", _fmt_value(rep_row.get("resample_stability_ari_mean")))
    c6.metric("Min cluster", _fmt_value(rep_row.get("min_cluster_share"), digits=1, pct=True))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Raw inputs",
        str(int(rep_row["raw_input_count"])) if pd.notna(rep_row.get("raw_input_count")) else "—",
    )
    c2.metric(
        "Latent dimensions",
        str(int(rep_row["latent_dimensions"]))
        if pd.notna(rep_row.get("latent_dimensions"))
        else "—",
    )
    c3.metric(
        "Variance summary", _fmt_value(rep_row.get("variance_retained_summary"), digits=1, pct=True)
    )
    c4.metric("Cross-rep ARI", _fmt_value(rep_row.get("mean_cross_representation_ari")))

    excluded = str(rep_row.get("excluded_features", "None"))
    st.markdown(f"**Excluded raw feature(s):** `{excluded}`")
    selected_inputs = rep_row.get("selected_input_features", "")
    if pd.notna(selected_inputs) and str(selected_inputs).strip():
        with st.expander("Raw input contract for this representation", expanded=False):
            st.write(str(selected_inputs))

    # ---- Full K=2..8 candidate evidence for this representation ----
    cm = candidate_metrics.copy()
    if not cm.empty and "representation_id" in cm.columns:
        cm = cm[cm["representation_id"].astype(str) == str(rep_id)].copy()
    if not cm.empty:
        st.markdown("##### KMeans / GMM candidate evidence for this representation")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(
                cm.sort_values(["algorithm", "k"]),
                x="k",
                y="silhouette",
                color="algorithm",
                markers=True,
                text="silhouette",
                title=f"{rep_id} — silhouette by K",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="top center")
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
                title=f"{rep_id} — {'subsample' if y.startswith('resample') else 'seed'} stability",
            )
            gate = rationale.get(
                "resample_stability_threshold"
                if y.startswith("resample")
                else "stability_threshold"
            )
            if gate is not None:
                fig.add_hline(y=float(gate), line_dash="dash", annotation_text="gate")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="top center")
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
                    title=f"{rep_id} — minimum cluster share",
                )
                gate = rationale.get("minimum_cluster_share_threshold")
                if gate is not None:
                    fig.add_hline(y=float(gate), line_dash="dash", annotation_text="balance gate")
                fig.update_yaxes(tickformat=".0%")
                fig.update_traces(texttemplate="%{text:.1%}", textposition="top center")
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
                    title=f"{rep_id} — supporting diagnostic: {support}",
                )
                fig.update_traces(texttemplate="%{text:.3f}", textposition="top center")
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
            f"{rep_id} — all candidate metrics",
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
        st.markdown("##### PCA / latent-space evidence for this representation")
        if (
            "family" in pv.columns
            and pv["family"].notna().any()
            and pv["family"].astype(str).nunique() > 1
        ):
            families = sorted(pv["family"].dropna().astype(str).unique())
            fam = st.selectbox("Family PCA", families, key=f"p3_{rep_id}_family_pca")
            pv_plot = pv[pv["family"].astype(str) == fam].copy()
        else:
            pv_plot = pv.copy()

        xcol = "component_number" if "component_number" in pv_plot.columns else "component"
        if xcol in pv_plot.columns and "explained_variance_ratio" in pv_plot.columns:
            fig = go.Figure()
            fig.add_bar(
                x=pv_plot[xcol],
                y=100 * pd.to_numeric(pv_plot["explained_variance_ratio"], errors="coerce"),
                name="Individual variance",
            )
            if "cumulative_variance" in pv_plot.columns:
                fig.add_scatter(
                    x=pv_plot[xcol],
                    y=100 * pd.to_numeric(pv_plot["cumulative_variance"], errors="coerce"),
                    mode="lines+markers",
                    name="Cumulative variance",
                )
            fig.update_layout(
                title=f"{rep_id} — PCA explained variance",
                xaxis_title="Component",
                yaxis_title="Variance (%)",
            )
            show_plot(st, fig, f"p3_{rep_id}_pca")
        downloadable_table(
            st, pv, f"{rep_id} — PCA component evidence", f"p3_{rep_id}_pca_table", height=340
        )

    # ---- Representation-specific top loadings ----
    ld = loadings_all.copy()
    if not ld.empty and "representation_id" in ld.columns:
        ld = ld[ld["representation_id"].astype(str) == str(rep_id)].copy()
    if not ld.empty and "absolute_loading" in ld.columns:
        st.markdown("##### Top encoded contributors to the representation")
        if "component" in ld.columns:
            components = list(dict.fromkeys(ld["component"].astype(str).tolist()))
            component = st.selectbox("Component", components, key=f"p3_{rep_id}_loading_component")
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
            title=f"{rep_id} — top absolute loadings",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        show_plot(st, fig, f"p3_{rep_id}_loadings")

    # Raw candidate assignments are intentionally shown as evidence, not recomputed.
    ra = rep_assignments.copy()
    if not ra.empty and "representation_id" in ra.columns:
        ra = ra[ra["representation_id"].astype(str) == str(rep_id)].copy()
    if not ra.empty:
        with st.expander(f"{rep_id} — persisted candidate assignments", expanded=False):
            downloadable_table(
                st, ra, f"{rep_id} candidate assignments", f"p3_{rep_id}_assignments", height=420
            )


def _render_representation_design_detail(
    st,
    rep_id: str,
    rep_row: pd.Series,
    pca_all: pd.DataFrame,
    loadings_all: pd.DataFrame,
) -> None:
    """Render representation construction evidence only (no clustering decision metrics)."""
    st.markdown(f"#### {rep_id} — {rep_row.get('representation_label', '')}")
    purpose = rep_row.get("purpose", "")
    if pd.notna(purpose) and str(purpose).strip():
        st.caption(str(purpose))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Raw inputs",
        str(int(rep_row["raw_input_count"])) if pd.notna(rep_row.get("raw_input_count")) else "—",
    )
    c2.metric(
        "Latent dimensions",
        str(int(rep_row["latent_dimensions"]))
        if pd.notna(rep_row.get("latent_dimensions"))
        else "—",
    )
    c3.metric(
        "Variance retained",
        _fmt_value(rep_row.get("variance_retained_summary"), digits=1, pct=True),
    )
    c4.metric("Excluded fields", str(rep_row.get("excluded_features", "None")))

    selected_inputs = rep_row.get("selected_input_features", "")
    if pd.notna(selected_inputs) and str(selected_inputs).strip():
        with st.expander("Raw input contract for this representation", expanded=False):
            st.write(str(selected_inputs))

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
            fam = st.selectbox("PCA family", families, key=f"p3_design_{rep_id}_family")
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
                name="Individual variance",
            )
            if "cumulative_variance" in pv_plot.columns:
                fig.add_scatter(
                    x=pv_plot[xcol],
                    y=100 * pd.to_numeric(pv_plot["cumulative_variance"], errors="coerce"),
                    mode="lines+markers",
                    name="Cumulative variance",
                )
            fig.update_layout(
                title=f"{rep_id} — explained variance",
                xaxis_title="Component",
                yaxis_title="Variance (%)",
            )
            show_plot(st, fig, f"p3_design_{rep_id}_pca")

    ld = loadings_all.copy()
    if not ld.empty and "representation_id" in ld.columns:
        ld = ld[ld["representation_id"].astype(str) == str(rep_id)].copy()
    if not ld.empty and "absolute_loading" in ld.columns:
        if "component" in ld.columns:
            components = list(dict.fromkeys(ld["component"].astype(str).tolist()))
            component = st.selectbox(
                "Component loading", components, key=f"p3_design_{rep_id}_component"
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
            title=f"{rep_id} — top encoded contributors",
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

    st.markdown("#### O1 vs O2 — feature-space option comparison")
    st.caption(
        "O1 is feature extraction: clustering uses retained PCA latent dimensions. "
        "O2 is feature selection: clustering uses the selected family-balanced encoded X features directly. "
        "Both options use the same KMeans/GMM K=2…8 search and the same robustness gates."
    )

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
    c1.metric("Selected feature space", selected_opt)
    if "cross_option_ari" in opt.columns and opt["cross_option_ari"].notna().any():
        c2.metric("O1 ↔ O2 assignment ARI", _fmt_value(opt["cross_option_ari"].dropna().iloc[0]))
    else:
        c2.metric("O1 ↔ O2 assignment ARI", "—")
    c3.metric("O2 correlation threshold", _fmt_value(rationale.get("correlation_threshold")))
    c4.metric(
        "Decision tolerance",
        _fmt_value(rationale.get("representation_silhouette_tolerance")),
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
                "silhouette": "Silhouette",
                "stability_ari": "Seed ARI",
                "resample_stability_ari_mean": "Subsample ARI",
                "min_cluster_share": "Min cluster share",
            }
            long["metric"] = long["metric"].map(label_map).fillna(long["metric"])
            fig = px.bar(
                long,
                x="option_id",
                y="value",
                color="metric",
                barmode="group",
                text="value",
                title="O1 vs O2 — common clustering evidence",
                labels={
                    "option_id": "Feature-space option",
                    "value": "Metric",
                    "metric": "Evidence",
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
                title="Decision view — separation vs robustness",
                labels={
                    "resample_stability_ari_mean": "Subsample stability ARI",
                    "silhouette": "Silhouette",
                },
            )
            if rationale.get("resample_stability_threshold") is not None:
                fig.add_vline(
                    x=float(rationale["resample_stability_threshold"]),
                    line_dash="dash",
                    annotation_text="stability gate",
                )
            sil = opt["silhouette"].dropna()
            tol = rationale.get("representation_silhouette_tolerance", 0.02)
            if len(sil):
                fig.add_hline(
                    y=float(sil.max()) - float(tol),
                    line_dash="dot",
                    annotation_text="near-best floor",
                )
            fig.update_traces(textposition="top center")
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
                title="Parsimony view — dimensions vs separation",
                labels={
                    "clustering_dimensions": "Clustering dimensions (lower is simpler)",
                    "silhouette": "Silhouette",
                },
            )
            fig.update_traces(textposition="top center")
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
                title="O1 vs O2 — assignment agreement (ARI)",
            )
            show_plot(st, fig, "p3_o1_o2_pairwise_ari")

    # Gate matrix at option level
    st.markdown("##### Option-level gate matrix")
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
                "Seed stability": int(
                    True
                    if seed_thr is None
                    else (
                        pd.notna(row.get("stability_ari"))
                        and float(row.get("stability_ari")) >= float(seed_thr)
                    )
                ),
                "Subsample stability": int(
                    True
                    if sub_thr is None
                    else (
                        pd.notna(row.get("resample_stability_ari_mean"))
                        and float(row.get("resample_stability_ari_mean")) >= float(sub_thr)
                    )
                ),
                "Cluster balance": int(
                    True
                    if bal_thr is None
                    else (
                        pd.notna(row.get("min_cluster_share"))
                        and float(row.get("min_cluster_share")) >= float(bal_thr)
                    )
                ),
                "Near-best separation": int(
                    pd.notna(row.get("silhouette"))
                    and pd.notna(best_sil)
                    and float(row.get("silhouette")) >= best_sil - float(tol)
                ),
                "Eligible": int(bool(row.get("eligible", False))),
            }
        )
    gate_df = pd.DataFrame(gate_rows)
    if not gate_df.empty:
        gate_cols = [
            "Seed stability",
            "Subsample stability",
            "Cluster balance",
            "Near-best separation",
            "Eligible",
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
            title="O1/O2 decision gates",
            xaxis_title="Decision check",
            yaxis_title="Option",
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
        "O1 vs O2 — Feature-space Decision Evidence",
        "p3_o1_o2_option_summary",
        height=330,
    )

    _show_saved_insight(
        st,
        insights,
        "feature_space_selection",
        "Why this feature-space option is selected",
    )


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

    st.markdown("#### R Compare — representation decision evidence")
    st.caption(
        "Compare every persisted representation side-by-side before interpreting the official R. "
        "The charts below explain the decision using the pipeline's saved evidence only."
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
    st.markdown("##### 1. Candidate landscape across all representations")
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
                title="All R — silhouette across algorithm × K",
            )
            fig.update_layout(legend_title_text="Representation / algorithm")
            show_plot(st, fig, "p3_rcompare_all_silhouette")
        else:
            st.info("Candidate-level silhouette evidence is not available for all representations.")

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
                    "All R — subsample robustness across algorithm × K"
                    if stability_col == "resample_stability_ari_mean"
                    else "All R — seed stability across algorithm × K"
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
            fig.update_layout(legend_title_text="Representation / algorithm")
            show_plot(st, fig, "p3_rcompare_all_stability")
        else:
            st.info("Candidate-level stability evidence is not available for all representations.")

    # ------------------------------------------------------------------
    # 2. Winner inside each R: common comparable metrics
    # ------------------------------------------------------------------
    st.markdown("##### 2. Best candidate inside each R — common quality metrics")
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
            "silhouette": "Silhouette",
            "stability_ari": "Seed ARI",
            "resample_stability_ari_mean": "Subsample ARI",
            "min_cluster_share": "Min cluster share",
            "mean_cross_representation_ari": "Cross-R ARI",
        }
        long["metric"] = long["metric"].map(label_map).fillna(long["metric"])
        fig = px.bar(
            long,
            x="representation_id",
            y="value",
            color="metric",
            barmode="group",
            text="value",
            title="R-level comparison — higher is better",
            labels={
                "representation_id": "Representation",
                "value": "Metric value",
                "metric": "Evidence",
            },
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_yaxes(range=[0, 1.05])
        show_plot(st, fig, "p3_rcompare_grouped_metrics")

    # ------------------------------------------------------------------
    # 3. Decision views
    # ------------------------------------------------------------------
    st.markdown("##### 3. Decision views — robustness, separation and parsimony")
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
                title="Separation vs subsample robustness",
                labels={
                    "resample_stability_ari_mean": "Subsample stability ARI",
                    "silhouette": "Silhouette",
                },
            )

            if rationale.get("resample_stability_threshold") is not None:
                fig.add_vline(
                    x=float(rationale["resample_stability_threshold"]),
                    line_dash="dash",
                    annotation_text="stability gate",
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
                    annotation_text="near-best floor",
                )

            fig.update_traces(textposition="top center")
            show_plot(st, fig, "p3_rcompare_sep_vs_stability")
        else:
            st.info("R-level separation/stability evidence is incomplete.")

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
                title="Parsimony vs separation",
                labels={
                    "latent_dimensions": "Latent dimensions (lower is simpler)",
                    "silhouette": "Silhouette",
                    "_selected": "Official R",
                },
            )
            fig.update_traces(textposition="top center")
            show_plot(st, fig, "p3_rcompare_parsimony")
        else:
            st.info("Latent-dimension evidence is not available for the parsimony view.")

    # ------------------------------------------------------------------
    # 4. Gate matrix for each selected-within-R candidate
    # ------------------------------------------------------------------
    st.markdown("##### 4. Representation gate matrix")

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
                "Convergence": int(convergence_pass),
                "Seed stability": int(seed_pass),
                "Subsample stability": int(subsample_pass),
                "Cluster balance": int(balance_pass),
                "Near-best separation": int(near_best),
            }
        )

    gate_df = pd.DataFrame(gate_rows)

    if not gate_df.empty:
        gate_cols = [
            "Convergence",
            "Seed stability",
            "Subsample stability",
            "Cluster balance",
            "Near-best separation",
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
                    "Representation=%{y}<br>Check=%{x}<br>Status=%{text}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title="Gate status by representation",
            xaxis_title="Decision check",
            yaxis_title="Representation",
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
        st.markdown("##### 5. Cross-representation assignment agreement")
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
            title="Cross-R ARI — assignment similarity after representation ablation",
        )
        show_plot(st, fig, "p3_rcompare_cross_ari")

    # ------------------------------------------------------------------
    # 6. Transparent decision table
    # ------------------------------------------------------------------
    st.markdown("##### 6. Transparent R decision table — no composite score")

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
            str(row["representation_id"]): bool(row["Near-best separation"])
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
                    row["Convergence"]
                    and row["Seed stability"]
                    and row["Subsample stability"]
                    and row["Cluster balance"]
                )
        eligible_flag = decision["representation_id"].map(hard_gate_lookup).fillna(False)

    decision["decision_cue"] = "Secondary alternative"

    decision.loc[
        ~eligible_flag,
        "decision_cue",
    ] = "Reject / review — hard gate failed"

    decision.loc[
        eligible_flag & decision["near_best_separation"],
        "decision_cue",
    ] = "Strong candidate — eligible and near-best"

    decision.loc[
        decision["representation_id"].eq(str(selected_representation)),
        "decision_cue",
    ] = "OFFICIAL representation"

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
        "R Comparison — Decision Evidence",
        "p3_rcompare_decision_table",
        height=430,
    )

    # ------------------------------------------------------------------
    # 7. Dynamic conclusion
    # ------------------------------------------------------------------
    selected = decision[decision["representation_id"].astype(str) == str(selected_representation)]

    if not selected.empty:
        s = selected.iloc[0]
        observed = (
            f"Official representation: {selected_representation}. "
            f"Within this R, the selected candidate is "
            f"{s.get('algorithm', '—')} K={s.get('k', '—')}; "
            f"silhouette={_fmt_value(s.get('silhouette'))}, "
            f"subsample ARI="
            f"{_fmt_value(s.get('resample_stability_ari_mean'))}, "
            f"minimum cluster share="
            f"{_fmt_value(s.get('min_cluster_share'), digits=1, pct=True)}, "
            f"cross-R ARI="
            f"{_fmt_value(s.get('mean_cross_representation_ari'))}, "
            f"latent dimensions="
            f"{int(s['latent_dimensions']) if pd.notna(s.get('latent_dimensions')) else '—'}."
        )

        interpretation = (
            "The R decision is not based on silhouette alone. "
            "First retain representations whose selected candidate is valid, "
            "stable and balanced. Then keep those within the near-best "
            "silhouette tolerance. Among that shortlist, compare "
            "cross-representation agreement, subsample robustness and "
            "parsimony. These visuals explain the persisted offline choice "
            "without replacing it."
        )

        action = (
            "Use the official R as the frozen Branch-A representation contract. "
            "Keep the other R variants as robustness evidence and re-evaluate "
            "them only when the dataset or feature policy changes."
        )

        interpretation_card(
            st,
            observed,
            interpretation,
            action,
            "success",
            title="R selection conclusion",
        )


def render(st, root, role="admin"):
    style_page(st)
    st.title("3. AI Job Market Segmentation")
    st.caption(
        "Branch A pipeline — A1 Shared Prepared Feature Base → A2 Feature Families → A3 Family Balancing → "
        "A4 Feature-space Options (O1 PCA / O2 Correlation) → A5 Same Clustering Search for Both Options → "
        "A6 Robustness & O1-vs-O2 Decision → A7 Official Selection → A8 Final Outputs"
    )

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
        st.error(
            "Core Branch-A outputs are missing in the active run. Execute `python pipeline.py` first, "
            "then relaunch Streamlit."
        )
        return

    metric_cols(
        st,
        [
            ("Records clustered", f"{len(assign):,}"),
            ("Feature families", "6"),
            ("Encoded dimensions", meta.get("encoded_dimensions", "—")),
            ("Feature-space option", meta.get("feature_space_option_id", "—")),
            ("Selected representation", meta.get("representation_id", "—")),
            ("Selected solution", f"{meta.get('algorithm', '—')} · K={meta.get('k', '—')}"),
            ("Subsample ARI", _fmt_value(meta.get("resample_stability_ari_mean"))),
        ],
    )
    st.info(
        "Interpret the page from left to right. Salary is excluded from clustering and is used only after labels are frozen "
        "for post-hoc interpretation. Streamlit reads persisted evidence only; it does not refit PCA, KMeans or GMM."
    )

    with st.expander("Display filters — outputs only", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        cluster_opts = sorted(assign["cluster"].astype(int).unique().tolist())
        clusters = c1.multiselect(
            "Official clusters", cluster_opts, default=cluster_opts, key="p3_clusters"
        )
        categories = c2.multiselect(
            "Job categories", _safe_options(assign, "job_category"), key="p3_categories"
        )
        countries = c3.multiselect(
            "Countries", _safe_options(assign, "country"), key="p3_countries"
        )
        family = c4.selectbox(
            "Feature family focus",
            ["Job Domain", "Skills", "Experience", "Company", "Geography", "Demand / Benefits"],
            key="p3_family",
        )
        st.caption(
            "These filters change displayed evidence only; the saved clustering solution is never recalculated in the UI."
        )

    focused_rows = _filter_rows(assign, clusters, categories, countries)
    if focused_rows.empty:
        st.warning("No persisted evidence rows match the current display filters.")
        return

    tabs = st.tabs(
        [
            "A1 · Prepared Base",
            "A2 · Feature Families",
            "A3 · Family Balancing",
            "A4 · O1 vs O2 Design",
            "A5 · Same Candidate Search",
            "A6 · Option Compare & Robustness",
            "A7 · Official Selection",
            "A8 · Final Outputs",
        ]
    )

    # ------------------------------------------------------------------
    # A1 — Shared Prepared Feature Base
    # ------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### A1 — Shared Prepared Feature Base")
        st.caption(
            "Start from the cleaned, leakage-safe feature base created by the common foundation. "
            "This stage verifies what information is available before any unsupervised representation is designed."
        )

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
        c1.metric("Records", f"{len(assign):,}")
        c2.metric(
            "Target used for clustering",
            "No" if not bool(meta.get("target_used_for_clustering", False)) else "Yes",
        )
        c3.metric(
            "Selected raw inputs",
            len(selected_inputs) if isinstance(selected_inputs, list) else "—",
        )
        c4.metric("Ablation exclusions", len(excluded) if isinstance(excluded, list) else "—")

        st.markdown("#### Prepared feature contract before representation selection")
        if r0_inputs is not None and pd.notna(r0_inputs):
            st.code(str(r0_inputs), language=None)
        else:
            contract = pd.DataFrame(FAMILY_CONTRACT)
            st.dataframe(
                contract[["family", "raw_fields"]], use_container_width=True, hide_index=True
            )

        st.markdown("#### Official feature contract after representation selection")
        st.code(str(selected_inputs), language=None)
        if excluded:
            st.warning("Official representation excludes: " + ", ".join(map(str, excluded)))
        st.success("Next → organize the prepared features into six interpretable feature families.")

    # ------------------------------------------------------------------
    # A2 — Feature Families / encoding
    # ------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### A2 — Six Feature Families")
        st.caption(
            "Group related variables before clustering so the representation remains interpretable. "
            "Encoding is performed inside each family: one-hot for nominal fields, multi-hot for skills, and scaling for numeric signals."
        )

        contract = pd.DataFrame(FAMILY_CONTRACT)
        dims = e["family_dimensions"].copy()
        if not dims.empty and "family" in dims.columns:
            contract = contract.merge(dims, on="family", how="left")
        st.dataframe(contract, use_container_width=True, hide_index=True)

        if not dims.empty and "encoded_dimensions" in dims.columns:
            fig = px.bar(
                dims.sort_values("encoded_dimensions"),
                y="family",
                x="encoded_dimensions",
                orientation="h",
                text="encoded_dimensions",
                hover_data=_available(dims, ["balance_weight"]),
                title="Encoded dimensions by feature family",
            )
            fig.update_traces(textposition="outside")
            show_plot(st, fig, "p3_a2_family_dims")
        st.info(
            "Interpretation: encoded dimension count is descriptive, not a reason to give a family more influence. "
            "The next stage explicitly balances family contribution."
        )

    # ------------------------------------------------------------------
    # A3 — Family Balancing
    # ------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### A3 — Family Balancing")
        st.caption(
            "High-dimensional blocks such as skills or geography can dominate Euclidean distance simply because they contain more columns. "
            "Family balancing equalizes structural influence before PCA and clustering."
        )
        dims = e["family_dimensions"].copy()
        diag = e["family_balance_diagnostics"].copy()
        if dims.empty or diag.empty:
            _render_missing(st, "family_dimensions.csv / family_balance_diagnostics.csv")
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
                        title="Applied family balance weight",
                    )
                    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                    show_plot(st, fig, "p3_a3_weights")
            with c2:
                fig = go.Figure()
                if "mean_row_l2_before" in diag.columns:
                    fig.add_bar(
                        name="Before balancing",
                        y=diag["family"],
                        x=diag["mean_row_l2_before"],
                        orientation="h",
                        text=diag["mean_row_l2_before"],
                        texttemplate="%{text:.2f}",
                    )
                if "mean_row_l2_after" in diag.columns:
                    fig.add_bar(
                        name="After balancing",
                        y=diag["family"],
                        x=diag["mean_row_l2_after"],
                        orientation="h",
                        text=diag["mean_row_l2_after"],
                        texttemplate="%{text:.2f}",
                    )
                fig.update_layout(
                    barmode="group", title="Mean row L2 norm — before vs after balancing"
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
                    title="Family block Frobenius norm — before vs after",
                )
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                show_plot(st, fig, "p3_a3_frobenius")

            downloadable_table(
                st, diag, "Family Balance Diagnostics", "p3_a3_family_balance", height=360
            )
        st.success(
            "Next → build alternative latent representations and test whether segmentation depends on dominant features."
        )

    # ------------------------------------------------------------------
    # A4 — Feature-space construction: O1 PCA vs O2 Correlation
    # ------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### A4 — Feature-space Construction: O1 PCA vs O2 Correlation")
        st.caption(
            "Two independent feature-space strategies are built from the same family-balanced DEV data. "
            "O1 performs feature extraction through PCA; O2 performs feature selection by removing highly correlated encoded features. "
            "Salary is not used in either option."
        )

        o1_tab, o2_tab = st.tabs(
            [
                "O1 · PCA-based latent representation",
                "O2 · Correlation-based feature selection",
            ]
        )

        with o1_tab:
            st.markdown("#### O1 — PCA-based latent representation")
            st.info(
                "O1 does not force clustering to use only PC1 and PC2. "
                "The pipeline retains enough PCs to meet the configured variance rule; PC1/PC2 are display-only. "
                "R0–R4 are internal O1 robustness variants."
            )

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
                    "O1 PCA Representation Design — R0 to R4",
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
                "O1 output = a lower-dimensional latent matrix PC1…PCn for clustering. "
                "PC1/PC2 are only used to visualize the final partition."
            )

        with o2_tab:
            st.markdown("#### O2 — Correlation-based feature selection")
            st.info(
                "O2 measures feature-to-feature Pearson correlation on DEV only. "
                "It does NOT correlate X with annual_salary_usd. "
                "When |r| reaches the configured threshold, a redundant encoded feature is removed. "
                "The retained encoded features are used directly for clustering."
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
            c1.metric("Correlation threshold", _fmt_value(threshold))
            c2.metric("Encoded features before", str(before))
            c3.metric("Selected encoded features", str(kept))
            if isinstance(before, (int, float)) and isinstance(kept, (int, float)) and before:
                c4.metric("Reduction", f"{100 * (1 - float(kept) / float(before)):.1f}%")
            else:
                c4.metric("Reduction", "—")

            if not fam.empty:
                fam_plot = fam.copy()
                fig = go.Figure()
                if "encoded_features_before" in fam_plot.columns:
                    fig.add_bar(
                        x=fam_plot["family"],
                        y=fam_plot["encoded_features_before"],
                        name="Before correlation filter",
                        text=fam_plot["encoded_features_before"],
                    )
                if "selected_encoded_features" in fam_plot.columns:
                    fig.add_bar(
                        x=fam_plot["family"],
                        y=fam_plot["selected_encoded_features"],
                        name="Selected for O2",
                        text=fam_plot["selected_encoded_features"],
                    )
                fig.update_layout(
                    barmode="group",
                    title="O2 feature retention by family",
                    xaxis_title="Feature family",
                    yaxis_title="Encoded features",
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
                    title="Top redundant encoded-feature pairs by |Pearson r|",
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
                    title="O2 KEEP / DROP decisions by feature family",
                )
                show_plot(st, fig, "p3_a4_o2_decisions")

                selected_only = sel[_truthy(sel["selected"])] if "selected" in sel.columns else sel
                downloadable_table(
                    st,
                    selected_only,
                    "O2 Selected Encoded Features",
                    "p3_a4_o2_selected_features",
                    height=390,
                )

            if not pairs.empty:
                downloadable_table(
                    st,
                    pairs,
                    "O2 High-correlation Pair Evidence",
                    "p3_a4_o2_corr_pairs",
                    height=390,
                )

            st.caption(
                "O2 output = selected original/encoded X dimensions. "
                "A separate 2D PCA may still be used later for visualization, but PCA does not define O2 clustering."
            )

        st.success(
            "Next → apply the SAME KMeans/GMM K=2…8 search to O1 and O2 so the feature-space choice is evidence-based."
        )

    # ------------------------------------------------------------------
    # A5 — Same clustering candidate search for O1 and O2
    # ------------------------------------------------------------------
    with tabs[4]:
        st.markdown("### A5 — Same Clustering Candidate Search for O1 and O2")
        st.caption(
            "Both feature-space options are evaluated under the same protocol: "
            "KMeans and GMM, K=2…8, the same seed policy and the same fit diagnostics. "
            "This prevents choosing O1 or O2 simply because one option received an easier search."
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
                "Feature-space option to inspect",
                option_choices,
                index=idx,
                key="p3_a5_option",
            )
            view = cm[cm["option_id"].astype(str) == chosen_option].copy()

            if "internal_representation" in view.columns:
                internal = ", ".join(
                    sorted(view["internal_representation"].dropna().astype(str).unique())
                )
                st.caption(f"Internal representation: {internal}")

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
                        title=f"{chosen_option} — silhouette by K",
                    )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition="top center",
                    )
                    show_plot(st, fig, "p3_a5_option_silhouette")

            with c2:
                support_choices = [
                    x for x in ["calinski_harabasz", "davies_bouldin"] if x in view.columns
                ]
                if support_choices:
                    support = st.selectbox(
                        "Supporting fit metric",
                        support_choices,
                        key="p3_a5_option_support",
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
                        textposition="top center",
                    )
                    show_plot(st, fig, "p3_a5_option_support_chart")

            c1, c2 = st.columns(2)
            with c1:
                km = (
                    view[view["algorithm"].astype(str) == "KMeans"].copy()
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
                        title=f"{chosen_option} — KMeans inertia ↓",
                    )
                    fig.update_traces(
                        texttemplate="%{text:.1f}",
                        textposition="top center",
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
                        textposition="top center",
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
                f"{chosen_option} — Candidate Search Evidence",
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
                    title="Direct O1 vs O2 silhouette landscape",
                )
                show_plot(st, fig, "p3_a5_o1_o2_silhouette_landscape")

    # ------------------------------------------------------------------
    # A6 — Robustness, option comparison and O1 internal R comparison
    # ------------------------------------------------------------------
    with tabs[5]:
        st.markdown("### A6 — Robustness & O1-vs-O2 Decision")
        st.caption(
            "Feature-space choice is made only after robustness evidence is available. "
            "The decision order is: validity/convergence → seed stability → subsample stability → "
            "minimum cluster share → near-best silhouette → robustness → parsimony."
        )

        cm = option_metrics.copy()
        if cm.empty:
            _render_missing(st, "feature_space_candidate_metrics.csv")
        else:
            option_choices = sorted(cm["option_id"].dropna().astype(str).unique().tolist())
            default_option = str(meta.get("feature_space_option_id", option_choices[0]))
            idx = option_choices.index(default_option) if default_option in option_choices else 0
            chosen_option = st.selectbox(
                "Option for detailed robustness review",
                option_choices,
                index=idx,
                key="p3_a6_option",
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
                        title=f"{chosen_option} — Seed stability ARI ↑",
                    )
                    if rationale.get("stability_threshold") is not None:
                        fig.add_hline(
                            y=float(rationale["stability_threshold"]),
                            line_dash="dash",
                            annotation_text="seed gate",
                        )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition="top center",
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
                        title=f"{chosen_option} — Subsample stability ARI ↑",
                    )
                    if rationale.get("resample_stability_threshold") is not None:
                        fig.add_hline(
                            y=float(rationale["resample_stability_threshold"]),
                            line_dash="dash",
                            annotation_text="subsample gate",
                        )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition="top center",
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
                        title=f"{chosen_option} — Minimum cluster share",
                    )
                    if rationale.get("minimum_cluster_share_threshold") is not None:
                        fig.add_hline(
                            y=float(rationale["minimum_cluster_share_threshold"]),
                            line_dash="dash",
                            annotation_text="balance gate",
                        )
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_traces(
                        texttemplate="%{text:.1%}",
                        textposition="top center",
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
                            lambda x: "Pass" if str(x).lower() in {"true", "1", "yes"} else "Fail"
                        )
                        fig = px.bar(
                            gm.sort_values("k"),
                            x="k",
                            y=[1] * len(gm),
                            color="convergence_status",
                            text="convergence_status",
                            title=f"{chosen_option} — GMM convergence gate",
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

        # R0-R4 remain the internal robustness study for O1 only.
        with st.expander(
            "O1 internal robustness — compare R0/R1/R2/R3/R4",
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
                    "O1 Dominant-feature Dependency Summary",
                    "p3_a6_o1_dependency",
                    height=280,
                )

        _show_saved_insight(
            st,
            insights,
            "feature_space_selection",
            "Feature-space decision interpretation",
        )

    # ------------------------------------------------------------------
    # A7 — Official Feature Space + Algorithm + K selection
    # ------------------------------------------------------------------
    with tabs[6]:
        st.markdown("### A7 — Official Selection")
        st.caption(
            "A7 freezes the complete Branch-A contract: feature-space option (O1 or O2) + "
            "internal representation where applicable + clustering algorithm + K."
        )

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
        c1.metric("Feature-space option", selected_option)
        c2.metric("Representation", selected_rep)
        c3.metric("Algorithm", selected_alg)
        c4.metric("K", str(selected_k))
        c5.metric(
            "Silhouette",
            _fmt_value(
                rationale.get(
                    "selected_silhouette",
                    meta.get("silhouette"),
                )
            ),
        )
        c6.metric(
            "Subsample ARI",
            _fmt_value(
                rationale.get(
                    "selected_resample_stability_ari_mean",
                    meta.get("resample_stability_ari_mean"),
                )
            ),
        )

        if selected_option == "O1_PCA":
            st.info(
                f"O1 won the feature-space comparison. Its internal PCA representation is {selected_rep}. "
                "The clustering model uses retained latent PCs; PC1/PC2 remain display-only."
            )
        elif selected_option == "O2_CORRELATION":
            st.info(
                "O2 won the feature-space comparison. Clustering uses the correlation-selected "
                "family-balanced encoded features directly; PC1/PC2 are created only for visualization."
            )

        st.markdown(
            "**Decision sequence:** same candidate search → convergence/fit validity → seed stability → "
            "subsample stability → cluster balance → near-best silhouette → robustness → parsimony."
        )

        gate_rows = [
            [
                "Optimizer seed stability",
                "stability_ari",
                rationale.get("stability_threshold"),
                "≥",
            ],
            [
                "Subsample stability",
                "resample_stability_ari_mean",
                rationale.get("resample_stability_threshold"),
                "≥",
            ],
            [
                "Minimum cluster share",
                "min_cluster_share",
                rationale.get("minimum_cluster_share_threshold"),
                "≥",
            ],
            [
                "Feature-space practical tie",
                "silhouette gap",
                rationale.get("representation_silhouette_tolerance"),
                "≤",
            ],
            ["Within-option K tie", "silhouette gap", rationale.get("silhouette_tolerance"), "≤"],
        ]
        st.dataframe(
            pd.DataFrame(
                gate_rows,
                columns=["Gate / rule", "Metric", "Threshold", "Direction"],
            ),
            use_container_width=True,
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
                "Official Feature-space Selection Summary",
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
                    {"candidate": "Selected option", **selected_row},
                    {"candidate": "Runner-up option", **runner},
                ]
            )
            st.markdown("#### Selected feature-space option vs runner-up")
            st.dataframe(
                comp,
                use_container_width=True,
                hide_index=True,
            )

        decision_note = rationale.get("decision_note")
        if decision_note:
            st.success(str(decision_note))

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
                "Official Feature Space — Eligible K Candidates",
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
                chosen["cluster_label"] = "Cluster " + chosen["cluster"].astype(int).astype(str)
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
                            f"Official solution — {selected_option} / "
                            f"{selected_rep} / {selected_alg} K={kval} "
                            "(PC1/PC2 display only)"
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
                            title="Official cluster balance",
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
            "Why this feature space is selected",
        )
        _show_saved_insight(
            st,
            insights,
            "k_selection",
            "Why this Algorithm / K is selected",
        )

    # ------------------------------------------------------------------
    # A8 — Final Outputs
    # ------------------------------------------------------------------
    with tabs[7]:
        st.markdown("### A8 — Final Outputs & Interpretation")
        st.caption(
            "After the official clustering solution is frozen, interpret the segment labels through cluster profiles, feature-family EDA, "
            "geography and post-hoc salary summaries. None of these post-hoc views is used to refit the clusters."
        )

        out_tabs = st.tabs(
            [
                "Cluster Labels & Map",
                "Cluster Profiles",
                "Feature-family EDA",
                "Market Geography",
                "Evidence Tables",
            ]
        )

        with out_tabs[0]:
            scatter = focused_rows.copy()
            if {"PC1", "PC2", "cluster"}.issubset(scatter.columns):
                scatter["cluster_label"] = "Cluster " + scatter["cluster"].astype(int).astype(str)
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
                    title="Official segmentation — 2D PCA display",
                )
                show_plot(st, fig, "p3_a8_map")
            downloadable_table(
                st,
                focused_rows,
                "Filtered Official Row-level Assignments",
                "p3_a8_assignments",
                height=420,
            )

        with out_tabs[1]:
            view = (
                prof[prof["cluster"].astype(int).isin(list(map(int, clusters)))]
                if clusters
                else prof.copy()
            )
            st.dataframe(view, use_container_width=True, hide_index=True)
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
                fig.update_layout(barmode="group", title="Numeric cluster profile")
                show_plot(st, fig, "p3_a8_profile")
            if "annual_salary_usd" in focused_rows.columns:
                fig = px.box(
                    focused_rows,
                    x=focused_rows["cluster"].astype(str),
                    y="annual_salary_usd",
                    points="outliers",
                    title="Observed salary distribution by segment — post-hoc only",
                    labels={"x": "Cluster", "annual_salary_usd": "Annual salary (USD)"},
                )
                show_plot(st, fig, "p3_a8_salary")
                st.caption("Salary is descriptive here; it was not an input to clustering.")
            _show_saved_insight(st, insights, "profiles", "Cluster profile interpretation")

        with out_tabs[2]:
            st.markdown(f"#### Feature-family EDA — {family}")
            fig = None
            if family == "Job Domain":
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
                            "Job-domain view",
                            metric_options,
                            horizontal=True,
                            key="p3_a8_job_metric",
                        )
                        focus = st.multiselect(
                            "Focus job categories",
                            sorted(d["job_category"].astype(str).unique()),
                            key="p3_a8_job_focus",
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
                            title="Job-domain profile by cluster",
                            labels={"color": "Cluster"},
                        )

            elif family == "Skills":
                d = e["cluster_skill_profile"].copy()
                if clusters and not d.empty:
                    d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                if not d.empty:
                    topn = st.slider("Top skill rows", 10, 50, 20, key="p3_a8_skill_topn")
                    d = d.sort_values("records", ascending=False).head(topn)
                    fig = px.bar(
                        d,
                        x="skill",
                        y="records",
                        color=d["cluster"].astype(str),
                        barmode="group",
                        text="records",
                        title="Normalized skill frequency by cluster",
                        labels={"color": "Cluster"},
                    )

            elif family == "Experience":
                mode = st.radio(
                    "Experience chart",
                    ["Years distribution", "Experience-level salary"],
                    horizontal=True,
                    key="p3_a8_exp_mode",
                )
                if mode == "Years distribution":
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
                            title="Years-of-experience distribution",
                            labels={"color": "Cluster"},
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
                            title="Post-hoc salary by experience level and cluster",
                            labels={"color": "Cluster"},
                        )
                        fig.update_traces(texttemplate="$%{text:,.0f}")

            elif family == "Company":
                d = e["cluster_company_profile"].copy()
                if not d.empty and "feature" in d.columns:
                    options = [
                        x
                        for x in ["company_size", "industry", "remote_work"]
                        if x in d["feature"].astype(str).unique()
                    ]
                    if options:
                        feature = st.selectbox(
                            "Company feature", options, key="p3_a8_company_feature"
                        )
                        d = d[d["feature"] == feature].copy()
                        if clusters:
                            d = d[d["cluster"].astype(int).isin(list(map(int, clusters)))]
                        focus = st.multiselect(
                            "Focus categories",
                            sorted(d["category"].astype(str).unique()),
                            key="p3_a8_company_focus",
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
                            title=f"{feature.replace('_', ' ')} profile",
                            labels={"color": "Cluster"},
                        )

            elif family == "Geography":
                d = e["cluster_geography_profile"].copy()
                if not d.empty and "geo_level" in d.columns:
                    options = [
                        x for x in ["country", "city"] if x in d["geo_level"].astype(str).unique()
                    ]
                    if options:
                        level = st.selectbox("Geography level", options, key="p3_a8_geo_level")
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
                            title=f"{level.title()} profile by cluster",
                            labels={"color": "Cluster"},
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
                            "Market signal", market_options, key="p3_a8_market_metric"
                        )
                        d = _filter_rows(d, clusters, categories, countries)
                        if not d.empty:
                            fig = px.box(
                                d,
                                x=d["cluster"].astype(str),
                                y=market_metric,
                                points="outliers",
                                title=f"{market_metric.replace('_', ' ')} by cluster",
                                labels={"x": "Cluster"},
                            )

            if fig is not None:
                show_plot(st, fig, "p3_a8_family_chart")
            else:
                st.info("No persisted evidence is available for this family/filter combination.")
            _show_saved_insight(st, insights, family, f"{family} interpretation")

        with out_tabs[3]:
            geo = e["country_cluster_profile"].copy()
            if clusters and not geo.empty:
                geo = geo[geo["cluster"].astype(int).isin(list(map(int, clusters)))]
            if countries and not geo.empty:
                geo = geo[geo["country"].astype(str).isin(countries)]
            if not geo.empty:
                geo["cluster_label"] = "Cluster " + geo["cluster"].astype(int).astype(str)
                fig = px.scatter_geo(
                    geo,
                    locations="country",
                    locationmode="country names",
                    size="records",
                    color="cluster_label",
                    hover_name="country",
                    hover_data={"salary_mean_posthoc": ":,.0f", "records": True}
                    if "salary_mean_posthoc" in geo.columns
                    else {"records": True},
                    projection="natural earth",
                    title="Country-level segment footprint",
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
                    title="City market view — posting volume vs post-hoc mean salary"
                    if ycol == "salary_mean_posthoc"
                    else "City market view — posting volume",
                    labels={
                        "color": "Cluster",
                        ycol: "Mean salary (USD)" if ycol == "salary_mean_posthoc" else "Records",
                    },
                )
                fig.update_traces(textposition="top center")
                show_plot(st, fig, "p3_a8_city_scatter")
            _show_saved_insight(st, insights, "geography", "Geography interpretation")

        with out_tabs[4]:
            st.caption("Raw persisted evidence for auditability and reproducibility.")
            table_specs = [
                (
                    "feature_space_option_summary",
                    "Feature-space Option Summary — O1 vs O2",
                    "p3_a8_option_summary",
                    360,
                ),
                (
                    "feature_space_candidate_metrics",
                    "Feature-space Candidate Metrics — O1/O2 × algorithm × K",
                    "p3_a8_option_metrics",
                    480,
                ),
                ("feature_space_pairwise_ari", "O1 vs O2 Pairwise ARI", "p3_a8_option_ari", 260),
                (
                    "correlation_feature_selection",
                    "O2 Correlation Feature Selection",
                    "p3_a8_o2_selection",
                    480,
                ),
                ("correlation_redundancy_pairs", "O2 Redundancy Pairs", "p3_a8_o2_pairs", 480),
                ("correlation_family_summary", "O2 Retention by Family", "p3_a8_o2_family", 320),
                (
                    "representation_summary",
                    "O1 Representation Summary — R0 to R4",
                    "p3_a8_ev_rep_summary",
                    360,
                ),
                (
                    "representation_candidate_metrics",
                    "Representation Candidate Metrics — all R × algorithm × K",
                    "p3_a8_ev_rep_metrics",
                    480,
                ),
                (
                    "representation_pairwise_ari",
                    "Representation Pairwise ARI",
                    "p3_a8_ev_rep_ari",
                    320,
                ),
                (
                    "representation_pca_variance",
                    "Representation PCA / Component Variance",
                    "p3_a8_ev_rep_pca",
                    480,
                ),
                (
                    "representation_top_loadings",
                    "Representation Top Loadings",
                    "p3_a8_ev_rep_loadings",
                    480,
                ),
                (
                    "representation_resample_stability_runs",
                    "Representation Resample Stability Runs",
                    "p3_a8_ev_rep_resample",
                    480,
                ),
                (
                    "feature_dependency_summary",
                    "Feature Dependency Summary",
                    "p3_a8_ev_dependency",
                    300,
                ),
                ("cluster_evaluation", "Official Cluster Candidate Evaluation", "p3_a8_eval", 440),
                (
                    "resample_stability_runs",
                    "Official Subsample Stability Runs",
                    "p3_a8_resample_runs",
                    420,
                ),
                ("pca_variance", "Selected PCA Variance Contract", "p3_a8_pca_variance", 420),
                (
                    "family_balance_diagnostics",
                    "Family Balance Diagnostics",
                    "p3_a8_family_balance",
                    360,
                ),
                ("cluster_profiles", "Official Cluster Profiles", "p3_a8_profiles", 360),
                (
                    "candidate_cluster_balance",
                    "Candidate Cluster Balance",
                    "p3_a8_candidate_balance",
                    420,
                ),
                (
                    "candidate_cluster_coordinates",
                    "Candidate Cluster Coordinates",
                    "p3_a8_candidate_coords",
                    480,
                ),
                ("cluster_skill_profile", "Offline Skill Profile", "p3_a8_skills", 450),
            ]
            for key, title, dl_key, height in table_specs:
                df = e.get(key, pd.DataFrame())
                if df is not None and not df.empty:
                    downloadable_table(st, df, title, dl_key, height=height)
