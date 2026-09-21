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

from .common import *


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("navigation.page07"))
    st.caption(
        _tr(
            "page07_integrated.decision_layer_combine_unsupervised_structural_segments_with_c108eca"
        )
    )
    seg = read_csv(root, "07_integrated_insight/segment_salary_summary.csv")
    pred = read_csv(root, "07_integrated_insight/predicted_salary_by_segment.csv")
    geo = read_csv(root, "07_integrated_insight/segment_city_country_summary.csv")
    assign = read_csv(root, "03_ai_job_market_segmentation/cluster_assignments.csv")

    cluster_opts = sorted(assign.cluster.astype(int).unique().tolist())
    with st.expander(_tr("page07_integrated.integrated_insight_filters_0fffb22"), expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        clusters = c1.multiselect(
            _tr("page07_integrated.clusters_92d5e4c"),
            cluster_opts,
            default=cluster_opts,
            key="p7_clusters",
            format_func=_option_label(),
        )
        countries = c2.multiselect(
            _tr("page03_segmentation.countries_8faf7ec"),
            sorted(assign.country.dropna().astype(str).unique()),
            key="p7_country",
            format_func=_option_label(),
        )
        cats = c3.multiselect(
            _tr("page03_segmentation.job_categories_1fd1537"),
            sorted(assign.job_category.dropna().astype(str).unique()),
            key="p7_cat",
            format_func=_option_label(),
        )
        metric = c4.selectbox(
            _tr("page07_integrated.salary_metric_3bb544b"),
            [
                _src("page01_data_basic_clean.mean_727a7d1"),
                _src("page01_data_basic_clean.median_daab7c4"),
            ],
            key="p7_metric",
            format_func=_option_label(),
        )
    d = apply_filters(
        assign, {"cluster": [str(x) for x in clusters], "country": countries, "job_category": cats}
    )
    if d.empty:
        st.warning(_tr("page07_integrated.no_records_match_the_selected_filters_9a7911b"))
        return

    actual_col = "mean" if metric == _src("page01_data_basic_clean.mean_727a7d1") else "median"
    g = (
        d.groupby("cluster")
        .agg(
            records=("annual_salary_usd", "size"),
            salary_mean=("annual_salary_usd", "mean"),
            salary_median=("annual_salary_usd", "median"),
            years_mean=("years_of_experience", "mean"),
            demand_mean=("demand_score", "mean"),
        )
        .reset_index()
    )
    top = g.sort_values(f"salary_{actual_col}", ascending=False).iloc[0]
    metric_cols(
        st,
        [
            (_tr("page07_integrated.segments_in_view_7788536"), len(g)),
            (_tr("page01_data_basic_clean.records_47a84e9"), f"{len(d):,}"),
            (
                _tr("page07_integrated.highest_salary_segment_e204457"),
                _tr("page07_integrated.cluster_value0_b49a215", value0=f"{int(top.cluster)}"),
            ),
            (
                _tr("page07_integrated.highest_value0_salary_3aabe1a", value0=f"{metric.lower()}"),
                money(top[f"salary_{actual_col}"]),
            ),
            (_tr("page07_integrated.mean_demand_e9d78ab"), f"{d.demand_score.mean():.1f}"),
        ],
    )

    tabs = st.tabs(
        [
            _tr("page07_integrated.segment_salary_d7cc939"),
            _tr("page07_integrated.prediction_alignment_5b8b2e5"),
            _tr("common.geography_f3c7380"),
            _tr("page07_integrated.job_market_structure_a65910e"),
            _tr("page01_data_basic_clean.interactive_chart_explorer_236cb24"),
            _tr("page01_data_basic_clean.evidence_tables_21295b1"),
        ]
    )
    with tabs[0]:
        m = g.melt(
            id_vars=["cluster", "records"],
            value_vars=["salary_mean", "salary_median"],
            var_name="metric",
            value_name="salary",
        )
        fig = px.bar(
            m,
            x="cluster",
            y="salary",
            color="metric",
            barmode="group",
            text="salary",
            hover_data=["records"],
            title=_src("page07_integrated.observed_salary_by_structural_segment_182c3d1"),
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        show_plot(st, fig, "p7_seg_salary")
        fig = px.box(
            d,
            x=d.cluster.astype(str),
            y="annual_salary_usd",
            color=d.cluster.astype(str),
            points="outliers",
            title=_src("page07_integrated.salary_distribution_by_segment_7158e74"),
            labels={
                "x": _src("page03_segmentation.cluster_c137a7f"),
                "color": _src("page03_segmentation.cluster_c137a7f"),
            },
        )
        show_plot(st, fig, "p7_seg_box")
        spread = float(g.salary_mean.max() - g.salary_mean.min())
        interpretation_card(
            st,
            _tr(
                "page07_integrated.observed_mean_salary_spread_across_visible_structural_19f27f7",
                value0=f"{money(spread)}",
                value1=f"{int(g.sort_values('salary_mean', ascending=False).iloc[0].cluster)}",
            ),
            _tr("page07_integrated.segmentation_did_not_use_salary_so_this_7b94523"),
            _tr("page07_integrated.use_the_spread_only_as_descriptive_market_1bbb636"),
            "warning",
        )

    with tabs[1]:
        pp = pred[pred.cluster.isin(clusters)] if clusters else pred
        x = pp.melt(
            id_vars=["cluster", "records", "MAE"],
            value_vars=["actual_salary_mean", "predicted_salary_mean"],
            var_name="series",
            value_name="salary",
        )
        fig = px.bar(
            x,
            x="cluster",
            y="salary",
            color="series",
            barmode="group",
            text="salary",
            hover_data=["MAE", "records"],
            title=_src("page07_integrated.locked_test_actual_vs_predicted_mean_salary_d763b5f"),
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        show_plot(st, fig, "p7_actual_pred")
        fig = px.bar(
            pp.sort_values("MAE"),
            x="cluster",
            y="MAE",
            text="MAE",
            title=_src("page07_integrated.locked_test_mae_by_segment_bc9b338"),
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        show_plot(st, fig, "p7_seg_mae")
        st.caption(_tr("page07_integrated.segment_labels_are_not_fed_into_the_b138df8"))
        if not pp.empty:
            worst = pp.sort_values("MAE", ascending=False).iloc[0]
            bestp = pp.sort_values("MAE").iloc[0]
            interpretation_card(
                st,
                _tr(
                    "page07_integrated.locked_test_segment_mae_ranges_from_value0_5868fd4",
                    value0=f"{money(bestp.MAE)}",
                    value1=f"{int(bestp.cluster)}",
                    value2=f"{money(worst.MAE)}",
                    value3=f"{int(worst.cluster)}",
                ),
                _tr(
                    "page07_integrated.prediction_error_varies_across_structural_segments_even_91bc4be"
                ),
                _tr("page07_integrated.monitor_segment_level_error_as_a_diagnostic_13e627d"),
                "warning",
            )

    with tabs[2]:
        ge = (
            d.groupby(["cluster", "country"])
            .agg(records=("annual_salary_usd", "size"), salary_mean=("annual_salary_usd", "mean"))
            .reset_index()
        )
        ge["cluster_label"] = _src("page03_segmentation.cluster_cac75ce") + ge.cluster.astype(
            int
        ).astype(str)
        fig = px.scatter_geo(
            ge,
            locations="country",
            locationmode=_src("page03_segmentation.country_names_28a73cb"),
            size="records",
            color="cluster_label",
            hover_name="country",
            hover_data={"salary_mean": ":,.0f", "records": True},
            projection=_src("page03_segmentation.natural_earth_c8d4369"),
            title=_src("page07_integrated.country_footprint_by_segment_5a103bc"),
        )
        show_plot(st, fig, "p7_country_map")
        city = (
            d.groupby(["cluster", "country", "city"])
            .agg(records=("annual_salary_usd", "size"), salary_mean=("annual_salary_usd", "mean"))
            .reset_index()
        )
        topn = st.slider(
            _tr("page07_integrated.top_cities_by_record_volume_e5d0ab9"),
            10,
            50,
            25,
            key="p7_city_topn",
        )
        city = city.nlargest(topn, "records")
        fig = px.scatter(
            city,
            x="records",
            y="salary_mean",
            size="records",
            color=city.cluster.astype(str),
            text="city",
            hover_data=["country"],
            title=_src("page07_integrated.city_volume_vs_mean_salary_886dfa7"),
            labels={
                "color": _src("page03_segmentation.cluster_c137a7f"),
                "salary_mean": _src("page03_segmentation.mean_salary_usd_5e08331"),
            },
        )
        fig.update_traces(textposition=_src("model_evidence.top_center_24b3167"))
        show_plot(st, fig, "p7_city")
        if len(city):
            topcity = city.sort_values("records", ascending=False).iloc[0]
            interpretation_card(
                st,
                _tr(
                    "page07_integrated.largest_city_footprint_in_the_current_filters_741d476",
                    value0=f"{topcity.city}",
                    value1=f"{topcity.country}",
                    value2=f"{int(topcity.records)}",
                    value3=f"{money(topcity.salary_mean)}",
                ),
                _tr(
                    "page07_integrated.high_volume_locations_can_dominate_aggregate_segment_5a2a9c6"
                ),
                _tr("page07_integrated.compare_country_city_mix_with_job_domain_3b2d12a"),
                "info",
            )

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            dom = d.groupby(["cluster", "job_category"]).size().rename("records").reset_index()
            fig = px.bar(
                dom,
                x="job_category",
                y="records",
                color=dom.cluster.astype(str),
                barmode="stack",
                title=_src("page07_integrated.job_domain_mix_by_segment_f71a16f"),
                labels={"color": _src("page03_segmentation.cluster_c137a7f")},
            )
            show_plot(st, fig, "p7_domain_mix")
        with c2:
            demand = (
                d.groupby("cluster")
                .agg(
                    demand_mean=("demand_score", "mean"),
                    benefits_mean=("benefits_score_10", "mean"),
                    years_mean=("years_of_experience", "mean"),
                )
                .reset_index()
            )
            dm = demand.melt(id_vars=["cluster"], var_name="metric", value_name="value")
            fig = px.bar(
                dm,
                x="cluster",
                y="value",
                color="metric",
                barmode="group",
                text="value",
                title=_src("page07_integrated.demand_benefits_experience_profile_1534dc7"),
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            show_plot(st, fig, "p7_market_profile")
        if len(demand):
            hi = demand.sort_values("demand_mean", ascending=False).iloc[0]
            interpretation_card(
                st,
                _tr(
                    "page07_integrated.cluster_value0_has_the_highest_mean_demand_e541693",
                    value0=f"{int(hi.cluster)}",
                    value1=f"{hi.demand_mean:.1f}",
                ),
                _tr(
                    "page07_integrated.integrated_profiles_combine_independent_branch_a_structure_3a5043d"
                ),
                _tr("page07_integrated.use_them_to_create_descriptive_segment_labels_05b34ce"),
                "info",
            )

    with tabs[4]:
        st.markdown(_tr("page01_data_basic_clean.interactive_chart_explorer_320e7ef"))
        chart = chart_selector(
            st,
            _tr("page07_integrated.choose_market_view_e867087"),
            [
                _tr("page07_integrated.salary_by_cluster_c89a781"),
                _tr("page01_data_basic_clean.salary_by_country_8072edc"),
                _tr("page07_integrated.salary_by_job_category_d159d2e"),
                _tr("page07_integrated.demand_vs_salary_52a951a"),
                _tr("page07_integrated.experience_vs_salary_4a979a5"),
                _tr("page07_integrated.cluster_composition_by_remote_work_4bd46ef"),
            ],
            "p7_chart",
        )
        if chart == _src("page07_integrated.salary_by_cluster_c89a781"):
            gg = d.groupby("cluster").annual_salary_usd.mean().reset_index()
            fig = px.bar(
                gg,
                x="cluster",
                y="annual_salary_usd",
                text="annual_salary_usd",
                title=_src("page07_integrated.mean_salary_by_cluster_d7d584d"),
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == _src("page01_data_basic_clean.salary_by_country_8072edc"):
            gg = (
                d.groupby("country")
                .agg(salary=("annual_salary_usd", "mean"), records=("annual_salary_usd", "size"))
                .reset_index()
                .sort_values("salary")
            )
            fig = px.bar(
                gg,
                y="country",
                x="salary",
                orientation="h",
                text="salary",
                hover_data=["records"],
                title=_src("page01_data_basic_clean.mean_salary_by_country_f4437b2"),
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == _src("page07_integrated.salary_by_job_category_d159d2e"):
            gg = (
                d.groupby("job_category")
                .agg(salary=("annual_salary_usd", "mean"), records=("annual_salary_usd", "size"))
                .reset_index()
                .sort_values("salary")
            )
            fig = px.bar(
                gg,
                y="job_category",
                x="salary",
                orientation="h",
                text="salary",
                hover_data=["records"],
                title=_src("page07_integrated.mean_salary_by_job_category_641bd86"),
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == _src("page07_integrated.demand_vs_salary_52a951a"):
            fig = px.scatter(
                d,
                x="demand_score",
                y="annual_salary_usd",
                color=d.cluster.astype(str),
                hover_data=["job_title", "job_category", "country"],
                title=_src("page07_integrated.demand_score_vs_salary_328cfeb"),
                labels={"color": _src("page03_segmentation.cluster_c137a7f")},
            )
        elif chart == _src("page07_integrated.experience_vs_salary_4a979a5"):
            fig = px.scatter(
                d,
                x="years_of_experience",
                y="annual_salary_usd",
                color=d.cluster.astype(str),
                hover_data=["job_title", "job_category", "country"],
                title=_src("page07_integrated.experience_vs_salary_f7454cd"),
                labels={"color": _src("page03_segmentation.cluster_c137a7f")},
            )
        else:
            gg = d.groupby(["cluster", "remote_work"]).size().rename("records").reset_index()
            fig = px.bar(
                gg,
                x="cluster",
                y="records",
                color="remote_work",
                barmode="stack",
                text="records",
                title=_src("page07_integrated.remote_work_mix_by_cluster_c3d4074"),
            )
        show_plot(st, fig, "p7_explorer")
        interpretation_card(
            st,
            _tr(
                "page07_integrated.explorer_view_value0_uses_value1_filtered_records_89880af",
                value0=f"{chart}",
                value1=f"{len(d):,}",
                value2=f"{d.cluster.nunique()}",
            ),
            _tr("page07_integrated.the_interactive_explorer_is_designed_to_test_100cc5e"),
            _tr("page07_integrated.if_a_conclusion_only_appears_in_one_9528a40"),
            "info",
        )

    with tabs[5]:
        downloadable_table(
            st, seg, _tr("page07_integrated.segment_salary_summary_686d225"), "p7_seg_table"
        )
        downloadable_table(
            st, pred, _tr("page07_integrated.predicted_salary_by_segment_caeafc0"), "p7_pred_table"
        )
        downloadable_table(
            st,
            geo,
            _tr("page07_integrated.city_country_segment_summary_9c2923e"),
            "p7_geo_table",
            height=450,
        )
