from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .common import *


def render(st, root, role="admin"):
    style_page(st)
    st.title("7. Integrated Market Insight")
    st.caption(
        "Decision Layer — combine unsupervised structural segments with supervised salary evidence after both branches finish"
    )
    seg = read_csv(root, "07_integrated_insight/segment_salary_summary.csv")
    pred = read_csv(root, "07_integrated_insight/predicted_salary_by_segment.csv")
    geo = read_csv(root, "07_integrated_insight/segment_city_country_summary.csv")
    assign = read_csv(root, "03_ai_job_market_segmentation/cluster_assignments.csv")

    cluster_opts = sorted(assign.cluster.astype(int).unique().tolist())
    with st.expander("Integrated insight filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        clusters = c1.multiselect("Clusters", cluster_opts, default=cluster_opts, key="p7_clusters")
        countries = c2.multiselect(
            "Countries", sorted(assign.country.dropna().astype(str).unique()), key="p7_country"
        )
        cats = c3.multiselect(
            "Job categories",
            sorted(assign.job_category.dropna().astype(str).unique()),
            key="p7_cat",
        )
        metric = c4.selectbox("Salary metric", ["Mean", "Median"], key="p7_metric")
    d = apply_filters(
        assign, {"cluster": [str(x) for x in clusters], "country": countries, "job_category": cats}
    )
    if d.empty:
        st.warning("No records match the selected filters.")
        return

    actual_col = "mean" if metric == "Mean" else "median"
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
            ("Segments in view", len(g)),
            ("Records", f"{len(d):,}"),
            ("Highest salary segment", f"Cluster {int(top.cluster)}"),
            (f"Highest {metric.lower()} salary", money(top[f"salary_{actual_col}"])),
            ("Mean demand", f"{d.demand_score.mean():.1f}"),
        ],
    )

    tabs = st.tabs(
        [
            "Segment Salary",
            "Prediction Alignment",
            "Geography",
            "Job-market Structure",
            "Interactive Chart Explorer",
            "Evidence Tables",
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
            title="Observed Salary by Structural Segment",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        show_plot(st, fig, "p7_seg_salary")
        fig = px.box(
            d,
            x=d.cluster.astype(str),
            y="annual_salary_usd",
            color=d.cluster.astype(str),
            points="outliers",
            title="Salary Distribution by Segment",
            labels={"x": "Cluster", "color": "Cluster"},
        )
        show_plot(st, fig, "p7_seg_box")
        spread = float(g.salary_mean.max() - g.salary_mean.min())
        interpretation_card(
            st,
            f"Observed mean-salary spread across visible structural segments is {money(spread)}; highest is Cluster {int(g.sort_values('salary_mean', ascending=False).iloc[0].cluster)}.",
            "Segmentation did not use salary, so this is post-hoc alignment evidence rather than a salary-driven cluster definition.",
            "Use the spread only as descriptive market context and verify whether the same segment differences persist on future data.",
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
            title="Locked-test Actual vs Predicted Mean Salary by Segment",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        show_plot(st, fig, "p7_actual_pred")
        fig = px.bar(
            pp.sort_values("MAE"),
            x="cluster",
            y="MAE",
            text="MAE",
            title="Locked-test MAE by Segment",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        show_plot(st, fig, "p7_seg_mae")
        st.caption(
            "Segment labels are not fed into the salary model. Integration happens only after unsupervised and supervised branches have completed independently."
        )
        if not pp.empty:
            worst = pp.sort_values("MAE", ascending=False).iloc[0]
            bestp = pp.sort_values("MAE").iloc[0]
            interpretation_card(
                st,
                f"Locked-test segment MAE ranges from {money(bestp.MAE)} (C{int(bestp.cluster)}) to {money(worst.MAE)} (C{int(worst.cluster)}).",
                "Prediction error varies across structural segments even though cluster labels are not model inputs; the pattern can reveal heterogeneity that global metrics hide.",
                "Monitor segment-level error as a diagnostic, but do not feed cluster labels back into the salary model unless an explicit ablation proves stable improvement.",
                "warning",
            )

    with tabs[2]:
        ge = (
            d.groupby(["cluster", "country"])
            .agg(records=("annual_salary_usd", "size"), salary_mean=("annual_salary_usd", "mean"))
            .reset_index()
        )
        ge["cluster_label"] = "Cluster " + ge.cluster.astype(int).astype(str)
        fig = px.scatter_geo(
            ge,
            locations="country",
            locationmode="country names",
            size="records",
            color="cluster_label",
            hover_name="country",
            hover_data={"salary_mean": ":,.0f", "records": True},
            projection="natural earth",
            title="Country Footprint by Segment",
        )
        show_plot(st, fig, "p7_country_map")
        city = (
            d.groupby(["cluster", "country", "city"])
            .agg(records=("annual_salary_usd", "size"), salary_mean=("annual_salary_usd", "mean"))
            .reset_index()
        )
        topn = st.slider("Top cities by record volume", 10, 50, 25, key="p7_city_topn")
        city = city.nlargest(topn, "records")
        fig = px.scatter(
            city,
            x="records",
            y="salary_mean",
            size="records",
            color=city.cluster.astype(str),
            text="city",
            hover_data=["country"],
            title="City Volume vs Mean Salary",
            labels={"color": "Cluster", "salary_mean": "Mean salary (USD)"},
        )
        fig.update_traces(textposition="top center")
        show_plot(st, fig, "p7_city")
        if len(city):
            topcity = city.sort_values("records", ascending=False).iloc[0]
            interpretation_card(
                st,
                f"Largest city footprint in the current filters is {topcity.city}, {topcity.country} with {int(topcity.records)} records and mean salary {money(topcity.salary_mean)}.",
                "High-volume locations can dominate aggregate segment profiles, so geography should be checked before generalizing segment behavior.",
                "Compare country/city mix with job-domain composition to distinguish geographic concentration from genuine structural segmentation.",
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
                title="Job Domain Mix by Segment",
                labels={"color": "Cluster"},
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
                title="Demand / Benefits / Experience Profile",
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            show_plot(st, fig, "p7_market_profile")
        if len(demand):
            hi = demand.sort_values("demand_mean", ascending=False).iloc[0]
            interpretation_card(
                st,
                f"Cluster {int(hi.cluster)} has the highest mean demand score ({hi.demand_mean:.1f}) in the current view.",
                "Integrated profiles combine independent Branch-A structure with observed market descriptors; these are segment characteristics, not treatment effects.",
                "Use them to create descriptive segment labels only after checking skills, company and geography views for consistency.",
                "info",
            )

    with tabs[4]:
        st.markdown("### Interactive Chart Explorer")
        chart = chart_selector(
            st,
            "Choose market view",
            [
                "Salary by cluster",
                "Salary by country",
                "Salary by job category",
                "Demand vs salary",
                "Experience vs salary",
                "Cluster composition by remote work",
            ],
            "p7_chart",
        )
        if chart == "Salary by cluster":
            gg = d.groupby("cluster").annual_salary_usd.mean().reset_index()
            fig = px.bar(
                gg,
                x="cluster",
                y="annual_salary_usd",
                text="annual_salary_usd",
                title="Mean Salary by Cluster",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == "Salary by country":
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
                title="Mean Salary by Country",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == "Salary by job category":
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
                title="Mean Salary by Job Category",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == "Demand vs salary":
            fig = px.scatter(
                d,
                x="demand_score",
                y="annual_salary_usd",
                color=d.cluster.astype(str),
                hover_data=["job_title", "job_category", "country"],
                title="Demand Score vs Salary",
                labels={"color": "Cluster"},
            )
        elif chart == "Experience vs salary":
            fig = px.scatter(
                d,
                x="years_of_experience",
                y="annual_salary_usd",
                color=d.cluster.astype(str),
                hover_data=["job_title", "job_category", "country"],
                title="Experience vs Salary",
                labels={"color": "Cluster"},
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
                title="Remote-work Mix by Cluster",
            )
        show_plot(st, fig, "p7_explorer")
        interpretation_card(
            st,
            f"Explorer view '{chart}' uses {len(d):,} filtered records across {d.cluster.nunique()} cluster(s).",
            "The interactive explorer is designed to test whether a narrative is robust to alternative market views.",
            "If a conclusion only appears in one chart but disappears under related views or filters, report it as exploratory rather than established.",
            "info",
        )

    with tabs[5]:
        downloadable_table(st, seg, "Segment Salary Summary", "p7_seg_table")
        downloadable_table(st, pred, "Predicted Salary by Segment", "p7_pred_table")
        downloadable_table(st, geo, "City / Country Segment Summary", "p7_geo_table", height=450)
