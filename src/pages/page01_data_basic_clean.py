from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ai_job_market.core import dynamic_insights
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


def build_clean_comparison_frame(audit: dict) -> pd.DataFrame:
    """Build an Arrow-safe comparison table with consistently textual values."""
    return pd.DataFrame(
        {
            _src("page01_data_basic_clean.aspect_0d608d9"): [
                _src("page01_data_basic_clean.records_47a84e9"),
                _src("source.columns"),
                _src("page01_data_basic_clean.missing_cells_a4b3af3"),
                _src("page01_data_basic_clean.duplicate_rows_d61929e"),
                _src("page01_data_basic_clean.unique_identifier_d9d04db"),
            ],
            _src("page01_data_basic_clean.raw_848123d"): [
                f"{int(audit['raw_rows']):,}",
                f"{int(audit['raw_columns']):,}",
                "0",
                f"{int(audit['duplicate_rows_removed']):,}",
                _src("page01_data_basic_clean.job_id_present_993f759"),
            ],
            _src("page01_data_basic_clean.basic_clean_c25fc5a"): [
                f"{int(audit['clean_rows']):,}",
                f"{int(audit['clean_columns']):,}",
                "0",
                "0",
                _src("page01_data_basic_clean.job_id_removed_b6c3205"),
            ],
        },
        dtype="string",
    )


def _filters(st, df):
    with st.expander(_tr("page01_data_basic_clean.interactive_filters_cdf723e"), expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        cats = c1.multiselect(
            _tr("page01_data_basic_clean.job_category_0175eef"),
            sorted(df.job_category.dropna().astype(str).unique()),
            key="p1_cat",
            format_func=_option_label(),
        )
        levels = c2.multiselect(
            _tr("page01_data_basic_clean.experience_level_71293da"),
            sorted(df.experience_level.dropna().astype(str).unique()),
            key="p1_exp",
            format_func=_option_label(),
        )
        countries = c3.multiselect(
            _tr("page01_data_basic_clean.country_701d021"),
            sorted(df.country.dropna().astype(str).unique()),
            key="p1_country",
            format_func=_option_label(),
        )
        remote = c4.multiselect(
            _tr("page01_data_basic_clean.remote_work_f3cb058"),
            sorted(df.remote_work.dropna().astype(str).unique()),
            key="p1_remote",
            format_func=_option_label(),
        )
    return apply_filters(
        df,
        {
            "job_category": cats,
            "experience_level": levels,
            "country": countries,
            "remote_work": remote,
        },
    )


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("navigation.page01"))
    st.caption(_tr("page01_data_basic_clean.common_foundation_stage_1_project_scope_raw_6818cfd"))
    audit = read_json(root, "01_data_basic_clean/basic_clean_audit.json")
    target = read_json(root, "01_data_basic_clean/target_summary.json")
    clean = read_csv(root, "01_data_basic_clean/basic_clean.csv")
    findings = read_csv(root, "01_data_basic_clean/contradiction_summary.csv")
    flags = (
        read_csv(root, "01_data_basic_clean/integrity_row_flags.csv")
        if (root / "outputs/01_data_basic_clean/integrity_row_flags.csv").exists()
        else pd.DataFrame()
    )
    metric_cols(
        st,
        [
            (_tr("page01_data_basic_clean.raw_records_21d5c9b"), f"{audit['raw_rows']:,}"),
            (_tr("page01_data_basic_clean.raw_columns_d1f17e6"), audit["raw_columns"]),
            (_tr("page01_data_basic_clean.clean_records_a8ec39e"), f"{audit['clean_rows']:,}"),
            (_tr("page01_data_basic_clean.clean_columns_63b5407"), audit["clean_columns"]),
            (
                _tr("page01_data_basic_clean.removed_corrupted_rows_01bc04e"),
                audit["invalid_category_rows_removed"],
            ),
        ],
    )

    filtered = _filters(st, clean)
    if filtered.empty:
        st.warning(
            _tr("page01_data_basic_clean.no_records_match_the_current_filters_clear_8c360a9")
        )
        return

    tabs = st.tabs(
        [
            _tr("page01_data_basic_clean.stage_1_scope_profiling_035de97"),
            _tr("page01_data_basic_clean.stage_2_basic_clean_c46d3cf"),
            _tr("page01_data_basic_clean.stage_3_contradictions_c5b98b0"),
            _tr("page01_data_basic_clean.interactive_chart_explorer_236cb24"),
            _tr("page01_data_basic_clean.evidence_tables_21295b1"),
        ]
    )
    with tabs[0]:
        st.markdown(_tr("page01_data_basic_clean.stage_1_project_scope_raw_data_ingestion_e928a08"))
        st.markdown(
            _tr("page01_data_basic_clean.div_class_stage_note_target_b_annual_586c820"),
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1.7, 1])
        with c1:
            fig = px.histogram(
                filtered,
                x="annual_salary_usd",
                nbins=24,
                marginal="box",
                title=_src("page01_data_basic_clean.observed_annual_salary_distribution_89a8ea7"),
                labels={"annual_salary_usd": _src("model_evidence.annual_salary_usd_46de225")},
            )
            fig.add_vline(
                x=float(filtered.annual_salary_usd.mean()),
                line_dash="dash",
                annotation_text=_tr(
                    "page01_data_basic_clean.mean_value0_8f3f242",
                    value0=f"{money(filtered.annual_salary_usd.mean())}",
                ),
            )
            fig.add_vline(
                x=float(filtered.annual_salary_usd.median()),
                line_dash="dot",
                annotation_text=_tr(
                    "page01_data_basic_clean.median_value0_c3edcf2",
                    value0=f"{money(filtered.annual_salary_usd.median())}",
                ),
            )
            show_plot(st, fig, "p1_target_hist")
        with c2:
            q = filtered.annual_salary_usd.quantile([0.25, 0.5, 0.75])
            st.dataframe(
                display_frame(
                    pd.DataFrame(
                        {
                            _src("page01_data_basic_clean.metric_2d275a7"): [
                                _src("source.rows"),
                                _src("page01_data_basic_clean.min_dea7933"),
                                "Q1",
                                _src("page01_data_basic_clean.median_daab7c4"),
                                _src("page01_data_basic_clean.mean_727a7d1"),
                                "Q3",
                                _src("page01_data_basic_clean.max_a1a5936"),
                                _src("page01_data_basic_clean.std_5684e60"),
                            ],
                            _src("page01_data_basic_clean.value_8e37953"): [
                                f"{len(filtered):,}",
                                money(filtered.annual_salary_usd.min()),
                                money(q.loc[0.25]),
                                money(q.loc[0.5]),
                                money(filtered.annual_salary_usd.mean()),
                                money(q.loc[0.75]),
                                money(filtered.annual_salary_usd.max()),
                                money(filtered.annual_salary_usd.std()),
                            ],
                        }
                    )
                ),
                hide_index=True,
                width="stretch",
            )
        skew_note = (
            "right-skewed"
            if filtered.annual_salary_usd.mean() > filtered.annual_salary_usd.median()
            else _src("page01_data_basic_clean.left_skewed_or_symmetric_c3b9286")
        )
        interpretation_card(
            st,
            _tr(
                "page01_data_basic_clean.filtered_salary_mean_is_value0_versus_median_4212e83",
                value0=f"{money(filtered.annual_salary_usd.mean())}",
                value1=f"{money(filtered.annual_salary_usd.median())}",
                value2=f"{money(filtered.annual_salary_usd.min())}",
                value3=f"{money(filtered.annual_salary_usd.max())}",
            ),
            _tr(
                "page01_data_basic_clean.the_target_is_value0_extreme_salaries_are_69fec5f",
                value0=f"{skew_note}",
            ),
            _tr("page01_data_basic_clean.retain_target_extremes_for_now_handle_any_3e696b4"),
            "info",
        )
        with st.expander(_tr("page01_data_basic_clean.raw_schema_profiling_evidence_18947ab")):
            st.dataframe(
                display_frame(read_csv(root, "01_data_basic_clean/raw_profile.csv")),
                width="stretch",
                hide_index=True,
            )
            if (root / "outputs/01_data_basic_clean/numeric_descriptive_summary.csv").exists():
                st.dataframe(
                    display_frame(
                        read_csv(root, "01_data_basic_clean/numeric_descriptive_summary.csv")
                    ),
                    width="stretch",
                    hide_index=True,
                )

    with tabs[1]:
        st.markdown(_tr("page01_data_basic_clean.stage_2_basic_clean_1beea6e"))
        st.success(
            _tr(
                "page01_data_basic_clean.basic_clean_removed_value0_corrupted_category_row_68fe398",
                value0=f"{audit['invalid_category_rows_removed']}",
                value1=f"{audit['clean_rows']:,}",
                value2=f"{audit['clean_columns']}",
            )
        )
        compare = build_clean_comparison_frame(audit)
        st.dataframe(display_frame(compare), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            prof = (
                read_csv(root, "01_data_basic_clean/raw_profile.csv")
                .sort_values("unique", ascending=False)
                .head(15)
            )
            fig = px.bar(
                prof.sort_values("unique"),
                y="column",
                x="unique",
                orientation="h",
                text="unique",
                title=_src("page01_data_basic_clean.top_feature_cardinalities_raw_data_23ba07d"),
            )
            fig.update_traces(textposition="outside")
            show_plot(st, fig, "p1_cardinality")
        with c2:
            # all missing counts are visible, including zeros
            prof = read_csv(root, "01_data_basic_clean/raw_profile.csv")
            miss = prof.assign(total_missing=prof.missing + prof.hidden_missing).sort_values(
                "total_missing", ascending=False
            )
            fig = px.bar(
                miss,
                y="column",
                x="total_missing",
                orientation="h",
                text="total_missing",
                title=_src(
                    "page01_data_basic_clean.missing_hidden_missing_audit_by_feature_b6cfb98"
                ),
            )
            show_plot(st, fig, "p1_missing")
        total_missing = int(
            (
                read_csv(root, "01_data_basic_clean/raw_profile.csv").missing
                + read_csv(root, "01_data_basic_clean/raw_profile.csv").hidden_missing
            ).sum()
        )
        interpretation_card(
            st,
            _tr(
                "page01_data_basic_clean.basic_clean_changed_value0_row_s_and_8d02053",
                value0=f"{audit['raw_rows'] - audit['clean_rows']}",
                value1=f"{audit['raw_columns'] - audit['clean_columns']}",
                value2=f"{total_missing}",
            ),
            _tr("page01_data_basic_clean.structural_cleanliness_alone_is_not_enough_the_ae359cd"),
            _tr("page01_data_basic_clean.proceed_to_stage_3_contradiction_checks_before_f7029f9"),
            "success" if total_missing == 0 else "warning",
        )

    with tabs[2]:
        st.markdown(
            _tr("page01_data_basic_clean.stage_3_contradictory_feature_investigation_ad5ef03")
        )
        fig = px.bar(
            findings.sort_values("affected_pct"),
            y="issue",
            x="affected_pct",
            orientation="h",
            text="affected_pct",
            title=_src(
                "page01_data_basic_clean.data_logic_integrity_issues_ranked_by_severity_a23b669"
            ),
            labels={
                "affected_pct": _src("page01_data_basic_clean.affected_records_37ec2ed"),
                "issue": _src("page01_data_basic_clean.issue_48dc76d"),
            },
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        show_plot(st, fig, "p1_issue_rank")
        insight_box(st, dynamic_insights(root / "outputs").get("data_quality", ""), "warning")
        st.dataframe(
            display_frame(findings.style.format({"affected_pct": "{:.2f}%"})),
            width="stretch",
            hide_index=True,
        )
        top_issue = findings.sort_values("affected_pct", ascending=False).iloc[0]
        interpretation_card(
            st,
            _tr(
                "page01_data_basic_clean.largest_logic_finding_value0_affects_value1_value2_3f6beb4",
                value0=f"{top_issue.issue}",
                value1=f"{top_issue.affected_pct:.1f}",
                value2=f"{int(top_issue.affected_rows):,}",
            ),
            _tr("page01_data_basic_clean.this_is_a_semantic_integrity_problem_not_3476a45"),
            str(top_issue.required_action),
            "warning",
        )

        s1, s2, s3 = st.tabs(
            [
                _tr("page01_data_basic_clean.experience_contradiction_0ddfef5"),
                _tr("page01_data_basic_clean.salary_range_inconsistency_b4cce38"),
                _tr("page01_data_basic_clean.salary_tier_inconsistency_5cc9000"),
            ]
        )
        with s1:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.box(
                    filtered,
                    x="experience_level",
                    y="years_of_experience",
                    points="outliers",
                    title=_src(
                        "page01_data_basic_clean.years_of_experience_by_experience_level_104d930"
                    ),
                )
                show_plot(st, fig, "p1_exp_box")
            with c2:
                g = (
                    filtered.groupby("experience_level", observed=True)
                    .agg(
                        mean_salary=("annual_salary_usd", "mean"),
                        median_salary=("annual_salary_usd", "median"),
                        records=("annual_salary_usd", "size"),
                    )
                    .reset_index()
                )
                m = g.melt(
                    id_vars=["experience_level", "records"],
                    value_vars=["mean_salary", "median_salary"],
                    var_name="metric",
                    value_name="salary",
                )
                fig = px.bar(
                    m,
                    x="experience_level",
                    y="salary",
                    color="metric",
                    barmode="group",
                    text="salary",
                    title=_src("page01_data_basic_clean.annual_salary_by_experience_level_15de358"),
                )
                fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
                show_plot(st, fig, "p1_exp_salary")
            byy = (
                filtered.groupby("years_of_experience", observed=True)
                .agg(
                    mean_salary=("annual_salary_usd", "mean"),
                    median_salary=("annual_salary_usd", "median"),
                    records=("annual_salary_usd", "size"),
                )
                .reset_index()
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=byy.years_of_experience,
                    y=byy.mean_salary,
                    mode="lines+markers+text",
                    text=[f"${x:,.0f}" for x in byy.mean_salary],
                    textposition=_src("model_evidence.top_center_24b3167"),
                    name=_src("page01_data_basic_clean.mean_727a7d1"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=byy.years_of_experience,
                    y=byy.median_salary,
                    mode="lines+markers",
                    name=_src("page01_data_basic_clean.median_daab7c4"),
                )
            )
            fig.update_layout(
                title=_src(
                    "page01_data_basic_clean.annual_salary_by_raw_years_of_experience_007bcc3"
                ),
                xaxis_title=_src("page01_data_basic_clean.years_of_experience_29eb74e"),
                yaxis_title=_src("page01_data_basic_clean.salary_usd_ac3be8f"),
            )
            show_plot(st, fig, "p1_year_salary")
            corr = (
                float(filtered[["years_of_experience", "annual_salary_usd"]].corr().iloc[0, 1])
                if len(filtered) > 2
                else float("nan")
            )
            spread = float(g.mean_salary.max() - g.mean_salary.min()) if len(g) > 1 else 0.0
            interpretation_card(
                st,
                _tr(
                    "page01_data_basic_clean.experience_level_mean_salary_spread_is_value0_9e0873e",
                    value0=f"{money(spread)}",
                    value1=f"{corr:+.2f}",
                ),
                _tr("page01_data_basic_clean.if_bucket_labels_and_numeric_years_tell_0d26731"),
                _tr(
                    "page01_data_basic_clean.keep_the_contradiction_visible_and_let_ablation_02b1225"
                ),
                "warning",
            )
        with s2:
            if not flags.empty:
                d = flags.copy()
                d["range_status"] = np.where(
                    d.salary_range_mismatch.astype(bool),
                    _src("page01_data_basic_clean.outside_stated_range_df8627f"),
                    _src("page01_data_basic_clean.inside_stated_range_d15e8bb"),
                )
                c1, c2 = st.columns(2)
                with c1:
                    g = (
                        d.range_status.value_counts()
                        .rename_axis("status")
                        .reset_index(name="records")
                    )
                    fig = px.bar(
                        g,
                        x="status",
                        y="records",
                        text="records",
                        title=_src("page01_data_basic_clean.salary_range_audit_status_84620f5"),
                    )
                    fig.update_traces(textposition="outside")
                    show_plot(st, fig, "p1_range_status")
                with c2:
                    fig = px.histogram(
                        d,
                        x="annual_salary_usd",
                        color="range_status",
                        nbins=25,
                        barmode="overlay",
                        title=_src(
                            "page01_data_basic_clean.actual_salary_distribution_by_range_status_ef99554"
                        ),
                    )
                    show_plot(st, fig, "p1_range_hist")
                sample = d.sort_values("annual_salary_usd").copy().reset_index(drop=True)
                sample["row"] = np.arange(len(sample))
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=sample.row,
                        y=sample.salary_min_usd,
                        mode="lines",
                        name=_src("page01_data_basic_clean.stated_min_61f4d6f"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=sample.row,
                        y=sample.salary_max_usd,
                        mode="lines",
                        name=_src("page01_data_basic_clean.stated_max_1f99b3d"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=sample.row,
                        y=sample.annual_salary_usd,
                        mode="markers",
                        name=_src("page01_data_basic_clean.actual_salary_12aadac"),
                        marker=dict(size=5),
                    )
                )
                fig.update_layout(
                    title=_src(
                        "page01_data_basic_clean.actual_salary_vs_stated_salary_bounds_09ae88b"
                    ),
                    xaxis_title=_src(
                        "page01_data_basic_clean.records_sorted_by_actual_salary_af6504d"
                    ),
                    yaxis_title="USD",
                )
                show_plot(st, fig, "p1_range_detail")
                rate = 100 * d.salary_range_mismatch.astype(bool).mean()
                interpretation_card(
                    st,
                    _tr(
                        "page01_data_basic_clean.value0_of_filtered_records_fall_outside_the_30bf93a",
                        value0=f"{rate:.1f}",
                    ),
                    _tr(
                        "page01_data_basic_clean.the_range_fields_are_target_adjacent_metadata_0541b1c"
                    ),
                    _tr("page01_data_basic_clean.block_salary_min_usd_and_salary_max_22635e5"),
                    "warning",
                )
        with s3:
            if not flags.empty:
                d = flags.copy()
                d["tier_status"] = np.where(
                    d.salary_tier_mismatch.astype(bool),
                    _src("page01_data_basic_clean.mismatch_0dae606"),
                    _src("page01_data_basic_clean.match_03c0e80"),
                )
                c1, c2 = st.columns(2)
                with c1:
                    g = (
                        d.tier_status.value_counts()
                        .rename_axis("status")
                        .reset_index(name="records")
                    )
                    fig = px.bar(
                        g,
                        x="status",
                        y="records",
                        text="records",
                        title=_src(
                            "page01_data_basic_clean.salary_tier_consistency_status_076072d"
                        ),
                    )
                    show_plot(st, fig, "p1_tier_status")
                with c2:
                    ct = pd.crosstab(d.salary_tier, d.expected_salary_tier)
                    fig = px.imshow(
                        ct,
                        text_auto=True,
                        aspect="auto",
                        title=_src(
                            "page01_data_basic_clean.stated_tier_vs_expected_tier_from_annual_143d019"
                        ),
                        labels=dict(
                            x=_src("page01_data_basic_clean.expected_tier_296ea85"),
                            y=_src("page01_data_basic_clean.stated_tier_706e84c"),
                            color=_src("page01_data_basic_clean.records_47a84e9"),
                        ),
                    )
                    show_plot(st, fig, "p1_tier_heat")
                fig = px.box(
                    d,
                    x="salary_tier",
                    y="annual_salary_usd",
                    color="tier_status",
                    points=False,
                    title=_src(
                        "page01_data_basic_clean.observed_salary_distribution_by_stated_tier_7dcbdb1"
                    ),
                )
                show_plot(st, fig, "p1_tier_box")
                rate = 100 * d.salary_tier_mismatch.astype(bool).mean()
                interpretation_card(
                    st,
                    _tr(
                        "page01_data_basic_clean.value0_of_filtered_records_have_a_stated_6b12b29",
                        value0=f"{rate:.1f}",
                    ),
                    _tr(
                        "page01_data_basic_clean.salary_tier_behaves_like_a_noisy_derivative_36f6598"
                    ),
                    _tr("page01_data_basic_clean.exclude_salary_tier_from_model_x_and_e6494e7"),
                    "warning",
                )

    with tabs[3]:
        st.markdown(_tr("page01_data_basic_clean.interactive_chart_explorer_320e7ef"))
        chart = chart_selector(
            st,
            _tr("page01_data_basic_clean.choose_chart_6e73d7f"),
            [
                _tr("page01_data_basic_clean.salary_by_job_domain_4ea91e3"),
                _tr("page01_data_basic_clean.salary_by_country_8072edc"),
                _tr("page01_data_basic_clean.salary_by_city_4003022"),
                _tr("page01_data_basic_clean.salary_by_years_of_experience_2cc0af0"),
                _tr("page01_data_basic_clean.demand_score_vs_salary_e86b78b"),
                _tr("page01_data_basic_clean.benefits_score_vs_salary_d55fd25"),
            ],
            "p1_chart_select",
        )
        focus = st.slider(
            _tr("page01_data_basic_clean.top_categories_to_display_546f7ef"),
            5,
            30,
            12,
            key="p1_topn",
        )
        if chart == _src("page01_data_basic_clean.salary_by_job_domain_4ea91e3"):
            g = (
                filtered.groupby("job_category")
                .agg(
                    mean_salary=("annual_salary_usd", "mean"), records=("annual_salary_usd", "size")
                )
                .reset_index()
                .nlargest(focus, "mean_salary")
            )
            fig = px.bar(
                g.sort_values("mean_salary"),
                y="job_category",
                x="mean_salary",
                orientation="h",
                text="mean_salary",
                hover_data=["records"],
                title=_src("page01_data_basic_clean.mean_salary_by_job_domain_51b02b0"),
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == _src("page01_data_basic_clean.salary_by_country_8072edc"):
            g = (
                filtered.groupby("country")
                .agg(
                    mean_salary=("annual_salary_usd", "mean"), records=("annual_salary_usd", "size")
                )
                .reset_index()
                .nlargest(focus, "mean_salary")
            )
            fig = px.bar(
                g.sort_values("mean_salary"),
                y="country",
                x="mean_salary",
                orientation="h",
                text="mean_salary",
                hover_data=["records"],
                title=_src("page01_data_basic_clean.mean_salary_by_country_f4437b2"),
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == _src("page01_data_basic_clean.salary_by_city_4003022"):
            g = (
                filtered.groupby(["city", "country"])
                .agg(
                    mean_salary=("annual_salary_usd", "mean"), records=("annual_salary_usd", "size")
                )
                .reset_index()
                .nlargest(focus, "mean_salary")
            )
            fig = px.bar(
                g.sort_values("mean_salary"),
                y="city",
                x="mean_salary",
                orientation="h",
                color="country",
                text="mean_salary",
                hover_data=["records"],
                title=_src("page01_data_basic_clean.mean_salary_by_city_c5c99e0"),
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        elif chart == _src("page01_data_basic_clean.salary_by_years_of_experience_2cc0af0"):
            g = (
                filtered.groupby("years_of_experience")
                .agg(
                    mean_salary=("annual_salary_usd", "mean"),
                    median_salary=("annual_salary_usd", "median"),
                    records=("annual_salary_usd", "size"),
                )
                .reset_index()
            )
            fig = px.line(
                g,
                x="years_of_experience",
                y=["mean_salary", "median_salary"],
                markers=True,
                title=_src("page01_data_basic_clean.salary_by_years_of_experience_11ba6f4"),
                labels={
                    "value": _src("page01_data_basic_clean.salary_usd_ac3be8f"),
                    "variable": _src("page01_data_basic_clean.metric_2d275a7"),
                },
            )
        elif chart == _src("page01_data_basic_clean.demand_score_vs_salary_e86b78b"):
            fig = px.scatter(
                filtered,
                x="demand_score",
                y="annual_salary_usd",
                color="job_category",
                hover_data=["job_title", "country", "years_of_experience"],
                title=_src("page01_data_basic_clean.demand_score_vs_annual_salary_bba119d"),
                trendline=None,
            )
        else:
            fig = px.scatter(
                filtered,
                x="benefits_score_10",
                y="annual_salary_usd",
                color="job_category",
                hover_data=["job_title", "country", "years_of_experience"],
                title=_src("page01_data_basic_clean.benefits_score_vs_annual_salary_b61e990"),
            )
        show_plot(st, fig, "p1_explorer")
        st.caption(
            _tr(
                "page01_data_basic_clean.current_chart_uses_value0_filtered_records_hover_850ca50",
                value0=f"{len(filtered):,}",
            )
        )
        if chart in {
            _src("page01_data_basic_clean.salary_by_job_domain_4ea91e3"),
            _src("page01_data_basic_clean.salary_by_country_8072edc"),
            _src("page01_data_basic_clean.salary_by_city_4003022"),
        }:
            top = g.sort_values("mean_salary", ascending=False).iloc[0]
            label_col = (
                "job_category"
                if "job_category" in g.columns
                else (
                    "country"
                    if chart == _src("page01_data_basic_clean.salary_by_country_8072edc")
                    else "city"
                )
            )
            interpretation_card(
                st,
                _tr(
                    "page01_data_basic_clean.highest_mean_salary_in_the_current_chart_b978c25",
                    value0=f"{top[label_col]}",
                    value1=f"{money(top.mean_salary)}",
                    value2=f"{int(top.records):,}",
                ),
                _tr(
                    "page01_data_basic_clean.this_is_a_descriptive_filtered_comparison_unequal_1a5e7fe"
                ),
                _tr("page01_data_basic_clean.use_hover_counts_and_filters_to_check_1c10e1d"),
                "info",
            )
        elif chart == _src("page01_data_basic_clean.salary_by_years_of_experience_2cc0af0"):
            corr = (
                float(filtered[["years_of_experience", "annual_salary_usd"]].corr().iloc[0, 1])
                if len(filtered) > 2
                else float("nan")
            )
            interpretation_card(
                st,
                _tr(
                    "page01_data_basic_clean.current_filter_pearson_correlation_between_years_of_9411d72",
                    value0=f"{corr:+.2f}",
                ),
                _tr("page01_data_basic_clean.the_direction_is_diagnostic_only_and_may_9426c55"),
                _tr("page01_data_basic_clean.compare_this_chart_with_stage_5_ablation_25f07aa"),
                "warning",
            )
        else:
            xcol = (
                "demand_score"
                if chart == _src("page01_data_basic_clean.demand_score_vs_salary_e86b78b")
                else "benefits_score_10"
            )
            corr = (
                float(filtered[[xcol, "annual_salary_usd"]].corr().iloc[0, 1])
                if len(filtered) > 2
                else float("nan")
            )
            interpretation_card(
                st,
                _tr(
                    "page01_data_basic_clean.current_filter_correlation_between_value0_and_salary_98f878a",
                    value0=f"{xcol}",
                    value1=f"{corr:+.2f}",
                ),
                _tr(
                    "page01_data_basic_clean.scatter_structure_shows_association_strength_and_heterogeneity_2b6f1c1"
                ),
                _tr("page01_data_basic_clean.use_this_as_exploratory_evidence_only_model_51d38c1"),
                "info",
            )

    with tabs[4]:
        downloadable_table(
            st,
            findings,
            _tr("page01_data_basic_clean.contradiction_summary_ac2f85b"),
            "p1_findings",
        )
        if not flags.empty:
            downloadable_table(
                st,
                flags,
                _tr("page01_data_basic_clean.row_level_integrity_flags_52f68cf"),
                "p1_flags",
                "integrity_row_flags.csv",
                height=420,
            )
        if (root / "outputs/01_data_basic_clean/categorical_frequency_summary.csv").exists():
            downloadable_table(
                st,
                read_csv(root, "01_data_basic_clean/categorical_frequency_summary.csv"),
                _tr("page01_data_basic_clean.categorical_frequency_summary_d0ad9d4"),
                "p1_catfreq",
                height=420,
            )
