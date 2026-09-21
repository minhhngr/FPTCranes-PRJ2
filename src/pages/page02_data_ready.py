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


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("page02_data_ready.2_data_ready_for_machine_learning_04ff341"))
    st.caption(
        _tr("page02_data_ready.common_foundation_stage_4_feature_importance_leakage_5ac64fa")
    )
    policy = read_csv(root, "02_data_ready_for_ml/feature_policy.csv")
    split = read_json(root, "02_data_ready_for_ml/temporal_split_summary.json")
    contract = read_json(root, "02_data_ready_for_ml/preprocessing_contract.json")
    corr = read_csv(root, "02_data_ready_for_ml/train_encoded_correlations.csv")
    abl = read_csv(root, "02_data_ready_for_ml/ablation_results.csv")
    monthly = (
        read_csv(root, "02_data_ready_for_ml/monthly_split_counts.csv")
        if (root / "outputs/02_data_ready_for_ml/monthly_split_counts.csv").exists()
        else pd.DataFrame()
    )
    fmap = (
        read_csv(root, "02_data_ready_for_ml/preprocessing_feature_map.csv")
        if (root / "outputs/02_data_ready_for_ml/preprocessing_feature_map.csv").exists()
        else pd.DataFrame()
    )
    scaling = (
        read_csv(root, "02_data_ready_for_ml/numeric_scaling_summary.csv")
        if (root / "outputs/02_data_ready_for_ml/numeric_scaling_summary.csv").exists()
        else pd.DataFrame()
    )
    skills = (
        read_csv(root, "02_data_ready_for_ml/skill_token_summary.csv")
        if (root / "outputs/02_data_ready_for_ml/skill_token_summary.csv").exists()
        else pd.DataFrame()
    )
    skill_corr = (
        read_csv(root, "02_data_ready_for_ml/skill_target_correlations.csv")
        if (root / "outputs/02_data_ready_for_ml/skill_target_correlations.csv").exists()
        else pd.DataFrame()
    )

    metric_cols(
        st,
        [
            (
                _tr("page02_data_ready.allowed_raw_inputs_ae0ff2e"),
                int((policy.policy == "ALLOW").sum()),
            ),
            (
                _tr("page02_data_ready.blocked_fields_e1e0811"),
                int((policy.policy == "BLOCK").sum()),
            ),
            (_tr("page02_data_ready.dev_rows_8eb0441"), f"{split['development_rows']:,}"),
            (_tr("page02_data_ready.locked_test_rows_31140b8"), f"{split['locked_test_rows']:,}"),
            (
                _tr("page02_data_ready.encoded_train_features_a2e26fe"),
                contract["encoded_feature_count"],
            ),
        ],
    )

    tabs = st.tabs(
        [
            _tr("page02_data_ready.stage_4_governance_correlation_d21376e"),
            _tr("page02_data_ready.stage_5_ablation_selection_57a130e"),
            _tr("page02_data_ready.b1_b3_temporal_split_preprocessing_0243618"),
            _tr("page02_data_ready.skills_analysis_4416f8c"),
            _tr("page01_data_basic_clean.interactive_chart_explorer_236cb24"),
            _tr("page01_data_basic_clean.evidence_tables_21295b1"),
        ]
    )
    with tabs[0]:
        st.markdown(_tr("page02_data_ready.stage_4_feature_importance_leakage_prevention_fa9748d"))
        c1, c2 = st.columns([1.1, 1.9])
        with c1:
            fig = px.pie(
                policy[policy.policy.isin(["ALLOW", "BLOCK"])],
                names="policy",
                title=_src("page02_data_ready.feature_governance_mix_49ae647"),
                hole=0.55,
            )
            show_plot(st, fig, "p2_policy_pie")
        with c2:
            st.dataframe(display_frame(policy), width="stretch", hide_index=True, height=360)
        insight_box(st, dynamic_insights(root / "outputs").get("correlation", ""))
        topn = st.slider(
            _tr("page02_data_ready.top_encoded_correlations_67d3c08"),
            10,
            60,
            30,
            key="p2_corr_topn",
        )
        direction = st.radio(
            _tr("page02_data_ready.correlation_view_d6de092"),
            [
                _src("page02_data_ready.absolute_strongest_b20ec5e"),
                _src("page02_data_ready.positive_only_6aedbe5"),
                _src("page02_data_ready.negative_only_6d620da"),
            ],
            horizontal=True,
            key="p2_corr_dir",
            format_func=_option_label(),
        )
        d = corr.copy()
        if direction == _src("page02_data_ready.positive_only_6aedbe5"):
            d = d[d.pearson_r >= 0].sort_values("pearson_r", ascending=False)
        elif direction == _src("page02_data_ready.negative_only_6d620da"):
            d = d[d.pearson_r < 0].sort_values("pearson_r")
        else:
            d = d.sort_values("abs_r", ascending=False)
        d = d.head(topn).sort_values("pearson_r")
        fig = px.bar(
            d,
            y="encoded_feature",
            x="pearson_r",
            orientation="h",
            text="pearson_r",
            title=_src(
                "page02_data_ready.train_only_encoded_feature_correlation_with_annual_9d37f29"
            ),
        )
        fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        fig.add_vline(x=0)
        show_plot(st, fig, "p2_corr")
        st.caption(_tr("page02_data_ready.correlation_is_computed_on_train_dev_only_a5baaab"))
        strongest = corr.sort_values("abs_r", ascending=False).iloc[0]
        blocked = int((policy.policy == "BLOCK").sum())
        allowed = int((policy.policy == "ALLOW").sum())
        interpretation_card(
            st,
            _tr(
                "page02_data_ready.governance_keeps_value0_raw_serving_inputs_and_ca1fdf4",
                value0=f"{allowed}",
                value1=f"{blocked}",
                value2=f"{strongest.encoded_feature}",
                value3=f"{strongest.pearson_r:+.2f}",
            ),
            _tr("page02_data_ready.the_leakage_gate_is_intentionally_stricter_than_6dd0ed7"),
            _tr("page02_data_ready.use_correlations_to_diagnose_signal_concentration_but_d486260"),
            "warning",
        )

    with tabs[1]:
        st.markdown(_tr("page02_data_ready.stage_5_feature_selection_ablation_84f48dd"))
        metric = st.selectbox(
            _tr("page02_data_ready.ablation_comparison_metric_7a5cadf"),
            ["MAE_mean", "RMSE_mean", "R2_mean"],
            key="p2_abl_metric",
            format_func=_option_label(),
        )
        ascending = metric != "R2_mean"
        d = abl.sort_values(metric, ascending=not ascending if metric == "R2_mean" else False)
        fig = px.bar(
            d.sort_values(metric, ascending=not ascending),
            y="option",
            x=metric,
            orientation="h",
            text=metric,
            hover_data=["feature_count", "features"],
            title=_tr("page02_data_ready.ablation_study_value0_9e46283", value0=f"{metric}"),
        )
        if metric in {"MAE_mean", "RMSE_mean"}:
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        else:
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        show_plot(st, fig, "p2_ablation")
        best = abl.sort_values("MAE_mean").iloc[0]
        st.info(
            _tr(
                "page02_data_ready.best_ablation_by_temporal_cv_mae_value0_608bdba",
                value0=f"{best['option']}",
                value1=f"{money(best['MAE_mean'])}",
            )
        )
        st.dataframe(
            display_frame(
                abl.style.format(
                    {
                        "MAE_mean": "${:,.0f}",
                        "MAE_std": "${:,.0f}",
                        "RMSE_mean": "${:,.0f}",
                        "R2_mean": "{:.3f}",
                    }
                )
            ),
            width="stretch",
            hide_index=True,
        )
        base = (
            abl.loc[
                abl.option.str.contains(
                    _src("page02_data_ready.conservative_75e1783"), case=False, na=False
                )
            ].iloc[0]
            if abl.option.str.contains(
                _src("page02_data_ready.conservative_75e1783"), case=False, na=False
            ).any()
            else abl.iloc[-1]
        )
        gain = (
            100 * (float(base.MAE_mean) - float(best.MAE_mean)) / float(base.MAE_mean)
            if float(base.MAE_mean)
            else 0.0
        )
        interpretation_card(
            st,
            _tr(
                "page02_data_ready.best_ablation_is_value0_with_mean_temporal_557ff52",
                value0=f"{best['option']}",
                value1=f"{money(best.MAE_mean)}",
                value2=f"{gain:.1f}",
            ),
            _tr(
                "page02_data_ready.ablation_isolates_incremental_predictive_contribution_it_does_243b55b"
            ),
            _tr("page02_data_ready.retain_only_feature_families_whose_out_of_e143968"),
            "success" if gain > 5 else "warning",
        )

    with tabs[2]:
        st.markdown(_tr("page02_data_ready.b1_b3_locked_temporal_test_train_only_a4065df"))
        st.info(
            _tr(
                "page02_data_ready.locked_period_value0_value1_rows_value2_the_c9b2c29",
                value0=f"{split['locked_test_period_label']}",
                value1=f"{split['locked_test_rows']:,}",
                value2=f"{split['locked_test_pct']:.1f}",
            )
        )
        if not monthly.empty:
            fig = px.bar(
                monthly,
                x="period",
                y="records",
                color="split",
                barmode="stack",
                text="records",
                title=_src("page02_data_ready.monthly_posting_volume_locked_temporal_test_1ba77ca"),
            )
            fig.update_traces(textposition="outside")
            show_plot(st, fig, "p2_monthly")
        st.success(
            _tr(
                "page02_data_ready.the_columntransformer_was_fitted_on_dev_only_485b52c",
                value0=f"{contract['encoded_feature_count']}",
            )
        )
        interpretation_card(
            st,
            _tr(
                "page02_data_ready.dev_contains_value0_rows_locked_value1_contains_96fd885",
                value0=f"{split['development_rows']:,}",
                value1=f"{split['locked_test_period_label']}",
                value2=f"{split['locked_test_rows']:,}",
                value3=f"{split['locked_test_pct']:.1f}",
                value4=f"{contract['encoded_feature_count']}",
            ),
            _tr("page02_data_ready.time_ordering_is_preserved_and_all_learned_9b31048"),
            _tr("page02_data_ready.do_not_inspect_locked_test_labels_during_6b4d7b8"),
            "success",
        )
        if not scaling.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    scaling,
                    x="feature",
                    y=["before_mean", "after_mean"],
                    barmode="group",
                    title=_src(
                        "page02_data_ready.numeric_feature_means_before_vs_after_scaling_e3b7af0"
                    ),
                )
                show_plot(st, fig, "p2_scaling_mean")
            with c2:
                fig = px.bar(
                    scaling,
                    x="feature",
                    y=["before_std", "after_std"],
                    barmode="group",
                    title=_src(
                        "page02_data_ready.numeric_feature_standard_deviation_before_vs_after_a9d2868"
                    ),
                )
                show_plot(st, fig, "p2_scaling_std")
        # Dynamic before/after numeric distribution view using saved DEV and encoded matrices.
        dev_raw = read_csv(root, "02_data_ready_for_ml/development_raw.csv")
        train_enc = read_csv(root, "02_data_ready_for_ml/train_preprocessed.csv")
        numeric_choices = [
            c
            for c in ["years_of_experience", "demand_score", "benefits_score_10", "skill_count"]
            if c in dev_raw.columns
        ]
        if numeric_choices:
            feature = st.selectbox(
                _tr(
                    "page02_data_ready.numeric_feature_distribution_before_vs_after_scaling_2bb7ac1"
                ),
                numeric_choices,
                key="p2_scale_feature",
                format_func=_option_label(),
            )
            enc_col = next(
                (c for c in train_enc.columns if c.endswith("__" + feature) or c == feature), None
            )
            if enc_col:
                before = pd.DataFrame(
                    {
                        "value": dev_raw[feature].astype(float),
                        "state": _src("page02_data_ready.before_scaling_6e0780b"),
                    }
                )
                after = pd.DataFrame(
                    {
                        "value": train_enc[enc_col].astype(float),
                        "state": _src("page02_data_ready.after_scaling_cf73f79"),
                    }
                )
                long = pd.concat([before, after], ignore_index=True)
                fig = px.histogram(
                    long,
                    x="value",
                    color="state",
                    nbins=30,
                    barmode="overlay",
                    opacity=0.65,
                    marginal="box",
                    title=_tr(
                        "page02_data_ready.value0_distribution_before_vs_after_scaling_cecccb9",
                        value0=f"{feature}",
                    ),
                )
                show_plot(st, fig, "p2_scale_distribution")
                r = (
                    scaling.loc[scaling.feature == feature].iloc[0]
                    if (scaling.feature == feature).any()
                    else None
                )
                if r is not None:
                    interpretation_card(
                        st,
                        _tr(
                            "page02_data_ready.value0_mean_std_change_from_value1_value2_5235386",
                            value0=f"{feature}",
                            value1=f"{r.before_mean:.2f}",
                            value2=f"{r.before_std:.2f}",
                            value3=f"{r.after_mean:.2f}",
                            value4=f"{r.after_std:.2f}",
                        ),
                        _tr(
                            "page02_data_ready.scaling_changes_units_not_information_content_the_ca8b6ec"
                        ),
                        _tr(
                            "page02_data_ready.use_the_post_scaling_distribution_to_verify_330044b"
                        ),
                        "info",
                    )
        if not fmap.empty:
            original = st.selectbox(
                _tr("page02_data_ready.inspect_encoded_columns_from_original_feature_27d6e82"),
                [_src("page02_data_ready.all_a52ace4")]
                + sorted(fmap.original_feature.dropna().astype(str).unique()),
                key="p2_map_original",
                format_func=_option_label(),
            )
            show = (
                fmap
                if original == _src("page02_data_ready.all_a52ace4")
                else fmap[fmap.original_feature.astype(str) == original]
            )
            st.dataframe(display_frame(show), width="stretch", hide_index=True, height=360)

    with tabs[3]:
        st.markdown(_tr("page02_data_ready.skills_multi_hot_encoding_evidence_1540be4"))
        if skills.empty:
            st.info(_tr("page02_data_ready.no_skill_token_summary_is_available_for_2639728"))
        else:
            topn = st.slider(
                _tr("page02_data_ready.top_skills_f3a4ad0"),
                10,
                60,
                30,
                key="p2_skill_topn",
            )
            metric = st.selectbox(
                _tr("page02_data_ready.skill_chart_metric_e707577"),
                ["records", "salary_mean", "years_mean"],
                key="p2_skill_metric",
                format_func=_option_label(),
            )
            d = skills.nlargest(topn, metric if metric != "years_mean" else "records")
            fig = px.bar(
                d.sort_values(metric),
                y="skill",
                x=metric,
                orientation="h",
                text=metric,
                hover_data=["records", "salary_mean", "salary_median", "years_mean"],
                title=_tr("page02_data_ready.top_skill_tokens_value0_fb0c94a", value0=f"{metric}"),
            )
            if metric == "salary_mean":
                fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            else:
                fig.update_traces(texttemplate="%{text:,.1f}", textposition="outside")
            show_plot(st, fig, "p2_skill_chart")
            if not skill_corr.empty:
                sc = skill_corr.head(topn).sort_values("pearson_r")
                fig = px.bar(
                    sc,
                    y="skill",
                    x="pearson_r",
                    orientation="h",
                    text="pearson_r",
                    hover_data=["records"],
                    title=_src(
                        "page02_data_ready.top_skill_tokens_by_train_only_correlation_d38a55f"
                    ),
                )
                fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
                fig.add_vline(x=0)
                show_plot(st, fig, "p2_skill_corr")
            low = skills[skills.records < 50].sort_values("records")
            st.caption(
                _tr(
                    "page02_data_ready.skills_with_fewer_than_50_records_value0_8eadd68",
                    value0=f"{len(low):,}",
                    value1=f"{len(skills):,}",
                )
            )
            with st.expander(_tr("page02_data_ready.skills_with_50_records_1eca427")):
                st.dataframe(display_frame(low), width="stretch", hide_index=True, height=360)
            top_skill = skills.sort_values("records", ascending=False).iloc[0]
            rare_share = 100 * len(low) / len(skills) if len(skills) else 0
            corr_note = ""
            if not skill_corr.empty:
                sc0 = skill_corr.sort_values("abs_r", ascending=False).iloc[0]
                corr_note = _tr(
                    "page02_data_ready.strongest_skill_target_r_is_value0_at_377d953",
                    value0=f"{sc0.skill}",
                    value1=f"{sc0.pearson_r:+.2f}",
                )
            interpretation_card(
                st,
                _tr(
                    "page02_data_ready.most_frequent_train_skill_is_value0_value1_9577e14",
                    value0=f"{top_skill.skill}",
                    value1=f"{int(top_skill.records)}",
                    value2=f"{rare_share:.1f}",
                    value3=f"{corr_note}",
                ),
                _tr("page02_data_ready.sparse_skills_can_be_useful_for_scenario_f849161"),
                _tr("page02_data_ready.keep_the_full_train_vocabulary_for_the_7c9193f"),
                "warning" if rare_share > 30 else "info",
            )

    with tabs[4]:
        st.markdown(_tr("page01_data_basic_clean.interactive_chart_explorer_320e7ef"))
        chart = chart_selector(
            st,
            _tr("page02_data_ready.choose_evidence_view_f26ffcd"),
            [
                _tr("page02_data_ready.top_feature_family_correlation_b9573ae"),
                _tr("page02_data_ready.top_encoded_pearson_correlation_dd29abc"),
                _tr("page02_data_ready.temporal_monthly_split_fb62991"),
                _tr("page02_data_ready.ablation_comparison_d94ac8f"),
                _tr("page02_data_ready.skill_frequency_vs_salary_241f0a4"),
            ],
            "p2_explorer_select",
        )
        topn = st.slider(_tr("page02_data_ready.top_n_314de6a"), 5, 50, 20, key="p2_explorer_topn")
        if chart == _src("page02_data_ready.top_feature_family_correlation_b9573ae"):
            if fmap.empty:
                st.info(_tr("page02_data_ready.feature_map_is_unavailable_6bbe167"))
                return
            x = corr.merge(fmap, on="encoded_feature", how="left")
            g = (
                x.groupby("original_feature", dropna=False)
                .agg(
                    max_abs_r=("abs_r", "max"),
                    signed_r=("pearson_r", lambda s: s.iloc[np.argmax(np.abs(s.to_numpy()))]),
                )
                .reset_index()
                .nlargest(topn, "max_abs_r")
                .sort_values("signed_r")
            )
            fig = px.bar(
                g,
                y="original_feature",
                x="signed_r",
                orientation="h",
                text="signed_r",
                title=_src("page02_data_ready.top_feature_family_correlation_on_train_b3ebcda"),
            )
            fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
            fig.add_vline(x=0)
        elif chart == _src("page02_data_ready.top_encoded_pearson_correlation_dd29abc"):
            d = corr.nlargest(topn, "abs_r").sort_values("pearson_r")
            fig = px.bar(
                d,
                y="encoded_feature",
                x="pearson_r",
                orientation="h",
                text="pearson_r",
                title=_src(
                    "page02_data_ready.top_encoded_features_by_pearson_correlation_on_bed35e0"
                ),
            )
            fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
            fig.add_vline(x=0)
        elif chart == _src("page02_data_ready.temporal_monthly_split_fb62991"):
            fig = px.bar(
                monthly,
                x="period",
                y="records",
                color="split",
                text="records",
                title=_src("page02_data_ready.temporal_split_d2364fe"),
            )
        elif chart == _src("page02_data_ready.ablation_comparison_d94ac8f"):
            fig = px.bar(
                abl.sort_values("MAE_mean", ascending=False),
                y="option",
                x="MAE_mean",
                orientation="h",
                text="MAE_mean",
                title=_src("page02_data_ready.ablation_mae_d29da73"),
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        else:
            d = skills.nlargest(topn, "records") if not skills.empty else pd.DataFrame()
            if d.empty:
                st.info(_tr("page02_data_ready.skill_summary_is_unavailable_21e49da"))
                return
            fig = px.scatter(
                d,
                x="records",
                y="salary_mean",
                size="records",
                text="skill",
                hover_data=["years_mean"],
                title=_src("page02_data_ready.skill_frequency_vs_mean_salary_42c5418"),
            )
            fig.update_traces(textposition=_src("model_evidence.top_center_24b3167"))
        show_plot(st, fig, "p2_explorer")
        interpretation_card(
            st,
            _tr(
                "page02_data_ready.explorer_view_value0_the_visual_is_generated_4b5c7da",
                value0=f"{chart}",
            ),
            _tr("page02_data_ready.use_the_selector_to_test_whether_the_b350dcd"),
            _tr("page02_data_ready.if_a_conclusion_changes_materially_by_view_b591d86"),
            "info",
        )

    with tabs[5]:
        downloadable_table(
            st, policy, _tr("page02_data_ready.feature_governance_policy_29973dd"), "p2_policy"
        )
        downloadable_table(
            st,
            corr.head(100),
            _tr("page02_data_ready.train_only_encoded_correlations_3467657"),
            "p2_corr_table",
        )
        downloadable_table(
            st, abl, _tr("page02_data_ready.ablation_results_5c74067"), "p2_ablation_table"
        )
        if not fmap.empty:
            downloadable_table(
                st,
                fmap,
                _tr("page02_data_ready.encoding_feature_construction_map_4d0bc0a"),
                "p2_feature_map",
            )
        if not scaling.empty:
            downloadable_table(
                st,
                scaling,
                _tr("page02_data_ready.numeric_scaling_summary_f577e14"),
                "p2_scaling_table",
            )
        if not skills.empty:
            downloadable_table(
                st,
                skills,
                _tr("page02_data_ready.skill_token_summary_60f5fb1"),
                "p2_skill_table",
                height=420,
            )
        if not skill_corr.empty:
            downloadable_table(
                st,
                skill_corr,
                _tr("page02_data_ready.skill_target_correlation_train_only_c9e2081"),
                "p2_skill_corr_table",
                height=420,
            )
