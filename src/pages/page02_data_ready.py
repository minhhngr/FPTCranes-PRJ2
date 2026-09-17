from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ai_job_market.core import dynamic_insights

from .common import *


def render(st, root, role="admin"):
    style_page(st)
    st.title("2. Data Ready for Machine Learning")
    st.caption(
        "Common Foundation — Stage 4 Feature Importance & Leakage Prevention → Stage 5 Feature Selection → Branch B B1–B3 Temporal Split & TRAIN-only Preprocessing"
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
            ("Allowed raw inputs", int((policy.policy == "ALLOW").sum())),
            ("Blocked fields", int((policy.policy == "BLOCK").sum())),
            ("DEV rows", f"{split['development_rows']:,}"),
            ("Locked test rows", f"{split['locked_test_rows']:,}"),
            ("Encoded train features", contract["encoded_feature_count"]),
        ],
    )

    tabs = st.tabs(
        [
            "Stage 4 · Governance & Correlation",
            "Stage 5 · Ablation & Selection",
            "B1–B3 · Temporal Split & Preprocessing",
            "Skills Analysis",
            "Interactive Chart Explorer",
            "Evidence Tables",
        ]
    )
    with tabs[0]:
        st.markdown("### Stage 4 — Feature Importance & Leakage Prevention")
        c1, c2 = st.columns([1.1, 1.9])
        with c1:
            fig = px.pie(
                policy[policy.policy.isin(["ALLOW", "BLOCK"])],
                names="policy",
                title="Feature Governance Mix",
                hole=0.55,
            )
            show_plot(st, fig, "p2_policy_pie")
        with c2:
            st.dataframe(policy, use_container_width=True, hide_index=True, height=360)
        insight_box(st, dynamic_insights(root / "outputs").get("correlation", ""))
        topn = st.slider("Top encoded correlations", 10, 60, 30, key="p2_corr_topn")
        direction = st.radio(
            "Correlation view",
            ["Absolute strongest", "Positive only", "Negative only"],
            horizontal=True,
            key="p2_corr_dir",
        )
        d = corr.copy()
        if direction == "Positive only":
            d = d[d.pearson_r >= 0].sort_values("pearson_r", ascending=False)
        elif direction == "Negative only":
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
            title="TRAIN-only Encoded Feature Correlation with annual_salary_usd",
        )
        fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        fig.add_vline(x=0)
        show_plot(st, fig, "p2_corr")
        st.caption(
            "Correlation is computed on TRAIN/DEV only. It is diagnostic association and does not establish causality."
        )
        strongest = corr.sort_values("abs_r", ascending=False).iloc[0]
        blocked = int((policy.policy == "BLOCK").sum())
        allowed = int((policy.policy == "ALLOW").sum())
        interpretation_card(
            st,
            f"Governance keeps {allowed} raw serving inputs and blocks {blocked} fields. Strongest TRAIN-only encoded association is {strongest.encoded_feature} at r={strongest.pearson_r:+.2f}.",
            "The leakage gate is intentionally stricter than simple correlation ranking: target-adjacent or contradictory fields remain blocked even if they look predictive.",
            "Use correlations to diagnose signal concentration, but let feature policy and temporal ablation decide what can enter the model.",
            "warning",
        )

    with tabs[1]:
        st.markdown("### Stage 5 — Feature Selection / Ablation")
        metric = st.selectbox(
            "Ablation comparison metric", ["MAE_mean", "RMSE_mean", "R2_mean"], key="p2_abl_metric"
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
            title=f"Ablation Study — {metric}",
        )
        if metric in {"MAE_mean", "RMSE_mean"}:
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        else:
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        show_plot(st, fig, "p2_ablation")
        best = abl.sort_values("MAE_mean").iloc[0]
        st.info(
            f"Best ablation by temporal-CV MAE: **{best['option']}** at **{money(best['MAE_mean'])}**. This evidence is used diagnostically; governance rules still control the serving contract."
        )
        st.dataframe(
            abl.style.format(
                {
                    "MAE_mean": "${:,.0f}",
                    "MAE_std": "${:,.0f}",
                    "RMSE_mean": "${:,.0f}",
                    "R2_mean": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        base = (
            abl.loc[abl.option.str.contains("Conservative", case=False, na=False)].iloc[0]
            if abl.option.str.contains("Conservative", case=False, na=False).any()
            else abl.iloc[-1]
        )
        gain = (
            100 * (float(base.MAE_mean) - float(best.MAE_mean)) / float(base.MAE_mean)
            if float(base.MAE_mean)
            else 0.0
        )
        interpretation_card(
            st,
            f"Best ablation is {best['option']} with mean temporal-CV MAE {money(best.MAE_mean)}, a {gain:.1f}% improvement versus the conservative reference shown here.",
            "Ablation isolates incremental predictive contribution; it does not automatically override leakage or deployment-governance rules.",
            "Retain only feature families whose out-of-time improvement is material and semantically defensible.",
            "success" if gain > 5 else "warning",
        )

    with tabs[2]:
        st.markdown("### B1–B3 — Locked Temporal Test & TRAIN-only Preprocessing")
        st.info(
            f"Locked period: **{split['locked_test_period_label']}** — {split['locked_test_rows']:,} rows ({split['locked_test_pct']:.1f}%). The locked period is excluded from CV, feature selection and preprocessing fitting."
        )
        if not monthly.empty:
            fig = px.bar(
                monthly,
                x="period",
                y="records",
                color="split",
                barmode="stack",
                text="records",
                title="Monthly Posting Volume & Locked Temporal Test",
            )
            fig.update_traces(textposition="outside")
            show_plot(st, fig, "p2_monthly")
        st.success(
            f"The ColumnTransformer was fitted on DEV only and produced **{contract['encoded_feature_count']} encoded features**. Test data are transformed with the frozen objects only — no re-fitting."
        )
        interpretation_card(
            st,
            f"DEV contains {split['development_rows']:,} rows; locked {split['locked_test_period_label']} contains {split['locked_test_rows']:,} rows ({split['locked_test_pct']:.1f}%). The frozen preprocessor expands the approved raw contract to {contract['encoded_feature_count']} columns.",
            "Time ordering is preserved and all learned preprocessing objects are fit on DEV only, preventing look-ahead through category vocabularies or scaling statistics.",
            "Do not inspect locked-test labels during model-family comparison or tuning; open them only after the configuration is frozen.",
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
                    title="Numeric Feature Means — Before vs After Scaling",
                )
                show_plot(st, fig, "p2_scaling_mean")
            with c2:
                fig = px.bar(
                    scaling,
                    x="feature",
                    y=["before_std", "after_std"],
                    barmode="group",
                    title="Numeric Feature Standard Deviation — Before vs After Scaling",
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
                "Numeric feature distribution — before vs after scaling",
                numeric_choices,
                key="p2_scale_feature",
            )
            enc_col = next(
                (c for c in train_enc.columns if c.endswith("__" + feature) or c == feature), None
            )
            if enc_col:
                before = pd.DataFrame(
                    {"value": dev_raw[feature].astype(float), "state": "Before scaling"}
                )
                after = pd.DataFrame(
                    {"value": train_enc[enc_col].astype(float), "state": "After scaling"}
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
                    title=f"{feature} — Distribution Before vs After Scaling",
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
                        f"{feature}: mean/std change from {r.before_mean:.2f}/{r.before_std:.2f} to {r.after_mean:.2f}/{r.after_std:.2f} after the TRAIN-fitted scaler.",
                        "Scaling changes units, not information content. The same frozen transform is applied to the locked test.",
                        "Use the post-scaling distribution to verify that preprocessing behaves as expected; do not fit a separate scaler on test data.",
                        "info",
                    )
        if not fmap.empty:
            original = st.selectbox(
                "Inspect encoded columns from original feature",
                ["All"] + sorted(fmap.original_feature.dropna().astype(str).unique()),
                key="p2_map_original",
            )
            show = (
                fmap if original == "All" else fmap[fmap.original_feature.astype(str) == original]
            )
            st.dataframe(show, use_container_width=True, hide_index=True, height=360)

    with tabs[3]:
        st.markdown("### Skills — Multi-hot Encoding Evidence")
        if skills.empty:
            st.info("No skill-token summary is available for this run.")
        else:
            topn = st.slider("Top skills", 10, 60, 30, key="p2_skill_topn")
            metric = st.selectbox(
                "Skill chart metric",
                ["records", "salary_mean", "years_mean"],
                key="p2_skill_metric",
            )
            d = skills.nlargest(topn, metric if metric != "years_mean" else "records")
            fig = px.bar(
                d.sort_values(metric),
                y="skill",
                x=metric,
                orientation="h",
                text=metric,
                hover_data=["records", "salary_mean", "salary_median", "years_mean"],
                title=f"Top Skill Tokens — {metric}",
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
                    title="Top Skill Tokens by TRAIN-only Correlation with annual_salary_usd",
                )
                fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
                fig.add_vline(x=0)
                show_plot(st, fig, "p2_skill_corr")
            low = skills[skills.records < 50].sort_values("records")
            st.caption(
                f"Skills with fewer than 50 records: {len(low):,} of {len(skills):,} tokens in the current TRAIN vocabulary."
            )
            with st.expander("Skills with <50 records"):
                st.dataframe(low, use_container_width=True, hide_index=True, height=360)
            top_skill = skills.sort_values("records", ascending=False).iloc[0]
            rare_share = 100 * len(low) / len(skills) if len(skills) else 0
            corr_note = ""
            if not skill_corr.empty:
                sc0 = skill_corr.sort_values("abs_r", ascending=False).iloc[0]
                corr_note = f" Strongest skill-target |r| is {sc0.skill} at r={sc0.pearson_r:+.2f}."
            interpretation_card(
                st,
                f"Most frequent TRAIN skill is {top_skill.skill} ({int(top_skill.records)} records). {rare_share:.1f}% of skill tokens occur in fewer than 50 records.{corr_note}",
                "Sparse skills can be useful for scenario detail but unstable for inference if they have little support.",
                "Keep the full TRAIN vocabulary for the serving contract, but avoid claiming rare skills are reliable salary drivers without external validation.",
                "warning" if rare_share > 30 else "info",
            )

    with tabs[4]:
        st.markdown("### Interactive Chart Explorer")
        chart = chart_selector(
            st,
            "Choose evidence view",
            [
                "Top feature-family correlation",
                "Top encoded Pearson correlation",
                "Temporal monthly split",
                "Ablation comparison",
                "Skill frequency vs salary",
            ],
            "p2_explorer_select",
        )
        topn = st.slider("Top N", 5, 50, 20, key="p2_explorer_topn")
        if chart == "Top feature-family correlation":
            if fmap.empty:
                st.info("Feature map is unavailable.")
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
                title="Top Feature-family Correlation on TRAIN",
            )
            fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
            fig.add_vline(x=0)
        elif chart == "Top encoded Pearson correlation":
            d = corr.nlargest(topn, "abs_r").sort_values("pearson_r")
            fig = px.bar(
                d,
                y="encoded_feature",
                x="pearson_r",
                orientation="h",
                text="pearson_r",
                title="Top Encoded Features by Pearson Correlation on TRAIN",
            )
            fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
            fig.add_vline(x=0)
        elif chart == "Temporal monthly split":
            fig = px.bar(
                monthly,
                x="period",
                y="records",
                color="split",
                text="records",
                title="Temporal Split",
            )
        elif chart == "Ablation comparison":
            fig = px.bar(
                abl.sort_values("MAE_mean", ascending=False),
                y="option",
                x="MAE_mean",
                orientation="h",
                text="MAE_mean",
                title="Ablation MAE",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        else:
            d = skills.nlargest(topn, "records") if not skills.empty else pd.DataFrame()
            if d.empty:
                st.info("Skill summary is unavailable.")
                return
            fig = px.scatter(
                d,
                x="records",
                y="salary_mean",
                size="records",
                text="skill",
                hover_data=["years_mean"],
                title="Skill Frequency vs Mean Salary",
            )
            fig.update_traces(textposition="top center")
        show_plot(st, fig, "p2_explorer")
        interpretation_card(
            st,
            f"Explorer view: {chart}. The visual is generated from the active run's saved TRAIN/DEV evidence, not hard-coded values.",
            "Use the selector to test whether the same conclusion persists across correlation, split, ablation and skills views.",
            "If a conclusion changes materially by view, treat the feature as uncertain and prioritize temporal-validation evidence.",
            "info",
        )

    with tabs[5]:
        downloadable_table(st, policy, "Feature Governance Policy", "p2_policy")
        downloadable_table(st, corr.head(100), "TRAIN-only Encoded Correlations", "p2_corr_table")
        downloadable_table(st, abl, "Ablation Results", "p2_ablation_table")
        if not fmap.empty:
            downloadable_table(st, fmap, "Encoding / Feature Construction Map", "p2_feature_map")
        if not scaling.empty:
            downloadable_table(st, scaling, "Numeric Scaling Summary", "p2_scaling_table")
        if not skills.empty:
            downloadable_table(st, skills, "Skill Token Summary", "p2_skill_table", height=420)
        if not skill_corr.empty:
            downloadable_table(
                st,
                skill_corr,
                "Skill-target Correlation (TRAIN only)",
                "p2_skill_corr_table",
                height=420,
            )
