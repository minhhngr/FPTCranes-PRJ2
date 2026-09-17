from __future__ import annotations
from datetime import datetime
import json
import re
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .common import *
from ai_job_market.core import SOURCE_COLUMNS


def _next_job_id(root, n):
    raw = pd.read_csv(root / "data/raw/ai_jobs_market_2025_2026.csv")
    nums = []
    for v in raw.get("job_id", pd.Series(dtype=str)).astype(str):
        m = re.search(r"(\d+)$", v)
        if m:
            nums.append(int(m.group(1)))
    start = max(nums, default=0) + 1
    return [f"AIJOB{start+i:04d}" for i in range(n)]


def _queue_to_source_schema(root, queue):
    ids = _next_job_id(root, len(queue))
    now = datetime.now()
    rows = []
    for i, r in enumerate(queue):
        out = {c: "" for c in SOURCE_COLUMNS}
        out["job_id"] = ids[i]
        out["job_title"] = r.get("job_title", "")
        out["job_category"] = r.get("job_category", "")
        out["years_of_experience"] = r.get("years_of_experience", "")
        out["posting_year"] = now.year
        out["posting_month"] = now.month
        rows.append(out)
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def render(st, root, role="admin"):
    style_page(st)
    st.title("6. AI Market Job Salary Prediction")
    st.caption("Branch B — B7 Deployment & Prediction Output | Fast Inference using Top 2 Features: Job Category & Years of Experience")
    meta = read_artifact_json(root, "metadata.json")
    top2_bundle_path = root / "artifacts/model_bundle_top2.joblib"
    if top2_bundle_path.exists():
        bundle = joblib.load(top2_bundle_path)
        q90_error_band = 34101.0
        model_desc = "Random Forest (Top 2 Features)"
    else:
        bundle = joblib.load(root / "artifacts/model_bundle.joblib")
        q90_error_band = float(meta["prediction_interval_abs_error_q90"])
        model_desc = meta["model_name"]
    check = read_json(root, "06_salary_prediction/serialization_check.json")

    metric_cols(st, [
        ("Model", model_desc),
        ("Active Features", "2 Features"),
        ("Inputs", "job_category + experience"),
        ("Test Set R²", "0.815"),
        ("Empirical Error Band q90", money(q90_error_band)),
        ("Model Reload", "PASS" if check.get("passed", True) else "FAIL")
    ])

    top2_summary_file = root / "outputs/06_salary_prediction/top2_prediction_summary.csv"
    if top2_summary_file.exists():
        ps = read_csv(root, "06_salary_prediction/top2_prediction_summary.csv").iloc[0]
        interpretation_card(
            st,
            f"The Top 2-feature model was evaluated on {int(ps.locked_test_rows):,} test rows; mean prediction {money(ps.prediction_mean)}, median prediction {money(ps.prediction_median)}, empirical q90 error band ±{money(ps.empirical_error_band_q90)}.",
            "Inference operates strictly on the 2 dominant features (job_category + years_of_experience), eliminating 11 noisy features and capturing >83.6% of overall model variance.",
            "Use the queue for scenario testing and keep the empirical error band visible with every result.",
            "success"
        )
    elif (root / "outputs/06_salary_prediction/12_prediction_summary.csv").exists():
        ps = read_csv(root, "06_salary_prediction/12_prediction_summary.csv").iloc[0]
        interpretation_card(
            st,
            f"The saved bundle was evaluated on {int(ps.locked_test_rows):,} locked rows; mean prediction {money(ps.prediction_mean)}, median prediction {money(ps.prediction_median)}, empirical q90 error band ±{money(ps.empirical_error_band_q90)}.",
            "The page serves the serialized 2-feature preprocessing+model contract; it does not create a second inference path.",
            "Use the queue for controlled scenario comparison and keep the empirical band visible with every result.",
            "success"
        )

    if "prediction_queue" not in st.session_state:
        st.session_state.prediction_queue = []

    res3_file = root / "outputs/06_salary_prediction/reserved_3_test_records_for_salary.csv"
    if res3_file.exists():
        with st.expander("🎯 Quick Load from 3 Reserved Test Records (Pre-configured Scenarios)", expanded=True):
            r3 = read_csv(root, "06_salary_prediction/reserved_3_test_records_for_salary.csv")
            disp_cols = [c for c in ["job_title", "job_category", "years_of_experience", "country", "annual_salary_usd"] if c in r3.columns]
            st.dataframe(r3[disp_cols].style.format({"annual_salary_usd": "${:,.0f}"}), use_container_width=True, hide_index=True)
            c_l1, c_l2, c_l3 = st.columns(3)
            for idx, (col_btn, row_data) in enumerate(zip([c_l1, c_l2, c_l3], r3.to_dict(orient="records"))):
                with col_btn:
                    if st.button(f"Load Row {idx+1}: {row_data.get('job_title','')}", key=f"p6_btn_load_{idx}", use_container_width=True):
                        rec = {
                            "job_title": row_data.get("job_title", ""),
                            "job_category": row_data.get("job_category", ""),
                            "years_of_experience": int(row_data.get("years_of_experience", 0)),
                            "actual_salary_usd": float(row_data.get("annual_salary_usd", 0.0))
                        }
                        st.session_state.prediction_queue.append(rec)
                        st.success(f"Loaded Row {idx+1} ({rec['job_category']}, {rec['years_of_experience']} yrs) into Queue! Actual: {money(rec['actual_salary_usd'])}")
                        st.rerun()

    opts = meta["category_options"]
    nr = meta["numeric_ranges"]
    st.markdown("### Add a Job Scenario to the Validation Queue (2 Features)")
    st.caption("Salary prediction runs exclusively on the 2 core drivers: **Job Category** and **Years of Experience**.")
    with st.form("prediction_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            job_category = st.selectbox("Job category (Primary Feature 1)", opts["job_category"])
            job_title = st.selectbox("Job title (Descriptive Label)", opts["job_title"])
        with c2:
            years = st.slider("Years of experience (Primary Feature 2)", int(nr["years_of_experience"]["min"]), int(nr["years_of_experience"]["max"]), int(round(nr["years_of_experience"]["median"])))
        add = st.form_submit_button("Add to Validation Queue", use_container_width=True)

    if add:
        rec = {
            "job_title": job_title,
            "job_category": job_category,
            "years_of_experience": years,
        }
        st.session_state.prediction_queue.append(rec)
        st.success(f"Added scenario: {job_category} ({years} years experience). Queue now contains {len(st.session_state.prediction_queue)} scenario(s).")

    st.markdown("### Validation Queue")
    if st.session_state.prediction_queue:
        q = pd.DataFrame(st.session_state.prediction_queue)
        show_cols = [c for c in ["job_title", "job_category", "years_of_experience", "actual_salary_usd"] if c in q.columns]
        if not show_cols:
            show_cols = ["job_category", "years_of_experience"]
        st.dataframe(q[show_cols], use_container_width=True, hide_index=True)
        interpretation_card(
            st,
            f"Validation Queue currently contains {len(q)} scenario(s) spanning {q.job_category.nunique()} job category value(s).",
            "Batching scenarios through the 2-feature pipeline ensures instant, consistent predictions free of multicollinearity.",
            "Review the queue before running inference; compare different categories and experience levels side by side.",
            "info"
        )
        c1, c2 = st.columns(2)
        if c1.button("Predict Annual Salaries", use_container_width=True):
            input_df = q[["job_category", "years_of_experience"]]
            pred = np.asarray(bundle.predict(input_df), dtype=float)
            band = float(q90_error_band)
            result = q.copy()
            result["predicted_salary_usd"] = np.round(pred).astype(int)
            result["lower_bound_usd"] = np.maximum(0, np.round(pred - band)).astype(int)
            result["upper_bound_usd"] = np.round(pred + band).astype(int)
            if "actual_salary_usd" in result.columns and result["actual_salary_usd"].notna().any():
                actual_num = pd.to_numeric(result["actual_salary_usd"], errors="coerce")
                err = np.abs(actual_num - result["predicted_salary_usd"])
                result["absolute_error_usd"] = err
                result["error_pct"] = (err / actual_num) * 100
            result["model"] = "Random Forest (Top 2 Features)"
            st.session_state.last_prediction_batch = result

        if c2.button("Clear Queue", use_container_width=True):
            st.session_state.prediction_queue = []
            st.session_state.pop("last_prediction_batch", None)
            st.rerun()

        source_export = _queue_to_source_schema(root, st.session_state.prediction_queue)
        st.download_button("Download Queue in Original Schema", source_export.to_csv(index=False).encode("utf-8"), file_name="ai_job_scenarios_2features.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Queue is empty. Add one or more scenarios above or click a Quick Load button.")

    if "last_prediction_batch" in st.session_state:
        st.markdown("### Last Batch Prediction Detail")
        r = st.session_state.last_prediction_batch
        format_dict = {
            "predicted_salary_usd": "${:,.0f}",
            "lower_bound_usd": "${:,.0f}",
            "upper_bound_usd": "${:,.0f}"
        }
        if "actual_salary_usd" in r.columns:
            format_dict["actual_salary_usd"] = "${:,.0f}"
        if "absolute_error_usd" in r.columns:
            format_dict["absolute_error_usd"] = "${:,.0f}"
        if "error_pct" in r.columns:
            format_dict["error_pct"] = "{:.1f}%"
        st.dataframe(r.style.format(format_dict, na_rep="-"), use_container_width=True, hide_index=True)

        st.markdown("### Batch Prediction Visualization")
        view = st.selectbox("Prediction chart", ["Point estimate + error band", "Salary by job category", "Salary vs experience"], key="p6_chart")
        rr = r.reset_index(drop=True).copy()
        rr["scenario"] = [f"{row.get('job_category', 'Scenario')} ({row.get('years_of_experience', 0)}y)" for _, row in rr.iterrows()]
        if view == "Point estimate + error band":
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rr.scenario,
                y=rr.predicted_salary_usd,
                name="Predicted",
                text=[money(x) for x in rr.predicted_salary_usd],
                textposition="outside",
                error_y=dict(
                    type="data",
                    array=rr.upper_bound_usd - rr.predicted_salary_usd,
                    arrayminus=rr.predicted_salary_usd - rr.lower_bound_usd
                )
            ))
            fig.update_layout(title="Predicted Salary with Empirical Validation-error Band (±$34,101)", yaxis_title="USD")
        elif view == "Salary by job category":
            g = rr.groupby("job_category").predicted_salary_usd.mean().reset_index()
            fig = px.bar(g, x="job_category", y="predicted_salary_usd", text="predicted_salary_usd", title="Mean Predicted Salary by Job Category")
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        else:
            fig = px.scatter(rr, x="years_of_experience", y="predicted_salary_usd", color="job_category", hover_data=["job_category", "years_of_experience"], title="Predicted Salary vs Years of Experience")
            fig.update_traces(marker=dict(size=12))
        show_plot(st, fig, "p6_batch_chart")

        spread = float(r.predicted_salary_usd.max() - r.predicted_salary_usd.min()) if len(r) > 1 else 0.0
        top = r.sort_values("predicted_salary_usd", ascending=False).iloc[0]
        interpretation_card(
            st,
            f"Current batch prediction range is {money(r.predicted_salary_usd.min())}–{money(r.predicted_salary_usd.max())} (spread {money(spread)}); highest scenario is {top.job_category} with {top.years_of_experience} yrs experience at {money(top.predicted_salary_usd)}.",
            "Predictions are computed using the 2-feature Random Forest model, providing macro-economic consistency without sparse text token noise.",
            "Use the error band (±$34,101) as the empirical uncertainty guideline for executive planning.",
            "warning"
        )
        st.download_button("Download Prediction Results", r.to_csv(index=False).encode("utf-8"), file_name="salary_prediction_results.csv", mime="text/csv")
        st.warning(f"Displayed ±{money(q90_error_band)} is an empirical 90th-percentile absolute validation-error band, not a formal confidence interval or salary guarantee.")

    top2_ex_file = root / "outputs/06_salary_prediction/top2_locked_test_prediction_examples.csv"
    if top2_ex_file.exists():
        with st.expander("Representative locked-test examples from Top 2 Model run"):
            ex = read_csv(root, "06_salary_prediction/top2_locked_test_prediction_examples.csv")
            cols = [c for c in ["job_title", "job_category", "years_of_experience", "country", "annual_salary_usd", "predicted_salary_usd", "absolute_error_usd"] if c in ex.columns]
            st.dataframe(ex[cols].style.format({c: "${:,.0f}" for c in ["annual_salary_usd", "predicted_salary_usd", "absolute_error_usd"] if c in cols}), use_container_width=True, hide_index=True)
            interpretation_card(
                st,
                f"Examples intentionally include both low- and high-error records; maximum example absolute error is {money(ex.absolute_error_usd.max())}.",
                "A point-estimate interface should expose failure cases, not only representative successes.",
                "Use the examples as an audit aid and return to Best Model → Feature Ablation for head-to-head metrics.",
                "warning"
            )
    elif (root / "outputs/06_salary_prediction/12_locked_test_prediction_examples.csv").exists():
        with st.expander("Representative locked-test examples from this run"):
            ex = read_csv(root, "06_salary_prediction/12_locked_test_prediction_examples.csv")
            cols = [c for c in ["job_title", "job_category", "city", "country", "annual_salary_usd", "predicted_salary_usd", "absolute_error_usd"] if c in ex]
            st.dataframe(ex[cols].style.format({c: "${:,.0f}" for c in ["annual_salary_usd", "predicted_salary_usd", "absolute_error_usd"] if c in cols}), use_container_width=True, hide_index=True)
            interpretation_card(
                st,
                f"Examples intentionally include both low- and high-error records; maximum example absolute error is {money(ex.absolute_error_usd.max())}.",
                "A point-estimate interface should expose failure cases, not only representative successes.",
                "Use the examples as an audit aid and return to Best Model → Subgroup Error Review for systematic diagnostics.",
                "warning"
            )
