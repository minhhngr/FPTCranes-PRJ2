from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from ai_job_market.salary_inference import (
    PredictionError,
    build_growth_inputs,
    build_prediction_audit_csv,
    build_source_schema_csv,
    predict_scenarios,
)
from ai_job_market.scenario_policy import ScenarioValidationError, validate_scenarios
from ai_job_market.ui_evidence_io import EvidenceContractError

from .common import money, show_plot, style_page
from .model_evidence import (
    PREDICTION_SERIES_COLORS,
    chart_container_width,
    load_evidence,
    load_json_file,
    load_top2_bundle,
    prediction_batch_conclusion,
    prediction_interval_figure,
    render_conclusion,
    render_metric_glossary,
    render_page_brief,
)


def _reset_results(st):
    st.session_state.pop("p6_results", None)
    st.session_state.pop("p6_growth_results", None)


def _append(st, row):
    queue = st.session_state.p6_queue
    if len(queue) >= 100:
        st.error("The queue limit is 100 scenarios.")
        return
    st.session_state.p6_next_id += 1
    row = dict(row)
    row["scenario_id"] = f"S{st.session_state.p6_next_id:03d}"
    st.session_state.p6_queue = [*queue, row]
    st.session_state.p6_revision += 1
    _reset_results(st)


def _metrics(st, values):
    with st.container(horizontal=True):
        for label, value in values:
            st.metric(label, value, border=True)


def render(st, root, role="admin"):
    style_page(st)
    st.title("6. Controlled salary scenarios")
    st.caption("Fixed Top-2 Random Forest · observed DEV policy · historical empirical uncertainty")
    try:
        manifest, tables = load_evidence(root)
        policy = load_json_file(root, manifest, "scenario_policy")
        metadata = load_json_file(root, manifest, "top2_metadata")
        bundle = load_top2_bundle(root, manifest)
    except (EvidenceContractError, KeyError, OSError, ValueError) as exc:
        st.error(f"Top-2 serving evidence is unavailable or incompatible: {exc}")
        st.code(
            f'PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "{root}"'
        )
        return

    context = (
        str(root.resolve()),
        manifest["evidence_id"],
        policy["policy_hash"],
        metadata["model_id"],
    )
    if st.session_state.get("p6_context") != context:
        st.session_state.p6_context = context
        st.session_state.p6_queue = []
        st.session_state.p6_revision = 0
        st.session_state.p6_next_id = 0
        st.session_state.pop("p6_results", None)
        st.session_state.pop("p6_growth_results", None)
        st.session_state.p6_context_reset = True
    if st.session_state.pop("p6_context_reset", False):
        st.info(
            "Scenario state was reset because the active workspace, evidence, policy, or model changed."
        )

    test_metrics = metadata["historical_test_metrics"]
    render_page_brief(
        st,
        question="How do validated job-category and experience scenarios compare under the fixed two-input salary model?",
        evidence_scope=f"Inputs: {', '.join(metadata['feature_order'])} · historical reference: {int(manifest['datasets']['historical_test']['rows'])} rows · {manifest['evidence_id']}",
        takeaway=f"The fixed two-input model records historical-test R² {test_metrics['R2']:.3f} with an empirical q90 absolute error of {money(metadata['q90_abs_error_usd'])}.",
        limitation="This is an academic scenario comparison, not a compensation recommendation or guaranteed salary range.",
    )
    _metrics(
        st,
        [
            ("Model", metadata["model_name"]),
            ("Active inputs", f"{len(metadata['feature_order'])}"),
            ("Historical test R²", f"{test_metrics['R2']:.3f}"),
            ("Empirical q90", f"±{money(metadata['q90_abs_error_usd'])}"),
        ],
    )
    st.caption(
        f"Evidence `{manifest['evidence_id']}` · fixed feature ablation, not independently tuned · no training occurs on this page"
    )
    render_metric_glossary(st, ["R²", "q90"])

    st.subheader("1. Quick-load historical benchmarks")
    examples = tables.get("benchmark_examples", pd.DataFrame())
    if examples.empty:
        st.warning("No strict-policy historical benchmark examples are available.")
    else:
        with st.container(horizontal=True):
            for index, example in examples.head(3).iterrows():
                label = f"{example.job_title} · {example.years_of_experience:g}y · actual {money(example.annual_salary_usd)}"
                if st.button(label, key=f"p6_benchmark_{index}"):
                    row = example.to_dict()
                    row.update(
                        {
                            "source": "benchmark",
                            "actual_salary_usd": float(example.annual_salary_usd),
                        }
                    )
                    try:
                        validate_scenarios([{**row, "scenario_id": "preview"}], policy)
                    except ScenarioValidationError as exc:
                        st.error(str(exc))
                    else:
                        _append(st, row)
                        st.rerun()
        st.caption(
            "Examples were already included in historical test scoring. They are traceable, not pristine or newly held out."
        )

    st.subheader("2. Controlled scenario builder")
    pairs = pd.DataFrame(policy["pairs"])
    categories = sorted(pairs.job_category.unique())
    category = st.selectbox("Job category", categories, key="p6_category")
    titles = sorted(pairs.loc[pairs.job_category == category, "job_title"].unique())
    if st.session_state.get("p6_title") not in titles:
        st.session_state.p6_title = titles[0]
    title = st.selectbox("Job title — observed DEV association", titles, key="p6_title")
    bounds = policy["titles"][title]
    experience_key = f"p6_experience_{title}"
    if experience_key not in st.session_state:
        st.session_state[experience_key] = float(bounds["median"])
    years = st.number_input(
        "Years of experience",
        min_value=float(bounds["min"]),
        max_value=float(bounds["max"]),
        step=0.5,
        key=experience_key,
    )
    support = int(
        pairs[(pairs.job_category == category) & (pairs.job_title == title)].iloc[0].support
    )
    st.success(
        f"Within observed DEV bounds: {bounds['min']:g}–{bounds['max']:g} years · title median {bounds['median']:g} · pair support {support}"
    )
    if st.button("Add validated scenario", type="primary", key="p6_add"):
        candidate = {
            "source": "manual",
            "job_category": category,
            "job_title": title,
            "years_of_experience": float(years),
        }
        try:
            validate_scenarios([{**candidate, "scenario_id": "preview"}], policy)
        except ScenarioValidationError as exc:
            st.error(str(exc))
        else:
            _append(st, candidate)
            st.rerun()

    st.subheader("3. Validation queue and actions")
    queue = st.session_state.p6_queue
    if not queue:
        st.info("The queue is empty. Add a validated scenario or quick-load a benchmark.")
    else:
        queue_frame = pd.DataFrame(queue)
        visible = [
            column
            for column in [
                "scenario_id",
                "source",
                "job_title",
                "job_category",
                "years_of_experience",
                "actual_salary_usd",
            ]
            if column in queue_frame
        ]
        queue_display = queue_frame[visible].rename(
            columns={
                "scenario_id": "Scenario ID",
                "source": "Source",
                "job_title": "Job title",
                "job_category": "Job category",
                "years_of_experience": "Experience (years)",
                "actual_salary_usd": "Known actual (USD)",
            }
        )
        st.dataframe(
            queue_display,
            hide_index=True,
            width="stretch",
            column_config={
                "Scenario ID": st.column_config.TextColumn(pinned=True),
                "Experience (years)": st.column_config.NumberColumn(format="%.1f"),
                "Known actual (USD)": st.column_config.NumberColumn(format="$%.0f"),
            },
        )
    with st.container(horizontal=True):
        run = st.button(
            "Run batch salary prediction", type="primary", disabled=not queue, key="p6_run"
        )
        clear = st.button("Clear queue", disabled=not queue, key="p6_clear")
    if clear:
        st.session_state.p6_queue = []
        st.session_state.p6_revision += 1
        _reset_results(st)
        st.rerun()
    if run:
        try:
            result = predict_scenarios(queue, bundle, metadata, policy)
        except (PredictionError, ScenarioValidationError) as exc:
            st.error(f"Prediction blocked: {exc}")
        else:
            st.session_state.p6_results = {
                "context": context,
                "revision": st.session_state.p6_revision,
                "rows": result,
            }

    snapshot = st.session_state.get("p6_results")
    if snapshot and (
        snapshot["context"] != context or snapshot["revision"] != st.session_state.p6_revision
    ):
        st.session_state.pop("p6_results", None)
        snapshot = None
    if snapshot:
        st.subheader("4. Visual prediction results")
        st.markdown(
            "**Question:** What salary range does this validated batch produce, how uncertain is the historical reference, and which rows have known actuals or extrapolation flags?"
        )
        results = pd.DataFrame(snapshot["rows"])
        page_count = max(1, math.ceil(len(results) / 10))
        page = st.number_input(
            "Prediction chart page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key="p6_result_page",
        )
        start = (int(page) - 1) * 10
        shown = results.iloc[start : start + 10]
        st.caption(f"Scenarios {start + 1}–{start + len(shown)} of {len(results)}")
        show_plot(st, prediction_interval_figure(shown), "p6_prediction_interval")
        render_conclusion(
            st,
            prediction_batch_conclusion(results, float(metadata["q90_abs_error_usd"])),
            title="Batch prediction conclusion",
        )
        st.warning(
            "Whiskers use a same-population historical absolute-error q90. They are not formal confidence intervals or salary guarantees; lower bounds are clipped at zero."
        )

        extend = st.checkbox("Extend growth curves to 0–15 years", key="p6_extend")
        acknowledged = False
        if extend:
            acknowledged = st.checkbox(
                "I understand extended points are outside observed DEV experience bounds and historical q90 coverage is unvalidated there.",
                key="p6_extend_ack",
            )
        if st.button(
            "Generate seniority growth view", disabled=extend and not acknowledged, key="p6_growth"
        ):
            try:
                points = build_growth_inputs(
                    snapshot["rows"], policy, extend=extend, acknowledgement=acknowledged
                )
                acknowledgements = {
                    point["scenario_id"]: policy["policy_hash"]
                    for point in points
                    if point["validation_mode"] == "experience_exception"
                }
                growth = predict_scenarios(
                    points,
                    bundle,
                    metadata,
                    policy,
                    allow_exceptions=extend,
                    acknowledgements=acknowledgements,
                )
                st.session_state.p6_growth_results = growth
            except (PredictionError, ScenarioValidationError) as exc:
                st.error(f"Growth view blocked: {exc}")
        growth_rows = st.session_state.get("p6_growth_results")
        if growth_rows:
            growth = pd.DataFrame(growth_rows)
            figure = go.Figure()
            grouped_curves = growth.groupby(
                ["job_category", "job_title", "validation_mode"], sort=True
            )
            for curve_index, ((category_name, title_name, mode), group) in enumerate(
                grouped_curves
            ):
                color = PREDICTION_SERIES_COLORS[curve_index % len(PREDICTION_SERIES_COLORS)]
                figure.add_scatter(
                    x=group.years_of_experience,
                    y=group.predicted_salary_usd,
                    mode="lines+markers",
                    line=dict(
                        color=color,
                        dash="dash" if mode == "experience_exception" else "solid",
                    ),
                    marker=dict(color=color),
                    name=f"{category_name} · {title_name} · {mode}",
                )
            figure.update_layout(
                title="Model scenario across years of experience",
                xaxis=dict(title="Years (validated points only)", automargin=True),
                yaxis=dict(
                    title="Predicted annual salary (USD)", tickformat=",.0f", automargin=True
                ),
                height=440,
                margin=dict(l=90, r=45, t=80, b=70),
            )
            with st.container(width=chart_container_width("P2")):
                show_plot(st, figure, "p6_growth_chart")
            st.caption(
                "This model response curve is not a causal or necessarily monotonic salary trajectory."
            )

        display_columns = [
            "scenario_id",
            "job_title",
            "job_category",
            "years_of_experience",
            "validation_mode",
            "lower_bound_usd",
            "predicted_salary_usd",
            "upper_bound_usd",
            "actual_salary_usd",
            "absolute_error_usd",
            "signed_variance_pct",
        ]
        result_display = results[display_columns].rename(
            columns={
                "scenario_id": "Scenario ID",
                "job_title": "Job title",
                "job_category": "Job category",
                "years_of_experience": "Experience (years)",
                "validation_mode": "Validation mode",
                "lower_bound_usd": "Lower reference (USD)",
                "predicted_salary_usd": "Prediction (USD)",
                "upper_bound_usd": "Upper reference (USD)",
                "actual_salary_usd": "Known actual (USD)",
                "absolute_error_usd": "Absolute error (USD)",
                "signed_variance_pct": "Signed variance (%)",
            }
        )
        st.dataframe(
            result_display,
            hide_index=True,
            width="stretch",
            column_config={
                "Scenario ID": st.column_config.TextColumn(pinned=True),
                "Experience (years)": st.column_config.NumberColumn(format="%.1f"),
                "Lower reference (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Prediction (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Upper reference (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Known actual (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Absolute error (USD)": st.column_config.NumberColumn(format="$%.0f"),
                "Signed variance (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.download_button(
            "Download prediction audit v1",
            build_prediction_audit_csv(snapshot["rows"]),
            file_name="salary_prediction_audit_v1.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "Download original enterprise-schema scenarios",
            build_source_schema_csv(queue),
            file_name="salary_scenarios_source_schema_v1.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(
            "The audit CSV uses signed variance. The original-schema file is an incomplete scenario interchange file; it does not place predictions into observed salary columns."
        )
