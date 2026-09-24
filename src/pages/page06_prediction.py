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
        st.error(_tr("page06_prediction.the_queue_limit_is_100_scenarios_c4e193e"))
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
            st.metric(_display(label), _display(value), border=True)


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("page06_prediction.6_controlled_salary_scenarios_06ec7a8"))
    st.caption(_tr("page06_prediction.fixed_top_2_random_forest_observed_dev_9bccb03"))
    try:
        manifest, tables = load_evidence(root)
        policy = load_json_file(root, manifest, "scenario_policy")
        metadata = load_json_file(root, manifest, "top2_metadata")
        bundle = load_top2_bundle(root, manifest)
    except (EvidenceContractError, KeyError, OSError, ValueError) as exc:
        st.error(
            _tr(
                "page06_prediction.top_2_serving_evidence_is_unavailable_or_c6a80b8",
                value0=f"{exc}",
            )
        )
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
        st.info(_tr("page06_prediction.scenario_state_was_reset_because_the_active_a0caaac"))

    test_metrics = metadata["historical_test_metrics"]
    render_page_brief(
        st,
        question=_tr("page06_prediction.how_do_validated_job_category_and_experience_bfd87d1"),
        evidence_scope=_tr(
            "page06_prediction.inputs_value0_historical_reference_value1_rows_value2_b08fba4",
            value0=f"{', '.join(metadata['feature_order'])}",
            value1=f"{int(manifest['datasets']['historical_test']['rows'])}",
            value2=f"{manifest['evidence_id']}",
        ),
        takeaway=_tr(
            "page06_prediction.the_fixed_two_input_model_records_historical_f214e18",
            value0=f"{test_metrics['R2']:.3f}",
            value1=f"{money(metadata['q90_abs_error_usd'])}",
        ),
        limitation=_tr("page06_prediction.this_is_an_academic_scenario_comparison_not_ecddbdd"),
    )
    _metrics(
        st,
        [
            (_tr("model_training_presentation.model_5e2c614"), metadata["model_name"]),
            (_tr("page06_prediction.active_inputs_53c7ec5"), f"{len(metadata['feature_order'])}"),
            (_tr("page05_best_model.historical_test_r_ee278ef"), f"{test_metrics['R2']:.3f}"),
            (
                _tr("page05_best_model.empirical_q90_382dadb"),
                f"±{money(metadata['q90_abs_error_usd'])}",
            ),
        ],
    )
    st.caption(
        _tr(
            "page06_prediction.evidence_value0_fixed_feature_ablation_not_independently_9d74e45",
            value0=f"{manifest['evidence_id']}",
        )
    )
    render_metric_glossary(st, ["R²", "q90"])

    st.subheader(_tr("page06_prediction.1_quick_load_historical_benchmarks_7fe9932"))
    examples = tables.get("benchmark_examples", pd.DataFrame())
    if examples.empty:
        st.warning(
            _tr("page06_prediction.no_strict_policy_historical_benchmark_examples_are_cfe38d1")
        )
    else:
        with st.container(horizontal=True):
            for index, example in examples.head(3).iterrows():
                label = _tr(
                    "page06_prediction.value0_value1_y_actual_value2_b4041bf",
                    value0=f"{example.job_title}",
                    value1=f"{example.years_of_experience:g}",
                    value2=f"{money(example.annual_salary_usd)}",
                )
                if st.button(_display(label), key=f"p6_benchmark_{index}"):
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
                        st.error(_display(str(exc)))
                    else:
                        _append(st, row)
                        st.rerun()
        st.caption(
            _tr("page06_prediction.examples_were_already_included_in_historical_test_148cea8")
        )

    st.subheader(_tr("page06_prediction.2_controlled_scenario_builder_0987fb6"))
    pairs = pd.DataFrame(policy["pairs"])
    categories = sorted(pairs.job_category.unique())
    category = st.selectbox(
        _tr("page01_data_basic_clean.job_category_0175eef"),
        categories,
        key="p6_category",
        format_func=_option_label(),
    )
    titles = sorted(pairs.loc[pairs.job_category == category, "job_title"].unique())
    if st.session_state.get("p6_title") not in titles:
        st.session_state.p6_title = titles[0]
    title = st.selectbox(
        _tr("page06_prediction.job_title_observed_dev_association_5f9e87d"),
        titles,
        key="p6_title",
        format_func=_option_label(),
    )
    bounds = policy["titles"][title]
    experience_key = f"p6_experience_{title}"
    if experience_key not in st.session_state:
        st.session_state[experience_key] = float(bounds["median"])
    years = st.number_input(
        _tr("page01_data_basic_clean.years_of_experience_29eb74e"),
        min_value=0.0,
        max_value=float(bounds["max"]),
        step=0.5,
        key=experience_key,
    )
    support = int(
        pairs[(pairs.job_category == category) & (pairs.job_title == title)].iloc[0].support
    )
    st.success(
        _tr(
            "page06_prediction.within_observed_dev_bounds_value0_value1_years_ba22ff3",
            value0=f"{bounds['min']:g}",
            value1=f"{bounds['max']:g}",
            value2=f"{bounds['median']:g}",
            value3=f"{support}",
        )
    )
    if st.button(
        _tr("page06_prediction.add_validated_scenario_3125e46"),
        type="primary",
        key="p6_add",
    ):
        candidate = {
            "source": "manual",
            "job_category": category,
            "job_title": title,
            "years_of_experience": float(years),
        }
        try:
            validate_scenarios([{**candidate, "scenario_id": "preview"}], policy)
        except ScenarioValidationError as exc:
            st.error(_display(str(exc)))
        else:
            _append(st, candidate)
            st.rerun()

    st.subheader(_tr("page06_prediction.3_validation_queue_and_actions_c94fd72"))
    queue = st.session_state.p6_queue
    if not queue:
        st.info(_tr("page06_prediction.the_queue_is_empty_add_a_validated_6cbabd8"))
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
                "scenario_id": _src("model_evidence.scenario_id_a8571ee"),
                "source": _src("page06_prediction.source_0e570ca"),
                "job_title": _src("page05_best_model.job_title_86db80a"),
                "job_category": _src("page01_data_basic_clean.job_category_0175eef"),
                "years_of_experience": _src("page05_best_model.experience_years_3453141"),
                "actual_salary_usd": _src("page06_prediction.known_actual_usd_c5cb27d"),
            }
        )
        st.dataframe(
            display_frame(queue_display),
            hide_index=True,
            width="stretch",
            column_config=display_column_config(
                {
                    _src("model_evidence.scenario_id_a8571ee"): st.column_config.TextColumn(
                        pinned=True
                    ),
                    _src(
                        "page05_best_model.experience_years_3453141"
                    ): st.column_config.NumberColumn(format="%.1f"),
                    _src(
                        "page06_prediction.known_actual_usd_c5cb27d"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                }
            ),
        )
    with st.container(horizontal=True):
        run = st.button(
            _tr("page06_prediction.run_batch_salary_prediction_64bfe5b"),
            type="primary",
            disabled=not queue,
            key="p6_run",
        )
        clear = st.button(
            _tr("page06_prediction.clear_queue_fd1d3b4"),
            disabled=not queue,
            key="p6_clear",
        )
    if clear:
        st.session_state.p6_queue = []
        st.session_state.p6_revision += 1
        _reset_results(st)
        st.rerun()
    if run:
        try:
            result = predict_scenarios(queue, bundle, metadata, policy)
        except (PredictionError, ScenarioValidationError) as exc:
            st.error(_tr("page06_prediction.prediction_blocked_value0_998d0dc", value0=f"{exc}"))
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
        st.subheader(_tr("page06_prediction.4_visual_prediction_results_2b3f873"))
        st.markdown(_tr("page06_prediction.question_what_salary_range_does_this_validated_fafe1a0"))
        results = pd.DataFrame(snapshot["rows"])
        page_count = max(1, math.ceil(len(results) / 10))
        page = (
            st.number_input(
                _tr("page06_prediction.prediction_chart_page_87c1d11"),
                min_value=1,
                max_value=page_count,
                value=1,
                step=1,
                key="p6_result_page",
            )
            if page_count > 1
            else 1
        )
        start = (int(page) - 1) * 10
        shown = results.iloc[start : start + 10]
        st.caption(
            _tr(
                "page06_prediction.scenarios_value0_value1_of_value2_6f00f54",
                value0=f"{start + 1}",
                value1=f"{start + len(shown)}",
                value2=f"{len(results)}",
            )
        )
        show_plot(st, prediction_interval_figure(shown), "p6_prediction_interval")
        render_conclusion(
            st,
            prediction_batch_conclusion(results, float(metadata["q90_abs_error_usd"])),
            title=_tr("page06_prediction.batch_prediction_conclusion_ac2a9d1"),
        )
        st.warning(
            _tr("page06_prediction.whiskers_use_a_same_population_historical_absolute_808f242")
        )

        extend = st.checkbox(
            _tr("page06_prediction.extend_growth_curves_to_0_15_years_4906144"),
            key="p6_extend",
        )
        acknowledged = False
        if extend:
            acknowledged = st.checkbox(
                _tr("page06_prediction.i_understand_extended_points_are_outside_observed_08318b5"),
                key="p6_extend_ack",
            )
        if st.button(
            _tr("page06_prediction.generate_seniority_growth_view_4398915"),
            disabled=extend and not acknowledged,
            key="p6_growth",
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
                st.error(
                    _tr("page06_prediction.growth_view_blocked_value0_d64382b", value0=f"{exc}")
                )
        growth_rows = st.session_state.get("p6_growth_results")
        if growth_rows:
            growth = pd.DataFrame(growth_rows)
            titles = sorted(growth["job_title"].dropna().unique())
            selected_titles = st.multiselect(
                _tr("page06_prediction.job_title_observed_dev_association_5f9e87d"),
                options=titles,
                default=titles,
                key="p6_growth_filter",
                format_func=_option_label(),
            )
            filtered_growth = growth[growth["job_title"].isin(selected_titles)]
            figure = go.Figure()
            grouped_curves = filtered_growth.groupby(
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
                title=_src("page06_prediction.model_scenario_across_years_of_experience_2c2a1bb"),
                xaxis=dict(
                    title=_src("page06_prediction.years_validated_points_only_06657b0"),
                    automargin=True,
                ),
                yaxis=dict(
                    title=_src("page06_prediction.predicted_annual_salary_usd_1c31401"),
                    tickformat=",.0f",
                    automargin=True,
                ),
                height=440,
                margin=dict(l=90, r=45, t=80, b=70),
            )
            with st.container(width=chart_container_width("P2")):
                show_plot(st, figure, "p6_growth_chart")
            st.caption(_tr("page06_prediction.this_model_response_curve_is_not_a_699b738"))

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
                "scenario_id": _src("model_evidence.scenario_id_a8571ee"),
                "job_title": _src("page05_best_model.job_title_86db80a"),
                "job_category": _src("page01_data_basic_clean.job_category_0175eef"),
                "years_of_experience": _src("page05_best_model.experience_years_3453141"),
                "validation_mode": _src("page06_prediction.validation_mode_2c5b802"),
                "lower_bound_usd": _src("page06_prediction.lower_reference_usd_05fba1a"),
                "predicted_salary_usd": _src("page06_prediction.prediction_usd_479e548"),
                "upper_bound_usd": _src("page06_prediction.upper_reference_usd_facbf2f"),
                "actual_salary_usd": _src("page06_prediction.known_actual_usd_c5cb27d"),
                "absolute_error_usd": _src("page05_best_model.absolute_error_usd_c68c45a"),
                "signed_variance_pct": _src("page06_prediction.signed_variance_8cb848c"),
            }
        )
        st.dataframe(
            display_frame(result_display),
            hide_index=True,
            width="stretch",
            column_config=display_column_config(
                {
                    _src("model_evidence.scenario_id_a8571ee"): st.column_config.TextColumn(
                        pinned=True
                    ),
                    _src(
                        "page05_best_model.experience_years_3453141"
                    ): st.column_config.NumberColumn(format="%.1f"),
                    _src(
                        "page06_prediction.lower_reference_usd_05fba1a"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src("page06_prediction.prediction_usd_479e548"): st.column_config.NumberColumn(
                        format="$%.0f"
                    ),
                    _src(
                        "page06_prediction.upper_reference_usd_facbf2f"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page06_prediction.known_actual_usd_c5cb27d"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page05_best_model.absolute_error_usd_c68c45a"
                    ): st.column_config.NumberColumn(format="$%.0f"),
                    _src(
                        "page06_prediction.signed_variance_8cb848c"
                    ): st.column_config.NumberColumn(format="%.1f%%"),
                }
            ),
        )
        st.download_button(
            _tr("page06_prediction.download_prediction_audit_v1_e1d1fcf"),
            build_prediction_audit_csv(snapshot["rows"]),
            file_name="salary_prediction_audit_v1.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            _tr("page06_prediction.download_original_enterprise_schema_scenarios_ec55f27"),
            build_source_schema_csv(queue),
            file_name="salary_scenarios_source_schema_v1.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(_tr("page06_prediction.the_audit_csv_uses_signed_variance_the_03bf7ec"))
