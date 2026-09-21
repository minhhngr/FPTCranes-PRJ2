from __future__ import annotations

import json

import pandas as pd
import plotly.express as px

from ai_job_market.core import SOURCE_COLUMNS
from components.presentation import (
    _display,
    _label,
    _option_label,
    _src,
    _tr,
    display_column_config,
    display_frame,
    display_record,
    translate_figure,
)

from .common import *


def render(st, root, role="admin"):
    style_page(st)
    st.title(_tr("navigation.page08"))
    st.caption(_tr("page08_full_pipeline.end_to_end_workflow_status_provenance_pipeline_e420cb0"))
    status = read_csv(root, "08_full_pipeline/pipeline_status.csv")
    summary = read_json(root, "08_full_pipeline/run_summary.json")
    seg = summary["segmentation"]
    metric_cols(
        st,
        [
            (_tr("page08_full_pipeline.run_id_26d3e7a"), summary["run_id"]),
            (
                _tr("page08_full_pipeline.raw_shape_7405341"),
                f"{summary['raw_shape'][0]} × {summary['raw_shape'][1]}",
            ),
            (
                _tr("page08_full_pipeline.basic_clean_shape_5bab4a6"),
                f"{summary['clean_shape'][0]} × {summary['clean_shape'][1]}",
            ),
            (_tr("page08_full_pipeline.segmentation_35cdf79"), f"{seg['algorithm']} K={seg['k']}"),
            (_tr("page08_full_pipeline.best_salary_model_a563048"), summary["best_model"]),
        ],
    )
    interpretation_card(
        st,
        _tr(
            "page08_full_pipeline.active_run_processes_value0_source_records_through_080d94f",
            value0=f"{summary['raw_shape'][0]:,}",
            value1=f"{seg['algorithm']}",
            value2=f"{seg['k']}",
            value3=f"{seg['silhouette']:.3f}",
            value4=f"{summary['best_model']}",
        ),
        _tr("page08_full_pipeline.all_numbers_on_the_dashboard_are_read_76b4ab5"),
        _tr("page08_full_pipeline.use_the_sidebar_data_source_control_to_9361a09"),
        "success",
    )

    tabs = st.tabs(
        [
            _tr("page08_full_pipeline.overall_pipeline_2c11f13"),
            _tr("page08_full_pipeline.branch_a_segmentation_0e8d77d"),
            _tr("page08_full_pipeline.branch_b_prediction_0039560"),
            _tr("page08_full_pipeline.stage_status_artifacts_6bc7c53"),
            _tr("page08_full_pipeline.new_dataset_processing_9a57a78"),
            _tr("page08_full_pipeline.provenance_1c4b8ff"),
        ]
    )
    with tabs[0]:
        img = root / "docs/pipelines/Pipeline_AI-salary-overall.png"
        if img.exists():
            st.image(
                str(img),
                caption=_tr(
                    "page08_full_pipeline.integrated_ai_job_market_segmentation_salary_prediction_4098f73"
                ),
                width="stretch",
            )
        st.markdown(_tr("page08_full_pipeline.presentation_order_used_in_this_release_1_cb72090"))
        interpretation_card(
            st,
            _tr("page08_full_pipeline.the_streamlit_order_mirrors_the_technical_design_60c6d53"),
            _tr("page08_full_pipeline.this_keeps_data_quality_evidence_before_feature_92e728e"),
            _tr("page08_full_pipeline.follow_the_numbered_pages_in_sequence_when_bd0861e"),
            "info",
        )

    with tabs[1]:
        img = root / "docs/pipelines/Pipeline_AI-salary-segmentation.png"
        if img.exists():
            st.image(
                str(img),
                caption=_tr("page08_full_pipeline.branch_a_ai_job_market_segmentation_cf68c3d"),
                width="stretch",
            )
        r = read_json(root, "03_ai_job_market_segmentation/k_selection_rationale.json")
        interpretation_card(
            st,
            _tr(
                "page08_full_pipeline.official_segmentation_value0_k_value1_silhouette_value2_ea2b35b",
                value0=f"{r['selected_algorithm']}",
                value1=f"{r['selected_k']}",
                value2=f"{r['selected_silhouette']:.3f}",
                value3=f"{r['selected_stability_ari']:.3f}",
                value4=f"{r['selected_min_cluster_share']:.1%}",
            ),
            _tr("page08_full_pipeline.k_is_selected_from_k_2_8_acd37ca"),
            _tr("page08_full_pipeline.open_page_3_how_k_is_selected_fd9efa1"),
            "success",
        )

    with tabs[2]:
        img = root / "docs/pipelines/Pipeline_AI-salary-prediction.png"
        if img.exists():
            st.image(
                str(img),
                caption=_tr("page08_full_pipeline.branch_b_salary_prediction_a6f874f"),
                width="stretch",
            )
        comp = read_csv(root, "04_model_comparison/model_comparison.csv").sort_values("MAE_mean")
        met = read_json(root, "05_best_model/locked_test_metrics.json")
        best = comp.iloc[0]
        interpretation_card(
            st,
            _tr(
                "page08_full_pipeline.value0_wins_expanding_temporal_cv_at_value1_6759359",
                value0=f"{best.model}",
                value1=f"{money(best.MAE_mean)}",
                value2=f"{money(met['MAE'])}",
                value3=f"{money(met['RMSE'])}",
                value4=f"{met['R2']:.3f}",
            ),
            _tr("page08_full_pipeline.branch_b_keeps_preprocessing_inside_each_fold_41b41e8"),
            _tr(
                "page08_full_pipeline.use_model_comparison_feature_family_ablation_importance_6097804"
            ),
            "warning",
        )

    with tabs[3]:
        fig = px.bar(
            status,
            x="order",
            y=[1] * len(status),
            color="status",
            hover_data=["stage", "artifact"],
            text="stage",
            title=_src("page08_full_pipeline.pipeline_stage_completion_05a87c2"),
        )
        fig.update_yaxes(visible=False)
        fig.update_traces(textposition="inside")
        show_plot(st, fig, "p8_status")
        st.dataframe(display_frame(status), width="stretch", hide_index=True)
        selected = st.selectbox(
            _tr("page08_full_pipeline.inspect_stage_artifact_4e86e56"),
            status.stage.tolist(),
            key="p8_artifact_stage",
            format_func=_option_label(),
        )
        row = status.loc[status.stage == selected].iloc[0]
        st.code(str(row.artifact))
        failed = int((status.status != "PASS").sum())
        interpretation_card(
            st,
            _tr(
                "page08_full_pipeline.value0_value1_workflow_checkpoints_are_marked_pass_f406fdb",
                value0=f"{len(status) - failed}",
                value1=f"{len(status)}",
            ),
            _tr("page08_full_pipeline.a_stage_is_considered_complete_only_when_af593f0"),
            _tr("page08_full_pipeline.if_a_stage_artifact_is_missing_or_bd6b09d"),
            "success" if failed == 0 else "error",
        )

    with tabs[4]:
        st.markdown(_tr("page08_full_pipeline.upload_schema_gate_full_processing_77e352e"))
        st.info(_tr("page08_full_pipeline.use_sidebar_data_source_upload_process_a_d910511"))
        schema = pd.DataFrame({"required_column": SOURCE_COLUMNS})
        st.dataframe(display_frame(schema), width="stretch", hide_index=True, height=430)
        st.markdown(
            _tr("page08_full_pipeline.blocking_checks_before_process_is_enabled_all_15dbb43")
        )
        interpretation_card(
            st,
            _tr("page08_full_pipeline.new_data_processing_is_isolated_by_run_b70fd20"),
            _tr(
                "page08_full_pipeline.this_preserves_provenance_and_prevents_one_exploratory_66d48aa"
            ),
            _tr("page08_full_pipeline.after_processing_review_page_1_onward_because_2970263"),
            "info",
        )

    with tabs[5]:
        st.markdown(_tr("page08_full_pipeline.reproducibility_provenance_f688f8d"))
        st.json(
            display_record(
                {
                    k: summary.get(k)
                    for k in [
                        "source_file",
                        "source_sha256",
                        "workspace_root",
                        "created_utc",
                        "python",
                        "pandas",
                        "numpy",
                        "scikit_learn",
                    ]
                }
            )
        )
        st.markdown(_tr("page08_full_pipeline.scientific_use_boundary_4cfee43"))
        st.warning(
            _tr("page08_full_pipeline.the_source_dataset_may_contain_strong_logical_4c68b7c")
        )
        interpretation_card(
            st,
            _tr(
                "page08_full_pipeline.run_source_sha_256_starts_with_value0_515c4a5",
                value0=f"{summary['source_sha256'][:12]}",
                value1=f"{summary['created_utc']}",
            ),
            _tr("page08_full_pipeline.the_hash_and_run_workspace_make_it_247b8c9"),
            _tr("page08_full_pipeline.keep_the_run_summary_and_artifact_bundle_2bb8e6f"),
            "info",
        )
