from __future__ import annotations

import json

import pandas as pd
import plotly.express as px

from ai_job_market.core import SOURCE_COLUMNS

from .common import *


def render(st, root, role="admin"):
    style_page(st)
    st.title("8. Full Pipeline")
    st.caption(
        "End-to-end workflow status, provenance, pipeline diagrams, active-run audit and new-dataset processing contract"
    )
    status = read_csv(root, "08_full_pipeline/pipeline_status.csv")
    summary = read_json(root, "08_full_pipeline/run_summary.json")
    seg = summary["segmentation"]
    metric_cols(
        st,
        [
            ("Run ID", summary["run_id"]),
            ("Raw shape", f"{summary['raw_shape'][0]} × {summary['raw_shape'][1]}"),
            ("Basic-clean shape", f"{summary['clean_shape'][0]} × {summary['clean_shape'][1]}"),
            ("Segmentation", f"{seg['algorithm']} K={seg['k']}"),
            ("Best salary model", summary["best_model"]),
        ],
    )
    interpretation_card(
        st,
        f"Active run processes {summary['raw_shape'][0]:,} source records through Common Foundation, Branch A and Branch B. Segmentation selected {seg['algorithm']} K={seg['k']} with silhouette {seg['silhouette']:.3f}; salary model is {summary['best_model']}.",
        "All numbers on the dashboard are read from this active workspace. Switching to a processed upload changes charts, tables and commentary together.",
        "Use the sidebar Data source control to validate and run a new schema-compatible CSV in an isolated workspace.",
        "success",
    )

    tabs = st.tabs(
        [
            "Overall Pipeline",
            "Branch A · Segmentation",
            "Branch B · Prediction",
            "Stage Status & Artifacts",
            "New-dataset Processing",
            "Provenance",
        ]
    )
    with tabs[0]:
        img = root / "docs/pipelines/Pipeline_AI-salary-overall.png"
        if img.exists():
            st.image(
                str(img),
                caption="Integrated AI Job Market Segmentation + Salary Prediction Pipeline",
                width="stretch",
            )
        st.markdown("""
        **Presentation order used in this release**
        1. Data Basic Clean — Stage 1 to Stage 3  
        2. Data Ready for ML — Stage 4, Stage 5 and B1–B3  
        3. AI Job Market Segmentation — Branch A A1–A8  
        4. Model Comparison — Branch B B4  
        5. Best Model & Importance — Branch B B5–B6  
        6. Salary Prediction — Branch B B7 / Streamlit inference  
        7. Integrated Market Insight — post-hoc decision layer  
        8. Full Pipeline — audit, provenance and new-data gate
        """)
        interpretation_card(
            st,
            "The Streamlit order mirrors the technical design rather than the physical order of source files.",
            "This keeps data-quality evidence before feature governance, separates unsupervised and supervised branches, and only integrates insights after both branches finish.",
            "Follow the numbered pages in sequence when presenting the project.",
            "info",
        )

    with tabs[1]:
        img = root / "docs/pipelines/Pipeline_AI-salary-segmentation.png"
        if img.exists():
            st.image(str(img), caption="Branch A — AI Job Market Segmentation", width="stretch")
        r = read_json(root, "03_ai_job_market_segmentation/k_selection_rationale.json")
        interpretation_card(
            st,
            f"Official segmentation: {r['selected_algorithm']} K={r['selected_k']}, silhouette {r['selected_silhouette']:.3f}, stability ARI {r['selected_stability_ari']:.3f}, minimum cluster share {r['selected_min_cluster_share']:.1%}.",
            "K is selected from K=2…8 using stability/balance gates, silhouette as the primary metric, and a parsimonious tie-break. Salary is excluded from the clustering matrix.",
            "Open Page 3 → How K is Selected to inspect every candidate and alternative K without refitting.",
            "success",
        )

    with tabs[2]:
        img = root / "docs/pipelines/Pipeline_AI-salary-prediction.png"
        if img.exists():
            st.image(str(img), caption="Branch B — Salary Prediction", width="stretch")
        comp = read_csv(root, "04_model_comparison/model_comparison.csv").sort_values("MAE_mean")
        met = read_json(root, "05_best_model/locked_test_metrics.json")
        best = comp.iloc[0]
        interpretation_card(
            st,
            f"{best.model} wins expanding temporal CV at {money(best.MAE_mean)} mean MAE. Frozen locked-test performance is MAE {money(met['MAE'])}, RMSE {money(met['RMSE'])}, R² {met['R2']:.3f}.",
            "Branch B keeps preprocessing inside each fold, tunes only on DEV, opens the future test once, and serializes preprocessing+model together.",
            "Use model comparison, feature-family ablation, importance drift and subgroup error evidence together before interpreting the selected model.",
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
            title="Pipeline Stage Completion",
        )
        fig.update_yaxes(visible=False)
        fig.update_traces(textposition="inside")
        show_plot(st, fig, "p8_status")
        st.dataframe(status, width="stretch", hide_index=True)
        selected = st.selectbox(
            "Inspect stage artifact", status.stage.tolist(), key="p8_artifact_stage"
        )
        row = status.loc[status.stage == selected].iloc[0]
        st.code(str(row.artifact))
        failed = int((status.status != "PASS").sum())
        interpretation_card(
            st,
            f"{len(status) - failed}/{len(status)} workflow checkpoints are marked PASS in the active run.",
            "A stage is considered complete only when its evidence artifact exists; Streamlit charts consume those persisted outputs rather than fitting models inside analysis pages.",
            "If a stage artifact is missing or stale, rerun the full pipeline from the sidebar upload/process control or command line.",
            "success" if failed == 0 else "error",
        )

    with tabs[4]:
        st.markdown("### Upload → Schema Gate → Full Processing")
        st.info(
            "Use **Sidebar → Data source → Upload & process a new dataset**. The Process button is enabled only when the blocking schema checks pass."
        )
        schema = pd.DataFrame({"required_column": SOURCE_COLUMNS})
        st.dataframe(schema, width="stretch", hide_index=True, height=430)
        st.markdown("""
        **Blocking checks before Process is enabled**
        - all 25 raw source fields are present (`job_category` is accepted as an alias for raw `AI Engineering`);
        - salary target and time fields are parseable and non-missing;
        - `posting_month` is between 1 and 12;
        - at least three monthly periods exist for temporal validation.

        **Warnings that do not block processing**
        - extra columns (ignored by the canonical contract);
        - duplicate rows / IDs;
        - hidden-missing text;
        - small sample size.

        A valid upload is saved under `runs/<run_id>/data/raw/`, then the *same* `run_pipeline()` executes Common Foundation, Branch A, Branch B, bundle serialization and integrated outputs. Baseline outputs are never overwritten.
        """)
        interpretation_card(
            st,
            "New-data processing is isolated by run ID; the dashboard then switches its evidence root to that run.",
            "This preserves provenance and prevents one exploratory upload from silently replacing the baseline release.",
            "After processing, review Page 1 onward because every data-dependent conclusion—including selected K and best salary model—may change.",
            "info",
        )

    with tabs[5]:
        st.markdown("### Reproducibility & provenance")
        st.json(
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
        st.markdown("### Scientific-use boundary")
        st.warning(
            "The source dataset may contain strong logical/synthetic artifacts. This project demonstrates data-quality-first ML systems design, segmentation and temporal regression; it does not establish causal real-world salary economics."
        )
        interpretation_card(
            st,
            f"Run source SHA-256 starts with {summary['source_sha256'][:12]}… and was created {summary['created_utc']}.",
            "The hash and run workspace make it possible to trace dashboard evidence back to the exact input file and environment version.",
            "Keep the run summary and artifact bundle together when sharing or grading the project.",
            "info",
        )
