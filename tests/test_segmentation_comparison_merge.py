"""Regression coverage for the merged, localized R0–R4 overview."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("language", ["EN", "VI"])
@pytest.mark.parametrize("persisted", [False, True])
def test_overview_renders_without_changing_evidence(language, persisted):
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
from pages.page03_segmentation import _render_representation_overview
rep = pd.DataFrame([{
    "representation_id": "R0_GLOBAL_PCA", "representation_label": "Global PCA",
    "silhouette": .4, "stability_ari": .9, "resample_stability_ari_mean": .8,
    "min_cluster_share": .2, "mean_cross_representation_ari": .7,
    "latent_dimensions": 3, "eligible": True,
    "within_representation_tolerance": True, "selected_representation": True,
}])
comparison = pd.DataFrame({"R": ["R0"], "Silhouette": [.123], "Eligible": ["PASS"],
                           "Selected": ["★ SELECTED"]}) if st.session_state["persisted"] else pd.DataFrame()
snapshot = rep.copy(deep=True)
_render_representation_overview(st, rep, comparison, {}, "R0_GLOBAL_PCA")
pd.testing.assert_frame_equal(rep, snapshot)
''')
    app.session_state["app_language"] = language
    app.session_state["persisted"] = persisted
    app.run()
    assert not app.exception
    assert len(app.metric) == 1
    assert len(app.get("plotly_chart")) == 2
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.iloc[0, 1] == (.123 if persisted else .4)
    text = "\n".join(item.value for item in app.markdown)
    assert ("official O1 robustness decision" in text) == (language == "EN")
    from ai_job_market.i18n import Translator, load_catalog
    translator = Translator(load_catalog(), language)
    assert translator.text("page03_segmentation.overview_selected") in [
        item.value for item in app.success
    ]
    assert translator.text("status.pass") in app.dataframe[0].value.iloc[0].tolist()


@pytest.mark.parametrize("empty", [True, False])
def test_overview_handles_missing_evidence(empty):
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
from pages.page03_segmentation import _render_representation_overview
rep = pd.DataFrame() if st.session_state["empty"] else pd.DataFrame({
    "representation_id": ["R0_GLOBAL_PCA"]
})
_render_representation_overview(st, rep, pd.DataFrame(), {}, "R0_GLOBAL_PCA")
''')
    app.session_state["empty"] = empty
    app.run()
    assert not app.exception
    if empty:
        assert app.warning
    else:
        assert len(app.info) == 2


def test_pipeline_returns_comparison_matching_canonical_summary():
    from ai_job_market.core import run_segmentation
    from threadpoolctl import threadpool_limits

    data = pd.read_csv(ROOT / "outputs/02_data_ready_for_ml/development_raw.csv").sample(
        n=80, random_state=42
    )
    with threadpool_limits(limits=1):
        _, tables, _ = run_segmentation(
            data, data, k_min=2, k_max=2, stability_seeds=[42], resample_n=2,
        )
    summary = tables["representation_summary"].set_index("representation_id")
    comparison = tables["representation_comparison"].set_index("representation_id")
    assert set(comparison["R"]) == {"R0", "R1", "R2", "R3", "R4"}
    pd.testing.assert_series_equal(
        comparison["Silhouette"], summary["silhouette"], check_names=False,
    )
    assert comparison["Selected"].eq("★ SELECTED").tolist() == summary["selected_representation"].tolist()
    assert comparison["Eligible"].eq("PASS").tolist() == summary["eligible"].tolist()
    assert comparison["Near-best separation"].eq("PASS").tolist() == summary["within_representation_tolerance"].tolist()


def test_comparison_is_optional_evidence():
    from pages.page03_segmentation import EVIDENCE_NAMES
    assert "representation_comparison" in EVIDENCE_NAMES
