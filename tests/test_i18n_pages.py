"""Feature-wide rendering and raw-data invariants for language migration."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("page", range(1, 9))
def test_every_page_changes_its_body_title_in_vietnamese(page):
    app = AppTest.from_file(str(ROOT / "streamlit.py"), default_timeout=60)
    app.session_state["auth_role"] = "admin"
    app.run()
    app.radio(key="workflow_page").set_value(f"page{page:02d}").run()
    assert not app.exception
    english_titles = [item.value for item in app.title]
    assert english_titles
    app.selectbox(key="app_language").set_value("VI").run()
    assert not app.exception
    assert [item.value for item in app.title] != english_titles
    assert app.radio(key="workflow_page").value == f"page{page:02d}"
    app.selectbox(key="app_language").set_value("EN").run()
    assert not app.exception
    assert [item.value for item in app.title] == english_titles


def test_display_figure_is_localized_without_mutating_source():
    from ai_job_market.i18n import Translator, load_catalog
    from components.presentation import translate_figure

    figure = go.Figure(go.Bar(x=[1, 2], y=[3, 4], name="Historical test"))
    figure.update_layout(title="Model comparison", xaxis_title="Years of experience")
    original = figure.to_json()
    translated = translate_figure(figure, Translator(load_catalog(), "VI"))
    assert translated.layout.title.text != figure.layout.title.text
    assert translated.layout.xaxis.title.text != figure.layout.xaxis.title.text
    assert list(translated.data[0].y) == [3, 4]
    assert figure.to_json() == original


def test_raw_fields_and_unknown_values_remain_unchanged():
    from ai_job_market.i18n import Translator, load_catalog

    raw = pd.DataFrame(
        {
            "years_of_experience": [4.5],
            "status": ["completed"],
            "user_note": ["Completed"],
            "job_title": ["My unique job"],
        }
    )
    snapshot = raw.copy(deep=True)
    localized = Translator(load_catalog(), "VI").frame(raw)
    assert "years_of_experience" not in localized.columns
    assert localized["user_note"].iloc[0] == "Completed"
    assert "My unique job" in localized.iloc[0].tolist()
    pd.testing.assert_frame_equal(raw, snapshot)
