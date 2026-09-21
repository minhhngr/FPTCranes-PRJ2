from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def open_app(role=None):
    app = AppTest.from_file(str(ROOT / "streamlit.py"), default_timeout=40)
    if role:
        app.session_state["auth_role"] = role
    return app.run()


def test_login_switches_language_before_authentication():
    app = open_app()
    assert not app.exception
    assert app.text_input[0].label == "Username"
    app.selectbox(key="app_language").set_value("VI").run()
    assert not app.exception
    assert app.text_input[0].label == "Tên đăng nhập"
    assert app.text_input[1].label == "Mật khẩu"
    app.text_input[0].set_value("invalid")
    app.text_input[1].set_value("invalid")
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Thông tin đăng nhập minh họa không hợp lệ."


def test_navigation_identity_survives_language_switch_without_processing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Language switching must not run the pipeline")

    monkeypatch.setattr("components.data_source.run_pipeline", forbidden)
    app = open_app("admin")
    assert not app.exception
    app.radio(key="workflow_page").set_value("page08").run()
    assert not app.exception
    workspace = app.session_state["analysis_root"]
    app.selectbox(key="app_language").set_value("VI").run()
    assert not app.exception
    assert app.radio(key="workflow_page").value == "page08"
    assert app.radio(key="workflow_page").label == "Quy trình"
    assert app.selectbox(key="workspace_selector").options[0] == "Bản phát hành gốc"
    assert app.session_state["analysis_root"] == workspace
    app.radio(key="workflow_page").set_value("page01").run()
    assert not app.exception
    assert app.selectbox(key="app_language").value == "VI"
    app.selectbox(key="app_language").set_value("EN").run()
    assert not app.exception
    assert app.radio(key="workflow_page").value == "page01"
    assert app.radio(key="workflow_page").label == "Workflow"


@pytest.mark.parametrize("language", ["EN", "VI"])
def test_standard_user_navigation_stays_restricted(language):
    app = AppTest.from_file(str(ROOT / "streamlit.py"), default_timeout=40)
    app.session_state["auth_role"] = "user"
    app.session_state["app_language"] = language
    app.run()
    assert not app.exception
    assert len(app.radio(key="workflow_page").options) == 1
    assert app.radio(key="workflow_page").value == "page06"


def test_common_table_and_interpretation_are_localized_without_mutating_data():
    def script():
        import pandas as pd
        import streamlit as st

        from pages.common import downloadable_table, interpretation_card

        st.session_state["app_language"] = "VI"
        raw = pd.DataFrame({"status": ["completed"], "user_text": ["completed"]})
        downloadable_table(st, raw, "", "sample")
        interpretation_card(st, "123", "456", "789")
        assert raw.status.tolist() == ["completed"]

    app = AppTest.from_function(script).run()
    assert not app.exception
    assert app.dataframe[0].value["Trạng thái"].tolist() == ["Hoàn tất"]
    assert app.dataframe[0].value["user_text"].tolist() == ["completed"]
    assert app.get("download_button")[0].label == "Tải xuống CSV"
    assert "Quan sát:" in app.markdown[-1].value
    assert "Hành động tiếp theo:" in app.markdown[-1].value


def test_language_is_session_local_and_invalid_state_is_reset():
    vi = open_app()
    vi.selectbox(key="app_language").set_value("VI").run()
    en = open_app()
    assert en.text_input[0].label == "Username"
    assert vi.text_input[0].label == "Tên đăng nhập"
    # Seed invalid server-side state before widget serialization by AppTest.
    invalid = AppTest.from_file(str(ROOT / "streamlit.py"))
    invalid.session_state["app_language"] = {"invalid": True}
    invalid.run()
    assert not invalid.exception
    assert invalid.selectbox(key="app_language").value == "EN"
