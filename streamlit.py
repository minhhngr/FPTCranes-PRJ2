from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from components.auth import require_login
from components.data_source import render_data_source_control
from components.language import get_translator, render_language_selector
from pages import (
    page01_data_basic_clean,
    page02_data_ready,
    page03_segmentation,
    page04_model_comparison,
    page05_best_model,
    page06_prediction,
    page07_integrated,
    page08_full_pipeline,
)

i18n = get_translator(st)
st.set_page_config(
    page_title=i18n.text("app.browser_title"),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
with st.sidebar:
    i18n = render_language_selector(st)
role = require_login(st)

ADMIN_PAGES = {
    "page01": page01_data_basic_clean,
    "page02": page02_data_ready,
    "page03": page03_segmentation,
    "page04": page04_model_comparison,
    "page05": page05_best_model,
    "page06": page06_prediction,
    "page07": page07_integrated,
    "page08": page08_full_pipeline,
}
USER_PAGES = {"page06": page06_prediction}
pages = ADMIN_PAGES if role == "admin" else USER_PAGES

with st.sidebar:
    st.markdown(i18n.text("app.title"))
    st.caption(i18n.text("app.caption"))
    ACTIVE_ROOT = render_data_source_control(st, ROOT, role)
    st.divider()
    if st.session_state.get("workflow_page") not in pages:
        st.session_state["workflow_page"] = next(iter(pages))
    selected = st.radio(
        i18n.text("navigation.workflow"),
        list(pages),
        format_func=lambda page: i18n.text(f"navigation.{page}"),
        key="workflow_page",
    )
    st.caption(i18n.text("app.active_evidence", name=ACTIVE_ROOT.name))

pages[selected].render(st, ACTIVE_ROOT, role)
