from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from components.auth import require_login
from components.data_source import render_data_source_control
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

st.set_page_config(
    page_title="AI Job Market | Segmentation + Salary Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
role = require_login(st)

ADMIN_PAGES = {
    "1. Data Basic Clean": page01_data_basic_clean,
    "2. Data Ready for ML": page02_data_ready,
    "3. AI Job Market Segmentation": page03_segmentation,
    "4. Model Comparison": page04_model_comparison,
    "5. Best Model & Importance": page05_best_model,
    "6. Salary Prediction": page06_prediction,
    "7. Integrated Market Insight": page07_integrated,
    "8. Full Pipeline": page08_full_pipeline,
}
USER_PAGES = {"6. Salary Prediction": page06_prediction}
pages = ADMIN_PAGES if role == "admin" else USER_PAGES

with st.sidebar:
    st.markdown("## AI Job Market 2025–2026")
    st.caption(
        "Outputs-first dashboard. New-data processing calls the same isolated offline orchestrator used by the release pipeline."
    )
    ACTIVE_ROOT = render_data_source_control(st, ROOT, role)
    st.divider()
    selected = st.radio("Workflow", list(pages.keys()), index=0)
    st.caption(f"Active evidence: `{ACTIVE_ROOT.name}`")

pages[selected].render(st, ACTIVE_ROOT, role)
