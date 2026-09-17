from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def read_csv(root: Path, rel: str) -> pd.DataFrame:
    return pd.read_csv(root / "outputs" / rel)


def read_json(root: Path, rel: str):
    with open(root / "outputs" / rel, "r", encoding="utf-8") as f:
        return json.load(f)


def read_artifact_json(root: Path, name: str):
    with open(root / "artifacts" / name, "r", encoding="utf-8") as f:
        return json.load(f)


def money(x: float) -> str:
    return f"${float(x):,.0f}"


def insight_box(st, text: str, kind: str = "info"):
    if not text:
        return
    getattr(st, kind if kind in {"info", "warning", "success", "error"} else "info")(text)


def style_page(st):
    st.markdown(
        """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1580px;}
    [data-testid="stMetric"] {
        background: rgba(125, 160, 210, 0.08) !important;
        border: 1px solid rgba(125, 160, 210, 0.22) !important;
        padding: 12px 14px !important;
        border-radius: 12px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        opacity: 0.88 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        word-break: break-word !important;
        white-space: normal !important;
    }
    .stage-note {background: rgba(88, 191, 163, 0.1); border-left: 5px solid #58bfa3; padding: 12px 16px; border-radius: 8px; margin: 8px 0 14px 0;}
    .small-note {background: rgba(246, 227, 178, 0.1); border: 1px solid rgba(246, 227, 178, 0.3); padding: 10px 14px; border-radius: 10px;}
    </style>
    """,
        unsafe_allow_html=True,
    )


def metric_cols(st, items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


def show_plot(st, fig, key=None):
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=25), hovermode="closest")
    st.plotly_chart(
        fig, use_container_width=True, key=key, config={"displaylogo": False, "responsive": True}
    )


def chart_selector(st, label, options, key, index=0):
    return st.selectbox(label, options, index=index, key=key)


def category_multiselect(st, df, col, label=None, key=None, default_all=True, max_default=12):
    if col not in df.columns:
        return []
    values = sorted(map(str, df[col].dropna().unique()))
    default = values if default_all and len(values) <= max_default else []
    return st.multiselect(label or col.replace("_", " ").title(), values, default=default, key=key)


def apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for c, vals in filters.items():
        if vals and c in out.columns:
            out = out[out[c].astype(str).isin(list(map(str, vals)))]
    return out


def downloadable_table(
    st, df: pd.DataFrame, title: str, key: str, file_name: str | None = None, height=350
):
    st.markdown(f"#### {title}")
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=file_name or f"{key}.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def ranked_bar(
    df, category, value, title, value_prefix="", value_suffix="", horizontal=True, top_n=None
):
    d = df.copy()
    if top_n is not None:
        d = d.nlargest(top_n, value)
    if horizontal:
        d = d.sort_values(value)
        fig = px.bar(d, y=category, x=value, orientation="h", text=value, title=title)
    else:
        fig = px.bar(d, x=category, y=value, text=value, title=title)
    if value_prefix or value_suffix:
        fig.update_traces(
            texttemplate=f"{value_prefix}%{{text:,.0f}}{value_suffix}", textposition="outside"
        )
    else:
        fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
    return fig


def dynamic_family_comment(assign: pd.DataFrame, family: str) -> str:
    if assign.empty:
        return "No data available for the current filters."
    if family == "Job Domain":
        g = (
            assign.groupby("cluster")["job_category"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        return (
            "Dominant job-domain labels by cluster: "
            + "; ".join(f"C{k}: {v}" for k, v in sorted(g.items()))
            + "."
        )
    if family == "Skills":
        s = assign.groupby("cluster")["skill_count"].mean().round(1).to_dict()
        return (
            "Average normalized skill count by cluster: "
            + "; ".join(f"C{k}: {v}" for k, v in sorted(s.items()))
            + "."
        )
    if family == "Experience":
        s = assign.groupby("cluster")["years_of_experience"].mean().round(1).to_dict()
        return (
            "Mean years of experience by cluster: "
            + "; ".join(f"C{k}: {v}" for k, v in sorted(s.items()))
            + "."
        )
    if family == "Company":
        g = (
            assign.groupby("cluster")["company_size"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        )
        return (
            "Most common company size by cluster: "
            + "; ".join(f"C{k}: {v}" for k, v in sorted(g.items()))
            + "."
        )
    if family == "Geography":
        g = assign.groupby("cluster")["country"].agg(lambda s: s.value_counts().index[0]).to_dict()
        return (
            "Most common country by cluster: "
            + "; ".join(f"C{k}: {v}" for k, v in sorted(g.items()))
            + "."
        )
    if family == "Demand / Benefits":
        g = assign.groupby("cluster")[["demand_score", "benefits_score_10"]].mean().round(1)
        return (
            "Demand/benefits means by cluster: "
            + "; ".join(
                f"C{i}: demand {r.demand_score}, benefits {r.benefits_score_10}"
                for i, r in g.iterrows()
            )
            + "."
        )
    return ""


def interpretation_card(
    st,
    observation: str,
    interpretation: str,
    action: str | None = None,
    tone: str = "info",
    title: str = "Data-driven interpretation",
):
    """Consistent narrative block placed under Streamlit evidence charts.

    Text passed to this helper is always computed from the currently loaded
    output tables / filtered dataframe.  It is intentionally separated into
    observation, interpretation and action so chart commentary remains
    scientific instead of becoming generic prose.
    """
    icon = {"info": "🔎", "success": "✅", "warning": "⚠️", "error": "⛔"}.get(tone, "🔎")
    bg = {"info": "#f4f9ff", "success": "#f2fbf5", "warning": "#fff9ed", "error": "#fff2f2"}.get(
        tone, "#f4f9ff"
    )
    border = {
        "info": "#9bc8f2",
        "success": "#9bd7ad",
        "warning": "#f3c96b",
        "error": "#ef9b9b",
    }.get(tone, "#9bc8f2")
    action_html = f"<div><b>Next action:</b> {action}</div>" if action else ""
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {border};border-left:5px solid {border};padding:12px 15px;border-radius:10px;margin:8px 0 16px 0;">
          <div style="font-weight:700;margin-bottom:6px;">{icon} {title}</div>
          <div><b>Observed:</b> {observation}</div>
          <div><b>Interpretation:</b> {interpretation}</div>
          {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_pct(numer: float, denom: float) -> float:
    return 0.0 if not denom else 100.0 * float(numer) / float(denom)


def strongest_category(df: pd.DataFrame, category: str, value: str, ascending: bool = False):
    if df.empty or category not in df or value not in df:
        return None
    d = df.dropna(subset=[category, value]).sort_values(value, ascending=ascending)
    return None if d.empty else d.iloc[0]
