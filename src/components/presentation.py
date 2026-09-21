"""Explicit localization at presentation boundaries; analytical inputs stay raw.

``_src`` supplies canonical labels used as selectors/table keys. ``_tr`` renders
messages for the current session. Do not store a translated value in module state.
These helpers never translate arbitrary user table cells: those require a field
mapping in the central catalog.
"""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go

from components.language import get_translator


def _current():
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is None:
            return get_translator(SimpleNamespace(session_state={}))
        return get_translator(st)
    except (AttributeError, ImportError):
        # CLI / batch context — no Streamlit runtime available.
        return get_translator(SimpleNamespace(session_state={}))


def _src(key: str, **values) -> str:
    """Canonical EN labels for programmatic comparisons, independent of locale."""
    catalog = get_translator(SimpleNamespace(session_state={})).catalog
    from ai_job_market.i18n import Translator

    return Translator(catalog, "EN").text(key, **values)


def _tr(key: str, **values) -> str:
    """Resolve a complete keyed message, with already formatted named values."""
    return _current().text(key, **values)


def _label(value, translator=None):
    """Known application labels only; unknown names and non-strings pass through."""
    translator = translator or _current()
    if not isinstance(value, str):
        return value
    mapped = translator.value(value, field="__ui__")
    return translator.column(value) if mapped == value else mapped


def _option_label():
    """Bind locale for widget serialization and keep raw option values unchanged."""
    translator = _current()
    return lambda value: str(_label(value, translator))


def _display(value, translator=None):
    """Localize known UI prose, including saved application narrative fields."""
    translator = translator or _current()
    if not isinstance(value, str):
        return value
    result = _label(value, translator)
    if result != value:
        return result
    for field in (
        "observation",
        "interpretation",
        "action",
        "recommendation",
        "rationale",
        "reason",
        "finding",
        "why_it_matters",
        "decision_or_use",
        "limit",
        "message",
        "conclusion",
    ):
        result = translator.value(value, field=field)
        if result != value:
            return result
    for marker in ("**", "*"):
        if value.startswith(marker) and value.endswith(marker):
            return marker + _label(value[len(marker) : -len(marker)], translator) + marker
    heading = re.fullmatch(r"(#{1,6} )(.*)", value, re.DOTALL)
    if heading:
        return heading[1] + _label(heading[2], translator)
    return value


def display_frame(value, translator=None):
    """Localize DataFrames/Styler at rendering, not at reader or calculation time."""
    translator = translator or _current()
    if isinstance(value, pd.DataFrame):
        return translator.frame(value)
    # A Styler must retain canonical data keys for pending style operations.
    from pandas.io.formats.style import Styler

    if isinstance(value, Styler):
        styled = copy.deepcopy(value)
        labels = [translator.column(field) for field in value.data.columns]
        if len(set(labels)) == len(labels):
            styled.relabel_index(labels, axis=1)
        mappings = translator.catalog.resources["EN"]["output_mappings"]["values"]
        for field in value.data.columns:
            if field in mappings:
                styled.format(
                    lambda cell, field=field: translator.value(cell, field=field), subset=[field]
                )
        return styled
    return value


def display_record(value):
    """Localize known top-level metadata labels without rewriting nested raw evidence."""
    translator = _current()
    if isinstance(value, dict):
        return translator.record(value)
    if isinstance(value, list):
        return [translator.record(item) if isinstance(item, dict) else item for item in value]
    return value


def display_column_config(config):
    """Match column config keys to translated DataFrame headers."""
    translator = _current()
    return {translator.column(key): value for key, value in config.items()}


def translate_figure(figure, translator=None):
    """Copy a Plotly figure and translate text, never coordinates or measurements."""
    translator = translator or _current()
    result = go.Figure(figure)

    def label(value):
        return _label(value, translator)

    def hover(value):
        if not isinstance(value, str):
            return value
        # Only replace field labels outside Plotly's %{...} expressions.
        return re.sub(r"([^<>=%{}]+)=", lambda match: str(label(match[1])) + "=", value)

    layout = result.layout
    if layout.title.text:
        layout.title.text = label(layout.title.text)
    if layout.legend.title.text:
        layout.legend.title.text = label(layout.legend.title.text)
    for name in layout:
        if name.startswith(("xaxis", "yaxis")):
            axis = layout[name]
            if axis.title.text:
                axis.title.text = label(axis.title.text)
            if axis.ticktext is not None:
                axis.ticktext = [label(value) for value in axis.ticktext]
        elif name.startswith("coloraxis"):
            colorbar = layout[name].colorbar
            if colorbar.title.text:
                colorbar.title.text = label(colorbar.title.text)
    for annotation in layout.annotations:
        annotation.text = label(annotation.text)
    category_ticks = {}
    for trace in result.data:
        for coordinate in ("x", "y"):
            values = getattr(trace, coordinate, None)
            if values is not None and len(values) and all(isinstance(v, str) for v in values):
                reference = getattr(trace, coordinate + "axis", None) or coordinate
                axis_name = coordinate + "axis" + reference[1:]
                ticks = category_ticks.setdefault(axis_name, {})
                ticks.update({v: label(v) for v in values})
        if trace.name:
            trace.name = label(trace.name)
        if hasattr(trace, "labels") and trace.labels is not None:
            trace.labels = [label(value) for value in trace.labels]
        if hasattr(trace, "hovertemplate") and trace.hovertemplate is not None:
            if isinstance(trace.hovertemplate, str):
                trace.hovertemplate = hover(trace.hovertemplate)
            else:
                trace.hovertemplate = [hover(value) for value in trace.hovertemplate]
        if hasattr(trace, "text") and trace.text is not None:
            trace.text = (
                label(trace.text) if isinstance(trace.text, str) else [label(v) for v in trace.text]
            )
    for axis_name, ticks in category_ticks.items():
        axis = layout[axis_name]
        if axis.ticktext is None and any(raw != text for raw, text in ticks.items()):
            axis.update(tickmode="array", tickvals=list(ticks), ticktext=list(ticks.values()))
    return result
