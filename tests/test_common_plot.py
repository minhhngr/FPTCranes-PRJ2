from __future__ import annotations

import plotly.graph_objects as go

from pages.common import show_plot


class PlotRecorder:
    def __init__(self):
        self.figure = None
        self.kwargs = None

    def plotly_chart(self, figure, **kwargs):
        self.figure = figure
        self.kwargs = kwargs


def test_show_plot_preserves_explicit_geometry_and_adds_only_missing_defaults():
    figure = go.Figure(go.Bar(x=["A"], y=[1]))
    figure.update_layout(
        height=520,
        margin=dict(l=210, r=30, t=80, b=90),
        hovermode="x unified",
    )
    recorder = PlotRecorder()

    show_plot(recorder, figure, key="readable")

    assert recorder.figure.layout.height == 520
    assert recorder.figure.layout.margin.l == 210
    assert recorder.figure.layout.margin.b == 90
    assert recorder.figure.layout.hovermode == "x unified"
    assert recorder.kwargs["width"] == "stretch"


def test_show_plot_supplies_defaults_when_figure_has_no_geometry():
    figure = go.Figure(go.Bar(x=["A"], y=[1]))
    recorder = PlotRecorder()

    show_plot(recorder, figure)

    assert recorder.figure.layout.margin.l == 40
    assert recorder.figure.layout.margin.b == 50
    assert recorder.figure.layout.hovermode == "closest"
