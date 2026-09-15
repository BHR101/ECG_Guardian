"""Plotly figures for the dashboard.

Kept out of ``app.py`` so the figures can be built and checked without a
Streamlit runtime.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import (
    QUALITY_DEGRADED_MIN,
    QUALITY_TRUSTED_MIN,
    TRUST_DEGRADED,
    TRUST_RECOVERED,
    TRUST_REJECTED,
    TRUST_TRUSTED,
)
from .evidence import ACCEPTED, MeasurementResult
from .pipeline import PipelineResult

TRUST_COLORS: dict[str, str] = {
    TRUST_TRUSTED: "#1a9850",
    TRUST_DEGRADED: "#e6a700",
    TRUST_RECOVERED: "#2b7fd4",
    TRUST_REJECTED: "#d73027",
}

ARTIFACT_FILL = "rgba(215, 48, 39, 0.13)"

# Figures are drawn on a transparent ground so they sit on whatever the page is
# painted in.  Streamlit follows the viewer's light/dark preference, and a solid
# white plot panel punched into a dark page is what made the dashboard hard to
# read.  Axis, grid and label colours are therefore semi-transparent greys that
# resolve against either background, and the data colours below clear both.
_AXIS_TEXT = "#8494a5"
_GRID = "rgba(132, 148, 165, 0.20)"
_AXIS_LINE = "rgba(132, 148, 165, 0.38)"

# Only keys that no call site also passes -- a figure supplying its own legend
# or subplot axes must not collide with these.
_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=55, r=20, t=40, b=40),
    hovermode="x unified",
    font=dict(size=12, color=_AXIS_TEXT),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def _theme_axes(fig: go.Figure) -> go.Figure:
    """Apply the neutral axis palette across every subplot, then return it."""
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_AXIS_LINE,
                     tickfont=dict(color=_AXIS_TEXT),
                     title_font=dict(color=_AXIS_TEXT))
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_AXIS_LINE,
                     tickfont=dict(color=_AXIS_TEXT),
                     title_font=dict(color=_AXIS_TEXT))
    fig.update_layout(legend=dict(font=dict(color=_AXIS_TEXT)))
    return fig


def signal_figure(result: PipelineResult) -> go.Figure:
    """Raw and processed ECG on a shared time axis, with artifact regions marked."""
    t = result.record.time
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Raw ECG", "Processed ECG (after recovery)"),
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=result.record.signal, name="raw",
            line=dict(color="#8494a5", width=1),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=result.processed_signal, name="processed",
            line=dict(color="#4a9ae0", width=1),
        ),
        row=2, col=1,
    )

    # Detected artifact spans, shaded on both panels.
    for det in result.detections:
        for row in (1, 2):
            fig.add_vrect(
                x0=det.start, x1=det.end,
                fillcolor=ARTIFACT_FILL, line_width=0, layer="below",
                row=row, col=1,
            )
        fig.add_annotation(
            x=0.5 * (det.start + det.end), y=1.0, yref="paper",
            text=det.label, showarrow=False,
            font=dict(size=10, color="#d1683f"), yanchor="bottom",
        )

    # Validated and rejected beats, from the heart-rate gate.
    hr = result.measurement("heart_rate")
    if hr is not None and hr.beat_verdicts:
        proc = result.processed_signal
        acc_t, acc_y, rej_t, rej_y = [], [], [], []
        for beat, verdict in zip(result.beats, hr.beat_verdicts):
            idx = min(max(beat.index, 0), proc.size - 1)
            if verdict.accepted:
                acc_t.append(beat.time)
                acc_y.append(proc[idx])
            else:
                rej_t.append(beat.time)
                rej_y.append(proc[idx])
        if acc_t:
            fig.add_trace(
                go.Scatter(
                    x=acc_t, y=acc_y, mode="markers", name="validated beat",
                    marker=dict(color=TRUST_COLORS[TRUST_TRUSTED], size=7, symbol="circle"),
                ),
                row=2, col=1,
            )
        if rej_t:
            fig.add_trace(
                go.Scatter(
                    x=rej_t, y=rej_y, mode="markers", name="rejected beat",
                    marker=dict(color=TRUST_COLORS[TRUST_REJECTED], size=8, symbol="x"),
                ),
                row=2, col=1,
            )

    fig.update_layout(height=470, legend=dict(orientation="h", y=-0.14), **_LAYOUT)
    fig.update_yaxes(title_text="mV", row=1, col=1)
    fig.update_yaxes(title_text="mV", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    return _theme_axes(fig)


def trust_map_figure(result: PipelineResult) -> go.Figure:
    """The trust timeline: which parts of the record may be used, and how."""
    fig = go.Figure()
    seen: set[str] = set()
    for seg in result.trust_map:
        fig.add_trace(
            go.Bar(
                x=[seg.end - seg.start],
                base=[seg.start],
                y=["trust"],
                orientation="h",
                marker=dict(color=TRUST_COLORS.get(seg.label, "#888888")),
                name=seg.label,
                legendgroup=seg.label,
                showlegend=seg.label not in seen,
                hovertemplate=(
                    f"<b>{seg.label}</b><br>{seg.start:.2f}-{seg.end:.2f} s"
                    f"<br>mean quality {seg.score:.0f}/100<br>{seg.note}<extra></extra>"
                ),
            )
        )
        seen.add(seg.label)

    fig.update_layout(
        height=170,
        barmode="stack",
        showlegend=True,
        legend=dict(orientation="h", y=-0.45),
        **_LAYOUT,
    )
    fig.update_xaxes(title_text="time (s)", range=[0, result.record.duration])
    fig.update_yaxes(showticklabels=False)
    return _theme_axes(fig)


def quality_timeline_figure(result: PipelineResult) -> go.Figure:
    """Per-window quality before and after recovery, against the decision bands."""
    fig = go.Figure()
    raw, proc = result.quality_raw, result.quality_processed

    fig.add_hrect(
        y0=QUALITY_TRUSTED_MIN, y1=100,
        fillcolor="rgba(26, 152, 80, 0.07)", line_width=0, layer="below",
    )
    fig.add_hrect(
        y0=QUALITY_DEGRADED_MIN, y1=QUALITY_TRUSTED_MIN,
        fillcolor="rgba(230, 167, 0, 0.09)", line_width=0, layer="below",
    )
    fig.add_hrect(
        y0=0, y1=QUALITY_DEGRADED_MIN,
        fillcolor="rgba(215, 48, 39, 0.07)", line_width=0, layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=raw.centers, y=raw.scores, name="before recovery",
            line=dict(color="#8494a5", width=2, dash="dot"),
        )
    )
    if proc is not raw:
        fig.add_trace(
            go.Scatter(
                x=proc.centers, y=proc.scores, name="after recovery",
                line=dict(color="#4a9ae0", width=2),
            )
        )

    fig.add_hline(
        y=QUALITY_TRUSTED_MIN, line=dict(color="#1a9850", width=1, dash="dash"),
        annotation_text="trusted", annotation_position="right",
    )
    fig.add_hline(
        y=QUALITY_DEGRADED_MIN, line=dict(color="#d73027", width=1, dash="dash"),
        annotation_text="unusable below", annotation_position="right",
    )

    fig.update_layout(height=280, legend=dict(orientation="h", y=-0.22), **_LAYOUT)
    fig.update_xaxes(title_text="time (s)", range=[0, result.record.duration])
    fig.update_yaxes(title_text="Prototype Signal Quality Index", range=[0, 103])
    return _theme_axes(fig)


def evidence_chain_figure(measurement: MeasurementResult) -> go.Figure:
    """Beat-by-beat evidence behind one measurement."""
    fig = go.Figure()
    if not measurement.beat_verdicts:
        fig.update_layout(height=190, **_LAYOUT)
        return fig

    acc = [v for v in measurement.beat_verdicts if v.accepted]
    rej = [v for v in measurement.beat_verdicts if not v.accepted]

    if acc:
        fig.add_trace(
            go.Scatter(
                x=[v.time for v in acc],
                y=[v.local_quality for v in acc],
                mode="markers", name=f"meets standard ({len(acc)})",
                marker=dict(color=TRUST_COLORS[TRUST_TRUSTED], size=9, symbol="circle"),
                hovertemplate="beat at %{x:.2f} s<br>local quality %{y:.0f}<extra></extra>",
            )
        )
    if rej:
        fig.add_trace(
            go.Scatter(
                x=[v.time for v in rej],
                y=[v.local_quality for v in rej],
                mode="markers", name=f"does not ({len(rej)})",
                marker=dict(color=TRUST_COLORS[TRUST_REJECTED], size=10, symbol="x"),
                text=["; ".join(v.reasons)[:110] for v in rej],
                hovertemplate="beat at %{x:.2f} s<br>local quality %{y:.0f}<br>%{text}<extra></extra>",
            )
        )

    fig.update_layout(height=230, legend=dict(orientation="h", y=-0.28), **_LAYOUT)
    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text="local quality", range=[-4, 104])
    return _theme_axes(fig)


def quality_factor_figure(result: PipelineResult) -> go.Figure:
    """What actually cost the record its quality points."""
    pens = result.quality_raw.evidence.get("mean_penalties", {})
    after = result.quality_processed.evidence.get("mean_penalties", {})
    labels = [k.replace("_", " ") for k in pens]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels, x=[pens[k] for k in pens], orientation="h",
            name="before recovery", marker=dict(color="#8494a5"),
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels, x=[after.get(k, 0.0) for k in pens], orientation="h",
            name="after recovery", marker=dict(color="#4a9ae0"),
        )
    )
    fig.update_layout(
        height=300, barmode="group", legend=dict(orientation="h", y=-0.2), **_LAYOUT
    )
    fig.update_xaxes(title_text="mean points deducted (0 = no problem found)")
    return _theme_axes(fig)


def confidence_breakdown_figure(measurement: MeasurementResult) -> go.Figure:
    """The weighted terms that produced a measurement's confidence."""
    terms = measurement.confidence_terms
    fig = go.Figure()
    if not terms:
        fig.update_layout(height=170, **_LAYOUT)
        return fig
    labels = list(terms)
    colour = (
        TRUST_COLORS[TRUST_TRUSTED]
        if measurement.status == ACCEPTED
        else TRUST_COLORS[TRUST_REJECTED]
    )
    fig.add_trace(
        go.Bar(
            y=labels, x=[terms[k] for k in labels], orientation="h",
            marker=dict(color=colour),
            hovertemplate="%{y}: contributes %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=190, showlegend=False, **_LAYOUT)
    fig.update_xaxes(title_text="contribution to confidence", range=[0, 0.55])
    return _theme_axes(fig)
