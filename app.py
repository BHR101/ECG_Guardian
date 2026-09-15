"""ECG Guardian -- Evidence-Gated ECG Analysis.

Run with::

    streamlit run app.py

Everything shown here is produced by running the pipeline in ``src/``.  No
figure, score or verdict is pre-computed or stored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DISCLAIMER,
    QUALITY_DEGRADED_MIN,
    QUALITY_TRUSTED_MIN,
    TRUST_RECOVERED,
    TRUST_REJECTED,
    TRUST_TRUSTED,
)
from src.demo import SCENARIOS, scenario_by_name
from src.evidence import ACCEPTED
from src.pipeline import run_pipeline
from src.plots import (
    TRUST_COLORS,
    confidence_breakdown_figure,
    evidence_chain_figure,
    quality_factor_figure,
    quality_timeline_figure,
    signal_figure,
    trust_map_figure,
)

st.set_page_config(
    page_title="ECG Guardian",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1500px; }
.eg-title { font-size: 2.3rem; font-weight: 700; margin-bottom: 0; line-height: 1.1; }
.eg-tag { font-size: 1.05rem; color: #2b7fd4; font-weight: 600; margin-top: .1rem; }
.eg-sub { font-size: 1rem; color: #444; margin-top: .55rem; font-style: italic; }
.eg-disclaimer {
  background: #fff6e0; border-left: 4px solid #e6a700; padding: .55rem .85rem;
  border-radius: 4px; font-size: .88rem; color: #6b4e00; margin: .9rem 0 .4rem 0;
}
.eg-card {
  border: 1px solid #e2e5ea; border-radius: 9px; padding: .85rem 1rem;
  background: #ffffff; height: 100%;
}
.eg-card-accept { border-left: 5px solid #1a9850; }
.eg-card-reject { border-left: 5px solid #d73027; background: #fdf4f4; }
.eg-card-name { font-size: .82rem; text-transform: uppercase; letter-spacing: .05em;
  color: #666; font-weight: 600; }
.eg-card-value { font-size: 1.85rem; font-weight: 700; margin: .18rem 0; color: #10233d;
  word-break: normal; overflow-wrap: normal; }
.eg-card-value-rej { font-size: 1.2rem; font-weight: 700; margin: .18rem 0; color: #a01010;
  word-break: normal; overflow-wrap: normal; hyphens: none; line-height: 1.25; }
.eg-card-conf { font-size: .88rem; color: #444; }
.eg-badge { display: inline-block; padding: .12rem .55rem; border-radius: 11px;
  font-size: .76rem; font-weight: 700; margin-top: .45rem; }
.eg-badge-a { background: #e3f4e7; color: #12703c; }
.eg-badge-r { background: #fbe3e2; color: #a01010; }
.eg-reason { font-size: .82rem; color: #7a1f1f; margin-top: .4rem; line-height: 1.35; }
.eg-kv { font-size: .9rem; color: #333; }
.eg-pill { display:inline-block; padding:.15rem .6rem; border-radius:11px;
  font-size:.8rem; font-weight:700; color:#fff; }
.eg-checklist { font-size: .78rem; line-height: 1.5; margin-top: .15rem; }
.eg-ok { color: #12703c; }
.eg-no { color: #a01010; font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "scenario_name" not in st.session_state:
    st.session_state.scenario_name = SCENARIOS[0].name


@st.cache_data(show_spinner=False)
def _analyse(scenario_key: str):
    """Run the pipeline for a scenario.  Cached because it is deterministic."""
    from src.demo import SCENARIOS_BY_KEY

    scenario = SCENARIOS_BY_KEY[scenario_key]
    return run_pipeline(scenario.record(), scenario.name)


@st.cache_data(show_spinner=False)
def _validate():
    from src.validation import (
        differential_gating_check,
        ground_truth_isolation_check,
        run_validation,
    )

    df, summary = run_validation()
    return df, summary, differential_gating_check(), ground_truth_isolation_check()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="eg-title">ECG GUARDIAN</div>', unsafe_allow_html=True)
st.markdown('<div class="eg-tag">Evidence-Gated ECG Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="eg-sub">"Before interpreting an ECG, determine whether the '
    'evidence is sufficient."</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="eg-disclaimer">⚠️ {DISCLAIMER} It does not detect, '
            'diagnose or rule out any medical condition.</div>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 1 -- demo controls
# ---------------------------------------------------------------------------
st.markdown("### 1 · Demo controls")
ctrl_left, ctrl_right = st.columns([3, 2])

with ctrl_left:
    names = [s.name for s in SCENARIOS]
    scenario_name = st.selectbox(
        "Scenario",
        names,
        index=names.index(st.session_state.scenario_name)
        if st.session_state.scenario_name in names
        else 0,
    )
    scenario = scenario_by_name(scenario_name)
    b1, b2 = st.columns(2)
    run_clicked = b1.button("▶ Run analysis", type="primary", use_container_width=True)
    reset_clicked = b2.button("↺ Reset", use_container_width=True)

with ctrl_right:
    st.caption(f"**{scenario.description}**")
    st.caption(f"What this scenario is for: {scenario.expectation}")

if reset_clicked:
    st.session_state.result = None
    st.session_state.scenario_name = scenario_name
    st.rerun()

if run_clicked:
    st.session_state.scenario_name = scenario_name
    with st.spinner("Running the pipeline..."):
        st.session_state.result = _analyse(scenario.key)

result = st.session_state.result


# ---------------------------------------------------------------------------
# Sidebar -- pipeline map and thresholds
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Pipeline")
    st.code(
        "ECG\n"
        " -> signal quality\n"
        " -> artifact detection\n"
        " -> localisation + class\n"
        " -> recovery\n"
        " -> REVALIDATION\n"
        "      |        |\n"
        "    pass     fail -> reject segment\n"
        " -> measurement analysis\n"
        " -> evidence engine\n"
        "      |        |\n"
        "   ACCEPT    REJECT\n"
        " -> dashboard",
        language="text",
    )

    st.markdown("### Decision thresholds")
    st.caption(
        f"Trusted window ≥ {QUALITY_TRUSTED_MIN:.0f} / 100 · "
        f"unusable below {QUALITY_DEGRADED_MIN:.0f} / 100. "
        "Each measurement then applies its own, separate evidence standard."
    )

if result is None:
    st.info("Choose a scenario and press **Run analysis**.")
    st.stop()


# ---------------------------------------------------------------------------
# Section 2 -- signal overview
# ---------------------------------------------------------------------------
st.markdown("### 2 · Signal overview")
st.caption(
    "Shaded spans are the regions the detector localised on its own. Markers on "
    "the processed trace show which beats met the heart-rate evidence standard."
)
st.plotly_chart(signal_figure(result), use_container_width=True)


# ---------------------------------------------------------------------------
# Section 3 -- trust map
# ---------------------------------------------------------------------------
st.markdown("### 3 · Trust map")
st.caption(
    "Different parts of one record carry different reliability. This is the "
    "output the rest of the system reasons over — not a single global verdict."
)
st.plotly_chart(trust_map_figure(result), use_container_width=True)

tm_cols = st.columns(4)
for i, label in enumerate((TRUST_TRUSTED, "DEGRADED", TRUST_RECOVERED, TRUST_REJECTED)):
    seconds = sum(s.end - s.start for s in result.trust_map if s.label == label)
    with tm_cols[i]:
        st.markdown(
            f'<span class="eg-pill" style="background:{TRUST_COLORS.get(label, "#888")}">'
            f"{label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="eg-kv">{seconds:.1f} s of {result.record.duration:.0f} s</div>',
            unsafe_allow_html=True,
        )

with st.expander("Trust map segments"):
    st.dataframe(
        pd.DataFrame([s.as_dict() for s in result.trust_map]),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# Section 4 -- signal quality
# ---------------------------------------------------------------------------
st.markdown("### 4 · Prototype Signal Quality Index")
q_left, q_right = st.columns([1, 2])

with q_left:
    st.metric(
        "Raw record",
        f"{result.quality_raw.score:.0f} / 100",
        help="Mean over all analysis windows of the raw signal.",
    )
    st.metric(
        "After recovery",
        f"{result.quality_processed.score:.0f} / 100",
        delta=f"{result.quality_processed.score - result.quality_raw.score:+.1f}",
    )
    ev = result.quality_raw.evidence
    st.markdown("**Factors behind the score**")
    for key in (
        "baseline_stability", "qrs_visibility", "high_frequency_noise",
        "powerline_interference", "amplitude_stability", "dropout", "artifact_burden",
    ):
        st.markdown(
            f'<div class="eg-kv">· {key.replace("_", " ")}: '
            f"<b>{ev.get(key, 'unknown')}</b></div>",
            unsafe_allow_html=True,
        )
    st.caption(
        f"Worst window {ev.get('worst_window_score', 0):.0f}/100 · "
        f"{ev.get('n_windows', 0)} windows analysed."
    )

with q_right:
    st.plotly_chart(quality_timeline_figure(result), use_container_width=True)

with st.expander("Why the score is what it is — measured features and point deductions"):
    st.plotly_chart(quality_factor_figure(result), use_container_width=True)
    st.caption(
        "The index is 100 minus these deductions. Every deduction is a "
        "documented function of a measured feature; none is learned or assigned."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"feature": k, "raw record": v,
                 "after recovery": result.quality_processed.evidence["mean_features"].get(k)}
                for k, v in result.quality_raw.evidence["mean_features"].items()
            ]
        ),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# Section 5 -- artifact intelligence
# ---------------------------------------------------------------------------
st.markdown("### 5 · Artifact intelligence")
if not result.detections:
    st.success("No disturbance found above the detection threshold.")
else:
    for det in result.detections:
        with st.container(border=True):
            a, b, c = st.columns([2, 1, 3])
            a.markdown(f"**{det.label}**")
            a.caption(f"classification confidence {det.confidence:.0%}")
            b.markdown(f"**{det.start:.2f} – {det.end:.2f} s**")
            b.caption(f"{det.duration:.2f} s long")
            c.markdown("**Supporting evidence**")
            for line in det.evidence:
                c.markdown(f'<div class="eg-kv">· {line}</div>', unsafe_allow_html=True)
            with st.expander("Class evidence scores for this region"):
                st.caption(
                    "Each artifact class is scored from the point deductions it "
                    "would produce. The highest score wins; how far it leads the "
                    "runner-up is what sets the confidence."
                )
                st.dataframe(
                    pd.DataFrame(
                        sorted(det.class_scores.items(), key=lambda kv: -kv[1]),
                        columns=["artifact class", "evidence score"],
                    ),
                    use_container_width=True, hide_index=True,
                )


# ---------------------------------------------------------------------------
# Section 6 -- recovery and revalidation
# ---------------------------------------------------------------------------
st.markdown("### 6 · Recovery and revalidation")
rec = result.recovery
status_colour = {
    "VALIDATED": TRUST_COLORS[TRUST_TRUSTED],
    "PARTIAL": "#e6a700",
    "FAILED": TRUST_COLORS[TRUST_REJECTED],
    "NOT_REQUIRED": "#777777",
}[rec.status]

r1, r2, r3 = st.columns([1, 1, 2])
with r1:
    if np.isnan(rec.affected_before):
        st.metric("Affected-region quality", "—")
    else:
        st.metric(
            "Affected-region quality",
            f"{rec.affected_before:.0f} → {rec.affected_after:.0f}",
            delta=f"{rec.affected_after - rec.affected_before:+.0f} points",
        )
with r2:
    st.markdown("**Status**")
    st.markdown(
        f'<span class="eg-pill" style="background:{status_colour}">'
        f'{"RECOVERY " + rec.status if rec.status != "NOT_REQUIRED" else "NOT REQUIRED"}'
        "</span>",
        unsafe_allow_html=True,
    )
with r3:
    st.markdown("**Method**")
    st.markdown(f'<div class="eg-kv">{rec.method_label}</div>', unsafe_allow_html=True)
    for reason in dict.fromkeys(rec.method_reasons):
        st.caption(reason)

if rec.regions:
    st.markdown("**Revalidation — every recovery attempt is independently re-checked**")
    st.caption(
        "Two tests must both pass. The quality score must reach an acceptable "
        "level, **and** the property that defined the artifact must have "
        "actually normalised. A rising score on its own is not evidence that "
        "the missing content came back — removing a step discontinuity lifts "
        "the score without restoring a single complex."
    )
    for r in rec.regions:
        icon = "✅" if r.recovered else "❌"
        st.markdown(
            f"{icon} **{r.start:.2f}–{r.end:.2f} s** "
            f"({r.artifact_type.replace('_', ' ')}) — "
            f"{r.score_before:.0f} → {r.score_after:.0f} · "
            f"{'RECOVERY VALIDATED' if r.recovered else 'RECOVERY FAILED — segment stays rejected'}"
        )
        # Show the two tests separately, so a case where the score improved but
        # the evidence did not is visible rather than buried in one sentence.
        parts = [p.strip() for p in r.reason.split(";")]
        score_part = parts[0] if parts else r.reason
        struct_part = parts[1] if len(parts) > 1 else ""
        c1, c2 = st.columns(2)
        with c1:
            ok = "did not" not in score_part and "only" not in score_part
            st.markdown(
                f"{'✓' if ok else '✗'} **Quality test** — {score_part}"
            )
        with c2:
            if struct_part:
                ok = "did not normalise" not in struct_part
                st.markdown(
                    f"{'✓' if ok else '✗'} **Structural test** — {struct_part}"
                )
        if not r.recovered:
            affected = [
                m.display_name for m in result.measurements
                if any(s <= r.start < e or s < r.end <= e
                       for s, e in m.rejected_segments)
            ]
            st.caption(
                "This segment stays rejected, so no measurement may draw "
                "evidence from it."
                + (f" Measurements excluding it: {', '.join(affected)}." if affected else "")
            )
for note in rec.notes:
    st.caption(note)


# ---------------------------------------------------------------------------
# Section 7 -- measurements
# ---------------------------------------------------------------------------
st.markdown("### 7 · Measurements")
st.caption(
    "Each measurement is judged against its own evidence standard. That is why "
    "one record can yield an accepted heart rate and a refused QRS duration."
)

cards = st.columns(len(result.measurements))
for col, m in zip(cards, result.measurements):
    accepted = m.status == ACCEPTED
    with col:
        value_html = (
            f'<div class="eg-card-value">{m.value:g} '
            f'<span style="font-size:1rem;color:#666">{m.unit}</span></div>'
            if accepted
            else '<div class="eg-card-value-rej">NOT REPORTED</div>'
        )
        badge = (
            '<span class="eg-badge eg-badge-a">✓ ACCEPTED</span>'
            if accepted
            else '<span class="eg-badge eg-badge-r">✗ REJECTED</span>'
        )
        reason = (
            "" if accepted else f'<div class="eg-reason">{m.reason}</div>'
        )
        st.markdown(
            f'<div class="eg-card {"eg-card-accept" if accepted else "eg-card-reject"}">'
            f'<div class="eg-card-name">{m.display_name}</div>'
            f"{value_html}"
            f'<div class="eg-card-conf">{m.confidence:.0%} confidence</div>'
            f"{badge}{reason}</div>",
            unsafe_allow_html=True,
        )
        # The evidence checklist, on the card itself: which specific checks
        # this measurement needed, and which of them it actually met.
        st.markdown(
            f'<div class="eg-checklist"><b>{m.evidence_standard}</b></div>',
            unsafe_allow_html=True,
        )
        for c in m.criteria:
            mark = "✓" if c.passed else "✗"
            cls = "eg-ok" if c.passed else "eg-no"
            st.markdown(
                f'<div class="eg-checklist {cls}">{mark} {c.name}: '
                f"<b>{c.value}</b> <span style='opacity:.65'>(needs {c.required})</span>"
                "</div>",
                unsafe_allow_html=True,
            )

st.markdown("")
tabs = st.tabs([m.display_name for m in result.measurements])
for tab, m in zip(tabs, result.measurements):
    with tab:
        st.markdown(f"**Evidence standard applied: {m.evidence_standard}**")
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Criteria**")
            st.dataframe(
                pd.DataFrame([c.as_dict() for c in m.criteria]).rename(
                    columns={
                        "name": "criterion", "value": "measured",
                        "required": "required", "passed": "met",
                    }
                ),
                use_container_width=True, hide_index=True,
            )
        with right:
            st.markdown("**Where the confidence came from**")
            st.plotly_chart(
                confidence_breakdown_figure(m), use_container_width=True,
                key=f"conf_{m.measurement}",
            )
            st.caption(
                "Confidence is the sum of these weighted, measured terms — not "
                "an assigned number."
            )
        st.markdown("**Evidence**")
        for line in m.evidence:
            st.markdown(f'<div class="eg-kv">· {line}</div>', unsafe_allow_html=True)
        if m.status != ACCEPTED:
            st.error(f"Rejected — {m.reason}")


# ---------------------------------------------------------------------------
# Section 8 -- evidence chain
# ---------------------------------------------------------------------------
st.markdown("### 8 · Evidence chain")
chain_target = st.selectbox(
    "Trace the evidence for",
    [m.display_name for m in result.measurements],
    index=0,
    key="chain_pick",
)
m = next(x for x in result.measurements if x.display_name == chain_target)

e1, e2, e3, e4 = st.columns(4)
e1.metric("Validated beats", m.validated_beats)
e2.metric("Rejected beats", m.rejected_beats)
rrc = result.measurement("rr_consistency")
e3.metric(
    "RR consistency",
    f"{rrc.value:g}%" if rrc and rrc.value is not None else "—",
)
e4.metric("Measurement confidence", f"{m.confidence:.0%}")

st.plotly_chart(evidence_chain_figure(m), use_container_width=True)

chain_left, chain_right = st.columns(2)
with chain_left:
    st.markdown("**Supporting regions**")
    if m.supporting_segments:
        for s, e in m.supporting_segments:
            st.markdown(f'<div class="eg-kv">· {s:.2f} – {e:.2f} s</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="eg-kv">· none</div>', unsafe_allow_html=True)
with chain_right:
    st.markdown("**Excluded regions**")
    if m.rejected_segments:
        for s, e in m.rejected_segments:
            st.markdown(f'<div class="eg-kv">· {s:.2f} – {e:.2f} s</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="eg-kv">· none</div>', unsafe_allow_html=True)

with st.expander("Every beat, and why it was or was not used"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "beat time (s)": b.time,
                    "local quality": round(b.local_quality, 1)
                    if not np.isnan(b.local_quality) else None,
                    "rel. amplitude": round(b.relative_amplitude, 2),
                    "rel. prominence": round(b.relative_prominence, 2),
                    "shape correlation": round(b.template_correlation, 3),
                    "QRS width (ms)": round(mo.duration_ms, 1)
                    if not np.isnan(mo.duration_ms) else None,
                    "used": v.accepted,
                    "why not": "; ".join(v.reasons),
                }
                for b, v, mo in zip(result.beats, m.beat_verdicts, result.morphology)
            ]
        ),
        use_container_width=True, hide_index=True, height=300,
    )


# ---------------------------------------------------------------------------
# Section 9 -- event log
# ---------------------------------------------------------------------------
st.markdown("### 9 · Event log")
st.caption("Emitted by the pipeline as it ran.")
log_df = pd.DataFrame([e.as_dict() for e in result.log])
st.dataframe(
    log_df[["time", "stage", "message", "level"]],
    use_container_width=True, hide_index=True, height=330,
)
if result.errors:
    st.warning("Stage failures recorded this run: " + "; ".join(result.errors))


# ---------------------------------------------------------------------------
# Section 10 -- validation
# ---------------------------------------------------------------------------
st.markdown("### 10 · Prototype validation")
st.caption(
    "Measured by running the pipeline over records whose corruption we injected "
    "ourselves, so ground truth is known. Nothing here is stored or hand-written."
)
if st.button("Run validation now"):
    with st.spinner("Running the validation set..."):
        df, summary, gate, iso = _validate()
    st.session_state.validation = (df, summary, gate, iso)

if "validation" in st.session_state:
    df, summary, gate, iso = st.session_state.validation
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("R-peak sensitivity", f"{summary['peak_sensitivity']:.3f}")
    v1.metric("R-peak PPV", f"{summary['peak_ppv']:.3f}")
    v2.metric("Timing error", f"{summary['peak_timing_error_ms']:.2f} ms")
    v2.metric("Localisation IoU", f"{summary['localisation_iou']:.3f}")
    v3.metric("Classification accuracy", f"{summary['classification_accuracy']:.3f}")
    v3.metric("False alarms on clean", f"{summary['false_alarm_regions_on_clean']:.0f}")
    v4.metric("Heart-rate error", f"{summary['hr_abs_error_bpm']:.2f} BPM")
    v4.metric("Worst heart-rate error", f"{summary['hr_max_abs_error_bpm']:.2f} BPM")

    w1, w2, w3 = st.columns(3)
    w1.metric(
        "Separable artifacts revalidated",
        f"{summary['separable_recovery_validated_rate']:.0%}",
        delta=f"{summary['separable_region_quality_gain']:+.1f} quality points",
    )
    w2.metric(
        "Artifact time excluded when recovery failed",
        f"{summary['failed_recovery_exclusion_fraction']:.0%}",
        help=f"Across {summary['n_failed_recovery_cases']:.0f} cases whose recovery "
             "did not pass revalidation.",
    )
    w3.metric(
        "Contact loss wrongly revalidated",
        f"{summary['contact_loss_recovery_validated_rate']:.0%}",
        help="Lower is better. A known limitation — see the README.",
    )

    if gate["different_verdicts"]:
        st.success(
            f"**Measurement-specific gating confirmed.** On *{gate['scenario']}*, "
            f"heart rate was {gate['heart_rate_status']} "
            f"({gate['heart_rate_value']} BPM) while QRS duration was "
            f"{gate['qrs_status']}: {gate['qrs_reason']}"
        )
    else:
        st.warning(
            "Differential gating check did not produce different verdicts on this run."
        )

    st.markdown(
        f"**Ground-truth isolation:** no analysis entry point accepts a "
        f"ground-truth argument — `{iso['clean']}`."
    )
    st.code("\n".join(f"{k}({', '.join(v)})" for k, v in iso["signatures"].items()),
            language="text")

    with st.expander("Per-case detail"):
        st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    f"{DISCLAIMER} Signal quality assessment, filtering, QRS detection and "
    "artifact detection are established techniques and are not claimed as novel "
    "here. The contribution is the architecture: reliability is resolved into a "
    "per-measurement accept/reject decision rather than a preprocessing step."
)
