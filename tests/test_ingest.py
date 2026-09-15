"""Tests for reading user-supplied ECG files.

The bar here is not "it parses".  An upload reader that silently mis-reads a
rate, a gain or a polarity produces a confident wrong number, which is the one
outcome this whole project exists to avoid.  So these tests check that the
values come back *correct*, and that the reader refuses rather than guesses
when the file does not say.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from src.ecg_generator import generate_ecg
from src.evidence import ACCEPTED
from src.ingest import (
    IngestError,
    LoadedRecording,
    load_recording,
    looks_inverted,
    read_edf,
    read_text,
    read_wfdb,
    to_ecg_record,
    validate_rate,
)
from src.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clean():
    """A known-good 30 s record at 250 Hz."""
    return generate_ecg(seed=42)


def _csv_bytes(rows, header=None, delim=","):
    out = []
    if header:
        out.append(delim.join(header))
    for r in rows:
        out.append(delim.join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r))
    return ("\n".join(out) + "\n").encode()


def _make_edf(channels, names, fs, record_sec=1.0):
    """Build a minimal but standards-correct EDF in memory."""
    n_sig = len(channels)
    spr = int(fs * record_sec)
    n_rec = len(channels[0]) // spr
    header = (
        f"{0:<8}{'X':<80}{'X':<80}{'01.01.24':<8}{'00.00.00':<8}"
        f"{256 * (n_sig + 1):<8}{'':<44}{n_rec:<8}{record_sec:<8}{n_sig:<4}"
    ).encode("ascii")
    assert len(header) == 256, len(header)

    def block(vals, width):
        return "".join(f"{v:<{width}}" for v in vals).encode("ascii")

    dmin, dmax = -32768, 32767
    pmin, pmax = -5.0, 5.0
    header += block([n[:16] for n in names], 16)          # labels
    header += block(["" for _ in names], 80)              # transducer
    header += block(["mV" for _ in names], 8)             # physical dimension
    header += block([f"{pmin:g}" for _ in names], 8)
    header += block([f"{pmax:g}" for _ in names], 8)
    header += block([f"{dmin:d}" for _ in names], 8)
    header += block([f"{dmax:d}" for _ in names], 8)
    header += block(["" for _ in names], 80)              # prefiltering
    header += block([f"{spr:d}" for _ in names], 8)
    header += block(["" for _ in names], 32)              # reserved

    body = bytearray()
    for r in range(n_rec):
        for c in channels:
            seg = np.asarray(c[r * spr:(r + 1) * spr], dtype=float)
            dig = np.round((seg - pmin) * (dmax - dmin) / (pmax - pmin) + dmin)
            dig = np.clip(dig, dmin, dmax).astype("<i2")
            body += dig.tobytes()
    return bytes(header) + bytes(body)


def _pack_212(values):
    """Pack int values as WFDB format 212 (two 12-bit samples per 3 bytes)."""
    v = np.asarray(values, dtype=np.int32).copy()
    if v.size % 2:
        v = np.append(v, 0)
    v[v < 0] += 4096
    out = bytearray()
    for a, b in zip(v[0::2], v[1::2]):
        out.append(a & 0xFF)
        out.append(((a >> 8) & 0x0F) | (((b >> 8) & 0x0F) << 4))
        out.append(b & 0xFF)
    return bytes(out)


# ---------------------------------------------------------------------------
# Delimited text
# ---------------------------------------------------------------------------
def test_csv_single_column_has_no_rate(clean):
    data = _csv_bytes([[v] for v in clean.signal[:1000]])
    rec = load_recording(data, "ecg.csv")
    assert rec.n_channels == 1
    assert rec.n_samples == 1000
    # A bare column of numbers cannot tell us the rate, and we must not invent one.
    assert rec.sampling_rate is None


def test_csv_without_rate_is_refused_not_guessed(clean):
    rec = load_recording(_csv_bytes([[v] for v in clean.signal[:1000]]), "ecg.csv")
    with pytest.raises(IngestError, match="sampling rate is required"):
        to_ecg_record(rec)


def test_csv_time_column_gives_the_right_rate(clean):
    rows = [[t, v] for t, v in zip(clean.time[:2000], clean.signal[:2000])]
    rec = load_recording(_csv_bytes(rows, header=["time", "ecg"]), "ecg.csv")
    assert rec.sampling_rate == pytest.approx(250.0, abs=0.01)
    assert rec.n_channels == 1                      # time column is not a lead
    assert rec.channel_names == ["ecg"]


def test_csv_time_column_in_milliseconds(clean):
    rows = [[t * 1000.0, v] for t, v in zip(clean.time[:2000], clean.signal[:2000])]
    rec = load_recording(_csv_bytes(rows, header=["time_ms", "ecg"]), "ecg.csv")
    assert rec.sampling_rate == pytest.approx(250.0, abs=0.01)


def test_csv_uneven_time_column_is_rejected_not_averaged(clean):
    t = clean.time[:500].copy()
    t[250:] += 3.0                                  # a gap: not uniformly sampled
    rows = [[a, b] for a, b in zip(t, clean.signal[:500])]
    rec = load_recording(_csv_bytes(rows, header=["time", "ecg"]), "ecg.csv")
    assert rec.sampling_rate is None
    assert any("unevenly sampled" in n for n in rec.notes)


def test_csv_multicolumn_keeps_every_lead(clean):
    rows = [[v, -v, 2 * v] for v in clean.signal[:500]]
    rec = load_recording(_csv_bytes(rows, header=["I", "II", "III"]), "leads.csv")
    assert rec.n_channels == 3
    assert rec.channel_names == ["I", "II", "III"]


def test_csv_tab_and_semicolon_delimiters(clean):
    rows = [[v] for v in clean.signal[:300]]
    for delim in ("\t", ";"):
        rec = load_recording(_csv_bytes(rows, delim=delim), "ecg.csv")
        assert rec.n_samples == 300


def test_csv_headerless_is_not_mistaken_for_a_header(clean):
    rec = load_recording(_csv_bytes([[v] for v in clean.signal[:300]]), "ecg.csv")
    assert rec.n_samples == 300                     # no row consumed as a header


def test_csv_ragged_rows_are_skipped_and_reported(clean):
    body = _csv_bytes([[v, v] for v in clean.signal[:200]]).decode()
    body += "1.0\n"                                 # one short row
    rec = read_text(body.encode(), "ecg.csv")
    assert any("skipped" in n for n in rec.notes)


def test_empty_and_garbage_files_are_refused():
    with pytest.raises(IngestError):
        load_recording(b"", "ecg.csv")
    with pytest.raises(IngestError):
        load_recording(b"not,an,ecg\nalso,not,one\n", "ecg.csv")


def test_too_few_samples_is_refused():
    with pytest.raises(IngestError, match="at least"):
        load_recording(_csv_bytes([[0.1], [0.2], [0.3]]), "ecg.csv")


# ---------------------------------------------------------------------------
# EDF
# ---------------------------------------------------------------------------
def test_edf_round_trip_preserves_rate_and_values(clean):
    n = 250 * 20
    data = _make_edf([clean.signal[:n]], ["ECG"], fs=250)
    rec = read_edf(data, "x.edf")
    assert rec.sampling_rate == pytest.approx(250.0)
    assert rec.n_samples == n
    # 16-bit over a +/-5 mV range: quantisation step is ~0.15 uV.
    assert np.allclose(rec.channels[0], clean.signal[:n], atol=1e-3)


def test_edf_multichannel_names_and_count(clean):
    n = 250 * 10
    data = _make_edf([clean.signal[:n], -clean.signal[:n]], ["Lead I", "Lead II"],
                     fs=250)
    rec = read_edf(data, "x.edf")
    assert rec.n_channels == 2
    assert rec.channel_names == ["Lead I", "Lead II"]


def test_edf_truncated_header_is_refused():
    with pytest.raises(IngestError, match="too short"):
        read_edf(b"\x00" * 100, "x.edf")


def test_bdf_is_refused_with_a_clear_message():
    with pytest.raises(IngestError, match="BDF"):
        read_edf(b"\xff" + b"BIOSEMI" + b"\x00" * 300, "x.bdf")


# ---------------------------------------------------------------------------
# WFDB
# ---------------------------------------------------------------------------
def test_wfdb_format_212_round_trip(clean):
    """The MIT-BIH packing: gain and baseline must be undone correctly."""
    gain, baseline = 200.0, 1024.0
    n = 3600
    adc = np.round(clean.signal[:n] * gain + baseline).astype(int)
    adc = np.clip(adc, 0, 4095)
    hea = (f"rec 1 360 {n}\n"
           f"rec.dat 212 {gain:g}({baseline:g})/mV 12 {baseline:g} 0 0 0 MLII\n")
    rec = read_wfdb(hea, _pack_212(adc), "rec")
    assert rec.sampling_rate == pytest.approx(360.0)
    assert rec.channel_names == ["MLII"]
    expected = (adc - baseline) / gain
    assert np.allclose(rec.channels[0][:n], expected[:n], atol=1e-9)


def test_wfdb_format_16_two_signals(clean):
    gain = 200.0
    n = 1000
    a = np.round(clean.signal[:n] * gain).astype("<i2")
    b = np.round(-clean.signal[:n] * gain).astype("<i2")
    interleaved = np.empty(2 * n, dtype="<i2")
    interleaved[0::2], interleaved[1::2] = a, b
    hea = (f"rec 2 250 {n}\n"
           f"rec.dat 16 {gain:g}/mV 16 0 0 0 0 MLII\n"
           f"rec.dat 16 {gain:g}/mV 16 0 0 0 0 V5\n")
    rec = read_wfdb(hea, interleaved.tobytes(), "rec")
    assert rec.n_channels == 2
    assert rec.channel_names == ["MLII", "V5"]
    assert np.allclose(rec.channels[0], a / gain, atol=1e-9)
    assert np.allclose(rec.channels[1], b / gain, atol=1e-9)


def test_wfdb_needs_both_files():
    with pytest.raises(IngestError, match="both files"):
        load_recording(b"rec 1 360 100\n", "rec.hea")


def test_wfdb_pair_loads_from_either_side(clean):
    gain = 200.0
    n = 500
    adc = np.round(clean.signal[:n] * gain).astype("<i2")
    hea = f"rec 1 250 {n}\nrec.dat 16 {gain:g}/mV 16 0 0 0 0 II\n".encode()
    dat = adc.tobytes()
    from_hea = load_recording(hea, "rec.hea", companion=("rec.dat", dat))
    from_dat = load_recording(dat, "rec.dat", companion=("rec.hea", hea))
    assert np.allclose(from_hea.channels, from_dat.channels)
    assert from_hea.sampling_rate == from_dat.sampling_rate == 250.0


def test_wfdb_unsupported_format_is_named():
    with pytest.raises(IngestError, match="format 24"):
        read_wfdb("rec 1 250 10\nrec.dat 24 200/mV\n", b"\x00" * 100, "rec")


# ---------------------------------------------------------------------------
# MATLAB and NumPy
# ---------------------------------------------------------------------------
def test_mat_round_trip(clean):
    from scipy.io import savemat
    buf = io.BytesIO()
    savemat(buf, {"val": clean.signal[:2000].reshape(1, -1)})
    rec = load_recording(buf.getvalue(), "x.mat")
    assert rec.n_samples == 2000
    assert np.allclose(rec.channels[0], clean.signal[:2000])


def test_mat_reads_an_embedded_sampling_rate(clean):
    from scipy.io import savemat
    buf = io.BytesIO()
    savemat(buf, {"val": clean.signal[:2000], "fs": 250.0})
    rec = load_recording(buf.getvalue(), "x.mat")
    assert rec.sampling_rate == pytest.approx(250.0)


def test_npy_round_trip(clean):
    buf = io.BytesIO()
    np.save(buf, clean.signal[:1500])
    rec = load_recording(buf.getvalue(), "x.npy")
    assert rec.n_samples == 1500
    assert rec.sampling_rate is None


def test_pickled_npy_is_refused():
    """allow_pickle would let a crafted file execute code on load."""
    buf = io.BytesIO()
    np.save(buf, np.array([{"malicious": "object"}], dtype=object), allow_pickle=True)
    with pytest.raises(IngestError):
        load_recording(buf.getvalue(), "x.npy")


def test_unsupported_extension_is_named():
    with pytest.raises(IngestError, match="Unsupported file type"):
        load_recording(b"\x00" * 100, "scan.pdf")


# ---------------------------------------------------------------------------
# Rate handling and windowing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, 0, -5, 1e9, float("nan"), "abc"])
def test_implausible_rates_are_refused(bad):
    with pytest.raises(IngestError):
        validate_rate(bad)


def test_supplied_rate_overrides_the_file(clean):
    rows = [[t, v] for t, v in zip(clean.time[:1000], clean.signal[:1000])]
    loaded = load_recording(_csv_bytes(rows, header=["time", "ecg"]), "e.csv")
    rec = to_ecg_record(loaded, sampling_rate=500.0)
    assert rec.sampling_rate == 500
    assert rec.duration == pytest.approx(2.0)


def test_window_selection_trims_to_the_requested_span(clean):
    loaded = LoadedRecording(
        channels=clean.signal.reshape(1, -1), channel_names=["ecg"],
        sampling_rate=250.0, source_format="test",
    )
    rec = to_ecg_record(loaded, start_sec=5.0, max_sec=10.0)
    assert rec.duration == pytest.approx(10.0)
    assert np.allclose(rec.signal, clean.signal[1250:3750])


def test_bad_channel_index_is_refused(clean):
    loaded = LoadedRecording(
        channels=clean.signal.reshape(1, -1), channel_names=["ecg"],
        sampling_rate=250.0, source_format="test",
    )
    with pytest.raises(IngestError, match="does not exist"):
        to_ecg_record(loaded, channel=3)


def test_nan_samples_are_interpolated_not_propagated(clean):
    sig = clean.signal.copy()
    sig[100:105] = np.nan
    loaded = LoadedRecording(channels=sig.reshape(1, -1), channel_names=["ecg"],
                             sampling_rate=250.0, source_format="test")
    rec = to_ecg_record(loaded)
    assert np.all(np.isfinite(rec.signal))


# ---------------------------------------------------------------------------
# Polarity: detected, reported, never silently corrected
# ---------------------------------------------------------------------------
def test_inversion_is_detected_both_ways(clean):
    assert not looks_inverted(clean.signal)
    assert looks_inverted(-clean.signal)


def test_inversion_is_not_applied_unless_asked(clean):
    loaded = LoadedRecording(channels=(-clean.signal).reshape(1, -1),
                             channel_names=["ecg"], sampling_rate=250.0,
                             source_format="test")
    same = to_ecg_record(loaded, invert=False)
    assert np.allclose(same.signal, -clean.signal)
    flipped = to_ecg_record(loaded, invert=True)
    assert np.allclose(flipped.signal, clean.signal)


# ---------------------------------------------------------------------------
# End to end: an upload is analysed by the real pipeline, with no ground truth
# ---------------------------------------------------------------------------
def test_uploaded_record_carries_no_ground_truth(clean):
    loaded = LoadedRecording(channels=clean.signal.reshape(1, -1),
                             channel_names=["ecg"], sampling_rate=250.0,
                             source_format="test")
    rec = to_ecg_record(loaded)
    assert rec.r_peaks_true.size == 0
    assert rec.metadata["ground_truth"].startswith("none")


def test_clean_upload_runs_the_real_pipeline_and_is_accepted(clean):
    """A clean upload should behave exactly like the clean scenario."""
    loaded = LoadedRecording(channels=clean.signal.reshape(1, -1),
                             channel_names=["ecg"], sampling_rate=250.0,
                             source_format="test")
    result = run_pipeline(to_ecg_record(loaded), "upload")
    assert result.errors == []
    hr = result.measurement("heart_rate")
    assert hr.status == ACCEPTED
    assert hr.value == pytest.approx(71.8, abs=2.0)


def test_upload_path_matches_the_scenario_path_exactly(clean):
    """Going through the ingest layer must not change a single verdict."""
    direct = run_pipeline(clean, "direct")
    loaded = LoadedRecording(channels=clean.signal.reshape(1, -1),
                             channel_names=["ecg"], sampling_rate=250.0,
                             source_format="test")
    uploaded = run_pipeline(to_ecg_record(loaded), "upload")
    assert (
        [(m.measurement, m.status, m.value) for m in uploaded.measurements]
        == [(m.measurement, m.status, m.value) for m in direct.measurements]
    )


def test_inverted_upload_is_refused_rather_than_mismeasured(clean):
    """The failure mode we care about: no confident wrong number."""
    loaded = LoadedRecording(channels=(-clean.signal).reshape(1, -1),
                             channel_names=["ecg"], sampling_rate=250.0,
                             source_format="test")
    result = run_pipeline(to_ecg_record(loaded, invert=False), "upside down")
    hr = result.measurement("heart_rate")
    assert hr.status != ACCEPTED
    assert hr.value is None


def test_wrong_declared_rate_does_not_produce_a_plausible_number(clean):
    """A rate off by 4x must not yield a quietly believable heart rate."""
    loaded = LoadedRecording(channels=clean.signal.reshape(1, -1),
                             channel_names=["ecg"], sampling_rate=250.0,
                             source_format="test")
    result = run_pipeline(to_ecg_record(loaded, sampling_rate=1000.0), "wrong rate")
    hr = result.measurement("heart_rate")
    # Either refused, or reported at the rate the user actually declared --
    # what must not happen is a normal-looking ~72 BPM from a 4x-wrong rate.
    if hr.status == ACCEPTED:
        assert hr.value == pytest.approx(71.8 / 4, rel=0.25)
