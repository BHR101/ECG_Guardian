"""Reading user-supplied ECG recordings in standard formats.

This module turns a file someone uploaded into the same :class:`ECGRecord` the
synthetic generator produces, so an uploaded recording goes through exactly the
same pipeline as every demo scenario.  There is no separate path for real data
and no relaxed mode: if the evidence is not there, an uploaded recording is
refused the same way a synthetic one is.

Supported formats
-----------------
====================  =======================================================
``.csv`` ``.tsv``     delimited text; header optional, multi-column allowed,
``.txt``              a time column is detected and used to derive the rate
``.edf``              European Data Format / EDF+ (the biosignal standard)
``.hea`` + ``.dat``   WFDB / PhysioNet (formats 16, 61, 80, 212)
``.mat``              MATLAB (v4-v7.2 via scipy)
``.npy`` ``.npz``     NumPy arrays
====================  =======================================================

Two things this module will not do
----------------------------------
It will not invent a sampling rate.  CSV and NumPy files usually carry no rate;
without one, every time-based measurement would be wrong by an unknown factor,
so the caller must supply it (the UI asks) or the file must contain a usable
time column.  A guessed rate is worse than a refused upload.

It will not silently flip an inverted trace.  The R-peak detector assumes an
upright R wave, so a lead recorded with reversed polarity is refused rather
than mismeasured.  Inversion is *detected* and reported, and the caller may
offer the user an explicit switch -- lead polarity is something the person who
recorded it knows, not something this code should decide for them.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .ecg_generator import ECGRecord

# Guard rails on what we will accept, so a malformed or hostile file cannot
# exhaust memory before we have looked at it.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 20_000_000
MIN_SAMPLES = 32

# Plausible acquisition rates.  Used only to reject nonsense, never to guess.
MIN_RATE_HZ = 10.0
MAX_RATE_HZ = 20_000.0

TEXT_SUFFIXES = (".csv", ".tsv", ".txt", ".dat.txt")
SUPPORTED_SUFFIXES = (
    ".csv", ".tsv", ".txt", ".edf", ".bdf", ".hea", ".dat", ".mat", ".npy", ".npz",
)

# Column names that mean "this is the time axis, not a lead".
_TIME_NAMES = ("time", "t", "sec", "secs", "second", "seconds", "ms",
               "millis", "millisecond", "milliseconds", "timestamp", "elapsed")


class IngestError(ValueError):
    """Raised when a file cannot be read as an ECG.  Message is user-facing."""


@dataclass
class LoadedRecording:
    """One uploaded recording, before a single channel has been chosen."""

    channels: np.ndarray                 # (n_channels, n_samples), float
    channel_names: list[str]
    sampling_rate: float | None          # None when the file did not say
    source_format: str
    filename: str = ""
    units: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return int(self.channels.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.channels.shape[1])

    def duration(self, rate: float | None = None) -> float | None:
        r = rate if rate is not None else self.sampling_rate
        return self.n_samples / r if r else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _finite_2d(rows: np.ndarray, name: str) -> np.ndarray:
    """Coerce to a finite (n_channels, n_samples) float array."""
    arr = np.atleast_2d(np.asarray(rows, dtype=float))
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T                       # samples are the long axis
    if arr.size == 0:
        raise IngestError(f"{name} contains no samples.")
    if arr.shape[1] < MIN_SAMPLES:
        raise IngestError(
            f"{name} has only {arr.shape[1]} samples; at least {MIN_SAMPLES} "
            "are needed to assess anything."
        )
    if arr.shape[1] > MAX_SAMPLES:
        raise IngestError(
            f"{name} has {arr.shape[1]:,} samples, over the {MAX_SAMPLES:,} limit."
        )
    return arr


def _clean_channel(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace non-finite samples by linear interpolation; report how many."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    n_bad = int(bad.sum())
    if n_bad and n_bad < x.size:
        idx = np.arange(x.size)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    elif n_bad:
        raise IngestError("Every sample in this channel is missing or non-numeric.")
    return x, n_bad


def _rate_from_time_column(t: np.ndarray) -> tuple[float | None, str]:
    """Derive a sampling rate from a time column, or explain why we cannot."""
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < MIN_SAMPLES:
        return None, "time column too short to derive a rate"
    d = np.diff(t)
    if np.any(d <= 0):
        return None, "time column is not strictly increasing"
    med = float(np.median(d))
    if med <= 0:
        return None, "time column has no usable spacing"
    # Reject a wildly irregular axis rather than average over it.
    if float(np.max(np.abs(d - med))) > 0.5 * med:
        return None, "time column is unevenly sampled"
    for unit, scale, label in ((1.0, 1.0, "seconds"),
                               (1e-3, 1e3, "milliseconds"),
                               (1e-6, 1e6, "microseconds")):
        rate = scale / med
        if MIN_RATE_HZ <= rate <= MAX_RATE_HZ:
            # Floating-point division leaves 249.99999999999977 where the file
            # plainly means 250.  Snapping a hair's breadth is arithmetic
            # cleanup, not a guess, so keep the tolerance tight.
            if abs(rate - round(rate)) < 0.01:
                rate = float(round(rate))
            return float(rate), f"derived from the time column, read as {label}"
    return None, "time column spacing implies an implausible rate"


def validate_rate(rate: float | None) -> float:
    """Check a rate the caller supplied."""
    if rate is None:
        raise IngestError("A sampling rate is required.")
    try:
        r = float(rate)
    except (TypeError, ValueError):
        raise IngestError("The sampling rate must be a number.") from None
    if not np.isfinite(r) or not (MIN_RATE_HZ <= r <= MAX_RATE_HZ):
        raise IngestError(
            f"Sampling rate {rate} Hz is outside the accepted "
            f"{MIN_RATE_HZ:g}-{MAX_RATE_HZ:g} Hz range."
        )
    return r


def looks_inverted(x: np.ndarray) -> bool:
    """True when the dominant deflection is downward.

    The R-peak detector assumes an upright R wave.  This is a report, not a
    correction: nothing here modifies the signal.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < MIN_SAMPLES:
        return False
    med = float(np.median(x))
    up = float(np.percentile(x, 99.5)) - med
    down = med - float(np.percentile(x, 0.5))
    return down > 1.25 * up


# ---------------------------------------------------------------------------
# Delimited text
# ---------------------------------------------------------------------------
def read_text(data: bytes, filename: str = "") -> LoadedRecording:
    """Read CSV / TSV / whitespace-delimited text."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        raise IngestError("The file is empty.")

    sample = "\n".join(lines[:40])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t| ")
        delim = dialect.delimiter
    except csv.Error:
        delim = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)

    def split(line: str) -> list[str]:
        return line.split(delim) if delim else line.split()

    first = split(lines[0])

    def numeric(fields: list[str]) -> bool:
        vals = [f.strip() for f in fields if f.strip() != ""]
        if not vals:
            return False
        try:
            [float(v) for v in vals]
            return True
        except ValueError:
            return False

    has_header = not numeric(first)
    names = [f.strip().strip('"').strip("'") for f in first] if has_header else []
    body = lines[1:] if has_header else lines
    if not body:
        raise IngestError("The file has a header but no data rows.")

    n_cols = len(split(body[0]))
    rows: list[list[float]] = []
    skipped = 0
    for line in body:
        fields = split(line)
        if len(fields) != n_cols:
            skipped += 1
            continue
        try:
            rows.append([float(f) if f.strip() != "" else np.nan for f in fields])
        except ValueError:
            skipped += 1
    if not rows:
        raise IngestError("No numeric data rows could be parsed.")

    table = np.asarray(rows, dtype=float)          # (n_samples, n_cols)
    notes: list[str] = []
    if skipped:
        notes.append(f"{skipped} row(s) skipped as unparseable or ragged.")

    if not names:
        names = [f"column {i + 1}" for i in range(table.shape[1])]

    # Identify a time column, by name first and then by shape.
    rate: float | None = None
    time_col: int | None = None
    for i, nm in enumerate(names):
        if re.sub(r"[^a-z]", "", nm.lower()) in _TIME_NAMES:
            time_col = i
            break
    if time_col is None and table.shape[1] > 1:
        col = table[:, 0]
        if np.all(np.isfinite(col)) and np.all(np.diff(col) > 0):
            time_col = 0
    if time_col is not None:
        rate, why = _rate_from_time_column(table[:, time_col])
        notes.append(
            f"Column '{names[time_col]}' read as a time axis; sampling rate {why}."
            if rate else
            f"Column '{names[time_col]}' looks like a time axis but "
            f"no rate could be derived ({why})."
        )
        keep = [i for i in range(table.shape[1]) if i != time_col]
        if not keep:
            raise IngestError("The file contains a time column and no signal column.")
        table = table[:, keep]
        names = [names[i] for i in keep]

    return LoadedRecording(
        channels=_finite_2d(table.T, filename or "file"),
        channel_names=names,
        sampling_rate=rate,
        source_format="delimited text",
        filename=filename,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# EDF / EDF+
# ---------------------------------------------------------------------------
def read_edf(data: bytes, filename: str = "") -> LoadedRecording:
    """Read European Data Format (EDF and EDF+)."""
    if len(data) < 256:
        raise IngestError("File is too short to be an EDF.")
    if data[:1] == b"\xff":
        raise IngestError(
            "This is a BDF file (24-bit BioSemi), which is not supported. "
            "Export it as EDF or CSV."
        )

    def txt(lo: int, hi: int) -> str:
        return data[lo:hi].decode("ascii", "replace").strip()

    def num(lo: int, hi: int, what: str) -> float:
        raw = txt(lo, hi)
        try:
            return float(raw)
        except ValueError:
            raise IngestError(f"EDF header field '{what}' is not a number ({raw!r}).") from None

    header_bytes = int(num(184, 192, "header length"))
    reserved = txt(192, 236)
    n_records = int(num(236, 244, "number of records"))
    record_sec = num(244, 252, "record duration")
    n_sig = int(num(252, 256, "number of signals"))

    if n_sig <= 0:
        raise IngestError("EDF header declares no signals.")
    if record_sec <= 0:
        raise IngestError("EDF header declares a non-positive record duration.")
    if len(data) < 256 + n_sig * 256:
        raise IngestError("EDF header is truncated.")

    def fields(offset: int, width: int) -> list[str]:
        base = 256 + offset * n_sig
        return [data[base + i * width: base + (i + 1) * width]
                .decode("ascii", "replace").strip() for i in range(n_sig)]

    labels = fields(0, 16)
    phys_dim = fields(16 + 80, 8)
    phys_min = [float(v or 0) for v in fields(16 + 80 + 8, 8)]
    phys_max = [float(v or 0) for v in fields(16 + 80 + 16, 8)]
    dig_min = [float(v or 0) for v in fields(16 + 80 + 24, 8)]
    dig_max = [float(v or 0) for v in fields(16 + 80 + 32, 8)]
    n_samp = [int(float(v or 0)) for v in fields(16 + 80 + 40 + 80, 8)]

    if n_records < 0:                      # -1 means "unknown", seen in EDF+
        body = len(data) - header_bytes
        per = 2 * sum(n_samp)
        n_records = body // per if per else 0
    if n_records <= 0:
        raise IngestError("EDF file declares no data records.")

    per_record = sum(n_samp)
    needed = header_bytes + 2 * per_record * n_records
    notes: list[str] = []
    if len(data) < needed:
        n_records = max(0, (len(data) - header_bytes) // (2 * per_record))
        if n_records <= 0:
            raise IngestError("EDF data section is truncated.")
        notes.append("File is shorter than its header claims; read what was there.")
    if reserved.upper().startswith("EDF+D"):
        notes.append(
            "EDF+D (discontinuous) file: segments are concatenated, so any gap "
            "between them is not represented in the time axis."
        )

    raw = np.frombuffer(data, dtype="<i2", count=per_record * n_records,
                        offset=header_bytes)
    raw = raw.reshape(n_records, per_record)

    bounds = np.cumsum([0] + n_samp)
    channels, names, rates = [], [], []
    for i in range(n_sig):
        if n_samp[i] <= 0:
            continue
        if "annotation" in labels[i].lower():   # EDF+ annotation channel
            notes.append(f"Skipped non-signal channel '{labels[i]}'.")
            continue
        block = raw[:, bounds[i]:bounds[i + 1]].reshape(-1).astype(float)
        span_d = dig_max[i] - dig_min[i]
        span_p = phys_max[i] - phys_min[i]
        if span_d and span_p:
            block = (block - dig_min[i]) * (span_p / span_d) + phys_min[i]
        channels.append(block)
        names.append(labels[i] or f"signal {i + 1}")
        rates.append(n_samp[i] / record_sec)

    if not channels:
        raise IngestError("The EDF contains no usable signal channels.")

    # Channels may legitimately differ in rate; we can only analyse one grid.
    if len(set(np.round(rates, 6))) > 1:
        fastest = int(np.argmax(rates))
        notes.append(
            "Channels have different sampling rates "
            f"({', '.join(f'{n}={r:g} Hz' for n, r in zip(names, rates))}). "
            f"Only channels at {rates[fastest]:g} Hz are offered."
        )
        keep = [i for i, r in enumerate(rates) if abs(r - rates[fastest]) < 1e-6]
        channels = [channels[i] for i in keep]
        names = [names[i] for i in keep]
        rates = [rates[i] for i in keep]

    width = min(c.size for c in channels)
    arr = np.vstack([c[:width] for c in channels])
    return LoadedRecording(
        channels=_finite_2d(arr, filename or "EDF"),
        channel_names=names,
        sampling_rate=float(rates[0]),
        source_format="EDF / EDF+",
        filename=filename,
        units=phys_dim[0] if phys_dim else "",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# WFDB / PhysioNet
# ---------------------------------------------------------------------------
def _parse_hea(text: str) -> dict:
    """Parse a WFDB header into the fields we need."""
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise IngestError("The .hea file is empty.")
    head = lines[0].split()
    if len(head) < 2:
        raise IngestError("The .hea record line is malformed.")
    try:
        n_sig = int(head[1])
        fs = float(head[2]) if len(head) > 2 else 250.0
        n_samp = int(float(head[3])) if len(head) > 3 else 0
    except ValueError:
        raise IngestError("The .hea record line has non-numeric fields.") from None

    signals = []
    for ln in lines[1:1 + n_sig]:
        f = ln.split()
        if len(f) < 2:
            continue
        gain_field = f[2] if len(f) > 2 else "200"
        m = re.match(r"([-\d.eE+]+)", gain_field)
        gain = float(m.group(1)) if m else 200.0
        if gain == 0:
            gain = 200.0                      # WFDB: 0 means "uncalibrated"
        baseline = 0.0
        b = re.search(r"\(([-\d.eE+]+)\)", gain_field)
        if b:
            baseline = float(b.group(1))
        units = gain_field.split("/")[-1] if "/" in gain_field else "mV"
        adc_zero = float(f[4]) if len(f) > 4 else 0.0
        if not b:
            baseline = adc_zero
        signals.append({
            "file": f[0],
            "format": re.sub(r"[^0-9]", "", f[1]) or "16",
            "gain": gain,
            "baseline": baseline,
            "units": units,
            "name": " ".join(f[8:]) if len(f) > 8 else f"signal {len(signals) + 1}",
        })
    if not signals:
        raise IngestError("The .hea file declares no signal lines.")
    return {"n_sig": len(signals), "fs": fs, "n_samp": n_samp, "signals": signals}


def _unpack_212(buf: bytes, n_sig: int) -> np.ndarray:
    """Unpack WFDB format 212: two 12-bit signed samples per 3 bytes."""
    b = np.frombuffer(buf, dtype=np.uint8)
    b = b[: (b.size // 3) * 3].reshape(-1, 3).astype(np.int32)
    first = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    second = b[:, 2] | ((b[:, 1] >> 4) << 8)
    out = np.empty(first.size * 2, dtype=np.int32)
    out[0::2], out[1::2] = first, second
    out[out > 2047] -= 4096                    # sign-extend 12-bit
    return out


def read_wfdb(hea_text: str, dat_bytes: bytes, filename: str = "") -> LoadedRecording:
    """Read a WFDB record from its header text and its signal file."""
    hea = _parse_hea(hea_text)
    n_sig = hea["n_sig"]
    fmt = hea["signals"][0]["format"]
    if any(s["format"] != fmt for s in hea["signals"]):
        raise IngestError("Mixed WFDB sample formats in one record are not supported.")

    if fmt == "212":
        flat = _unpack_212(dat_bytes, n_sig)
    elif fmt in ("16", "160"):
        flat = np.frombuffer(dat_bytes, dtype="<i2").astype(np.int32)
    elif fmt == "61":
        flat = np.frombuffer(dat_bytes, dtype=">i2").astype(np.int32)
    elif fmt == "80":
        flat = np.frombuffer(dat_bytes, dtype=np.uint8).astype(np.int32) - 128
    else:
        raise IngestError(
            f"WFDB format {fmt} is not supported. Supported: 16, 61, 80, 212."
        )

    usable = (flat.size // n_sig) * n_sig
    if usable == 0:
        raise IngestError("The .dat file holds no complete samples.")
    frame = flat[:usable].reshape(-1, n_sig).T.astype(float)

    notes: list[str] = []
    if hea["n_samp"] and frame.shape[1] != hea["n_samp"]:
        notes.append(
            f"Header declares {hea['n_samp']:,} samples per signal; "
            f"the .dat file holds {frame.shape[1]:,}."
        )

    names = []
    for i, s in enumerate(hea["signals"]):
        frame[i] = (frame[i] - s["baseline"]) / s["gain"]
        names.append(s["name"] or f"signal {i + 1}")

    return LoadedRecording(
        channels=_finite_2d(frame, filename or "WFDB record"),
        channel_names=names,
        sampling_rate=float(hea["fs"]),
        source_format=f"WFDB (format {fmt})",
        filename=filename,
        units=hea["signals"][0]["units"],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# MATLAB and NumPy
# ---------------------------------------------------------------------------
def read_mat(data: bytes, filename: str = "") -> LoadedRecording:
    """Read a MATLAB .mat file (v4 to v7.2)."""
    from scipy.io import loadmat
    try:
        mat = loadmat(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(
            "Could not read this .mat file. MATLAB v7.3 files are HDF5 and are "
            f"not supported -- re-save as v7 or export to CSV. ({exc})"
        ) from None

    rate = None
    for key in ("fs", "Fs", "FS", "sampling_rate", "samplingRate", "srate", "freq"):
        if key in mat:
            try:
                rate = float(np.asarray(mat[key]).ravel()[0])
            except (ValueError, IndexError):
                rate = None
            if rate is not None:
                break

    candidates = {
        k: np.asarray(v, dtype=float)
        for k, v in mat.items()
        if not k.startswith("__") and hasattr(v, "shape")
        and np.asarray(v).size >= MIN_SAMPLES
        and np.issubdtype(np.asarray(v).dtype, np.number)
    }
    if not candidates:
        raise IngestError("No numeric signal array found in this .mat file.")
    name, arr = max(candidates.items(), key=lambda kv: kv[1].size)
    notes = [f"Variable '{name}' used as the signal."]
    if len(candidates) > 1:
        notes.append("Other variables present: "
                     + ", ".join(k for k in candidates if k != name) + ".")
    if rate:
        notes.append(f"Sampling rate {rate:g} Hz read from the file.")

    arr2 = _finite_2d(arr, filename or ".mat file")
    return LoadedRecording(
        channels=arr2,
        channel_names=[f"{name} [{i + 1}]" for i in range(arr2.shape[0])]
        if arr2.shape[0] > 1 else [name],
        sampling_rate=rate if rate and MIN_RATE_HZ <= rate <= MAX_RATE_HZ else None,
        source_format="MATLAB .mat",
        filename=filename,
        notes=notes,
    )


def read_numpy(data: bytes, filename: str = "") -> LoadedRecording:
    """Read .npy / .npz.  Pickled arrays are refused -- they execute code."""
    buf = io.BytesIO(data)
    notes: list[str] = []
    try:
        if filename.lower().endswith(".npz"):
            with np.load(buf, allow_pickle=False) as z:
                keys = list(z.files)
                if not keys:
                    raise IngestError("The .npz archive is empty.")
                key = max(keys, key=lambda k: z[k].size)
                arr = np.asarray(z[key], dtype=float)
                notes.append(f"Array '{key}' used as the signal.")
        else:
            arr = np.asarray(np.load(buf, allow_pickle=False), dtype=float)
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(
            f"Could not read this NumPy file (pickled arrays are refused): {exc}"
        ) from None

    arr2 = _finite_2d(arr, filename or "NumPy file")
    return LoadedRecording(
        channels=arr2,
        channel_names=[f"channel {i + 1}" for i in range(arr2.shape[0])],
        sampling_rate=None,
        source_format="NumPy array",
        filename=filename,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def load_recording(
    data: bytes,
    filename: str,
    companion: tuple[str, bytes] | None = None,
) -> LoadedRecording:
    """Read an uploaded file, choosing the reader from its extension.

    Args:
        data: the file's bytes.
        filename: used only to pick the format and to label the result.
        companion: for WFDB, the other half of the pair -- ``(name, bytes)``
            of the ``.dat`` when ``data`` is the ``.hea``, or vice versa.
    """
    if not data:
        raise IngestError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(
            f"File is {len(data) / 1e6:.1f} MB, over the "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit."
        )

    suffix = os.path.splitext(filename)[1].lower()

    if suffix in (".hea", ".dat"):
        if companion is None:
            raise IngestError(
                "A WFDB record needs both files: upload the .hea header and "
                "the .dat signal file together."
            )
        cname, cbytes = companion
        if suffix == ".hea":
            hea_bytes, dat_bytes = data, cbytes
        else:
            hea_bytes, dat_bytes = cbytes, data
        return read_wfdb(
            hea_bytes.decode("utf-8", "replace"), dat_bytes,
            filename=os.path.splitext(filename)[0],
        )

    if suffix == ".edf" or suffix == ".bdf":
        return read_edf(data, filename)
    if suffix == ".mat":
        return read_mat(data, filename)
    if suffix in (".npy", ".npz"):
        return read_numpy(data, filename)
    if suffix in TEXT_SUFFIXES or suffix == "":
        return read_text(data, filename)

    raise IngestError(
        f"Unsupported file type '{suffix}'. Supported: "
        + ", ".join(SUPPORTED_SUFFIXES) + "."
    )


def to_ecg_record(
    loaded: LoadedRecording,
    channel: int = 0,
    sampling_rate: float | None = None,
    invert: bool = False,
    start_sec: float = 0.0,
    max_sec: float | None = None,
) -> ECGRecord:
    """Build the ECGRecord the pipeline analyses.

    ``r_peaks_true`` is empty: an uploaded recording has no ground truth, and
    the pipeline never reads that field in any case.

    Args:
        loaded: the parsed upload.
        channel: which channel to analyse.
        sampling_rate: overrides the file's rate; required when it had none.
        invert: flip polarity, for a lead recorded upside down.  The caller
            must have obtained this from the user -- it is never inferred.
        start_sec: offset of the analysis window.
        max_sec: length of the analysis window; None analyses to the end.
    """
    rate = validate_rate(
        sampling_rate if sampling_rate is not None else loaded.sampling_rate
    )
    if not 0 <= channel < loaded.n_channels:
        raise IngestError(
            f"Channel {channel} does not exist; the file has {loaded.n_channels}."
        )

    x, _ = _clean_channel(loaded.channels[channel])

    lo = int(max(0.0, start_sec) * rate)
    hi = loaded.n_samples if max_sec is None else lo + int(max_sec * rate)
    x = x[lo:min(hi, x.size)]
    if x.size < MIN_SAMPLES:
        raise IngestError(
            f"The selected window holds only {x.size} samples; "
            f"at least {MIN_SAMPLES} are needed."
        )
    if invert:
        x = -x

    return ECGRecord(
        time=np.arange(x.size, dtype=float) / rate,
        signal=x,
        sampling_rate=int(round(rate)),
        r_peaks_true=np.asarray([], dtype=int),
        metadata={
            "source": "user upload",
            "filename": loaded.filename,
            "format": loaded.source_format,
            "channel": loaded.channel_names[channel]
            if channel < len(loaded.channel_names) else str(channel),
            "sampling_rate_hz": rate,
            "inverted": invert,
            "window_start_sec": start_sec,
            "ground_truth": "none -- uploaded recording",
        },
    )
