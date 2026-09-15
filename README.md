# ECG Guardian

**Evidence-Gated ECG Analysis**

> *Prototype engineering system. Not intended for clinical diagnosis.*
> It does not detect, diagnose or rule out any medical condition.

---

## What this is

ECG Guardian does not ask what an ECG *means*. It asks whether there is enough
trustworthy evidence in the recording to justify making a particular
measurement — and it answers that question **separately for every
measurement**.

A moderately degraded recording may still contain perfectly good evidence for
*when* the beats happened, while containing nothing like enough evidence for
*how wide* the complexes were. A conventional pipeline computes one
signal-quality number and applies it as a single gate to everything downstream,
so both measurements share a fate. ECG Guardian does not:

```
HEART RATE          QRS DURATION
72.1 BPM            NOT REPORTED
86% confidence      58% confidence
✓ ACCEPTED          ✗ REJECTED
                    beat-to-beat spread of QRS duration (IQR)
                    was 58.2 ms (required ≤ 14 ms)
```

Both of those come from the same 30-second record, in the same run. You can
reproduce them: choose *Muscle noise (mild degradation)* in the dashboard.

**NOT REPORTED is a decision, not a failure.** Where a gate refuses, the
dashboard names the specific criterion that failed and shows — struck through,
so it can be read but never mistaken for a result — the number the arithmetic
actually produced. On the severely corrupted scenario that withheld value is
72.5 BPM: close to the true rate, and entirely plausible-looking. It is
withheld anyway, because the evidence behind it did not survive. False
precision is worse than missing information.

## What is and is not claimed as novel

Signal-quality assessment, band-pass filtering, notch filtering, Pan-Tompkins
QRS detection and rule-based artifact detection are **established techniques**.
None of them were invented here, and the code says so where it uses them.

The proposed contribution is the **architecture**: ECG Guardian turns ECG
reliability from a preprocessing concern into a measurement-level decision.

```
ECG
 → segment / beat quality
 → artifact reasoning
 → recovery
 → REVALIDATION
 → measurement-specific evidence
 → ACCEPT / REJECT
```

Three properties follow from that, and each is tested:

1. **Recovery is never assumed to have worked.** Every filter is followed by an
   independent re-assessment, *and* by a re-measurement of the property that
   defined the artifact in the first place. Removing a step discontinuity
   raises a quality score without restoring a single missing complex, so a
   rising score alone is not accepted as proof.
   (`test_contact_loss_is_not_declared_recovered`)
2. **Rejection is a valid output.** Where the evidence is insufficient, no
   number is produced and the failing criterion is named.
   (`test_rejected_measurement_reports_no_number`)
3. **Different measurements reach different verdicts from one record.** A
   single global quality gate cannot produce this.
   (`test_one_record_can_produce_different_verdicts`)

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

That is the whole demo. It is software-only: it runs offline and needs no
database, no account and no hardware.

Other entry points:

```bash
python -m pytest tests/ -q             # 233 tests
python -m src.validation               # validation report, measured live
python tools/adversarial_run.py        # the full adversarial corpus
python tools/final_review.py           # the project's own review checklist
python tools/demo_flow.py              # the five-act walkthrough, headless
python tools/make_sample_uploads.py    # example files for the upload panel
python tools/real_data_evaluation.py   # real-ECG evaluation (dataset optional)
```

## Analysing your own recording

The dashboard's **Upload a recording** mode runs your own ECG through the same
pipeline as the demo scenarios — same thresholds, same evidence gates, same
willingness to report nothing.

| format | extension | carries its own sampling rate |
|---|---|---|
| Delimited text | `.csv` `.tsv` `.txt` | only via a time column |
| European Data Format | `.edf` (EDF / EDF+) | yes |
| WFDB / PhysioNet | `.hea` **and** `.dat` together (formats 16, 61, 80, 212) | yes |
| MATLAB | `.mat` (v7.2 and older) | if it stores `fs` |
| NumPy | `.npy` `.npz` | no |

Multi-column text and multi-channel EDF/WFDB are read in full and you pick the
lead. The EDF and WFDB readers are written out in `src/ingest.py`; no new
dependency was added for either.

Two refusals are deliberate, and they exist for the same reason the evidence
gates do:

- **The sampling rate is never guessed.** It is the one quantity the pipeline
  is not invariant to — every timing measurement scales linearly with it. A
  time column is used only when it is strictly increasing *and* uniformly
  spaced; a file with a gap in it is refused as *"unevenly sampled"* rather
  than averaged into a plausible-looking wrong rate. Where the file states no
  rate, you must.
- **An inverted lead is never silently flipped.** `peaks.py` assumes an upright
  R wave, so reversed polarity is detected, reported, and left alone, with a
  switch you set yourself. Lead polarity is something the person who recorded
  it knows; it is not the software's to assume. Left unflipped, such a record
  is refused rather than mismeasured
  (`test_inverted_upload_is_refused_rather_than_mismeasured`).

Amplitude scale and DC offset, by contrast, need no handling at all. Every
quality feature normalises against a record-level reference, so millivolts,
microvolts and raw ADC counts give bit-identical verdicts — which is why there
is deliberately no units control. That invariance was measured before the
reader was written, not assumed.

```bash
python tools/make_sample_uploads.py   # writes one example per format
```

`data/samples/muscle_noise_with_time.csv` is the differential-verdict record
round-tripped through CSV: heart rate accepted, QRS duration refused.

42 of the 233 tests cover this path. They check that values come back
*correct*, not merely parsed: a misread gain, rate or packing would produce a
confident wrong number, which is the one outcome this project exists to avoid.

---

## Repository layout

```
ecg-guardian/
├── app.py                  Streamlit dashboard
├── requirements.txt
├── README.md
├── src/
│   ├── config.py           every threshold, in one auditable place
│   ├── ecg_generator.py    Phase 1  synthetic ECG + ground truth
│   ├── artifacts.py        Phase 2  controlled artifact injection
│   ├── ingest.py           read user-supplied ECG files (the other input path)
│   ├── quality.py          Phase 3  windowed quality index
│   ├── detection.py        Phase 4  artifact detection / localisation / class
│   ├── filtering.py        Phase 5+6 recovery AND revalidation
│   ├── peaks.py            Phase 7  R-peak detection, per-beat evidence
│   ├── measurements.py     Phase 8  heart rate, RR, QRS boundaries
│   ├── evidence.py         Phase 9  measurement-specific gating  ← the core
│   ├── pipeline.py         orchestration, trust map, event log
│   ├── demo.py             Phase 13 deterministic scenarios
│   ├── validation.py       Phase 14 validation, measured live
│   ├── adversarial.py      adversarial corpus + ground-truth scoring
│   └── plots.py            Plotly figures
├── tests/                  233 tests
├── tools/                  diagnostics, not runtime:
│   ├── final_review.py     answers the project's own review checklist, live
│   ├── calibrate.py        measured feature values used to set the thresholds
│   ├── check_*.py          per-stage accuracy against ground truth
│   ├── adversarial_run.py  run the corpus, report false acceptance
│   ├── demo_flow.py        the five-act walkthrough, headless
│   ├── real_data_evaluation.py   independent real-ECG evaluation
│   ├── make_sample_uploads.py    example files for the upload panel
│   └── export_demo.py      write scenarios to data/generated/
└── data/generated/
```

`detection.py` and `plots.py` are additions to the originally sketched module
list: keeping artifact *detection* in the same file as artifact *injection*
would have put ground truth and the detector in one namespace, which is exactly
the mistake the project is about.

---

## How it works

### 1 · Synthetic ECG (`ecg_generator.py`)

A sum of Gaussian P, Q, R, S and T components placed relative to each R peak,
with mild respiratory-style RR variability and a realistic sensor-noise floor.
Deterministic: the same seed gives a byte-identical signal.

Ground truth (R-peak sample indices, nominal QRS geometry) is returned for
validation only. **No analysis stage can see it** — the entry points take a
waveform and a sampling rate and nothing else, which the validation report
verifies by inspecting their signatures.

### 2 · Artifacts (`artifacts.py`)

Baseline wander, 50 Hz mains interference, muscle/EMG noise, motion, and
electrode-contact loss. Each is injected into a known span with a known
severity, tapered so it starts and stops smoothly, and each records its own
ground truth. Injection is provably localised: a test asserts that nothing
outside the requested span changes by more than 1e-12.

### 3 · Quality (`quality.py`)

Per-window (1.0 s, 0.5 s hop) measurement of baseline drift, high-frequency
power, mains-band power, amplitude relative to the record, QRS contrast, and
dropout. The **Prototype Signal Quality Index** is `100` minus a set of
documented piecewise-linear penalties over those measured features. No learned
weights, no assigned numbers; the dashboard shows the deduction behind every
point lost.

Two details worth pointing out, because both were real failure modes found
during development and fixed rather than tuned around:

- **The amplitude reference** is taken from the windows where a QRS is most
  clearly visible, not from a plain median. A plain median fails in two
  opposite ways — if most of the record is dropped out it describes the
  dropout, and the few intact stretches then measure as huge excursions.
- **Dropout is a question about time, not about one window.** At 48 BPM the
  beats are 1.25 s apart, so a 1.0 s window can legitimately fall between two
  of them. Dropout is therefore scored from the length of the QRS-free gap a
  window sits inside, combined with an absolute energy measure over a
  beat-spanning context. Each covers the other's blind spot.
- **A complex returns to baseline; a step does not.** Both deposit broadband
  energy, so an energy-only measure cannot tell them apart — which is precisely
  how an electrode step used to hide a dropout, by splitting the QRS-free gap
  in two and making missing signal look occupied. `baseline_return_ratio`
  compares the median level just before an event with the median level just
  after it. Measured on the corpus, real complexes sit at 0.016–0.037 and never
  exceed 0.29 even under severe wander or at 180 BPM; contact-loss steps measure
  around 0.80.
- **Several features need a beat's worth of context, and not the same ones.**
  An amplitude *excursion* is local and transient (motion), so it is measured on
  the window. An amplitude *collapse* is sustained (a failing electrode), so it
  is measured over a rhythm-adaptive context — otherwise a window falling
  between two slow beats, or one whose boundary cuts through a complex, reads as
  a collapse. Interference is normalised against a record-level power reference
  rather than the window's own total, for the same reason: a beat-free window
  has almost no power of its own to divide by.

### 4 · Detection (`detection.py`)

Windows below the trusted threshold are classified individually from the same
penalty breakdown, then grouped into regions of a single class. Two of the
rules are worth stating because they encode real distinctions:

- An amplitude excursion is only *motion* if it also obscures the complexes.
  Amplitude alone is not enough.
- A baseline that moves while the signal stays intact is *wander*; the same
  shift with the QRS content missing is a *failing electrode*, which is the
  more specific finding.

### 5+6 · Recovery and revalidation (`filtering.py`)

Artifact-specific: high-pass for drift, notch for mains, low-pass for EMG,
conservative band limitation for motion. Contact loss is explicitly marked
non-restorable — filtering cannot recreate absent QRS content.

**Revalidation is mandatory and has two independent tests.** A region counts as
recovered only if (a) it is acceptable now — either already at the trusted
threshold, or improved by a real margin and clear of the post-recovery floor —
*and* (b) the feature that defined the artifact has normalised across at least
70% of the region's windows. A region that fails stays `REJECTED`, and every
beat inside it is refused regardless of how it looks.

Both halves of (a) matter. Demanding a fixed improvement unconditionally
condemns a stretch already sitting at 70/100 that filtering can only take to 71,
and every measurement depending on it is then refused for no reason. Demanding
only the improvement, without (b), lets a filter that merely removed a step
discontinuity claim it restored the complexes it never touched.

### 7 · R-peaks (`peaks.py`)

Band-pass → differentiate → square → integrate → adaptive threshold, with the
threshold blended from a local and a record-wide reference. **Tuned for
sensitivity, not precision**: a missed beat destroys evidence nothing
downstream can recover, whereas an extra candidate costs nothing because the
evidence engine judges every beat before using it. T-wave false positives are
cleanly separable (relative prominence ≈ 0.05 and shape correlation ≈ 0.37,
versus ≈ 1.0 for real beats) and a test asserts none of them ever reaches a
measurement.

After the trust map is known, the beat references are rebuilt from beats in
trustworthy regions, so a heavily corrupted record cannot define "normal" from
its own corruption.

### 9 · Evidence gating (`evidence.py`) — the core

Each measurement declares what evidence it needs. The same beat is judged
against a **different standard** depending on what is being asked of it:

| | timing standard (heart rate, RR) | morphological standard (QRS duration) |
|---|---|---|
| local quality | ≥ 55 | ≥ 72 |
| relative prominence | ≥ 0.25 | ≥ 0.55 |
| relative amplitude | 0.40 – 2.50 | 0.70 – 1.60 |
| shape correlation | ≥ 0.70 | ≥ 0.90 |
| local HF noise | — | ≤ 0.35 |
| QRS onset/offset resolved | — | required |

A beat can be good enough to say *when* it happened without being good enough
to say *how wide* it was. That sentence is the whole project.

Confidence is a weighted sum of measured quantities — validated beat fraction,
RR consistency, local quality, onset/offset agreement, shape consistency — and
the dashboard shows each term's contribution. A test asserts that every
reported confidence equals the sum of its own terms, so no number can drift
away from its evidence.

---

## Validation

Two independent harnesses, both measured by running the pipeline over records
whose corruption we injected, so ground truth is known. Nothing below is
stored or hand-written.

```bash
python -m src.validation          # per-stage metrics
python tools/adversarial_run.py   # the adversarial corpus
```

### Per-stage metrics (31 cases)

| | |
|---|---|
| R-peak sensitivity / PPV | 0.978 / 0.913 |
| R-peak timing error | 0.60 ms |
| Artifact localisation (IoU) | 0.832 |
| Artifact classification accuracy | 0.962 |
| False-alarm regions on clean records | 0 |
| Mean score, clean records | 100.0 / 100 |
| Clean seconds wrongly rejected | 0.00 s |
| Separable artifacts revalidated | 100% (+38.8 points) |
| Contact loss wrongly revalidated | 0% |
| Heart-rate error when reported | 0.14 BPM mean, 0.52 BPM worst |
| QRS error vs noise-free reference | 0.32 ms |
| Artifact time excluded when recovery failed | 91.8% |
| Uncaught pipeline errors | 0 |

### Adversarial corpus (328 cases, 200 randomised)

The corpus exists to find cases where the system reports a measurement it
cannot support. It sweeps heart rates from 40 to 150 BPM, sampling rates from
200 to 500 Hz, durations from 8 to 60 s, every artifact at five severities both
localised and record-wide, ten cross-artifact combinations, and a set of shapes
designed specifically to fool it: pure DC steps, flatlines, dropouts placed
exactly over complexes, repeated contact loss, polarity inversion, and trains of
non-cardiac spikes at plausible heart-rate intervals.

**The safety-critical metric is the false acceptance rate:** how often a
reported value was materially wrong against ground truth.

| | |
|---|---|
| Cases | 328 (200 randomised, fixed seeds) |
| Crashes / uncaught stage errors | 0 / 0 |
| Values reported | 474 |
| **False acceptances** | **0** |
| **False acceptance rate** | **0.00%** |
| P0 (wrong value, high confidence) | 0 |
| P1 (wrong value, low confidence) | 0 |
| Heart-rate error when reported | 0.46 BPM mean, 3.34 BPM worst |
| QRS error when reported | 0.31 ms mean, 1.35 ms worst |
| Conservative refusals (P2) | 71 heart rate, 95 QRS duration |

Ground truth is scored two ways, deliberately. The **outcome-based** measure
asks whether a reported number was actually right, which cannot be argued with.
The **evidence-based** measure compares every beat against the same beat in a
noise-free twin and asks whether the record could have supported the
measurement at all. Both report zero false acceptances.

### Independent real-ECG evaluation (1000 fragments)

Everything above is measured on records this project generated and corrupted
itself. That is what makes the ground truth trustworthy, and it is equally the
limitation: the corruption was injected by the same codebase being tested.

A separate harness answers a narrower question — *does the architecture behave
sensibly on ECG signals this project did not create?*

```bash
python tools/real_data_evaluation.py
```

It runs over [Mendeley Data 7dybx7wyfn v3](https://data.mendeley.com/datasets/7dybx7wyfn/3):
1000 × 10 s single-lead (MLII) fragments cut from 45 MIT-BIH Arrhythmia
recordings. The dataset is **optional** — nothing else in the project needs it,
and the script exits cleanly with download instructions when it is absent.

**What this evaluation cannot claim.** The fragments carry a rhythm/beat class
per fragment (the folder name) and nothing else — no R-peak annotations, no
beat timings, no artifact or signal-quality annotations, no reference
measurements. There is therefore no way to score R-peak sensitivity, heart-rate
accuracy, QRS accuracy, artifact localisation, or whether any individual
refusal was correct. Deriving any of those from the ECG itself would mean
scoring the pipeline against another algorithm's opinion and calling it ground
truth; the script does not do that, and it prints what it could *not* evaluate
alongside what it could. The class label is used only to group results for
reporting — it never enters the pipeline.

**What it does show:**

| | |
|---|---|
| Fragments processed / succeeded | 1000 / 1000 |
| Crashes / caught stage errors | 0 / 0 |
| Signal quality (raw) | median 99.1, p05 80.9, range 10.6–100.0 |
| Heart rate accepted | 390 (39.0%) |
| RR interval accepted | 376 (37.6%) |
| RR consistency accepted | 429 (42.9%) |
| QRS duration accepted | 353 (35.3%) |
| Accepted values outside physiological rails | 0 |
| Heart rate vs RR interval self-consistency | 359 jointly accepted, max disagreement 0.06 BPM |
| Determinism (same fragment analysed twice) | identical |
| Ground-truth isolation, re-proved on real signals | holds |

The result that matters for the thesis: **197 of the 1000 fragments produced
differing heart-rate and QRS verdicts** — 117 with heart rate accepted and QRS
refused, 80 the other way round. The two gates decouple on signals this project
did not create, which is not something a single global quality score can do.

The fragments the dataset labels *AFIB* show it most clearly: QRS duration was
accepted on 73.3% of them while heart rate was accepted on 24.4%. Morphological
evidence survives while the RR-consistency criterion declines to report one
representative rate for an irregular sequence of intervals. That is the gate
behaving as specified, on a real signal, with no knowledge of the label — and
it is a statement about evidence, not a clinical finding.

### Ground-truth isolation

Verified by experiment, not only by inspection (`tests/test_isolation.py`).
Each record is analysed twice — once with correct ground truth attached, once
with `r_peaks_true` and the metadata replaced by deliberately *wrong* values —
and every quality score, detection, recovery decision, trust label, beat verdict
and measurement must come out bit-for-bit identical. It does, for every demo
scenario and every artifact class.

### Known limitations

Stated plainly, because a report that hides its failures is worth nothing:

- **The system is conservative, and measurably so.** 71 of 328 cases refused a
  heart rate that ground truth says was supportable. The largest group is
  record-wide motion: the detector is deliberately tuned for sensitivity, so
  motion inflates the candidate count with spurious peaks, and the
  `validated beats / detected candidates` fraction then falls below its
  threshold even when twenty perfectly good beats were validated. This is a
  known, deliberate trade: with zero false acceptances, loosening an evidence
  criterion to recover these cases risks converting them into the failure mode
  that actually matters.
- **Heart rates below about 31 BPM are refused by design.** `RR_MAX_SEC` is
  2.0 s, so intervals longer than that are treated as implausible and
  discarded. This is a documented physiological rail in `src/config.py`, and it
  fails safe.
- **Localisation IoU is ~0.5 for mild artifacts**, rising to ~0.85 at moderate
  and high severity. A weak disturbance only crosses the detection threshold
  near its centre.
- **Electrode contact loss at severity 0.4 is classified as baseline wander.**
  At that severity the disturbance genuinely *is* mostly a baseline step with
  the signal still present, and the call is made at low confidence.
- **On real ECGs the system refuses considerably more often than on synthetic
  ones.** Heart rate was accepted on 39% of the 1000 fragments, and on only 35%
  of the normal-sinus ones — despite a median quality of 99.8/100 across the
  refused group. The cause is characterised rather than guessed: real MLII
  recordings produce roughly one spurious sub-threshold candidate per beat, so
  the sensitivity-tuned detector emits about twice as many candidates as there
  are beats. The evidence engine classifies them correctly — on record 100,
  all 13 true beats accepted and all 12 spurious ones refused — but
  `validated beats / detected candidates` then lands at a median of exactly
  0.50, just under its 0.55 threshold. This is the denominator limitation above,
  surfacing harder on real morphology than on synthetic. It is not a
  sampling-rate effect: resampling to 250 Hz reproduces the candidate counts and
  the verdicts exactly.
- **Three of the 45 source recordings have inverted R waves in MLII** (107, 108
  and 217). `peaks.py` assumes an upright R wave and those records sit at 0%
  acceptance. They fail safe rather than mismeasuring — no implausible value was
  accepted anywhere in the corpus — but they are not analysed.
- **10 s fragments are structurally tight.** `HR_MIN_VALID_BEATS` is 8, so a
  10 s record at 60 BPM offers about 10 beats and very little headroom.
- QRS duration is a **prototype estimate from a slope-threshold rule**. It is
  internally consistent and reproducible, but it has not been compared against
  any clinical annotation and no clinical meaning is claimed.
- **No clinical validation of any kind, and no accuracy claim on real signals.**
  The architecture is validated on controlled synthetic and adversarial data,
  where ground truth is known, and separately exercised on 1000 real ECG
  fragments for robustness. That real-ECG run is a *behaviour* evaluation, not
  an accuracy benchmark — the dataset carries nothing to score against. All data
  throughout is single-lead.

## Demo scenarios

All deterministic, all built from fixed seeds.

| scenario | what it shows |
|---|---|
| Clean ECG | quality 100/100, no artifact, all four measurements reported |
| Baseline wander | drift localised, high-pass recovery, revalidation confirms the drift is gone |
| Power-line interference | narrowband energy identified, notch filter, quality 63 → 100 |
| **Muscle noise (mild degradation)** | **heart rate accepted, QRS duration refused — the central case** |
| Motion artifact (recoverable) | localised to 8.2–10.8 s, region quality 54 → 71, recovery validated |
| Motion artifact (unrecoverable) | same artifact class, higher severity; filtering raises the score but revalidation **fails**, so the region stays rejected and the measurements are rebuilt from the rest of the record |
| Severe corruption | recovery fails, all four measurements refused |

The last two are the pair worth showing together: identical artifact class,
identical filter, opposite verdict — decided by revalidation rather than by
assumption.

### The five-act walkthrough

The dashboard's **Guided walkthrough** steps through five of those scenarios in
an order that tells one story, with narration for each act. The same sequence
runs in the terminal:

```bash
python tools/demo_flow.py
```

| act | scenario | heart rate | QRS duration |
|---|---|---|---|
| 1 | Clean ECG | ACCEPTED | ACCEPTED |
| 2 | Motion artifact (unrecoverable) | ACCEPTED | ACCEPTED |
| 3 | Motion artifact (recoverable) | ACCEPTED | ACCEPTED |
| 4 | **Muscle noise** | **ACCEPTED** | **NOT REPORTED** |
| 5 | Severe corruption | NOT REPORTED | NOT REPORTED |

Acts 2 and 3 are the revalidation pair — same artifact class, opposite
revalidation outcome. Act 4 is the row where the two columns disagree, which is
the whole argument. Act 5 is the refusal, with the withheld 72.5 BPM shown as
withheld.

Every act calls the same `run_pipeline` the dashboard calls. There is no
demo-only code path and no stored result: if the architecture stopped producing
the behaviour an act describes, the tool would show it.

## Language

The system reports *signal pattern detected*, *measurement accepted*,
*measurement rejected*, *insufficient evidence*, *potential artifact*,
*prototype signal-quality assessment*. It never states or implies a diagnosis,
and it makes no claim of clinical validation.
