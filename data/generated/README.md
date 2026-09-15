# Generated data

Nothing is written here by default — the pipeline runs entirely in memory and
the dashboard needs no files on disk.

`python tools/export_demo.py` fills this directory with, for each demo
scenario, a CSV of the raw and processed waveform and a JSON of the detections,
trust map and per-measurement verdicts. Useful for handing a fixed record to
the hardware side, or for inspecting a run outside the dashboard.
