# mainoteD

Audio → chart IR → **maidata** toolkit for maimai DX / simai charts.

This repo currently contains the deterministic foundation (P0), a timing frontend and
rule-based baseline (P1), and dataset ingestion/reporting tools (P2). The neural
audio-to-chart model (P3) is intentionally not trained here yet: it needs paired
audio/chart data and a GPU.

## Quick start

No third-party dependency is needed for the P0 toolchain:

```bash
cd /root/mainoteD
PYTHONPATH=src python3 -m mainote parse dataset/xxdw/maidata --json out/sample.ir.json
PYTHONPATH=src python3 -m mainote validate dataset/xxdw/maidata
PYTHONPATH=src python3 -m mainote stats dataset/xxdw/maidata
PYTHONPATH=src python3 -m mainote roundtrip dataset/xxdw/maidata
PYTHONPATH=src python3 -m mainote eval dataset/xxdw/maidata dataset/xxdw/maidata --index 5
```

Compile an IR back to maidata:

```bash
PYTHONPATH=src python3 -m mainote compile out/sample.ir.json -o out/roundtrip.txt
```

Run the test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Audio phase (P1)

The audio frontend is numpy-only (ffmpeg is used for non-WAV decoding):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[audio]'   # numpy + mutagen
# If pip cannot resolve packages in this environment, add:
#   -i https://pypi.org/simple
```

Then:

```bash
mainote analyze song.mp3 --out out/timing.json
mainote generate song.mp3 --difficulty master --level 12.9 -o out/maidata.txt
```

If you do not have audio yet, generate a synthetic click track and exercise the
whole pipeline:

```bash
PYTHONPATH=src .venv/bin/python scripts/make_demo.py
```

## Dataset phase (P2)

Expected raw layout:

```text
data/raw/<song_id>/
  audio.wav|mp3|flac|m4a|ogg
  maidata.txt
  meta.json          # optional: {"title": ..., "artist": ...}
```

Ingest and report:

```bash
PYTHONPATH=src .venv/bin/python -m mainote dataset ingest data/raw --out data/processed --align
PYTHONPATH=src .venv/bin/python -m mainote dataset report data/processed --out out/dataset_report.json
PYTHONPATH=src .venv/bin/python -m mainote dataset split data/processed --out out/splits.json
```

## Design notes

- The model-facing representation is an event IR, not raw maidata text.
- The compiler is deterministic and is covered by IR → maidata → IR round-trip tests.
- Times use 960 ticks per quarter note (3840 ticks per 4/4 bar) internally; the
  model can quantize to a coarser grid later.
- `first` is the audio offset of chart tick 0. The rule baseline sets it to 0 and
  leaves leading rests, which is valid simai.

## Status

- [x] P0: parser, IR, compiler, validator, stats, round-trip tests
- [x] P1: timing frontend, rule baseline, end-to-end CLI (tested with synthetic audio)
- [x] P2: dataset ingestion, alignment scoring, per-difficulty report
- [ ] P3: neural audio encoder + event decoder (needs paired data + GPU)
