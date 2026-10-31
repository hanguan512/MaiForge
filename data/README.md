# Dataset layout

Training data is never committed to this repo.  Use this layout under
`data/raw/` (gitignored):

```text
data/raw/
  <song_id>/
    audio.wav | audio.mp3 | audio.flac | audio.m4a | audio.ogg
    maidata.txt          # or maidata
    meta.json            # optional: {"title": "...", "artist": "..."}
```

Ingest:

```bash
PYTHONPATH=src .venv/bin/python -m mainote dataset ingest data/raw \
  --out data/processed --align
```

This produces, per song:

```text
data/processed/<song_id>/
  meta.json          # metadata, stats, alignment, timing
  maidata.ir.json    # canonical IR for all charts
```

And a dataset-level `data/processed/ingest_summary.json`.

Report, split, and per-difficulty sample export:

```bash
PYTHONPATH=src .venv/bin/python -m mainote dataset report data/processed \
  --out out/dataset_report.json --csv out/dataset_report.csv
PYTHONPATH=src .venv/bin/python -m mainote dataset split data/processed \
  --out out/splits.json --train 0.8 --val 0.1 --test 0.1 --seed 0
PYTHONPATH=src .venv/bin/python -m mainote dataset samples data/processed \
  --out data/samples --splits out/splits.json --require-audio
```

`dataset samples` writes `manifest.jsonl` (one row per difficulty), one
self-contained `inote_N.json` per chart, `summary.json`, and `splits.json`.
The audio is referenced by path rather than copied, and all difficulties of a
song share the same split.

## Quality rules

- Audio and maidata must be the **same edit/version**.  Check duration, BPM,
  `first`, and the alignment confidence in `meta.json`.
- Do not split a song across train/val/test.  Split by `song_id` (and ideally by
  artist/BPM/genre for generalization tests).
- Low-confidence alignments (< 0.5) should be manually checked or excluded
  before training.
- Keep copyrighted audio local; document sources and licenses.
