# MaiForge

**Audio → maidata chart generation for maimai DX / simai.**

MaiForge turns an audio file into a simai chart by generating a structured event
IR and compiling it deterministically to `maidata.txt`. The repo currently
contains the P0–P2 foundation: a full format toolchain, an audio timing frontend,
a rule-based baseline, a dataset pipeline, and note-level evaluation. The neural
audio-to-chart model (P3) is planned but not trained yet.

> **Naming:** MaiForge is the project name. The Python package and CLI are
> currently named `mainote`; a package rename is planned.

## Why MaiForge?

Generating `maidata` text directly is brittle: commas, `{n}` divisions, `[a:b]`
durations, slide paths, and the final `E` are easy to get wrong. MaiForge splits
the problem into two layers:

```text
audio → timing → event IR → deterministic compiler → maidata.txt
```

- The generator only decides **what note, where, when, and how long**.
- The compiler decides **formatting**: commas, divisions, duration fractions,
  metadata order, and `E`.
- Every output can be parsed back and checked with round-trip tests.

This makes the format layer reliable even while the musical generation layer is
still being developed.

## Features

- **maidata parser / IR / compiler / validator** for multi-difficulty charts.
- Supports Tap, Hold, Slide, Touch, Touch Hold, Break, EX, fireworks, comments,
  BPM changes, per-difficulty `first`, `*` slide chains, and connected path slides.
- **Round-trip guarantee:** IR → maidata → IR is covered by tests.
- **Audio frontend:** ffmpeg decoding, numpy onset detection, BPM/beat/downbeat
  tracking, and manual `--bpm` / `--first` correction.
- **Rule-based baseline:** generates valid charts from audio with difficulty and
  level conditioning.
- **Dataset pipeline:** ingest, audio-chart alignment scoring, per-difficulty
  statistics, leakage-safe song splits, and per-difficulty sample export.
- **Evaluation:** onset F1, position/type accuracy, duration error, slide
  endpoint accuracy, and difficulty-profile deltas.
- **Synthetic demo:** exercises the whole audio → maidata pipeline without any
  copyrighted audio.

## Status

| Phase | Scope | Status |
|---|---|---|
| P0 | Parser, IR, compiler, validator, stats, round-trip tests | Done |
| P1 | Audio timing frontend, rule baseline, end-to-end CLI | Done |
| P2 | Dataset ingest/alignment/report/split/samples, evaluation | Done |
| P3 | Neural audio encoder + event decoder | Planned |

There is no trained neural checkpoint in this repository. The current generator
is a rule baseline; it follows detected onsets/beats and a target difficulty
profile, but it does not understand melody or chart style.

## Quick start

### P0 toolchain (no third-party dependencies)

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

### Full environment

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[audio]'   # numpy + mutagen
# If pip cannot resolve packages, add: -i https://pypi.org/simple
```

Then use `.venv/bin/python` for audio and dataset alignment commands.

```bash
# Analyze
PYTHONPATH=src .venv/bin/python -m mainote analyze song.mp3 --out out/timing.json

# Generate
PYTHONPATH=src .venv/bin/python -m mainote generate song.mp3 \
  --difficulty master --level 12.9 -o out/maidata.txt
```

If the BPM or downbeat is wrong, override it and regenerate:

```bash
PYTHONPATH=src .venv/bin/python -m mainote analyze song.mp3 \
  --bpm 160 --first 0.0 --out out/timing.json
PYTHONPATH=src .venv/bin/python -m mainote generate song.mp3 \
  --timing out/timing.json --difficulty master --level 12.9 -o out/maidata.txt
```

### Synthetic demo

No audio file is required:

```bash
PYTHONPATH=src .venv/bin/python scripts/make_demo.py out/demo
```

This writes a click track, timing JSON, generated chart, IR, and a small raw
dataset under `out/demo/`.

## CLI overview

```text
mainote parse <maidata> [--json out.ir.json]
mainote compile <ir.json> -o <maidata.txt>
mainote validate <maidata|ir.json> [--json]
mainote stats <maidata|ir.json> [--json]
mainote roundtrip <maidata> [--out recompiled.txt]
mainote eval <generated> <reference> [--index N] [--tolerance 0.06]
mainote analyze <audio> --out timing.json [--bpm N] [--first S]
mainote generate <audio> -o maidata.txt [--difficulty master] [--level 12.9] [--timing timing.json]
mainote demo [--out out/demo]
mainote dataset ingest <raw_dir> --out <processed_dir> [--align]
mainote dataset report <processed_dir> [--out report.json] [--csv report.csv]
mainote dataset split <processed_dir> --out splits.json [--train 0.8 --val 0.1 --test 0.1]
mainote dataset samples <processed_dir> --out samples_dir [--splits splits.json] [--require-audio]
```

## Architecture

```text
audio
  │
  ├─ decode / mono / resample                  ffmpeg + numpy
  ├─ onset, BPM, beat, downbeat                audio/features.py
  │
  ├─ timing map (BPM, first, bars, onsets)     audio/timing.py
  │
  ├─ generator
  │    ├─ P1: rule baseline                    generate/rule.py
  │    └─ P3: neural event decoder             planned
  │
  ├─ event IR                                  maidata/model.py
  │    ├─ tick (960 per beat, 3840 per 4/4 bar)
  │    ├─ kind / position / flags / duration
  │    └─ slide chain (same_start or path)
  │
  ├─ validation                                maidata/validate.py
  ├─ deterministic compiler                    maidata/compiler.py
  └─ maidata.txt
```

The model-facing representation is the event IR, not raw maidata text. The
compiler chooses the smallest `{n}` per bar that aligns all events and BPM
changes, then writes commas, `[a:b]` durations, `&lv`/`&des` metadata, and `E`.

## Dataset pipeline

Raw layout:

```text
data/raw/<song_id>/
  audio.mp3 | audio.wav | audio.flac | ...
  maidata.txt
  meta.json            # optional: {"title": "...", "artist": "..."}
```

Ingest and inspect:

```bash
PYTHONPATH=src .venv/bin/python -m mainote dataset ingest data/raw \
  --out data/processed --align
PYTHONPATH=src .venv/bin/python -m mainote dataset report data/processed \
  --out out/dataset_report.json
```

Create song-level splits and per-difficulty samples:

```bash
PYTHONPATH=src .venv/bin/python -m mainote dataset split data/processed \
  --out out/splits.json --train 0.8 --val 0.1 --test 0.1 --seed 0

PYTHONPATH=src .venv/bin/python -m mainote dataset samples data/processed \
  --out data/samples --splits out/splits.json --require-audio
```

`dataset samples` keeps one IR per song but writes one self-contained training
sample per difficulty:

```text
data/samples/
  manifest.jsonl          # one row per (song_id, inote_N)
  summary.json
  splits.json
  <song_id>/
    inote_1.json ... inote_5.json
```

All difficulties of a song stay in the same split. Never split by chart or
window; that would leak the same audio across train/test. The manifest records
`sample_id`, `chart_index`, `difficulty`, `level_num`, `audio`, `ir`, `first`,
`initial_bpm`, `event_count`, `stats`, `alignment_confidence`, `valid`, and
`split`.

## Project layout

```text
src/mainote/
  maidata/      parser, IR model, compiler, validator, stats
  audio/        ffmpeg decode, onset/BPM/beat frontend, TimingInfo
  generate/     rule-based chart generator
  dataset/      ingest, alignment, report, split, per-difficulty samples
  eval/         note-level evaluation metrics
  cli.py        argparse CLI
tests/          stdlib unittest suite
scripts/        demo helper
dataset/xxdw/   committed maidata fixture (4 charts, BPM 143, no audio)
```

## Evaluation

The evaluator compares a generated chart with a reference chart using a time
tolerance (default 60 ms):

- onset precision / recall / F1
- position, type, and exact-match accuracy
- Hold duration MAE
- slide endpoint accuracy
- notes/sec, max simultaneous, and type-ratio deltas

```bash
PYTHONPATH=src python3 -m mainote eval generated.txt reference.txt --index 5
```

## Testing

```bash
# Full suite (audio tests need numpy)
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

# P0-only: 8 audio/dataset-alignment tests are skipped
PYTHONPATH=src python3 -m unittest discover -s tests

# Single test
PYTHONPATH=src python3 -m unittest tests.test_parser.TestTokenSyntax.test_path_chain -v
```

The suite currently has 70 tests. There is no separate lint/typecheck/CI setup;
`python3 -m compileall -q src tests scripts` is a useful quick check.

## Roadmap

- **P2.5:** calibrate difficulty profiles from real data, improve beat/downbeat
  tracking, inspect low-confidence alignments.
- **P3:** frozen audio features (MERT/PaSST) or a small Conformer, event
  Transformer decoder, difficulty conditioning, constrained decoding, and
  training/evaluation against the rule baseline. Requires paired data and a GPU.
- **P4:** full-song generation, cross-window Hold/Slide state, dynamic BPM,
  Touch/Break/EX, complex slides, and candidate reranking.
- **P5:** preview/editor, manual timing correction, partial regeneration, and
  packaging.

## Limitations

- The current generator is a **rule baseline**, not a learned model.
- The timing frontend assumes a **constant BPM** for a song.
- Beat/downbeat tracking is lightweight; real songs may need `--bpm`/`--first`
  correction.
- The committed fixture has no audio; audio is supplied by the user.
- Neural chart generation needs a substantial paired audio/chart dataset and a
  GPU. One song is not enough to train a general model.

## Documentation

- [`MAIDATA_FORMAT.md`](MAIDATA_FORMAT.md) — maidata/simai format reference.
- [`AUDIO_TO_MAIDATA_PLAN.md`](AUDIO_TO_MAIDATA_PLAN.md) — design plan for the
  full system. It is aspirational in places; the code and tests are the source
  of truth.
- [`data/README.md`](data/README.md) — dataset layout and quality rules.
- [`AGENTS.md`](AGENTS.md) — guidance for AI coding agents working in this repo.

## License

No license has been selected yet. Add a `LICENSE` file before publishing or
distributing the project.
