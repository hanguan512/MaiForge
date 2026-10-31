# AGENTS.md

## What this repo is
- Audio → event IR → **maidata** (simai/maimai DX) toolkit. P0/P1/P2 are implemented. P3 training/inference code exists in `src/mainote/ml/`, but no checkpoint is trained or shipped.
- Core invariant: generators produce `Event` IR; `src/mainote/maidata/compiler.py` owns all maidata formatting (`{n}`, commas, `[a:b]`, `E`). Do not generate raw maidata text directly.
- `MAIDATA_FORMAT.md` is the format reference. `AUDIO_TO_MAIDATA_PLAN.md` is a design plan and is stale in places (it says 384 ticks/bar; code uses 960 ticks/beat and 3840 ticks/bar). Trust code and tests.
- P3 workflow is documented in `docs/TRAINING.md`; training must run on a GPU server, not in this CPU-only workspace.

## Commands
- P0 has no third-party deps: `PYTHONPATH=src python3 -m mainote validate|stats|roundtrip|parse|compile|eval ...`
- Full test suite (audio tests need numpy): `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` (84 tests).
- P0-only tests: `PYTHONPATH=src python3 -m unittest discover -s tests` (12 numpy/torch-dependent tests skip).
- Single test: `PYTHONPATH=src python3 -m unittest tests.test_parser.TestTokenSyntax.test_path_chain -v`
- Demo: `PYTHONPATH=src .venv/bin/python scripts/make_demo.py out/demo`
- Audio, demo, and `dataset ... --align` require `.venv/bin/python`; system `python3` has no numpy. The editable `.venv/bin/mainote` entrypoint also exists.
- P3 commands (on a GPU server with `.[audio,train]`):
  `mainote ml prepare-features ...`, `mainote ml train --config configs/p3_base.yaml`, `mainote ml infer --checkpoint ... --audio ... -o ...`.
- P3 needs `torch`/`pyyaml` (the `train` extra). This workspace has CPU-only PyTorch for smoke tests; use a GPU server for real training.
- No lint/typecheck/CI config; verify with unittest plus `PYTHONPATH=src python3 -m compileall -q src tests scripts`.
- If pip cannot resolve packages, add `-i https://pypi.org/simple` (the default index failed in this environment).

## Layout
- `src/mainote/maidata/`: `model.py` (IR, tick constants, `Event.chain_type`), `parser.py`, `compiler.py`, `validate.py`, `stats.py`.
- `src/mainote/audio/`: ffmpeg decode + numpy onset/BPM/beat analysis; `TimingInfo` JSON. Keep this frontend numpy-only unless there is a strong reason.
- `src/mainote/generate/rule.py`: hand-tuned rule baseline; `DIFFICULTY_RULES` and `LEVEL_DENSITY_ANCHORS` are priors, not learned truth.
- `src/mainote/dataset/`: `ingest`, `align`, `report`, `split`, `samples`.
- `src/mainote/eval/metrics.py`: note-level comparison (onset F1, position/type, duration, slide endpoints).
- `src/mainote/ml/`: P3 tokenizer, log-mel features, window dataset, model, train loop, greedy grammar decoder, inference.
- `configs/p3_base.yaml`: base training config.
- `docs/TRAINING.md`: server data prep, training, inference, evaluation, troubleshooting.
- `tests/`: stdlib `unittest`, not pytest.

## Dataset / training samples
- Raw layout: `data/raw/<song_id>/{audio.*,maidata.txt,meta.json?}`.
- `dataset ingest --align` → `data/processed/<song_id>/{meta.json,maidata.ir.json}`. One IR per song keeps all difficulties and shares audio.
- `dataset samples --splits ... --require-audio` → one sample per `(song_id, inote_N)`:
  `data/samples/<song_id>/inote_N.json`, `manifest.jsonl`, `summary.json`, `splits.json`.
- `manifest.jsonl` fields include `sample_id`, `chart_index`, `difficulty`, `level_num`, `audio`, `ir`, `first`, `initial_bpm`, `event_count`, `stats`, `alignment_confidence`, `valid`, `split`.
- Split **by song only**; all difficulties of a song stay in one split. Never split by chart or window.
- Ingest stores absolute audio/IR paths; moving the dataset requires re-ingest.
- `dataset samples` does not replace `maidata.ir.json`; the IR remains the source of truth, sample files are chart-level training views.
- P3 training also needs `mainote ml prepare-features data/processed --out data/features`; each song becomes `data/features/<song_id>.npz` (mel, fps, sr, n_mels, duration).

## Gotchas
- `B2` is Touch, `2b` is Break; `E1` is Touch, `E` is End; `1/8` is EACH, `1-5` is Slide. See `MAIDATA_FORMAT.md`.
- Parser supports `*` same-start chains and connected path slides (`4-1>2-7[16:7]`) via `Event.chain_type` (`same_start` / `path`). Do not collapse them into one case.
- `Duration.raw` preserves odd `[...]` forms in IR JSON but is excluded from semantic round-trip signatures. Compare semantic fields, not raw formatting.
- Compiler groups all `&lv`/`&des` before all `&inote` bodies and keeps generated metadata minimal (`title`, `artist`, `wholebpm`; no `&first=0`/`&clock_count`). This is reader-compatibility behavior; do not revert casually.
- The committed `dataset/xxdw/maidata` is the only fixture: 4 charts, BPM 143, no audio. Use it for parser/compiler goldens.
- `out/`, `data/raw/`, `data/processed/`, `data/features/`, `data/samples/`, `runs/`, `assets/synthetic/`, `.venv/`, and audio files are gitignored. `audio-FM/` is local user audio; never commit it.
- The current machine has no GPU and there is no training checkpoint. Do not claim a trained neural model exists.
- If real-song BPM/downbeat is wrong: `analyze --bpm N --first S`, then `generate --timing timing.json`.
