#!/usr/bin/env python3
"""Teacher-force a P3 checkpoint and report event-type calibration.

The 2026-11-09 posdur run generated 51% slides / 44% taps against a dataset
prior of ~14% / ~70% (master charts).  Note-level validation missed it because
``type_accuracy`` only judges notes that already matched.  This script answers
the diagnostic questions without running the slow decoder:

* gold vs teacher-forced argmax type-token ratios;
* the full gold -> predicted type confusion matrix;
* whether ``position_proj`` / ``duration_proj`` output has grown past the
  token embedding it was warm-started from.

    PYTHONPATH=src .venv/bin/python scripts/diagnose_type_bias.py \
        --checkpoint runs/p3_posdur_warm/best.pt --split val --device cpu
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from mainote.ml.dataset import ChartWindowDataset
from mainote.ml.infer import load_checkpoint
from mainote.ml.tokenizer import TYPE_FROM_TOKEN
from mainote.ml.train import resolve_device

NON_TYPE = "<non-type>"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teacher-forced event-type calibration for a P3 checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None, help="defaults to checkpoint config")
    parser.add_argument(
        "--features-dir", default=None, help="defaults to checkpoint config"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None, help="write JSON here as well")
    return parser


def _choose_windows(dataset: ChartWindowDataset, limit: int) -> list[int]:
    total = len(dataset)
    if total <= limit:
        return list(range(total))
    step = total / max(1, limit)
    return [min(total - 1, int(index * step)) for index in range(limit)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, vocab, tokenizer, checkpoint = load_checkpoint(
        args.checkpoint, device
    )
    cfg = checkpoint.get("config") or {}
    data_cfg = cfg.get("data", {})
    manifest = Path(
        args.manifest
        or data_cfg.get("manifest")
        or "data/samples/manifest.jsonl"
    )
    features_dir = Path(
        args.features_dir or data_cfg.get("features_dir") or "data/features"
    )
    # Checkpoints trained on another machine (e.g. Windows) store absolute
    # paths that do not exist here; fall back to the standard local layout
    # unless the user explicitly chose a path.
    if not manifest.exists():
        fallback = Path("data/samples/manifest.jsonl")
        if not args.manifest and fallback.exists():
            manifest = fallback
        else:
            raise SystemExit(
                f"manifest not found: {manifest} "
                "(checkpoint config points at the training machine?); "
                "pass --manifest"
            )
    if not features_dir.exists():
        fallback = Path("data/features")
        if not args.features_dir and fallback.exists():
            features_dir = fallback
        else:
            raise SystemExit(
                f"features dir not found: {features_dir} "
                "(checkpoint config points at the training machine?); "
                "pass --features-dir"
            )
    dataset = ChartWindowDataset(
        manifest,
        features_dir,
        args.split,
        tokenizer,
        window_bars=int(data_cfg.get("window_bars", 16)),
        stride_bars=int(data_cfg.get("stride_bars", 8)),
        context_bars=int(data_cfg.get("context_bars", 2)),
        min_events=int(data_cfg.get("min_events", 1)),
        max_audio_frames=int(
            data_cfg.get("max_audio_frames", model.config.max_audio_frames)
        ),
        max_tokens=model.config.max_tokens,
        min_alignment=float(data_cfg.get("min_alignment", 0.0)),
        shuffle_same_tick_events=bool(
            data_cfg.get("shuffle_same_tick_events", False)
        ),
    )
    if len(dataset) == 0:
        raise SystemExit(
            f"no usable windows in split {args.split!r} ({manifest})"
        )

    selected = _choose_windows(dataset, max(1, args.max_windows))
    type_token_ids = [
        vocab.token_to_id[token]
        for token in TYPE_FROM_TOKEN
        if token in vocab.token_to_id
    ]
    type_mask = torch.zeros(len(vocab.tokens), dtype=torch.bool)
    type_mask[type_token_ids] = True
    type_mask = type_mask.to(device)

    gold_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    total = 0
    embed_norm_sum = 0.0
    position_delta_sum = 0.0
    duration_delta_sum = 0.0
    token_count = 0

    model.eval()
    batch_size = max(1, args.batch_size)
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            items = [dataset[index] for index in selected[start : start + batch_size]]
            tokens = torch.full(
                (len(items), max(len(item["tokens"]) for item in items)),
                vocab.pad_id,
                dtype=torch.long,
            )
            token_mask = torch.ones(tokens.shape, dtype=torch.bool)
            max_frames = max(item["mel"].shape[0] for item in items)
            mel = torch.zeros(
                (len(items), max_frames, items[0]["mel"].shape[1]),
                dtype=torch.float32,
            )
            mel_mask = torch.ones((len(items), max_frames), dtype=torch.bool)
            for row, item in enumerate(items):
                length = len(item["tokens"])
                tokens[row, :length] = torch.tensor(
                    item["tokens"], dtype=torch.long
                )
                token_mask[row, :length] = False
                frames = item["mel"].shape[0]
                mel[row, :frames] = torch.from_numpy(item["mel"])
                mel_mask[row, :frames] = False
            tokens = tokens.to(device)
            if tokens.size(1) < 2:
                continue
            logits = model(
                mel.to(device),
                mel_mask.to(device),
                tokens[:, :-1],
                token_mask[:, :-1].to(device),
            )
            preds = logits.argmax(dim=-1)
            labels = tokens[:, 1:]
            mask = labels.ne(vocab.pad_id) & type_mask[labels]
            gold_ids = labels[mask].tolist()
            pred_ids = preds[mask].tolist()
            for gold_id, pred_id in zip(gold_ids, pred_ids):
                gold = TYPE_FROM_TOKEN[vocab.id_to_token[gold_id]]
                pred = TYPE_FROM_TOKEN.get(
                    vocab.id_to_token[pred_id], NON_TYPE
                )
                gold_counts[gold] += 1
                pred_counts[pred] += 1
                confusion[gold][pred] += 1
            total += len(gold_ids)

            # Embedding-scale check: the added geometry/duration delta should
            # stay small relative to the token content it modifies.
            inputs = tokens[:, :-1]
            valid = inputs.ne(vocab.pad_id)
            token_count += int(valid.sum())
            base = model.token_embed(inputs)
            embed_norm_sum += float(base.norm(dim=-1)[valid].sum())
            if model.position_proj is not None:
                delta = model.position_proj(
                    model.position_feature_table[inputs]
                )
                position_delta_sum += float(delta.norm(dim=-1)[valid].sum())
            if model.duration_proj is not None:
                delta = model.duration_proj(
                    model.duration_feature_table[inputs]
                )
                duration_delta_sum += float(delta.norm(dim=-1)[valid].sum())

    def _ratios(counts: Counter[str]) -> dict[str, float]:
        return {
            kind: round(counts[kind] / total, 4)
            for kind in sorted(counts)
        } if total else {}

    predicted_ratios = _ratios(pred_counts)
    gold_ratios = _ratios(gold_counts)
    slide_gap = predicted_ratios.get("slide", 0.0) - gold_ratios.get(
        "slide", 0.0
    )

    state = model.state_dict()
    weight_norms: dict[str, float] = {}
    for name in ("token_embed.weight", "position_proj.weight", "duration_proj.weight"):
        if name in state:
            weight_norms[name] = round(float(state[name].norm()), 4)

    def _mean(total_norm: float) -> float | None:
        return round(total_norm / token_count, 4) if token_count else None

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metric": checkpoint.get("best_metric_name"),
        "split": args.split,
        "windows": len(selected),
        "type_tokens": total,
        "gold_ratios": gold_ratios,
        "predicted_ratios": predicted_ratios,
        "type_accuracy": round(
            sum(confusion[kind][kind] for kind in confusion) / total, 4
        )
        if total
        else None,
        "confusion": {
            gold: dict(sorted(counts.items()))
            for gold, counts in sorted(confusion.items())
        },
        "embedding_norms": {
            "token_embed_mean": _mean(embed_norm_sum),
            "position_delta_mean": _mean(position_delta_sum)
            if model.position_proj is not None
            else None,
            "duration_delta_mean": _mean(duration_delta_sum)
            if model.duration_proj is not None
            else None,
        },
        "weight_norms": weight_norms,
    }
    if total and slide_gap > 0.15:
        result["warning"] = (
            f"teacher-forced argmax over-predicts slides by {slide_gap:.1%}; "
            "the checkpoint itself is type-biased, not just the sampler"
        )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
